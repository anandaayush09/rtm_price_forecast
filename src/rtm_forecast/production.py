"""The final day-ahead model for use: fit on all published days, save, forecast a new day.

The model is the cap-aware two-stage forecast of ``models.py`` (a classifier for "the block settles
at the cap" and quantile regressions on the blocks below it) with prediction intervals corrected
by a margin learned on the days before (``intervals.py``). In the backtest it was refitted every
day; here it is fitted once on the days given and saved, and the next day is refitted the same way.

Saved files are LightGBM text files plus a JSON manifest with the input columns, settings, last
training day and a checksum of every model file, so a changed or incomplete folder is refused on
load. Nothing is pickled.

The margins need the forecasts of the last days as they were made, together with the prices that
were then published (the calibration table). It starts from the backtest and grows by one day each
time ``update_calibration`` is given the published prices of a forecast day.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import Booster

from rtm_forecast.intervals import cap_state, conformal_adjust
from rtm_forecast.models import (
    check_levels,
    fit_two_stage,
    mixture_quantile,
    predict_two_stage,
    two_stage_settings,
)
from rtm_forecast.transforms import AsinhTransform

FORMAT_VERSION = 1
POINT_LEVELS = (0.5, 0.75, 0.9)
LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
COVERAGES = (0.5, 0.9)
WINDOW_DAYS = 60
MIN_DAYS = 14
RAW_COLUMNS = ["point", "lo50", "hi50", "lo90", "hi90"]
CALIBRATION_COLUMNS = ["date", "block", "actual", *RAW_COLUMNS]


@dataclass(frozen=True)
class Spec:
    """What the fit needs besides the data: input columns, LightGBM settings (without the
    objective), seed, price cap, and the scale of the asinh transform (None: raw prices)."""

    columns: tuple[str, ...]
    params: dict[str, Any]
    seed: int
    cap: float
    asinh_scale: float | None = None
    levels: tuple[float, ...] = LEVELS

    def __post_init__(self) -> None:
        check_levels(np.asarray(self.levels, dtype=float))
        if not set(POINT_LEVELS) <= set(self.levels):
            raise ValueError(f"levels must include {POINT_LEVELS}")

    @property
    def transform(self) -> AsinhTransform | None:
        return None if self.asinh_scale is None else AsinhTransform(self.asinh_scale)


class ProductionModel:
    """A fitted two-stage model: ``raw`` gives the point forecast and the uncorrected intervals."""

    def __init__(
        self,
        spec: Spec,
        trained_until: date,
        train_days: int,
        train_rows: int,
        classifier: Booster | None,
        regressors: Sequence[Booster],
    ) -> None:
        self.spec = spec
        self.trained_until = trained_until
        self.train_days = train_days
        self.train_rows = train_rows
        self.classifier = classifier
        self.regressors = list(regressors)

    def raw(self, rows: pd.DataFrame) -> pd.DataFrame:
        """``point`` (median of the mixture, the cap-aware forecast) and the interval ends
        ``lo50``, ``hi50``, ``lo90``, ``hi90`` before the correction, one row per input row."""
        spec = self.spec
        columns, cap, transform = spec.columns, spec.cap, spec.transform
        # the point forecast uses the fits at its own levels, as the backtest model does
        chosen = [spec.levels.index(level) for level in POINT_LEVELS]
        p, below = predict_two_stage(
            self.classifier, [self.regressors[i] for i in chosen], rows, columns, transform
        )
        out = {"point": mixture_quantile(p, below, np.asarray(POINT_LEVELS), 0.5, cap)}
        p, quantiles = predict_two_stage(self.classifier, self.regressors, rows, columns, transform)
        grid = np.asarray(spec.levels)
        for coverage in COVERAGES:
            tail = (1.0 - coverage) / 2.0
            tag = round(coverage * 100)
            out[f"lo{tag}"] = mixture_quantile(p, quantiles, grid, tail, cap)
            out[f"hi{tag}"] = mixture_quantile(p, quantiles, grid, 1 - tail, cap)
        return pd.DataFrame(out, index=rows.index)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        boosters = {f"quantile_{i}.txt": b for i, b in enumerate(self.regressors)}
        if self.classifier is not None:
            boosters["classifier.txt"] = self.classifier
        for name, booster in boosters.items():
            booster.save_model(str(folder / name))
            files[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        manifest = {
            "format_version": FORMAT_VERSION,
            "columns": list(self.spec.columns),
            "params": self.spec.params,
            "seed": self.spec.seed,
            "cap": self.spec.cap,
            "asinh_scale": self.spec.asinh_scale,
            "levels": list(self.spec.levels),
            "trained_until": self.trained_until.isoformat(),
            "train_days": self.train_days,
            "train_rows": self.train_rows,
            "lightgbm_version": lightgbm.__version__,
            "files": files,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")

    @classmethod
    def load(cls, folder: Path) -> "ProductionModel":
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format_version"] != FORMAT_VERSION:
            raise ValueError(f"unknown model format {manifest['format_version']}")
        for name, checksum in manifest["files"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != checksum:
                raise ValueError(f"{name} does not match the manifest")
        spec = Spec(
            columns=tuple(manifest["columns"]),
            params=manifest["params"],
            seed=manifest["seed"],
            cap=manifest["cap"],
            asinh_scale=manifest["asinh_scale"],
            levels=tuple(manifest["levels"]),
        )
        classifier = (
            Booster(model_file=str(folder / "classifier.txt"))
            if "classifier.txt" in manifest["files"]
            else None
        )
        regressors = [
            Booster(model_file=str(folder / f"quantile_{i}.txt")) for i in range(len(spec.levels))
        ]
        return cls(
            spec,
            date.fromisoformat(manifest["trained_until"]),
            manifest["train_days"],
            manifest["train_rows"],
            classifier,
            regressors,
        )


def fit_production(
    features: pd.DataFrame, spec: Spec, before: pd.Timestamp | None = None
) -> ProductionModel:
    """Fit on every day with published prices (only those before ``before`` if it is given)."""
    train = features[features["target_observed"]]
    if before is not None:
        train = train[train["date"] < before]
    if train.empty:
        raise ValueError("no training days")
    settings = two_stage_settings(spec.params, spec.seed)
    classifier, regressors = fit_two_stage(
        train, spec.columns, spec.transform, settings, spec.cap, np.asarray(spec.levels)
    )
    last = pd.Timestamp(train["date"].max()).date()
    return ProductionModel(
        spec, last, int(train["date"].nunique()), len(train), classifier, regressors
    )


def forecast_day(
    model: ProductionModel,
    rows: pd.DataFrame,
    calibration: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
    min_days: int = MIN_DAYS,
) -> pd.DataFrame:
    """Forecast one day: ``rows`` are its 96 input rows (the published price may be missing).

    Returns ``date``, ``block``, ``point``, the corrected intervals ``lo50``, ``hi50``, ``lo90``,
    ``hi90``, ``state`` and the uncorrected ends as ``raw_lo50`` and so on. The margin of each
    cap state comes from the last ``window_days`` days of ``calibration`` before the day.
    """
    days = pd.DatetimeIndex(rows["date"]).unique()
    if len(days) != 1:
        raise ValueError("forecast one day at a time")
    day = days[0]
    earlier = calibration[calibration["date"] < day]
    if earlier["date"].nunique() < min_days:
        raise ValueError(
            f"the calibration table needs at least {min_days} days before {day:%Y-%m-%d}"
        )
    cap = model.spec.cap
    raw = model.raw(rows)
    today = pd.concat([rows[["date", "block"]], raw], axis=1).assign(actual=np.nan)
    table = pd.concat([earlier[CALIBRATION_COLUMNS], today[CALIBRATION_COLUMNS]], ignore_index=True)
    table["state"] = cap_state(
        table["point"].to_numpy(), table["hi50"].to_numpy(), table["hi90"].to_numpy(), cap
    )
    result = today[["date", "block", "point"]].copy()
    for coverage in COVERAGES:
        tag = round(coverage * 100)
        adjusted = conformal_adjust(
            table.assign(lo=table[f"lo{tag}"], hi=table[f"hi{tag}"]),
            coverage,
            window_days,
            min_days,
            cap=cap,
            group="state",
        )
        mine = adjusted[adjusted["date"] == day].set_index("block")
        blocks = result["block"].to_numpy()
        result[f"lo{tag}"] = mine.loc[blocks, "lo_adj"].to_numpy()
        result[f"hi{tag}"] = mine.loc[blocks, "hi_adj"].to_numpy()
    result["state"] = table["state"].to_numpy()[len(earlier) :]
    for name in RAW_COLUMNS[1:]:
        result[f"raw_{name}"] = raw[name].to_numpy()
    return result


def update_calibration(
    calibration: pd.DataFrame, forecast: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Add a forecast day to the calibration table once its prices are published.

    ``forecast`` is the result of ``forecast_day`` and ``prices`` has ``date``, ``block`` and
    ``rtm_price`` (NaN where nothing was published). The forecast is stored as it was made.
    """
    day = pd.DatetimeIndex(forecast["date"]).unique()
    if len(day) != 1:
        raise ValueError("one forecast day at a time")
    if (calibration["date"] == day[0]).any():
        raise ValueError(f"{day[0]:%Y-%m-%d} is already in the calibration table")
    known = prices.set_index(["date", "block"])["rtm_price"]
    keys = pd.MultiIndex.from_frame(forecast[["date", "block"]])
    if not keys.isin(known.index).all():
        raise ValueError("the prices do not cover every forecast block")
    added = forecast[["date", "block", "point"]].assign(actual=known.reindex(keys).to_numpy())
    for name in RAW_COLUMNS[1:]:
        added[name] = forecast[f"raw_{name}"].to_numpy()
    return pd.concat([calibration, added[CALIBRATION_COLUMNS]], ignore_index=True)
