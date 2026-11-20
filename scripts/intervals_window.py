"""Prediction intervals for the day-ahead task on the 2025-26 window.

    python scripts/intervals_window.py fit      # quantiles of the two-stage model, all folds
    python scripts/intervals_window.py report   # conformal correction and coverage scores

``fit`` refits the cap-aware model every day (as ``benchmark_cap_aware.py`` does) and keeps the
50 and 90 percent interval ends of its mixture distribution. ``report`` shifts the ends by a
margin learned on the 60 days before each day (conformal correction) and scores the intervals on
the validation and test days: share of published prices inside, mean width, interval score, and
the share inside on cap blocks and on the others. Two reference intervals use the same
correction around a point forecast: the cap-aware median and the plain LightGBM forecast.

Results are private files under .project-local/results_window. Intervals are in
``intervals_cap_mixture.parquet``, not in ``predictions_*`` files, so the point-forecast report
does not read them.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from benchmark_window import Inputs, read_setup

from rtm_forecast import models
from rtm_forecast.backtest import rolling_origin
from rtm_forecast.intervals import conformal_adjust, interval_scores

COVERAGES = (0.5, 0.9)
WINDOW_DAYS = 60


def fit(inputs: Inputs, redo: bool) -> None:
    target = inputs.out / "intervals_cap_mixture.parquet"
    if target.exists() and not redo:
        print("intervals already there (use --redo)")
        return
    setup = read_setup(inputs)
    best = json.loads((inputs.out / "best_params.json").read_text(encoding="utf-8"))
    predictors = models.cap_aware_interval_predictors(
        setup["columns"],
        setup["transform"],
        best["lightgbm"]["params"],
        inputs.experiment.seed,
        inputs.experiment.price_cap,
        COVERAGES,
    )
    days = [d for fold in inputs.folds for d in fold]
    fold_of = {day: k + 1 for k, fold in enumerate(inputs.folds) for day in fold}
    started = time.time()
    frame: pd.DataFrame | None = None
    for name, predictor in predictors.items():
        result = rolling_origin(setup["frame"], days, predictor, name)
        if frame is None:
            frame = result[["date", "block", "actual"]].copy()
        frame[name] = result["prediction"].to_numpy()
        print(f"{name}: {time.time() - started:.0f}s", flush=True)
    assert frame is not None
    frame["fold"] = frame["date"].map(fold_of)
    frame["period"] = np.select(
        [frame["date"].isin(inputs.periods.test), frame["date"].isin(inputs.periods.validation)],
        ["test", "validation"],
        "earlier",
    )
    frame.to_parquet(target, index=False)


STATES = {3: "cap called", 2: "cap likely", 1: "cap possible", 0: "cap unlikely"}


def scored_variants(inputs: Inputs) -> pd.DataFrame:
    """Every interval variant for every coverage, one row per block and variant.

    ``state`` is how much the model expects the cap, from quantities known when the day is
    forecast: the cap-aware median is the cap (probability at least 0.5), the upper end of the
    50 percent interval is the cap (at least 0.25), the upper end of the 90 percent interval is
    the cap (at least 0.05), or none of these.
    """
    cap = inputs.experiment.price_cap
    raw = pd.read_parquet(inputs.out / "intervals_cap_mixture.parquet")
    points = {
        name: pd.read_parquet(inputs.out / f"predictions_{name}.parquet").set_index(
            ["date", "block"]
        )["prediction"]
        for name in ("cap_mixture", "lightgbm")
    }
    keys = raw.set_index(["date", "block"]).index
    around = {name: series.reindex(keys).to_numpy() for name, series in points.items()}
    base = raw[["date", "block", "actual", "fold", "period"]].assign(
        state=np.select(
            [around["cap_mixture"] >= cap, raw["hi50"] >= cap, raw["hi90"] >= cap], [3, 2, 1], 0
        ),
        below=raw["actual"] < cap,
    )
    parts = []
    for coverage in COVERAGES:
        tag = round(coverage * 100)
        ends = (raw[f"lo{tag}"].to_numpy(), raw[f"hi{tag}"].to_numpy())
        variants: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object] | None]] = {
            "two-stage, raw": (*ends, None),
            "two-stage, corrected": (*ends, {}),
            "two-stage, corrected by cap state": (*ends, {"group": "state"}),
            "two-stage, taught by below-cap blocks": (*ends, {"only": "below"}),
            "two-stage, by state and below-cap": (*ends, {"group": "state", "only": "below"}),
            "cap-aware median, corrected": (around["cap_mixture"], around["cap_mixture"], {}),
            "LightGBM, corrected": (around["lightgbm"], around["lightgbm"], {}),
        }
        for label, (low, high, options) in variants.items():
            table = base.assign(lo=low, hi=high)
            if options is not None:
                table = conformal_adjust(table, coverage, WINDOW_DAYS, cap=cap, **options)  # type: ignore[arg-type]
                table = table.assign(lo=table["lo_adj"], hi=table["hi_adj"])
            parts.append(table.assign(variant=label, coverage=coverage))
    return pd.concat(parts, ignore_index=True)


def report(inputs: Inputs) -> None:
    frame = scored_variants(inputs)
    # the first days of the first fold have no earlier days to learn the margin from
    frame = frame.dropna(subset=["lo", "hi"])
    rows = []
    groups = {"test": ("period", "test"), "validation": ("period", "validation")}
    groups |= {"fold 4": ("fold", 4), "fold 5": ("fold", 5)}
    for label, (column, value) in groups.items():
        part = frame[frame[column] == value]
        for (coverage, variant), table in part.groupby(["coverage", "variant"], sort=False):
            level = float(coverage)
            scores = interval_scores(table, level)
            scores["cov_cap"] = interval_scores(table[~table["below"]], level)["coverage"]
            scores["cov_below"] = interval_scores(table[table["below"]], level)["coverage"]
            for code, name in STATES.items():
                subset = table[table["state"] == code]
                scores[f"cov {name}"] = (
                    interval_scores(subset, level)["coverage"] if len(subset) else float("nan")
                )
            rows.append({"group": label, "nominal": coverage, "variant": variant, **scores})
    result = pd.DataFrame(rows)
    result.to_csv(inputs.out / "interval_scores.csv", index=False)
    shown = result.assign(variant=result["variant"].str[:38])
    pd.set_option("display.width", 220)
    for (group, nominal), table in shown.groupby(["group", "nominal"], sort=False):
        print()
        print(f"== {group}, nominal {nominal:.0%} ==")
        print(table.drop(columns=["group", "nominal"]).round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "report"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    inputs = Inputs(args.smoke)
    if args.stage == "fit":
        fit(inputs, args.redo)
    else:
        report(inputs)


if __name__ == "__main__":
    main()
