"""What the forecast app shows, computed from the files the command line writes.

The forecast files (``forecast_YYYY-MM-DD.csv``) and the calibration table come from
``rtm-forecast forecast`` and ``rtm-forecast settle``. Nothing here fits a model. The record of
recent days is computed the way a forecast is: the interval of a settled day is corrected with a
margin from the days before it, using only what was known then.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from rtm_forecast.intervals import cap_state, conformal_adjust
from rtm_forecast.production import COVERAGES, MIN_DAYS, WINDOW_DAYS

BLOCK_MINUTES = 15
STATE_NAMES = {3: "cap called", 2: "cap likely", 1: "cap possible", 0: "cap unlikely"}
_FILE = re.compile(r"forecast_(\d{4}-\d{2}-\d{2})\.csv$")


def block_start(date: pd.Series, block: pd.Series) -> pd.Series:
    """The time a block starts (block 1 starts at midnight)."""
    return pd.DatetimeIndex(date).to_series(index=date.index) + pd.to_timedelta(
        (block - 1) * BLOCK_MINUTES, unit="min"
    )


def forecast_days(folder: Path) -> list[pd.Timestamp]:
    """The days that have a forecast file, latest first."""
    if not folder.is_dir():
        return []
    found = [m.group(1) for f in folder.iterdir() if (m := _FILE.search(f.name))]
    return sorted((pd.Timestamp(d) for d in found), reverse=True)


def read_forecast(folder: Path, day: pd.Timestamp) -> pd.DataFrame:
    table = pd.read_csv(folder / f"forecast_{day:%Y-%m-%d}.csv", parse_dates=["date"])
    return table.assign(start=block_start(table["date"], table["block"]))


def day_summary(forecast: pd.DataFrame, cap: float) -> dict[str, float]:
    """Numbers shown above the chart: blocks where the cap is expected, mean and highest point."""
    called = forecast["point"] >= cap
    top = forecast.iloc[int(forecast["point"].to_numpy().argmax())]
    return {
        "blocks_at_cap": float(called.sum()),
        "blocks_possible": float((forecast["hi90"] >= cap).sum()),
        "mean_point": float(forecast["point"].mean()),
        "highest_point": float(top["point"]),
        "highest_block": float(top["block"]),
    }


def corrected_history(
    calibration: pd.DataFrame,
    cap: float,
    window_days: int = WINDOW_DAYS,
    min_days: int = MIN_DAYS,
) -> pd.DataFrame:
    """The forecasts of the settled days as they were issued: the point forecast and the
    corrected intervals (``lo50`` ... ``hi90``), with the published price in ``actual``.

    Days with fewer than ``min_days`` earlier days in the table have no corrected interval and
    are left out.
    """
    table = calibration.copy()
    table["state"] = cap_state(
        table["point"].to_numpy(), table["hi50"].to_numpy(), table["hi90"].to_numpy(), cap
    )
    out = table[["date", "block", "actual", "point", "state"]].copy()
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
        ends = adjusted.set_index(["date", "block"])[["lo_adj", "hi_adj"]]
        keys = pd.MultiIndex.from_frame(out[["date", "block"]])
        out[f"lo{tag}"] = ends["lo_adj"].reindex(keys).to_numpy()
        out[f"hi{tag}"] = ends["hi_adj"].reindex(keys).to_numpy()
    return out.dropna(subset=["lo90"]).sort_values(["date", "block"], ignore_index=True)


def daily_record(history: pd.DataFrame) -> pd.DataFrame:
    """One row per settled day: mean absolute error of the forecast and of yesterday's price
    (the same block a day earlier), and the share of published prices inside each interval.

    Both errors use the same blocks: those with a published price whose day before is also in
    the table (NaN for a day that has none).
    """
    table = history.copy()
    earlier = table[["date", "block", "actual"]].assign(
        date=lambda t: t["date"] + pd.Timedelta(days=1)
    )
    table = table.merge(
        earlier.rename(columns={"actual": "yesterday"}), on=["date", "block"], how="left"
    )
    table = table[table["actual"].notna()]
    rows = []
    for day, part in table.groupby("date"):
        both = part.dropna(subset=["yesterday"])
        row: dict[str, object] = {
            "date": day,
            "blocks": len(part),
            "model_mae": float((both["point"] - both["actual"]).abs().mean())
            if len(both)
            else float("nan"),
            "yesterday_mae": float((both["yesterday"] - both["actual"]).abs().mean())
            if len(both)
            else float("nan"),
        }
        for coverage in COVERAGES:
            tag = round(coverage * 100)
            inside = (part["actual"] >= part[f"lo{tag}"]) & (part["actual"] <= part[f"hi{tag}"])
            row[f"inside{tag}"] = float(inside.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def record_totals(record: pd.DataFrame) -> dict[str, float]:
    """Averages over the days in ``record``; the errors only over days that have both."""
    both = record.dropna(subset=["yesterday_mae"])
    return {
        "days": float(len(record)),
        "model_mae": float(both["model_mae"].mean()) if len(both) else float("nan"),
        "yesterday_mae": float(both["yesterday_mae"].mean()) if len(both) else float("nan"),
        "inside50": float(np.average(record["inside50"], weights=record["blocks"])),
        "inside90": float(np.average(record["inside90"], weights=record["blocks"])),
    }
