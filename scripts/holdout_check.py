"""Score the final model on the period fixed in docs/holdout_protocol.md.

    python scripts/holdout_check.py --params <params file> \
        --first-day 2025-03-01 --last-day 2025-10-31

For each day in turn the model is fitted on the days before it and the day is forecast, as the
commands ``fit`` and ``forecast`` do; the day is then added to the calibration table, as ``settle``
does. The table starts empty, so the first days have no corrected intervals (they need
``MIN_DAYS`` days of history): their point forecasts count, their intervals are not scored.

The scores follow the protocol: the mean absolute error of the model, of the same block
yesterday, of the same block a week earlier and of the DAM price on the same blocks, the
Diebold-Mariano test against yesterday, the months in which the model beats yesterday, the share
of published prices inside the intervals and the F1 of calling the cap. Nothing is tuned. Results
go to .project-local/results_window/holdout and are not committed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rtm_forecast.cli import _spec, _window_features
from rtm_forecast.production import (
    CALIBRATION_COLUMNS,
    MIN_DAYS,
    RAW_COLUMNS,
    ProductionModel,
    fit_production,
    forecast_day,
    update_calibration,
)
from rtm_forecast.significance import diebold_mariano

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".project-local" / "results_window" / "holdout"


def _forecast(
    model: ProductionModel, rows: pd.DataFrame, calibration: pd.DataFrame
) -> pd.DataFrame:
    if calibration["date"].nunique() >= MIN_DAYS:
        return forecast_day(model, rows, calibration)
    raw = model.raw(rows)
    table = rows[["date", "block"]].reset_index(drop=True).join(raw.reset_index(drop=True))
    for name in ("lo50", "hi50", "lo90", "hi90"):
        table[f"raw_{name}"] = table[name]
        table[name] = np.nan
    return table


def run(params: Path, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    _, spec = _spec(params)
    features = _window_features(params)
    calibration = pd.DataFrame(columns=CALIBRATION_COLUMNS).astype(
        {
            "date": "datetime64[ns]",
            "block": int,
            "actual": float,
            **dict.fromkeys(RAW_COLUMNS, float),
        }
    )
    parts = []
    for day in map(pd.Timestamp, sorted(features["date"].unique())):
        if not first <= day <= last:
            continue
        rows = features[features["date"] == day]
        model = fit_production(features, spec, before=day)
        forecast = _forecast(model, rows, calibration)
        prices = rows[["date", "block", "rtm_price"]]
        calibration = update_calibration(calibration, forecast, prices)
        keep = ["date", "block", "rtm_price", "rtm_price_lag1d", "rtm_price_lag7d", "dam_price"]
        parts.append(forecast.merge(rows[keep], on=["date", "block"]))
        print(f"{day:%Y-%m-%d} done (fitted on {model.train_days} days)", flush=True)
    return pd.DataFrame(pd.concat(parts, ignore_index=True))


def score(result: pd.DataFrame, cap: float) -> None:
    common = result.dropna(subset=["rtm_price", "rtm_price_lag1d", "rtm_price_lag7d", "dam_price"])
    actual = common["rtm_price"]
    forecasts = {
        "final model": common["point"],
        "same block yesterday": common["rtm_price_lag1d"],
        "same block last week": common["rtm_price_lag7d"],
        "DAM price": common["dam_price"],
    }
    days = common["date"].nunique()
    print(f"\n{len(common)} blocks on {days} days (blocks with every forecast and a price)")
    errors = {name: (actual - f).abs() for name, f in forecasts.items()}
    daily = pd.DataFrame({name: e.groupby(common["date"]).mean() for name, e in errors.items()})
    for name, e in errors.items():
        print(f"MAE {name}: {e.mean():.1f}")
    for other in list(forecasts)[1:]:
        stat, p = diebold_mariano(daily["final model"].to_numpy(), daily[other].to_numpy())
        print(f"Diebold-Mariano, final model against {other}: statistic {stat:.2f}, p {p:.4f}")
    month = daily.groupby(pd.DatetimeIndex(daily.index).to_period("M")).mean()
    win = month["final model"] < month["same block yesterday"]
    print(f"\nmonths where the final model beats yesterday: {int(win.sum())} of {len(win)}")
    cap_share = (actual >= cap).groupby(common["date"].dt.to_period("M")).mean()
    table = month.assign(cap_share=cap_share).round(3)
    print(table.to_string())
    at_cap, called = (actual >= cap).to_numpy(), (common["point"] >= cap).to_numpy()
    tp = float((at_cap & called).sum())
    print(
        f"\nshare of blocks at the cap {at_cap.mean():.3f}; F1 of calling the cap "
        f"{2 * tp / max(1.0, float(at_cap.sum() + called.sum())):.3f}"
    )
    covered = common.dropna(subset=["lo50", "hi50", "lo90", "hi90"])
    print(f"intervals scored on {covered['date'].nunique()} days")
    for tag in (50, 90):
        inside = (covered["rtm_price"] >= covered[f"lo{tag}"]) & (
            covered["rtm_price"] <= covered[f"hi{tag}"]
        )
        print(f"share inside the {tag} percent interval: {inside.mean():.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / "daily_mae.csv")
    table.to_csv(OUT / "monthly_mae.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--first-day", required=True)
    parser.add_argument("--last-day", required=True)
    args = parser.parse_args()
    first, last = pd.Timestamp(args.first_day), pd.Timestamp(args.last_day)
    _, spec = _spec(args.params)
    result = run(args.params, first, last)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT / "holdout_forecasts.parquet", index=False)
    score(result, spec.cap)


if __name__ == "__main__":
    main()
