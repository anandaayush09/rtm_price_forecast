"""Run the daily loop over days that came after the comparison and score it.

    python scripts/new_days_check.py --first-day 2026-11-01 --last-day 2026-11-22

For each day D in turn: fit the final model on the days before D, forecast D with the margins
from the calibration table, and then add D to the table once its prices are known, as the
commands ``fit``, ``forecast`` and ``settle`` do. The window table must hold the DAM results,
the demand and the published RTM prices of every day from ``--first-day`` to ``--last-day``
(set ``window.end`` in params.yaml and run ``rtm-forecast clean-window``). The calibration table
of the production folder is read and not changed.

The scores are the ones of the comparison: mean absolute error against the same block yesterday,
the share of published prices inside the 50 and 90 percent intervals, and the F1 of calling the
cap. Results go to .project-local/results_window/checks and are not committed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rtm_forecast.cli import _spec, _window_features
from rtm_forecast.production import fit_production, forecast_day, update_calibration

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".project-local" / "results_window" / "checks"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-day", required=True)
    parser.add_argument("--last-day", required=True)
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    parser.add_argument(
        "--calibration", type=Path, help="default: the table of the production folder"
    )
    args = parser.parse_args()
    first, last = pd.Timestamp(args.first_day), pd.Timestamp(args.last_day)
    production, spec = _spec(args.params)
    features = _window_features(args.params)
    calibration = pd.read_parquet(args.calibration or production.calibration_path)
    calibration = calibration[calibration["date"] < first].reset_index(drop=True)
    days = [d for d in sorted(features["date"].unique()) if first <= d <= last]
    if not days:
        raise SystemExit("the window table has no day in that range")
    parts = []
    for day in map(pd.Timestamp, days):
        rows = features[features["date"] == day]
        model = fit_production(features, spec, before=day)
        forecast = forecast_day(model, rows, calibration)
        prices = rows[["date", "block", "rtm_price"]]
        if prices["rtm_price"].notna().any():
            calibration = update_calibration(calibration, forecast, prices)
        table = forecast.merge(
            rows[["date", "block", "rtm_price", "rtm_price_lag1d"]], on=["date", "block"]
        )
        parts.append(table)
        print(f"{day:%Y-%m-%d} done (fitted on {model.train_days} days)", flush=True)
    result = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT / "new_days_forecasts.parquet", index=False)
    scored = result[result["rtm_price"].notna() & result["rtm_price_lag1d"].notna()]
    if scored.empty:
        print("no published prices in the range: nothing to score")
        return
    actual = scored["rtm_price"].to_numpy()
    cap = spec.cap
    at_cap, called = actual >= cap, scored["point"].to_numpy() >= cap
    tp = float((at_cap & called).sum())
    f1 = 2 * tp / max(1.0, float(at_cap.sum() + called.sum()))
    print(f"\n{len(scored)} blocks on {scored['date'].nunique()} days")
    print(f"MAE of the model: {np.abs(actual - scored['point']).mean():.1f}")
    print(
        f"MAE of the same block yesterday: {np.abs(actual - scored['rtm_price_lag1d']).mean():.1f}"
    )
    for tag in (50, 90):
        inside = (actual >= scored[f"lo{tag}"]) & (actual <= scored[f"hi{tag}"])
        print(f"share inside the {tag} percent interval: {inside.mean():.3f}")
    print(f"share of blocks at the cap: {at_cap.mean():.3f}, F1 of calling the cap: {f1:.3f}")


if __name__ == "__main__":
    main()
