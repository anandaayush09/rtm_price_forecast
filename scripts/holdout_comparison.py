"""Run the plain tuned models and the 2021 models on the days scored by holdout_check.py.

    python scripts/holdout_comparison.py --params <params file> \\
        --first-day 2025-03-01 --last-day 2025-10-31

The settings are the ones chosen on the 2025-26 window (inputs, price transform and tuned
LightGBM and CatBoost settings from the private results folder); nothing is tuned here. Each
model is refitted every day on the days before it, as in the benchmark. The forecasts of the final
model come from ``holdout_check.py``. Results go to .project-local/results_window/holdout.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtm_forecast import models
from rtm_forecast.backtest import Predictor, rolling_origin
from rtm_forecast.calendar import load_holidays
from rtm_forecast.cli import _spec, _window_features
from rtm_forecast.config import load_feature_params, load_window_params
from rtm_forecast.reference import build_features as phase1_features
from rtm_forecast.reference import prepare_frame
from rtm_forecast.significance import diebold_mariano, model_confidence_set

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / ".project-local" / "results_window"
OUT = RESULTS / "holdout"
ORDER = ["lightgbm", "phase1_xgboost", "phase1_random_forest", "catboost"]


def phase1_inputs(
    window: pd.DataFrame, holidays: list[pd.Timestamp], frame: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """The 2021 model inputs rebuilt from the window table (as in benchmark_window.py)."""
    raw = pd.DataFrame(
        {
            "Date": window["date"].dt.strftime("%d-%m-%Y"),
            "TimeblockId": window["block"].astype(str),
            "DAM_Price": window["dam_price"],
            "RTM_Price": window["rtm_price"],
            "DAM_Purchase_Bid": window["dam_purchase_bid"],
            "DAM_Selling_Bid": window["dam_selling_bid"],
            "RTM_Purchase_Bid": window["rtm_purchase_bid"],
            "RTM_Selling_Bid": window["rtm_selling_bid"],
            "Demand": window["demand"],
        }
    )
    old = phase1_features(prepare_frame(raw, dates="day_first", holidays=holidays))
    columns = [c for c in old.columns if c != "RTM_Price_SD"]
    if len(old) != len(frame):
        raise ValueError("the 2021 inputs and the window inputs do not line up")
    base = frame[["date", "block", "timestamp", "rtm_price", "target_observed"]]
    return pd.concat([base, old[columns]], axis=1), columns


def run_models(params: Path, days: list[pd.Timestamp], only: list[str]) -> None:
    _, spec = _spec(params)
    features = _window_features(params)
    feature_params = load_feature_params(params)
    window = pd.read_parquet(load_window_params(params).clean_path)
    holidays = [pd.Timestamp(d) for d in sorted(load_holidays(feature_params.holidays_path))]
    best: dict[str, Any] = json.loads((RESULTS / "best_params.json").read_text())
    columns = list(spec.columns)
    p1_frame, p1_columns = phase1_inputs(window, holidays, features)
    runs: dict[str, tuple[pd.DataFrame, Predictor]] = {
        "lightgbm": (
            features,
            models.lightgbm_predictor(columns, None, best["lightgbm"]["params"], spec.seed),
        ),
        "catboost": (
            features,
            models.catboost_predictor(columns, None, best["catboost"]["params"], spec.seed),
        ),
        "phase1_xgboost": (p1_frame, models.phase1_xgboost_predictor(p1_columns)),
        "phase1_random_forest": (
            p1_frame,
            models.phase1_random_forest_predictor(p1_columns, spec.seed),
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name in only:
        target = OUT / f"predictions_{name}.parquet"
        if target.exists():
            print(f"{name}: already done, skipped", flush=True)
            continue
        started = time.time()
        data, predictor = runs[name]
        rolling_origin(data, days, predictor, name).to_parquet(target, index=False)
        print(f"{name}: {time.time() - started:.0f}s", flush=True)


def score() -> None:
    final = pd.read_parquet(OUT / "holdout_forecasts.parquet")
    wide = final[["date", "block", "rtm_price", "rtm_price_lag1d", "dam_price", "point"]].rename(
        columns={"point": "final model", "rtm_price_lag1d": "same block yesterday"}
    )
    names = ["final model", "same block yesterday", "DAM price"]
    wide = wide.rename(columns={"dam_price": "DAM price"})
    for name in ORDER:
        path = OUT / f"predictions_{name}.parquet"
        if path.exists():
            part = pd.read_parquet(path)[["date", "block", "prediction"]]
            wide = wide.merge(part.rename(columns={"prediction": name}), on=["date", "block"])
            names.append(name)
    common = wide.dropna(subset=["rtm_price", *names])
    print(
        f"{len(common)} blocks on {common['date'].nunique()} days with every forecast and a price"
    )
    errors = pd.DataFrame({n: (common["rtm_price"] - common[n]).abs() for n in names})
    daily = errors.groupby(common["date"]).mean()
    mean = errors.mean()
    yesterday = float(mean["same block yesterday"])
    rows = []
    for name in names:
        stat, p = (
            (np.nan, np.nan)
            if name == "final model"
            else diebold_mariano(daily["final model"].to_numpy(), daily[name].to_numpy())
        )
        rows.append(
            {
                "model": name,
                "mae": round(float(mean[name]), 1),
                "rmae": round(float(mean[name]) / yesterday, 3),
                "dm_stat_final_vs": round(stat, 2),
                "dm_p_final_vs": round(p, 4),
            }
        )
    table = pd.DataFrame(rows).sort_values("mae")
    print(table.to_string(index=False))
    months = daily.groupby(pd.DatetimeIndex(daily.index).to_period("M")).mean()
    best_other = months.drop(columns="final model").min(axis=1)
    print(
        f"\nmonths where the final model has the lowest error: "
        f"{int((months['final model'] < best_other).sum())} of {len(months)}"
    )
    mcs = model_confidence_set(daily[names], alpha=0.10, block=5, draws=5000, seed=0)
    print("\nmodel confidence set (90 percent):")
    print(mcs.round(4).to_string())
    table.to_csv(OUT / "comparison_scores.csv", index=False)
    months.round(1).to_csv(OUT / "comparison_monthly_mae.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--first-day", required=True)
    parser.add_argument("--last-day", required=True)
    parser.add_argument("--models", nargs="*", default=ORDER)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if not args.score_only:
        first, last = pd.Timestamp(args.first_day), pd.Timestamp(args.last_day)
        all_days = _window_features(args.params)["date"].unique()
        days = [pd.Timestamp(d) for d in sorted(all_days) if first <= pd.Timestamp(d) <= last]
        run_models(args.params, days, args.models)
    score()


if __name__ == "__main__":
    main()
