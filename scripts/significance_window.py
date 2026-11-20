"""Significance tests for the day-ahead comparison on the 2025-26 window.

    python scripts/significance_window.py

Reads the prediction files of the comparison (private, under .project-local/results_window) and
tests, on the daily mean absolute error of the 60 test days:

* Diebold-Mariano (one-sided, small-sample correction): does model A beat the reference?
  References: the best naive forecast, and both Phase 1 models, as in the success criterion.
* the 90 percent model confidence set over all models.

The validation days were used for tuning, so only the test result is used for a claim. The
validation and all-days results are printed for comparison. Results go to the same folder.
"""

from pathlib import Path

import pandas as pd

from rtm_forecast.config import load_window_experiment_params
from rtm_forecast.metrics import daily_mae
from rtm_forecast.significance import bootstrap_difference, diebold_mariano, model_confidence_set

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".project-local" / "results_window"
REFERENCES = ["naive_yesterday", "phase1_xgboost", "phase1_random_forest", "lightgbm", "catboost"]
# forecasts that are baselines, not candidates: they are only used as references
BASELINES = {"naive_yesterday", "naive_lastweek", "dam_price"}


def losses_for(frame: pd.DataFrame, period: str | None) -> pd.DataFrame:
    part = frame if period is None else frame[frame["period"] == period]
    return daily_mae(part).dropna(axis=1)


def dm_table(losses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reference in REFERENCES:
        if reference not in losses.columns:
            continue
        for model in losses.columns:
            if model == reference:
                continue
            a, b = losses[model].to_numpy(), losses[reference].to_numpy()
            statistic, p_value = diebold_mariano(a, b)
            low, high = bootstrap_difference(a, b)
            rows.append(
                {
                    "model": model,
                    "reference": reference,
                    "mean_daily_MAE_diff": float((a - b).mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "DM_statistic": statistic,
                    "p_model_better": p_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    exp = load_window_experiment_params(ROOT / "params.yaml")
    files = sorted(OUT.glob("predictions_*.parquet"))
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    frame = frame[~frame["model"].str.endswith("_check")]
    reference = {"rtm_price_lag1d": "naive_yesterday", "rtm_price_lag7d": "naive_lastweek"}
    assert reference[exp.naive_reference] == "naive_yesterday"
    dm_all, mcs_all = [], []
    for label, period in [("test", "test"), ("validation", "validation"), ("all days", None)]:
        losses = losses_for(frame, period)
        print(f"\n===== {label}: {len(losses)} days, {losses.shape[1]} models =====")
        table = dm_table(losses)
        table.insert(0, "group", label)
        dm_all.append(table)
        shown = table[~table["model"].isin(BASELINES)]
        print("\nDiebold-Mariano (p = chance of such a gap if the model were no better):")
        print(shown.drop(columns="group").round(4).to_string(index=False))
        mcs = model_confidence_set(losses, alpha=0.10)
        mcs.insert(0, "group", label)
        mcs_all.append(mcs.reset_index(names="model"))
        print("\n90 percent model confidence set:")
        print(mcs.drop(columns="group").round(4).to_string())
    pd.concat(dm_all).to_csv(OUT / "significance_dm.csv", index=False)
    pd.concat(mcs_all).to_csv(OUT / "significance_mcs.csv", index=False)


if __name__ == "__main__":
    main()
