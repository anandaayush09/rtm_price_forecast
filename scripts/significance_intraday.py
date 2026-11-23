"""Significance tests for the intraday comparison on the 2025-26 window.

    python scripts/significance_intraday.py

Reads the prediction files of the intraday comparison (private, under
.project-local/results_intraday) and, for each lead, tests on the daily mean absolute error of
the 60 test days:

* Diebold-Mariano (one-sided, small-sample correction): does model A beat the reference?
  References: the last known price (the natural intraday baseline) and yesterday's price (the
  day-ahead baseline).
* the 90 percent model confidence set over all models.

Only blocks that every model has are used, as in the report. The tuned settings of the day-ahead
task were reused, so no setting was chosen on the test days. Results go to the same folder.
"""

from pathlib import Path

import pandas as pd

from rtm_forecast.metrics import daily_mae
from rtm_forecast.significance import bootstrap_difference, diebold_mariano, model_confidence_set

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".project-local" / "results_intraday"
REFERENCES = ["naive_last", "naive_yesterday"]
BASELINES = {"naive_last", "naive_last_dam_adjusted", "naive_yesterday", "dam_price"}


def collect() -> pd.DataFrame:
    files = sorted(OUT.glob("lead*/predictions_*.parquet"))
    files += [f for f in sorted(OUT.glob("intraday_*.parquet")) if "_check" not in f.name]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def dm_table(losses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reference in REFERENCES:
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
    result = collect()
    dm_all, mcs_all = [], []
    for lead, by_lead in result.groupby("lead"):
        part = by_lead[by_lead["period"] == "test"]
        seen = part.groupby(["date", "block"])["model"].transform("nunique")
        part = part[seen == part["model"].nunique()]
        losses = daily_mae(part).dropna(axis=1)
        print(f"\n===== lead {lead}, test: {len(losses)} days, {losses.shape[1]} models =====")
        table = dm_table(losses)
        table.insert(0, "lead", lead)
        dm_all.append(table)
        shown = table[~table["model"].isin(BASELINES)]
        print("\nDiebold-Mariano (p = chance of such a gap if the model were no better):")
        print(shown.drop(columns="lead").round(4).to_string(index=False))
        mcs = model_confidence_set(losses, alpha=0.10)
        mcs.insert(0, "lead", lead)
        mcs_all.append(mcs.reset_index(names="model"))
        print("\n90 percent model confidence set:")
        print(mcs.drop(columns="lead").round(4).to_string())
    pd.concat(dm_all).to_csv(OUT / "significance_dm.csv", index=False)
    pd.concat(mcs_all).to_csv(OUT / "significance_mcs.csv", index=False)


if __name__ == "__main__":
    main()
