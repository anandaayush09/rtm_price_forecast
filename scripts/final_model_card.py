"""Write the private model card of the chosen day-ahead model from the saved result files.

    python scripts/final_model_card.py

No model is fitted here. Every number is read from the csv files that the comparison, the
significance script and the interval script wrote under .project-local/results_window, so the
card cannot drift from them. Output: .project-local/FINAL_MODEL.md.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / ".project-local" / "results_window"
FINAL = "cap_mixture"
SHOWN = [FINAL, "cap_hard", "lightgbm", "catboost", "naive_yesterday", "phase1_xgboost"]
SHOWN += ["phase1_random_forest"]
INTERVAL_VARIANTS = ["two-stage, raw", "two-stage, corrected by cap state"]


def table(frame: pd.DataFrame, digits: int = 3) -> str:
    """A markdown table of the frame."""
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    line = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = [
        "| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in r) + " |"
        for r in frame.itertuples(index=False)
    ]
    return "\n".join([head, line, *rows])


def point_scores() -> str:
    scores = pd.read_csv(RESULTS / "scores.csv")
    parts = []
    for group in ("test", "validation", "fold 4", "fold 5"):
        part = scores[(scores["group"] == group) & scores["model"].isin(SHOWN)]
        part = part.set_index("model").loc[SHOWN].reset_index()
        part = part[["model", "MAE", "RMSE", "rMAE", "cap_f1"]].round(3)
        parts.append(f"### {group}\n\n{table(part, 3)}")
    return "\n\n".join(parts)


def tests() -> str:
    dm = pd.read_csv(RESULTS / "significance_dm.csv")
    dm = dm[dm["group"] == "test"]
    rows = []
    for model in (
        "naive_yesterday",
        "phase1_xgboost",
        "phase1_random_forest",
        "lightgbm",
        "catboost",
    ):
        found = dm[(dm["model"] == FINAL) & (dm["reference"] == model)]
        if found.empty:
            other = dm[(dm["model"] == model) & (dm["reference"] == FINAL)]
            if other.empty:
                continue
            row = other.iloc[0]
            diff, low, high = -row["mean_daily_MAE_diff"], -row["ci_high"], -row["ci_low"]
            p_value = float(1.0 - row["p_model_better"])
        else:
            row = found.iloc[0]
            diff, low, high = row["mean_daily_MAE_diff"], row["ci_low"], row["ci_high"]
            p_value = float(row["p_model_better"])
        rows.append(
            {
                "reference": model,
                "daily MAE difference": round(float(diff), 1),
                "95% interval low": round(float(low), 1),
                "95% interval high": round(float(high), 1),
                "one-sided p (final model better)": (
                    "<0.0001" if p_value < 0.0001 else f"{p_value:.4f}"
                ),
            }
        )
    mcs = pd.read_csv(RESULTS / "significance_mcs.csv")
    mcs = mcs[mcs["group"] == "test"]
    in_set = mcs[mcs["in_set"]]["model"].tolist()
    return (
        table(pd.DataFrame(rows))
        + f"\n\nThe 90 percent model confidence set on the test days holds: {', '.join(in_set)}."
    )


def intervals() -> str:
    scores = pd.read_csv(RESULTS / "interval_scores.csv")
    scores = scores[scores["variant"].isin(INTERVAL_VARIANTS)]
    parts = []
    for group in ("test", "validation"):
        part = scores[scores["group"] == group]
        for nominal in (0.9, 0.5):
            sub = part[part["nominal"] == nominal]
            cols = ["variant", "coverage", "width", "interval_score", "cov_below"]
            cols += ["cov cap unlikely"]
            parts.append(f"### {group}, nominal {nominal:.0%}\n\n{table(sub[cols].round(3), 3)}")
    return "\n\n".join(parts)


TEXT = """# Final day-ahead model (private card)

Written by scripts/final_model_card.py from the saved result files. Nothing here is fitted.

## What the model is

Task: forecast the 96 RTM blocks of day D once the DAM results of day D are known, using the
published prices of all earlier days. Refit every day on all earlier days.

* Inputs: the same input columns as the plain tree models (all groups, raw price scale, demand
  as one definition; chosen on the validation days).
* Stage 1: a LightGBM classifier for "the block settles at the cap" (price >= 10000).
* Stage 2: LightGBM quantile regressions on the blocks below the cap, at levels 0.5, 0.75, 0.9
  (point forecast) and 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975 (intervals).
* Point forecast (`cap_mixture`): the median of the two-part distribution (the cap when the cap
  probability is at least one half, otherwise a quantile of the below-cap part). It was chosen
  over the hard switch on the validation days only.
* Intervals: the quantiles of the same distribution, then a conformal margin learned on the
  60 days before each day, separately for four groups of how much the model expects the cap
  (from quantities known when the day is forecast).
* Settings: the tuned LightGBM settings of the plain model are reused; nothing is tuned on the
  test days. Seed 0.

## Results

Scores are read from scores.csv. MAE and RMSE are in Rs/MWh; rMAE is MAE divided by the MAE of
the same block yesterday; cap_f1 is the F1 of calling a block at the cap.

{points}

## Tests against the references (60 test days, daily mean absolute error)

{tests}

## Prediction intervals

Rows are read from interval_scores.csv. coverage is the share of published prices inside the
interval; cov_below is that share among prices below the cap; cov cap unlikely is that share
among blocks where the model expects no cap. interval_score: lower is better.

{intervals}

## Checks done

* Refitting the model on the last three test days reproduces the saved point forecast and the
  four raw interval ends exactly (same seed, deterministic training).
* Tests: the forecast does not read the prices of the days it forecasts; the conformal margin of
  a day uses only earlier days; the mixture quantile and the interval scores are tested.

## Limits to state in the report

* The success criterion was fixed before tuning, but this model was designed after the test days
  showed the plain models did not meet it, so the test period is not a clean one-shot for it.
* P-values are not adjusted for the number of comparisons. The confidence set is large because
  60 days and many models give it little power.
* The cap classifier has not been checked for calibration and has no recency weighting, although
  the share of cap blocks rises through the window. Fold 4 (the regime change) is the weakest
  fold for the intervals.
* RMSE is worse than the plain tree models because a wrong all-or-nothing cap call costs a lot.
* The 50 percent intervals cover too much because of the atom at the cap.
* The intraday task has no cap-aware version yet; its best models are the plain LightGBM and
  CatBoost with the day-ahead settings.
* The data are private (IEX terms of use unchecked); results are not for public release as
  price statistics until that is settled.

## Reproduce

    python scripts/benchmark_window.py final      # plain models (once)
    python scripts/benchmark_cap_aware.py         # the point forecasts (about 8 minutes)
    python scripts/intervals_window.py fit        # raw interval ends (about 16 minutes)
    python scripts/benchmark_window.py report
    python scripts/significance_window.py
    python scripts/intervals_window.py report
    python scripts/final_model_card.py
"""


def main() -> None:
    text = TEXT.replace("{points}", point_scores())
    text = text.replace("{tests}", tests()).replace("{intervals}", intervals())
    path = ROOT / ".project-local" / "FINAL_MODEL.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
