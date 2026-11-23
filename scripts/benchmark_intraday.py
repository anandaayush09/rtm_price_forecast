"""Intraday comparison on the 2025-26 window: forecast a block a few blocks after the latest RTM.

    python scripts/benchmark_intraday.py final --leads 1 2 4 8   # models, one folder per lead
    python scripts/benchmark_intraday.py report                  # scores per lead

The day-ahead task forecasts all 96 blocks of day D with RTM data up to the end of day D - 1.
This task keeps the same folds and the same DAM inputs, and adds the RTM blocks up to ``lead``
blocks before the block that is forecast. The tuned settings of the day-ahead task are reused
(no new tuning), so the intraday scores are a floor for what tuning could reach. Add --smoke
for a quick run on a few days that writes to a separate folder.

Results are private files under .project-local/results_intraday and are not committed.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_window import ROOT, Inputs, linear_columns, read_setup

from rtm_forecast import models
from rtm_forecast.backtest import Predictor, rolling_origin
from rtm_forecast.intraday_features import INTRADAY_GROUPS, build_intraday_features
from rtm_forecast.metrics import cap_scores, summarise

LEADS = [1, 2, 4, 8]
MODELS = [
    "naive_last",
    "naive_last_dam_adjusted",
    "naive_yesterday",
    "dam_price",
    "lear",
    "lightgbm",
    "catboost",
    "cap_hard",
    "cap_mixture",
]
REFERENCE = "naive_last"


def out_dir(inputs: Inputs) -> Path:
    name = "results_intraday_smoke" if inputs.smoke else "results_intraday"
    return ROOT / ".project-local" / name


def final(inputs: Inputs, leads: list[int], only: list[str], redo: bool) -> None:
    setup = read_setup(inputs)
    best = json.loads((inputs.out / "best_params.json").read_text(encoding="utf-8"))
    seed = inputs.experiment.seed
    transform = setup["transform"]
    days = [d for fold in inputs.folds for d in fold]
    fold_of = {day: k + 1 for k, fold in enumerate(inputs.folds) for day in fold}
    extra = [c for names in INTRADAY_GROUPS.values() for c in names]
    columns = [*setup["columns"], *extra]
    one_definition = setup["chosen"]["demand"] == "one_definition"
    for lead in leads:
        folder = out_dir(inputs) / f"lead{lead:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        frame = build_intraday_features(
            inputs.window,
            inputs.feature_params,
            inputs.holiday_dates,
            lead,
            inputs.experiment.price_cap,
            one_definition,
        )
        runs: dict[str, Predictor] = {
            "naive_last": models.column_predictor("rtm_last_price"),
            "naive_last_dam_adjusted": models.column_predictor("rtm_last_dam_adjusted"),
            "naive_yesterday": models.column_predictor("rtm_price_lag1d"),
            "dam_price": models.column_predictor("dam_price"),
            "lear": models.lear_predictor(
                linear_columns(columns), transform, best["lear"]["params"]["alpha"], seed, "lars"
            ),
            "lightgbm": models.lightgbm_predictor(
                columns, transform, best["lightgbm"]["params"], seed
            ),
            "catboost": models.catboost_predictor(
                columns, transform, best["catboost"]["params"], seed
            ),
            # the two cap-aware forecasts share their fits, so the second run costs almost nothing
            **models.cap_aware_predictors(
                columns,
                transform,
                best["lightgbm"]["params"],
                seed,
                inputs.experiment.price_cap,
            ),
        }
        for name in only:
            target = folder / f"predictions_{name}.parquet"
            marker = folder / f".running_{name}"
            if (target.exists() or marker.exists()) and not redo:
                print(f"lead {lead} {name}: already done or running, skipped", flush=True)
                continue
            marker.write_text("running", encoding="utf-8")
            started = time.time()
            result = rolling_origin(frame, days, runs[name], name)
            result["lead"] = lead
            result["fold"] = result["date"].map(fold_of)
            result["period"] = np.select(
                [
                    result["date"].isin(inputs.periods.test),
                    result["date"].isin(inputs.periods.validation),
                ],
                ["test", "validation"],
                "earlier",
            )
            result.to_parquet(target, index=False)
            marker.unlink(missing_ok=True)
            print(f"lead {lead} {name}: {time.time() - started:.0f}s", flush=True)


def collect(inputs: Inputs) -> pd.DataFrame:
    """Every prediction file: one folder per lead from this script, and the files that the GPU
    notebook wrote (``intraday_<model>.parquet``, all leads in one file) in the results folder."""
    files = sorted(out_dir(inputs).glob("lead*/predictions_*.parquet"))
    files += [
        f for f in sorted(out_dir(inputs).glob("intraday_*.parquet")) if "_check" not in f.name
    ]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def report(inputs: Inputs) -> None:
    """Scores per lead for the test and validation periods. Rows are models; MAPE is left out.

    Only the blocks that every model has are scored (the notebook's neural models have no window
    for the last few blocks of the last fold)."""
    exp = inputs.experiment
    result = collect(inputs)
    tables = []
    for lead, by_lead in result.groupby("lead"):
        for period in ("test", "validation"):
            part = by_lead[by_lead["period"] == period]
            if part.empty:
                continue
            seen = part.groupby(["date", "block"])["model"].transform("nunique")
            part = part[seen == part["model"].nunique()]
            scores = summarise(part, REFERENCE).drop(columns="MAPE")
            scores = scores.join(cap_scores(part, exp.price_cap, exp.cap_forecast_share))
            scores.insert(0, "period", period)
            scores.insert(0, "lead", lead)
            tables.append(scores.reset_index(names="model"))
            print()
            print(f"== lead {lead}, {period}, {len(part) // part['model'].nunique()} blocks ==")
            print(scores.round(3).drop(columns=["lead", "period"]).to_string())
    if tables:
        pd.concat(tables).to_csv(out_dir(inputs) / "scores.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["final", "report"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--leads", nargs="+", type=int, default=LEADS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS[:7])
    args = parser.parse_args()
    inputs = Inputs(args.smoke)
    actions: dict[str, Callable[[], None]] = {
        "final": lambda: final(inputs, args.leads, list(args.models), args.redo),
        "report": lambda: report(inputs),
    }
    actions[args.stage]()


if __name__ == "__main__":
    main()
