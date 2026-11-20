"""Cap-aware two-stage forecasts on the 2025-26 window (day-ahead task).

    python scripts/benchmark_cap_aware.py            # all five folds, both forecasts
    python scripts/benchmark_cap_aware.py --smoke    # three validation days, separate folder

A classifier decides whether a block settles at the price cap and a quantile regression handles the
blocks below it (see ``rtm_forecast.models.cap_aware_predictors``). The tuned LightGBM settings
of the plain model are reused, so nothing is tuned on the test days. Predictions go next to the
other models of the comparison (``predictions_cap_hard.parquet`` and
``predictions_cap_mixture.parquet``), so ``benchmark_window.py report`` and
``significance_window.py`` pick them up. Which of the two forecasts is the headline one is decided
on the validation days only.
"""

import argparse
import json
import time

import numpy as np
from benchmark_window import Inputs, read_setup

from rtm_forecast import models
from rtm_forecast.backtest import rolling_origin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    inputs = Inputs(args.smoke)
    setup = read_setup(inputs)
    best = json.loads((inputs.out / "best_params.json").read_text(encoding="utf-8"))
    predictors = models.cap_aware_predictors(
        setup["columns"],
        setup["transform"],
        best["lightgbm"]["params"],
        inputs.experiment.seed,
        inputs.experiment.price_cap,
    )
    days = [d for fold in inputs.folds for d in fold]
    fold_of = {day: k + 1 for k, fold in enumerate(inputs.folds) for day in fold}
    inputs.out.mkdir(parents=True, exist_ok=True)
    for name, predictor in predictors.items():
        target = inputs.out / f"predictions_{name}.parquet"
        marker = inputs.out / f".running_{name}"
        if (target.exists() or marker.exists()) and not args.redo:
            print(f"{name}: already done or running, skipped", flush=True)
            continue
        marker.write_text("running", encoding="utf-8")
        started = time.time()
        # the two predictors share their fits, so the second run costs almost nothing
        result = rolling_origin(setup["frame"], days, predictor, name)
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
        print(f"{name}: {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
