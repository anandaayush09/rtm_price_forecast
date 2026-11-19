"""Model comparison on the 111-day window, in three stages.

    python scripts/benchmark_old_window.py choose   # input settings, decided on validation days
    python scripts/benchmark_old_window.py tune     # Optuna on validation days
    python scripts/benchmark_old_window.py final    # every model on the validation and test days

The last 21 days are the test period. The first two stages never forecast them.
"""

import argparse
import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from rtm_forecast import models
from rtm_forecast.backtest import Predictor, rolling_origin
from rtm_forecast.calendar import load_holidays
from rtm_forecast.config import (
    ExperimentParams,
    FeatureParams,
    load_experiment_params,
    load_feature_params,
)
from rtm_forecast.features import build_features, feature_groups, fill_rtm
from rtm_forecast.ingest import load_raw
from rtm_forecast.metrics import daily_mae
from rtm_forecast.reference import build_features as phase1_features
from rtm_forecast.reference import prepare_frame
from rtm_forecast.splits import Periods, make_periods
from rtm_forecast.transforms import AsinhTransform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "old_window"
# Row and column sampling make the seed matter, so three seeds show how much a score moves by
# chance.
LIGHTGBM_START = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}
SEEDS = (0, 1, 2)


class Inputs:
    """Everything the stages share."""

    def __init__(self) -> None:
        self.feature_params: FeatureParams = load_feature_params(ROOT / "params.yaml")
        self.experiment: ExperimentParams = load_experiment_params(ROOT / "params.yaml")
        clean = pd.read_parquet(ROOT / "data" / "interim" / "clean.parquet")
        holidays = load_holidays(ROOT / self.feature_params.holidays_path)
        self.clean = clean
        self.features = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
        keep = dataclasses.replace(self.feature_params, demand_policy="keep")
        self.features_keep = build_features(clean, keep, holidays)
        days = [pd.Timestamp(d) for d in sorted(self.features["date"].unique())]
        self.periods: Periods = make_periods(
            days, self.experiment.validation_days, self.experiment.test_days
        )
        self.groups = feature_groups(self.feature_params)
        self.all_columns = [c for names in self.groups.values() for c in names]
        filled = fill_rtm(clean, self.feature_params.imputation)
        self.history = pd.Series(filled["rtm_price"].to_numpy(), index=filled["timestamp"])
        self.asinh = AsinhTransform(
            float(json.loads((ROOT / "reports" / "feature_report.json").read_text())["asinh_scale"])
        )


def columns_without(inputs: Inputs, groups: list[str]) -> list[str]:
    return [c for g, names in inputs.groups.items() if g not in groups for c in names]


def validation_mae(forecast: pd.DataFrame, name: str) -> float:
    return float(daily_mae(forecast)[name].mean())


def choose(inputs: Inputs) -> None:
    """Compare settings with LightGBM at fixed parameters (three seeds each) on validation days."""
    settings: dict[str, tuple[pd.DataFrame, list[str], AsinhTransform | None]] = {
        "raw price, all inputs": (inputs.features, inputs.all_columns, None),
        "asinh price, all inputs": (inputs.features, inputs.all_columns, inputs.asinh),
        "asinh, demand not used": (
            inputs.features,
            columns_without(inputs, ["demand_lags"]),
            inputs.asinh,
        ),
        "asinh, demand as published": (inputs.features_keep, inputs.all_columns, inputs.asinh),
    }
    for group in inputs.groups:
        if group != "demand_lags":
            settings[f"asinh, without {group}"] = (
                inputs.features,
                columns_without(inputs, [group]),
                inputs.asinh,
            )
    rows = []
    for label, (frame, columns, transform) in settings.items():
        started = time.time()
        scores = []
        for seed in SEEDS:
            predictor = models.lightgbm_predictor(columns, transform, LIGHTGBM_START, seed)
            forecast = rolling_origin(frame, inputs.periods.validation, predictor, "m")
            scores.append(validation_mae(forecast, "m"))
        rows.append(
            {
                "setting": label,
                "inputs": len(columns),
                "MAE mean": float(np.mean(scores)),
                "MAE min": float(np.min(scores)),
                "MAE max": float(np.max(scores)),
            }
        )
        print(f"{label}: {np.mean(scores):.2f} ({time.time() - started:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values("MAE mean")
    table.to_csv(OUT / "choose_setup.csv", index=False)
    print(table.round(2).to_string(index=False))


def lightgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "objective": trial.suggest_categorical(
            "objective", ["regression", "regression_l1", "huber"]
        ),
    }


def catboost_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 100, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 7),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "loss_function": trial.suggest_categorical("loss_function", ["RMSE", "MAE"]),
    }


def lear_space(trial: optuna.Trial) -> dict[str, Any]:
    return {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}


def builders(
    inputs: Inputs, setup: dict[str, Any]
) -> dict[str, tuple[Callable[..., Predictor], Callable[..., dict[str, Any]]]]:
    columns: list[str] = setup["columns"]
    transform: AsinhTransform | None = setup["transform"]
    seed = inputs.experiment.seed
    no_demand = [c for c in columns if not c.startswith("demand_")]
    return {
        "lightgbm": (
            lambda p: models.lightgbm_predictor(columns, transform, p, seed),
            lightgbm_space,
        ),
        "catboost": (
            lambda p: models.catboost_predictor(columns, transform, p, seed),
            catboost_space,
        ),
        "lear": (
            lambda p: models.lear_predictor(no_demand, transform, p["alpha"], seed),
            lear_space,
        ),
    }


def read_setup(inputs: Inputs) -> dict[str, Any]:
    chosen = json.loads((OUT / "setup.json").read_text())
    frame = inputs.features_keep if chosen["demand"] == "keep" else inputs.features
    columns = columns_without(inputs, chosen["drop_groups"])
    transform = inputs.asinh if chosen["transform"] == "asinh" else None
    return {"frame": frame, "columns": columns, "transform": transform, "chosen": chosen}


def tune_all(inputs: Inputs) -> None:
    from rtm_forecast.tuning import tune

    setup = read_setup(inputs)
    storage = f"sqlite:///{(ROOT / 'optuna' / 'old_window.db').as_posix()}"
    (ROOT / "optuna").mkdir(exist_ok=True)
    best: dict[str, Any] = {}
    for name, (build, space) in builders(inputs, setup).items():
        started = time.time()
        study = tune(
            f"old_window_{name}",
            build,
            space,
            setup["frame"],
            inputs.periods.validation,
            inputs.experiment.tuning_trials,
            inputs.experiment.seed,
            inputs.experiment.tuning_refit_every_days,
            storage,
        )
        # The full settings, including the values the search keeps fixed.
        full = space(optuna.trial.FixedTrial(study.best_params))
        best[name] = {"params": full, "validation_MAE": study.best_value}
        history = study.trials_dataframe()
        history.to_csv(OUT / f"trials_{name}.csv", index=False)
        print(f"{name}: best {study.best_value:.2f} in {time.time() - started:.0f}s", flush=True)
        (OUT / "best_params.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")


def final(inputs: Inputs) -> None:
    setup = read_setup(inputs)
    best = json.loads((OUT / "best_params.json").read_text())
    frame, columns, transform = setup["frame"], setup["columns"], setup["transform"]
    seed = inputs.experiment.seed
    no_demand = [c for c in columns if not c.startswith("demand_")]
    days = inputs.periods.validation + inputs.periods.test
    runs: dict[str, tuple[pd.DataFrame, Predictor]] = {
        "naive_yesterday": (frame, models.column_predictor("rtm_price_lag1d")),
        "naive_lastweek": (frame, models.column_predictor("rtm_price_lag7d")),
        "dam_price": (frame, models.column_predictor("dam_price")),
        "mstl_ets": (frame, models.mstl_ets_predictor(inputs.history)),
        "lear": (
            frame,
            models.lear_predictor(no_demand, transform, best["lear"]["params"]["alpha"], seed),
        ),
        "lightgbm": (
            frame,
            models.lightgbm_predictor(columns, transform, best["lightgbm"]["params"], seed),
        ),
        "catboost": (
            frame,
            models.catboost_predictor(columns, transform, best["catboost"]["params"], seed),
        ),
    }
    raw = load_raw(ROOT / "data" / "raw" / "price_data.csv")
    old = phase1_features(prepare_frame(raw, dates="day_first"))
    old_inputs = [c for c in old.columns if c != "RTM_Price_SD"]
    p1 = pd.concat(
        [frame[["date", "block", "timestamp", "rtm_price", "target_observed"]], old[old_inputs]],
        axis=1,
    )
    runs["phase1_xgboost"] = (p1, models.phase1_xgboost_predictor(old_inputs))
    runs["phase1_random_forest"] = (p1, models.phase1_random_forest_predictor(old_inputs, seed))
    parts = []
    for name, (data, predictor) in runs.items():
        started = time.time()
        parts.append(rolling_origin(data, days, predictor, name))
        print(f"{name}: {time.time() - started:.0f}s", flush=True)
    result = pd.concat(parts, ignore_index=True)
    result["period"] = np.where(result["date"].isin(inputs.periods.test), "test", "validation")
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT / "predictions.parquet", index=False)
    print("wrote", OUT / "predictions.parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["choose", "tune", "final"])
    stage = parser.parse_args().stage
    inputs = Inputs()
    {"choose": choose, "tune": tune_all, "final": final}[stage](inputs)


if __name__ == "__main__":
    main()
