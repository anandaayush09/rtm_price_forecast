"""Model comparison on the 2025-26 window, in four stages.

    python scripts/benchmark_window.py choose   # input settings, decided on validation days
    python scripts/benchmark_window.py tune     # Optuna on validation days
    python scripts/benchmark_window.py final    # models on the five rolling folds (--models ...)
    python scripts/benchmark_window.py report   # scores per period and fold

The last 60 days are the test period (folds 4 and 5), the 30 days before them are the validation
period (fold 3), and folds 1 and 2 are earlier months. The first two stages never forecast the
test days. Add --smoke for a quick run on a few days that writes to a separate folder.

The inputs are private files, so results go to .project-local/results_window and are not
committed.
"""

import argparse
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
    FeatureParams,
    WindowExperimentParams,
    load_feature_params,
    load_window_experiment_params,
    load_window_params,
)
from rtm_forecast.features import fill_rtm
from rtm_forecast.metrics import cap_scores, daily_mae, summarise
from rtm_forecast.reference import build_features as phase1_features
from rtm_forecast.reference import prepare_frame
from rtm_forecast.splits import Periods, make_periods, rolling_folds
from rtm_forecast.transforms import AsinhTransform, choose_scale
from rtm_forecast.window_features import build_window_features, window_feature_groups

ROOT = Path(__file__).resolve().parents[1]
LIGHTGBM_START = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}
SEEDS = (0, 1, 2)
# Inputs that come from the demand sheets can be missing; the linear model leaves them out.
SHEET_PREFIXES = ("demand_", "wind_", "solar_", "net_demand_")
SHEET_GROUPS = ["demand_lags", "weather_lags", "net_demand"]


class Inputs:
    """Everything the stages share."""

    def __init__(self, smoke: bool) -> None:
        self.smoke = smoke
        self.out = ROOT / ".project-local" / ("results_window_smoke" if smoke else "results_window")
        self.feature_params: FeatureParams = load_feature_params(ROOT / "params.yaml")
        self.experiment: WindowExperimentParams = load_window_experiment_params(
            ROOT / "params.yaml"
        )
        window = pd.read_parquet(ROOT / load_window_params(ROOT / "params.yaml").clean_path)
        self.holiday_dates = load_holidays(ROOT / self.feature_params.holidays_path)
        self.features = build_window_features(window, self.feature_params, self.holiday_dates)
        self.features_published = build_window_features(
            window, self.feature_params, self.holiday_dates, one_definition=False
        )
        days = [pd.Timestamp(d) for d in sorted(self.features["date"].unique())]
        self.periods: Periods = make_periods(
            days, self.experiment.validation_days, self.experiment.test_days
        )
        self.folds = rolling_folds(days, self.experiment.folds, self.experiment.fold_days)
        self.trials = self.experiment.tuning_trials
        if smoke:
            self.periods = Periods(
                self.periods.train, self.periods.validation[:3], self.periods.test[:0]
            )
            self.folds = [self.periods.validation]
            self.trials = 2
        self.groups = window_feature_groups(self.feature_params)
        self.all_columns = [c for names in self.groups.values() for c in names]
        filled = fill_rtm(window, self.feature_params.imputation)
        self.history = pd.Series(filled["rtm_price"].to_numpy(), index=filled["timestamp"])
        self.window = window
        train = self.features[
            self.features["date"].isin(self.periods.train) & self.features["target_observed"]
        ]
        # The scale comes from the training days only. The old window used the first 90 days; the
        # price regime changes during this window, so all training days are used.
        self.asinh = AsinhTransform(
            choose_scale(train["rtm_price"].to_numpy(), self.feature_params.asinh_scales)
        )


def columns_without(inputs: Inputs, groups: list[str]) -> list[str]:
    return [c for g, names in inputs.groups.items() if g not in groups for c in names]


def linear_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if not c.startswith(SHEET_PREFIXES)]


def validation_mae(forecast: pd.DataFrame, name: str) -> float:
    return float(daily_mae(forecast)[name].mean())


def settings_to_compare(inputs: Inputs) -> dict[str, dict[str, Any]]:
    settings: dict[str, dict[str, Any]] = {
        "raw price, all inputs": {"transform": "raw", "drop": [], "demand": "one_definition"},
        "asinh price, all inputs": {"transform": "asinh", "drop": [], "demand": "one_definition"},
        "asinh, demand as published": {"transform": "asinh", "drop": [], "demand": "published"},
        "asinh, no demand-sheet inputs": {
            "transform": "asinh",
            "drop": SHEET_GROUPS,
            "demand": "one_definition",
        },
    }
    for group in inputs.groups:
        settings[f"asinh, without {group}"] = {
            "transform": "asinh",
            "drop": [group],
            "demand": "one_definition",
        }
    if inputs.smoke:
        return dict(list(settings.items())[:2])
    return settings


def frame_for(inputs: Inputs, demand: str) -> pd.DataFrame:
    return inputs.features if demand == "one_definition" else inputs.features_published


def transform_for(inputs: Inputs, name: str) -> AsinhTransform | None:
    return inputs.asinh if name == "asinh" else None


def choose(inputs: Inputs) -> None:
    """Compare settings with LightGBM at fixed parameters (three seeds each) on validation days."""
    rows = []
    refit = inputs.experiment.tuning_refit_every_days
    seeds = SEEDS[:1] if inputs.smoke else SEEDS
    for label, setting in settings_to_compare(inputs).items():
        started = time.time()
        columns = columns_without(inputs, setting["drop"])
        frame = frame_for(inputs, setting["demand"])
        transform = transform_for(inputs, setting["transform"])
        scores = []
        for seed in seeds:
            predictor = models.lightgbm_predictor(columns, transform, LIGHTGBM_START, seed)
            forecast = rolling_origin(frame, inputs.periods.validation, predictor, "m", refit)
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
    inputs.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values("MAE mean")
    table.to_csv(inputs.out / "choose_setup.csv", index=False)
    print(table.round(2).to_string(index=False))
    best = settings_to_compare(inputs)[str(table.iloc[0]["setting"])]
    chosen = {
        "transform": best["transform"],
        "drop_groups": best["drop"],
        "demand": best["demand"],
        "asinh_scale": inputs.asinh.scale,
        "setting": str(table.iloc[0]["setting"]),
        "reason": "Lowest mean validation MAE of LightGBM at fixed parameters over three seeds "
        "(choose_setup.csv). The seeds overlap for close settings, so read the table too.",
    }
    (inputs.out / "setup.json").write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")


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


def read_setup(inputs: Inputs) -> dict[str, Any]:
    chosen = json.loads((inputs.out / "setup.json").read_text())
    return {
        "frame": frame_for(inputs, chosen["demand"]),
        "columns": columns_without(inputs, chosen["drop_groups"]),
        "transform": transform_for(inputs, chosen["transform"]),
        "chosen": chosen,
    }


def builders(
    inputs: Inputs, setup: dict[str, Any]
) -> dict[str, tuple[Callable[..., Predictor], Callable[..., dict[str, Any]]]]:
    columns: list[str] = setup["columns"]
    transform: AsinhTransform | None = setup["transform"]
    seed = inputs.experiment.seed
    plain = linear_columns(columns)
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
            lambda p: models.lear_predictor(plain, transform, p["alpha"], seed, "lars"),
            lear_space,
        ),
    }


def tune_all(inputs: Inputs) -> None:
    from rtm_forecast.tuning import tune

    setup = read_setup(inputs)
    name_prefix = "window_smoke" if inputs.smoke else "window"
    (ROOT / "optuna").mkdir(exist_ok=True)
    storage = f"sqlite:///{(ROOT / 'optuna' / (name_prefix + '.db')).as_posix()}"
    best: dict[str, Any] = {}
    for name, (build, space) in builders(inputs, setup).items():
        started = time.time()
        study = tune(
            f"{name_prefix}_{name}",
            build,
            space,
            setup["frame"],
            inputs.periods.validation,
            inputs.trials,
            inputs.experiment.seed,
            inputs.experiment.tuning_refit_every_days,
            storage,
        )
        full = space(optuna.trial.FixedTrial(study.best_params))
        best[name] = {"params": full, "validation_MAE": study.best_value}
        study.trials_dataframe().to_csv(inputs.out / f"trials_{name}.csv", index=False)
        print(f"{name}: best {study.best_value:.2f} in {time.time() - started:.0f}s", flush=True)
        (inputs.out / "best_params.json").write_text(json.dumps(best, indent=2) + "\n", "utf-8")


def phase1_inputs(inputs: Inputs, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The 2021 model inputs rebuilt from the window table.

    The old notebook had six fixed holiday dates; the window list is used here instead.
    """
    window = inputs.window
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
    holidays = [pd.Timestamp(d) for d in sorted(inputs.holiday_dates)]
    old = phase1_features(prepare_frame(raw, dates="day_first", holidays=holidays))
    columns = [c for c in old.columns if c != "RTM_Price_SD"]
    if len(old) != len(frame):
        raise ValueError("the 2021 inputs and the window inputs do not line up")
    base = frame[["date", "block", "timestamp", "rtm_price", "target_observed"]]
    return pd.concat([base, old[columns]], axis=1), columns


INDEPENDENT = [
    "naive_yesterday",
    "naive_lastweek",
    "dam_price",
    "mstl_ets",
    "phase1_xgboost",
    "phase1_random_forest",
]
TUNED = ["lear", "lightgbm", "catboost"]


def final(inputs: Inputs, only: list[str], redo: bool = False) -> None:
    """Forecast every fold day with each model in ``only``; one file of predictions per model.

    The simple models and the 2021 models need neither the chosen setup nor the tuned settings,
    so they can run while the search is still going.
    """
    seed = inputs.experiment.seed
    runs: dict[str, Callable[[], tuple[pd.DataFrame, Predictor]]] = {}
    frame = inputs.features
    runs["naive_yesterday"] = lambda: (frame, models.column_predictor("rtm_price_lag1d"))
    runs["naive_lastweek"] = lambda: (frame, models.column_predictor("rtm_price_lag7d"))
    runs["dam_price"] = lambda: (frame, models.column_predictor("dam_price"))
    runs["mstl_ets"] = lambda: (frame, models.mstl_ets_predictor(inputs.history))

    def tuned(name: str) -> tuple[pd.DataFrame, Predictor]:
        setup = read_setup(inputs)
        best = json.loads((inputs.out / "best_params.json").read_text())
        columns, transform = setup["columns"], setup["transform"]
        if name == "lear":
            alpha = best["lear"]["params"]["alpha"]
            predictor = models.lear_predictor(
                linear_columns(columns), transform, alpha, seed, "lars"
            )
        elif name == "lightgbm":
            predictor = models.lightgbm_predictor(columns, transform, best[name]["params"], seed)
        else:
            predictor = models.catboost_predictor(columns, transform, best[name]["params"], seed)
        return setup["frame"], predictor

    for name in TUNED:
        runs[name] = lambda name=name: tuned(name)
    p1_frame, p1_columns = phase1_inputs(inputs, inputs.features)
    runs["phase1_xgboost"] = lambda: (p1_frame, models.phase1_xgboost_predictor(p1_columns))
    runs["phase1_random_forest"] = lambda: (
        p1_frame,
        models.phase1_random_forest_predictor(p1_columns, seed),
    )
    days = [d for fold in inputs.folds for d in fold]
    fold_of = {day: k + 1 for k, fold in enumerate(inputs.folds) for day in fold}
    inputs.out.mkdir(parents=True, exist_ok=True)
    for name in only:
        target = inputs.out / f"predictions_{name}.parquet"
        marker = inputs.out / f".running_{name}"
        if (target.exists() or marker.exists()) and not redo:
            print(f"{name}: already done or running, skipped (use --redo to run again)", flush=True)
            continue
        marker.write_text("running", encoding="utf-8")
        started = time.time()
        data, predictor = runs[name]()
        result = rolling_origin(data, days, predictor, name)
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


def report(inputs: Inputs) -> None:
    """Scores per period and per fold. MAPE is left out: some prices are exactly 0."""
    files = sorted(inputs.out.glob("predictions_*.parquet"))
    result = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    exp = inputs.experiment
    names = {"rtm_price_lag1d": "naive_yesterday", "rtm_price_lag7d": "naive_lastweek"}
    reference = names[exp.naive_reference]
    groups: dict[str, pd.DataFrame] = {
        "test": result[result["period"] == "test"],
        "validation": result[result["period"] == "validation"],
    }
    for fold, part in result.groupby("fold"):
        groups[f"fold {int(fold)}"] = part
    tables = []
    for label, part in groups.items():
        if part.empty:
            continue
        scores = summarise(part, reference).drop(columns="MAPE")
        scores = scores.join(cap_scores(part, exp.price_cap, exp.cap_forecast_share))
        scores.insert(0, "group", label)
        tables.append(scores.reset_index(names="model"))
        print(f"\n== {label} ==")
        print(scores.round(3).drop(columns="group").to_string())
    pd.concat(tables).to_csv(inputs.out / "scores.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["choose", "tune", "final", "report"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--redo", action="store_true", help="run models that already have results")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*INDEPENDENT, *TUNED],
        default=[*INDEPENDENT, *TUNED],
        help="for the final stage: which models to forecast",
    )
    args = parser.parse_args()
    inputs = Inputs(args.smoke)
    if args.stage == "final":
        final(inputs, list(args.models), args.redo)
    else:
        {"choose": choose, "tune": tune_all, "report": report}[args.stage](inputs)


if __name__ == "__main__":
    main()
