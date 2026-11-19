"""Hyperparameter search with Optuna on the validation days."""

from collections.abc import Callable, Sequence
from typing import Any

import optuna
import pandas as pd

from rtm_forecast.backtest import Predictor, rolling_origin
from rtm_forecast.metrics import daily_mae


def tune(
    name: str,
    build: Callable[[dict[str, Any]], Predictor],
    space: Callable[[optuna.Trial], dict[str, Any]],
    features: pd.DataFrame,
    validation_days: Sequence[pd.Timestamp],
    trials: int,
    seed: int,
    refit_every: int,
    storage: str,
) -> optuna.Study:
    """Search ``space`` for the settings with the lowest mean daily MAE on the validation days.

    Every trial refits the model on the days before each group of validation days, so no trial sees
    a validation day it forecasts. The study is stored and resumes if it is run again with the same
    name; only the missing trials are run.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        forecast = rolling_origin(features, validation_days, build(params), name, refit_every)
        return float(daily_mae(forecast)[name].mean())

    remaining = trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    return study
