"""Forecasting models. Each builder returns a predictor: (training rows, rows to forecast).

All predictors see only the training rows they are given, which the backtest limits to earlier
days. Prices are INR/MWh. When a transform is given the model is fit on transformed prices and
its output is transformed back.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import MSTL
from xgboost import XGBRegressor

from rtm_forecast.backtest import Predictor
from rtm_forecast.ingest import BLOCKS_PER_DAY
from rtm_forecast.reference import RF_PARAMS, XGB_PARAMS
from rtm_forecast.transforms import AsinhTransform

THREADS = 2


def _fit_target(train: pd.DataFrame, transform: AsinhTransform | None) -> np.ndarray:
    target = train["rtm_price"].to_numpy(dtype=float)
    return target if transform is None else transform.forward(target)


def _back(values: np.ndarray, transform: AsinhTransform | None) -> np.ndarray:
    return values if transform is None else transform.inverse(values)


def column_predictor(column: str) -> Predictor:
    """Forecast equal to one input column (used for the simple forecasts)."""

    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return test[column].to_numpy(dtype=float)

    return predict


def lightgbm_predictor(
    columns: Sequence[str], transform: AsinhTransform | None, params: dict[str, Any], seed: int
) -> Predictor:
    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = LGBMRegressor(
            random_state=seed,
            n_jobs=THREADS,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
            **params,
        )
        model.fit(train[list(columns)], _fit_target(train, transform))
        return _back(np.asarray(model.predict(test[list(columns)])), transform)

    return predict


def catboost_predictor(
    columns: Sequence[str], transform: AsinhTransform | None, params: dict[str, Any], seed: int
) -> Predictor:
    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = CatBoostRegressor(
            random_seed=seed,
            thread_count=THREADS,
            verbose=0,
            allow_writing_files=False,
            **params,
        )
        model.fit(train[list(columns)], _fit_target(train, transform))
        return _back(np.asarray(model.predict(test[list(columns)])), transform)

    return predict


def lear_predictor(
    columns: Sequence[str], transform: AsinhTransform | None, alpha: float, seed: int
) -> Predictor:
    """Lasso on the numeric inputs plus one-hot block and weekday, after standardising.

    Standardising uses the training rows only. The columns must not contain missing values.
    """
    numeric = [c for c in columns if c not in ("block", "weekday")]

    def design(frame: pd.DataFrame) -> pd.DataFrame:
        blocks = pd.get_dummies(
            pd.Categorical(frame["block"], categories=range(1, BLOCKS_PER_DAY + 1)), prefix="b"
        )
        weekdays = pd.get_dummies(pd.Categorical(frame["weekday"], categories=range(7)), prefix="w")
        parts = [frame[numeric].reset_index(drop=True), blocks, weekdays]
        return pd.concat(parts, axis=1).astype(float)

    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        x_train, x_test = design(train), design(test)
        if x_train.isna().any().any() or x_test.isna().any().any():
            raise ValueError("LEAR inputs contain missing values")
        scaler = StandardScaler().fit(x_train)
        model = Lasso(alpha=alpha, max_iter=10000, random_state=seed)
        model.fit(scaler.transform(x_train), _fit_target(train, transform))
        return _back(np.asarray(model.predict(scaler.transform(x_test))), transform)

    return predict


def phase1_xgboost_predictor(columns: Sequence[str]) -> Predictor:
    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = XGBRegressor(
            tree_method="exact", base_score=0.5, random_state=0, n_jobs=THREADS, **XGB_PARAMS
        )
        model.fit(train[list(columns)], train["rtm_price"])
        return np.asarray(model.predict(test[list(columns)]))

    return predict


def phase1_random_forest_predictor(columns: Sequence[str], seed: int) -> Predictor:
    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = RandomForestRegressor(random_state=seed, n_jobs=THREADS, **RF_PARAMS)
        model.fit(train[list(columns)], train["rtm_price"])
        return np.asarray(model.predict(test[list(columns)]))

    return predict


def mstl_ets_predictor(history: pd.Series) -> Predictor:
    """Seasonal decomposition (daily and weekly), a damped-trend exponential smoothing of what is
    left, and the last daily and weekly seasonal pattern repeated.

    ``history`` is the filled RTM price on the 15-minute grid, indexed by timestamp. For each day
    only values before midnight are used. It uses no day-ahead market information.
    """

    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        first = test["date"].min()
        zone = pd.DatetimeIndex(history.index).tz
        past = history[history.index < first.tz_localize(zone)].to_numpy()
        daily, weekly = BLOCKS_PER_DAY, 7 * BLOCKS_PER_DAY
        parts = MSTL(past, periods=(daily, weekly)).fit()
        adjusted = past - parts.seasonal.sum(axis=1)
        smooth = ExponentialSmoothing(adjusted, trend="add", damped_trend=True).fit()
        horizon = len(test)
        level = np.asarray(smooth.forecast(horizon))
        seasonal_daily = np.tile(parts.seasonal[:, 0][-daily:], horizon // daily + 1)[:horizon]
        seasonal_weekly = parts.seasonal[:, 1][-weekly:][:horizon]
        return np.asarray(level + seasonal_daily + seasonal_weekly)

    return predict
