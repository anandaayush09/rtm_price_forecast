"""Forecasting models. Each builder returns a predictor: (training rows, rows to forecast).

All predictors see only the training rows they are given, which the backtest limits to earlier
days. Prices are INR/MWh. When a transform is given the model is fit on transformed prices and
its output is transformed back.
"""

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoLars
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


def _fingerprint(*parts: pd.DataFrame | pd.Series) -> bytes:
    """A hash of the numbers in the given tables, to tell whether two calls get the same data."""
    digest = hashlib.blake2b()
    for part in parts:
        digest.update(np.ascontiguousarray(part.to_numpy(dtype=float)).tobytes())
    return digest.digest()


def mixture_quantile(
    p: np.ndarray, quantiles: np.ndarray, grid: np.ndarray, tau: float, cap: float
) -> np.ndarray:
    """The ``tau`` quantile of a price that is at ``cap`` with probability ``p`` and otherwise
    follows a below-cap distribution known through its quantiles.

    ``quantiles`` has one row per block and one column per level in ``grid`` (rising). The
    mixture puts mass 1 - p below the cap, so the tau quantile is the cap when tau >= 1 - p,
    and otherwise the quantile of the below-cap part at level tau / (1 - p), read from the
    fitted levels by linear interpolation (the nearest fitted level beyond the ends).
    """
    below_mass = 1.0 - np.asarray(p, dtype=float)
    inside = tau < below_mass
    level = np.where(inside, tau / np.where(inside, below_mass, 1.0), 1.0)
    below = np.array([np.interp(a, grid, row) for a, row in zip(level, quantiles, strict=True)])
    return np.where(inside, below, cap)


def _two_stage_fit(
    columns: Sequence[str],
    transform: AsinhTransform | None,
    params: dict[str, Any],
    seed: int,
    cap: float,
    grid: np.ndarray,
) -> Callable[[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, np.ndarray]]:
    """A function (train, test) -> (cap probability, below-cap quantiles at ``grid``).

    Fits are cached on a hash of the data, so predictors that share it fit once per day.
    """
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    shared = {k: v for k, v in params.items() if k not in ("objective", "alpha")}
    base: dict[str, Any] = {
        "random_state": seed,
        "n_jobs": THREADS,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    if grid[0] <= 0.0 or np.any(np.diff(grid) <= 0) or grid[-1] >= 1.0:
        raise ValueError("levels must rise and stay between 0 and 1")

    def fit(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_train, x_test = train[list(columns)], test[list(columns)]
        key = _fingerprint(x_train, train["rtm_price"], x_test)
        if key not in cache:
            at_cap = train["rtm_price"].to_numpy(dtype=float) >= cap
            if at_cap.all():
                raise ValueError("every training block is at the cap")
            if at_cap.any():
                classifier = LGBMClassifier(**base, **shared).fit(x_train, at_cap)
                p = np.asarray(classifier.predict_proba(x_test))[:, 1]
            else:
                p = np.zeros(len(test))
            below = train[~at_cap]
            target = _fit_target(below, transform)
            quantiles = []
            for level in grid:
                regressor = LGBMRegressor(**base, **shared, objective="quantile", alpha=level)
                regressor.fit(below[list(columns)], target)
                quantiles.append(_back(np.asarray(regressor.predict(x_test)), transform))
            cache[key] = (p, np.sort(np.stack(quantiles, axis=1), axis=1))
        return cache[key]

    return fit


def cap_aware_predictors(
    columns: Sequence[str],
    transform: AsinhTransform | None,
    params: dict[str, Any],
    seed: int,
    cap: float,
    levels: Sequence[float] = (0.5, 0.75, 0.9),
) -> dict[str, Predictor]:
    """Two-stage forecasts for prices that sit at the exchange cap in a large share of blocks.

    Stage one is a classifier for "the block settles at the cap" (price >= ``cap``). Stage two is
    quantile regression on the blocks below the cap only, at each of ``levels``. Two forecasts
    come out of the same fits, so both predictors share them:

    * ``hard``: the cap when the cap probability p is at least 0.5, else the median below the cap.
    * ``mixture``: the median of the mixture of the two stages, which is what minimises the mean
      absolute error (see ``mixture_quantile``).

    ``params`` are LightGBM settings without the objective (the tuned ones of the plain model
    are reused). The levels must start at 0.5.
    """
    grid = np.asarray(levels, dtype=float)
    if grid[0] != 0.5:
        raise ValueError("levels must start at 0.5")
    fit = _two_stage_fit(columns, transform, params, seed, cap, grid)

    def hard(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        p, quantiles = fit(train, test)
        return np.where(p >= 0.5, cap, quantiles[:, 0])

    def mixture(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        p, quantiles = fit(train, test)
        return mixture_quantile(p, quantiles, grid, 0.5, cap)

    return {"cap_hard": hard, "cap_mixture": mixture}


def cap_aware_interval_predictors(
    columns: Sequence[str],
    transform: AsinhTransform | None,
    params: dict[str, Any],
    seed: int,
    cap: float,
    coverages: Sequence[float] = (0.5, 0.9),
    levels: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975),
) -> dict[str, Predictor]:
    """Prediction intervals from the two-stage model: for each coverage c, the quantiles of the
    mixture at (1 - c) / 2 and 1 - (1 - c) / 2. Names are ``lo50``, ``hi50``, ``lo90`` and so on.
    """
    grid = np.asarray(levels, dtype=float)
    fit = _two_stage_fit(columns, transform, params, seed, cap, grid)
    predictors: dict[str, Predictor] = {}
    for coverage in coverages:
        tail = (1.0 - coverage) / 2.0
        for name, tau in (
            (f"lo{round(coverage * 100)}", tail),
            (f"hi{round(coverage * 100)}", 1 - tail),
        ):

            def predict(train: pd.DataFrame, test: pd.DataFrame, tau: float = tau) -> np.ndarray:
                p, quantiles = fit(train, test)
                return mixture_quantile(p, quantiles, grid, tau, cap)

            predictors[name] = predict
    return predictors


def lear_predictor(
    columns: Sequence[str],
    transform: AsinhTransform | None,
    alpha: float,
    seed: int,
    solver: Literal["coordinate", "lars"] = "coordinate",
) -> Predictor:
    """Lasso on the numeric inputs plus one-hot block and weekday, after standardising.

    Standardising uses the training rows only. The columns must not contain missing values.
    ``lars`` follows the exact lasso path and is much faster on many rows and small alpha;
    ``coordinate`` is the solver the first benchmark used.
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
        model: Lasso | LassoLars
        if solver == "lars":
            model = LassoLars(alpha=alpha)
        else:
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
