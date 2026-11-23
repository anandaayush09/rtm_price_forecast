from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rtm_forecast import models as m
from rtm_forecast.backtest import rolling_origin
from rtm_forecast.calendar import load_holidays
from rtm_forecast.config import FeatureParams, load_feature_params
from rtm_forecast.features import build_features, feature_groups, fill_rtm
from rtm_forecast.transforms import AsinhTransform

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def feature_params() -> FeatureParams:
    return load_feature_params(ROOT / "params.yaml")


@pytest.fixture(scope="module")
def table(clean_table: pd.DataFrame, feature_params: FeatureParams) -> pd.DataFrame:
    holidays: frozenset[date] = load_holidays(ROOT / feature_params.holidays_path)
    return build_features(clean_table, feature_params, holidays)


def _days(table: pd.DataFrame) -> list[pd.Timestamp]:
    return [pd.Timestamp(d) for d in sorted(table["date"].unique())[-3:]]


def _columns(feature_params: FeatureParams, with_demand: bool) -> list[str]:
    groups = feature_groups(feature_params)
    return [c for g, v in groups.items() if with_demand or g != "demand_lags" for c in v]


@pytest.mark.parametrize(
    "build",
    [
        lambda cols: m.lightgbm_predictor(cols, AsinhTransform(1000.0), {"n_estimators": 30}, 0),
        lambda cols: m.catboost_predictor(cols, None, {"iterations": 30, "depth": 4}, 0),
    ],
)
def test_tree_models_return_one_finite_value_per_row(
    table: pd.DataFrame, feature_params: FeatureParams, build: object
) -> None:
    columns = _columns(feature_params, with_demand=True)
    predictor = build(columns)  # type: ignore[operator]
    result = rolling_origin(table, _days(table), predictor, "m")
    assert len(result) == 3 * 96
    assert np.isfinite(result["prediction"]).all()
    assert result["prediction"].between(0, 20000).all()


def test_tree_models_are_repeatable(table: pd.DataFrame, feature_params: FeatureParams) -> None:
    columns = _columns(feature_params, with_demand=True)
    days = _days(table)[:1]
    first = rolling_origin(
        table, days, m.lightgbm_predictor(columns, None, {"n_estimators": 40}, 3), "a"
    )
    second = rolling_origin(
        table, days, m.lightgbm_predictor(columns, None, {"n_estimators": 40}, 3), "a"
    )
    np.testing.assert_array_equal(first["prediction"], second["prediction"])


def test_lear_needs_complete_inputs_and_runs_without_demand(
    table: pd.DataFrame, feature_params: FeatureParams
) -> None:
    columns = _columns(feature_params, with_demand=False)
    result = rolling_origin(table, _days(table), m.lear_predictor(columns, None, 1.0, 0), "lear")
    assert np.isfinite(result["prediction"]).all()
    with pytest.raises(ValueError):
        rolling_origin(
            table,
            _days(table),
            m.lear_predictor(_columns(feature_params, with_demand=True), None, 1.0, 0),
            "lear",
        )


def test_simple_forecasts_copy_their_column(table: pd.DataFrame) -> None:
    days = _days(table)
    result = rolling_origin(table, days, m.column_predictor("dam_price"), "dam")
    expected = table[table["date"].isin(days)]["dam_price"].to_numpy()
    np.testing.assert_array_equal(result["prediction"].to_numpy(), expected)


def test_mstl_forecast_uses_only_earlier_values(
    clean_table: pd.DataFrame, table: pd.DataFrame, feature_params: FeatureParams
) -> None:
    filled = fill_rtm(clean_table, feature_params.imputation)
    history = pd.Series(filled["rtm_price"].to_numpy(), index=filled["timestamp"])
    day = [_days(table)[-1]]
    first = rolling_origin(table, day, m.mstl_ets_predictor(history), "mstl")
    changed = history.copy()
    zone = pd.DatetimeIndex(changed.index).tz
    changed[changed.index >= day[0].tz_localize(zone)] *= 3.0
    second = rolling_origin(table, day, m.mstl_ets_predictor(changed), "mstl")
    np.testing.assert_array_equal(first["prediction"], second["prediction"])
    assert np.isfinite(first["prediction"]).all()


def test_lars_solver_gives_the_same_lasso_forecasts(
    table: pd.DataFrame, feature_params: FeatureParams
) -> None:
    columns = _columns(feature_params, with_demand=False)
    transform = AsinhTransform(1000.0)
    days = _days(table)[:1]
    slow = rolling_origin(
        table, days, m.lear_predictor(columns, transform, 0.01, 0, "coordinate"), "a"
    )
    fast = rolling_origin(table, days, m.lear_predictor(columns, transform, 0.01, 0, "lars"), "a")
    np.testing.assert_allclose(fast["prediction"], slow["prediction"], rtol=2e-3)


CAP = 10000.0


def _capped_prices(rows: int, seed: int, cap_when_positive: bool = True) -> pd.DataFrame:
    """Price is at the cap when x > 0 and about 3000 + 500 x otherwise: the cap is predictable."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    below = 3000 + 500 * x + rng.normal(scale=100, size=rows)
    price = np.where(x > 0, CAP, below) if cap_when_positive else below
    dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(rows) // 96, unit="D")
    return pd.DataFrame({"date": dates, "x": x, "rtm_price": price})


CAP_PARAMS = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 8, "min_child_samples": 5}


def test_cap_aware_forecasts_call_the_cap_and_stay_below_it_otherwise() -> None:
    train, test = _capped_prices(4000, 0), _capped_prices(960, 1)
    predictors = m.cap_aware_predictors(["x"], None, CAP_PARAMS, 0, CAP)
    hard = predictors["cap_hard"](train, test)
    mixture = predictors["cap_mixture"](train, test)
    clear = np.abs(test["x"].to_numpy()) > 0.5
    at_cap = test["x"].to_numpy() > 0
    assert (hard[clear & at_cap] == CAP).all()
    assert (hard[clear & ~at_cap] < CAP).all()
    # where the cap is called both forecasts agree; below it the mixture median is not lower
    called = hard == CAP
    np.testing.assert_array_equal(mixture[called], hard[called])
    assert (mixture[~called] >= hard[~called] - 1e-9).all()
    assert (mixture <= CAP).all()


def test_cap_aware_forecasts_do_not_read_the_test_prices() -> None:
    train, test = _capped_prices(3000, 2), _capped_prices(480, 3)
    changed = test.assign(rtm_price=0.0)
    first = m.cap_aware_predictors(["x"], None, CAP_PARAMS, 0, CAP)["cap_mixture"](train, test)
    second = m.cap_aware_predictors(["x"], None, CAP_PARAMS, 0, CAP)["cap_mixture"](train, changed)
    np.testing.assert_array_equal(first, second)


def test_cap_aware_forecast_without_any_cap_in_training_is_a_plain_median_forecast() -> None:
    train, test = _capped_prices(2000, 4, cap_when_positive=False), _capped_prices(480, 5)
    predictors = m.cap_aware_predictors(["x"], None, CAP_PARAMS, 0, CAP)
    assert (predictors["cap_hard"](train, test) < CAP).all()
    with pytest.raises(ValueError, match="every training block"):
        predictors["cap_hard"](train.assign(rtm_price=CAP), test)


def test_cap_aware_levels_are_checked() -> None:
    for levels in [(0.6, 0.9), (0.5, 0.5), (0.5, 1.0)]:
        with pytest.raises(ValueError, match="levels"):
            m.cap_aware_predictors(["x"], None, CAP_PARAMS, 0, CAP, levels)
