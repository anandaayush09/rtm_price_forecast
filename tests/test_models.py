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
