import dataclasses
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rtm_forecast.calendar import load_holidays
from rtm_forecast.config import FeatureParams, load_feature_params
from rtm_forecast.features import build_features, feature_groups
from rtm_forecast.reference import build_features as phase1_features
from rtm_forecast.reference import prepare_frame

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def feature_params() -> FeatureParams:
    return load_feature_params(ROOT / "params.yaml")


@pytest.fixture(scope="module")
def holidays(feature_params: FeatureParams) -> frozenset[date]:
    return load_holidays(ROOT / feature_params.holidays_path)


@pytest.fixture(scope="module")
def features(
    clean_table: pd.DataFrame, feature_params: FeatureParams, holidays: frozenset[date]
) -> pd.DataFrame:
    return build_features(clean_table, feature_params, holidays)


def test_shape_columns_and_target(features: pd.DataFrame, feature_params: FeatureParams) -> None:
    assert len(features) == 9984
    assert features["date"].min() == pd.Timestamp("2020-10-08")
    assert not features.columns.duplicated().any()
    names = [n for group in feature_groups(feature_params).values() for n in group]
    assert set(names) <= set(features.columns)
    assert int((~features["target_observed"]).sum()) == 8


def test_only_demand_lags_have_missing_inputs(features: pd.DataFrame) -> None:
    missing = features.drop(columns=["rtm_price"]).isna().sum()
    assert list(missing[missing > 0].index) == ["demand_lag1d", "demand_lag7d"]


def test_matches_the_first_notebook_where_they_overlap(
    clean_table: pd.DataFrame,
    raw_table: pd.DataFrame,
    feature_params: FeatureParams,
    holidays: frozenset[date],
) -> None:
    """With forward fill, unmasked demand and correct dates the lags equal the first notebook's."""
    same = dataclasses.replace(feature_params, imputation="ffill", demand_policy="keep")
    mine = build_features(clean_table, same, holidays)
    old = phase1_features(prepare_frame(raw_table, dates="day_first"))
    pairs = {
        "dam_price": "DAM_Price",
        "dam_purchase_bid": "DAM_Purchase_Bid",
        "dam_selling_bid": "DAM_Selling_Bid",
        "rtm_price_lag1d": "RTM_Price_PD",
        "rtm_purchase_bid_lag1d": "RTM_Purchase_Bid_PD",
        "rtm_selling_bid_lag1d": "RTM_Selling_Bid_PD",
        "demand_lag1d": "Demand_PD",
        "rtm_price_lag7d": "RTM_Price_SDLW",
        "rtm_purchase_bid_lag7d": "RTM_Purchase_Bid_SDLW",
        "rtm_selling_bid_lag7d": "RTM_Selling_Bid_SDLW",
    }
    for new_name, old_name in pairs.items():
        np.testing.assert_array_equal(mine[new_name].to_numpy(), old[old_name].to_numpy(), new_name)
    observed = mine["target_observed"].to_numpy()
    np.testing.assert_array_equal(
        mine["rtm_price"].to_numpy()[observed], old["RTM_Price_SD"].to_numpy()[observed]
    )


def test_holiday_flags(features: pd.DataFrame) -> None:
    flagged = set(features.loc[features["is_holiday"] == 1, "date"].dt.date)
    assert flagged == {
        date(2020, 10, 25),
        date(2020, 10, 30),
        date(2020, 11, 14),
        date(2020, 11, 30),
        date(2020, 12, 25),
    }
    assert date(2020, 12, 31) not in flagged


def test_weekday_is_read_from_the_correct_date(features: pd.DataFrame) -> None:
    first = features.iloc[0]
    assert first["date"] == pd.Timestamp("2020-10-08")
    assert first["weekday"] == 3  # Thursday


def _scramble(values: pd.Series) -> pd.Series:
    return values * 1.7 + 123.0


@pytest.mark.parametrize("cut_day", ["2020-11-20", "2020-12-10", "2021-01-12"])
def test_no_input_uses_information_from_the_forecast_day_or_later(
    clean_table: pd.DataFrame,
    feature_params: FeatureParams,
    holidays: frozenset[date],
    features: pd.DataFrame,
    cut_day: str,
) -> None:
    """Change every value a forecaster for ``cut_day`` could not know and check its inputs stay put.

    Unknown: RTM prices, RTM bids and demand from the cut day on, and DAM values after the cut day.
    """
    cut = pd.Timestamp(cut_day)
    changed = clean_table.copy()
    later_or_same = changed["date"] >= cut
    for column in ("rtm_price", "rtm_purchase_bid", "rtm_selling_bid", "demand"):
        changed.loc[later_or_same, column] = _scramble(changed.loc[later_or_same, column])
    after = changed["date"] > cut
    for column in ("dam_price", "dam_purchase_bid", "dam_selling_bid"):
        changed.loc[after, column] = _scramble(changed.loc[after, column])
    altered = build_features(changed, feature_params, holidays)
    inputs = [c for c in features.columns if c not in ("rtm_price", "target_observed")]
    day = features["date"] == cut
    pd.testing.assert_frame_equal(
        altered.loc[day, inputs].reset_index(drop=True),
        features.loc[day, inputs].reset_index(drop=True),
    )
    before = features["date"] < cut
    pd.testing.assert_frame_equal(
        altered.loc[before].reset_index(drop=True), features.loc[before].reset_index(drop=True)
    )


def test_lag_inputs_do_react_to_the_past(
    clean_table: pd.DataFrame,
    feature_params: FeatureParams,
    holidays: frozenset[date],
    features: pd.DataFrame,
) -> None:
    """The test above also passes for constant inputs, so check that the lags are not constant."""
    changed = clean_table.copy()
    yesterday = changed["date"] == pd.Timestamp("2020-12-09")
    changed.loc[yesterday, "rtm_price"] = _scramble(changed.loc[yesterday, "rtm_price"])
    altered = build_features(changed, feature_params, holidays)
    day = features["date"] == pd.Timestamp("2020-12-10")
    assert not np.allclose(
        altered.loc[day, "rtm_price_lag1d"], features.loc[day, "rtm_price_lag1d"]
    )
    assert not np.allclose(
        altered.loc[day, "rtm_price_prev_day_mean"], features.loc[day, "rtm_price_prev_day_mean"]
    )


def test_a_gap_at_the_end_of_a_day_is_not_filled_from_the_next_day(
    clean_table: pd.DataFrame, feature_params: FeatureParams, holidays: frozenset[date]
) -> None:
    gapped = clean_table.copy()
    last_two = (gapped["date"] == pd.Timestamp("2020-12-09")) & (gapped["block"] >= 95)
    for column in ("rtm_price", "rtm_purchase_bid", "rtm_selling_bid"):
        gapped.loc[last_two, column] = np.nan
    first = build_features(gapped, feature_params, holidays)
    later = gapped.copy()
    next_day = later["date"] == pd.Timestamp("2020-12-10")
    later.loc[next_day, "rtm_price"] = _scramble(later.loc[next_day, "rtm_price"])
    second = build_features(later, feature_params, holidays)
    day = first["date"] == pd.Timestamp("2020-12-10")
    np.testing.assert_array_equal(
        first.loc[day, "rtm_price_lag1d"].to_numpy(), second.loc[day, "rtm_price_lag1d"].to_numpy()
    )


def test_unknown_demand_policy_is_rejected(
    clean_table: pd.DataFrame, feature_params: FeatureParams, holidays: frozenset[date]
) -> None:
    bad = dataclasses.replace(feature_params, demand_policy="drop")
    with pytest.raises(ValueError):
        build_features(clean_table, bad, holidays)
