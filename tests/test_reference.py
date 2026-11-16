"""Numbers printed by the original notebook, reproduced by the port.

The expected values are the ones the notebook printed, compared at the precision it printed them.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rtm_forecast.reference import (
    build_features,
    legacy_daily_mae,
    mae,
    mape,
    parse_dates,
    prepare_frame,
    quarter_mapes,
    split_last_day,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "original" / "results"


@pytest.fixture(scope="module")
def notebook_frame(raw_table: pd.DataFrame) -> pd.DataFrame:
    return prepare_frame(raw_table)


@pytest.fixture(scope="module")
def features(notebook_frame: pd.DataFrame) -> pd.DataFrame:
    return build_features(notebook_frame)


def test_month_first_parsing_changes_44_dates(raw_table: pd.DataFrame) -> None:
    unique = raw_table["Date"].drop_duplicates()
    wrong = parse_dates(unique, "month_first") != parse_dates(unique, "day_first")
    assert int(wrong.sum()) == 44
    assert parse_dates(raw_table["Date"].head(1), "month_first").iloc[0] == pd.Timestamp(
        "2020-01-10"
    )


def test_missing_values_before_and_after_fill(raw_table: pd.DataFrame) -> None:
    converted = raw_table.assign(RTM_Price=pd.to_numeric(raw_table["RTM_Price"], errors="coerce"))
    assert int(converted["RTM_Price"].isna().sum()) == 12
    assert int(prepare_frame(raw_table).isna().sum().sum()) == 0


def test_day_type_counts(raw_table: pd.DataFrame) -> None:
    weekend_only = prepare_frame(raw_table, holidays=[])
    assert weekend_only["HolidayId"].value_counts().sort_index().to_dict() == {0: 7680, 1: 2976}
    full = prepare_frame(raw_table)
    assert full["HolidayId"].value_counts().sort_index().to_dict() == {0: 7392, 1: 2784, 2: 480}


def test_feature_shapes(features: pd.DataFrame) -> None:
    assert features.shape == (9984, 110)
    split = split_last_day(features)
    assert split.x_train.shape == (8832, 109)
    assert split.y_train.shape == (8832,)
    assert split.x_test.shape == (96, 109)
    assert split.y_test.shape == (96,)


def test_first_feature_row(features: pd.DataFrame) -> None:
    first = features.iloc[0]
    assert first["TimeblockId_1"] == 1
    assert first["HolidayId_0"] == 1
    assert first["DAM_Price"] == 2849.39
    assert first["DAM_Purchase_Bid"] == 10274.1
    assert first["RTM_Price_PD"] == 3673.80
    assert first["RTM_Purchase_Bid_PD"] == 2115.6
    assert first["Demand_PD"] == 145315
    assert first["RTM_Price_SDLW"] == 2089.45
    assert first["RTM_Selling_Bid_SDLW"] == 3145.7
    assert first["RTM_Price_SD"] == 3921.56


def test_test_day_is_the_last_day(features: pd.DataFrame, notebook_frame: pd.DataFrame) -> None:
    split = split_last_day(features)
    last_day = notebook_frame["RTM_Price"].to_numpy()[-96:]
    np.testing.assert_array_equal(split.y_test, last_day)


def _forecast(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pd.read_csv(RESULTS / f"Forecast_{name}.csv")
    actual = table["Actual RTM Price"].to_numpy()
    difference = table["Difference"].to_numpy()
    return actual, actual - difference, difference


def test_average_price_of_the_test_day() -> None:
    actual, _, _ = _forecast("XGB")
    assert np.mean(actual) == pytest.approx(3989.9464583333333, rel=1e-12)


def test_printed_xgboost_evaluation() -> None:
    actual, predicted, difference = _forecast("XGB")
    assert np.max(np.abs(difference)) == pytest.approx(2760.33984375, rel=1e-12)
    assert np.min(np.abs(difference)) == pytest.approx(7.1733691406248, rel=1e-9)
    assert legacy_daily_mae(difference) == pytest.approx(756.5341135152185, rel=1e-9)
    assert quarter_mapes(actual, predicted) == pytest.approx(
        [11.95734952581325, 22.49845547564474, 22.06340889151566, 8.165047204889216], rel=1e-9
    )


def test_printed_random_forest_evaluation() -> None:
    actual, predicted, difference = _forecast("RF")
    assert np.max(np.abs(difference)) == pytest.approx(2598.4647404600973, rel=1e-12)
    assert np.min(np.abs(difference)) == pytest.approx(15.490902050112709, rel=1e-9)
    assert legacy_daily_mae(difference) == pytest.approx(707.4508663797345, rel=1e-9)
    assert quarter_mapes(actual, predicted) == pytest.approx(
        [12.912854536744248, 21.037841105083967, 17.95532627563596, 9.382668273659263], rel=1e-9
    )
    assert mape(actual, predicted) == pytest.approx(15.322172547780857, rel=1e-9)


def test_the_printed_daily_error_leaves_out_one_block() -> None:
    _, _, difference = _forecast("XGB")
    assert mae(np.zeros(96), difference) == pytest.approx(757.87, abs=0.005)
    assert legacy_daily_mae(difference) < mae(np.zeros(96), difference)
    assert mae(np.zeros(96), difference) - legacy_daily_mae(difference) == pytest.approx(
        abs(difference[95]) / 96, rel=1e-9
    )
