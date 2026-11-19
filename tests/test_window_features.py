from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rtm_forecast.config import (
    FeatureParams,
    load_feature_params,
    load_window_experiment_params,
)
from rtm_forecast.iex_export import parse_snapshot
from rtm_forecast.metrics import cap_scores
from rtm_forecast.psp import parse_timeseries
from rtm_forecast.splits import rolling_folds
from rtm_forecast.window import build_window
from rtm_forecast.window_features import (
    EXTRA_GROUPS,
    build_window_features,
    one_demand_definition,
    window_feature_groups,
)
from test_iex_export import sheet as iex_sheet
from test_psp import sheet as psp_sheet

ROOT = Path(__file__).resolve().parents[1]
FIRST = date(2026, 6, 1)
DAYS = 12


@pytest.fixture(scope="module")
def feature_params() -> FeatureParams:
    return load_feature_params(ROOT / "params.yaml")


def _window() -> pd.DataFrame:
    rtm_parts, dam_parts, psp_parts = [], [], []
    for offset in range(DAYS):
        day = FIRST + timedelta(days=offset)
        text = day.strftime("%d-%m-%Y")
        price = str(2000 + 100 * offset)
        rtm_parts.append(parse_snapshot(iex_sheet(text, price=price), "rtm"))
        dam_parts.append(parse_snapshot(iex_sheet(text, market="dam", price=price), "dam"))
        report = (day + timedelta(days=1)).strftime("%d-%b-%Y")
        psp_parts.append(parse_timeseries(psp_sheet(3, report=report)))
    return build_window(
        pd.concat(rtm_parts, ignore_index=True),
        pd.concat(dam_parts, ignore_index=True),
        pd.concat(psp_parts, ignore_index=True),
        FIRST,
        FIRST + timedelta(days=DAYS - 1),
    )


def test_storage_load_is_taken_out_of_demand() -> None:
    window = _window()
    assert one_demand_definition(window).eq(200000.0 - 100.0).all()
    window["storage_demand"] = float("nan")
    assert one_demand_definition(window).eq(200000.0).all()


def test_extra_inputs_exist_and_a_day_never_sees_itself(
    feature_params: FeatureParams,
) -> None:
    holidays: frozenset[date] = frozenset()
    window = _window()
    features = build_window_features(window, feature_params, holidays)
    columns = sum(window_feature_groups(feature_params).values(), [])
    assert set(sum(EXTRA_GROUPS.values(), [])) <= set(columns)
    assert features["date"].min() == pd.Timestamp(FIRST) + pd.Timedelta(
        days=feature_params.min_history_days
    )
    changed_day = pd.Timestamp(FIRST) + pd.Timedelta(days=9)
    altered = window.copy()
    mask = altered["date"] == changed_day
    for column in ("rtm_price", "rtm_purchase_bid", "wind", "solar", "demand"):
        altered.loc[mask, column] = altered.loc[mask, column] * 1.5
    after = build_window_features(altered, feature_params, holidays)
    lagged = [c for c in columns if "lag" in c or "prev_day" in c or c == "rtm_dam_spread_lag1d"]
    same_day = features["date"] <= changed_day
    pd.testing.assert_frame_equal(
        features.loc[same_day, lagged].reset_index(drop=True),
        after.loc[after["date"] <= changed_day, lagged].reset_index(drop=True),
    )
    next_day = features["date"] == changed_day + pd.Timedelta(days=1)
    assert not np.allclose(features.loc[next_day, "wind_lag1d"], after.loc[next_day, "wind_lag1d"])
    assert np.isfinite(features["dam_bid_ratio"]).all()


def test_rolling_folds_are_consecutive_and_end_on_the_last_day() -> None:
    days = list(pd.date_range("2026-01-01", periods=100))
    folds = rolling_folds(days, 3, 10)
    assert [len(f) for f in folds] == [10, 10, 10]
    assert folds[-1][-1] == days[-1] and folds[1][0] == folds[0][-1] + pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        rolling_folds(days, 10, 10)


def test_cap_scores_count_cap_blocks_and_forecasts_of_the_cap() -> None:
    frame = pd.DataFrame(
        {
            "model": "m",
            "actual": [10000.0, 10000.0, 3000.0, 3000.0, np.nan],
            "prediction": [9800.0, 4000.0, 9700.0, 3100.0, 10000.0],
        }
    )
    row = cap_scores(frame, 10000.0, 0.95).loc["m"]
    assert row["cap_share"] == 0.5
    assert row["cap_precision"] == 0.5 and row["cap_recall"] == 0.5 and row["cap_f1"] == 0.5
    assert row["MAE_cap"] == (200 + 6000) / 2 and row["MAE_other"] == (6700 + 100) / 2


def test_experiment_params_load() -> None:
    params = load_window_experiment_params(ROOT / "params.yaml")
    assert params.test_days == 60 and params.folds * params.fold_days == 150
    assert params.price_cap == 10000.0 and params.naive_reference == "rtm_price_lag1d"
