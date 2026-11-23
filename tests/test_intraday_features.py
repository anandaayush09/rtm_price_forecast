from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rtm_forecast.config import FeatureParams, load_feature_params
from rtm_forecast.intraday_features import (
    INTRADAY_GROUPS,
    build_intraday_features,
    intraday_feature_groups,
)
from test_window_features import FIRST, _window

ROOT = Path(__file__).resolve().parents[1]
CAP = 10000.0
HOLIDAYS: frozenset[date] = frozenset()


@pytest.fixture(scope="module")
def feature_params() -> FeatureParams:
    return load_feature_params(ROOT / "params.yaml")


def _ramp_window() -> pd.DataFrame:
    """The test window with a price that rises by 1 each block, so every block is distinct."""
    window = _window()
    steps = np.arange(len(window), dtype=float)
    window["rtm_price"] = 100.0 + steps
    window["dam_price"] = 50.0 + 0.5 * steps
    return window


def _position(frame: pd.DataFrame) -> np.ndarray:
    """Number of blocks since the first block of the window."""
    days = (frame["date"] - pd.Timestamp(FIRST)).dt.days.to_numpy()
    return np.asarray(days * 96 + frame["block"].to_numpy() - 1)


@pytest.mark.parametrize("lead", [1, 4])
def test_recent_blocks_are_the_blocks_lead_positions_back_even_across_midnight(
    feature_params: FeatureParams, lead: int
) -> None:
    features = build_intraday_features(_ramp_window(), feature_params, HOLIDAYS, lead, CAP)
    position = _position(features)
    assert (features["rtm_last_price"] == 100.0 + position - lead).all()
    assert (features["rtm_last_price_m3"] == 100.0 + position - lead - 3).all()
    first_block = features[features["block"] == 1]
    assert (first_block["rtm_last_price"] == 100.0 + _position(first_block) - lead).all()
    # the dam moves by 0.5 a block, so the change since the last known block is 0.5 * lead
    assert np.allclose(features["dam_change_since_last"], 0.5 * lead)
    assert np.allclose(features["rtm_last_spread"], (100.0 - 50.0) + 0.5 * (position - lead))


def test_a_block_never_sees_prices_from_the_last_lead_blocks(feature_params: FeatureParams) -> None:
    lead, changed = 4, 900  # inside the days that have feature rows
    window = _ramp_window()
    altered = window.copy()
    altered.loc[changed:, "rtm_price"] = altered.loc[changed:, "rtm_price"] + 777.0
    before = build_intraday_features(window, feature_params, HOLIDAYS, lead, CAP)
    after = build_intraday_features(altered, feature_params, HOLIDAYS, lead, CAP)
    columns = sum(INTRADAY_GROUPS.values(), [])
    # a block that is ``lead`` blocks after the first changed block is the first that may differ
    early = _position(before) < changed + lead
    pd.testing.assert_frame_equal(
        before.loc[early, columns].reset_index(drop=True),
        after.loc[early, columns].reset_index(drop=True),
    )
    assert early.sum() > 0 and (~early).sum() > 0
    late_before, late_after = before.loc[~early], after.loc[~early]
    assert not np.allclose(late_before["rtm_last_price"], late_after["rtm_last_price"])


def test_dam_adjusted_forecast_stays_inside_the_price_range(feature_params: FeatureParams) -> None:
    window = _ramp_window()
    window["dam_price"] = np.where(window["block"] % 2 == 0, 9990.0, 5.0)
    window["rtm_price"] = np.where(window["block"] % 3 == 0, 9990.0, 5.0)
    features = build_intraday_features(window, feature_params, HOLIDAYS, 2, CAP)
    assert features["rtm_last_dam_adjusted"].between(0.0, CAP).all()
    assert features["rtm_last_cap_share8"].dropna().between(0.0, 1.0).all()


def test_groups_extend_the_day_ahead_groups_and_the_lead_is_checked(
    feature_params: FeatureParams,
) -> None:
    groups = intraday_feature_groups(feature_params)
    assert set(INTRADAY_GROUPS) <= set(groups) and "dam_today" in groups
    features = build_intraday_features(_ramp_window(), feature_params, HOLIDAYS, 2, CAP)
    assert set(sum(groups.values(), [])) <= set(features.columns)
    with pytest.raises(ValueError):
        build_intraday_features(_ramp_window(), feature_params, HOLIDAYS, 0, CAP)
