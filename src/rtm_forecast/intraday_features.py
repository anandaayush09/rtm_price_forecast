"""Inputs for forecasting an RTM block a few blocks after the latest published RTM block.

The day-ahead inputs of ``window_features`` stay: the DAM results of day D and the RTM of earlier
days. On top of them the blocks right before the target are added. ``lead`` is the number of
blocks between the latest known RTM block and the target block, so with ``lead`` 4 the block
4 positions earlier is the latest known one (one hour for 15-minute blocks). The latest known
block may lie on the day before. The exact lead depends on the exchange's gate closure and
result times, which is why the lead is a setting and not fixed here.
"""

from datetime import date

import numpy as np
import pandas as pd

from rtm_forecast.config import FeatureParams
from rtm_forecast.features import fill_rtm
from rtm_forecast.window_features import build_window_features, window_feature_groups

RECENT = 8  # blocks in the rolling summaries (two hours)
INTRADAY_GROUPS = {
    "rtm_recent": [
        "rtm_last_price",
        "rtm_last_price_m1",
        "rtm_last_price_m2",
        "rtm_last_price_m3",
        "rtm_last_price_m7",
        "rtm_last_mean8",
        "rtm_last_max8",
        "rtm_last_min8",
        "rtm_last_cap_share8",
    ],
    "rtm_recent_market": [
        "rtm_last_spread",
        "rtm_last_bid_ratio",
        "dam_change_since_last",
        "rtm_last_dam_adjusted",
    ],
}


def intraday_feature_groups(params: FeatureParams) -> dict[str, list[str]]:
    return {**window_feature_groups(params), **INTRADAY_GROUPS}


def build_intraday_features(
    window: pd.DataFrame,
    params: FeatureParams,
    holidays: frozenset[date],
    lead: int,
    cap: float,
    one_definition: bool = True,
) -> pd.DataFrame:
    """The day-ahead table plus the recent RTM blocks, all at least ``lead`` blocks before the row.

    Missing RTM values are filled inside a day as in the day-ahead table; where the earlier
    block still has no value the input stays missing. ``rtm_last_dam_adjusted`` is the latest
    price moved by the change in the DAM price since that block, kept inside 0 to ``cap``.
    """
    if lead < 1:
        raise ValueError("lead must be at least 1 block")
    base = build_window_features(window, params, holidays, one_definition)
    table = window.sort_values(["date", "block"]).reset_index(drop=True)
    filled = fill_rtm(table, params.imputation)
    price = filled["rtm_price"]
    dam = filled["dam_price"]
    known = price.shift(lead)  # the latest block known when the row is forecast
    at_cap = (price >= cap).astype(float).where(price.notna())
    ratio = (filled["rtm_purchase_bid"] / filled["rtm_selling_bid"]).transform(np.log)
    recent = known.rolling(RECENT, min_periods=RECENT // 2)
    change = dam - dam.shift(lead)
    columns = {
        "rtm_last_price": known,
        "rtm_last_price_m1": price.shift(lead + 1),
        "rtm_last_price_m2": price.shift(lead + 2),
        "rtm_last_price_m3": price.shift(lead + 3),
        "rtm_last_price_m7": price.shift(lead + 7),
        "rtm_last_mean8": recent.mean(),
        "rtm_last_max8": recent.max(),
        "rtm_last_min8": recent.min(),
        "rtm_last_cap_share8": at_cap.shift(lead).rolling(RECENT, min_periods=RECENT // 2).mean(),
        "rtm_last_spread": (price - dam).shift(lead),
        "rtm_last_bid_ratio": ratio.shift(lead),
        "dam_change_since_last": change,
        "rtm_last_dam_adjusted": (known + change).clip(0.0, cap),
    }
    added = pd.DataFrame({"date": table["date"], "block": table["block"], **columns})
    return base.merge(added, on=["date", "block"], how="left", validate="one_to_one")
