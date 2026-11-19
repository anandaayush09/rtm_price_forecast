"""Forecast-time inputs for the 2025-26 window.

Same rule as ``features.py``: when day D is forecast, everything up to the end of day D - 1
is known, plus the day-ahead market (DAM) results for day D. The joined window table has the
column names of the old clean table, so the old builder is reused. A few inputs that the audit
pointed to are added: the DAM bid ratio of day D, and wind, solar and net demand of earlier days.
"""

import dataclasses
from datetime import date

import numpy as np
import pandas as pd

from rtm_forecast.config import FeatureParams
from rtm_forecast.features import (
    _day_matrix,
    _lagged,
    _long,
    build_features,
    feature_groups,
    fill_rtm,
)

EXTRA_GROUPS = {
    "dam_bids": ["dam_bid_ratio"],
    "rtm_bids": ["rtm_bid_ratio_lag1d"],
    "weather_lags": ["wind_lag1d", "wind_lag7d", "solar_lag1d", "solar_lag7d"],
    "net_demand": ["net_demand_lag1d", "net_demand_lag7d"],
}


def one_demand_definition(window: pd.DataFrame) -> pd.Series:
    """Demand without storage load. From June 2026 the sheet counts it; before that it did not."""
    return window["demand"] - window["storage_demand"].fillna(0.0)


def window_feature_groups(params: FeatureParams) -> dict[str, list[str]]:
    return {**feature_groups(params), **EXTRA_GROUPS}


def build_window_features(
    window: pd.DataFrame, params: FeatureParams, holidays: frozenset[date]
) -> pd.DataFrame:
    """One row per (day, block) with the target ``rtm_price`` and the inputs known beforehand.

    Missing demand, wind and solar are not filled: the lagged inputs stay missing where the
    sheet had no usable reading, and a model that cannot take missing values leaves them out.
    """
    table = window.copy()
    table["demand"] = one_demand_definition(window)
    # Missing demand was masked when the window was built, so the old test for stale or copied
    # demand and for early days is switched off.
    base = build_features(table, dataclasses.replace(params, demand_policy="keep"), holidays)

    demand = _day_matrix(table, "demand")
    wind = _day_matrix(table, "wind")
    solar = _day_matrix(table, "solar")
    net = demand - wind - solar
    filled = fill_rtm(table, params.imputation)
    rtm_ratio = (
        _day_matrix(filled, "rtm_purchase_bid") / _day_matrix(filled, "rtm_selling_bid")
    ).transform(np.log)
    dam_ratio = (
        _day_matrix(table, "dam_purchase_bid") / _day_matrix(table, "dam_selling_bid")
    ).transform(np.log)
    extra: dict[str, pd.DataFrame] = {
        "dam_bid_ratio": dam_ratio,
        "rtm_bid_ratio_lag1d": _lagged(rtm_ratio, 1),
        "wind_lag1d": _lagged(wind, 1),
        "wind_lag7d": _lagged(wind, 7),
        "solar_lag1d": _lagged(solar, 1),
        "solar_lag7d": _lagged(solar, 7),
        "net_demand_lag1d": _lagged(net, 1),
        "net_demand_lag7d": _lagged(net, 7),
    }
    added = pd.concat([_long(matrix, name) for name, matrix in extra.items()], axis=1)
    return base.merge(added.reset_index(), on=["date", "block"], how="left", validate="one_to_one")
