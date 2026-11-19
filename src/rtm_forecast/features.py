"""Inputs for forecasting the 96 RTM blocks of a day once that day's DAM results are known.

Information available when day D is forecast: everything published up to the end of day D - 1,
and the day-ahead market (DAM) results for day D. RTM prices, RTM bids and demand of day D are
not available.
"""

from datetime import date

import numpy as np
import pandas as pd

from rtm_forecast.audit import identical_days, stale_mask
from rtm_forecast.calendar import calendar_features
from rtm_forecast.config import FeatureParams
from rtm_forecast.imputation import fill_within_days
from rtm_forecast.ingest import BLOCKS_PER_DAY, RTM_COLUMNS

DAM_COLUMNS = ("dam_price", "dam_purchase_bid", "dam_selling_bid")


def fill_rtm(clean: pd.DataFrame, method: str) -> pd.DataFrame:
    """Fill the unpublished RTM values. ``rtm_missing`` is kept so the fill stays visible."""
    filled = clean.copy()
    for column in RTM_COLUMNS:
        filled[column] = fill_within_days(clean[column].to_numpy(), method)
    return filled


def demand_suspect(clean: pd.DataFrame, params: FeatureParams) -> np.ndarray:
    """Rows whose demand value should not be trusted (see the data audit)."""
    early = clean["date"].dt.date < params.demand_reliable_from
    stale = stale_mask(clean["demand"], params.stale_min_blocks)
    copied = np.zeros(len(clean), dtype=bool)
    for _, later in identical_days(clean, "demand"):
        copied |= (clean["date"].dt.date == date.fromisoformat(later)).to_numpy()
    return np.asarray(early.to_numpy() | stale | copied)


def _day_matrix(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per day, one column per block."""
    return frame.pivot(index="date", columns="block", values=column)


def _lagged(matrix: pd.DataFrame, days: int) -> pd.DataFrame:
    return matrix.shift(days)


def _long(matrix: pd.DataFrame, name: str) -> pd.Series:
    index = pd.MultiIndex.from_product([matrix.index, matrix.columns], names=["date", "block"])
    return pd.Series(matrix.to_numpy().ravel(), index=index, name=name)


def feature_groups(params: FeatureParams) -> dict[str, list[str]]:
    lags = params.lags_days
    return {
        "calendar": ["block", "weekday", "is_holiday"],
        "dam_today": [*DAM_COLUMNS, "dam_price_day_mean", "dam_price_day_max"],
        "dam_change": ["dam_price_change_1d"],
        "rtm_lags": [f"{c}_lag{d}d" for c in RTM_COLUMNS for d in lags]
        + ["rtm_price_prev_day_mean", "rtm_price_prev_day_std"],
        "spread_lag": ["rtm_dam_spread_lag1d"],
        "demand_lags": [f"demand_lag{d}d" for d in lags],
    }


def build_features(
    clean: pd.DataFrame, params: FeatureParams, holidays: frozenset[date]
) -> pd.DataFrame:
    """One row per (day, block) with the target ``rtm_price`` and the inputs known beforehand.

    ``rtm_price`` is the published value and is NaN where the exchange did not publish one
    (``target_observed`` is then False). Gaps are filled only in the lagged inputs.
    """
    filled = fill_rtm(clean, params.imputation)
    if params.demand_policy == "mask":
        filled["demand"] = filled["demand"].mask(demand_suspect(clean, params))
    elif params.demand_policy != "keep":
        raise ValueError(f"unknown demand policy {params.demand_policy!r}")

    price = _day_matrix(filled, "rtm_price")
    dam = _day_matrix(filled, "dam_price")
    columns: list[pd.Series] = []
    for column in DAM_COLUMNS:
        columns.append(_long(_day_matrix(filled, column), column))
    dam_mean = pd.DataFrame(
        np.repeat(dam.mean(axis=1).to_numpy()[:, None], BLOCKS_PER_DAY, axis=1),
        index=dam.index,
        columns=dam.columns,
    )
    dam_max = pd.DataFrame(
        np.repeat(dam.max(axis=1).to_numpy()[:, None], BLOCKS_PER_DAY, axis=1),
        index=dam.index,
        columns=dam.columns,
    )
    columns += [_long(dam_mean, "dam_price_day_mean"), _long(dam_max, "dam_price_day_max")]
    columns.append(_long(dam - _lagged(dam, 1), "dam_price_change_1d"))
    for column in RTM_COLUMNS:
        matrix = _day_matrix(filled, column)
        for days in params.lags_days:
            columns.append(_long(_lagged(matrix, days), f"{column}_lag{days}d"))
    previous = price.shift(1)
    day_mean = price.mean(axis=1).shift(1)
    day_std = price.std(axis=1).shift(1)
    for name, series in [
        ("rtm_price_prev_day_mean", day_mean),
        ("rtm_price_prev_day_std", day_std),
    ]:
        block = pd.DataFrame(
            np.repeat(series.to_numpy()[:, None], BLOCKS_PER_DAY, axis=1),
            index=price.index,
            columns=price.columns,
        )
        columns.append(_long(block, name))
    columns.append(_long(previous - _lagged(dam, 1), "rtm_dam_spread_lag1d"))
    demand = _day_matrix(filled, "demand")
    for days in params.lags_days:
        columns.append(_long(_lagged(demand, days), f"demand_lag{days}d"))

    table = pd.concat(columns, axis=1).reset_index()
    table["timestamp"] = (
        table["date"] + pd.to_timedelta((table["block"] - 1) * 15, unit="min")
    ).dt.tz_localize("Asia/Kolkata")
    table = pd.concat(
        [table, calendar_features(table["date"], holidays)[["weekday", "is_holiday"]]], axis=1
    )
    target = clean.set_index(["date", "block"])["rtm_price"]
    table["rtm_price"] = target.reindex(
        pd.MultiIndex.from_frame(table[["date", "block"]])
    ).to_numpy()
    table["target_observed"] = table["rtm_price"].notna()
    first_day = table["date"].min() + pd.Timedelta(days=params.min_history_days)
    table = table[table["date"] >= first_day].reset_index(drop=True)
    ordered = ["date", "timestamp", "rtm_price", "target_observed"]
    ordered += [name for names in feature_groups(params).values() for name in names]
    return pd.DataFrame(table[ordered])
