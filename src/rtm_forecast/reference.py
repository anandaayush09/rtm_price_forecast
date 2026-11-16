"""Port of the original notebook: data preparation, features, split and evaluation.

The port keeps the behaviour of the first notebook, including its known defects, so that its
printed numbers can be reproduced and each defect can be switched off on its own. Nothing here
is meant to be a good way to forecast prices. The corrected pipeline lives in the other modules.

Defects kept on purpose:
    * dates are parsed month-first whenever that is possible (``dates="month_first"``)
    * the Saturday and Sunday flag and the six holiday dates follow from those parsed dates
    * the "mean absolute error of the day" adds up 95 of the 96 errors (``legacy_daily_mae``)
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

BLOCKS_PER_DAY = 96
DAYS_PER_WEEK = 7
N_TRAIN_DAYS = 92
N_PREDICTION_BLOCKS = 96

# The six dates the notebook marked as special days.
NOTEBOOK_HOLIDAYS = (
    pd.Timestamp("2020-10-02"),
    pd.Timestamp("2020-10-25"),
    pd.Timestamp("2020-10-30"),
    pd.Timestamp("2020-11-14"),
    pd.Timestamp("2020-12-25"),
    pd.Timestamp("2020-12-31"),
)

NUMERIC_COLUMNS = (
    "RTM_Price",
    "DAM_Price",
    "RTM_Purchase_Bid",
    "RTM_Selling_Bid",
    "DAM_Purchase_Bid",
    "DAM_Selling_Bid",
    "Demand",
)

# Order of the 109 model inputs in the notebook, followed by the target.
FEATURE_TAIL = (
    "DAM_Price",
    "DAM_Purchase_Bid",
    "DAM_Selling_Bid",
    "RTM_Price_PD",
    "RTM_Purchase_Bid_PD",
    "Demand_PD",
    "RTM_Selling_Bid_PD",
    "RTM_Price_SDLW",
    "RTM_Purchase_Bid_SDLW",
    "RTM_Selling_Bid_SDLW",
)
TARGET = "RTM_Price_SD"

# Grid-search winners printed by the notebook (2021 environment). The learning rate comment in the
# notebook (0.05) is not the winner; 0.07 is.
XGB_PARAMS = {
    "colsample_bytree": 0.5,
    "gamma": 0.05,
    "learning_rate": 0.07,
    "max_depth": 4,
    "n_estimators": 100,
}
RF_PARAMS = {
    "bootstrap": True,
    "max_depth": 10,
    "max_features": 18,
    "min_samples_leaf": 4,
    "min_samples_split": 36,
    "n_estimators": 700,
}

DateParsing = Literal["month_first", "day_first"]


def parse_dates(raw_dates: pd.Series, dates: DateParsing) -> pd.Series:
    """Parse the raw dd-mm-yyyy strings.

    ``month_first`` parses each value on its own and reads the day as the month whenever that
    is valid.
    That is what the notebook's ``pd.to_datetime`` did: 01-10-2020 became 10 January 2020.
    """
    if dates == "month_first":
        return pd.to_datetime(raw_dates, format="mixed", dayfirst=False)
    return pd.to_datetime(raw_dates, format="%d-%m-%Y")


def prepare_frame(
    raw: pd.DataFrame,
    dates: DateParsing = "month_first",
    holidays: Sequence[pd.Timestamp] = NOTEBOOK_HOLIDAYS,
) -> pd.DataFrame:
    """Numeric conversion, Saturday/Sunday and holiday flag, forward fill (notebook cells 22 to 36).

    ``HolidayId`` is 0 for Monday to Friday, 1 for Saturday and Sunday and 2 on the given holidays.
    Unpublished RTM values ("-") become NaN and are then filled from the previous block.
    """
    frame = raw.copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["TimeblockId"] = frame["TimeblockId"].astype(int)
    frame["Date"] = parse_dates(raw["Date"], dates)
    frame["HolidayId"] = (frame["Date"].dt.day_of_week >= 5).astype(int)
    frame = frame.ffill(axis=0)
    frame.loc[frame["Date"].isin(list(holidays)), "HolidayId"] = 2
    return frame


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Previous-day and same-day-last-week inputs plus one-hot block and day type (cells 52 to 59).

    The first seven days are used only to build lags. The result has 9984 rows: 109 inputs and
    the target.
    """
    start = DAYS_PER_WEEK * BLOCKS_PER_DAY
    day = BLOCKS_PER_DAY
    week = DAYS_PER_WEEK * BLOCKS_PER_DAY
    built = pd.DataFrame(
        {
            "TimeblockId": frame["TimeblockId"],
            "HolidayId": frame["HolidayId"],
            "DAM_Price": frame["DAM_Price"],
            "DAM_Purchase_Bid": frame["DAM_Purchase_Bid"],
            "DAM_Selling_Bid": frame["DAM_Selling_Bid"],
            "RTM_Price_PD": frame["RTM_Price"].shift(day),
            "RTM_Purchase_Bid_PD": frame["RTM_Purchase_Bid"].shift(day),
            "RTM_Selling_Bid_PD": frame["RTM_Selling_Bid"].shift(day),
            "Demand_PD": frame["Demand"].shift(day),
            "RTM_Price_SDLW": frame["RTM_Price"].shift(week),
            "RTM_Purchase_Bid_SDLW": frame["RTM_Purchase_Bid"].shift(week),
            "RTM_Selling_Bid_SDLW": frame["RTM_Selling_Bid"].shift(week),
            TARGET: frame["RTM_Price"],
        }
    ).iloc[start:]
    built = built.reset_index(drop=True)
    blocks = pd.get_dummies(
        pd.Categorical(built["TimeblockId"], categories=range(1, BLOCKS_PER_DAY + 1)),
        prefix="TimeblockId",
    )
    day_types = pd.get_dummies(
        pd.Categorical(built["HolidayId"], categories=[0, 1, 2]), prefix="HolidayId"
    )
    dummies = pd.concat([blocks, day_types], axis=1).astype(int)
    return pd.concat([dummies, built[[*FEATURE_TAIL, TARGET]]], axis=1)


@dataclass(frozen=True)
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def split_last_day(
    features: pd.DataFrame,
    n_train_days: int = N_TRAIN_DAYS,
    n_prediction_blocks: int = N_PREDICTION_BLOCKS,
) -> Split:
    """The notebook's split: ``n_train_days`` days of training, then the last day as the test."""
    values = features.iloc[:, :-1].to_numpy()
    target = features.iloc[:, -1].to_numpy()
    day = BLOCKS_PER_DAY
    train = slice(-(n_train_days + 1) * day, -day)
    test = (
        slice(-day, None) if n_prediction_blocks == day else slice(-day, -day + n_prediction_blocks)
    )
    return Split(values[train], target[train], values[test], target[test])


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error in percent. The notebook calls 100 minus this 'accuracy'."""
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def legacy_daily_mae(errors: np.ndarray, blocks: int = BLOCKS_PER_DAY) -> float:
    """The notebook's 'Mean Absolute Error of the day'.

    Its loop closes the day on the last block without adding that block's error, so it sums the
    first ``blocks - 1`` errors and divides by ``blocks``.
    """
    return float(np.abs(errors[: blocks - 1]).sum() / blocks)


def quarter_mapes(actual: np.ndarray, predicted: np.ndarray) -> list[float]:
    """MAPE for 00:00-06:00, 06:00-12:00, 12:00-18:00 and 18:00-24:00 (24 blocks each)."""
    bounds = [(0, 24), (24, 48), (48, 72), (72, len(actual))]
    return [mape(actual[a:b], predicted[a:b]) for a, b in bounds]
