"""The price table that goes to the GPU notebook: RTM and DAM prices and the fold labels only.

No demand or generation value is included, so nothing from the Grid-India sheets leaves the
machine. Prices come from the joined window table.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rtm_forecast.features import fill_rtm
from rtm_forecast.splits import Periods

COLUMNS = [
    "timestamp",
    "date",
    "block",
    "rtm_price",
    "rtm_price_filled",
    "dam_price",
    "fold",
    "period",
]


def build_export(
    window: pd.DataFrame,
    folds: Sequence[Sequence[pd.Timestamp]],
    periods: Periods,
    imputation: str,
) -> pd.DataFrame:
    """One row per block of the window.

    ``rtm_price`` is the published price (missing where none was published) and is the value a
    forecast is scored against. ``rtm_price_filled`` is the history a model may read. ``fold``
    is 1 to ``len(folds)`` on the days to forecast and missing before.
    """
    filled = fill_rtm(window, imputation)
    table = pd.DataFrame(
        {
            "timestamp": window["timestamp"],
            "date": window["date"],
            "block": window["block"],
            "rtm_price": window["rtm_price"],
            "rtm_price_filled": filled["rtm_price"],
            "dam_price": window["dam_price"],
        }
    )
    fold_of = {day: k + 1 for k, fold in enumerate(folds) for day in fold}
    table["fold"] = table["date"].map(fold_of).astype("Int64")
    table["period"] = np.select(
        [table["date"].isin(periods.test), table["date"].isin(periods.validation)],
        ["test", "validation"],
        np.where(table["fold"].notna(), "earlier", ""),
    )
    return table[COLUMNS]
