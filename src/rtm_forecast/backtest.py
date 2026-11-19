"""Rolling-origin evaluation: refit on the past, forecast the next days."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

Predictor = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def rolling_origin(
    features: pd.DataFrame,
    days: Sequence[pd.Timestamp],
    predictor: Predictor,
    model: str,
    refit_every: int = 1,
) -> pd.DataFrame:
    """Forecast every block of each day in ``days``.

    For each group of ``refit_every`` consecutive days the predictor gets the training rows
    (published targets of all earlier days) and the rows to forecast, and must return one value
    per row. The result has one row per block with the published price in ``actual`` (NaN if
    none was published).
    """
    parts = []
    ordered = list(days)
    for first in range(0, len(ordered), refit_every):
        chunk = ordered[first : first + refit_every]
        start = chunk[0]
        train = features[(features["date"] < start) & features["target_observed"]]
        test = features[features["date"].isin(chunk)]
        predicted = predictor(train, test)
        if len(predicted) != len(test):
            raise ValueError(f"{model} returned {len(predicted)} values for {len(test)} rows")
        part = test[["date", "block", "rtm_price"]].rename(columns={"rtm_price": "actual"})
        part["prediction"] = np.asarray(predicted, dtype=float)
        part["model"] = model
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
