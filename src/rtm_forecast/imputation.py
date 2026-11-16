"""Ways to fill unpublished RTM values, and a test of each on gaps made on purpose."""

from collections.abc import Callable

import numpy as np
import pandas as pd

BLOCKS_PER_DAY = 96


def ffill(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).ffill().bfill().to_numpy()


def linear(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).interpolate(limit_direction="both").to_numpy()


def yesterday(values: np.ndarray) -> np.ndarray:
    """Same block one day earlier, where that value exists; straight-line fill elsewhere."""
    filled = values.copy()
    for position in np.flatnonzero(np.isnan(values)):
        if position >= BLOCKS_PER_DAY and not np.isnan(values[position - BLOCKS_PER_DAY]):
            filled[position] = values[position - BLOCKS_PER_DAY]
    return np.asarray(linear(filled))


def blend(values: np.ndarray) -> np.ndarray:
    """Mean of the straight-line fill and yesterday's value."""
    return np.asarray((linear(values) + yesterday(values)) / 2)


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "ffill": ffill,
    "linear": linear,
    "yesterday": yesterday,
    "blend": blend,
}


def fill(values: np.ndarray, method: str) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"unknown imputation method {method!r}; choose from {sorted(METHODS)}")
    return METHODS[method](values)


def fill_within_days(values: np.ndarray, method: str) -> np.ndarray:
    """Fill each day using only values up to the end of that day.

    A forecast for day D may use filled values of day D - 1 or earlier, so a fill must not look at
    values from a later day. A gap that runs to the end of a day is filled from the values
    before it.
    """
    filled = values.astype(float).copy()
    for first in range(0, len(values), BLOCKS_PER_DAY):
        last = first + BLOCKS_PER_DAY
        begin = max(0, first - 2 * BLOCKS_PER_DAY)
        window = np.concatenate([filled[begin:first], values[first:last]])
        filled[first:last] = fill(window, method)[first - begin :]
    return filled


def candidate_gaps(values: np.ndarray, length: int) -> np.ndarray:
    """Start positions of gaps that can be tested.

    A gap starts on an odd block (index 0, 2, 4 ... within a day), as the real gaps do, and
    the gap stays inside one day, and its values, the values around it and the values one day
    earlier must be known.
    """
    known = ~np.isnan(values)
    starts = []
    for start in range(BLOCKS_PER_DAY, len(values) - length - 1):
        if start % 2 != 0 or start % BLOCKS_PER_DAY + length > BLOCKS_PER_DAY:
            continue
        window = known[start - 1 : start + length + 1]
        day_before = known[start - BLOCKS_PER_DAY : start - BLOCKS_PER_DAY + length]
        if window.all() and day_before.all():
            starts.append(start)
    return np.array(starts)


def evaluate_methods(values: np.ndarray, length: int, trials: int, seed: int) -> pd.DataFrame:
    """Hide ``length`` known values at random places, fill them, and measure the error.

    Each trial hides one gap, so the other known values stay as they are. A method only sees values
    up to the end of the day that holds the gap, as in the feature builder.
    """
    rng = np.random.default_rng(seed)
    starts = rng.choice(candidate_gaps(values, length), size=trials, replace=False)
    errors: dict[str, list[float]] = {name: [] for name in METHODS}
    relative: dict[str, list[float]] = {name: [] for name in METHODS}
    for start in starts:
        hidden = values.copy()
        truth = values[start : start + length]
        hidden[start : start + length] = np.nan
        window = slice(
            max(0, start - 3 * BLOCKS_PER_DAY), (start // BLOCKS_PER_DAY + 1) * BLOCKS_PER_DAY
        )
        local = hidden[window]
        offset = start - window.start
        for name, method in METHODS.items():
            guess = method(local)[offset : offset + length]
            errors[name].extend(np.abs(guess - truth))
            relative[name].extend(np.abs(guess - truth) / truth * 100)
    return pd.DataFrame(
        {
            "MAE": {name: float(np.mean(e)) for name, e in errors.items()},
            "MAPE": {name: float(np.mean(r)) for name, r in relative.items()},
            "median abs error": {name: float(np.median(e)) for name, e in errors.items()},
        }
    )
