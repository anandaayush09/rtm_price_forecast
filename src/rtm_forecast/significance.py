"""Tests of equal forecast accuracy."""

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano test on daily losses for one-day-ahead forecasts.

    ``loss_a`` and ``loss_b`` hold one loss per day (here the day's mean absolute error). The
    statistic uses the Harvey, Leybourne and Newbold small-sample correction. The p-value is
    one-sided: a small value means model A has the lower loss. Returns (statistic, p-value).
    """
    difference = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(difference)
    variance = float(np.var(difference, ddof=1))
    if variance == 0.0:
        mean = float(np.mean(difference))
        if mean == 0.0:
            return 0.0, 0.5
        return (float("-inf"), 0.0) if mean < 0 else (float("inf"), 1.0)
    statistic = float(np.mean(difference) / np.sqrt(variance / n))
    statistic *= float(np.sqrt((n - 1) / n))
    p_value = float(stats.t.cdf(statistic, df=n - 1))
    return statistic, p_value


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm's step-down adjustment of a family of p-values.

    The smallest p-value is multiplied by the number of tests, the next by one less, and so on;
    each adjusted value is at least as large as the one before it and at most 1. Rejecting the
    tests with an adjusted value below 0.05 keeps the chance of any false rejection below 0.05.
    """
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    scaled = (len(p) - np.arange(len(p))) * p[order]
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return adjusted


def bootstrap_difference(
    loss_a: np.ndarray, loss_b: np.ndarray, draws: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """95 percent interval for the mean daily loss of A minus that of B, resampling days."""
    difference = np.asarray(loss_a) - np.asarray(loss_b)
    rng = np.random.default_rng(seed)
    means = rng.choice(difference, size=(draws, len(difference)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _block_indices(days: int, draws: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap: ``draws`` resamples of ``days`` positions in blocks of ``block``."""
    starts = rng.integers(0, days - block + 1, size=(draws, -(-days // block)))
    positions = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)
    return positions[:, :days]


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    block: int = 5,
    draws: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Model confidence set of Hansen, Lunde and Nason (2011) with the maximum-t statistic.

    ``losses`` has one row per day and one column per model. Models are dropped one at a time
    (the one with the largest t statistic against the average of the models still in) until the
    hypothesis of equal accuracy is no longer rejected at ``alpha``. The p-value of the bootstrap
    is taken over moving blocks of ``block`` days, because daily losses are correlated.
    Returns one row per model: the elimination order (0 for the survivors), the set p-value
    and whether the model is in the set. A model is in the set when its p-value is at least
    ``alpha``.
    """
    values = losses.to_numpy(dtype=float)
    days, count = values.shape
    if count < 2:
        raise ValueError("a confidence set needs at least two models")
    rng = np.random.default_rng(seed)
    resampled = np.stack(
        [values[index].mean(axis=0) for index in _block_indices(days, draws, block, rng)]
    )
    mean = values.mean(axis=0)
    alive = list(range(count))
    eliminated: list[int] = []
    p_values = np.zeros(count)
    running = 0.0
    while True:
        part = np.array(alive)
        relative = mean[part] - mean[part].mean()
        relative_boot = resampled[:, part] - resampled[:, part].mean(axis=1, keepdims=True)
        spread = (relative_boot - relative_boot.mean(axis=0)).std(axis=0, ddof=1)
        spread = np.where(spread == 0.0, np.inf, spread)
        statistic = relative / spread
        worst = int(np.argmax(statistic))
        boot_max = ((relative_boot - relative_boot.mean(axis=0)) / spread).max(axis=1)
        p_value = float(np.mean(boot_max >= statistic[worst]))
        running = max(running, p_value)
        if len(alive) == 1 or p_value >= alpha:
            for model in alive:
                p_values[model] = 1.0 if len(alive) == 1 else running
            break
        p_values[alive[worst]] = running
        eliminated.append(alive.pop(worst))
    order = np.zeros(count, dtype=int)
    for step, model in enumerate(eliminated, start=1):
        order[model] = step
    result = pd.DataFrame(
        {"elimination_order": order, "mcs_p_value": p_values}, index=losses.columns
    )
    result["in_set"] = result["mcs_p_value"] >= alpha
    return result.sort_values(["in_set", "mcs_p_value"], ascending=False)
