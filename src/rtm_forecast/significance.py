"""Tests of equal forecast accuracy."""

import numpy as np
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


def bootstrap_difference(
    loss_a: np.ndarray, loss_b: np.ndarray, draws: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """95 percent interval for the mean daily loss of A minus that of B, resampling days."""
    difference = np.asarray(loss_a) - np.asarray(loss_b)
    rng = np.random.default_rng(seed)
    means = rng.choice(difference, size=(draws, len(difference)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)
