"""Prediction intervals: a conformal correction from earlier days, and interval scores.

All functions work on a table with one row per forecast block and the columns ``date``,
``actual`` (NaN if none was published), and the interval ends ``lo`` and ``hi``.
"""

import numpy as np
import pandas as pd


def cap_state(point: np.ndarray, hi50: np.ndarray, hi90: np.ndarray, cap: float) -> np.ndarray:
    """How much the model expects the cap, from quantities known when the day is forecast:
    3 the point forecast is the cap (probability at least 0.5), 2 the upper end of the 50 percent
    interval is the cap (at least 0.25), 1 the upper end of the 90 percent interval is the cap
    (at least 0.05), 0 none of these."""
    return np.select([point >= cap, hi50 >= cap, hi90 >= cap], [3, 2, 1], 0)


def _margin(scores: np.ndarray, coverage: float) -> float:
    """The score at rank ceil((n + 1) * coverage) of n scores (the largest if above n)."""
    count = len(scores)
    if count == 0:
        return float("nan")
    rank = min(int(np.ceil((count + 1) * coverage)), count)
    return float(np.sort(scores)[rank - 1])


def conformal_adjust(
    frame: pd.DataFrame,
    coverage: float,
    window_days: int = 60,
    min_days: int = 14,
    cap: float | None = None,
    group: str | None = None,
    only: str | None = None,
    min_blocks: int = 100,
) -> pd.DataFrame:
    """Widen or narrow each day's interval by a margin learned on the days before it.

    For a day D the conformity score of an earlier block is how far its published price lay
    outside its own interval (negative if inside): ``max(lo - actual, actual - hi)``. The margin is
    the score at rank ceil((n + 1) * coverage) among the n blocks of the last ``window_days``
    days before D (the largest score if that rank exceeds n). The interval of D becomes
    ``[lo - margin, hi + margin]``, cut to ``[0, cap]`` when ``cap`` is given. Days with fewer
    than ``min_days`` earlier days get NaN ends. Only days before D are used, so the margin is
    known when D is forecast. Returns the frame with ``lo_adj`` and ``hi_adj`` added.

    ``group`` names a column that is known when the block is forecast (for example whether the
    cap is expected). The margin of a block then comes from earlier blocks of its own group, which
    gives the coverage inside each group; a group with fewer than ``min_blocks`` earlier blocks
    uses the margin of all blocks. ``only`` names a boolean column (for example "the published
    price is below the cap"): only earlier blocks where it is true teach the margin, which gives
    the coverage among such blocks.
    """
    ordered = frame.sort_values(["date", "block"]).copy()
    observed = ordered["actual"].notna().to_numpy()
    lo, hi = ordered["lo"].to_numpy(dtype=float), ordered["hi"].to_numpy(dtype=float)
    actual = ordered["actual"].to_numpy(dtype=float)
    score = np.where(observed, np.maximum(lo - actual, actual - hi), np.nan)
    teaches = observed & (ordered[only].to_numpy(dtype=bool) if only else True)
    groups = ordered[group].to_numpy() if group else np.zeros(len(ordered), dtype=int)
    dates = pd.DatetimeIndex(ordered["date"])
    days = list(dates.unique().sort_values())
    shift = np.full(len(ordered), np.nan)
    for position, day in enumerate(days):
        recent = days[max(0, position - window_days) : position]
        if len(recent) < min_days:
            continue
        earlier = dates.isin(recent) & teaches
        today = np.asarray(dates == day)
        pooled = _margin(score[earlier], coverage)
        for value in np.unique(groups[today]):
            members = earlier & (groups == value)
            own = _margin(score[members], coverage) if members.sum() >= min_blocks else pooled
            shift[today & (groups == value)] = own
    low, high = lo - shift, hi + shift
    if cap is not None:
        low, high = np.clip(low, 0.0, cap), np.clip(high, 0.0, cap)
    ordered["lo_adj"], ordered["hi_adj"] = low, high
    return ordered


def interval_scores(
    frame: pd.DataFrame, coverage: float, lo: str = "lo", hi: str = "hi"
) -> dict[str, float]:
    """Share of published prices inside the interval, mean width, and the interval score
    (width plus 2 / (1 - coverage) times the distance outside; lower is better)."""
    part = frame[frame["actual"].notna() & frame[lo].notna()]
    actual = part["actual"].to_numpy(dtype=float)
    low, high = part[lo].to_numpy(dtype=float), part[hi].to_numpy(dtype=float)
    outside = np.maximum(low - actual, 0.0) + np.maximum(actual - high, 0.0)
    return {
        "coverage": float(np.mean((actual >= low) & (actual <= high))),
        "width": float(np.mean(high - low)),
        "interval_score": float(np.mean(high - low + 2.0 / (1.0 - coverage) * outside)),
    }
