"""Chronological split of the forecast days."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Periods:
    train: list[pd.Timestamp]
    validation: list[pd.Timestamp]
    test: list[pd.Timestamp]


def make_periods(days: list[pd.Timestamp], validation_days: int, test_days: int) -> Periods:
    """The last ``test_days`` days are the test, the ``validation_days`` before them validation."""
    ordered = sorted(days)
    if validation_days + test_days >= len(ordered):
        raise ValueError("not enough days for the requested validation and test periods")
    test_start = len(ordered) - test_days
    validation_start = test_start - validation_days
    return Periods(
        train=ordered[:validation_start],
        validation=ordered[validation_start:test_start],
        test=ordered[test_start:],
    )


def rolling_folds(days: list[pd.Timestamp], folds: int, fold_days: int) -> list[list[pd.Timestamp]]:
    """The last ``folds * fold_days`` days cut into consecutive folds, earliest fold first.

    A fold is forecast with a model that has seen every earlier day and nothing later.
    """
    ordered = sorted(days)
    if folds < 1 or fold_days < 1 or folds * fold_days >= len(ordered):
        raise ValueError("not enough days for the requested folds")
    first = len(ordered) - folds * fold_days
    return [ordered[first + k * fold_days : first + (k + 1) * fold_days] for k in range(folds)]
