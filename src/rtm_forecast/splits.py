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
