"""Holiday list and calendar features."""

from datetime import date
from pathlib import Path

import pandas as pd
import yaml


def load_holidays(path: Path = Path("config/holidays.yaml")) -> frozenset[date]:
    with path.open(encoding="utf-8") as handle:
        entries = yaml.safe_load(handle)["holidays"]
    return frozenset(entry["date"] for entry in entries)


def calendar_features(dates: pd.Series, holidays: frozenset[date]) -> pd.DataFrame:
    """Day of week (Monday = 0) and a holiday flag for a series of dates."""
    return pd.DataFrame(
        {
            "weekday": dates.dt.day_of_week.astype(int),
            "is_holiday": dates.dt.date.isin(holidays).astype(int),
        },
        index=dates.index,
    )
