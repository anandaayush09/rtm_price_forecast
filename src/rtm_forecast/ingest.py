"""Read the raw IEX export and turn it into a typed, timestamped table."""

from pathlib import Path

import pandas as pd

TIMEZONE = "Asia/Kolkata"
BLOCK_MINUTES = 15
BLOCKS_PER_DAY = 96
MISSING_TOKEN = "-"

RAW_TO_CLEAN = {
    "DAM_Price": "dam_price",
    "RTM_Price": "rtm_price",
    "RTM_Purchase_Bid": "rtm_purchase_bid",
    "RTM_Selling_Bid": "rtm_selling_bid",
    "DAM_Purchase_Bid": "dam_purchase_bid",
    "DAM_Selling_Bid": "dam_selling_bid",
    "Demand": "demand",
}
RTM_COLUMNS = ("rtm_price", "rtm_purchase_bid", "rtm_selling_bid")


def load_raw(path: Path) -> pd.DataFrame:
    """Read every cell as text so that nothing is converted before the contract has checked it."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse the raw table without filling or dropping anything.

    Dates are day-month-year. Block 1 is taken to start at 00:00 IST, so a row's timestamp is the
    start of its 15-minute interval. The export marks an unpublished RTM value with "-"; it becomes
    NaN and ``rtm_missing`` is set. Any other non-numeric text raises.
    """
    date = pd.to_datetime(raw["Date"], format="%d-%m-%Y")
    block = raw["TimeblockId"].astype(int)
    offset = pd.to_timedelta((block - 1) * BLOCK_MINUTES, unit="min")
    table = pd.DataFrame(
        {
            "timestamp": (date + offset).dt.tz_localize(TIMEZONE),
            "date": date,
            "block": block,
        }
    )
    for raw_name, name in RAW_TO_CLEAN.items():
        text = raw[raw_name]
        table[name] = pd.to_numeric(text.where(text != MISSING_TOKEN), errors="raise").astype(float)
    table["rtm_missing"] = table[list(RTM_COLUMNS)].isna().any(axis=1)
    return table
