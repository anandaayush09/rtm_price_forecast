"""Data contracts. A table that breaks its contract stops the pipeline instead of flowing on."""

import pandas as pd
import pandera.pandas as pa

from rtm_forecast.config import DataParams
from rtm_forecast.ingest import BLOCK_MINUTES, BLOCKS_PER_DAY, RAW_TO_CLEAN, RTM_COLUMNS, TIMEZONE

NUMBER = r"^\d+(\.\d+)?$"
NUMBER_OR_DASH = r"^(\d+(\.\d+)?|-)$"


def raw_schema() -> pa.DataFrameSchema:
    columns: dict[str, pa.Column] = {
        "Date": pa.Column(str, pa.Check.str_matches(r"^\d{2}-\d{2}-\d{4}$")),
        "TimeblockId": pa.Column(str, pa.Check.str_matches(r"^\d{1,2}$")),
    }
    for raw_name, name in RAW_TO_CLEAN.items():
        pattern = NUMBER_OR_DASH if name in RTM_COLUMNS else NUMBER
        columns[raw_name] = pa.Column(str, pa.Check.str_matches(pattern))
    return pa.DataFrameSchema(columns, strict=True)


def _regular_grid(timestamps: pd.Series) -> bool:
    step = pd.Timedelta(minutes=BLOCK_MINUTES)
    return bool((timestamps.diff().dropna() == step).all())


def _full_days(frame: pd.DataFrame) -> bool:
    return bool((frame.groupby("date").size() == BLOCKS_PER_DAY).all())


def clean_schema(params: DataParams) -> pa.DataFrameSchema:
    """Prices are INR/MWh. Bids and demand are MW; the bid columns are volumes judged by scale."""
    columns: dict[str, pa.Column] = {
        "timestamp": pa.Column(
            checks=pa.Check(lambda s: str(s.dt.tz) == TIMEZONE, error="timestamps are not in IST"),
            unique=True,
        ),
        "date": pa.Column(checks=pa.Check(lambda s: s.dt.tz is None, error="date must be naive")),
        "block": pa.Column(int, pa.Check.in_range(1, BLOCKS_PER_DAY)),
        "rtm_missing": pa.Column(bool),
        "demand": pa.Column(float, pa.Check.in_range(params.demand_min, params.demand_max)),
    }
    for name in RAW_TO_CLEAN.values():
        if name == "demand":
            continue
        low, high = (
            (params.price_min, params.price_max)
            if name.endswith("_price")
            else (params.bid_min, params.bid_max)
        )
        columns[name] = pa.Column(float, pa.Check.in_range(low, high), nullable=name in RTM_COLUMNS)
    return pa.DataFrameSchema(
        columns,
        strict=True,
        checks=[
            pa.Check(
                lambda f: _regular_grid(f["timestamp"]),
                error="timestamps are not a 15-minute grid",
            ),
            pa.Check(_full_days, error="a day does not have 96 blocks"),
        ],
    )
