"""Data contracts. A table that breaks its contract stops the pipeline instead of flowing on."""

import pandas as pd
import pandera.pandas as pa

from rtm_forecast.config import DataParams
from rtm_forecast.ingest import BLOCK_MINUTES, BLOCKS_PER_DAY, RAW_TO_CLEAN, RTM_COLUMNS, TIMEZONE

# ASSUMPTION: a source reading below -5000 MW is not a metering offset.
NEAR_ZERO_FLOOR = -5000.0
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


def _timestamp_column() -> pa.Column:
    return pa.Column(
        checks=pa.Check(lambda s: str(s.dt.tz) == TIMEZONE, error="timestamps are not in IST"),
        unique=True,
    )


def iex_schema(market: str, params: DataParams) -> pa.DataFrameSchema:
    """A parsed IEX snapshot. Days may be incomplete (RTM publishes in half-hour sessions)."""
    volume = pa.Check.in_range(params.bid_min, params.bid_max)
    columns: dict[str, pa.Column] = {
        "date": pa.Column(checks=pa.Check(lambda s: s.dt.tz is None, error="date must be naive")),
        "block": pa.Column(int, pa.Check.in_range(1, BLOCKS_PER_DAY)),
        "timestamp": _timestamp_column(),
        "hour": pa.Column(int, pa.Check.in_range(1, 24)),
        "purchase_bid": pa.Column(float, volume),
        "sell_bid": pa.Column(float, volume),
        "mcv": pa.Column(float, volume),
        "scheduled_volume": pa.Column(float, volume),
        "mcp": pa.Column(float, pa.Check.in_range(params.price_min, params.price_max)),
    }
    checks = [
        pa.Check(
            lambda f: ((f["block"] - 1) // 4 + 1 == f["hour"]).all(),
            error="hour does not match block",
        ),
        pa.Check(
            lambda f: not f.duplicated(["date", "block"]).any(), error="a block appears twice"
        ),
    ]
    if market == "rtm":
        columns["session_id"] = pa.Column(int, pa.Check.in_range(1, BLOCKS_PER_DAY // 2))
        # Observed on the sample day: a session is two consecutive blocks from :00 or :30.
        checks.append(
            pa.Check(
                lambda f: ((f["block"] - 1) // 2 + 1 == f["session_id"]).all(),
                error="session id does not match block",
            )
        )
    return pa.DataFrameSchema(columns, strict=True, checks=checks)


def psp_schema(params: DataParams) -> pa.DataFrameSchema:
    """Parsed daily power-supply sheets: at most 96 readings a day, MW except frequency (Hz)."""
    non_negative = pa.Check.ge(0)
    # Solar, storage and the others group read slightly below zero at night or while charging.
    near_zero = pa.Check.ge(NEAR_ZERO_FLOOR)
    columns: dict[str, pa.Column] = {
        "date": pa.Column(checks=pa.Check(lambda s: s.dt.tz is None, error="date must be naive")),
        "block": pa.Column(int, pa.Check.in_range(1, BLOCKS_PER_DAY)),
        "timestamp": _timestamp_column(),
        "layout": pa.Column(int, pa.Check.in_range(1, 3)),
        # A reading of 0 Hz is a telemetry loss (4 in the year); psp.suspect_rows flags it.
        "frequency": pa.Column(float, pa.Check.in_range(0.0, 52.0)),
        "demand_met": pa.Column(float, pa.Check.in_range(params.demand_min, params.demand_max)),
        "net_exchange": pa.Column(float),
    }
    for name in ("nuclear", "wind", "gas", "thermal", "net_demand_met", "total_generation"):
        columns[name] = pa.Column(float, non_negative)
    # Hydro had one reading of -4141 MW on 30 Apr 2026; psp.suspect_rows flags negative hydro.
    for name in ("solar", "hydro", "others"):
        columns[name] = pa.Column(float, near_zero)
    columns["storage_demand"] = pa.Column(float, non_negative, nullable=True)
    columns["storage_generation"] = pa.Column(float, near_zero, nullable=True)
    return pa.DataFrameSchema(
        columns,
        strict=True,
        checks=[
            pa.Check(
                lambda f: bool((f.groupby("date").size() <= BLOCKS_PER_DAY).all()),
                error="a day has more than 96 readings",
            ),
            pa.Check(
                lambda f: not f.duplicated(["date", "block"]).any(), error="a reading appears twice"
            ),
        ],
    )


def window_schema(params: DataParams) -> pa.DataFrameSchema:
    """The joined 2025-26 table: a full 96-block grid, values missing where a source has none."""
    non_negative = pa.Check.ge(0)
    near_zero = pa.Check.ge(NEAR_ZERO_FLOOR)
    volume = pa.Check.in_range(params.bid_min, params.bid_max)
    columns: dict[str, pa.Column] = {
        "timestamp": _timestamp_column(),
        "date": pa.Column(checks=pa.Check(lambda s: s.dt.tz is None, error="date must be naive")),
        "block": pa.Column(int, pa.Check.in_range(1, BLOCKS_PER_DAY)),
    }
    for market in ("rtm", "dam"):
        columns[f"{market}_price"] = pa.Column(
            float, pa.Check.in_range(params.price_min, params.price_max), nullable=True
        )
        for name in ("purchase_bid", "selling_bid", "mcv", "scheduled_volume"):
            columns[f"{market}_{name}"] = pa.Column(float, volume, nullable=True)
        columns[f"{market}_empty"] = pa.Column(bool)
        columns[f"{market}_missing"] = pa.Column(bool)
    columns["demand"] = pa.Column(
        float, pa.Check.in_range(params.demand_min, params.demand_max), nullable=True
    )
    # Readings outside 48 to 52 Hz are masked as telemetry loss before this point.
    columns["frequency"] = pa.Column(float, pa.Check.in_range(48.0, 52.0), nullable=True)
    for name in ("nuclear", "wind", "gas", "thermal", "net_demand_met", "total_generation"):
        columns[name] = pa.Column(float, non_negative, nullable=True)
    for name in ("solar", "hydro", "others", "storage_generation"):
        columns[name] = pa.Column(float, near_zero, nullable=True)
    columns["storage_demand"] = pa.Column(float, non_negative, nullable=True)
    columns["net_exchange"] = pa.Column(float, nullable=True)
    columns["demand_suspect"] = pa.Column(bool)
    columns["demand_missing"] = pa.Column(bool)
    columns["psp_layout"] = pa.Column("Int64", pa.Check.in_range(1, 3), nullable=True)
    return pa.DataFrameSchema(
        columns,
        strict=True,
        checks=[
            pa.Check(
                lambda f: _regular_grid(f["timestamp"]),
                error="timestamps are not a 15-minute grid",
            ),
            pa.Check(_full_days, error="a day does not have 96 blocks"),
            pa.Check(
                lambda f: bool((f["demand_missing"] == f["demand"].isna()).all()),
                error="demand_missing does not match demand",
            ),
            pa.Check(
                lambda f: bool(f.loc[f["demand_suspect"], "demand"].isna().all()),
                error="a suspect reading kept its demand",
            ),
            pa.Check(
                lambda f: bool(f.loc[f["rtm_empty"], "rtm_price"].isna().all()),
                error="an empty RTM block kept its price",
            ),
        ],
    )
