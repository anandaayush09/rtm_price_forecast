"""Read the market snapshot spreadsheets exported from the IEX website (RTM and DAM).

Cells are read as text. The table is found by its header row, so the title lines above it and the
summary rows below it do not matter, and a range export that holds several days is read the same way
as a one-day file. A header that is not exactly the known one stops the load (schema drift).
"""

import re
from pathlib import Path
from typing import Literal

import pandas as pd

from rtm_forecast.ingest import BLOCK_MINUTES, BLOCKS_PER_DAY, MISSING_TOKEN, TIMEZONE

Market = Literal["rtm", "dam"]

HEADER_TO_NAME = {
    "date": "date",
    "hour": "hour",
    "session id": "session_id",
    "time block": "time_block",
    "purchase bid (mw)": "purchase_bid",
    "sell bid (mw)": "sell_bid",
    "mcv (mw)": "mcv",
    "final scheduled volume (mw)": "scheduled_volume",
    "mcp (rs/mwh)": "mcp",
}
EXPECTED = {
    "rtm": set(HEADER_TO_NAME.values()),
    "dam": set(HEADER_TO_NAME.values()) - {"session_id"},
}
NUMERIC = ("purchase_bid", "sell_bid", "mcv", "scheduled_volume", "mcp")
DATE_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{4}$")
BLOCK_PATTERN = re.compile(r"^(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2})$")


class SchemaDriftError(ValueError):
    """The export no longer has the columns or layout this loader was written for."""


def read_sheet(path: Path) -> pd.DataFrame:
    """The first sheet as text, with no header row assumed."""
    return pd.read_excel(path, sheet_name=0, header=None, dtype=str, engine="calamine")


def _header_name(cell: object) -> str:
    text = str(cell).replace("\n", " ").strip().rstrip("*").strip().lower()
    return re.sub(r"\s+", " ", text)


def _find_header(raw: pd.DataFrame) -> tuple[int, list[str]]:
    for position in range(len(raw)):
        names = [_header_name(cell) for cell in raw.iloc[position]]
        if "time block" in names:
            return position, names
    raise SchemaDriftError("no header row with a 'Time Block' column was found")


def _block_number(text: str) -> int:
    match = BLOCK_PATTERN.match(text.strip())
    if match is None:
        raise SchemaDriftError(f"time block {text!r} is not in the form HH:MM-HH:MM")
    start = int(match[1]) * 60 + int(match[2])
    end = int(match[3]) * 60 + int(match[4])
    if start % BLOCK_MINUTES or end - start != BLOCK_MINUTES:
        raise SchemaDriftError(f"time block {text!r} is not a 15-minute block on the quarter hour")
    return start // BLOCK_MINUTES + 1


def _number(text: pd.Series) -> pd.Series:
    cleaned = text.astype("string").str.strip().str.replace(",", "", regex=False)
    cleaned = cleaned.where(~cleaned.isin([MISSING_TOKEN, ""]))
    return pd.to_numeric(cleaned, errors="raise").astype(float)


def parse_snapshot(raw: pd.DataFrame, market: Market) -> pd.DataFrame:
    """One typed row per published 15-minute block found in the sheet."""
    position, names = _find_header(raw)
    known = [HEADER_TO_NAME.get(name) for name in names]
    unknown = [n for n, k in zip(names, known, strict=True) if k is None and n not in ("", "nan")]
    present = {k for k in known if k is not None}
    if unknown or present != EXPECTED[market]:
        raise SchemaDriftError(
            f"{market} header differs from the known one: unknown {unknown}, "
            f"missing {sorted(EXPECTED[market] - present)}"
        )
    keep = [i for i, k in enumerate(known) if k is not None]
    body = raw.iloc[position + 1 :, keep].copy()
    body.columns = pd.Index([known[i] for i in keep])
    rows = body[
        body["date"].astype("string").str.match(DATE_PATTERN)
        & body["time_block"].astype("string").str.match(BLOCK_PATTERN)
    ]
    if rows.empty:
        raise SchemaDriftError("the sheet holds no block rows")
    date = pd.to_datetime(rows["date"], format="%d-%m-%Y")
    block = rows["time_block"].map(_block_number).astype(int)
    table = pd.DataFrame(
        {
            "date": date,
            "block": block,
            "timestamp": (
                date + pd.to_timedelta((block - 1) * BLOCK_MINUTES, unit="min")
            ).dt.tz_localize(TIMEZONE),
            "hour": pd.to_numeric(rows["hour"], errors="raise").astype(int),
        }
    )
    if market == "rtm":
        table["session_id"] = pd.to_numeric(rows["session_id"], errors="raise").astype(int)
    for name in NUMERIC:
        table[name] = _number(rows[name])
    return table.sort_values(["date", "block"]).reset_index(drop=True)


def combine(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack several exports. A block present twice must be identical; a conflict stops the load."""
    stacked = pd.concat(tables, ignore_index=True)
    key = ["date", "block"]
    distinct = stacked.drop_duplicates()
    if distinct.duplicated(key).any():
        clash = distinct[distinct.duplicated(key, keep=False)].sort_values(key)
        raise ValueError(f"the same block has different values in two files:\n{clash.head(6)}")
    return distinct.sort_values(key).reset_index(drop=True)


def load_folder(folder: Path, market: Market) -> pd.DataFrame:
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"no .xlsx files in {folder}")
    return combine([parse_snapshot(read_sheet(path), market) for path in files])


def missing_blocks(table: pd.DataFrame) -> pd.DataFrame:
    """Days with fewer than 96 blocks and how many are absent."""
    counts = table.groupby("date").size()
    short = counts[counts < BLOCKS_PER_DAY]
    return pd.DataFrame({"blocks": short, "absent": BLOCKS_PER_DAY - short})


def mask_empty_blocks(table: pd.DataFrame) -> pd.DataFrame:
    """Blocks whose bids, volumes and price are all 0 carry no trading data: make them missing.

    ASSUMPTION: a published block with every value 0 means no data, not a price of 0.
    The flag column ``empty_block`` records which blocks were masked.
    """
    out = table.copy()
    empty = out[list(NUMERIC)].eq(0).all(axis=1)
    out.loc[empty, list(NUMERIC)] = float("nan")
    out["empty_block"] = empty
    return out
