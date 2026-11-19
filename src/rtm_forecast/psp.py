"""Read the 15-minute all-India demand and generation sheet of the daily power supply report.

Columns are found by their header words, not by position, because the layout changed during the
year: a storage generation column appeared in May 2026 and a storage demand column in June 2026,
and the meaning of "demand met" changed with them. Any header word that is not known stops the load.
"""

import re
from pathlib import Path

import pandas as pd

from rtm_forecast.ingest import BLOCK_MINUTES, BLOCKS_PER_DAY, TIMEZONE

SHEET = "TimeSeries"
# ASSUMPTION: a report dated D describes day D-1. In all 364 files the date in the name plus one day
# equals the report date.
REPORT_LAG_DAYS = 1

HEADER_TO_NAME = {
    "TIME": "time",
    "FREQUENCY": "frequency",
    "DEMAND MET": "demand_met",
    "STORAGE DEMAND": "storage_demand",
    "NUCLEAR": "nuclear",
    "WIND": "wind",
    "SOLAR": "solar",
    "HYDRO": "hydro",
    "GAS": "gas",
    "THERMAL": "thermal",
    "STORAGE": "storage_generation",
    "OTHERS": "others",
    "NET DEMAND MET": "net_demand_met",
    "TOTAL GENERATION": "total_generation",
    "NET TRANSNATIONAL EXCHANGE": "net_exchange",
}
REQUIRED = set(HEADER_TO_NAME.values()) - {"storage_demand", "storage_generation"}
VALUE_COLUMNS = [name for name in HEADER_TO_NAME.values() if name != "time"]
TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")
REPORT_DATE_PATTERN = re.compile(r"\d{2}-[A-Za-z]{3}-\d{4}")
FILE_DATE_PATTERN = re.compile(r"^(\d{2})\.(\d{2})\.(\d{2})_")


class SchemaDriftError(ValueError):
    """The sheet no longer has the columns or layout this loader was written for."""


def read_sheet(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=SHEET, header=None, dtype=str, engine="calamine")


def _header_name(cell: object) -> str:
    """The words of a header cell up to its unit line, upper case, letters and spaces only."""
    if not isinstance(cell, str):
        return ""
    words: list[str] = []
    for line in cell.split("\n"):
        line = line.strip()
        if line.startswith("("):
            break
        words.append(re.sub(r"[^A-Z ]", "", line.upper()).strip())
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def _report_date(raw: pd.DataFrame) -> pd.Timestamp:
    for cell in raw.iloc[:2].to_numpy().ravel():
        if isinstance(cell, str) and (match := REPORT_DATE_PATTERN.search(cell)):
            return pd.to_datetime(match[0], format="%d-%b-%Y")
    raise SchemaDriftError("no 'Date of Reporting' value in the first rows")


def parse_timeseries(raw: pd.DataFrame) -> pd.DataFrame:
    """One typed row per 15-minute reading of the day the report describes.

    A reading that is missing from the sheet stays absent; the cleaning step puts it back as a gap.
    """
    header = next(
        (i for i in range(len(raw)) if str(raw.iloc[i, 0]).strip().upper() == "TIME"), None
    )
    if header is None:
        raise SchemaDriftError("no header row starting with TIME")
    names = [_header_name(cell) for cell in raw.iloc[header]]
    unknown = [n for n in names if n and n not in HEADER_TO_NAME]
    mapped = {HEADER_TO_NAME[n] for n in names if n in HEADER_TO_NAME}
    if unknown or not REQUIRED <= mapped or len(mapped) != sum(n in HEADER_TO_NAME for n in names):
        raise SchemaDriftError(
            f"header differs from the known ones: unknown {unknown}, "
            f"missing {sorted(REQUIRED - mapped)}"
        )
    keep = [i for i, n in enumerate(names) if n]
    rows = raw.iloc[header + 1 :]
    rows = rows[rows.iloc[:, 0].astype("string").str.match(TIME_PATTERN).fillna(False)]
    empty = [i for i, n in enumerate(names) if not n]
    if empty and rows.iloc[:, empty].notna().any().any():
        raise SchemaDriftError("a column without a header holds values")
    body = rows.iloc[:, keep].copy()
    body.columns = pd.Index([HEADER_TO_NAME[names[i]] for i in keep])
    clock = pd.to_datetime(body["time"], format="%H:%M")
    minutes = clock.dt.hour * 60 + clock.dt.minute
    if (minutes % BLOCK_MINUTES).any():
        raise SchemaDriftError("a reading is not on the quarter hour")
    day = _report_date(raw) - pd.Timedelta(days=REPORT_LAG_DAYS)
    block = (minutes // BLOCK_MINUTES + 1).astype(int)
    table = pd.DataFrame(
        {
            "date": day,
            "block": block,
            "timestamp": (day + pd.to_timedelta(minutes, unit="min")).dt.tz_localize(TIMEZONE),
        }
    )
    for name in VALUE_COLUMNS:
        if name in body:
            table[name] = pd.to_numeric(
                body[name].astype("string").str.strip(), errors="raise"
            ).astype(float)
        else:
            table[name] = float("nan")
    table["layout"] = 1 + int("storage_generation" in body) + int("storage_demand" in body)
    if len(table) > BLOCKS_PER_DAY or table["block"].duplicated().any():
        raise SchemaDriftError(
            f"expected at most {BLOCKS_PER_DAY} distinct readings, found {len(table)}"
        )
    return table.sort_values("block").reset_index(drop=True)


def load_folder(root: Path) -> pd.DataFrame:
    """All daily files under ``root``. The date in each file name must match the date inside."""
    files = sorted(root.rglob("*.xls*"))
    if not files:
        raise FileNotFoundError(f"no .xls files under {root}")
    days = []
    for path in files:
        table = parse_timeseries(read_sheet(path))
        name = FILE_DATE_PATTERN.match(path.name)
        if name is None:
            raise SchemaDriftError(f"file name {path.name!r} does not start with dd.mm.yy_")
        stamp = pd.Timestamp(2000 + int(name[3]), int(name[2]), int(name[1]))
        if table["date"].iloc[0] != stamp:
            raise SchemaDriftError(
                f"{path.name}: the file name says {stamp.date()}, "
                f"the sheet {table['date'].iloc[0].date()}"
            )
        days.append(table)
    stacked = pd.concat(days, ignore_index=True)
    if stacked.duplicated(["date", "block"]).any():
        raise ValueError("two files describe the same day")
    return stacked.sort_values(["date", "block"]).reset_index(drop=True)


def suspect_rows(table: pd.DataFrame) -> pd.Series:
    """Readings that look like telemetry loss: frequency outside 48 to 52 Hz, no generation at all,
    or a negative value for a source that cannot be negative."""
    cannot_be_negative = table[["nuclear", "wind", "hydro", "gas", "thermal"]].lt(0).any(axis=1)
    return (
        ~table["frequency"].between(48.0, 52.0)
        | (table["total_generation"] <= 0)
        | cannot_be_negative
    )
