"""Join the IEX RTM and DAM exports and the daily power-supply sheets on one 96-block grid."""

from datetime import date
from typing import Any

import pandas as pd

from rtm_forecast.iex_export import NUMERIC, mask_empty_blocks
from rtm_forecast.ingest import BLOCK_MINUTES, BLOCKS_PER_DAY, TIMEZONE
from rtm_forecast.psp import VALUE_COLUMNS, suspect_rows

IEX_NAMES = {
    "purchase_bid": "purchase_bid",
    "sell_bid": "selling_bid",
    "mcv": "mcv",
    "scheduled_volume": "scheduled_volume",
    "mcp": "price",
}
RTM_VALUES = [f"rtm_{name}" for name in IEX_NAMES.values()]
DAM_VALUES = [f"dam_{name}" for name in IEX_NAMES.values()]
PSP_VALUES = [name for name in VALUE_COLUMNS if name != "demand_met"]


def _grid(start: date, end: date) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    table = pd.DataFrame(
        {
            "date": days.repeat(BLOCKS_PER_DAY),
            "block": list(range(1, BLOCKS_PER_DAY + 1)) * len(days),
        }
    )
    offset = pd.to_timedelta((table["block"] - 1) * BLOCK_MINUTES, unit="min")
    table.insert(0, "timestamp", (table["date"] + offset).dt.tz_localize(TIMEZONE))
    return table


def _market(table: pd.DataFrame, market: str) -> pd.DataFrame:
    """One market's blocks with its own column names. All-zero blocks become missing."""
    masked = mask_empty_blocks(table)
    columns = {name: f"{market}_{new}" for name, new in IEX_NAMES.items()}
    out = masked[["date", "block", *NUMERIC, "empty_block"]].rename(columns=columns)
    return out.rename(columns={"empty_block": f"{market}_empty"})


def _power_supply(psp: pd.DataFrame) -> pd.DataFrame:
    """Readings that look like telemetry loss lose every value; the flag keeps the reason."""
    out = psp.copy()
    suspect = suspect_rows(out)
    out.loc[suspect, VALUE_COLUMNS] = float("nan")
    out["demand_suspect"] = suspect
    out = out.rename(columns={"demand_met": "demand", "layout": "psp_layout"})
    keep = ["date", "block", "demand", *PSP_VALUES, "demand_suspect", "psp_layout"]
    return out[keep]


def build_window(
    rtm: pd.DataFrame, dam: pd.DataFrame, psp: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    """Every block from ``start`` to ``end`` with RTM, DAM and demand side by side.

    Nothing is filled. A block a source does not have, or has but with no usable value, is missing
    there and the flags say which source: ``rtm_missing``, ``dam_missing``, ``demand_missing``.
    Rows of a source outside the window are left out.
    """
    table = _grid(start, end)
    for part in (_market(rtm, "rtm"), _market(dam, "dam"), _power_supply(psp)):
        table = table.merge(part, on=["date", "block"], how="left", validate="one_to_one")
    for market in ("rtm", "dam"):
        values = [f"{market}_{name}" for name in IEX_NAMES.values()]
        table[f"{market}_empty"] = table[f"{market}_empty"].fillna(False).astype(bool)
        table[f"{market}_missing"] = table[values].isna().any(axis=1)
    table["demand_suspect"] = table["demand_suspect"].fillna(False).astype(bool)
    table["demand_missing"] = table["demand"].isna()
    table["psp_layout"] = table["psp_layout"].astype("Int64")
    return table


def summarise(table: pd.DataFrame) -> dict[str, int]:
    """Counts for the data-quality report."""
    return {
        "days": int(table["date"].nunique()),
        "blocks": len(table),
        "rtm_missing": int(table["rtm_missing"].sum()),
        "rtm_empty": int(table["rtm_empty"].sum()),
        "dam_missing": int(table["dam_missing"].sum()),
        "demand_missing": int(table["demand_missing"].sum()),
        "demand_suspect": int(table["demand_suspect"].sum()),
    }


def _runs(table: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    """Runs of consecutive blocks where a flag is set, as start time, first block and length."""
    flag = table[column]
    run_id = (flag != flag.shift()).cumsum()
    runs: list[dict[str, Any]] = []
    for _, group in table[flag].groupby(run_id[flag]):
        runs.append(
            {
                "start": group["timestamp"].iloc[0].isoformat(),
                "block": int(group["block"].iloc[0]),
                "length": len(group),
            }
        )
    return runs


def window_report(
    table: pd.DataFrame, outside_window: dict[str, int], layouts: dict[str, int]
) -> dict[str, Any]:
    """Counts and positions only. No price, bid or demand value goes into the report."""
    return {
        "first_day": table["date"].min().date().isoformat(),
        "last_day": table["date"].max().date().isoformat(),
        **summarise(table),
        "rows_outside_window": outside_window,
        "demand_sheet_layouts": layouts,
        "rtm_missing_runs": _runs(table, "rtm_missing"),
        "dam_missing_runs": _runs(table, "dam_missing"),
        "demand_missing_runs": _runs(table, "demand_missing"),
        "demand_suspect_runs": _runs(table, "demand_suspect"),
    }
