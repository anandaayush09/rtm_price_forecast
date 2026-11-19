from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from rtm_forecast.config import DataParams
from rtm_forecast.contracts import iex_schema
from rtm_forecast.iex_export import (
    NUMERIC,
    SchemaDriftError,
    combine,
    load_folder,
    mask_empty_blocks,
    missing_blocks,
    parse_snapshot,
    read_sheet,
)

RTM_HEADER = [
    "Date",
    "Hour",
    "Session ID",
    "Time Block",
    "Purchase Bid (MW)",
    "Sell Bid (MW)",
    "MCV (MW)",
    "Final Scheduled Volume (MW)",
    "MCP (Rs/MWh) *",
]


def _label(block: int, spaced: bool) -> str:
    start = (block - 1) * 15
    end = start + 15
    joiner = " - " if spaced else "-"
    return f"{start // 60:02d}:{start % 60:02d}{joiner}{end // 60:02d}:{end % 60:02d}"


def sheet(
    day: str = "16-11-2026",
    blocks: range = range(1, 97),
    market: str = "rtm",
    price: str = "3000.50",
) -> pd.DataFrame:
    """A sheet laid out like the website export: title lines, header, block rows, summary rows."""
    header = RTM_HEADER if market == "rtm" else [h for h in RTM_HEADER if h != "Session ID"]
    width = len(header)
    pad: list[Any] = [np.nan] * (width - 1)
    rows: list[list[Any]] = [
        pad + [np.nan],
        ["Market Snapshot"] + pad,
        [f"Date: {day}"] + pad,
        [np.nan] * width,
        header,
    ]
    for block in blocks:
        row: list[Any] = [day, str((block - 1) // 4 + 1)]
        if market == "rtm":
            row.append(str((block - 1) // 2 + 1))
        row += [
            _label(block, spaced=market == "dam"),
            "9898.40",
            "15094.50",
            "8755.80",
            "8755.80",
            price,
        ]
        rows.append(row)
    rows.append(
        [day, "Summary", "Purchase Bid", "Sell Bid", "MCV", "x", "MCP (Rs/MWh) *"]
        + [np.nan] * (width - 7)
    )
    rows.append(
        [day, "Total (MWh)", "225974.83", "331805.10", "188599.42", "1", np.nan]
        + [np.nan] * (width - 7)
    )
    return pd.DataFrame(rows[: len(rows)]).astype(object)


def test_rtm_sheet_parses_to_one_row_per_block() -> None:
    table = parse_snapshot(sheet(blocks=range(1, 77)), "rtm")
    assert len(table) == 76
    assert table["block"].tolist() == list(range(1, 77))
    assert str(table["timestamp"].dt.tz) == "Asia/Kolkata"
    assert table["timestamp"].dt.strftime("%H:%M").iloc[0] == "00:00"
    assert table.loc[1, "session_id"] == 1 and table.loc[2, "session_id"] == 2
    assert table["mcp"].eq(3000.5).all()


def test_dam_sheet_with_spaced_time_blocks_and_no_session() -> None:
    table = parse_snapshot(sheet(market="dam"), "dam")
    assert len(table) == 96 and "session_id" not in table
    assert table["block"].iloc[-1] == 96


def test_a_range_export_with_two_days_reads_both() -> None:
    both = pd.concat([sheet("15-11-2026"), sheet("16-11-2026").iloc[5:]], ignore_index=True)
    table = parse_snapshot(both, "rtm")
    assert table["date"].nunique() == 2 and len(table) == 192


def test_dash_is_missing_and_other_text_raises() -> None:
    table = parse_snapshot(sheet(price="-"), "rtm")
    assert table["mcp"].isna().all()
    with pytest.raises(ValueError):
        parse_snapshot(sheet(price="n/a"), "rtm")


def test_unknown_or_missing_column_is_drift() -> None:
    extra = sheet()
    extra.iloc[4, 8] = "Weighted Price (Rs/MWh)"
    with pytest.raises(SchemaDriftError, match="unknown"):
        parse_snapshot(extra, "rtm")
    with pytest.raises(SchemaDriftError, match="missing"):
        parse_snapshot(sheet(market="dam"), "rtm")


def test_a_block_not_on_the_quarter_hour_is_drift() -> None:
    broken = sheet()
    broken.iloc[5, 3] = "00:10-00:25"
    with pytest.raises(SchemaDriftError):
        parse_snapshot(broken, "rtm")


def test_sheet_without_header_or_rows_is_drift() -> None:
    with pytest.raises(SchemaDriftError):
        parse_snapshot(pd.DataFrame([["nothing", "here"]]), "rtm")
    with pytest.raises(SchemaDriftError, match="no block rows"):
        parse_snapshot(sheet(blocks=range(1, 1)), "rtm")


def test_combine_drops_identical_overlap_and_rejects_conflict() -> None:
    first = parse_snapshot(sheet(blocks=range(1, 61)), "rtm")
    second = parse_snapshot(sheet(blocks=range(41, 97)), "rtm")
    assert len(combine([first, second])) == 96
    changed = parse_snapshot(sheet(blocks=range(41, 97), price="1.0"), "rtm")
    with pytest.raises(ValueError, match="different values"):
        combine([first, changed])


def test_missing_blocks_reports_short_days() -> None:
    table = parse_snapshot(sheet(blocks=range(1, 77)), "rtm")
    assert missing_blocks(table).to_dict("records") == [{"blocks": 76, "absent": 20}]


def test_workbook_round_trip_and_folder(tmp_path: Path) -> None:
    frame = sheet()
    frame.to_excel(tmp_path / "RTM_one.xlsx", header=False, index=False)
    assert read_sheet(tmp_path / "RTM_one.xlsx").shape == frame.shape
    table = load_folder(tmp_path, "rtm")
    assert len(table) == 96
    with pytest.raises(FileNotFoundError):
        load_folder(tmp_path / "empty", "rtm")


def test_contract_accepts_a_day_and_rejects_bad_rows(params: DataParams) -> None:
    good = parse_snapshot(sheet(blocks=range(1, 77)), "rtm")
    iex_schema("rtm", params).validate(good)
    high = good.copy()
    high.loc[3, "mcp"] = params.price_max + 1
    with pytest.raises(pa.errors.SchemaError):
        iex_schema("rtm", params).validate(high)
    for wrong in (1, 40):  # block 4 belongs to session 2
        wrong_session = good.copy()
        wrong_session.loc[3, "session_id"] = wrong
        with pytest.raises(pa.errors.SchemaError):
            iex_schema("rtm", params).validate(wrong_session)
    twice = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    with pytest.raises(pa.errors.SchemaError):
        iex_schema("rtm", params).validate(twice)
    iex_schema("dam", params).validate(parse_snapshot(sheet(market="dam"), "dam"))


def test_all_zero_blocks_become_missing_and_only_those() -> None:
    table = parse_snapshot(sheet(blocks=range(1, 9)), "rtm")
    table.loc[2:3, list(NUMERIC)] = 0.0  # no data in two blocks
    table.loc[5, "mcp"] = 0.0  # a real price of 0 with trading volume
    table.loc[6, ["purchase_bid", "sell_bid", "mcv", "scheduled_volume"]] = 0.0  # price still set
    out = mask_empty_blocks(table)
    assert out["empty_block"].tolist() == [False, False, True, True, False, False, False, False]
    assert out.loc[[2, 3], list(NUMERIC)].isna().all().all()
    assert out.loc[5, "mcp"] == 0.0 and out.loc[5, "mcv"] == 8755.8
    assert out.loc[6, "mcp"] == 3000.5
    assert table.loc[2, "mcp"] == 0.0  # the input is not changed
