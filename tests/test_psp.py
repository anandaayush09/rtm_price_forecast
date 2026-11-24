from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from rtm_forecast.config import DataParams
from rtm_forecast.contracts import psp_schema
from rtm_forecast.psp import (
    SchemaDriftError,
    _header_name,
    load_folder,
    parse_timeseries,
    read_sheet,
    suspect_rows,
)

HEADERS: dict[int, list[Any]] = {
    1: [
        "TIME",
        "FREQUENCY\n(Hz)",
        "DEMAND MET\n(MW)",
        "NUCLEAR\n(MW)",
        "WIND\n(MW)",
        "SOLAR\n(MW)",
        "HYDRO**\n(MW)",
        "GAS\n(MW)",
        "THERMAL\n(MW)",
        "OTHERS*\n(MW)",
        "NET DEMAND MET\n(MW)",
        "TOTAL GENERATION\n(MW)",
        "NET TRANSNATIONAL EXCHANGE\n(MW)\n(+ve) Import, (-ve) Export",
    ],
    2: [
        "TIME",
        "FREQUENCY\n(Hz)",
        "DEMAND\nMET\u00b9\n(MW)",
        "NUCLEAR\n(MW)",
        "WIND\n(MW)",
        "SOLAR\n(MW)",
        "HYDRO\u00b2\n(MW)",
        "GAS\n(MW)",
        "THERMAL\n(MW)",
        "STORAGE\n(PSP&BESS)\n(MW)",
        "OTHERS\u00b3\n(MW)",
        "NET DEMAND MET\n(MW)",
        "TOTAL GENERATION\n(MW)",
        np.nan,
        "NET TRANSNATIONAL EXCHANGE\n(MW)\n(+ve) Import, (-ve) Export",
    ],
    3: [
        "TIME",
        "FREQUENCY\n(Hz)",
        "DEMAND\nMET\u00b9\n(MW)",
        "STORAGE\nDEMAND\n(PSP&BESS)\n(MW)",
        "NUCLEAR\n(MW)",
        "WIND\n(MW)",
        "SOLAR\n(MW)",
        "HYDRO\u00b2\n(MW)",
        "GAS\n(MW)",
        "THERMAL\n(MW)",
        "STORAGE\n(PSP&BESS)\n(MW)",
        "OTHERS\u00b3\n(MW)",
        "NET DEMAND MET\n(MW)",
        "TOTAL GENERATION\n(MW)",
        np.nan,
        "NET \nTRANSNATIONAL\nEXCHANGE\n(MW)\n+ve Imp,-ve Exp",
    ],
}
VALUES = {
    "FREQUENCY": 50.0,
    "DEMAND": 200000.0,
    "STORAGE DEMAND": 100.0,
    "NUCLEAR": 7000.0,
    "WIND": 9000.0,
    "SOLAR": 1000.0,
    "HYDRO": 20000.0,
    "GAS": 2000.0,
    "THERMAL": 150000.0,
    "STORAGE": 50.0,
    "OTHERS": 2000.0,
    "NET": 190000.0,
    "TOTAL": 201000.0,
    "NET TRANSNATIONAL": 500.0,
}


def _value(header: Any) -> Any:
    if not isinstance(header, str):
        return np.nan
    words = []
    for line in header.split("\n"):
        if line.strip().startswith("(") or line.strip().startswith("+"):
            break
        words.append(line.strip())
    name = (
        " ".join(words)
        .replace("*", "")
        .replace("\u00b9", "")
        .replace("\u00b2", "")
        .replace("\u00b3", "")
    )
    for key in sorted(VALUES, key=len, reverse=True):
        if name.startswith(key):
            return str(VALUES[key])
    return np.nan


def sheet(layout: int = 1, report: str = "17-Nov-2025", skip: tuple[str, ...] = ()) -> pd.DataFrame:
    header: list[Any] = HEADERS[layout]
    width = len(header)
    first: list[Any] = [np.nan] * width
    first[0], first[-1] = "Date of Reporting:", report
    title = ["15 Min (INSTANTANEOUS) ALL INDIA GRID FREQUENCY, GENERATION & DEMAND MET"]
    rows: list[list[Any]] = [
        first,
        title + [np.nan] * (width - 1),
        header,
        ["", "(A)"] + [np.nan] * (width - 2),
    ]
    for block in range(96):
        clock = f"{block * 15 // 60}:{block * 15 % 60:02d}"
        if clock in skip:
            continue
        rows.append([clock] + [_value(h) for h in header[1:]])
    rows.append(["*Others include biomass. Disclaimer: ..."] + [np.nan] * (width - 1))
    return pd.DataFrame(rows).astype(object)


@pytest.mark.parametrize("layout", [1, 2, 3])
def test_every_known_layout_parses(layout: int) -> None:
    table = parse_timeseries(sheet(layout))
    assert len(table) == 96 and table["layout"].eq(layout).all()
    assert table["date"].eq(pd.Timestamp("2025-11-16")).all()
    assert table["block"].tolist() == list(range(1, 97))
    assert str(table["timestamp"].dt.tz) == "Asia/Kolkata"
    assert table["demand_met"].eq(200000.0).all() and table["net_exchange"].eq(500.0).all()
    has_storage = layout >= 2
    assert table["storage_generation"].notna().all() == has_storage
    assert table["storage_demand"].notna().all() == (layout == 3)


def test_headers_of_2024_write_the_unit_on_the_same_line() -> None:
    assert _header_name("FREQUENCY(Hz)") == "FREQUENCY"
    assert _header_name("NUCLEAR (MW)") == "NUCLEAR"
    assert _header_name("HYDRO**\n(MW)") == "HYDRO"
    name = _header_name("Net Transanational Exchange(MW)\n(Import:+ve,Export:-ve)")
    assert name == "NET TRANSANATIONAL EXCHANGE"


def test_a_missing_reading_stays_absent() -> None:
    table = parse_timeseries(sheet(skip=("12:15",)))
    assert len(table) == 95 and 50 not in table["block"].tolist()


def test_unknown_header_or_filled_blank_column_is_drift() -> None:
    renamed = sheet(2)
    renamed.iloc[2, 9] = "BATTERY\n(MW)"
    with pytest.raises(SchemaDriftError, match="BATTERY"):
        parse_timeseries(renamed)
    filled = sheet(2)
    filled.iloc[10, 13] = "5"
    with pytest.raises(SchemaDriftError, match="without a header"):
        parse_timeseries(filled)
    with pytest.raises(SchemaDriftError, match="missing"):
        parse_timeseries(sheet(1).drop(columns=[4]))


def test_off_grid_time_text_value_and_extra_readings_are_rejected() -> None:
    off = sheet()
    off.iloc[10, 0] = "0:07"
    with pytest.raises(SchemaDriftError, match="quarter hour"):
        parse_timeseries(off)
    text = sheet()
    text.iloc[10, 2] = "n/a"
    with pytest.raises(ValueError):
        parse_timeseries(text)
    same = pd.concat([sheet(), sheet().iloc[[10]]], ignore_index=True)
    assert len(parse_timeseries(same)) == 96
    changed = same.copy()
    changed.iloc[-1, 2] = "123456.0"
    with pytest.raises(SchemaDriftError, match="at most"):
        parse_timeseries(changed)


def test_no_report_date_or_no_header_is_drift() -> None:
    nodate = sheet()
    nodate.iloc[0, :] = np.nan
    with pytest.raises(SchemaDriftError, match="Reporting"):
        parse_timeseries(nodate)
    with pytest.raises(SchemaDriftError, match="TIME"):
        parse_timeseries(pd.DataFrame([["x", "y"]]))


def test_suspect_rows_flag_telemetry_loss_only() -> None:
    table = parse_timeseries(sheet())
    table.loc[3, "frequency"] = 0.0
    table.loc[4, "total_generation"] = 0.0
    table.loc[5, "hydro"] = -4141.0
    table.loc[6, "solar"] = -900.0
    assert suspect_rows(table).to_numpy().nonzero()[0].tolist() == [3, 4, 5]


def test_folder_checks_file_name_against_report_date(tmp_path: Path) -> None:
    sheet(1, report="17-Nov-2025").to_excel(
        tmp_path / "16.11.25_NLDC_PSP_1.xlsx", sheet_name="TimeSeries", header=False, index=False
    )
    assert read_sheet(tmp_path / "16.11.25_NLDC_PSP_1.xlsx").shape[1] == 13
    assert len(load_folder(tmp_path)) == 96
    sheet(1, report="18-Nov-2025").to_excel(
        tmp_path / "16.11.25_NLDC_PSP_2.xlsx", sheet_name="TimeSeries", header=False, index=False
    )
    (tmp_path / "16.11.25_NLDC_PSP_1.xlsx").unlink()
    with pytest.raises(SchemaDriftError, match="file name says"):
        load_folder(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_folder(tmp_path / "none")


def test_contract_accepts_a_day_and_rejects_bad_values(params: DataParams) -> None:
    good = parse_timeseries(sheet(3))
    psp_schema(params).validate(good)
    for column, value in (
        ("demand_met", 10.0),
        ("frequency", 60.0),
        ("wind", -1.0),
        ("solar", -9000.0),
    ):
        broken = good.copy()
        broken.loc[2, column] = value
        with pytest.raises(pa.errors.SchemaError):
            psp_schema(params).validate(broken)
    twice = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    with pytest.raises(pa.errors.SchemaError):
        psp_schema(params).validate(twice)
