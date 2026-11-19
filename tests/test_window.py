from datetime import date

import pandas as pd
import pandera.pandas as pa
import pytest

from rtm_forecast.config import DataParams
from rtm_forecast.contracts import window_schema
from rtm_forecast.iex_export import parse_snapshot
from rtm_forecast.psp import parse_timeseries
from rtm_forecast.window import build_window, summarise
from test_iex_export import sheet as iex_sheet
from test_psp import sheet as psp_sheet

DAY = date(2026, 11, 16)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rtm = parse_snapshot(iex_sheet("16-11-2026"), "rtm")
    dam = parse_snapshot(iex_sheet("16-11-2026", market="dam"), "dam")
    psp = parse_timeseries(psp_sheet(3, report="17-Nov-2026"))
    return rtm, dam, psp


def test_grid_is_full_and_sources_line_up() -> None:
    rtm, dam, psp = _inputs()
    table = build_window(rtm, dam, psp, DAY, DAY)
    assert len(table) == 96 and table["block"].tolist() == list(range(1, 97))
    assert str(table["timestamp"].dt.tz) == "Asia/Kolkata"
    assert table["rtm_price"].eq(3000.5).all() and table["dam_price"].eq(3000.5).all()
    assert table["demand"].eq(200000.0).all()
    assert summarise(table)["rtm_missing"] == 0
    assert not table[["rtm_missing", "dam_missing", "demand_missing", "demand_suspect"]].any().any()


def test_a_block_a_source_does_not_have_stays_missing_and_nothing_is_filled() -> None:
    rtm, dam, psp = _inputs()
    short_rtm = rtm[rtm["block"] <= 76]
    short_psp = psp[psp["block"] != 50]
    table = build_window(short_rtm, dam, short_psp, DAY, DAY)
    assert table["rtm_missing"].sum() == 20 and table["rtm_price"].isna().sum() == 20
    assert table.loc[table["block"] == 50, "demand_missing"].item()
    assert not table.loc[table["block"] == 50, "demand_suspect"].item()
    assert table["dam_missing"].sum() == 0


def test_all_zero_rtm_blocks_and_suspect_readings_are_masked_with_a_flag() -> None:
    rtm, dam, psp = _inputs()
    rtm.loc[4:5, ["purchase_bid", "sell_bid", "mcv", "scheduled_volume", "mcp"]] = 0.0
    psp.loc[10, "frequency"] = 0.0
    table = build_window(rtm, dam, psp, DAY, DAY)
    assert table["rtm_empty"].to_numpy().nonzero()[0].tolist() == [4, 5]
    assert table.loc[[4, 5], "rtm_price"].isna().all() and table["rtm_missing"].sum() == 2
    assert table["demand_suspect"].to_numpy().nonzero()[0].tolist() == [10]
    assert table.loc[10, ["demand", "wind", "frequency"]].isna().all()
    assert table.loc[11, "demand"] == 200000.0


def test_days_outside_the_window_are_left_out_and_empty_days_are_missing() -> None:
    rtm, dam, psp = _inputs()
    table = build_window(rtm, dam, psp, date(2026, 11, 15), date(2026, 11, 15))
    assert len(table) == 96 and table["rtm_missing"].all() and table["demand_missing"].all()
    assert table["psp_layout"].isna().all()
    two = build_window(rtm, dam, psp, date(2026, 11, 16), date(2026, 11, 17))
    assert len(two) == 192 and two["rtm_missing"].sum() == 96


def test_contract_accepts_the_table_and_rejects_broken_ones(params: DataParams) -> None:
    rtm, dam, psp = _inputs()
    rtm.loc[4, ["purchase_bid", "sell_bid", "mcv", "scheduled_volume", "mcp"]] = 0.0
    psp.loc[10, "frequency"] = 0.0
    good = build_window(rtm, dam, psp, DAY, DAY)
    window_schema(params).validate(good)
    cases = {
        "price": ("rtm_price", params.price_max + 1),
        "volume": ("dam_purchase_bid", params.bid_max + 1),
        "demand": ("demand", 10.0),
        "frequency": ("frequency", 47.0),
        "wind": ("wind", -1.0),
    }
    for column, value in cases.values():
        broken = good.copy()
        broken.loc[20, column] = value
        with pytest.raises(pa.errors.SchemaError):
            window_schema(params).validate(broken)
    kept = good.copy()
    kept.loc[10, "demand"] = 200000.0  # a suspect reading must not keep its demand
    with pytest.raises(pa.errors.SchemaError):
        window_schema(params).validate(kept)
    priced = good.copy()
    priced.loc[4, "rtm_price"] = 0.0  # an empty block must not keep a price
    with pytest.raises(pa.errors.SchemaError):
        window_schema(params).validate(priced)
    with pytest.raises(pa.errors.SchemaError):
        window_schema(params).validate(good.iloc[:-1])
