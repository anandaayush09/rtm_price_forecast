import pandas as pd
import pytest

from rtm_forecast.ingest import clean


def test_shape_and_span(clean_table: pd.DataFrame) -> None:
    assert len(clean_table) == 10656
    assert clean_table["date"].nunique() == 111
    assert clean_table["timestamp"].iloc[0] == pd.Timestamp("2020-10-01 00:00", tz="Asia/Kolkata")
    assert clean_table["timestamp"].iloc[-1] == pd.Timestamp("2021-01-19 23:45", tz="Asia/Kolkata")


def test_dates_are_day_first(clean_table: pd.DataFrame) -> None:
    # 01-10-2020 is 1 October. Read month-first it would become 10 January.
    first = clean_table.iloc[0]
    assert (first["date"].month, first["date"].day) == (10, 1)
    assert clean_table["date"].dt.month.nunique() == 4


def test_block_to_clock_time(clean_table: pd.DataFrame) -> None:
    row = clean_table[(clean_table["date"] == "2020-10-06") & (clean_table["block"] == 7)].iloc[0]
    assert row["timestamp"].strftime("%H:%M") == "01:30"


def test_dash_becomes_missing_not_a_number(clean_table: pd.DataFrame) -> None:
    assert int(clean_table["rtm_missing"].sum()) == 12
    assert int(clean_table["rtm_price"].isna().sum()) == 12
    assert int(clean_table["dam_price"].isna().sum()) == 0


def test_other_text_is_rejected() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["01-10-2020"],
            "TimeblockId": ["1"],
            "DAM_Price": ["abc"],
            "RTM_Price": ["1.0"],
            "RTM_Purchase_Bid": ["1.0"],
            "RTM_Selling_Bid": ["1.0"],
            "DAM_Purchase_Bid": ["1.0"],
            "DAM_Selling_Bid": ["1.0"],
            "Demand": ["1"],
        }
    )
    with pytest.raises(ValueError):
        clean(raw)
