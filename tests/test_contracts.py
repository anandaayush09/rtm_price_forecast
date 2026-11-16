import pandas as pd
import pandera.pandas as pa
import pytest

from rtm_forecast.config import DataParams
from rtm_forecast.contracts import clean_schema, raw_schema


def test_real_raw_table_passes(raw_table: pd.DataFrame) -> None:
    raw_schema().validate(raw_table)


def test_real_clean_table_passes(clean_table: pd.DataFrame, params: DataParams) -> None:
    clean_schema(params).validate(clean_table)


def test_raw_rejects_dash_in_a_dam_column(raw_table: pd.DataFrame) -> None:
    broken = raw_table.copy()
    broken.loc[0, "DAM_Price"] = "-"
    with pytest.raises(pa.errors.SchemaError):
        raw_schema().validate(broken)


def test_raw_rejects_unexpected_column(raw_table: pd.DataFrame) -> None:
    broken = raw_table.assign(Extra="1")
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        raw_schema().validate(broken)


def test_clean_rejects_price_above_bound(clean_table: pd.DataFrame, params: DataParams) -> None:
    broken = clean_table.copy()
    broken.loc[5, "rtm_price"] = params.price_max + 1
    with pytest.raises(pa.errors.SchemaError):
        clean_schema(params).validate(broken)


def test_clean_rejects_a_missing_block(clean_table: pd.DataFrame, params: DataParams) -> None:
    broken = clean_table.drop(index=100).reset_index(drop=True)
    with pytest.raises(pa.errors.SchemaError):
        clean_schema(params).validate(broken)


def test_clean_rejects_duplicate_timestamp(clean_table: pd.DataFrame, params: DataParams) -> None:
    broken = pd.concat([clean_table.iloc[:10], clean_table.iloc[[9]], clean_table.iloc[11:]])
    with pytest.raises(pa.errors.SchemaError):
        clean_schema(params).validate(broken)
