from pathlib import Path

import pandas as pd
import pytest

from rtm_forecast.config import DataParams, load_data_params
from rtm_forecast.ingest import clean, load_raw

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def params() -> DataParams:
    return load_data_params(ROOT / "params.yaml")


@pytest.fixture(scope="session")
def raw_table() -> pd.DataFrame:
    return load_raw(ROOT / "data" / "raw" / "price_data.csv")


@pytest.fixture(scope="session")
def clean_table(raw_table: pd.DataFrame) -> pd.DataFrame:
    return clean(raw_table)
