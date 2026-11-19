"""The original notebook, its results and its data must not change."""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SHA256 = {
    "original/README.md": "6ecf6846afb1866c818d69af69ab043ddf110b670528149897af39c2a8de8987",
    "original/LD_Price_Forecast_Bagging_Boosting_Ensembles.ipynb": (
        "32f4d0d20de8a0c17df45c50e21971baf2f7baf06f9c2468edfe8774b2e23aa3"
    ),
    "original/results/Forecast_RF.csv": (
        "e2212234e56a54de71163b8587225fc9cd1ac657244a529a9489af03e863091c"
    ),
    "original/results/Forecast_XGB.csv": (
        "5361e9d4b74521f9b5c44e076e33e5d0bce6aea7446c21890b012915d8db6a12"
    ),
    "data/raw/price_data.csv": "29905622555babf56a6885dae2c1e41b6f6fda60f2469eabf76855b98890cd13",
}


@pytest.mark.parametrize(("relative_path", "expected"), sorted(EXPECTED_SHA256.items()))
def test_original_file_is_unchanged(relative_path: str, expected: str) -> None:
    digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
    assert digest == expected
