import numpy as np
import pandas as pd
import pytest

from rtm_forecast.imputation import evaluate_methods, fill, fill_within_days
from rtm_forecast.transforms import AsinhTransform, choose_scale


def test_ffill_and_linear_on_a_small_gap() -> None:
    values = np.array([1.0, np.nan, np.nan, 4.0])
    assert fill(values, "ffill").tolist() == [1.0, 1.0, 1.0, 4.0]
    assert fill(values, "linear").tolist() == [1.0, 2.0, 3.0, 4.0]


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        fill(np.array([1.0]), "median")


def test_fill_does_not_look_into_the_next_day() -> None:
    values = np.arange(192, dtype=float)
    values[94:96] = np.nan
    filled = fill_within_days(values, "linear")
    assert filled[94] == filled[95] == 93.0
    values[96] = 1e6
    assert fill_within_days(values, "linear")[94] == 93.0


def test_yesterday_uses_the_same_block_a_day_earlier() -> None:
    values = np.arange(200, dtype=float)
    values[150] = np.nan
    assert fill(values, "yesterday")[150] == values[54]


def test_linear_fill_is_best_on_the_real_prices(clean_table: pd.DataFrame) -> None:
    result = evaluate_methods(clean_table["rtm_price"].to_numpy(), length=2, trials=300, seed=0)
    assert result["MAE"].idxmin() == "linear"
    mae = result["MAE"].to_dict()
    assert mae["linear"] < mae["ffill"]


def test_asinh_round_trip() -> None:
    transform = AsinhTransform(scale=100.0)
    values = np.array([-50.0, 0.0, 400.0, 8000.0])
    np.testing.assert_allclose(transform.inverse(transform.forward(values)), values)


def test_asinh_removes_most_of_the_skew(clean_table: pd.DataFrame) -> None:
    prices = clean_table["rtm_price"].dropna().to_numpy()
    scale = choose_scale(prices[: 90 * 96], [50, 100, 250, 500, 1000, 2000])
    raw_skew = float(abs(pd.Series(prices).skew()))  # type: ignore[arg-type]
    new_skew = float(abs(pd.Series(AsinhTransform(scale).forward(prices)).skew()))  # type: ignore[arg-type]
    assert raw_skew > 1.5
    assert new_skew < 0.3


def test_choose_scale_picks_the_most_symmetric_candidate() -> None:
    rng = np.random.default_rng(0)
    values = np.exp(rng.normal(8, 0.5, 5000))
    assert choose_scale(values, [10.0, 1e6]) == 10.0
