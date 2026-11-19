import numpy as np
import pandas as pd
import pytest

from rtm_forecast.backtest import rolling_origin
from rtm_forecast.metrics import daily_mae, mae, mape, rmse, smape, summarise
from rtm_forecast.significance import bootstrap_difference, diebold_mariano
from rtm_forecast.splits import make_periods


def test_error_measures_on_a_hand_worked_example() -> None:
    actual = np.array([100.0, 200.0, 400.0])
    predicted = np.array([110.0, 180.0, 400.0])
    assert mae(actual, predicted) == pytest.approx(10.0)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt((100 + 400 + 0) / 3))
    assert mape(actual, predicted) == pytest.approx((10 + 10 + 0) / 3)
    assert smape(actual, predicted) == pytest.approx(100 * (2 * 10 / 210 + 2 * 20 / 380 + 0) / 3)


def test_summary_reference_and_missing_actuals() -> None:
    frame = pd.DataFrame(
        {
            "model": ["ref", "ref", "ref", "m", "m", "m"],
            "date": pd.to_datetime(["2021-01-01"] * 6),
            "block": [1, 2, 3, 1, 2, 3],
            "actual": [10.0, 20.0, np.nan, 10.0, 20.0, np.nan],
            "prediction": [12.0, 24.0, 99.0, 11.0, 22.0, 1.0],
        }
    )
    table = summarise(frame, reference="ref")
    assert table.loc["ref", "MAE"] == pytest.approx(3.0)
    assert table.loc["m", "MAE"] == pytest.approx(1.5)
    assert table.loc["m", "rMAE"] == pytest.approx(0.5)
    assert daily_mae(frame).loc["2021-01-01", "m"] == pytest.approx(1.5)


def test_periods_are_consecutive_and_disjoint() -> None:
    days = list(pd.date_range("2021-01-01", periods=30))
    periods = make_periods(days, validation_days=5, test_days=7)
    assert (len(periods.train), len(periods.validation), len(periods.test)) == (18, 5, 7)
    assert periods.train[-1] < periods.validation[0] < periods.validation[-1] < periods.test[0]
    with pytest.raises(ValueError):
        make_periods(days, validation_days=20, test_days=10)


def test_dm_test_detects_a_clear_difference_and_ignores_noise() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(100, 5, 40)
    statistic, p_value = diebold_mariano(base - 10 + rng.normal(0, 2, 40), base)
    assert statistic < 0 and p_value < 0.001
    assert diebold_mariano(base - 10, base) == (float("-inf"), 0.0)
    _, p_same = diebold_mariano(base, base + rng.normal(0, 5, 40))
    assert 0.05 < p_same < 0.95
    assert diebold_mariano(base, base) == (0.0, 0.5)


def test_bootstrap_interval_contains_the_true_difference() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(100, 5, 60)
    low, high = bootstrap_difference(base - 3 + rng.normal(0, 1, 60), base, seed=0)
    assert low < -3 < high


def _frame() -> pd.DataFrame:
    days = pd.date_range("2021-01-01", periods=6)
    rows = pd.DataFrame({"date": np.repeat(days, 4), "block": np.tile(np.arange(1, 5), 6)})
    rows["rtm_price"] = np.arange(len(rows), dtype=float)
    rows["target_observed"] = True
    rows.loc[5, "target_observed"] = False
    rows.loc[5, "rtm_price"] = np.nan
    return rows


def test_backtest_gives_each_model_only_earlier_days() -> None:
    seen = []

    def predictor(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        seen.append((train["date"].max(), test["date"].min(), len(train)))
        assert train["target_observed"].all()
        return np.zeros(len(test))

    days = list(pd.date_range("2021-01-03", periods=3))
    result = rolling_origin(_frame(), days, predictor, "zero")
    assert all(latest < first for latest, first, _ in seen)
    assert len(result) == 12 and set(result["model"]) == {"zero"}
    assert seen[0][2] == 8 - 1  # two earlier days of 4 rows, minus the unpublished one


def test_backtest_refit_every_two_days_uses_the_start_of_the_group() -> None:
    starts = []

    def predictor(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        starts.append((train["date"].max(), sorted(test["date"].unique())))
        return np.zeros(len(test))

    days = list(pd.date_range("2021-01-03", periods=4))
    rolling_origin(_frame(), days, predictor, "zero", refit_every=2)
    assert len(starts) == 2
    assert starts[0][0] == pd.Timestamp("2021-01-02")
    assert starts[1][0] == pd.Timestamp("2021-01-04")


def test_backtest_rejects_a_wrong_length() -> None:
    def predictor(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(test) - 1)

    with pytest.raises(ValueError):
        rolling_origin(_frame(), [pd.Timestamp("2021-01-03")], predictor, "bad")
