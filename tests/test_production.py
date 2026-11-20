from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rtm_forecast import models as m
from rtm_forecast.production import (
    COVERAGES,
    LEVELS,
    ProductionModel,
    Spec,
    fit_production,
    forecast_day,
    update_calibration,
)

CAP = 10000.0
PARAMS = {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 8, "min_child_samples": 5}
SPEC = Spec(columns=("x",), params=PARAMS, seed=0, cap=CAP)


def _days(count: int, seed: int, first: str = "2026-01-01", cap_share: float = 0.4) -> pd.DataFrame:
    """The price is at the cap when x is large, and about 3000 + 500 x otherwise."""
    rng = np.random.default_rng(seed)
    rows = count * 96
    x = rng.normal(size=rows)
    below = 3000 + 500 * x + rng.normal(scale=150, size=rows)
    price = np.where(x > np.quantile(x, 1 - cap_share), CAP, below)
    return pd.DataFrame(
        {
            "date": pd.Timestamp(first) + pd.to_timedelta(np.arange(rows) // 96, unit="D"),
            "block": np.tile(np.arange(1, 97), count),
            "x": x,
            "rtm_price": price,
            "target_observed": True,
        }
    )


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return _days(40, 0)


@pytest.fixture(scope="module")
def model(history: pd.DataFrame) -> ProductionModel:
    return fit_production(history, SPEC)


def _calibration(model: ProductionModel, days: pd.DataFrame) -> pd.DataFrame:
    """Forecasts of ``days`` made by ``model`` next to their prices (stands in for the backtest)."""
    raw = model.raw(days)
    table = days[["date", "block"]].assign(actual=days["rtm_price"].to_numpy())
    return pd.concat([table, raw], axis=1)


def test_fit_covers_the_days_given_and_nothing_after(history: pd.DataFrame) -> None:
    fitted = fit_production(history, SPEC, before=pd.Timestamp("2026-02-01"))
    assert fitted.trained_until.isoformat() == "2026-01-31"
    assert fitted.train_days == 31
    assert fitted.train_rows == 31 * 96
    unpublished = history.assign(target_observed=history["date"] < "2026-01-20")
    assert fit_production(unpublished, SPEC).trained_until.isoformat() == "2026-01-19"
    with pytest.raises(ValueError, match="no training days"):
        fit_production(history, SPEC, before=pd.Timestamp("2025-12-01"))


def test_point_and_intervals_equal_the_backtest_model(
    history: pd.DataFrame, model: ProductionModel
) -> None:
    train, test = history[history["date"] < "2026-02-05"], _days(2, 7, "2026-02-05")
    fitted = fit_production(history, SPEC, before=pd.Timestamp("2026-02-05"))
    raw = fitted.raw(test)
    point = m.cap_aware_predictors(["x"], None, PARAMS, 0, CAP)["cap_mixture"](train, test)
    np.testing.assert_array_equal(raw["point"], point)
    ends = m.cap_aware_interval_predictors(["x"], None, PARAMS, 0, CAP, COVERAGES, LEVELS)
    for name, predictor in ends.items():
        np.testing.assert_array_equal(raw[name], predictor(train, test))


def test_saved_model_gives_the_same_forecasts(
    tmp_path: Path, history: pd.DataFrame, model: ProductionModel
) -> None:
    model.save(tmp_path)
    loaded = ProductionModel.load(tmp_path)
    test = _days(1, 3, "2026-03-01")
    pd.testing.assert_frame_equal(loaded.raw(test), model.raw(test))
    assert loaded.trained_until == model.trained_until
    assert loaded.spec == model.spec


def test_changed_or_unknown_files_are_refused(tmp_path: Path, model: ProductionModel) -> None:
    model.save(tmp_path)
    (tmp_path / "quantile_3.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        ProductionModel.load(tmp_path)
    model.save(tmp_path)
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        manifest.replace('"format_version": 1', '"format_version": 9'), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown model format"):
        ProductionModel.load(tmp_path)


def test_a_history_without_the_cap_gives_a_model_without_a_classifier(tmp_path: Path) -> None:
    history = _days(20, 1, cap_share=0.0)
    fitted = fit_production(history, SPEC)
    assert fitted.classifier is None
    fitted.save(tmp_path)
    loaded = ProductionModel.load(tmp_path)
    assert loaded.classifier is None
    test = _days(1, 2, "2026-02-01")
    pd.testing.assert_frame_equal(loaded.raw(test), fitted.raw(test))
    assert (loaded.raw(test)["point"] < CAP).all()


def test_levels_must_hold_the_point_levels() -> None:
    with pytest.raises(ValueError, match="levels must include"):
        Spec(columns=("x",), params=PARAMS, seed=0, cap=CAP, levels=(0.1, 0.5, 0.9))
    with pytest.raises(ValueError, match="levels"):
        Spec(columns=("x",), params=PARAMS, seed=0, cap=CAP, levels=(0.5, 0.9, 0.75))


def test_forecast_is_the_point_and_a_corrected_interval_around_it(model: ProductionModel) -> None:
    calibration = _calibration(model, _days(30, 4, "2026-02-01"))
    day = _days(1, 5, "2026-03-05")
    forecast = forecast_day(model, day.assign(rtm_price=np.nan), calibration)
    assert len(forecast) == 96
    np.testing.assert_array_equal(forecast["point"], model.raw(day)["point"])
    for tag in ("50", "90"):
        assert (forecast[f"lo{tag}"] <= forecast[f"hi{tag}"]).all()
        assert forecast[f"lo{tag}"].between(0, CAP).all()
        assert forecast[f"hi{tag}"].between(0, CAP).all()
    assert (forecast["lo90"] <= forecast["lo50"] + 1e-9).all()
    assert (forecast["hi90"] >= forecast["hi50"] - 1e-9).all()
    assert set(forecast["state"]) <= {0, 1, 2, 3}


def test_forecast_does_not_read_the_prices_of_the_day_or_later_calibration_days(
    model: ProductionModel,
) -> None:
    calibration = _calibration(model, _days(30, 4, "2026-02-01"))
    day = _days(1, 5, "2026-03-05")
    first = forecast_day(model, day.assign(rtm_price=np.nan), calibration)
    later = _calibration(model, _days(3, 6, "2026-03-05")).assign(actual=0.0)
    second = forecast_day(model, day.assign(rtm_price=0.0), pd.concat([calibration, later]))
    pd.testing.assert_frame_equal(first, second)


def test_forecast_needs_enough_calibration_days_and_one_day(model: ProductionModel) -> None:
    short = _calibration(model, _days(10, 4, "2026-02-01"))
    day = _days(1, 5, "2026-03-05")
    with pytest.raises(ValueError, match="at least 14 days"):
        forecast_day(model, day, short)
    with pytest.raises(ValueError, match="one day at a time"):
        forecast_day(model, _days(2, 5, "2026-03-05"), _calibration(model, _days(30, 4)))


def test_margin_follows_the_latest_prices(model: ProductionModel) -> None:
    """Prices far from the forecasts in the last days widen the interval of the next day."""
    calm = _calibration(model, _days(30, 4, "2026-02-01"))
    shifted = calm.assign(actual=np.where(calm["actual"] < CAP, calm["actual"] + 1500, CAP))
    day = _days(1, 5, "2026-03-05")
    narrow = forecast_day(model, day, calm)
    wide = forecast_day(model, day, shifted)
    assert (wide["hi90"] - wide["lo90"]).mean() > (narrow["hi90"] - narrow["lo90"]).mean()


def test_calibration_grows_by_one_day_with_the_forecast_as_it_was_made(
    model: ProductionModel,
) -> None:
    calibration = _calibration(model, _days(20, 4, "2026-02-01"))
    day = _days(1, 5, "2026-03-05")
    forecast = forecast_day(model, day.assign(rtm_price=np.nan), calibration)
    prices = day[["date", "block", "rtm_price"]]
    grown = update_calibration(calibration, forecast, prices)
    added = grown[grown["date"] == "2026-03-05"]
    assert len(grown) == len(calibration) + 96
    np.testing.assert_array_equal(added["actual"], day["rtm_price"])
    np.testing.assert_array_equal(added["point"], forecast["point"])
    np.testing.assert_array_equal(added["hi90"], forecast["raw_hi90"])
    with pytest.raises(ValueError, match="already in the calibration"):
        update_calibration(grown, forecast, prices)
    with pytest.raises(ValueError, match="do not cover"):
        update_calibration(calibration, forecast, prices.iloc[:50])
