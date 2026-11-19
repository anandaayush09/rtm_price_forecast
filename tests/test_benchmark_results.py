"""Checks on the stored model comparison for the 111-day window."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "old_window"


def test_every_model_forecast_the_same_rows() -> None:
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    counts = predictions.groupby(["model", "period"]).size().unstack()
    assert len(counts) == 9
    assert (counts["test"] == 21 * 96).all()
    assert (counts["validation"] == 14 * 96).all()
    assert predictions["prediction"].notna().all()


def test_test_days_come_after_validation_days() -> None:
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    validation = predictions.loc[predictions["period"] == "validation", "date"]
    test = predictions.loc[predictions["period"] == "test", "date"]
    assert validation.max() < test.min()
    assert test.max() == pd.Timestamp("2021-01-19")


def test_the_forecasts_use_the_published_prices_as_actuals() -> None:
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    features = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    one = predictions[predictions["model"] == "lear"].merge(
        features[["date", "block", "rtm_price"]], on=["date", "block"]
    )
    same = (one["actual"] == one["rtm_price"]) | (one["actual"].isna() & one["rtm_price"].isna())
    assert same.all()


def test_stored_tuned_settings_are_complete() -> None:
    best = json.loads((OUT / "best_params.json").read_text())
    lightgbm = best["lightgbm"]["params"]
    # Row sampling only takes effect together with subsample_freq, which the search keeps fixed.
    assert lightgbm["subsample"] < 1.0 and lightgbm["subsample_freq"] == 1
    assert set(best) == {"lightgbm", "catboost", "lear"}


def test_the_chosen_setup_was_recorded() -> None:
    setup = json.loads((OUT / "setup.json").read_text())
    assert setup["transform"] == "asinh" and setup["drop_groups"] == ["demand_lags"]
