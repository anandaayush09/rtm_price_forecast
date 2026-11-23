from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from rtm_forecast import dashboard
from rtm_forecast.production import ProductionModel, forecast_day

CAP = 10000.0


def calibration_table(days: int, seed: int = 0) -> pd.DataFrame:
    """Forecasts around 3000 with intervals that are too narrow, and the prices that followed."""
    rng = np.random.default_rng(seed)
    rows = days * 96
    point = 3000 + 500 * np.sin(np.arange(rows) * 2 * np.pi / 96)
    actual = point + rng.normal(scale=400, size=rows)
    actual[rng.random(rows) < 0.1] = CAP
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2026-08-01") + pd.to_timedelta(np.arange(rows) // 96, unit="D"),
            "block": np.tile(np.arange(1, 97), days),
            "actual": actual,
            "point": point,
            "lo50": point - 150,
            "hi50": point + 150,
            "lo90": point - 350,
            "hi90": point + 350,
        }
    )


def test_block_one_starts_at_midnight_and_block_96_at_a_quarter_to_midnight() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-10-01"] * 2), "block": [1, 96]})
    start = dashboard.block_start(frame["date"], frame["block"])
    assert list(start.dt.strftime("%H:%M")) == ["00:00", "23:45"]


def test_forecast_days_lists_forecast_files_latest_first(tmp_path: Path) -> None:
    assert dashboard.forecast_days(tmp_path / "missing") == []
    for name in (
        "forecast_2026-10-30.csv",
        "forecast_2026-11-02.csv",
        "notes.csv",
        "forecast_x.csv",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert dashboard.forecast_days(tmp_path) == [
        pd.Timestamp("2026-11-02"),
        pd.Timestamp("2026-10-30"),
    ]


def test_day_summary_counts_blocks_at_the_cap() -> None:
    forecast = pd.DataFrame(
        {
            "block": [1, 2, 3, 4],
            "point": [2000.0, CAP, 3000.0, CAP],
            "hi90": [4000.0, CAP, CAP, CAP],
        }
    )
    summary = dashboard.day_summary(forecast, CAP)
    assert summary["blocks_at_cap"] == 2
    assert summary["blocks_possible"] == 3
    assert summary["mean_point"] == pytest.approx(6250.0)
    assert (summary["highest_point"], summary["highest_block"]) == (CAP, 2)


def test_history_leaves_out_days_without_enough_earlier_days() -> None:
    calibration = calibration_table(30)
    history = dashboard.corrected_history(calibration, CAP)
    assert history["date"].min() == calibration["date"].min() + pd.Timedelta(days=14)
    assert history["date"].max() == calibration["date"].max()
    assert history[["lo50", "hi50", "lo90", "hi90"]].notna().all().all()
    assert (history["lo90"] <= history["hi90"]).all()
    assert history["hi90"].max() <= CAP and history["lo90"].min() >= 0


def test_corrected_history_equals_what_a_new_forecast_of_that_day_would_have_given() -> None:
    calibration = calibration_table(30)
    last = calibration["date"].max()
    today = calibration[calibration["date"] == last]

    class Stub:
        """Gives back the stored forecasts of the day, as the model did when it issued them."""

        spec = type("Spec", (), {"cap": CAP})()

        def raw(self, rows: pd.DataFrame) -> pd.DataFrame:
            names = ["point", "lo50", "hi50", "lo90", "hi90"]
            return pd.DataFrame(today[names].to_numpy(), index=rows.index, columns=names)

    rows = today[["date", "block"]]
    issued = forecast_day(
        cast(ProductionModel, Stub()), rows, calibration[calibration["date"] < last]
    )
    history = dashboard.corrected_history(calibration, CAP)
    again = history[history["date"] == last].reset_index(drop=True)
    for name in ("point", "lo50", "hi50", "lo90", "hi90"):
        np.testing.assert_allclose(again[name], issued[name])


def test_daily_record_compares_with_yesterday_on_the_same_blocks() -> None:
    blocks = [1, 2]
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-01"] * 2 + ["2026-09-02"] * 2),
            "block": blocks * 2,
            "actual": [1000.0, 2000.0, 1500.0, 1800.0],
            "point": [1100.0, 1900.0, 1400.0, 2000.0],
            "lo50": [900.0] * 4,
            "hi50": [1600.0] * 4,
            "lo90": [500.0] * 4,
            "hi90": [2100.0] * 4,
        }
    )
    record = dashboard.daily_record(history).set_index("date")
    first, second = record.loc["2026-09-01"], record.loc["2026-09-02"]
    assert np.isnan(first["model_mae"]) and np.isnan(first["yesterday_mae"])
    assert second["model_mae"] == pytest.approx(150.0)
    assert second["yesterday_mae"] == pytest.approx(350.0)
    assert first["inside90"] == 1.0
    assert first["inside50"] == pytest.approx(0.5)
    totals = dashboard.record_totals(record.reset_index())
    assert totals["days"] == 2
    assert totals["model_mae"] == pytest.approx(150.0)
    assert totals["inside50"] == pytest.approx(0.5)
