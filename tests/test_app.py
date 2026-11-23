from pathlib import Path

import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from test_dashboard import calibration_table

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")


@pytest.fixture()
def folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    settings["production"] |= {
        "model_dir": str(tmp_path / "model"),
        "calibration_path": str(tmp_path / "calibration.parquet"),
        "forecast_dir": str(tmp_path / "forecasts"),
    }
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setenv("RTM_PARAMS", str(tmp_path / "params.yaml"))
    return tmp_path


def _forecast_file(folder: Path, calibration: pd.DataFrame, day: pd.Timestamp) -> None:
    today = calibration[calibration["date"] == day]
    forecast = today[["date", "block", "point", "lo50", "hi50", "lo90", "hi90"]].assign(
        state=0, raw_lo50=0.0, raw_hi50=0.0, raw_lo90=0.0, raw_hi90=0.0
    )
    (folder / "forecasts").mkdir(exist_ok=True)
    forecast.to_csv(folder / "forecasts" / f"forecast_{day:%Y-%m-%d}.csv", index=False)


def test_without_files_the_app_says_what_to_run(folders: Path) -> None:
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    texts = " ".join(str(item.value) for item in app.info)
    assert "No forecast files" in texts
    assert "rtm-forecast fit" in texts
    assert "No calibration table yet" in texts


def test_with_forecasts_and_a_calibration_table_it_shows_the_day_and_the_record(
    folders: Path,
) -> None:
    calibration = calibration_table(30)
    calibration.to_parquet(folders / "calibration.parquet", index=False)
    for day in calibration["date"].drop_duplicates().tail(2):
        _forecast_file(folders, calibration, day)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    chosen = app.selectbox[0]
    assert len(chosen.options) == 2
    labels = [metric.label for metric in app.metric]
    assert "Blocks at the cap" in labels
    assert "Forecast error" in labels
    assert len(app.get("vega_lite_chart")) >= 3


def test_a_day_that_is_not_settled_has_no_price_line(folders: Path) -> None:
    calibration = calibration_table(30)
    later = calibration["date"].max() + pd.Timedelta(days=1)
    coming = calibration[calibration["date"] == calibration["date"].max()].assign(date=later)
    _forecast_file(folders, pd.concat([calibration, coming]), later)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    captions = " ".join(str(item.value) for item in app.caption)
    assert "Line: forecast" in captions
    assert "Orange" not in captions
