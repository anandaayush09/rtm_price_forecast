import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

from rtm_forecast.cli import main
from rtm_forecast.iex_export import parse_snapshot
from rtm_forecast.psp import parse_timeseries
from rtm_forecast.window import build_window
from test_iex_export import sheet as iex_sheet
from test_psp import sheet as psp_sheet

ROOT = Path(__file__).resolve().parents[1]


def test_clean_then_audit_writes_outputs(tmp_path: Path) -> None:
    settings = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    settings["data"]["raw_path"] = str(ROOT / "data" / "raw" / "price_data.csv")
    settings["data"]["clean_path"] = str(tmp_path / "clean.parquet")
    settings["data"]["report_path"] = str(tmp_path / "report.json")
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(settings), encoding="utf-8")

    main(["clean", "--params", str(params_file)])
    main(["audit", "--params", str(params_file)])

    assert len(pd.read_parquet(tmp_path / "clean.parquet")) == 10656
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["rtm_missing_rows"] == 12


DAYS = 40
LIGHTGBM = {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 8, "min_child_samples": 5}


def _window_table() -> pd.DataFrame:
    """Forty days from 1 June 2026; every third day settles at the cap."""
    rtm, dam, psp = [], [], []
    for offset in range(DAYS):
        day = date(2026, 6, 1) + timedelta(days=offset)
        price = "10000" if offset % 3 == 0 else str(2000 + 37 * offset)
        rtm.append(parse_snapshot(iex_sheet(day.strftime("%d-%m-%Y"), price=price), "rtm"))
        dam.append(
            parse_snapshot(iex_sheet(day.strftime("%d-%m-%Y"), market="dam", price=price), "dam")
        )
        report = (day + timedelta(days=1)).strftime("%d-%b-%Y")
        psp.append(parse_timeseries(psp_sheet(3, report=report)))
    return build_window(
        pd.concat(rtm, ignore_index=True),
        pd.concat(dam, ignore_index=True),
        pd.concat(psp, ignore_index=True),
        date(2026, 6, 1),
        date(2026, 6, 1) + timedelta(days=DAYS - 1),
    )


@pytest.fixture()
def production(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    """A params file whose paths are in ``tmp_path``, and the full window table."""
    settings = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    settings["window"]["clean_path"] = str(tmp_path / "window.parquet")
    settings["features"]["holidays_path"] = str(ROOT / "config" / "holidays.yaml")
    settings["production"] = {
        "model_dir": str(tmp_path / "model"),
        "calibration_path": str(tmp_path / "calibration.parquet"),
        "forecast_dir": str(tmp_path / "forecasts"),
        "lightgbm": LIGHTGBM,
    }
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return params_file, tmp_path, _window_table()


def test_fit_calibrate_forecast_and_settle_one_day(
    production: tuple[Path, Path, pd.DataFrame], capsys: pytest.CaptureFixture[str]
) -> None:
    params_file, folder, full = production
    last = pd.Timestamp(date(2026, 6, 1) + timedelta(days=DAYS - 1))
    # on the morning of the last day its RTM prices are not published yet
    morning = full.assign(rtm_price=full["rtm_price"].mask(full["date"] == last))
    morning.to_parquet(folder / "window.parquet", index=False)
    args = ["--params", str(params_file)]

    main(["fit", *args])
    main(["calibrate", "--days", "15", *args])
    table = pd.read_parquet(folder / "calibration.parquet")
    assert table["date"].nunique() == 15 and table["date"].max() == last - pd.Timedelta(days=1)
    with pytest.raises(SystemExit, match="--force"):
        main(["calibrate", "--days", "15", *args])
    main(["calibrate", "--days", "15", "--force", *args])

    main(["forecast", "--day", f"{last:%Y-%m-%d}", *args])
    forecast = pd.read_csv(folder / "forecasts" / f"forecast_{last:%Y-%m-%d}.csv")
    assert len(forecast) == 96
    assert forecast["point"].between(0, 10000).all()
    assert (forecast["lo90"] <= forecast["hi90"]).all()
    with pytest.raises(SystemExit, match="inside the days"):
        main(["forecast", "--day", f"{last - pd.Timedelta(days=1):%Y-%m-%d}", *args])

    with pytest.raises(SystemExit, match="no RTM prices"):
        main(["settle", "--day", f"{last:%Y-%m-%d}", *args])
    full.to_parquet(folder / "window.parquet", index=False)
    main(["settle", "--day", f"{last:%Y-%m-%d}", *args])
    assert pd.read_parquet(folder / "calibration.parquet")["date"].nunique() == 16
    with pytest.raises(SystemExit, match="already in the calibration"):
        main(["settle", "--day", f"{last:%Y-%m-%d}", *args])
    assert "calibration table now holds 16 days" in capsys.readouterr().out


def test_forecast_needs_the_model_to_match_the_settings_and_the_day_to_exist(
    production: tuple[Path, Path, pd.DataFrame], capsys: pytest.CaptureFixture[str]
) -> None:
    params_file, folder, full = production
    full.to_parquet(folder / "window.parquet", index=False)
    args = ["--params", str(params_file)]
    with pytest.raises(SystemExit):
        main(["forecast", *args])
    assert "forecast needs --day" in capsys.readouterr().err
    main(["fit", *args])
    main(["calibrate", "--days", "15", *args])
    with pytest.raises(SystemExit, match="not in the window table"):
        main(["forecast", "--day", "2026-07-11", *args])
    main(["fit", "--before", "2026-07-05", *args])
    with pytest.raises(SystemExit, match="run fit again"):
        main(["forecast", "--day", "2026-07-10", *args])
    settings = yaml.safe_load(params_file.read_text(encoding="utf-8"))
    settings["production"]["lightgbm"]["n_estimators"] = 31
    params_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(SystemExit, match="other settings"):
        main(["forecast", "--day", "2026-07-06", *args])
