import json
from pathlib import Path

import pandas as pd
import yaml

from rtm_forecast.cli import main

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
