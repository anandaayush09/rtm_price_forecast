"""Command line entry points used by the DVC pipeline."""

import argparse
import json
from pathlib import Path

import pandas as pd

from rtm_forecast.audit import build_report
from rtm_forecast.config import load_data_params
from rtm_forecast.contracts import clean_schema, raw_schema
from rtm_forecast.ingest import clean, load_raw


def run_clean(params_path: Path) -> None:
    params = load_data_params(params_path)
    raw = raw_schema().validate(load_raw(params.raw_path))
    table = clean_schema(params).validate(clean(raw))
    params.clean_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(params.clean_path, index=False)
    print(f"wrote {len(table)} rows to {params.clean_path}")


def run_audit(params_path: Path) -> None:
    params = load_data_params(params_path)
    table = pd.read_parquet(params.clean_path)
    report = build_report(table, params.stale_min_blocks)
    params.report_path.parent.mkdir(parents=True, exist_ok=True)
    params.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {params.report_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rtm-forecast")
    parser.add_argument("command", choices=["clean", "audit"])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args(argv)
    {"clean": run_clean, "audit": run_audit}[args.command](args.params)


if __name__ == "__main__":
    main()
