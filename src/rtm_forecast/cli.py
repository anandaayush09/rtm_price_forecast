"""Command line entry points used by the DVC pipeline."""

import argparse
import json
from pathlib import Path
from typing import cast

import pandas as pd

from rtm_forecast import iex_export, psp
from rtm_forecast.audit import build_report
from rtm_forecast.calendar import load_holidays
from rtm_forecast.config import load_data_params, load_feature_params, load_window_params
from rtm_forecast.contracts import clean_schema, raw_schema, window_schema
from rtm_forecast.features import build_features, feature_groups
from rtm_forecast.ingest import BLOCKS_PER_DAY, clean, load_raw
from rtm_forecast.transforms import AsinhTransform, choose_scale
from rtm_forecast.window import build_window, window_report


def run_clean(params_path: Path) -> None:
    params = load_data_params(params_path)
    raw = raw_schema().validate(load_raw(params.raw_path))
    table = clean_schema(params).validate(clean(raw))
    params.clean_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(params.clean_path, index=False)
    print(f"wrote {len(table)} rows to {params.clean_path}")


def run_clean_window(params_path: Path) -> None:
    data = load_data_params(params_path)
    window = load_window_params(params_path)
    rtm = iex_export.load_folder(window.rtm_dir, "rtm")
    dam = iex_export.load_folder(window.dam_dir, "dam")
    sheets = psp.load_folder(window.psp_dir)
    table = window_schema(data).validate(build_window(rtm, dam, sheets, window.start, window.end))
    first, last = pd.Timestamp(window.start), pd.Timestamp(window.end)
    outside = {
        name: int((~source["date"].between(first, last)).sum())
        for name, source in (("rtm", rtm), ("dam", dam), ("demand", sheets))
    }
    layouts = {
        str(layout): int(count) for layout, count in table["psp_layout"].value_counts().items()
    }
    report = window_report(table, outside, layouts)
    window.clean_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(window.clean_path, index=False)
    window.report_path.parent.mkdir(parents=True, exist_ok=True)
    window.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(table)} rows to {window.clean_path}")


def run_audit(params_path: Path) -> None:
    params = load_data_params(params_path)
    table = pd.read_parquet(params.clean_path)
    report = build_report(table, params.stale_min_blocks)
    params.report_path.parent.mkdir(parents=True, exist_ok=True)
    params.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {params.report_path}")


def run_features(params_path: Path) -> None:
    data = load_data_params(params_path)
    params = load_feature_params(params_path)
    table = pd.read_parquet(data.clean_path)
    holidays = load_holidays(params.holidays_path)
    features = build_features(table, params, holidays)
    params.features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(params.features_path, index=False)
    training = features["rtm_price"].dropna().to_numpy()[: params.asinh_train_days * BLOCKS_PER_DAY]
    scale = choose_scale(training, params.asinh_scales)
    transformed = AsinhTransform(scale).forward(features["rtm_price"].dropna().to_numpy())
    groups = feature_groups(params)
    report = {
        "rows": int(len(features)),
        "first_day": features["date"].min().date().isoformat(),
        "last_day": features["date"].max().date().isoformat(),
        "target_observed_rows": int(features["target_observed"].sum()),
        "feature_groups": groups,
        "missing_inputs": {
            name: int(features[name].isna().sum())
            for name in sum(groups.values(), [])
            if features[name].isna().any()
        },
        "imputation": params.imputation,
        "demand_policy": params.demand_policy,
        "asinh_scale": scale,
        "price_skew_raw": round(cast(float, features["rtm_price"].skew()), 3),
        "price_skew_asinh": round(cast(float, pd.Series(transformed).skew()), 3),
    }
    params.report_path.parent.mkdir(parents=True, exist_ok=True)
    params.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(features)} rows to {params.features_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rtm-forecast")
    parser.add_argument("command", choices=["clean", "clean-window", "audit", "features"])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args(argv)
    commands = {
        "clean": run_clean,
        "clean-window": run_clean_window,
        "audit": run_audit,
        "features": run_features,
    }
    commands[args.command](args.params)


if __name__ == "__main__":
    main()
