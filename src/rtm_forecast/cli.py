"""Command line entry points: the DVC pipeline stages and the day-ahead forecast.

Forecast of day D, once its DAM results and demand are in the window table (set ``window.end`` to D
and run ``clean-window``; the RTM prices of D are not there yet):

    rtm-forecast fit                      fit the final model on all published days
    rtm-forecast calibrate                once: forecasts of the last 60 days for the margins
    rtm-forecast forecast --day D         point forecast and intervals, one csv per day
    rtm-forecast settle --day D           next day, after the prices of D are published

``fit`` is run again every day, and ``settle`` adds the forecast of D to the calibration table.
"""

import argparse
import json
from pathlib import Path
from typing import cast

import pandas as pd

from rtm_forecast import iex_export, psp
from rtm_forecast.audit import build_report
from rtm_forecast.calendar import load_holidays
from rtm_forecast.config import (
    ProductionParams,
    load_data_params,
    load_feature_params,
    load_production_params,
    load_window_params,
)
from rtm_forecast.contracts import clean_schema, raw_schema, window_schema
from rtm_forecast.features import build_features, feature_groups
from rtm_forecast.ingest import BLOCKS_PER_DAY, clean, load_raw
from rtm_forecast.production import (
    WINDOW_DAYS,
    ProductionModel,
    Spec,
    build_calibration,
    fit_production,
    forecast_day,
    update_calibration,
)
from rtm_forecast.transforms import AsinhTransform, choose_scale
from rtm_forecast.window import build_window, window_report
from rtm_forecast.window_features import build_window_features, window_feature_groups


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


def _window_features(params_path: Path) -> pd.DataFrame:
    """Inputs of every block in the window table (the demand definition of the final model)."""
    features = load_feature_params(params_path)
    window = pd.read_parquet(load_window_params(params_path).clean_path)
    holidays = load_holidays(features.holidays_path)
    return build_window_features(window, features, holidays)


def _spec(params_path: Path) -> tuple[ProductionParams, Spec]:
    production = load_production_params(params_path)
    columns = sum(window_feature_groups(load_feature_params(params_path)).values(), [])
    spec = Spec(
        columns=tuple(columns),
        params=production.lightgbm,
        seed=production.seed,
        cap=production.price_cap,
    )
    return production, spec


def run_fit(params_path: Path, before: str | None) -> None:
    """Fit the final model on all published days (or those before ``before``) and save it."""
    production, spec = _spec(params_path)
    stop = None if before is None else pd.Timestamp(before)
    model = fit_production(_window_features(params_path), spec, stop)
    model.save(production.model_dir)
    print(
        f"fitted on {model.train_days} days ({model.train_rows} blocks) up to "
        f"{model.trained_until}, saved to {production.model_dir}"
    )


def run_calibrate(params_path: Path, days: int, force: bool) -> None:
    """Start the calibration table: forecast each of the last ``days`` published days with a
    model fitted on the days before it. Takes a few seconds per day."""
    production, spec = _spec(params_path)
    if production.calibration_path.exists() and not force:
        raise SystemExit(f"{production.calibration_path} exists; use --force to replace it")
    features = _window_features(params_path)
    published = features[features["target_observed"]]["date"].drop_duplicates().sort_values()
    table = build_calibration(features, spec, [pd.Timestamp(d) for d in published.tail(days)])
    production.calibration_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(production.calibration_path, index=False)
    print(f"wrote {table['date'].nunique()} days to {production.calibration_path}")


def run_forecast(params_path: Path, day: str) -> None:
    """Forecast one day with the saved model; the day needs its DAM results and demand."""
    production, spec = _spec(params_path)
    target = pd.Timestamp(day)
    model = ProductionModel.load(production.model_dir)
    if model.spec != spec:
        raise SystemExit("the saved model was fitted with other settings; run fit again")
    if target.date() <= model.trained_until:
        raise SystemExit(
            f"{day} is inside the days the model was fitted on (up to {model.trained_until}); "
            "to forecast it, fit on the days before it"
        )
    if model.trained_until < (target - pd.Timedelta(days=1)).date():
        raise SystemExit(f"the model was fitted up to {model.trained_until}; run fit again")
    features = _window_features(params_path)
    rows = features[features["date"] == target]
    if rows.empty:
        raise SystemExit(f"{day} is not in the window table; set window.end and run clean-window")
    calibration = pd.read_parquet(production.calibration_path)
    try:
        result = forecast_day(model, rows.assign(rtm_price=float("nan")), calibration)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    production.forecast_dir.mkdir(parents=True, exist_ok=True)
    path = production.forecast_dir / f"forecast_{day}.csv"
    result.to_csv(path, index=False)
    print(f"wrote {len(result)} blocks to {path}")


def run_settle(params_path: Path, day: str) -> None:
    """Add a forecast day to the calibration table once its prices are in the window table."""
    production, _ = _spec(params_path)
    path = production.forecast_dir / f"forecast_{day}.csv"
    if not path.exists():
        raise SystemExit(f"no forecast file {path}")
    forecast = pd.read_csv(path, parse_dates=["date"])
    window = pd.read_parquet(load_window_params(params_path).clean_path)
    prices = window.loc[window["date"] == pd.Timestamp(day), ["date", "block", "rtm_price"]]
    if prices["rtm_price"].notna().sum() == 0:
        raise SystemExit(f"no RTM prices published for {day} in the window table")
    calibration = pd.read_parquet(production.calibration_path)
    try:
        grown = update_calibration(calibration, forecast, prices)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    grown.to_parquet(production.calibration_path, index=False)
    print(f"calibration table now holds {grown['date'].nunique()} days, last {day}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rtm-forecast")
    commands = [
        "clean",
        "clean-window",
        "audit",
        "features",
        "fit",
        "calibrate",
        "forecast",
        "settle",
    ]
    parser.add_argument("command", choices=commands)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--day", help="forecast and settle: the day, YYYY-MM-DD")
    parser.add_argument("--before", help="fit: use only days before this one, YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=WINDOW_DAYS, help="calibrate: days to forecast")
    parser.add_argument("--force", action="store_true", help="calibrate: replace the table")
    args = parser.parse_args(argv)
    if args.command in ("forecast", "settle") and not args.day:
        parser.error(f"{args.command} needs --day YYYY-MM-DD")
    pipeline = {
        "clean": run_clean,
        "clean-window": run_clean_window,
        "audit": run_audit,
        "features": run_features,
    }
    if args.command in pipeline:
        pipeline[args.command](args.params)
    elif args.command == "fit":
        run_fit(args.params, args.before)
    elif args.command == "calibrate":
        run_calibrate(args.params, args.days, args.force)
    elif args.command == "forecast":
        run_forecast(args.params, args.day)
    else:
        run_settle(args.params, args.day)


if __name__ == "__main__":
    main()
