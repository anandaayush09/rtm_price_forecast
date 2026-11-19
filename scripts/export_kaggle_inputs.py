"""Write the price table for the GPU notebook to .project-local/kaggle_export (private)."""

from pathlib import Path

import pandas as pd

from rtm_forecast.config import (
    load_feature_params,
    load_window_experiment_params,
    load_window_params,
)
from rtm_forecast.kaggle_export import build_export
from rtm_forecast.splits import make_periods, rolling_folds

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    features = load_feature_params(ROOT / "params.yaml")
    experiment = load_window_experiment_params(ROOT / "params.yaml")
    window = pd.read_parquet(ROOT / load_window_params(ROOT / "params.yaml").clean_path)
    first = window["date"].min() + pd.Timedelta(days=features.min_history_days)
    days = [pd.Timestamp(d) for d in sorted(window["date"].unique()) if d >= first]
    periods = make_periods(days, experiment.validation_days, experiment.test_days)
    folds = rolling_folds(days, experiment.folds, experiment.fold_days)
    table = build_export(window, folds, periods, features.imputation)
    out = ROOT / ".project-local" / "kaggle_export"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "rtm_dam_prices.parquet", index=False)
    print(f"wrote {len(table)} rows, {table['fold'].notna().sum()} on forecast days, to {out}")


if __name__ == "__main__":
    main()
