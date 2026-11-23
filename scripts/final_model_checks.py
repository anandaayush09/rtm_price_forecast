"""Checks of the final day-ahead model that come after the comparison (2025-26 window).

    python scripts/final_model_checks.py holm          # tests corrected for many comparisons
    python scripts/final_model_checks.py calibration   # how well the cap probability is calibrated
    python scripts/final_model_checks.py ablation      # plain LightGBM with cap-history inputs
    python scripts/final_model_checks.py timing        # how long one daily cycle takes

``holm`` and ``ablation`` read the prediction files of the comparison, and ``calibration`` and
``ablation`` refit the models day by day on the 150 days of the five rolling folds, so they need
the private window table. Results go to .project-local/results_window/checks and are not
committed.
"""

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_window import Inputs, read_setup
from lightgbm import LGBMClassifier
from production_fit import backtest_calibration, spec_for
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from rtm_forecast import models
from rtm_forecast.backtest import rolling_origin
from rtm_forecast.metrics import daily_mae, expected_calibration_error
from rtm_forecast.production import ProductionModel, fit_production, forecast_day
from rtm_forecast.significance import bootstrap_difference, diebold_mariano, holm_adjust

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / ".project-local" / "results_window"
OUT = RESULTS / "checks"


def read_predictions() -> pd.DataFrame:
    files = sorted(RESULTS.glob("predictions_*.parquet"))
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return frame[~frame["model"].str.endswith("_check")]


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Mean daily difference of two forecasts, its 95 percent interval and the one-sided p-value."""
    low, high = bootstrap_difference(a, b)
    return {
        "mean_daily_diff": float((a - b).mean()),
        "ci_low": low,
        "ci_high": high,
        "p": diebold_mariano(a, b)[1],
    }


def run_holm() -> None:
    frame = read_predictions()
    test = daily_mae(frame[frame["period"] == "test"]).dropna(axis=1)
    rows = [
        {"comparison": f"cap_mixture vs {model}"}
        | compare(test["cap_mixture"].to_numpy(), test[model].to_numpy())
        for model in test.columns
        if model != "cap_mixture"
    ]
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p"].to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "holm_test.csv", index=False)
    print(f"Test days ({len(test)}): the final model against every other forecast")
    print(table.round(6).to_string(index=False))
    folds = []
    for fold in sorted(frame["fold"].dropna().unique()):
        part = daily_mae(frame[frame["fold"] == fold]).dropna(axis=1)
        a, b = part["cap_mixture"].to_numpy(), part["naive_yesterday"].to_numpy()
        folds.append({"fold": int(fold), "days": len(part)} | compare(a, b))
    by_fold = pd.DataFrame(folds)
    by_fold["p_holm"] = holm_adjust(by_fold["p"].to_numpy())
    by_fold.to_csv(OUT / "holm_folds.csv", index=False)
    print("\nEach fold: the final model against the same block yesterday")
    print(by_fold.round(6).to_string(index=False))


def run_calibration() -> None:
    inputs = Inputs(False)
    spec, frame = spec_for(inputs)
    settings = models.two_stage_settings(spec.params, spec.seed)
    days = [d for fold in inputs.folds for d in fold]
    fold_of = {d: k + 1 for k, fold in enumerate(inputs.folds) for d in fold}
    parts = []
    for i, day in enumerate(days):
        train = frame[(frame["date"] < day) & frame["target_observed"]]
        test = frame[frame["date"] == day]
        at_cap = train["rtm_price"].to_numpy(dtype=float) >= spec.cap
        classifier = LGBMClassifier(**settings).fit(train[list(spec.columns)], at_cap)
        part = test[["date", "block", "rtm_price", "rtm_price_lag1d"]].copy()
        part["p"] = classifier.predict_proba(test[list(spec.columns)])[:, 1]
        part["base_rate"] = float(at_cap.mean())
        parts.append(part)
        if i % 25 == 0:
            print(f"day {i + 1} of {len(days)}", flush=True)
    result = pd.concat(parts, ignore_index=True)
    result["fold"] = result["date"].map(fold_of)
    result["period"] = np.select(
        [result["date"].isin(inputs.periods.test), result["date"].isin(inputs.periods.validation)],
        ["test", "validation"],
        "earlier",
    )
    result = result[result["rtm_price"].notna()].copy()
    result["at_cap"] = (result["rtm_price"] >= spec.cap).astype(int)
    result["yesterday_at_cap"] = (result["rtm_price_lag1d"] >= spec.cap).astype(float)
    result.loc[result["rtm_price_lag1d"].isna(), "yesterday_at_cap"] = np.nan
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT / "cap_probability.parquet", index=False)
    rows = []
    groups = {name: result["period"] == name for name in ("test", "validation", "earlier")}
    for label, chosen in {**groups, "all days": result["period"].notna()}.items():
        d = result[chosen]
        y, p = d["at_cap"].to_numpy(), d["p"].to_numpy()
        known = d["yesterday_at_cap"].notna().to_numpy()
        brier = brier_score_loss(y, p)
        rows.append(
            {
                "group": label,
                "blocks": len(d),
                "cap_share": float(y.mean()),
                "mean_p": float(p.mean()),
                "brier": brier,
                "brier_base_rate": brier_score_loss(y, d["base_rate"].to_numpy()),
                "brier_yesterday": brier_score_loss(y[known], d["yesterday_at_cap"][known]),
                "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                "auc": roc_auc_score(y, p),
                "accuracy_at_half": float(((p >= 0.5) == (y == 1)).mean()),
                "calibration_error": expected_calibration_error(y, p),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "cap_calibration.csv", index=False)
    print(summary.round(4).to_string(index=False))
    test = result[result["period"] == "test"]
    cut = pd.cut(test["p"], np.linspace(0, 1, 11), include_lowest=True)
    table = test.groupby(cut, observed=True).agg(
        blocks=("p", "size"), mean_p=("p", "mean"), observed=("at_cap", "mean")
    )
    table.to_csv(OUT / "cap_reliability_test.csv")
    print("\nReliability on the test days")
    print(table.round(3).to_string())


def run_ablation() -> None:
    inputs = Inputs(False)
    spec, frame = spec_for(inputs)
    frame = frame.copy()
    # cap history that is known on day D: the same block a day and a week earlier, and the
    # share of cap blocks on the previous day and on the previous seven days
    daily = frame.groupby("date")["rtm_price"].apply(lambda s: float((s >= spec.cap).mean()))
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    previous = daily.shift(1)
    for lag in ("1d", "7d"):
        price = frame[f"rtm_price_lag{lag}"]
        frame[f"cap_lag{lag}"] = (price >= spec.cap).astype(float).where(price.notna())
    frame["cap_share_prev_day"] = frame["date"].map(previous)
    frame["cap_share_prev_7d"] = frame["date"].map(previous.rolling(7, min_periods=3).mean())
    extra = ["cap_lag1d", "cap_lag7d", "cap_share_prev_day", "cap_share_prev_7d"]
    predictor = models.lightgbm_predictor(
        [*spec.columns, *extra], read_setup(inputs)["transform"], spec.params, spec.seed
    )
    days = [d for fold in inputs.folds for d in fold]
    fold_of = {d: k + 1 for k, fold in enumerate(inputs.folds) for d in fold}
    result = rolling_origin(frame, days, predictor, "lightgbm_cap_history")
    result["fold"] = result["date"].map(fold_of)
    result["period"] = np.select(
        [result["date"].isin(inputs.periods.test), result["date"].isin(inputs.periods.validation)],
        ["test", "validation"],
        "earlier",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT / "predictions_lightgbm_cap_history.parquet", index=False)
    others = [
        pd.read_parquet(RESULTS / f"predictions_{name}.parquet")
        for name in ("lightgbm", "cap_mixture", "naive_yesterday")
    ]
    frame_all = pd.concat([result, *others], ignore_index=True)
    rows = []
    for period in ("test", "validation", "earlier"):
        for model, g in frame_all[frame_all["period"] == period].groupby("model"):
            g = g[g["actual"].notna()]
            error = (g["actual"] - g["prediction"]).abs()
            at_cap = g["actual"] >= spec.cap
            rows.append(
                {
                    "period": period,
                    "model": model,
                    "MAE": float(error.mean()),
                    "MAE_cap": float(error[at_cap].mean()),
                    "MAE_other": float(error[~at_cap].mean()),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ablation_scores.csv", index=False)
    print(table.round(1).to_string(index=False))
    test = daily_mae(frame_all[frame_all["period"] == "test"]).dropna(axis=1)
    for a, b in (("cap_mixture", "lightgbm_cap_history"), ("lightgbm_cap_history", "lightgbm")):
        c = compare(test[a].to_numpy(), test[b].to_numpy())
        print(
            f"test days, {a} against {b}: difference {c['mean_daily_diff']:.1f} "
            f"({c['ci_low']:.1f} to {c['ci_high']:.1f}), p {c['p']:.4f}"
        )


def run_timing() -> None:
    inputs = Inputs(False)
    spec, frame = spec_for(inputs)
    fits = []
    for _ in range(3):
        started = time.perf_counter()
        model = fit_production(frame, spec)
        fits.append(time.perf_counter() - started)
    print(
        f"fit on {model.train_days} days ({model.train_rows} blocks), seconds:",
        [round(x, 1) for x in fits],
        "median",
        round(statistics.median(fits), 1),
    )
    with tempfile.TemporaryDirectory() as folder:
        started = time.perf_counter()
        model.save(Path(folder) / "model")
        saved = time.perf_counter() - started
        size = sum(f.stat().st_size for f in (Path(folder) / "model").rglob("*") if f.is_file())
        started = time.perf_counter()
        ProductionModel.load(Path(folder) / "model")
        loaded = time.perf_counter() - started
    print(f"save {saved:.2f} s, load {loaded:.2f} s, model folder {size / 1e6:.2f} MB")
    day = pd.Timestamp(frame["date"].max())
    calibration = backtest_calibration(inputs)
    earlier = fit_production(frame, spec, before=day)
    rows = frame[frame["date"] == day]
    times = []
    for _ in range(5):
        started = time.perf_counter()
        forecast_day(earlier, rows, calibration)
        times.append(time.perf_counter() - started)
    print(f"forecast of one day, seconds: {[round(x, 3) for x in times]}")
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "rtm_forecast.cli", "--help"], capture_output=True, check=False
    )
    print(f"start-up of the command line: {time.perf_counter() - started:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["holm", "calibration", "ablation", "timing"])
    args = parser.parse_args()
    {
        "holm": run_holm,
        "calibration": run_calibration,
        "ablation": run_ablation,
        "timing": run_timing,
    }[args.check]()


if __name__ == "__main__":
    main()
