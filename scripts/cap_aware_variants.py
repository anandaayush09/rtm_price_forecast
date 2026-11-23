"""Two changes to the final day-ahead model, tried on the 2025-26 window.

    python scripts/cap_aware_variants.py fit        # refit the model with each recency weighting
    python scripts/cap_aware_variants.py evaluate   # recalibration of the cap probability, scores

* Recency weighting: each training block gets the weight 0.5 ** (age in days / half-life), so
  that recent days count more. The half-lives are 30, 60 and 120 days, and ``none`` is the final
  model as it is (every day counts the same).
* Recalibration: the cap probability of a day is mapped through an isotonic regression fitted on
  the probabilities of the 60 days before it and what happened (at least 14 days, else the
  probability is left as it is). The fits are out-of-sample, so no day is used to calibrate itself.

The tuned LightGBM settings of the plain model are reused and nothing is tuned here. The
variant is chosen on the validation days only; the test days are only scored. Results are
private files under .project-local/results_window/variants and are not committed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_window import Inputs, read_setup
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from rtm_forecast import models
from rtm_forecast.metrics import daily_mae, expected_calibration_error
from rtm_forecast.significance import bootstrap_difference, diebold_mariano

HALF_LIVES: dict[str, float | None] = {"none": None, "hl30": 30.0, "hl60": 60.0, "hl120": 120.0}
LEVELS = np.array([0.5, 0.75, 0.9])
WINDOW_DAYS = 60
MIN_DAYS = 14


def out_dir(inputs: Inputs) -> Path:
    return inputs.out / "variants"


def fit_day(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    settings: dict[str, object],
    cap: float,
    half_life: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap probability and below-cap quantiles (one column per level) for the rows of ``test``."""
    age = (test["date"].iloc[0] - train["date"]).dt.days.to_numpy(dtype=float)
    weight = None if half_life is None else 0.5 ** (age / half_life)
    at_cap = train["rtm_price"].to_numpy(dtype=float) >= cap
    classifier = LGBMClassifier(**settings).fit(train[columns], at_cap, sample_weight=weight)
    p = classifier.predict_proba(test[columns])[:, 1]
    below = ~at_cap
    quantiles = np.stack(
        [
            LGBMRegressor(**settings, objective="quantile", alpha=float(level))
            .fit(
                train.loc[below, columns],
                train.loc[below, "rtm_price"],
                sample_weight=None if weight is None else weight[below],
            )
            .predict(test[columns])
            for level in LEVELS
        ],
        axis=1,
    )
    return p, np.sort(quantiles, axis=1)


def fit(inputs: Inputs, redo: bool) -> None:
    setup = read_setup(inputs)
    if setup["transform"] is not None:
        raise ValueError("this script uses raw prices, as the final model does")
    best = json.loads((inputs.out / "best_params.json").read_text(encoding="utf-8"))
    settings = models.two_stage_settings(best["lightgbm"]["params"], inputs.experiment.seed)
    frame: pd.DataFrame = setup["frame"]
    columns: list[str] = setup["columns"]
    cap = inputs.experiment.price_cap
    days = [d for fold in inputs.folds for d in fold]
    fold_of = {day: k + 1 for k, fold in enumerate(inputs.folds) for day in fold}
    out_dir(inputs).mkdir(parents=True, exist_ok=True)
    for name, half_life in HALF_LIVES.items():
        target = out_dir(inputs) / f"{name}.parquet"
        if target.exists() and not redo:
            print(f"{name}: done, skipped", flush=True)
            continue
        started = time.time()
        parts = []
        for day in days:
            train = frame[(frame["date"] < day) & frame["target_observed"]]
            test = frame[frame["date"] == day]
            p, q = fit_day(train, test, columns, settings, cap, half_life)
            part = test[["date", "block", "rtm_price"]].rename(columns={"rtm_price": "actual"})
            part["p"] = p
            for k, level in enumerate(LEVELS):
                part[f"q{round(level * 100)}"] = q[:, k]
            parts.append(part)
        result = pd.concat(parts, ignore_index=True)
        result["fold"] = result["date"].map(fold_of)
        result["period"] = np.select(
            [
                result["date"].isin(inputs.periods.test),
                result["date"].isin(inputs.periods.validation),
            ],
            ["test", "validation"],
            "earlier",
        )
        result.to_parquet(target, index=False)
        print(f"{name}: {time.time() - started:.0f}s", flush=True)


def recalibrate(result: pd.DataFrame, cap: float) -> np.ndarray:
    """Isotonic map of each day's probability, fitted on the 60 days before it."""
    calibrated = result["p"].to_numpy(dtype=float).copy()
    observed = result["actual"].notna().to_numpy()
    at_cap = (result["actual"] >= cap).to_numpy(dtype=float)
    dates = result["date"].to_numpy()
    for day in np.unique(dates):
        earlier = (dates < day) & (dates >= day - np.timedelta64(WINDOW_DAYS, "D")) & observed
        if len(np.unique(dates[earlier])) < MIN_DAYS:
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(result["p"].to_numpy()[earlier], at_cap[earlier])
        today = dates == day
        calibrated[today] = iso.predict(result["p"].to_numpy()[today])
    return calibrated


def forecast(result: pd.DataFrame, p: np.ndarray, cap: float) -> np.ndarray:
    quantiles = result[["q50", "q75", "q90"]].to_numpy()
    return models.mixture_quantile(p, quantiles, LEVELS, 0.5, cap)


def evaluate(inputs: Inputs) -> None:
    cap = inputs.experiment.price_cap
    rows, forecasts = [], []
    for name in HALF_LIVES:
        result = pd.read_parquet(out_dir(inputs) / f"{name}.parquet")
        for source in ("raw", "isotonic"):
            p = result["p"].to_numpy() if source == "raw" else recalibrate(result, cap)
            prediction = forecast(result, p, cap)
            variant = f"{name} / {source}"
            table = result[["date", "block", "actual", "fold", "period"]].copy()
            table["prediction"], table["model"], table["p"] = prediction, variant, p
            forecasts.append(table)
            ok = table["actual"].notna()
            for period in ("validation", "test"):
                d = table[ok & (table["period"] == period)]
                y = (d["actual"] >= cap).to_numpy()
                rows.append(
                    {
                        "variant": variant,
                        "period": period,
                        "MAE": float((d["actual"] - d["prediction"]).abs().mean()),
                        "brier": brier_score_loss(y, d["p"]),
                        "calibration_error": expected_calibration_error(y, d["p"].to_numpy()),
                    }
                )
    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir(inputs) / "scores.csv", index=False)
    wide = scores.pivot(
        index="variant", columns="period", values=["MAE", "brier", "calibration_error"]
    )
    print(wide.round(4).to_string())
    validation = scores[scores["period"] == "validation"].set_index("variant")["MAE"]
    chosen = str(validation.idxmin())
    base = "none / raw"
    print(
        f"\nchosen on the validation days only: {chosen} (validation MAE {validation[chosen]:.1f}, "
        f"final model {validation[base]:.1f})"
    )
    allp = pd.concat(forecasts, ignore_index=True)
    test = daily_mae(allp[allp["period"] == "test"]).dropna(axis=1)
    a, b = test[chosen].to_numpy(), test[base].to_numpy()
    low, high = bootstrap_difference(a, b)
    print(
        f"test days: {chosen} against {base}: mean daily difference {(a - b).mean():.1f} "
        f"({low:.1f} to {high:.1f}), p {diebold_mariano(a, b)[1]:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "evaluate"])
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    inputs = Inputs(False)
    {"fit": lambda: fit(inputs, args.redo), "evaluate": lambda: evaluate(inputs)}[args.stage]()


if __name__ == "__main__":
    main()
