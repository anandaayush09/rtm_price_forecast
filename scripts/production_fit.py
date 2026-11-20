"""Fit the final day-ahead model on all published days, save it, and forecast a day with it.

    python scripts/production_fit.py fit                  # fit, save, start the calibration
    python scripts/production_fit.py check                # refit the last test days, compare
    python scripts/production_fit.py forecast --day DATE  # forecast one new day

Everything is written under .project-local/production (private: the model is fitted on IEX data).

* ``fit`` uses the setup chosen on the validation days and the tuned LightGBM settings of the
  comparison, fits on every day with published prices, and saves the model. It also starts the
  calibration table from the forecasts of the backtest, which the margins of the intervals need.
* ``check`` is the test of the production code: for each of the last three test days it fits on
  the days before, forecasts the day with prices hidden, and compares the point forecast and the
  corrected intervals with the backtest results. It also reloads the saved model and checks that it
  gives the same forecasts as the one in memory.
* ``forecast`` refuses a day that the saved model was fitted on. To forecast a day after the last
  one in the data, rebuild the window table up to that day first; the day then needs its DAM
  results and demand, and no RTM price.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_window import Inputs, read_setup
from intervals_window import scored_variants

from rtm_forecast.production import (
    CALIBRATION_COLUMNS,
    ProductionModel,
    Spec,
    fit_production,
    forecast_day,
)

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / ".project-local" / "production"
MODEL_DIR = FOLDER / "model"
CALIBRATION = FOLDER / "calibration.parquet"
CHECKED_DAYS = 3


def spec_for(inputs: Inputs) -> tuple[Spec, pd.DataFrame]:
    setup = read_setup(inputs)
    best = json.loads((inputs.out / "best_params.json").read_text(encoding="utf-8"))
    transform = setup["transform"]
    spec = Spec(
        columns=tuple(setup["columns"]),
        params=best["lightgbm"]["params"],
        seed=inputs.experiment.seed,
        cap=inputs.experiment.price_cap,
        asinh_scale=None if transform is None else float(transform.scale),
    )
    return spec, setup["frame"]


def backtest_calibration(inputs: Inputs) -> pd.DataFrame:
    """The forecasts of the backtest as they were made, with the published prices."""
    raw = pd.read_parquet(inputs.out / "intervals_cap_mixture.parquet")
    point = pd.read_parquet(inputs.out / "predictions_cap_mixture.parquet")
    point = point[["date", "block", "prediction"]].rename(columns={"prediction": "point"})
    table = raw.merge(point, on=["date", "block"], how="left", validate="one_to_one")
    return table[CALIBRATION_COLUMNS].sort_values(["date", "block"], ignore_index=True)


def fit(inputs: Inputs) -> None:
    spec, frame = spec_for(inputs)
    model = fit_production(frame, spec)
    model.save(MODEL_DIR)
    print(
        f"fitted on {model.train_days} days ({model.train_rows} blocks) up to "
        f"{model.trained_until}, saved to {MODEL_DIR}"
    )
    if CALIBRATION.exists():
        print(f"calibration table kept: {CALIBRATION}")
        return
    table = backtest_calibration(inputs)
    table.to_parquet(CALIBRATION, index=False)
    print(f"calibration table started from the backtest: {table['date'].nunique()} days")


def check(inputs: Inputs) -> None:
    spec, frame = spec_for(inputs)
    calibration = backtest_calibration(inputs)
    variants = scored_variants(inputs)
    variants = variants[variants["variant"] == "two-stage, corrected by cap state"]
    saved = pd.read_parquet(inputs.out / "predictions_cap_mixture.parquet")
    saved = saved.set_index(["date", "block"])["prediction"]
    worst = {"point": 0.0, "lo50": 0.0, "hi50": 0.0, "lo90": 0.0, "hi90": 0.0}
    for day in inputs.periods.test[-CHECKED_DAYS:]:
        model = fit_production(frame, spec, before=day)
        rows = frame[frame["date"] == day].assign(rtm_price=np.nan)
        forecast = forecast_day(model, rows, calibration[calibration["date"] < day])
        keys = pd.MultiIndex.from_frame(forecast[["date", "block"]])
        worst["point"] = max(
            worst["point"],
            float(np.abs(forecast["point"].to_numpy() - saved.reindex(keys).to_numpy()).max()),
        )
        for coverage in (0.5, 0.9):
            tag = round(coverage * 100)
            theirs = variants[variants["coverage"] == coverage].set_index(["date", "block"])
            for end in ("lo", "hi"):
                gap = forecast[f"{end}{tag}"].to_numpy() - theirs[end].reindex(keys).to_numpy()
                worst[f"{end}{tag}"] = max(worst[f"{end}{tag}"], float(np.abs(gap).max()))
        print(f"{day:%Y-%m-%d}: trained until {model.trained_until}", flush=True)
    print("largest difference to the backtest (Rs/MWh):")
    for name, value in worst.items():
        print(f"  {name}: {value:.9f}")
    if (MODEL_DIR / "manifest.json").exists():
        loaded = ProductionModel.load(MODEL_DIR)
        rows = frame[frame["date"] == inputs.periods.test[-1]]
        gap = (loaded.raw(rows) - fit_production(frame, spec).raw(rows)).abs().to_numpy().max()
        print(
            f"saved model (trained until {loaded.trained_until}) against a fresh fit on all days:"
        )
        print(f"  largest difference {gap:.9f} (Rs/MWh)")


def forecast(inputs: Inputs, day: str) -> None:
    spec, frame = spec_for(inputs)
    model = ProductionModel.load(MODEL_DIR)
    if model.spec != spec:
        raise SystemExit("the saved model was fitted with other settings; run fit again")
    target = pd.Timestamp(day)
    if target.date() <= model.trained_until:
        raise SystemExit(
            f"{day} is inside the days the model was fitted on ({model.trained_until})"
        )
    rows = frame[frame["date"] == target]
    if rows.empty:
        raise SystemExit(f"{day} is not in the window table; rebuild it up to that day first")
    calibration = pd.read_parquet(CALIBRATION)
    result = forecast_day(model, rows.assign(rtm_price=np.nan), calibration)
    out = FOLDER / "forecasts"
    out.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out / f"forecast_{day}.parquet", index=False)
    print(f"wrote {out / f'forecast_{day}.parquet'} ({len(result)} blocks)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "check", "forecast"])
    parser.add_argument("--day")
    args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    inputs = Inputs(False)
    if args.stage == "fit":
        fit(inputs)
    elif args.stage == "check":
        check(inputs)
    else:
        if not args.day:
            raise SystemExit("forecast needs --day YYYY-MM-DD")
        forecast(inputs, args.day)


if __name__ == "__main__":
    main()
