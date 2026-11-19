"""Re-run the notebook's XGBoost grid search with two cross-validation schemes.

The notebook used scikit-learn's default for regressors: five consecutive folds, each validated on a
block that sits between training blocks, so training folds contain days after the validation days.
Here the same grid and the same inputs are scored with that scheme and with forward-chaining folds
(``TimeSeriesSplit``), and each winner is refit and scored on the notebook's test day.
"""

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit
from xgboost import XGBRegressor

from rtm_forecast.ingest import load_raw
from rtm_forecast.reference import build_features, mae, mape, prepare_frame, split_last_day

ROOT = Path(__file__).resolve().parents[1]
GRID = {
    "colsample_bytree": [0.5, 0.6, 0.7],
    "gamma": [0.05, 0.06],
    "learning_rate": [0.06, 0.07, 0.1],
    "max_depth": [4, 5, 6],
    "n_estimators": [100, 110],
}


def evaluate(cv: object, label: str, split: object) -> dict[str, object]:
    started = time.time()
    model = XGBRegressor(tree_method="exact", base_score=0.5, random_state=0, n_jobs=1)
    search = GridSearchCV(model, GRID, cv=cv, scoring="neg_mean_squared_error", n_jobs=1)
    search.fit(split.x_train, split.y_train)
    predicted = search.best_estimator_.predict(split.x_test)
    return {
        "scheme": label,
        "best_params": search.best_params_,
        "cv_score_neg_mse": round(float(search.best_score_), 2),
        "test_day_mape": round(mape(split.y_test, predicted), 3),
        "test_day_mae": round(mae(split.y_test, predicted), 2),
        "seconds": round(time.time() - started),
    }


def main() -> None:
    raw = load_raw(ROOT / "data" / "raw" / "price_data.csv")
    split = split_last_day(build_features(prepare_frame(raw)))
    results = [
        evaluate(KFold(n_splits=5), "KFold(5), consecutive folds (notebook default)", split),
        evaluate(TimeSeriesSplit(n_splits=5), "TimeSeriesSplit(5), forward chaining", split),
    ]
    out = ROOT / "results" / "phase1_cv_comparison.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(results).to_string())


if __name__ == "__main__":
    main()
