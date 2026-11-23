"""Forecast accuracy measures. Prices are in INR/MWh."""

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, in percent."""
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE, in percent: the error over the mean of the two values."""
    return float(
        np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100
    )


def summarise(frame: pd.DataFrame, reference: str = "naive_lastweek") -> pd.DataFrame:
    """One row per model from long predictions (columns model, date, block, actual, prediction).

    rMAE is the model's MAE divided by the MAE of ``reference`` on the same rows, so a value
    below 1 means better than the reference.
    """
    observed = frame[frame["actual"].notna()]
    rows = {}
    for model, part in observed.groupby("model"):
        actual = part["actual"].to_numpy()
        predicted = part["prediction"].to_numpy()
        rows[model] = {
            "MAE": mae(actual, predicted),
            "RMSE": rmse(actual, predicted),
            "MAPE": mape(actual, predicted),
            "sMAPE": smape(actual, predicted),
        }
    table = pd.DataFrame(rows).T
    if reference in table.index:
        table["rMAE"] = table["MAE"] / float(table["MAE"].loc[reference])
    return table.sort_values("MAE")


def daily_mae(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error of each model on each day (rows: days, columns: models)."""
    observed = frame[frame["actual"].notna()].copy()
    observed["error"] = (observed["actual"] - observed["prediction"]).abs()
    return observed.pivot_table(index="date", columns="model", values="error", aggfunc="mean")


def cap_scores(frame: pd.DataFrame, cap: float, forecast_share: float) -> pd.DataFrame:
    """How well each model handles blocks at the price cap.

    A block is at the cap when the published price is at least ``cap``; a forecast calls the cap
    when it is at least ``forecast_share * cap``. Also the MAE on cap blocks and on the others.
    """
    observed = frame[frame["actual"].notna()]
    rows = {}
    for model, part in observed.groupby("model"):
        actual = part["actual"].to_numpy()
        predicted = part["prediction"].to_numpy()
        is_cap = actual >= cap
        called = predicted >= forecast_share * cap
        hits = int((is_cap & called).sum())
        precision = hits / int(called.sum()) if called.any() else float("nan")
        recall = hits / int(is_cap.sum()) if is_cap.any() else float("nan")
        both = precision + recall
        rows[model] = {
            "cap_share": float(is_cap.mean()),
            "cap_precision": precision,
            "cap_recall": recall,
            "cap_f1": 2 * precision * recall / both if both > 0 else float("nan"),
            "MAE_cap": mae(actual[is_cap], predicted[is_cap]) if is_cap.any() else float("nan"),
            "MAE_other": mae(actual[~is_cap], predicted[~is_cap])
            if (~is_cap).any()
            else float("nan"),
        }
    return pd.DataFrame(rows).T


def expected_calibration_error(
    outcome: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    """Expected calibration error of probabilities for a yes/no outcome.

    The probabilities are cut into ``bins`` equal-width bins. In each bin the absolute gap
    between the mean probability and the observed share of yes is weighted by the share of
    blocks in the bin.
    """
    outcome = np.asarray(outcome, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        inside = index == b
        if inside.any():
            gap = abs(outcome[inside].mean() - probability[inside].mean())
            total += float(inside.mean()) * gap
    return total
