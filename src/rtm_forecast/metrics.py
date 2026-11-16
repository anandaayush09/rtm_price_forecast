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
