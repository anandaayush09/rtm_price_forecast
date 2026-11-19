"""Run the code cells of the original notebook unchanged, in a 2021-era environment.

Only these edits are made, all listed in PATCHES below: the data path, the imports that are needed only
for plots, and n_jobs in the two grid searches (the original used 48 cores; this machine has 8 cores and
7.7 GB of memory). Plot cells are skipped. Cell 90 calls a function that the
notebook never defines, so it is replaced by a stand-in that is described in the output.
"""

import ast
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "original" / "LD_Price_Forecast_Bagging_Boosting_Ensembles.ipynb"
DATA = ROOT / "data" / "raw" / "price_data.csv"
OUT = Path(os.environ.get("REPRO_OUT", Path(__file__).resolve().parent / "out"))

# Cells that run, in notebook order (0-based index in the notebook).
RUN = [
    11,
    12,
    14,
    15,
    17,
    18,
    22,
    25,
    27,
    29,
    31,
    34,
    36,
    52,
    53,
    54,
    55,
    58,
    59,
    63,
    64,
    65,
    66,
    67,
    72,
    74,
    76,
    78,
    86,
    88,
    90,
]

PATCHES = {
    12: [
        ("import matplotlib.pyplot as plt, seaborn as sns\n", ""),
        ("import statsmodels.api as sm\n", ""),
    ],
    72: [("n_jobs=-1", "n_jobs=4")],
    86: [("n_jobs = -1", "n_jobs = 4")],
    14: [("./Documents/PriceForecasting/v1/price_data.csv", str(DATA).replace("\\", "/"))],
}


def evalute_model(y_pred, y_test, n_pred_block):
    """Stand-in for the function cell 90 calls but the notebook never defines.

    It prints the same lines as evalute_xgb, but uses y_pred for the per-quarter errors and adds
    the full-day MAPE that the original cell 90 output contains.
    """
    test_diff = np.abs(y_pred - y_test)
    mape = lambda a, p: np.mean(np.abs((a - p) / a)) * 100  # noqa: E731
    diff_day, total, count = [], 0.0, 0
    for value in test_diff:
        if count != n_pred_block - 1:
            total += value
            count += 1
        else:
            diff_day.append(total / n_pred_block)
            total, count = 0.0, 0
    print("Maximum Error of the day = ", max(test_diff))
    print("Minimum Error of the day = ", min(test_diff))
    print("Mean Absolute Error of the day = ", diff_day)
    print("Average RTM Price = ", np.mean(y_test))
    for label, sl in [
        ("00:00-06:00", slice(0, 24)),
        ("06:00-12:00", slice(24, 48)),
        ("12:00-18:00", slice(48, 72)),
        ("18:00-24:00", slice(72, None)),
    ]:
        print(f"MAPE during {label} hrs = ", mape(y_test[sl], y_pred[sl]))
    print("Full Day MAPE = ", mape(y_test, y_pred))


def run_cell(source: str, namespace: dict, show_last: bool) -> str:
    buffer = io.StringIO()
    tree = ast.parse(source)
    last = (
        tree.body.pop() if show_last and tree.body and isinstance(tree.body[-1], ast.Expr) else None
    )
    with redirect_stdout(buffer):
        exec(compile(tree, "<cell>", "exec"), namespace)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
            if value is not None:
                print(repr(value))
    return buffer.getvalue()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace: dict = {"evalute_model": evalute_model}
    # Jupyter prints frames with up to 20 columns; the plain console would shorten them.
    import pandas

    pandas.set_option("display.max_columns", 20)
    log = []
    until = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    for index in [i for i in RUN if i <= until]:
        source = "".join(cells[index]["source"])
        for old, new in PATCHES.get(index, []):
            assert old in source, (index, old)
            source = source.replace(old, new)
        if index == 90:
            source = source.replace(
                "best_regr.fit(X_train,y_train)", "best_regr.fit(X_train,y_train)"
            )
        started = time.time()
        text = run_cell(source, namespace, show_last=True)
        seconds = time.time() - started
        log.append({"cell": index, "seconds": round(seconds, 1), "output": text})
        print(f"--- cell {index} ({seconds:.1f}s)\n{text}", flush=True)
        (OUT / "cells.json").write_text(json.dumps(log, indent=1), encoding="utf-8")

    if until < 90:
        return
    import pandas as pd

    for name, preds in [("RF", namespace["rf_predictions"]), ("XGB", namespace["xgb_predictions"])]:
        frame = pd.DataFrame(
            {
                "Actual RTM Price": namespace["y_test"],
                "Predicted RTM Price": preds,
                "Difference": namespace["y_test"] - preds,
            }
        )
        frame.to_csv(OUT / f"Forecast_{name}.csv", index=False)
    summary = {
        "xgb_best_params": namespace["gs_xgb"].best_params_,
        "xgb_best_score": namespace["gs_xgb"].best_score_,
        "rf_best_params": namespace["gs_rf"].best_params_,
        "rf_best_score": namespace["gs_rf"].best_score_,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
