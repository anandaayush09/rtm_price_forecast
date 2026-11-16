"""Compare what the 2021-era rerun printed with what the original notebook printed.

Reads the outputs stored in the original notebook and the two output files written by
``repro2020/run_original_cells.py``, and writes ``results/phase1_reproduction.json``.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "original" / "LD_Price_Forecast_Bagging_Boosting_Ensembles.ipynb"
PREP = ROOT / "repro2020" / "out_dataprep" / "cells.json"
FULL = ROOT / "repro2020" / "out" / "cells.json"

WHOLE_CELLS = [15, 17, 18, 25, 27, 29, 31, 36, 53, 54, 55, 58, 59, 64, 67]
LINE_CELLS = {
    72: ["Best Params", "Best Score"],
    78: [
        "Maximum Error",
        "Minimum Error",
        "Mean Absolute Error",
        "Average RTM Price",
        "MAPE during 00:00",
        "MAPE during 06:00",
        "MAPE during 12:00",
        "MAPE during 18:00",
    ],
    86: ["Best Params", "Best Score"],
    88: ["Average Error", "Accuracy"],
    90: [
        "Maximum Error",
        "Minimum Error",
        "Mean Absolute Error",
        "Average RTM Price",
        "MAPE during 00:00",
        "MAPE during 06:00",
        "MAPE during 12:00",
        "MAPE during 18:00",
        "Full Day MAPE",
    ],
}


def original_text(cell: dict[str, Any]) -> str:
    parts = []
    for output in cell.get("outputs", []):
        if output["output_type"] == "stream":
            parts.append("".join(output["text"]))
        elif output["output_type"] == "execute_result":
            parts.append("".join(output["data"].get("text/plain", "")))
    return "\n".join(parts)


def normal(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def find_line(text: str, start: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip().startswith(start)), "")


def estimator_text(text: str) -> str:
    """The printed XGBoost estimator, with n_jobs removed (48 cores then, 8 now)."""
    match = re.search(r"Best Estimator: (XGBRegressor\(.*?\))\n", text, re.S)
    return re.sub(r"n_jobs=\d+,\s*", "", match.group(1)) if match else ""


def forecast_difference(name: str) -> float:
    """Largest absolute gap between the rerun forecast and the forecast saved in 2021."""

    def predictions(path: Path) -> list[float]:
        with path.open(newline="", encoding="utf-8") as handle:
            return [float(row["Predicted RTM Price"]) for row in csv.DictReader(handle)]

    saved = predictions(ROOT / "original" / "results" / f"Forecast_{name}.csv")
    rerun = predictions(ROOT / "repro2020" / "out" / f"Forecast_{name}.csv")
    return max(abs(a - b) for a, b in zip(saved, rerun, strict=True))


def main() -> None:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    prep = {c["cell"]: c["output"] for c in json.loads(PREP.read_text(encoding="utf-8"))}
    full = {c["cell"]: c["output"] for c in json.loads(FULL.read_text(encoding="utf-8"))}
    items = []
    for index in WHOLE_CELLS:
        original, rerun = normal(original_text(cells[index])), normal(prep[index])
        items.append({"cell": index, "what": "whole printed output", "same": original == rerun})
    for index, starts in LINE_CELLS.items():
        original_all, rerun_all = original_text(cells[index]), full[index]
        for start in starts:
            original, rerun = find_line(original_all, start), find_line(rerun_all, start)
            items.append(
                {
                    "cell": index,
                    "what": start,
                    "same": original == rerun,
                    "original": original,
                    "rerun": rerun,
                }
            )
    original_estimator = estimator_text(original_text(cells[72]))
    items.append(
        {
            "cell": 72,
            "what": "best XGBoost estimator (n_jobs left out)",
            "same": original_estimator == estimator_text(full[72]) and original_estimator != "",
        }
    )
    summary = {
        "items": len(items),
        "identical": sum(item["same"] for item in items),
        "different": [
            {key: item[key] for key in item if key != "same"} for item in items if not item["same"]
        ],
    }
    forecast_gap = {name: forecast_difference(name) for name in ("XGB", "RF")}
    record = {"summary": summary, "forecast_max_abs_difference": forecast_gap, "items": items}
    out = ROOT / "results" / "phase1_reproduction.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"{summary['identical']} of {summary['items']} identical; forecast gaps {forecast_gap}")
    for item in summary["different"]:
        print(json.dumps(item))


if __name__ == "__main__":
    main()
