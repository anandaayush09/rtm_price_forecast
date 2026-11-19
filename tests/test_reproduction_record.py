"""The stored rerun of the first notebook: what matched and what did not."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _load_compare_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "compare_reproduction", ROOT / "scripts" / "compare_reproduction.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stored_comparison_matches_a_fresh_one(tmp_path: Path) -> None:
    module = _load_compare_module()
    module.main()
    record = json.loads((ROOT / "results" / "phase1_reproduction.json").read_text(encoding="utf-8"))
    assert record["summary"]["items"] == 39
    assert record["summary"]["identical"] == 27


def test_only_the_random_forest_cells_differ() -> None:
    record = json.loads((ROOT / "results" / "phase1_reproduction.json").read_text(encoding="utf-8"))
    differing_cells = {item["cell"] for item in record["summary"]["different"]}
    assert differing_cells == {86, 88, 90}
    same_cells = {item["cell"] for item in record["items"] if item["same"]}
    assert {72, 78} <= same_cells
