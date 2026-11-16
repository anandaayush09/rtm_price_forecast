"""Execute the notebooks in order. Saves outputs in place unless --check is given."""

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="run without saving the outputs")
    args = parser.parse_args()
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        started = time.time()
        NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(NOTEBOOKS)}},
        ).execute()
        print(f"{path.name}: ok in {time.time() - started:.0f}s")
        if not args.check:
            nbformat.write(notebook, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
