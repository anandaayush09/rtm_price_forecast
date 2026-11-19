# Rerunning the first notebook in a 2021-era environment

`run_original_cells.py` runs the code cells of `original/LD_Price_Forecast_Bagging_Boosting_Ensembles.ipynb`
unchanged, in Python 3.8 with the library versions that were current in February 2021. It makes only these edits,
listed in `PATCHES` in the script: the data path, two plot-only imports, `n_jobs` in the two grid searches (4
workers instead of 48 cores), and a stand-in for `evalute_model`, which cell 90 calls but the notebook never defines.

`requirements.txt` was resolved with `--exclude-newer 2021-02-10` from `requirements.in`.

```bash
uv venv --python 3.8 repro2020/.venv
uv pip install --python repro2020/.venv/Scripts/python.exe -r repro2020/requirements.txt
repro2020/.venv/Scripts/python.exe repro2020/run_original_cells.py
```

The XGBoost grid search took about 13 minutes. The RandomForest grid search took 3.2 hours of wall time on a
busy 8-core laptop that was asleep for part of it, so the compute time is lower. Passing a cell number, for example `run_original_cells.py 67`, stops after that cell. The data preparation cells
then run in seconds. `REPRO_OUT` changes the output folder.

`out/` holds the full run and `out_dataprep/` the data preparation cells. `scripts/compare_reproduction.py` compares them
with the outputs stored in the original notebook and writes `results/phase1_reproduction.json`.
