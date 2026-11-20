# GPU notebook: pretrained and neural time-series models

`foundation_models.ipynb` forecasts the same 150 days, folds and scores as the model comparison on the
2025-26 window, one cell per model:

* Chronos-Bolt and TimesFM, pretrained and used without any training on this data. Each forecast reads
  only the RTM price history up to the end of the previous day. They use no day-ahead price and no
  demand input, so they are a univariate reference.
* N-HiTS and TFT, trained in the notebook once per fold on all earlier days. Besides the price history
  they read the DAM price, the block and the weekday of the day being forecast, which are known in advance.

Every cell saves a checkpoint after each fold and its full predictions as soon as it finishes. Running
a cell again skips the folds already saved.

The price file is not part of this repository. The market data is private, so the dataset must stay
private too.

## Steps

1. Write the price file on the machine that holds the data:

   ```
   python scripts/export_kaggle_inputs.py
   ```

   This creates `.project-local/kaggle_export/rtm_dam_prices.parquet`. It holds RTM and DAM prices and the
   fold labels. It holds no demand or generation value.

2. On Kaggle, create a new dataset named `rtm-dam-prices`, upload that one file and set the visibility to
   private.
3. Create a notebook from `foundation_models.ipynb`, attach the dataset, turn on a GPU and turn on internet
   access (the model weights are downloaded when the notebook runs).
4. Run the cells in order. Each model cell can be run on its own. The last cells print the scores for the test
   period and for each fold, and bundle the files into `predictions_bundle.zip`.
5. Download the `predictions_<model>.parquet` files (or the bundle) into `.project-local/results_window/`,
   leaving out `predictions_naive_yesterday_check.parquet`, and run:

   ```
   python scripts/benchmark_window.py report
   ```

   The report then lists the four models next to the others.

## Intraday task

After the day-ahead cells the notebook runs the same four models on a second task: forecast a block 1, 2,
4 or 8 blocks after the latest known RTM block (the "Intraday task" section). It follows the CPU script
`scripts/benchmark_intraday.py`, which scores the naive forecasts, LEAR, LightGBM and CatBoost on the same task.

* Chronos-Bolt and TimesFM read the filled RTM price up to the latest known block and forecast the next 8
  blocks, once per block, so one pass serves all four leads.
* N-HiTS and TFT are trained once per fold on the rows up to 8 blocks before the fold, with a horizon of
  8 blocks, and then used for every block of the fold without retraining. The last fold has no window for its
  final few blocks at leads below 8, so the scores use only the blocks that every model has.
* Each model cell writes `intraday_<model>.parquet` (one file, a `lead` column) and a checkpoint after every
  fold. `RUN_INTRADAY` in the setup cell chooses which models run. Running a cell again skips finished folds.
* The `intraday_naive_last_check` cell must give the same scores as the `naive_last` row of the CPU run.

To compare, download the `intraday_<model>.parquet` files into `.project-local/results_intraday/` and run:

```
python scripts/benchmark_intraday.py report
```

Status: all four models have been run on Kaggle, for both tasks. The last-known-price check gives exactly the
CPU script's `naive_last` forecasts, and the predictions of every model contain no missing values. TFT needs a
smaller batch (`windows_batch_size=64`) to fit in GPU memory, and the last cell writes only the prediction files
into the bundle, because the model weights and logs fill the disk of the working folder.

## Check before trusting the numbers

The notebook includes a yesterday forecast built the same way as the comparison's own. Its scores must
match the `naive_yesterday` row of the report. If they do not, the data or the folds have drifted.
