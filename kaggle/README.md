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

## Check before trusting the numbers

The notebook includes a yesterday forecast built the same way as the comparison's own. Its scores must
match the `naive_yesterday` row of the report. If they do not, the data or the folds have drifted.
