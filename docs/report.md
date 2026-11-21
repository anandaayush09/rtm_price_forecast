# Forecasting real-time electricity prices on the Indian Energy Exchange

Report on Phase 2. This is the structure of the report. Sections marked **[to be filled]** wait for results that
come from exchange prices, which I publish only after I have checked the exchange's terms of use. Sections without
a mark are written.

## Summary

**[to be filled, written last]** Three short paragraphs: the question, the main result against the baselines and
against the Phase 1 models, and what the result does not show.

## 1. Question and setup

### 1.1 The problem

The real-time market (RTM) of the Indian Energy Exchange (IEX) sets a price for each 15-minute block of the day, 96
blocks per day. The day-ahead market (DAM) has already set its own prices for the same blocks when the RTM trades.
A forecast of the RTM price helps a buyer or seller decide how much to leave for the real-time market.

### 1.2 Data

The sources, the conventions, the cleaning rules and the quality counts are in the [data card](data_card.md). The
exchange files and the demand reports are not part of this repository.

### 1.3 Tasks

* **Day ahead (main task).** Forecast the 96 RTM blocks of day D once the DAM results of day D are known. Everything
  up to the end of day D - 1 is known, plus the DAM results of day D.
* **Intraday.** Forecast a block 1, 2, 4 or 8 blocks after the latest known RTM block. I have not tied these leads to
  the exchange's gate-closure and result times (see the limitations).

### 1.4 Evaluation protocol, fixed before any tuning

* **Splits.** The last days of the window are the test days. The 30 days before them are the validation days, and the
  days before those are the training days. Every choice of inputs and settings uses validation days only.
* **Rolling origin.** Each day is forecast by a model fitted on all earlier days with a published price (an expanding
  window). Final runs refit every day. While tuning, I refit every second validation day to save time.
* **Rolling folds as a check.** Five consecutive folds of 30 days, each forecast by models fitted on all days before it.
* **Measures.** Mean absolute error (MAE) and root mean squared error (RMSE) in INR/MWh; rMAE, the MAE divided by the
  MAE of the same block a day earlier; and for the price cap, precision, recall and F1 of calling a block at the cap.
  Blocks without a published price are left out.
* **Significance.** Diebold-Mariano tests on the daily mean absolute error (one-sided, with the Harvey-Leybourne-Newbold
  small-sample correction and a bootstrap interval for the mean daily difference), and the 90 percent model confidence
  set. I do not correct for the number of comparisons, and 60 test days give the tests little power.
* **Success criterion, fixed before tuning.** A model succeeds if it beats the best simple forecast and both Phase 1
  models with a Diebold-Mariano p-value below 0.05 on the test days and is in the 90 percent model confidence set.
* **Tuning.** The same number of Optuna trials for each tuned model family (LEAR, LightGBM and CatBoost).

### 1.5 How the parts fit together

The flowchart in the [README](../README.md#how-it-fits-together) shows the path from the source files to the daily
forecast.

## 2. Steps

Each step has four parts: why I did it, what I did, what I found, and what I understand from it.

### 2.1 Data pipeline and audit

* **Why.** A forecast is only as good as the table behind it, and Phase 1 had no contract that its data had to meet.
* **What I did.** Typed loaders and contracts for the exchange exports and the demand sheets, a full grid of 96 blocks per
  day, and a data-quality report for each run. The steps run through DVC and read their settings from `params.yaml`.
* **What I found.** **[to be filled]** Counts of missing and suspect values, the changes in the demand sheet layout,
  and what the audit led me to change.
* **What I understand.** **[to be filled]**

### 2.2 Exploration

* **Why.** To see which patterns a model can use, and which breaks it must live with.
* **What I did.** Daily and weekly profiles, the gap between the DAM and RTM prices, autocorrelation, a seasonal
  decomposition, volatility clusters, the relation to demand, and the price cap.
* **What I found.** **[to be filled]**
* **What I understand.** **[to be filled]**

### 2.3 Check of Phase 1

* **Why.** Before I compare with Phase 1, I need to know what it computed.
* **What I did.** A faithful port with a test for every number the 2021 notebook printed, a rerun in a pinned 2020
  environment, and each defect fixed on its own so that I could measure its effect. Naive forecasts on the same day.
* **What I found.** **[to be filled]** The table of defects is in Appendix A.
* **What I understand.** **[to be filled]**

### 2.4 Preprocessing and features

* **Why.** A model must see only what is known when the forecast is made.
* **What I did.** Gap filling compared on hidden gaps, a variance-stabilising transform compared with raw prices,
  calendar and holiday inputs, lags of price, demand, wind and solar, and tests that change future values and check
  that no input moves.
* **What I found.** **[to be filled]**
* **What I understand.** **[to be filled]**

### 2.5 Benchmark and tuning

* **Why.** To compare simple and complex models on the same footing.
* **What I did.** Naive forecasts, LEAR, MSTL with exponential smoothing, LightGBM, CatBoost, the two Phase 1 models,
  Chronos-Bolt and TimesFM (pretrained), and N-HiTS and TFT (trained per fold), on the same splits.
* **What I found.** **[to be filled]**
  * Table R1: day-ahead scores on the test days, one row per model (MAE, RMSE, rMAE, cap F1).
  * Table R2: the same on the validation days, marked as optimistic for tuned models.
  * Table R3: the five rolling folds.
  * Table R4: intraday scores for the leads 1, 2, 4 and 8.
* **What I understand.** **[to be filled]**

### 2.6 Significance, uncertainty and errors

* **Why.** A lower error on 60 days can be luck.
* **What I did.** Diebold-Mariano tests against the best simple forecast and both Phase 1 models, bootstrap intervals,
  the 90 percent model confidence set, and an error analysis by block, weekday and price level.
* **What I found.** **[to be filled]**
  * Table R5: pairwise tests on the day-ahead task.
  * Table R6: model confidence set membership, day ahead and by lead.
  * Whether the success criterion is met, stated with the limits in section 4.
* **What I understand.** **[to be filled]**

### 2.7 The final model

* **Why.** A large share of the error comes from blocks that settle at the exchange cap, and a forecast of the average
  cannot be right both on those blocks and on the others.
* **What I did.** A two-stage forecast: a classifier decides whether a block settles at the cap, and a quantile
  regression forecasts the price on the blocks below it. The point forecast is the median of the mixture of the two
  stages. Prediction intervals come from the same mixture and are corrected with a margin learned on the 60 days before,
  separately for four levels of how much the model expects the cap.
* **What I found.** **[to be filled]**
  * Table R7: the final model against the baselines on the test days.
  * Table R8: interval coverage, width and interval score, by cap state.
  * Note for the reader: I designed this model after the plain models had not met the criterion on the test days, so
    the test days are not a clean one-shot for it.
* **What I understand.** **[to be filled]**

### 2.8 Engineering

* **Why.** A forecast is useful only if it can be made again tomorrow.
* **What I did.** A package with tests and typed code, a command line to fit, calibrate, forecast and settle a day, an
  app that shows the forecast and the record of recent days, continuous integration, and a Docker image that never
  contains data.
* **What I found.** On a fresh clone the install from the lock file, the checks and the tests pass, the pipeline
  rebuilds its report identically, and the notebooks run. The saved model reproduces the backtest forecasts of the
  days I checked. I have not forecast a day after the last day in the data.
* **What I understand.** **[to be filled]**

## 3. Conclusion

**[to be filled]** The answer to the question, in a paragraph, followed by what I would do next.

## 4. Limitations

See [`LIMITATIONS_AND_FUTURE_SCOPE.md`](../LIMITATIONS_AND_FUTURE_SCOPE.md).

## Appendix A. Phase 1: inconsistencies, defects and fixes

One row per item: where it is, how I fixed it, and the effect of the fix. All scores are on the single test day of
Phase 1 (19 Jan 2021, 96 blocks, 92 training days) and in INR/MWh; MAPE is the mean absolute percentage error. One day
cannot rank models, so I read each effect as a size, not as a result. Same block yesterday scores MAE 431.57 and MAPE
10.98 on that day, better than every row below. The numbers come from notebook 03 and from the two files in
`results/` that it reads.

| Item | Where | Fix | Effect of the fix |
|---|---|---|---|
| Dates read with the month first: 44 of 111 dates were read wrongly, for example 01-10-2020 as 10 January | data loading | Read the dates day first | XGBoost MAPE 16.17 to 14.10 (MAE 757.87 to 650.14). Random forest MAPE, mean of five seeds, 15.37 to 14.70 (MAE 711.96 to 678.94) |
| Weekend flags: computed from the wrongly read dates, so 19 days had the wrong Saturday or Sunday flag. Of the six holiday dates, 2 Oct only exists in the data when the dates are read correctly | feature construction | Fixed by the date fix | Included in the date row, I did not separate it. Dropping the holiday flag as well gives XGBoost 13.88 and random forest 14.44, but one day cannot say whether the flag hurts |
| Random forest without a seed | model fitting | Fix the seed and average over several | Across five seeds the MAPE moves between 15.28 and 15.47, and my 2021 value (15.32) lies inside. In the rerun in the 2021 environment the grid picks `min_samples_split` 60 instead of 36, the full-day MAPE is 15.418 instead of 15.322, and a block of the forecast moves by up to 129.35 |
| A function used but never defined (`evalute_model`, called in cell 90) | evaluation | Write the function the cell evidently meant | No effect on the XGBoost numbers, which I reproduce exactly. The random forest values printed by that cell are reproduced only approximately |
| An error sum that leaves out one block | evaluation | Sum over all 96 blocks | The printed daily error is too low by 1.34 for XGBoost (756.53 against 757.87) and by 1.32 for the random forest (707.45 against 708.77) |
| Cross-validation by plain K folds on a time series | tuning | Forward-chaining folds | The default folds reproduce my 2021 winner (learning rate 0.07, score -288694.97, MAPE 16.171). Forward chaining picks learning rate 0.06 with a worse score (-310720.03) and a test-day MAPE of 15.617 (MAE 730.07). The worse score is expected, since early folds have little to train on |
| Accuracy figure in the README that does not match the output | README | State the figure that the saved forecasts give | The README says about 85 percent for both models. The saved forecasts give 84.68 for the random forest and 83.83 for XGBoost, as 100 minus MAPE |

Besides these, 27 of the 39 printed values of the original notebook are identical when I rerun it unchanged in the
2021 environment. The other 12 all belong to the random forest, which had no seed.

## Appendix B. How I checked the work

* Tests for the leakage of future values into inputs, for the borders between training, validation and test days, for
  each measure against values worked out by hand, and for repeatability with fixed seeds.
* A fresh clone: install from the lock file, lint, type check, the full test suite, the data pipeline with its report
  compared to the committed one, and the notebooks.
* A scan of every commit and the tree for private paths, non-ASCII text, trailers and secrets.
* **[to be filled]** Every figure looked at once, and every citation checked at its source.

## Appendix C. Data quality

See the [data card](data_card.md) and `reports/`.

## Appendix D. References

**[to be filled]** Only references that I have checked at the source. Until then, the methods named in the text are
Diebold and Mariano for the forecast comparison test, Harvey, Leybourne and Newbold for its small-sample correction,
and Hansen, Lunde and Nason for the model confidence set.
