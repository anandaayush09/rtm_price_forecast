# Limitations and future scope

This page lists what the work does not show, and what I would do next. It is a draft and will be revised
together with the report.

## Limitations

### Data

* **One year, one exchange.** The main comparison uses 365 days of one market (1 Nov 2025 to 31 Oct 2026). The
  share of blocks at the price cap changes during that year, so a result for this year may not hold for another.
* **Missing blocks.** A block where every exchange value is zero is treated as missing. This is an assumption: I
  have not confirmed that zero everywhere means that no trading data exist.
* **Demand changes its definition.** From May and June 2026 the demand figure includes storage load, before that it
  did not. I take the storage load out so that the definition stays the same, but I did not measure how much the
  break matters before doing so. Demand and wind and solar enter the models only as values of earlier days.
* **Demand day.** I assume that a daily demand report describes the day before its report date. I checked this
  only for internal consistency, not against a second source.
* **Bid volumes and block times.** The bound on bid volumes is an assumption, because I found no published limit,
  and the convention that a block is stamped with its start time is not confirmed against exchange documentation.
* **Holidays.** Only central government holidays are used. State holidays and local festivals are not included.

### Evaluation

* **Sixty test days.** Significance tests on 60 days have little power. The model confidence set keeps several
  models in, and the pairwise tests are not corrected for the number of comparisons.
* **The final model was designed after the test days were seen.** The plain models did not meet the success
  criterion I had set before tuning. I then built the cap-aware model. The test period is therefore not a clean
  one-shot for it. Its choice between two variants was made on the validation days only, but the results should be
  read as evidence, not as confirmation.
* **Validation scores are optimistic** for every model whose settings were tuned on those days.
* **Intraday task.** The leads 1, 2, 4 and 8 are not tied to the exchange's gate-closure and result times, which I
  have not verified. The intraday models reuse the settings tuned for the day-ahead task, and there is no cap-aware
  intraday model.
* **Intervals.** The 50 percent intervals cover more than they should, because the mass of prices at the cap makes
  them impossible to narrow with this correction. One of the rolling folds stays under-covered even after the
  correction. I did not check how well the cap probability itself is calibrated, and I did not weight recent days
  more heavily.
* **A new day.** The saved model reproduces the backtest forecasts of the days I checked, but I have not forecast a
  day that came after the last day in the data.

### Models

* The pretrained models read the price history only, with no day-ahead price and no demand. They are a reference,
  not a fair match for models that see those inputs.
* I did not study how sensitive the neural and pretrained models are to their settings.

### Use

* A forecast of a published price is not a trading decision. I did not model transaction costs, bidding rules or
  the effect of an order on the price.
* The app runs on one computer and reads local files. The Docker image is large, because the command line imports
  libraries that the production path does not need.
* The exchange's terms of use limit its content to personal, non-commercial use and do not allow copying or
  distributing it without permission (see the data card). This repository therefore publishes no 2025-26 price
  figures and no result tables derived from them. I have not checked the terms of the other sources.

## Future scope

* Verify the gate-closure and result times, fix the real intraday lead, and add a cap-aware intraday model.
* Check the calibration of the cap probability, try weighting recent days more, and look at coverage fold by fold.
* Add TabPFN and Chronos-2 to the GPU notebook.
* Score the frozen final model on days after 31 Oct 2026 as a true hold-out. The daily commands already support
  this: each forecast is stored and settled when the prices are published.
* Add state holidays, and handle the change in the demand definition explicitly.
* Watch the share of blocks at the cap over time, and run the daily commands on a schedule.
* Slim the image by loading the heavy libraries only where they are used.
* Publish the app and the result tables once the exchange allows it.
