# Limitations and future scope

This page lists what the work does not show, and what I would do next.

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
  models in, and the tables of pairwise tests are not corrected for the number of comparisons. I checked afterwards
  with Holm's correction (`scripts/final_model_checks.py`): the final model stays significantly better than each other
  forecast except its own variant. Against the same block yesterday it is better in every rolling fold, but the
  difference is significant after the correction in three of the five folds only.
* **Twenty-two new days.** The check on 1 to 22 Nov 2026 is short, and in that stretch few blocks settle at the
  cap, so it says little about months with many. The model calls the cap less well there (F1 0.765 against 0.923 on
  the test days).
* **The final model was designed after the test days were seen.** The plain models did not meet the success
  criterion I had set before tuning. I then built the cap-aware model. The 2025-26 test period is therefore not a
  clean one-shot for it. I answered this with a plan written down before scoring and a single run on 245 days of an
  earlier period (1 Mar to 31 Oct 2025, see `docs/holdout_protocol.md`), where it met all three conditions. That test
  has limits of its own: the model's settings were chosen on the 2025-26 days, which come after it, so it tests the
  design on other days and is not a forecast made ahead of time; only 8.8 percent of its blocks settle at the cap; the
  demand reports of that period have gaps (424 blocks in the scored months). I added a comparison with the tuned
  LightGBM and CatBoost models and the Phase 1 models after the result: CatBoost is not significantly worse than the
  final model there, so that test does not show that the two-stage structure is needed when the cap is rare.
* **Validation scores are optimistic** for every model whose settings were tuned on those days.
* **Intraday task.** The leads 1, 2, 4 and 8 are not tied to the exchange's gate-closure and result times, which I
  have not verified. The intraday models reuse the settings tuned for the day-ahead task. A cap-aware intraday model
  exists and has the lowest error at all four leads on the test days, but it has the same limits as the day-ahead
  one (it was added after the test days were seen, with no tuning of its own).
* **Intervals.** The 50 percent intervals cover more than they should, because the mass of prices at the cap makes
  them impossible to narrow with this correction. One of the rolling folds stays under-covered even after the
  correction. I checked the cap probability afterwards. It separates cap blocks from the others well and is close to
  calibrated overall, but it is over-confident where it gives a probability between 0.6 and 0.9. I then tried
  recalibrating it with an isotonic regression on the 60 days before each day, and weighting recent days more
  (half-lives of 30, 60 and 120 days). The variant that was best on the validation days (a half-life of 60 days with
  recalibration) was worse on the test days, so the final model keeps equal weights and the probability as it is.
* **A new day.** The saved model reproduces the backtest forecasts of the days I checked. I ran the daily loop over
  22 days after the window, but not yet as a routine, one day at a time, as the data arrive.

### Models

* The pretrained models read the price history only, with no day-ahead price and no demand. They are a reference,
  not a fair match for models that see those inputs.
* I did not study how sensitive the neural and pretrained models are to their settings.

### Use

* A forecast of a published price is not a trading decision. I did not model transaction costs, bidding rules or
  the effect of an order on the price.
* The app runs on one computer and reads local files. The command line takes six to eleven seconds to start on this
  computer, mostly the time to import LightGBM and the libraries it loads.
* The exchange's terms of use limit its content to personal, non-commercial use and do not allow copying or
  distributing it without permission (see the data card). This repository therefore holds no 2025-26 price files
  or prices per block. It does publish scores and summary statistics computed from them, and I have not asked the
  exchange whether that is allowed. I have not checked the terms of the other sources.

## Future scope

* Verify the gate-closure and result times, and fix the real intraday lead.
* Look at coverage fold by fold, and try a calibration that treats the regime of the last weeks separately.
* Add TabPFN and Chronos-2 to the GPU notebook.
* Repeat the comparison of the earlier period with the pretrained and neural models, and test the final model on a
  period with many cap blocks that it was not designed on, when one exists.
* Keep scoring the final model on days after 31 Oct 2026. The 22 days of 1 to 22 Nov 2026 are scored (the model
  is refitted each day), but they are few and only 7.3 percent of their blocks settle at the cap. The daily commands
  already support this: each forecast is stored and settled when the prices are published.
* Add state holidays, and handle the change in the demand definition explicitly.
* Watch the share of blocks at the cap over time, and run the daily commands on a schedule.
* Ask the exchange whether the published scores and summary statistics are allowed, and publish the app with data
  only once it allows it.
