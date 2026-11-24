# Test on a period the model was not designed on

This page was written and committed before any price of the test period was scored. It fixes what is tested and
what counts as success, so that the result cannot be adjusted afterwards.

## Why

I designed the final model after I had seen the 60 test days of the 2025-26 window, so those days are not a clean
test of it (see the report and the limits page). The period 4 Nov 2024 to 31 Oct 2025 played no part in the design:
I did not look at its prices, its price cap share or its demand when I chose the model, its inputs or its settings.

**A change made before anything was scored.** The first version of this page named calendar year 2024. The grid
operator's daily reports hold the 15-minute demand and generation sheet only from 4 Nov 2024, and the model needs
it, so the test period became 4 Nov 2024 to 31 Oct 2025. I made the change before I had scored a day or read a
price of 2025. Nothing else on this page changed except the dates that follow from it.

## What is frozen

* **The model and its settings.** The two-stage cap-aware forecast of version 0.1.0, with the settings in
  `params.yaml` as they are at the commit tagged `holdout-freeze`. The day-ahead task only.
* **No tuning on the test period.** No setting, input, window length, threshold or code path changes after I have
  seen a score from it. If something breaks (a file layout the loader cannot read, for example), I fix the loader
  only, and I list the fix in the result.
* **The comparison forecasts.** The same block yesterday, the same block last week and the DAM price, scored on the
  same days.

## What is scored

* **Data.** Exchange prices and bids for 4 Nov 2024 to 31 Oct 2025, and the all-India demand and generation reports
  of the same days. Two of the demand reports (30 and 31 Dec 2024, in the fitting period) are listed on the grid
  operator's site but cannot be downloaded, so those two days have no demand figures. I report how many blocks that
  affects.
* **Method.** The daily loop of the command line: for each day, the model is fitted on the days before it, the day
  is forecast, and the day is added to the calibration table once its prices are known.
* **Warm-up and scored days.** The first four months (4 Nov 2024 to 28 Feb 2025) are used for fitting only. The
  scored days are 1 Mar to 31 Oct 2025. The calibration table starts empty on 1 Mar, so the intervals are scored
  from the 15th day on (15 Mar).
* **No selection of days.** Every scored day counts, including days with many blocks at the cap and days with few.

## What counts as success

All three must hold on the scored days:

1. **Point forecast.** The mean absolute error of the final model is lower than that of the same block yesterday, and
   the Diebold-Mariano test (with the Harvey-Leybourne-Newbold correction, as in the report) gives p below 0.05.
2. **Against the DAM price.** The final model's mean absolute error is also lower than that of the DAM price.
3. **Not only one season.** The final model has a lower error than the same block yesterday in at least two thirds of
   the calendar months of the scored period.

I also report, without a pass mark: the share of published prices inside the 50 and 90 percent intervals, the F1
of calling the cap, and the share of blocks at the cap in each month.

## What I will report

The scores as they come out, whether or not they meet the criterion, together with the share of blocks at the
cap in the scored period, so that the reader can see how different it is from 2025-26. If the model does not meet
the criterion, the report and the README say so and the earlier results are described as specific to the 2025-26
year.

## Result

I ran the plan once, on 245 scored days (1 Mar to 31 Oct 2025). All three conditions are met: the mean absolute
error of the final model is 817.5, against 1011.1 for the same block yesterday (Diebold-Mariano p below 0.0001) and
981.1 for the DAM price, and it is lower than yesterday's in all 8 months. The full tables are in section 2.9 of the
[report](report.md). Only the loaders changed between the freeze and the run (monthly exports that label the shared
day differently, older demand report headers, one report with two repeated readings, empty reading sheets), each with
a test.

After seeing this result I added a comparison, which was not part of the plan: the tuned LightGBM and CatBoost models and
the two Phase 1 models on the same days, with the settings chosen on the 2025-26 window. The final model has the lowest
error (817.5 against 825.0 for CatBoost, 840.0 for LightGBM, 865.8 and 877.9 for the Phase 1 models), but its advantage
over CatBoost is not significant (p = 0.189) and both are in the 90 percent model confidence set.
