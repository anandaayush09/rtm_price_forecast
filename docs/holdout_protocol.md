# Test on a year the model was not designed on

This page was written and committed before any 2024 price was scored. It fixes what is tested and what counts as
success, so that the result cannot be adjusted afterwards.

## Why

I designed the final model after I had seen the 60 test days of the 2025-26 window, so those days are not a clean
test of it (see the report and the limits page). Calendar year 2024 is a year that played no part in the design: I
did not look at its prices, its price cap share or its demand when I chose the model, its inputs or its settings.

## What is frozen

* **The model and its settings.** The two-stage cap-aware forecast of version 0.1.0, with the settings in
  `params.yaml` as they are at the commit tagged `holdout-freeze`. The day-ahead task only.
* **No tuning on 2024.** No setting, input, window length, threshold or code path changes after I have seen a 2024
  score. If something breaks (a file layout the loader cannot read, for example), I fix the loader only, and I list
  the fix in the result.
* **The comparison forecasts.** The same block yesterday, the same block last week and the DAM price, scored on the
  same days.

## What is scored

* **Data.** Exchange prices and bids for 1 Jan 2024 to 31 Dec 2024, and the all-India demand and generation reports
  of the same days. Two of the demand reports (30 and 31 Dec 2024) are listed on the grid operator's site but
  cannot be downloaded, so those two days have no demand figures. I report how many blocks that affects.
* **Method.** The daily loop of the command line: for each day, the model is fitted on the days before it, the day
  is forecast, and the day is added to the calibration table once its prices are known.
* **Warm-up and scored days.** The first four months (1 Jan to 30 Apr 2024) are used for fitting only. The scored
  days are 1 May to 31 Dec 2024. The calibration table starts empty on 1 May, so the intervals are scored from the
  15th day on (15 May).
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
cap in 2024, so that the reader can see how different that year is from 2025-26. If the model does not meet the
criterion, the report and the README say so and the earlier results are described as specific to the 2025-26 year.
