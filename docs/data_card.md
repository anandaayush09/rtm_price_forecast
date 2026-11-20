# Data card

What data the project uses, where it comes from, and what I did to it. This is a draft.

## Summary

| | 2020-21 data (Phase 1) | 2025-26 window (Phase 2) |
|---|---|---|
| Period | 1 Oct 2020 to 19 Jan 2021 | 1 Nov 2025 to 31 Oct 2026 |
| Size | 111 days, 10656 blocks | 365 days, 35040 blocks |
| In this repository | `data/raw/price_data.csv`, unchanged | no, built locally from files I do not redistribute |

A block is 15 minutes, so a day has 96 blocks.

## Sources

* **IEX real-time market (RTM) and day-ahead market (DAM) snapshots.** Per block: price, purchase bid, selling bid,
  matched volume and scheduled volume. The files are range exports from the exchange's market-data pages.
* **Grid-India daily power supply position reports.** From each report, the sheet with 15-minute readings gives all-India demand
  met, frequency, generation by source (nuclear, wind, gas, thermal, hydro, solar, others), storage generation
  and storage demand (the last two only in the later layouts) and net exchange.
* **Government of India holiday lists.** The office memoranda of the Department of Personnel and Training for
  2020, 2025 and 2026 (see `config/holidays.yaml`, which names each document). They list central government
  holidays only.
* **The 2020-21 file.** `data/raw/price_data.csv` comes with the 2021 notebook. Its 2021 README names the IEX
  website for prices and bids and the Vidyut Pravah site for demand.

I have not checked the terms of use of any of these sources. The 2025-26 files are therefore not in the repository,
and no price figures from them are published here.

## Conventions

* Times are Indian Standard Time (UTC+05:30). Block 1 starts at 00:00 and block 96 at 23:45. I stamp each block
  with its start time, which I assume and have not confirmed against exchange documentation.
* Prices are in INR/MWh and the exchange price cap is 10000. Bid and volume columns are in MW, which I judged
  from their size. Demand is in MW.

## What I did to the data

* **Grid.** The 2025-26 tables are joined onto a full grid of 96 blocks per day. Nothing is interpolated at this
  step; a value that a source does not have stays missing.
* **Missing exchange blocks.** A block where every exchange value is zero is set to missing. This is an
  assumption: I take zero everywhere to mean that no trading data exist.
* **Demand and generation readings.** A reading with a frequency outside 48 to 52 Hz, or with total generation
  of zero or less, or with a negative nuclear, wind, hydro, gas or thermal value, is treated as lost telemetry,
  and all values of that reading are set to missing. Readings that look wrong but pass these rules are flagged
  (`demand_suspect`), not dropped or edited.
* **Demand definition.** The demand figure counts storage load from mid-2026 and did not before. For the models I
  use demand without storage load, so that it means the same thing all year.
* **Gaps for the models.** Where a model needs a filled price history, I fill short gaps in a straight line
  inside the day, using only earlier values. I chose this over carrying the last value forward by hiding known
  values and comparing the two.
* **Bounds.** Contracts reject prices outside 0 to 20000, bid volumes outside 0 to 200000 MW and demand outside
  50000 to 300000 MW. These bounds only catch broken rows; they are not market rules, and the bid bound is an
  assumption because I found no published limit.

## Quality of the 2025-26 window

These counts are read from `reports/window_quality.json`:

* RTM: 14 blocks missing, in two runs of consecutive blocks.
* DAM: no block missing.
* Demand: 104 blocks missing and 6 flagged as suspect.
* The demand sheet changed layout twice during the year: three layouts, with storage columns added in the
  later two.

For the 2020-21 file, `reports/data_quality.json` counts 12 missing RTM values, and its demand column has
long runs of identical values that I treat as unreliable.

## Columns of the 2025-26 table

| Group | Columns |
|---|---|
| Key | `timestamp`, `date`, `block` |
| RTM and DAM, each | `price`, `purchase_bid`, `selling_bid`, `mcv`, `scheduled_volume`, `empty`, `missing` (prefixed `rtm_` or `dam_`) |
| Demand | `demand`, `demand_suspect`, `demand_missing`, `net_demand_met`, `net_exchange` |
| Grid | `frequency`, `total_generation` |
| Generation | `nuclear`, `wind`, `gas`, `thermal`, `solar`, `hydro`, `others`, `storage_generation`, `storage_demand` |
| Source layout | `psp_layout` (1 to 3) |

## Intended use

The data serve one purpose here: comparing forecasts of the RTM price, and building a day-ahead forecast from the
best of them. Do not read the models' forecasts as advice on bidding. If you rebuild the 2025-26 table, get the
files from the sources yourself and check their terms of use first.
