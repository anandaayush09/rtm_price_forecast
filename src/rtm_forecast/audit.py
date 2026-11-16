"""Data-quality report for the clean table."""

from typing import Any, cast

import pandas as pd

from rtm_forecast.ingest import BLOCKS_PER_DAY, RAW_TO_CLEAN, RTM_COLUMNS


def missing_runs(clean: pd.DataFrame) -> list[dict[str, Any]]:
    """Runs of consecutive blocks with unpublished RTM values.

    ``aligned_pair`` is true when the run starts on an odd block and has even length, which is what
    a missing half-hour auction session would look like.
    """
    flag = clean["rtm_missing"].to_numpy()
    runs: list[dict[str, Any]] = []
    start = None
    for position, is_missing in enumerate([*flag, False]):
        if is_missing and start is None:
            start = position
        elif not is_missing and start is not None:
            length = position - start
            first = clean.iloc[start]
            runs.append(
                {
                    "start": first["timestamp"].isoformat(),
                    "block": int(first["block"]),
                    "length": length,
                    "aligned_pair": bool(first["block"] % 2 == 1 and length % 2 == 0),
                }
            )
            start = None
    return runs


def repeated_runs(clean: pd.DataFrame, column: str, min_length: int) -> list[dict[str, Any]]:
    """Runs where a column keeps exactly the same value for at least ``min_length`` blocks."""
    run_id = (clean[column].diff() != 0).cumsum()
    runs: list[dict[str, Any]] = []
    for _, group in clean.groupby(run_id):
        if len(group) >= min_length:
            runs.append(
                {
                    "start": group["timestamp"].iloc[0].isoformat(),
                    "length": len(group),
                    "value": float(group[column].iloc[0]),
                }
            )
    return runs


def identical_days(clean: pd.DataFrame, column: str) -> list[list[str]]:
    """Pairs of different days whose 96 values of a column are exactly the same."""
    profiles = clean.pivot(index="date", columns="block", values=column)
    seen: dict[tuple[float, ...], str] = {}
    pairs: list[list[str]] = []
    for day, row in profiles.iterrows():
        key = tuple(row.to_numpy())
        label = pd.Timestamp(str(day)).date().isoformat()
        if key in seen:
            pairs.append([seen[key], label])
        else:
            seen[key] = label
    return pairs


def build_report(clean: pd.DataFrame, stale_min_blocks: int = 4) -> dict[str, Any]:
    columns = list(RAW_TO_CLEAN.values())
    stats = clean[columns].describe(percentiles=[0.01, 0.5, 0.99]).T
    runs = missing_runs(clean)
    stale = repeated_runs(clean, "demand", stale_min_blocks)
    return {
        "rows": int(len(clean)),
        "days": int(clean["date"].nunique()),
        "first_timestamp": clean["timestamp"].iloc[0].isoformat(),
        "last_timestamp": clean["timestamp"].iloc[-1].isoformat(),
        "blocks_per_day_ok": bool((clean.groupby("date").size() == BLOCKS_PER_DAY).all()),
        "duplicate_timestamps": int(clean["timestamp"].duplicated().sum()),
        "missing_cells": {name: int(clean[name].isna().sum()) for name in columns},
        "rtm_missing_rows": int(clean["rtm_missing"].sum()),
        "rtm_missing_runs": runs,
        "rtm_columns_missing_together": bool(
            clean[list(RTM_COLUMNS)].isna().nunique(axis=1).eq(1).all()
        ),
        "demand_stale_min_blocks": stale_min_blocks,
        "demand_stale_runs": stale,
        "demand_stale_rows": sum(run["length"] for run in stale),
        "demand_identical_days": identical_days(clean, "demand"),
        "rtm_price_skew": round(cast(float, clean["rtm_price"].skew()), 3),
        "rtm_price_equal_to_previous": int((clean["rtm_price"].diff() == 0).sum()),
        "summary": {
            str(name): {str(key): round(value, 2) for key, value in row.items()}
            for name, row in stats.astype(float).to_dict(orient="index").items()
        },
    }
