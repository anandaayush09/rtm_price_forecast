import pandas as pd

from rtm_forecast.audit import build_report, identical_days, missing_runs, repeated_runs


def test_missing_runs_on_real_data(clean_table: pd.DataFrame) -> None:
    runs = missing_runs(clean_table)
    assert len(runs) == 6
    assert all(run["length"] == 2 and run["aligned_pair"] for run in runs)
    assert [run["block"] for run in runs] == [7, 71, 85, 15, 9, 5]


def test_missing_run_at_end_of_table() -> None:
    table = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=4, freq="15min", tz="Asia/Kolkata"),
            "block": [1, 2, 3, 4],
            "rtm_missing": [False, False, True, True],
        }
    )
    runs = missing_runs(table)
    assert runs == [
        {"start": "2021-01-01T00:30:00+05:30", "block": 3, "length": 2, "aligned_pair": True}
    ]


def test_report_counts(clean_table: pd.DataFrame) -> None:
    report = build_report(clean_table)
    assert report["rows"] == 10656
    assert report["blocks_per_day_ok"] is True
    assert report["missing_cells"]["rtm_price"] == 12
    assert report["rtm_columns_missing_together"] is True


def test_repeated_runs_on_real_demand(clean_table: pd.DataFrame) -> None:
    runs = repeated_runs(clean_table, "demand", 4)
    assert len(runs) == 16
    assert max(run["length"] for run in runs) == 87
    assert sum(run["length"] for run in runs) == 359


def test_repeated_runs_threshold_is_inclusive() -> None:
    table = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=6, freq="15min", tz="Asia/Kolkata"),
            "demand": [1.0, 5.0, 5.0, 5.0, 7.0, 8.0],
        }
    )
    assert [run["length"] for run in repeated_runs(table, "demand", 3)] == [3]
    assert repeated_runs(table, "demand", 4) == []


def test_identical_days_on_real_demand(clean_table: pd.DataFrame) -> None:
    assert identical_days(clean_table, "demand") == [
        ["2020-12-10", "2020-12-17"],
        ["2020-12-13", "2020-12-20"],
        ["2020-12-19", "2020-12-26"],
    ]
