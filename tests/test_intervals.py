import numpy as np
import pandas as pd
import pytest

from rtm_forecast.intervals import conformal_adjust, interval_scores
from rtm_forecast.models import mixture_quantile

CAP = 10000.0


def _frame(days: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    """Prices around 3000 with noise 300; the interval is 3000 +- 300 (too narrow for 90%)."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2026-01-01", periods=days), 4)
    price = 3000 + shift + rng.normal(scale=300, size=len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "block": np.tile(np.arange(1, 5), days),
            "actual": price,
            "lo": 2700.0,
            "hi": 3300.0,
        }
    )


def test_adjustment_restores_coverage_that_the_raw_interval_lacks() -> None:
    frame = _frame(400, 0)
    raw = interval_scores(frame, 0.9)
    adjusted = interval_scores(conformal_adjust(frame, 0.9), 0.9, "lo_adj", "hi_adj")
    assert raw["coverage"] < 0.75  # +-1 sigma holds about 68 percent
    assert adjusted["coverage"] == pytest.approx(0.9, abs=0.03)
    assert adjusted["width"] > raw["width"]


def test_margin_of_a_day_uses_only_earlier_days() -> None:
    frame = _frame(60, 1)
    changed = frame.copy()
    day = pd.Timestamp("2026-02-10")
    changed.loc[changed["date"] >= day, "actual"] = 99999.0  # this day and later
    first = conformal_adjust(frame, 0.9).set_index(["date", "block"])
    second = conformal_adjust(changed, 0.9).set_index(["date", "block"])
    before = first.index.get_level_values("date") <= day
    pd.testing.assert_series_equal(first.loc[before, "lo_adj"], second.loc[before, "lo_adj"])
    after = ~before
    assert (first.loc[after, "lo_adj"] != second.loc[after, "lo_adj"]).any()


def test_margin_is_the_conformal_rank_of_the_scores_and_early_days_stay_empty() -> None:
    # 20 earlier days, 1 block per day, score = actual - hi = 1..20, coverage 0.9:
    # rank ceil(21 * 0.9) = 19, so the margin is 19
    days = pd.date_range("2026-01-01", periods=21)
    frame = pd.DataFrame(
        {
            "date": days,
            "block": 1,
            "actual": [101.0 + i for i in range(20)] + [100.0],
            "lo": 0.0,
            "hi": 100.0,
        }
    )
    result = conformal_adjust(frame, 0.9, window_days=60, min_days=14).set_index("date")
    assert result["hi_adj"].iloc[:14].isna().all()
    assert result["hi_adj"].iloc[20] == 100.0 + 19.0
    assert result["lo_adj"].iloc[20] == 0.0 - 19.0


def test_adjusted_ends_are_cut_to_the_price_range() -> None:
    frame = _frame(100, 2, shift=9000.0).assign(lo=8900.0, hi=9200.0)
    result = conformal_adjust(frame, 0.9, cap=CAP).dropna()
    assert result["hi_adj"].max() <= CAP
    assert result["lo_adj"].min() >= 0.0


def test_interval_score_counts_width_and_misses() -> None:
    frame = pd.DataFrame(
        {"actual": [5.0, 15.0], "lo": [0.0, 0.0], "hi": [10.0, 10.0], "date": ["a", "b"]}
    )
    scores = interval_scores(frame, 0.9)
    assert scores["coverage"] == 0.5
    assert scores["width"] == 10.0
    # miss of 5 above: 10 + 2 / 0.1 * 5 = 110 for the second block, 10 for the first
    assert scores["interval_score"] == pytest.approx((10 + 110) / 2)


def test_mixture_quantile_is_the_cap_when_the_cap_is_likely_enough() -> None:
    grid = np.array([0.1, 0.5, 0.9])
    quantiles = np.tile([1000.0, 2000.0, 3000.0], (3, 1))
    p = np.array([0.0, 0.3, 0.9])
    # tau 0.5: no cap mass -> below-cap median; p 0.3 -> below-cap quantile at 0.5/0.7 (between 0.5
    # and 0.9); p 0.9 -> cap
    result = mixture_quantile(p, quantiles, grid, 0.5, CAP)
    assert result[0] == 2000.0
    assert 2000.0 < result[1] < 3000.0
    assert result[2] == CAP
    # a low quantile is below the cap even when the cap is likely: tau 0.05 < 1 - 0.9 is false
    low = mixture_quantile(np.array([0.5, 0.97]), quantiles[:2], grid, 0.05, CAP)
    assert low[0] < 2000.0
    assert low[1] == CAP
    # quantiles rise with tau
    taus = [mixture_quantile(p, quantiles, grid, t, CAP) for t in (0.05, 0.25, 0.5, 0.75, 0.95)]
    assert all((later >= earlier).all() for earlier, later in zip(taus, taus[1:], strict=False))


def _two_groups(days: int, seed: int) -> pd.DataFrame:
    """Blocks 1-2 are quiet (noise 100), blocks 3-4 are noisy (noise 1000); same interval."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2026-01-01", periods=days), 4)
    block = np.tile(np.arange(1, 5), days)
    noise = np.where(block <= 2, 100.0, 1000.0) * rng.normal(size=len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "block": block,
            "actual": 3000 + noise,
            "lo": 2500.0,
            "hi": 3500.0,
            "state": np.where(block <= 2, "quiet", "noisy"),
        }
    )


def test_group_margins_give_coverage_inside_each_group() -> None:
    frame = _two_groups(500, 0)

    def coverage_by_state(result: pd.DataFrame) -> dict[str, float]:
        out = result.dropna(subset=["lo_adj"])
        return {
            str(state): interval_scores(part, 0.9, "lo_adj", "hi_adj")["coverage"]
            for state, part in out.groupby("state")
        }

    pooled = coverage_by_state(conformal_adjust(frame, 0.9))
    grouped = coverage_by_state(conformal_adjust(frame, 0.9, group="state", min_blocks=50))
    assert pooled["quiet"] > 0.97 and pooled["noisy"] < 0.85  # pooled: right on average only
    assert grouped["quiet"] == pytest.approx(0.9, abs=0.04)
    assert grouped["noisy"] == pytest.approx(0.9, abs=0.04)


def test_small_groups_fall_back_to_the_margin_of_all_blocks() -> None:
    frame = _two_groups(100, 1)
    pooled = conformal_adjust(frame, 0.9)
    grouped = conformal_adjust(frame, 0.9, group="state", min_blocks=10**6)
    pd.testing.assert_series_equal(pooled["hi_adj"], grouped["hi_adj"])


def test_only_the_marked_earlier_blocks_teach_the_margin() -> None:
    frame = _two_groups(100, 2).assign(teach=lambda d: d["block"] <= 2)
    wild = frame.assign(actual=np.where(frame["teach"], frame["actual"], 1e6))
    first = conformal_adjust(frame, 0.9, only="teach")
    second = conformal_adjust(wild, 0.9, only="teach")
    pd.testing.assert_series_equal(first["hi_adj"], second["hi_adj"])
