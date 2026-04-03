"""
Unit tests for breakout.range_quality — range breakout trader primitives.

TDD Red -> Green -> Refactor cycle:
  - All tests written BEFORE implementation exists.
  - Smoke tests use the SCM.MI 2016-08-25 to 2016-10-17 consolidation window
    (38 bars, rbo_20=0 throughout). Values verified against real parquet data:
      OLS slope = 0.0087%/day, resistance touches = 2, support touches = 2.

Each class tests one public function. Test names encode the invariant being checked.
"""

import numpy as np
import pandas as pd
import pytest

import logging

from ta.breakout.range_quality import (
    RangeQualityConfig,
    VolatilityState,
    breakout_prior_consolidation_length,
    classify_trend,
    count_touches,
    measure_volatility_compression,
    assess_range,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# SCM.MI 2016-08-25 to 2016-10-17: 38 bars, rbo_20=0 throughout.
# range_pct and rclose extracted from analysis_results.parquet.
SCM_RANGE_PCT = pd.Series([
    54.92, 72.03, 81.49, 69.71, 82.40, 83.61, 68.13, 67.78, 75.84,
    61.81, 57.05, 69.77, 88.63, 97.87, 98.37, 69.50, 94.51, 81.52,
    85.13, 64.06, 22.95, 49.19, 90.65, 99.48, 93.32, 75.46, 66.09,
    85.35, 80.17, 54.72, 37.60, 54.68, 10.46, 41.10, 42.17, 82.33,
    18.49, 10.91,
])

SCM_RCLOSE = pd.Series([
    10.2039, 10.4176, 10.5358, 10.3886, 10.3567, 10.3690, 10.2115,
    10.2079, 10.2899, 10.1471, 10.0987, 10.2282, 10.4201, 10.5141,
    10.5192, 10.2254, 10.4799, 10.3477, 10.4702, 10.3787, 10.1990,
    10.3137, 10.4799, 10.5170, 10.4911, 10.4160, 10.3766, 10.4576,
    10.4358, 10.3288, 10.3194, 10.3741, 10.2325, 10.3306, 10.3331,
    10.4608, 10.2578, 10.2337,
])


# ===========================================================================
# count_touches
# ===========================================================================


class TestCountTouches:

    def test_midrange_flat_no_touches(self):
        """Price oscillating at midrange never reaches resistance or support."""
        s = pd.Series([50.0] * 10)
        assert count_touches(s) == (0, 0)

    def test_single_resistance_spike(self):
        s = pd.Series([50, 50, 90, 90, 50, 50])
        n_res, n_sup = count_touches(s)
        assert n_res == 1
        assert n_sup == 0

    def test_two_resistance_touches_with_retreat(self):
        """Genuine retreat below retreat=65 separates two distinct touch events."""
        # spike 1: indices 1-2; retreat at index 3 (50 < 65 -> exit); spike 2: index 4
        s = pd.Series([50, 90, 90, 50, 90, 50])
        assert count_touches(s) == (2, 0)

    def test_gray_zone_does_not_split_touch(self):
        """
        Price at 70% (in gray zone [65, 85)) between two spikes must NOT create
        two touch events. The state machine stays in the current touch until price
        drops below retreat=65.
        """
        s = pd.Series([50, 90, 70, 88, 50])
        assert count_touches(s) == (1, 0)

    def test_below_retreat_splits_touch(self):
        """Price at 60% (below retreat=65) between two spikes creates two events."""
        s = pd.Series([50, 90, 60, 90, 50])
        assert count_touches(s) == (2, 0)

    def test_two_support_touches(self):
        s = pd.Series([50, 10, 50, 10, 50])
        assert count_touches(s) == (0, 2)

    def test_support_gray_zone_does_not_split(self):
        """
        Price at 30% (gray zone [15, 35]) does not exit the support touch.
        bounce=35, so we need val > 35 to exit. 30 is not > 35.
        """
        s = pd.Series([50, 10, 30, 12, 50])
        assert count_touches(s) == (0, 1)

    def test_mixed_resistance_and_support(self):
        """Single series with both resistance and support touches."""
        # res touch #1: index 0 (90); retreat at index 1 (50 < 65)
        # sup touch:    index 2 (10); bounce at index 3 (50 > 35)
        # res touch #2: index 4 (90)
        s = pd.Series([90, 50, 10, 50, 90])
        assert count_touches(s) == (2, 1)

    def test_nan_values_skipped(self):
        """NaN values are ignored; state machine continues across them."""
        s = pd.Series([90, float("nan"), 50, float("nan"), 90, 50])
        # spike at index 0; nan skipped; 50 < retreat -> exit; nan skipped; spike again
        assert count_touches(s) == (2, 0)

    def test_series_never_touches_either_level(self):
        s = pd.Series([40, 45, 55, 60, 55, 45])
        assert count_touches(s) == (0, 0)

    def test_invalid_retreat_gte_hi_thresh_raises(self):
        with pytest.raises(ValueError, match="retreat"):
            count_touches(pd.Series([50.0] * 5), hi_thresh=85, retreat=85)

    def test_invalid_bounce_lte_lo_thresh_raises(self):
        with pytest.raises(ValueError, match="bounce"):
            count_touches(pd.Series([50.0] * 5), lo_thresh=15, bounce=15)

    def test_smoke_scm_mi(self):
        """
        SCM.MI 2016-08-25 to 2016-10-17: 38-bar consolidation.
        Expected: 2 resistance touches, 2 support touches.
        Verified manually from parquet data.
        """
        n_res, n_sup = count_touches(SCM_RANGE_PCT)
        assert n_res == 2, f"expected 2 resistance touches, got {n_res}"
        assert n_sup == 2, f"expected 2 support touches, got {n_sup}"


# ===========================================================================
# classify_trend
# ===========================================================================


class TestClassifyTrend:

    def test_perfectly_flat_is_sideways(self):
        s = pd.Series([1.0] * 20)
        is_sw, slope = classify_trend(s)
        assert is_sw is True
        assert slope == pytest.approx(0.0, abs=1e-6)

    def test_strong_uptrend_is_not_sideways(self):
        # +0.5%/bar from 1.0 -> slope >> 0.15 threshold
        s = pd.Series([1.0 + i * 0.005 for i in range(20)])
        is_sw, slope = classify_trend(s)
        assert is_sw is False
        assert slope > 0.15

    def test_strong_downtrend_is_not_sideways(self):
        s = pd.Series([1.0 - i * 0.005 for i in range(20)])
        is_sw, slope = classify_trend(s)
        assert is_sw is False
        assert slope < -0.15

    def test_noisy_flat_is_sideways(self):
        """Small random noise around a flat mean stays below the 0.15%/day threshold."""
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.001, 30)  # ~0.1% std noise
        s = pd.Series(1.0 + noise)
        is_sw, slope = classify_trend(s)
        assert is_sw is True
        assert abs(slope) < 0.15

    def test_slope_positive_for_uptrend(self):
        s = pd.Series([1.0 + i * 0.01 for i in range(10)])
        _, slope = classify_trend(s)
        assert slope > 0

    def test_slope_negative_for_downtrend(self):
        s = pd.Series([1.0 - i * 0.01 for i in range(10)])
        _, slope = classify_trend(s)
        assert slope < 0

    def test_nan_values_are_dropped(self):
        """NaN values are dropped; regression runs on clean bars only."""
        # 5 NaN + 10 flat bars = 10 clean bars (meets MIN_TREND_BARS=10).
        s = pd.Series([float("nan")] * 5 + [1.0] * 10)
        is_sw, slope = classify_trend(s)  # must not raise
        assert is_sw is True

    def test_minimum_bars_exact_boundary(self):
        """Exactly 10 non-NaN bars: must not raise (MIN_TREND_BARS=10)."""
        s = pd.Series([1.0] * 10)
        classify_trend(s)  # no exception

    def test_minimum_bars_below_boundary_raises(self):
        """Fewer than 10 non-NaN bars: must raise ValueError mentioning the minimum."""
        with pytest.raises(ValueError, match="10"):
            classify_trend(pd.Series([1.0] * 9))

    def test_nan_heavy_series_drops_below_minimum_raises(self):
        """6 NaN + 4 real = only 4 clean bars: must raise (MIN_TREND_BARS=10)."""
        s = pd.Series([float("nan")] * 6 + [1.0] * 4)
        with pytest.raises(ValueError, match="10"):
            classify_trend(s)

    def test_smoke_scm_mi(self):
        """
        SCM.MI 2016-08-25 to 2016-10-17: 38 bars.
        Verified OLS slope = 0.0087%/day -> is_sideways=True.
        """
        is_sw, slope = classify_trend(SCM_RCLOSE)
        assert is_sw is True, f"expected sideways, got slope={slope:.4f}%/day"
        assert slope == pytest.approx(0.0087, abs=0.001)


# ===========================================================================
# consolidation_age
# ===========================================================================


class TestBreakoutPriorConsolidationLength:

    def test_basic_five_bar_consolidation(self):
        """5-bar consolidation then breakout: flip bar reports prior age=5."""
        rbo = pd.Series([0, 0, 0, 0, 0, 1, 1])
        result = breakout_prior_consolidation_length(rbo)
        assert result.iloc[5] == 5
        assert result.iloc[0:5].isna().all()
        assert pd.isna(result.iloc[6])

    def test_no_breakout_all_nan(self):
        rbo = pd.Series([0, 0, 0, 0, 0])
        assert breakout_prior_consolidation_length(rbo).isna().all()

    def test_already_in_trend_no_zero_all_nan(self):
        """Series starts and stays at +1: no flip bar -> all NaN."""
        rbo = pd.Series([1, 1, 1, 1])
        assert breakout_prior_consolidation_length(rbo).isna().all()

    def test_one_bar_consolidation(self):
        """Single 0 between two +1 runs: flip bar reports prior age=1."""
        rbo = pd.Series([1, 1, 0, 1, 1])
        result = breakout_prior_consolidation_length(rbo)
        assert result.iloc[3] == 1
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[4])

    def test_bearish_breakout_detected(self):
        """0 to -1 flip also counts as a breakout."""
        rbo = pd.Series([0, 0, 0, -1])
        result = breakout_prior_consolidation_length(rbo)
        assert result.iloc[3] == 3
        assert result.iloc[0:3].isna().all()

    def test_multiple_breakouts(self):
        """Consolidation, breakout, re-consolidation, second breakout."""
        rbo = pd.Series([0, 0, 1, 1, 0, 0, 0, 1])
        result = breakout_prior_consolidation_length(rbo)
        assert result.iloc[2] == 2   # first breakout: 2 prior zeros
        assert result.iloc[7] == 3   # second breakout: 3 prior zeros
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[4])

    def test_flip_bar_is_only_non_nan_in_run(self):
        """Only the breakout flip bar (first bar of the non-zero run) is non-NaN."""
        rbo = pd.Series([0, 0, 1, 1, 1])
        result = breakout_prior_consolidation_length(rbo)
        assert result.iloc[2] == 2
        # continuation bars 3 and 4 must be NaN
        assert pd.isna(result.iloc[3])
        assert pd.isna(result.iloc[4])

    def test_result_is_always_positive_at_flip(self):
        """Returned age is always >= 1 at any flip bar."""
        rbo = pd.Series([0, 1])
        result = breakout_prior_consolidation_length(rbo)
        assert result.iloc[1] >= 1


# ===========================================================================
# measure_volatility_compression
# ===========================================================================

# SCM.MI 38-bar window band_width_pct values (verified from parquet).
# Derived as (rhi_20 - rlo_20) / rclose * 100 for each bar.
_SCM_BW = [
    12.2385, 11.9874, 11.8529, 12.0209,  9.8265,  9.8148,  9.9662,  9.9697,  9.8903,
    10.0295, 10.0775,  9.9499,  9.7667,  9.6794,  9.6747,  9.9527,  9.7110,  9.8350,
     4.2139,  4.2115,  4.2857,  4.2381,  4.0124,  3.9983,  4.0082,  4.0371,  4.0524,
     4.0210,  4.0294,  4.0711,  3.1029,  3.0865,  3.1292,  3.0995,  3.0775,  3.0399,
     3.1001,  3.1074,
]


def _bw_df(bw_values: list) -> pd.DataFrame:
    """
    Build a minimal DataFrame from band_width_pct values for compression tests.
    Sets rclose=1.0, rhi_20 = 1 + bw/200, rlo_20 = 1 - bw/200 so that
    (rhi_20 - rlo_20) / rclose * 100 == bw_value exactly.
    """
    bw = np.array(bw_values) / 100.0
    return pd.DataFrame({
        "rhi_20": 1.0 + bw / 2,
        "rlo_20": 1.0 - bw / 2,
        "rclose": np.ones(len(bw)),
    })


class TestVolatilityCompression:

    def test_decreasing_bw_is_compressed(self):
        """Band width shrinking linearly from 10% to 2% over 30 bars.
        Slope is negative, rank is at the bottom -> is_compressed=True.
        """
        n = 30
        bw = [10.0 - i * (8.0 / (n - 1)) for i in range(n)]
        result = measure_volatility_compression(_bw_df(bw), window_bars=n, history_bars=n)
        assert result.band_width_slope < 0
        assert result.band_width_pct_rank < 25.0
        assert result.is_compressed is True

    def test_increasing_bw_not_compressed(self):
        """Band width expanding from 2% to 10% -> slope > 0, is_compressed=False."""
        n = 30
        bw = [2.0 + i * (8.0 / (n - 1)) for i in range(n)]
        result = measure_volatility_compression(_bw_df(bw), window_bars=n, history_bars=n)
        assert result.band_width_slope > 0
        assert result.is_compressed is False

    def test_flat_bw_slope_zero_not_compressed(self):
        """Constant band width -> slope=0, is_compressed=False (slope not strictly < 0)."""
        result = measure_volatility_compression(_bw_df([5.0] * 30), window_bars=30, history_bars=30)
        assert result.band_width_slope == pytest.approx(0.0, abs=1e-9)
        assert result.is_compressed is False

    def test_current_bw_is_last_bar_value(self):
        """band_width_pct reports the value at the last bar (needs >= MIN_TREND_BARS=10 bars)."""
        bw = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.0]
        result = measure_volatility_compression(_bw_df(bw), window_bars=10, history_bars=10)
        assert result.band_width_pct == pytest.approx(2.0, abs=1e-4)

    def test_rank_at_minimum_is_low(self):
        """Current bw is the smallest value in the history window -> rank = 1/N * 100."""
        n = 10
        bw = [float(10 - i) for i in range(n)]   # 10, 9, ..., 1
        result = measure_volatility_compression(_bw_df(bw), window_bars=n, history_bars=n)
        # current = 1.0 = minimum; only 1 value (itself) is <= 1.0
        assert result.band_width_pct_rank == pytest.approx(1.0 / n * 100, abs=0.1)

    def test_rank_at_maximum_is_100(self):
        """Current bw is the largest value in the history window -> rank = 100."""
        n = 10
        bw = [float(i + 1) for i in range(n)]   # 1, 2, ..., 10
        result = measure_volatility_compression(_bw_df(bw), window_bars=n, history_bars=n)
        assert result.band_width_pct_rank == pytest.approx(100.0, abs=0.1)

    def test_high_rank_prevents_compression(self):
        """slope < 0 but rank >= 25 -> is_compressed=False (not historically tight)."""
        # Build: history is wide (high bw), window narrows only slightly.
        # Wide history: bw goes from 50 down to 26 (rank of 26 in [26..50] = 2/26 ~ 7.7%)
        # Wait, I need rank >= 25. So current bw should be in the upper half of history.
        # Strategy: long history of low bw, then recent high bw (so current is at top rank).
        # But then slope would be positive. Let me think differently:
        # history has mostly high bw. Recent window has slightly decreasing but still high bw.
        # E.g. history=[8..10] (20 bars), window=[10, 9.5, 9.0, 8.5, 8.0] (last 5)
        # current=8.0, count(bw<=8.0)/25 * 100 -- need to compute.
        history_part = [float(8 + (i % 3)) for i in range(20)]   # oscillates 8/9/10
        window_part  = [10.0, 9.5, 9.0, 8.5, 8.0]
        bw = history_part + window_part
        result = measure_volatility_compression(_bw_df(bw), window_bars=5, history_bars=25)
        # slope over the 5-bar window is negative
        assert result.band_width_slope < 0
        # current=8.0 is NOT low relative to history (which has many 8.0, 9.0, 10.0 values)
        assert result.band_width_pct_rank >= 25.0
        assert result.is_compressed is False

    def test_nan_rows_in_rhi_are_dropped(self):
        """NaN values in rhi_20 / rlo_20 are excluded before any computation."""
        # 2 NaN bars + 10 clean bars → 10 clean bars meets MIN_TREND_BARS=10.
        # bw values (clean bars): [10, 9, 8, 7, 6, 5, 4, 3, 2.5, 2]
        clean_hi = [1.0 + v / 200 for v in [10, 9, 8, 7, 6, 5, 4, 3, 2.5, 2]]
        clean_lo = [1.0 - v / 200 for v in [10, 9, 8, 7, 6, 5, 4, 3, 2.5, 2]]
        df = pd.DataFrame({
            "rhi_20": [float("nan"), float("nan")] + clean_hi,
            "rlo_20": [float("nan"), float("nan")] + clean_lo,
            "rclose": [1.0] * 12,
        })
        # After dropping NaN: 10 clean bars; last bar bw=2.0
        result = measure_volatility_compression(df, window_bars=10, history_bars=10)
        assert result.band_width_pct == pytest.approx(2.0, abs=0.01)

    def test_too_few_bars_raises(self):
        with pytest.raises(ValueError, match="10"):
            measure_volatility_compression(_bw_df([5.0] * 9), window_bars=9, history_bars=9)

    def test_missing_rlo_20_raises(self):
        df = pd.DataFrame({"rhi_20": [1.05] * 5, "rclose": [1.0] * 5})
        with pytest.raises(ValueError, match="rlo_20"):
            measure_volatility_compression(df)

    def test_missing_rhi_20_raises(self):
        df = pd.DataFrame({"rlo_20": [0.95] * 5, "rclose": [1.0] * 5})
        with pytest.raises(ValueError, match="rhi_20"):
            measure_volatility_compression(df)

    def test_smoke_scm_mi(self):
        """
        SCM.MI 2016-08-25 to 2016-10-17: 38-bar consolidation.
        band_width_pct compresses 12.24% -> 3.11%.
        Verified values: slope=-0.2871 %/bar, rank=18.42%, is_compressed=True.
        """
        df = _bw_df(_SCM_BW)
        result = measure_volatility_compression(df, window_bars=38, history_bars=38)
        assert result.band_width_slope   == pytest.approx(-0.2871, abs=0.001)
        assert result.band_width_pct     == pytest.approx(3.1074,  abs=0.001)
        assert result.band_width_pct_rank == pytest.approx(18.42,  abs=0.1)
        assert result.is_compressed is True


# ===========================================================================
# assess_range
# ===========================================================================


def _make_range_df(
    n_zero: int = 20,
    n_breakout: int = 0,
    rclose: list | None = None,
    range_pct: list | None = None,
    band_width_pct: float = 10.0,
) -> pd.DataFrame:
    """
    Build a minimal DataFrame for assess_range tests.

    Constructs rhi_20 / rlo_20 from rclose, range_pct, and band_width_pct
    so that all three are mathematically consistent:
        band_width = (rhi - rlo) / rclose * 100
        range_pct  = (rclose - rlo) / (rhi - rlo) * 100

    Any trailing n_breakout rows have rbo_20=1.0; all preceding rows have rbo_20=0.
    """
    n = n_zero + n_breakout

    rclose_arr = np.ones(n) if rclose is None else np.asarray(rclose, dtype=float)

    # range_pct defaults to flat 50% for zero bars; breakout bars are set to 50% too
    # (they are outside the zero-run and not used by assess_range's analysis).
    if range_pct is None:
        pct_arr = np.full(n, 50.0)
    else:
        pct_arr = np.concatenate([range_pct, [50.0] * n_breakout]).astype(float)

    bw_frac = band_width_pct / 100.0
    delta   = bw_frac * rclose_arr          # rhi - rlo
    rlo_arr = rclose_arr - (pct_arr / 100.0) * delta
    rhi_arr = rlo_arr + delta

    rbo_arr = np.array([0.0] * n_zero + [1.0] * n_breakout)

    return pd.DataFrame({
        "rbo_20": rbo_arr,
        "rclose": rclose_arr,
        "rhi_20": rhi_arr,
        "rlo_20": rlo_arr,
    })


class TestAssessRange:

    def test_flat_consolidation_no_touches_sideways(self):
        """All rbo_20=0, price flat at midrange → (0,0) touches, is_sideways=True."""
        df = _make_range_df(n_zero=20)
        result = assess_range(df)
        assert result.n_resistance_touches == 0
        assert result.n_support_touches    == 0
        assert result.is_sideways          is True
        assert result.consolidation_bars   == 20
        assert result.band_width_pct       == pytest.approx(10.0, abs=0.01)

    def test_resistance_touches_counted(self):
        """range_pct alternating 50/90/50/90... → at least 2 resistance touches."""
        pct = ([50.0, 90.0] * 10)   # 20 bars, 10 touches
        df  = _make_range_df(n_zero=20, range_pct=pct)
        result = assess_range(df)
        assert result.n_resistance_touches >= 2

    def test_support_touches_counted(self):
        """range_pct alternating 50/10/50/10... → at least 2 support touches."""
        pct = ([50.0, 10.0] * 10)
        df  = _make_range_df(n_zero=20, range_pct=pct)
        result = assess_range(df)
        assert result.n_support_touches >= 2

    def test_sideways_false_for_trending_rclose(self):
        """rclose rising +1%/bar → slope >> 0.15 threshold → is_sideways=False."""
        n      = 20
        rcl    = [1.0 + i * 0.01 for i in range(n)]   # +1%/bar
        df     = _make_range_df(n_zero=n, rclose=rcl)
        result = assess_range(df)
        assert result.is_sideways is False
        assert result.slope_pct_per_day > 0.15

    def test_consolidation_bars_full_run_length(self):
        """consolidation_bars always reports the full zero-run, not the capped window."""
        df     = _make_range_df(n_zero=15)
        result = assess_range(df, window_bars=40)
        assert result.consolidation_bars == 15

    def test_window_bars_does_not_affect_consolidation_bars(self):
        """consolidation_bars is the full run; window_bars caps touch/trend only."""
        df = _make_range_df(n_zero=30)
        result_narrow = assess_range(df, window_bars=10)
        result_wide   = assess_range(df, window_bars=30)
        # Both must report the full 30-bar run.
        assert result_narrow.consolidation_bars == 30
        assert result_wide.consolidation_bars   == 30

    def test_window_bars_limits_touch_counting(self):
        """window_bars=14 excludes early resistance touches from the count."""
        # First 6 bars: 3 resistance touch events (90/50/90/50/90/50).
        # Last 14 bars: flat at 50% (no touches).
        pct = [90.0, 50.0, 90.0, 50.0, 90.0, 50.0] + [50.0] * 14   # 20 bars
        df  = _make_range_df(n_zero=20, range_pct=pct)

        result_narrow = assess_range(df, window_bars=14)
        result_wide   = assess_range(df, window_bars=20)

        assert result_narrow.n_resistance_touches == 0    # last 14 bars are flat
        assert result_wide.n_resistance_touches   == 3    # full 20 bars includes 3 touches

    def test_last_bar_is_breakout_uses_preceding_zero_run(self):
        """When last bar is rbo_20=1, consolidation_bars = length of preceding zero-run."""
        df     = _make_range_df(n_zero=10, n_breakout=1)
        result = assess_range(df)
        assert result.consolidation_bars == 10

    def test_band_width_pct_from_last_consolidation_bar(self):
        """band_width_pct = (rhi_20 - rlo_20) / rclose * 100 at last zero-run bar."""
        df     = _make_range_df(n_zero=10, band_width_pct=12.0)
        result = assess_range(df)
        assert result.band_width_pct == pytest.approx(12.0, abs=0.01)

    def test_band_width_pct_at_last_zero_bar_not_breakout_bar(self):
        """band_width_pct reads from the last rbo_20=0 bar, not the breakout bar."""
        # Consolidation: 10 bars, bw=8%.
        # Breakout bar: rhi/rlo set very wide (bw≈50%) — should NOT affect band_width_pct.
        n_zero = 10
        n_bo   = 1
        # Build manually: last bar (breakout) has much wider rhi/rlo.
        df = _make_range_df(n_zero=n_zero, n_breakout=n_bo, band_width_pct=8.0)
        # Overwrite the breakout bar's rhi/rlo to be very wide.
        df.loc[n_zero, "rhi_20"] = 2.0
        df.loc[n_zero, "rlo_20"] = 0.5
        result = assess_range(df)
        # band_width_pct must reflect the zero-run's last bar (≈8%), not 150%.
        assert result.band_width_pct == pytest.approx(8.0, abs=0.01)

    def test_slope_sign_correct(self):
        """slope_pct_per_day is positive for uptrend, negative for downtrend."""
        n    = 20
        up   = [1.0 + i * 0.01 for i in range(n)]
        down = [1.0 - i * 0.01 for i in range(n)]
        _, slope_up   = assess_range(_make_range_df(n_zero=n, rclose=up)).slope_pct_per_day, \
                        assess_range(_make_range_df(n_zero=n, rclose=up)).slope_pct_per_day
        slope_up   = assess_range(_make_range_df(n_zero=n, rclose=up)).slope_pct_per_day
        slope_down = assess_range(_make_range_df(n_zero=n, rclose=down)).slope_pct_per_day
        assert slope_up   > 0
        assert slope_down < 0

    def test_missing_rbo_20_raises(self):
        df = pd.DataFrame({"rclose": [1.0]*10, "rhi_20": [1.1]*10, "rlo_20": [0.9]*10})
        with pytest.raises(ValueError, match="rbo_20"):
            assess_range(df)

    def test_missing_rclose_raises(self):
        df = pd.DataFrame({"rbo_20": [0.0]*10, "rhi_20": [1.1]*10, "rlo_20": [0.9]*10})
        with pytest.raises(ValueError, match="rclose"):
            assess_range(df)

    def test_missing_rhi_20_raises(self):
        df = pd.DataFrame({"rbo_20": [0.0]*10, "rclose": [1.0]*10, "rlo_20": [0.9]*10})
        with pytest.raises(ValueError, match="rhi_20"):
            assess_range(df)

    def test_missing_rlo_20_raises(self):
        df = pd.DataFrame({"rbo_20": [0.0]*10, "rclose": [1.0]*10, "rhi_20": [1.1]*10})
        with pytest.raises(ValueError, match="rlo_20"):
            assess_range(df)

    def test_no_consolidation_raises(self):
        """Entire history in trend (no rbo_20=0) raises ValueError."""
        df = pd.DataFrame({
            "rbo_20": [1.0]*10,
            "rclose": [1.0]*10,
            "rhi_20": [1.1]*10,
            "rlo_20": [0.9]*10,
        })
        with pytest.raises(ValueError, match="consolidation"):
            assess_range(df)

    def test_too_few_bars_raises(self):
        """Zero-run shorter than MIN_TREND_BARS=10 raises ValueError from classify_trend."""
        df = _make_range_df(n_zero=9)
        with pytest.raises(ValueError, match="10"):
            assess_range(df)

    def test_smoke_scm_mi(self):
        """
        SCM.MI 2016-08-25 to 2016-10-17: 38-bar consolidation.

        Reconstructs rhi_20 / rlo_20 from the existing SCM_RANGE_PCT, SCM_RCLOSE,
        and _SCM_BW fixtures, which are individually verified by other smoke tests.

        Expected (derived from per-primitive smoke tests):
            n_resistance_touches = 2
            n_support_touches    = 2
            is_sideways          = True
            slope_pct_per_day    ≈ 0.0087
            consolidation_bars   = 38
            band_width_pct       ≈ 3.11  (last bar of _SCM_BW)
        """
        bw_arr     = np.array(_SCM_BW)
        rclose_arr = SCM_RCLOSE.to_numpy()
        pct_arr    = SCM_RANGE_PCT.to_numpy()

        # Derive rhi_20 / rlo_20 from band_width_pct and range_position_pct.
        # rlo = rclose - (range_pct / 100) * delta; delta = bw * rclose / 100
        delta   = bw_arr * rclose_arr / 100.0
        rlo_arr = rclose_arr - (pct_arr / 100.0) * delta
        rhi_arr = rlo_arr + delta

        df = pd.DataFrame({
            "rbo_20": np.zeros(38),
            "rclose": rclose_arr,
            "rhi_20": rhi_arr,
            "rlo_20": rlo_arr,
        })

        result = assess_range(df, window_bars=38)

        assert result.n_resistance_touches == 2,      f"n_res: {result.n_resistance_touches}"
        assert result.n_support_touches    == 2,      f"n_sup: {result.n_support_touches}"
        assert result.is_sideways          is True,   f"slope: {result.slope_pct_per_day}"
        assert result.consolidation_bars   == 38
        assert result.slope_pct_per_day    == pytest.approx(0.0087, abs=0.001)
        assert result.band_width_pct       == pytest.approx(3.1074, abs=0.001)


# ===========================================================================
# VolatilityState — new fields: history_available, is_rank_reliable
# ===========================================================================


class TestVolatilityStateNewFields:

    def test_history_available_equals_actual_bars(self):
        """history_available reports the number of bars passed to the rank computation."""
        n = 30
        bw = [5.0] * n
        result = measure_volatility_compression(_bw_df(bw), window_bars=n, history_bars=n)
        assert result.history_available == n

    def test_is_rank_reliable_true_when_full_history(self):
        """is_rank_reliable is True when actual history >= requested history_bars."""
        n = 30
        result = measure_volatility_compression(_bw_df([5.0] * n), window_bars=n, history_bars=n)
        assert result.is_rank_reliable is True

    def test_is_rank_reliable_false_when_short_history(self):
        """is_rank_reliable is False when fewer bars available than history_bars requests."""
        # 20 actual bars, requesting 252-bar reference window.
        n = 20
        result = measure_volatility_compression(_bw_df([5.0] * n), window_bars=n, history_bars=252)
        assert result.is_rank_reliable is False
        assert result.history_available == n

    def test_is_rank_reliable_false_triggers_warning(self, caplog):
        """Thin history must emit a WARNING about unreliable rank."""
        n = 20
        with caplog.at_level(logging.WARNING, logger="ta.breakout.range_quality"):
            measure_volatility_compression(_bw_df([5.0] * n), window_bars=n, history_bars=252)
        assert any("rank" in r.message.lower() or "history" in r.message.lower()
                   for r in caplog.records), "expected a warning about thin history"

    def test_volatility_state_is_dataclass(self):
        """VolatilityState is a frozen dataclass with the expected fields."""
        n = 30
        vs = measure_volatility_compression(_bw_df([5.0] * n), window_bars=n, history_bars=n)
        assert isinstance(vs, VolatilityState)
        assert hasattr(vs, "history_available")
        assert hasattr(vs, "is_rank_reliable")


# ===========================================================================
# breakout_prior_consolidation_length — DatetimeIndex guard
# ===========================================================================


class TestBreakoutPriorConsolidationLengthDatetimeIndex:

    def test_datetimeindex_with_gaps_works_correctly(self):
        """
        A DatetimeIndex with holiday gaps must not produce NaN via shift(1).
        reset_index(drop=True) inside the function ensures positional shift semantics.
        """
        # Build a DatetimeIndex with a gap (skipping one day — simulates a holiday).
        dates = pd.to_datetime([
            "2024-01-02", "2024-01-03", "2024-01-04",
            # 2024-01-05 is skipped (holiday)
            "2024-01-08", "2024-01-09", "2024-01-10",
        ])
        # 3 zero-run bars, then breakout on 2024-01-08 (first bar after gap).
        rbo = pd.Series([0, 0, 0, 1, 1, 1], index=dates)
        result = breakout_prior_consolidation_length(rbo)
        # The breakout flip (index 3, date 2024-01-08) must report age=3, not NaN.
        assert result.iloc[3] == 3, (
            f"Expected flip bar at iloc[3] to report age=3, got {result.iloc[3]}"
        )
        # Non-flip bars must be NaN.
        assert result.iloc[0:3].isna().all()
        assert pd.isna(result.iloc[4])
        assert pd.isna(result.iloc[5])


# ===========================================================================
# assess_range — NaN range_pct warning and max_trend_skip warning
# ===========================================================================


def _make_zero_range_df(n_zero: int = 20, n_zero_range: int = 0) -> pd.DataFrame:
    """
    Build a DataFrame where n_zero_range bars have rhi_20 == rlo_20 (zero-width range),
    which causes range_pct to be NaN. Used to test NaN detection in assess_range.
    """
    n = n_zero
    rclose_arr = np.ones(n)
    rhi_arr    = np.where(np.arange(n) < n_zero_range, 1.0, 1.05)  # first n_zero_range are degenerate
    rlo_arr    = np.where(np.arange(n) < n_zero_range, 1.0, 0.95)

    return pd.DataFrame({
        "rbo_20": np.zeros(n),
        "rclose": rclose_arr,
        "rhi_20": rhi_arr,
        "rlo_20": rlo_arr,
    })


class TestAssessRangeNewBehaviours:

    def test_nan_range_pct_warning_emitted(self, caplog):
        """
        When >10% of window bars have zero-width range (rhi==rlo → NaN range_pct),
        assess_range must emit a WARNING.
        """
        # 20-bar window, 3 degenerate bars = 15% NaN → above the 10% threshold.
        df = _make_zero_range_df(n_zero=20, n_zero_range=3)
        with caplog.at_level(logging.WARNING, logger="ta.breakout.range_quality"):
            assess_range(df, window_bars=20)
        assert any("nan" in r.message.lower() or "range_pct" in r.message.lower()
                   for r in caplog.records), "expected a warning about NaN range_pct values"

    def test_nan_range_pct_below_threshold_no_warning(self, caplog):
        """
        When ≤10% of window bars have zero-width range, no NaN warning is emitted.
        """
        # 20-bar window, 2 degenerate bars = 10% — right at the threshold, should not warn.
        df = _make_zero_range_df(n_zero=20, n_zero_range=2)
        with caplog.at_level(logging.WARNING, logger="ta.breakout.range_quality"):
            assess_range(df, window_bars=20)
        nan_warnings = [
            r for r in caplog.records
            if ("nan" in r.message.lower() or "range_pct" in r.message.lower())
            and r.levelno >= logging.WARNING
        ]
        assert len(nan_warnings) == 0

    def test_max_trend_skip_warning_emitted(self, caplog):
        """
        When assess_range skips more non-zero bars than config.max_trend_skip,
        it must emit a WARNING before continuing.

        The backward walk starts at the LAST bar and skips trailing non-zero bars,
        so the non-zero (trend) run must be at the END of the series.
        """
        # 20 consolidation bars (rbo=0) FIRST, then 60 trend bars (rbo=1) at the TAIL.
        # The backward walk skips all 60 trailing non-zero bars to find the zero-run.
        n_consol = 20
        n_trend  = 60   # exceeds max_trend_skip=50
        n        = n_consol + n_trend
        df = pd.DataFrame({
            "rbo_20": np.array([0.0] * n_consol + [1.0] * n_trend),
            "rclose": np.ones(n),
            "rhi_20": np.full(n, 1.05),
            "rlo_20": np.full(n, 0.95),
        })
        cfg = RangeQualityConfig(max_trend_skip=50)
        with caplog.at_level(logging.WARNING, logger="ta.breakout.range_quality"):
            assess_range(df, config=cfg)
        assert any("skipped" in r.message.lower()
                   for r in caplog.records), "expected a warning about excessive trend skip"

    def test_max_trend_skip_not_exceeded_no_warning(self, caplog):
        """
        When the trailing trend bars are within max_trend_skip, no skip warning is emitted.
        """
        # 20 consolidation bars, then 10 trend bars at the TAIL (within max_trend_skip=50).
        n_consol = 20
        n_trend  = 10
        n        = n_consol + n_trend
        df = pd.DataFrame({
            "rbo_20": np.array([0.0] * n_consol + [1.0] * n_trend),
            "rclose": np.ones(n),
            "rhi_20": np.full(n, 1.05),
            "rlo_20": np.full(n, 0.95),
        })
        with caplog.at_level(logging.WARNING, logger="ta.breakout.range_quality"):
            assess_range(df)
        skip_warnings = [
            r for r in caplog.records
            if "skipped" in r.message.lower()
            and r.levelno >= logging.WARNING
        ]
        assert len(skip_warnings) == 0
