"""
test_trend_quality.py — Test suite for ta.ma.trend_quality.

Coverage
--------
_wilder_running_sum:
  - known values (hand-verified), constant input, period > len, period == len, period=1

_wilder_ewm:
  - constant series stays constant, single-element period, passthrough shape

compute_rsi:
  - all-gains → RSI=100, all-losses → RSI=0, constant → RSI=50
  - too-short series raises, period < 1 raises
  - result invariant: RSI in [0, 100]

compute_adx:
  - trending data → ADX > 0, flat series → ADX near 0
  - missing columns raise, too-short raises, period < 1 raises
  - custom column names, result invariant: ADX in [0, 100]

compute_ma_gap_pct:
  - fast > slow → positive, fast < slow → negative, fast == slow → 0
  - rclose = 0 raises, rclose < 0 raises

compute_ma_slope_pct:
  - rising / falling / constant MA
  - window < 1 raises, fewer than 2 bars → 0.0

assess_ma_trend:
  - returns MATrendStrength instance with all fields including r2 fields
  - missing columns raise, custom column names accepted
  - rsi in [0, 100], adx in [0, 100]
  - adx_slope_r2 and ma_gap_slope_r2 in [0, 1]
  - zero rclose mid-series does not produce inf in gap series
  - RSI_PERIOD and ADX_PERIOD constants exist
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ta.ma.trend_quality import (
    ADX_PERIOD,
    ADX_SLOPE_WINDOW,
    ADX_TREND_THRESHOLD,
    MA_SLOPE_WINDOW,
    RSI_PERIOD,
    MATrendStrength,
    _wilder_ewm,
    _wilder_running_sum,
    assess_ma_trend,
    compute_adx,
    compute_ma_gap_pct,
    compute_ma_slope_pct,
    compute_rsi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adx_df(
    n: int = 100,
    trend: str = "up",
    high_col: str = "rhigh",
    low_col: str  = "rlow",
    close_col: str = "rclose",
) -> pd.DataFrame:
    """Minimal DataFrame for compute_adx."""
    if trend == "up":
        close = np.linspace(1.0, 3.0, n)
    elif trend == "flat":
        close = np.ones(n)
    else:
        raise ValueError(f"Unknown trend: {trend}")
    high = close + 0.05
    low  = close - 0.05
    return pd.DataFrame({high_col: high, low_col: low, close_col: close})


def _make_full_df(n: int = 120) -> pd.DataFrame:
    """Full DataFrame for assess_ma_trend with all default columns."""
    close   = np.linspace(1.0, 2.0, n)
    high    = close * 1.01
    low     = close * 0.99
    fast_ma = np.linspace(1.1, 2.0, n)
    slow_ma = np.linspace(1.0, 1.7, n)
    med_ma  = (fast_ma + slow_ma) / 2
    return pd.DataFrame({
        "rclose":          close,
        "rhigh":           high,
        "rlow":            low,
        "rema_short_50":   fast_ma,
        "rema_medium_100": med_ma,
        "rema_long_150":   slow_ma,
    })


# ---------------------------------------------------------------------------
# _wilder_running_sum
# ---------------------------------------------------------------------------


class TestWilderRunningSum:
    def test_known_values_period_2(self):
        """
        arr=[1,2,3,4,5], period=2:
          seed=3, then S_j=(1-0.5)*S_{j-1}+arr[j]
          → [3, 4.5, 6.25, 8.125]
        """
        arr    = np.array([1., 2., 3., 4., 5.])
        result = _wilder_running_sum(arr, period=2)
        assert np.allclose(result, [3.0, 4.5, 6.25, 8.125], rtol=1e-6)

    def test_constant_input_stays_at_period_sum(self):
        """All-ones with period=14: seed=14, stays 14 forever."""
        arr    = np.ones(30)
        result = _wilder_running_sum(arr, period=14)
        assert np.allclose(result, 14.0, rtol=1e-6)

    def test_output_length(self):
        arr = np.ones(20)
        for p in [1, 3, 7, 14]:
            result = _wilder_running_sum(arr, period=p)
            assert len(result) == 20 - p + 1, f"period={p}"

    def test_period_larger_than_array_returns_empty(self):
        arr = np.array([1., 2., 3.])
        result = _wilder_running_sum(arr, period=5)
        assert len(result) == 0

    def test_period_equals_array_length(self):
        """Single output: just the sum of all elements."""
        arr    = np.array([1., 2., 3., 4.])
        result = _wilder_running_sum(arr, period=4)
        assert len(result) == 1
        assert result[0] == pytest.approx(10.0)

    def test_period_one_is_identity(self):
        """With period=1: S_j = 0*S_{j-1} + arr[j] = arr[j]."""
        arr    = np.array([3., 7., 2., 5.])
        result = _wilder_running_sum(arr, period=1)
        assert np.allclose(result, arr)

    # ------------------------------------------------------------------
    # Overflow safety for long time series
    # ------------------------------------------------------------------

    def test_period2_long_series_no_overflow(self):
        """
        With period=2, beta=0.5, (1/beta)^k = 2^k.
        2^1024 overflows float64 (~1.8e308), so the geometric-decay identity
        produces NaN/Inf at ~1024 bars when computed naïvely.

        For all-ones with period=2 the recurrence S[k] = 0.5*S[k-1] + 1
        reaches a steady state of exactly 2.0 (seed=2, stays there).
        All 2000 output values must be finite and equal 2.0.
        """
        arr    = np.ones(2000, dtype=float)
        result = _wilder_running_sum(arr, period=2)
        assert np.all(np.isfinite(result)), "Overflow detected: result contains NaN or Inf"
        assert np.allclose(result, 2.0, rtol=1e-6), (
            f"Steady-state value should be 2.0; got range [{result.min():.4f}, {result.max():.4f}]"
        )

    def test_period3_long_series_no_overflow(self):
        """
        period=3: beta=2/3, 1/beta=1.5. 1.5^1750 ≈ 10^307 (near float64 max);
        1.5^2000 overflows.  The batched approach must remain finite.

        For all-ones with period=3:
          seed = 3, steady state S* = 0.667*S* + 1 → S* = 3.0.
        """
        arr    = np.ones(2000, dtype=float)
        result = _wilder_running_sum(arr, period=3)
        assert np.all(np.isfinite(result)), "Overflow detected for period=3"
        assert np.allclose(result, 3.0, rtol=1e-6)

    def test_long_series_matches_recursive_ground_truth(self):
        """
        Verify the batched output against a simple Python-loop ground truth
        on a 3000-bar random series with period=2, so overflow would corrupt
        results in the current implementation.
        """
        rng    = np.random.default_rng(42)
        arr    = rng.uniform(0.0, 0.1, 3000)
        period = 2
        beta   = 1.0 - 1.0 / period

        # Ground truth via plain Python loop (slow but unambiguously correct)
        gt = [float(arr[:period].sum())]
        for v in arr[period:]:
            gt.append(beta * gt[-1] + float(v))
        gt = np.array(gt)

        result = _wilder_running_sum(arr, period=period)
        assert np.allclose(result, gt, rtol=1e-6), (
            "Batched result diverges from ground truth"
        )


# ---------------------------------------------------------------------------
# _wilder_ewm
# ---------------------------------------------------------------------------


class TestWilderEwm:
    def test_constant_series_stays_constant(self):
        """Constant input: EWM seed = mean = value; stays there."""
        arr    = np.full(20, 2.5)
        result = _wilder_ewm(arr, period=7)
        assert np.allclose(result, 2.5, rtol=1e-6)

    def test_output_length(self):
        arr = np.ones(25)
        for p in [2, 5, 10, 14]:
            result = _wilder_ewm(arr, period=p)
            assert len(result) == 25 - p + 1, f"period={p}"

    def test_period_equals_array_length(self):
        """Single output: just the mean of all elements."""
        arr    = np.array([1., 3., 5., 7.])
        result = _wilder_ewm(arr, period=4)
        assert len(result) == 1
        assert result[0] == pytest.approx(4.0)

    def test_period_larger_than_array_returns_empty(self):
        arr    = np.array([1., 2.])
        result = _wilder_ewm(arr, period=5)
        assert len(result) == 0

    def test_convergence_to_new_level(self):
        """Step-change input: later values converge toward the new level."""
        # First 14 bars at 1.0, then many bars at 2.0 — output should approach 2.0
        arr    = np.concatenate([np.ones(14), np.full(100, 2.0)])
        result = _wilder_ewm(arr, period=14)
        # Last value should be well above 1.5 (converging toward 2.0)
        assert result[-1] > 1.9


# ---------------------------------------------------------------------------
# compute_rsi
# ---------------------------------------------------------------------------


class TestComputeRsi:
    def test_all_gains_gives_rsi_100(self):
        close = pd.Series(np.linspace(1.0, 3.0, RSI_PERIOD + 1))
        result = compute_rsi(close)
        assert result == pytest.approx(100.0, abs=1e-4)

    def test_all_losses_gives_rsi_0(self):
        close = pd.Series(np.linspace(3.0, 1.0, RSI_PERIOD + 1))
        result = compute_rsi(close)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_constant_series_gives_rsi_50(self):
        close = pd.Series(np.ones(RSI_PERIOD + 5))
        result = compute_rsi(close)
        assert result == pytest.approx(50.0, abs=1e-4)

    def test_result_in_unit_range(self):
        rng = np.random.default_rng(42)
        close = pd.Series(rng.standard_normal(50).cumsum() + 10)
        result = compute_rsi(close)
        assert 0.0 <= result <= 100.0

    def test_too_short_raises(self):
        close = pd.Series(np.ones(RSI_PERIOD))  # needs period+1
        with pytest.raises(ValueError, match="at least"):
            compute_rsi(close)

    def test_period_zero_raises(self):
        with pytest.raises(ValueError, match="period must be >= 1"):
            compute_rsi(pd.Series(np.ones(20)), period=0)

    def test_period_negative_raises(self):
        with pytest.raises(ValueError, match="period must be >= 1"):
            compute_rsi(pd.Series(np.ones(20)), period=-5)

    def test_custom_period(self):
        close = pd.Series(np.linspace(1.0, 2.0, 30))
        result = compute_rsi(close, period=7)
        assert 0.0 <= result <= 100.0

    def test_nan_in_series_are_dropped(self):
        """NaN values are silently dropped; result still computed on clean data."""
        clean   = pd.Series(np.linspace(1.0, 2.0, RSI_PERIOD + 2))
        with_nan = clean.copy()
        with_nan.iloc[3] = np.nan
        # Both should produce a valid RSI (not necessarily equal, but in range)
        result = compute_rsi(with_nan)
        assert 0.0 <= result <= 100.0


# ---------------------------------------------------------------------------
# compute_adx
# ---------------------------------------------------------------------------


class TestComputeAdx:
    def test_strong_uptrend_gives_positive_adx(self):
        """Strongly trending prices should produce ADX > 0."""
        df = _make_adx_df(n=100, trend="up")
        result = compute_adx(df)
        assert result > 0.0

    def test_flat_series_gives_near_zero_adx(self):
        """Flat prices: +DM = -DM = 0 → ADX should be near 0."""
        df = _make_adx_df(n=100, trend="flat")
        result = compute_adx(df)
        assert result == pytest.approx(0.0, abs=1.0)

    def test_result_in_unit_range(self):
        df = _make_adx_df(n=100, trend="up")
        assert 0.0 <= compute_adx(df) <= 100.0

    def test_missing_high_column_raises(self):
        df = pd.DataFrame({"rlow": [1.0]*30, "rclose": [1.0]*30})
        with pytest.raises(ValueError, match="missing required columns"):
            compute_adx(df)

    def test_missing_low_column_raises(self):
        df = pd.DataFrame({"rhigh": [1.0]*30, "rclose": [1.0]*30})
        with pytest.raises(ValueError, match="missing required columns"):
            compute_adx(df)

    def test_too_short_raises(self):
        df = _make_adx_df(n=2*ADX_PERIOD)  # needs 2*period+1
        with pytest.raises(ValueError, match="at least"):
            compute_adx(df)

    def test_period_zero_raises(self):
        df = _make_adx_df()
        with pytest.raises(ValueError, match="period must be >= 1"):
            compute_adx(df, period=0)

    def test_custom_column_names_accepted(self):
        df = _make_adx_df(high_col="hi", low_col="lo", close_col="cl")
        result = compute_adx(df, high_col="hi", low_col="lo", close_col="cl")
        assert 0.0 <= result <= 100.0

    def test_custom_col_missing_raises(self):
        df = _make_adx_df()  # has "rhigh", "rlow", "rclose"
        with pytest.raises(ValueError, match="missing required columns"):
            compute_adx(df, high_col="hi")  # "hi" not in df


# ---------------------------------------------------------------------------
# compute_ma_gap_pct
# ---------------------------------------------------------------------------


class TestComputeMaGapPct:
    def test_fast_above_slow_is_positive(self):
        result = compute_ma_gap_pct(fast_ma=1.5, slow_ma=1.0, rclose=1.0)
        assert result > 0.0

    def test_fast_below_slow_is_negative(self):
        result = compute_ma_gap_pct(fast_ma=1.0, slow_ma=1.5, rclose=1.0)
        assert result < 0.0

    def test_fast_equals_slow_is_zero(self):
        result = compute_ma_gap_pct(fast_ma=1.2, slow_ma=1.2, rclose=1.0)
        assert result == pytest.approx(0.0)

    def test_rclose_zero_raises(self):
        with pytest.raises(ValueError, match="rclose must be > 0"):
            compute_ma_gap_pct(fast_ma=1.5, slow_ma=1.0, rclose=0.0)

    def test_rclose_negative_raises(self):
        with pytest.raises(ValueError, match="rclose must be > 0"):
            compute_ma_gap_pct(fast_ma=1.5, slow_ma=1.0, rclose=-1.0)

    def test_formula_correctness(self):
        """gap = (fast - slow) / rclose * 100."""
        result = compute_ma_gap_pct(fast_ma=2.0, slow_ma=1.0, rclose=2.0)
        assert result == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# compute_ma_slope_pct
# ---------------------------------------------------------------------------


class TestComputeMaSlopePct:
    def test_rising_ma_gives_positive_slope(self):
        ma = pd.Series(np.linspace(1.0, 2.0, 30))
        result = compute_ma_slope_pct(ma, window=20)
        assert result > 0.0

    def test_falling_ma_gives_negative_slope(self):
        ma = pd.Series(np.linspace(2.0, 1.0, 30))
        result = compute_ma_slope_pct(ma, window=20)
        assert result < 0.0

    def test_constant_ma_gives_zero_slope(self):
        ma = pd.Series(np.ones(30))
        result = compute_ma_slope_pct(ma, window=20)
        assert result == pytest.approx(0.0, abs=1e-8)

    def test_window_zero_raises(self):
        ma = pd.Series(np.ones(20))
        with pytest.raises(ValueError, match="window must be >= 1"):
            compute_ma_slope_pct(ma, window=0)

    def test_window_negative_raises(self):
        ma = pd.Series(np.ones(20))
        with pytest.raises(ValueError, match="window must be >= 1"):
            compute_ma_slope_pct(ma, window=-3)

    def test_fewer_than_two_bars_returns_zero(self):
        ma = pd.Series([2.5])
        result = compute_ma_slope_pct(ma, window=5)
        assert result == 0.0


# ---------------------------------------------------------------------------
# assess_ma_trend
# ---------------------------------------------------------------------------


class TestAssessMaTrend:
    def test_returns_ma_trend_strength_instance(self):
        result = assess_ma_trend(_make_full_df())
        assert isinstance(result, MATrendStrength)

    def test_all_fields_populated(self):
        result = assess_ma_trend(_make_full_df())
        for field in ("rsi", "adx", "adx_slope", "adx_slope_r2",
                      "ma_gap_pct", "ma_gap_slope", "ma_gap_slope_r2",
                      "is_trending"):
            assert hasattr(result, field), f"Missing field: {field}"

    def test_rsi_in_unit_range(self):
        result = assess_ma_trend(_make_full_df())
        assert 0.0 <= result.rsi <= 100.0

    def test_adx_in_unit_range(self):
        result = assess_ma_trend(_make_full_df())
        assert 0.0 <= result.adx <= 100.0

    def test_adx_slope_r2_in_unit_interval(self):
        result = assess_ma_trend(_make_full_df())
        assert 0.0 <= result.adx_slope_r2 <= 1.0

    def test_ma_gap_slope_r2_in_unit_interval(self):
        result = assess_ma_trend(_make_full_df())
        assert 0.0 <= result.ma_gap_slope_r2 <= 1.0

    def test_bullish_alignment_gives_positive_gap(self):
        """fast_ma > slow_ma throughout → ma_gap_pct > 0."""
        result = assess_ma_trend(_make_full_df())
        assert result.ma_gap_pct > 0.0

    def test_is_trending_is_bool(self):
        result = assess_ma_trend(_make_full_df())
        assert isinstance(result.is_trending, bool)

    def test_missing_rclose_column_raises(self):
        df = _make_full_df().drop(columns=["rclose"])
        with pytest.raises(ValueError, match="missing required columns"):
            assess_ma_trend(df)

    def test_missing_fast_ma_column_raises(self):
        df = _make_full_df().drop(columns=["rema_short_50"])
        with pytest.raises(ValueError, match="missing required columns"):
            assess_ma_trend(df)

    def test_custom_column_names_accepted(self):
        df = _make_full_df().rename(columns={
            "rclose": "cl", "rhigh": "hi", "rlow": "lo",
            "rema_short_50": "fast", "rema_long_150": "slow",
            "rema_medium_100": "med",
        })
        result = assess_ma_trend(
            df,
            close_col="cl", high_col="hi", low_col="lo",
            fast_ma_col="fast", slow_ma_col="slow",
        )
        assert isinstance(result, MATrendStrength)

    def test_zero_rclose_mid_series_does_not_produce_inf(self):
        """A zero rclose in the middle of the series must be guarded (→ NaN gap, not inf)."""
        df = _make_full_df()
        df.loc[df.index[50], "rclose"] = 0.0  # inject zero mid-series
        result = assess_ma_trend(df)
        assert np.isfinite(result.ma_gap_slope), "ma_gap_slope must be finite even with a zero rclose bar"

    def test_datetimeindex_accepted(self):
        df = _make_full_df()
        df.index = pd.date_range("2023-01-01", periods=len(df), freq="D")
        result = assess_ma_trend(df)
        assert isinstance(result, MATrendStrength)

    def test_rsi_period_parameter_accepted(self):
        result = assess_ma_trend(_make_full_df(), rsi_period=7)
        assert 0.0 <= result.rsi <= 100.0

    def test_adx_period_parameter_accepted(self):
        result = assess_ma_trend(_make_full_df(), adx_period=7)
        assert 0.0 <= result.adx <= 100.0

    def test_module_constants_exist(self):
        """Ensure the named constants are exported and have sane defaults."""
        assert RSI_PERIOD == 14
        assert ADX_PERIOD == 14
        assert ADX_TREND_THRESHOLD == pytest.approx(25.0)
        assert ADX_SLOPE_WINDOW >= 1
        assert MA_SLOPE_WINDOW >= 1
