"""
Unit tests for ta.ma.trend_quality — MA crossover trader strength primitives.

TDD Red → Green → Refactor cycle:
  All tests written BEFORE implementation exists.

Functions under test:
  compute_rsi(rclose, period)        → float
  compute_adx(df, period)            → float
  compute_ma_gap_pct(fast, slow, rc) → float
  compute_ma_slope_pct(ma, window)   → float
  assess_ma_trend(df)                → MATrendStrength
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ta.ma.trend_quality import (
    MATrendStrength,
    assess_ma_trend,
    compute_adx,
    compute_ma_gap_pct,
    compute_ma_slope_pct,
    compute_rsi,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _rclose_gains(n: int = 30, period: int = 14) -> pd.Series:
    """All-gain series of length n: RSI should approach 100."""
    return pd.Series([1.0 + i * 0.01 for i in range(n)])


def _rclose_losses(n: int = 30, period: int = 14) -> pd.Series:
    """All-loss series of length n: RSI should approach 0."""
    return pd.Series([1.0 - i * 0.005 for i in range(n)])


def _rclose_flat(n: int = 30) -> pd.Series:
    return pd.Series([1.0] * n)


def _trending_df(n: int = 50, step: float = 0.002) -> pd.DataFrame:
    """
    Build a minimal DataFrame with rclose, rhigh, rlow rising linearly.
    A steadily trending series should produce ADX > 25 and still rising after warm-up.
    step=0.002 (0.2%/bar) is moderate enough that ADX stays between 25–80, not pinned at 100.
    """
    rclose = np.array([1.0 + i * step for i in range(n)])
    rhigh  = rclose + 0.005
    rlow   = rclose - 0.005
    return pd.DataFrame({
        "rclose": rclose,
        "rhigh":  rhigh,
        "rlow":   rlow,
        "rema_short_50":   rclose * 0.99,   # fast MA slightly below rclose (bullish)
        "rema_long_150":   rclose * 0.97,   # slow MA further below
        "rema_medium_100": rclose * 0.98,
    })


def _flat_df(n: int = 50) -> pd.DataFrame:
    """Flat series: ADX should be low, RSI near 50."""
    rclose = np.ones(n)
    return pd.DataFrame({
        "rclose": rclose,
        "rhigh":  rclose + 0.001,
        "rlow":   rclose - 0.001,
        "rema_short_50":   rclose,
        "rema_long_150":   rclose,
        "rema_medium_100": rclose,
    })


# ===========================================================================
# compute_rsi
# ===========================================================================


class TestComputeRsi:

    def test_all_gains_rsi_high(self):
        """All-gain series: RSI approaches 100."""
        rsi = compute_rsi(_rclose_gains(30), period=14)
        assert rsi > 80.0

    def test_all_losses_rsi_low(self):
        """All-loss series: RSI approaches 0."""
        rsi = compute_rsi(_rclose_losses(30), period=14)
        assert rsi < 20.0

    def test_flat_series_rsi_near_50(self):
        """Constant series (no gains, no losses): RSI = 50."""
        rsi = compute_rsi(_rclose_flat(30), period=14)
        assert rsi == pytest.approx(50.0, abs=5.0)

    def test_returns_float(self):
        assert isinstance(compute_rsi(_rclose_gains(20)), float)

    def test_rsi_bounded_0_to_100(self):
        """RSI must always be in [0, 100]."""
        for s in [_rclose_gains(30), _rclose_losses(30), _rclose_flat(30)]:
            rsi = compute_rsi(s)
            assert 0.0 <= rsi <= 100.0

    def test_too_few_bars_raises(self):
        """Fewer than period+1 non-NaN values: raises ValueError."""
        with pytest.raises(ValueError, match="period"):
            compute_rsi(pd.Series([1.0] * 5), period=14)

    def test_exactly_period_plus_1_bars_ok(self):
        """Exactly period+1 bars: must not raise."""
        compute_rsi(pd.Series([float(i) for i in range(16)]), period=14)

    def test_nan_values_handled(self):
        """NaN values are dropped before computation."""
        s = pd.Series([float("nan")] * 5 + [1.0 + i * 0.01 for i in range(20)])
        rsi = compute_rsi(s, period=14)
        assert 0.0 <= rsi <= 100.0

    def test_alternating_gains_losses_midrange(self):
        """Alternating +0.01 / -0.01: RSI should be near 50."""
        vals = [1.0]
        for i in range(29):
            vals.append(vals[-1] + (0.01 if i % 2 == 0 else -0.01))
        rsi = compute_rsi(pd.Series(vals), period=14)
        assert 35.0 < rsi < 65.0


# ===========================================================================
# compute_adx
# ===========================================================================


class TestComputeAdx:

    def test_strongly_trending_adx_above_threshold(self):
        """Steadily rising series: ADX > 25 after warm-up."""
        df = _trending_df(n=60, step=0.01)
        adx = compute_adx(df, period=14)
        assert adx > 25.0, f"expected ADX > 25 for trending series, got {adx:.2f}"

    def test_flat_series_adx_low(self):
        """Flat series: ADX < 20 (weak or no trend)."""
        df = _flat_df(n=60)
        adx = compute_adx(df, period=14)
        assert adx < 20.0, f"expected ADX < 20 for flat series, got {adx:.2f}"

    def test_returns_float(self):
        assert isinstance(compute_adx(_trending_df(), period=14), float)

    def test_adx_non_negative(self):
        """ADX is always >= 0."""
        assert compute_adx(_trending_df(), period=14) >= 0.0
        assert compute_adx(_flat_df(), period=14) >= 0.0

    def test_missing_rhigh_raises(self):
        df = pd.DataFrame({"rclose": [1.0] * 30, "rlow": [0.99] * 30})
        with pytest.raises(ValueError, match="rhigh"):
            compute_adx(df)

    def test_missing_rlow_raises(self):
        df = pd.DataFrame({"rclose": [1.0] * 30, "rhigh": [1.01] * 30})
        with pytest.raises(ValueError, match="rlow"):
            compute_adx(df)

    def test_missing_rclose_raises(self):
        df = pd.DataFrame({"rhigh": [1.01] * 30, "rlow": [0.99] * 30})
        with pytest.raises(ValueError, match="rclose"):
            compute_adx(df)

    def test_too_few_bars_raises(self):
        """Fewer than 2*period bars: raises ValueError."""
        df = _trending_df(n=10)
        with pytest.raises(ValueError, match="period"):
            compute_adx(df, period=14)

    def test_adx_bounded_0_to_100(self):
        """ADX is always in [0, 100]."""
        adx = compute_adx(_trending_df(n=60), period=14)
        assert 0.0 <= adx <= 100.0


# ===========================================================================
# compute_ma_gap_pct
# ===========================================================================


class TestComputeMaGapPct:

    def test_fast_above_slow_positive_gap(self):
        """fast_ma > slow_ma: gap is positive (bullish)."""
        gap = compute_ma_gap_pct(fast_ma=1.05, slow_ma=1.00, rclose=1.02)
        assert gap > 0.0

    def test_fast_below_slow_negative_gap(self):
        """fast_ma < slow_ma: gap is negative (bearish)."""
        gap = compute_ma_gap_pct(fast_ma=0.98, slow_ma=1.00, rclose=1.00)
        assert gap < 0.0

    def test_equal_mas_gap_zero(self):
        gap = compute_ma_gap_pct(fast_ma=1.00, slow_ma=1.00, rclose=1.00)
        assert gap == pytest.approx(0.0, abs=1e-9)

    def test_formula_correct(self):
        """(fast - slow) / rclose * 100 = (1.05 - 1.00) / 1.02 * 100 ≈ 4.902."""
        gap = compute_ma_gap_pct(fast_ma=1.05, slow_ma=1.00, rclose=1.02)
        assert gap == pytest.approx((1.05 - 1.00) / 1.02 * 100, rel=1e-4)

    def test_zero_rclose_raises(self):
        with pytest.raises(ValueError, match="rclose"):
            compute_ma_gap_pct(fast_ma=1.0, slow_ma=1.0, rclose=0.0)

    def test_negative_rclose_raises(self):
        with pytest.raises(ValueError, match="rclose"):
            compute_ma_gap_pct(fast_ma=1.0, slow_ma=1.0, rclose=-1.0)

    def test_returns_float(self):
        assert isinstance(compute_ma_gap_pct(1.0, 1.0, 1.0), float)


# ===========================================================================
# compute_ma_slope_pct
# ===========================================================================


class TestComputeMaSlopePct:

    def test_rising_ma_positive_slope(self):
        s = pd.Series([1.0 + i * 0.01 for i in range(20)])
        slope = compute_ma_slope_pct(s, window=20)
        assert slope > 0.0

    def test_falling_ma_negative_slope(self):
        s = pd.Series([1.0 - i * 0.01 for i in range(20)])
        slope = compute_ma_slope_pct(s, window=20)
        assert slope < 0.0

    def test_flat_ma_slope_near_zero(self):
        s = pd.Series([1.0] * 20)
        slope = compute_ma_slope_pct(s, window=20)
        assert slope == pytest.approx(0.0, abs=1e-6)

    def test_window_limits_bars_used(self):
        """window=5 uses only the last 5 bars; early trend is ignored."""
        # First 15 bars rising sharply; last 5 bars flat.
        rising = [1.0 + i * 0.1 for i in range(15)]
        flat   = [rising[-1]] * 5
        s = pd.Series(rising + flat)
        slope = compute_ma_slope_pct(s, window=5)
        assert slope == pytest.approx(0.0, abs=1e-5)

    def test_normalised_by_mean(self):
        """Slope is %/day: scale-invariant across different rclose levels."""
        s1 = pd.Series([1.0 + i * 0.01 for i in range(20)])
        s2 = s1 * 100  # same shape, different level
        slope1 = compute_ma_slope_pct(s1, window=20)
        slope2 = compute_ma_slope_pct(s2, window=20)
        assert slope1 == pytest.approx(slope2, rel=1e-3)

    def test_returns_float(self):
        assert isinstance(compute_ma_slope_pct(pd.Series([1.0] * 10), window=10), float)


# ===========================================================================
# assess_ma_trend → MATrendStrength
# ===========================================================================


class TestAssessMaTrend:

    def test_trending_df_is_trending_true(self):
        """Steadily trending df: is_trending should be True."""
        df = _trending_df(n=80, step=0.002)
        result = assess_ma_trend(df)
        assert result.is_trending is True

    def test_flat_df_is_trending_false(self):
        """Flat df: is_trending should be False."""
        df = _flat_df(n=60)
        result = assess_ma_trend(df)
        assert result.is_trending is False

    def test_returns_ma_trend_strength_type(self):
        df = _trending_df(n=60)
        result = assess_ma_trend(df)
        assert isinstance(result, MATrendStrength)

    def test_all_fields_populated(self):
        """MATrendStrength must have all required fields and they must be finite."""
        df = _trending_df(n=60)
        r = assess_ma_trend(df)
        assert isinstance(r.rsi, float)
        assert isinstance(r.adx, float)
        assert isinstance(r.adx_slope, float)
        assert isinstance(r.ma_gap_pct, float)
        assert isinstance(r.ma_gap_slope, float)
        assert isinstance(r.is_trending, bool)
        assert 0.0 <= r.rsi <= 100.0
        assert r.adx >= 0.0

    def test_trending_rsi_above_50(self):
        """Rising series: RSI should be above 50."""
        df = _trending_df(n=60, step=0.01)
        result = assess_ma_trend(df)
        assert result.rsi > 50.0

    def test_trending_ma_gap_positive(self):
        """In _trending_df, rema_short_50 > rema_long_150 → ma_gap_pct > 0."""
        df = _trending_df(n=60)
        result = assess_ma_trend(df)
        assert result.ma_gap_pct > 0.0

    def test_missing_rclose_raises(self):
        df = _trending_df(n=60).drop(columns=["rclose"])
        with pytest.raises(ValueError, match="rclose"):
            assess_ma_trend(df)

    def test_missing_rhigh_raises(self):
        df = _trending_df(n=60).drop(columns=["rhigh"])
        with pytest.raises(ValueError, match="rhigh"):
            assess_ma_trend(df)

    def test_missing_ma_level_raises(self):
        df = _trending_df(n=60).drop(columns=["rema_short_50"])
        with pytest.raises(ValueError, match="rema_short_50"):
            assess_ma_trend(df)

    def test_is_trending_requires_adx_above_25_and_not_declining(self):
        """is_trending = adx > 25 AND adx_slope >= 0. Verify the rule directly."""
        df = _trending_df(n=80, step=0.002)
        result = assess_ma_trend(df)
        expected = result.adx > 25.0 and result.adx_slope >= 0.0
        assert result.is_trending is expected
