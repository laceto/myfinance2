"""
test_volume.py — Test suite for ta.breakout.volume.

Coverage
--------
_find_zero_run:
  - all zeros, trailing non-zero, mid-series zero-run, single zero, no zeros, breakout flip bar

assess_volume_profile — validation:
  - missing volume column, missing signal column, custom col names missing
  - window_bars < 1, vol_ma_window < 1, quiet_threshold <= 0, breakout_vol_threshold <= 0
  - no consolidation (all rbo != 0), zero-run shorter than min_vol_bars

assess_volume_profile — happy path:
  - returns VolumeProfile instance
  - quiet consolidation (is_quiet=True), noisy consolidation (is_quiet=False)
  - breakout confirmed, breakout not confirmed, not a flip bar, still in consolidation
  - vol_trend_slope_r2 in [0, 1], declining volume, custom column names
  - window_bars caps zero-run, custom min_vol_bars allows short history

zero-volume handling:
  - all-zero volume → NaN vol_trend → ValueError
  - partial zero volume → enough bars remain → succeeds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ta.breakout.volume import (
    DEFAULT_BREAKOUT_VOL_THR,
    DEFAULT_QUIET_THRESHOLD,
    DEFAULT_VOL_MA_WINDOW,
    DEFAULT_WINDOW_BARS,
    MIN_VOL_BARS,
    VolumeProfile,
    _find_zero_run,
    assess_volume_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    volume,
    rbo,
    volume_col: str = "volume",
    signal_col: str = "rbo_20",
) -> pd.DataFrame:
    """Build a minimal DataFrame from plain lists."""
    return pd.DataFrame({volume_col: volume, signal_col: rbo})


def _consolidation_df(
    n_bars: int = 30,
    vol_level: float = 80.0,
    warmup_vol: float = 200.0,
    warmup: int = 20,
) -> pd.DataFrame:
    """
    DataFrame where the first `warmup` bars have high volume (establishes the rolling
    baseline) and the remaining bars are in consolidation (rbo=0) with lower volume.
    vol_trend during consolidation will be < 1.0 → is_quiet=True.
    """
    volumes = [warmup_vol] * warmup + [vol_level] * (n_bars - warmup)
    rbo = [0] * n_bars
    return _make_df(volumes, rbo)


# ---------------------------------------------------------------------------
# _find_zero_run
# ---------------------------------------------------------------------------


class TestFindZeroRun:
    def test_all_zeros(self):
        rbo = np.array([0, 0, 0, 0, 0], dtype=np.int8)
        start, end = _find_zero_run(rbo)
        assert start == 0
        assert end == 4

    def test_trailing_nonzero(self):
        """Zero-run must exclude trailing non-zero bars."""
        rbo = np.array([0, 0, 0, 1, 1], dtype=np.int8)
        start, end = _find_zero_run(rbo)
        assert start == 0
        assert end == 2

    def test_most_recent_zero_run_selected(self):
        """When multiple zero-runs exist, the most recent one is chosen."""
        # Zeros at [0,1], then non-zero at [2], then zeros at [3,4,5], then non-zero at [6,7]
        rbo = np.array([0, 0, 1, 0, 0, 0, 1, 1], dtype=np.int8)
        start, end = _find_zero_run(rbo)
        assert start == 3
        assert end == 5

    def test_single_zero_at_end(self):
        rbo = np.array([1, 1, 0], dtype=np.int8)
        start, end = _find_zero_run(rbo)
        assert start == 2
        assert end == 2

    def test_no_zeros_raises(self):
        rbo = np.array([1, -1, 1, 1], dtype=np.int8)
        with pytest.raises(ValueError, match="No consolidation window"):
            _find_zero_run(rbo)

    def test_breakout_flip_bar_excludes_last_bar(self):
        """When last bar is non-zero (breakout flip), zero-run ends at bar before it."""
        rbo = np.array([0, 0, 0, 0, 1], dtype=np.int8)
        start, end = _find_zero_run(rbo)
        assert end == 3  # not 4 (the breakout bar)
        assert start == 0


# ---------------------------------------------------------------------------
# assess_volume_profile — input validation
# ---------------------------------------------------------------------------


class TestAssessVolumeProfileValidation:
    def test_missing_volume_column_raises(self):
        df = pd.DataFrame({"rbo_20": [0] * 10})
        with pytest.raises(ValueError, match="missing required columns"):
            assess_volume_profile(df)

    def test_missing_signal_column_raises(self):
        df = pd.DataFrame({"volume": [100.0] * 10})
        with pytest.raises(ValueError, match="missing required columns"):
            assess_volume_profile(df)

    def test_custom_col_names_not_present_raises(self):
        df = _make_df([100.0] * 10, [0] * 10)  # has "volume" and "rbo_20"
        with pytest.raises(ValueError, match="missing required columns"):
            assess_volume_profile(df, volume_col="vol", signal_col="sig")

    def test_window_bars_zero_raises(self):
        with pytest.raises(ValueError, match="window_bars must be >= 1"):
            assess_volume_profile(_consolidation_df(), window_bars=0)

    def test_window_bars_negative_raises(self):
        with pytest.raises(ValueError, match="window_bars must be >= 1"):
            assess_volume_profile(_consolidation_df(), window_bars=-10)

    def test_vol_ma_window_zero_raises(self):
        with pytest.raises(ValueError, match="vol_ma_window must be >= 1"):
            assess_volume_profile(_consolidation_df(), vol_ma_window=0)

    def test_quiet_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="quiet_threshold must be > 0"):
            assess_volume_profile(_consolidation_df(), quiet_threshold=0.0)

    def test_quiet_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="quiet_threshold must be > 0"):
            assess_volume_profile(_consolidation_df(), quiet_threshold=-1.0)

    def test_breakout_vol_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="breakout_vol_threshold must be > 0"):
            assess_volume_profile(_consolidation_df(), breakout_vol_threshold=0.0)

    def test_no_consolidation_history_raises(self):
        """All rbo != 0 → no zero-run → ValueError."""
        df = _make_df([100.0] * 10, [1] * 10)
        with pytest.raises(ValueError, match="No consolidation window"):
            assess_volume_profile(df)

    def test_zero_run_too_short_raises(self):
        """Only 4 bars of rbo==0 < default MIN_VOL_BARS=5 → ValueError."""
        rbo = [1] * 6 + [0] * 4
        df = _make_df([100.0] * 10, rbo)
        with pytest.raises(ValueError, match="at least"):
            assess_volume_profile(df)

    # ------------------------------------------------------------------
    # signal_col dtype / value validation (int8 cast safety)
    # ------------------------------------------------------------------

    def test_non_integer_float_signal_raises(self):
        """
        A signal value of 0.5 is not a valid breakout code — it would silently
        truncate to 0 under a bare .astype(np.int8) cast, hiding the upstream bug.
        Validation must raise immediately with a message identifying the bad values.
        """
        rbo = [0.0] * 8 + [0.5, 0.0]          # 0.5 is not integral
        df = _make_df([100.0] * 10, rbo)
        with pytest.raises(ValueError, match="non-integer"):
            assess_volume_profile(df)

    def test_signal_out_of_range_raises(self):
        """
        A signal value of 2 rounds to an integer but is outside {-1, 0, 1}.
        Could be an error sentinel (e.g. 99) or a different signal convention.
        Must raise clearly rather than wrap via int8 overflow.
        """
        rbo = [0] * 9 + [2]                    # 2 is integral but not in {-1, 0, 1}
        df = _make_df([100.0] * 10, rbo)
        with pytest.raises(ValueError, match="outside"):
            assess_volume_profile(df)

    def test_signal_overflow_value_raises(self):
        """
        128 overflows int8 to -128, silently turning an error sentinel into a
        bearish breakout signal.  Must be caught before the cast.
        """
        rbo = [0] * 9 + [128]
        df = _make_df([100.0] * 10, rbo)
        with pytest.raises(ValueError, match="outside"):
            assess_volume_profile(df)

    def test_nan_in_signal_column_raises(self):
        """
        NaN in the signal column cannot be cast to int8 deterministically.
        Must raise with a clear message.
        """
        rbo = [0.0] * 9 + [float("nan")]
        df = _make_df([100.0] * 10, rbo)
        with pytest.raises(ValueError, match="NaN"):
            assess_volume_profile(df)

    def test_integral_float_signal_accepted(self):
        """
        1.0 / 0.0 / -1.0 are the valid signal values expressed as float64
        (common after parquet round-trips).  Must not raise.
        """
        rbo = [0.0] * 9 + [1.0]               # last bar is a breakout flip as float
        df = _make_df([100.0] * 10, rbo)
        result = assess_volume_profile(df)     # must not raise
        assert isinstance(result, VolumeProfile)

    def test_int64_signal_accepted(self):
        """int64 dtype (default pandas integer) must be accepted without error."""
        rbo = pd.array([0] * 9 + [1], dtype="int64")
        df = _make_df([100.0] * 10, rbo)
        result = assess_volume_profile(df)
        assert isinstance(result, VolumeProfile)


# ---------------------------------------------------------------------------
# assess_volume_profile — happy path
# ---------------------------------------------------------------------------


class TestAssessVolumeProfileHappyPath:
    def test_returns_volume_profile_instance(self):
        result = assess_volume_profile(_consolidation_df())
        assert isinstance(result, VolumeProfile)

    def test_quiet_consolidation_detected(self):
        """Low volume during consolidation (vs high warmup) → is_quiet=True."""
        df = _consolidation_df(n_bars=30, vol_level=80.0, warmup_vol=200.0, warmup=20)
        result = assess_volume_profile(df)
        # vol_trend during consolidation << 1.0 because rolling_mean is anchored on 200
        assert result.is_quiet is True
        assert result.vol_trend_mean < DEFAULT_QUIET_THRESHOLD

    def test_noisy_consolidation_not_quiet(self):
        """Constant volume → vol_trend_mean ≈ 1.0 → is_quiet=False (not strictly below 1.0)."""
        df = _make_df([100.0] * 30, [0] * 30)
        result = assess_volume_profile(df)
        assert result.vol_trend_mean == pytest.approx(1.0, abs=0.01)
        assert result.is_quiet is False

    def test_breakout_confirmed_when_volume_spikes(self):
        """Flip bar with 2× vol → vol_trend_now >> 1.2 → breakout_confirmed=True."""
        volumes = [100.0] * 29 + [200.0]  # last bar spikes
        rbo     = [0]       * 29 + [1]
        df = _make_df(volumes, rbo)
        result = assess_volume_profile(df, breakout_vol_threshold=1.2)
        # rolling_mean at last bar ≈ 105 (dominated by 100s), vol_trend ≈ 1.9 → True
        assert result.breakout_confirmed is True
        assert result.vol_trend_now > 1.2

    def test_breakout_not_confirmed_when_volume_flat(self):
        """Flip bar with same vol as consolidation → vol_trend_now ≈ 1.0 < 1.2 → False."""
        volumes = [100.0] * 30
        rbo     = [0]     * 29 + [1]
        df = _make_df(volumes, rbo)
        result = assess_volume_profile(df, breakout_vol_threshold=1.2)
        assert result.breakout_confirmed is False

    def test_mid_trend_returns_none_for_breakout_confirmed(self):
        """rbo[-1] != 0 AND rbo[-2] != 0 → not a flip bar → breakout_confirmed=None."""
        rbo = [0] * 20 + [1] * 10
        df = _make_df([100.0] * 30, rbo)
        result = assess_volume_profile(df)
        assert result.breakout_confirmed is None

    def test_in_consolidation_returns_none_for_breakout_confirmed(self):
        """Still in range (last bar rbo==0) → breakout_confirmed=None."""
        result = assess_volume_profile(_consolidation_df())
        assert result.breakout_confirmed is None

    def test_r2_field_is_in_unit_interval(self):
        """vol_trend_slope_r2 must always be in [0, 1]."""
        result = assess_volume_profile(_consolidation_df())
        assert 0.0 <= result.vol_trend_slope_r2 <= 1.0

    def test_perfect_linear_decline_has_high_r2(self):
        """Perfectly linearly declining vol_trend → R² should be close to 1."""
        # Construct: 20 warmup bars then 20 bars of strictly declining volume
        warmup   = [200.0] * 20
        # Make decline steep enough that vol_trend within the window is near-linear
        decline  = [200.0 - i * 8 for i in range(20)]  # 200, 192, ..., 48
        volumes  = warmup + decline
        rbo      = [0] * 40
        df       = _make_df(volumes, rbo)
        result   = assess_volume_profile(df, window_bars=20)
        # Only the declining 20 bars are in the capped window → near-linear trend
        assert result.is_declining is True
        assert result.vol_trend_slope_r2 > 0.7, (
            f"Expected R² > 0.7 for a near-linear decline, got {result.vol_trend_slope_r2}"
        )

    def test_constant_series_has_zero_r2(self):
        """Flat vol_trend → slope = 0, R² = 0."""
        df = _make_df([100.0] * 30, [0] * 30)
        result = assess_volume_profile(df)
        assert result.vol_trend_slope == pytest.approx(0.0, abs=1e-6)
        assert result.vol_trend_slope_r2 == pytest.approx(0.0, abs=1e-6)

    def test_declining_volume_sets_is_declining(self):
        """Volume declining during consolidation → is_declining=True, slope < 0."""
        warmup   = [200.0] * 20
        declining = [200.0 - i * 10 for i in range(10)]  # 200, 190, ..., 110
        df       = _make_df(warmup + declining, [0] * 30)
        result   = assess_volume_profile(df)
        assert result.is_declining is True
        assert result.vol_trend_slope < 0.0

    def test_custom_column_names_accepted(self):
        """volume_col and signal_col parameters override the defaults."""
        df = pd.DataFrame({"vol": [100.0] * 30, "sig": [0] * 30})
        result = assess_volume_profile(df, volume_col="vol", signal_col="sig")
        assert isinstance(result, VolumeProfile)

    def test_window_bars_caps_zero_run(self):
        """window_bars limits the consolidation window; function still succeeds."""
        df = _make_df([100.0] * 60, [0] * 60)
        result = assess_volume_profile(df, window_bars=10)
        assert isinstance(result, VolumeProfile)

    def test_custom_min_vol_bars_allows_short_history(self):
        """min_vol_bars=3 allows consolidation windows shorter than the default 5."""
        rbo = [1] * 16 + [0] * 4  # only 4 bars in consolidation
        df  = _make_df([100.0] * 20, rbo)
        result = assess_volume_profile(df, min_vol_bars=3)
        assert isinstance(result, VolumeProfile)

    def test_vol_trend_now_reflects_last_bar(self):
        """vol_trend_now = vol_trend at the last bar regardless of rbo state."""
        volumes = [100.0] * 29 + [150.0]
        rbo     = [0] * 30
        df      = _make_df(volumes, rbo)
        result  = assess_volume_profile(df)
        # rolling_mean at last bar ≈ 100 → vol_trend ≈ 1.5
        assert result.vol_trend_now > 1.0

    def test_datetimeindex_preserved_internally(self):
        """Function must not crash when df has a DatetimeIndex."""
        idx     = pd.date_range("2024-01-01", periods=30, freq="D")
        df      = pd.DataFrame({"volume": [100.0] * 30, "rbo_20": [0] * 30}, index=idx)
        result  = assess_volume_profile(df)
        assert isinstance(result, VolumeProfile)


# ---------------------------------------------------------------------------
# Zero-volume edge cases
# ---------------------------------------------------------------------------


class TestZeroVolumeHandling:
    def test_all_zero_volume_raises(self):
        """
        All-zero volume → rolling_mean = 0 → vol_trend = NaN for every bar.
        clean_vt will be empty → ValueError (fewer than MIN_VOL_BARS bars).
        """
        df = _make_df([0.0] * 30, [0] * 30)
        with pytest.raises(ValueError, match="at least"):
            assess_volume_profile(df)

    def test_partial_zero_volume_ok_when_enough_bars_remain(self):
        """
        First 3 bars have zero volume (→ NaN vol_trend).
        Bars 3–29 have positive volume → 27 clean bars ≥ MIN_VOL_BARS=5.
        Should succeed.
        """
        volumes = [0.0] * 3 + [100.0] * 27
        rbo     = [0] * 30
        df      = _make_df(volumes, rbo)
        result  = assess_volume_profile(df)
        assert isinstance(result, VolumeProfile)

    def test_zero_volume_produces_nan_not_inf(self):
        """
        vol_trend at a zero-volume bar where rolling_mean is also zero must be NaN,
        not inf or -inf. The function may raise ValueError due to too many NaNs,
        but must never silently return inf.
        """
        volumes = [0.0] * 10 + [100.0] * 20
        rbo     = [0] * 30
        df      = _make_df(volumes, rbo)
        # Even if we call the internal vol_trend computation, no inf should appear.
        # We verify indirectly: if enough clean bars exist, the result is finite.
        result = assess_volume_profile(df)
        assert np.isfinite(result.vol_trend_mean)
        assert np.isfinite(result.vol_trend_now)
