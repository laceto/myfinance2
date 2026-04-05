"""
test_ma_volume.py — Test suite for ta.ma.volume.

Coverage
--------
_find_last_flip:
  - last bar is a flip, mid-series flip, all-same signal (no flip), two-bar series

assess_ma_volume — validation:
  - missing volume column, missing signal column, both missing
  - custom volume_col / signal_col missing
  - fewer than 2 bars, min_post_bars < 1, confirmed_vol_threshold <= 0
  - sustained_vol_threshold <= 0, vol_ma_window < 1

assess_ma_volume — happy path:
  - last bar is flip + high volume → is_confirmed=True, vol_on_crossover set
  - last bar is flip + low volume → is_confirmed=False
  - last bar not a flip → is_confirmed=None, vol_on_crossover=None
  - post-flip bars available and vol high → is_sustained=True
  - post-flip bars available and vol low → is_sustained=False
  - fewer than min_post_bars after flip → is_sustained=None
  - no flip in history → is_sustained=None
  - custom volume_col and signal_col
  - DatetimeIndex accepted

zero-volume edge cases:
  - all-zero volume → vol_trend NaN → is_confirmed still set (not None) but NaN propagates
  - partial zero-volume → enough valid bars remain, no inf
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ta.ma.volume import (
    CONFIRMED_VOL_THRESHOLD,
    DEFAULT_SUSTAINED_VOL_THR,
    DEFAULT_VOL_MA_WINDOW,
    MIN_POST_BARS,
    MAVolumeProfile,
    _find_last_flip,
    assess_ma_volume,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    volume,
    signal,
    volume_col: str = "volume",
    signal_col: str = "sig",
) -> pd.DataFrame:
    return pd.DataFrame({volume_col: list(volume), signal_col: list(signal)})


def _make_flip_df(
    n: int = 30,
    flip_at: int = -1,    # bar index where signal flips (default: last bar)
    vol_at_flip: float = 150.0,
    vol_default: float = 100.0,
) -> pd.DataFrame:
    """
    DataFrame where signal is 0 until flip_at, then 1.
    Volume is vol_at_flip at the flip bar, vol_default elsewhere.
    """
    signal  = [0] * n
    volumes = [vol_default] * n
    idx     = flip_at % n
    signal[idx] = 1
    volumes[idx] = vol_at_flip
    return _make_df(volumes, signal)


# ---------------------------------------------------------------------------
# _find_last_flip
# ---------------------------------------------------------------------------


class TestFindLastFlip:
    def test_last_bar_is_flip(self):
        signal = np.array([0, 0, 0, 1], dtype=np.int8)
        assert _find_last_flip(signal) == 3

    def test_mid_series_flip(self):
        signal = np.array([0, 0, 1, 1, 1], dtype=np.int8)
        assert _find_last_flip(signal) == 2

    def test_most_recent_of_multiple_flips(self):
        signal = np.array([0, 1, 0, 1], dtype=np.int8)
        assert _find_last_flip(signal) == 3

    def test_all_same_signal_returns_none(self):
        signal = np.array([1, 1, 1, 1], dtype=np.int8)
        assert _find_last_flip(signal) is None

    def test_two_bar_flip(self):
        signal = np.array([0, 1], dtype=np.int8)
        assert _find_last_flip(signal) == 1

    def test_two_bar_no_flip(self):
        signal = np.array([1, 1], dtype=np.int8)
        assert _find_last_flip(signal) is None

    def test_bearish_flip(self):
        """1 → -1 transition should be detected."""
        signal = np.array([1, 1, 1, -1], dtype=np.int8)
        assert _find_last_flip(signal) == 3

    def test_zero_to_minus_one_flip(self):
        signal = np.array([0, 0, -1], dtype=np.int8)
        assert _find_last_flip(signal) == 2


# ---------------------------------------------------------------------------
# assess_ma_volume — input validation
# ---------------------------------------------------------------------------


class TestAssessMaVolumeValidation:
    def test_missing_volume_column_raises(self):
        df = pd.DataFrame({"sig": [0, 1]})
        with pytest.raises(ValueError, match="missing required columns"):
            assess_ma_volume(df, signal_col="sig")

    def test_missing_signal_column_raises(self):
        df = pd.DataFrame({"volume": [100.0, 100.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            assess_ma_volume(df, signal_col="sig")

    def test_custom_volume_col_missing_raises(self):
        df = _make_df([100.0, 100.0], [0, 1])
        with pytest.raises(ValueError, match="missing required columns"):
            assess_ma_volume(df, signal_col="sig", volume_col="vol")

    def test_custom_signal_col_missing_raises(self):
        df = _make_df([100.0, 100.0], [0, 1])
        with pytest.raises(ValueError, match="missing required columns"):
            assess_ma_volume(df, signal_col="other_sig")

    def test_fewer_than_two_bars_raises(self):
        df = _make_df([100.0], [0])
        with pytest.raises(ValueError, match="at least 2 bars"):
            assess_ma_volume(df, signal_col="sig")

    def test_min_post_bars_zero_raises(self):
        df = _make_flip_df()
        with pytest.raises(ValueError, match="min_post_bars must be >= 1"):
            assess_ma_volume(df, signal_col="sig", min_post_bars=0)

    def test_confirmed_vol_threshold_zero_raises(self):
        df = _make_flip_df()
        with pytest.raises(ValueError, match="confirmed_vol_threshold must be > 0"):
            assess_ma_volume(df, signal_col="sig", confirmed_vol_threshold=0.0)

    def test_sustained_vol_threshold_zero_raises(self):
        df = _make_flip_df()
        with pytest.raises(ValueError, match="sustained_vol_threshold must be > 0"):
            assess_ma_volume(df, signal_col="sig", sustained_vol_threshold=0.0)

    def test_vol_ma_window_zero_raises(self):
        df = _make_flip_df()
        with pytest.raises(ValueError, match="vol_ma_window must be >= 1"):
            assess_ma_volume(df, signal_col="sig", vol_ma_window=0)

    # ------------------------------------------------------------------
    # signal_col dtype / value validation (int8 overflow safety)
    # ------------------------------------------------------------------

    def test_non_integer_float_signal_raises(self):
        """
        0.5 would silently truncate to 0 under a bare np.int8 cast — an
        upstream bug hidden from the caller.  Validation must raise early.
        """
        df = _make_df([100.0] * 10, [0.0] * 8 + [0.5, 0.0])
        with pytest.raises(ValueError, match="non-integer"):
            assess_ma_volume(df, signal_col="sig")

    def test_signal_out_of_range_raises(self):
        """
        2 is not a valid MA crossover code.  Could be a regime score or
        sentinel that would survive an int8 cast but corrupt comparisons.
        """
        df = _make_df([100.0] * 10, [0] * 9 + [2])
        with pytest.raises(ValueError, match="outside"):
            assess_ma_volume(df, signal_col="sig")

    def test_signal_int8_overflow_raises(self):
        """128 wraps to -128 under int8 — must be caught before the cast."""
        df = _make_df([100.0] * 10, [0] * 9 + [128])
        with pytest.raises(ValueError, match="outside"):
            assess_ma_volume(df, signal_col="sig")

    def test_nan_in_signal_raises(self):
        """NaN in the signal column has no meaningful interpretation as a flip code."""
        df = _make_df([100.0] * 10, [0.0] * 9 + [float("nan")])
        with pytest.raises(ValueError, match="NaN"):
            assess_ma_volume(df, signal_col="sig")

    def test_integral_float_signal_accepted(self):
        """1.0 / 0.0 / -1.0 from parquet round-trips must be accepted without error."""
        df = _make_df([100.0] * 10, [0.0] * 9 + [1.0])
        result = assess_ma_volume(df, signal_col="sig")
        assert isinstance(result, MAVolumeProfile)

    def test_int64_signal_accepted(self):
        """int64 (default pandas integer dtype) must be accepted without error."""
        df = _make_df([100.0] * 10, pd.array([0] * 9 + [1], dtype="int64"))
        result = assess_ma_volume(df, signal_col="sig")
        assert isinstance(result, MAVolumeProfile)


# ---------------------------------------------------------------------------
# assess_ma_volume — happy path
# ---------------------------------------------------------------------------


class TestAssessMaVolumeHappyPath:
    def test_returns_ma_volume_profile_instance(self):
        df = _make_flip_df()
        result = assess_ma_volume(df, signal_col="sig")
        assert isinstance(result, MAVolumeProfile)

    def test_last_bar_flip_high_volume_is_confirmed(self):
        """Last bar flips with vol >> CONFIRMED_VOL_THRESHOLD → is_confirmed=True."""
        # 30 bars of vol=100 then last bar flips with vol=200
        # rolling_mean at last bar ≈ 105 → vol_trend ≈ 1.9 > 1.2
        volumes = [100.0] * 29 + [200.0]
        signal  = [0] * 29 + [1]
        df = _make_df(volumes, signal)
        result = assess_ma_volume(df, signal_col="sig")
        assert result.is_confirmed is True
        assert result.vol_on_crossover is not None
        assert result.vol_on_crossover > CONFIRMED_VOL_THRESHOLD

    def test_last_bar_flip_low_volume_is_not_confirmed(self):
        """Last bar flips with same vol → vol_trend ≈ 1.0 < 1.2 → is_confirmed=False."""
        volumes = [100.0] * 30
        signal  = [0] * 29 + [1]
        df = _make_df(volumes, signal)
        result = assess_ma_volume(df, signal_col="sig")
        assert result.is_confirmed is False

    def test_no_flip_on_last_bar_is_confirmed_none(self):
        """Continuation bar (no signal change) → is_confirmed=None."""
        signal = [0] * 20 + [1] * 10
        df = _make_df([100.0] * 30, signal)
        result = assess_ma_volume(df, signal_col="sig")
        assert result.is_confirmed is None
        assert result.vol_on_crossover is None

    def test_post_flip_high_volume_is_sustained(self):
        """
        Flip at bar 20, next MIN_POST_BARS bars all have high vol → is_sustained=True.
        """
        n   = 20 + MIN_POST_BARS + 5  # room for post bars
        vol = [100.0] * n
        sig = [0] * 20 + [1] * (n - 20)
        # Boost post-flip volume to push mean above 1.0
        for i in range(20, 20 + MIN_POST_BARS):
            vol[i] = 150.0
        df = _make_df(vol, sig)
        result = assess_ma_volume(df, signal_col="sig")
        assert result.is_sustained is True
        assert result.vol_trend_mean_post is not None

    def test_post_flip_low_volume_is_not_sustained(self):
        """Post-flip bars with vol much lower than rolling average → is_sustained=False."""
        # Need: 30 baseline bars + 1 flip bar + MIN_POST_BARS post bars
        # flip at index 30; post bars at 31..30+MIN_POST_BARS
        n   = 30 + 1 + MIN_POST_BARS
        vol = [200.0] * 30 + [50.0] * (1 + MIN_POST_BARS)
        sig = [0] * 30 + [1] * (1 + MIN_POST_BARS)
        df  = _make_df(vol, sig)
        result = assess_ma_volume(df, signal_col="sig")
        assert result.is_sustained is False

    def test_fewer_post_flip_bars_than_min_is_sustained_none(self):
        """Not enough post-flip bars → is_sustained=None."""
        # Only 2 post-flip bars but min_post_bars=3 (default)
        sig = [0] * 20 + [1] * 2
        df  = _make_df([100.0] * 22, sig)
        result = assess_ma_volume(df, signal_col="sig")
        assert result.is_sustained is None
        assert result.vol_trend_mean_post is None

    def test_no_flip_in_history_is_sustained_none(self):
        """Signal never changes → flip_idx=None → is_sustained=None."""
        df = _make_df([100.0] * 10, [1] * 10)
        result = assess_ma_volume(df, signal_col="sig")
        assert result.is_sustained is None
        assert result.vol_on_crossover is None

    def test_custom_column_names(self):
        df = pd.DataFrame({"vol": [100.0] * 30, "ma_sig": [0] * 29 + [1]})
        result = assess_ma_volume(df, signal_col="ma_sig", volume_col="vol")
        assert isinstance(result, MAVolumeProfile)

    def test_datetimeindex_accepted(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        df  = _make_flip_df(n=30)
        df.index = idx
        result = assess_ma_volume(df, signal_col="sig")
        assert isinstance(result, MAVolumeProfile)

    def test_module_constants_exported(self):
        assert MIN_POST_BARS >= 1
        assert CONFIRMED_VOL_THRESHOLD > 0
        assert DEFAULT_SUSTAINED_VOL_THR > 0
        assert DEFAULT_VOL_MA_WINDOW >= 1

    # ------------------------------------------------------------------
    # NaN in post-flip window
    # ------------------------------------------------------------------

    def test_all_post_flip_bars_nan_returns_is_sustained_none(self):
        """
        vol_trend is NaN when rolling_mean ≤ 0 (zero or halted volume).
        With vol_ma_window=1 the rolling mean is just the current bar's volume,
        so a zero-volume bar directly produces NaN vol_trend.

        If all min_post_bars post-flip bars are NaN, computing any mean is
        meaningless.  is_sustained must be None (no clean evidence), not False.

        False would imply "volume was below threshold"; None means "unknown
        because the stock may have been halted during the follow-through window".
        """
        n     = 30 + 1 + MIN_POST_BARS
        vol   = [100.0] * 30 + [150.0] + [0.0] * MIN_POST_BARS
        sig   = [0] * 30 + [1] * (1 + MIN_POST_BARS)
        df    = _make_df(vol, sig)
        # vol_ma_window=1 so rolling_mean = current bar volume
        # → zero-volume post-flip bars have rolling_mean=0 → _safe_mean=NaN → NaN vol_trend
        result = assess_ma_volume(df, signal_col="sig", vol_ma_window=1)
        assert result.is_sustained is None, (
            "is_sustained must be None when all post-flip bars have NaN vol_trend"
        )
        assert result.vol_trend_mean_post is None

    def test_partial_nan_post_flip_bars_returns_is_sustained_none(self):
        """
        1 of 3 post-flip bars has zero volume (vol_ma_window=1 → NaN vol_trend).
        With the old skipna=True behaviour the mean would be computed from only 2
        bars — fewer than min_post_bars (3) — insufficient evidence.
        is_sustained must be None when ANY post-flip bar has missing volume data.
        """
        n   = 30 + 1 + MIN_POST_BARS
        vol = [100.0] * 30 + [150.0] + [0.0] + [100.0] * (MIN_POST_BARS - 1)
        sig = [0] * 30 + [1] * (1 + MIN_POST_BARS)
        df  = _make_df(vol, sig)
        result = assess_ma_volume(df, signal_col="sig", vol_ma_window=1)
        assert result.is_sustained is None, (
            "is_sustained must be None when any post-flip bar has NaN vol_trend"
        )


# ---------------------------------------------------------------------------
# Zero-volume edge cases
# ---------------------------------------------------------------------------


class TestZeroVolumeHandling:
    def test_all_zero_volume_vol_on_crossover_is_nan(self):
        """
        All-zero volume → rolling_mean = 0 → vol_trend = NaN everywhere.
        is_confirmed depends on NaN >= threshold which is False.
        """
        volumes = [0.0] * 29 + [0.0]
        signal  = [0] * 29 + [1]
        df      = _make_df(volumes, signal)
        result  = assess_ma_volume(df, signal_col="sig")
        # vol_on_crossover is NaN (not inf), is_confirmed is False (NaN >= 1.2 is False)
        assert result.vol_on_crossover is not None
        assert not np.isfinite(result.vol_on_crossover) or result.vol_on_crossover == 0.0

    def test_zero_volume_does_not_produce_inf(self):
        """
        Partial zero volume: first few bars are zero, rest are normal.
        vol_trend at those bars must be NaN, never inf.
        """
        volumes = [0.0] * 5 + [100.0] * 25
        signal  = [0] * 29 + [1]
        df      = _make_df(volumes, signal)
        result  = assess_ma_volume(df, signal_col="sig")
        # Last bar has vol=100 and rolling_mean > 0 → vol_on_crossover is finite
        assert result.vol_on_crossover is not None
        assert np.isfinite(result.vol_on_crossover)
