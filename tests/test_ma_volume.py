"""
Unit tests for ta.ma.volume — MA crossover volume behaviour primitive.

TDD Red → Green → Refactor cycle:
  All tests written BEFORE implementation exists.

Unlike breakout volume (which checks for QUIET consolidation + spike at flip),
MA crossover volume checks for EXPANDING volume ON the crossover flip bar and
SUSTAINED above-average volume after the flip.

Key invariants:
  - is_confirmed: True only when signal flipped on the last bar AND vol_trend >= 1.2
  - is_confirmed: False when signal flipped but vol_trend < 1.2
  - is_confirmed: None when signal did NOT flip on the last bar
  - is_sustained: True when mean vol_trend over post-flip bars >= 1.0
  - is_sustained: None when there are not enough post-flip bars (< MIN_POST_BARS)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ta.ma.volume import MAVolumeProfile, assess_ma_volume, MIN_POST_BARS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WARMUP_BARS = 30


def _make_df(
    signal: list[float],
    vol: list[float],
    warmup_bars: int = WARMUP_BARS,
    warmup_vol: float = 100.0,
) -> pd.DataFrame:
    """
    Build a minimal DataFrame for testing.

    Prepends `warmup_bars` rows (signal=0, volume=warmup_vol) so that
    the rolling-20 vol_trend mean is stable before the test window begins.

    Args:
        signal: Signal column values (e.g. rema_50100). Appended after warmup.
        vol:    Volume values for the test window. Same length as signal.
        warmup_bars: Number of pre-history rows with vol=warmup_vol, signal=0.
        warmup_vol:  Volume during warmup (vol_trend baseline = 1.0).
    """
    assert len(signal) == len(vol), "signal and vol must have the same length"
    all_signal = [0.0] * warmup_bars + list(signal)
    all_vol    = [warmup_vol] * warmup_bars + list(vol)
    return pd.DataFrame({
        "rema_50100": all_signal,
        "volume":     all_vol,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAssessMaVolume:

    # --- is_confirmed ---

    def test_flip_high_volume_confirmed_true(self):
        """Signal flips 0→+1 on last bar, vol_trend >= 1.2 → is_confirmed=True."""
        # Warmup at 100, then last bar at 200 → vol_trend = 2.0
        df = _make_df(signal=[0] * 9 + [1], vol=[100.0] * 9 + [200.0])
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.is_confirmed is True

    def test_flip_low_volume_confirmed_false(self):
        """Signal flips 0→+1 on last bar, vol_trend < 1.2 → is_confirmed=False."""
        df = _make_df(signal=[0] * 9 + [1], vol=[100.0] * 9 + [100.0])
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.is_confirmed is False

    def test_no_flip_confirmed_none(self):
        """Signal unchanged on last bar → is_confirmed=None (not a crossover bar)."""
        df = _make_df(signal=[1] * 10, vol=[100.0] * 10)
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.is_confirmed is None

    def test_mid_trend_continuation_confirmed_none(self):
        """Signal held +1 for multiple bars: last bar is a continuation, not a flip."""
        df = _make_df(signal=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1], vol=[100.0] * 10)
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.is_confirmed is None

    def test_bearish_flip_high_volume_confirmed_true(self):
        """Bearish crossover (0 → -1) with high volume → is_confirmed=True."""
        df = _make_df(signal=[0] * 9 + [-1], vol=[100.0] * 9 + [200.0])
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.is_confirmed is True

    def test_direction_reversal_flip_confirmed(self):
        """Signal reverses from +1 to -1: that counts as a flip → check confirmation."""
        df = _make_df(signal=[0, 0, 0, 1, 1, 1, 1, 1, 1, -1], vol=[100.0] * 9 + [200.0])
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.is_confirmed is True

    # --- vol_on_crossover ---

    def test_vol_on_crossover_at_flip_bar(self):
        """vol_on_crossover reflects vol_trend at the flip bar."""
        # warmup at 100 → vol_trend = 1.0. Last bar at 180 → vol_trend = 1.8.
        df = _make_df(signal=[0] * 9 + [1], vol=[100.0] * 9 + [180.0])
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.vol_on_crossover == pytest.approx(1.8, rel=0.05)

    def test_vol_on_crossover_none_when_no_flip(self):
        """vol_on_crossover is None when last bar is not a flip."""
        df = _make_df(signal=[1] * 10, vol=[150.0] * 10)
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.vol_on_crossover is None

    # --- is_sustained ---

    def test_is_sustained_true_when_post_flip_vol_above_avg(self):
        """Post-flip vol_trend mean >= 1.0 → is_sustained=True."""
        # 5 bars at signal=0 (vol=100), then 6 bars at signal=+1 (vol=150 → vt=1.5)
        sig = [0] * 5 + [1] * 6
        vol = [100.0] * 5 + [150.0] * 6
        df  = _make_df(signal=sig, vol=vol)
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.is_sustained is True

    def test_is_sustained_false_when_post_flip_vol_below_avg(self):
        """Post-flip vol_trend mean < 1.0 → is_sustained=False."""
        sig = [0] * 5 + [1] * 6
        vol = [100.0] * 5 + [60.0] * 6   # vol_trend ≈ 0.6
        df  = _make_df(signal=sig, vol=vol)
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.is_sustained is False

    def test_is_sustained_none_when_insufficient_post_flip_bars(self):
        """Fewer than MIN_POST_BARS post-flip bars → is_sustained=None."""
        # Only 1 bar after the flip (fewer than MIN_POST_BARS required)
        sig = [0] * 9 + [1]
        vol = [100.0] * 10
        df  = _make_df(signal=sig, vol=vol)
        result = assess_ma_volume(df, signal_col="rema_50100")
        # is_confirmed is set (it's a flip), but is_sustained = None (1 < MIN_POST_BARS)
        assert result.is_sustained is None

    def test_is_sustained_none_when_no_flip_ever(self):
        """Signal always 0: no flip ever found → is_sustained=None."""
        df = _make_df(signal=[0] * 10, vol=[100.0] * 10)
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.is_sustained is None

    # --- vol_trend_mean_post ---

    def test_vol_trend_mean_post_reflects_post_flip_window(self):
        """vol_trend_mean_post is the mean vol_trend over MIN_POST_BARS after the flip."""
        # 5 consolidation bars, then MIN_POST_BARS+1 trend bars at 2x volume
        n_post = MIN_POST_BARS + 1
        sig = [0] * 5 + [1] * n_post
        vol = [100.0] * 5 + [200.0] * n_post
        df  = _make_df(signal=sig, vol=vol)
        result = assess_ma_volume(df, signal_col="rema_50100")
        # vol_trend in post-flip window ≈ 2.0 (200/100-bar rolling average)
        assert result.vol_trend_mean_post is not None
        assert result.vol_trend_mean_post > 1.5

    def test_vol_trend_mean_post_none_when_no_flip(self):
        df = _make_df(signal=[0] * 10, vol=[100.0] * 10)
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert result.vol_trend_mean_post is None

    # --- Return type ---

    def test_returns_ma_volume_profile_type(self):
        df = _make_df(signal=[0] * 9 + [1], vol=[100.0] * 10)
        result = assess_ma_volume(df, signal_col="rema_50100")
        assert isinstance(result, MAVolumeProfile)

    # --- Error handling ---

    def test_missing_volume_column_raises(self):
        df = pd.DataFrame({"rema_50100": [0.0] * 10})
        with pytest.raises(ValueError, match="volume"):
            assess_ma_volume(df, signal_col="rema_50100")

    def test_missing_signal_column_raises(self):
        df = pd.DataFrame({"volume": [100.0] * 10})
        with pytest.raises(ValueError, match="rema_50100"):
            assess_ma_volume(df, signal_col="rema_50100")

    def test_too_few_bars_raises(self):
        """Fewer than 2 bars: cannot detect a flip."""
        df = pd.DataFrame({"rema_50100": [1.0], "volume": [100.0]})
        with pytest.raises(ValueError, match="bars"):
            assess_ma_volume(df, signal_col="rema_50100")
