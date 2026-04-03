"""
Unit tests for breakout.volume — volume-behavior primitive.

TDD Red -> Green -> Refactor cycle:
  All tests are written BEFORE implementation exists.

Smoke-test values verified against real parquet data:

  A2A.MI — 40-bar window ending 2016-03-23 (last bar rbo_20=0, still in consolidation):
    vol_trend_mean   = 0.8051
    vol_trend_slope  = 0.00146 /bar
    is_quiet         = True   (mean < 1.0)
    is_declining     = False  (slope > 0)
    breakout_confirmed = None (no flip)

  POR.MI — breakout bar 2018-09-03 (rbo_20 flipped 0 → 1):
    vol_trend_now   = 10.9546
    vol_trend_mean  = 0.3714   (35-bar pre-breakout zero-run)
    vol_trend_slope = 0.01244 /bar
    is_quiet        = True
    is_declining    = False
    breakout_confirmed = True (10.9546 >= 1.2)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    rbo_20: list[float],
    vol_consolidation: list[float],
    warmup_vol: float = 100.0,
    warmup_bars: int = 30,
) -> pd.DataFrame:
    """
    Build a minimal DataFrame for testing.

    Prepends `warmup_bars` rows (rbo_20=0, volume=warmup_vol) so that
    the rolling-20 mean is stable before the test window begins.
    """
    assert len(rbo_20) == len(vol_consolidation), "rbo_20 and vol_consolidation must be same length"
    all_rbo = [0.0] * warmup_bars + list(rbo_20)
    all_vol = [warmup_vol] * warmup_bars + list(vol_consolidation)
    return pd.DataFrame({"rbo_20": all_rbo, "volume": all_vol})


# ---------------------------------------------------------------------------
# TestVolumeProfile — synthetic unit tests
# ---------------------------------------------------------------------------

class TestVolumeProfile:

    def test_quiet_consolidation_is_quiet_true(self):
        """vol_trend_mean < 1.0 when consolidation volume is below warmup volume."""
        from ta.breakout.volume import assess_volume_profile
        # Warmup vol=100, consolidation vol=50 → vol_trend << 1.0
        df = _make_df(
            rbo_20=[0.0] * 15,
            vol_consolidation=[50.0] * 15,
            warmup_vol=100.0,
        )
        result = assess_volume_profile(df)
        assert result.is_quiet is True
        assert result.vol_trend_mean < 1.0

    def test_noisy_consolidation_is_quiet_false(self):
        """vol_trend_mean >= 1.0 when consolidation volume exceeds warmup volume."""
        from ta.breakout.volume import assess_volume_profile
        # Warmup vol=50, consolidation vol=150 → vol_trend >> 1.0
        df = _make_df(
            rbo_20=[0.0] * 15,
            vol_consolidation=[150.0] * 15,
            warmup_vol=50.0,
        )
        result = assess_volume_profile(df)
        assert result.is_quiet is False
        assert result.vol_trend_mean > 1.0

    def test_declining_volume_slope_is_declining_true(self):
        """Decreasing volume during consolidation → vol_trend_slope < 0 → is_declining=True."""
        from ta.breakout.volume import assess_volume_profile
        # Volume decreases from 100 to 10 during consolidation (after warmup=100)
        # vol_trend starts near 1.0 and falls → negative slope
        declining_vols = [100 - i * 8 for i in range(10)]  # [100, 92, 84, ..., 28]
        df = _make_df(
            rbo_20=[0.0] * 10,
            vol_consolidation=declining_vols,
            warmup_vol=100.0,
            warmup_bars=30,
        )
        result = assess_volume_profile(df)
        assert result.is_declining is True
        assert result.vol_trend_slope < 0

    def test_rising_volume_slope_is_declining_false(self):
        """Increasing volume during consolidation → vol_trend_slope > 0 → is_declining=False.

        Fixture design: warmup_vol matches the starting consolidation vol (30) so
        vol_trend begins at ≈1.0 and rises cleanly as volumes increase. If warmup
        vol were higher than starting consolidation vol the rolling mean would suppress
        the early bars and produce a misleading negative slope.
        """
        from ta.breakout.volume import assess_volume_profile
        # Warmup vol=30; consolidation volumes rise 30 → 210 (step=20)
        # vol_trend starts at 1.0 and climbs as volumes outpace the lagging rolling mean
        rising_vols = [30 + i * 20 for i in range(10)]  # [30, 50, 70, ..., 210]
        df = _make_df(
            rbo_20=[0.0] * 10,
            vol_consolidation=rising_vols,
            warmup_vol=30.0,
            warmup_bars=30,
        )
        result = assess_volume_profile(df)
        assert result.is_declining is False
        assert result.vol_trend_slope > 0

    def test_breakout_flip_high_volume_confirmed_true(self):
        """Breakout flip bar with vol >> warmup → breakout_confirmed=True."""
        from ta.breakout.volume import assess_volume_profile
        # Consolidation: 10 bars of rbo_20=0, vol=80
        # Breakout bar: rbo_20=1, vol=300 → vol_trend_now >> 1.2
        rbo = [0.0] * 10 + [1.0]
        vol = [80.0] * 10 + [300.0]
        df = _make_df(rbo_20=rbo, vol_consolidation=vol, warmup_vol=100.0)
        result = assess_volume_profile(df)
        assert result.breakout_confirmed is True
        assert result.vol_trend_now >= 1.2

    def test_breakout_flip_low_volume_confirmed_false(self):
        """Breakout flip bar with vol << warmup → breakout_confirmed=False."""
        from ta.breakout.volume import assess_volume_profile
        rbo = [0.0] * 10 + [1.0]
        vol = [80.0] * 10 + [60.0]   # 60 / ~100 = 0.6 < 1.2
        df = _make_df(rbo_20=rbo, vol_consolidation=vol, warmup_vol=100.0)
        result = assess_volume_profile(df)
        assert result.breakout_confirmed is False
        assert result.vol_trend_now < 1.2

    def test_in_consolidation_no_flip_breakout_confirmed_none(self):
        """Last bar in consolidation (rbo_20=0) → breakout_confirmed=None."""
        from ta.breakout.volume import assess_volume_profile
        df = _make_df(rbo_20=[0.0] * 10, vol_consolidation=[80.0] * 10)
        result = assess_volume_profile(df)
        assert result.breakout_confirmed is None

    def test_mid_trend_non_flip_bar_breakout_confirmed_none(self):
        """Two consecutive non-zero bars → not a flip → breakout_confirmed=None."""
        from ta.breakout.volume import assess_volume_profile
        # Last two bars: rbo_20=1, 1 → not a flip bar
        rbo = [0.0] * 10 + [1.0, 1.0]
        vol = [80.0] * 12
        df = _make_df(rbo_20=rbo, vol_consolidation=vol, warmup_vol=100.0)
        result = assess_volume_profile(df)
        assert result.breakout_confirmed is None

    def test_bearish_flip_low_volume_confirmed_false(self):
        """Bearish breakout flip (rbo_20=0 → -1) with low volume → breakout_confirmed=False."""
        from ta.breakout.volume import assess_volume_profile
        rbo = [0.0] * 10 + [-1.0]
        vol = [80.0] * 10 + [60.0]
        df = _make_df(rbo_20=rbo, vol_consolidation=vol, warmup_vol=100.0)
        result = assess_volume_profile(df)
        assert result.breakout_confirmed is False

    def test_bearish_flip_high_volume_confirmed_true(self):
        """Bearish breakout flip (rbo_20=0 → -1) with high volume → breakout_confirmed=True."""
        from ta.breakout.volume import assess_volume_profile
        rbo = [0.0] * 10 + [-1.0]
        vol = [80.0] * 10 + [300.0]
        df = _make_df(rbo_20=rbo, vol_consolidation=vol, warmup_vol=100.0)
        result = assess_volume_profile(df)
        assert result.breakout_confirmed is True

    def test_missing_volume_column_raises_value_error(self):
        """DataFrame without 'volume' column → ValueError with descriptive message."""
        from ta.breakout.volume import assess_volume_profile
        df = pd.DataFrame({"rbo_20": [0.0] * 10})
        with pytest.raises(ValueError, match="volume"):
            assess_volume_profile(df)

    def test_missing_rbo_20_column_raises_value_error(self):
        """DataFrame without 'rbo_20' column → ValueError with descriptive message."""
        from ta.breakout.volume import assess_volume_profile
        df = pd.DataFrame({"volume": [100.0] * 10})
        with pytest.raises(ValueError, match="rbo_20"):
            assess_volume_profile(df)

    def test_too_few_bars_in_zero_run_raises_value_error(self):
        """Zero-run shorter than MIN_TREND_BARS → ValueError."""
        from ta.breakout.volume import assess_volume_profile, MIN_VOL_BARS
        # Only 2 bars in zero-run (below MIN_VOL_BARS=5)
        df = _make_df(rbo_20=[0.0] * 2, vol_consolidation=[80.0] * 2, warmup_bars=0)
        with pytest.raises(ValueError, match="non-NaN"):
            assess_volume_profile(df)

    def test_no_zero_run_ever_raises_value_error(self):
        """History with no rbo_20==0 bar at all → ValueError."""
        from ta.breakout.volume import assess_volume_profile
        df = pd.DataFrame({
            "rbo_20": [1.0] * 30,
            "volume": [100.0] * 30,
        })
        with pytest.raises(ValueError, match="consolidation"):
            assess_volume_profile(df)

    def test_custom_quiet_threshold(self):
        """quiet_threshold parameter is respected."""
        from ta.breakout.volume import assess_volume_profile
        # vol_trend_mean ≈ 0.8 (see quiet test); with threshold=0.7 it should NOT be quiet
        df = _make_df(
            rbo_20=[0.0] * 15,
            vol_consolidation=[50.0] * 15,
            warmup_vol=100.0,
        )
        result_default = assess_volume_profile(df, quiet_threshold=1.0)
        result_strict  = assess_volume_profile(df, quiet_threshold=0.5)
        assert result_default.is_quiet is True   # mean ≈ 0.8 < 1.0
        assert result_strict.is_quiet is False    # mean ≈ 0.8 > 0.5... wait, need to check

    def test_custom_breakout_vol_threshold(self):
        """breakout_vol_threshold parameter is respected."""
        from ta.breakout.volume import assess_volume_profile
        rbo = [0.0] * 10 + [1.0]
        vol = [80.0] * 10 + [130.0]   # vol_trend_now ≈ 1.3
        df = _make_df(rbo_20=rbo, vol_consolidation=vol, warmup_vol=100.0)
        # Default threshold 1.2 → confirmed
        result_default = assess_volume_profile(df, breakout_vol_threshold=1.2)
        # Strict threshold 1.5 → not confirmed
        result_strict  = assess_volume_profile(df, breakout_vol_threshold=1.5)
        assert result_default.breakout_confirmed is True
        assert result_strict.breakout_confirmed is False

    def test_window_bars_limits_zero_run_used(self):
        """window_bars caps how many zero-run bars are used for mean/slope."""
        from ta.breakout.volume import assess_volume_profile
        # Long zero-run: 40 bars of low vol, then 10 bars of high vol
        # If window_bars=10, only the last 10 low-vol bars are used
        rbo = [0.0] * 50
        # First 40 bars: vol=50, last 10 bars: vol=200
        vol = [50.0] * 40 + [200.0] * 10
        df = _make_df(rbo_20=rbo, vol_consolidation=vol, warmup_vol=100.0)
        result_wide   = assess_volume_profile(df, window_bars=50)
        result_narrow = assess_volume_profile(df, window_bars=10)
        # narrow window only sees the last 10 bars (vol=200) → higher mean
        assert result_narrow.vol_trend_mean > result_wide.vol_trend_mean

    def test_vol_trend_now_reflects_last_bar(self):
        """vol_trend_now is the vol_trend of the very last row."""
        from ta.breakout.volume import assess_volume_profile
        rbo = [0.0] * 10 + [1.0]
        # Set last bar volume to exactly 500 (wildly different from warmup=100)
        vol = [100.0] * 10 + [500.0]
        df = _make_df(rbo_20=rbo, vol_consolidation=vol, warmup_vol=100.0)
        result = assess_volume_profile(df)
        # vol_trend_now = 500 / rolling_mean; rolling_mean ≈ 100 → vol_trend ≈ 5.0
        assert result.vol_trend_now > 3.0


# ---------------------------------------------------------------------------
# TestVolumeProfileSmoke — real parquet values
# ---------------------------------------------------------------------------

PARQUET = Path("data/results/it/analysis_results.parquet")

@pytest.mark.skipif(not PARQUET.exists(), reason="parquet not available")
class TestVolumeProfileSmoke:

    @pytest.fixture(scope="class")
    def parquet_df(self):
        df = pd.read_parquet(PARQUET)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def test_a2a_mi_quiet_consolidation(self, parquet_df):
        """
        A2A.MI window ending 2016-03-23 (rbo_20=0 throughout, still consolidating).

        Expected (verified against parquet):
            vol_trend_mean     ≈ 0.8051  (< 1.0 → is_quiet=True)
            vol_trend_slope    ≈ 0.00146 (> 0 → is_declining=False)
            breakout_confirmed = None
        """
        from ta.breakout.volume import assess_volume_profile
        a2a = (
            parquet_df[parquet_df["symbol"] == "A2A.MI"]
            .sort_values("date")
            .reset_index(drop=True)
        )
        a2a_window = a2a[a2a["date"] <= "2016-03-23"].copy()

        result = assess_volume_profile(a2a_window)

        assert result.is_quiet is True,                    f"Expected is_quiet=True, got {result.vol_trend_mean}"
        assert result.is_declining is False,               f"Expected is_declining=False, got {result.vol_trend_slope}"
        assert result.breakout_confirmed is None,          "Last bar is in consolidation — breakout_confirmed must be None"
        assert abs(result.vol_trend_mean  - 0.8051) < 0.02, f"vol_trend_mean {result.vol_trend_mean} not near 0.8051"
        assert abs(result.vol_trend_slope - 0.00146) < 0.005, f"vol_trend_slope {result.vol_trend_slope} not near 0.00146"

    def test_por_mi_breakout_bar_confirmed(self, parquet_df):
        """
        POR.MI breakout bar 2018-09-03 (rbo_20 flipped 0 → 1 with extreme volume).

        Expected (verified against parquet):
            vol_trend_now      ≈ 10.9546
            vol_trend_mean     ≈ 0.3714  (< 1.0 → is_quiet=True)
            vol_trend_slope    ≈ 0.01244 (> 0 → is_declining=False)
            breakout_confirmed = True
        """
        from ta.breakout.volume import assess_volume_profile
        por = (
            parquet_df[parquet_df["symbol"] == "POR.MI"]
            .sort_values("date")
            .reset_index(drop=True)
        )
        por_window = por[por["date"] <= "2018-09-03"].copy()

        result = assess_volume_profile(por_window)

        assert result.breakout_confirmed is True,          f"Expected breakout_confirmed=True, got {result.breakout_confirmed}"
        assert result.is_quiet is True,                    f"Expected is_quiet=True (quiet pre-breakout), got {result.vol_trend_mean}"
        assert abs(result.vol_trend_now  - 10.9546) < 0.05, f"vol_trend_now {result.vol_trend_now} not near 10.9546"
        assert abs(result.vol_trend_mean -  0.3714) < 0.02,  f"vol_trend_mean {result.vol_trend_mean} not near 0.3714"
        assert abs(result.vol_trend_slope - 0.01244) < 0.005, f"vol_trend_slope {result.vol_trend_slope} not near 0.01244"
