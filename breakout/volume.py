"""
volume.py — Volume-behaviour primitive for the range breakout trader.

Public API:

    assess_volume_profile(df, window_bars, quiet_threshold, breakout_vol_threshold)
        Analyse volume behaviour during consolidation and at breakout bars.
        Returns a VolumeProfile dataclass answering two independent questions:
          1. Was volume quiet (below average) during the consolidation window?
          2. Did volume confirm the breakout on the flip bar?

Design decisions
----------------
is_quiet uses vol_trend_mean < quiet_threshold ONLY (not slope).
Real data (POR.MI 2018-09-03) shows healthy consolidations with a slightly positive
volume slope — a slope gate would silently reject valid accumulation setups.
Slope is reported separately via is_declining so callers can apply stricter filters
if they choose.

The zero-run window is always the most recent consecutive run of rbo_20==0,
limited to window_bars bars.  This is computed even when the last bar is a
breakout bar (rbo_20 != 0), so that mean/slope reflect the pre-breakout
consolidation quality regardless of when assess_volume_profile is called.

breakout_confirmed is only set (True/False) on the exact flip bar.
On all other bars — whether inside consolidation or mid-trend — it is None.

See range_breakout_trader.md §"Primitive 5 — Volume Behavior" for the full
derivation and the smoke-test values used in tests/test_volume.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from breakout.utils import ols_slope

# ---------------------------------------------------------------------------
# Module-level constants (exported so tests can reference them by name)
# ---------------------------------------------------------------------------

MIN_VOL_BARS:              int   = 5    # minimum non-NaN bars needed in the zero-run
DEFAULT_WINDOW_BARS:       int   = 40   # max zero-run bars used for mean/slope
DEFAULT_QUIET_THRESHOLD:   float = 1.0  # vol_trend_mean < this → is_quiet
DEFAULT_BREAKOUT_VOL_THR:  float = 1.2  # vol_trend_now >= this on flip bar → confirmed

_REQUIRED_COLS = frozenset({"volume", "rbo_20"})


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolumeProfile:
    """
    Snapshot of volume behaviour at the last bar of a ticker series.

    Attributes
    ----------
    vol_trend_now
        vol_trend at the very last bar of the input DataFrame.
        vol_trend = volume / volume.rolling(20, min_periods=1).mean()
        A value > 1 means the last bar's volume is above its 20-bar average.

    vol_trend_mean
        Mean vol_trend over the most recent consolidation window
        (last consecutive run of rbo_20 == 0, capped at window_bars).
        Reflects typical volume during consolidation.

    vol_trend_slope
        OLS slope of vol_trend over the same consolidation window, in units of
        vol_trend change per bar.
        Negative  = volume declining inside the range (drying up — healthy).
        Positive  = volume building inside the range (accumulation or distribution).

    is_quiet
        True when vol_trend_mean < quiet_threshold.
        The sole gate for "was the consolidation quiet?" — slope is informational only.
        Invariant: never None.

    is_declining
        True when vol_trend_slope < 0.
        Informational; callers may use it as a secondary filter.
        Invariant: never None.

    breakout_confirmed
        True   — last bar is a breakout flip AND vol_trend_now >= breakout_vol_threshold.
        False  — last bar is a breakout flip AND vol_trend_now <  breakout_vol_threshold.
        None   — last bar is NOT a breakout flip (in consolidation or mid-trend).

        A "breakout flip" is defined as: rbo_20[-1] != 0 AND rbo_20[-2] == 0.
    """

    vol_trend_now:       float
    vol_trend_mean:      float
    vol_trend_slope:     float
    is_quiet:            bool
    is_declining:        bool
    breakout_confirmed:  bool | None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_zero_run(rbo: np.ndarray) -> tuple[int, int]:
    """
    Find the (start_idx, end_idx) of the most recent consecutive run of rbo == 0.

    Walks backward from the end of the array, skipping any trailing non-zero bars,
    then identifies the zero-run immediately before them.

    Args:
        rbo: 1-D float/int array of rbo_20 signal values, sorted ascending by date.

    Returns:
        (start_idx, end_idx) — inclusive indices into the original array.

    Raises:
        ValueError: if no rbo == 0 bar exists in the entire array.

    Invariants:
        - All values rbo[start_idx:end_idx+1] are == 0.
        - end_idx < len(rbo).
        - For a last-bar breakout (rbo[-1] != 0, rbo[-2] == 0):
            end_idx = len(rbo) - 2  (i.e., the bar immediately before the flip).
    """
    n = len(rbo)

    # Skip trailing non-zero bars (e.g., an active trend following a breakout)
    end = n - 1
    while end >= 0 and rbo[end] != 0:
        end -= 1

    if end < 0:
        raise ValueError(
            "No consolidation window (rbo_20 == 0) found in the provided history. "
            "The entire series is in a trend or breakout state. "
            "Provide a longer history that includes at least one consolidation period."
        )

    # Walk backward to find the start of this zero-run
    start = end
    while start > 0 and rbo[start - 1] == 0:
        start -= 1

    return start, end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_volume_profile(
    df:                    pd.DataFrame,
    window_bars:           int   = DEFAULT_WINDOW_BARS,
    quiet_threshold:       float = DEFAULT_QUIET_THRESHOLD,
    breakout_vol_threshold: float = DEFAULT_BREAKOUT_VOL_THR,
) -> VolumeProfile:
    """
    Analyse volume behaviour during the most recent consolidation window and at
    the breakout flip bar (if applicable).

    Algorithm
    ---------
    1. Validate that 'volume' and 'rbo_20' columns are present.
    2. Compute vol_trend = volume / volume.rolling(20, min_periods=1).mean().
    3. Find the most recent consecutive run of rbo_20 == 0 using _find_zero_run().
       This is always the pre-breakout consolidation window, whether the last bar
       is still in consolidation or is a just-completed breakout.
    4. Cap the zero-run to the last window_bars bars.
    5. Drop NaN from vol_trend within the capped window.  Raise if < MIN_VOL_BARS remain.
    6. Compute vol_trend_mean and vol_trend_slope (OLS) over the clean window.
    7. Derive is_quiet (mean gate) and is_declining (slope sign).
    8. vol_trend_now = vol_trend at the very last bar of df.
    9. Breakout flip check:
         flip = (rbo_20[-1] != 0) AND (rbo_20[-2] == 0)
         breakout_confirmed = vol_trend_now >= breakout_vol_threshold  if flip else None

    Args:
        df:                     Full single-ticker history sorted ascending by date.
                                Must contain columns 'volume' and 'rbo_20'.
        window_bars:            Maximum number of zero-run bars used for mean/slope.
                                Matches the consolidation window length. Default 40.
        quiet_threshold:        vol_trend_mean < this → is_quiet=True. Default 1.0.
        breakout_vol_threshold: vol_trend_now >= this on a flip bar → breakout_confirmed=True.
                                Default 1.2 (20% above 20-bar average).

    Returns:
        VolumeProfile with all fields populated.

    Raises:
        ValueError: if 'volume' or 'rbo_20' column is missing.
        ValueError: if no rbo_20 == 0 bar exists in df (no consolidation history).
        ValueError: if the zero-run contains fewer than MIN_VOL_BARS non-NaN vol_trend bars.

    Failure modes:
        - Very short history (< 20 bars): vol_trend is computed with min_periods=1,
          so no NaN from rolling warmup; but the zero-run may still be short and raise.
        - Entire history in trend (no rbo_20 == 0): raises ValueError with clear message.
        - Zero-run has all-NaN volume: raises ValueError after dropping NaN.
    """
    # --- 1. Column validation ---
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        col = sorted(missing)[0]
        raise ValueError(
            f"DataFrame is missing required column '{col}'. "
            f"Expected columns: {sorted(_REQUIRED_COLS)}."
        )

    # --- 2. vol_trend series ---
    vol_trend: pd.Series = (
        df["volume"] / df["volume"].rolling(20, min_periods=1).mean()
    ).reset_index(drop=True)

    # --- 3. Find the most recent zero-run ---
    rbo = df["rbo_20"].to_numpy(dtype=float)
    zero_start, zero_end = _find_zero_run(rbo)

    # --- 4. Cap to window_bars (take the tail of the zero-run) ---
    capped_start = max(zero_start, zero_end - window_bars + 1)
    window_vt = vol_trend.iloc[capped_start : zero_end + 1]

    # --- 5. Drop NaN; guard minimum bars ---
    clean_vt = window_vt.dropna()
    if len(clean_vt) < MIN_VOL_BARS:
        raise ValueError(
            f"assess_volume_profile requires at least {MIN_VOL_BARS} non-NaN vol_trend bars "
            f"in the zero-run; found {len(clean_vt)}. "
            "Provide a longer history or check for excessive NaN in 'volume'."
        )

    # --- 6. Mean and slope ---
    y = clean_vt.to_numpy(dtype=float)
    vt_mean  = float(y.mean())
    vt_slope = ols_slope(y)

    # --- 7. Qualitative flags ---
    is_quiet     = bool(vt_mean  < quiet_threshold)
    is_declining = bool(vt_slope < 0.0)

    # --- 8. vol_trend at last bar ---
    vt_now = float(vol_trend.iloc[-1])

    # --- 9. Breakout flip check ---
    breakout_confirmed: bool | None = None
    if len(rbo) >= 2 and rbo[-1] != 0 and rbo[-2] == 0:
        breakout_confirmed = bool(vt_now >= breakout_vol_threshold)

    return VolumeProfile(
        vol_trend_now      = round(vt_now,    4),
        vol_trend_mean     = round(vt_mean,   4),
        vol_trend_slope    = round(vt_slope,  5),
        is_quiet           = is_quiet,
        is_declining       = is_declining,
        breakout_confirmed = breakout_confirmed,
    )
