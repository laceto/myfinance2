"""
volume.py — Volume-behaviour primitive for the range breakout trader.

Public API
----------
assess_volume_profile(df, *, volume_col, signal_col, vol_ma_window,
                      window_bars, min_vol_bars, quiet_threshold,
                      breakout_vol_threshold)
    Analyse volume behaviour during consolidation and at breakout bars.
    Returns a VolumeProfile dataclass answering two independent questions:
      1. Was volume quiet (below average) during the consolidation window?
      2. Did volume confirm the breakout on the flip bar?

Design decisions
----------------
is_quiet uses vol_trend_mean < quiet_threshold ONLY (not slope).
Real data (POR.MI 2018-09-03) shows healthy consolidations with a slightly
positive volume slope — a slope gate would silently reject valid accumulation
setups.  Slope is reported separately via is_declining so callers can apply
stricter filters if they choose.

vol_trend_slope_r2 is reported alongside vol_trend_slope so callers can
assess statistical reliability. R² < 0.3 on short windows (< ~20 bars) means
the slope is dominated by noise and should be treated as informational only.

The zero-run window is always the most recent consecutive run of signal_col==0,
limited to window_bars bars.  This is computed even when the last bar is a
breakout bar (signal_col[-1] != 0), so mean/slope reflect the pre-breakout
consolidation quality regardless of when assess_volume_profile is called.

breakout_confirmed is only set (True/False) on the exact flip bar.
On all other bars — whether inside consolidation or mid-trend — it is None.

Lookahead note
--------------
This function returns a snapshot of bar t using data up to and including bar t.
If called in a rolling loop for backtesting, the caller MUST delay the result
by one bar before using it in signal logic:
    # LOOKAHEAD PROTECTED: signal delayed by 1 bar
    signals = signals.shift(1)

See range_breakout_trader.md §"Primitive 5 — Volume Behavior" for the full
derivation and the smoke-test values used in ta/breakout/tests/test_volume.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ta.utils import ols_slope_r2

# ---------------------------------------------------------------------------
# Module-level constants (exported so tests can reference them by name)
# ---------------------------------------------------------------------------

MIN_VOL_BARS:             int   = 5     # minimum non-NaN bars required in zero-run
DEFAULT_WINDOW_BARS:      int   = 40    # max zero-run bars used for mean/slope
DEFAULT_VOL_MA_WINDOW:    int   = 20    # rolling window for vol_trend denominator
DEFAULT_QUIET_THRESHOLD:  float = 1.0   # vol_trend_mean < this → is_quiet
DEFAULT_BREAKOUT_VOL_THR: float = 1.2   # vol_trend_now >= this on flip bar → confirmed


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolumeProfile:
    """
    Snapshot of volume behaviour at the last bar of a ticker series.

    Attributes
    ----------
    vol_trend_now : float
        vol_trend at the very last bar of the input DataFrame.
        vol_trend = volume / volume.rolling(vol_ma_window, min_periods=1).mean()
        NaN when the rolling mean is zero (halted stock / bad data).
        A value > 1 means the last bar's volume is above its rolling average.

    vol_trend_mean : float
        Mean vol_trend over the most recent consolidation window
        (last consecutive run of signal_col == 0, capped at window_bars).

    vol_trend_slope : float
        OLS slope of vol_trend over the same consolidation window, in units of
        vol_trend change per bar.
        Negative = volume drying up inside the range (healthy).
        Positive = volume building inside the range (accumulation or distribution).

    vol_trend_slope_r2 : float
        R² of the OLS fit for vol_trend_slope, in [0, 1].
        R² < 0.3 → slope is dominated by noise (window too short or data too noisy).
        R² ≥ 0.7 → strong linear trend; slope is reliable as a secondary filter.
        Invariant: always in [0, 1].

    is_quiet : bool
        True when vol_trend_mean < quiet_threshold.
        The sole gate for "was the consolidation quiet?" — slope is informational only.
        Invariant: never None.

    is_declining : bool
        True when vol_trend_slope < 0.
        Informational; callers may use it as a secondary filter.
        Invariant: never None.

    breakout_confirmed : bool | None
        True   — last bar is a breakout flip AND vol_trend_now >= breakout_vol_threshold.
        False  — last bar is a breakout flip AND vol_trend_now <  breakout_vol_threshold.
        None   — last bar is NOT a breakout flip (in consolidation or mid-trend).

        A "breakout flip" is defined as: signal_col[-1] != 0 AND signal_col[-2] == 0.
    """

    vol_trend_now:        float
    vol_trend_mean:       float
    vol_trend_slope:      float
    vol_trend_slope_r2:   float
    is_quiet:             bool
    is_declining:         bool
    breakout_confirmed:   bool | None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_zero_run(rbo: np.ndarray) -> tuple[int, int]:
    """
    Find (start_idx, end_idx) of the most recent consecutive run of rbo == 0.

    Fully vectorised — no Python while loops.

    Walks backward from the end of the array (skipping any trailing non-zero bars
    by taking the last zero via np.where), then identifies the start of the
    contiguous zero-run that ends there.

    Args:
        rbo: 1-D integer array of signal values (e.g. dtype=np.int8).

    Returns:
        (start_idx, end_idx) — inclusive indices into the original array.

    Raises:
        ValueError: if no rbo == 0 bar exists in the entire array.

    Invariants:
        - All values rbo[start_idx : end_idx + 1] == 0.
        - For a last-bar breakout (rbo[-1] != 0, rbo[-2] == 0):
            end_idx = len(rbo) - 2  (the bar immediately before the flip).
    """
    is_zero = rbo == 0

    if not is_zero.any():
        raise ValueError(
            "No consolidation window (signal_col == 0) found in the provided history. "
            "The entire series is in a trend or breakout state. "
            "Provide a longer history that includes at least one consolidation period."
        )

    # end = last position where rbo is zero (ignores trailing non-zero bars)
    end = int(np.where(is_zero)[0][-1])

    # start = first index of the contiguous zero-run ending at `end`
    # Any non-zero bar at or before `end` is a boundary; start after the last one.
    non_zero_before_end = np.where(~is_zero[: end + 1])[0]
    start = int(non_zero_before_end[-1] + 1) if len(non_zero_before_end) > 0 else 0

    return start, end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_volume_profile(
    df:                     pd.DataFrame,
    *,
    volume_col:             str   = "volume",
    signal_col:             str   = "rbo_20",
    vol_ma_window:          int   = DEFAULT_VOL_MA_WINDOW,
    window_bars:            int   = DEFAULT_WINDOW_BARS,
    min_vol_bars:           int   = MIN_VOL_BARS,
    quiet_threshold:        float = DEFAULT_QUIET_THRESHOLD,
    breakout_vol_threshold: float = DEFAULT_BREAKOUT_VOL_THR,
) -> VolumeProfile:
    """
    Analyse volume behaviour during the most recent consolidation window and at
    the breakout flip bar (if applicable).

    Algorithm
    ---------
    1. Validate all parameters and required columns.
    2. Compute vol_trend = volume / rolling_mean(vol_ma_window), with a zero-
       denominator guard: rolling_mean <= 0 → vol_trend = NaN (not inf).
    3. Find the most recent consecutive run of signal_col == 0 via _find_zero_run().
       This always reflects the pre-breakout consolidation, whether the last bar
       is still in consolidation or is a just-completed breakout.
    4. Cap the zero-run to the last window_bars bars.
    5. Drop NaN from vol_trend within the capped window. Raise if < min_vol_bars remain.
    6. Compute vol_trend_mean and (vol_trend_slope, vol_trend_slope_r2) via OLS.
    7. Derive is_quiet (mean gate) and is_declining (slope sign).
    8. vol_trend_now = vol_trend at the very last bar of df.
    9. Breakout flip check:
         flip = (signal_col[-1] != 0) AND (signal_col[-2] == 0)
         breakout_confirmed = vol_trend_now >= breakout_vol_threshold  if flip else None

    Args:
        df:
            Full single-ticker history sorted ascending by date.
            Must contain columns named by volume_col and signal_col.
        volume_col:
            Name of the volume column. Default "volume".
        signal_col:
            Name of the range-breakout signal column (values: -1, 0, 1).
            Default "rbo_20". 0 = in consolidation; ±1 = active breakout/breakdown.
            Note: cast to np.int8 internally — upstream values must be integral
            (1.0, 0.0, -1.0 are fine; non-integer floats like 0.9 would be truncated).
        vol_ma_window:
            Rolling window (bars) for the vol_trend denominator. Default 20.
        window_bars:
            Maximum number of zero-run bars used for mean/slope. Default 40.
        min_vol_bars:
            Minimum non-NaN vol_trend bars required in the capped zero-run.
            Default 5 (module constant MIN_VOL_BARS).
        quiet_threshold:
            vol_trend_mean < this → is_quiet=True. Must be > 0. Default 1.0.
        breakout_vol_threshold:
            vol_trend_now >= this on a flip bar → breakout_confirmed=True.
            Must be > 0. Default 1.2 (20% above the rolling average).

    Returns:
        VolumeProfile with all fields populated.

    Raises:
        ValueError: if any parameter is out of range.
        ValueError: if volume_col or signal_col is not present in df.
        ValueError: if no signal_col == 0 bar exists in df.
        ValueError: if the capped zero-run contains fewer than min_vol_bars
                    non-NaN vol_trend bars.

    Failure modes:
        - Halted stock (all-zero volume): rolling_mean = 0 → vol_trend = NaN.
          If the zero-run is dominated by NaN bars, ValueError is raised.
        - Very short history: vol_trend is computed with min_periods=1 so no NaN
          from rolling warmup; but the zero-run may still be too short.
        - Entire history in trend (no signal_col == 0): raises ValueError.
    """
    # --- 1. Parameter validation (fail fast) ---
    if window_bars < 1:
        raise ValueError(
            f"window_bars must be >= 1, got {window_bars}. "
            "This parameter caps the consolidation window length used for mean/slope."
        )
    if vol_ma_window < 1:
        raise ValueError(
            f"vol_ma_window must be >= 1, got {vol_ma_window}. "
            "This is the rolling window for the vol_trend denominator."
        )
    if min_vol_bars < 1:
        raise ValueError(
            f"min_vol_bars must be >= 1, got {min_vol_bars}."
        )
    if quiet_threshold <= 0:
        raise ValueError(
            f"quiet_threshold must be > 0, got {quiet_threshold}. "
            "Example: quiet_threshold=1.0 (vol_trend_mean below the rolling average)."
        )
    if breakout_vol_threshold <= 0:
        raise ValueError(
            f"breakout_vol_threshold must be > 0, got {breakout_vol_threshold}. "
            "Example: breakout_vol_threshold=1.2 (20% above the rolling average)."
        )

    required = {volume_col, signal_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}. "
            f"DataFrame has: {sorted(df.columns)}. "
            "Pass the correct column names via volume_col= and signal_col= parameters."
        )

    # --- 2. vol_trend series (zero-denominator guarded) ---
    # rolling_mean <= 0 (halted stock / bad data) → NaN, never inf.
    rolling_mean: pd.Series = (
        df[volume_col].rolling(vol_ma_window, min_periods=1).mean()
    )
    _safe_mean                = rolling_mean.values.copy()
    _safe_mean[_safe_mean <= 0] = np.nan        # guard: zero/negative → NaN propagation

    vol_trend: pd.Series = pd.Series(
        df[volume_col].values / _safe_mean,
        index=df.index,
    ).reset_index(drop=True)                     # positional alignment with rbo array

    # --- 3. Find the most recent zero-run (vectorised, no Python loops) ---
    # Cast to int8: signal is categorical {-1, 0, 1}; avoids float != 0 comparisons.
    rbo = df[signal_col].to_numpy(dtype=np.int8)
    zero_start, zero_end = _find_zero_run(rbo)

    # --- 4. Cap to window_bars (take the tail of the zero-run) ---
    capped_start = max(zero_start, zero_end - window_bars + 1)
    window_vt    = vol_trend.iloc[capped_start : zero_end + 1]

    # --- 5. Drop NaN; guard minimum bars ---
    clean_vt = window_vt.dropna()
    if len(clean_vt) < min_vol_bars:
        raise ValueError(
            f"assess_volume_profile requires at least {min_vol_bars} non-NaN vol_trend "
            f"bars in the consolidation window; found {len(clean_vt)}. "
            "Provide a longer history or check for excessive NaN in the volume column. "
            f"(vol_ma_window={vol_ma_window}, window_bars={window_bars})"
        )

    # --- 6. Mean and OLS slope with R² for noise detection ---
    y          = clean_vt.to_numpy(dtype=float)
    vt_mean    = float(y.mean())
    vt_slope, vt_r2 = ols_slope_r2(y)
    # R² < 0.3 → treat slope as noise; R² ≥ 0.7 → reliable trend signal.

    # --- 7. Qualitative flags ---
    is_quiet     = bool(vt_mean  < quiet_threshold)
    is_declining = bool(vt_slope < 0.0)

    # --- 8. vol_trend at last bar ---
    vt_now = float(vol_trend.iloc[-1])

    # --- 9. Breakout flip check (integer comparison, no float ambiguity) ---
    breakout_confirmed: bool | None = None
    if len(rbo) >= 2 and rbo[-1] != 0 and rbo[-2] == 0:
        breakout_confirmed = bool(vt_now >= breakout_vol_threshold)

    return VolumeProfile(
        vol_trend_now       = round(vt_now,   4),
        vol_trend_mean      = round(vt_mean,  4),
        vol_trend_slope     = round(vt_slope, 5),
        vol_trend_slope_r2  = round(vt_r2,    4),
        is_quiet            = is_quiet,
        is_declining        = is_declining,
        breakout_confirmed  = breakout_confirmed,
    )
