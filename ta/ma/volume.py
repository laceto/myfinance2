"""
volume.py — Volume-behaviour primitive for the MA crossover trader.

Public API
----------
assess_ma_volume(df, signal_col, *, volume_col, vol_ma_window,
                 confirmed_vol_threshold, sustained_vol_threshold, min_post_bars)
    Analyse volume behaviour at the MA crossover flip bar and in the
    post-crossover bars.
    Returns a MAVolumeProfile dataclass answering two independent questions:
      1. Was volume expanding on the crossover flip bar? (is_confirmed)
      2. Is volume staying above average post-crossover? (is_sustained)

Design decisions vs ta.breakout.volume
---------------------------------------
The breakout volume primitive checks for QUIET volume during consolidation
and a SPIKE on the breakout flip bar.

For MA crossovers the question is different:
  - There is no meaningful "consolidation window" — MA signals transition
    continuously between +1 and -1 without a flat 0 state.
  - The quality checks are:
    1. vol_trend_on_crossover >= confirmed_vol_threshold (default 1.2) at the flip bar.
       Expanding volume on the crossover = institutional participation.
    2. Mean vol_trend over the first min_post_bars post-flip bars
       >= sustained_vol_threshold (default 1.0).
       Sustained above-average volume = the move has follow-through.

A "flip" is any bar where signal[-1] != signal[-2] (any direction:
0→+1, +1→-1, -1→0, etc.).

Lookahead note
--------------
This function returns a snapshot of bar t using data up to and including bar t.
If called in a rolling loop for backtesting, the caller MUST delay results
by one bar before use in signal logic:
    # LOOKAHEAD PROTECTED: signal delayed by 1 bar
    signals = signals.shift(1)

See ma_crossovers_trader.md §"Confirmation Indicators Checklist" for the
trading rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants (exported so tests can reference them)
# ---------------------------------------------------------------------------

MIN_POST_BARS:            int   = 3    # minimum post-flip bars needed to compute is_sustained
CONFIRMED_VOL_THRESHOLD:  float = 1.2  # vol_trend_on_flip >= this → is_confirmed=True
DEFAULT_VOL_MA_WINDOW:    int   = 20   # rolling window for vol_trend denominator
DEFAULT_SUSTAINED_VOL_THR: float = 1.0  # mean post-flip vol_trend >= this → is_sustained=True


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MAVolumeProfile:
    """
    Snapshot of volume behaviour at and after the most recent MA crossover flip bar.

    Attributes
    ----------
    vol_on_crossover : float | None
        vol_trend at the flip bar (volume / vol_ma_window-bar rolling mean).
        None if the last bar is not a crossover flip.
        NaN when volume is zero on that bar (zero rolling mean → NaN, not inf).

    vol_trend_mean_post : float | None
        Mean vol_trend over the first min_post_bars bars after the flip.
        None when: no flip ever found, or fewer than min_post_bars post-flip bars.

    is_confirmed : bool | None
        True   — last bar is a flip AND vol_on_crossover >= confirmed_vol_threshold.
        False  — last bar is a flip AND vol_on_crossover <  confirmed_vol_threshold.
        None   — last bar is NOT a flip (continuation bar or no signal change).

    is_sustained : bool | None
        True   — vol_trend_mean_post >= sustained_vol_threshold.
        False  — vol_trend_mean_post <  sustained_vol_threshold.
        None   — insufficient post-flip history (< min_post_bars bars).

    Relationship between is_confirmed and is_sustained
        is_confirmed answers: "was the crossover itself convincing?"
        is_sustained answers: "is the follow-through real?"
        Both True = high-quality MA trend entry.
        is_confirmed=True, is_sustained=False = possible fakeout / distribution.
    """

    vol_on_crossover:    float | None
    vol_trend_mean_post: float | None
    is_confirmed:        bool  | None
    is_sustained:        bool  | None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_last_flip(signal: np.ndarray) -> int | None:
    """
    Find the index of the most recent flip bar (last bar where signal changed).

    Fully vectorised — no Python loops.

    A flip is defined as signal[i] != signal[i-1] for any i >= 1.

    Args:
        signal: 1-D integer array (dtype np.int8), sorted ascending by date.
                Must have length >= 2.

    Returns:
        Index (into the original array) of the most recent flip bar,
        or None if no flip was ever found.

    Example:
        >>> _find_last_flip(np.array([0, 0, 1, 1, -1], dtype=np.int8))
        4
    """
    diff  = signal[1:] != signal[:-1]   # boolean array of length n-1
    flips = np.where(diff)[0]           # positions in diff where signal changed
    return int(flips[-1] + 1) if len(flips) > 0 else None  # +1: diff[i] = signal[i+1] vs signal[i]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_ma_volume(
    df:                       pd.DataFrame,
    signal_col:               str,
    *,
    volume_col:               str   = "volume",
    vol_ma_window:            int   = DEFAULT_VOL_MA_WINDOW,
    confirmed_vol_threshold:  float = CONFIRMED_VOL_THRESHOLD,
    sustained_vol_threshold:  float = DEFAULT_SUSTAINED_VOL_THR,
    min_post_bars:            int   = MIN_POST_BARS,
) -> MAVolumeProfile:
    """
    Analyse volume behaviour at and after the most recent MA crossover flip bar.

    Algorithm
    ---------
    1. Validate all parameters and required columns.
    2. Compute vol_trend = volume / rolling_mean(vol_ma_window), with a zero-
       denominator guard: rolling_mean <= 0 → vol_trend = NaN (not inf).
    3. Cast signal to np.int8 to guarantee exact integer comparisons
       (parquet round-trips can silently widen to float64).
    4. Determine whether the last bar is a flip (signal[-1] != signal[-2]).
    5. vol_on_crossover / is_confirmed: set if last bar is a flip, else None.
    6. Find the most recent flip via _find_last_flip() (vectorised).
    7. Slice vol_trend[flip_idx+1 : flip_idx+1+min_post_bars].
    8. vol_trend_mean_post / is_sustained: set if min_post_bars bars available, else None.

    Args:
        df:                      Full single-ticker history sorted ascending by date.
        signal_col:              MA crossover signal column to monitor for flips
                                 (e.g. "rema_50100"). Must be present in df.
        volume_col:              Volume column name. Default "volume".
        vol_ma_window:           Rolling window for vol_trend denominator. Default 20.
                                 Must be >= 1.
        confirmed_vol_threshold: vol_trend >= this at flip bar → is_confirmed=True.
                                 Must be > 0. Default CONFIRMED_VOL_THRESHOLD (1.2).
        sustained_vol_threshold: Mean post-flip vol_trend >= this → is_sustained=True.
                                 Must be > 0. Default DEFAULT_SUSTAINED_VOL_THR (1.0).
        min_post_bars:           Minimum post-flip bars required to set is_sustained.
                                 Must be >= 1. Default MIN_POST_BARS (3).

    Returns:
        MAVolumeProfile with all fields populated.

    Raises:
        ValueError: if any parameter is out of range.
        ValueError: if volume_col or signal_col is missing from df.
        ValueError: if fewer than 2 bars are provided (cannot detect a flip).

    Failure modes:
        - All-zero volume: rolling_mean = 0 → vol_trend = NaN. is_confirmed will be
          False (NaN >= threshold is False in Python). vol_on_crossover will be NaN.
        - Signal never changes: flip_idx=None → is_sustained=None.
    """
    # --- 1. Parameter validation (fail fast) ---
    if vol_ma_window < 1:
        raise ValueError(
            f"vol_ma_window must be >= 1, got {vol_ma_window}. "
            "This is the rolling window for the vol_trend denominator."
        )
    if confirmed_vol_threshold <= 0:
        raise ValueError(
            f"confirmed_vol_threshold must be > 0, got {confirmed_vol_threshold}. "
            "Example: confirmed_vol_threshold=1.2 (20% above the rolling average)."
        )
    if sustained_vol_threshold <= 0:
        raise ValueError(
            f"sustained_vol_threshold must be > 0, got {sustained_vol_threshold}. "
            "Example: sustained_vol_threshold=1.0 (at or above the rolling average)."
        )
    if min_post_bars < 1:
            raise ValueError(
            f"min_post_bars must be >= 1, got {min_post_bars}. "
            "This is the number of post-flip bars required to assess follow-through."
        )

    required = {volume_col, signal_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"assess_ma_volume: DataFrame missing required columns: {sorted(missing)}. "
            f"DataFrame has: {sorted(df.columns)}. "
            "Pass the correct column names via volume_col= and signal_col= parameters."
        )

    if len(df) < 2:
        raise ValueError(
            "assess_ma_volume requires at least 2 bars to detect a signal flip; "
            f"got {len(df)}."
        )

    # --- 2. vol_trend series (zero-denominator guarded) ---
    rolling_mean: pd.Series = (
        df[volume_col].rolling(vol_ma_window, min_periods=1).mean()
    )
    _safe_mean                  = rolling_mean.values.copy()
    _safe_mean[_safe_mean <= 0] = np.nan        # zero/negative volume → NaN, never inf

    vol_trend: pd.Series = pd.Series(
        df[volume_col].values / _safe_mean,
        index=df.index,
    ).reset_index(drop=True)                     # positional alignment with signal array

    # --- 3. Signal as int8 (safe integer comparison; guards float parquet round-trips) ---
    signal_arr = df[signal_col].to_numpy(dtype=np.int8)

    # --- 4. Is the last bar a flip? (int != int: exact, no float ambiguity) ---
    last_is_flip = bool(signal_arr[-1] != signal_arr[-2])

    # --- 5. vol_on_crossover and is_confirmed ---
    if last_is_flip:
        vt_now           = float(vol_trend.iloc[-1])
        vol_on_crossover = round(vt_now, 4)
        is_confirmed     = bool(vt_now >= confirmed_vol_threshold)
    else:
        vol_on_crossover = None
        is_confirmed     = None

    # --- 6. Find the most recent flip (vectorised, no loops) ---
    flip_idx = _find_last_flip(signal_arr)

    if flip_idx is None:
        return MAVolumeProfile(
            vol_on_crossover    = vol_on_crossover,
            vol_trend_mean_post = None,
            is_confirmed        = is_confirmed,
            is_sustained        = None,
        )

    # --- 7. Post-flip bars (flip_idx+1 onward, capped at min_post_bars) ---
    post_start = flip_idx + 1
    post_vt    = vol_trend.iloc[post_start : post_start + min_post_bars]

    if len(post_vt) < min_post_bars:
        return MAVolumeProfile(
            vol_on_crossover    = vol_on_crossover,
            vol_trend_mean_post = None,
            is_confirmed        = is_confirmed,
            is_sustained        = None,
        )

    # --- 8–9. vol_trend_mean_post and is_sustained ---
    mean_post    = float(post_vt.mean())
    is_sustained = bool(mean_post >= sustained_vol_threshold)

    return MAVolumeProfile(
        vol_on_crossover    = vol_on_crossover,
        vol_trend_mean_post = round(mean_post, 4),
        is_confirmed        = is_confirmed,
        is_sustained        = is_sustained,
    )
