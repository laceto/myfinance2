"""
volume.py — Volume-behaviour primitive for the MA crossover trader.

Public API:

    assess_ma_volume(df, signal_col, ...)
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
    1. vol_on_crossover >= CONFIRMED_VOL_THRESHOLD (1.2) at the flip bar.
       Expanding volume on the crossover = institutional participation.
    2. Mean vol_trend over the first MIN_POST_BARS post-flip bars >= 1.0.
       Sustained above-average volume = the move has follow-through.

A "flip" is any bar where signal[-1] != signal[-2] (value change in any direction:
0→+1, +1→-1, -1→0, etc.).

See ma_crossovers_trader.md §"Confirmation Indicators Checklist" for the
trading rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants (exported so tests can reference them)
# ---------------------------------------------------------------------------

MIN_POST_BARS:           int   = 3     # minimum post-flip bars needed to compute is_sustained
CONFIRMED_VOL_THRESHOLD: float = 1.2  # vol_trend_on_flip >= this → is_confirmed=True

_REQUIRED_COLS_BASE = {"volume"}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MAVolumeProfile:
    """
    Snapshot of volume behaviour at and after the most recent MA crossover flip bar.

    Attributes:
        vol_on_crossover:
            vol_trend at the flip bar (volume / 20-bar rolling mean).
            None if the last bar is not a crossover flip.
        vol_trend_mean_post:
            Mean vol_trend over the first MIN_POST_BARS bars after the flip.
            None when: no flip ever found, or fewer than MIN_POST_BARS post-flip bars.
        is_confirmed:
            True   — last bar is a flip AND vol_on_crossover >= CONFIRMED_VOL_THRESHOLD.
            False  — last bar is a flip AND vol_on_crossover < CONFIRMED_VOL_THRESHOLD.
            None   — last bar is NOT a flip (continuation bar or no signal change).
        is_sustained:
            True   — vol_trend_mean_post >= 1.0 (volume stayed above average).
            False  — vol_trend_mean_post < 1.0 (volume faded after crossover).
            None   — insufficient post-flip history (< MIN_POST_BARS bars).

    Relationship between is_confirmed and is_sustained:
        - is_confirmed answers: "was the crossover itself convincing?"
        - is_sustained answers: "is the follow-through real?"
        Both True = high-quality MA trend entry.
        is_confirmed=True, is_sustained=False = possible fakeout / distribution.
    """

    vol_on_crossover:    float | None
    vol_trend_mean_post: float | None
    is_confirmed:        bool  | None
    is_sustained:        bool  | None


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _find_last_flip(signal: list[float]) -> int | None:
    """
    Find the index of the most recent flip bar (last bar where signal changed).

    A flip is defined as signal[i] != signal[i-1] for any i >= 1.

    Args:
        signal: Signal values as a plain Python list, sorted ascending by date.

    Returns:
        Index of the most recent flip bar, or None if no flip was ever found.
    """
    for i in range(len(signal) - 1, 0, -1):
        if signal[i] != signal[i - 1]:
            return i
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_ma_volume(
    df:                      pd.DataFrame,
    signal_col:              str,
    confirmed_vol_threshold: float = CONFIRMED_VOL_THRESHOLD,
    min_post_bars:           int   = MIN_POST_BARS,
) -> MAVolumeProfile:
    """
    Analyse volume behaviour at and after the most recent MA crossover flip bar.

    Algorithm:
        1. Validate that 'volume' and signal_col columns are present.
        2. Validate minimum bar count (>= 2 to detect a flip).
        3. Compute vol_trend = volume / volume.rolling(20, min_periods=1).mean().
        4. Find the most recent flip bar using _find_last_flip().
        5. vol_on_crossover = vol_trend at the flip bar.
        6. is_confirmed: True/False if last bar is a flip, else None.
        7. Find post-flip bars (flip_idx+1 onward, capped at min_post_bars).
        8. vol_trend_mean_post = mean of those bars; None if < min_post_bars available.
        9. is_sustained = vol_trend_mean_post >= 1.0 if available, else None.

    Args:
        df:                      Full single-ticker history sorted ascending by date.
                                 Must contain 'volume' and signal_col columns.
        signal_col:              MA crossover signal column to monitor for flips
                                 (e.g. "rema_50100", "rema_50100150").
        confirmed_vol_threshold: vol_trend >= this at flip bar → is_confirmed=True.
                                 Default 1.2 (20% above 20-bar average).
        min_post_bars:           Minimum post-flip bars required to set is_sustained.
                                 Default 3.

    Returns:
        MAVolumeProfile with all fields populated.

    Raises:
        ValueError: if 'volume' column is missing.
        ValueError: if signal_col column is missing.
        ValueError: if fewer than 2 bars are provided (cannot detect a flip).
    """
    # --- 1. Column validation ---
    required = _REQUIRED_COLS_BASE | {signal_col}
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"assess_ma_volume: DataFrame is missing required column '{col}'. "
                f"Expected columns: {sorted(required)}."
            )

    # --- 2. Minimum bars ---
    if len(df) < 2:
        raise ValueError(
            "assess_ma_volume requires at least 2 bars to detect a signal flip; "
            f"got {len(df)}."
        )

    # --- 3. vol_trend series ---
    vol_trend: pd.Series = (
        df["volume"] / df["volume"].rolling(20, min_periods=1).mean()
    ).reset_index(drop=True)

    signal = df[signal_col].reset_index(drop=True).tolist()
    n      = len(signal)

    # --- 4. Is the last bar a flip? ---
    last_is_flip = (signal[-1] != signal[-2])

    # --- 5. vol_on_crossover and is_confirmed ---
    if last_is_flip:
        vt_now           = float(vol_trend.iloc[-1])
        vol_on_crossover = round(vt_now, 4)
        is_confirmed     = bool(vt_now >= confirmed_vol_threshold)
    else:
        vol_on_crossover = None
        is_confirmed     = None

    # --- 6. Find the most recent flip to compute post-flip vol stats ---
    flip_idx = _find_last_flip(signal)

    if flip_idx is None:
        return MAVolumeProfile(
            vol_on_crossover    = vol_on_crossover,
            vol_trend_mean_post = None,
            is_confirmed        = is_confirmed,
            is_sustained        = None,
        )

    # --- 7. Post-flip bars (flip_idx+1 onward) ---
    post_start = flip_idx + 1
    post_vt    = vol_trend.iloc[post_start : post_start + min_post_bars]
    n_post     = len(post_vt)

    if n_post < min_post_bars:
        return MAVolumeProfile(
            vol_on_crossover    = vol_on_crossover,
            vol_trend_mean_post = None,
            is_confirmed        = is_confirmed,
            is_sustained        = None,
        )

    # --- 8–9. vol_trend_mean_post and is_sustained ---
    mean_post    = float(post_vt.mean())
    is_sustained = bool(mean_post >= 1.0)

    return MAVolumeProfile(
        vol_on_crossover    = vol_on_crossover,
        vol_trend_mean_post = round(mean_post, 4),
        is_confirmed        = is_confirmed,
        is_sustained        = is_sustained,
    )
