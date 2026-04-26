"""
swing_range_quality.py — Range-quality assessment using RegimeFC structural levels.

Problem with rhi_20 / rlo_20 as reference levels:
    They are mechanical 20-day rolling highs/lows.  A multi-day stall at the same
    price narrows the band without a genuine structural level being present.

What RegimeFC adds:
    _regime_floor_ceiling() discovers floors/ceilings from confirmed swing structure
    (alternating highs and lows filtered by ATR distance and retracement depth).
    These structural levels — stored as rclg / rflr in analysis_results.parquet —
    represent genuine price memory: the market has already tested and confirmed them
    at least once.

This module is a thin adapter.  It:
    1. Forward-fills the sparse rclg / rflr columns (event-based, 4–5 non-NaN per
       2 600 bars) so every bar has a structural reference.
    2. Calls assess_range() and measure_volatility_compression() — unchanged — via
       their existing rhi_col / rlo_col parameters.
    3. Keeps rbo_20 as the consolidation-window selector (rrg has no 0-state).

No new signal logic.  All primitives remain in range_quality.py.

Public API:
    assess_swing_range(df, window_bars, config)
        Run assess_range using structural swing ceiling/floor as reference levels.

    measure_swing_volatility(df, window_bars, history_bars, config)
        Run measure_volatility_compression using structural swing ceiling/floor.

Column contract (all present in analysis_results.parquet):
    rclg   — RegimeFC ceiling (relative price), sparse — written by _regime_floor_ceiling
    rflr   — RegimeFC floor  (relative price), sparse — written by _regime_floor_ceiling
    rrg    — Regime signal (+1 / -1), dense
    rbo_20 — Range-breakout signal (+1 / 0 / -1) — used as consolidation selector
    rclose — Relative close price
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ta.breakout.range_quality import (
    RangeQualityConfig,
    RangeSetup,
    VolatilityState,
    assess_range,
    measure_volatility_compression,
)

log = logging.getLogger(__name__)

# Column names written by RegimeFC._regime_floor_ceiling (relative=True, lowercase)
_CLG_COL = "rclg"  # structural ceiling — sparse, needs ffill
_FLR_COL = "rflr"  # structural floor   — sparse, needs ffill

# Derived column names added by _prepare_swing_levels (never written to the parquet)
_SWING_HI = "swing_hi"
_SWING_LO = "swing_lo"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_swing_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add swing_hi and swing_lo columns by forward-filling rclg / rflr.

    rclg and rflr are event-driven sparse columns: only the bar where a new
    ceiling/floor is confirmed carries a value.  Forward-filling carries each
    confirmed level forward until a new one is discovered — exactly how a
    trader would use a structural swing level.

    Args:
        df: Single-ticker DataFrame from analysis_results.parquet.
            Must contain rclg and rflr.

    Returns:
        Copy of df with swing_hi and swing_lo added.

    Raises:
        ValueError: if rclg or rflr are missing, or if all values are NaN
                    after ffill+bfill (no regime history for this ticker).
    """
    required = {_CLG_COL, _FLR_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"_prepare_swing_levels: columns {sorted(missing)} not found. "
            f"Run RegimeFC.compute_regime() first, or use analysis_results.parquet "
            f"which already contains {_CLG_COL} and {_FLR_COL}."
        )

    out = df.copy()

    # ffill carries each confirmed level forward; bfill seeds the start of history.
    out[_SWING_HI] = out[_CLG_COL].ffill().bfill()
    out[_SWING_LO] = out[_FLR_COL].ffill().bfill()

    if out[_SWING_HI].isna().all():
        raise ValueError(
            f"_prepare_swing_levels: {_CLG_COL} is all-NaN — "
            "no ceiling has been discovered for this ticker. "
            "The ticker may have too short a history for _regime_floor_ceiling to fire."
        )
    if out[_SWING_LO].isna().all():
        raise ValueError(
            f"_prepare_swing_levels: {_FLR_COL} is all-NaN — "
            "no floor has been discovered for this ticker. "
            "The ticker may have too short a history for _regime_floor_ceiling to fire."
        )

    n_swing_hi = int(df[_CLG_COL].notna().sum())
    n_swing_lo = int(df[_FLR_COL].notna().sum())
    log.debug(
        "_prepare_swing_levels: %d ceiling events, %d floor events over %d bars.",
        n_swing_hi, n_swing_lo, len(df),
    )

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_swing_range(
    df:          pd.DataFrame,
    window_bars: int = 40,
    config:      RangeQualityConfig | None = None,
    rbo_col:     str = "rbo_20",
    rclose_col:  str = "rclose",
) -> RangeSetup:
    """
    Run assess_range using RegimeFC structural swing ceiling/floor as reference levels.

    Difference from assess_range(df):
        assess_range uses rhi_20/rlo_20 (20-day rolling band) as resistance/support.
        This function uses rclg_ff / rflr_ff (RegimeFC confirmed swing levels, forward-filled)
        as resistance/support — levels that have been validated by the alternating
        swing structure and ATR-distance filtering in _regime_floor_ceiling.

    What stays identical:
        - Consolidation window detection: rbo_20 zero-run, same as assess_range.
        - Touch counting state machine: count_touches, same thresholds.
        - Sideways classification: classify_trend, same OLS slope logic.
        - All invariants and failure modes of assess_range.

    Args:
        df:          Full single-ticker DataFrame. Must contain rclg, rflr, rbo_20, rclose.
        window_bars: Maximum zero-run bars used for touch counting. Default 40.
        config:      RangeQualityConfig for threshold overrides. Defaults to module defaults.
        rbo_col:     Consolidation signal column. Default 'rbo_20'.
        rclose_col:  Relative close column. Default 'rclose'.

    Returns:
        RangeSetup — same shape as assess_range(), but counts reflect tests against
        structural swing levels rather than the 20-day rolling band.

    Raises:
        ValueError: if rclg/rflr missing, all-NaN, or any assess_range invariant violated.

    Interpretation of results vs assess_range:
        n_resistance_touches — how many times rclose reached ≥ 85% of the
            swing ceiling band (structural resistance) and then retreated.
        n_support_touches    — symmetric for structural floor.
        band_width_pct       — (swing_ceiling - swing_floor) / rclose × 100
            at the last consolidation bar.  Wider than band_width from rhi_20/rlo_20
            because structural levels span a longer historical window.
    """
    df_sw = _prepare_swing_levels(df)

    return assess_range(
        df_sw,
        window_bars = window_bars,
        config      = config,
        rbo_col     = rbo_col,
        rhi_col     = _SWING_HI,
        rlo_col     = _SWING_LO,
        rclose_col  = rclose_col,
    )


def measure_swing_volatility(
    df:            pd.DataFrame,
    window_bars:   int   = 40,
    history_bars:  int   = 252,
    config:        RangeQualityConfig | None = None,
    rclose_col:    str   = "rclose",
) -> VolatilityState:
    """
    Run measure_volatility_compression using RegimeFC swing levels as the band.

    Difference from measure_volatility_compression(df):
        The band is (swing_ceiling - swing_floor) / rclose rather than
        (rhi_20 - rlo_20) / rclose.  Structural band width compresses more slowly
        and represents a longer-term volatility regime — a useful complement to the
        fast 20-day band.

    Args:
        df:           Full single-ticker DataFrame. Must contain rclg, rflr, rclose.
        window_bars:  Bars used to compute band_width_slope. Default 40.
        history_bars: Bars used for band_width_pct_rank. Default 252.
        config:       RangeQualityConfig for threshold overrides.
        rclose_col:   Relative close column. Default 'rclose'.

    Returns:
        VolatilityState — same shape as measure_volatility_compression().
        band_width_pct reflects the structural swing range, not the 20-day band.

    Raises:
        ValueError: if rclg/rflr missing or all-NaN.
    """
    rank_threshold = config.compression_rank_threshold if config else 25.0

    df_sw = _prepare_swing_levels(df)

    return measure_volatility_compression(
        df_sw,
        window_bars    = window_bars,
        history_bars   = history_bars,
        rank_threshold = rank_threshold,
        rhi_col        = _SWING_HI,
        rlo_col        = _SWING_LO,
        rclose_col     = rclose_col,
    )
