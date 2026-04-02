"""
range_quality.py — Foundational analytical primitives for the range breakout trader.

Public API:

    count_touches(range_pct)
        How many distinct times price approached resistance / support without breaking through.
        Uses range_position_pct thresholds with a state machine so a prolonged approach
        near a level counts as one touch, not many.

    classify_trend(rclose)
        Whether relative price is moving sideways over a window.
        Uses OLS slope normalised by mean(rclose), expressed as %/day.
        More robust than rclose_chg_Nd which only compares first and last bar.

    breakout_prior_consolidation_length(rbo)  [backtesting utility, not used in snapshot]
        Pre-breakout consolidation length at each historical flip bar.
        For the live snapshot use rbo_N_age from the parquet directly.

    measure_volatility_compression(df)
        Whether the 20-day relative range envelope is actively narrowing.
        Returns a VolatilityState dataclass with band_width_pct, OLS slope,
        historical percentile rank, and a composite is_compressed flag.

    assess_range(df, window_bars=40)
        Integrates all three range-quality primitives into a single RangeSetup snapshot.
        Returns n_resistance/support touches, sideways classification, slope,
        consolidation length, and band width at the last consolidation bar.

All functions are:
    - Pure (no I/O, no side-effects).
    - Typed (PEP 484).
    - Independently testable with synthetic pd.Series / pd.DataFrame.
    - Documented with invariants and failure modes.

See range_breakout_trader.md §"Foundational analytical questions" for derivations
and the SCM.MI smoke-test values used in tests/test_range_quality.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from breakout.utils import ols_slope as _ols_slope  # shared; avoids duplicated math

# ---------------------------------------------------------------------------
# Module-level constants (exposed so callers can reference thresholds by name)
# ---------------------------------------------------------------------------

TOUCH_HI_THRESHOLD: float = 85.0   # range_pct >= this -> "at resistance"
TOUCH_LO_THRESHOLD: float = 15.0   # range_pct <= this -> "at support"
RETREAT_THRESHOLD:  float = 65.0   # must drop below this to close a resistance touch
BOUNCE_THRESHOLD:   float = 35.0   # must rise above this to close a support touch

SIDEWAYS_SLOPE_THRESHOLD: float = 0.15   # |slope_pct_per_day| below this -> sideways
MIN_TREND_BARS:           int   = 5      # minimum bars for a reliable OLS estimate

COMPRESSION_RANK_THRESHOLD: float = 25.0  # band_width_pct_rank below this -> historically tight


# ===========================================================================
# Primitive 1 — Touch counting
# ===========================================================================


def count_touches(
    range_pct: pd.Series,
    hi_thresh: float = TOUCH_HI_THRESHOLD,
    lo_thresh: float = TOUCH_LO_THRESHOLD,
    retreat:   float = RETREAT_THRESHOLD,
    bounce:    float = BOUNCE_THRESHOLD,
) -> tuple[int, int]:
    """
    Count distinct resistance and support touch events within a consolidation window.

    A resistance touch event starts when range_pct >= hi_thresh and ends when
    range_pct drops below retreat.  A new touch is only counted after the previous
    one has ended (no double-counting of a prolonged approach).  Support touch events
    are symmetric.

    The gray zones [retreat, hi_thresh) for resistance and (lo_thresh, bounce] for
    support keep the state machine inside the current touch without triggering a new
    count.  This models the real-world pattern where price lingers near a level for
    several bars before finally retreating — which is one touch, not many.

    Args:
        range_pct: range_position_pct series.
                   0%   = rclose is at rlo_N (support).
                   100% = rclose is at rhi_N (resistance).
                   >100% or <0% = price outside the range (active breakout).
                   NaN values are skipped; state machine continues across gaps.
        hi_thresh: Entry threshold for a resistance touch event. Default 85.0.
        lo_thresh: Entry threshold for a support touch event. Default 15.0.
        retreat:   Exit threshold for a resistance touch (must drop below). Default 65.0.
        bounce:    Exit threshold for a support touch (must rise above). Default 35.0.

    Returns:
        (n_resistance_touches, n_support_touches)

    Raises:
        ValueError: if retreat >= hi_thresh (no gray zone for resistance), or
                    bounce <= lo_thresh (no gray zone for support).

    Invariants:
        - Two consecutive bars at 90% count as ONE touch event.
        - A bar at 70% between two 90% bars does NOT split them (70 > retreat=65).
        - A bar at 60% between two 90% bars DOES split them (60 < retreat=65).
    """
    if retreat >= hi_thresh:
        raise ValueError(
            f"retreat ({retreat}) must be strictly less than hi_thresh ({hi_thresh}). "
            "The gray zone [retreat, hi_thresh) would not exist."
        )
    if bounce <= lo_thresh:
        raise ValueError(
            f"bounce ({bounce}) must be strictly greater than lo_thresh ({lo_thresh}). "
            "The gray zone (lo_thresh, bounce] would not exist."
        )

    n_res        = 0
    n_sup        = 0
    in_res_touch = False
    in_sup_touch = False

    for val in range_pct:
        if pd.isna(val):
            continue

        # --- Resistance state machine ---
        # Enter a new touch only when not already in one.
        # Stay inside the touch (gray zone) until price retreats below `retreat`.
        if not in_res_touch:
            if val >= hi_thresh:
                in_res_touch = True
                n_res += 1
        else:
            if val < retreat:
                in_res_touch = False

        # --- Support state machine (symmetric) ---
        if not in_sup_touch:
            if val <= lo_thresh:
                in_sup_touch = True
                n_sup += 1
        else:
            if val > bounce:
                in_sup_touch = False

    return n_res, n_sup


# ===========================================================================
# Primitive 2 — Sideways classification
# ===========================================================================


def classify_trend(
    rclose:    pd.Series,
    threshold: float = SIDEWAYS_SLOPE_THRESHOLD,
) -> tuple[bool, float]:
    """
    Classify whether relative price is moving sideways within a window.

    Uses OLS linear regression of rclose vs bar index via _ols_slope().
    The slope is normalised by mean(rclose) within the window and expressed as
    percentage per day, making it comparable across tickers regardless of their
    absolute relative price level.

    Why not rclose_chg_Nd:
        rclose_chg_Nd only compares the first and last bar and ignores the path.
        A stock that oscillates +3%/-3%/+3% has chg~0% but is not consolidating.
        The OLS slope captures the full trajectory.

    Args:
        rclose:    Relative close price series for the window, sorted ascending by date.
                   Must contain at least MIN_TREND_BARS (5) non-NaN values.
        threshold: Maximum |slope_pct_per_day| to be classified as sideways. Default 0.15.

    Returns:
        (is_sideways, slope_pct_per_day)
        slope_pct_per_day is signed:
            > 0   uptrend (rclose rising on average)
            < 0   downtrend (rclose falling on average)
            ~0    sideways / consolidation
        Value is rounded to 4 decimal places.

    Raises:
        ValueError: if rclose has fewer than MIN_TREND_BARS non-NaN values.

    Failure modes:
        - Constant series (all values equal): slope = 0, is_sideways = True. Correct.
        - Very short windows (< 5 bars): raises ValueError to prevent noisy estimates.
    """
    clean = rclose.dropna()
    if len(clean) < MIN_TREND_BARS:
        raise ValueError(
            f"classify_trend requires at least {MIN_TREND_BARS} non-NaN bars; "
            f"received {len(clean)}. "
            "Provide a longer window or check for excessive NaN values."
        )

    y = clean.to_numpy(dtype=float)
    y_mean = y.mean()

    # Normalise raw OLS slope by mean relative price level -> %/day
    slope_pct_per_day = _ols_slope(y) / y_mean * 100
    is_sideways = bool(abs(slope_pct_per_day) < threshold)

    return is_sideways, round(float(slope_pct_per_day), 4)


# ===========================================================================
# Primitive 3 — Consolidation age
# ===========================================================================


def breakout_prior_consolidation_length(rbo: pd.Series) -> pd.Series:
    """
    Return the pre-breakout consolidation length at each historical breakout flip bar.

    BACKTESTING UTILITY — not used in the live snapshot.

    For the live snapshot, the current consolidation length is simply
    `rbo_N_age` at the last bar (already present in analysis_results.parquet).
    This function is for retrospective analysis: scanning all historical flip bars
    to ask "how long did that consolidation last before it broke out?"

    At the bar where rbo transitions from 0 to a non-zero value (breakout flip),
    this returns the number of consecutive bars that rbo held 0 BEFORE the flip.
    All other bars return NaN.

    Why the shift is necessary:
        rbo_N_age resets to 1 on the flip bar itself. Reading rbo_N_age at the
        breakout bar gives 1, not the consolidation length. This function shifts
        the age series by one to recover the pre-breakout count at the flip bar.

    Args:
        rbo: Breakout signal series (+1 / 0 / -1) for a single ticker, sorted
             ascending by date.

    Returns:
        pd.Series with the same index as rbo.
        - Breakout flip bars (rbo transitions 0 -> non-zero): prior consolidation length.
        - All other bars: NaN.

    Invariants:
        - The returned value at any flip bar is always >= 1.
        - 0 -> -1 (bearish breakout) is counted symmetrically.

    Example:
        rbo    = [0, 0, 0, 0, 0, 1, 1]
        result = [NaN, NaN, NaN, NaN, NaN, 5, NaN]
    """
    groups = (rbo != rbo.shift()).cumsum()
    age    = rbo.groupby(groups).cumcount() + 1

    is_breakout_flip = (rbo != 0) & (rbo.shift(1) == 0)
    prior_age        = age.shift(1)

    return prior_age.where(is_breakout_flip)


# ===========================================================================
# Primitive 4 — Volatility compression
# ===========================================================================


@dataclass(frozen=True)
class VolatilityState:
    """
    Snapshot of the volatility compression state at the last bar of a ticker series.

    Attributes:
        band_width_pct:       Current (rhi_20 - rlo_20) / rclose * 100.
                              Normalised range envelope width in %.
                              Low value = price confined to a narrow band.
        band_width_slope:     OLS slope of band_width_pct over the last window_bars bars,
                              in raw %/bar (not normalised by mean).
                              Negative = range is actively narrowing = energy building.
                              Positive = range expanding = consolidation may be dissolving.
        band_width_pct_rank:  Percentile rank of band_width_pct vs the last history_bars.
                              Range [0, 100]. Low rank = historically compressed for this ticker.
        is_compressed:        True when band_width_slope < 0 AND band_width_pct_rank < 25.
                              Both conditions required: the range must be actively narrowing
                              AND already in the bottom quartile of its own history.
    """

    band_width_pct:       float
    band_width_slope:     float
    band_width_pct_rank:  float
    is_compressed:        bool


_REQUIRED_COLS = frozenset({"rhi_20", "rlo_20", "rclose"})


def measure_volatility_compression(
    df:           pd.DataFrame,
    window_bars:  int = 40,
    history_bars: int = 252,
) -> VolatilityState:
    """
    Measure whether the 20-day relative range envelope is actively compressing.

    Computes three independent signals and combines them into a VolatilityState:

    1. band_width_pct = (rhi_20 - rlo_20) / rclose * 100 at the last bar.
       Already normalised for ticker price level — directly comparable across tickers.

    2. band_width_slope = OLS slope of band_width_pct over the last window_bars clean bars.
       Uses the same _ols_slope() helper as classify_trend() — no duplicated math.
       A negative slope means the 20-day envelope is getting tighter bar by bar.

    3. band_width_pct_rank = fraction of the last history_bars values that are <=
       the current band_width_pct, expressed as a percentile (0–100).
       Normalises for ticker-specific volatility: a 5% band width on a quiet utility
       stock means something different than on a volatile small-cap.

    is_compressed requires BOTH signals to be true:
       - Actively narrowing (slope < 0): range is in motion toward a breakout.
       - Historically tight (rank < 25): range is not just quiet, it is compressed
         relative to this ticker's own baseline.

    Args:
        df:           Full single-ticker DataFrame sorted ascending by date.
                      Must contain columns: rhi_20, rlo_20, rclose.
        window_bars:  Number of recent bars used to compute band_width_slope.
                      Should match the consolidation window length. Default 40.
        history_bars: Number of recent bars used to compute band_width_pct_rank.
                      Typically 252 (one trading year). Default 252.

    Returns:
        VolatilityState with all fields populated.

    Raises:
        ValueError: if any required column is missing.
        ValueError: if fewer than MIN_TREND_BARS non-NaN rows remain after dropping NaN.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}. "
            f"Expected: {sorted(_REQUIRED_COLS)}."
        )

    # Compute band_width_pct for all rows; drop NaN (from rolling-window warmup).
    bw: pd.Series = (
        (df["rhi_20"] - df["rlo_20"]) / df["rclose"] * 100
    ).dropna()

    if len(bw) < MIN_TREND_BARS:
        raise ValueError(
            f"measure_volatility_compression requires at least {MIN_TREND_BARS} "
            f"non-NaN rows after computing band_width_pct; got {len(bw)}. "
            "Check for insufficient history or excessive NaN in rhi_20/rlo_20/rclose."
        )

    current_bw = float(bw.iloc[-1])

    # --- Signal 1: slope over the recent window ---
    # If window_bars > len(bw), use all available bars (no error — we just have short history).
    bw_window = bw.iloc[-window_bars:].to_numpy(dtype=float)
    band_width_slope = _ols_slope(bw_window)

    # --- Signal 2: percentile rank vs historical baseline ---
    bw_history = bw.iloc[-history_bars:]
    band_width_pct_rank = float((bw_history <= current_bw).mean() * 100)

    is_compressed = bool(band_width_slope < 0 and band_width_pct_rank < COMPRESSION_RANK_THRESHOLD)

    return VolatilityState(
        band_width_pct      = round(current_bw,          4),
        band_width_slope    = round(band_width_slope,     6),
        band_width_pct_rank = round(band_width_pct_rank,  2),
        is_compressed       = is_compressed,
    )


# ===========================================================================
# Primitive 5 — Integrated range setup
# ===========================================================================


@dataclass(frozen=True)
class RangeSetup:
    """
    Snapshot of consolidation quality at the last bar of a single-ticker series.

    Integrates the outputs of count_touches, classify_trend, and band-width
    measurement into one cohesive object.  All fields are derived from the most
    recent consecutive run of rbo_20 == 0, capped at window_bars bars.

    Attributes:
        n_resistance_touches: Distinct times price approached rhi_20 without breaking through.
                              A clean range has 2–4 touches; < 2 = poorly-defined range.
        n_support_touches:    Distinct times price approached rlo_20 without breaking through.
        is_sideways:          True when the OLS slope of rclose over the window is below
                              SIDEWAYS_SLOPE_THRESHOLD (0.15 %/day).
        slope_pct_per_day:    Signed OLS slope of rclose in %/day, normalised by mean(rclose).
                              Positive = rclose drifting up; negative = drifting down; ~0 = flat.
        consolidation_bars:   Full length of the most recent consecutive zero-run, NOT capped
                              by window_bars.  Use this to judge how long price has been coiling.
        band_width_pct:       (rhi_20 - rlo_20) / rclose * 100 at the last rbo_20 == 0 bar.
                              Low value = narrow range = compressed energy.
    """

    n_resistance_touches: int
    n_support_touches:    int
    is_sideways:          bool
    slope_pct_per_day:    float
    consolidation_bars:   int
    band_width_pct:       float


_ASSESS_REQUIRED_COLS = frozenset({"rbo_20", "rclose", "rhi_20", "rlo_20"})


def assess_range(
    df:          pd.DataFrame,
    window_bars: int = 40,
) -> RangeSetup:
    """
    Assess consolidation quality for a single ticker.

    Walks backward from the last row to find the most recent consecutive run of
    rbo_20 == 0 (the current or just-completed consolidation window), caps it at
    window_bars, and runs all three range-quality primitives over the capped slice.

    Algorithm:
        1. Validate required columns: rbo_20, rclose, rhi_20, rlo_20.
        2. Walk backward from the last row, skipping any trailing rbo_20 != 0 bars
           (e.g., the current breakout bar).
        3. Walk backward to find the start of the zero-run.
        4. consolidation_bars = full run length (never capped — always the real age).
        5. Cap the slice to the last window_bars rows of the zero-run.
        6. Compute range_position_pct over the capped window.
        7. count_touches(range_position_pct) → (n_resistance_touches, n_support_touches).
        8. classify_trend(rclose over capped window) → (is_sideways, slope_pct_per_day).
        9. band_width_pct = (rhi_20 - rlo_20) / rclose * 100 at the last zero-run bar.

    Args:
        df:          Full single-ticker DataFrame sorted ascending by date.
                     Must contain columns: rbo_20, rclose, rhi_20, rlo_20.
        window_bars: Maximum zero-run bars used for touch counting and sideways
                     classification. Does NOT affect consolidation_bars. Default 40.

    Returns:
        RangeSetup with all fields populated.

    Raises:
        ValueError: if any required column is missing.
        ValueError: if no rbo_20 == 0 bar exists in the history (no consolidation).
        ValueError: if the capped zero-run has fewer than MIN_TREND_BARS non-NaN bars
                    (raised by classify_trend — prevents noisy estimates on tiny windows).

    Failure modes:
        - Last bar is a breakout (rbo_20 != 0): the preceding zero-run is used automatically.
        - Zero-run shorter than window_bars: all available bars are used without error.
        - Zero-run shorter than MIN_TREND_BARS (5): raises ValueError via classify_trend.
    """
    missing = _ASSESS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}. "
            f"Expected: {sorted(_ASSESS_REQUIRED_COLS)}."
        )

    rbo = df["rbo_20"].to_numpy(dtype=float)
    n   = len(rbo)

    # --- Find the most recent zero-run ---
    # Skip any trailing non-zero bars (active trend / breakout bar).
    end = n - 1
    while end >= 0 and rbo[end] != 0:
        end -= 1

    if end < 0:
        raise ValueError(
            "No consolidation window (rbo_20 == 0) found in the provided history. "
            "The entire series is in a trend or breakout state. "
            "Provide a longer history that includes at least one consolidation period."
        )

    # Walk backward to find the start of the zero-run.
    start = end
    while start > 0 and rbo[start - 1] == 0:
        start -= 1

    consolidation_bars = end - start + 1

    # --- Cap to window_bars for analysis (preserves consolidation_bars) ---
    capped_start = max(start, end - window_bars + 1)
    window_df    = df.iloc[capped_start : end + 1]

    # --- range_position_pct for touch counting ---
    rng       = window_df["rhi_20"] - window_df["rlo_20"]
    range_pct = (
        (window_df["rclose"] - window_df["rlo_20"]) / rng.where(rng != 0) * 100
    ).reset_index(drop=True)

    n_res, n_sup = count_touches(range_pct)

    # --- Sideways classification (raises if < MIN_TREND_BARS clean bars) ---
    is_sideways, slope_pct_per_day = classify_trend(
        window_df["rclose"].reset_index(drop=True)
    )

    # --- Band width at the last consolidation bar (last bar of the zero-run) ---
    last_rhi       = float(df["rhi_20"].iloc[end])
    last_rlo       = float(df["rlo_20"].iloc[end])
    last_rclose    = float(df["rclose"].iloc[end])
    band_width_pct = (last_rhi - last_rlo) / last_rclose * 100

    return RangeSetup(
        n_resistance_touches = n_res,
        n_support_touches    = n_sup,
        is_sideways          = is_sideways,
        slope_pct_per_day    = slope_pct_per_day,
        consolidation_bars   = consolidation_bars,
        band_width_pct       = round(band_width_pct, 4),
    )
