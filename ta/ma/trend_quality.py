"""
trend_quality.py — Analytical primitives for the MA crossover trader.

Public API:

    compute_rsi(rclose, period=14)
        Wilder's RSI on a relative close price series.
        Returns the last-bar RSI as a float in [0, 100].

    compute_adx(df, period=14)
        Wilder's Average Directional Index using rhigh/rlow/rclose.
        Returns the last-bar ADX as a float in [0, 100].
        A value > 25 (and rising) signals a trend with institutional momentum.

    compute_ma_gap_pct(fast_ma, slow_ma, rclose)
        Percentage spread between fast and slow MA relative to rclose.
        Acts as a MACD proxy using the MA level columns already in the parquet
        (rema_short_50 vs rema_long_150) — no new formula needed.
        Positive = bullish alignment, negative = bearish.

    compute_ma_slope_pct(ma_series, window)
        OLS slope of an MA series over `window` bars, normalised by the mean
        and expressed as %/day.  Reuses ta.utils.ols_slope.

    assess_ma_trend(df)
        Integrates all four primitives into a single MATrendStrength snapshot.
        Preferred MA pair: rema_short_50 (fast) vs rema_long_150 (slow).
        is_trending = adx > ADX_TREND_THRESHOLD and adx_slope > 0.

All functions are:
    - Pure (no I/O, no side-effects).
    - Typed (PEP 484).
    - Independently testable with synthetic pd.Series / pd.DataFrame.
    - Documented with invariants and failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ta.utils import ols_slope

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ADX_TREND_THRESHOLD: float = 25.0   # ADX above this = trend has institutional strength
ADX_SLOPE_WINDOW:    int   = 14     # bars used to compute adx_slope
MA_SLOPE_WINDOW:     int   = 20     # bars used to compute ma_gap_slope

_RSI_REQUIRED_COLS = frozenset({"rclose"})
_ADX_REQUIRED_COLS = frozenset({"rclose", "rhigh", "rlow"})
_TREND_REQUIRED_COLS = frozenset({
    "rclose", "rhigh", "rlow",
    "rema_short_50", "rema_medium_100", "rema_long_150",
})


# ===========================================================================
# Primitive 1 — RSI (Wilder's smoothing)
# ===========================================================================


def compute_rsi(rclose: pd.Series, period: int = 14) -> float:
    """
    Compute Wilder's RSI on the last bar of a relative close price series.

    Uses Wilder's exponential smoothing (alpha = 1/period) for average gain
    and average loss, seeded with the simple mean over the first `period` bars.

    Args:
        rclose: Relative close price series (rclose = ticker / FTSEMIB.MI).
                Must have at least period+1 non-NaN values.
        period: RSI lookback window. Default 14.

    Returns:
        RSI value at the last bar, in [0.0, 100.0].

    Raises:
        ValueError: if fewer than period+1 non-NaN values are present.

    Invariants:
        - Constant series → RSI = 50 (no gains, no losses → avg_gain = avg_loss = 0).
        - All-gain series → RSI approaches 100 as the series lengthens.
        - All-loss series → RSI approaches 0.
    """
    clean = rclose.dropna()
    if len(clean) < period + 1:
        raise ValueError(
            f"compute_rsi requires at least period+1 = {period + 1} non-NaN values; "
            f"got {len(clean)}. Provide a longer series or reduce the period."
        )

    closes = clean.to_numpy(dtype=float)
    deltas = np.diff(closes)

    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple mean over the first `period` changes.
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    # Wilder's smoothing over the remaining bars.
    alpha = 1.0 / period
    for i in range(period, len(deltas)):
        avg_gain = alpha * gains[i]  + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0

    if avg_loss == 0.0:
        return 100.0

    rs  = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(float(rsi), 4)


# ===========================================================================
# Primitive 2 — ADX (Wilder's DMI)
# ===========================================================================


def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute Wilder's Average Directional Index (ADX) on relative OHLC data.

    ADX measures the strength of a trend, not its direction:
    - ADX > 25 (and rising) = trend is gaining institutional momentum.
    - ADX < 20              = weak or no trend; MA crossover may be unreliable.

    Algorithm (Wilder):
        1. True Range (TR)   = max(rhigh-rlow, |rhigh-prev_rclose|, |rlow-prev_rclose|)
        2. +DM = max(rhigh - prev_rhigh, 0) if > max(prev_rlow - rlow, 0), else 0
        3. -DM = max(prev_rlow - rlow, 0) if > max(rhigh - prev_rhigh, 0), else 0
        4. Wilder-smooth TR, +DM, -DM over `period` bars.
        5. +DI = 100 * (+DM_smooth / TR_smooth), -DI symmetric.
        6. DX  = 100 * |+DI - -DI| / (+DI + -DI)
        7. ADX = Wilder-smooth DX over `period` bars.

    Args:
        df:     Single-ticker DataFrame sorted ascending by date.
                Must contain columns: rclose, rhigh, rlow.
        period: Smoothing window. Default 14.

    Returns:
        ADX value at the last bar, in [0.0, 100.0].

    Raises:
        ValueError: if any required column is missing.
        ValueError: if fewer than 2*period+1 bars are available (insufficient warm-up).

    Failure modes:
        - Flat series (rhigh == rlow all bars): TR = 0 → +DI = -DI = 0 → ADX = 0.
    """
    missing = _ADX_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_adx: DataFrame missing required columns: {sorted(missing)}. "
            f"Expected: {sorted(_ADX_REQUIRED_COLS)}."
        )

    required_bars = 2 * period + 1
    n = len(df)
    if n < required_bars:
        raise ValueError(
            f"compute_adx requires at least 2*period+1 = {required_bars} bars; "
            f"got {n}. Provide a longer history or reduce the period."
        )

    high   = df["rhigh"].to_numpy(dtype=float)
    low    = df["rlow"].to_numpy(dtype=float)
    close  = df["rclose"].to_numpy(dtype=float)

    m = n - 1  # number of change bars

    # --- Step 1: True Range ---
    tr   = np.empty(m)
    p_dm = np.empty(m)
    n_dm = np.empty(m)

    for i in range(m):
        h, l, pc = high[i + 1], low[i + 1], close[i]
        ph, pl   = high[i], low[i]

        tr[i] = max(h - l, abs(h - pc), abs(l - pc))

        up_move   = h - ph
        down_move = pl - l

        if up_move > down_move and up_move > 0:
            p_dm[i] = up_move
        else:
            p_dm[i] = 0.0

        if down_move > up_move and down_move > 0:
            n_dm[i] = down_move
        else:
            n_dm[i] = 0.0

    # --- Step 2: Wilder smoothing of TR, +DM, -DM ---
    def _wilder_smooth(arr: np.ndarray) -> np.ndarray:
        """Seed with sum of first `period` bars; then Wilder's rolling sum."""
        result = np.empty(len(arr) - period + 1)
        result[0] = arr[:period].sum()
        for j in range(1, len(result)):
            result[j] = result[j - 1] - result[j - 1] / period + arr[period + j - 1]
        return result

    tr_s   = _wilder_smooth(tr)
    p_dm_s = _wilder_smooth(p_dm)
    n_dm_s = _wilder_smooth(n_dm)

    # --- Step 3: +DI, -DI, DX ---
    with np.errstate(divide="ignore", invalid="ignore"):
        p_di = np.where(tr_s != 0, 100.0 * p_dm_s / tr_s, 0.0)
        n_di = np.where(tr_s != 0, 100.0 * n_dm_s / tr_s, 0.0)
        di_sum  = p_di + n_di
        di_diff = np.abs(p_di - n_di)
        dx = np.where(di_sum != 0, 100.0 * di_diff / di_sum, 0.0)

    # --- Step 4: Wilder smooth DX → ADX ---
    if len(dx) < period:
        return 0.0

    adx_seed = dx[:period].mean()
    adx_val  = adx_seed
    alpha    = 1.0 / period
    for i in range(period, len(dx)):
        adx_val = alpha * dx[i] + (1 - alpha) * adx_val

    return round(float(np.clip(adx_val, 0.0, 100.0)), 4)


# ===========================================================================
# Primitive 3 — MA gap % (MACD proxy)
# ===========================================================================


def compute_ma_gap_pct(fast_ma: float, slow_ma: float, rclose: float) -> float:
    """
    Compute the percentage spread between fast and slow MA, normalised by rclose.

    Acts as a MACD-line proxy:
        gap = (fast_ma - slow_ma) / rclose * 100

    Positive value = fast MA above slow MA = bullish alignment.
    Negative value = fast MA below slow MA = bearish alignment.
    Widening gap (rising ma_gap_slope) = trend accelerating.
    Narrowing gap (falling ma_gap_slope) = trend losing steam.

    Args:
        fast_ma: Fast MA value at the last bar (e.g. rema_short_50).
        slow_ma: Slow MA value at the last bar (e.g. rema_long_150).
        rclose:  Relative close price at the last bar. Must be > 0.

    Returns:
        Gap as a signed float (%); rounded to 4 decimal places.

    Raises:
        ValueError: if rclose <= 0 (undefined ratio).
    """
    if rclose <= 0:
        raise ValueError(
            f"compute_ma_gap_pct: rclose must be > 0; got {rclose}. "
            "Relative close price cannot be zero or negative."
        )
    return round(float((fast_ma - slow_ma) / rclose * 100), 4)


# ===========================================================================
# Primitive 4 — MA slope %/day (trend momentum)
# ===========================================================================


def compute_ma_slope_pct(ma_series: pd.Series, window: int) -> float:
    """
    Compute the OLS slope of an MA level series over `window` bars.

    Normalises by mean(ma_series[-window:]) and expresses as %/day — the same
    normalisation as classify_trend() in ta.breakout.range_quality, making
    slopes comparable across tickers regardless of absolute price level.

    Args:
        ma_series: MA level values (e.g. rema_short_50 column), sorted ascending.
                   Must contain at least 2 non-NaN values in the last `window` bars.
        window:    Number of recent bars to use. Series is sliced to its tail.

    Returns:
        Slope as %/day; signed. Positive = MA rising, negative = falling.
        Returns 0.0 if fewer than 2 non-NaN values after slicing (delegates to ols_slope).

    Raises:
        No exceptions raised (ols_slope returns 0.0 for < 2 values).
    """
    tail = ma_series.iloc[-window:].dropna()
    if len(tail) < 2:
        return 0.0
    y      = tail.to_numpy(dtype=float)
    y_mean = y.mean()
    if y_mean == 0.0:
        return 0.0
    slope_pct = ols_slope(y) / y_mean * 100
    return round(float(slope_pct), 6)


# ===========================================================================
# Primitive 5 — Integrated trend strength snapshot
# ===========================================================================


@dataclass(frozen=True)
class MATrendStrength:
    """
    Snapshot of MA crossover trend quality at the last bar of a ticker series.

    Integrates RSI, ADX, and MA gap/slope into one cohesive object.

    Attributes:
        rsi:          Wilder's RSI (14-period) on rclose. In [0, 100].
                      > 50 and rising = bullish momentum.
                      < 50 and falling = bearish momentum.
        adx:          ADX at the last bar. In [0, 100].
                      > 25 = trend has institutional strength.
        adx_slope:    OLS slope of ADX over the last ADX_SLOPE_WINDOW bars (%/bar).
                      Positive = ADX rising = trend gaining strength.
                      Negative = ADX falling = trend weakening.
        ma_gap_pct:   (rema_short_50 - rema_long_150) / rclose * 100 at last bar.
                      MACD-line proxy. Positive = bullish alignment.
        ma_gap_slope: OLS slope of ma_gap_pct over MA_SLOPE_WINDOW bars (%/bar).
                      Widening gap = trend accelerating.
                      Narrowing gap = trend losing steam.
        is_trending:  True when adx > ADX_TREND_THRESHOLD AND adx_slope >= 0.
                      Conditions: the trend must be strong AND not actively declining.
                      adx_slope > 0 = fresh/gaining trend.
                      adx_slope == 0 = trend stable at a high level (still valid).
                      adx_slope < 0 = trend weakening — caution, not a fresh entry.
    """

    rsi:          float
    adx:          float
    adx_slope:    float
    ma_gap_pct:   float
    ma_gap_slope: float
    is_trending:  bool


_ASSESS_REQUIRED_COLS = _TREND_REQUIRED_COLS


def assess_ma_trend(df: pd.DataFrame) -> MATrendStrength:
    """
    Assess MA crossover trend quality for a single ticker.

    Uses the full ticker history for RSI and ADX computation, then slices
    the last ADX_SLOPE_WINDOW / MA_SLOPE_WINDOW bars for slope estimates.

    Algorithm:
        1. Validate required columns.
        2. Compute RSI on rclose.
        3. Compute ADX on rhigh/rlow/rclose.
        4. Compute adx_slope: OLS of ADX values over last ADX_SLOPE_WINDOW bars.
           ADX series is re-computed per-bar by rolling the Wilder smoother.
           For simplicity, adx_slope is estimated from the daily ADX values
           computed via a vectorised rolling approach.
        5. Compute ma_gap_pct at last bar: fast=rema_short_50, slow=rema_long_150.
        6. Compute ma_gap_slope: OLS of the ma_gap_pct series over MA_SLOPE_WINDOW bars.
        7. is_trending = adx > ADX_TREND_THRESHOLD AND adx_slope > 0.

    Args:
        df: Full single-ticker DataFrame sorted ascending by date.
            Must contain: rclose, rhigh, rlow, rema_short_50, rema_medium_100,
            rema_long_150.

    Returns:
        MATrendStrength with all fields populated.

    Raises:
        ValueError: if any required column is missing.
        ValueError: propagated from compute_rsi or compute_adx on insufficient history.
    """
    missing = _ASSESS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"assess_ma_trend: DataFrame missing required columns: {sorted(missing)}. "
            f"Expected: {sorted(_ASSESS_REQUIRED_COLS)}."
        )

    # --- RSI on full rclose series ---
    rsi = compute_rsi(df["rclose"], period=14)

    # --- ADX at the last bar ---
    adx = compute_adx(df, period=14)

    # --- adx_slope: computed from the rolling ADX series over recent bars ---
    # Re-use compute_adx on a rolling tail to approximate the ADX time series.
    # We compute ADX on [df[-window-pad:]] slices to build a short slope series.
    adx_slope = _compute_adx_slope(df, slope_window=ADX_SLOPE_WINDOW, period=14)

    # --- MA gap at last bar ---
    last = df.iloc[-1]
    ma_gap_pct = compute_ma_gap_pct(
        fast_ma=float(last["rema_short_50"]),
        slow_ma=float(last["rema_long_150"]),
        rclose=float(last["rclose"]),
    )

    # --- MA gap slope: OLS over last MA_SLOPE_WINDOW bars of the gap series ---
    gap_series = (
        (df["rema_short_50"] - df["rema_long_150"]) / df["rclose"] * 100
    ).dropna()
    ma_gap_slope = compute_ma_slope_pct(gap_series, window=MA_SLOPE_WINDOW)

    is_trending = bool(adx > ADX_TREND_THRESHOLD and adx_slope >= 0.0)

    return MATrendStrength(
        rsi          = rsi,
        adx          = adx,
        adx_slope    = round(adx_slope, 6),
        ma_gap_pct   = ma_gap_pct,
        ma_gap_slope = ma_gap_slope,
        is_trending  = is_trending,
    )


def _compute_adx_slope(df: pd.DataFrame, slope_window: int, period: int) -> float:
    """
    Estimate the slope of the ADX series over the last `slope_window` bars.

    Builds the ADX series by computing ADX on rolling windows of the DataFrame,
    then fits OLS to the resulting values.  The minimum warmup required for each
    ADX computation is 2*period+1 bars, so the overall minimum df length is
    2*period+1+slope_window bars.

    If insufficient data is available, returns 0.0 (no trend acceleration signal).
    """
    warmup = 2 * period + 1
    min_len = warmup + slope_window

    if len(df) < min_len:
        return 0.0

    adx_values = []
    for i in range(slope_window, 0, -1):
        # Use df up to (and including) the bar that is `i` bars before the last.
        slice_end = len(df) - i + 1
        if slice_end < warmup:
            continue
        adx_i = compute_adx(df.iloc[:slice_end], period=period)
        adx_values.append(adx_i)

    # Include the current last-bar ADX
    adx_values.append(compute_adx(df, period=period))

    if len(adx_values) < 2:
        return 0.0

    return ols_slope(np.array(adx_values, dtype=float))
