"""
trend_quality.py — Analytical primitives for the MA crossover trader.

Public API
----------
compute_rsi(rclose, period=RSI_PERIOD)
    Wilder's RSI on a relative close price series.
    Returns the last-bar RSI as a float in [0, 100].

compute_adx(df, period=ADX_PERIOD, *, high_col, low_col, close_col)
    Wilder's Average Directional Index using relative OHLC columns.
    Returns the last-bar ADX as a float in [0, 100].
    A value > ADX_TREND_THRESHOLD (and rising) signals a trend with institutional
    momentum.

compute_ma_gap_pct(fast_ma, slow_ma, rclose)
    Percentage spread between fast and slow MA relative to rclose.
    Acts as a MACD proxy using the MA level columns already in the parquet.
    Positive = bullish alignment, negative = bearish.

compute_ma_slope_pct(ma_series, window)
    OLS slope of an MA series over `window` bars, normalised by the mean
    and expressed as %/day.

assess_ma_trend(df, *, close_col, high_col, low_col, fast_ma_col, slow_ma_col,
                rsi_period, adx_period, adx_slope_window, ma_slope_window)
    Integrates all four primitives into a single MATrendStrength snapshot.
    ADX series is computed in a single vectorised pass and reused for both the
    last-bar value and the slope — no redundant computation.
    is_trending = adx > ADX_TREND_THRESHOLD AND adx_slope >= 0.

All functions are:
    - Pure (no I/O, no side-effects).
    - Typed (PEP 484).
    - Vectorised — no Python loops in any calculation path.
    - Independently testable with synthetic pd.Series / pd.DataFrame.
    - Documented with invariants and failure modes.

Lookahead note
--------------
All functions return a snapshot of bar t using data up to and including bar t.
If called in a rolling loop for backtesting, the caller MUST delay results
by one bar before use in signal logic:
    # LOOKAHEAD PROTECTED: signal delayed by 1 bar
    signals = signals.shift(1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ta.utils import ols_slope_r2

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (exported; tests reference by name)
# ---------------------------------------------------------------------------

RSI_PERIOD:          int   = 14    # Wilder RSI lookback window
ADX_PERIOD:          int   = 14    # Wilder ADX/DMI lookback window
ADX_TREND_THRESHOLD: float = 25.0  # ADX above this = trend has institutional strength
ADX_SLOPE_WINDOW:    int   = 14    # bars used to compute adx_slope
MA_SLOPE_WINDOW:     int   = 20    # bars used to compute ma_gap_slope


# ---------------------------------------------------------------------------
# Private vectorised smoothing helpers
# ---------------------------------------------------------------------------


def _wilder_running_sum(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's running sum: S_j = (1 - 1/period) * S_{j-1} + arr[j].

    Seed: S_0 = sum(arr[:period])  (a sum, not an average).
    The new observation has coefficient **1** — used for TR, +DM, −DM smoothing.

    Vectorised via the geometric-decay cumsum identity:
        S_j = beta^j * (seed + cumsum(tail * (1/beta)^k)[j-1])
    where beta = 1 - 1/period.  No Python loops.

    Special case period=1: beta=0, so S_j = arr[j] (identity).

    Args:
        arr:    1-D float array of length n.
        period: Smoothing period >= 1.

    Returns:
        Float array of length max(0, n - period + 1).

    Example:
        >>> _wilder_running_sum(np.array([1.,2.,3.,4.,5.]), 2)
        array([3., 4.5, 6.25, 8.125])
    """
    n = len(arr)
    if n < period:
        return np.array([], dtype=float)

    if period == 1:
        # beta = 0 → S_j = 0 * S_{j-1} + arr[j] = arr[j]
        return arr.astype(float)

    seed = float(arr[:period].sum())
    tail = arr[period:].astype(float)

    if len(tail) == 0:
        return np.array([seed], dtype=float)

    beta = 1.0 - 1.0 / period

    # --- Overflow-safe batched implementation ---
    #
    # The naive vectorised identity S_j = β^j * (seed + Σ tail[k]*(1/β)^k)
    # is mathematically exact but creates intermediate values of order
    # (1/β)^m which overflow float64 (max ~1.8e308) when m is large:
    #   period=2  (β=0.5):  2^1024 overflows — hits the limit at ~1024 bars.
    #   period=3  (β=0.667): 1.5^1751 overflows — hits the limit at ~1751 bars.
    #   period=14 (β=0.929): overflow at ~9568 bars — safe for most histories.
    #
    # Fix: split tail into chunks no larger than max_batch, where max_batch is
    # the largest k such that (1/β)^k < float64_max ≈ e^709.
    # Each chunk uses the geometric identity within the safe range, then carries
    # the last computed value as the seed for the next chunk.
    # The Python loop runs over at most ceil(N/max_batch) chunks; for period=14
    # and N=2500 this is one iteration — no performance regression for typical data.
    max_batch = max(1, int(708.0 / np.log(1.0 / beta)))

    chunks: list[np.ndarray] = [np.array([seed])]
    s = seed
    for i in range(0, len(tail), max_batch):
        batch      = tail[i : i + max_batch]
        m          = len(batch)
        inv_powers = (1.0 / beta) ** np.arange(1, m + 1)   # safe: m ≤ max_batch
        dec_powers = beta          ** np.arange(1, m + 1)
        chunk      = (s + np.cumsum(batch * inv_powers)) * dec_powers
        chunks.append(chunk)
        s = float(chunk[-1])

    return np.concatenate(chunks)


def _wilder_ewm(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Standard Wilder EWM: S_j = (1/period) * arr[j] + (1 - 1/period) * S_{j-1}.

    Seed: S_0 = mean(arr[:period])  (an average, not a sum).
    The new observation has coefficient **1/period** — used for RSI avg_gain/avg_loss
    and DX → ADX smoothing.

    Vectorised via pandas `.ewm(alpha=1/period, adjust=False)` after prepending
    the seed value.  No Python loops.

    Args:
        arr:    1-D float array of length n.
        period: Smoothing period >= 1.

    Returns:
        Float array of length max(0, n - period + 1).

    Example:
        >>> _wilder_ewm(np.full(20, 2.5), period=7)
        array([2.5, 2.5, ..., 2.5])  # constant series stays constant
    """
    n = len(arr)
    if n < period:
        return np.array([], dtype=float)

    seed   = float(arr[:period].mean())
    series = np.concatenate([[seed], arr[period:].astype(float)])
    return pd.Series(series).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(dtype=float)


def _compute_adx_series(
    high:   np.ndarray,
    low:    np.ndarray,
    close:  np.ndarray,
    period: int,
) -> np.ndarray:
    """
    Compute the full ADX time series in a single vectorised pass.

    Algorithm (Wilder):
        1. Vectorised TR, +DM, -DM using NumPy shifts.
        2. Wilder running-sum smoothing (coefficient 1 on new value).
        3. +DI = 100 * (+DM_s / TR_s), guarded against zero TR.
        4. DX  = 100 * |+DI - -DI| / (+DI + -DI), guarded against zero sum.
        5. Wilder EWM smoothing of DX (coefficient 1/period on new value).

    Args:
        high, low, close: Equal-length 1-D float arrays, sorted ascending by date.
        period:           Smoothing period.

    Returns:
        ADX values clipped to [0, 100], length max(0, n - 2*period + 1).
        Empty array if insufficient data.
    """
    # --- Step 1: Vectorised TR, +DM, -DM ---
    prev_close = close[:-1]
    curr_high  = high[1:]
    curr_low   = low[1:]
    prev_high  = high[:-1]
    prev_low   = low[:-1]

    tr = np.maximum(
        curr_high - curr_low,
        np.maximum(np.abs(curr_high - prev_close), np.abs(curr_low - prev_close)),
    )

    up_move   = curr_high - prev_high
    down_move = prev_low  - curr_low

    p_dm = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    n_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # --- Step 2: Wilder running-sum smoothing ---
    tr_s   = _wilder_running_sum(tr,   period)
    p_dm_s = _wilder_running_sum(p_dm, period)
    n_dm_s = _wilder_running_sum(n_dm, period)

    if len(tr_s) == 0:
        return np.array([], dtype=float)

    # --- Step 3: +DI, -DI with zero guard (Rule 4: tolerance, not ==) ---
    # np.errstate suppresses the divide/invalid warning that fires when np.where
    # evaluates the false-branch expression before applying the mask.
    with np.errstate(divide="ignore", invalid="ignore"):
        _zero_tr = tr_s < 1e-10
        p_di = np.where(_zero_tr, 0.0, 100.0 * p_dm_s / tr_s)
        n_di = np.where(_zero_tr, 0.0, 100.0 * n_dm_s / tr_s)

    # --- Step 4: DX ---
    di_sum  = p_di + n_di
    di_diff = np.abs(p_di - n_di)
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(di_sum < 1e-10, 0.0, 100.0 * di_diff / di_sum)

    # --- Step 5: Wilder EWM of DX → ADX ---
    if len(dx) < period:
        return np.array([], dtype=float)

    adx = _wilder_ewm(dx, period)
    return np.clip(adx, 0.0, 100.0)


# ===========================================================================
# Primitive 1 — RSI (Wilder's smoothing, fully vectorised)
# ===========================================================================


def compute_rsi(rclose: pd.Series, period: int = RSI_PERIOD) -> float:
    """
    Compute Wilder's RSI on the last bar of a relative close price series.

    Uses Wilder's exponential smoothing (alpha = 1/period) for average gain
    and average loss, seeded with the simple mean over the first `period` bars.
    Fully vectorised via `_wilder_ewm` — no Python loops.

    Args:
        rclose: Relative close price series (rclose = ticker / FTSEMIB.MI).
                Must have at least period+1 non-NaN values.
        period: RSI lookback window. Must be >= 1. Default RSI_PERIOD (14).

    Returns:
        RSI value at the last bar, in [0.0, 100.0].

    Raises:
        ValueError: if period < 1.
        ValueError: if fewer than period+1 non-NaN values are present.

    Invariants:
        - Constant series → RSI = 50 (avg_gain = avg_loss = 0).
        - All-gain series → RSI = 100 (avg_loss converges to 0).
        - All-loss series → RSI = 0.

    Convergence note:
        Wilder's EWM seeds from the simple mean of the first `period` bars.
        The smoothing needs ~3×period bars to forget the seed and reflect
        the current market regime.  With fewer than 2×period bars the result
        is dominated by the seed value and may be misleading.
        assess_ma_trend logs a WARNING in this case; callers can silence it by
        providing a longer lead-in history.
    """
    if period < 1:
        raise ValueError(
            f"period must be >= 1, got {period}. "
            "RSI requires at least one bar of history per period unit."
        )

    clean = rclose.dropna()
    if len(clean) < period + 1:
        raise ValueError(
            f"compute_rsi requires at least period+1 = {period + 1} non-NaN values; "
            f"got {len(clean)}. Provide a longer series or reduce the period."
        )

    if len(clean) < 2 * period:
        log.warning(
            "compute_rsi: only %d non-NaN bars available (period=%d). "
            "Wilder's EWM requires ~3×period (%d) bars to converge; "
            "this RSI estimate is seed-dominated and less accurate.",
            len(clean), period, 3 * period,
        )

    closes = clean.to_numpy(dtype=float)
    deltas = np.diff(closes)

    gains  = np.where(deltas > 0,  deltas,  0.0)
    losses = np.where(deltas < 0, -deltas,  0.0)

    # _wilder_ewm seeds with mean(arr[:period]) = initial avg_gain / avg_loss
    avg_gain = float(_wilder_ewm(gains,  period)[-1])
    avg_loss = float(_wilder_ewm(losses, period)[-1])

    # Guard: both near-zero → flat series → RSI = 50 (Rule 4: tolerance not ==)
    if np.isclose(avg_gain, 0.0, atol=1e-10) and np.isclose(avg_loss, 0.0, atol=1e-10):
        return 50.0
    if np.isclose(avg_loss, 0.0, atol=1e-10):
        return 100.0

    rs  = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(float(rsi), 4)


# ===========================================================================
# Primitive 2 — ADX (Wilder's DMI, fully vectorised)
# ===========================================================================


def compute_adx(
    df:        pd.DataFrame,
    period:    int = ADX_PERIOD,
    *,
    high_col:  str = "rhigh",
    low_col:   str = "rlow",
    close_col: str = "rclose",
) -> float:
    """
    Compute Wilder's Average Directional Index (ADX) on relative OHLC data.

    Delegates entirely to `_compute_adx_series` which runs a single vectorised
    pass — no Python loops.

    ADX measures trend strength, not direction:
    - ADX > ADX_TREND_THRESHOLD (and rising) = trend gaining institutional momentum.
    - ADX < 20                               = weak or no trend.

    Args:
        df:        Single-ticker DataFrame sorted ascending by date.
                   Must contain columns named by high_col, low_col, close_col.
        period:    Smoothing window. Must be >= 1. Default ADX_PERIOD (14).
        high_col:  Name of the high price column. Default "rhigh".
        low_col:   Name of the low price column. Default "rlow".
        close_col: Name of the close price column. Default "rclose".

    Returns:
        ADX value at the last bar, in [0.0, 100.0].

    Raises:
        ValueError: if period < 1.
        ValueError: if any required column is missing.
        ValueError: if fewer than 2*period+1 bars are available.

    Failure modes:
        - Flat series (high == low all bars): TR = 0 → ADX = 0.
    """
    if period < 1:
        raise ValueError(
            f"period must be >= 1, got {period}. "
            "ADX requires sufficient bars to seed the Wilder smoother."
        )

    required = {high_col, low_col, close_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_adx: DataFrame missing required columns: {sorted(missing)}. "
            f"DataFrame has: {sorted(df.columns)}. "
            "Pass correct names via high_col=, low_col=, close_col= parameters."
        )

    min_bars = 2 * period + 1
    if len(df) < min_bars:
        raise ValueError(
            f"compute_adx requires at least 2*period+1 = {min_bars} bars; "
            f"got {len(df)}. Provide a longer history or reduce the period."
        )

    high  = df[high_col].to_numpy(dtype=float)
    low   = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)

    adx_series = _compute_adx_series(high, low, close, period)
    return round(float(adx_series[-1]), 4) if len(adx_series) > 0 else 0.0


# ===========================================================================
# Primitive 3 — MA gap % (MACD proxy)
# ===========================================================================


def compute_ma_gap_pct(fast_ma: float, slow_ma: float, rclose: float) -> float:
    """
    Compute the percentage spread between fast and slow MA, normalised by rclose.

    Acts as a MACD-line proxy:
        gap = (fast_ma - slow_ma) / rclose * 100

    Positive = fast MA above slow MA = bullish alignment.
    Negative = fast MA below slow MA = bearish alignment.
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
        ma_series: MA level values, sorted ascending. Must be a pd.Series.
                   Returns 0.0 if fewer than 2 non-NaN values in the window.
        window:    Number of recent bars to use. Must be >= 1.

    Returns:
        Slope as %/day; signed. Positive = MA rising, negative = falling.
        Returns 0.0 if fewer than 2 non-NaN values after slicing.

    Raises:
        ValueError: if window < 1.
    """
    if window < 1:
        raise ValueError(
            f"window must be >= 1, got {window}. "
            "This parameter controls the slope regression window length."
        )

    tail = ma_series.iloc[-window:].dropna()
    if len(tail) < 2:
        return 0.0

    y      = tail.to_numpy(dtype=float)
    y_mean = y.mean()

    if abs(y_mean) < 1e-10:    # Rule 4: tolerance not == (prevents div/zero)
        return 0.0

    slope, _ = ols_slope_r2(y)
    return round(float(slope / y_mean * 100), 6)


# ===========================================================================
# Primitive 5 — Integrated trend strength snapshot
# ===========================================================================


@dataclass(frozen=True)
class MATrendStrength:
    """
    Snapshot of MA crossover trend quality at the last bar of a ticker series.

    Integrates RSI, ADX, and MA gap/slope into one cohesive object.
    R² fields accompany every OLS slope to quantify statistical reliability:
    R² < 0.3 → slope is dominated by noise; R² ≥ 0.7 → reliable trend signal.

    Attributes
    ----------
    rsi : float
        Wilder's RSI (rsi_period-period) on rclose. In [0, 100].
        > 50 and rising = bullish momentum.
    adx : float
        ADX at the last bar. In [0, 100].
        > ADX_TREND_THRESHOLD = trend has institutional strength.
    adx_slope : float
        OLS slope of ADX over the last adx_slope_window bars.
        Units: ADX points/bar (raw, not normalised by mean ADX).
        Positive = ADX rising = trend gaining strength.
        Negative = ADX falling = trend weakening.
    adx_slope_r2 : float
        R² of the adx_slope OLS fit. In [0, 1].
        R² < 0.3 on short windows → treat slope as noise.
    ma_gap_pct : float
        (fast_ma_col - slow_ma_col) / rclose * 100 at last bar.
        Units: % (already rclose-normalised, comparable across tickers).
        MACD-line proxy. Positive = bullish alignment.
    ma_gap_slope : float
        OLS slope of the ma_gap_pct time series over ma_slope_window bars.
        Units: %-points/bar (absolute change in gap percentage per bar).
        NOT normalised by mean gap — the gap series is already in % terms,
        so a further mean-normalisation would produce an unintuitive
        "% of gap change per bar as a fraction of mean gap".
        Compare with compute_ma_slope_pct(), which DOES normalise by mean
        and returns a dimensionless %/day — a different quantity.
        Widening positive gap (positive slope) = trend accelerating bullishly.
        Narrowing gap (negative slope) = trend losing steam.
    ma_gap_slope_r2 : float
        R² of the ma_gap_slope OLS fit. In [0, 1].
    is_trending : bool
        True when adx > ADX_TREND_THRESHOLD AND adx_slope >= 0.
        Conditions: trend must be strong AND not actively declining.
        adx_slope > 0  = fresh/gaining trend.
        adx_slope == 0 = trend stable at a high level (still valid).
        adx_slope < 0  = trend weakening — not a fresh entry.
    """

    rsi:             float
    adx:             float
    adx_slope:       float
    adx_slope_r2:    float
    ma_gap_pct:      float
    ma_gap_slope:    float
    ma_gap_slope_r2: float
    is_trending:     bool


def assess_ma_trend(
    df:               pd.DataFrame,
    *,
    close_col:        str = "rclose",
    high_col:         str = "rhigh",
    low_col:          str = "rlow",
    fast_ma_col:      str = "rema_short_50",
    slow_ma_col:      str = "rema_long_150",
    rsi_period:       int = RSI_PERIOD,
    adx_period:       int = ADX_PERIOD,
    adx_slope_window: int = ADX_SLOPE_WINDOW,
    ma_slope_window:  int = MA_SLOPE_WINDOW,
) -> MATrendStrength:
    """
    Assess MA crossover trend quality for a single ticker.

    The ADX series is computed in a single vectorised pass and reused for both
    the last-bar ADX value and the adx_slope — no redundant computation.

    Algorithm
    ---------
    1. Validate all required columns.
    2. Compute RSI on the close column.
    3. Compute full ADX series via _compute_adx_series (single pass, vectorised).
       - adx       = adx_series[-1]
       - adx_slope = OLS slope of adx_series[-adx_slope_window:]
    4. Compute ma_gap_pct at last bar: fast_ma_col - slow_ma_col.
    5. Compute ma_gap_slope: OLS of gap_series over last ma_slope_window bars.
       Zero close prices guarded before the gap series is built (→ NaN, not inf).
    6. is_trending = adx > ADX_TREND_THRESHOLD AND adx_slope >= 0.

    Args:
        df:               Full single-ticker DataFrame sorted ascending by date.
        close_col:        Close price column. Default "rclose".
        high_col:         High price column. Default "rhigh".
        low_col:          Low price column. Default "rlow".
        fast_ma_col:      Fast MA column. Default "rema_short_50".
        slow_ma_col:      Slow MA column. Default "rema_long_150".
        rsi_period:       RSI lookback. Default RSI_PERIOD (14).
        adx_period:       ADX/DMI lookback. Default ADX_PERIOD (14).
        adx_slope_window: Bars used for adx_slope OLS. Default ADX_SLOPE_WINDOW (14).
        ma_slope_window:  Bars used for ma_gap_slope OLS. Default MA_SLOPE_WINDOW (20).

    Returns:
        MATrendStrength with all fields populated.

    Raises:
        ValueError: if any required column is missing.
        ValueError: propagated from compute_rsi on insufficient close history.
    """
    # --- 1. Column validation (fail fast) ---
    required = {close_col, high_col, low_col, fast_ma_col, slow_ma_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"assess_ma_trend: DataFrame missing required columns: {sorted(missing)}. "
            f"DataFrame has: {sorted(df.columns)}. "
            "Pass correct names via close_col=, high_col=, low_col=, "
            "fast_ma_col=, slow_ma_col= parameters."
        )

    # --- 2. RSI ---
    rsi = compute_rsi(df[close_col], period=rsi_period)

    # --- 3. ADX series — computed ONCE; value + slope reuse the same array ---
    high  = df[high_col].to_numpy(dtype=float)
    low   = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)

    adx_series = _compute_adx_series(high, low, close, adx_period)

    if len(adx_series) == 0:
        adx, adx_slope, adx_slope_r2_val = 0.0, 0.0, 0.0
    else:
        adx = round(float(adx_series[-1]), 4)
        slope_tail = adx_series[-adx_slope_window:]
        if len(slope_tail) >= 2:
            _slope, _r2   = ols_slope_r2(slope_tail)
            adx_slope     = round(float(_slope), 6)
            adx_slope_r2_val = round(float(_r2), 4)
        else:
            adx_slope, adx_slope_r2_val = 0.0, 0.0

    # --- 4. MA gap at last bar ---
    last = df.iloc[-1]
    ma_gap_pct = compute_ma_gap_pct(
        fast_ma = float(last[fast_ma_col]),
        slow_ma = float(last[slow_ma_col]),
        rclose  = float(last[close_col]),
    )

    # --- 5. MA gap slope over the full series with zero-close guard (Rule 5) ---
    rclose_safe = df[close_col].replace(0, np.nan)  # zero rclose → NaN, not inf
    gap_series  = (
        (df[fast_ma_col] - df[slow_ma_col]) / rclose_safe * 100
    ).dropna()

    if len(gap_series) >= 2:
        _gslope, _gr2   = ols_slope_r2(
            gap_series.iloc[-ma_slope_window:].to_numpy(dtype=float)
        )
        # _gslope is in %-points/bar: the gap series is already rclose-normalised
        # to %, so the raw OLS slope is directly interpretable without further
        # mean-normalisation.  Do NOT substitute compute_ma_slope_pct() here —
        # that function normalises by mean, producing a different quantity (%/day).
        ma_gap_slope    = round(float(_gslope), 6)
        ma_gap_slope_r2 = round(float(_gr2), 4)
    else:
        ma_gap_slope, ma_gap_slope_r2 = 0.0, 0.0

    # --- 6. Trend gate (docstring: adx_slope >= 0) ---
    is_trending = bool(adx > ADX_TREND_THRESHOLD and adx_slope >= 0.0)

    return MATrendStrength(
        rsi             = rsi,
        adx             = adx,
        adx_slope       = adx_slope,
        adx_slope_r2    = adx_slope_r2_val,
        ma_gap_pct      = ma_gap_pct,
        ma_gap_slope    = ma_gap_slope,
        ma_gap_slope_r2 = ma_gap_slope_r2,
        is_trending     = is_trending,
    )
