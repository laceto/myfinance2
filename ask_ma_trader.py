"""
ask_ma_trader.py — Moving average crossover trader AI assistant.

Usage:
    python ask_ma_trader.py --ticker A2A.MI
    python ask_ma_trader.py --ticker A2A.MI --question "Is the trend strengthening?"

Scope: MA crossover analysis only.
    Range breakout signals (rbo_*, rhi_*, rlo_*, rtt_5020) are intentionally excluded —
    they belong to ask_bo_trader.py.

What it does:
    1. Loads analysis_results.parquet
    2. Filters to the requested ticker
    3. Computes signal age and flip flags for all 6 MA signal columns
    4. Builds the last-bar snapshot: MA signals, MA levels, distances, momentum,
       swing levels, vol_trend, stop-loss
    5. Enriches the snapshot with MATrendStrength and MAVolumeProfile (computed
       over full ticker history)
    6. Sends the JSON to the OpenAI model with a MA-specific system prompt
    7. Prints the structured analysis to stdout

Environment:
    OPENAI_API_KEY must be set.

Columns included (last bar only):
    - Identity:        symbol, date, rrg (regime direction)
    - Relative OHLC:   rclose only
    - MA signals:      rema_50100, rema_100150, rema_50100150 (EMA — primary)
                       rsma_50100, rsma_100150, rsma_50100150 (SMA — confirmation)
    - MA levels:       rema_short_50, rema_medium_100, rema_long_150
                       rsma_short_50, rsma_medium_100, rsma_long_150
    - Derived (age):   {signal}_age — consecutive bars signal has held its value
    - Derived (flip):  {signal}_flip — 1 if signal changed on last bar
    - Derived (dist):  dist_to_rema_{n}_pct, dist_to_rsma_{n}_pct — % from rclose to MA
    - Derived (mom):   rclose_chg_50d, rclose_chg_150d
    - Volume:          vol_trend (last bar vs 20-bar mean)
    - Stop-loss:       rema_50100_stop_loss, rema_100150_stop_loss, rema_50100150_stop_loss
    - Swing levels:    rh3, rh4, rl3, rl4
    - Enrichments:     trend_strength (MATrendStrength), volume_profile (MAVolumeProfile)

Excluded on purpose:
    - rbo_*, rhi_*, rlo_*, rtt_5020 → range breakout assistant (ask_bo_trader.py)
    - ropen, rhigh, rlow — intraday OHLC; enrichments computed from full history but
      only EOD rclose in the snapshot
    - *_cumul, *_returns, *_chg* (intermediate analytics), *_PL_cum
    - rsma_*_stop_loss — stop-loss used from EMA signals only
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Literal

import openai
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ta.ma.trend_quality import MATrendStrength, assess_ma_trend
from ta.ma.volume import MAVolumeProfile, assess_ma_volume

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_PATH = Path("data/results/it/analysis_results.parquet")

# ---------------------------------------------------------------------------
# MA signal column sets (derived from confirmed parquet schema)
# ---------------------------------------------------------------------------

EMA_SIGNAL_COLS = ["rema_50100", "rema_100150", "rema_50100150"]
SMA_SIGNAL_COLS = ["rsma_50100", "rsma_100150", "rsma_50100150"]
ALL_SIGNAL_COLS = EMA_SIGNAL_COLS + SMA_SIGNAL_COLS

EMA_LEVEL_COLS = ["rema_short_50", "rema_medium_100", "rema_long_150"]
SMA_LEVEL_COLS = ["rsma_short_50", "rsma_medium_100", "rsma_long_150"]

EMA_STOP_COLS = [
    "rema_50100_stop_loss",
    "rema_100150_stop_loss",
    "rema_50100150_stop_loss",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a professional equity trader assistant specialising in Italian equities (Borsa Italiana).
You receive a JSON snapshot of a single ticker's last trading bar and provide a concise, \
actionable technical analysis focused on moving average crossover signals.

Field definitions:

RELATIVE PRICE
- rclose: relative price of the stock vs FTSEMIB.MI (rclose = ticker_close / benchmark_close).
  Rising rclose = stock outperforms the index regardless of absolute market direction.
- rrg: regime of the relative price series. +1 = bullish, -1 = bearish, 0 = sideways.

MA CROSSOVER SIGNALS (+1 long / -1 short / 0 flat)
  EMA signals (primary — faster, more responsive):
- rema_50100:   EMA 50 vs EMA 100 crossover. Fast/medium-term direction.
- rema_100150:  EMA 100 vs EMA 150 crossover. Medium/long-term direction.
- rema_50100150: Triple EMA confluence. +1 only when all three EMAs are in bullish stack
  (EMA50 > EMA100 > EMA150). -1 when all bearish. This is the highest-quality signal.
  SMA signals (confirmation — slower, more reliable against noise):
- rsma_50100, rsma_100150, rsma_50100150: identical logic on simple MAs.
  Agreement between EMA and SMA signals = strong conviction.
  EMA+/SMA- or EMA-/SMA+ disagreement = low conviction — wait for alignment.

SIGNAL AGE AND FLIP
- {signal}_age: consecutive bars the signal has held its current value.
  rema_50100150_age=1 = just crossed (fresh entry opportunity).
  rema_50100150_age=30 = mature trend — entry risk is higher.
- {signal}_flip: 1 if the signal changed on the last bar (fresh crossover), else 0.
  A flip=1 on the triple confluence signal is the strongest entry trigger.

MA LEVEL VALUES (relative price of each moving average)
- rema_short_50, rema_medium_100, rema_long_150: EMA values at each window.
- rsma_short_50, rsma_medium_100, rsma_long_150: SMA values at each window.
  Price relationship to these levels:
  rclose > rema_long_150 = price above long-term MA = structurally bullish.
  rclose < rema_long_150 = price below long-term MA = structurally bearish.

DISTANCES TO MA LEVELS
- dist_to_rema_{n}_pct: (rema_{n} - rclose) / rclose * 100.
  Negative = rclose already above the MA (bullish position).
  Positive = rclose below the MA (bearish position or approaching support from above).
  Large negative = price extended above MA — risk of mean reversion.
- dist_to_rsma_{n}_pct: same for SMA levels.

MOMENTUM
- rclose_chg_50d: % change in rclose over 50 bars. Measures medium-term outperformance.
- rclose_chg_150d: % change in rclose over 150 bars. Measures long-term trend direction.
  Both positive = sustained outperformance vs benchmark.

VOLUME
- vol_trend: current volume / 20-bar average. > 1.0 = expanding (confirms signal).
  < 1.0 = contracting (low conviction — treat crossover with caution).

STOP-LOSS (ATR-based, EMA signals)
- rema_50100_stop_loss:   ATR stop for the 50/100 EMA crossover.
- rema_100150_stop_loss:  ATR stop for the 100/150 EMA crossover.
- rema_50100150_stop_loss: ATR stop for the triple EMA confluence.
  For longs: entry invalidated if rclose falls below this stop.
  For shorts: entry invalidated if rclose rises above this stop.

SWING LEVELS
- rh3, rh4: magnitude-ordered swing highs. rh4 = strongest resistance (long target / short entry near).
- rl3, rl4: magnitude-ordered swing lows. rl4 = deepest floor (long target / short stop zone).

TREND STRENGTH ENRICHMENT (computed over full ticker history)
- trend_strength.rsi: Wilder's RSI (14-period) on rclose.
  Longs: RSI > 50 and rising (not > 70 overbought). Shorts: RSI < 50 and falling (not < 30 oversold).
- trend_strength.adx: Average Directional Index. > 25 = trend has institutional strength.
  ADX < 20 = weak trend — MA crossover may be unreliable / choppy.
- trend_strength.adx_slope: Is the ADX rising (trend gaining momentum) or falling (weakening)?
  Positive = ADX rising = trend strengthening.
  Negative = ADX falling = trend may be reversing.
- trend_strength.adx_slope_r2: R² of the ADX slope OLS fit (0–1).
  High (> 0.7) = clean, directional ADX move. Low (< 0.3) = noisy — slope is unreliable.
- trend_strength.ma_gap_pct: (rema_short_50 - rema_long_150) / rclose * 100.
  MACD-line proxy. Positive = bullish MA stack. Negative = bearish.
  Widening gap = trend accelerating. Narrowing gap = trend losing steam.
- trend_strength.ma_gap_slope: OLS slope of ma_gap_pct over 20 bars (%/bar).
- trend_strength.ma_gap_slope_r2: R² of the MA gap slope OLS fit (0–1).
  High = gap moving cleanly in one direction. Low = erratic / mean-reverting.
- trend_strength.is_trending: True when ADX > 25 AND ADX not declining.
  null: assess_ma_trend failed (insufficient history).

VOLUME PROFILE ENRICHMENT (computed over full ticker history)
- volume_profile.vol_on_crossover: vol_trend at the most recent signal flip bar.
  >= 1.2 = institutional participation on the crossover = high conviction.
  < 1.2 = low-volume crossover = fakeout risk.
  null = last bar was not a crossover (continuation).
- volume_profile.vol_trend_mean_post: mean vol_trend over the first 3 bars after the flip.
  > 1.0 = volume sustained after crossover = follow-through is real.
  < 1.0 = volume faded = possible distribution / fakeout.
- volume_profile.is_confirmed: True/False on the flip bar, null otherwise.
- volume_profile.is_sustained: True/False if 3+ post-flip bars, null otherwise.
  null: assess_ma_volume failed or insufficient history.

Output format — start with a description, then analyse each section.
Apply symmetric logic for longs (+1) and shorts (-1) throughout.
0. description: 3-5 sentence narrative. Regime, EMA/SMA confluence, ADX/RSI state,
   vol_trend. Exact numbers. No filler.
1. Short-term (50/100): rema_50100 signal, age, flip. dist_to_rema_50_pct and
   dist_to_rema_100_pct. Note SMA agreement (rsma_50100).
2. Medium-term (100/150): rema_100150 signal, age, flip. rclose_chg_150d trend.
   Note SMA agreement (rsma_100150).
3. Triple confluence: rema_50100150 signal, age, flip.
   Compare to rsma_50100150 — agree = high conviction, disagree = caution.
4. Trend strength: ADX (level + slope), RSI (value + direction), ma_gap_pct + ma_gap_slope.
   State is_trending. Call out any divergence (price rising but RSI / ma_gap falling).
5. Volume quality: vol_trend at last bar. vol_on_crossover, is_confirmed, is_sustained.
   Distinguish between fresh crossover and continuation bar.
6. Risk — always state both sides.
   Long stop  = rema_50100150_stop_loss (long invalidated if rclose drops below this).
   Long structural = rl4 (deepest swing low — structural floor for longs).
   Short stop = rema_50100150_stop_loss (short invalidated if rclose rises above this).
   Short structural = rh4 (peak resistance — structural ceiling for shorts).
   State long_stop, long_structural_stop, short_stop, short_structural_stop explicitly,
   even when only one direction is currently active.
7. Verdict: one actionable sentence — direction (long/short/flat), exact entry trigger
   (which signal flip + which confirmation), and the relevant stop level.

Be concise. No padding. Use numbers from the data, not generic statements.\
"""

# ---------------------------------------------------------------------------
# Column selection and derivation
# ---------------------------------------------------------------------------


def _compute_signal_age_and_flip(
    df: pd.DataFrame, signal_cols: list[str]
) -> pd.DataFrame:
    """
    Compute age and flip flags for each MA signal column.

    Age: consecutive bars the signal has held its current value (resets to 1 on change).
    Flip: 1 if the signal changed on that bar vs the previous bar, else 0.

    These columns are not present in analysis_results.parquet — they are derived here
    using the same algorithm as the rbo_*_age / rbo_*_flip derivation in ask_bo_trader.py.

    Args:
        df:          Single-ticker DataFrame with signal columns present.
        signal_cols: List of signal column names to process.

    Returns:
        df with {col}_age and {col}_flip columns added (in-place copy).
    """
    df = df.copy()
    for sig in signal_cols:
        if sig not in df.columns:
            continue
        # --- flip flag ---
        df[f"{sig}_flip"] = (df[sig] != df[sig].shift(1)).astype(int)
        # --- age: consecutive bars at the same signal value ---
        groups = (df[sig] != df[sig].shift()).cumsum()
        df[f"{sig}_age"] = df.groupby(groups).cumcount() + 1
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and compute the MA crossover column set for a single-ticker DataFrame.

    Columns kept:
      - Identity:        date, rrg
      - Relative close:  rclose
      - EMA signals:     rema_50100, rema_100150, rema_50100150
      - SMA signals:     rsma_50100, rsma_100150, rsma_50100150
      - EMA level vals:  rema_short_50, rema_medium_100, rema_long_150
      - SMA level vals:  rsma_short_50, rsma_medium_100, rsma_long_150
      - Stop-loss:       rema_50100_stop_loss, rema_100150_stop_loss, rema_50100150_stop_loss
      - Swing levels:    rh3, rh4, rl3, rl4
      - Computed:        {signal}_age, {signal}_flip, dist_to_rema_{n}_pct,
                         dist_to_rsma_{n}_pct, rclose_chg_50d, rclose_chg_150d, vol_trend

    Columns excluded:
      - rbo_*, rhi_*, rlo_*, rtt_5020 → range breakout assistant
      - ropen, rhigh, rlow — intraday OHLC, not needed for EOD MA logic
      - rh1, rh2, rl1, rl2 — minor pivots, too noisy
      - *_cumul, *_returns, *_chg*, *_PL_cum — intermediate analytics
      - rsma_*_stop_loss — EMA stops are sufficient
    """
    cols = df.columns.tolist()

    # --- Compute age and flip before column selection (needs full signal history) ---
    df = _compute_signal_age_and_flip(df, ALL_SIGNAL_COLS)

    # --- Base column selection ---
    selected = ["symbol", "date", "rrg", "rclose"]

    for col in (
        EMA_SIGNAL_COLS + SMA_SIGNAL_COLS
        + EMA_LEVEL_COLS + SMA_LEVEL_COLS
        + EMA_STOP_COLS
    ):
        if col in df.columns:
            selected.append(col)

    # Significant swing levels (rh3/rh4/rl3/rl4 — skip minor pivots)
    for sw in ["rh3", "rh4", "rl3", "rl4"]:
        if sw in df.columns:
            selected.append(sw)

    # Age and flip columns (computed above)
    age_cols  = [c for c in df.columns if c.endswith("_age")  and any(s in c for s in ALL_SIGNAL_COLS)]
    flip_cols = [c for c in df.columns if c.endswith("_flip") and any(s in c for s in ALL_SIGNAL_COLS)]
    selected += sorted(age_cols) + sorted(flip_cols)

    df = df[selected].copy()

    # Forward-fill swing levels
    swing_cols = [c for c in ["rh3", "rh4", "rl3", "rl4"] if c in df.columns]
    if swing_cols:
        df[swing_cols] = df[swing_cols].ffill().bfill()

    # --- Derived: distance from rclose to each MA level (% of rclose) ---
    # Negative = rclose already above the MA (bullish).
    # Positive = rclose below the MA.
    level_map = {
        "rema_short_50":   "dist_to_rema_50_pct",
        "rema_medium_100": "dist_to_rema_100_pct",
        "rema_long_150":   "dist_to_rema_150_pct",
        "rsma_short_50":   "dist_to_rsma_50_pct",
        "rsma_medium_100": "dist_to_rsma_100_pct",
        "rsma_long_150":   "dist_to_rsma_150_pct",
    }
    for ma_col, dist_col in level_map.items():
        if ma_col in df.columns:
            df[dist_col] = (
                (df[ma_col] - df["rclose"]) / df["rclose"] * 100
            ).round(2)

    # --- Derived: momentum (% change in rclose over 50 / 150 bars) ---
    for lookback in [50, 150]:
        df[f"rclose_chg_{lookback}d"] = (
            df["rclose"].pct_change(periods=lookback) * 100
        ).round(2)

    # --- Derived: vol_trend ---
    if "volume" in df.columns:
        vol_avg = df["volume"].rolling(20, min_periods=1).mean()
        df["vol_trend"] = (df["volume"] / vol_avg.where(vol_avg != 0)).round(2)

    # Drop volume (raw) and symbol (redundant — injected as ticker in snapshot)
    df = df.drop(columns=["volume", "symbol"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj) -> dict:
    """Convert a frozen dataclass to a plain JSON-safe dict."""
    return {k: getattr(obj, k) for k in obj.__dataclass_fields__}


def _compute_trend_strength(df: pd.DataFrame) -> dict | None:
    """
    Run assess_ma_trend over the full ticker history and return a JSON-safe dict.

    Returns None when insufficient history or missing columns.
    Logs a warning instead of crashing.
    """
    try:
        ts: MATrendStrength = assess_ma_trend(df)
        return {
            "rsi":             ts.rsi,
            "adx":             ts.adx,
            "adx_slope":       ts.adx_slope,
            "adx_slope_r2":    ts.adx_slope_r2,
            "ma_gap_pct":      ts.ma_gap_pct,
            "ma_gap_slope":    ts.ma_gap_slope,
            "ma_gap_slope_r2": ts.ma_gap_slope_r2,
            "is_trending":     ts.is_trending,
        }
    except (ValueError, KeyError) as exc:
        log.warning("assess_ma_trend failed: %s", exc)
        return None


def _compute_volume_profile(df: pd.DataFrame) -> dict | None:
    """
    Run assess_ma_volume over the full ticker history and return a JSON-safe dict.

    Uses rema_50100 as the primary signal col for flip detection.
    Returns None on failure. Logs a warning.
    """
    try:
        vp: MAVolumeProfile = assess_ma_volume(df, signal_col="rema_50100")
        return {
            "vol_on_crossover":    vp.vol_on_crossover,
            "vol_trend_mean_post": vp.vol_trend_mean_post,
            "is_confirmed":        vp.is_confirmed,
            "is_sustained":        vp.is_sustained,
        }
    except (ValueError, KeyError) as exc:
        log.warning("assess_ma_volume failed: %s", exc)
        return None


def build_snapshot(df_ticker: pd.DataFrame) -> dict:
    """
    Return the last bar of a prepared ticker DataFrame as a JSON-safe dict,
    enriched with trend-strength and volume-behaviour summaries.

    The enrichments are computed over the full ticker history before slicing to
    last bar. Each returns None on failure so a failed enrichment never prevents
    the snapshot from being built.

    Raises:
        ValueError: If the DataFrame is empty after filtering.
    """
    if df_ticker.empty:
        raise ValueError("DataFrame is empty — ticker may not exist in the parquet.")

    # --- Enrichments over full history BEFORE column selection ---
    trend_strength  = _compute_trend_strength(df_ticker)
    volume_profile  = _compute_volume_profile(df_ticker)

    # --- Last-bar snapshot ---
    df_prepared = select_columns(df_ticker)
    last = df_prepared.tail(1).copy()
    last["date"] = last["date"].dt.strftime("%Y-%m-%d")

    records  = json.loads(last.to_json(orient="records", double_precision=4))
    snapshot = records[0]

    snapshot["trend_strength"]  = trend_strength
    snapshot["volume_profile"]  = volume_profile

    return snapshot


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class MATimeframeAnalysis(BaseModel):
    ema_signal:       int   = Field(description="EMA signal: +1 long, 0 flat, -1 short.")
    sma_signal:       int   = Field(description="SMA signal (confirmation). Same scale.")
    ema_sma_agree:    bool  = Field(description="True if EMA and SMA signal have the same non-zero sign.")
    signal_age:       int   = Field(description="Bars the EMA signal has held its current value.")
    fresh_flip:       bool  = Field(description="True if EMA signal changed on the last bar.")
    dist_fast_ma_pct: float = Field(description="% distance from rclose to fast MA for this pair. Negative = above MA.")
    dist_slow_ma_pct: float = Field(description="% distance from rclose to slow MA for this pair. Negative = above MA.")
    commentary:       str   = Field(description="One sentence on the signal state and key MA level context.")


class MATripleConfluence(BaseModel):
    ema_signal:    int   = Field(description="rema_50100150: +1 all EMAs bullish, -1 all bearish, 0 mixed.")
    sma_signal:    int   = Field(description="rsma_50100150: same for SMAs.")
    agree:         bool  = Field(description="True if EMA and SMA triple signals agree.")
    signal_age:    int   = Field(description="Bars rema_50100150 has held its current value.")
    fresh_flip:    bool  = Field(description="True if rema_50100150 changed on the last bar.")
    commentary:    str   = Field(description="One sentence on triple confluence quality and conviction.")


class MATrendStrengthOutput(BaseModel):
    rsi:             float = Field(description="RSI (14-period) on rclose. 0–100.")
    adx:             float = Field(description="ADX at last bar. > 25 = institutional trend strength.")
    adx_slope:       float = Field(description="ADX slope (%/bar). Positive = strengthening. Negative = weakening.")
    adx_slope_r2:    float = Field(description="R² of the ADX slope OLS fit (0–1). High = clean trend signal. Low = noisy.")
    ma_gap_pct:      float = Field(description="(EMA50 - EMA150) / rclose * 100. MACD proxy. Positive = bullish stack.")
    ma_gap_slope:    float = Field(description="OLS slope of ma_gap_pct over 20 bars. Widening = accelerating trend.")
    ma_gap_slope_r2: float = Field(description="R² of the MA gap slope OLS fit (0–1). High = gap changing cleanly. Low = erratic.")
    is_trending:     bool  = Field(description="True when ADX > 25 and ADX not declining.")
    commentary:      str   = Field(description="One sentence on trend quality: RSI position, ADX state, gap direction.")


class MAVolumeQuality(BaseModel):
    vol_trend:             float       = Field(description="Current volume / 20-bar average. > 1.0 = expanding.")
    vol_on_crossover:      float | None = Field(description="vol_trend at the most recent crossover flip bar. null = continuation bar.")
    is_confirmed:          bool  | None = Field(description="True = flip bar with vol_trend >= 1.2. False = weak volume on flip. null = not a flip bar.")
    is_sustained:          bool  | None = Field(description="True = post-flip vol mean >= 1.0. False = volume faded. null = insufficient post-flip history.")
    commentary:            str          = Field(description="One sentence: was the crossover volume-confirmed and is the follow-through real?")


class MARiskLevels(BaseModel):
    long_stop:             float = Field(description="rema_50100150_stop_loss — ATR stop for a long; long invalidated if rclose drops below this.")
    long_structural_stop:  float = Field(description="rl4 — deepest swing low; structural floor for longs.")
    short_stop:            float = Field(description="rema_50100150_stop_loss — ATR stop for a short; short invalidated if rclose rises above this.")
    short_structural_stop: float = Field(description="rh4 — peak resistance; structural ceiling for shorts.")
    peak_resistance:       float = Field(description="rh4 — absolute highest swing high.")
    major_floor:           float = Field(description="rl4 — absolute deepest swing low.")


class MATraderAnalysis(BaseModel):
    description: str = Field(
        description=(
            "3-5 sentence narrative summary. Regime, EMA/SMA confluence, ADX/RSI state, "
            "vol_trend. Exact numbers. No generic statements."
        )
    )
    regime:       int                                              = Field(description="rrg: +1 bullish, 0 sideways, -1 bearish.")
    confluence:   Literal["full_long", "full_short", "mixed", "flat"] = Field(
        description=(
            "full_long: rema_50100150=+1 AND rsma_50100150=+1. "
            "full_short: both -1. mixed: EMA/SMA disagree or partial alignment. flat: signals unclear."
        )
    )
    short_term:         MATimeframeAnalysis  = Field(description="50/100 crossover analysis.")
    medium_term:        MATimeframeAnalysis  = Field(description="100/150 crossover analysis.")
    triple_confluence:  MATripleConfluence   = Field(description="Triple EMA/SMA confluence analysis.")
    trend_strength:     MATrendStrengthOutput | None = Field(
        description="ADX/RSI/gap trend quality. null when insufficient history."
    )
    volume_quality:     MAVolumeQuality      = Field(description="Volume confirmation at and after crossover.")
    risk:               MARiskLevels         = Field(description="Stop-loss and swing level targets.")
    verdict: str = Field(
        description="Actionable one-sentence conclusion. Direction, entry trigger, and relevant stop. Exact numbers."
    )


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"


def ask_ma_trader(
    snapshot: dict, ticker: str, question: str | None
) -> MATraderAnalysis:
    """
    Send the ticker snapshot to an OpenAI model and return a structured MA analysis.

    Args:
        snapshot: Dict from build_snapshot — the last-bar MA data payload.
        ticker:   Ticker symbol string.
        question: Optional follow-up question from the CLI caller.

    Returns:
        MATraderAnalysis Pydantic model parsed from the model response.
    """
    client = openai.OpenAI()

    user_content = f"Ticker: {ticker}\n\nSnapshot:\n{json.dumps(snapshot, indent=2)}"
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending MA snapshot for %s to %s (%d fields)", ticker, MODEL, len(snapshot))

    response = client.beta.chat.completions.parse(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        response_format=MATraderAnalysis,
    )

    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MA crossover trader AI assistant — sends a ticker snapshot to OpenAI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Yahoo Finance ticker symbol (e.g. A2A.MI)",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Optional follow-up question (e.g. 'Is the trend strengthening?')",
    )
    parser.add_argument(
        "--data",
        default=str(RESULTS_PATH),
        help=f"Path to analysis_results.parquet (default: {RESULTS_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args      = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        log.error("Parquet not found: %s", data_path)
        sys.exit(1)

    log.info("Loading %s", data_path)
    df = pd.read_parquet(data_path)

    df_ticker = df[df["symbol"] == args.ticker].copy()
    if df_ticker.empty:
        log.error(
            "Ticker '%s' not found in parquet. Available sample: %s",
            args.ticker,
            df["symbol"].unique()[:10].tolist(),
        )
        sys.exit(1)

    snapshot = {"ticker": args.ticker, **build_snapshot(df_ticker)}
    log.info("Snapshot built: %d fields, date=%s", len(snapshot), snapshot.get("date"))

    print("\n" + "=" * 60)
    print(f"  MA Snapshot — {args.ticker}")
    print("=" * 60)
    print(json.dumps(snapshot, indent=2))
    print("=" * 60 + "\n")

    analysis: MATraderAnalysis = ask_ma_trader(
        snapshot, ticker=args.ticker, question=args.question
    )

    a   = analysis
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  AI MA Trader Analysis — {args.ticker}")
    print(sep)
    print(f"  {a.description}")
    print()
    print(f"  Regime     : {a.regime}   Confluence: {a.confluence}")
    print()
    for label, tf in [("SHORT (50/100)", a.short_term), ("MED   (100/150)", a.medium_term)]:
        flip = " [FLIP]" if tf.fresh_flip else ""
        agree = "✓" if tf.ema_sma_agree else "✗ EMA/SMA DISAGREE"
        print(f"  {label}  ema={tf.ema_signal}{flip}  sma={tf.sma_signal}  {agree}  age={tf.signal_age}")
        print(f"            fast_dist={tf.dist_fast_ma_pct:+.2f}%  slow_dist={tf.dist_slow_ma_pct:+.2f}%")
        print(f"            {tf.commentary}")
    print()
    tc = a.triple_confluence
    flip3 = " [FLIP]" if tc.fresh_flip else ""
    agree3 = "✓ EMA+SMA agree" if tc.agree else "✗ EMA/SMA disagree"
    print(f"  Triple     : ema={tc.ema_signal}{flip3}  sma={tc.sma_signal}  {agree3}  age={tc.signal_age}")
    print(f"               {tc.commentary}")
    print()
    if a.trend_strength is not None:
        ts = a.trend_strength
        print(f"  Trend      : RSI={ts.rsi:.1f}  ADX={ts.adx:.1f}  "
              f"adx_slope={ts.adx_slope:+.4f}/bar (r²={ts.adx_slope_r2:.2f})  "
              f"trending={ts.is_trending}")
        print(f"               gap={ts.ma_gap_pct:+.2f}%  "
              f"gap_slope={ts.ma_gap_slope:+.4f}/bar (r²={ts.ma_gap_slope_r2:.2f})")
        print(f"               {ts.commentary}")
    else:
        print("  Trend      : insufficient history for ADX/RSI computation")
    print()
    vq = a.volume_quality
    confirm_str = ("n/a" if vq.is_confirmed is None
                   else ("CONFIRMED" if vq.is_confirmed else "WEAK"))
    sustain_str = ("n/a" if vq.is_sustained is None
                   else ("SUSTAINED" if vq.is_sustained else "FADED"))
    print(f"  Volume     : vol_trend={vq.vol_trend:.2f}x  crossover={confirm_str}  post={sustain_str}")
    print(f"               {vq.commentary}")
    print()
    r = a.risk
    print(f"  Risk (long) : stop={r.long_stop}  structural={r.long_structural_stop}")
    print(f"  Risk (short): stop={r.short_stop}  structural={r.short_structural_stop}")
    print(f"               resistance={r.peak_resistance}  floor={r.major_floor}")
    print()
    print(f"  Verdict    : {a.verdict}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
