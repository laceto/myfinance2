"""
ask_bo_trader.py — Range breakout trader AI assistant.

Usage:
    python ask_bo_trader.py --ticker A2A.MI
    python ask_bo_trader.py --ticker A2A.MI --question "Should I add to this position?"

Scope: range breakout analysis only.
    Moving average crossover signals (rema_*, rsma_*) are intentionally excluded —
    they belong to a separate MA trader assistant.  The turtle signal (rtt_5020) is
    kept as an independent price-channel breakout confirmation, not an MA signal.

What it does:
    1. Loads analysis_results.parquet
    2. Filters to the requested ticker
    3. Computes the range breakout snapshot (last bar): rbo/rhi/rlo levels, swing levels,
       range_position_pct, distances, momentum, signal age/flip flags, vol_trend, turtle
    4. Enriches the snapshot with RangeSetup and VolumeProfile (computed over full history)
    5. Sends the JSON to the OpenAI model with a range-breakout-specific system prompt
    6. Prints the structured analysis to stdout

Environment:
    OPENAI_API_KEY must be set.

Columns included (last bar only):
    - Identity:       symbol, date, rrg (regime direction)
    - Relative OHLC:  rclose (EOD close — open/high/low excluded as intraday noise)
    - Breakout bands: rhi_20/50/150 (resistance), rlo_20/50/150 (support)
    - Breakout sigs:  rbo_20/50/150 (+1 long / -1 short / 0 flat), age, flip
    - Swing levels:   rh3, rh4, rl3, rl4 (forward-filled; minor pivots rh1/2/rl1/2 excluded)
    - Turtle:         rtt_5020 (independent price-channel confirmation)
    - Derived:        range_position_pct, dist_to_rhi/rlo per window, rclose_chg_Nd, vol_trend
    - Enrichments:    range_setup (RangeSetup), volatility_compression (VolatilityState),
                      volume_profile (VolumeProfile)

Excluded on purpose:
    - rema_*, rsma_* (MA crossover signals and MA level values) → MA trader assistant
    - ropen, rhigh, rlow — intraday OHLC; only EOD rclose matters for signal logic
    - *_stop_loss — ATR stops not in scope for the breakout snapshot
    - *_cumul, *_returns, *_chg* (intermediate analytics), *_PL_cum
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

from ta.breakout.range_quality import assess_range, RangeSetup, measure_volatility_compression, VolatilityState
from ta.breakout.volume import assess_volume_profile, VolumeProfile

load_dotenv()  # reads .env from the project root into os.environ

# Windows stdout must be UTF-8 before any logging/print.
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
# System prompt — loaded once, defines field semantics for the model
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a professional equity trader assistant specialising in Italian equities (Borsa Italiana).
You receive a JSON snapshot of a single ticker's last trading bar and provide a concise, \
actionable technical analysis.

Field definitions:
- rclose: relative price of the stock vs FTSEMIB.MI benchmark
  (rclose = ticker_close / benchmark_close). A rising rclose means the stock
  outperforms the index regardless of absolute market direction.
- rrg: regime of the stock on the relative price series.
    +1 = bullish regime, -1 = bearish regime, 0 = sideways.
- rhi_20, rhi_50, rhi_150: relative resistance levels at 20, 50, 150-day windows.
- rlo_20, rlo_50, rlo_150: relative support levels at 20, 50, 150-day windows.
  Wider windows = stronger, more structural levels.
  rhi_150/rlo_150 are the major trend boundaries; rhi_20/rlo_20 are the short-term trigger zone.
- rbo_20: range breakout signal on the 20-day relative price window.
    +1 = enter long (price broke above rhi_20),
    -1 = enter short (price broke below rlo_20),
     0 = exit / no trade (price back inside the range).
- rbo_50, rbo_150: same breakout signal logic on the 50 and 150-day windows.
  Full confluence: rbo_20 = rbo_50 = rbo_150 = +1 → strongest long setup.
                   rbo_20 = rbo_50 = rbo_150 = -1 → strongest short setup.
  Fighting the 150-day signal (e.g., rbo_150=-1 while rbo_20=+1) is a low-quality trade.
- range_position_pct: where rclose sits between rlo_20 and rhi_20.
  0%   = at support. 100% = at resistance.
  >100% = breakout long in progress (price above rhi_20).
  <0%   = breakout short in progress (price below rlo_20).
- rbo_20_age, rbo_50_age, rbo_150_age: bars each signal has held its current value.
  rbo_150_age=90, rbo_20_age=1 = mature structural trend, fresh short-term re-entry.
- rbo_20_flip, rbo_50_flip, rbo_150_flip: 1 if signal changed on the last bar, else 0.
- dist_to_rhi_{N}_pct: % distance from rclose to resistance.
  Negative = price already above resistance (long breakout active).
  For shorts: this is the stop distance (how far price must rally to invalidate the trade).
- dist_to_rlo_{N}_pct: % distance from rclose to support.
  Negative = price already below support (short breakout active).
  For longs: this is the stop distance (how far price must fall to invalidate the trade).
- rclose_chg_20d/50d/150d: % change in relative price over 20/50/150 bars.
  Use these to assess whether outperformance vs benchmark is accelerating or fading.
- vol_trend: current volume / 20-bar average volume. >1.0 = expanding (confirming),
  <1.0 = contracting (caution — move lacks conviction).
- rtt_5020: turtle breakout signal (fast=20-day high, slow=50-day high).
  +1 = price broke above the 20-day high (bullish channel breakout).
  -1 = price broke below the 20-day low.
   0 = inside channel.
  Independent confirmation of rbo_*: both signals firing together is stronger than either alone.
- rh3, rh4: magnitude-ordered swing highs. rh4 = highest (strongest resistance).
- rl3, rl4: magnitude-ordered swing lows. rl4 = deepest (strongest support / major floor).

Range setup (range_setup — computed over the most recent consolidation window):
- n_resistance_touches: distinct times price approached rhi_20 without breaking through.
  2–4 = clean range; < 2 = poorly defined; > 5 = very mature, energy highly compressed.
- n_support_touches: same for rlo_20.
- is_sideways: True when OLS slope of rclose is below 0.15%/day — confirms real consolidation.
- slope_pct_per_day: signed OLS slope of rclose (%/day). ~0 = ideal sideways; >0 = drifting up.
- consolidation_bars: total consecutive bars rbo_20 has held 0. Higher = more mature range.
- band_width_pct: (rhi_20 - rlo_20) / rclose * 100 at the last consolidation bar.
  Low = tight range = compressed energy. Watch for compressing band_width over time.
- null: assess_range failed (e.g., ticker currently in a trend with no recent consolidation).

Volatility compression (volatility_compression — computed over the full ticker history):
- band_width_pct: (rhi_20 - rlo_20) / rclose * 100 at the last bar.
  Measures how wide the current 20-day range envelope is. Low = price in a tight cage.
- band_width_slope: OLS slope of band_width_pct over the last 40 bars (%/bar).
  Negative = the envelope is actively narrowing = energy building toward a breakout.
  Positive = the envelope is expanding = consolidation may be dissolving.
- band_width_pct_rank: percentile rank of band_width_pct vs the last 252 bars (0–100).
  Low rank = historically tight for this specific ticker — not just quiet, but compressed
  relative to its own baseline. This normalises for ticker-specific volatility.
- is_compressed: True when band_width_slope < 0 AND band_width_pct_rank < 25.
  Both conditions required: the range must be both actively narrowing AND in the bottom
  quartile of its own history. This is the "coiled spring" signal.
- history_available: actual number of bars used for the percentile rank computation.
  Fewer bars = less reliable rank. Treat is_compressed with caution when history_available < 252.
- is_rank_reliable: True when history_available >= 252 (full reference window).
  When False, band_width_pct_rank is computed on partial history and may not be comparable
  across tickers — flag this in the analysis and weight is_compressed accordingly.
- null: measure_volatility_compression failed (e.g., insufficient history).

Volume profile (volume_profile — computed over the same consolidation window):
- vol_trend_mean: mean vol_trend over the consolidation window. < 1.0 = quiet (healthy).
- vol_trend_slope: OLS slope of vol_trend per bar. Negative = volume drying up (ideal).
  A slightly positive slope with low mean = quiet accumulation — also valid.
- is_quiet: True when vol_trend_mean < 1.0 (the primary gate for consolidation quality).
- is_declining: True when vol_trend_slope < 0 (informational; callers may use as extra filter).
- breakout_confirmed: True/False only on the exact breakout flip bar (rbo_20 just changed to
  non-zero). True = vol_trend_now >= 1.2 (institutional participation). False = fakeout risk.
  None = not a flip bar (in consolidation or mid-trend).
- null: assess_volume_profile failed (e.g., no consolidation history in the series).

Output format — start with a description, then analyse each section.
Apply symmetric logic for longs (+1) and shorts (-1) throughout.
0. description: 3-5 sentence narrative. Regime, multi-timeframe breakout confluence
   (both long and short signals treated equally), range quality, key levels, vol_trend.
   Exact numbers. No filler.
1. Short term (20-day): rbo_20 signal, age, flip, range_position_pct, vol_trend.
   For longs: dist_to_rhi_20_pct (how close to resistance / already above).
   For shorts: dist_to_rlo_20_pct (how close to support / already below).
   Note rtt_5020 turtle confirmation.
2. Medium term (50-day): rbo_50 signal, age, flip, rclose_chg_50d.
3. Long term (150-day): rbo_150 signal, age, rclose_chg_150d,
   rh4 (peak resistance / short target) and rl4 (major floor / long target).
4. Range quality (range_quality): touch count, sideways confirmation, consolidation_bars,
   band_width_pct. State direction bias (slope_pct_per_day sign).
   If null, ticker is in an active trend with no recent consolidation.
5. Volatility compression (volatility_compression): band_width_slope, band_width_pct_rank,
   is_compressed. State whether the range is actively coiling and historically tight.
   If null, state insufficient history.
6. Volume quality (volume_quality): is_quiet, is_declining, breakout_confirmed.
   If breakout_confirmed is True/False, comment on conviction. If null, state unavailable.
7. Risk — always state both sides:
   Long stop  = rlo_20 (short-term), rlo_150 (structural).
   Short stop = rhi_20 (short-term), rhi_150 (structural).
   Highlight the relevant side given the active signal.
8. Verdict: one actionable sentence — state direction (long/short/flat), exact entry
   trigger levels, and the relevant stop.

Be concise. No padding. Use numbers from the data, not generic statements.\
"""

# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and compute the range-breakout column set for a single-ticker DataFrame.

    Columns kept:
      - Identity:        date, rrg (regime: +1 bullish / 0 sideways / -1 bearish)
      - Relative close:  rclose only (ropen/rhigh/rlow excluded — intraday noise)
      - Breakout bands:  rhi_20/50/150 (resistance), rlo_20/50/150 (support)
      - Breakout sigs:   rbo_20/50/150 (+1 enter long / -1 enter short / 0 flat)
      - Swing levels:    rh3, rh4, rl3, rl4 (forward-filled; minor pivots excluded)
      - Turtle:          rtt_5020 (price-channel breakout confirmation)
      - Volume:          absolute volume for liquidity context
      - Derived:         range_position_pct, dist_to_rhi/rlo per window,
                         rclose_chg_Nd, vol_trend, rbo_*_age, rbo_*_flip

    Columns excluded:
      - rema_*, rsma_* (MA crossover signals + MA level values) → MA trader assistant
      - ropen, rhigh, rlow — intraday OHLC, not needed for EOD range breakout logic
      - rh1, rh2, rl1, rl2 — minor recent pivots, too noisy
      - *_stop_loss, *_cumul, *_returns, *_chg*, *_PL_cum — intermediate analytics
    """
    cols = df.columns

    rbo_cols = [c for c in cols if "rbo" in c]
    # rhi_* / rlo_* — breakout band columns only (rhi_20/50/150, rlo_20/50/150).
    # startswith("rhi_") / startswith("rlo_") excludes "rhigh" / "rlow".
    rhi_cols = [c for c in cols if c.startswith("rhi_")]
    rlo_cols = [c for c in cols if c.startswith("rlo_")]
    # Turtle signal only — no rema_*/rsma_* in this assistant.
    turtle_cols = [c for c in cols if c == "rtt_5020"]

    selected = (
        ["symbol", "rrg", "date", "rclose"]
        + ["volume"]
        + rbo_cols
        + rhi_cols + rlo_cols
        + turtle_cols
    )
    # Add significant swing levels only (rh3/rh4/rl3/rl4; skip minor pivots rh1/2/rl1/2).
    for sw in ["rh3", "rh4", "rl3", "rl4"]:
        if sw in cols:
            selected.append(sw)

    df = df[selected].copy()

    # Drop intermediate / noisy columns.
    drop_patterns = [r"cumul$", r"returns$", r"chg", r"cum"]
    drop_exact    = ["rbo_20_stop_loss", "rbo_150_stop_loss", "rbo_50_stop_loss"]
    to_drop = [
        c for c in df.columns
        if any(re.search(p, c) for p in drop_patterns)
    ] + drop_exact
    df = df.drop(columns=to_drop, errors="ignore")

    # Forward-fill swing highs/lows so NaN gaps don't propagate into the snapshot.
    swing_cols = [c for c in ["rh1", "rh2", "rh3", "rh4", "rl1", "rl2", "rl3", "rl4"] if c in df.columns]
    if swing_cols:
        df[swing_cols] = df[swing_cols].ffill().bfill()

    # --- Derived field 1: range_position_pct ---
    # Measures where rclose sits within the 20-day relative range.
    # NaN when rhi_20 == rlo_20 (flat/no-range — guard against zero division).
    rng = df["rhi_20"] - df["rlo_20"]
    df["range_position_pct"] = (
        (df["rclose"] - df["rlo_20"]) / rng.where(rng != 0) * 100
    ).round(2)

    # --- Derived field 2: distance to breakout levels as % of rclose ---
    # Positive = price hasn't reached the level yet.
    # Negative = price already above resistance / below support.
    for window in [20, 50, 150]:
        hi_col, lo_col = f"rhi_{window}", f"rlo_{window}"
        if hi_col in df.columns:
            df[f"dist_to_rhi_{window}_pct"] = (
                (df[hi_col] - df["rclose"]) / df["rclose"] * 100
            ).round(2)
        if lo_col in df.columns:
            df[f"dist_to_rlo_{window}_pct"] = (
                (df["rclose"] - df[lo_col]) / df["rclose"] * 100
            ).round(2)

    # --- Derived field 3: relative price momentum over 20 / 50 / 150 bars ---
    # % change in rclose over the lookback window.
    # Captures whether the outperformance vs benchmark is accelerating or fading.
    for lookback in [20, 50, 150]:
        df[f"rclose_chg_{lookback}d"] = (
            df["rclose"].pct_change(periods=lookback) * 100
        ).round(2)

    # --- Derived field 4: volume trend (current vs 20-bar rolling average) ---
    # > 1.0 = volume expanding (breakout confirmation).
    # < 1.0 = volume contracting (caution — move may lack conviction).
    vol_avg = df["volume"].rolling(20, min_periods=1).mean()
    df["vol_trend"] = (df["volume"] / vol_avg.where(vol_avg != 0)).round(2)

    # --- Derived field 5: signal flip flags ---
    # 1 = signal changed value on the last bar (fresh entry/exit).
    # 0 = signal held same value (continuation).
    for sig in ["rbo_20", "rbo_50", "rbo_150"]:
        if sig in df.columns:
            df[f"{sig}_flip"] = (df[sig] != df[sig].shift(1)).astype(int)

    # --- Derived field 6: signal_age_bars per window ---
    # For each breakout signal, count consecutive bars it has held its current value.
    # Resets to 1 on every value change (fresh breakout = 1, mature trend = N).
    # Computed independently per window — rbo_150 may be active far longer than rbo_20.
    for sig in ["rbo_20", "rbo_50", "rbo_150"]:
        if sig in df.columns:
            groups = (df[sig] != df[sig].shift()).cumsum()
            df[f"{sig}_age"] = df.groupby(groups).cumcount() + 1

    # Final column order: rclose → breakout bands/signals → swing levels → derived.
    # rh3/rh4 = 2nd-highest / highest swing high (magnitude order, not time order).
    # rl3/rl4 = 2nd-deepest / deepest swing low — major structural floor.
    # symbol is redundant with ticker injected by main() — excluded.
    significant_swings = [
        c for c in ["rh3", "rh4", "rl3", "rl4"]
        if c in df.columns and df[c].notna().any()
    ]
    breakout_levels = [
        c for c in df.columns
        if re.match(r"r(hi|lo|bo)_(20|50|150)$", c)
    ]
    age_cols      = [c for c in df.columns if c.endswith("_age")]
    flip_cols     = [c for c in df.columns if c.endswith("_flip")]
    dist_cols     = sorted(c for c in df.columns if re.match(r"dist_to_r(hi|lo)_\d+_pct", c))
    momentum_cols = sorted(c for c in df.columns if re.match(r"rclose_chg_\d+d", c))
    turtle        = [c for c in df.columns if c == "rtt_5020"]

    final_cols = (
        ["rclose"]
        + breakout_levels        # rbo/rhi/rlo per 20/50/150
        + significant_swings     # rh3, rh4, rl3, rl4
        + ["range_position_pct"]
        + dist_cols              # dist_to_rhi/rlo per window
        + momentum_cols          # rclose_chg_20d/50d/150d
        + age_cols               # rbo_20/50/150_age
        + flip_cols              # rbo_20/50/150_flip
        + ["vol_trend"]
        + turtle                 # rtt_5020 (price-channel breakout confirmation)
        # rema_*/rsma_* intentionally omitted → MA trader assistant
    )
    # Deduplicate while preserving order.
    seen: set[str] = set()
    final_cols = [c for c in final_cols if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

    return df[["date", "rrg"] + final_cols]


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj) -> dict:
    """Convert a frozen dataclass to a plain JSON-safe dict (handles bool | None fields)."""
    return {k: v for k, v in obj.__dataclass_fields__.items()
            if True} | {k: getattr(obj, k) for k in obj.__dataclass_fields__}


def _compute_range_setup(df: pd.DataFrame) -> dict | None:
    """
    Run assess_range over the full ticker history and return a JSON-safe dict.

    Returns None (serialised as null) when the ticker has no recent consolidation
    window (e.g., the entire history is in an active trend) or too few bars.
    Logs a warning instead of crashing so the rest of the snapshot is unaffected.
    """
    try:
        rs: RangeSetup = assess_range(df)
        return {
            "n_resistance_touches": rs.n_resistance_touches,
            "n_support_touches":    rs.n_support_touches,
            "is_sideways":          rs.is_sideways,
            "slope_pct_per_day":    rs.slope_pct_per_day,
            "consolidation_bars":   rs.consolidation_bars,
            "band_width_pct":       rs.band_width_pct,
        }
    except ValueError as exc:
        log.warning("assess_range failed: %s", exc)
        return None


def _compute_volatility_state(df: pd.DataFrame) -> dict | None:
    """
    Run measure_volatility_compression over the full ticker history and return a
    JSON-safe dict.

    Returns None when the ticker has insufficient history for the 252-bar percentile
    rank or the slope window.  Logs a warning so the snapshot is unaffected.
    """
    try:
        vs: VolatilityState = measure_volatility_compression(df)
        return {
            "band_width_pct":      vs.band_width_pct,
            "band_width_slope":    vs.band_width_slope,
            "band_width_pct_rank": vs.band_width_pct_rank,
            "is_compressed":       vs.is_compressed,
            "history_available":   vs.history_available,
            "is_rank_reliable":    vs.is_rank_reliable,
        }
    except ValueError as exc:
        log.warning("measure_volatility_compression failed: %s", exc)
        return None


def _compute_volume_profile(df: pd.DataFrame) -> dict | None:
    """
    Run assess_volume_profile over the full ticker history and return a JSON-safe dict.

    Returns None when the ticker has no consolidation history or insufficient bars.
    Logs a warning so the rest of the snapshot is unaffected.
    """
    try:
        vp: VolumeProfile = assess_volume_profile(df)
        return {
            "vol_trend_now":      vp.vol_trend_now,
            "vol_trend_mean":     vp.vol_trend_mean,
            "vol_trend_slope":    vp.vol_trend_slope,
            "is_quiet":           vp.is_quiet,
            "is_declining":       vp.is_declining,
            "breakout_confirmed": vp.breakout_confirmed,
        }
    except ValueError as exc:
        log.warning("assess_volume_profile failed: %s", exc)
        return None


def build_snapshot(df_ticker: pd.DataFrame) -> dict:
    """
    Return the last bar of a prepared ticker DataFrame as a JSON-safe dict,
    enriched with range-quality and volume-behaviour summaries.

    The enrichments (range_setup, volatility_compression, volume_profile) are computed
    over the full ticker history before the per-bar snapshot is extracted.  Each returns
    None if the underlying primitive raises (e.g., no consolidation window found) so
    a failed enrichment never prevents the snapshot from being built.

    Raises:
        ValueError: If the DataFrame is empty after filtering.
    """
    if df_ticker.empty:
        raise ValueError("DataFrame is empty — ticker may not exist in the parquet.")

    # --- Compute enrichments over full history BEFORE slicing to last bar ---
    range_setup           = _compute_range_setup(df_ticker)
    volatility_compression = _compute_volatility_state(df_ticker)
    volume_profile        = _compute_volume_profile(df_ticker)

    # --- Last-bar snapshot ---
    df_prepared = select_columns(df_ticker)
    last = df_prepared.tail(1).copy()
    last["date"] = last["date"].dt.strftime("%Y-%m-%d")

    records = json.loads(last.to_json(orient="records", double_precision=4))
    snapshot = records[0]

    # --- Merge enrichments ---
    snapshot["range_setup"]            = range_setup
    snapshot["volatility_compression"] = volatility_compression
    snapshot["volume_profile"]         = volume_profile

    return snapshot


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class TimeframeAnalysis(BaseModel):
    signal: int = Field(description="Breakout signal value: +1 long, 0 exit/flat, -1 short.")
    signal_age: int = Field(description="Bars the signal has held its current value.")
    fresh_flip: bool = Field(description="True if the signal changed value on the last bar.")
    resistance: float = Field(description="Relative resistance level for this window.")
    support: float = Field(description="Relative support level for this window.")
    dist_to_resistance_pct: float = Field(description="% distance from rclose to resistance. Negative = already above.")
    dist_to_support_pct: float = Field(description="% distance from rclose to support.")
    momentum_pct: float | None = Field(description="% change in rclose over this window's lookback period.")
    commentary: str = Field(description="One sentence on the signal state and key level for this timeframe.")


class TurtleSignal(BaseModel):
    signal: int = Field(description="rtt_5020: +1 = broke above 20-day high, -1 = below 20-day low, 0 = inside channel.")
    aligned_with_rbo_20: bool = Field(description="True if rtt_5020 and rbo_20 have the same non-zero sign.")
    commentary: str = Field(description="One sentence on turtle signal and whether it agrees with rbo_20.")


class RiskLevels(BaseModel):
    long_stop:            float = Field(description="rlo_20 — short-term stop for a long: exit if price re-enters range below this.")
    long_structural_stop: float = Field(description="rlo_150 — major structural stop for longs: breach invalidates the long thesis.")
    short_stop:            float = Field(description="rhi_20 — short-term stop for a short: exit if price re-enters range above this.")
    short_structural_stop: float = Field(description="rhi_150 — major structural stop for shorts: breach invalidates the short thesis.")
    peak_resistance:       float = Field(description="rh4 — absolute highest swing high. Resistance for longs; target / cover level for shorts.")
    major_floor:           float = Field(description="rl4 — absolute deepest swing low. Target for longs; structural floor for shorts.")


class VolatilityCompression(BaseModel):
    band_width_pct:       float = Field(description="(rhi_20 - rlo_20) / rclose * 100 at the last bar. Low = price in a tight cage.")
    band_width_slope:     float = Field(description="OLS slope of band_width_pct over the last 40 bars (%/bar). Negative = range actively narrowing.")
    band_width_pct_rank:  float = Field(description="Percentile rank of band_width_pct vs 252-bar history (0–100). Low = historically compressed for this ticker.")
    is_compressed:        bool  = Field(description="True when band_width_slope < 0 AND band_width_pct_rank < 25. The coiled-spring signal.")
    commentary:           str   = Field(description="One sentence on compression state: is the range coiling, stable, or dissolving?")


class RangeQuality(BaseModel):
    n_resistance_touches: int   = Field(description="Distinct times price tested rhi_20 without breaking through. 2-4 = clean range.")
    n_support_touches:    int   = Field(description="Distinct times price tested rlo_20 without breaking through.")
    is_sideways:          bool  = Field(description="True when OLS slope of rclose is below 0.15%/day.")
    slope_pct_per_day:    float = Field(description="Signed OLS slope of rclose (%/day). ~0 = ideal consolidation.")
    consolidation_bars:   int   = Field(description="Total consecutive bars rbo_20 has held 0. Higher = more mature range.")
    band_width_pct:       float = Field(description="(rhi_20 - rlo_20) / rclose * 100. Low = tight range = compressed energy.")
    commentary:           str   = Field(description="One sentence assessing setup quality: touch count, sideways confirmation, compression.")


class VolumeQuality(BaseModel):
    vol_trend_mean:       float       = Field(description="Mean vol_trend over the consolidation window. < 1.0 = quiet (healthy).")
    vol_trend_slope:      float       = Field(description="OLS slope of vol_trend per bar. Negative = volume drying up (ideal).")
    is_quiet:             bool        = Field(description="True when vol_trend_mean < 1.0 — primary gate for consolidation quality.")
    is_declining:         bool        = Field(description="True when vol_trend_slope < 0 — volume actively contracting inside range.")
    breakout_confirmed:   bool | None = Field(description="True = flip bar with vol_trend >= 1.2. False = flip bar, low volume. None = not a flip bar.")
    commentary:           str         = Field(description="One sentence on volume behaviour: quiet/noisy consolidation, breakout confirmation or caution.")


class TraderAnalysis(BaseModel):
    description: str = Field(
        description=(
            "A 3-5 sentence narrative summary of the full technical picture for this stock. "
            "Written for a professional trader. Cover: regime, multi-timeframe breakout "
            "confluence (long or short — treat both directions equally), range quality, "
            "key levels, and vol_trend. Use exact numbers. No generic statements."
        )
    )
    regime: int = Field(description="rrg: +1 bullish, 0 sideways, -1 bearish.")
    confluence: Literal["full_long", "full_short", "mixed", "flat"] = Field(
        description="full_long: rbo_20/50/150 all +1. full_short: all -1. mixed: disagree. flat: all 0."
    )
    short_term: TimeframeAnalysis = Field(description="Analysis of the 20-day window.")
    medium_term: TimeframeAnalysis = Field(description="Analysis of the 50-day window.")
    long_term: TimeframeAnalysis = Field(description="Analysis of the 150-day window.")
    turtle: TurtleSignal = Field(description="Turtle price-channel breakout signal and alignment with rbo_20.")
    vol_trend: float = Field(description="Volume ratio vs 20-bar average. >1 expanding, <1 contracting.")
    range_quality: RangeQuality | None = Field(
        description="Range setup quality metrics. null when no consolidation window is available (ticker in trend)."
    )
    volatility_compression: VolatilityCompression | None = Field(
        description="Volatility compression state. null when insufficient history for slope/rank computation."
    )
    volume_quality: VolumeQuality | None = Field(
        description="Volume behaviour during consolidation. null when no consolidation history available."
    )
    risk: RiskLevels
    verdict: str = Field(
        description="Actionable one-sentence conclusion for a professional trader. Use exact numbers."
    )


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"


def ask_bo_trader(snapshot: dict, ticker: str, question: str | None) -> TraderAnalysis:
    """
    Send the ticker snapshot to an OpenAI model and return a structured analysis.

    Uses OpenAI structured output (beta.chat.completions.parse) to guarantee
    the response conforms to TraderAnalysis schema — no post-processing needed.

    Args:
        snapshot: Dict from build_snapshot — the last-bar data payload.
        ticker:   Ticker symbol string (used in the user message for clarity).
        question: Optional follow-up question from the CLI caller.

    Returns:
        TraderAnalysis Pydantic model parsed directly from the model response.
    """
    client = openai.OpenAI()

    user_content = f"Ticker: {ticker}\n\nSnapshot:\n{json.dumps(snapshot, indent=2)}"
    if question:
        user_content += f"\n\nQuestion: {question}"

    log.info("Sending snapshot for %s to %s (%d fields)", ticker, MODEL, len(snapshot))

    response = client.beta.chat.completions.parse(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=TraderAnalysis,
    )

    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a ticker snapshot to an OpenAI model as a professional trader assistant.",
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
        help="Optional follow-up question appended to the snapshot (e.g. 'Should I add?')",
    )
    parser.add_argument(
        "--data",
        default=str(RESULTS_PATH),
        help=f"Path to analysis_results.parquet (default: {RESULTS_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        log.error("Parquet not found: %s", data_path)
        sys.exit(1)

    log.info("Loading %s", data_path)
    df = pd.read_parquet(data_path)

    df_ticker = df[df["symbol"] == args.ticker].copy()
    if df_ticker.empty:
        log.error("Ticker '%s' not found in parquet. Available sample: %s", args.ticker, df["symbol"].unique()[:10].tolist())
        sys.exit(1)

    snapshot = {"ticker": args.ticker, **build_snapshot(df_ticker)}
    log.info("Snapshot built: %d fields, date=%s", len(snapshot), snapshot.get("date"))

    print("\n" + "=" * 60)
    print(f"  Snapshot — {args.ticker}")
    print("=" * 60)
    print(json.dumps(snapshot, indent=2))
    print("=" * 60 + "\n")

    analysis: TraderAnalysis = ask_bo_trader(snapshot, ticker=args.ticker, question=args.question)

    a = analysis
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  AI Trader Analysis — {args.ticker}")
    print(sep)
    print(f"  {a.description}")
    print()
    print(f"  Regime     : {a.regime}   Confluence: {a.confluence}")
    print(f"  Vol trend  : {a.vol_trend}x")
    print()
    for label, tf in [("SHORT (20d)", a.short_term), ("MED   (50d)", a.medium_term), ("LONG (150d)", a.long_term)]:
        flip = " [FLIP]" if tf.fresh_flip else ""
        print(f"  {label}  sig={tf.signal}{flip}  age={tf.signal_age}  chg={tf.momentum_pct}%")
        print(f"            res={tf.resistance}  ({tf.dist_to_resistance_pct:+.2f}%)  sup={tf.support}  ({tf.dist_to_support_pct:+.2f}%)")
        print(f"            {tf.commentary}")
    print()
    tt = a.turtle
    print(f"  Turtle     : sig={tt.signal}  aligns_with_rbo_20={tt.aligned_with_rbo_20}")
    print(f"               {tt.commentary}")
    print()
    if a.range_quality is not None:
        rq = a.range_quality
        print(f"  Range      : res_touches={rq.n_resistance_touches}  sup_touches={rq.n_support_touches}  "
              f"sideways={rq.is_sideways}  slope={rq.slope_pct_per_day:+.4f}%/day")
        print(f"               bars={rq.consolidation_bars}  bw={rq.band_width_pct:.2f}%")
        print(f"               {rq.commentary}")
    else:
        print("  Range      : no consolidation window available (ticker in trend)")
    print()
    if a.volatility_compression is not None:
        vc = a.volatility_compression
        print(f"  Volatility : bw={vc.band_width_pct:.2f}%  slope={vc.band_width_slope:+.6f}/bar  "
              f"rank={vc.band_width_pct_rank:.1f}%ile  compressed={vc.is_compressed}")
        print(f"               {vc.commentary}")
    else:
        print("  Volatility : insufficient history for compression estimate")
    print()
    if a.volume_quality is not None:
        vq = a.volume_quality
        bc = ("n/a" if vq.breakout_confirmed is None
              else ("CONFIRMED" if vq.breakout_confirmed else "WEAK"))
        print(f"  Volume     : quiet={vq.is_quiet}  declining={vq.is_declining}  "
              f"mean={vq.vol_trend_mean:.3f}  slope={vq.vol_trend_slope:+.5f}/bar  "
              f"breakout={bc}")
        print(f"               {vq.commentary}")
    else:
        print("  Volume     : no volume history available")
    print()
    r = a.risk
    print(f"  Risk (long) : stop={r.long_stop}  structural={r.long_structural_stop}")
    print(f"  Risk (short): stop={r.short_stop}  structural={r.short_structural_stop}")
    print(f"               peak res={r.peak_resistance}  major floor={r.major_floor}")
    print()
    print(f"  Verdict    : {a.verdict}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
