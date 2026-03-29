"""
ask_trader.py — Send a ticker snapshot to an OpenAI model as an AI trader assistant.

Usage:
    python ask_trader.py --ticker A2A.MI
    python ask_trader.py --ticker A2A.MI --question "Should I add to this position?"

What it does:
    1. Loads analysis_results.parquet
    2. Filters to the requested ticker and selects the actionable columns
    3. Computes two derived fields:
         - range_position_pct: where rclose sits in the 20-day relative range (0=low, 100=high)
         - signal_age_bars:    how many bars rbo_20 has held its current value
    4. Takes the last bar (current state) and serialises it to JSON
    5. Sends the JSON to gpt-4o with a professional trader system prompt
    6. Prints the model's analysis to stdout

Environment:
    OPENAI_API_KEY must be set.

Columns included (last bar only):
    - Identity:       symbol, date, rrg (regime direction)
    - Relative OHLC:  ropen, rhigh, rlow, rclose
    - 20-day context: rhi_20, rlo_20, rbo_20 (breakout signal), rbo_20_stop_loss excluded
    - Swing levels:   rh1..rh4, rl1..rl4 (forward-filled to avoid NaN gaps)
    - Derived:        range_position_pct, signal_age_bars
    - Volume:         volume (absolute — used for liquidity context)

Excluded on purpose (too noisy / intermediate):
    - All *_cumul, *_returns, *_chg*, *_PL_cum columns
    - rbo_20_stop_loss, rbo_50_stop_loss, rbo_150_stop_loss
    - 50/150-day breakout detail (focus is the 20-day window per user intent)
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
  Confluence of rbo_20 = rbo_50 = rbo_150 = +1 signals the strongest long setup.
- range_position_pct: where rclose sits between rlo_20 and rhi_20.
  0% = at support, 100% = at resistance. Above 100% = breakout long in progress.
- rbo_20_age, rbo_50_age, rbo_150_age: bars each signal has held its current value.
  A rbo_150_age=90 with rbo_20_age=3 = mature long-term trend, fresh short-term re-entry.
- rbo_20_flip, rbo_50_flip, rbo_150_flip: 1 if signal changed on the last bar, else 0.
- dist_to_rhi_{N}_pct: % distance from rclose to resistance. Negative = already above it.
- dist_to_rlo_{N}_pct: % distance from rclose to support (= stop distance for longs).
- rclose_chg_20d/50d/150d: % change in relative price over 20/50/150 bars.
  Use these to assess whether outperformance vs benchmark is accelerating or fading.
- vol_trend: current volume / 20-bar average volume. >1.0 = expanding (confirming),
  <1.0 = contracting (caution — move lacks conviction).
- rema_50100150: triple EMA stack signal (+1 = EMA50>EMA100>EMA150, -1 = inverted, 0 = mixed).
- rsma_50100150: triple SMA stack signal (+1 = SMA50>SMA100>SMA150, -1 = inverted, 0 = mixed).
- rema_short_50, rema_medium_100, rema_long_150: actual EMA levels on relative price.
- rsma_short_50, rsma_medium_100, rsma_long_150: actual SMA levels on relative price.
- rtt_5020: turtle breakout signal (fast=20, slow=50). Independent confirmation of rbo_*.
- rh3, rh4: magnitude-ordered swing highs. rh4 = highest (strongest resistance).
- rl3, rl4: magnitude-ordered swing lows. rl4 = deepest (strongest support / major floor).

Output format — start with a description, then analyse each timeframe separately:
0. description: 3-5 sentence narrative of the full technical picture. Regime, signal
   confluence, MA alignment, key levels, trend quality. Exact numbers. No filler.
1. Short term (20-day): rbo_20 signal, age, flip, range_position_pct, dist_to_rhi_20_pct,
   rclose_chg_20d, vol_trend.
2. Medium term (50-day): rbo_50 signal, age, flip, dist_to_rhi_50_pct, rclose_chg_50d.
   Note if EMA/SMA crossover signals (rema_50100, rsma_50100) agree with rbo_50.
3. Long term (150-day): rbo_150 signal, age, dist_to_rhi_150_pct, rclose_chg_150d,
   rh4 (peak resistance) and rl4 (major floor). Note turtle signal (rtt_5020) alignment.
4. MA crossover synthesis: rema_50100150 and rsma_50100150 are the triple stack signals.
   Both +1 = full EMA and SMA alignment. Add turtle (rtt_5020) for independent confirmation.
5. Risk: short-term stop = rlo_20, structural stop = rlo_150.
6. Verdict: one actionable sentence using exact numbers.

Be concise. No padding. Use numbers from the data, not generic statements.\
"""

# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and compute the actionable column set for a single-ticker DataFrame.

    Columns kept:
      - Identity:    symbol, date, rrg (regime: +1 bullish / 0 sideways / -1 bearish)
      - Relative OHLC: ropen, rhigh, rlow, rclose (stock / FTSEMIB.MI)
      - 20-day levels: rhi_20 (resistance), rlo_20 (support)
      - 20-day signal: rbo_20 (+1 enter long / -1 enter short / 0 exit–no trade)
      - Swing levels:  rh1..rh4, rl1..rl4 (forward-filled)
      - Volume:        absolute volume for liquidity context
      - Derived:       range_position_pct, signal_age_bars

    Columns excluded:
      - *_cumul, *_returns, *_chg*, *_PL_cum  — intermediate analytics, not needed per-bar
      - rbo_*_stop_loss                        — ATR stop not in scope for 20-day snapshot
    """
    cols = df.columns

    range_cols = cols[cols.get_loc("ropen"): cols.get_loc("rclose") + 1].tolist()
    rbo_cols   = [c for c in cols if "rbo" in c]
    # rhi_* / rlo_* — breakout band columns (rhi_20, rhi_50, rhi_150, rlo_*).
    # Underscore suffix excludes "rhigh" and "rlow" which are already in range_cols.
    rhi_cols   = [c for c in cols if c.startswith("rhi_")]
    rlo_cols   = [c for c in cols if c.startswith("rlo_")]
    # rh\d / rl\d — swing level columns only (rh1..rh4, rl1..rl4).
    # Deliberately excludes "rhigh" and "rlow" which are already in range_cols.
    rh_cols    = [c for c in cols if re.match(r"rh\d", c)]
    rl_cols    = [c for c in cols if re.match(r"rl\d", c)]

    # MA value columns: actual EMA/SMA levels at each window (used for alignment context).
    ema_val_cols = [c for c in cols if re.match(r"rema_(short|medium|long)_\d+$", c)]
    sma_val_cols = [c for c in cols if re.match(r"rsma_(short|medium|long)_\d+$", c)]
    # MA crossover signal columns: triple stack only (50/100/150).
    # +1 = EMA50 > EMA100 > EMA150 (full bullish stack).
    # -1 = inverted. 0 = mixed / flat.
    ema_sig_cols = [c for c in cols if c == "rema_50100150"]
    sma_sig_cols = [c for c in cols if c == "rsma_50100150"]
    # Turtle signal: independent breakout confirmation.
    turtle_cols  = [c for c in cols if c == "rtt_5020"]

    selected = (
        ["symbol", "rrg", "date"]
        + range_cols
        + ["volume"]
        + rbo_cols
        + rhi_cols + rlo_cols
        + rh_cols + rl_cols
        + ema_val_cols + sma_val_cols
        + ema_sig_cols + sma_sig_cols
        + turtle_cols
    )
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

    # Final column order: rclose + 20-day levels/signal + significant swing levels + derived.
    # rh3/rh4 = 2nd-highest / highest relative swing high (magnitude order, not time order).
    # rl3/rl4 = 2nd-deepest / deepest relative swing low.
    # rh1/rh2 and rl1/rl2 are minor recent pivots — excluded as noise.
    # ropen/rhigh/rlow are intraday relative OHLC — excluded; only rclose matters EOD.
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
    ma_sig_cols   = [c for c in df.columns if re.match(r"r(ema|sma)_\d+", c) and "_stop" not in c]
    ma_val_cols   = [c for c in df.columns if re.match(r"r(ema|sma)_(short|medium|long)_\d+", c)]
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
        + turtle                 # rtt_5020
        + ma_sig_cols            # rema/rsma crossover signals
        + ma_val_cols            # rema/rsma actual MA levels
    )
    # Deduplicate while preserving order.
    seen: set[str] = set()
    final_cols = [c for c in final_cols if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

    return df[["date", "rrg"] + final_cols]


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


def build_snapshot(df_ticker: pd.DataFrame) -> dict:
    """
    Return the last bar of a prepared ticker DataFrame as a JSON-safe dict.

    Raises:
        ValueError: If the DataFrame is empty after filtering.
    """
    if df_ticker.empty:
        raise ValueError("DataFrame is empty — ticker may not exist in the parquet.")

    df_prepared = select_columns(df_ticker)
    last = df_prepared.tail(1).copy()
    last["date"] = last["date"].dt.strftime("%Y-%m-%d")

    records = json.loads(last.to_json(orient="records", double_precision=4))
    return records[0]


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


class MACrossover(BaseModel):
    ema_triple: int = Field(description="rema_50100150: +1 = EMA50>EMA100>EMA150, -1 = inverted stack, 0 = mixed.")
    sma_triple: int = Field(description="rsma_50100150: +1 = SMA50>SMA100>SMA150, -1 = inverted stack, 0 = mixed.")
    turtle: int = Field(description="rtt_5020 signal: +1 long, -1 short, 0 flat.")
    aligned_bullish: bool = Field(description="True if ema_triple, sma_triple, and turtle are all +1.")
    commentary: str = Field(description="One sentence on MA stack alignment and what it implies.")


class RiskLevels(BaseModel):
    short_term_stop: float = Field(description="rlo_20 — short-term stop zone for a long.")
    structural_stop: float = Field(description="rlo_150 — major structural floor.")
    peak_resistance: float = Field(description="rh4 — absolute highest swing high.")
    major_floor: float = Field(description="rl4 — absolute deepest swing low.")


class TraderAnalysis(BaseModel):
    description: str = Field(
        description=(
            "A 3-5 sentence narrative summary of the full technical picture for this stock. "
            "Written for a professional trader. Cover: regime, multi-timeframe signal state, "
            "MA stack alignment, key levels to watch, and overall trend quality. "
            "Use exact numbers from the data. No generic statements."
        )
    )
    regime: int = Field(description="rrg: +1 bullish, 0 sideways, -1 bearish.")
    confluence: Literal["full_long", "full_short", "mixed", "flat"] = Field(
        description="full_long: rbo_20/50/150 all +1. full_short: all -1. mixed: disagree. flat: all 0."
    )
    short_term: TimeframeAnalysis = Field(description="Analysis of the 20-day window.")
    medium_term: TimeframeAnalysis = Field(description="Analysis of the 50-day window.")
    long_term: TimeframeAnalysis = Field(description="Analysis of the 150-day window.")
    ma_crossover: MACrossover = Field(description="Moving average crossover signals and stack alignment.")
    vol_trend: float = Field(description="Volume ratio vs 20-bar average. >1 expanding, <1 contracting.")
    risk: RiskLevels
    verdict: str = Field(
        description="Actionable one-sentence conclusion for a professional trader. Use exact numbers."
    )


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"


def ask_trader(snapshot: dict, ticker: str, question: str | None) -> TraderAnalysis:
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

    analysis: TraderAnalysis = ask_trader(snapshot, ticker=args.ticker, question=args.question)

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
    ma = a.ma_crossover
    print(f"  MA cross   : ema_triple={ma.ema_triple}  sma_triple={ma.sma_triple}  turtle={ma.turtle}  aligned={ma.aligned_bullish}")
    print(f"               {ma.commentary}")
    print()
    r = a.risk
    print(f"  Risk       : ST stop={r.short_term_stop}  struct stop={r.structural_stop}")
    print(f"               peak res={r.peak_resistance}  major floor={r.major_floor}")
    print()
    print(f"  Verdict    : {a.verdict}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
