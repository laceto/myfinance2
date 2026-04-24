"""
ta/breakout/bo_snapshot.py — Range-breakout snapshot builder.

Responsibility: pure data layer.
    - Column selection and derived-field computation from analysis_results.parquet
    - Enrichment with RangeSetup, VolatilityState, VolumeProfile
    - Assembly of the last-bar JSON-safe snapshot dict

No AI, no OpenAI, no CLI concerns live here.

Public API:
    build_snapshot(df_ticker: pd.DataFrame) -> dict
        Build snapshot from a pre-filtered single-ticker DataFrame.
        Use when the caller already holds the parquet data in memory
        (e.g. app.py serving multiple tickers, batch_trader.py bulk runs).

    build_snapshot_from_parquet(ticker: str, data_path: Path = RESULTS_PATH) -> dict
        Convenience entry point: load parquet, filter to ticker, build snapshot.
        Use when the caller only has a ticker string and a file path
        (e.g. ask_bo_trader.py CLI, one-off scripts).

    select_columns(df: pd.DataFrame) -> pd.DataFrame
    RESULTS_PATH
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

from ta.breakout.range_quality import (
    RangeSetup,
    VolatilityState,
    assess_range,
    measure_volatility_compression,
)
from ta.breakout.volume import VolumeProfile, assess_volume_profile

__all__ = [
    "RESULTS_PATH",
    "select_columns",
    "build_snapshot",
    "build_snapshot_from_parquet",
]

log = logging.getLogger(__name__)

RESULTS_PATH = Path("data/results/it/analysis_results.parquet")


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
    # startswith("rhi_") / startswith("rlo_") excludes "rhigh" / "rlow"
    rhi_cols = [c for c in cols if c.startswith("rhi_")]
    rlo_cols = [c for c in cols if c.startswith("rlo_")]
    turtle_cols = [c for c in cols if c == "rtt_5020"]

    selected = (
        ["symbol", "rrg", "date", "rclose"]
        + ["volume"]
        + rbo_cols
        + rhi_cols + rlo_cols
        + turtle_cols
    )
    for sw in ["rh3", "rh4", "rl3", "rl4"]:
        if sw in cols:
            selected.append(sw)

    df = df[selected].copy()

    drop_patterns = [r"cumul$", r"returns$", r"chg", r"cum"]
    drop_exact    = ["rbo_20_stop_loss", "rbo_150_stop_loss", "rbo_50_stop_loss"]
    to_drop = [
        c for c in df.columns
        if any(re.search(p, c) for p in drop_patterns)
    ] + drop_exact
    df = df.drop(columns=to_drop, errors="ignore")

    swing_cols = [c for c in ["rh1", "rh2", "rh3", "rh4", "rl1", "rl2", "rl3", "rl4"] if c in df.columns]
    if swing_cols:
        df[swing_cols] = df[swing_cols].ffill().bfill()

    # --- Derived field 1: range_position_pct ---
    rng = df["rhi_20"] - df["rlo_20"]
    df["range_position_pct"] = (
        (df["rclose"] - df["rlo_20"]) / rng.where(rng != 0) * 100
    ).round(2)

    # --- Derived field 2: distance to breakout levels as % of rclose ---
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
    for lookback in [20, 50, 150]:
        df[f"rclose_chg_{lookback}d"] = (
            df["rclose"].pct_change(periods=lookback) * 100
        ).round(2)

    # --- Derived field 4: volume trend (current vs 20-bar rolling average) ---
    vol_avg = df["volume"].rolling(20, min_periods=1).mean()
    df["vol_trend"] = (df["volume"] / vol_avg.where(vol_avg != 0)).round(2)

    # --- Derived field 5: signal flip flags ---
    for sig in ["rbo_20", "rbo_50", "rbo_150"]:
        if sig in df.columns:
            df[f"{sig}_flip"] = (df[sig] != df[sig].shift(1)).astype(int)

    # --- Derived field 6: signal age in bars per window ---
    for sig in ["rbo_20", "rbo_50", "rbo_150"]:
        if sig in df.columns:
            groups = (df[sig] != df[sig].shift()).cumsum()
            df[f"{sig}_age"] = df.groupby(groups).cumcount() + 1

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
        + breakout_levels
        + significant_swings
        + ["range_position_pct"]
        + dist_cols
        + momentum_cols
        + age_cols
        + flip_cols
        + ["vol_trend"]
        + turtle
    )
    seen: set[str] = set()
    final_cols = [c for c in final_cols if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

    return df[["date", "rrg"] + final_cols]


# ---------------------------------------------------------------------------
# Enrichment helpers (private — called only by build_snapshot)
# ---------------------------------------------------------------------------


def _compute_range_setup(df: pd.DataFrame) -> dict | None:
    """
    Run assess_range over the full ticker history and return a JSON-safe dict.

    Returns None when the ticker has no recent consolidation window or too few bars.
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
    Run measure_volatility_compression over the full ticker history.

    Returns None when the ticker has insufficient history for the 252-bar rank.
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
    Run assess_volume_profile over the full ticker history.

    Returns None when no consolidation history is available.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_snapshot(df_ticker: pd.DataFrame) -> dict:
    """
    Return the last bar of a prepared ticker DataFrame as a JSON-safe dict,
    enriched with range-quality and volume-behaviour summaries.

    Enrichments are computed over the full ticker history before the per-bar
    snapshot is extracted.  Each returns None if the underlying primitive raises
    (e.g., no consolidation window found) — a failed enrichment never blocks
    the snapshot from being built.

    Args:
        df_ticker: Full history DataFrame for a single ticker, as loaded from
                   analysis_results.parquet and pre-filtered by symbol.

    Returns:
        JSON-safe dict with last-bar fields + range_setup, volatility_compression,
        volume_profile enrichments.

    Raises:
        ValueError: If df_ticker is empty.
    """
    if df_ticker.empty:
        raise ValueError("DataFrame is empty — ticker may not exist in the parquet.")

    range_setup            = _compute_range_setup(df_ticker)
    volatility_compression = _compute_volatility_state(df_ticker)
    volume_profile         = _compute_volume_profile(df_ticker)

    df_prepared = select_columns(df_ticker)
    last = df_prepared.tail(1).copy()
    last["date"] = last["date"].dt.strftime("%Y-%m-%d")

    records = json.loads(last.to_json(orient="records", double_precision=4))
    snapshot = records[0]

    snapshot["range_setup"]            = range_setup
    snapshot["volatility_compression"] = volatility_compression
    snapshot["volume_profile"]         = volume_profile

    return snapshot


def build_snapshot_from_parquet(
    ticker: str,
    data_path: Path = RESULTS_PATH,
) -> dict:
    """
    Load the parquet at data_path, filter to ticker, and return its last-bar snapshot.

    This is the convenience entry point for callers that only have a ticker string.
    Callers that already hold the full DataFrame (e.g. app.py, batch_trader.py) should
    call build_snapshot(df_ticker) directly to avoid redundant parquet I/O.

    Args:
        ticker:    Yahoo Finance ticker symbol (e.g. "A2A.MI").
        data_path: Path to analysis_results.parquet. Defaults to RESULTS_PATH.

    Returns:
        JSON-safe dict — same shape as build_snapshot().

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If ticker is not found in the parquet.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Parquet not found: {data_path}")

    log.info("Loading %s", data_path)
    df = pd.read_parquet(data_path)

    df_ticker = df[df["symbol"] == ticker].copy()
    if df_ticker.empty:
        sample = df["symbol"].unique()[:10].tolist()
        raise ValueError(
            f"Ticker '{ticker}' not found in {data_path}. "
            f"Available sample: {sample}"
        )

    return build_snapshot(df_ticker)
