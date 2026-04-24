"""
ta/ma/ma_snapshot.py — MA crossover snapshot builder.

Responsibility: pure data layer.
    - Column selection and derived-field computation from analysis_results.parquet
    - Enrichment with MATrendStrength and MAVolumeProfile
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
        (e.g. ask_ma_trader.py CLI, one-off scripts, notebooks).

    select_columns(df: pd.DataFrame) -> pd.DataFrame
    RESULTS_PATH
"""

import json
import logging
from pathlib import Path

import pandas as pd

from ta.ma.trend_quality import MATrendStrength, assess_ma_trend
from ta.ma.volume import MAVolumeProfile, assess_ma_volume

__all__ = [
    "RESULTS_PATH",
    "select_columns",
    "build_snapshot",
    "build_snapshot_from_parquet",
]

log = logging.getLogger(__name__)

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
# Column selection
# ---------------------------------------------------------------------------


def _compute_signal_age_and_flip(
    df: pd.DataFrame, signal_cols: list[str]
) -> pd.DataFrame:
    """
    Compute age and flip flags for each MA signal column.

    Age: consecutive bars the signal has held its current value (resets to 1 on change).
    Flip: 1 if the signal changed on that bar vs the previous bar, else 0.

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
        df[f"{sig}_flip"] = (df[sig] != df[sig].shift(1)).astype(int)
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
    # --- Compute age and flip before column selection (needs full signal history) ---
    df = _compute_signal_age_and_flip(df, ALL_SIGNAL_COLS)

    selected = ["symbol", "date", "rrg", "rclose"]

    for col in (
        EMA_SIGNAL_COLS + SMA_SIGNAL_COLS
        + EMA_LEVEL_COLS + SMA_LEVEL_COLS
        + EMA_STOP_COLS
    ):
        if col in df.columns:
            selected.append(col)

    for sw in ["rh3", "rh4", "rl3", "rl4"]:
        if sw in df.columns:
            selected.append(sw)

    age_cols  = [c for c in df.columns if c.endswith("_age")  and any(s in c for s in ALL_SIGNAL_COLS)]
    flip_cols = [c for c in df.columns if c.endswith("_flip") and any(s in c for s in ALL_SIGNAL_COLS)]
    selected += sorted(age_cols) + sorted(flip_cols)

    df = df[selected].copy()

    swing_cols = [c for c in ["rh3", "rh4", "rl3", "rl4"] if c in df.columns]
    if swing_cols:
        df[swing_cols] = df[swing_cols].ffill().bfill()

    # --- Derived: distance from rclose to each MA level (% of rclose) ---
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

    df = df.drop(columns=["volume", "symbol"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Enrichment helpers (private — called only by build_snapshot)
# ---------------------------------------------------------------------------


def _compute_trend_strength(df: pd.DataFrame) -> dict | None:
    """
    Run assess_ma_trend over the full ticker history and return a JSON-safe dict.

    Returns None when insufficient history or missing columns.
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
    Returns None on failure.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_snapshot(df_ticker: pd.DataFrame) -> dict:
    """
    Return the last bar of a prepared ticker DataFrame as a JSON-safe dict,
    enriched with trend-strength and volume-behaviour summaries.

    Enrichments are computed over the full ticker history before slicing to the
    last bar. Each returns None on failure so a failed enrichment never prevents
    the snapshot from being built.

    Args:
        df_ticker: Full history DataFrame for a single ticker, as loaded from
                   analysis_results.parquet and pre-filtered by symbol.

    Returns:
        JSON-safe dict with last-bar fields + trend_strength, volume_profile enrichments.

    Raises:
        ValueError: If df_ticker is empty.
    """
    if df_ticker.empty:
        raise ValueError("DataFrame is empty — ticker may not exist in the parquet.")

    trend_strength = _compute_trend_strength(df_ticker)
    volume_profile = _compute_volume_profile(df_ticker)

    df_prepared = select_columns(df_ticker)
    last = df_prepared.tail(1).copy()
    last["date"] = last["date"].dt.strftime("%Y-%m-%d")

    records  = json.loads(last.to_json(orient="records", double_precision=4))
    snapshot = records[0]

    snapshot["trend_strength"] = trend_strength
    snapshot["volume_profile"] = volume_profile

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
