"""
Pure scoring functions for the trending ticker dashboard.

Two scores are computed per ticker and summed into a total:
  score_vs_bm:     sum of signal values on rclose (ticker / FTSEMIB.MI)
  score_vs_sector: sum of signal values on ticker / sector_peer_avg_close (exclude-self)

Each signal contributes +1 (long), 0 (neutral), or -1 (short).
Score range: [-N, +N] where N = number of signal columns detected.

Sector peer average uses a vectorised exclude-self approach:
  peer_avg = (sector_sum_close - ticker_close) / (n_peers - 1)
This avoids recomputing the group mean for each ticker (O(n) not O(n²)).

Singleton sectors (only one ticker) receive score_vs_sector = None.
"""

import logging
from typing import Optional

import pandas as pd

from algoshort.ohlcprocessor import OHLCProcessor
from algoshort.wrappers import generate_signals

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def detect_signal_columns(df: pd.DataFrame) -> list[str]:
    """
    Return signal column names that have a matching ``<signal>_stop_loss`` column.

    This convention is established by the algoshort pipeline: every signal
    processed by StopLossCalculator gets a companion stop-loss column.

    Args:
        df: Per-ticker DataFrame from analysis_results.parquet.

    Returns:
        Sorted list of signal column names.
    """
    cols = set(df.columns)
    return sorted(c for c in cols if f"{c}_stop_loss" in cols)


# ---------------------------------------------------------------------------
# Score vs benchmark
# ---------------------------------------------------------------------------


def compute_score_vs_bm(
    data_dict: dict[str, pd.DataFrame],
    signal_cols: list[str],
) -> pd.DataFrame:
    """
    For each ticker, sum the last-bar signal values on the relative-to-benchmark series.

    The signal values in ``analysis_results.parquet`` are already computed on
    ``rclose = ticker_close / benchmark_close``, so no additional computation
    is needed here — we only read the last row.

    Args:
        data_dict: symbol -> DataFrame loaded from analysis_results.parquet.
        signal_cols: Signal column names from ``detect_signal_columns``.

    Returns:
        DataFrame with columns:
            ticker | score_vs_bm | active_signals_bm | short_signals_bm | n_signals

    Raises:
        ValueError: If signal_cols is empty.
    """
    if not signal_cols:
        raise ValueError("signal_cols cannot be empty — run detect_signal_columns first")

    if not data_dict:
        return pd.DataFrame(
            columns=["ticker", "score_vs_bm", "active_signals_bm", "short_signals_bm", "n_signals"]
        )

    rows = []
    for ticker, df in data_dict.items():
        missing = [c for c in signal_cols if c not in df.columns]
        if missing:
            log.warning("Ticker %s missing signal columns %s — skipping", ticker, missing)
            continue

        last = df[signal_cols].iloc[-1]
        score = int(last.sum())
        active = sorted(c for c in signal_cols if last[c] == 1)
        short = sorted(c for c in signal_cols if last[c] == -1)

        rows.append(
            {
                "ticker": ticker,
                "score_vs_bm": score,
                "active_signals_bm": ", ".join(active),
                "short_signals_bm": ", ".join(short),
                "n_signals": len(signal_cols),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sector peer average (vectorised, exclude-self)
# ---------------------------------------------------------------------------


def compute_sector_peer_avg(
    ohlc_df: pd.DataFrame,
    ticker: str,
    sector_tickers: list[str],
) -> pd.DataFrame:
    """
    Compute the equal-weight average close of all sector peers, EXCLUDING ``ticker``.

    Uses a vectorised exclude-self formula to avoid recomputing the sector mean
    from scratch for each ticker:
        peer_avg = (sector_sum - ticker_close) / (n_peers - 1)

    Args:
        ohlc_df: Raw OHLC DataFrame with columns: symbol, date, close.
        ticker: The ticker to exclude from its own sector average.
        sector_tickers: All tickers in the sector (including ``ticker``).

    Returns:
        DataFrame with columns: date, close — representing the peer average close.
        Returns an empty DataFrame for singleton sectors (only one ticker).

    Invariant:
        If ``ticker`` is not present in ``sector_tickers``, all tickers in
        ``sector_tickers`` are treated as peers (no exclusion needed).
    """
    peers = [t for t in sector_tickers if t != ticker]
    if not peers:
        log.warning(
            "Ticker %s is a singleton in its sector — sector score will be NaN", ticker
        )
        return pd.DataFrame()

    peer_data = ohlc_df[ohlc_df["symbol"].isin(peers)][["date", "close"]]
    sector_avg = peer_data.groupby("date")["close"].mean().reset_index()
    # columns: date, close
    return sector_avg


# ---------------------------------------------------------------------------
# Score vs sector
# ---------------------------------------------------------------------------


def compute_score_vs_sector(
    ohlc_df: pd.DataFrame,
    ticker_to_sector: dict[str, str],
    search_spaces: tuple[dict, list, dict],
) -> pd.DataFrame:
    """
    For each ticker, compute trend score on ``ticker_close / sector_peer_avg_close``.

    Reuses the same OHLCProcessor + generate_signals pipeline as the benchmark
    score, treating the sector peer average as a synthetic "benchmark".

    Singleton sectors (only one ticker) receive score_vs_sector = None.
    Tickers where signal generation fails also receive score_vs_sector = None
    with a WARNING log entry.

    Args:
        ohlc_df: Raw OHLC DataFrame (symbol, date, open, high, low, close).
        ticker_to_sector: ticker -> sector name mapping.
        search_spaces: (tt_search_space, bo_search_space, ma_search_space)
            as returned by ``pipeline.build_search_spaces()``.

    Returns:
        DataFrame with columns:
            ticker | score_vs_sector | active_signals_sector | short_signals_sector
    """
    tt_search_space, bo_search_space, ma_search_space = search_spaces

    # Build sector -> [tickers] lookup once to avoid O(n²) dict scans.
    sector_map: dict[str, list[str]] = {}
    for t, s in ticker_to_sector.items():
        sector_map.setdefault(s, []).append(t)

    processor = OHLCProcessor()
    rows = []

    tickers = sorted(ticker_to_sector.keys())
    log.info("Computing sector scores for %d tickers", len(tickers))

    for ticker in tickers:
        sector = ticker_to_sector[ticker]
        sector_tickers = sector_map.get(sector, [])
        peer_avg = compute_sector_peer_avg(ohlc_df, ticker, sector_tickers)

        if peer_avg.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "score_vs_sector": None,
                    "active_signals_sector": "",
                    "short_signals_sector": "",
                }
            )
            continue

        ticker_df = ohlc_df[ohlc_df["symbol"] == ticker].copy()
        if ticker_df.empty:
            log.warning("No OHLC data for %s — skipping sector score", ticker)
            rows.append(
                {
                    "ticker": ticker,
                    "score_vs_sector": None,
                    "active_signals_sector": "",
                    "short_signals_sector": "",
                }
            )
            continue

        ticker_df["fx"] = 1

        try:
            rel_df = processor.calculate_relative_prices(
                stock_data=ticker_df,
                benchmark_data=peer_avg,
            )
            rel_df, sig_cols = generate_signals(
                df=rel_df,
                tt_search_space=tt_search_space,
                bo_search_space=bo_search_space,
                ma_search_space=ma_search_space,
                relative=True,
            )
            # Exclude the regime column (rrg) — it is a direction indicator,
            # not a tradeable signal, and must not contribute to the score.
            sig_cols = [c for c in sig_cols if c != "rrg"]

            last = rel_df[sig_cols].iloc[-1]
            score: Optional[int] = int(last.sum())
            active = sorted(c for c in sig_cols if last[c] == 1)
            short = sorted(c for c in sig_cols if last[c] == -1)

        except Exception as exc:
            log.warning(
                "Sector score computation failed for %s (sector=%s): %s",
                ticker,
                sector,
                exc,
            )
            score = None
            active = []
            short = []

        rows.append(
            {
                "ticker": ticker,
                "score_vs_sector": score,
                "active_signals_sector": ", ".join(active),
                "short_signals_sector": ", ".join(short),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Score history (for heatmap)
# ---------------------------------------------------------------------------


def compute_score_history(
    data_dict: dict[str, pd.DataFrame],
    signal_cols: list[str],
    n_days: int,
) -> pd.DataFrame:
    """
    Compute score_vs_bm for each ticker over the last ``n_days`` trading dates.

    The score for a given date is the sum of all signal values on that row
    (each in {-1, 0, +1}), identical to ``compute_score_vs_bm`` but applied
    to every row rather than just the last one.

    Args:
        data_dict: symbol -> DataFrame from analysis_results.parquet.
        signal_cols: Signal column names from ``detect_signal_columns``.
        n_days: Number of most recent trading dates to include. If a ticker
            has fewer rows than n_days, all available rows are used.

    Returns:
        Wide DataFrame:
            - Index (name="ticker"): ticker symbols
            - Columns: date values in ascending order
            - Values: integer score_vs_bm for that ticker on that date
    """
    series: dict[str, pd.Series] = {}

    for ticker, df in data_dict.items():
        # Deduplicate by date (keep last row per date) before slicing,
        # so duplicate-date tickers don't cause reindex failures.
        deduped = df.drop_duplicates(subset="date", keep="last")
        tail = deduped.tail(n_days)
        scores = tail[signal_cols].sum(axis=1).astype(int)
        scores.index = pd.to_datetime(tail["date"]).dt.normalize()
        series[ticker] = scores

    result = pd.DataFrame(series).T
    result.index.name = "ticker"

    # Ensure columns (dates) are in ascending order.
    result = result[sorted(result.columns)]
    return result


# ---------------------------------------------------------------------------
# Supporting per-ticker metrics
# ---------------------------------------------------------------------------


def compute_supporting_metrics(
    data_dict: dict[str, pd.DataFrame],
    signal_cols: list[str],
    lookback: int,
) -> pd.DataFrame:
    """
    Extract per-ticker metrics from the last bar of each ticker's DataFrame.

    Metrics:
      - rclose_chg_{lookback}d: percentage change in relative price over lookback window
      - stop_loss: stop-loss price from the first active (+1) signal's stop column
      - sl_dist_pct: (close - stop_loss) / close * 100
      - last_date: date of the most recent bar

    Args:
        data_dict: symbol -> DataFrame from analysis_results.parquet.
        signal_cols: Signal column names (used to find stop_loss companion columns).
        lookback: Number of bars to look back for relative return calculation.

    Returns:
        DataFrame with columns:
            ticker | rclose_chg_{lookback}d | stop_loss | sl_dist_pct | last_date
    """
    chg_col = f"rclose_chg_{lookback}d"
    rows = []

    for ticker, df in data_dict.items():
        last_row = df.iloc[-1]
        close = last_row.get("close")

        rclose_chg = None
        if len(df) > lookback and "rclose" in df.columns:
            rclose_now = df["rclose"].iloc[-1]
            rclose_prev = df["rclose"].iloc[-(lookback + 1)]
            if rclose_prev and rclose_prev != 0:
                rclose_chg = round((rclose_now / rclose_prev - 1) * 100, 2)

        # Use stop-loss from the first active (+1) signal for risk context.
        stop_loss = None
        sl_dist_pct = None
        for sig in signal_cols:
            sl_col = f"{sig}_stop_loss"
            if sl_col in df.columns and last_row.get(sig) == 1:
                sl_val = last_row[sl_col]
                if pd.notna(sl_val):
                    stop_loss = round(float(sl_val), 4)
                    if close and close != 0:
                        sl_dist_pct = round((float(close) - stop_loss) / float(close) * 100, 2)
                break

        rows.append(
            {
                "ticker": ticker,
                chg_col: rclose_chg,
                "stop_loss": stop_loss,
                "sl_dist_pct": sl_dist_pct,
                "last_date": last_row.get("date"),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dashboard assembly
# ---------------------------------------------------------------------------


def build_dashboard(
    score_bm: pd.DataFrame,
    score_sector: pd.DataFrame,
    metrics: pd.DataFrame,
    sectors_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all per-ticker data into a single ranked DataFrame.

    Sort order:
        Primary:   score_total descending
        Tiebreaker: rclose_chg_{N}d descending (relative momentum)

    Args:
        score_bm:     Output of compute_score_vs_bm.
        score_sector: Output of compute_score_vs_sector.
        metrics:      Output of compute_supporting_metrics.
        sectors_df:   DataFrame with columns: ticker, name, sector.

    Returns:
        Ranked DataFrame with a leading ``rank`` column (1-indexed).
    """
    df = score_bm.merge(score_sector, on="ticker", how="left")
    df = df.merge(metrics, on="ticker", how="left")
    df = df.merge(
        sectors_df[["ticker", "name", "sector"]].drop_duplicates("ticker"),
        on="ticker",
        how="left",
    )

    # NaN sector score (singleton) contributes 0 to the total.
    df["score_vs_sector"] = pd.to_numeric(df["score_vs_sector"], errors="coerce")
    df["score_total"] = df["score_vs_bm"].fillna(0) + df["score_vs_sector"].fillna(0)

    # Sort: score_total desc, then rclose_chg desc as tiebreaker.
    chg_col = next((c for c in df.columns if c.startswith("rclose_chg_")), None)
    sort_cols = ["score_total", chg_col] if chg_col else ["score_total"]
    sort_ascending = [False] * len(sort_cols)
    df = df.sort_values(sort_cols, ascending=sort_ascending).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    # Column order: identity → scores → signal detail → metrics
    front = ["rank", "ticker", "name", "sector", "score_total", "score_vs_bm", "score_vs_sector"]
    signal_detail = [
        "active_signals_bm",
        "short_signals_bm",
        "active_signals_sector",
        "short_signals_sector",
        "n_signals",
    ]
    remaining = [c for c in df.columns if c not in front + signal_detail]
    ordered = front + signal_detail + remaining

    return df[[c for c in ordered if c in df.columns]]
