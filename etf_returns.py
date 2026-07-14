"""
etf_returns.py — Multi-timeframe return analysis for the ETF universe.

Computes each ETF's total return over several lookback windows from the
long/tidy historical OHLC parquet, then ranks the top out-performers per
window. Returns use the close **on or before** each target date, so there is no
lookahead and a fund without enough history simply yields NaN for that window
(and drops out of the ranking rather than distorting it).

Return definition
-----------------
    return(window) = close(as_of) / close(as_of - window) - 1

where ``close(d)`` is the last available bar on or before date ``d``. YTD uses
the prior calendar year-end close as the base.

Usage
-----
    python etf_returns.py            # writes data/results/etf/ + prints top-N
    python etf_returns.py --top 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from etf_liquidity import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_ACTIVE_FRAC,
    DEFAULT_MIN_TRADED_VALUE,
    liquid_symbols,
    liquidity_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HISTORICAL_PATH = Path("data/ohlc/historical/etf/ohlc_data.parquet")
SEED_PATH = Path("data/ticker/etf/ticker_active.xlsx")
RESULTS_DIR = Path("data/results/etf")

# Calendar-day lookbacks. Ordered short -> long for readable output.
DEFAULT_WINDOWS_DAYS: Dict[str, int] = {
    "1W": 7,
    "1M": 30,
    "3M": 91,
    "6M": 182,
    "1Y": 365,
    "3Y": 1095,
}

# Very-short-term lookbacks for the up/down snapshot. "1D" = last trading-day
# move (a calendar day back over a weekend lands on the previous session's
# close, so on a Monday "1D" is the Fri->Mon move).
SHORT_TERM_WINDOWS_DAYS: Dict[str, int] = {
    "1D": 1,
    "1W": 7,
}

# Any |single-day close change| above this within a window is treated as a
# split / re-denomination data artifact (real ETFs don't move this much in a
# day), and the affected window return is dropped.
DEFAULT_MAX_DAILY_MOVE = 0.50


def latest_close_asof(ohlc: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Last close on or before ``as_of`` for each symbol (index: symbol)."""
    upto = ohlc[ohlc["date"] <= as_of]
    if upto.empty:
        return pd.Series(dtype="float64")
    # Sort by date then take the last row per symbol = the most recent close.
    last = upto.sort_values("date").groupby("symbol")["close"].last()
    return last


def period_return(
    ohlc: pd.DataFrame,
    as_of: pd.Timestamp,
    base_date: pd.Timestamp,
    method: str = "simple",
) -> pd.Series:
    """Per-symbol return between ``base_date`` and ``as_of``.

    ``method`` selects the return convention:
      - ``"simple"``: discrete return ``close(as_of)/close(base) - 1``
      - ``"log"``: continuously-compounded ``ln(close(as_of)/close(base))``

    NaN for any symbol lacking a bar on/before ``base_date`` (insufficient
    history) — such symbols must not appear in a ranking. Both conventions
    preserve ranking order, so top/bottom lists are identical; only the
    magnitudes differ.
    """
    if method not in ("simple", "log"):
        raise ValueError(f"unknown method {method!r}; expected 'simple' or 'log'")

    latest = latest_close_asof(ohlc, as_of)
    base = latest_close_asof(ohlc, base_date)
    # Align on symbols present at as_of; symbols missing a base close -> NaN.
    base = base.reindex(latest.index)
    ratio = latest / base

    if method == "log":
        return np.log(ratio)
    return ratio - 1.0


def price_artifact_symbols(
    ohlc: pd.DataFrame,
    base_date: pd.Timestamp,
    as_of: pd.Timestamp,
    max_daily_move: float = DEFAULT_MAX_DAILY_MOVE,
) -> list:
    """Symbols whose close makes an implausible single-day move within
    ``(base_date, as_of]`` — a signature of an unadjusted split / re-denomination.

    Daily moves are computed over full history (so the first in-window bar is
    measured against the base bar), then sliced to the window. A ``max_daily_move``
    of 0.5 means "flag any |day-over-day close change| > 50%"; real ETFs (even
    2x leveraged) do not move that much in a day, so such a jump is a data
    artifact, and the affected window return is unreliable.
    """
    df = ohlc[["symbol", "date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])
    df["dmove"] = df.groupby("symbol")["close"].pct_change().abs()

    window = df[(df["date"] > pd.Timestamp(base_date)) & (df["date"] <= pd.Timestamp(as_of))]
    worst = window.groupby("symbol")["dmove"].max()
    return sorted(worst[worst > max_daily_move].index.tolist())


def compute_returns(
    ohlc: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    windows_days: Optional[Dict[str, int]] = None,
    include_ytd: bool = True,
    method: str = "simple",
    max_daily_move: Optional[float] = None,
) -> pd.DataFrame:
    """Return a wide table (index: symbol) with one column per window.

    ``as_of`` defaults to the latest date in ``ohlc``. YTD (when included) uses
    the prior year-end close as its base. ``method`` is ``"simple"`` or ``"log"``
    (see ``period_return``). When ``max_daily_move`` is set, any symbol with an
    implausible single-day move within a window (a split/re-denomination
    artifact) gets NaN for **that** window, so it drops from that window's
    ranking while its clean windows are unaffected.
    """
    if windows_days is None:
        windows_days = DEFAULT_WINDOWS_DAYS

    ohlc = ohlc.copy()
    ohlc["date"] = pd.to_datetime(ohlc["date"])
    if as_of is None:
        as_of = ohlc["date"].max()
    as_of = pd.Timestamp(as_of)

    def _column(base_date: pd.Timestamp) -> pd.Series:
        col = period_return(ohlc, as_of, base_date, method=method)
        if max_daily_move is not None:
            flagged = price_artifact_symbols(ohlc, base_date, as_of, max_daily_move)
            col.loc[col.index.isin(flagged)] = np.nan
        return col

    columns = {}
    for label, days in windows_days.items():
        columns[label] = _column(as_of - pd.Timedelta(days=days))

    if include_ytd:
        columns["YTD"] = _column(pd.Timestamp(year=as_of.year - 1, month=12, day=31))

    return pd.DataFrame(columns)


def _ranked(returns: pd.DataFrame, window: str, n: int, ascending: bool) -> pd.DataFrame:
    return (
        returns[[window]]
        .dropna()
        .sort_values(window, ascending=ascending)
        .head(n)
        .reset_index()
        .rename(columns={"index": "symbol"})
    )


def top_outperformers(returns: pd.DataFrame, window: str, n: int = 15) -> pd.DataFrame:
    """Top ``n`` symbols by return in ``window``, best first, NaNs dropped.

    Returns a tidy frame with columns ``symbol`` and ``window``.
    """
    return _ranked(returns, window, n, ascending=False)


def bottom_underperformers(returns: pd.DataFrame, window: str, n: int = 15) -> pd.DataFrame:
    """Bottom ``n`` symbols by return in ``window``, worst first, NaNs dropped."""
    return _ranked(returns, window, n, ascending=True)


# ---------------------------------------------------------------------------
# Report / CLI
# ---------------------------------------------------------------------------

def _name_map(seed_path: Path = SEED_PATH) -> Dict[str, str]:
    if not seed_path.exists():
        return {}
    seed = pd.read_excel(seed_path)
    return dict(zip(seed["ticker"], seed["name"]))


def build_report(returns: pd.DataFrame, names: Dict[str, str], top: int,
                 method: str = "simple", screen: str = "") -> str:
    """Human-readable top-N-per-window report with the universe median for context."""
    lines = [
        "ETF multi-timeframe return analysis — top out-performers",
        f"universe: {len(returns)} ETFs   (returns: {method}, cumulative, close-to-close)",
    ]
    if screen:
        lines.append(f"liquidity screen: {screen}")
    lines.append("")
    for window in [c for c in returns.columns]:
        median = returns[window].median()
        lines.append(f"== {window}  (universe median {median:+.1%}) ==")
        table = top_outperformers(returns, window, n=top)
        for rank, (_, row) in enumerate(table.iterrows(), start=1):
            symbol = row["symbol"]
            name = names.get(symbol, "")
            lines.append(f"  {rank:>2}. {symbol:<10} {row[window]:+7.1%}  {name}")
        lines.append("")
    return "\n".join(lines)


def build_up_down_report(returns: pd.DataFrame, names: Dict[str, str], top: int,
                         method: str = "simple", screen: str = "") -> str:
    """Short-term who's-up / who's-down report: per window, the up/down split
    plus the top gainers and top decliners."""
    lines = [
        "ETF short-term movers — who is up and who is down",
        f"universe: {len(returns)} ETFs   (windows: {', '.join(returns.columns)}; returns: {method})",
    ]
    if screen:
        lines.append(f"liquidity screen: {screen}")
    lines.append("")
    for window in returns.columns:
        col = returns[window].dropna()
        up, down, flat = int((col > 0).sum()), int((col < 0).sum()), int((col == 0).sum())
        lines.append(f"== {window}  —  {up} up / {down} down / {flat} flat  (median {col.median():+.1%}) ==")

        lines.append(f"  UP (top {top}):")
        for rank, (_, row) in enumerate(top_outperformers(returns, window, top).iterrows(), 1):
            sym = row["symbol"]
            lines.append(f"    {rank:>2}. {sym:<10} {row[window]:+7.1%}  {names.get(sym, '')}")

        lines.append(f"  DOWN (bottom {top}):")
        for rank, (_, row) in enumerate(bottom_underperformers(returns, window, top).iterrows(), 1):
            sym = row["symbol"]
            lines.append(f"    {rank:>2}. {sym:<10} {row[window]:+7.1%}  {names.get(sym, '')}")
        lines.append("")
    return "\n".join(lines)


def write_short_term_report(short: pd.DataFrame, names: Dict[str, str], top: int,
                            method: str = "simple", screen: str = "",
                            results_dir: Path = RESULTS_DIR) -> Path:
    """Write the short-term up/down text report and return its path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    report = build_up_down_report(short, names, top, method=method, screen=screen)
    path = results_dir / "short_term_movers.txt"
    path.write_text(report, encoding="utf-8")
    log.info("Wrote %s", path)
    print(report)
    return path


def write_outputs(returns: pd.DataFrame, names: Dict[str, str], top: int,
                  method: str = "simple", screen: str = "",
                  results_dir: Path = RESULTS_DIR) -> None:
    """Write an xlsx (one sheet per window) and a text report."""
    results_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = results_dir / "returns_ranking.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        for window in returns.columns:
            table = top_outperformers(returns, window, n=top)
            table["name"] = table["symbol"].map(names)
            table.to_excel(writer, sheet_name=window, index=False)
    log.info("Wrote %s", xlsx_path)

    report = build_report(returns, names, top, method=method, screen=screen)
    txt_path = results_dir / "returns_report.txt"
    txt_path.write_text(report, encoding="utf-8")
    log.info("Wrote %s", txt_path)
    print(report)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-timeframe ETF return analysis (top out-performers).")
    p.add_argument("--top", type=int, default=15, help="How many out-performers to list per window.")
    p.add_argument("--method", choices=["simple", "log"], default="simple",
                   help="Return convention: 'simple' (discrete) or 'log' (continuously compounded).")
    p.add_argument("--min-traded-value", type=float, default=DEFAULT_MIN_TRADED_VALUE,
                   help="Liquidity floor: min median daily traded value (close*volume).")
    p.add_argument("--min-active", type=float, default=DEFAULT_MIN_ACTIVE_FRAC,
                   help="Liquidity floor: min fraction of days with volume > 0.")
    p.add_argument("--liquidity-lookback", type=int, default=DEFAULT_LOOKBACK_DAYS,
                   help="Lookback (days) for the liquidity screen.")
    p.add_argument("--no-liquidity-filter", action="store_true",
                   help="Disable the illiquidity screen (rank the full universe).")
    p.add_argument("--max-daily-move", type=float, default=DEFAULT_MAX_DAILY_MOVE,
                   help="Flag |single-day close change| above this as a split/re-denomination "
                        "artifact and drop the affected window return.")
    p.add_argument("--no-artifact-filter", action="store_true",
                   help="Disable the split/re-denomination artifact screen.")
    return p.parse_args()


def _write_flagged_artifacts(ohlc: pd.DataFrame, names: Dict[str, str], as_of: pd.Timestamp,
                             max_daily_move: float, results_dir: Path = RESULTS_DIR) -> None:
    """List symbols with a suspect price jump over the analysis horizon (~3Y),
    for review — these are excluded per-window from the rankings."""
    horizon = max(DEFAULT_WINDOWS_DAYS.values())
    flagged = price_artifact_symbols(ohlc, as_of - pd.Timedelta(days=horizon), as_of, max_daily_move)
    lines = [
        "ETF price artifacts (suspect split / re-denomination — excluded from rankings)",
        f"criterion: |single-day close change| > {max_daily_move:.0%} within the last {horizon}d",
        f"flagged: {len(flagged)}",
        "",
    ]
    lines += [f"  {sym:<10} {names.get(sym, '')}" for sym in flagged]
    (results_dir / "flagged_artifacts.txt").write_text("\n".join(lines), encoding="utf-8")
    log.info("Flagged %d price-artifact symbol(s)", len(flagged))


def _apply_liquidity_filter(ohlc, args):
    """Return (kept_symbols, screen_note). Empty screen_note means no filter."""
    if args.no_liquidity_filter:
        return None, ""
    stats = liquidity_stats(ohlc, lookback_days=args.liquidity_lookback)
    kept = liquid_symbols(stats, args.min_traded_value, args.min_active)
    note = (f"median daily traded value >= {args.min_traded_value:,.0f} "
            f"& active >= {args.min_active:.0%} over {args.liquidity_lookback}d "
            f"-> {len(kept)} of {len(stats)} ETFs")
    log.info("Liquidity screen kept %d of %d ETFs", len(kept), len(stats))
    return set(kept), note


def main() -> None:
    args = _parse_args()
    ohlc = pd.read_parquet(HISTORICAL_PATH)
    names = _name_map()

    kept, screen = _apply_liquidity_filter(ohlc, args)
    max_move = None if args.no_artifact_filter else args.max_daily_move
    if max_move is not None:
        screen = f"{screen}; drop |1-day move| > {max_move:.0%} (split/re-denomination)"

    def _screened(returns: pd.DataFrame) -> pd.DataFrame:
        return returns if kept is None else returns[returns.index.isin(kept)]

    returns = _screened(compute_returns(ohlc, method=args.method, max_daily_move=max_move))
    write_outputs(returns, names, top=args.top, method=args.method, screen=screen)

    short = _screened(compute_returns(ohlc, windows_days=SHORT_TERM_WINDOWS_DAYS,
                                      include_ytd=False, method=args.method, max_daily_move=max_move))
    write_short_term_report(short, names, top=args.top, method=args.method, screen=screen)

    if max_move is not None:
        _write_flagged_artifacts(ohlc, names, pd.to_datetime(ohlc["date"]).max(), max_move)


if __name__ == "__main__":
    main()
