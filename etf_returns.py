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

import pandas as pd

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
) -> pd.Series:
    """Per-symbol total return between ``base_date`` and ``as_of``.

    NaN for any symbol lacking a bar on/before ``base_date`` (insufficient
    history) — such symbols must not appear in a ranking.
    """
    latest = latest_close_asof(ohlc, as_of)
    base = latest_close_asof(ohlc, base_date)
    # Align on symbols present at as_of; symbols missing a base close -> NaN.
    base = base.reindex(latest.index)
    return latest / base - 1.0


def compute_returns(
    ohlc: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    windows_days: Optional[Dict[str, int]] = None,
    include_ytd: bool = True,
) -> pd.DataFrame:
    """Return a wide table (index: symbol) with one column per window.

    ``as_of`` defaults to the latest date in ``ohlc``. YTD (when included) uses
    the prior year-end close as its base.
    """
    if windows_days is None:
        windows_days = DEFAULT_WINDOWS_DAYS

    ohlc = ohlc.copy()
    ohlc["date"] = pd.to_datetime(ohlc["date"])
    if as_of is None:
        as_of = ohlc["date"].max()
    as_of = pd.Timestamp(as_of)

    columns = {}
    for label, days in windows_days.items():
        columns[label] = period_return(ohlc, as_of, as_of - pd.Timedelta(days=days))

    if include_ytd:
        prior_year_end = pd.Timestamp(year=as_of.year - 1, month=12, day=31)
        columns["YTD"] = period_return(ohlc, as_of, prior_year_end)

    return pd.DataFrame(columns)


def top_outperformers(returns: pd.DataFrame, window: str, n: int = 15) -> pd.DataFrame:
    """Top ``n`` symbols by return in ``window``, best first, NaNs dropped.

    Returns a tidy frame with columns ``symbol`` and ``window``.
    """
    ranked = (
        returns[[window]]
        .dropna()
        .sort_values(window, ascending=False)
        .head(n)
        .reset_index()
        .rename(columns={"index": "symbol"})
    )
    return ranked


# ---------------------------------------------------------------------------
# Report / CLI
# ---------------------------------------------------------------------------

def _name_map(seed_path: Path = SEED_PATH) -> Dict[str, str]:
    if not seed_path.exists():
        return {}
    seed = pd.read_excel(seed_path)
    return dict(zip(seed["ticker"], seed["name"]))


def build_report(returns: pd.DataFrame, names: Dict[str, str], top: int) -> str:
    """Human-readable top-N-per-window report with the universe median for context."""
    lines = [
        "ETF multi-timeframe return analysis — top out-performers",
        f"universe: {len(returns)} ETFs",
        "",
    ]
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


def write_outputs(returns: pd.DataFrame, names: Dict[str, str], top: int,
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

    report = build_report(returns, names, top)
    txt_path = results_dir / "returns_report.txt"
    txt_path.write_text(report, encoding="utf-8")
    log.info("Wrote %s", txt_path)
    print(report)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-timeframe ETF return analysis (top out-performers).")
    p.add_argument("--top", type=int, default=15, help="How many out-performers to list per window.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ohlc = pd.read_parquet(HISTORICAL_PATH)
    returns = compute_returns(ohlc)
    write_outputs(returns, _name_map(), top=args.top)


if __name__ == "__main__":
    main()
