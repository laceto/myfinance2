"""
etf_sector_rotation.py — detect sector rotation from the ETF return data.

WHAT
    Groups the per-window ETF returns (produced by ``etf_returns``) into sectors
    and ranks them by *momentum acceleration* — how the last month's pace
    compares to the prior quarter's trend — to surface money rotating **into**
    or **out of** sectors.

WHY
    ``returns_ranking.xlsx`` only holds the top-N individual funds per window.
    Rotation is a *sector-level* signal: it needs every liquid fund grouped by
    theme, then a leader/laggard read across timeframes. This script is a
    read-only, on-demand companion to the daily pipeline — it is **not** wired
    into GitHub Actions.

HOW
    1. Load the historical OHLC parquet.
    2. Apply the *same* screens as the daily pipeline (``etf_liquidity`` for the
       illiquidity screen, ``etf_returns.compute_returns`` with the split /
       re-denomination artifact screen) — single source of truth, no re-derived
       return math.
    3. Classify each fund into a sector from its name (keyword rules).
    4. Median return per sector per window + fund count; drop thin buckets.
    5. Acceleration metrics + a human-readable rotating-in / rotating-out report.

INVARIANTS & FAILURE MODES
    - First matching rule wins: ``SECTOR_RULES`` order is significant (specific
      before broad, e.g. "Banks" before "Financials").
    - Leveraged / inverse funds are bucketed separately and excluded from
      medians — their amplified returns would distort a sector's median.
    - Thin buckets (fewer than ``min_funds`` funds) are dropped: a 1-2 fund
      "sector" is idiosyncratic, not a rotation signal. Callers should still
      treat small ``n`` as directional, not statistical.
    - Sector labels are a name-keyword heuristic, not an official taxonomy.
    - Returns are simple cumulative close-to-close (inherited from
      ``compute_returns``); the 1W column is noisier than 1M/3M.
    - ``accel_1M_vs_3M`` > 0 => last month ran hotter than the quarter's
      average monthly pace (rotating in); < 0 => cooling (rotating out).

OBSERVABILITY
    Structured logging at INFO (as-of date, universe sizes, classified vs
    unclassified counts) and WARNING (missing name file). Run with ``-v`` to see
    it; the report itself always prints to stdout.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

import etf_liquidity as L
import etf_returns as R

logger = logging.getLogger("etf_sector_rotation")

# --- paths (match the daily pipeline layout) -------------------------------
HISTORICAL_PARQUET = Path("data/ohlc/historical/etf/ohlc_data.parquet")
NAME_FILE = Path("data/ticker/etf/ticker_active.xlsx")
DEFAULT_OUTPUT = Path("data/results/etf/sector_rotation.txt")

# --- classification --------------------------------------------------------
LEVERAGED_INVERSE = "Leveraged/Inverse"
UNCLASSIFIED = "Other/Unclassified"

# Ordered, FIRST MATCH WINS. Specific sectors/themes before broad regions.
SECTOR_RULES: List[Tuple[str, str]] = [
    ("Gold/Precious miners", r"gold|silver|precious"),
    ("Copper/Base-metal miners", r"copper|base metal|industrial metal"),
    ("Basic Resources/Materials", r"basic resource|materials|mining|metals|steel"),
    ("Energy", r"\benergy\b|oil|gas|uranium|nuclear"),
    ("Banks", r"\bbanks?\b"),
    ("Financials", r"financ|insurance"),
    ("Biotech/Genomics", r"biotech|genomic"),
    ("Health Care", r"health|medical|pharma"),
    ("Information Technology", r"information technology|\bit sector\b|technology|semiconduc|software"),
    ("AI/Robotics", r"\bai\b|artificial intelligence|robotic|automation"),
    ("Communication Services", r"communication|telecom|\bmedia\b"),
    ("Consumer Discretionary", r"consumer discretionary|consumer disc"),
    ("Consumer Staples", r"consumer staples|consumer stap|\bfood\b|beverage"),
    ("Industrials", r"industrial"),
    ("Utilities", r"utilit"),
    ("Real Estate", r"real estate|reit|property"),
    ("Defence/Aerospace", r"defen|aerospace|military"),
    ("Clean energy/Climate", r"clean energy|solar|renewable|climate|hydrogen"),
    ("Crypto/Blockchain", r"crypto|bitcoin|ethereum|blockchain|digital asset"),
    # Regional broad equity — rotation happens across regions too.
    ("Korea", r"korea"),
    ("Japan", r"japan"),
    ("China", r"china|\bcsi\b"),
    ("India", r"india|nifty"),
    ("Latin America", r"latin america|brazil|mexico"),
    ("Emerging Markets", r"emerging market"),
    ("Europe broad", r"stoxx europe 600|msci europe|euro stoxx 50|european"),
    ("US broad", r"\bs&p 500\b|msci usa|nasdaq|us large|russell"),
    ("World/Global broad", r"msci world|ftse all|global|acwi|developed"),
]
_LEVERAGED = re.compile(r"leverag|daily \(?-?\d?x|\(2x\)|\(-1x\)|inverse|\bshort\b", re.I)
_COMPILED: List[Tuple[str, "re.Pattern[str]"]] = [
    (label, re.compile(pat, re.I)) for label, pat in SECTOR_RULES
]

# --- windows & acceleration ------------------------------------------------
ROTATION_WINDOWS: List[str] = ["1W", "1M", "3M", "6M", "YTD"]
ACCEL_1M_VS_3M = "accel_1M_vs_3M"
ACCEL_1W_VS_1M = "accel_1W_vs_1M"
_WEEKS_PER_MONTH = 4.3  # ~30/7


def classify_sector(name: str) -> str:
    """Map a fund name to a sector label (first matching rule wins).

    Leveraged/inverse funds short-circuit to ``LEVERAGED_INVERSE`` so their
    amplified returns never enter a sector median. Unknown names return
    ``UNCLASSIFIED``.
    """
    if not isinstance(name, str) or not name.strip():
        return UNCLASSIFIED
    if _LEVERAGED.search(name):
        return LEVERAGED_INVERSE
    for label, pat in _COMPILED:
        if pat.search(name):
            return label
    return UNCLASSIFIED


def sector_table(
    returns: pd.DataFrame,
    names: Mapping[str, str],
    *,
    windows: Sequence[str] = ROTATION_WINDOWS,
    min_funds: int = 2,
    exclude: Sequence[str] = (LEVERAGED_INVERSE, UNCLASSIFIED),
) -> pd.DataFrame:
    """Aggregate per-fund returns into a per-sector median table.

    Parameters
    ----------
    returns : DataFrame indexed by symbol, one column per window.
    names   : symbol -> fund name (drives classification).
    windows : which window columns to aggregate.
    min_funds : drop sectors with fewer than this many funds (thin = noisy).
    exclude : sector labels never reported (leveraged/inverse, unclassified).

    Returns a DataFrame indexed by sector with the median of each window plus an
    integer ``n`` fund count, sorted by descending 3M (established trend).
    """
    if returns.empty:
        return pd.DataFrame(columns=list(windows) + ["n"])

    cols = [w for w in windows if w in returns.columns]
    df = returns[cols].copy()
    df["sector"] = [classify_sector(names.get(sym, "")) for sym in df.index]
    df = df[~df["sector"].isin(set(exclude))]

    grouped = df.groupby("sector")
    table = grouped[cols].median()
    table["n"] = grouped.size().astype(int)
    table = table[table["n"] >= min_funds]
    sort_col = "3M" if "3M" in cols else cols[-1]
    return table.sort_values(sort_col, ascending=False)


def add_acceleration(table: pd.DataFrame) -> pd.DataFrame:
    """Append momentum-acceleration columns (recent pace vs trailing trend).

    ``accel_1M_vs_3M`` = 1M - 3M/3   (last month vs quarter-implied monthly pace)
    ``accel_1W_vs_1M`` = 1W - 1M/4.3 (last week vs month-implied weekly pace)

    Positive => accelerating (rotating in); negative => decelerating (out).
    A column is only added when its inputs are present.
    """
    out = table.copy()
    if {"1M", "3M"}.issubset(out.columns):
        out[ACCEL_1M_VS_3M] = out["1M"] - out["3M"] / 3.0
    if {"1W", "1M"}.issubset(out.columns):
        out[ACCEL_1W_VS_1M] = out["1W"] - out["1M"] / _WEEKS_PER_MONTH
    return out


def _pct_row(row: pd.Series, cols: Sequence[str]) -> str:
    return "  ".join(f"{c} {row[c] * 100:+5.1f}%" for c in cols if c in row)


def build_rotation_report(
    table: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    universe_median: Mapping[str, float],
    n_liquid: int,
    top: int = 8,
) -> str:
    """Render the human-readable rotating-in / rotating-out report."""
    acc = add_acceleration(table)
    windows = [w for w in ROTATION_WINDOWS if w in table.columns]
    lines: List[str] = []
    lines.append("ETF sector rotation")
    lines.append(
        f"as_of {as_of.date()}   liquid universe {n_liquid} ETFs   "
        f"(returns: simple, cumulative, close-to-close)"
    )
    lines.append(
        "universe median: "
        + "  ".join(f"{w} {universe_median.get(w, float('nan')) * 100:+.1f}%" for w in windows)
    )
    lines.append(
        "sectors: name-keyword classification; leveraged/inverse excluded; "
        "thin buckets (n<min) dropped — small n is directional, not statistical."
    )
    lines.append("")

    def _block(title: str, sub: pd.DataFrame, extra: Optional[str] = None) -> None:
        lines.append(f"== {title} ==")
        for sector, row in sub.iterrows():
            tail = f"  [{extra} {row[extra] * 100:+.1f}%]" if extra and extra in row else ""
            lines.append(f"  {sector:<26} n={int(row['n']):<3} {_pct_row(row, windows)}{tail}")
        lines.append("")

    if ACCEL_1M_VS_3M in acc.columns:
        ranked = acc.sort_values(ACCEL_1M_VS_3M, ascending=False)
        _block("ROTATING IN  (1M pace above 3M trend)", ranked.head(top), ACCEL_1M_VS_3M)
        _block("ROTATING OUT (1M pace below 3M trend)", ranked.tail(top).iloc[::-1], ACCEL_1M_VS_3M)

    if "3M" in table.columns:
        by3m = table.sort_values("3M", ascending=False)
        _block("3M/6M LEADERS (established trend)", by3m.head(top))
        _block("3M/6M LAGGARDS", by3m.tail(top).iloc[::-1])

    return "\n".join(lines).rstrip() + "\n"


def load_names(path: Path = NAME_FILE) -> Dict[str, str]:
    """Load symbol -> fund name from the active ticker seed (name, ticker)."""
    if not path.exists():
        logger.warning("name file %s not found; sectors will be Unclassified", path)
        return {}
    df = pd.read_excel(path)
    return dict(zip(df["ticker"].astype(str), df["name"].astype(str)))


def compute_sector_rotation(
    ohlc: pd.DataFrame,
    names: Mapping[str, str],
    *,
    as_of: Optional[pd.Timestamp] = None,
    min_funds: int = 2,
    max_daily_move: Optional[float] = R.DEFAULT_MAX_DAILY_MOVE,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.Timestamp, int]:
    """End-to-end: liquid + artifact screens -> per-window returns -> sectors.

    Returns ``(table, universe_median, as_of, n_liquid)`` where ``table`` is the
    per-sector median table (before acceleration is appended).
    """
    as_of = as_of if as_of is not None else ohlc["date"].max()
    stats = L.liquidity_stats(ohlc, as_of=as_of, lookback_days=L.DEFAULT_LOOKBACK_DAYS)
    liquid = set(L.liquid_symbols(stats))
    returns = R.compute_returns(ohlc, as_of=as_of, method="simple", max_daily_move=max_daily_move)
    returns = returns.loc[returns.index.isin(liquid)]
    logger.info("as_of=%s liquid=%d of %d", as_of.date(), len(returns), len(stats))

    windows = [w for w in ROTATION_WINDOWS if w in returns.columns]
    universe_median = {w: float(returns[w].median()) for w in windows}
    table = sector_table(returns, names, windows=windows, min_funds=min_funds)

    classified = sum(classify_sector(names.get(s, "")) not in (LEVERAGED_INVERSE, UNCLASSIFIED)
                     for s in returns.index)
    logger.info("classified %d/%d funds into %d sectors (min_funds=%d)",
                classified, len(returns), len(table), min_funds)
    return table, universe_median, as_of if isinstance(as_of, pd.Timestamp) else pd.Timestamp(as_of), len(returns)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Detect ETF sector rotation.")
    parser.add_argument("--parquet", type=Path, default=HISTORICAL_PARQUET)
    parser.add_argument("--names", type=Path, default=NAME_FILE)
    parser.add_argument("--min-funds", type=int, default=2,
                        help="drop sectors with fewer funds than this (default 2)")
    parser.add_argument("--top", type=int, default=8, help="rows per block (default 8)")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"also write the report here (e.g. {DEFAULT_OUTPUT})")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ohlc = pd.read_parquet(args.parquet)
    names = load_names(args.names)
    table, universe_median, as_of, n_liquid = compute_sector_rotation(
        ohlc, names, min_funds=args.min_funds
    )
    report = build_rotation_report(
        table, as_of=as_of, universe_median=universe_median, n_liquid=n_liquid, top=args.top
    )
    print(report, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
