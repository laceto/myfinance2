"""
generate_etf_ticker_list.py
============================
Build the ETF ticker list for myfinance2 by reading the ISIN→YF mapping
produced by the justETF repo (generate_isin_to_yf_map.py) and applying
quality / cost filters.

Outputs data/ticker/etf/ticker.xlsx with one column: ticker

Usage
-----
    python generate_etf_ticker_list.py
    python generate_etf_ticker_list.py --isin-map /path/to/isin_to_yf.json
    python generate_etf_ticker_list.py --min-fund-size 200 --max-ter 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)

BENCHMARK        = "IWDA.AS"
DEFAULT_ISIN_MAP = Path("../justETF/data/ticker/isin_to_yf.json")
DEFAULT_OUT      = Path("data/ticker/etf/ticker.xlsx")
DEFAULT_MIN_FUND = 500.0   # EUR million
DEFAULT_MAX_TER  = 0.30    # percent


def load_mapping(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data.values())


def apply_filters(
    entries: list[dict],
    min_fund_size: float,
    max_ter: float,
) -> list[dict]:
    kept = []
    skipped_size = skipped_ter = 0
    for e in entries:
        fund_size = e.get("fund_size_eur_mln")
        if fund_size is not None and fund_size < min_fund_size:
            skipped_size += 1
            continue
        ter = e.get("ter_pct")
        if ter is not None and ter > max_ter:
            skipped_ter += 1
            continue
        kept.append(e)
    LOG.info(
        "Filters: kept %d  |  dropped by fund_size < %.0f: %d  |  dropped by ter > %.2f%%: %d",
        len(kept), min_fund_size, skipped_size, max_ter, skipped_ter,
    )
    return kept


def build_ticker_list(entries: list[dict], benchmark: str) -> list[str]:
    tickers = []
    seen: set[str] = set()
    for e in entries:
        t = (e.get("yf_ticker") or "").strip()
        if t and t not in seen and t != benchmark:
            tickers.append(t)
            seen.add(t)
    if benchmark not in seen:
        tickers.append(benchmark)
    return tickers


def save_xlsx(tickers: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_excel(path, index=False)
    LOG.info("Saved %d tickers → %s  (includes benchmark)", len(tickers), path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build ETF ticker list for myfinance2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--isin-map",      type=Path, default=DEFAULT_ISIN_MAP)
    p.add_argument("--out",           type=Path, default=DEFAULT_OUT)
    p.add_argument("--benchmark",     default=BENCHMARK)
    p.add_argument("--min-fund-size", type=float, default=DEFAULT_MIN_FUND,
                   help="Minimum fund size in EUR million (None-values are kept)")
    p.add_argument("--max-ter",       type=float, default=DEFAULT_MAX_TER,
                   help="Maximum TER percent (None-values are kept)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    if not args.isin_map.exists():
        LOG.error("ISIN map not found: %s", args.isin_map)
        raise SystemExit(1)

    entries = load_mapping(args.isin_map)
    LOG.info("Loaded %d entries from %s", len(entries), args.isin_map)

    filtered = apply_filters(entries, args.min_fund_size, args.max_ter)
    tickers  = build_ticker_list(filtered, args.benchmark)

    LOG.info(
        "Final ticker list: %d symbols (including benchmark %s)",
        len(tickers), args.benchmark,
    )
    save_xlsx(tickers, args.out)


if __name__ == "__main__":
    main()
