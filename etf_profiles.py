"""
etf_profiles.py — AUM-ranked ETF selection from the justETF profiles export.

`data/ticker/etf/profiles.jsonl` is a justETF export with one JSON object per
fund: `isin`, `name`, `category`, `fund_size_eur_mln` (assets under management),
`ter_pct`, `distribution`, `replication`, `ytd_pct`, `index`. It has **no Yahoo
ticker**, so funds are mapped to tickers by exact `name` against the existing
`data/ticker/etf/ticker.xlsx` (name -> Yahoo ticker) before use.

This lets the active universe be enriched with the **largest funds by real AUM**
rather than by justETF list order (`etf_universe`).

Invariants (enforced by tests/test_etf_profiles.py)
--------------------------------------------------
- output is a (name, ticker) table ordered by descending fund size
- funds whose name has no ticker mapping are dropped (can't be downloaded)
- duplicate tickers collapse to the larger-AUM row
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROFILES_FILE = Path("data/ticker/etf/profiles.jsonl")
_AUM_COLUMN = "fund_size_eur_mln"


def load_profiles(path: Path = PROFILES_FILE) -> pd.DataFrame:
    """Read the justETF profiles JSON Lines export into a DataFrame."""
    return pd.read_json(Path(path), lines=True)


def profiles_ticker_table(
    profiles: pd.DataFrame,
    name_to_ticker: Mapping[str, str],
    n: int = 100,
) -> pd.DataFrame:
    """Return the top-``n`` funds by AUM as a (name, ticker) table.

    Maps each fund's ``name`` to its Yahoo ticker via ``name_to_ticker``, drops
    funds with no mapping (they can't be downloaded), orders by descending
    ``fund_size_eur_mln``, collapses duplicate tickers to the larger-AUM row,
    and keeps the first ``n``.

    Pure function (the name -> ticker mapping is injected) so the ranking logic
    is unit-tested without touching the filesystem.
    """
    table = profiles.copy()
    table["ticker"] = table["name"].map(dict(name_to_ticker))
    table = table.dropna(subset=["ticker"])

    table = table.sort_values(_AUM_COLUMN, ascending=False, kind="stable")
    table = table.drop_duplicates(subset=["ticker"], keep="first")

    table = table.head(n)
    return table[["name", "ticker"]].reset_index(drop=True)
