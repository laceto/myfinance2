"""
etf_top15.py — Curated seed of the 15 largest European UCITS ETFs by AUM.

This is the **single source of truth** for the ETF market's cold-start
universe. Both the historical backfill (`cold_start_etf.py`) and the daily
append (`get_daily_ohlc_data_etf.py`) read the seed produced here, so the two
pipelines always operate on the same 15 tickers.

Why a hand-curated list
-----------------------
Assets-under-management figures drift daily and are not stored in this repo,
and ranking the full 2217-ticker justETF universe live would mean thousands of
unreliable `yfinance.info` calls. A curated, version-controlled list is
deterministic, reviewable, and trivially editable. The figures below are
**approximate** and stamped with `TOP15_AS_OF` — refresh them periodically.

Each entry is a distinct fund (no duplicate listings of the same ETF) with a
Yahoo Finance symbol (`.L` London, `.DE` Xetra, `.AS` Amsterdam, `.PA` Paris).

Invariants (enforced by tests/test_etf_top15.py)
-----------------------------------------------
- exactly 15 records, ordered by descending AUM (largest first)
- tickers are unique and non-blank
- the generated table uses the (name, ticker) schema shared by
  data/ticker/*/ticker.xlsx
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Provenance — see module docstring. Figures are approximate.
TOP15_AS_OF = "2026-01"
TOP15_SOURCE = (
    "Curated from public UCITS ETF AUM rankings (justETF / issuer factsheets); "
    "aum_usd_bn values are approximate and should be refreshed periodically."
)

# The seed path shared by the cold-start and daily ETF scripts. Kept separate
# from the full 2217-ticker data/ticker/etf/ticker.xlsx so the active pipeline
# uses the curated 15 without discarding the wider universe.
TICKER_FILE = Path("data/ticker/etf/ticker_top15.xlsx")

# Ordered by descending approximate AUM (USD bn). Each is a distinct fund.
TOP15_ETFS: List[Dict[str, Union[str, float]]] = [
    {"name": "iShares Core S&P 500 UCITS ETF (Acc)",          "ticker": "CSPX.L",  "provider": "iShares",   "aum_usd_bn": 110.0},
    {"name": "iShares Core MSCI World UCITS ETF (Acc)",        "ticker": "SWDA.L",  "provider": "iShares",   "aum_usd_bn": 95.0},
    {"name": "Vanguard S&P 500 UCITS ETF (Dist)",             "ticker": "VUSA.L",  "provider": "Vanguard",  "aum_usd_bn": 48.0},
    {"name": "Vanguard FTSE All-World UCITS ETF (Acc)",       "ticker": "VWCE.DE", "provider": "Vanguard",  "aum_usd_bn": 28.0},
    {"name": "iShares Core MSCI EM IMI UCITS ETF (Acc)",      "ticker": "EIMI.L",  "provider": "iShares",   "aum_usd_bn": 24.0},
    {"name": "Invesco EQQQ Nasdaq-100 UCITS ETF",            "ticker": "EQQQ.L",  "provider": "Invesco",   "aum_usd_bn": 14.0},
    {"name": "iShares Core FTSE 100 UCITS ETF (Dist)",        "ticker": "ISF.L",   "provider": "iShares",   "aum_usd_bn": 13.5},
    {"name": "Xtrackers MSCI World UCITS ETF (Acc)",          "ticker": "XDWD.DE", "provider": "Xtrackers", "aum_usd_bn": 13.0},
    {"name": "Amundi Stoxx Europe 600 UCITS ETF (Acc)",       "ticker": "MEUD.PA", "provider": "Amundi",    "aum_usd_bn": 12.0},
    {"name": "SPDR MSCI World UCITS ETF",                    "ticker": "SWRD.L",  "provider": "SPDR",      "aum_usd_bn": 11.0},
    {"name": "iShares Core MSCI Europe UCITS ETF (Acc)",      "ticker": "IMEU.L",  "provider": "iShares",   "aum_usd_bn": 10.0},
    {"name": "iShares Core Global Aggregate Bond UCITS ETF",  "ticker": "AGGG.L",  "provider": "iShares",   "aum_usd_bn": 9.0},
    {"name": "Vanguard FTSE All-World High Div Yield UCITS",  "ticker": "VHYL.L",  "provider": "Vanguard",  "aum_usd_bn": 6.0},
    {"name": "Vanguard FTSE Emerging Markets UCITS ETF",     "ticker": "VFEM.L",  "provider": "Vanguard",  "aum_usd_bn": 5.5},
    {"name": "Vanguard FTSE Developed World UCITS ETF (Acc)", "ticker": "VEVE.L",  "provider": "Vanguard",  "aum_usd_bn": 5.0},
]


def build_top15_ticker_table() -> pd.DataFrame:
    """Return the curated seed as a (name, ticker) table, largest AUM first.

    Pure function: derives the download-ready ticker table from TOP15_ETFS
    without touching the filesystem, so its shape/integrity can be unit-tested.
    """
    table = pd.DataFrame(TOP15_ETFS)[["name", "ticker"]].copy()
    return table.reset_index(drop=True)


def write_top15_ticker_file(path: Path = TICKER_FILE) -> Path:
    """Materialise the seed as an .xlsx matching data/ticker/*/ticker.xlsx.

    Creates parent directories as needed. Returns the written path so callers
    (and the CLI) can log it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = build_top15_ticker_table()
    table.to_excel(path, index=False)

    logger.info("Wrote ETF top-15 seed: %s (%d rows, as of %s)", path, len(table), TOP15_AS_OF)
    return path


def load_tickers(path: Path = TICKER_FILE) -> List[str]:
    """Load the ticker symbols from a seed .xlsx, preserving row order.

    Fails fast if the seed is missing — the cold-start/daily scripts must not
    proceed to call Yahoo Finance with an empty universe.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ETF ticker seed not found: {path}. "
            "Run `python cold_start_etf.py` (or generate it via etf_top15.write_top15_ticker_file)."
        )

    return pd.read_excel(path)["ticker"].tolist()


def main() -> None:
    """Regenerate the seed file from the curated list."""
    write_top15_ticker_file()


if __name__ == "__main__":
    main()
