"""
etf_universe.py — The active ETF universe (curated core + justETF extras).

Single source of truth for the tickers the ETF pipeline actually trades. The
active universe is the curated top-15 UCITS ETFs (``etf_top15``) followed by up
to ``DEFAULT_EXTRA_COUNT`` additional tickers pulled from the justETF list
(``data/ticker/etf/ticker.xlsx``). Both the cold-start backfill
(``cold_start_etf.py``) and the daily download (``get_daily_ohlc_data_etf.py``)
read the seed produced here, so the two pipelines stay on the same universe.

Why justETF for the extras
--------------------------
Assets-under-management figures are not stored in the repo, so the extras cannot
be AUM-ranked reliably; the justETF list is the real, in-repo universe of
downloadable Yahoo symbols. The curated top-15 remain the reviewed core (first
in order); the extras broaden coverage for a larger-scale run. Swap the source
or ordering here without touching the pipeline.

Invariants (enforced by tests/test_etf_universe.py)
--------------------------------------------------
- the curated core appears first, in its original order
- extras are the first N justETF tickers NOT already in the core
- tickers are unique; the (name, ticker) schema is preserved
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from etf_top15 import build_top15_ticker_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# The active seed the pipeline reads. Renamed from ticker_top15.xlsx: the file
# now holds the full active universe (core + extras), not just the top 15.
ACTIVE_SEED_FILE = Path("data/ticker/etf/ticker_active.xlsx")
JUSTETF_FILE = Path("data/ticker/etf/ticker.xlsx")
DEFAULT_EXTRA_COUNT = 350

# Yahoo symbols that returned zero rows in the 2026-07-12 cold-start runs
# (delisted, renamed, or no Yahoo history for that listing). Pruned from the
# active universe so the daily job does not waste calls on them. Pruned, NOT
# replaced: the universe simply shrinks by these. Remove an entry if a symbol
# becomes downloadable again.
KNOWN_DEAD = frozenset(
    {
        # found in the 115-ticker run (extra_count=100)
        "10AI.PA", "AEMD.PA", "AEMU.PA", "AHYQ.PA", "AMINA.AS", "POLY.AS", "STXH.SW", "XDCN.AS",
        # found in the 257-ticker run (extra_count=250)
        "10AJ.PA", "ACWU.SW", "AHYC.PA", "AHYI.PA", "CUIK.PA", "DECR.PA",
        "ESTE.PA", "GCSG.PA", "IBX35.SW", "PRAM.PA", "UCRH.PA",
    }
)

_REQUIRED_COLUMNS = ["name", "ticker"]


def build_active_universe_table(
    justetf: pd.DataFrame,
    core: pd.DataFrame,
    extra_count: int = DEFAULT_EXTRA_COUNT,
    exclude=KNOWN_DEAD,
) -> pd.DataFrame:
    """Return the active universe: the core, then up to ``extra_count`` extras.

    Extras are the first justETF tickers not already present in the core, in
    justETF order. The core keeps its own order and comes first. Duplicate
    tickers are dropped (core wins). Any ticker in ``exclude`` is pruned from
    the result and is **not** replaced — the universe shrinks by the excluded
    count (so pruning known-dead symbols from the first ``extra_count`` window
    yields fewer than ``extra_count`` extras, on purpose).

    Args:
        justetf: DataFrame with at least ``name`` and ``ticker`` columns.
        core: the curated core table (``name``, ``ticker``), kept first.
        extra_count: maximum number of extra tickers to consider from justETF.
        exclude: tickers to prune (default: ``KNOWN_DEAD``).

    Raises:
        ValueError: if ``justetf`` lacks the required columns.
    """
    missing = [col for col in _REQUIRED_COLUMNS if col not in justetf.columns]
    if missing:
        raise ValueError(
            f"justetf is missing required column(s) {missing}; "
            f"expected {_REQUIRED_COLUMNS}, got {list(justetf.columns)}"
        )

    core_tickers = set(core["ticker"])
    extras = (
        justetf[~justetf["ticker"].isin(core_tickers)][_REQUIRED_COLUMNS]
        .head(extra_count)
    )

    combined = pd.concat([core[_REQUIRED_COLUMNS], extras], ignore_index=True)
    combined = combined[~combined["ticker"].isin(set(exclude))]
    combined = combined.drop_duplicates(subset=["ticker"], keep="first")
    return combined.reset_index(drop=True)


def write_active_seed(
    path: Path = ACTIVE_SEED_FILE,
    justetf_file: Path = JUSTETF_FILE,
    extra_count: int = DEFAULT_EXTRA_COUNT,
) -> Path:
    """Materialise the active universe seed (.xlsx) that the pipeline reads.

    ``path`` is first so this matches the ``seed_writer(path)`` contract used by
    ``cold_start_etf.cold_start``.
    """
    path = Path(path)
    justetf = pd.read_excel(justetf_file)
    core = build_top15_ticker_table()

    table = build_active_universe_table(justetf, core, extra_count)

    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(path, index=False)

    logger.info(
        "Wrote active ETF universe seed: %s (%d tickers = %d core + %d extras)",
        path, len(table), len(core), len(table) - len(core),
    )
    return path


def main() -> None:
    """Regenerate the active universe seed from the curated core + justETF."""
    write_active_seed()


if __name__ == "__main__":
    main()
