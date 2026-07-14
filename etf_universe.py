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
from typing import Optional

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
DEFAULT_EXTRA_COUNT = 550

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
        # found in the 346-ticker run (extra_count=350) — mostly crypto ETPs
        # and thin .PA/.SW listings with no Yahoo history
        "0W76.L", "500D.PA", "AAVE.AS", "ACD27.PA", "ACD29.PA", "ACD32.PA",
        "ACESD.SW", "AGESD.SW", "AHYK.PA", "ALGO.AS", "ALINK.AS", "ALTS.AS",
        "AVAX.AS", "AWDS.PA", "AXLM.AS", "AXTZ.AS", "BNBA.AS", "BOLD.AS",
        "ETHC.AS", "GRAM.AS", "HODL.AS", "PRAE.PA", "PRAZ.PA", "QGHY.PA",
        "QGLO.PA", "QUIG.PA", "QUSA.PA", "SRIC7.SW", "SRID7.SW",
        # found in the 395-ticker run (AUM enrichment from profiles.jsonl)
        "BNKS.SW", "D28A.AS", "EEJD.AS", "EEWD.AS", "IEXXF",
        # found in the 488-ticker run (extra_count=450)
        "ALLC.PA", "BGUS.PA", "BRIXU.PA", "BTCE.AS", "BTCG.AS", "CASHD.PA",
        "DA20.PA", "EDEC.PA", "EEE.AS", "EMEXC.PA", "EMGXC.PA", "GLOBD.PA",
        "GLOBU.PA", "SPTR.PA", "WETF.PA", "WEWE.PA", "ZETH.AS",
        # found in the 571-ticker run (extra_count=550)
        "0MPZ.L", "0MQ3.L", "0W81.L", "BEWGF", "BMAC.PA", "CASL.SW",
        "CFMOM.SW", "ESOLGBP.SW", "FEIG.MI", "FUIE.MI", "FUIG.MI", "XBTI.AS",
    }
)

_REQUIRED_COLUMNS = ["name", "ticker"]


def build_active_universe_table(
    justetf: pd.DataFrame,
    core: pd.DataFrame,
    extra_count: int = DEFAULT_EXTRA_COUNT,
    exclude=KNOWN_DEAD,
    aum_extras: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return the active universe: core, justETF extras, then AUM extras.

    Order: the curated ``core`` first (own order), then the first
    ``extra_count`` justETF tickers not already in the core (justETF order),
    then ``aum_extras`` (AUM-ranked funds — see ``etf_profiles``). Duplicate
    tickers are dropped keeping the first occurrence, so a fund already present
    from an earlier source is **not** re-added ("if not already there"). Any
    ticker in ``exclude`` is pruned and not replaced.

    Args:
        justetf: DataFrame with at least ``name`` and ``ticker`` columns.
        core: the curated core table (``name``, ``ticker``), kept first.
        extra_count: maximum number of justETF extra tickers to consider.
        exclude: tickers to prune (default: ``KNOWN_DEAD``).
        aum_extras: optional (``name``, ``ticker``) table of AUM-ranked funds to
            union in after the justETF extras.

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

    parts = [core[_REQUIRED_COLUMNS], extras]
    if aum_extras is not None and len(aum_extras) > 0:
        parts.append(aum_extras[_REQUIRED_COLUMNS])

    combined = pd.concat(parts, ignore_index=True)
    combined = combined[~combined["ticker"].isin(set(exclude))]
    combined = combined.drop_duplicates(subset=["ticker"], keep="first")
    return combined.reset_index(drop=True)


PROFILES_TOP_N = 100


def _aum_extras(justetf: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Top-``PROFILES_TOP_N`` funds by real AUM from profiles.jsonl, mapped to
    Yahoo tickers via the justETF name -> ticker table. Returns None when the
    profiles export is absent (keeps the seed reproducible without it)."""
    from etf_profiles import PROFILES_FILE, load_profiles, profiles_ticker_table

    if not PROFILES_FILE.exists():
        logger.info("No profiles export at %s; skipping AUM enrichment", PROFILES_FILE)
        return None

    name_to_ticker = dict(zip(justetf["name"], justetf["ticker"]))
    return profiles_ticker_table(load_profiles(), name_to_ticker, n=PROFILES_TOP_N)


def write_active_seed(
    path: Path = ACTIVE_SEED_FILE,
    justetf_file: Path = JUSTETF_FILE,
    extra_count: int = DEFAULT_EXTRA_COUNT,
) -> Path:
    """Materialise the active universe seed (.xlsx) that the pipeline reads.

    The universe = curated core + justETF list-order extras + the top-N funds by
    real AUM from ``profiles.jsonl`` (deduped, ``KNOWN_DEAD`` pruned).

    ``path`` is first so this matches the ``seed_writer(path)`` contract used by
    ``cold_start_etf.cold_start``.
    """
    path = Path(path)
    justetf = pd.read_excel(justetf_file)
    core = build_top15_ticker_table()
    aum_extras = _aum_extras(justetf)

    table = build_active_universe_table(justetf, core, extra_count, aum_extras=aum_extras)

    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(path, index=False)

    n_aum = 0 if aum_extras is None else len(aum_extras)
    logger.info(
        "Wrote active ETF universe seed: %s (%d tickers = %d core + justETF extras "
        "+ up to %d AUM-ranked)",
        path, len(table), len(core), n_aum,
    )
    return path


def main() -> None:
    """Regenerate the active universe seed from the curated core + justETF."""
    write_active_seed()


if __name__ == "__main__":
    main()
