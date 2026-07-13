"""
cold_start_etf.py — One-time historical backfill for the ETF market.

Seeds ``data/ohlc/historical/etf/ohlc_data.parquet`` with full daily history
(default from 2016-01-01) for the curated top-15 UCITS ETFs defined in
``etf_top15.py``. After the cold start, ``download_daily_ohlc.yml`` appends each
new trading day's bar via ``append_daily_to_historical.py``.

Design
------
The Yahoo download client (``algoshort.YFinanceDataHandler``) is
**dependency-injected** through ``handler_factory`` and imported lazily, so this
module imports (and its orchestration tests run) without algoshort installed and
without any network access. The live client is only constructed when run for
real (CLI / CI).

Contract / invariants
---------------------
- Backfill is idempotent: it overwrites the historical Parquet with a fresh
  full pull, so re-running cannot create duplicate rows.
- Symbols that return zero rows are retried **once**; any still empty are
  reported in the summary and logged at WARNING — never silently dropped.
- An empty ticker universe is a fail-fast ValueError (never call Yahoo blank).

Usage
-----
    python cold_start_etf.py                       # 2016-01-01 -> today, top-15 seed
    python cold_start_etf.py --start 2010-01-01
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, List, Optional

from etf_top15 import load_tickers
from etf_universe import ACTIVE_SEED_FILE, write_active_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_START = "2016-01-01"
HISTORICAL_OUTPUT = Path("data/ohlc/historical/etf/ohlc_data.parquet")


@dataclass
class ColdStartSummary:
    """Outcome of a backfill run, for logging and assertions."""

    requested: int
    output_path: Path
    missing_after_retry: List[str] = field(default_factory=list)


def _default_handler_factory(cache_dir: Path):
    """Construct the real algoshort client. Imported lazily so tests/imports
    do not require algoshort or network."""
    from algoshort.yfinance_handler import YFinanceDataHandler

    return YFinanceDataHandler(
        cache_dir=str(cache_dir),
        enable_logging=True,
        chunk_size=20,               # small chunks for stability, as in the daily script
        log_level=logging.INFO,
    )


def _zero_row_symbols(handler) -> List[str]:
    """Symbols the handler downloaded but for which it has zero rows."""
    summary = handler.list_available_data()
    return [symbol for symbol, info in summary.items() if info["rows"] == 0]


def _download_and_save(handler, symbols: List[str], output_path: Path,
                       start: str, end: str, interval: str) -> None:
    handler.download_data(
        symbols=symbols,
        start=start,
        end=end,
        interval=interval,
        use_cache=False,
        threads=True,
    )
    handler.save_data(
        filepath=str(output_path),
        format="parquet",
        multi_symbol_strategy="single_file",
        combine_column=["open", "high", "low", "close", "volume"],
    )


def backfill_historical(
    handler,
    ticker_list: List[str],
    output_path: Path,
    start: str,
    end: str,
    interval: str = "1d",
) -> ColdStartSummary:
    """Download full history for ``ticker_list`` and save it to ``output_path``.

    Retries any zero-row symbols once. Returns a ColdStartSummary; symbols still
    empty after the retry are logged at WARNING and listed in the summary.

    Raises:
        ValueError: if ``ticker_list`` is empty.
    """
    if not ticker_list:
        raise ValueError("ticker_list is empty; refusing to start a backfill with no symbols")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Backfilling %d tickers %s -> %s into %s",
             len(ticker_list), start, end, output_path)
    _download_and_save(handler, ticker_list, output_path, start, end, interval)

    missing = _zero_row_symbols(handler)
    if missing:
        log.warning("Retrying %d zero-row symbol(s): %s", len(missing), missing)
        _download_and_save(handler, missing, output_path, start, end, interval)
        missing = _zero_row_symbols(handler)

    if missing:
        log.warning("%d symbol(s) still empty after retry: %s", len(missing), missing)
    else:
        log.info("Backfill complete: all %d symbols returned data", len(ticker_list))

    return ColdStartSummary(
        requested=len(ticker_list),
        output_path=output_path,
        missing_after_retry=missing,
    )


def cold_start(
    ticker_file: Path = ACTIVE_SEED_FILE,
    output_path: Path = HISTORICAL_OUTPUT,
    start: str = DEFAULT_START,
    end: Optional[str] = None,
    seed_writer: Callable[[Path], object] = write_active_seed,
    handler_factory: Callable[[Path], object] = _default_handler_factory,
) -> ColdStartSummary:
    """(Re)write the active universe seed, then backfill its full history.

    Args:
        ticker_file: seed .xlsx to (re)write and read (default: the active seed).
        output_path: destination historical Parquet.
        start: inclusive start date (YYYY-MM-DD).
        end: inclusive end date; defaults to today when None.
        seed_writer: writes the seed for ``ticker_file`` (injectable for testing;
            default builds the curated core + justETF extras).
        handler_factory: builds the download client given a cache dir (injectable
            for testing).
    """
    # Coerce blank inputs to sensible defaults. A push-triggered workflow run
    # supplies an empty --start (workflow_dispatch inputs are absent on push);
    # a blank start must not reach yfinance, which would silently fall back to
    # period=1y and backfill only the last year instead of the full history.
    start = start or DEFAULT_START
    end = end or date.today().isoformat()

    seed_writer(ticker_file)
    ticker_list = load_tickers(ticker_file)

    handler = handler_factory(output_path.parent)
    summary = backfill_historical(handler, ticker_list, output_path, start, end)

    log.info("Cold start done: %d requested, %d missing, output %s",
             summary.requested, len(summary.missing_after_retry), summary.output_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cold-start the ETF market: backfill historical OHLC for the active universe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Inclusive end date (YYYY-MM-DD); defaults to today.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cold_start(start=args.start, end=args.end)


if __name__ == "__main__":
    main()
