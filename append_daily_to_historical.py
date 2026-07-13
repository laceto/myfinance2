"""
append_daily_to_historical.py — Append a market's daily OHLC bar to history.

Single source of truth for the CI step that folds a freshly-downloaded daily
Parquet (``data/ohlc/today/<market>/ohlc_data.parquet``) into the committed
historical Parquet (``data/ohlc/historical/<market>/ohlc_data.parquet``).

Why this module exists
----------------------
The append logic previously lived as an inline heredoc inside
``download_daily_ohlc.yml`` for the ``it`` market. Adding the ``etf`` market
would have meant copy-pasting that logic. Extracting it here keeps one
market-agnostic implementation (DRY), makes the behaviour unit-testable
(YAML cannot be), and gives each market a self-documenting CLI invocation.

Contract / invariants
---------------------
- Append is a plain ``concat(historical, today)`` — **no deduplication**.
  This matches the pre-existing inline behaviour; the once-daily cron plus
  ``[skip ci]`` on the data commit guard against accidental double-appends.
- The long/tidy OHLC schema is preserved (both frames share it).
- If the historical file is absent, it is created from today's frame.

Usage
-----
    python append_daily_to_historical.py --market it
    python append_daily_to_historical.py --market etf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

_OHLC_ROOT = Path("data/ohlc")
_PARQUET_NAME = "ohlc_data.parquet"


def market_paths(market: str) -> Tuple[Path, Path]:
    """Resolve the (today, historical) Parquet paths for a market code.

    Fail fast on a blank market code so a misconfigured CI step surfaces an
    actionable error instead of silently writing to ``data/ohlc/today//``.
    """
    if not market or not market.strip():
        raise ValueError("market code must be a non-empty string, e.g. 'it' or 'etf'")

    code = market.strip()
    today_path = _OHLC_ROOT / "today" / code / _PARQUET_NAME
    historical_path = _OHLC_ROOT / "historical" / code / _PARQUET_NAME
    return today_path, historical_path


def append_daily_frames(
    today_df: pd.DataFrame,
    historical_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Return the combined history: historical rows first, then today's rows.

    Pure function (no I/O) so the append semantics can be tested against
    hand-built frames. When ``historical_df`` is ``None`` the caller has no
    prior history yet and today's frame becomes the seed.
    """
    if historical_df is None:
        return today_df

    return pd.concat([historical_df, today_df], ignore_index=True)


def append_market_file(today_path: Path, historical_path: Path) -> pd.DataFrame:
    """Read today's Parquet, append it to historical on disk, return the result.

    Fails fast if today's Parquet is missing — that means the upstream
    download step did not produce output and we must not silently commit a
    stale or empty history.
    """
    if not today_path.exists():
        raise FileNotFoundError(
            f"today's OHLC Parquet not found: {today_path} "
            "(did the download step run and succeed?)"
        )

    today_df = pd.read_parquet(today_path)

    historical_df: Optional[pd.DataFrame] = None
    if historical_path.exists():
        historical_df = pd.read_parquet(historical_path)
    else:
        historical_path.parent.mkdir(parents=True, exist_ok=True)

    combined = append_daily_frames(today_df, historical_df)
    combined.to_parquet(historical_path, index=False)

    log.info(
        "Appended %d today rows to %s -> %d total rows",
        len(today_df),
        historical_path,
        len(combined),
    )
    return combined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a market's daily OHLC bar to its historical Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--market",
        required=True,
        help="Market code selecting data/ohlc/{today,historical}/<market>/ "
             "(e.g. 'it' for Borsa Italiana, 'etf' for the ETF universe).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    today_path, historical_path = market_paths(args.market)
    log.info("Appending market '%s': %s -> %s", args.market, today_path, historical_path)
    append_market_file(today_path, historical_path)


if __name__ == "__main__":
    main()
