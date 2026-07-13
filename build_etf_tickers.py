"""
build_etf_tickers.py — Builds data/ticker/etf/ticker.xlsx from the justETF isin_to_yf.csv export.

Mirrors the shape of data/ticker/it/ticker.xlsx (columns: name, ticker) so the ETF market
can reuse the same downstream OHLC scripts (get_daily_ohlc_data_etf.py,
get_historical_ohlc_data_etf.py) as the "it" market does with algoshort.YFinanceDataHandler.

Source columns consumed: name, yf_ticker
    isin, exch_code, currency, yf_suffix, ter_pct, fund_size_eur_mln, index,
    distribution, replication are intentionally dropped — not needed by the OHLC pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUIRED_SOURCE_COLUMNS = ["name", "yf_ticker"]


def build_etf_ticker_table(source: pd.DataFrame) -> pd.DataFrame:
    """
    Map a justETF isin_to_yf export to the (name, ticker) schema used by data/ticker/*/ticker.xlsx.

    Drops rows with a missing/empty name or yf_ticker, and drops duplicate tickers
    (keeping the first occurrence) so downstream Yahoo Finance downloads never request
    the same symbol twice.

    Args:
        source: DataFrame with at least `name` and `yf_ticker` columns (e.g. loaded from
            justETF's isin_to_yf.csv).

    Returns:
        DataFrame with exactly two columns, `name` and `ticker`, row order preserved.

    Raises:
        ValueError: if `source` is missing the `name` or `yf_ticker` column.
    """
    missing = [col for col in REQUIRED_SOURCE_COLUMNS if col not in source.columns]
    if missing:
        raise ValueError(
            f"source DataFrame is missing required column(s) {missing}; "
            f"expected at least {REQUIRED_SOURCE_COLUMNS}, got {list(source.columns)}"
        )

    table = source[REQUIRED_SOURCE_COLUMNS].rename(columns={"yf_ticker": "ticker"})

    table["name"] = table["name"].replace("", pd.NA)
    table["ticker"] = table["ticker"].replace("", pd.NA)
    table = table.dropna(subset=["name", "ticker"])

    table = table.drop_duplicates(subset=["ticker"], keep="first")

    return table.reset_index(drop=True)


def main(
    source_csv: str = r"C:\Users\l_ace\Desktop\projects\justETF\data\ticker\isin_to_yf.csv",
    output_xlsx: str = "data/ticker/etf/ticker.xlsx",
) -> None:
    source_path = Path(source_csv)
    if not source_path.exists():
        raise FileNotFoundError(f"justETF source CSV not found: {source_path}")

    logger.info("Reading justETF source CSV: %s", source_path)
    source = pd.read_csv(source_path)

    table = build_etf_ticker_table(source)
    dropped = len(source) - len(table)
    logger.info(
        "Built ETF ticker table: %d rows kept, %d rows dropped (missing/duplicate)",
        len(table),
        dropped,
    )

    output_path = Path(output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(table))


if __name__ == "__main__":
    main()
