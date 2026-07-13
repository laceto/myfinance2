"""
test_etf_top15.py — Unit tests for etf_top15 (curated top-15 UCITS ETF seed).

etf_top15 is the single source of truth for the ETF market's cold-start
universe: the 15 largest European UCITS ETFs by AUM. Both cold_start_etf.py
(historical backfill) and get_daily_ohlc_data_etf.py (daily append) consume it,
so the seed's shape and integrity are load-bearing.

Coverage
--------
build_top15_ticker_table (pure):
  - returns the (name, ticker) schema used by data/ticker/*/ticker.xlsx
  - contains exactly 15 rows
  - tickers are unique (no symbol downloaded twice)
  - no missing/blank name or ticker
  - ordered by AUM rank (largest first)
provenance:
  - TOP15_AS_OF and TOP15_SOURCE are populated (auditability of the ranking)
round-trip:
  - write_top15_ticker_file + load_tickers returns the 15 tickers in order
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from etf_top15 import (
    TOP15_AS_OF,
    TOP15_ETFS,
    TOP15_SOURCE,
    build_top15_ticker_table,
    load_tickers,
    write_top15_ticker_file,
)


class TestBuildTop15TickerTable:

    def test_returns_name_ticker_schema(self):
        table = build_top15_ticker_table()

        assert list(table.columns) == ["name", "ticker"]

    def test_has_exactly_15_rows(self):
        table = build_top15_ticker_table()

        assert len(table) == 15

    def test_tickers_are_unique(self):
        table = build_top15_ticker_table()

        assert table["ticker"].is_unique

    def test_no_missing_or_blank_values(self):
        table = build_top15_ticker_table()

        assert table["name"].notna().all()
        assert table["ticker"].notna().all()
        assert (table["name"].str.strip() != "").all()
        assert (table["ticker"].str.strip() != "").all()

    def test_ordered_by_aum_rank_largest_first(self):
        # The curated records carry a descending aum_usd_bn; the table must
        # preserve that order so row 0 is the single biggest fund.
        aums = [etf["aum_usd_bn"] for etf in TOP15_ETFS]

        assert aums == sorted(aums, reverse=True)


class TestProvenance:

    def test_as_of_and_source_are_populated(self):
        assert TOP15_AS_OF.strip() != ""
        assert TOP15_SOURCE.strip() != ""

    def test_every_record_has_required_fields(self):
        required = {"name", "ticker", "provider", "aum_usd_bn"}

        for etf in TOP15_ETFS:
            assert required.issubset(etf.keys())


class TestSeedFileRoundTrip:

    def test_write_then_load_returns_15_tickers_in_order(self, tmp_path: Path):
        seed_path = tmp_path / "ticker_top15.xlsx"

        write_top15_ticker_file(seed_path)
        tickers = load_tickers(seed_path)

        expected = build_top15_ticker_table()["ticker"].tolist()
        assert tickers == expected
        assert len(tickers) == 15

    def test_written_file_matches_name_ticker_schema(self, tmp_path: Path):
        seed_path = tmp_path / "ticker_top15.xlsx"

        write_top15_ticker_file(seed_path)

        written = pd.read_excel(seed_path)
        assert list(written.columns) == ["name", "ticker"]

    def test_load_tickers_missing_file_raises(self, tmp_path: Path):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_tickers(tmp_path / "nope.xlsx")
