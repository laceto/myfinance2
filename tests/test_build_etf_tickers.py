"""
test_build_etf_tickers.py — Unit tests for build_etf_tickers.build_etf_ticker_table.

Coverage
--------
build_etf_ticker_table:
  - happy path: maps (name, yf_ticker) -> (name, ticker), preserves row order
  - drops rows with missing/NaN yf_ticker
  - drops rows with missing/NaN name
  - drops rows with duplicate ticker (keeps first occurrence)
  - empty input DataFrame -> empty output with correct columns
  - missing expected source columns -> raises ValueError with actionable message
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from build_etf_tickers import build_etf_ticker_table


def _source_df(rows):
    """Build a minimal justETF-shaped source DataFrame from (name, yf_ticker) tuples."""
    return pd.DataFrame(rows, columns=["name", "yf_ticker"])


class TestBuildEtfTickerTableHappyPath:

    def test_maps_name_and_yf_ticker_to_name_and_ticker(self):
        source = _source_df(
            [
                ("iShares Core MSCI World", "IWDA.AS"),
                ("Vanguard FTSE All-World", "VWCE.DE"),
            ]
        )

        result = build_etf_ticker_table(source)

        assert list(result.columns) == ["name", "ticker"]
        assert result["name"].tolist() == ["iShares Core MSCI World", "Vanguard FTSE All-World"]
        assert result["ticker"].tolist() == ["IWDA.AS", "VWCE.DE"]

    def test_preserves_row_order(self):
        source = _source_df([("C ETF", "C.MI"), ("A ETF", "A.MI"), ("B ETF", "B.MI")])

        result = build_etf_ticker_table(source)

        assert result["ticker"].tolist() == ["C.MI", "A.MI", "B.MI"]


class TestBuildEtfTickerTableEdgeCases:

    def test_drops_rows_with_missing_ticker(self):
        source = _source_df(
            [("Valid ETF", "OK.AS"), ("No Ticker ETF", np.nan), ("Empty Ticker ETF", "")]
        )

        result = build_etf_ticker_table(source)

        assert result["ticker"].tolist() == ["OK.AS"]

    def test_drops_rows_with_missing_name(self):
        source = _source_df([(np.nan, "NOTNAME.AS"), ("Valid ETF", "OK.AS")])

        result = build_etf_ticker_table(source)

        assert result["name"].tolist() == ["Valid ETF"]

    def test_drops_duplicate_tickers_keeping_first(self):
        source = _source_df(
            [
                ("First Listing", "DUP.MI"),
                ("Second Listing Same Ticker", "DUP.MI"),
                ("Unique ETF", "UNQ.MI"),
            ]
        )

        result = build_etf_ticker_table(source)

        assert result["ticker"].tolist() == ["DUP.MI", "UNQ.MI"]
        assert result["name"].tolist() == ["First Listing", "Unique ETF"]

    def test_empty_input_returns_empty_output_with_correct_columns(self):
        source = _source_df([])

        result = build_etf_ticker_table(source)

        assert list(result.columns) == ["name", "ticker"]
        assert len(result) == 0

    def test_all_nan_ticker_column_returns_empty_output(self):
        source = _source_df([("ETF One", np.nan), ("ETF Two", np.nan)])

        result = build_etf_ticker_table(source)

        assert len(result) == 0


class TestBuildEtfTickerTableFailureModes:

    def test_missing_required_source_columns_raises_value_error(self):
        source = pd.DataFrame({"isin": ["IE00ABC"], "some_other_col": ["x"]})

        with pytest.raises(ValueError, match="name|yf_ticker"):
            build_etf_ticker_table(source)
