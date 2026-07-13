"""
test_append_daily_to_historical.py — Unit tests for append_daily_to_historical.

The module is the single source of truth for the CI step that appends a
market's freshly-downloaded daily bar to its committed historical Parquet.
It is exercised by download_daily_ohlc.yml for both the `it` and `etf`
markets, so its behaviour must be identical and market-agnostic.

Coverage
--------
append_daily_frames (pure core, no I/O):
  - happy path: concatenates historical + today preserving order and schema
  - no historical (None): returns today's frame unchanged
  - empty today: returns historical unchanged (no phantom rows)
  - preserves the long/tidy column schema
  - does NOT deduplicate (matches the pre-existing inline CI behaviour)

market_paths (path resolver):
  - maps a market code to the today/historical Parquet paths
  - rejects an empty/blank market code with an actionable error

append_market_file (I/O wrapper):
  - creates historical from today when historical file is absent
  - appends when historical file already exists
  - missing today file -> FileNotFoundError with actionable message
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from append_daily_to_historical import (
    append_daily_frames,
    append_market_file,
    market_paths,
)


def _ohlc(rows):
    """Build a minimal long/tidy OHLC frame from (symbol, date, close) tuples."""
    return pd.DataFrame(
        [
            {"symbol": s, "date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1}
            for s, d, c in rows
        ]
    )


class TestAppendDailyFramesHappyPath:

    def test_concatenates_historical_then_today_in_order(self):
        historical = _ohlc([("A.MI", "2026-07-10", 10.0), ("B.MI", "2026-07-10", 20.0)])
        today = _ohlc([("A.MI", "2026-07-11", 11.0), ("B.MI", "2026-07-11", 21.0)])

        result = append_daily_frames(today, historical)

        assert len(result) == 4
        assert result["date"].tolist() == ["2026-07-10", "2026-07-10", "2026-07-11", "2026-07-11"]
        assert result["close"].tolist() == [10.0, 20.0, 11.0, 21.0]

    def test_preserves_long_tidy_schema(self):
        historical = _ohlc([("A.MI", "2026-07-10", 10.0)])
        today = _ohlc([("A.MI", "2026-07-11", 11.0)])

        result = append_daily_frames(today, historical)

        assert list(result.columns) == ["symbol", "date", "open", "high", "low", "close", "volume"]


class TestAppendDailyFramesEdgeCases:

    def test_returns_today_when_no_historical(self):
        today = _ohlc([("A.MI", "2026-07-11", 11.0)])

        result = append_daily_frames(today, None)

        assert result["close"].tolist() == [11.0]
        assert len(result) == 1

    def test_returns_historical_when_today_empty(self):
        historical = _ohlc([("A.MI", "2026-07-10", 10.0)])
        empty_today = historical.iloc[0:0].copy()

        result = append_daily_frames(empty_today, historical)

        assert result["close"].tolist() == [10.0]

    def test_does_not_deduplicate_repeated_rows(self):
        # The pre-existing inline CI step did a plain concat with no dedup;
        # preserve that contract so behaviour is unchanged by the extraction.
        bar = _ohlc([("A.MI", "2026-07-11", 11.0)])

        result = append_daily_frames(bar, bar)

        assert len(result) == 2


class TestMarketPaths:

    def test_resolves_today_and_historical_paths_for_market(self):
        today_path, historical_path = market_paths("etf")

        assert today_path == Path("data/ohlc/today/etf/ohlc_data.parquet")
        assert historical_path == Path("data/ohlc/historical/etf/ohlc_data.parquet")

    def test_blank_market_raises_value_error(self):
        with pytest.raises(ValueError, match="market"):
            market_paths("  ")


class TestAppendMarketFile:

    def test_creates_historical_from_today_when_absent(self, tmp_path: Path):
        today_path = tmp_path / "today.parquet"
        historical_path = tmp_path / "hist" / "historical.parquet"
        _ohlc([("A.MI", "2026-07-11", 11.0)]).to_parquet(today_path, index=False)

        combined = append_market_file(today_path, historical_path)

        assert historical_path.exists()
        assert len(combined) == 1
        assert pd.read_parquet(historical_path)["close"].tolist() == [11.0]

    def test_appends_today_to_existing_historical(self, tmp_path: Path):
        today_path = tmp_path / "today.parquet"
        historical_path = tmp_path / "historical.parquet"
        _ohlc([("A.MI", "2026-07-10", 10.0)]).to_parquet(historical_path, index=False)
        _ohlc([("A.MI", "2026-07-11", 11.0)]).to_parquet(today_path, index=False)

        combined = append_market_file(today_path, historical_path)

        assert combined["close"].tolist() == [10.0, 11.0]
        assert pd.read_parquet(historical_path)["close"].tolist() == [10.0, 11.0]

    def test_missing_today_file_raises_file_not_found(self, tmp_path: Path):
        missing_today = tmp_path / "does_not_exist.parquet"
        historical_path = tmp_path / "historical.parquet"

        with pytest.raises(FileNotFoundError, match="does_not_exist"):
            append_market_file(missing_today, historical_path)
