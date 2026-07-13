"""
test_cold_start_etf.py — Unit tests for cold_start_etf (historical backfill).

The cold start seeds data/ohlc/historical/etf/ohlc_data.parquet with full
history for the curated top-15 UCITS ETFs. The download is dependency-injected
(a YFinanceDataHandler-shaped object) so the orchestration — request, save,
zero-row retry, fail-fast — is tested deterministically without any network,
per the project testing rules.

Coverage
--------
backfill_historical:
  - happy path: downloads all tickers once, saves once, reports 0 missing
  - zero-row symbols are retried exactly once, then saved again
  - symbols still empty after retry are reported (and warned), not dropped
  - empty ticker list raises ValueError (fail fast — never call Yahoo blank)
cold_start:
  - writes the seed file then backfills using the loaded tickers
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

from cold_start_etf import backfill_historical, cold_start


class FakeHandler:
    """Records calls and returns a scripted list_available_data() per call.

    `avail_sequence` supplies one dict per list_available_data() invocation;
    the last entry is reused if calls exceed the sequence length. This lets a
    test simulate a symbol that is empty on the first pass and populated after
    the retry (or one that stays empty).
    """

    def __init__(self, avail_sequence: List[Dict[str, Dict[str, int]]]):
        self._avail_sequence = avail_sequence
        self._avail_idx = 0
        self.download_calls: List[List[str]] = []
        self.download_kwargs: List[dict] = []
        self.save_calls: int = 0

    def download_data(self, symbols, **kwargs):
        self.download_calls.append(list(symbols))
        self.download_kwargs.append(kwargs)
        return None

    def save_data(self, **kwargs):
        self.save_calls += 1

    def list_available_data(self):
        idx = min(self._avail_idx, len(self._avail_sequence) - 1)
        self._avail_idx += 1
        return self._avail_sequence[idx]


def _all_populated(tickers):
    return {t: {"rows": 100} for t in tickers}


def _seed_writer_for(tickers):
    """Return a seed_writer that materialises a known (name, ticker) seed.

    Decouples cold_start's integration tests from the real justETF file so the
    backfilled universe is exactly `tickers`.
    """
    import pandas as pd

    def writer(path):
        pd.DataFrame({"name": [f"n-{t}" for t in tickers], "ticker": tickers}).to_excel(path, index=False)
        return path

    return writer


class TestBackfillHistoricalHappyPath:

    def test_downloads_all_tickers_once_and_saves(self, tmp_path: Path):
        tickers = ["CSPX.L", "SWDA.L", "VUSA.L"]
        handler = FakeHandler([_all_populated(tickers)])

        summary = backfill_historical(handler, tickers, tmp_path / "hist.parquet",
                                      start="2016-01-01", end="2026-07-12")

        assert handler.download_calls == [tickers]      # exactly one download, all tickers
        assert handler.save_calls == 1
        assert summary.requested == 3
        assert summary.missing_after_retry == []


class TestBackfillHistoricalRetry:

    def test_zero_row_symbols_retried_once_then_resolved(self, tmp_path: Path):
        tickers = ["CSPX.L", "SWDA.L"]
        first_pass = {"CSPX.L": {"rows": 100}, "SWDA.L": {"rows": 0}}
        after_retry = _all_populated(tickers)
        handler = FakeHandler([first_pass, after_retry])

        summary = backfill_historical(handler, tickers, tmp_path / "hist.parquet",
                                      start="2016-01-01", end="2026-07-12")

        assert handler.download_calls == [tickers, ["SWDA.L"]]  # full, then retry the empty one
        assert handler.save_calls == 2
        assert summary.missing_after_retry == []

    def test_symbols_empty_after_retry_are_reported(self, tmp_path: Path):
        tickers = ["CSPX.L", "DEAD.L"]
        persistent = {"CSPX.L": {"rows": 100}, "DEAD.L": {"rows": 0}}
        handler = FakeHandler([persistent, persistent])

        summary = backfill_historical(handler, tickers, tmp_path / "hist.parquet",
                                      start="2016-01-01", end="2026-07-12")

        assert handler.download_calls == [tickers, ["DEAD.L"]]
        assert summary.missing_after_retry == ["DEAD.L"]


class TestBackfillHistoricalFailFast:

    def test_empty_ticker_list_raises_value_error(self, tmp_path: Path):
        handler = FakeHandler([{}])

        with pytest.raises(ValueError, match="ticker"):
            backfill_historical(handler, [], tmp_path / "hist.parquet",
                                start="2016-01-01", end="2026-07-12")


class TestColdStartIntegration:

    def test_writes_seed_then_backfills_loaded_tickers(self, tmp_path: Path):
        seed_path = tmp_path / "ticker_active.xlsx"
        out_path = tmp_path / "hist.parquet"
        tickers = ["CSPX.L", "SWDA.L", "AAA.L"]

        summary = cold_start(
            ticker_file=seed_path,
            output_path=out_path,
            start="2016-01-01",
            end="2026-07-12",
            seed_writer=_seed_writer_for(tickers),
            handler_factory=lambda cache_dir: FakeHandler([_all_populated(tickers)]),
        )

        assert seed_path.exists()               # seed materialised by the injected writer
        assert summary.requested == 3           # backfilled exactly the loaded universe
        assert summary.missing_after_retry == []

    def test_empty_start_falls_back_to_default_not_blank(self, tmp_path: Path):
        # Regression: a push-triggered run passes an empty --start; that empty
        # string must NOT reach yfinance (which silently falls back to period=1y
        # and downloads only the last year instead of the full 2016 history).
        seed_path = tmp_path / "ticker_active.xlsx"
        tickers = ["CSPX.L", "SWDA.L"]
        captured = {}

        def handler_factory(cache_dir):
            handler = FakeHandler([_all_populated(tickers)])
            captured["handler"] = handler
            return handler

        from cold_start_etf import DEFAULT_START

        cold_start(
            ticker_file=seed_path,
            output_path=tmp_path / "hist.parquet",
            start="",                       # simulate the push-event empty input
            end="2026-07-12",
            seed_writer=_seed_writer_for(tickers),
            handler_factory=handler_factory,
        )

        first_download_start = captured["handler"].download_kwargs[0]["start"]
        assert first_download_start == DEFAULT_START
