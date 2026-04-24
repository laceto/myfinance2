"""
tests/agents/test_prepare_tools.py

Covers load_analysis_data (Mode A) with deterministic fixture DataFrames.
No real parquet reads — all I/O uses temp files written by the test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agents._tools.prepare_tools import load_analysis_data, HISTORY_BARS


def _make_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / "analysis_results.parquet"
    df.to_parquet(p, index=False)
    return p


def _base_row(symbol: str, date_str: str) -> dict:
    return {
        "symbol": symbol,
        "date":   pd.Timestamp(date_str),
        "close":  100.0,
        "rclose": 1.0,
        "rrg":    1,
    }


class TestLoadAnalysisData:
    def test_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "no_such.parquet"
        with pytest.raises(FileNotFoundError, match="Parquet not found"):
            load_analysis_data(missing, "A.MI", None)

    def test_raises_on_invalid_date(self, tmp_path):
        p = _make_parquet(tmp_path, [_base_row("A.MI", "2026-01-01")])
        with pytest.raises(ValueError, match="not found in parquet"):
            load_analysis_data(p, "A.MI", "2020-01-01")

    def test_raises_when_symbol_not_found(self, tmp_path):
        p = _make_parquet(tmp_path, [_base_row("A.MI", "2026-01-01")])
        with pytest.raises(ValueError):
            load_analysis_data(p, "NONEXISTENT.MI", None)

    def test_resolves_latest_date_when_none(self, tmp_path):
        rows = [
            _base_row("A.MI", "2026-01-01"),
            _base_row("A.MI", "2026-01-02"),
        ]
        p = _make_parquet(tmp_path, rows)
        resolved, df = load_analysis_data(p, "A.MI", None)
        assert resolved == "2026-01-02"

    def test_resolves_specified_date(self, tmp_path):
        rows = [
            _base_row("A.MI", "2026-01-01"),
            _base_row("A.MI", "2026-01-02"),
        ]
        p = _make_parquet(tmp_path, rows)
        resolved, df = load_analysis_data(p, "A.MI", "2026-01-01")
        assert resolved == "2026-01-01"

    def test_returns_only_requested_symbol(self, tmp_path):
        rows = [
            _base_row("A.MI", "2026-01-01"),
            _base_row("B.MI", "2026-01-01"),
        ]
        p = _make_parquet(tmp_path, rows)
        _, df = load_analysis_data(p, "A.MI", None)
        assert set(df["symbol"].unique()) == {"A.MI"}

    def test_returns_dataframe(self, tmp_path):
        rows = [_base_row("A.MI", "2026-01-01"), _base_row("A.MI", "2026-01-02")]
        p = _make_parquet(tmp_path, rows)
        resolved, df = load_analysis_data(p, "A.MI", None)
        assert isinstance(resolved, str)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_history_bars_cap(self, tmp_path):
        base = pd.Timestamp("2020-01-01")
        rows = [
            _base_row("A.MI", str((base + pd.Timedelta(days=i)).date()))
            for i in range(HISTORY_BARS + 50)
        ]
        p = _make_parquet(tmp_path, rows)
        _, df = load_analysis_data(p, "A.MI", None)
        assert len(df) == HISTORY_BARS

    def test_data_sliced_up_to_analysis_date(self, tmp_path):
        rows = [_base_row("A.MI", f"2026-0{m}-01") for m in range(1, 7)]
        p = _make_parquet(tmp_path, rows)
        _, df = load_analysis_data(p, "A.MI", "2026-03-01")
        dates = pd.to_datetime(df["date"]).dt.date
        assert all(d <= pd.Timestamp("2026-03-01").date() for d in dates)
