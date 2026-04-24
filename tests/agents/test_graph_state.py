"""
tests/agents/test_graph_state.py

Covers: _last reducer and TechnicalAnalysisState field contract.
All tests are deterministic — no I/O, no parquet, no network.
"""

import pytest

from agents.graph_state import TechnicalAnalysisState, _last


class TestLastReducer:
    def test_returns_second_value(self):
        assert _last("old", "new") == "new"

    def test_returns_second_when_first_is_none(self):
        assert _last(None, "value") == "value"

    def test_returns_none_when_second_is_none(self):
        assert _last("value", None) is None

    def test_works_with_dicts(self):
        a = {"x": 1}
        b = {"y": 2}
        assert _last(a, b) == {"y": 2}

    def test_works_with_lists(self):
        assert _last([1, 2], [3, 4]) == [3, 4]


class TestStateSchema:
    """TechnicalAnalysisState must accept all documented fields without error."""

    def test_empty_state_is_valid(self):
        state: TechnicalAnalysisState = {}
        assert isinstance(state, dict)

    def test_full_state_is_valid(self):
        state: TechnicalAnalysisState = {
            "symbol":        "A2A.MI",
            "analysis_date": "2026-04-24",
            "data_source":   "parquet",
            "benchmark":     "FTSEMIB.MI",
            "fx":            None,
            "resolved_date": "2026-04-24",
            "payload_json":  '{"date":"2026-04-24","symbol":"A2A.MI","breakout_snapshot":{},"ma_snapshot":{}}',
            "breakout_result": {"verdict": "bullish", "regime": "trending"},
            "ma_result":       {"verdict": "neutral", "regime": "ranging"},
            "final_output":  "Technical Analysis — A2A.MI — 2026-04-24\n...",
        }
        assert state["symbol"] == "A2A.MI"
        assert state["data_source"] == "parquet"
        assert state["benchmark"] == "FTSEMIB.MI"
        assert state["fx"] is None

    def test_data_source_defaults_absent(self):
        state: TechnicalAnalysisState = {"symbol": "A2A.MI"}
        assert "data_source" not in state

    def test_parallel_merge_uses_last(self):
        # Simulate what LangGraph does when two parallel branches return results:
        # the second update overwrites the first via _last.
        base: TechnicalAnalysisState = {
            "symbol": "A2A.MI",
            "breakout_result": None,
            "ma_result": None,
        }

        breakout_update = {"breakout_result": {"verdict": "bullish"}}
        ma_update       = {"ma_result":       {"verdict": "neutral"}}

        merged = dict(base)
        for key, val in {**breakout_update, **ma_update}.items():
            merged[key] = _last(merged.get(key), val)

        assert merged["breakout_result"]["verdict"] == "bullish"
        assert merged["ma_result"]["verdict"] == "neutral"
        assert merged["symbol"] == "A2A.MI"  # untouched field preserved
