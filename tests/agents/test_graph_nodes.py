"""
tests/agents/test_graph_nodes.py

Covers:
  - prepare_node raises on missing parquet (bad state → abort)
  - subgraph worker mocks ask_bo_trader / ask_ma_trader; never raises on exception
  - synthesise_node handles None/error results gracefully and returns final_output str
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.graph_nodes import prepare_node, create_subgraph, synthesise_node


# ---------------------------------------------------------------------------
# prepare_node
# ---------------------------------------------------------------------------


class TestPrepareNode:
    def test_raises_on_missing_parquet(self, tmp_path):
        state = {
            "symbol":        "A.MI",
            "data_source":   "parquet",
            "analysis_date": None,
        }
        missing_path = tmp_path / "no_such.parquet"

        with patch("agents.graph_nodes.RESULTS_PATH", missing_path):
            with pytest.raises((FileNotFoundError, ValueError)):
                prepare_node(state)


# ---------------------------------------------------------------------------
# create_subgraph (worker)
# ---------------------------------------------------------------------------


def _make_payload(bo_snapshot: dict = None, ma_snapshot: dict = None) -> str:
    return json.dumps({
        "date":              "2026-04-24",
        "symbol":            "A.MI",
        "breakout_snapshot": bo_snapshot or {},
        "ma_snapshot":       ma_snapshot or {},
    })


def _mock_trader_result(fields: dict) -> MagicMock:
    mock = MagicMock()
    mock.model_dump.return_value = fields
    return mock


class TestSubgraphWorker:
    def test_breakout_worker_returns_result(self):
        mock_result = _mock_trader_result({"verdict": "bullish", "regime": "trending"})
        payload = _make_payload(bo_snapshot={"rbo_150": 1})
        state = {"payload_json": payload}

        with patch("agents.graph_nodes.ask_bo_trader", return_value=mock_result):
            subgraph = create_subgraph("breakout")
            out = subgraph.invoke(state)

        assert "breakout_result" in out
        assert "error" not in out["breakout_result"]
        assert out["breakout_result"]["verdict"] == "bullish"

    def test_ma_worker_returns_result(self):
        mock_result = _mock_trader_result({"verdict": "neutral", "regime": "ranging"})
        payload = _make_payload(ma_snapshot={"rema_50100": 1})
        state = {"payload_json": payload}

        with patch("agents.graph_nodes.ask_ma_trader", return_value=mock_result):
            subgraph = create_subgraph("ma")
            out = subgraph.invoke(state)

        assert "ma_result" in out
        assert "error" not in out["ma_result"]
        assert out["ma_result"]["verdict"] == "neutral"

    def test_worker_returns_error_dict_on_exception(self):
        # Corrupt payload_json → worker must return {"error": ...} not raise
        state = {"payload_json": "NOT VALID JSON"}
        subgraph = create_subgraph("breakout")
        out = subgraph.invoke(state)
        assert "breakout_result" in out
        assert "error" in out["breakout_result"]

    def test_unknown_worker_name_does_not_raise(self):
        # Invariant: subgraph never propagates exceptions to the caller.
        payload = _make_payload()
        state = {"payload_json": payload}
        subgraph = create_subgraph("unknown_strategy")
        out = subgraph.invoke(state)
        assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# synthesise_node
# ---------------------------------------------------------------------------


class TestSynthesiseNode:
    def _bo_result(self):
        return {"verdict": "bullish", "regime": "trending", "description": "Breakout forming"}

    def _ma_result(self):
        return {"verdict": "neutral", "regime": "ranging", "description": "MA crossover pending"}

    def test_returns_final_output_string(self):
        state = {
            "resolved_date":  "2026-04-24",
            "symbol":         "A.MI",
            "breakout_result": self._bo_result(),
            "ma_result":       self._ma_result(),
        }
        out = synthesise_node(state)
        assert "final_output" in out
        assert isinstance(out["final_output"], str)
        assert len(out["final_output"]) > 0

    def test_symbol_appears_in_output(self):
        state = {
            "resolved_date":  "2026-04-24",
            "symbol":         "A.MI",
            "breakout_result": self._bo_result(),
            "ma_result":       self._ma_result(),
        }
        out = synthesise_node(state)
        assert "A.MI" in out["final_output"]

    def test_date_appears_in_output(self):
        state = {
            "resolved_date":  "2026-04-24",
            "symbol":         "A.MI",
            "breakout_result": self._bo_result(),
            "ma_result":       self._ma_result(),
        }
        out = synthesise_node(state)
        assert "2026-04-24" in out["final_output"]

    def test_handles_none_breakout_result(self):
        state = {
            "resolved_date":  "2026-04-24",
            "symbol":         "A.MI",
            "breakout_result": None,
            "ma_result":       self._ma_result(),
        }
        out = synthesise_node(state)
        assert isinstance(out["final_output"], str)

    def test_handles_error_breakout_result(self):
        state = {
            "resolved_date":  "2026-04-24",
            "symbol":         "A.MI",
            "breakout_result": {"error": "scoring failed"},
            "ma_result":       self._ma_result(),
        }
        out = synthesise_node(state)
        brief = out["final_output"]
        assert isinstance(brief, str)
        assert "unavailable" in brief.lower() or "scoring failed" in brief

    def test_handles_both_results_missing(self):
        state = {"resolved_date": "2026-04-24", "symbol": "A.MI"}
        out = synthesise_node(state)
        assert isinstance(out["final_output"], str)
