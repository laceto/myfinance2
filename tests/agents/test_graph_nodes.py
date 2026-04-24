"""
tests/agents/test_graph_nodes.py

Covers:
  - prepare_node raises on missing parquet (bad state → abort)
  - subgraph worker mocks ask_bo_trader / ask_ma_trader; never raises on exception
  - synthesise_node delegates to _call_synthesis_llm; formats inputs correctly;
    handles None/error results gracefully; never raises
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

    # ── Happy path ─────────────────────────────────────────────────────────────

    def test_returns_final_output_string(self):
        state = {
            "symbol":          "A.MI",
            "breakout_result": self._bo_result(),
            "ma_result":       self._ma_result(),
        }
        with patch("agents.graph_nodes._call_synthesis_llm", return_value="LONG — High\n..."):
            out = synthesise_node(state)
        assert "final_output" in out
        assert out["final_output"] == "LONG — High\n..."

    def test_passes_ticker_to_llm(self):
        state = {
            "symbol":          "A.MI",
            "breakout_result": self._bo_result(),
            "ma_result":       self._ma_result(),
        }
        with patch("agents.graph_nodes._call_synthesis_llm", return_value="report") as mock_fn:
            synthesise_node(state)
        assert mock_fn.call_args[0][0] == "A.MI"

    def test_passes_both_analyses_as_json(self):
        state = {
            "symbol":          "A.MI",
            "breakout_result": self._bo_result(),
            "ma_result":       self._ma_result(),
        }
        with patch("agents.graph_nodes._call_synthesis_llm", return_value="report") as mock_fn:
            synthesise_node(state)
        _, bo_arg, ma_arg = mock_fn.call_args[0]
        assert "bullish" in bo_arg
        assert "neutral" in ma_arg

    # ── None / error inputs ───────────────────────────────────────────────────

    def test_passes_unavailable_for_none_breakout_result(self):
        state = {
            "symbol":          "A.MI",
            "breakout_result": None,
            "ma_result":       self._ma_result(),
        }
        with patch("agents.graph_nodes._call_synthesis_llm", return_value="report") as mock_fn:
            out = synthesise_node(state)
        assert isinstance(out["final_output"], str)
        assert mock_fn.call_args[0][1] == "unavailable"

    def test_passes_error_message_for_error_breakout_result(self):
        state = {
            "symbol":          "A.MI",
            "breakout_result": {"error": "scoring failed"},
            "ma_result":       self._ma_result(),
        }
        with patch("agents.graph_nodes._call_synthesis_llm", return_value="report") as mock_fn:
            synthesise_node(state)
        bo_arg = mock_fn.call_args[0][1]
        assert "unavailable" in bo_arg
        assert "scoring failed" in bo_arg

    def test_handles_both_results_missing(self):
        state = {"symbol": "A.MI"}
        with patch("agents.graph_nodes._call_synthesis_llm", return_value="report"):
            out = synthesise_node(state)
        assert isinstance(out["final_output"], str)
