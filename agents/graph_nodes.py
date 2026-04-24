"""
agents/graph_nodes.py — Node implementations for the TechnicalAnalysis graph.

Nodes:
  prepare_node      — loads data and builds breakout + MA snapshots for one symbol,
                      serialises everything to payload_json.
  create_subgraph() — factory: returns a compiled single-node subgraph that calls
                      one AI trader (ask_bo_trader or ask_ma_trader).
  synthesise_node   — formats both AI reports side by side into final_output.

Invariant: payload_json is the sole data channel from prepare_node to workers.
           No worker reads from disk or network.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agents.graph_state import TechnicalAnalysisState
from agents._tools.prepare_tools import load_analysis_data, load_live_data
from ask_bo_trader import ask_bo_trader
from ask_ma_trader import ask_ma_trader
from ta.breakout.bo_snapshot import build_snapshot as bo_build_snapshot
from ta.ma.ma_snapshot import build_snapshot as ma_build_snapshot

log = logging.getLogger(__name__)

RESULTS_PATH = Path("data/results/it/analysis_results.parquet")


# ---------------------------------------------------------------------------
# prepare_node
# ---------------------------------------------------------------------------


def prepare_node(state: TechnicalAnalysisState) -> dict:
    """
    Load data for one symbol, build breakout and MA snapshots, serialise to payload_json.

    Raises:
        ValueError: If the symbol cannot be found or snapshots cannot be built.
        FileNotFoundError: If the parquet file is missing (Mode A).
    """
    t0 = time.perf_counter()
    symbol        = state["symbol"]
    data_source   = state.get("data_source", "parquet")
    analysis_date = state.get("analysis_date")
    benchmark     = state.get("benchmark", "FTSEMIB.MI")
    fx            = state.get("fx")

    if data_source == "live":
        resolved_date, df = load_live_data(symbol, benchmark=benchmark, fx=fx)
    else:
        resolved_date, df = load_analysis_data(RESULTS_PATH, symbol, analysis_date)

    log.info("[prepare] symbol=%s resolved_date=%s", symbol, resolved_date)

    breakout_snapshot = bo_build_snapshot(df)
    ma_snapshot       = ma_build_snapshot(df)

    log.info("[prepare] snapshots built in %.2fs", time.perf_counter() - t0)

    payload = {
        "date":              resolved_date,
        "symbol":            symbol,
        "breakout_snapshot": breakout_snapshot,
        "ma_snapshot":       ma_snapshot,
    }

    return {
        "payload_json":  json.dumps(payload),
        "resolved_date": resolved_date,
    }


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------


def create_subgraph(worker_name: str):
    """
    Build a compiled single-node subgraph that calls one AI trader.

    The subgraph never raises — exceptions are caught and returned as
    {"error": str(exc)} so the other worker can continue.

    Args:
        worker_name: "breakout" → calls ask_bo_trader; "ma" → calls ask_ma_trader.
    """
    result_key = f"{worker_name}_result"

    def run_worker(state: TechnicalAnalysisState) -> dict:
        try:
            payload = json.loads(state["payload_json"])
            symbol  = payload["symbol"]
            if worker_name == "breakout":
                snapshot = payload["breakout_snapshot"]
                result   = ask_bo_trader(snapshot, ticker=symbol)
                log.info("[%s] analysis complete for %s", worker_name, symbol)
                return {result_key: result.model_dump()}
            elif worker_name == "ma":
                snapshot = payload["ma_snapshot"]
                result   = ask_ma_trader(snapshot, ticker=symbol)
                log.info("[%s] analysis complete for %s", worker_name, symbol)
                return {result_key: result.model_dump()}
            else:
                raise ValueError(f"Unknown worker: {worker_name!r}")
        except Exception as exc:
            log.error("[%s] worker failed: %s", worker_name, exc, exc_info=True)
            return {result_key: {"error": str(exc)}}

    graph = StateGraph(TechnicalAnalysisState)
    graph.add_node("run_worker", run_worker)
    graph.add_edge(START, "run_worker")
    graph.add_edge("run_worker", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# synthesise_node
# ---------------------------------------------------------------------------


def _format_report(result: dict, title: str) -> list[str]:
    """Format a TraderAnalysis / MATraderAnalysis model_dump() as a readable section."""
    lines = [f"## {title}"]
    if not result:
        lines.append("  (no result)")
        return lines
    if "error" in result:
        lines.append(f"  unavailable — {result['error']}")
        return lines
    # Verdict first for quick scan
    if "verdict" in result:
        lines.append(f"**Verdict:** {result['verdict']}")
        lines.append("")
    for key, val in result.items():
        if key == "verdict":
            continue
        label = key.replace("_", " ").capitalize()
        lines.append(f"**{label}:** {val}")
    return lines


def synthesise_node(state: TechnicalAnalysisState) -> dict:
    """
    Format both AI reports side by side into final_output.

    Never raises — missing or errored results are reported as "unavailable".
    """
    resolved_date   = state.get("resolved_date", "unknown")
    symbol          = state.get("symbol", "unknown")
    breakout_result = state.get("breakout_result") or {}
    ma_result       = state.get("ma_result") or {}

    lines: list[str] = [
        f"# Technical Analysis — {symbol} — {resolved_date}",
        "",
    ]
    lines.extend(_format_report(breakout_result, "Breakout Analysis"))
    lines.append("")
    lines.extend(_format_report(ma_result, "MA Crossover Analysis"))

    brief = "\n".join(lines)
    log.info("[synthesise] brief chars=%d", len(brief))

    return {"final_output": brief}
