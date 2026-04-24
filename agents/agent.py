"""
agents/agent.py — create_manager() graph factory.

Builds the LangGraph StateGraph:
  START → prepare_node → [breakout_worker, ma_worker] → synthesise_node → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.graph_state import TechnicalAnalysisState
from agents.graph_nodes import prepare_node, synthesise_node
from agents._subagents import WORKER_NAMES, build_subgraphs


def create_manager(
    symbol: str,
    analysis_date: str | None = None,
    data_source: str = "parquet",   # "parquet" | "live"
    benchmark: str = "FTSEMIB.MI",
    fx: str | None = None,
    checkpointer=None,
):
    """
    Build and compile the TechnicalAnalysis LangGraph for a single ticker.

    Args:
        symbol:        Ticker to analyse (required, e.g. "A2A.MI").
        analysis_date: ISO date string; None → latest available bar.
        data_source:   "parquet" (default) or "live" (downloads via YFinanceDataHandler).
        benchmark:     Benchmark ticker.
                       Mode A: excluded from result set (default "FTSEMIB.MI").
                       Mode B: used for calculate_relative_prices (e.g. "H4ZX.DE").
        fx:            Optional FX ticker for currency conversion (e.g. "EURUSD=X").
                       Pass None when stock and benchmark share the same currency.
        checkpointer:  Optional LangGraph checkpointer for persistence / resumption.

    Returns:
        CompiledStateGraph ready for .invoke() / .stream().
    """
    builder = StateGraph(TechnicalAnalysisState)

    builder.add_node("prepare", prepare_node)
    builder.add_node("synthesise", synthesise_node)

    subgraphs = build_subgraphs()
    for name, subgraph in subgraphs.items():
        builder.add_node(name, subgraph)

    builder.add_edge(START, "prepare")

    for name in WORKER_NAMES:
        builder.add_edge("prepare", name)

    for name in WORKER_NAMES:
        builder.add_edge(name, "synthesise")

    builder.add_edge("synthesise", END)

    initial_state: TechnicalAnalysisState = {
        "symbol":        symbol,
        "analysis_date": analysis_date,
        "data_source":   data_source,
        "benchmark":     benchmark,
        "fx":            fx,
    }

    graph = builder.compile(checkpointer=checkpointer)
    graph._initial_state = initial_state
    return graph
