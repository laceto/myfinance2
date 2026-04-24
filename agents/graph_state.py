"""
agents/graph_state.py — Typed state for the TechnicalAnalysis LangGraph.

Single source of truth for all fields that flow between nodes.
Every field uses the _last reducer so parallel branches can merge without conflict.
"""

from __future__ import annotations

from typing import Annotated, Optional

from typing_extensions import TypedDict


def _last(a, b):  # noqa: ANN001
    """Reducer: always keep the most recent value. Required for parallel fan-out/fan-in."""
    return b


class TechnicalAnalysisState(TypedDict, total=False):
    # ── Caller inputs ──────────────────────────────────────────────────────────
    symbol:        Annotated[str,           _last]  # single ticker to analyse (required)
    analysis_date: Annotated[Optional[str], _last]  # ISO date; None → latest bar
    data_source:   Annotated[str,           _last]  # "parquet" (default) | "live"
    benchmark:     Annotated[str,           _last]  # benchmark ticker
    # Mode A default: "FTSEMIB.MI" (excluded from result set)
    # Mode B default: user-supplied (used for calculate_relative_prices)
    fx:            Annotated[Optional[str], _last]  # FX ticker for currency conversion (e.g. "EURUSD=X")
    # None → no FX conversion (stock and benchmark share the same currency)

    # ── Set by prepare_node ────────────────────────────────────────────────────
    resolved_date: Annotated[str, _last]  # actual date resolved from the data
    payload_json:  Annotated[str, _last]
    # payload_json shape:
    # {
    #   "date":               "YYYY-MM-DD",
    #   "symbol":             str,
    #   "breakout_snapshot":  dict,   ← ta.breakout.bo_snapshot.build_snapshot()
    #   "ma_snapshot":        dict,   ← ta.ma.ma_snapshot.build_snapshot()
    # }

    # ── Set by subgraph workers ────────────────────────────────────────────────
    breakout_result: Annotated[Optional[dict], _last]  # TraderAnalysis.model_dump()
    ma_result:       Annotated[Optional[dict], _last]  # MATraderAnalysis.model_dump()
    # {"error": str} when the worker caught an exception

    # ── Set by synthesise_node ─────────────────────────────────────────────────
    final_output: Annotated[str, _last]  # both AI reports formatted side by side
