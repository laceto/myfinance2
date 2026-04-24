"""
agents/graph_nodes.py — Node implementations for the TechnicalAnalysis graph.

Nodes:
  prepare_node         — loads data and builds breakout + MA snapshots for one symbol,
                         serialises everything to payload_json.
  create_subgraph()    — factory: returns a compiled single-node subgraph that calls
                         one AI trader (ask_bo_trader or ask_ma_trader).
  _call_synthesis_llm  — calls the LLM once to compile both AI reports into a final brief.
  synthesise_node      — formats inputs and delegates to _call_synthesis_llm.

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

RESULTS_PATH   = Path("data/results/it/analysis_results.parquet")
_DEFAULT_MODEL = "gpt-4o"

# ---------------------------------------------------------------------------
# Stage 3 — Report compilation prompt
# ---------------------------------------------------------------------------

_REPORT_SYSTEM = """\
You are a senior technical analyst at a long/short equity proprietary trading desk.
You have received two independent technical assessments for {ticker}:
a range-breakout analysis and an MA-crossover analysis.

Your audience is a portfolio manager who needs actionable entry/exit conviction,
not a retail summary. Be direct, precise, and opinionated. Every claim must be
backed by a specific signal or metric from the assessments. Do not fabricate data.
If a figure is unavailable, say so explicitly.

---

## Position Recommendation

State clearly: LONG · SHORT · NEUTRAL — and the conviction level: High / Medium / Low.

Two to three paragraphs covering:
- The primary technical catalyst driving the recommended position.
- The key signal that would invalidate the thesis (stop-loss trigger).
- Suggested holding horizon: short-term (< 4 weeks), medium-term (1–3 months),
  or structural (> 3 months).

---

## Signal Confluence Scorecard

Summary table with columns:
| Dimension | Breakout | MA Crossover | Confluence |

Where Confluence is one of: ✅ Aligned · ⚠️ Mixed · 🔴 Diverging

Include these dimensions: Trend direction, Regime (rrg), Entry timing, \
Volume confirmation, Risk/Stop level.

---

## Breakout Analysis Deep-Dive

1. **Verdict** — direction and conviction from the breakout assessment.
2. **Key signals** — bullet list of every active signal, age, and flip status.
3. **Range quality** — range setup, volatility compression, touch count.
4. **Volume** — breakout volume confirmation or consolidation quiet.
5. **Stop level** — the specific stop-loss level from the assessment.
6. **Trigger to watch** — the exact signal or price event that would change the call.

---

## MA Crossover Analysis Deep-Dive

1. **Verdict** — direction and conviction from the MA assessment.
2. **Key signals** — EMA/SMA alignment, triple-confluence status, signal age/flip.
3. **Trend quality** — ADX level and slope, RSI, MA gap and gap slope.
4. **Volume** — crossover volume confirmation and post-crossover sustainability.
5. **Stop level** — the specific stop-loss level from the assessment.
6. **Trigger to watch** — the exact signal or metric that would change the call.

---

## Entry & Exit Plan

- **Entry trigger**: the specific signal flip or price level that confirms entry.
- **Stop-loss**: the signal reversal or price level that invalidates the thesis.
- **First target**: the nearest resistance or measured-move projection.

---

## Bottom Line

One paragraph: net technical conviction, the single most important signal to
monitor, and the specific event or price level that would change the call.
"""

_REPORT_HUMAN = """\
Breakout analysis:    {breakout_analysis}

MA crossover analysis: {ma_analysis}
"""


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


def _call_synthesis_llm(ticker: str, breakout_analysis: str, ma_analysis: str) -> str:
    """
    Call the LLM once to compile both AI reports into a final technical brief.

    Isolated into its own function so tests can mock the LLM call without
    wrestling with LangChain's LCEL pipe internals.

    Args:
        ticker:            Ticker symbol (e.g. "A2A.MI") — injected into the system prompt.
        breakout_analysis: JSON string of the breakout worker's result (or "unavailable").
        ma_analysis:       JSON string of the MA worker's result (or "unavailable").

    Returns:
        Markdown string containing the compiled report.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages([
        ("system", _REPORT_SYSTEM),
        ("human",  _REPORT_HUMAN),
    ])
    llm = ChatOpenAI(model=_DEFAULT_MODEL, temperature=0)

    response = (prompt | llm).invoke({
        "ticker":             ticker,
        "breakout_analysis":  breakout_analysis,
        "ma_analysis":        ma_analysis,
    })
    return response.content if hasattr(response, "content") else str(response)


def synthesise_node(state: TechnicalAnalysisState) -> dict:
    """
    Compile both AI reports into a final technical brief via an LLM call.

    Never raises — missing or errored worker results are passed as "unavailable"
    to the LLM so it can still produce a partial report.
    """
    ticker          = state.get("symbol", "unknown")
    breakout_result = state.get("breakout_result") or {}
    ma_result       = state.get("ma_result") or {}

    def _fmt(result: dict) -> str:
        if not result:
            return "unavailable"
        if "error" in result:
            return f"unavailable — {result['error']}"
        return json.dumps(result, indent=2)

    bo_analysis = _fmt(breakout_result)
    ma_analysis = _fmt(ma_result)

    log.info("[synthesise] generating report for %s", ticker)
    brief = _call_synthesis_llm(ticker, bo_analysis, ma_analysis)
    log.info("[synthesise] report generated (%d chars)", len(brief))

    return {"final_output": brief}
