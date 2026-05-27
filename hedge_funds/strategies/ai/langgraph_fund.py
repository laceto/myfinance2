"""
hedge_funds/strategies/ai/langgraph_fund.py — LangGraph AI-powered funds (F086–F095).

Adapts the existing agents.create_manager() (built for Italian equities) to
US equities by switching to:
    data_source = "live"    — downloads via Yahoo Finance (works for US tickers)
    benchmark   = "SPY"     — S&P 500 as the US reference index

The LLM report includes a Position Recommendation section. We parse the
first line for BUY / SELL / HOLD to produce a Signal.

Config params:
    benchmark     : str  — benchmark ticker (default "SPY")
    signal_parser : str  — "first_line" | "regex" (default "first_line")

Failure mode:
    LLM calls can fail (rate limit, timeout, API error). The fund catches
    all exceptions from the graph and returns an empty signal list so the
    portfolio circuit-breaker can log and continue.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from hedge_funds.base import BaseHedgeFund
from hedge_funds.config import FundConfig
from hedge_funds.signals import Order, OrderType, Signal, SignalAction

LOG = logging.getLogger(__name__)

# Pattern that matches "BUY", "SELL", or "HOLD" (case-insensitive) anywhere in the first 200 chars.
_ACTION_RE = re.compile(r"\b(BUY|SELL|HOLD)\b", re.IGNORECASE)


class LangGraphFund(BaseHedgeFund):
    """
    AI-powered hedge fund using the myfinance2 LangGraph multi-agent system.

    Processes each symbol in the universe through the breakout + MA parallel
    subgraph and synthesises a structured brief. The Position Recommendation
    in the brief determines the signal action.

    Used by funds F086–F095.
    """

    def __init__(self, config: FundConfig) -> None:
        super().__init__(config)
        self._benchmark: str = str(config.params.get("benchmark", "SPY"))

    # ── Batch signal generation ────────────────────────────────────────────────

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Run the LangGraph analysis pipeline for each universe symbol.

        Calls the LLM; can be slow (1–5 seconds per symbol). Recommend
        running a small universe (≤5 symbols) per fund for daily paper trading.
        """
        try:
            from agents import create_manager  # type: ignore[import]
        except ImportError as exc:
            LOG.error("[%s] agents package not available: %s", self.fund_id, exc)
            return []

        signals: list[Signal] = []
        for symbol in self._config.universe:
            try:
                signal = self._analyse_symbol(symbol, create_manager)
                if signal is not None:
                    signals.append(signal)
            except Exception as exc:
                LOG.error("[%s] LangGraph analysis failed for %s: %s", self.fund_id, symbol, exc, exc_info=True)
                # Do not re-raise — one bad symbol must not block the rest.

        return signals

    # ── Incremental bar update ─────────────────────────────────────────────────

    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        """
        Run the full LangGraph pipeline for a single bar's symbol.

        This is intentionally slow (LLM call per bar) — suitable only for
        end-of-day daily bar processing, not intraday streaming.
        """
        symbol = str(bar["symbol"])
        if symbol not in self._config.universe:
            return None

        try:
            from agents import create_manager  # type: ignore[import]
        except ImportError as exc:
            LOG.error("[%s] agents package not available: %s", self.fund_id, exc)
            return None

        try:
            signal = self._analyse_symbol(symbol, create_manager)
        except Exception as exc:
            LOG.error("[%s] LangGraph on_bar failed for %s: %s", self.fund_id, symbol, exc, exc_info=True)
            return None

        if signal is None:
            return None

        price = float(bar["close"])
        qty = self.position_size(symbol, price)
        if qty <= 0:
            return None

        return Order(
            fund_id=self.fund_id,
            symbol=symbol,
            side=signal.action.value,
            quantity=qty,
            order_type=OrderType.MARKET,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _analyse_symbol(self, symbol: str, create_manager) -> Optional[Signal]:  # noqa: ANN001
        """Run the graph and extract a Signal from the final_output brief."""
        graph = create_manager(
            symbol=symbol,
            data_source="live",
            benchmark=self._benchmark,
        )
        result = graph.invoke({
            "symbol": symbol,
            "data_source": "live",
            "benchmark": self._benchmark,
        })

        brief: str = result.get("final_output", "")
        if not brief:
            LOG.warning("[%s] LangGraph returned empty brief for %s", self.fund_id, symbol)
            return None

        action = self._parse_action(brief)
        if action is None:
            LOG.info("[%s] Could not extract BUY/SELL/HOLD from brief for %s", self.fund_id, symbol)
            return None

        return Signal(
            fund_id=self.fund_id,
            symbol=symbol,
            action=action,
            confidence=0.7,  # LLM doesn't give a numeric score; use moderate default
            signal_type="langgraph_multi_agent_analysis",
            metadata={"brief_preview": brief[:200]},
        )

    @staticmethod
    def _parse_action(brief: str) -> Optional[SignalAction]:
        """Extract the first BUY / SELL / HOLD keyword from the brief text."""
        match = _ACTION_RE.search(brief[:500])  # only check the opening summary
        if match is None:
            return None
        keyword = match.group(1).upper()
        mapping = {"BUY": SignalAction.BUY, "SELL": SignalAction.SELL, "HOLD": SignalAction.HOLD}
        return mapping.get(keyword)
