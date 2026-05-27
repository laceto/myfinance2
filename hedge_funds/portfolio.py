"""
hedge_funds/portfolio.py — Team orchestrator for all hedge funds.

HedgeFundPortfolio is the single entry point that:
  1. Drives signal generation across all registered funds
  2. Routes Orders to the execution client
  3. Isolates fund failures — one rogue agent cannot halt the team
  4. Surfaces a team-level summary for the dashboard

Design invariants:
  - A fund failure (exception) is caught, logged with full traceback, and
    skipped for that cycle. Other funds continue unaffected.
  - The execution client is injected (not constructed internally), so the
    full team can be run against a mock in tests.
  - All per-cycle results are returned as a dict keyed by fund_id for
    downstream observability (dashboard, audit log).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .base import BaseExecutionClient, BaseHedgeFund
from .config import TradingMode
from .signals import Order, Signal, SignalAction

LOG = logging.getLogger(__name__)


class HedgeFundPortfolio:
    """
    Orchestrates a team of BaseHedgeFund instances against one execution client.

    Usage:
        portfolio = HedgeFundPortfolio(funds=[...], execution_client=client)
        orders = portfolio.run_cycle(ohlc_df)
        print(portfolio.summary())
    """

    def __init__(
        self,
        funds: list[BaseHedgeFund],
        execution_client: BaseExecutionClient,
    ) -> None:
        if not funds:
            raise ValueError("HedgeFundPortfolio requires at least one fund.")

        self._funds: dict[str, BaseHedgeFund] = {f.fund_id: f for f in funds}
        self._client = execution_client
        self._cycle_count: int = 0
        LOG.info(
            "HedgeFundPortfolio initialised [funds=%d paper=%d live=%d]",
            len(funds),
            sum(1 for f in funds if f.mode == TradingMode.PAPER),
            sum(1 for f in funds if f.mode == TradingMode.LIVE),
        )

    # ── Execution cycle ────────────────────────────────────────────────────────

    def run_cycle(self, data: pd.DataFrame) -> dict[str, list[Order]]:
        """
        Run one full signal → order → submit cycle across all funds.

        Args:
            data: OHLC DataFrame (long/tidy) for all symbols in all universes.
                  Columns: [symbol, date, open, high, low, close, volume]

        Returns:
            Dict mapping fund_id → list of orders submitted this cycle.
            Funds that raised an exception map to an empty list.
        """
        self._cycle_count += 1
        all_orders: dict[str, list[Order]] = {}

        for fund_id, fund in self._funds.items():
            try:
                signals = fund.compute_signals(data)
                actionable = [s for s in signals if s.action != SignalAction.HOLD]
                orders = [self._signal_to_order(s, data) for s in actionable]
                submitted: list[Order] = []
                for order in orders:
                    try:
                        self._client.submit_order(order)
                        submitted.append(order)
                    except Exception as exc:
                        LOG.error(
                            "Order submission failed [fund=%s symbol=%s]: %s",
                            fund_id, order.symbol, exc, exc_info=True,
                        )
                all_orders[fund_id] = submitted

            except Exception as exc:
                LOG.error(
                    "Fund %s raised an exception during cycle %d: %s",
                    fund_id, self._cycle_count, exc, exc_info=True,
                )
                all_orders[fund_id] = []

        LOG.info(
            "Cycle %d complete [total_orders=%d]",
            self._cycle_count,
            sum(len(v) for v in all_orders.values()),
        )
        return all_orders

    def run_bar(self, bar: pd.Series) -> dict[str, Optional[Order]]:
        """
        Push a single new price bar to all funds (live incremental path).

        Returns a dict mapping fund_id → Order (or None if the fund was silent).
        """
        results: dict[str, Optional[Order]] = {}
        for fund_id, fund in self._funds.items():
            try:
                order = fund.on_bar(bar)
                results[fund_id] = order
                if order is not None:
                    self._client.submit_order(order)
            except Exception as exc:
                LOG.error(
                    "Fund %s failed on bar [symbol=%s date=%s]: %s",
                    fund_id, bar.get("symbol"), bar.get("date"), exc, exc_info=True,
                )
                results[fund_id] = None
        return results

    # ── Observability ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Team-level summary suitable for the dashboard."""
        fund_statuses = [f.status() for f in self._funds.values()]
        return {
            "total_funds": len(self._funds),
            "paper_funds": sum(1 for f in self._funds.values() if f.mode == TradingMode.PAPER),
            "live_funds": sum(1 for f in self._funds.values() if f.mode == TradingMode.LIVE),
            "total_unrealized_pnl": sum(s["unrealized_pnl"] for s in fund_statuses),
            "total_realized_pnl": sum(s["realized_pnl"] for s in fund_statuses),
            "cycles_run": self._cycle_count,
            "funds": fund_statuses,
        }

    def get_fund(self, fund_id: str) -> BaseHedgeFund:
        """Retrieve a specific fund by ID. Raises KeyError if not found."""
        if fund_id not in self._funds:
            raise KeyError(f"Fund {fund_id!r} not in portfolio. Known: {list(self._funds)}")
        return self._funds[fund_id]

    # ── Private helpers ────────────────────────────────────────────────────────

    def _signal_to_order(self, signal: Signal, data: pd.DataFrame) -> Order:
        """Convert a Signal into an Order, computing quantity from latest price."""
        from .signals import Order, OrderType

        latest = data[data["symbol"] == signal.symbol].sort_values("date")
        if latest.empty:
            raise ValueError(
                f"Cannot size order for {signal.symbol!r}: no price data in current cycle."
            )
        price = float(latest["close"].iloc[-1])
        fund = self._funds[signal.fund_id]
        quantity = fund.position_size(signal.symbol, price)

        return Order(
            fund_id=signal.fund_id,
            symbol=signal.symbol,
            side="buy" if signal.action == SignalAction.BUY else "sell",
            quantity=quantity,
            order_type=OrderType.MARKET,
        )
