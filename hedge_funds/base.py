"""
hedge_funds/base.py — Abstract contracts for funds and execution clients.

Design decisions:
  - BaseExecutionClient is a pure interface so tests can inject a mock without
    importing alpaca-py at all.
  - BaseHedgeFund depends on BaseExecutionClient, NOT AlpacaExecutionClient
    (dependency inversion — easy to test, easy to swap brokers).
  - Subclasses only need to implement compute_signals() and on_bar().
    All bookkeeping (capital, P&L, position sizing) lives here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from .config import FundConfig, TradingMode
from .signals import Order, Position, Signal, SignalAction


class BaseExecutionClient(ABC):
    """Broker-agnostic interface for order submission and account state queries."""

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order. Returns broker-assigned order_id."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return all open positions (fund_id may be empty — broker knows no funds)."""
        ...

    @abstractmethod
    def get_account_cash(self) -> float:
        """Return available buying power / cash."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending (unfilled) order."""
        ...


class BaseHedgeFund(ABC):
    """
    Abstract base for all 100 hedge fund implementations.

    Subclasses must implement:
      - compute_signals(data)  — batch signal generation from full OHLC history
      - on_bar(bar)            — incremental update on a single new price bar

    The base class provides:
      - Configuration access (fund_id, name, mode, universe)
      - Capital and P&L tracking
      - Position sizing helper

    Invariant: execution is NOT the fund's responsibility. The fund produces
    Signals/Orders; HedgeFundPortfolio routes them to the execution client.
    """

    def __init__(self, config: FundConfig) -> None:
        self._config = config
        self._log = logging.getLogger(f"hedge_fund.{config.fund_id}")
        self._positions: dict[str, Position] = {}
        self._realized_pnl: float = 0.0
        self._capital: float = config.initial_capital

    # ── Read-only properties ───────────────────────────────────────────────────

    @property
    def fund_id(self) -> str:
        return self._config.fund_id

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def mode(self) -> TradingMode:
        return self._config.mode

    @property
    def capital(self) -> float:
        return self._capital

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_pnl(self) -> float:
        return self._realized_pnl + self.unrealized_pnl

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Compute trading signals from a full OHLC DataFrame.

        Args:
            data: long/tidy DataFrame with columns
                  [symbol, date, open, high, low, close, volume]

        Returns:
            List of Signal objects for symbols with actionable signals.
            Return an empty list (not HOLD signals) when nothing to do.
        """
        ...

    @abstractmethod
    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        """
        Process a single new price bar (incremental / live-trading path).

        Args:
            bar: Series with keys [symbol, date, open, high, low, close, volume]

        Returns:
            An Order to submit, or None if no action is required.
        """
        ...

    # ── Helpers ────────────────────────────────────────────────────────────────

    def position_size(self, symbol: str, price: float) -> float:
        """
        Compute order quantity from config.position_size_pct and available capital.

        Returns 0.0 when price is zero or capital is exhausted.
        """
        if price <= 0:
            self._log.warning(f"[{self.fund_id}] position_size called with price={price} for {symbol}")
            return 0.0
        max_value = self._capital * self._config.position_size_pct
        return max_value / price

    def status(self) -> dict:
        """Serialisable snapshot of the fund's current state — used by dashboard."""
        return {
            "fund_id": self.fund_id,
            "name": self.name,
            "mode": self.mode.value,
            "capital": self._capital,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self._realized_pnl,
            "total_pnl": self.total_pnl,
            "open_positions": len(self._positions),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.fund_id!r}, name={self.name!r}, mode={self.mode.value!r})"
        )
