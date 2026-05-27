"""
hedge_funds/alpaca_client.py — Alpaca broker adapter (paper and live).

Uses alpaca-py (official SDK). Install with:
    pip install alpaca-py>=0.35.0

Mode switching is a constructor argument:
    TradingMode.PAPER  → paper-api.alpaca.markets
    TradingMode.LIVE   → api.alpaca.markets

Alpaca does NOT track which orders belong to which fund — that is the
portfolio layer's responsibility. Position objects returned here have an
empty fund_id by design.

Failure modes:
    - alpaca-py not installed    → ImportError at construction time
    - Invalid credentials        → alpaca SDK raises APIError on first call
    - Order rejected by Alpaca   → submit_order raises AlpacaOrderError
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseExecutionClient
from .config import TradingMode
from .signals import Order, OrderType, Position

LOG = logging.getLogger(__name__)


class AlpacaOrderError(RuntimeError):
    """Raised when Alpaca rejects an order submission."""


class AlpacaExecutionClient(BaseExecutionClient):
    """
    Thin wrapper around alpaca-py TradingClient.

    Paper vs live is controlled entirely by the mode constructor argument —
    no other code change needed to switch between environments.
    """

    def __init__(self, api_key: str, secret_key: str, mode: TradingMode) -> None:
        try:
            from alpaca.trading.client import TradingClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is required for live/paper execution. "
                "Install with: pip install alpaca-py>=0.35.0"
            ) from exc

        self._mode = mode
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=(mode == TradingMode.PAPER),
        )
        LOG.info("AlpacaExecutionClient initialised [mode=%s]", mode.value)

    def submit_order(self, order: Order) -> str:
        """
        Submit a market or limit order to Alpaca.

        Returns the Alpaca-assigned order UUID on success.
        Raises AlpacaOrderError if the broker rejects the order.
        """
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore[import]
        from alpaca.trading.requests import (  # type: ignore[import]
            LimitOrderRequest,
            MarketOrderRequest,
        )

        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        tif = TimeInForce.GTC if order.time_in_force == "gtc" else TimeInForce.DAY

        try:
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                )
            elif order.order_type == OrderType.LIMIT:
                if order.limit_price is None:
                    raise AlpacaOrderError(
                        f"Limit order for {order.symbol} is missing limit_price"
                    )
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                )
            else:
                raise AlpacaOrderError(
                    f"Order type {order.order_type.value!r} not yet supported"
                )

            result = self._client.submit_order(request)
            LOG.info(
                "Order submitted [id=%s fund=%s symbol=%s side=%s qty=%s]",
                result.id, order.fund_id, order.symbol, order.side, order.quantity,
            )
            return str(result.id)

        except Exception as exc:
            raise AlpacaOrderError(
                f"Alpaca rejected order [{order.fund_id} {order.symbol} "
                f"{order.side} {order.quantity}]: {exc}"
            ) from exc

    def get_positions(self) -> list[Position]:
        """Return all open positions. fund_id is empty — broker has no fund concept."""
        raw = self._client.get_all_positions()
        return [
            Position(
                fund_id="",
                symbol=p.symbol,
                quantity=float(p.qty),
                avg_cost=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pnl=float(p.unrealized_pl),
            )
            for p in raw
        ]

    def get_account_cash(self) -> float:
        """Return available buying power."""
        account = self._client.get_account()
        return float(account.cash)

    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending order by its Alpaca UUID."""
        self._client.cancel_order_by_id(order_id)
        LOG.info("Order cancelled [id=%s]", order_id)
