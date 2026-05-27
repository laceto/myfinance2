"""
hedge_funds/signals.py — Value objects for the signal → order → fill pipeline.

These are plain dataclasses (not Pydantic) because they are created at high
frequency inside strategy hot-paths and never need serialisation validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SignalAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"  # explicit position close (regardless of direction)


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Signal:
    """Trading signal produced by a fund's strategy logic."""

    fund_id: str
    symbol: str
    action: SignalAction
    # Normalised confidence [0.0, 1.0] — used for position sizing or filtering
    confidence: float
    signal_type: str  # e.g. "sma_50_200_golden_cross"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Signal.confidence must be in [0, 1], got {self.confidence}"
            )


@dataclass
class Order:
    """Execution order derived from a Signal — sent to AlpacaExecutionClient."""

    fund_id: str
    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Order.side must be 'buy' or 'sell', got {self.side!r}")
        if self.quantity <= 0:
            raise ValueError(f"Order.quantity must be > 0, got {self.quantity}")


@dataclass
class Fill:
    """Confirmed execution of an order, returned by the broker."""

    order_id: str
    fund_id: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    filled_at: datetime
    commission: float = 0.0


@dataclass
class Position:
    """Current open position for a (fund, symbol) pair."""

    fund_id: str
    symbol: str
    quantity: float  # positive = long, negative = short
    avg_cost: float
    current_price: float
    unrealized_pnl: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0
