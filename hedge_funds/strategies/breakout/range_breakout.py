"""
hedge_funds/strategies/breakout/range_breakout.py — N-day range breakout (F021–F035).

Reuses ta.breakout.range_quality primitives where applicable.

Signal logic (Donchian-style):
    BUY  when close > max(high, last N bars)  — breakout above range
    SELL when close < min(low,  last N bars)  — breakdown below range

Config params:
    bo_window    : int  — lookback window for high/low range (default 20)
    long_only    : bool — if True, only BUY signals are emitted (default False)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from hedge_funds.base import BaseHedgeFund
from hedge_funds.config import FundConfig
from hedge_funds.signals import Order, OrderType, Signal, SignalAction

LOG = logging.getLogger(__name__)


class RangeBreakoutFund(BaseHedgeFund):
    """
    Donchian channel breakout strategy for US equities.

    Buys when close exceeds the N-day high, sells (or shorts) when close
    falls below the N-day low. Used by funds F021–F035.
    """

    def __init__(self, config: FundConfig) -> None:
        super().__init__(config)
        self._bo_window: int = int(config.params.get("bo_window", 20))
        self._long_only: bool = bool(config.params.get("long_only", False))
        self._history: dict[str, list[dict]] = {}

        if self._bo_window < 2:
            raise ValueError(f"[{config.fund_id}] bo_window must be >= 2, got {self._bo_window}")

    # ── Batch signal generation ────────────────────────────────────────────────

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []
        for symbol in self._config.universe:
            df = data[data["symbol"] == symbol].sort_values("date")

            if len(df) <= self._bo_window:
                LOG.debug("[%s] %s: only %d bars (need >%d)", self.fund_id, symbol, len(df), self._bo_window)
                continue

            signal = self._breakout_signal(symbol, df)
            if signal is not None:
                signals.append(signal)

        return signals

    # ── Incremental bar update ─────────────────────────────────────────────────

    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        symbol = str(bar["symbol"])
        if symbol not in self._config.universe:
            return None

        self._history.setdefault(symbol, []).append(bar.to_dict())
        df = pd.DataFrame(self._history[symbol])

        if len(df) <= self._bo_window:
            return None

        signal = self._breakout_signal(symbol, df)
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

    def _breakout_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """
        Compare the latest close against the N-day high/low of the prior window.

        The prior window excludes the current bar ([-N-1 : -1]) so we are
        measuring whether today breaks yesterday's range, not tomorrow's.
        """
        prior = df.iloc[-(self._bo_window + 1):-1]
        current_close = float(df["close"].iloc[-1])

        range_high = float(prior["high"].max())
        range_low = float(prior["low"].min())
        range_span = range_high - range_low

        # Normalised distance from the boundary (confidence proxy)
        if range_span > 0:
            conf = min(abs(current_close - range_high) / range_span, 1.0)
        else:
            conf = 0.0

        if current_close > range_high:
            LOG.info("[%s] BREAKOUT %s above %.2f (close=%.2f)", self.fund_id, symbol, range_high, current_close)
            return Signal(
                fund_id=self.fund_id,
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=conf,
                signal_type=f"range_breakout_{self._bo_window}d_high",
            )

        if not self._long_only and current_close < range_low:
            conf_down = min(abs(current_close - range_low) / range_span, 1.0) if range_span > 0 else 0.0
            LOG.info("[%s] BREAKDOWN %s below %.2f (close=%.2f)", self.fund_id, symbol, range_low, current_close)
            return Signal(
                fund_id=self.fund_id,
                symbol=symbol,
                action=SignalAction.SELL,
                confidence=conf_down,
                signal_type=f"range_breakout_{self._bo_window}d_low",
            )

        return None
