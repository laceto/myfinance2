"""
hedge_funds/strategies/mean_reversion/bollinger_mr.py — Bollinger Band MR (F036–F050).

Signal logic:
    BUY  when close crosses below lower band  (oversold)
    SELL when close crosses above upper band  (overbought, or close long)

Config params:
    bb_window    : int   — MA period for band midline (default 20)
    bb_std       : float — band width in standard deviations (default 2.0)
    long_only    : bool  — if True, only BUY (entry) signals emitted (default True)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from hedge_funds.base import BaseHedgeFund
from hedge_funds.config import FundConfig
from hedge_funds.signals import Order, OrderType, Signal, SignalAction

LOG = logging.getLogger(__name__)


class BollingerMRFund(BaseHedgeFund):
    """
    Bollinger Band mean-reversion strategy for US equities.

    Enters long when price touches the lower band, closes (or shorts) on
    upper band touch. Used by funds F036–F050 with different windows.
    """

    def __init__(self, config: FundConfig) -> None:
        super().__init__(config)
        self._bb_window: int = int(config.params.get("bb_window", 20))
        self._bb_std: float = float(config.params.get("bb_std", 2.0))
        self._long_only: bool = bool(config.params.get("long_only", True))
        self._history: dict[str, list[dict]] = {}

        if self._bb_window < 2:
            raise ValueError(f"[{config.fund_id}] bb_window must be >= 2")
        if self._bb_std <= 0:
            raise ValueError(f"[{config.fund_id}] bb_std must be > 0")

    # ── Batch signal generation ────────────────────────────────────────────────

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []
        for symbol in self._config.universe:
            df = data[data["symbol"] == symbol].sort_values("date")

            if len(df) < self._bb_window + 1:
                continue

            signal = self._band_signal(symbol, df["close"])
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

        if len(df) < self._bb_window + 1:
            return None

        signal = self._band_signal(symbol, df["close"])
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

    def _band_signal(self, symbol: str, close: pd.Series) -> Optional[Signal]:
        """Compute bands and detect a band touch on the latest bar."""
        mid = close.rolling(window=self._bb_window).mean()
        std = close.rolling(window=self._bb_window).std()
        upper = mid + self._bb_std * std
        lower = mid - self._bb_std * std

        curr = float(close.iloc[-1])
        curr_upper = float(upper.iloc[-1])
        curr_lower = float(lower.iloc[-1])
        curr_mid = float(mid.iloc[-1])

        if pd.isna(curr_upper) or pd.isna(curr_lower):
            return None

        band_width = curr_upper - curr_lower

        if curr <= curr_lower:
            # Distance below lower band as confidence
            conf = min(abs(curr - curr_lower) / band_width if band_width > 0 else 0.0, 1.0)
            LOG.info("[%s] BB TOUCH LOWER %s (close=%.2f lower=%.2f)", self.fund_id, symbol, curr, curr_lower)
            return Signal(
                fund_id=self.fund_id,
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=conf,
                signal_type=f"bollinger_{self._bb_window}_{self._bb_std}_lower_touch",
            )

        if not self._long_only and curr >= curr_upper:
            conf = min(abs(curr - curr_upper) / band_width if band_width > 0 else 0.0, 1.0)
            LOG.info("[%s] BB TOUCH UPPER %s (close=%.2f upper=%.2f)", self.fund_id, symbol, curr, curr_upper)
            return Signal(
                fund_id=self.fund_id,
                symbol=symbol,
                action=SignalAction.SELL,
                confidence=conf,
                signal_type=f"bollinger_{self._bb_window}_{self._bb_std}_upper_touch",
            )

        return None
