"""
hedge_funds/strategies/crypto/btc_momentum.py — Crypto MA momentum (F081–F085).

Identical mechanics to MACrossoverFund but targets Alpaca's crypto pairs
(24/7 markets, no market-hours constraint).

Alpaca crypto symbols use the format "BTC/USD", "ETH/USD", etc.

Config params: same as MACrossoverFund
    fast_window, slow_window, ma_type
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from hedge_funds.base import BaseHedgeFund
from hedge_funds.config import FundConfig
from hedge_funds.signals import Order, OrderType, Signal, SignalAction
from hedge_funds.strategies.trend.ma_crossover import _compute_ma

LOG = logging.getLogger(__name__)


class CryptoMomentumFund(BaseHedgeFund):
    """
    MA crossover momentum for Alpaca-listed crypto pairs (F081–F085).

    Re-uses _compute_ma from the equity MA crossover module — crypto OHLC
    data is structurally identical to equity OHLC data.
    """

    def __init__(self, config: FundConfig) -> None:
        super().__init__(config)
        self._fast_window: int = int(config.params.get("fast_window", 20))
        self._slow_window: int = int(config.params.get("slow_window", 50))
        self._ma_type: str = str(config.params.get("ma_type", "ema"))
        self._history: dict[str, list[dict]] = {}

        if self._fast_window >= self._slow_window:
            raise ValueError(
                f"[{config.fund_id}] fast_window ({self._fast_window}) must be "
                f"< slow_window ({self._slow_window})"
            )

    # ── Batch signal generation ────────────────────────────────────────────────

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []
        for symbol in self._config.universe:
            df = data[data["symbol"] == symbol].sort_values("date")

            min_bars = self._slow_window + 1
            if len(df) < min_bars:
                LOG.debug("[%s] %s: only %d bars (need %d)", self.fund_id, symbol, len(df), min_bars)
                continue

            fast = _compute_ma(df["close"], self._fast_window, self._ma_type)
            slow = _compute_ma(df["close"], self._slow_window, self._ma_type)

            prev_diff = float(fast.iloc[-2] - slow.iloc[-2])
            curr_diff = float(fast.iloc[-1] - slow.iloc[-1])
            last_close = float(df["close"].iloc[-1])

            conf = min(abs(curr_diff) / last_close if last_close > 0 else 0.0, 1.0)
            sig_base = f"crypto_{self._ma_type}_{self._fast_window}_{self._slow_window}"

            if prev_diff <= 0 and curr_diff > 0:
                LOG.info("[%s] CRYPTO BUY %s (conf=%.4f)", self.fund_id, symbol, conf)
                signals.append(Signal(
                    fund_id=self.fund_id,
                    symbol=symbol,
                    action=SignalAction.BUY,
                    confidence=conf,
                    signal_type=f"{sig_base}_golden_cross",
                ))
            elif prev_diff >= 0 and curr_diff < 0:
                LOG.info("[%s] CRYPTO SELL %s (conf=%.4f)", self.fund_id, symbol, conf)
                signals.append(Signal(
                    fund_id=self.fund_id,
                    symbol=symbol,
                    action=SignalAction.SELL,
                    confidence=conf,
                    signal_type=f"{sig_base}_death_cross",
                ))

        return signals

    # ── Incremental bar update ─────────────────────────────────────────────────

    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        symbol = str(bar["symbol"])
        if symbol not in self._config.universe:
            return None

        self._history.setdefault(symbol, []).append(bar.to_dict())
        df = pd.DataFrame(self._history[symbol])

        if len(df) < self._slow_window + 1:
            return None

        fast = _compute_ma(df["close"], self._fast_window, self._ma_type)
        slow = _compute_ma(df["close"], self._slow_window, self._ma_type)

        prev_diff = float(fast.iloc[-2] - slow.iloc[-2])
        curr_diff = float(fast.iloc[-1] - slow.iloc[-1])

        if prev_diff <= 0 and curr_diff > 0:
            action = SignalAction.BUY
        elif prev_diff >= 0 and curr_diff < 0:
            action = SignalAction.SELL
        else:
            return None

        price = float(bar["close"])
        qty = self.position_size(symbol, price)
        if qty <= 0:
            return None

        return Order(
            fund_id=self.fund_id,
            symbol=symbol,
            side=action.value,
            quantity=qty,
            order_type=OrderType.MARKET,
        )
