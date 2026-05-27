"""
hedge_funds/strategies/trend/ma_crossover.py — MA crossover strategy (F001–F020).

Supports SMA and EMA with configurable fast/slow windows.
Detects golden cross (BUY) and death cross (SELL) on each bar for every
ticker in the fund's universe.

Config params:
    fast_window  : int  — short MA period
    slow_window  : int  — long MA period
    ma_type      : str  — "sma" | "ema"

Signal logic:
    prev_bar:  fast < slow  →  curr_bar: fast > slow  →  BUY  (golden cross)
    prev_bar:  fast > slow  →  curr_bar: fast < slow  →  SELL (death cross)
    otherwise → no signal emitted (HOLD signals clutter the log)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from hedge_funds.base import BaseHedgeFund
from hedge_funds.config import FundConfig
from hedge_funds.signals import Order, OrderType, Signal, SignalAction

LOG = logging.getLogger(__name__)


def _compute_ma(series: pd.Series, window: int, ma_type: str) -> pd.Series:
    """Compute SMA or EMA. Raises ValueError for unknown ma_type."""
    if ma_type == "sma":
        return series.rolling(window=window).mean()
    if ma_type == "ema":
        return series.ewm(span=window, adjust=False).mean()
    raise ValueError(f"Unknown ma_type {ma_type!r}. Expected 'sma' or 'ema'.")


class MACrossoverFund(BaseHedgeFund):
    """
    Moving-average crossover strategy for US equities.

    Used by funds F001–F020 with different windows, MA types, and universes.
    """

    def __init__(self, config: FundConfig) -> None:
        super().__init__(config)
        self._fast_window: int = int(config.params.get("fast_window", 50))
        self._slow_window: int = int(config.params.get("slow_window", 200))
        self._ma_type: str = str(config.params.get("ma_type", "sma"))
        self._history: dict[str, list[dict]] = {}  # symbol → list of bar dicts

        if self._fast_window >= self._slow_window:
            raise ValueError(
                f"[{config.fund_id}] fast_window ({self._fast_window}) must be "
                f"< slow_window ({self._slow_window})"
            )
        if self._ma_type not in ("sma", "ema"):
            raise ValueError(f"[{config.fund_id}] ma_type must be 'sma' or 'ema', got {self._ma_type!r}")

    # ── Batch signal generation ────────────────────────────────────────────────

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Scan all universe symbols in the given OHLC history.
        Emit one Signal per symbol that crossed over on the latest bar.
        """
        signals: list[Signal] = []
        for symbol in self._config.universe:
            df = data[data["symbol"] == symbol].sort_values("date")

            min_bars = self._slow_window + 1
            if len(df) < min_bars:
                LOG.debug("[%s] %s: only %d bars (need %d)", self.fund_id, symbol, len(df), min_bars)
                continue

            signal = self._crossover_signal(symbol, df["close"])
            if signal is not None:
                signals.append(signal)

        return signals

    # ── Incremental bar update ─────────────────────────────────────────────────

    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        """Append the bar to the history buffer and emit an order on crossover."""
        symbol = str(bar["symbol"])
        if symbol not in self._config.universe:
            return None

        self._history.setdefault(symbol, []).append(bar.to_dict())
        df = pd.DataFrame(self._history[symbol])

        min_bars = self._slow_window + 1
        if len(df) < min_bars:
            return None

        signal = self._crossover_signal(symbol, df["close"])
        if signal is None:
            return None

        price = float(bar["close"])
        qty = self.position_size(symbol, price)
        if qty <= 0:
            LOG.warning("[%s] position_size returned 0 for %s at price %.2f", self.fund_id, symbol, price)
            return None

        return Order(
            fund_id=self.fund_id,
            symbol=symbol,
            side=signal.action.value,  # "buy" or "sell"
            quantity=qty,
            order_type=OrderType.MARKET,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _crossover_signal(self, symbol: str, close: pd.Series) -> Optional[Signal]:
        """
        Detect a golden or death cross on the last two bars.

        Returns a Signal or None (no crossover).
        Confidence is the normalised gap between fast and slow MA at the
        current bar (|fast - slow| / close), capped at 1.0.
        """
        fast = _compute_ma(close, self._fast_window, self._ma_type)
        slow = _compute_ma(close, self._slow_window, self._ma_type)

        prev_diff = float(fast.iloc[-2] - slow.iloc[-2])
        curr_diff = float(fast.iloc[-1] - slow.iloc[-1])
        last_close = float(close.iloc[-1])

        confidence = min(abs(curr_diff) / last_close if last_close > 0 else 0.0, 1.0)
        signal_base = f"{self._ma_type}_{self._fast_window}_{self._slow_window}"

        if prev_diff <= 0 and curr_diff > 0:
            LOG.info("[%s] GOLDEN CROSS %s (conf=%.4f)", self.fund_id, symbol, confidence)
            return Signal(
                fund_id=self.fund_id,
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=confidence,
                signal_type=f"{signal_base}_golden_cross",
            )

        if prev_diff >= 0 and curr_diff < 0:
            LOG.info("[%s] DEATH CROSS  %s (conf=%.4f)", self.fund_id, symbol, confidence)
            return Signal(
                fund_id=self.fund_id,
                symbol=symbol,
                action=SignalAction.SELL,
                confidence=confidence,
                signal_type=f"{signal_base}_death_cross",
            )

        return None
