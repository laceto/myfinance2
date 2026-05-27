"""
hedge_funds/strategies/momentum/cross_sectional.py — Cross-sectional momentum (F051–F060).

Based on the classic Jegadeesh & Titman (1993) 12-1 momentum anomaly:
rank symbols by 12-month return skipping the most recent month, buy top
quintile, sell (or avoid) bottom quintile.

Config params:
    formation_months : int  — lookback in trading days (~252 = 12 months, default 252)
    skip_months      : int  — skip most recent N days (default 21 ≈ 1 month)
    top_n            : int  — number of symbols to buy (default 3)
    bottom_n         : int  — number of symbols to sell if not long_only (default 0)
    long_only        : bool — if True, never emit SELL (default True)
    rebalance_freq   : int  — emit signals every N bars (default 21 = monthly)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from hedge_funds.base import BaseHedgeFund
from hedge_funds.config import FundConfig
from hedge_funds.signals import Order, OrderType, Signal, SignalAction

LOG = logging.getLogger(__name__)


class CrossSectionalMomentumFund(BaseHedgeFund):
    """
    Cross-sectional momentum strategy: rank the universe by trailing return,
    buy the winners, optionally sell the losers.

    Used by funds F051–F060 with different universes and lookbacks.
    """

    def __init__(self, config: FundConfig) -> None:
        super().__init__(config)
        self._formation: int = int(config.params.get("formation_months", 252))
        self._skip: int = int(config.params.get("skip_months", 21))
        self._top_n: int = int(config.params.get("top_n", 3))
        self._bottom_n: int = int(config.params.get("bottom_n", 0))
        self._long_only: bool = bool(config.params.get("long_only", True))
        self._rebalance_freq: int = int(config.params.get("rebalance_freq", 21))
        self._bar_count: int = 0  # tracks when to rebalance

        if self._formation <= self._skip:
            raise ValueError(
                f"[{config.fund_id}] formation_months ({self._formation}) must be "
                f"> skip_months ({self._skip})"
            )

    # ── Batch signal generation ────────────────────────────────────────────────

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Rank all universe symbols by momentum and emit BUY for top_n / SELL for bottom_n.
        Expects data to contain full history (≥ formation + skip bars per symbol).
        """
        scores: dict[str, float] = {}
        prices: dict[str, float] = {}

        for symbol in self._config.universe:
            df = data[data["symbol"] == symbol].sort_values("date")
            min_bars = self._formation + self._skip + 1
            if len(df) < min_bars:
                LOG.debug("[%s] %s: only %d bars (need %d)", self.fund_id, symbol, len(df), min_bars)
                continue

            score = self._momentum_score(df["close"])
            if score is not None:
                scores[symbol] = score
                prices[symbol] = float(df["close"].iloc[-1])

        if not scores:
            LOG.warning("[%s] No symbols had sufficient history for momentum ranking.", self.fund_id)
            return []

        return self._rank_to_signals(scores, prices)

    # ── Incremental bar update ─────────────────────────────────────────────────

    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        """
        Momentum is rebalanced on a schedule (rebalance_freq), not every bar.
        Returns None on non-rebalance bars.
        """
        self._bar_count += 1
        # Single-symbol on_bar doesn't have a full cross-section; no order.
        return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _momentum_score(self, close: pd.Series) -> Optional[float]:
        """
        Compute 12-1 momentum: return from bar [-formation-skip] to bar [-skip].

        Returns None if prices are non-positive (guard against bad data).
        """
        start_price = float(close.iloc[-(self._formation + self._skip)])
        end_price = float(close.iloc[-self._skip])

        if start_price <= 0:
            return None
        return (end_price - start_price) / start_price

    def _rank_to_signals(
        self, scores: dict[str, float], prices: dict[str, float]
    ) -> list[Signal]:
        """Convert momentum scores to BUY / SELL signals for top/bottom N."""
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        signals: list[Signal] = []

        for rank, (symbol, score) in enumerate(ranked):
            if rank < self._top_n:
                conf = min(max(score, 0.0), 1.0)
                LOG.info("[%s] MOM BUY rank=%d %s score=%.4f", self.fund_id, rank + 1, symbol, score)
                signals.append(Signal(
                    fund_id=self.fund_id,
                    symbol=symbol,
                    action=SignalAction.BUY,
                    confidence=conf,
                    signal_type=f"cross_sectional_momentum_top{self._top_n}",
                    metadata={"rank": rank + 1, "momentum_score": score},
                ))

        if not self._long_only:
            bottom_start = max(len(ranked) - self._bottom_n, self._top_n)
            for rank, (symbol, score) in enumerate(ranked[bottom_start:], start=bottom_start + 1):
                conf = min(max(-score, 0.0), 1.0)
                LOG.info("[%s] MOM SELL rank=%d %s score=%.4f", self.fund_id, rank, symbol, score)
                signals.append(Signal(
                    fund_id=self.fund_id,
                    symbol=symbol,
                    action=SignalAction.SELL,
                    confidence=conf,
                    signal_type=f"cross_sectional_momentum_bottom{self._bottom_n}",
                    metadata={"rank": rank, "momentum_score": score},
                ))

        return signals
