"""
hedge_funds/strategies/factor/multi_factor.py — Factor strategies (F071–F080).

Planned implementations:
    F071 QualityFactorFund    — ROE + low debt screen
    F072 ValueFactorFund      — P/E + P/B relative to universe
    F073 MomentumFactorFund   — 12-1 momentum within factor framework
    F074 LowVolFactorFund     — Low-volatility anomaly (buy low-beta)
    F075–F080                 — Multi-factor combined, Jensen alpha, sector rotation

Status: STUB — raises NotImplementedError until implemented.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from hedge_funds.base import BaseHedgeFund
from hedge_funds.signals import Order, Signal


class FactorFund(BaseHedgeFund):
    """Generic placeholder for all factor-based funds F071–F080. (STUB)"""

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        raise NotImplementedError(
            f"[{self.fund_id}] FactorFund.compute_signals() not yet implemented."
        )

    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        raise NotImplementedError(
            f"[{self.fund_id}] FactorFund.on_bar() not yet implemented."
        )
