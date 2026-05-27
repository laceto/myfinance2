"""
hedge_funds/strategies/volatility/vol_targeting.py — Volatility strategies (F061–F070).

Planned implementations:
    F061 VolTargetingFund  — Scale position size by inverse of realised vol
    F062 VIXRegimeFund     — VIX-based regime: risk-on / risk-off positioning
    F063 GARCHVolFund      — GARCH(1,1) vol forecast-based sizing
    F064–F070              — Further vol-based strategies

Status: STUB — raises NotImplementedError until implemented.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from hedge_funds.base import BaseHedgeFund
from hedge_funds.signals import Order, Signal


class VolTargetingFund(BaseHedgeFund):
    """Volatility-targeting strategy — scales exposure by realised vol. (STUB)"""

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        raise NotImplementedError(
            f"[{self.fund_id}] VolTargetingFund.compute_signals() not yet implemented."
        )

    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        raise NotImplementedError(
            f"[{self.fund_id}] VolTargetingFund.on_bar() not yet implemented."
        )


class VIXRegimeFund(BaseHedgeFund):
    """VIX-regime-based positioning fund. (STUB)"""

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        raise NotImplementedError(
            f"[{self.fund_id}] VIXRegimeFund.compute_signals() not yet implemented."
        )

    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        raise NotImplementedError(
            f"[{self.fund_id}] VIXRegimeFund.on_bar() not yet implemented."
        )
