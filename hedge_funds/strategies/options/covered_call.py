"""
hedge_funds/strategies/options/covered_call.py — Options strategies (F096–F100).

Planned implementations:
    F096 CoveredCallFund       — Write OTM calls against long equity position
    F097 CashSecuredPutFund    — Sell OTM puts on high-quality stocks
    F098 IronCondorFund        — Sell call spread + put spread (range-bound)
    F099 WheelStrategyFund     — CSP → CC cycle
    F100 CalendarSpreadFund    — Buy far month, sell near month

Note: Alpaca's Options API has different order types (legs) and requires
separate handling from equity orders. Full implementation deferred until
the equity + crypto layer is proven in paper trading.

Status: STUB — raises NotImplementedError until implemented.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from hedge_funds.base import BaseHedgeFund
from hedge_funds.signals import Order, Signal


class OptionsFund(BaseHedgeFund):
    """Generic placeholder for all options-based funds F096–F100. (STUB)"""

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        raise NotImplementedError(
            f"[{self.fund_id}] OptionsFund.compute_signals() not yet implemented. "
            "Options strategies require Alpaca Options API integration (phase 2)."
        )

    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        raise NotImplementedError(
            f"[{self.fund_id}] OptionsFund.on_bar() not yet implemented."
        )
