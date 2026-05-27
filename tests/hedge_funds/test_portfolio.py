"""
Tests for hedge_funds.portfolio — HedgeFundPortfolio team orchestration.

Uses a mock execution client and stub funds to test:
  - Cycle runs across all funds
  - Fund failures are isolated (circuit breaker)
  - Summary returns accurate counts
  - Portfolio raises on empty fund list
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from hedge_funds.base import BaseExecutionClient, BaseHedgeFund
from hedge_funds.config import FundConfig, TradingMode, AssetClass, StrategyCategory
from hedge_funds.portfolio import HedgeFundPortfolio
from hedge_funds.signals import Order, OrderType, Position, Signal, SignalAction


# ── Test doubles ───────────────────────────────────────────────────────────────

def _make_config(fund_id: str = "F001") -> FundConfig:
    return FundConfig(
        fund_id=fund_id,
        name=f"Test Fund {fund_id}",
        description="stub",
        category=StrategyCategory.TREND_FOLLOWING,
        asset_class=AssetClass.EQUITY,
        strategy_class="hedge_funds.strategies.trend.ma_crossover.MACrossoverFund",
        universe=["SPY"],
    )


class _AlwaysBuyFund(BaseHedgeFund):
    """Stub: always emits a BUY signal for SPY."""

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        return [Signal(
            fund_id=self.fund_id,
            symbol="SPY",
            action=SignalAction.BUY,
            confidence=0.9,
            signal_type="stub_buy",
        )]

    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        return None


class _BrokenFund(BaseHedgeFund):
    """Stub: always raises RuntimeError to test the circuit breaker."""

    def compute_signals(self, data: pd.DataFrame) -> list[Signal]:
        raise RuntimeError("Intentional test failure")

    def on_bar(self, bar: pd.Series) -> Optional[Order]:
        raise RuntimeError("Intentional test failure")


class _MockExecutionClient(BaseExecutionClient):
    """Records every submit_order call."""

    def __init__(self) -> None:
        self.submitted: list[Order] = []

    def submit_order(self, order: Order) -> str:
        self.submitted.append(order)
        return f"mock-order-{len(self.submitted)}"

    def get_positions(self) -> list[Position]:
        return []

    def get_account_cash(self) -> float:
        return 100_000.0

    def cancel_order(self, order_id: str) -> None:
        pass


def _make_data() -> pd.DataFrame:
    """Minimal OHLC DataFrame with SPY — enough to have a price for sizing."""
    import pandas as pd
    return pd.DataFrame({
        "symbol": ["SPY"] * 5,
        "date": pd.date_range("2026-01-01", periods=5),
        "open": [450.0] * 5,
        "high": [455.0] * 5,
        "low": [445.0] * 5,
        "close": [452.0] * 5,
        "volume": [1_000_000] * 5,
    })


# ── Portfolio construction ─────────────────────────────────────────────────────

class TestPortfolioConstruction:

    def test_empty_fund_list_raises(self):
        client = _MockExecutionClient()
        with pytest.raises(ValueError, match="at least one fund"):
            HedgeFundPortfolio(funds=[], execution_client=client)

    def test_single_fund_portfolio(self):
        fund = _AlwaysBuyFund(_make_config("F001"))
        client = _MockExecutionClient()
        portfolio = HedgeFundPortfolio(funds=[fund], execution_client=client)
        assert portfolio.get_fund("F001") is fund

    def test_get_fund_unknown_id_raises_key_error(self):
        fund = _AlwaysBuyFund(_make_config("F001"))
        portfolio = HedgeFundPortfolio(funds=[fund], execution_client=_MockExecutionClient())
        with pytest.raises(KeyError):
            portfolio.get_fund("F999")


# ── run_cycle ──────────────────────────────────────────────────────────────────

class TestRunCycle:

    def test_buy_signal_results_in_submitted_order(self):
        fund = _AlwaysBuyFund(_make_config("F001"))
        client = _MockExecutionClient()
        portfolio = HedgeFundPortfolio(funds=[fund], execution_client=client)

        result = portfolio.run_cycle(_make_data())

        assert "F001" in result
        assert len(result["F001"]) == 1
        assert result["F001"][0].side == "buy"
        assert len(client.submitted) == 1

    def test_broken_fund_does_not_block_healthy_fund(self):
        healthy = _AlwaysBuyFund(_make_config("F001"))
        broken = _BrokenFund(_make_config("F002"))
        client = _MockExecutionClient()
        portfolio = HedgeFundPortfolio(funds=[healthy, broken], execution_client=client)

        result = portfolio.run_cycle(_make_data())

        assert len(result["F001"]) == 1   # healthy fund submitted
        assert result["F002"] == []        # broken fund returns empty, not exception

    def test_cycle_count_increments(self):
        fund = _AlwaysBuyFund(_make_config("F001"))
        portfolio = HedgeFundPortfolio(funds=[fund], execution_client=_MockExecutionClient())
        portfolio.run_cycle(_make_data())
        portfolio.run_cycle(_make_data())
        assert portfolio.summary()["cycles_run"] == 2


# ── summary ────────────────────────────────────────────────────────────────────

class TestSummary:

    def test_summary_counts_paper_funds(self):
        funds = [_AlwaysBuyFund(_make_config(f"F00{i}")) for i in range(1, 4)]
        portfolio = HedgeFundPortfolio(funds=funds, execution_client=_MockExecutionClient())
        s = portfolio.summary()
        assert s["total_funds"] == 3
        assert s["paper_funds"] == 3
        assert s["live_funds"] == 0

    def test_summary_includes_per_fund_statuses(self):
        fund = _AlwaysBuyFund(_make_config("F001"))
        portfolio = HedgeFundPortfolio(funds=[fund], execution_client=_MockExecutionClient())
        s = portfolio.summary()
        assert len(s["funds"]) == 1
        assert s["funds"][0]["fund_id"] == "F001"
