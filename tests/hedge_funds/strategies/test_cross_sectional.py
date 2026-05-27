"""
Tests for CrossSectionalMomentumFund — momentum ranking and signal generation.

Key invariants:
  - Top-N symbols by 12-1 month return receive BUY signals
  - Bottom-N receive SELL signals when long_only=False
  - Insufficient data per symbol is gracefully skipped
  - formation_months must be > skip_months (guard clause)
  - Confidence is non-negative and in [0, 1]
"""

from __future__ import annotations

import pandas as pd
import pytest

from hedge_funds.config import AssetClass, FundConfig, StrategyCategory
from hedge_funds.signals import SignalAction
from hedge_funds.strategies.momentum.cross_sectional import CrossSectionalMomentumFund


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_config(
    universe: list,
    formation: int = 20,
    skip: int = 5,
    top_n: int = 2,
    bottom_n: int = 0,
    long_only: bool = True,
) -> FundConfig:
    return FundConfig(
        fund_id="F051",
        name="Test Momentum Fund",
        description="unit test",
        category=StrategyCategory.MOMENTUM,
        asset_class=AssetClass.EQUITY,
        strategy_class="hedge_funds.strategies.momentum.cross_sectional.CrossSectionalMomentumFund",
        universe=universe,
        params={
            "formation_months": formation,
            "skip_months": skip,
            "top_n": top_n,
            "bottom_n": bottom_n,
            "long_only": long_only,
        },
    )


def _make_ranked_data(symbols_prices: dict[str, float], n_bars: int = 30) -> pd.DataFrame:
    """
    Build a DataFrame where each symbol has a constant price equal to the
    given value — except the last `skip` bars where it stays at the same
    level. This makes momentum score = 0 for all but the winner.

    To give symbol A a positive score, set its price to 110 (10% > 100).
    """
    rows = []
    for symbol, final_price in symbols_prices.items():
        # First (n_bars - 5) bars at 100 — then last 5 at final_price
        prices = [100.0] * (n_bars - 5) + [final_price] * 5
        for i, p in enumerate(prices):
            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                "open": p, "high": p + 1, "low": p - 1, "close": p, "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


# ── Construction ───────────────────────────────────────────────────────────────

class TestConstruction:

    def test_valid_config_creates_fund(self):
        fund = CrossSectionalMomentumFund(_make_config(["SPY", "QQQ"]))
        assert fund.fund_id == "F051"

    def test_formation_le_skip_raises(self):
        with pytest.raises(ValueError, match="formation_months"):
            CrossSectionalMomentumFund(_make_config(["SPY"], formation=5, skip=5))

    def test_formation_lt_skip_raises(self):
        with pytest.raises(ValueError, match="formation_months"):
            CrossSectionalMomentumFund(_make_config(["SPY"], formation=3, skip=5))


# ── compute_signals ────────────────────────────────────────────────────────────

class TestComputeSignals:

    def test_top_winner_gets_buy_signal(self):
        """Symbol with highest return in formation window should be in BUY signals."""
        fund = CrossSectionalMomentumFund(
            _make_config(["AAPL", "MSFT", "GOOG"], formation=20, skip=5, top_n=1)
        )
        # AAPL jumps most, MSFT less, GOOG stays flat
        data = _make_ranked_data({"AAPL": 150.0, "MSFT": 120.0, "GOOG": 100.0}, n_bars=30)
        signals = fund.compute_signals(data)

        buy_symbols = [s.symbol for s in signals if s.action == SignalAction.BUY]
        assert "AAPL" in buy_symbols

    def test_top_n_buy_count(self):
        fund = CrossSectionalMomentumFund(
            _make_config(["AAPL", "MSFT", "GOOG", "AMZN"], formation=20, skip=5, top_n=2)
        )
        data = _make_ranked_data(
            {"AAPL": 150.0, "MSFT": 120.0, "GOOG": 110.0, "AMZN": 100.0},
            n_bars=30,
        )
        signals = fund.compute_signals(data)
        buy_signals = [s for s in signals if s.action == SignalAction.BUY]
        assert len(buy_signals) == 2

    def test_long_only_no_sell_signals(self):
        fund = CrossSectionalMomentumFund(
            _make_config(["AAPL", "MSFT"], formation=20, skip=5, top_n=1, bottom_n=1, long_only=True)
        )
        data = _make_ranked_data({"AAPL": 130.0, "MSFT": 90.0}, n_bars=30)
        signals = fund.compute_signals(data)
        sell_signals = [s for s in signals if s.action == SignalAction.SELL]
        assert sell_signals == []

    def test_not_long_only_emits_sell_for_bottom(self):
        fund = CrossSectionalMomentumFund(
            _make_config(
                ["AAPL", "MSFT", "GOOG"],
                formation=20, skip=5, top_n=1, bottom_n=1, long_only=False,
            )
        )
        data = _make_ranked_data({"AAPL": 150.0, "MSFT": 100.0, "GOOG": 70.0}, n_bars=30)
        signals = fund.compute_signals(data)
        sell_signals = [s for s in signals if s.action == SignalAction.SELL]
        assert len(sell_signals) >= 1

    def test_insufficient_data_returns_no_signals(self):
        fund = CrossSectionalMomentumFund(
            _make_config(["SPY"], formation=100, skip=10, top_n=1)
        )
        # Only 5 bars — far less than formation+skip
        data = _make_ranked_data({"SPY": 100.0}, n_bars=5)
        signals = fund.compute_signals(data)
        assert signals == []

    def test_confidence_in_unit_interval(self):
        fund = CrossSectionalMomentumFund(
            _make_config(["AAPL", "MSFT"], formation=20, skip=5, top_n=1)
        )
        data = _make_ranked_data({"AAPL": 200.0, "MSFT": 100.0}, n_bars=30)
        signals = fund.compute_signals(data)
        for s in signals:
            assert 0.0 <= s.confidence <= 1.0

    def test_metadata_contains_rank_and_score(self):
        fund = CrossSectionalMomentumFund(
            _make_config(["AAPL", "MSFT"], formation=20, skip=5, top_n=1)
        )
        data = _make_ranked_data({"AAPL": 130.0, "MSFT": 100.0}, n_bars=30)
        signals = fund.compute_signals(data)
        buy_signals = [s for s in signals if s.action == SignalAction.BUY]
        assert buy_signals[0].metadata["rank"] == 1
        assert "momentum_score" in buy_signals[0].metadata
