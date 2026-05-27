"""
Tests for MACrossoverFund — signal logic for golden/death cross.

TDD invariants:
  - Golden cross (fast crosses above slow) → BUY signal
  - Death cross  (fast crosses below slow) → SELL signal
  - No crossover                           → no signal
  - Insufficient data                      → no signal (guard clause)
  - EMA behaves differently than SMA (different crossover timing)
  - Invalid fast >= slow raises ValueError at construction
  - confidence is in [0, 1]
"""

from __future__ import annotations

import pandas as pd
import pytest

from hedge_funds.config import AssetClass, FundConfig, StrategyCategory
from hedge_funds.signals import SignalAction
from hedge_funds.strategies.trend.ma_crossover import MACrossoverFund, _compute_ma


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_config(fast: int = 5, slow: int = 10, ma_type: str = "sma") -> FundConfig:
    return FundConfig(
        fund_id="F001",
        name="Test MA Fund",
        description="unit test",
        category=StrategyCategory.TREND_FOLLOWING,
        asset_class=AssetClass.EQUITY,
        strategy_class="hedge_funds.strategies.trend.ma_crossover.MACrossoverFund",
        universe=["SPY"],
        params={"fast_window": fast, "slow_window": slow, "ma_type": ma_type},
    )


def _make_prices_with_golden_cross(n: int = 15) -> pd.DataFrame:
    """
    Build a price series where the fast SMA crosses above the slow SMA on the last bar.
    Prices trend down for the first half (fast < slow), then spike up.
    """
    close = [100.0] * (n - 1) + [200.0]  # spike on last bar forces golden cross
    return pd.DataFrame({
        "symbol": ["SPY"] * n,
        "date": pd.date_range("2026-01-01", periods=n),
        "open": close,
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
        "volume": [1_000_000] * n,
    })


def _make_prices_with_death_cross(n: int = 15) -> pd.DataFrame:
    """
    Build a series where fast SMA crosses below slow SMA on the last bar.
    Prices trend up then crash.
    """
    close = [100.0] * (n - 1) + [1.0]  # crash on last bar forces death cross
    return pd.DataFrame({
        "symbol": ["SPY"] * n,
        "date": pd.date_range("2026-01-01", periods=n),
        "open": close,
        "high": [c + 1 for c in close],
        "low": [max(c - 1, 0.0) for c in close],
        "close": close,
        "volume": [1_000_000] * n,
    })


def _make_flat_prices(n: int = 15, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["SPY"] * n,
        "date": pd.date_range("2026-01-01", periods=n),
        "open": [price] * n,
        "high": [price + 1] * n,
        "low": [price - 1] * n,
        "close": [price] * n,
        "volume": [1_000_000] * n,
    })


# ── _compute_ma ────────────────────────────────────────────────────────────────

class TestComputeMA:

    def test_sma_last_value_equals_rolling_mean(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_ma(series, window=3, ma_type="sma")
        assert abs(result.iloc[-1] - 4.0) < 1e-9

    def test_ema_returns_series_same_length(self):
        series = pd.Series(range(1, 11), dtype=float)
        result = _compute_ma(series, window=3, ma_type="ema")
        assert len(result) == 10

    def test_unknown_ma_type_raises(self):
        with pytest.raises(ValueError, match="Unknown ma_type"):
            _compute_ma(pd.Series([1.0, 2.0]), window=1, ma_type="wma")


# ── MACrossoverFund construction ───────────────────────────────────────────────

class TestConstruction:

    def test_valid_config_creates_fund(self):
        fund = MACrossoverFund(_make_config(5, 10))
        assert fund.fund_id == "F001"

    def test_fast_ge_slow_raises(self):
        with pytest.raises(ValueError, match="fast_window"):
            MACrossoverFund(_make_config(fast=10, slow=10))

    def test_fast_gt_slow_raises(self):
        with pytest.raises(ValueError, match="fast_window"):
            MACrossoverFund(_make_config(fast=20, slow=10))

    def test_invalid_ma_type_raises(self):
        with pytest.raises(ValueError, match="ma_type"):
            MACrossoverFund(_make_config(ma_type="wma"))


# ── compute_signals ────────────────────────────────────────────────────────────

class TestComputeSignals:

    def test_golden_cross_emits_buy_signal(self):
        fund = MACrossoverFund(_make_config(fast=5, slow=10))
        data = _make_prices_with_golden_cross(n=15)
        signals = fund.compute_signals(data)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.BUY
        assert signals[0].symbol == "SPY"
        assert signals[0].fund_id == "F001"

    def test_death_cross_emits_sell_signal(self):
        fund = MACrossoverFund(_make_config(fast=5, slow=10))
        data = _make_prices_with_death_cross(n=15)
        signals = fund.compute_signals(data)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.SELL

    def test_flat_prices_no_crossover_no_signal(self):
        fund = MACrossoverFund(_make_config(fast=5, slow=10))
        data = _make_flat_prices(n=15)
        signals = fund.compute_signals(data)
        assert signals == []

    def test_insufficient_data_no_signal(self):
        fund = MACrossoverFund(_make_config(fast=5, slow=10))
        data = _make_flat_prices(n=5)  # need 11 bars minimum
        signals = fund.compute_signals(data)
        assert signals == []

    def test_signal_confidence_in_unit_interval(self):
        fund = MACrossoverFund(_make_config(fast=5, slow=10))
        data = _make_prices_with_golden_cross(n=15)
        signals = fund.compute_signals(data)
        for s in signals:
            assert 0.0 <= s.confidence <= 1.0

    def test_symbol_not_in_universe_ignored(self):
        fund = MACrossoverFund(_make_config(fast=5, slow=10))
        data = _make_prices_with_golden_cross(n=15)
        data["symbol"] = "AAPL"  # not in universe ["SPY"]
        signals = fund.compute_signals(data)
        assert signals == []

    def test_ema_crossover_detected(self):
        fund = MACrossoverFund(_make_config(fast=3, slow=8, ma_type="ema"))
        data = _make_prices_with_golden_cross(n=15)
        signals = fund.compute_signals(data)
        # EMA should also detect the golden cross (spike is large enough)
        assert any(s.action == SignalAction.BUY for s in signals)


# ── on_bar ─────────────────────────────────────────────────────────────────────

class TestOnBar:

    def test_on_bar_returns_none_before_enough_history(self):
        fund = MACrossoverFund(_make_config(fast=5, slow=10))
        bar = pd.Series({
            "symbol": "SPY", "date": "2026-01-01",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000,
        })
        result = fund.on_bar(bar)
        assert result is None

    def test_on_bar_ignores_unknown_symbol(self):
        fund = MACrossoverFund(_make_config(fast=5, slow=10))
        bar = pd.Series({
            "symbol": "UNKNOWN", "date": "2026-01-01",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000,
        })
        result = fund.on_bar(bar)
        assert result is None

    def test_on_bar_order_has_correct_fund_id(self):
        """Feed enough bars to trigger a golden cross via on_bar."""
        fund = MACrossoverFund(_make_config(fast=5, slow=10))
        prices = _make_prices_with_golden_cross(n=15)

        last_order = None
        for _, row in prices.iterrows():
            last_order = fund.on_bar(row)

        # The last bar is a spike — should trigger a golden cross order
        assert last_order is not None
        assert last_order.fund_id == "F001"
        assert last_order.side == "buy"
        assert last_order.quantity > 0
