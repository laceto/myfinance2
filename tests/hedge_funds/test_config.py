"""
Tests for hedge_funds.config — FundConfig validation invariants.

TDD Red → Green: these tests specify the contract; implementation must satisfy them.
"""

import pytest
from pydantic import ValidationError

from hedge_funds.config import AssetClass, FundConfig, StrategyCategory, TradingMode


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _valid_config(**overrides) -> dict:
    base = {
        "fund_id": "F001",
        "name": "Test Fund",
        "description": "A test fund.",
        "category": StrategyCategory.TREND_FOLLOWING,
        "asset_class": AssetClass.EQUITY,
        "strategy_class": "hedge_funds.strategies.trend.ma_crossover.MACrossoverFund",
        "universe": ["SPY", "QQQ"],
    }
    base.update(overrides)
    return base


# ── TradingMode ────────────────────────────────────────────────────────────────

class TestTradingMode:

    def test_paper_value(self):
        assert TradingMode.PAPER.value == "paper"

    def test_live_value(self):
        assert TradingMode.LIVE.value == "live"

    def test_default_is_paper(self):
        cfg = FundConfig(**_valid_config())
        assert cfg.mode == TradingMode.PAPER


# ── FundConfig validation ──────────────────────────────────────────────────────

class TestFundConfigValidation:

    def test_valid_config_creates_successfully(self):
        cfg = FundConfig(**_valid_config())
        assert cfg.fund_id == "F001"
        assert cfg.initial_capital == 100_000.0

    def test_fund_id_must_match_F_pattern(self):
        with pytest.raises(ValidationError, match="F###"):
            FundConfig(**_valid_config(fund_id="001"))

    def test_fund_id_plain_number_rejected(self):
        with pytest.raises(ValidationError):
            FundConfig(**_valid_config(fund_id="F000"))

    def test_fund_id_above_100_rejected(self):
        with pytest.raises(ValidationError):
            FundConfig(**_valid_config(fund_id="F101"))

    def test_fund_id_F100_is_valid(self):
        cfg = FundConfig(**_valid_config(fund_id="F100"))
        assert cfg.fund_id == "F100"

    def test_empty_universe_rejected(self):
        with pytest.raises(ValidationError):
            FundConfig(**_valid_config(universe=[]))

    def test_position_size_zero_rejected(self):
        with pytest.raises(ValidationError):
            FundConfig(**_valid_config(position_size_pct=0.0))

    def test_position_size_above_one_rejected(self):
        with pytest.raises(ValidationError):
            FundConfig(**_valid_config(position_size_pct=1.01))

    def test_position_size_exactly_one_allowed(self):
        cfg = FundConfig(**_valid_config(position_size_pct=1.0))
        assert cfg.position_size_pct == 1.0

    def test_negative_capital_rejected(self):
        with pytest.raises(ValidationError):
            FundConfig(**_valid_config(initial_capital=-1000.0))

    def test_zero_capital_rejected(self):
        with pytest.raises(ValidationError):
            FundConfig(**_valid_config(initial_capital=0.0))

    def test_strategy_class_without_dot_rejected(self):
        with pytest.raises(ValidationError):
            FundConfig(**_valid_config(strategy_class="MACrossoverFund"))

    def test_strategy_class_with_dot_accepted(self):
        cfg = FundConfig(**_valid_config(strategy_class="a.b.C"))
        assert cfg.strategy_class == "a.b.C"

    def test_params_defaults_to_empty_dict(self):
        cfg = FundConfig(**_valid_config())
        assert cfg.params == {}

    def test_custom_params_stored(self):
        cfg = FundConfig(**_valid_config(params={"fast_window": 10, "slow_window": 50}))
        assert cfg.params["fast_window"] == 10
