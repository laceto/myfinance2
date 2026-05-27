"""
Tests for hedge_funds.registry — invariants over all 100 fund definitions.

Key invariants checked:
  - Exactly 100 funds exist
  - All fund_ids are unique and match their key
  - All fund_ids are in range F001–F100
  - strategy_class is always a dotted path
  - universe is always non-empty
  - Category and asset-class groupings have expected sizes
"""

import pytest

from hedge_funds.registry import (
    FUND_REGISTRY,
    funds_by_asset_class,
    funds_by_category,
    get_fund,
)
from hedge_funds.config import FundConfig


class TestRegistryCompleteness:

    def test_exactly_100_funds(self):
        assert len(FUND_REGISTRY) == 100

    def test_all_keys_are_unique(self):
        keys = list(FUND_REGISTRY.keys())
        assert len(keys) == len(set(keys))

    def test_all_keys_match_fund_id(self):
        for key, cfg in FUND_REGISTRY.items():
            assert cfg.fund_id == key, f"Key {key!r} does not match fund_id {cfg.fund_id!r}"

    def test_all_fund_ids_in_F001_to_F100_range(self):
        for fund_id in FUND_REGISTRY:
            num = int(fund_id[1:])
            assert 1 <= num <= 100, f"fund_id {fund_id!r} out of range"

    def test_no_gaps_in_fund_ids(self):
        numbers = sorted(int(fid[1:]) for fid in FUND_REGISTRY)
        assert numbers == list(range(1, 101)), "Fund ID sequence has gaps or duplicates"

    def test_all_configs_are_fundconfig_instances(self):
        for key, cfg in FUND_REGISTRY.items():
            assert isinstance(cfg, FundConfig), f"{key} is not a FundConfig"


class TestRegistryStructure:

    def test_all_strategy_classes_are_dotted_paths(self):
        for fid, cfg in FUND_REGISTRY.items():
            assert "." in cfg.strategy_class, (
                f"{fid}: strategy_class {cfg.strategy_class!r} must be a dotted path"
            )

    def test_all_universes_are_non_empty(self):
        for fid, cfg in FUND_REGISTRY.items():
            assert len(cfg.universe) >= 1, f"{fid}: universe is empty"

    def test_all_names_are_non_empty(self):
        for fid, cfg in FUND_REGISTRY.items():
            assert cfg.name.strip(), f"{fid}: name is blank"

    def test_all_descriptions_are_non_empty(self):
        for fid, cfg in FUND_REGISTRY.items():
            assert cfg.description.strip(), f"{fid}: description is blank"

    def test_all_funds_default_to_paper_mode(self):
        from hedge_funds.config import TradingMode
        for fid, cfg in FUND_REGISTRY.items():
            assert cfg.mode == TradingMode.PAPER, f"{fid}: not in PAPER mode"


class TestCategoryGroupings:

    def test_trend_following_funds_count(self):
        funds = funds_by_category("trend_following")
        assert len(funds) == 20, f"Expected 20 trend funds, got {len(funds)}"

    def test_breakout_funds_count(self):
        funds = funds_by_category("breakout")
        assert len(funds) == 15, f"Expected 15 breakout funds, got {len(funds)}"

    def test_mean_reversion_funds_count(self):
        funds = funds_by_category("mean_reversion")
        assert len(funds) == 15, f"Expected 15 MR funds, got {len(funds)}"

    def test_momentum_funds_count(self):
        funds = funds_by_category("momentum")
        assert len(funds) == 10, f"Expected 10 momentum funds, got {len(funds)}"

    def test_volatility_funds_count(self):
        funds = funds_by_category("volatility")
        assert len(funds) == 10, f"Expected 10 vol funds, got {len(funds)}"

    def test_factor_funds_count(self):
        funds = funds_by_category("factor")
        assert len(funds) == 10, f"Expected 10 factor funds, got {len(funds)}"

    def test_crypto_funds_count(self):
        funds = funds_by_category("crypto")
        assert len(funds) == 5, f"Expected 5 crypto funds, got {len(funds)}"

    def test_ai_funds_count(self):
        funds = funds_by_category("ai")
        assert len(funds) == 10, f"Expected 10 AI funds, got {len(funds)}"

    def test_options_funds_count(self):
        funds = funds_by_category("options")
        assert len(funds) == 5, f"Expected 5 options funds, got {len(funds)}"

    def test_total_category_sums_to_100(self):
        categories = [
            "trend_following", "breakout", "mean_reversion", "momentum",
            "volatility", "factor", "crypto", "ai", "options",
        ]
        total = sum(len(funds_by_category(c)) for c in categories)
        assert total == 100


class TestGetFund:

    def test_get_fund_by_id(self):
        cfg = get_fund("F001")
        assert cfg.fund_id == "F001"

    def test_get_fund_invalid_raises_key_error(self):
        with pytest.raises(KeyError):
            get_fund("F999")

    def test_funds_by_asset_class_equity(self):
        equity_funds = funds_by_asset_class("equity")
        assert len(equity_funds) > 0

    def test_funds_by_asset_class_crypto(self):
        crypto_funds = funds_by_asset_class("crypto")
        assert len(crypto_funds) == 5
