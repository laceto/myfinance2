"""
hedge_funds/config.py — Pydantic models for fund configuration.

FundConfig is the single source of truth for every hedge fund's identity,
strategy class, universe, parameters, and trading mode.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class AssetClass(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    OPTIONS = "options"


class StrategyCategory(str, Enum):
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    FACTOR = "factor"
    CRYPTO = "crypto"
    AI = "ai"
    OPTIONS = "options"


class FundConfig(BaseModel):
    """
    Complete specification for one hedge fund.

    Invariants:
      - fund_id must match pattern "F001" … "F100"
      - strategy_class must be a fully qualified dotted Python path
      - universe must be non-empty
      - position_size_pct in (0, 1]
      - initial_capital > 0
    """

    fund_id: str
    name: str
    description: str
    category: StrategyCategory
    asset_class: AssetClass
    strategy_class: str  # e.g. "hedge_funds.strategies.trend.ma_crossover.MACrossoverFund"
    universe: list[str] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    mode: TradingMode = TradingMode.PAPER
    initial_capital: float = 100_000.0
    max_positions: int = 10
    position_size_pct: float = 0.1  # fraction of capital per position

    @field_validator("fund_id")
    @classmethod
    def _validate_fund_id(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"F\d{3}", v):
            raise ValueError(f"fund_id must match 'F###', got: {v!r}")
        num = int(v[1:])
        if not 1 <= num <= 100:
            raise ValueError(f"fund_id number must be 001–100, got: {num}")
        return v

    @field_validator("position_size_pct")
    @classmethod
    def _validate_position_size(cls, v: float) -> float:
        if not 0 < v <= 1.0:
            raise ValueError(f"position_size_pct must be in (0, 1], got: {v}")
        return v

    @field_validator("initial_capital")
    @classmethod
    def _validate_capital(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"initial_capital must be > 0, got: {v}")
        return v

    @field_validator("strategy_class")
    @classmethod
    def _validate_strategy_class(cls, v: str) -> str:
        if "." not in v:
            raise ValueError(
                f"strategy_class must be a fully qualified dotted path, got: {v!r}"
            )
        return v
