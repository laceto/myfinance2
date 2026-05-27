"""
hedge_funds — 100-fund algorithmic trading team on Alpaca.

Public API:
    FUND_REGISTRY          — dict[str, FundConfig] with all 100 fund definitions
    get_fund(fund_id)      — retrieve a single FundConfig
    HedgeFundPortfolio     — team orchestrator
    AlpacaExecutionClient  — paper/live Alpaca broker adapter
    BaseHedgeFund          — abstract base for all strategy implementations
    TradingMode            — PAPER | LIVE enum
"""

from .config import AssetClass, FundConfig, StrategyCategory, TradingMode
from .base import BaseExecutionClient, BaseHedgeFund
from .alpaca_client import AlpacaExecutionClient
from .portfolio import HedgeFundPortfolio
from .registry import FUND_REGISTRY, funds_by_asset_class, funds_by_category, get_fund
from .signals import Fill, Order, OrderType, Position, Signal, SignalAction

__all__ = [
    "AssetClass",
    "BaseExecutionClient",
    "BaseHedgeFund",
    "AlpacaExecutionClient",
    "Fill",
    "FundConfig",
    "FUND_REGISTRY",
    "funds_by_asset_class",
    "funds_by_category",
    "get_fund",
    "HedgeFundPortfolio",
    "Order",
    "OrderType",
    "Position",
    "Signal",
    "SignalAction",
    "StrategyCategory",
    "TradingMode",
]
