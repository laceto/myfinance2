"""
etf_liquidity.py — Illiquidity screen for the ETF universe.

Thinly-traded ETFs (obscure crypto ETPs, tiny listings) produce stale/erratic
prints whose "returns" dominate rankings without being investable. This module
scores each symbol's recent liquidity so the return analysis can drop the
illiquid tail before ranking.

Liquidity measures (over a recent lookback)
------------------------------------------
- ``median_traded_value`` = median of ``close * volume`` per day — a robust
  proxy for daily turnover in the listing currency (median, not mean, so a
  single spike does not rescue an otherwise dead symbol).
- ``active_frac`` = share of days with ``volume > 0`` — catches symbols that
  simply do not trade most days, independent of currency.

A symbol is "liquid" when it clears BOTH floors. Note ``median_traded_value``
mixes listing currencies (EUR/USD/GBP/CHF); they are within ~1x, so the floor
is a coarse screen, not an exact currency threshold.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_MIN_TRADED_VALUE = 50_000.0
DEFAULT_MIN_ACTIVE_FRAC = 0.80


def liquidity_stats(
    ohlc: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Per-symbol liquidity over the ``lookback_days`` ending at ``as_of``.

    Returns a frame indexed by symbol with ``median_traded_value``,
    ``active_frac`` and ``n_days``. ``as_of`` defaults to the latest date.
    """
    ohlc = ohlc.copy()
    ohlc["date"] = pd.to_datetime(ohlc["date"])
    if as_of is None:
        as_of = ohlc["date"].max()
    as_of = pd.Timestamp(as_of)

    window = ohlc[(ohlc["date"] > as_of - pd.Timedelta(days=lookback_days))
                  & (ohlc["date"] <= as_of)].copy()
    window["traded_value"] = window["close"] * window["volume"]

    grouped = window.groupby("symbol")
    stats = pd.DataFrame({
        "median_traded_value": grouped["traded_value"].median(),
        "active_frac": grouped["volume"].apply(lambda v: float((v > 0).mean())),
        "n_days": grouped.size().astype(int),
    })
    return stats


def liquid_symbols(
    stats: pd.DataFrame,
    min_traded_value: float = DEFAULT_MIN_TRADED_VALUE,
    min_active_frac: float = DEFAULT_MIN_ACTIVE_FRAC,
) -> List[str]:
    """Symbols clearing BOTH the traded-value and active-fraction floors."""
    passing = stats[
        (stats["median_traded_value"] >= min_traded_value)
        & (stats["active_frac"] >= min_active_frac)
    ]
    return passing.index.tolist()
