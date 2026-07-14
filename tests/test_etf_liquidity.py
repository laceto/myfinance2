"""
test_etf_liquidity.py — Unit tests for etf_liquidity (illiquidity screen).

Thinly-traded ETFs (obscure crypto ETPs, tiny listings) produce unreliable
prints that dominate return rankings. This screen keeps only symbols that trade
enough: median daily traded value (close x volume) over a recent lookback, and
the fraction of days they actually traded (volume > 0).

Coverage
--------
liquidity_stats:
  - median traded value and active-day fraction per symbol
  - lookback restricts to the recent window
liquid_symbols:
  - keeps only symbols passing BOTH the traded-value and active-fraction floors
"""

from __future__ import annotations

import pandas as pd

from etf_liquidity import liquid_symbols, liquidity_stats


def _ohlc(rows):
    """rows: (symbol, 'YYYY-MM-DD', close, volume)."""
    return pd.DataFrame(
        [{"symbol": s, "date": pd.Timestamp(d), "open": c, "high": c,
          "low": c, "close": float(c), "volume": v} for s, d, c, v in rows]
    )


class TestLiquidityStats:

    def test_median_traded_value_and_active_fraction(self):
        ohlc = _ohlc([
            ("A.L", "2026-07-06", 10, 100),   # tv 1000
            ("A.L", "2026-07-07", 10, 200),   # tv 2000
            ("A.L", "2026-07-08", 10, 300),   # tv 3000
            ("B.L", "2026-07-06", 10, 0),     # tv 0
            ("B.L", "2026-07-07", 10, 0),     # tv 0
            ("B.L", "2026-07-08", 10, 50),    # tv 500
        ])

        stats = liquidity_stats(ohlc, as_of=pd.Timestamp("2026-07-08"), lookback_days=30)

        assert stats.loc["A.L", "median_traded_value"] == 2000
        assert stats.loc["A.L", "active_frac"] == 1.0
        assert stats.loc["B.L", "median_traded_value"] == 0
        assert abs(stats.loc["B.L", "active_frac"] - 1/3) < 1e-9

    def test_lookback_restricts_to_recent_window(self):
        ohlc = _ohlc([
            ("A.L", "2026-01-01", 10, 9999),   # old, outside a 30-day lookback
            ("A.L", "2026-07-07", 10, 100),
            ("A.L", "2026-07-08", 10, 100),
        ])

        stats = liquidity_stats(ohlc, as_of=pd.Timestamp("2026-07-08"), lookback_days=30)

        assert stats.loc["A.L", "median_traded_value"] == 1000   # only recent bars


class TestLiquidSymbols:

    def test_keeps_only_symbols_passing_both_floors(self):
        stats = pd.DataFrame({
            "median_traded_value": [2000.0, 0.0, 60000.0, 40000.0],
            "active_frac": [1.0, 0.33, 0.5, 0.9],
        }, index=["A.L", "B.L", "C.L", "D.L"])

        keep = liquid_symbols(stats, min_traded_value=1000, min_active_frac=0.8)

        # A passes (2000 & 1.0). B fails value+active. C fails active (0.5).
        # D fails value (40000 < 50000-equivalent? here min=1000 so D passes both).
        assert set(keep) == {"A.L", "D.L"}
