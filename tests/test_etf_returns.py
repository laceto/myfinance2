"""
test_etf_returns.py — Unit tests for etf_returns (multi-timeframe returns).

etf_returns computes each symbol's total return over several lookback windows
from the long/tidy OHLC parquet, and ranks the top out-performers per window.
Returns use close-on-or-before the target dates (no lookahead) so a symbol
without enough history yields NaN for that window and is excluded from ranking.

Coverage
--------
latest_close_asof:
  - picks the last close on or before the as-of date
period_return:
  - return = latest_close / base_close - 1
  - NaN when the symbol has no bar on/before the base date (insufficient history)
compute_returns:
  - one column per requested window (+ YTD)
top_outperformers:
  - orders by descending window return, drops NaN, respects the top-n limit
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_returns import (
    bottom_underperformers,
    compute_returns,
    latest_close_asof,
    period_return,
    top_outperformers,
)


def _ohlc(rows):
    """rows: list of (symbol, 'YYYY-MM-DD', close) -> long/tidy OHLC frame."""
    return pd.DataFrame(
        [
            {"symbol": s, "date": pd.Timestamp(d), "open": c, "high": c,
             "low": c, "close": float(c), "volume": 1}
            for s, d, c in rows
        ]
    )


class TestLatestCloseAsof:

    def test_picks_last_close_on_or_before(self):
        ohlc = _ohlc([("A.L", "2026-01-01", 100), ("A.L", "2026-02-01", 110),
                      ("A.L", "2026-03-01", 121)])

        s = latest_close_asof(ohlc, pd.Timestamp("2026-02-15"))

        assert s["A.L"] == 110  # 2026-02-01 is the last bar on/before 02-15


class TestPeriodReturn:

    def test_return_is_ratio_minus_one(self):
        ohlc = _ohlc([("A.L", "2026-01-01", 100), ("A.L", "2026-02-01", 110),
                      ("A.L", "2026-03-01", 121)])

        r = period_return(ohlc, as_of=pd.Timestamp("2026-03-01"),
                          base_date=pd.Timestamp("2026-02-01"))

        assert abs(r["A.L"] - 0.10) < 1e-9   # 121/110 - 1

    def test_base_uses_last_close_on_or_before_base_date(self):
        ohlc = _ohlc([("A.L", "2026-01-01", 100), ("A.L", "2026-02-01", 110),
                      ("A.L", "2026-03-01", 121)])

        r = period_return(ohlc, as_of=pd.Timestamp("2026-03-01"),
                          base_date=pd.Timestamp("2026-01-15"))

        assert abs(r["A.L"] - 0.21) < 1e-9   # 121/100 - 1

    def test_nan_when_no_history_before_base_date(self):
        ohlc = _ohlc([("A.L", "2026-02-01", 110), ("A.L", "2026-03-01", 121)])

        r = period_return(ohlc, as_of=pd.Timestamp("2026-03-01"),
                          base_date=pd.Timestamp("2026-01-01"))

        assert np.isnan(r["A.L"])   # no bar on/before 2026-01-01


class TestComputeReturns:

    def test_has_a_column_per_window_plus_ytd(self):
        ohlc = _ohlc([("A.L", "2024-12-31", 100), ("A.L", "2026-07-13", 130)])

        out = compute_returns(ohlc, as_of=pd.Timestamp("2026-07-13"),
                              windows_days={"1M": 30, "1Y": 365}, include_ytd=True)

        assert set(["1M", "1Y", "YTD"]).issubset(out.columns)
        assert "A.L" in out.index


class TestTopOutperformers:

    def test_orders_descending_drops_nan_and_limits_n(self):
        returns = pd.DataFrame(
            {"1M": [0.05, 0.20, np.nan, 0.12]},
            index=["A.L", "B.L", "C.L", "D.L"],
        )

        top = top_outperformers(returns, "1M", n=2)

        assert top["symbol"].tolist() == ["B.L", "D.L"]   # 0.20, 0.12
        assert "C.L" not in top["symbol"].tolist()        # NaN dropped


class TestBottomUnderperformers:

    def test_orders_ascending_drops_nan_and_limits_n(self):
        returns = pd.DataFrame(
            {"1W": [0.05, -0.20, np.nan, -0.12]},
            index=["A.L", "B.L", "C.L", "D.L"],
        )

        bottom = bottom_underperformers(returns, "1W", n=2)

        assert bottom["symbol"].tolist() == ["B.L", "D.L"]   # -0.20, -0.12 (worst first)
        assert "C.L" not in bottom["symbol"].tolist()        # NaN dropped
