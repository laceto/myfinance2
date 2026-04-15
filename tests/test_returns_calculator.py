"""
Tests for returns_calculator module.

TDD cycle: these tests are written BEFORE the implementation.
Run with: python -m pytest tests/test_returns_calculator.py -v
"""

import math
import pandas as pd
import pytest

from returns_calculator import compute_returns, compute_cumulative_returns, build_returns_dashboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ticker_df(close_values: list[float], rclose_values: list[float]) -> pd.DataFrame:
    """Build a minimal per-ticker DataFrame matching the parquet schema."""
    n = len(close_values)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=n, freq="B"),
            "symbol": ["TEST.MI"] * n,
            "close": close_values,
            "rclose": rclose_values,
        }
    ).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# A2.1 — Happy path: known values for compute_returns
# ---------------------------------------------------------------------------


class TestComputeReturns:
    """compute_returns(df, col, windows) returns correct pct returns."""

    def test_single_window_correct_value(self):
        """Window=3: return = (p[-1] - p[-4]) / p[-4]."""
        # close: [100, 110, 120, 130, 143]
        # window=3: (143 - 110) / 110 = 0.3
        df = _make_ticker_df(
            close_values=[100.0, 110.0, 120.0, 130.0, 143.0],
            rclose_values=[1.0, 1.1, 1.2, 1.3, 1.43],
        )
        result = compute_returns(df, col="close", windows=[3])
        assert math.isclose(result["ret_close_3d"], (143 - 110) / 110, rel_tol=1e-9)

    def test_multiple_windows_all_present(self):
        """All requested windows appear in the output dict."""
        n = 70  # enough rows for window=60
        close = [100.0 + i for i in range(n)]
        rclose = [1.0 + i * 0.01 for i in range(n)]
        df = _make_ticker_df(close_values=close, rclose_values=rclose)
        windows = [3, 5, 7, 10, 15, 30, 45, 60]
        result = compute_returns(df, col="close", windows=windows)
        expected_keys = {f"ret_close_{w}d" for w in windows}
        assert set(result.keys()) == expected_keys

    def test_rclose_column(self):
        """Works identically when col='rclose'."""
        df = _make_ticker_df(
            close_values=[100.0, 110.0, 120.0, 130.0, 143.0],
            rclose_values=[1.0, 1.1, 1.2, 1.3, 1.43],
        )
        result = compute_returns(df, col="rclose", windows=[3])
        # rclose: (1.43 - 1.1) / 1.1 ≈ 0.3
        assert math.isclose(result["ret_rclose_3d"], (1.43 - 1.1) / 1.1, rel_tol=1e-9)

    def test_window_1_equals_last_bar_pct_change(self):
        """Window=1 equals the last bar's simple return."""
        df = _make_ticker_df(
            close_values=[100.0, 105.0],
            rclose_values=[1.0, 1.05],
        )
        result = compute_returns(df, col="close", windows=[1])
        assert math.isclose(result["ret_close_1d"], 0.05, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# A2.2 — Edge case: window >= available rows → NaN (not crash)
# ---------------------------------------------------------------------------


class TestComputeReturnsEdgeCases:
    """Insufficient data yields NaN, not an exception."""

    def test_window_equals_row_count_returns_nan(self):
        """If window == len(df), there is no earlier price to compare to."""
        df = _make_ticker_df(
            close_values=[100.0, 110.0, 120.0],
            rclose_values=[1.0, 1.1, 1.2],
        )
        result = compute_returns(df, col="close", windows=[3])
        assert math.isnan(result["ret_close_3d"])

    def test_window_exceeds_row_count_returns_nan(self):
        """Window larger than the series → NaN."""
        df = _make_ticker_df(
            close_values=[100.0, 110.0],
            rclose_values=[1.0, 1.1],
        )
        result = compute_returns(df, col="close", windows=[5])
        assert math.isnan(result["ret_close_5d"])

    def test_empty_dataframe_returns_all_nan(self):
        """Empty ticker DataFrame → all NaN."""
        df = _make_ticker_df(close_values=[], rclose_values=[])
        result = compute_returns(df, col="close", windows=[3, 5])
        assert math.isnan(result["ret_close_3d"])
        assert math.isnan(result["ret_close_5d"])

    def test_zero_base_price_returns_nan(self):
        """Division by zero (base price == 0) → NaN, not crash."""
        df = _make_ticker_df(
            close_values=[0.0, 110.0, 120.0, 130.0],
            rclose_values=[1.0, 1.1, 1.2, 1.3],
        )
        # window=3: base is index 0, price = 0.0 → NaN
        result = compute_returns(df, col="close", windows=[3])
        assert math.isnan(result["ret_close_3d"])


# ---------------------------------------------------------------------------
# A2.3 — build_returns_dashboard: schema validation
# ---------------------------------------------------------------------------


class TestBuildReturnsDashboard:
    """build_returns_dashboard produces a correct wide-format DataFrame."""

    WINDOWS = [3, 5, 7, 10, 15, 30, 45, 60]

    def _make_data_dict(self) -> dict[str, pd.DataFrame]:
        n = 70
        return {
            "AAA.MI": _make_ticker_df(
                close_values=[100.0 + i for i in range(n)],
                rclose_values=[1.0 + i * 0.005 for i in range(n)],
            ),
            "BBB.MI": _make_ticker_df(
                close_values=[200.0 - i * 0.5 for i in range(n)],
                rclose_values=[2.0 - i * 0.003 for i in range(n)],
            ),
        }

    def test_one_row_per_ticker(self):
        data_dict = self._make_data_dict()
        df = build_returns_dashboard(data_dict, self.WINDOWS)
        assert len(df) == 2
        assert set(df["ticker"]) == {"AAA.MI", "BBB.MI"}

    def test_all_expected_columns_present(self):
        data_dict = self._make_data_dict()
        df = build_returns_dashboard(data_dict, self.WINDOWS)
        expected_cols = (
            {"ticker", "last_date", "close", "rclose"}
            | {f"ret_close_{w}d" for w in self.WINDOWS}
            | {f"ret_rclose_{w}d" for w in self.WINDOWS}
        )
        assert expected_cols.issubset(set(df.columns))

    def test_no_nan_for_sufficient_data(self):
        """70-row tickers have enough history for all windows up to 60."""
        data_dict = self._make_data_dict()
        df = build_returns_dashboard(data_dict, self.WINDOWS)
        ret_cols = [f"ret_close_{w}d" for w in self.WINDOWS] + [
            f"ret_rclose_{w}d" for w in self.WINDOWS
        ]
        assert not df[ret_cols].isnull().any().any()

    def test_last_date_is_max_date_per_ticker(self):
        data_dict = self._make_data_dict()
        df = build_returns_dashboard(data_dict, self.WINDOWS)
        for _, row in df.iterrows():
            ticker = row["ticker"]
            expected_last = data_dict[ticker]["date"].max()
            assert pd.to_datetime(row["last_date"]) == pd.to_datetime(expected_last)

    def test_close_price_is_last_bar(self):
        data_dict = self._make_data_dict()
        df = build_returns_dashboard(data_dict, self.WINDOWS)
        for _, row in df.iterrows():
            ticker = row["ticker"]
            expected_close = data_dict[ticker].sort_values("date")["close"].iloc[-1]
            assert math.isclose(row["close"], expected_close, rel_tol=1e-9)

    def test_cumret_columns_present(self):
        """build_returns_dashboard includes cumret_* columns alongside ret_* columns."""
        data_dict = self._make_data_dict()
        df = build_returns_dashboard(data_dict, self.WINDOWS)
        expected_cumret_cols = (
            {f"cumret_close_{w}d" for w in self.WINDOWS}
            | {f"cumret_rclose_{w}d" for w in self.WINDOWS}
        )
        assert expected_cumret_cols.issubset(set(df.columns))

    def test_cumret_no_nan_for_sufficient_data(self):
        """70-row tickers → no NaN in any cumret column."""
        data_dict = self._make_data_dict()
        df = build_returns_dashboard(data_dict, self.WINDOWS)
        cumret_cols = [f"cumret_close_{w}d" for w in self.WINDOWS] + [
            f"cumret_rclose_{w}d" for w in self.WINDOWS
        ]
        assert not df[cumret_cols].isnull().any().any()


# ---------------------------------------------------------------------------
# compute_cumulative_returns — unit tests
# ---------------------------------------------------------------------------


class TestComputeCumulativeReturns:
    """compute_cumulative_returns uses log → sum → exp (not price ratio)."""

    def test_correct_compounded_value_not_naive_sum(self):
        """
        Three consecutive +1% days: compounded = 1.01^3 - 1 = 3.0301%,
        NOT 3.0% (naive arithmetic sum of daily returns).
        This test proves the log→sum→exp path is being used.
        """
        close = [100.0, 101.0, 102.01, 103.0301]  # exact 1% each day
        df = _make_ticker_df(close_values=close, rclose_values=[1.0] * 4)
        result = compute_cumulative_returns(df, col="close", windows=[3])
        expected = 1.01 ** 3 - 1  # 0.030301
        assert math.isclose(result["cumret_close_3d"], expected, rel_tol=1e-6)
        # Must NOT equal the naive sum (0.03)
        assert not math.isclose(result["cumret_close_3d"], 0.03, rel_tol=1e-6)

    def test_single_window_telescopes_to_price_ratio(self):
        """
        sum(log(P_i/P_{i-1})) = log(P_last / P_{last-N}).
        exp of that - 1 == (P_last / P_{last-N}) - 1 == ret_close_Nd.
        """
        close = [100.0, 110.0, 120.0, 130.0, 143.0]
        df = _make_ticker_df(close_values=close, rclose_values=[1.0] * 5)
        cumret = compute_cumulative_returns(df, col="close", windows=[3])
        simple_ret = compute_returns(df, col="close", windows=[3])
        assert math.isclose(
            cumret["cumret_close_3d"], simple_ret["ret_close_3d"], rel_tol=1e-9
        )

    def test_all_windows_keys_present(self):
        """Output dict contains exactly one key per window."""
        n = 70
        df = _make_ticker_df(
            close_values=[100.0 + i for i in range(n)],
            rclose_values=[1.0 + i * 0.01 for i in range(n)],
        )
        windows = [3, 5, 7, 10, 15, 30, 45, 60]
        result = compute_cumulative_returns(df, col="close", windows=windows)
        assert set(result.keys()) == {f"cumret_close_{w}d" for w in windows}

    def test_rclose_prefix_in_keys(self):
        """col='rclose' produces 'cumret_rclose_*' keys."""
        df = _make_ticker_df(
            close_values=[1.0, 1.01, 1.0201, 1.030301],
            rclose_values=[1.0, 1.01, 1.0201, 1.030301],
        )
        result = compute_cumulative_returns(df, col="rclose", windows=[3])
        assert "cumret_rclose_3d" in result

    def test_insufficient_history_returns_nan(self):
        """Window > available bars → NaN."""
        df = _make_ticker_df(close_values=[100.0, 110.0], rclose_values=[1.0, 1.1])
        result = compute_cumulative_returns(df, col="close", windows=[5])
        assert math.isnan(result["cumret_close_5d"])

    def test_empty_dataframe_returns_nan(self):
        """Empty DataFrame → all NaN."""
        df = _make_ticker_df(close_values=[], rclose_values=[])
        result = compute_cumulative_returns(df, col="close", windows=[3])
        assert math.isnan(result["cumret_close_3d"])

    def test_zero_price_in_window_returns_nan(self):
        """log(0) is -inf; result must be NaN, not crash."""
        # P[0]=0 means log(P[1]/P[0]) = log(inf) = inf
        close = [0.0, 110.0, 120.0, 130.0]
        df = _make_ticker_df(close_values=close, rclose_values=[1.0] * 4)
        result = compute_cumulative_returns(df, col="close", windows=[3])
        assert math.isnan(result["cumret_close_3d"])
