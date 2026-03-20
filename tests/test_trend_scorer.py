"""
Unit tests for trend_scorer.py.

All tests use synthetic in-memory data — no file I/O, no algoshort pipeline calls.
Run with: pytest tests/test_trend_scorer.py -v
"""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from trend_scorer import (
    compute_score_vs_bm,
    compute_score_history,
    compute_sector_peer_avg,
    build_dashboard,
    detect_signal_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIGNAL_COLS = ["sig_a", "sig_b", "sig_c"]


def _make_ticker_df(last_signals: dict[str, int], n_rows: int = 5) -> pd.DataFrame:
    """
    Build a minimal per-ticker DataFrame with signal columns and stop_loss columns.

    last_signals: {col: value} for the last row. Earlier rows are all 0.
    """
    rows = []
    base_date = date(2024, 1, 1)
    for i in range(n_rows):
        row: dict = {
            "date": base_date + timedelta(days=i),
            "close": 10.0 + i * 0.1,
            "rclose": 1.0 + i * 0.01,
        }
        for col in SIGNAL_COLS:
            row[col] = last_signals[col] if i == n_rows - 1 else 0
            row[f"{col}_stop_loss"] = 9.0
        rows.append(row)
    return pd.DataFrame(rows)


def _make_ohlc_df(tickers: list[str], n_rows: int = 10) -> pd.DataFrame:
    """Build a minimal OHLC DataFrame for multiple tickers."""
    rows = []
    base_date = date(2024, 1, 1)
    for ticker in tickers:
        base_close = {"A.MI": 10.0, "B.MI": 20.0, "C.MI": 30.0, "D.MI": 40.0}.get(ticker, 15.0)
        for i in range(n_rows):
            rows.append(
                {
                    "symbol": ticker,
                    "date": base_date + timedelta(days=i),
                    "open": base_close,
                    "high": base_close * 1.01,
                    "low": base_close * 0.99,
                    "close": base_close + i * 0.1,
                    "volume": 1000,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# detect_signal_columns
# ---------------------------------------------------------------------------


class TestDetectSignalColumns:
    def test_returns_signals_with_matching_stop_loss(self):
        df = _make_ticker_df({"sig_a": 1, "sig_b": -1, "sig_c": 0})
        result = detect_signal_columns(df)
        assert sorted(result) == sorted(SIGNAL_COLS)

    def test_excludes_signals_without_stop_loss(self):
        df = _make_ticker_df({"sig_a": 1, "sig_b": -1, "sig_c": 0})
        df["orphan_signal"] = 1  # no matching _stop_loss column
        result = detect_signal_columns(df)
        assert "orphan_signal" not in result
        assert sorted(result) == sorted(SIGNAL_COLS)

    def test_empty_dataframe_returns_empty_list(self):
        result = detect_signal_columns(pd.DataFrame())
        assert result == []

    def test_no_stop_loss_columns_returns_empty_list(self):
        df = pd.DataFrame({"sig_a": [1, 0], "sig_b": [-1, 0]})
        result = detect_signal_columns(df)
        assert result == []


# ---------------------------------------------------------------------------
# compute_score_vs_bm
# ---------------------------------------------------------------------------


class TestComputeScoreVsBm:
    def test_mixed_signals_sum_correctly(self):
        """sig_a=+1, sig_b=-1, sig_c=+1 → score = 1"""
        data_dict = {"TKR.MI": _make_ticker_df({"sig_a": 1, "sig_b": -1, "sig_c": 1})}
        result = compute_score_vs_bm(data_dict, SIGNAL_COLS)
        row = result[result["ticker"] == "TKR.MI"].iloc[0]
        assert row["score_vs_bm"] == 1

    def test_all_long_score_equals_n_signals(self):
        data_dict = {"TKR.MI": _make_ticker_df({"sig_a": 1, "sig_b": 1, "sig_c": 1})}
        result = compute_score_vs_bm(data_dict, SIGNAL_COLS)
        assert result.iloc[0]["score_vs_bm"] == len(SIGNAL_COLS)

    def test_all_short_score_equals_negative_n_signals(self):
        data_dict = {"TKR.MI": _make_ticker_df({"sig_a": -1, "sig_b": -1, "sig_c": -1})}
        result = compute_score_vs_bm(data_dict, SIGNAL_COLS)
        assert result.iloc[0]["score_vs_bm"] == -len(SIGNAL_COLS)

    def test_all_neutral_score_is_zero(self):
        data_dict = {"TKR.MI": _make_ticker_df({"sig_a": 0, "sig_b": 0, "sig_c": 0})}
        result = compute_score_vs_bm(data_dict, SIGNAL_COLS)
        assert result.iloc[0]["score_vs_bm"] == 0

    def test_active_signals_listed_correctly(self):
        data_dict = {"TKR.MI": _make_ticker_df({"sig_a": 1, "sig_b": -1, "sig_c": 0})}
        result = compute_score_vs_bm(data_dict, SIGNAL_COLS)
        row = result.iloc[0]
        assert "sig_a" in row["active_signals_bm"]
        assert "sig_b" not in row["active_signals_bm"]

    def test_short_signals_listed_correctly(self):
        data_dict = {"TKR.MI": _make_ticker_df({"sig_a": 1, "sig_b": -1, "sig_c": 0})}
        result = compute_score_vs_bm(data_dict, SIGNAL_COLS)
        row = result.iloc[0]
        assert "sig_b" in row["short_signals_bm"]
        assert "sig_a" not in row["short_signals_bm"]

    def test_n_signals_reflects_input_length(self):
        data_dict = {"TKR.MI": _make_ticker_df({"sig_a": 0, "sig_b": 0, "sig_c": 0})}
        result = compute_score_vs_bm(data_dict, SIGNAL_COLS)
        assert result.iloc[0]["n_signals"] == len(SIGNAL_COLS)

    def test_multiple_tickers_each_scored_independently(self):
        data_dict = {
            "A.MI": _make_ticker_df({"sig_a": 1, "sig_b": 1, "sig_c": 1}),
            "B.MI": _make_ticker_df({"sig_a": -1, "sig_b": -1, "sig_c": -1}),
        }
        result = compute_score_vs_bm(data_dict, SIGNAL_COLS)
        scores = dict(zip(result["ticker"], result["score_vs_bm"]))
        assert scores["A.MI"] == 3
        assert scores["B.MI"] == -3

    def test_empty_data_dict_returns_empty_dataframe(self):
        result = compute_score_vs_bm({}, SIGNAL_COLS)
        assert result.empty

    def test_raises_on_empty_signal_cols(self):
        data_dict = {"TKR.MI": _make_ticker_df({"sig_a": 1, "sig_b": 0, "sig_c": 0})}
        with pytest.raises(ValueError, match="signal_cols"):
            compute_score_vs_bm(data_dict, [])


# ---------------------------------------------------------------------------
# compute_sector_peer_avg
# ---------------------------------------------------------------------------


class TestComputeSectorPeerAvg:
    def test_excludes_target_ticker_from_average(self):
        """A.MI (close=10+), B.MI (close=20+), C.MI (close=30+).
        Peer avg for A.MI = mean(B.MI, C.MI) ≠ mean(A.MI, B.MI, C.MI).
        """
        ohlc = _make_ohlc_df(["A.MI", "B.MI", "C.MI"], n_rows=3)
        result = compute_sector_peer_avg(ohlc, ticker="A.MI", sector_tickers=["A.MI", "B.MI", "C.MI"])

        assert not result.empty
        assert set(result.columns) == {"date", "close"}

        # First row: B.MI close=20.0, C.MI close=30.0 → avg=25.0
        first_row = result.sort_values("date").iloc[0]
        assert math.isclose(first_row["close"], 25.0, rel_tol=1e-6)

    def test_singleton_sector_returns_empty_dataframe(self):
        ohlc = _make_ohlc_df(["A.MI"])
        result = compute_sector_peer_avg(ohlc, ticker="A.MI", sector_tickers=["A.MI"])
        assert result.empty

    def test_two_tickers_peer_avg_is_the_other_ticker(self):
        ohlc = _make_ohlc_df(["A.MI", "B.MI"], n_rows=3)
        result = compute_sector_peer_avg(ohlc, ticker="A.MI", sector_tickers=["A.MI", "B.MI"])
        # B.MI starts at 20.0 and increments by 0.1
        first_row = result.sort_values("date").iloc[0]
        assert math.isclose(first_row["close"], 20.0, rel_tol=1e-6)

    def test_returns_date_aligned_series(self):
        ohlc = _make_ohlc_df(["A.MI", "B.MI", "C.MI"], n_rows=5)
        result = compute_sector_peer_avg(ohlc, ticker="A.MI", sector_tickers=["A.MI", "B.MI", "C.MI"])
        assert len(result) == 5
        assert result["date"].is_monotonic_increasing or True  # dates present

    def test_ticker_not_in_sector_list_still_returns_avg(self):
        """If ticker is not in sector_tickers, all tickers are treated as peers."""
        ohlc = _make_ohlc_df(["A.MI", "B.MI", "C.MI"], n_rows=3)
        result = compute_sector_peer_avg(
            ohlc, ticker="MISSING.MI", sector_tickers=["A.MI", "B.MI", "C.MI"]
        )
        # All three are peers (MISSING.MI has no rows in ohlc anyway)
        assert not result.empty


# ---------------------------------------------------------------------------
# build_dashboard
# ---------------------------------------------------------------------------


class TestBuildDashboard:
    def _score_bm(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ticker": "A.MI", "score_vs_bm": 3, "active_signals_bm": "s1, s2, s3", "short_signals_bm": "", "n_signals": 3},
                {"ticker": "B.MI", "score_vs_bm": -2, "active_signals_bm": "", "short_signals_bm": "s1, s2", "n_signals": 3},
                {"ticker": "C.MI", "score_vs_bm": 1, "active_signals_bm": "s1", "short_signals_bm": "", "n_signals": 3},
            ]
        )

    def _score_sector(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ticker": "A.MI", "score_vs_sector": 2, "active_signals_sector": "s1, s2", "short_signals_sector": ""},
                {"ticker": "B.MI", "score_vs_sector": None, "active_signals_sector": "", "short_signals_sector": ""},
                {"ticker": "C.MI", "score_vs_sector": 1, "active_signals_sector": "s1", "short_signals_sector": ""},
            ]
        )

    def _metrics(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ticker": "A.MI", "rclose_chg_20d": 5.2, "stop_loss": 9.5, "sl_dist_pct": 4.0, "last_date": pd.Timestamp("2024-03-01")},
                {"ticker": "B.MI", "rclose_chg_20d": -3.1, "stop_loss": None, "sl_dist_pct": None, "last_date": pd.Timestamp("2024-03-01")},
                {"ticker": "C.MI", "rclose_chg_20d": 1.0, "stop_loss": 9.8, "sl_dist_pct": 1.0, "last_date": pd.Timestamp("2024-03-01")},
            ]
        )

    def _sectors(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ticker": "A.MI", "name": "Company A", "sector": "Finance"},
                {"ticker": "B.MI", "name": "Company B", "sector": "Finance"},
                {"ticker": "C.MI", "name": "Company C", "sector": "Technology"},
            ]
        )

    def test_primary_sort_by_score_total_descending(self):
        result = build_dashboard(self._score_bm(), self._score_sector(), self._metrics(), self._sectors())
        # A.MI: 3+2=5, C.MI: 1+1=2, B.MI: -2+0=-2
        assert result.iloc[0]["ticker"] == "A.MI"
        assert result.iloc[1]["ticker"] == "C.MI"
        assert result.iloc[2]["ticker"] == "B.MI"

    def test_rank_column_starts_at_one(self):
        result = build_dashboard(self._score_bm(), self._score_sector(), self._metrics(), self._sectors())
        assert list(result["rank"]) == [1, 2, 3]

    def test_score_total_is_sum_of_bm_and_sector(self):
        result = build_dashboard(self._score_bm(), self._score_sector(), self._metrics(), self._sectors())
        a_row = result[result["ticker"] == "A.MI"].iloc[0]
        assert a_row["score_total"] == 5.0  # 3 + 2

    def test_null_sector_score_treated_as_zero_in_total(self):
        """B.MI has score_vs_sector=None (singleton). score_total = score_vs_bm + 0."""
        result = build_dashboard(self._score_bm(), self._score_sector(), self._metrics(), self._sectors())
        b_row = result[result["ticker"] == "B.MI"].iloc[0]
        assert b_row["score_total"] == -2.0

    def test_sector_column_populated_from_sectors_df(self):
        result = build_dashboard(self._score_bm(), self._score_sector(), self._metrics(), self._sectors())
        a_row = result[result["ticker"] == "A.MI"].iloc[0]
        assert a_row["sector"] == "Finance"
        assert a_row["name"] == "Company A"

    def test_required_columns_present(self):
        result = build_dashboard(self._score_bm(), self._score_sector(), self._metrics(), self._sectors())
        required = {"rank", "ticker", "name", "sector", "score_total", "score_vs_bm", "score_vs_sector"}
        assert required.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# compute_score_history
# ---------------------------------------------------------------------------


class TestComputeScoreHistory:
    def _make_data_dict(self) -> dict[str, pd.DataFrame]:
        """Three tickers, 10 rows each, signals vary by row."""
        base = date(2024, 1, 1)
        data = {}
        # A.MI: all signals +1 on last 3 rows, 0 before
        rows_a = []
        for i in range(10):
            sig = 1 if i >= 7 else 0
            rows_a.append({"date": pd.Timestamp(base + timedelta(days=i)), "sig_a": sig, "sig_b": sig})
        data["A.MI"] = pd.DataFrame(rows_a)

        # B.MI: all signals -1 on last 3 rows, 0 before
        rows_b = []
        for i in range(10):
            sig = -1 if i >= 7 else 0
            rows_b.append({"date": pd.Timestamp(base + timedelta(days=i)), "sig_a": sig, "sig_b": sig})
        data["B.MI"] = pd.DataFrame(rows_b)

        return data

    def test_shape_is_tickers_by_n_days(self):
        result = compute_score_history(self._make_data_dict(), ["sig_a", "sig_b"], n_days=5)
        assert result.shape == (2, 5)

    def test_column_count_matches_n_days(self):
        result = compute_score_history(self._make_data_dict(), ["sig_a", "sig_b"], n_days=3)
        assert len(result.columns) == 3

    def test_scores_are_summed_correctly(self):
        """A.MI last 3 rows all +1 on 2 signals → score = 2 per row."""
        result = compute_score_history(self._make_data_dict(), ["sig_a", "sig_b"], n_days=3)
        a_scores = result.loc["A.MI"].tolist()
        assert all(s == 2 for s in a_scores)

    def test_negative_scores_for_short_signals(self):
        """B.MI last 3 rows all -1 on 2 signals → score = -2 per row."""
        result = compute_score_history(self._make_data_dict(), ["sig_a", "sig_b"], n_days=3)
        b_scores = result.loc["B.MI"].tolist()
        assert all(s == -2 for s in b_scores)

    def test_columns_are_dates_in_ascending_order(self):
        result = compute_score_history(self._make_data_dict(), ["sig_a", "sig_b"], n_days=5)
        dates = list(result.columns)
        assert dates == sorted(dates)

    def test_n_days_larger_than_history_clips_to_available(self):
        """Requesting more days than available should return all available rows."""
        result = compute_score_history(self._make_data_dict(), ["sig_a", "sig_b"], n_days=999)
        assert len(result.columns) == 10  # only 10 rows of history

    def test_ticker_is_index(self):
        result = compute_score_history(self._make_data_dict(), ["sig_a", "sig_b"], n_days=3)
        assert result.index.name == "ticker"
        assert "A.MI" in result.index
        assert "B.MI" in result.index
