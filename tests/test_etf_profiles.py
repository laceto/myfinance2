"""
test_etf_profiles.py — Unit tests for etf_profiles (AUM-ranked ETF selection).

etf_profiles turns the justETF `profiles.jsonl` export (ISIN, name, fund size,
...) into a Yahoo-ticker table ranked by real assets under management, so the
active universe can be enriched with the largest funds instead of relying on
justETF list order. profiles.jsonl has no Yahoo ticker, so names are mapped to
tickers via the existing name -> ticker table.

Coverage
--------
load_profiles:
  - reads a JSON Lines file into a DataFrame with the expected columns
profiles_ticker_table (pure):
  - maps name -> Yahoo ticker and returns the (name, ticker) schema
  - orders by descending fund size (largest AUM first)
  - drops rows whose name has no ticker mapping
  - de-duplicates tickers, keeping the larger-AUM row
  - respects the top-n limit
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from etf_profiles import load_profiles, profiles_ticker_table


def _profiles(rows):
    """rows: list of (name, fund_size_eur_mln) -> profiles-shaped DataFrame."""
    return pd.DataFrame(
        [{"name": n, "fund_size_eur_mln": s, "isin": f"ISIN{i}"} for i, (n, s) in enumerate(rows)]
    )


class TestLoadProfiles:

    def test_reads_jsonl_into_dataframe(self, tmp_path: Path):
        p = tmp_path / "profiles.jsonl"
        p.write_text(
            '{"rank": 1, "isin": "IE00X", "name": "Fund A", "fund_size_eur_mln": 900.0}\n'
            '{"rank": 2, "isin": "IE00Y", "name": "Fund B", "fund_size_eur_mln": 800.0}\n',
            encoding="utf-8",
        )

        df = load_profiles(p)

        assert len(df) == 2
        assert {"name", "fund_size_eur_mln", "isin"}.issubset(df.columns)
        assert df.iloc[0]["name"] == "Fund A"


class TestProfilesTickerTable:

    def test_maps_names_to_tickers_with_correct_schema(self):
        profiles = _profiles([("Fund A", 900.0), ("Fund B", 800.0)])
        name2tick = {"Fund A": "AAA.L", "Fund B": "BBB.DE"}

        result = profiles_ticker_table(profiles, name2tick, n=100)

        assert list(result.columns) == ["name", "ticker"]
        assert set(result["ticker"]) == {"AAA.L", "BBB.DE"}

    def test_orders_by_descending_fund_size(self):
        profiles = _profiles([("Small", 100.0), ("Big", 900.0), ("Mid", 500.0)])
        name2tick = {"Small": "S.L", "Big": "B.L", "Mid": "M.L"}

        result = profiles_ticker_table(profiles, name2tick, n=100)

        assert result["ticker"].tolist() == ["B.L", "M.L", "S.L"]

    def test_drops_names_without_a_ticker_mapping(self):
        profiles = _profiles([("Mapped", 900.0), ("Unmapped", 800.0)])
        name2tick = {"Mapped": "OK.L"}   # "Unmapped" absent

        result = profiles_ticker_table(profiles, name2tick, n=100)

        assert result["ticker"].tolist() == ["OK.L"]

    def test_dedupes_ticker_keeping_larger_aum(self):
        # Two funds mapping to the same ticker: the larger-AUM one wins.
        profiles = _profiles([("Listing Small", 100.0), ("Listing Big", 900.0)])
        name2tick = {"Listing Small": "DUP.L", "Listing Big": "DUP.L"}

        result = profiles_ticker_table(profiles, name2tick, n=100)

        assert result["ticker"].tolist() == ["DUP.L"]
        assert result["name"].tolist() == ["Listing Big"]

    def test_respects_top_n(self):
        profiles = _profiles([("A", 900.0), ("B", 800.0), ("C", 700.0)])
        name2tick = {"A": "A.L", "B": "B.L", "C": "C.L"}

        result = profiles_ticker_table(profiles, name2tick, n=2)

        assert result["ticker"].tolist() == ["A.L", "B.L"]
