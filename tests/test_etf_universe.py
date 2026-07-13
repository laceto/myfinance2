"""
test_etf_universe.py — Unit tests for etf_universe (the active ETF universe).

The active universe = the curated top-15 UCITS ETFs (etf_top15) followed by N
additional tickers drawn from the justETF list. Both the cold-start backfill
and the daily download read the seed this module produces, so its composition
and ordering are load-bearing.

Coverage
--------
build_active_universe_table (pure):
  - curated core comes first, in its original order
  - appends the first `extra_count` justETF tickers not already in the core
  - total = core + extra_count when enough non-overlapping extras exist
  - excludes justETF rows whose ticker is already in the core (no duplicates)
  - fewer available extras than requested -> returns all available (no error)
  - (name, ticker) schema preserved
  - missing source columns -> ValueError
"""

from __future__ import annotations

import pandas as pd
import pytest

from etf_universe import KNOWN_DEAD, build_active_universe_table


def _core(tickers):
    return pd.DataFrame({"name": [f"core-{t}" for t in tickers], "ticker": tickers})


def _justetf(tickers):
    return pd.DataFrame({"name": [f"js-{t}" for t in tickers], "ticker": tickers})


class TestBuildActiveUniverseHappyPath:

    def test_core_comes_first_in_order(self):
        core = _core(["CSPX.L", "SWDA.L"])
        justetf = _justetf(["AAA.L", "BBB.L", "CCC.L"])

        result = build_active_universe_table(justetf, core, extra_count=2)

        assert result["ticker"].tolist()[:2] == ["CSPX.L", "SWDA.L"]

    def test_appends_first_n_extras_not_in_core(self):
        core = _core(["CSPX.L"])
        justetf = _justetf(["AAA.L", "BBB.L", "CCC.L", "DDD.L"])

        result = build_active_universe_table(justetf, core, extra_count=2)

        assert result["ticker"].tolist() == ["CSPX.L", "AAA.L", "BBB.L"]

    def test_total_is_core_plus_extra_count(self):
        core = _core([f"C{i}.L" for i in range(15)])
        justetf = _justetf([f"J{i}.L" for i in range(200)])

        result = build_active_universe_table(justetf, core, extra_count=100)

        assert len(result) == 115


class TestBuildActiveUniverseEdgeCases:

    def test_excludes_justetf_rows_already_in_core(self):
        core = _core(["DUP.L", "CSPX.L"])
        justetf = _justetf(["DUP.L", "AAA.L"])   # DUP.L overlaps the core

        result = build_active_universe_table(justetf, core, extra_count=5)

        # DUP.L appears once (from the core), not duplicated from justETF.
        assert result["ticker"].tolist() == ["DUP.L", "CSPX.L", "AAA.L"]
        assert result["ticker"].is_unique

    def test_fewer_available_extras_than_requested_returns_all(self):
        core = _core(["CSPX.L"])
        justetf = _justetf(["AAA.L", "BBB.L"])

        result = build_active_universe_table(justetf, core, extra_count=100)

        assert result["ticker"].tolist() == ["CSPX.L", "AAA.L", "BBB.L"]

    def test_preserves_name_ticker_schema(self):
        core = _core(["CSPX.L"])
        justetf = _justetf(["AAA.L"])

        result = build_active_universe_table(justetf, core, extra_count=1)

        assert list(result.columns) == ["name", "ticker"]


class TestBuildActiveUniverseExcludesDead:

    def test_drops_excluded_tickers_without_replacing(self):
        core = _core(["CSPX.L"])
        justetf = _justetf(["AAA.L", "DEAD.L", "BBB.L", "CCC.L"])

        # First 3 non-core = AAA, DEAD, BBB. DEAD is pruned and is NOT
        # replaced by CCC — the result shrinks, matching "drop, don't backfill".
        result = build_active_universe_table(justetf, core, extra_count=3, exclude={"DEAD.L"})

        assert result["ticker"].tolist() == ["CSPX.L", "AAA.L", "BBB.L"]

    def test_known_dead_contains_the_flagged_symbols(self):
        flagged = {"10AI.PA", "AEMD.PA", "AEMU.PA", "AHYQ.PA",
                   "AMINA.AS", "POLY.AS", "STXH.SW", "XDCN.AS"}

        assert flagged.issubset(set(KNOWN_DEAD))

    def test_default_exclude_applies_known_dead(self):
        core = _core(["CSPX.L"])
        justetf = _justetf(["AEMD.PA", "AAA.L"])   # AEMD.PA is a known-dead symbol

        result = build_active_universe_table(justetf, core, extra_count=5)

        assert "AEMD.PA" not in result["ticker"].tolist()


class TestBuildActiveUniverseFailureModes:

    def test_missing_source_columns_raises_value_error(self):
        core = _core(["CSPX.L"])
        bad_justetf = pd.DataFrame({"isin": ["IE00X"], "symbol": ["AAA.L"]})

        with pytest.raises(ValueError, match="name|ticker"):
            build_active_universe_table(bad_justetf, core, extra_count=1)
