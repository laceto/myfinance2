"""
test_etf_sector_rotation.py — Unit tests for etf_sector_rotation.

etf_sector_rotation reads the per-window ETF returns (from etf_returns) and
groups them into sectors (keyword classification on the fund name) to surface
rotation: sectors whose recent (1M) pace runs hotter or colder than their
quarter (3M) trend.

Coverage
--------
classify_sector:
  - keyword rules match (banks, utilities, biotech, ...)
  - first rule wins when several could match (order matters)
  - leveraged/inverse funds are bucketed separately (excluded from medians)
  - unknown names fall through to Other/Unclassified
sector_table:
  - median per window per sector, with a fund count `n`
  - thin buckets (n < min_funds) are dropped
  - excluded buckets (leveraged/inverse, unclassified) never appear
add_acceleration:
  - accel_1M_vs_3M = 1M - 3M/3   (recent month vs quarter-implied monthly pace)
  - accel_1W_vs_1M = 1W - 1M/4.3 (recent week vs month-implied weekly pace)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_sector_rotation import (
    LEVERAGED_INVERSE,
    UNCLASSIFIED,
    add_acceleration,
    classify_sector,
    sector_table,
)


def test_classify_sector_matches_keywords():
    assert classify_sector("Amundi STOXX Europe 600 Banks UCITS ETF Acc") == "Banks"
    assert classify_sector("Xtrackers MSCI World Utilities UCITS ETF 1C") == "Utilities"
    assert classify_sector("ARK Genomic Revolution UCITS ETF USD Acc") == "Biotech/Genomics"


def test_classify_sector_first_rule_wins():
    # "Banks" is ordered before the broader "Financials" rule, so a bank fund
    # classifies as Banks, not Financials.
    assert classify_sector("SPDR MSCI Europe Financials Banks ETF") == "Banks"


def test_classify_sector_leveraged_bucketed_separately():
    assert classify_sector("Amundi MSCI USA Daily (2x) Leveraged UCITS ETF Acc") == LEVERAGED_INVERSE
    assert classify_sector("Amundi MSCI USA Daily (-1x) Inverse UCITS ETF Acc") == LEVERAGED_INVERSE


def test_classify_sector_unknown_is_unclassified():
    assert classify_sector("Some Totally Generic Multi-Asset Wrapper") == UNCLASSIFIED


def _returns_frame():
    # Two banks, one utility (thin), one leveraged (excluded), one unknown.
    idx = ["BNK1", "BNK2", "UTIL1", "LEV1", "MISC1"]
    return pd.DataFrame(
        {
            "1W": [0.02, 0.04, 0.01, 0.10, 0.00],
            "1M": [0.06, 0.08, 0.03, 0.20, 0.01],
            "3M": [0.12, 0.16, 0.02, 0.40, 0.05],
            "6M": [0.10, 0.14, 0.12, 0.50, 0.05],
            "YTD": [0.15, 0.20, 0.15, 0.70, 0.05],
        },
        index=idx,
    )


def _names():
    return {
        "BNK1": "Amundi STOXX Europe 600 Banks UCITS ETF Acc",
        "BNK2": "Amundi Euro Stoxx Banks UCITS ETF Acc",
        "UTIL1": "Xtrackers MSCI World Utilities UCITS ETF 1C",
        "LEV1": "Amundi MSCI USA Daily (2x) Leveraged UCITS ETF Acc",
        "MISC1": "Some Totally Generic Multi-Asset Wrapper",
    }


def test_sector_table_medians_and_counts():
    table = sector_table(_returns_frame(), _names(), min_funds=2)
    # Only Banks survives (n=2); Utilities is thin (n=1); leveraged/unknown excluded.
    assert list(table.index) == ["Banks"]
    assert table.loc["Banks", "n"] == 2
    # Median of the two banks.
    assert table.loc["Banks", "1M"] == np.median([0.06, 0.08])
    assert table.loc["Banks", "3M"] == np.median([0.12, 0.16])


def test_sector_table_excludes_leveraged_and_unclassified():
    table = sector_table(_returns_frame(), _names(), min_funds=1)
    assert LEVERAGED_INVERSE not in table.index
    assert UNCLASSIFIED not in table.index
    # With min_funds=1 the thin Utilities bucket now appears.
    assert "Utilities" in table.index


def test_add_acceleration_formula():
    table = pd.DataFrame(
        {"1W": [0.02], "1M": [0.06], "3M": [0.12]},
        index=["Banks"],
    )
    out = add_acceleration(table)
    assert np.isclose(out.loc["Banks", "accel_1M_vs_3M"], 0.06 - 0.12 / 3.0)
    assert np.isclose(out.loc["Banks", "accel_1W_vs_1M"], 0.02 - 0.06 / 4.3)
