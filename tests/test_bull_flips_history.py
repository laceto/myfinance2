"""
test_bull_flips_history.py — Unit tests for bull_flips_history.

bull_flips_history reconstructs the last N days of `bull_flip` events from the
git history of data/results/it/daily_brief.txt. The brief is overwritten each
run and only shows the latest bar's flips in its "SIGNAL FLIPS" section, so the
month view comes from walking committed versions.

Coverage
--------
brief_date:
  - extracts the YYYY-MM-DD from the "DAILY BRIEF — <date>" banner
bull_flips_in_brief:
  - parses the SIGNAL FLIPS section, one record per bull_flip row
  - ignores bear_flip rows and everything outside the SIGNAL FLIPS section
  - attaches the method (from the [method] sub-headers) and the before→after state
collect_bull_flips:
  - walks injected brief versions and flattens their bull flips
  - de-duplicates (date, method, symbol) so a symbol repeated across snapshots
    is counted once
"""

from __future__ import annotations

from bull_flips_history import (
    BullFlip,
    brief_date,
    bull_flips_in_brief,
    collect_bull_flips,
)

SAMPLE = """
====================================================================================================
  DAILY BRIEF — 2026-07-23   (min_score=3  max_days=60  top_n=30  LM=on)
====================================================================================================

  BULL CANDIDATES  (2 stocks)
------------------------------------------------------------
conviction score ... symbol name
    50.8    10 ...  UNI.MI UNIPOL

  SIGNAL FLIPS — last bar  (3 flips across 2 methods)
------------------------------------------------------------
  [rbo_20]
direction symbol name    sector marginabile change
bull_flip BPE.MI BPER BANCA Finance si -1 → 1
bear_flip ACE.MI ACEA Utilities si 1 → -1

  [rtt_5020]
direction symbol name    sector marginabile change
bull_flip G.MI GENERALI ASS Finance NaN 0 → 1

====================================================================================================
  Full detail -> data/results/it/daily_brief.xlsx
====================================================================================================
"""


def test_brief_date_extracts_banner_date():
    assert brief_date(SAMPLE) == "2026-07-23"
    assert brief_date("no banner here") is None


def test_bull_flips_in_brief_extracts_only_bull_flips():
    flips = bull_flips_in_brief(SAMPLE)
    # Two bull_flip rows (BPE via rbo_20, G via rtt_5020); the bear_flip is ignored.
    assert {f.symbol for f in flips} == {"BPE.MI", "G.MI"}
    assert all(isinstance(f, BullFlip) for f in flips)


def test_bull_flips_in_brief_captures_method_and_state():
    flips = {f.symbol: f for f in bull_flips_in_brief(SAMPLE)}
    assert flips["BPE.MI"].method == "rbo_20"
    assert flips["BPE.MI"].date == "2026-07-23"
    assert (flips["BPE.MI"].before, flips["BPE.MI"].after) == (-1, 1)
    assert flips["G.MI"].method == "rtt_5020"
    assert (flips["G.MI"].before, flips["G.MI"].after) == (0, 1)


def test_bull_flips_in_brief_ignores_flips_outside_signal_section():
    # A bull_flip-looking line before the SIGNAL FLIPS header must not be picked up.
    text = "bull_flip ZZZ.MI FAKE Finance NaN -1 → 1\n" + SAMPLE
    flips = bull_flips_in_brief(text)
    assert "ZZZ.MI" not in {f.symbol for f in flips}


def test_collect_bull_flips_walks_versions_and_dedups():
    day1 = SAMPLE  # 2026-07-23: BPE.MI/rbo_20, G.MI/rtt_5020
    day2 = SAMPLE.replace("2026-07-23", "2026-07-24")  # same flips, different date
    # Same day repeated (duplicate commit) must not double-count.
    versions = [day1, day1, day2]
    records = collect_bull_flips(versions)
    keys = {(f.date, f.method, f.symbol) for f in records}
    assert keys == {
        ("2026-07-23", "rbo_20", "BPE.MI"),
        ("2026-07-23", "rtt_5020", "G.MI"),
        ("2026-07-24", "rbo_20", "BPE.MI"),
        ("2026-07-24", "rtt_5020", "G.MI"),
    }
    assert len(records) == 4  # 6 raw rows -> 4 after dedup
