"""
test_signal_flips_history.py — Unit tests for signal_flips_history.

signal_flips_history reconstructs the last N days of bull/bear flip events from
the git history of data/results/it/daily_brief.txt. The brief is overwritten
each run and only shows the latest bar's flips, so the month view comes from
walking committed versions.

Coverage
--------
brief_date:
  - extracts the YYYY-MM-DD from the "DAILY BRIEF — <date>" banner
flips_in_brief:
  - bull direction -> only bull_flip rows; bear direction -> only bear_flip rows
  - ignores rows outside the SIGNAL FLIPS section
  - captures method, before/after state, and direction
  - rejects an unknown direction
collect_flips:
  - walks injected brief versions and de-dups (date, method, symbol) per direction
"""

from __future__ import annotations

import pytest

from signal_flips_history import (
    SignalFlip,
    brief_date,
    collect_flips,
    flips_in_brief,
)

SAMPLE = """
====================================================================================================
  DAILY BRIEF — 2026-07-23   (min_score=3  max_days=60  top_n=30  LM=on)
====================================================================================================

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


def test_flips_in_brief_bull_only():
    flips = flips_in_brief(SAMPLE, "bull")
    assert {f.symbol for f in flips} == {"BPE.MI", "G.MI"}
    assert all(f.direction == "bull" for f in flips)
    assert all(isinstance(f, SignalFlip) for f in flips)


def test_flips_in_brief_bear_only():
    flips = flips_in_brief(SAMPLE, "bear")
    # The single bear_flip row (ACEA) — bull rows are excluded.
    assert [f.symbol for f in flips] == ["ACE.MI"]
    ace = flips[0]
    assert ace.direction == "bear"
    assert ace.method == "rbo_20"
    assert (ace.before, ace.after) == (1, -1)


def test_marginabile_is_parsed_and_split_from_description():
    bull = {f.symbol: f for f in flips_in_brief(SAMPLE, "bull")}
    # "BPER BANCA Finance si" -> description without the trailing marginabile token.
    assert bull["BPE.MI"].marginabile == "si"
    assert bull["BPE.MI"].is_marginable is True
    assert bull["BPE.MI"].description == "BPER BANCA Finance"
    # "GENERALI ASS Finance NaN" -> not marginable.
    assert bull["G.MI"].marginabile == "NaN"
    assert bull["G.MI"].is_marginable is False


def test_marginable_filter_keeps_only_shortable_bear_flips():
    # Two bear flips: one marginable (si), one not (NaN).
    text = """
  DAILY BRIEF — 2026-07-23
  SIGNAL FLIPS — last bar
  [rbo_20]
bear_flip ACE.MI ACEA Utilities si 1 → -1
bear_flip XYZ.MI SOME NAME Finance NaN 1 → -1
====================================================================================================
"""
    bears = flips_in_brief(text, "bear")
    assert {f.symbol for f in bears} == {"ACE.MI", "XYZ.MI"}
    shortable = [f for f in bears if f.is_marginable]
    assert [f.symbol for f in shortable] == ["ACE.MI"]


def test_flips_in_brief_captures_method_and_state():
    bull = {f.symbol: f for f in flips_in_brief(SAMPLE, "bull")}
    assert bull["BPE.MI"].method == "rbo_20"
    assert (bull["BPE.MI"].before, bull["BPE.MI"].after) == (-1, 1)
    assert bull["G.MI"].method == "rtt_5020"
    assert (bull["G.MI"].before, bull["G.MI"].after) == (0, 1)


def test_flips_in_brief_ignores_outside_signal_section():
    text = "bear_flip ZZZ.MI FAKE Finance NaN 1 → -1\n" + SAMPLE
    assert "ZZZ.MI" not in {f.symbol for f in flips_in_brief(text, "bear")}


def test_flips_in_brief_rejects_unknown_direction():
    with pytest.raises(ValueError):
        flips_in_brief(SAMPLE, "sideways")


def test_collect_flips_walks_versions_and_dedups():
    day2 = SAMPLE.replace("2026-07-23", "2026-07-24")
    versions = [SAMPLE, SAMPLE, day2]  # duplicate day1 must not double-count
    bull = collect_flips(versions, "bull")
    assert {(f.date, f.method, f.symbol) for f in bull} == {
        ("2026-07-23", "rbo_20", "BPE.MI"),
        ("2026-07-23", "rtt_5020", "G.MI"),
        ("2026-07-24", "rbo_20", "BPE.MI"),
        ("2026-07-24", "rtt_5020", "G.MI"),
    }
    bear = collect_flips(versions, "bear")
    assert {(f.date, f.symbol) for f in bear} == {
        ("2026-07-23", "ACE.MI"),
        ("2026-07-24", "ACE.MI"),
    }
