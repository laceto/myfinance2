"""
signal_flips_history.py — reconstruct N days of signal flips from the brief.

WHAT
    ``data/results/it/daily_brief.txt`` is regenerated every run and its
    "SIGNAL FLIPS — last bar" section only shows the *latest* bar's flips. This
    script walks the git history of that file (one commit per trading day) and
    collects every flip of a chosen direction — ``bull_flip`` (turned bullish)
    or ``bear_flip`` (turned bearish) — giving a month-at-a-glance view of which
    symbols flipped, when, and on which signal method.

WHY
    The live brief can't answer "show me the last month of bull/bear flips" —
    that history exists only across committed snapshots. Runs both on demand and
    daily in CI: the ``refresh-signal-flips`` job in ``analyze_and_report.yml``
    regenerates ``bull_flips_last_month.txt`` and ``bear_flips_last_month.txt``
    right after the analyze job commits each day's brief (needs a full-history
    checkout — see that workflow).

HOW
    1. ``git log`` the brief file over the window -> commit hashes.
    2. For each, ``git show <hash>:<file>`` -> that day's brief text.
    3. Parse the SIGNAL FLIPS section for rows of the requested direction.
    4. De-duplicate ``(date, method, symbol)`` and report newest-first.

INVARIANTS & FAILURE MODES
    - Parsing is anchored to the "SIGNAL FLIPS" banner: flip lines elsewhere
      (or in prose) are ignored.
    - A flip is any row the brief itself labels ``bull_flip`` / ``bear_flip``.
      This includes soft flips (e.g. a bull ``-1 → 0`` or a bear ``1 → 0`` under
      ``rtt_5020``) as well as full flips (``* → 1`` bull, ``* → -1`` bear). The
      before/after state is preserved so ``--full-only`` can keep just the flips
      that reached the terminal state for the direction (bull ``after==1``,
      bear ``after==-1``).
    - Only marginable names (``marginabile == si``) can be short-sold, so a bear
      flip is only *actionable* if the name is marginable. Bear output therefore
      defaults to marginable-only; bull (a long entry) has no such restriction.
      Both defaults are overridable (``--marginable-only`` / ``--all-margins``).
    - De-dup key is ``(date, method, symbol)`` — the same commit appearing
      twice, or a flip persisting in a re-commit, is counted once per day.
    - Depends on committed brief history: days where the brief wasn't committed
      simply don't contribute (no interpolation).

OBSERVABILITY
    INFO logging (commits scanned, events found); ``-v`` to see it.

SEPARATION OF CONCERNS
    Pure text parsing (``brief_date``, ``flips_in_brief``, ``collect_flips``) is
    decoupled from git I/O (``iter_brief_versions``), so the parser is
    unit-tested without a repository.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

logger = logging.getLogger("signal_flips_history")

BRIEF_FILE = "data/results/it/daily_brief.txt"
DEFAULT_REVISION = "origin/main"
DEFAULT_SINCE = "1 month ago"

# The two flip directions and their terminal ("full") after-state.
DIRECTIONS = ("bull", "bear")
_LABEL = {"bull": "bull_flip", "bear": "bear_flip"}
_TERMINAL = {"bull": 1, "bear": -1}
DEFAULT_OUTPUTS: Dict[str, Path] = {
    "bull": Path("data/results/it/bull_flips_last_month.txt"),
    "bear": Path("data/results/it/bear_flips_last_month.txt"),
}

_DATE_RE = re.compile(r"DAILY BRIEF\s+[—-]\s+(\d{4}-\d{2}-\d{2})")
_METHOD_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_ROW_RE = re.compile(
    r"^(bull_flip|bear_flip)\s+(\S+)\s+(.*?)\s+(-?\d+)\s*[—→>-]+\s*(-?\d+)\s*$"
)
_SIGNAL_BANNER = "SIGNAL FLIPS"


@dataclass(frozen=True)
class SignalFlip:
    """One signal flip from a daily brief's SIGNAL FLIPS section."""

    date: str          # YYYY-MM-DD (the brief's bar date)
    direction: str     # "bull" or "bear"
    method: str        # e.g. "rbo_20", "rsma_50100150", "rtt_5020"
    symbol: str        # e.g. "BPE.MI"
    description: str   # name + sector, whitespace-collapsed
    marginabile: str   # "si" (marginable -> short-sellable) or "NaN"/other
    before: int        # signal state before the flip (-1 / 0 / 1)
    after: int         # signal state after the flip

    @property
    def is_marginable(self) -> bool:
        """True when the name is marginable (``marginabile == si``) — required to short."""
        return self.marginabile.strip().lower() == "si"


def brief_date(text: str) -> Optional[str]:
    """Return the YYYY-MM-DD from the 'DAILY BRIEF — <date>' banner, or None."""
    m = _DATE_RE.search(text)
    return m.group(1) if m else None


def flips_in_brief(text: str, direction: str) -> List[SignalFlip]:
    """Parse one brief's SIGNAL FLIPS section into flips of the given direction.

    Only rows inside the SIGNAL FLIPS section and labelled with the requested
    direction's tag (``bull_flip`` / ``bear_flip``) are returned. The section
    ends at the next banner line (``===``) or the "Full detail" pointer.
    """
    if direction not in _LABEL:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    tag = _LABEL[direction]
    date = brief_date(text) or "?"
    in_signal = False
    method: Optional[str] = None
    out: List[SignalFlip] = []
    for line in text.splitlines():
        if _SIGNAL_BANNER in line:
            in_signal = True
            method = None
            continue
        if not in_signal:
            continue
        stripped = line.strip()
        if stripped.startswith("=") or "Full detail" in line:
            in_signal = False
            continue
        mm = _METHOD_RE.match(line)
        if mm:
            method = mm.group(1)
            continue
        rr = _ROW_RE.match(stripped)
        if rr and rr.group(1) == tag:
            # group(3) is "<name> <sector> <marginabile>"; marginabile is always
            # the final column (si / NaN), so split it off the end.
            body = re.sub(r"\s+", " ", rr.group(3)).strip()
            if " " in body:
                description, marginabile = body.rsplit(" ", 1)
            else:
                description, marginabile = body, ""
            out.append(
                SignalFlip(
                    date=date,
                    direction=direction,
                    method=method or "?",
                    symbol=rr.group(2),
                    description=description,
                    marginabile=marginabile,
                    before=int(rr.group(4)),
                    after=int(rr.group(5)),
                )
            )
    return out


def collect_flips(briefs: Iterable[str], direction: str) -> List[SignalFlip]:
    """Flatten flips of one direction across brief versions, de-duped by (date, method, symbol)."""
    seen = set()
    records: List[SignalFlip] = []
    for text in briefs:
        for flip in flips_in_brief(text, direction):
            key = (flip.date, flip.method, flip.symbol)
            if key in seen:
                continue
            seen.add(key)
            records.append(flip)
    return records


def _run(args: Sequence[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def iter_brief_versions(
    file: str = BRIEF_FILE,
    since: str = DEFAULT_SINCE,
    *,
    revision: str = DEFAULT_REVISION,
    run: Callable[[Sequence[str]], str] = _run,
) -> Iterator[str]:
    """Yield the brief's text at each commit touching it in the window (oldest first)."""
    log = run(["git", "log", revision, f"--since={since}", "--reverse",
               "--pretty=%H", "--", file])
    hashes = [h for h in log.split() if h]
    logger.info("scanning %d brief commits from %s", len(hashes), revision)
    for h in hashes:
        text = run(["git", "show", f"{h}:{file}"])
        if text:
            yield text


def build_report(
    records: Sequence[SignalFlip],
    *,
    direction: str,
    since_label: str,
    marginable_only: bool = False,
) -> str:
    """Render a newest-first, grouped-by-date flips report for one direction."""
    lines: List[str] = []
    lines.append(f"{direction.upper()} FLIPS from daily_brief.txt (SIGNAL FLIPS section)")
    lines.append(f"window: {since_label}   events: {len(records)}")
    lines.append("change: state before->after  (1=bull, 0=neutral, -1=bear)")
    if marginable_only:
        lines.append("filter: marginabile=si only (short-sellable names)")
    lines.append("")
    by_date: Dict[str, List[SignalFlip]] = {}
    for f in records:
        by_date.setdefault(f.date, []).append(f)
    for date in sorted(by_date, reverse=True):
        day = sorted(by_date[date], key=lambda f: (f.symbol, f.method))
        lines.append(f"-- {date}  ({len(day)} {direction} flips) --")
        for f in day:
            marg = "si " if f.is_marginable else f.marginabile or "-"
            lines.append(
                f"   {f.symbol:<9} {f.before}->{f.after:<4} {f.method:<13} "
                f"marg={marg:<4} {f.description}"
            )
        lines.append("")
    if records:
        freq = Counter(f.symbol for f in records)
        lines.append("Most frequent flippers: "
                     + ", ".join(f"{s}x{c}" for s, c in freq.most_common(10)))
    return "\n".join(lines).rstrip() + "\n"


def _directions_arg(value: str) -> List[str]:
    return list(DIRECTIONS) if value == "both" else [value]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Reconstruct signal flips from daily_brief.txt history.")
    parser.add_argument("--direction", choices=["bull", "bear", "both"], default="bull",
                        help="which flips to report (default bull)")
    parser.add_argument("--since", default=DEFAULT_SINCE,
                        help="git --since window (default '1 month ago')")
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="git revision to read history from (default origin/main)")
    parser.add_argument("--file", default=BRIEF_FILE)
    parser.add_argument("--full-only", action="store_true",
                        help="keep only flips that reached the terminal state (bull after==1, bear after==-1)")
    # Marginable filter: only marginabile=si names can be short-sold, so bear
    # flips default to marginable-only (that's the actionable set). Bull flips
    # are long entries -> no such restriction. Either default is overridable.
    parser.add_argument("--marginable-only", dest="marginable_only",
                        action="store_true", default=None,
                        help="keep only marginabile=si (short-sellable) names; default ON for bear, OFF for bull")
    parser.add_argument("--all-margins", dest="marginable_only", action="store_false",
                        help="include non-marginable names too (overrides the per-direction default)")
    parser.add_argument("--output", type=Path, default=None,
                        help="write report here (single direction only; default paths used for --direction both)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    directions = _directions_arg(args.direction)
    if args.output is not None and len(directions) > 1:
        parser.error("--output cannot be combined with --direction both (writes two files)")

    # Read history once; reuse the same brief texts for both directions.
    briefs = list(iter_brief_versions(args.file, args.since, revision=args.revision))

    for direction in directions:
        records = collect_flips(briefs, direction)
        if args.full_only:
            records = [f for f in records if f.after == _TERMINAL[direction]]
        # Resolve the marginable filter: explicit flag wins; otherwise bear
        # defaults to marginable-only (short-sellable), bull to all names.
        marginable_only = args.marginable_only
        if marginable_only is None:
            marginable_only = direction == "bear"
        if marginable_only:
            records = [f for f in records if f.is_marginable]
        logger.info("collected %d %s_flip events (marginable_only=%s)",
                    len(records), direction, marginable_only)

        report = build_report(records, direction=direction, since_label=args.since,
                              marginable_only=marginable_only)
        print(report, end="")

        out = args.output if args.output is not None else DEFAULT_OUTPUTS[direction]
        if args.output is not None or args.direction == "both":
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(report)
            logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
