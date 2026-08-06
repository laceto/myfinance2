"""
bull_flips_history.py — reconstruct N days of `bull_flip` events from the brief.

WHAT
    ``data/results/it/daily_brief.txt`` is regenerated every run and its
    "SIGNAL FLIPS — last bar" section only shows the *latest* bar's flips. This
    script walks the git history of that file (one commit per trading day) and
    collects every ``bull_flip`` row, giving a month-at-a-glance view of which
    symbols turned bullish, when, and on which signal method.

WHY
    The live brief can't answer "show me the last month of bull flips" — that
    history exists only across committed snapshots. Runs both on demand and
    daily in CI: the ``refresh-bull-flips`` job in ``analyze_and_report.yml``
    regenerates ``data/results/it/bull_flips_last_month.txt`` right after the
    analyze job commits each day's brief (needs a full-history checkout).

HOW
    1. ``git log`` the brief file over the window -> commit hashes.
    2. For each, ``git show <hash>:<file>`` -> that day's brief text.
    3. Parse the SIGNAL FLIPS section for ``bull_flip`` rows.
    4. De-duplicate ``(date, method, symbol)`` and report newest-first.

INVARIANTS & FAILURE MODES
    - Parsing is anchored to the "SIGNAL FLIPS" banner: ``bull_flip`` lines
      elsewhere (or in prose) are ignored.
    - A "bull flip" is any row the brief itself labels ``bull_flip``; this
      includes soft flips (e.g. ``-1 → 0`` under ``rtt_5020``) as well as full
      ``* → 1`` flips. The before/after state is preserved so callers can
      filter to ``after == 1`` if they want only fully-bullish flips.
    - De-dup key is ``(date, method, symbol)`` — the same commit appearing
      twice, or a flip persisting in a re-commit, is counted once per day.
    - Depends on committed brief history: days where the brief wasn't committed
      simply don't contribute (no interpolation).

OBSERVABILITY
    INFO logging (commits scanned, events found); ``-v`` to see it.

SEPARATION OF CONCERNS
    Pure text parsing (``brief_date``, ``bull_flips_in_brief``,
    ``collect_bull_flips``) is decoupled from git I/O (``iter_brief_versions``),
    so the parser is unit-tested without a repository.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

logger = logging.getLogger("bull_flips_history")

BRIEF_FILE = "data/results/it/daily_brief.txt"
DEFAULT_REVISION = "origin/main"
DEFAULT_SINCE = "1 month ago"
DEFAULT_OUTPUT = Path("data/results/it/bull_flips_last_month.txt")

_DATE_RE = re.compile(r"DAILY BRIEF\s+[—-]\s+(\d{4}-\d{2}-\d{2})")
_METHOD_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_ROW_RE = re.compile(
    r"^(bull_flip|bear_flip)\s+(\S+)\s+(.*?)\s+(-?\d+)\s*[—→>-]+\s*(-?\d+)\s*$"
)
_SIGNAL_BANNER = "SIGNAL FLIPS"


@dataclass(frozen=True)
class BullFlip:
    """One bull_flip event from a daily brief's SIGNAL FLIPS section."""

    date: str        # YYYY-MM-DD (the brief's bar date)
    method: str      # e.g. "rbo_20", "rsma_50100150", "rtt_5020"
    symbol: str      # e.g. "BPE.MI"
    description: str  # name + sector + marginabile, whitespace-collapsed
    before: int      # signal state before the flip (-1 / 0 / 1)
    after: int       # signal state after the flip


def brief_date(text: str) -> Optional[str]:
    """Return the YYYY-MM-DD from the 'DAILY BRIEF — <date>' banner, or None."""
    m = _DATE_RE.search(text)
    return m.group(1) if m else None


def bull_flips_in_brief(text: str) -> List[BullFlip]:
    """Parse one brief's SIGNAL FLIPS section into BullFlip records.

    Only rows inside the SIGNAL FLIPS section and labelled ``bull_flip`` are
    returned. The section ends at the next banner line (``===``) or the
    "Full detail" pointer.
    """
    date = brief_date(text) or "?"
    in_signal = False
    method: Optional[str] = None
    out: List[BullFlip] = []
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
        if rr and rr.group(1) == "bull_flip":
            out.append(
                BullFlip(
                    date=date,
                    method=method or "?",
                    symbol=rr.group(2),
                    description=re.sub(r"\s+", " ", rr.group(3)).strip(),
                    before=int(rr.group(4)),
                    after=int(rr.group(5)),
                )
            )
    return out


def collect_bull_flips(briefs: Iterable[str]) -> List[BullFlip]:
    """Flatten bull flips across many brief versions, de-duped by (date, method, symbol)."""
    seen = set()
    records: List[BullFlip] = []
    for text in briefs:
        for flip in bull_flips_in_brief(text):
            key = (flip.date, flip.method, flip.symbol)
            if key in seen:
                continue
            seen.add(key)
            records.append(flip)
    return records


def _run(args: Sequence[str]) -> str:
    return subprocess.run(
        args, capture_output=True, text=True, check=True
    ).stdout


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


def build_report(records: Sequence[BullFlip], *, since_label: str) -> str:
    """Render a newest-first, grouped-by-date bull-flips report."""
    lines: List[str] = []
    lines.append("BULL FLIPS from daily_brief.txt (SIGNAL FLIPS section)")
    lines.append(f"window: {since_label}   events: {len(records)}")
    lines.append("change: state before->after  (1=bull, 0=neutral, -1=bear)")
    lines.append("")
    by_date: dict[str, List[BullFlip]] = {}
    for f in records:
        by_date.setdefault(f.date, []).append(f)
    for date in sorted(by_date, reverse=True):
        day = sorted(by_date[date], key=lambda f: (f.symbol, f.method))
        lines.append(f"-- {date}  ({len(day)} bull flips) --")
        for f in day:
            lines.append(
                f"   {f.symbol:<9} {f.before}->{f.after:<4} {f.method:<13} {f.description}"
            )
        lines.append("")
    if records:
        freq = Counter(f.symbol for f in records)
        lines.append("Most frequent flippers: "
                     + ", ".join(f"{s}x{c}" for s, c in freq.most_common(10)))
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Reconstruct bull flips from daily_brief.txt history.")
    parser.add_argument("--since", default=DEFAULT_SINCE,
                        help="git --since window (default '1 month ago')")
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="git revision to read history from (default origin/main)")
    parser.add_argument("--file", default=BRIEF_FILE)
    parser.add_argument("--full-only", action="store_true",
                        help="keep only fully-bullish flips (after == 1)")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"also write the report here (e.g. {DEFAULT_OUTPUT})")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    versions = iter_brief_versions(args.file, args.since, revision=args.revision)
    records = collect_bull_flips(versions)
    if args.full_only:
        records = [f for f in records if f.after == 1]
    logger.info("collected %d bull_flip events", len(records))

    report = build_report(records, since_label=args.since)
    print(report, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
