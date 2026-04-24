"""
run_ta_agents.py — Run the multi-agent technical analysis system for a single ticker.

Usage:
    # Mode A — read from parquet, latest bar
    python run_ta_agents.py --symbol A2A.MI

    # Mode A — specific date
    python run_ta_agents.py --symbol ENI.MI --date 2026-04-23

    # Mode A — custom benchmark to exclude (default: FTSEMIB.MI)
    python run_ta_agents.py --symbol UCG.MI --benchmark FTSEMIB.MI

    # Mode B — live download: stock and benchmark in the same currency (no FX)
    python run_ta_agents.py --live --symbol UCG.MI --benchmark FTSEMIB.MI

    # Mode B — live download: benchmark in EUR, stock quoted in USD → FX conversion
    python run_ta_agents.py --live --symbol TCEHY --benchmark H4ZX.DE --fx EURUSD=X

    # Save brief to a file
    python run_ta_agents.py --symbol A2A.MI --out data/results/it/daily_brief.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the LangGraph multi-agent technical analysis system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--symbol",
        required=True,
        metavar="TICKER",
        help="Ticker symbol to analyse (e.g. A2A.MI).",
    )
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Analysis date (default: latest available bar in parquet).",
    )
    p.add_argument(
        "--benchmark",
        default="FTSEMIB.MI",
        metavar="TICKER",
        help=(
            "Benchmark ticker symbol. "
            "Mode A: excluded from the result set (default: FTSEMIB.MI). "
            "Mode B: used for relative-price computation (e.g. H4ZX.DE)."
        ),
    )
    p.add_argument(
        "--fx",
        default=None,
        metavar="TICKER",
        help=(
            "FX ticker for currency conversion (e.g. EURUSD=X). "
            "Omit when stock and benchmark share the same currency."
        ),
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Mode B: download live OHLC instead of reading the parquet.",
    )
    p.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Write the brief to this file (in addition to stdout).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from agents import create_manager

    data_source = "live" if args.live else "parquet"

    log.info(
        "Starting TA agents | symbol=%s mode=%s date=%s benchmark=%s fx=%s",
        args.symbol,
        data_source,
        args.date or "latest",
        args.benchmark,
        args.fx or "none",
    )

    t0 = time.perf_counter()
    graph = create_manager(
        symbol=args.symbol,
        analysis_date=args.date,
        data_source=data_source,
        benchmark=args.benchmark,
        fx=args.fx,
    )

    result = graph.invoke({
        "symbol":        args.symbol,
        "analysis_date": args.date,
        "data_source":   data_source,
        "benchmark":     args.benchmark,
        "fx":            args.fx,
    })

    elapsed = time.perf_counter() - t0
    log.info("Graph completed in %.1fs", elapsed)

    brief: str = result.get("final_output", "(no output)")
    print("\n" + brief)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(brief, encoding="utf-8")
        log.info("Brief written to %s", out_path)


if __name__ == "__main__":
    main()
