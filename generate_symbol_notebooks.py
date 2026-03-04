"""
generate_symbol_notebooks.py
-----------------------------
Generate one executed Jupyter notebook per symbol using papermill.

Each output notebook is saved to:
    notebooks/it/<SYMBOL>/analysis.ipynb

Usage:
    python generate_symbol_notebooks.py                  # all symbols
    python generate_symbol_notebooks.py ENI.MI ENEL.MI   # selected symbols only
    python generate_symbol_notebooks.py --dry-run        # list symbols, no execution

Requirements:
    pip install papermill

Failure policy:
    Errors for individual symbols are logged and skipped so the batch
    continues. A summary is printed at the end.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
ANALYSIS_PATH = PROJECT_ROOT / "data" / "results" / "it" / "analysis_results.parquet"
TEMPLATE_PATH = PROJECT_ROOT / "symbol_analysis_template.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "it"


def _load_symbols(path: Path) -> list[str]:
    """Return sorted list of unique symbols from the analysis results."""
    df = pd.read_parquet(path, columns=["symbol"])
    return sorted(df["symbol"].unique().tolist())


def _output_path(symbol: str) -> Path:
    """Return the output notebook path for a given symbol."""
    # Replace dots with underscores to avoid filesystem edge-cases (e.g. ENI.MI → ENI_MI)
    safe_name = symbol.replace(".", "_")
    return OUTPUT_DIR / safe_name / "analysis.ipynb"


def _generate(symbol: str, dry_run: bool) -> bool:
    """Execute the template notebook for one symbol. Returns True on success."""
    out_path = _output_path(symbol)

    if dry_run:
        logger.info("[dry-run] would write: %s", out_path)
        return True

    try:
        import papermill as pm  # noqa: PLC0415  (import inside function to give helpful error)
    except ImportError:
        logger.error(
            "papermill is not installed. Run: pip install papermill"
        )
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating notebook for %s → %s", symbol, out_path)
    try:
        pm.execute_notebook(
            input_path=str(TEMPLATE_PATH),
            output_path=str(out_path),
            parameters={"SYMBOL": symbol},
            # kernel_name can be overridden via CLI if needed
        )
        return True
    except Exception:
        logger.exception("Failed to generate notebook for %s", symbol)
        return False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Symbols to process (default: all symbols in analysis_results.parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be generated without executing notebooks",
    )
    args = parser.parse_args()

    if not ANALYSIS_PATH.exists():
        logger.error("Analysis results not found: %s", ANALYSIS_PATH)
        sys.exit(1)

    if not TEMPLATE_PATH.exists():
        logger.error("Template notebook not found: %s", TEMPLATE_PATH)
        sys.exit(1)

    all_symbols = _load_symbols(ANALYSIS_PATH)

    if args.symbols:
        unknown = set(args.symbols) - set(all_symbols)
        if unknown:
            logger.error("Unknown symbols: %s", sorted(unknown))
            logger.error("Available symbols: %s", all_symbols)
            sys.exit(1)
        symbols = sorted(args.symbols)
    else:
        symbols = all_symbols

    logger.info("Processing %d symbol(s) — dry_run=%s", len(symbols), args.dry_run)

    succeeded = []
    failed = []

    for symbol in symbols:
        ok = _generate(symbol, dry_run=args.dry_run)
        (succeeded if ok else failed).append(symbol)

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"  Total   : {len(symbols)}")
    print(f"  Success : {len(succeeded)}")
    print(f"  Failed  : {len(failed)}")
    if failed:
        print(f"\n  Failed symbols:")
        for s in failed:
            print(f"    - {s}")
    print("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
