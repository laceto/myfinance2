import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# algoshort prints unicode characters (e.g. ✓) that cp1252 (Windows default)
# cannot encode. Reconfigure stdout/stderr to UTF-8 before any import that
# triggers those prints.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from algoshort.combiner import HybridSignalCombiner, SignalGridSearch
from algoshort.position_sizing import PositionSizing

from pipeline import (
    REGIME_COL,
    load_config,
    build_search_spaces,
    load_data,
    build_symbol_dfs,
    compute_relative_prices,
    generate_all_signals,
    run_grid_search,
    calculate_returns,
    calculate_stop_losses,
    calculate_position_sizing,
    extract_cumul_snapshot,
    save_results,
)

# ---------------------------------------------------------------------------
# Defaults (used when config has no universe section)
# ---------------------------------------------------------------------------
_DEFAULT_DATA_PATH  = Path("./data/ohlc/historical/it/ohlc_data.parquet")
_DEFAULT_OUTPUT_DIR = Path("./data/results/it")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the signal analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=Path("config.json"),
        help="Config JSON file. Optional universe.ohlc_historical_path and "
             "universe.results_dir keys override the default Italian-equity paths.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.config.exists():
        log.error("Config not found: %s", args.config)
        raise SystemExit(1)

    cfg = load_config(args.config)
    benchmark: str = cfg["benchmark"]
    stop_loss_cfg  = cfg["stop_loss"]

    universe   = cfg.get("universe", {})
    data_path  = Path(universe.get("ohlc_historical_path", str(_DEFAULT_DATA_PATH)))
    output_dir = Path(universe.get("results_dir",          str(_DEFAULT_OUTPUT_DIR)))

    tt_search_space, bo_search_space, ma_search_space = build_search_spaces(cfg)

    ohlc_data, symbols = load_data(data_path, benchmark)
    bmk = ohlc_data[ohlc_data["symbol"] == benchmark].copy()

    dfs = build_symbol_dfs(ohlc_data, symbols)

    log.info("Computing relative prices for %d symbols", len(dfs))
    dfs = compute_relative_prices(dfs, bmk)

    log.info("Generating signals")
    dfs, signal_columns = generate_all_signals(
        dfs, tt_search_space, bo_search_space, ma_search_space
    )

    # # log.info("Running grid search")
    # # dfs, combined_signals = run_grid_search(dfs, signal_columns)
    # # all_signals = signal_columns + combined_signals
    all_signals = signal_columns

    log.info("Calculating returns")
    dfs = calculate_returns(dfs, all_signals)

    log.info("Extracting cumulative return snapshots")
    cumul_snapshot = (
        extract_cumul_snapshot(dfs, all_signals)
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )
    print("\n--- Cumulative Return Snapshot (last bar) ---")
    print(cumul_snapshot.to_string(index=False))
    output_dir.mkdir(parents=True, exist_ok=True)
    cumul_snapshot.to_excel(output_dir / "cumul_snapshot.xlsx", index=False)
    log.info("Cumulative return snapshot saved to %s", output_dir / "cumul_snapshot.xlsx")

    log.info("Calculating stop losses")
    dfs = calculate_stop_losses(
        dfs,
        all_signals,
        atr_window=stop_loss_cfg["atr_window"],
        atr_multiplier=stop_loss_cfg["atr_multiplier"],
    )

    save_results(dfs, output_dir)


if __name__ == "__main__":
    main()
