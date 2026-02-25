import logging
import sys
from pathlib import Path

import pandas as pd

# algoshort prints unicode characters (e.g. ✓) that cp1252 (Windows default)
# cannot encode. Reconfigure stdout/stderr to UTF-8 before any import that
# triggers those prints.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from algoshort.combiner import HybridSignalCombiner
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
# Paths
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("config.json")
DATA_PATH = Path("./data/ohlc/historical/it/ohlc_data.parquet")
OUTPUT_PATH = Path("./data/results/it/")
TRADE_SUMMARY_PATH = OUTPUT_PATH / "trade_summary.xlsx"
CUMUL_SNAPSHOT_PATH = OUTPUT_PATH / "cumul_snapshot.xlsx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

cfg = load_config(CONFIG_PATH)
benchmark: str = cfg["benchmark"]
stop_loss_cfg: dict = cfg["stop_loss"]
ps_cfg: dict = cfg["position_sizing"]

tt_search_space, bo_search_space, ma_search_space = build_search_spaces(cfg)

ohlc_data, symbols = load_data(DATA_PATH, benchmark)
bmk = ohlc_data[ohlc_data["symbol"] == benchmark].copy()

dfs = build_symbol_dfs(ohlc_data, symbols)

log.info("Computing relative prices for %d symbols", len(dfs))
dfs = compute_relative_prices(dfs, bmk)

log.info("Generating signals")
dfs, signal_columns = generate_all_signals(
    dfs, tt_search_space, bo_search_space, ma_search_space
)

log.info("Running grid search")
dfs, combined_signals = run_grid_search(dfs, signal_columns)
# Downstream stages receive both the original signals and every
# combined signal produced by the grid search.
all_signals = signal_columns 
# all_signals = signal_columns + combined_signals
log.info(
    "Total signals for downstream stages: %d original + %d combined = %d",
    len(signal_columns), len(combined_signals), len(all_signals),
)

# -----------------------------------------------------------------------
# Trade summary — one row per (symbol, signal), sorted by total_trades asc
# -----------------------------------------------------------------------
log.info("Computing trade summaries")
# entry_col / exit_col are not referenced inside get_trade_summary /
# add_signal_metadata, so one instance is reused across all signals.
_summarizer = HybridSignalCombiner(
    direction_col=REGIME_COL,
    entry_col=all_signals[0],
    exit_col=all_signals[0],
)
_summary_rows = []
for _df in dfs:
    _symbol = _df["symbol"].iloc[0]
    for _signal in all_signals:
        # Pass only the signal column to avoid a full-DataFrame copy.
        _summary = _summarizer.get_trade_summary(_df[[_signal]].copy(), _signal)
        _summary_rows.append({"symbol": _symbol, "signal": _signal, **_summary})

trade_summary = (
    pd.DataFrame(_summary_rows)
    .sort_values("total_entries", ascending=True)
    .reset_index(drop=True)
)
trade_summary = trade_summary[["symbol", "signal", "total_entries"]]
print(trade_summary.to_string())
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
trade_summary.to_excel(OUTPUT_PATH / "trade_summary.xlsx", index=False)
log.info("Trade summary saved to %s", OUTPUT_PATH / "trade_summary.xlsx")

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
cumul_snapshot.to_excel(CUMUL_SNAPSHOT_PATH, index=False)
log.info("Cumulative return snapshot saved to %s", CUMUL_SNAPSHOT_PATH)

# log.info("Calculating stop losses")
# dfs = calculate_stop_losses(
#     dfs,
#     all_signals,
#     atr_window=stop_loss_cfg["atr_window"],
#     atr_multiplier=stop_loss_cfg["atr_multiplier"],
# )

# # log.info("Calculating position sizing")
# # sizer = PositionSizing(
# #     tolerance=-0.1,
# #     mn=0.0025,
# #     mx=0.05,
# #     equal_weight=ps_cfg["equal_weight"],
# #     avg=0.03,
# #     lot=ps_cfg["lot"],
# # )
# # dfs = calculate_position_sizing(dfs, all_signals, sizer)

save_results(dfs, OUTPUT_PATH)
