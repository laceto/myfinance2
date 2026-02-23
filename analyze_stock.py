import logging
import sys
from pathlib import Path

# algoshort prints unicode characters (e.g. ✓) that cp1252 (Windows default)
# cannot encode. Reconfigure stdout/stderr to UTF-8 before any import that
# triggers those prints.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from algoshort.position_sizing import PositionSizing

from pipeline import (
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
    save_results,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("config.json")
DATA_PATH = Path("./data/ohlc/historical/it/ohlc_data.parquet")
OUTPUT_PATH = Path("./data/results/it/")

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
dfs = run_grid_search(dfs, signal_columns)

log.info("Calculating returns")
dfs = calculate_returns(dfs, signal_columns)

log.info("Calculating stop losses")
dfs = calculate_stop_losses(
    dfs,
    signal_columns,
    atr_window=stop_loss_cfg["atr_window"],
    atr_multiplier=stop_loss_cfg["atr_multiplier"],
)

log.info("Calculating position sizing")
sizer = PositionSizing(
    tolerance=-0.1,
    mn=0.0025,
    mx=0.05,
    equal_weight=ps_cfg["equal_weight"],
    avg=0.03,
    lot=ps_cfg["lot"],
)
dfs = calculate_position_sizing(dfs, signal_columns, sizer)

save_results(dfs, OUTPUT_PATH)
