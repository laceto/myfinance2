"""
Analysis pipeline steps for Italian equity signal generation.

Each function is a self-contained, stateless transformation that takes
DataFrames in and returns DataFrames out, making the pipeline easy to
test and reorder independently.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from algoshort.ohlcprocessor import OHLCProcessor
from algoshort.wrappers import generate_signals
from tqdm import tqdm

from algoshort.combiner import HybridSignalCombiner, SignalGridSearch
from algoshort.returns import ReturnsCalculator
from algoshort.stop_loss import StopLossCalculator
from algoshort.position_sizing import PositionSizing

# ---------------------------------------------------------------------------
# Column name constants — change once here if the schema ever evolves
# ---------------------------------------------------------------------------
OHLC_COLS = {"open": "open", "high": "high", "low": "low", "close": "close"}
RELATIVE_PREFIX = "r"
REGIME_COL = "rrg"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_search_spaces(cfg: dict) -> tuple[dict, list, dict]:
    turtle = cfg["regimes"]["turtle"]
    tt_search_space = {
        "fast": [turtle["fast_window"]],
        "slow": [turtle["slow_window"]],
    }

    bo_search_space = [cfg["regimes"]["breakout"]["bo_window"]]

    ma = cfg["regimes"]["ma_crossover"]
    ma_search_space = {
        "short_ma": [ma["short_window"]],
        "medium_ma": [ma["medium_window"]],
        "long_ma": [ma["long_window"]],
    }

    return tt_search_space, bo_search_space, ma_search_space


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path: Path, benchmark: str) -> tuple[pd.DataFrame, list[str]]:
    log.info("Loading OHLC data from %s", data_path)
    ohlc_data = pd.read_parquet(data_path)
    symbols = [s for s in ohlc_data["symbol"].unique() if s != benchmark]
    symbols = symbols[:1]  # limit to first symbol for testing; remove for full run
    log.info("Found %d symbols (excluding benchmark %s)", len(symbols), benchmark)
    return ohlc_data, symbols


def build_symbol_dfs(ohlc_data: pd.DataFrame, symbols: list[str]) -> list[pd.DataFrame]:
    dfs = []
    for symbol in symbols:
        df = ohlc_data[ohlc_data["symbol"] == symbol].copy()
        df["fx"] = 1
        dfs.append(df)
    return dfs


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def compute_relative_prices(
    dfs: list[pd.DataFrame], bmk: pd.DataFrame
) -> list[pd.DataFrame]:
    processor = OHLCProcessor()
    return [
        processor.calculate_relative_prices(stock_data=df, benchmark_data=bmk)
        for df in dfs
    ]


def generate_all_signals(
    dfs: list[pd.DataFrame],
    tt_search_space: dict,
    bo_search_space: list,
    ma_search_space: dict,
) -> tuple[list[pd.DataFrame], list[str]]:
    result_dfs = []
    signal_sets: list[set] = []

    for df in dfs:
        updated_df, signal_columns = generate_signals(
            df=df,
            tt_search_space=tt_search_space,
            bo_search_space=bo_search_space,
            ma_search_space=ma_search_space,
        )
        result_dfs.append(updated_df)
        signal_sets.append(set(signal_columns) - {REGIME_COL})

    # Intersection guarantees every column exists in every dataframe.
    # Union would cause KeyErrors downstream for symbols missing a column.
    common_signals = list(set.intersection(*signal_sets)) if signal_sets else []
    log.info("Common signal columns across all symbols: %s", sorted(common_signals))
    return result_dfs, common_signals


def run_grid_search(
    dfs: list[pd.DataFrame], signal_columns: list[str]
) -> tuple[list[pd.DataFrame], list[str]]:
    # Generate the combination grid once — it only depends on the signal
    # columns, not on any individual stock's data.
    searcher = SignalGridSearch(
        df=dfs[0],
        available_signals=signal_columns,
        direction_col=REGIME_COL,
    )
    grid = searcher.generate_grid()
    # Extract combined signal names before the loop so callers can merge them
    # with the original signal list and pass everything to downstream stages.
    combined_signal_names = [combo["name"] for combo in grid]
    log.info("Grid search: %d combinations × %d symbols", len(grid), len(dfs))

    result_dfs = []
    for df in dfs:
        symbol = df["symbol"].iloc[0]
        for combo in tqdm(grid, desc=symbol, leave=False):
            combiner = HybridSignalCombiner(
                direction_col=REGIME_COL,
                entry_col=combo["entry"],
                exit_col=combo["exit"],
                verbose=False,
            )
            output_col = combo["name"]
            df = combiner.combine_signals(
                df,
                output_col=output_col,
                allow_flips=True,
                require_regime_alignment=True,
            )
            df = combiner.add_signal_metadata(df, output_col)
        log.info("Grid search complete for %s: %d combinations applied", symbol, len(grid))
        result_dfs.append(df)

    return result_dfs, combined_signal_names


def calculate_returns(
    dfs: list[pd.DataFrame], signal_columns: list[str]
) -> list[pd.DataFrame]:
    result_dfs = []
    for df in dfs:
        calc = ReturnsCalculator(
            df,
            open_col=OHLC_COLS["open"],
            high_col=OHLC_COLS["high"],
            low_col=OHLC_COLS["low"],
            close_col=OHLC_COLS["close"],
            relative_prefix=RELATIVE_PREFIX,
        )
        for signal in signal_columns:
            df = calc.get_returns(df, signal=signal, relative=True, inplace=False)
        result_dfs.append(df)
    return result_dfs


def calculate_stop_losses(
    dfs: list[pd.DataFrame],
    signal_columns: list[str],
    atr_window: int,
    atr_multiplier: float,
) -> list[pd.DataFrame]:
    result_dfs = []
    for df in dfs:
        calc = StopLossCalculator(df)
        for signal in signal_columns:
            calc.data = calc.atr_stop_loss(
                signal=signal,
                window=atr_window,
                multiplier=atr_multiplier,
            )
        # ATR requires a warm-up period of `atr_window` rows, leaving NaN in
        # those early stop-loss cells. ffill propagates the first computed
        # value forward; bfill then backfills the leading NaN rows (before
        # the first valid ATR) so position sizing never receives a NaN
        # risk-per-share value.
        sl_cols = [c for c in calc.data.columns if c.endswith("_stop_loss")]
        calc.data[sl_cols] = calc.data[sl_cols].ffill().bfill()
        result_dfs.append(calc.data)
    return result_dfs


def calculate_position_sizing(
    dfs: list[pd.DataFrame],
    signal_columns: list[str],
    sizer: PositionSizing,
) -> list[pd.DataFrame]:
    result_dfs = []
    for df in dfs:
        symbol = df["symbol"].iloc[0]
        for signal in signal_columns:
            df = sizer.calculate_shares_for_signal(
                df=df,
                signal=signal,
                daily_chg=f"{signal}_chg1D_fx",
                sl=f"{signal}_stop_loss",
                close=OHLC_COLS["close"],
            )
        log.info("Position sizing complete for %s", symbol)
        result_dfs.append(df)
    return result_dfs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(dfs: list[pd.DataFrame], output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(dfs, ignore_index=True)
    out_file = output_path / "analysis_results.parquet"
    combined.to_parquet(out_file, index=False)
    log.info(
        "Saved combined results to %s (%d rows, %d symbols)",
        out_file,
        len(combined),
        len(dfs),
    )
