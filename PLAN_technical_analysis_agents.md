# Plan: Technical Analysis Multi-Agent System (LangGraph)

## Context

The project has two standalone CLI tools — `ask_bo_trader.py` (breakout) and `ask_ma_trader.py`
(MA crossover) — that each read a parquet file, build a per-symbol snapshot, and call OpenAI.
They work well but run sequentially and in isolation.

This plan wraps them in a LangGraph **manager + parallel subgraph** topology so both agents
run concurrently, share data through typed state, and their results flow into a synthesised
output — replacing the separate CLI invocations with a single composable graph.

The manager can:
1- reads from
`RESULTS_PATH = Path("data/results/it/analysis_results.parquet")` and distributes data to
subagents via `payload_json` in the graph state (no disk reads in the subgraph hot path).

2a- download data in real time with the following

``` python
from algoshort.yfinance_handler import YFinanceDataHandler
from algoshort.ohlcprocessor import OHLCProcessor
from algoshort.wrappers import generate_signals, calculate_return
from algoshort.stop_loss import StopLossCalculator

from ta.breakout.bo_snapshot import build_snapshot
from ask_bo_trader import ask_bo_trader
from pipeline import (
    REGIME_COL,
    load_config,
    build_search_spaces,
    # load_data,
    # build_symbol_dfs,
    # compute_relative_prices,
    # generate_all_signals,
    # run_grid_search,
    # calculate_returns,
    # calculate_stop_losses,
    # calculate_position_sizing,
    # extract_cumul_snapshot,
    # save_results,
)

import rich

import logging
from pathlib import Path
from datetime import date 
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,              # or DEBUG / WARNING / ERROR
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

stock = 'TCEHY'
benchmark = 'H4ZX.DE' # HSBC Hang Seng Tech UCITS ETF (H4ZX.DE)
fx = 'EURUSD=X' # EUR/USD exchange rate in us dollars

ticker_list = [stock, benchmark, fx]


massive_handler = YFinanceDataHandler(
    cache_dir="../data/ohlc/it",  # Cache directory
    enable_logging=True,
    chunk_size=20,                       # Smaller chunks for stability
    log_level=logging.INFO
)

today = date.today()

data = massive_handler.download_data(
    symbols=ticker_list,  # Example ticker
    start='2016-01-01',  # Start date for historical data
    # end='2026-02-20',    # End date for historical data
    end = today,
    # period='5y',           # 5 years of historical data
    interval='1d',         # Daily data
    use_cache=False,        # Use cache to avoid re-downloading
    threads=True           # Enable multi-threading
)
```

2b - data must be enriched using the following code:

``` python
CONFIG_PATH = Path("config.json")
cfg = load_config(CONFIG_PATH)
tt_search_space, bo_search_space, ma_search_space = build_search_spaces(cfg)

dfs = massive_handler.get_ohlc_data(benchmark)
bmk = massive_handler.get_ohlc_data(stock)
fx = massive_handler.get_ohlc_data(fx)
fx = fx.reset_index()
fx = fx.rename(columns={'close': 'fx_close'})
fx = fx[['date', 'fx_close']]

dfs = pd.merge(
    dfs, 
    fx, 
    how='left', 
    on='date'
)

dfs['close'] = dfs['close'] / dfs['fx_close']
dfs = dfs.drop(columns=['fx_close'])

processor = OHLCProcessor()
dfs = processor.calculate_relative_prices(dfs, bmk)

dfs, signal_columns = generate_signals(
    df=dfs,
    config_path='./config.json',
    tt_search_space=tt_search_space,
    bo_search_space=bo_search_space,
    ma_search_space=ma_search_space
)


dfs = calculate_return(dfs, signal_columns)

stop_loss_cfg: dict = cfg["stop_loss"]
calc = StopLossCalculator(dfs)

for signal in signal_columns:
    calc.data = calc.atr_stop_loss(
        signal=signal,
        window=stop_loss_cfg["atr_window"],
        multiplier=stop_loss_cfg["atr_multiplier"],
    )

dfs = calc.data
dfs["symbol"] = stock
```

2c- now data are ready to be snapshotted and sent to be trader

``` python
snapshot = build_snapshot(dfs)
bo_analysis =ask_bo_trader(snapshot, ticker=stock)
rich.print(bo_analysis)
```


---

## Architecture

```
START
  → set_config          (inject runtime config into state)
  → prepare_node        (read parquet, build snapshot JSON for all symbols)
      ├─► breakout_worker  (score/rank breakout signals)
      └─► ma_worker        (score/rank MA signals)
  → synthesise_node     (combine rankings, format brief)
END
```

- `set_config → prepare`: linear preamble
- `prepare → [breakout_worker, ma_worker]`: fan-out (concurrent)
- `[breakout_worker, ma_worker] → synthesise`: fan-in
- Every state field has `_last` reducer (required for parallel merge)

---

## File Layout

```
agents/
├── __init__.py               # re-exports create_manager only
├── agent.py                  # create_manager() — main graph factory  ← build first
├── graph_state.py            # TechnicalAnalysisState TypedDict       ← build first
├── graph_nodes.py            # prepare_node, create_subgraph(), synthesise_node
├── _subagents.py             # WORKER_NAMES + build_subgraphs()
└── _tools/
    ├── __init__.py
    ├── prepare_tools.py      # load parquet → per-symbol snapshot dicts
    ├── breakout_tools.py     # score_breakout(), rank_breakout_signals()
    └── ma_tools.py           # score_ma(), rank_ma_signals()
```

---

## Build Order

Build the main graph contract first, then subgraphs.

```
Step 1 — agents/graph_state.py            State schema + _last reducers
Step 2 — agents/agent.py                  Main graph skeleton (stubs acceptable)
Step 3 — agents/graph_nodes.py            prepare_node, create_subgraph(), synthesise_node
Step 4 — agents/_subagents.py             WORKER_NAMES + build_subgraphs()
Step 5 — agents/_tools/prepare_tools.py  Parquet loading + snapshot building
Step 6 — agents/_tools/breakout_tools.py Breakout scoring + ranking
Step 7 — agents/_tools/ma_tools.py       MA scoring + ranking
```

---

## Step 1 — State Schema (`graph_state.py`)

```python
def _last(a, b): return b

class TechnicalAnalysisState(TypedDict, total=False):
    # ── Caller inputs ──────────────────────────────────────────────────
    analysis_date:   Annotated[Optional[str],  _last]  # ISO date; None → latest
    symbols_filter:  Annotated[Optional[list], _last]  # None → all symbols
    top_n:           Annotated[int,            _last]  # symbols to rank (default 10)
    data_source:     Annotated[str,            _last]  # "parquet" (default) | "live"
    # Mode A (parquet): reads data/results/it/analysis_results.parquet
    # Mode B (live):    downloads via YFinanceDataHandler for symbols_filter tickers

    # ── Set by set_config ──────────────────────────────────────────────
    resolved_date:   Annotated[str,  _last]  # actual date resolved from data

    # ── Set by prepare_node ────────────────────────────────────────────
    payload_json:    Annotated[str,  _last]
    # payload_json shape:
    # {
    #   "date": "YYYY-MM-DD",
    #   "n_symbols": int,
    #   "breakout_snapshots": {symbol: dict},   ← from ta.breakout.bo_snapshot.build_snapshot()
    #   "ma_snapshots":       {symbol: dict},   ← from ta.ma.ma_snapshot.build_snapshot()
    # }

    # ── Set by subgraph workers ────────────────────────────────────────
    breakout_result: Annotated[Optional[dict], _last]
    ma_result:       Annotated[Optional[dict], _last]
    # Each result shape:
    # {
    #   "top_symbols": [{"symbol": str, "score": float, "signals": [...], ...}],
    #   "signal_flips": [{"symbol": str, "signal": str}],
    #   "regime_aligned_count": int
    # }

    # ── Set by synthesise_node ─────────────────────────────────────────
    final_output:    Annotated[str, _last]
```

**Invariant**: `payload_json` is the sole data channel from `prepare_node` to subgraphs.
After `prepare_node`, no subgraph reads from disk.

---

## Step 2 — Main Graph (`agent.py`)

```python
def create_manager(
    analysis_date: str | None = None,
    symbols_filter: list[str] | None = None,
    top_n: int = 10,
    data_source: str = "parquet",  # "parquet" | "live"
    checkpointer=None,
) -> CompiledStateGraph:
```

`WORKER_NAMES = ["breakout", "ma"]` drives all fan-out/fan-in loops — adding a third
subagent means one line in `_subagents.py`, no wiring changes.

---

## Step 3 — Node Implementations (`graph_nodes.py`)

### `prepare_node`

**Raises** `ValueError` on failure (no valid state = subgraphs cannot run).

Two data modes selected by `state["data_source"]` (default `"parquet"`):

**Mode A — parquet (default, IT universe):**
```
1. Parse state: analysis_date, symbols_filter, top_n
2. Call prepare_tools.load_analysis_data(RESULTS_PATH, analysis_date, symbols_filter)
   → (resolved_date, {symbol: df})
3. For each symbol:
   a. bo_snapshot.build_snapshot(df_symbol)   → bo_snapshot_dict   # ta.breakout.bo_snapshot
   b. ma_snapshot.build_snapshot(df_symbol)   → ma_snapshot_dict   # ta.ma.ma_snapshot
4. Serialize to payload_json
5. Return {"payload_json": ..., "resolved_date": ...}
```

**Mode B — live download (single ticker, real-time):**
```
1. Parse state: symbols_filter (must be non-empty list), top_n
   Requires: benchmark ticker ("H4ZX.DE") and fx ticker ("EURUSD=X") added automatically
2. Call prepare_tools.load_live_data(symbols_filter)
   → YFinanceDataHandler.download_data(ticker_list, start, end, interval='1d')
   → OHLCProcessor.calculate_relative_prices(dfs, bmk)   # bmk = symbols_filter[0]
   → generate_signals(dfs, config_path, search_spaces)
   → calculate_return(dfs, signal_columns)
   → StopLossCalculator ATR stop-loss for each signal
   → {symbol: df}  (resolved_date = today)
3. Same as Mode A step 3 onwards
```

> **Note on 2b variable naming**: in the plan's code snippet, `dfs` is the benchmark ETF
> (H4ZX.DE) and `bmk` is the stock under analysis — the names are swapped vs. convention.
> `load_live_data` will use the canonical names: `df_stock`, `df_benchmark`, `df_fx`.

Both `build_snapshot()` calls compute all TA enrichments (ADX, RSI, range quality, volume
profile) **once** in `prepare_node`. Workers reuse the pre-computed results from the snapshot
— no redundant computation.

### `create_subgraph(worker_name)` factory

Single-node StateGraph: `START → run_worker → END`.

`run_worker` deserializes `payload_json`, calls the worker tool, **never raises**
(errors → `{"error": str(exc)}`), returns `{f"{worker_name}_result": result_dict}`.

### `synthesise_node`

Rule-based for v1 (no LLM — validates results before adding latency/cost).

```
1. Read breakout_result and ma_result via state.get() — handles None/error gracefully
2. Rank tables: top breakout symbols, top MA symbols
3. Overlap: symbols in both → highest conviction ("double-confirmed")
4. Format structured brief:
   - Date + universe summary
   - Double-confirmed setups
   - Top breakout candidates with scores
   - Top MA candidates with scores
   - Signal flips (new entries today)
5. Return {"final_output": brief_text}
```

---

## Step 4 — Subagent Registry (`_subagents.py`)

```python
WORKER_NAMES: list[str] = ["breakout", "ma"]

def build_subgraphs() -> dict[str, CompiledStateGraph]:
    return {name: create_subgraph(name) for name in WORKER_NAMES}
```

---

## Step 5 — Prepare Tools (`_tools/prepare_tools.py`)

```python
RESULTS_PATH = Path("data/results/it/analysis_results.parquet")
HISTORY_BARS = 300  # enough for ADX(14), MA(150), RSI(14), slope windows
BENCHMARK_LIVE = "H4ZX.DE"   # Hang Seng Tech ETF — benchmark for live mode
FX_LIVE        = "EURUSD=X"  # EUR/USD — FX conversion for live mode

def load_analysis_data(
    path: Path,
    analysis_date: str | None,
    symbols_filter: list[str] | None,
) -> tuple[str, dict[str, pd.DataFrame]]:
    """
    Mode A. Load parquet, resolve analysis date, return (resolved_date, {symbol: df}).
    Keeps last HISTORY_BARS per symbol. Excludes benchmark (FTSEMIB.MI).
    """

def load_live_data(
    symbols: list[str],
    config_path: Path = Path("config.json"),
) -> tuple[str, dict[str, pd.DataFrame]]:
    """
    Mode B. Download live OHLC + enrich with signals/stops.
    Appends BENCHMARK_LIVE and FX_LIVE to the download list automatically.
    Returns (today_iso, {symbol: enriched_df}).

    Pipeline:
      YFinanceDataHandler.download_data(symbols + [BENCHMARK_LIVE, FX_LIVE], ...)
      → OHLCProcessor.calculate_relative_prices(df_stock, df_benchmark)
      → generate_signals(dfs, config_path, search_spaces)
      → calculate_return(dfs, signal_columns)
      → StopLossCalculator ATR stop-loss per signal
    """
```

**Reuses directly** (no copying):
- `ta.breakout.bo_snapshot.build_snapshot`
- `ta.ma.ma_snapshot.build_snapshot`
- `algoshort.YFinanceDataHandler`, `algoshort.OHLCProcessor`
- `algoshort.wrappers.generate_signals`, `algoshort.wrappers.calculate_return`
- `algoshort.StopLossCalculator`

---

## Step 6 — Breakout Tools (`_tools/breakout_tools.py`)

```python
def score_breakout(snapshot: dict) -> float:
    """
    Deterministic conviction score [0.0–1.0]:
      n_active_long_signals (rbo_150/50/20, rtt_5020 == +1)   +0–0.40
      regime_aligned (rrg == +1)                               +0.15
      fresh_flip today (any signal → +1)                       +0.15
      range_quality.n_resistance_touches >= 2                  +0.10
      volatility_compression.is_compressed                     +0.10
      volume_profile.breakout_confirmed                        +0.10
    """

def rank_breakout_signals(snapshots: dict[str, dict], top_n: int) -> dict:
    """Score all symbols, sort descending, return top_n with metadata."""
```

TA enrichments (`RangeSetup`, `VolatilityState`) are already embedded in the snapshot
by `prepare_node` — no re-computation in this layer.

---

## Step 7 — MA Tools (`_tools/ma_tools.py`)

```python
def score_ma(snapshot: dict) -> float:
    """
    Deterministic conviction score [0.0–1.0]:
      n_active_long_ma_signals (6 EMA/SMA == +1)   +0–0.40
      regime_aligned (rrg == +1)                    +0.15
      fresh_flip today (any MA signal → +1)         +0.15
      trend_strength.is_trending                    +0.10
      trend_strength.adx >= 25                      +0.10
      volume_profile.is_confirmed                   +0.10
    """

def rank_ma_signals(snapshots: dict[str, dict], top_n: int) -> dict:
    """Score all symbols, sort descending, return top_n with metadata."""
```

---

## Dependencies to Add to `requirements.txt`

```
langgraph>=0.2
langchain-core>=0.2
```

No `langchain-openai` needed for v1 — `synthesise_node` is rule-based.
LLM enrichment for top-N symbols is planned for v2.

---

## Error Handling

| Node | Failure mode | Action |
|------|-------------|--------|
| `prepare_node` | Missing parquet / bad date / no valid symbols | **Raises `ValueError`** — aborts graph |
| `breakout_worker` | Exception in scoring | Returns `{"breakout_result": {"error": "..."}}` — MA worker continues |
| `ma_worker` | Exception in scoring | Returns `{"ma_result": {"error": "..."}}` — breakout worker continues |
| `synthesise_node` | Either result is `None` / error dict | Formats `"unavailable"` for that section — never raises |

---

## Observability

All log entries carry `resolved_date` as the correlation key.

| Node | What is logged |
|------|---------------|
| `prepare_node` | Resolved date, n_symbols loaded, n_symbols with valid snapshots, duration |
| `breakout_worker` | n_symbols scored, top 3 by score, n_regime_aligned, n_signal_flips |
| `ma_worker` | Same as breakout_worker |
| `synthesise_node` | n_overlap symbols, brief character count |

---

## TDD Plan

Write tests **before** implementation. Confirm each test fails before writing the target code.

| Test file | Covers |
|-----------|--------|
| `tests/agents/test_graph_state.py` | `_last` reducer merges correctly across parallel branches |
| `tests/agents/test_prepare_tools.py` | Date resolution, symbol filtering, `symbols_filter`, invalid date raises |
| `tests/agents/test_breakout_tools.py` | `score_breakout` in [0,1]; regime-aligned scores higher; correct ordering |
| `tests/agents/test_ma_tools.py` | Same pattern for MA scoring |
| `tests/agents/test_graph_nodes.py` | `prepare_node` raises on bad date; subgraph node returns `{"error": ...}` |
| `tests/agents/test_agent.py` | Full graph `invoke` with fixture data returns `{"final_output": str}` |

All unit tests use **deterministic fixture DataFrames** — no real parquet reads.
Integration tests reading the real parquet are marked `@pytest.mark.integration`.

---

## Critical Files (Reuse, Do Not Copy)

| File | What to reuse |
|------|--------------|
| `ta/breakout/bo_snapshot.py` | `select_columns()`, `build_snapshot()` |
| `ta/ma/ma_snapshot.py` | `select_columns()`, `build_snapshot()` |
| `ta/breakout/range_quality.py` | `RangeSetup`, `VolatilityState` (snapshot field types) |
| `ta/ma/trend_quality.py` | `MATrendStrength` (snapshot field types) |
| `ta/ma/volume.py` | `MAVolumeProfile` (snapshot field types) |

---

## Verification

```bash
# Unit tests (fixture data, no parquet needed)
pytest tests/agents/ -v -m "not integration"

# Integration test (reads real parquet)
pytest tests/agents/test_agent.py -v -m integration

# Smoke test
python -c "
from agents import create_manager
g = create_manager()
r = g.invoke({'analysis_date': None, 'top_n': 5})
print(r['final_output'])
"
```

Expected: structured brief with date header, double-confirmed setups, ranked candidates.
Target runtime: < 30 s on a warm filesystem cache.

---

## Out of Scope (v2)

- LLM enrichment per symbol (call `ask_bo_trader` / `ask_ma_trader` for top N only)
- Conditional routing: skip MA worker if breakout worker errored
- Streaming output from `synthesise_node`
- Persisting results to `signals/daily_brief.txt`
