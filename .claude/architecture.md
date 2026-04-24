# Architecture Reference — myfinance2

Quantitative finance project focused on Italian stock market (Borsa Italiana) analysis
with automated daily data collection via GitHub Actions.

## Data Flow

```
data/ticker/it/ticker.xlsx          ← ticker list (Yahoo Finance symbols, e.g. AVIO.MI)
        │
        ▼
algoshort.YFinanceDataHandler       ← bulk download with caching and chunking
        │
        ├─→ data/ohlc/today/it/ohlc_data.parquet       (current day's bar)
        └─→ data/ohlc/historical/it/ohlc_data.parquet  (full history, appended daily by CI)
                │
                ▼
        algoshort.OHLCProcessor     ← relative prices vs benchmark (FTSEMIB.MI)
                │
                ▼
        wrappers.generate_signals   ← turtle / breakout / MA crossover signals
                │
                ▼
        SignalGridSearch (combiner) ← parallel grid search with regime alignment
                │
                ├─→ ReturnsCalculator    ← per-signal returns
                └─→ StopLossCalculator   ← ATR-based stop-loss levels
```

## algoshort Package Modules

| Module | Role |
|--------|------|
| `YFinanceDataHandler` | Wraps yfinance for bulk downloads with caching and chunking |
| `OHLCProcessor` | Computes relative prices (stock vs benchmark) |
| `wrappers.generate_signals` | Generates turtle/breakout/MA crossover signals |
| `SignalGridSearch` | Parallel grid search to combine signals with regime alignment |
| `ReturnsCalculator` | Computes returns for each signal |
| `StopLossCalculator` | Computes ATR-based stop-loss levels |

`algoshort` is a **private local wheel** (`algoshort-0.1.1-py3-none-any.whl`). Not on PyPI.

## Configuration (`config.json`)

Central parameter store — the canonical reference for all strategy values.

| Section | Key parameters |
|---------|---------------|
| `regimes` | `bo_window`, `fast/slow_window`, `short/medium/long_window`, `ma_type` |
| `stop_loss` | ATR window/multiplier, swing window, retracement level |
| `position_sizing` | starting capital, lot size, equal weight, amortized root |
| `benchmark` | `FTSEMIB.MI` |
| `metrics` | risk window (252 days), percentile, limit |

> **Note**: `config.json` is not loaded programmatically yet — parameters are hardcoded
> inline in scripts. The config is the source of truth; scripts should eventually read from it.

## Data Schema

All Parquet files use a **long/tidy format**:

| Column | Type | Notes |
|--------|------|-------|
| `symbol` | str | Yahoo Finance ticker, e.g. `AVIO.MI` |
| `date` | date | Trading date |
| `open` | float | |
| `high` | float | |
| `low` | float | |
| `close` | float | |
| `volume` | int | |

`.MI` suffix = Milan-listed (Borsa Italiana).

## CI / GitHub Actions

Workflow runs **Monday–Friday at 21:00 UTC** (after Borsa Italiana close):
1. Install dependencies (including the wheel)
2. Run `get_daily_ohlc_data.py`
3. Append today's bar to `data/ohlc/historical/it/ohlc_data.parquet`
4. Commit and push updated Parquet with `[skip ci]` in the message

Parquet files are committed directly to the repo as the persistence layer (no external DB).

## TA Packages (`ta/breakout/`, `ta/ma/`)

Pure analytical primitives — no AI, no CLI, no side effects.

### `ta/breakout/`

| Module | Exports |
|--------|---------|
| `range_quality.py` | `count_touches`, `classify_trend`, `assess_range` → `RangeSetup`, `measure_volatility_compression` → `VolatilityState` |
| `volume.py` | `assess_volume_profile` → `VolumeProfile` |
| `bo_snapshot.py` | `select_columns(df)`, `build_snapshot(df_ticker)`, `build_snapshot_from_parquet(ticker, path)` |

Enrichments in snapshot: `range_setup`, `volatility_compression`, `volume_profile`.

### `ta/ma/`

| Module | Exports |
|--------|---------|
| `trend_quality.py` | `assess_ma_trend` → `MATrendStrength` (RSI, ADX, MA gap) |
| `volume.py` | `assess_ma_volume` → `MAVolumeProfile` (crossover volume confirmation) |
| `ma_snapshot.py` | `select_columns(df)`, `build_snapshot(df_ticker)`, `build_snapshot_from_parquet(ticker, path)` |

Enrichments in snapshot: `trend_strength`, `volume_profile`.

### Shared entry-point contract (both snapshot modules)

- `build_snapshot(df_ticker)` — caller already holds a filtered DataFrame (`app.py`, `batch_trader.py`)
- `build_snapshot_from_parquet(ticker, data_path)` — caller has a ticker string; loads parquet internally (CLI, notebooks, one-off scripts)

Both raise `FileNotFoundError` (missing parquet) or `ValueError` (ticker not found).

## AI Trader Assistants

| Script | Scope | Snapshot source |
|--------|-------|----------------|
| `ask_bo_trader.py` | Range breakout (rbo/rhi/rlo signals, turtle) | `ta.breakout.bo_snapshot.build_snapshot_from_parquet` |
| `ask_ma_trader.py` | MA crossover signals (rema/rsma) | `ta.ma.ma_snapshot.build_snapshot_from_parquet` |
| `batch_trader.py` | Bulk run across all tickers, both strategies | loads parquet once; calls `build_snapshot(df)` directly |

## Entry Point Scripts

| Script | Purpose |
|--------|---------|
| `get_daily_ohlc_data.py` | Download today's OHLC bar for all tickers |
| `get_historical_ohlc_data.py` | Download full history (2016 → present) |
| `analyze_stock.py` | Run signal analysis pipeline |
| `ask_bo_trader.py` | CLI: breakout AI analysis for a single ticker |
| `ask_ma_trader.py` | CLI: MA crossover AI analysis for a single ticker |
| `batch_trader.py` | CLI: bulk AI analysis across the full universe |
| `run_ta_agents.py` | CLI: LangGraph multi-agent TA system (breakout + MA, parallel) |

## LangGraph Multi-Agent System (`agents/`)

Runs breakout and MA AI analysis concurrently for a **single ticker** via a LangGraph manager +
parallel subgraph topology. Each worker calls `ask_bo_trader` / `ask_ma_trader` (OpenAI) directly;
`synthesise_node` calls an LLM a third time to compile both reports into a structured brief
(Position Recommendation → Signal Confluence → Deep-Dives → Entry/Exit Plan → Bottom Line).

### Package layout

```
agents/
├── __init__.py          # re-exports create_manager
├── agent.py             # create_manager() — graph factory
├── graph_state.py       # TechnicalAnalysisState TypedDict + _last reducer
├── graph_nodes.py       # prepare_node, create_subgraph(), synthesise_node
├── _subagents.py        # WORKER_NAMES + build_subgraphs()
└── _tools/
    └── prepare_tools.py # load_analysis_data() + load_live_data()
```

### Graph topology

```
START → prepare_node → [breakout_worker, ma_worker] → synthesise_node → END
```

- `prepare_node → workers`: fan-out (concurrent)
- `workers → synthesise_node`: fan-in
- All state fields use `_last` reducer — required for parallel merge

### `create_manager()` — public API

```python
from agents import create_manager

graph = create_manager(
    symbol         = "A2A.MI",       # required — single ticker to analyse
    analysis_date  = None,           # ISO date; None → latest bar in parquet
    data_source    = "parquet",      # "parquet" | "live"
    benchmark      = "FTSEMIB.MI",   # Mode A: excluded from results; Mode B: relative-price base
    fx             = None,           # FX ticker for currency conversion; None = same currency
)
result = graph.invoke({...})        # returns TechnicalAnalysisState dict
brief  = result["final_output"]     # structured report compiled by LLM (markdown)
```

### Two data modes

| Mode | `data_source` | Description |
|------|--------------|-------------|
| A (default) | `"parquet"` | Reads `data/results/it/analysis_results.parquet`; loads history for the requested `symbol` |
| B (live) | `"live"` | Downloads via `YFinanceDataHandler`; `benchmark` used for `calculate_relative_prices`; `fx` triggers currency conversion when set |

### `TechnicalAnalysisState` key fields

| Field | Set by | Notes |
|-------|--------|-------|
| `symbol` | caller | required — single ticker to analyse (e.g. `"A2A.MI"`) |
| `benchmark` | caller | default `"FTSEMIB.MI"` |
| `fx` | caller | `None` = no FX conversion |
| `data_source` | caller | `"parquet"` or `"live"` |
| `payload_json` | `prepare_node` | sole data channel to workers; shape: `{"date", "symbol", "breakout_snapshot", "ma_snapshot"}` |
| `breakout_result` | `breakout_worker` | `TraderAnalysis.model_dump()` or `{"error": ...}` — never raises |
| `ma_result` | `ma_worker` | `MATraderAnalysis.model_dump()` or `{"error": ...}` — never raises |
| `final_output` | `synthesise_node` | LLM-compiled report (Position Rec → Scorecard → Deep-Dives → Entry/Exit → Bottom Line) |

**Invariant**: `payload_json` is the sole data channel from `prepare_node` to workers.
After `prepare_node` completes, no worker reads from disk or network.

### Dependencies added

```
langgraph>=0.2         (installed: 1.1.9)
langchain-core>=0.2    (installed: 1.3.1)
langchain-openai>=0.2  (installed: 1.2.0)   ← used by synthesise_node / ChatOpenAI
```

### CLI usage

```bash
# Mode A — latest bar from parquet
python run_ta_agents.py --symbol A2A.MI

# Mode A — specific date
python run_ta_agents.py --symbol ENI.MI --date 2026-04-14

# Mode B — live download, same currency (no FX step)
python run_ta_agents.py --live --symbol UCG.MI --benchmark FTSEMIB.MI

# Mode B — live download, EUR benchmark + USD stock with FX conversion
python run_ta_agents.py --live --symbol TCEHY --benchmark H4ZX.DE --fx EURUSD=X

# Save to file
python run_ta_agents.py --symbol A2A.MI --out data/results/it/daily_brief.txt
```
