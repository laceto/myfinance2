# Architecture Reference — myfinance2

Quantitative finance project focused on Italian stock market (Borsa Italiana) analysis
with automated daily data collection via GitHub Actions. Daily OHLC collection now
covers two markets — `it` (Borsa Italiana) and `etf` (ETF universe) — sharing the same
pipeline shape; downstream analysis currently runs for `it` only (see CI section).

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

`download_daily_ohlc.yml` runs **Monday–Friday at 21:00 UTC** (after Borsa
Italiana close) and downloads **two markets** as independent jobs:

| Market | Ticker source | Download script | Output paths |
|--------|---------------|-----------------|--------------|
| `it` | `data/ticker/it/ticker.xlsx` | `get_daily_ohlc_data.py` | `data/ohlc/{today,historical}/it/` |
| `etf` | `data/ticker/etf/ticker_active.xlsx` | `get_daily_ohlc_data_etf.py` | `data/ohlc/{today,historical}/etf/` |

Each market runs as an **independent matrix leg** (`fail-fast: false`):
1. Install dependencies (including the wheel)
2. Run that market's download script
3. Append today's bar via `append_daily_to_historical.py --market <it|etf>`
   (single, unit-tested source of truth — see `tests/test_append_daily_to_historical.py`)
4. Commit and push **only that market's** two Parquet files with `[skip ci]`

**Why two independent legs, not one job:** the markets must not block each
other — an `etf` download failure should never stop the `it` update (or vice
versa). Each leg commits its own files. Because the two legs run in parallel and
push to the same branch, the commit step **rebase-retries** on a rejected push
(they touch disjoint files, so the rebase never conflicts). A leg that fails to
download fails only its own market, before committing — no partial/stale data.

**ETF cold start (one-time backfill):** the `etf` market's *active universe* is
the curated 15 largest UCITS ETFs by AUM (the reviewed core) plus 100 more
tickers drawn from the justETF list, minus symbols that returned no Yahoo data
(pruned via `KNOWN_DEAD`), **plus** the top-100 funds by real AUM from
`data/ticker/etf/profiles.jsonl` (justETF export; `fund_size_eur_mln` mapped to
Yahoo tickers by name via `etf_profiles.py`). At `DEFAULT_EXTRA_COUNT=550` the
universe is **559** tickers (core + justETF list-order extras + AUM top-100,
deduped; known-dead pruned).

| Concern | Where |
|---------|-------|
| Curated 15-ETF core (single source of truth) | `etf_top15.py` (`TOP15_ETFS`, ordered by AUM; `TOP15_AS_OF` provenance) |
| Active universe = core + `DEFAULT_EXTRA_COUNT` justETF extras | `etf_universe.py` (`build_active_universe_table`, `write_active_seed`) |
| Seed ticker file (name, ticker) | `data/ticker/etf/ticker_active.xlsx` — read by **both** the cold-start and daily scripts |
| Historical backfill (2016 → today) | `cold_start_etf.py` → `data/ohlc/historical/etf/ohlc_data.parquet` |
| Trigger | `.github/workflows/cold_start_etf.yml` (`workflow_dispatch` only — never scheduled) |

The justETF extras are taken in list order (not AUM-ranked — AUM is not in the
repo), so expect a higher zero-row rate than the vetted core; `cold_start_etf.py`
retries each empty symbol once and reports the survivors at WARNING rather than
failing the run. Symbols confirmed dead in a run are added to `KNOWN_DEAD` in
`etf_universe.py` (pruned, not replaced). Tune the count via `DEFAULT_EXTRA_COUNT`
or swap the extras' source in `etf_universe.py`.

Run the cold start **once** (manually) to seed history; the scheduled daily
workflow then appends each trading day. `cold_start_etf.py` injects the Yahoo
client via a factory (lazy `algoshort` import) so its orchestration — download,
save, retry zero-row symbols once, fail-fast on an empty universe — is unit-
tested with a fake handler and no network (`tests/test_cold_start_etf.py`).
The AUM figures in `etf_top15.py` are approximate; refresh and regenerate the
seed with `python etf_top15.py`. The wider 2217-ticker `data/ticker/etf/ticker.xlsx`
(justETF universe) is retained for future full-universe use.

**ETF return analysis (daily):** `etf_returns.py` computes each ETF's
cumulative return over 1W/1M/3M/6M/1Y/3Y + YTD (and a 1D/1W short-term movers
view) from the historical parquet, ranking top out-performers and bottom
decliners per window. Returns are close-to-close, simple by default
(`--method log` for continuously-compounded); no lookahead (close on/before each
date), so funds with too little history drop out. An **illiquidity screen**
(`etf_liquidity.py`) drops thin symbols before ranking — default: median daily
traded value (close*volume) >= 50,000 and >= 80% active days over ~90d
(`--min-traded-value` / `--min-active` / `--liquidity-lookback` / `--no-liquidity-filter`).
A separate **split/re-denomination screen** drops a symbol's window return when the
close makes an implausible single-day move (>50% by default, `--max-daily-move` /
`--no-artifact-filter`) within that window — per-window, so a fund is only removed
from the windows its glitch spans. Flagged symbols (e.g. `LQQ.PA`, `JPXX.L`, `SPEQ.MI`)
are listed in `data/results/etf/flagged_artifacts.txt` for review. This is orthogonal
to the liquidity screen (those artifacts are liquid). `.github/workflows/etf_returns.yml`
runs it after each **Download Daily OHLC Data** completes (`workflow_run`) and
commits `data/results/etf/{returns_ranking.xlsx,returns_report.txt,short_term_movers.txt,flagged_artifacts.txt}`.
It is pure pandas over the committed parquet — no network or `algoshort` wheel.

**ETF sector rotation (on-demand, not in CI):** `etf_sector_rotation.py` is a
read-only companion that groups the liquid universe into sectors (keyword
classification on the fund name) and ranks them by *momentum acceleration*
(`accel_1M_vs_3M = 1M − 3M/3`) to surface money rotating **into** vs **out of**
sectors — the sector-level view that `returns_ranking.xlsx` (top-N funds only)
can't give. It reuses `etf_liquidity` and `etf_returns.compute_returns` so the
screens match the daily pipeline exactly. Run `python etf_sector_rotation.py`
(add `--output data/results/etf/sector_rotation.txt` to save, `-v` for logs).
Leveraged/inverse funds and thin buckets (`--min-funds`, default 2) are excluded
— small `n` is directional, not statistical. Deliberately **not** wired into any
workflow; logic is unit-tested in `tests/test_etf_sector_rotation.py`.

### CI concurrency model (invariant: workflows that can push concurrently rebase-retry)

`workflow_run` fans out: **one** successful **Download Daily OHLC Data** run
triggers **both** `analyze_and_report.yml` (results/it) **and**
`etf_returns.yml` (results/etf) at the same time. Both push commits to the same
default branch, so their pushes race even though they touch **disjoint files**.
The two download matrix legs (`it`, `etf`) likewise push in parallel.

**Invariant:** any workflow that commits and pushes to the shared branch **and
can run concurrently with another pusher** must use the rebase-retry loop, never
a plain `git push`. Concretely that is the three scheduled/`workflow_run`
pushers: the **download** legs, **analyze_and_report**, and **etf_returns**. The
pattern is:

```
git add <only this workflow's own files>
git diff --cached --quiet && exit 0        # nothing to commit is success
git commit -m "... [skip ci]"
for attempt in 1..5:
    git push origin HEAD:<ref> && exit 0
    git fetch origin <ref>
    git rebase --autostash origin/<ref> || { git rebase --abort; exit 1; }
```

Because the concurrent workflows write disjoint paths, the rebase never
conflicts — it just replays this workflow's commit on top of the other's.
**Failure mode this prevents:** without the loop, the second pusher gets
`! [rejected] ... (fetch first)` and the whole job fails (this was the
analyze-vs-etf_returns race). **When adding a new committing workflow that
fires on the same `workflow_run`, it must adopt this same loop** — otherwise it
reintroduces the race.

**`--autostash` is mandatory, not optional.** A job may run a script that
regenerates *tracked* files it deliberately does **not** commit — the analyze
job commits only 4 result files but `analyze_stock.py` also rewrites
`cumul_snapshot.xlsx`, `returns_dashboard.xlsx`, `trending_dashboard.xlsx`, and
`trending_heatmap.png`, leaving the working tree dirty. Plain `git rebase`
refuses on a dirty tree (`error: cannot rebase: You have unstaged changes`),
which failed the analyze job **every day** even though the analysis itself
succeeded. `--autostash` stashes those changes, rebases, and re-applies them; on
a clean tree it is a harmless no-op. (Reproduced and fixed with a two-repo git
race harness before shipping.)

`generate_symbol_notebooks.yml` is **`workflow_dispatch`-only** (manual). It
does not commit to the branch (it only uploads an artifact), so it is outside
the race. Its trigger is `workflow_dispatch:` — **never `on: {}`**, which is an
invalid trigger that makes GitHub emit a *startup-failure* on every push that
touches `.github/workflows/`.

`cold_start_etf.yml` also pushes (plain `git push`, no loop) but is the
deliberate **exception**: it is `workflow_dispatch`-only, never scheduled, and
is a one-time manual history overwrite you run when nothing else is pushing —
so it cannot race the daily fan-out. If it ever becomes scheduled or `workflow_run`-triggered, it must adopt the rebase-retry loop.

**Scope note:** CI currently downloads OHLC for both markets, but the
analysis/report pipeline (`analyze_stock.py` → `trading_report.py` →
`get_insights.py`, wired by `analyze_and_report.yml`) still targets `it` only.
`analyze_stock.py` already reads `universe.ohlc_historical_path` /
`universe.results_dir` from its config; extending analysis to `etf` requires
parametrizing `trading_report.py` and `get_insights.py` the same way.

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
