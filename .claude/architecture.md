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

`algoshort` is a **private local wheel** (`algoshort-0.1.0-py3-none-any.whl`). Not on PyPI.

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

## Entry Point Scripts

| Script | Purpose |
|--------|---------|
| `get_daily_ohlc_data.py` | Download today's OHLC bar for all tickers |
| `get_historical_ohlc_data.py` | Download full history (2016 → present) |
| `analyze_stock.py` | Run signal analysis pipeline |
