# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

The project uses a local `venv/` (gitignored) and requires the `algoshort` private package installed separately. Install dependencies:

```bash
pip install -r requirements.txt
pip install algoshort-0.1.0-py3-none-any.whl
# Additional packages not in requirements.txt but needed:
pip install yfinance openpyxl pyarrow joblib
```

The `algoshort` package is a local wheel file (`algoshort-0.1.0-py3-none-any.whl`) that must be installed manually. It is not on PyPI.

## Running Scripts

```bash
# Download today's OHLC data for Italian market tickers
python get_daily_ohlc_data.py

# Download full historical OHLC data (2016 to present)
python get_historical_ohlc_data.py

# Run stock signal analysis
python analyze_stock.py
```

## Architecture

This is a quantitative finance project focused on Italian stock market (Borsa Italiana) analysis with automated daily data collection via GitHub Actions.

### Data Flow

1. **Ticker list** is read from `data/ticker/it/ticker.xlsx` (Excel file with a `ticker` column of Yahoo Finance symbols, e.g. `AVIO.MI`)
2. **OHLC data** is downloaded from Yahoo Finance via `algoshort.YFinanceDataHandler` and saved as Parquet files:
   - `data/ohlc/today/it/ohlc_data.parquet` — current day's bar
   - `data/ohlc/historical/it/ohlc_data.parquet` — full historical dataset (appended daily by CI)
3. **Analysis** reads from the historical Parquet, computes relative prices vs benchmark (`FTSEMIB.MI`), generates trading signals, and calculates returns and stop-losses.

### `algoshort` Package Modules Used

- `YFinanceDataHandler` — wraps yfinance for bulk downloads with caching and chunking
- `OHLCProcessor` — computes relative prices (stock vs benchmark)
- `wrappers.generate_signals` — generates turtle/breakout/MA crossover signals
- `SignalGridSearch` (combiner) — parallel grid search to combine signals with regime alignment
- `ReturnsCalculator` — computes returns for each signal
- `StopLossCalculator` — computes ATR-based stop-loss levels

### Configuration (`config.json`)

Central parameter store for all strategy settings:
- **regimes**: parameters for breakout (`bo_window`), turtle (`fast/slow_window`), MA crossover (`short/medium/long_window`, `ma_type`), and floor-ceiling regimes
- **stop_loss**: ATR window/multiplier, swing window, retracement level
- **position_sizing**: starting capital, lot size, equal weight, amortized root
- **benchmark**: `FTSEMIB.MI` (FTSE MIB index)
- **metrics**: risk window (252 days), percentile, limit

`config.json` is not currently loaded programmatically in the scripts — parameters are hardcoded inline. The config serves as the canonical reference for parameter values.

### GitHub Actions CI

`.github/workflows/` contains a scheduled workflow that runs Monday–Friday at 21:00 UTC (after Borsa Italiana close). It:
1. Installs dependencies including the wheel file
2. Runs `get_daily_ohlc_data.py`
3. Appends today's data to `data/ohlc/historical/it/ohlc_data.parquet`
4. Commits and pushes the updated Parquet files with `[skip ci]` in the commit message

Parquet data files are committed directly to the repository as the persistence layer.

### Data Schema

All Parquet files use a long/tidy format with columns: `symbol`, `date`, `open`, `high`, `low`, `close`, `volume`. The `symbol` column uses Yahoo Finance tickers (`.MI` suffix for Milan-listed stocks).


### Coding Style

You are a Senior Staff Software Engineer and Architect who writes production-grade, maintainable, and high-performance code. Your work is guided by the principle that code is read far more often than it is written.

#### Core Engineering Principles
##### DRY (Don't Repeat Yourself)

Abstract repeated logic into reusable components
Avoid premature abstraction—only generalize when patterns emerge organically
Prioritize clarity and explicitness over cleverness

##### Fail Fast

Validate inputs and preconditions at function entry points
Use guard clauses to exit early when conditions aren't met
Throw informative, actionable exceptions immediately when invalid state is detected
Example: Check for null/undefined values, empty collections, or invalid ranges before processing

##### Auditability & Observability

Make every significant action and state change traceable
Implement structured logging with appropriate severity levels
Include relevant context in logs (user IDs, request IDs, timestamps)
Design clear error propagation chains that preserve context

##### Review-Friendly Code

Write code that peers can understand in a single read-through
Use self-documenting, expressive names: isUserSubscriptionActive not checkSub
Keep cognitive load low—avoid nested ternaries, complex conditionals, and long functions
Make intent explicit rather than requiring readers to infer behavior

##### Style & Best Practices
##### Modularity & Design

Follow SOLID principles (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion)
Keep functions small and focused on a single responsibility
Prefer composition over inheritance
Design for testability from the start

##### Type Safety

Use strong, explicit typing (TypeScript, Python type hints, generics)
Avoid any, unknown, object, or overly broad types
Leverage type systems to catch errors at compile time
Use union types, discriminated unions, and type guards where appropriate

##### Documentation

Use JSDoc/Docstrings for all public APIs, exported functions, and complex logic
Document the why (rationale, business context, trade-offs), not the what (which should be self-evident from code)
Include examples for non-obvious usage patterns
Keep documentation synchronized with code changes

##### Testing & Reliability

Structure code for testability using Dependency Injection and inversion of control
Suggest unit tests for complex business logic, edge cases, and critical paths
Design interfaces and abstractions that enable easy mocking
Consider test scenarios when architecting solutions

##### Output Requirements
##### Context First

Begin by briefly explaining the architectural approach and key design decisions
Highlight trade-offs considered and why this solution was chosen
Note any assumptions or constraints

##### Robust Error Handling

##### Never silently ignore errors
Use try-catch blocks with specific error types where appropriate
Consider Result/Either types or custom error classes for better error modeling
Provide actionable error messages that guide debugging
Include retry logic, fallbacks, or graceful degradation where relevant

##### Code Formatting & Standards

Follow established industry standards for the language:

Python: PEP 8
JavaScript/TypeScript: Prettier + Airbnb or Standard style
Java: Google Java Style Guide
C++: Google C++ Style Guide
Go: gofmt conventions


Ensure consistent indentation, spacing, and naming conventions
Use linters and formatters to enforce consistency automatically

##### Performance & Scalability Awareness

Consider algorithmic complexity (time and space)
Identify potential bottlenecks or scalability concerns
Suggest optimizations where appropriate, but not at the expense of readability
Note when premature optimization should be avoided

When providing code, always include:

Brief architectural explanation upfront
Complete, runnable code examples
Inline comments explaining non-obvious logic
Suggestions for testing approach
Notes on potential improvements or future considerations

### Git guideline

Never Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com> in the commit message