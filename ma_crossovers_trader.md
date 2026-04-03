# MA Crossover Trader

## Confirmation Indicators Checklist

A successful trend-following MA crossover trader looks at these indicators before entering a trade:

1. **Moving Average Crossover (primary signal):** Fast MA (e.g. 9 or 20 EMA) crosses above slow MA for long trades, or crosses below for short trades.
2. **Volume:** Volume should be above average or clearly expanding on the crossover candle/bar (shows strong participation).
3. **MACD:** MACD line should cross its signal line in the same direction as the MA crossover, or the MACD histogram should be expanding in the direction of the trade.
4. **RSI (Relative Strength Index, 14-period):** For long trades, RSI should be above 50 and rising (not overbought above 70). For short trades, RSI should be below 50 and falling (not oversold below 30).
5. **ADX (Average Directional Index):** ADX should be above 20–25 and preferably rising (confirms the market has sufficient trend strength).
6. **Support and Resistance levels:** The crossover should happen near or with a clear breakout above resistance (for longs) or breakdown below support (for shorts).
7. **Higher timeframe trend alignment:** The trend direction on the higher timeframe (e.g., daily chart) should agree with the MA crossover on the trading timeframe.
8. **ATR (Average True Range):** Used to set a logical stop-loss (typically 1–2x ATR away from entry) and to determine position size based on volatility.
9. **Bollinger Bands (optional):** Price should be breaking out of the bands or the bands should be expanding (indicating increasing volatility for a new trend).

---

## Parquet Schema — MA Columns (confirmed)

### Signal columns (+1 long / -1 short / 0 flat)
- `rema_50100` — EMA 50/100 crossover (short-term vs medium-term)
- `rema_100150` — EMA 100/150 crossover (medium-term vs long-term)
- `rema_50100150` — EMA triple confluence (all three windows aligned) ← highest quality
- `rsma_50100`, `rsma_100150`, `rsma_50100150` — SMA equivalents

### MA level values (relative price, i.e. vs FTSEMIB.MI benchmark)
- `rema_short_50`, `rema_medium_100`, `rema_long_150`
- `rsma_short_50`, `rsma_medium_100`, `rsma_long_150`

### Stop-loss columns
- `rema_50100_stop_loss`, `rema_100150_stop_loss`, `rema_50100150_stop_loss`
- SMA equivalents

### Other columns available for enrichment
- `rhigh`, `rlow`, `rclose` — relative OHLC (needed for ADX computation)
- `volume` — for volume expansion confirmation
- `rh1–rh4`, `rl1–rl4` — swing levels (shared with breakout assistant)
- `rrg` — regime direction (+1 bullish / -1 bearish / 0 sideways)

### NOT present (must compute in select_columns)
- No `rema_*_age` or `rema_*_flip` columns — must derive from signal series

---

## Implementation Breakdown

**Goal:** Migrate `breakout/` → `ta/breakout/`, build `ta/ma/` with TDD, then build `ask_ma_trader.py` mirroring `ask_trader.py`.

---

### A1  Inspect & scaffold (prerequisite)

**A1.1** ✅ Inspect parquet schema — confirm MA column names, level values, age/flip availability
- Done. Findings documented above.

**A1.2**  Create `ta/` package skeleton (empty files, no logic)
- Create: `ta/__init__.py`, `ta/utils.py`, `ta/breakout/__init__.py`, `ta/breakout/range_quality.py`, `ta/breakout/volume.py`, `ta/ma/__init__.py`, `ta/ma/trend_quality.py`, `ta/ma/volume.py`
- Done when: all files exist and `python -c "import ta"` succeeds

---

### A2  Migrate `breakout/` → `ta/breakout/` (import-only, no logic changes)

**A2.1**  Copy `breakout/utils.py` → `ta/utils.py`
- No changes to content. `ols_slope` is now the single shared math primitive.
- Done when: `from ta.utils import ols_slope` works

**A2.2**  Copy `breakout/range_quality.py` → `ta/breakout/range_quality.py`
- One import change: `from breakout.utils` → `from ta.utils`
- Done when: file is in place with updated import

**A2.3**  Copy `breakout/volume.py` → `ta/breakout/volume.py`
- One import change: `from breakout.utils` → `from ta.utils`
- Done when: file is in place with updated import

**A2.4**  Update `ta/breakout/__init__.py` and `ta/__init__.py` docstrings
- Document subpackages: breakout (range_quality, volume), ma (trend_quality, volume)
- Done when: docstrings reflect the full ta/ structure

**A2.5**  Update all callers to use new import paths
- `ask_trader.py` (2 lines): `from breakout.X` → `from ta.breakout.X`
- `tests/test_range_quality.py`: `from breakout.range_quality` → `from ta.breakout.range_quality`
- `tests/test_volume.py`: all `from breakout.volume` → `from ta.breakout.volume` (~20 occurrences)
- Done when: grep finds zero remaining `from breakout.` imports

**A2.6**  Run existing tests — confirm all pass with new import paths
- `python -m pytest tests/test_range_quality.py tests/test_volume.py -v`
- Done when: green

**A2.7**  Delete old `breakout/` directory
- Done when: `breakout/` no longer exists and tests still pass

---

### A3  Build `ta/ma/trend_quality.py` (TDD)

#### A3.1  Write failing tests — `tests/test_ma_trend_quality.py`

**A3.1.1**  Test `compute_rsi`
- Happy path: known RSI on a synthetic 20-bar series
- Edge: constant series → RSI = 50 (no gains, no losses)
- Edge: all gains → RSI approaches 100
- Edge: fewer than `period+1` bars → raises ValueError
- Done when: tests exist and fail with ImportError

**A3.1.2**  Test `compute_adx`
- Happy path: trending synthetic df (rhigh rising, rlow rising) → ADX > 25
- Edge: flat series → ADX < 20
- Edge: missing columns → raises ValueError
- Edge: fewer than `2*period` bars → raises ValueError
- Done when: tests exist and fail with ImportError

**A3.1.3**  Test `compute_ma_gap_pct`
- Happy path: fast_ma > slow_ma → positive gap %
- Happy path: fast_ma < slow_ma → negative gap %
- Edge: fast_ma == slow_ma → gap = 0.0
- Edge: rclose = 0 → raises ValueError (zero division guard)
- Done when: tests exist and fail with ImportError

**A3.1.4**  Test `compute_ma_slope_pct`
- Happy path: rising MA series → positive slope
- Happy path: falling MA series → negative slope
- Edge: flat series → slope ≈ 0
- Edge: fewer than 2 bars → returns 0.0 (delegated to ols_slope)
- Done when: tests exist and fail with ImportError

**A3.1.5**  Test `assess_ma_trend` → `MATrendStrength`
- Happy path: trending df → `is_trending=True`, ADX > 25, RSI > 50
- Happy path: flat/weak df → `is_trending=False`
- Edge: missing required columns → raises ValueError
- Done when: tests exist and fail with ImportError

#### A3.2  Implement `ta/ma/trend_quality.py` (Green)

**A3.2.1**  Implement `compute_rsi(rclose, period=14) → float`
- Wilder's smoothing (EMA of gains/losses)
- Returns last-bar RSI value
- Raises ValueError if fewer than `period + 1` non-NaN values

**A3.2.2**  Implement `compute_adx(df, period=14) → float`
- Inputs: df with `rhigh`, `rlow`, `rclose` (relative OHLC)
- Computes: True Range, +DM, -DM → Wilder-smoothed ADX
- Returns last-bar ADX value
- Raises ValueError on missing columns or insufficient bars

**A3.2.3**  Implement `compute_ma_gap_pct(fast_ma, slow_ma, rclose) → float`
- `(fast_ma - slow_ma) / rclose * 100`
- MACD proxy: positive = fast above slow = bullish alignment
- Raises ValueError if rclose ≤ 0

**A3.2.4**  Implement `compute_ma_slope_pct(ma_series, window) → float`
- OLS slope of last `window` bars of the MA series, normalised by mean, expressed as %/day
- Reuses `ta.utils.ols_slope` — no new math

**A3.2.5**  Implement `MATrendStrength` dataclass + `assess_ma_trend(df) → MATrendStrength`
- Fields: `rsi`, `adx`, `adx_slope`, `ma_gap_pct`, `ma_gap_slope`, `is_trending`
- `is_trending = adx > 25 and adx_slope > 0` (ADX above threshold AND rising)
- Preferred MA type: EMA (`rema_*`) as primary; SMA as secondary/confirmation
- MA gap uses: `rema_short_50` vs `rema_long_150` (widest spread = strongest trend read)

#### A3.3  Run A3 tests — all green

---

### A4  Build `ta/ma/volume.py` (TDD)

#### A4.1  Write failing tests — `tests/test_ma_volume.py`

**A4.1.1**  Test `assess_ma_volume` — crossover confirmation
- Signal flips on last bar + vol_trend ≥ 1.2 → `is_confirmed=True`
- Signal flips on last bar + vol_trend < 1.2 → `is_confirmed=False`
- No flip on last bar → `is_confirmed=None`

**A4.1.2**  Test `assess_ma_volume` — post-crossover volume sustainability
- 5-bar mean vol_trend ≥ 1.0 after flip → `is_sustained=True`
- Mean < 1.0 → `is_sustained=False`
- Not enough post-flip bars → `is_sustained=None`

**A4.1.3**  Test edge cases
- Signal always 0 (no crossover in history) → `is_confirmed=None`, `is_sustained=None`
- Missing `volume` column → raises ValueError
- Fewer than `MIN_VOL_BARS` bars → raises ValueError

#### A4.2  Implement `ta/ma/volume.py` (Green)

**A4.2.1**  Implement `MAVolumeProfile` dataclass
- Fields: `vol_on_crossover`, `vol_trend_mean_post`, `is_confirmed`, `is_sustained`
- Note on inverted semantics vs breakout: breakout wants quiet consolidation then spike;
  MA wants expansion ON the crossover bar and sustained above average after

**A4.2.2**  Implement `assess_ma_volume(df, signal_col) → MAVolumeProfile`
- `signal_col`: which MA signal to watch (e.g. `"rema_50100"`)
- Detects flip bar (signal changed from 0 to non-zero on last bar)
- `vol_on_crossover`: vol_trend value on the flip bar
- `vol_trend_mean_post`: mean vol_trend over post-flip bars (up to last 5)
- `is_confirmed`: True if vol_on_crossover ≥ DEFAULT_BREAKOUT_VOL_THR (1.2) — reuse threshold
- `is_sustained`: True if vol_trend_mean_post ≥ 1.0

#### A4.3  Run A4 tests — all green

---

### A5  Compute signal age and flip in select_columns (no new module)

**A5.1**  Add `_compute_signal_age_and_flip(df, signal_cols) → pd.DataFrame` helper inside `ask_ma_trader.py`
- For each signal column, computes:
  - `{col}_age`: consecutive bars the signal has held its current value (same logic as rbo_*_age)
  - `{col}_flip`: 1 if signal changed on the last bar vs previous, else 0
- Pure function, operates on a single-ticker DataFrame
- Done when: function exists and produces correct output on a synthetic 5-bar series

---

### A6  Build `ask_ma_trader.py`

**A6.1**  Write CLI skeleton — argparse, logging, paths, load_dotenv
- Mirror `ask_trader.py` exactly for infrastructure
- `--ticker` required, `--question` optional
- Done when: `python ask_ma_trader.py --ticker A2A.MI` prints "not implemented"

**A6.2**  Implement `select_columns(df) → pd.DataFrame`
- Keep: `symbol`, `date`, `rrg`, `rclose`
- Keep: `rema_50100`, `rema_100150`, `rema_50100150` (EMA signals, primary)
- Keep: `rsma_50100`, `rsma_100150`, `rsma_50100150` (SMA signals, confirmation)
- Keep: `rema_short_50`, `rema_medium_100`, `rema_long_150` (EMA levels for slope/gap)
- Keep: `rsma_short_50`, `rsma_medium_100`, `rsma_long_150` (SMA levels)
- Keep: `rema_50100_stop_loss`, `rema_100150_stop_loss`, `rema_50100150_stop_loss`
- Keep: `volume`, `rh3`, `rh4`, `rl3`, `rl4` (swing levels)
- Compute: `dist_to_ma_pct` for each MA level (% distance from rclose)
- Compute: `rclose_chg_50d`, `rclose_chg_150d` (momentum context)
- Compute: `vol_trend` (volume / 20-bar mean volume)
- Compute: `{signal}_age` and `{signal}_flip` for all 6 MA signal columns via A5.1
- Exclude: `rbo_*`, `rhi_*`, `rlo_*`, `rtt_5020`, `*_cumul`, `*_returns`, `*_chg*`, `*_PL_cum`
- Done when: last-bar snapshot JSON contains exactly the intended columns

**A6.3**  Build enrichment call
- Load full ticker df (before column selection) → pass to `assess_ma_trend(df)` and `assess_ma_volume(df, signal_col="rema_50100")`
- Attach results to snapshot dict as `"trend_strength"` and `"volume_profile"`
- Done when: snapshot JSON includes both enrichment dicts

**A6.4**  Write SYSTEM_PROMPT
- Field definitions for all MA columns (mirror depth of ask_trader.py prompt)
- Key concepts to define explicitly:
  - `rema_*` vs `rsma_*`: EMA leads faster, SMA lags — disagreement = low conviction
  - `rema_50100150 = +1`: all three windows aligned = highest quality long signal
  - `{signal}_age`: bars signal has held — older = more established trend
  - `{signal}_flip`: 1 = just changed this bar = fresh entry opportunity
  - `ma_gap_pct`: how far fast MA is above slow MA as % — widening = accelerating trend
  - `ma_gap_slope`: is the gap expanding or narrowing — narrowing = trend losing steam
  - `is_trending` (ADX): ADX > 25 and rising = trend has institutional momentum
  - `rsi`: momentum confirmation — >50 rising for longs, <50 falling for shorts
  - Stop-loss: `rema_50100_stop_loss` = ATR-based stop for the 50/100 crossover signal
- Output sections (parallel to ask_trader.py):
  0. Description: 3–5 sentence narrative. Signal confluence, RSI/ADX, vol_trend, regime.
  1. Short-term (50/100): `rema_50100` signal, age, flip, `dist_to_ma_pct` for rema_medium_100
  2. Medium-term (100/150): `rema_100150` signal, age, flip, `rclose_chg_150d`
  3. Triple confluence: `rema_50100150` value, SMA agreement/disagreement
  4. Trend quality: ADX level + slope (`is_trending`), RSI, `ma_gap_pct` + `ma_gap_slope`
  5. Volume: `is_confirmed` (crossover bar), `is_sustained` (post-crossover)
  6. Risk: stop-loss level from `rema_50100150_stop_loss`; swing levels `rh4`/`rl4` as targets
  7. Verdict: one actionable sentence — direction, entry trigger, stop

**A6.5**  Wire OpenAI call + structured output
- Mirror ask_trader.py: `client.beta.chat.completions.parse(...)` with Pydantic response model
- Done when: full run on a real ticker prints structured analysis

---

### A7  Integration test

**A7.1**  Run `python ask_ma_trader.py --ticker A2A.MI` end-to-end
- Confirm: no import errors, snapshot JSON is well-formed, all 7 output sections present

**A7.2**  Run `python ask_trader.py --ticker A2A.MI` — confirm breakout assistant still works after migration
- Done when: both CLIs produce output without errors

**A7.3**  Run full test suite
- `python -m pytest tests/ -v`
- Done when: all tests green

---

## Dependency Order

```
A1.1 (done) → A1.2
A1.2 → A2.1 → A2.2 → A2.3 → A2.4 → A2.5 → A2.6 → A2.7
A2.7 → A3.1 → A3.2 → A3.3
A2.7 → A4.1 → A4.2 → A4.3
A3.3 + A4.3 → A5.1 → A6.1 → A6.2 → A6.3 → A6.4 → A6.5
A6.5 → A7.1 → A7.2 → A7.3
```

---

## Key Invariants

- `ta/utils.py` is the single source of truth for `ols_slope` — never duplicated
- All `ta/ma/` functions are pure: no I/O, no side-effects, independently testable
- `rhigh`/`rlow` are used only inside enrichment functions — never sent to OpenAI (intraday noise)
- EMA signals are primary; SMA signals are confirmation — the AI prompt must make this explicit
- Signal age/flip are computed fresh in `select_columns()` — not read from parquet (they don't exist)
- `rema_50100150 = +1` is the highest-quality long signal (triple confluence); always surface it prominently
