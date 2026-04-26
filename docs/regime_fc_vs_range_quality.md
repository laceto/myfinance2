# regime_fc.py vs range_quality.py — Comparative Analysis

Both modules detect price tests of structural levels and use hysteresis to avoid noise.
They operate at different timescales, use different level sources, and serve different
analytical purposes — but their core state-machine logic is the same idea.

---

## 1. What each module is for

| | `RegimeFC` (`regime_fc.py`) | `range_quality` (`ta/breakout/range_quality.py`) |
|---|---|---|
| **Purpose** | Define the primary trend regime via dynamically discovered swing floors/ceilings | Assess consolidation quality before a range breakout |
| **Output** | `rg` column (+1 / 0 / -1), floor/ceiling levels, regime change dates | `RangeSetup` (touch counts, sideways flag, slope, band width) |
| **Level source** | Swing structure — discovered from OHLC peaks/troughs via `_historical_swings` | Fixed rolling bands — `rhi_20` / `rlo_20` (N-day high/low) |
| **Timescale** | Multi-level hierarchy (up to 9 swing levels) | Single level, capped consolidation window (default 40 bars) |
| **Price input** | Raw OHLC or relative OHLC | `range_position_pct` — already normalised to [0, 100] |

---

## 2. Structural parallels

### 2.1 Level discovery

`RegimeFC._historical_swings` uses `scipy.signal.find_peaks` + iterative
`_hilo_alternation` to reduce price data to alternating swing highs and lows.
These become the floors and ceilings that `_regime_floor_ceiling` uses.

`range_quality` skips the discovery step — it delegates to the signal pipeline.
`rhi_20` / `rlo_20` are the pre-computed N-day rolling bands. The analysis only
runs over the current zero-run of `rbo_20`.

> **Implication:** RegimeFC adapts the level to the current market structure.
> range_quality uses a fixed lookback band. The trade-off is adaptivity vs simplicity.

---

### 2.2 Retest and retreat — the same idea, different encoding

This is the most direct overlap. Both modules ask:
*"Has price approached a key level, then definitively pulled back — so the next
approach counts as a new, independent test?"*

#### RegimeFC — `_retest_swing`

```python
# After a swing high forms at hh_ll:
rt_hurdle = rt_sgmt.max()           # highest retest level reached
rt_px = df.loc[rt_dt:, _c].cummin() # how far price falls after the retest peak
# Confirmed retest = price crosses BACK through the retest hurdle
if (np.sign(rt_px - rt_hurdle) == -np.sign(_sign)).any():
    df.at[hh_ll_dt, _swg] = hh_ll   # write confirmed swing
```

A swing is confirmed only when price makes a new extreme AND then retreats back
through the retest level. The **retreat** is what validates the swing as structural.

#### range_quality — `count_touches`

```python
if not in_res_touch:
    if val >= 85:            # price reaches resistance → touch starts
        in_res_touch = True
        n_res += 1
else:
    if val < 65:             # price retreats below gray zone → touch ends
        in_res_touch = False # next approach will count as a new touch
```

A touch is confirmed the moment price enters ≥ 85%.
The touch ends — and a new one becomes countable — only after price retreats
below 65%, exactly as in RegimeFC.

#### Side-by-side

| Concept | RegimeFC | range_quality |
|---|---|---|
| "Test begins" | Price exceeds the retest hurdle | `range_pct >= 85` |
| "Still the same test" | cummin / cummax hasn't crossed back | `range_pct` in gray zone [65, 85) |
| "Retreat confirmed" | `np.sign(rt_px - rt_hurdle)` flips | `range_pct < 65` |
| "New test allowed" | `_swg` confirmed, flags reset | `in_res_touch = False` |
| Noise guard | `dist_vol` / `dist_pct` (ATR-based min distance) | Gray zone 20% wide, fixed |

---

### 2.3 Retracement — `_retracement_swing` vs gray zone

`_retracement_swing` confirms a swing when price retraces a minimum amount
(expressed in ATR multiples `retrace_vol` or as a fraction `retrace_pct`)
after reaching a new extreme:

```python
# Bullish case: new high must be followed by a pullback >= threshold
retracement = df.loc[hh_ll_dt:, _c].min() - hh_ll
if abs(retracement / vlty) - retrace_vol > 0:   # ATR-normalised
    df.at[hh_ll_dt, _swg] = hh_ll
```

The gray zone in `count_touches` plays the same role: it requires price to fall
*at least 20 percentage points from the touch entry before a new touch is allowed*.
The difference is calibration:

- RegimeFC normalises by ATR — adapts to current volatility.
- range_quality uses a fixed percentage of the 20-day range — simpler, but
  already normalised because `range_position_pct` is range-relative.

---

### 2.4 Boolean state machines — same pattern

| Flag | RegimeFC | range_quality |
|---|---|---|
| "Inside a test" | `ceiling_found`, `floor_found` | `in_res_touch`, `in_sup_touch` |
| "Breakout active" | `breakout`, `breakdown` | `rbo_20 != 0` (pre-computed) |
| NaN / gap handling | N/A (swing detection skips gaps naturally) | `consecutive_nan` counter → reset on `max_gap_bars` |

RegimeFC manages four flags simultaneously (two levels of state: found + direction).
range_quality manages two independent flags (one per side), which is simpler because
the range boundaries are already defined externally.

---

### 2.5 Regime output — same encoding

| | RegimeFC | range_quality / signal pipeline |
|---|---|---|
| Bullish | `rg = +1` | `rbo_20 = +1` |
| Neutral / consolidation | `rg ≈ 0` (transitional) | `rbo_20 = 0` |
| Bearish | `rg = -1` | `rbo_20 = -1` |

`rg` in RegimeFC is the `rrg` column in the stored DataFrame.
range_quality operates **inside** the `rbo_20 == 0` window — it characterises
the consolidation that RegimeFC's regime would label transitional.

---

## 3. Where they diverge

### 3.1 Level adaptivity

RegimeFC discovers floors/ceilings from the current swing structure, so the
reference level shifts as new swings form. `rhi_20` is a mechanical rolling max —
it does not distinguish a genuine swing high from a multi-day stall.

### 3.2 Volatility normalisation

RegimeFC normalises thresholds by ATR (`dist_vol`, `retrace_vol`) and by rolling
standard deviation (`stdev` in `_regime_floor_ceiling`). This means a volatile
period requires a larger absolute move to confirm a swing.

range_quality uses a fixed percentage of the 20-day range. Because
`range_position_pct` is already normalised, a 20-point retreat from 85% is always
the same fraction of the current range — implicitly volatility-adjusted at the
band level, not the threshold level.

### 3.3 Hierarchy

RegimeFC builds up to 9 swing levels (`rh1/rl1` … `rh9/rl9`), allowing multi-
timeframe structure analysis in a single pass. range_quality is single-level
(20-day band), with 50-day and 150-day equivalents in separate signal columns.

### 3.4 Purpose

RegimeFC answers: *"Is the market in a floor regime or ceiling regime — and did
that change?"*

range_quality answers: *"Is the current consolidation well-defined enough to trade
a breakout from?"*

They are complementary: RegimeFC defines the structural context; range_quality
assesses the consolidation quality within that context.

---

## 4. Integration — implementation

### 4.1 Key data discovery

`rclg`, `rflr`, `rrg`, and `rrg_ch` are **already computed and stored** in
`analysis_results.parquet` by the algoshort signal pipeline — no need to re-run
`compute_regime()` at runtime.

| Column | Content | Sparsity (A2A.MI, 2 619 bars) |
|---|---|---|
| `rclg` | Structural ceiling (relative price) | 4 non-NaN — event-based |
| `rflr` | Structural floor  (relative price) | 5 non-NaN — event-based |
| `rrg`  | Regime signal (+1 / -1)            | 2 369 non-NaN — dense |
| `rrg_ch` | Regime-change price level (ffilled) | 2 369 non-NaN — dense |

`rclg`/`rflr` are sparse because they only carry a value at the bar where a new
ceiling/floor is confirmed by `_regime_floor_ceiling`. Forward-filling them gives
every bar the most recently confirmed structural level.

### 4.2 Why rrg cannot replace rbo_20 as the window selector

`rrg` is ±1 only — it has no 0-state (consolidation). `assess_range` needs a
signal with 0 to define the zero-run window. `rbo_20` is therefore kept as the
consolidation-window selector. The integration only replaces the **reference levels**
(what counts as "near resistance"), not the **window** (which bars to analyse).

### 4.3 Column mapping — the full contract

```
analysis_results.parquet                  range_quality primitive
─────────────────────────────────         ──────────────────────────────────────
rclg  (sparse) → ffill → swing_hi    →   rhi_col  (resistance reference)
rflr  (sparse) → ffill → swing_lo    →   rlo_col  (support reference)
rbo_20 (dense, ±1/0)                 →   rbo_col  (consolidation window selector)
rclose (dense)                        →   rclose_col
```

No new signal logic.  `assess_range` and `measure_volatility_compression` receive
`swing_hi` / `swing_lo` via their existing `rhi_col` / `rlo_col` parameters —
both functions are called unchanged.

### 4.4 Implementation — ta/breakout/swing_range_quality.py

The adapter is ~100 lines:

```python
def _prepare_swing_levels(df) -> pd.DataFrame:
    out = df.copy()
    out["swing_hi"] = out["rclg"].ffill().bfill()   # forward-fill sparse ceiling
    out["swing_lo"] = out["rflr"].ffill().bfill()   # forward-fill sparse floor
    return out

def assess_swing_range(df, window_bars=40, config=None, ...) -> RangeSetup:
    df_sw = _prepare_swing_levels(df)
    return assess_range(df_sw, ..., rhi_col="swing_hi", rlo_col="swing_lo")

def measure_swing_volatility(df, ...) -> VolatilityState:
    df_sw = _prepare_swing_levels(df)
    return measure_volatility_compression(df_sw, ..., rhi_col="swing_hi", rlo_col="swing_lo")
```

### 4.5 Integration into bo_snapshot.py

`_compute_swing_range_setup(df)` follows the existing `_compute_range_setup` pattern
exactly — same try/except shape, same None-on-failure contract.  `build_snapshot()`
gains a `swing_range_setup` key alongside the existing `range_setup` key.

Example output (A2A.MI, 2026-04-24):

```json
"range_setup": {
  "n_resistance_touches": 1,
  "n_support_touches":    1,
  "band_width_pct":       6.47        ← tight 20-day rolling band
},
"swing_range_setup": {
  "n_resistance_touches": 0,
  "n_support_touches":    1,
  "band_width_pct":       45.42,      ← wide structural swing range
  "swing_bw_rank":        98.02,      ← near top of its own history
  "swing_vol_compressed": false
}
```

Reading both together: the 20-day band is tight (6.5%), but the structural swing
band is near its widest ever (rank 98p). The current micro-consolidation is a
small pause inside a historically large structural range — useful context for
sizing and stop placement.

### 4.6 What is NOT yet integrated (phase 2)

| Idea | Complexity | Benefit |
|---|---|---|
| ATR-normalised retreat threshold in `count_touches` | Medium — needs per-bar threshold series | Adapts gray zone to current volatility, matching `_retracement_swing` philosophy |
| Use `rh3`/`rl3` swing columns (hierarchical) as rhi/rlo | Low — already ffilled in bo_snapshot | Adds a mid-tier structural level between 20-day band and full swing range |
| Detect zero-runs within `rrg` regime changes | High — requires redesigning window finder | Enables "consolidation between RegimeFC regime changes" analysis |

---

## 5. Summary

| Criterion | RegimeFC | range_quality |
|---|---|---|
| **Core question** | What is the structural regime? | Is this consolidation worth trading? |
| **Level source** | Dynamic swing discovery | Fixed rolling bands |
| **Retest logic** | Cumulative price crosses back through rt_hurdle | `range_pct` retreats below exit threshold |
| **Noise filter** | ATR / % min distance | Fixed gray zone [65%, 85%) |
| **Volatility adapt** | ATR-normalised thresholds | Implicit via range normalisation |
| **State machine** | 4 flags, mutually exclusive | 2 independent flags |
| **Output** | `rg` column, floor/ceiling levels | `RangeSetup` dataclass |
| **Composability** | Produces structural context | Consumes it (via `rbo_20`) |
