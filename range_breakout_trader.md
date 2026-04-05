# Range Breakout Trader — Playbook

A range breakout trader hunts for tight, well-defined consolidation followed by an explosive
directional move. Before entering, multiple independent confirmations must align.

---

## 1. Clear Range Structure (The Setup)

Identify a clean, mature range:

- Multiple touches of relative resistance (`rhi_N`) and relative support (`rlo_N`)
- Price moving sideways — `rrg = 0` (sideways regime) or transitioning from `-1 → 0`
- The longer and tighter the range, the more powerful the breakout potential
- Longer signal age (`rbo_N_age` high while `rbo_N = 0`) indicates time spent consolidating

**Red flag**: A messy or uneven range. Check `rh3/rh4` (swing resistance) vs `rl3/rl4` (swing
support) — if they are widely spread, the range has no clean boundary.

---

## 2. Volatility Compression

Look for decreasing volatility inside the range:

- `range_position_pct` oscillating narrowly between 20–80% without touching extremes
- `rclose_chg_20d` near zero while `rclose_chg_150d` still shows trend context

The spread `(rhi_N - rlo_N) / rclose` is a synthetic "band width." A narrow band = compressed
energy ready to release. Ideal entry is when this compresses to multi-week lows.

---

## 3. Volume Behavior

Volume confirms whether the breakout is genuine:

| Condition | Meaning |
|-----------|---------|
| `vol_trend < 0.8` inside range | Normal compression — healthy |
| `vol_trend > 1.5` on breakout bar | Strong confirmation — institutional participation |
| `vol_trend < 1.0` on breakout bar | Caution — likely fakeout or retail-driven move |

**Rule**: Never enter a breakout with `vol_trend < 1.2`. Wait one bar to see if volume follows.

---

## 4. Catalysts and Context

Technical traders also keep an eye on:

- Earnings, macro data, sector rotations
- Sector strength: if peer stocks are already breaking out, the setup has tailwind
- Check `rrg` — a stock entering `rrg = +1` (bullish regime) simultaneously with `rbo_20 = +1`
  is a high-probability setup; the regime flip adds a second independent confirmation

---

## 5. Market Environment (Relative Lens)

This system uses **relative prices** (stock / `FTSEMIB.MI`), not absolute prices. This is
critical:

- `rclose` rising means the stock outperforms the index **regardless of absolute direction**
- A breakout in relative terms works in both bull and bear markets
- Still check: if the broader market is in free fall, even strong relative breakouts can stall
  due to forced selling and liquidity withdrawal

Practical rule: prefer entries where both `rrg = +1` and the index itself is not in a
downtrend (check `FTSEMIB.MI` separately as a market-regime filter).

---

## 6. False Breakout Risk (Liquidity Traps)

Smart traders anticipate traps:

- Price briefly breaks above `rhi_20`, then snaps back: `rbo_20` goes `+1` then flips back `0`
- `rbo_20_flip = 1` on the **exit** bar is the trap signal — confirms a failed breakout
- Big players trigger stops above `rhi_20` to accumulate at lower prices before the real move

**Fakeout filters** — require ALL of the following before entering:
1. `rbo_20 = +1` **and** `rbo_20_flip = 1` (fresh breakout bar)
2. `vol_trend ≥ 1.2` on the breakout bar
3. `rclose > rhi_20` at bar close, not just intrabar
4. Optional: wait for the first **retest bar** (`range_position_pct` dips to ~100% then holds)

---

## 7. Multi-Timeframe Signal Confluence

The system's three breakout windows give a natural timeframe stack:

| Signal | Window | Role |
|--------|--------|------|
| `rbo_20` | 20-day | Short-term trigger — entry signal |
| `rbo_50` | 50-day | Medium-term filter — must agree |
| `rbo_150` | 150-day | Structural trend — directional bias |

**Grade A setup (full confluence)**: `rbo_20 = rbo_50 = rbo_150 = +1`
**Grade B setup**: `rbo_20 = +1`, `rbo_50 = +1`, `rbo_150 = 0` (neutral, not opposed)
**Skip**: `rbo_150 = -1` with `rbo_20 = +1` — you are fighting the structural trend

Additionally, validate with the MA stack:
- `rema_50100150 = +1` **and** `rsma_50100150 = +1` = full EMA+SMA bullish alignment
- `rtt_5020 = +1` (turtle signal) = independent breakout confirmation

**Ideal entry checklist**:
- [ ] `confluence = full_long` (all three breakout signals = +1)
- [ ] `rrg = +1` or `rrg = 0` transitioning bullish
- [ ] `rema_50100150 = +1` and `rsma_50100150 = +1`
- [ ] `rtt_5020 = +1`
- [ ] `vol_trend ≥ 1.2` on breakout bar
- [ ] `rbo_20_flip = 1` (fresh signal — not a stale continuation)

---

## 8. Entry Trigger

Define exactly what confirms entry — no guessing, rules are predefined:

**Primary trigger** (intraday / next open):
- `rbo_20` flips from `0 → +1` (`rbo_20_flip = 1`)
- `rclose > rhi_20` at EOD close
- `vol_trend ≥ 1.2`

**Conservative trigger** (retest entry — lower risk, smaller position):
- `rbo_20 = +1` (already active, `rbo_20_flip = 0`)
- `range_position_pct` retraced to 95–105% (retesting the breakout level from above)
- Volume contracts on retest (`vol_trend < 0.9`) — price holds without pressure

---

## 9. Risk Management (Non-negotiable)

Before entering, all three parameters must be defined:

| Parameter | Rule |
|-----------|------|
| **Short-term stop** | `rlo_20` — short-term support; exit if price re-enters range |
| **Structural stop** | `rlo_150` — major floor; breach = thesis invalidated |
| **Stop distance** | `dist_to_rlo_20_pct` — position size inversely proportional to stop distance |
| **Min R:R** | `dist_to_rhi_150_pct` / `dist_to_rlo_20_pct` ≥ 2.0 — if below, skip the trade |
| **Position size** | Capital × risk_pct / stop_distance_in_pct |

**Rule**: If `dist_to_rlo_20_pct < 1.0%`, the stop is too tight — likely inside normal noise.
Wait for the range to re-establish or widen before entering.

---

## 10. Target Projection

Estimate how far the move can go before entering:

1. **Minimum target**: height of the `rhi_20 - rlo_20` range projected upward from `rhi_20`
2. **Intermediate target**: `rh3` (2nd-highest swing resistance)
3. **Major target**: `rh4` (absolute highest swing high — multi-year resistance)
4. **Extended target**: `rhi_150 + (rhi_150 - rlo_150)` — full 150-day range projection

Use `dist_to_rhi_150_pct` as a quick check: if the stock is already near the 150-day high,
the remaining upside may be small and the R:R degrades.

---

## 11. Signal Age as Trade Quality Filter

`rbo_N_age` tells you how long a signal has been active:

| Age (bars) | Interpretation |
|------------|---------------|
| 1–3 | Fresh breakout — maximum opportunity, maximum risk |
| 4–10 | Early trend — good entry on retest (conservative trigger) |
| 11–30 | Established trend — add on pullback to `rhi_20` |
| 31+ | Mature/extended — avoid new longs; use `rl3/rl4` as trailing stop reference |

A combination like `rbo_150_age = 90`, `rbo_50_age = 15`, `rbo_20_age = 1` means:
- Structural trend mature (150d)
- Medium trend established (50d)
- Short-term re-entry just triggered (20d) — **ideal re-entry within ongoing uptrend**

---

## 12. Exit Rules

Define exit before entering:

| Condition | Action |
|-----------|--------|
| `rbo_20` flips `+1 → 0` (`rbo_20_flip = 1`) | Partial exit (50%) — stop to breakeven |
| `rbo_20` and `rbo_50` both flip to `0` | Full exit |
| `rclose < rlo_20` (price back inside range) | Stop triggered — full exit immediately |
| `rbo_150_flip = 1` and `rbo_150 = -1` | Structural breakdown — exit all regardless |

**Trailing stop approach**: after `rbo_20_age > 10`, trail stop to the most recent `rlo_20`
update. Each new 20-day high resets `rhi_20`, so `rlo_20` rises too — natural trailing stop.

---

## Summary Decision Matrix

```
rbo_150  rbo_50  rbo_20  vol_trend  rrg  → Action
  +1       +1      +1      ≥ 1.2    +1   → Grade A long — full size
  +1       +1      +1      ≥ 1.2     0   → Grade B long — 3/4 size
  +1       +1       0       any     +1   → Wait for rbo_20 flip
  +1        0      +1      ≥ 1.2    +1   → Grade B — confirm rbo_50 next day
   0       +1      +1      ≥ 1.2     0   → Grade C — half size, structural trend unclear
  -1      any     +1       any     any   → Skip — fighting 150-day trend
  any     any      0       any     any   → No position
  any     any     -1       any     any   → Short side (apply symmetric rules)
```

---

## Bottom Line

A range breakout trader is not just buying "price going up." The edge comes from:

1. **Price compression** setting up the energy
2. **Volume confirmation** proving institutional participation
3. **Multi-timeframe alignment** ensuring the breakout is with the structural trend
4. **Pre-defined risk** so a loss is a planned cost, not a surprise
5. **Signal freshness** (`_flip`, `_age`) to avoid chasing stale moves

The `range_position_pct` is the single most useful number: above 100% with `vol_trend ≥ 1.2`
and `rbo_20_flip = 1` is the core entry signal. Everything else is confirmation layering.

---

## Implementation Plan

**Goal**: Implement three foundational analytical primitives (touch counting, sideways
classification, consolidation time), then build a deterministic screener on top of them.

**Current state**: `analysis_results.parquet` contains `rbo_*`, `rhi_*`, `rlo_*`, `rrg`,
`rema_50100150`, `rsma_50100150`, `rtt_5020`, swing levels, and volume. The relative price
series `rclose` vs `FTSEMIB.MI` is the input for all three primitives.

---

### Foundational analytical questions

Before any screener or rule engine can exist, three primitives must be defined and
implemented as pure, tested functions. All three operate on the `rclose` series for a
single ticker within a defined window.

---

#### Primitive 1 — Touch counting

**Problem**: count how many distinct times price approached the current resistance (`rhi_N`)
or support (`rlo_N`) without breaking through (i.e., while `rbo_N = 0`).

**Why not use absolute % distance from `rhi_N`**: `rhi_N` is a rolling max that decays as
old bars roll off. An absolute tolerance like `dist_rhi < 0.5%` ties the count to the
current level, not the width of the range. A narrow-range stock needs a different absolute
tolerance than a wide-range one.

**Solution**: use `range_position_pct` (where `rclose` sits between `rlo_N` and `rhi_N`).
This normalizes for range width automatically.

```
TOUCH_HI_THRESHOLD = 85   # range_pct >= 85% → near resistance
TOUCH_LO_THRESHOLD = 15   # range_pct <= 15% → near support
RETREAT_THRESHOLD  = 65   # range_pct must drop below 65% to end a resistance touch event
BOUNCE_THRESHOLD   = 35   # range_pct must rise above 35% to end a support touch event
DEFAULT_MAX_GAP_BARS = 5  # consecutive NaN bars that reset the touch state machine

def count_touches(
    range_pct:    pd.Series,  # range_position_pct for the consolidation window
    hi_thresh:    float = TOUCH_HI_THRESHOLD,
    lo_thresh:    float = TOUCH_LO_THRESHOLD,
    retreat:      float = RETREAT_THRESHOLD,
    bounce:       float = BOUNCE_THRESHOLD,
    max_gap_bars: int   = DEFAULT_MAX_GAP_BARS,
) -> tuple[int, int]:
    """
    Returns (n_resistance_touches, n_support_touches).

    Algorithm:
    1. Mark bars where range_pct >= hi_thresh as "at resistance".
    2. Group consecutive True bars into a single touch event.
    3. Between two resistance touch events, price must have dipped below
       retreat — otherwise merge the two clusters into one.
    4. Count distinct clusters = n_resistance_touches.
    5. Repeat symmetrically for support using lo_thresh / bounce.
    6. State machine resets (both in_res_touch and in_sup_touch → False)
       when max_gap_bars consecutive NaN bars are seen — trading halts or
       data gaps make the pre-gap touch state unreliable.
    """
```

**Tests**:
- flat series at 50% → (0, 0)
- spike to 90% then drop to 40% then spike to 92% → (2, 0)
- spike to 90% then only drop to 70% (stays above RETREAT=65) then 88% → (1, 0)
  (two near-resistance bars without a real retreat = one touch event)
- dip to 10% then rise to 50% then dip to 12% → (0, 2)

---

#### Primitive 2 — Sideways classification

**Problem**: determine whether `rclose` is moving sideways (consolidating) vs trending
during a given window.

**Why not use `rclose_chg_Nd`**: it only compares the first and last bar, ignoring the path.
A stock that drifts up 3%, then falls 3%, then rises 3% back shows `chg ≈ 0%` but is
trending, not consolidating.

**Solution**: OLS linear regression slope on `rclose` vs bar index, normalized by
`mean(rclose)` within the window, expressed as `%/day`.

```
SIDEWAYS_SLOPE_THRESHOLD = 0.15   # |slope_pct_per_day| < 0.15% → sideways

def classify_trend(
    rclose: pd.Series,    # relative close prices for the window
) -> tuple[bool, float]:
    """
    Returns (is_sideways, slope_pct_per_day).

    slope_pct_per_day = OLS_slope / mean(rclose) * 100
    is_sideways       = abs(slope_pct_per_day) < SIDEWAYS_SLOPE_THRESHOLD
    """
```

**Invariants**:
- Window must have at least `MIN_TREND_BARS` (10) non-NaN bars to produce a
  reliable OLS estimate. Raised from the original 5-bar minimum: OLS on 5
  noisy financial bars has a confidence interval that dwarfs the 0.15%/day
  sideways threshold at typical Borsa Italiana volatility.
- `slope_pct_per_day` is signed: positive = uptrend, negative = downtrend.

**Tests**:
- perfectly flat series → slope = 0.0, is_sideways = True
- series rising 0.5%/bar → slope ≈ 0.5, is_sideways = False
- series with noise but slope 0.05%/bar → is_sideways = True
- 5-bar minimum: 4-bar series → raises ValueError

---

#### Primitive 3 — Consolidation time

**Problem**: how many bars has price been consolidating (inside the range) before the
current bar?

**Answer**: already computed. `rbo_N_age` when `rbo_N = 0` is exactly the consecutive bar
count during which the signal has held `0`. No new function is needed.

**Invariant to document**: `rbo_N_age` resets to `1` on the bar the signal changes value
(whether `0 → +1`, `+1 → 0`, etc.). So at the moment of breakout, `rbo_N_age = 1` (the
breakout bar itself) and the prior consolidation length = the `rbo_N_age` from the
**previous bar**.

```
# Preceding consolidation length at the moment of breakout:
prior_consolidation_bars = df["rbo_20_age"].shift(1).where(df["rbo_20_flip"] == 1)
```

---

### Build sequence

```
A1  Implement touch counting → breakout/range_quality.py
  A1.1  count_touches(range_pct, hi_thresh=85, lo_thresh=15, retreat=65, bounce=35)
  A1.2  Unit tests: 6 fixture cases covering the invariants above
  A1.3  Smoke-test on SCM.MI 2016-08 consolidation window (38 bars):
          expected ≈ 3 resistance touches, 2 support touches

A2  Implement sideways classifier → breakout/range_quality.py (same module)
  A2.1  classify_trend(rclose, threshold=0.15) → (bool, float)
  A2.2  Unit tests: flat, rising, noisy-flat, 4-bar minimum guard
  A2.3  Smoke-test on SCM.MI 2016-08 window: expect is_sideways=True, slope≈0.008%/day

A3  Document consolidation_time invariant → breakout/range_quality.py docstring
  A3.1  Add consolidation_age(rbo_series) → pd.Series:
          returns rbo_age shifted by 1 on flip bars, else NaN
          (surfaces the pre-breakout consolidation length at the exact breakout bar)
  A3.2  Unit test: verify correct shift and NaN masking

A4  Integrate primitives into RangeSetup dataclass → breakout/range_quality.py
  A4.1  Define RangeSetup dataclass:
          {
            n_resistance_touches: int,
            n_support_touches:    int,
            is_sideways:          bool,
            slope_pct_per_day:    float,
            consolidation_bars:   int,    ← rbo_20_age on previous bar
            band_width_pct:       float,  ← (rhi_20 - rlo_20) / rclose * 100
          }
  A4.2  Implement assess_range(ticker_df, window_bars=40) → RangeSetup:
          - Slice last `window_bars` rows where rbo_20 = 0
          - Compute range_pct → count_touches
          - Pass rclose slice → classify_trend
          - Read rbo_20_age[-1] for consolidation_bars
          - Compute band_width_pct
  A4.3  Unit tests: full RangeSetup from synthetic DataFrame

A5  Expose range quality in ask_trader.py snapshot
  A5.1  Import assess_range from breakout.range_quality
  A5.2  Compute RangeSetup for the ticker before building the JSON snapshot
  A5.3  Merge RangeSetup fields into the snapshot dict
  A5.4  Add field definitions to SYSTEM_PROMPT
  A5.5  Smoke-test: run ask_trader.py --ticker SCM.MI, verify range quality fields appear
```

**Status**: A1–A5 complete — 340/340 tests green (full suite including `ta/ma/`).
**Dependency order**: A1 → A2 → A3 → A4 → A5 ✓

**Files created / modified**:
```
ta/breakout/__init__.py
ta/breakout/range_quality.py  ← count_touches (+ max_gap_bars), classify_trend,
                                 breakout_prior_consolidation_length,
                                 measure_volatility_compression, RangeSetup, assess_range,
                                 RangeQualityConfig (loadable from config.json)
ta/breakout/volume.py         ← VolumeProfile, assess_volume_profile
                                 (signal validation: NaN / non-integer / out-of-range guards)
ta/utils.py                   ← ols_slope, ols_slope_r2 (NaN guard added)
tests/test_range_quality.py   ← 79 unit tests, all passing
ta/breakout/tests/test_volume.py  ← 42 unit tests, all passing
ask_trader.py                 ← enriched with range_setup + volume_profile in snapshot
```

---

### Primitive 4 — Volatility Compression (Point 2 from playbook)

**Problem**: measure whether volatility is *decreasing* inside the range (energy buildup).
The playbook mentions "narrowing candles" and "Bollinger Bands tightening" — these are two
independent signals that must be defined precisely before implementation.

#### Two independent measures

**Signal A — Band width** (`band_width_pct`):
```
band_width_pct = (rhi_20 - rlo_20) / rclose * 100
```
This is the normalised 20-day relative range envelope. It captures how wide the current
"cage" is. Low value = price confined to a narrow band = range is tight.

**Signal B — Band width slope** (`band_width_slope`):
OLS slope of `band_width_pct` over the last N bars of the consolidation window.
- Negative slope = the cage is actively **narrowing** = energy building.
- Positive slope = range is **expanding** = consolidation may be dissolving.

These two are independent:
- Narrow AND narrowing → ideal compressed setup
- Narrow but widening → range breaking down, caution
- Wide but narrowing → early compression, not yet actionable
- Wide and widening → no setup

**Signal C — Historical percentile** (`band_width_pct_rank`):
Where the current `band_width_pct` sits in its own 252-bar rolling history (0–100).
Low rank = historically compressed for this ticker.

Why this is needed: a `band_width_pct` of 5% means different things for a volatile
small-cap vs a defensive large-cap. The percentile normalises for ticker-specific
volatility characteristics.

#### Compression definition

```
is_compressed = (band_width_slope < 0) AND (band_width_pct_rank < 25)
```

Interpretation: the range is both **actively narrowing** (slope < 0) and **historically
tight** (bottom quartile of its own history). This is the "energy buildup" signal.

#### Plan

```
B1  Implement measure_volatility_compression -> breakout/range_quality.py (same module)

  B1.1  Add VolatilityState dataclass:
          {
            band_width_pct:      float,  current (rhi_20 - rlo_20) / rclose * 100
            band_width_slope:    float,  OLS slope of band_width_pct over window_bars
                                         negative = range narrowing = energy building
            band_width_pct_rank: float,  percentile rank vs 252-bar history (0-100)
                                         low rank = historically compressed
            is_compressed:       bool,   band_width_slope < 0 AND rank < 25
            history_available:   int,    actual non-NaN bars used for rank computation
                                         may be < history_bars for new listings
            is_rank_reliable:    bool,   True when history_available >= history_bars
                                         False = rank based on thin history, treat cautiously
          }

  B1.2  Implement measure_volatility_compression(
            df: pd.DataFrame,        full ticker history sorted ascending
            window_bars: int = 40,   look-back for slope computation
            history_bars: int = 252, look-back for percentile rank
        ) -> VolatilityState

        Algorithm:
          1. Compute band_width_pct = (rhi_20 - rlo_20) / rclose * 100 for all bars.
          2. Take last `window_bars` of band_width_pct -> apply classify_trend() reuse
             (same OLS slope method). This gives band_width_slope in %/bar.
          3. Compute percentile rank: count how many of the last `history_bars` values
             are <= current band_width_pct, divide by history_bars * 100.
          4. is_compressed = band_width_slope < 0 AND band_width_pct_rank < 25.

        Invariants:
          - band_width_pct is always >= 0 (rhi_20 >= rlo_20 by definition).
          - band_width_pct_rank is in [0, 100].
          - Requires at least MIN_TREND_BARS bars where rhi_20 is non-NaN.
          - Uses reuse of classify_trend() for slope computation — no duplicated OLS code.

  B1.3  Write unit tests (tests/test_range_quality.py — same file, new class):
          TestVolatilityCompression:
          - flat band_width_pct series: slope = 0, rank at 50th pct
          - shrinking band_width: slope < 0, is_compressed = True if rank < 25
          - expanding band_width: slope > 0, is_compressed = False
          - rank at historical minimum: band_width_pct_rank < 25 -> True
          - rank at historical maximum: band_width_pct_rank >= 75 -> False
          - smoke test: SCM.MI consolidation window:
              band_width_pct at start ~11.6%, at end ~3.1% -> strongly compressing
              is_compressed = True

  B1.4  Smoke-test values (verified from parquet):
          SCM.MI 38-bar window band widths:
          start: (10.7669 - 9.5181) / 10.2039 = 12.24%
          end:   (10.5170 - 10.1990) / 10.2337 =  3.11%
          -> slope clearly negative -> is_compressed = True
```

**Note on intrabar candle size** (`rhigh - rlow`):
The parquet contains `rhigh` and `rlow` (relative OHLC). `(rhigh - rlow) / rclose * 100`
is a per-bar measure of daily price range in relative terms. A rolling mean of this
series over 10 bars gives a short-term ATR-equivalent. This is a complementary signal
to `band_width_pct` (which captures the 20-day envelope). Implementation is deferred
to after B1 is verified, as it adds complexity for marginal additional information
given the 20-day band width already captures the compression well.

---

### Primitive 5 — Volume Behavior (Point 3 from playbook)

**Two distinct questions.** Volume behavior inside a consolidation and volume behavior
at the breakout bar are answered with the same metric (`vol_trend`) but at different
points in time, with different thresholds, and with different interpretations.

---

#### Question 1: Is volume quiet inside the range?

**Why this matters**: Low volume during consolidation = no one is eager to trade at
current prices = range is genuine. High volume inside the range = distribution or
accumulation in progress = setup may be less clean.

**Metric**: `vol_trend_mean` — mean of `vol_trend` over the current consolidation
window (bars where `rbo_20 = 0`).

Where `vol_trend = volume / volume.rolling(20).mean()`.

**Threshold**: `vol_trend_mean < 1.0` = volume has been below its 20-bar average
during consolidation = quiet range = healthy.

**Secondary metric**: `vol_trend_slope` — OLS slope of `vol_trend` over the window.

This is informational, NOT a hard gate. Real data shows two valid consolidation
patterns:

| Pattern | `vol_trend_mean` | `vol_trend_slope` | Meaning |
|---------|-----------------|-------------------|---------|
| Classic compression | < 0.8 | negative | Volume drying up — ideal |
| Accumulation creep | < 0.8 | slightly positive | Low absolute volume but quietly building — still valid |
| Distribution | >= 1.0 | positive | Active selling inside range — caution |

POR.MI (2018-09-03 breakout): pre-breakout `mean=0.475`, `slope=+0.074` — low absolute
volume with a slight positive trend → accumulation. Classified as quiet (`is_quiet=True`)
because `mean < 1.0`, not rejected because `slope > 0`.

**`is_quiet` definition**: `vol_trend_mean < 1.0` (mean threshold only).
Slope is reported separately — callers decide if they want to require `slope < 0` as
a stricter filter.

---

#### Question 2: Is volume confirming at the breakout bar?

**Why this matters**: A breakout on thin volume is a fakeout candidate. Institutional
participation shows up as a volume spike on the breakout bar.

**Metric**: `vol_trend_now` at the current bar, but only when that bar IS a breakout
flip (`rbo_20` transitioned from `0 → +1` or `0 → -1`).

**Threshold**: `vol_trend_now >= 1.2` = at least 20% above the 20-bar average.

**`breakout_confirmed` definition**:
- `True`  — last bar is a breakout flip AND `vol_trend_now >= 1.2`
- `False` — last bar is a breakout flip AND `vol_trend_now < 1.2`
- `None`  — last bar is NOT a breakout flip (still in consolidation or mid-trend)

POR.MI smoke test: breakout bar `vol_trend = 10.95` >> 1.2 → `breakout_confirmed=True`.

---

#### Plan

```
C1  Implement assess_volume_profile -> breakout/volume.py  (new file, same module family)

  C1.1  Add VolumeProfile dataclass:
          {
            vol_trend_now:       float,        vol_trend at the last bar
            vol_trend_mean:      float,        mean vol_trend over consolidation window
            vol_trend_slope:     float,        OLS slope of vol_trend over window (/bar)
            vol_trend_slope_r2:  float,        R² of OLS fit (0-1); < 0.3 = noisy slope
            is_quiet:            bool,         vol_trend_mean < quiet_threshold (default 1.0)
            is_declining:        bool,         vol_trend_slope < 0  (informational)
            breakout_confirmed:  bool | None,  see Question 2 definition above
          }

  C1.2  Implement assess_volume_profile(
            df:                      pd.DataFrame,  full ticker history sorted ascending
            window_bars:             int   = 40,    max bars of zero-run to use for mean/slope
            quiet_threshold:         float = 1.0,   vol_trend_mean < this -> is_quiet
            breakout_vol_threshold:  float = 1.2,   vol_trend_now >= this -> breakout_confirmed
        ) -> VolumeProfile

        Required columns: volume, rbo_20

        Algorithm:
          1. Compute vol_trend = volume / volume.rolling(20, min_periods=1).mean()
          2. Identify the current consolidation window:
               last consecutive run of bars where rbo_20 == 0, up to window_bars bars.
               If the last bar is NOT in consolidation (rbo_20 != 0), look back to find
               the most recent zero-run for mean/slope.
          3. vol_trend_now   = vol_trend at the last bar.
          4. vol_trend_mean  = mean(vol_trend over the zero-run slice).
          5. vol_trend_slope = _ols_slope(vol_trend over the zero-run slice).
          6. is_quiet        = vol_trend_mean < quiet_threshold.
          7. is_declining    = vol_trend_slope < 0.
          8. Breakout flip check:
               flip = (rbo_20[-1] != 0) AND (rbo_20[-2] == 0)
               if flip:   breakout_confirmed = vol_trend_now >= breakout_vol_threshold
               otherwise: breakout_confirmed = None

        Invariants:
          - vol_trend_mean and vol_trend_slope are always computed over the most recent
            zero-run, even if the last bar is a breakout or trend bar.
          - breakout_confirmed is None for any bar that is not a flip bar.
          - Requires at least MIN_TREND_BARS non-NaN vol_trend bars in the zero-run.

  C1.3  Write unit tests (tests/test_volume.py — new file):
          TestVolumeProfile:
          - quiet consolidation (mean < 1.0): is_quiet=True
          - noisy consolidation (mean >= 1.0): is_quiet=False
          - declining slope: is_declining=True
          - rising slope: is_declining=False
          - breakout flip with vol_trend >= 1.2: breakout_confirmed=True
          - breakout flip with vol_trend < 1.2: breakout_confirmed=False
          - no flip (in consolidation): breakout_confirmed=None
          - missing column raises ValueError
          - too few bars raises ValueError

  C1.4  Smoke-test values (verified from parquet):
          A2A.MI 39-bar consolidation (2016-01-29 to 2016-03-23):
            vol_trend_mean  = 0.8847
            vol_trend_slope = -0.00632 /bar
            is_quiet        = True
            is_declining    = True
            breakout_confirmed = None  (last bar still in consolidation)

          POR.MI breakout bar (2018-09-03):
            vol_trend_now   = 10.9546
            breakout_confirmed = True  (10.9546 >= 1.2)
            (pre-breakout mean = 0.475, slope = +0.074 -> is_quiet=True despite slope > 0)
```

**File to be created**:
```
breakout/volume.py   <- VolumeProfile, assess_volume_profile
tests/test_volume.py <- unit tests for volume behavior
```

**Key design decision**: `is_quiet` uses `vol_trend_mean < 1.0` only, not slope.
Real data (POR.MI) shows healthy consolidations with slightly positive volume slope.
Gating on slope would silently drop valid accumulation setups.
