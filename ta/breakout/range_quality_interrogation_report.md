# INTERROGATION REPORT: ta/breakout/range_quality.py

**Date:** 2026-04-03
**Auditor:** Evil Code Reviewer
**Verdict before reading:** Guilty

---

## Opening Statement

This file presents itself as a "foundational analytical primitive" — the kind of language used by engineers who want you to believe the foundations were carefully laid. They were not. What we have is a collection of hardcoded thresholds that were clearly eyeballed on one or two real tickers, a state machine that silently makes decisions about what constitutes a "touch" based on values that have never been validated against a distribution of real-world consolidation windows, and a Wilder-smoothing-free OLS slope being passed off as a trading signal. The code is clean-looking. That is the most dangerous kind of code. It reads like a professional wrote it. A professional *would have measured things.*

---

## The Charges

---

**Constant #1: `TOUCH_HI_THRESHOLD = 85.0` and `TOUCH_LO_THRESHOLD = 15.0`**

*What it is:* Lines 53–54. Hardcoded percentile thresholds that define whether price is "at resistance" or "at support."

*Why? (Layer 1):* Presumably these felt symmetrical and round. 85 and 15 are equidistant from 50. Someone drew a line.
*Why? (Layer 2):* Why 85 and not 80 or 90? The difference between calling something a resistance touch at 80% vs 85% changes the touch count — which feeds directly into `n_resistance_touches`, which feeds directly into the AI system prompt that drives trading decisions.
*Why? (Layer 3):* There is no reference to any empirical study, backtest, or distribution analysis of where price *actually* reverses in range-bound Italian equities. The smoke test uses *one* ticker (SCM.MI). One.

*Production Explosion #1:* A ticker with a structural upward drift during consolidation spends 40% of its consolidation bars between 80–85%. With threshold=85, zero touches are counted. With threshold=80, four are counted. The AI is told this is a "poorly-defined range" and recommends flat. It was a coiled spring.
*Production Explosion #2:* Volatile small-caps on Borsa Italiana routinely whipsaw. A threshold of 85% for a high-ATR ticker means "at resistance" requires the ticker to be within the top 15% of its range — which it may overshoot and immediately reverse from, getting counted as zero touches because the retreat happened in one bar before the state machine could register it.

*Evidence demanded:* A backtest across all tickers in `ticker.xlsx` showing that the distribution of reversal points clusters around 85/15, not 80/20 or 90/10. Or a sensitivity analysis showing that changing these thresholds by ±5pp does not materially change the touch count distribution.

*Verdict:* These must be promoted to config or at minimum have a docstring citation explaining *why* 85/15. "It felt right" is not documented here, but it's the real reason.

---

**Constant #2: `RETREAT_THRESHOLD = 65.0` and `BOUNCE_THRESHOLD = 35.0`**

*What it is:* Lines 55–56. The gray zone boundaries.

*Why? (Layer 1):* These create a 20-point gray zone between the touch entry and exit thresholds. 85→65 for resistance, 15→35 for support.
*Why? (Layer 2):* A 20-point gray zone is 20% of the entire range width. On a 10%-band-width range, this is a 2% price move in absolute relative terms. On a 3%-band-width compressed range (the SCM.MI smoke test ends at 3.1%), the gray zone collapses to 0.6% — which may be inside the bid-ask spread of a thin Italian small-cap. The constants are not scale-aware.
*Why? (Layer 3):* The gray zone width was chosen as a fixed percentage of the range, but the range itself varies by an order of magnitude across the ticker universe (3%–15% from the smoke test data alone). What is a "meaningful retreat" in a wide range is noise in a compressed range.

*Production Explosion #1:* A compressed range (band_width=3%) has an effective gray zone of 0.6% of rclose. A single noisy bar prints at range_pct=64% (one tick below the retreat threshold), ending the touch, then immediately bounces back to 90%. The state machine counts 2 touches. The actual range has 1 resistance level. The setup is assessed as "clean range, 2 touches." It is not.
*Production Explosion #2:* The gray zone is evaluated on `range_position_pct`, not absolute price. When `rhi_20 == rlo_20` (zero-width range), any NaN values skip the state machine silently. An entire consolidation phase with zero-range bars is invisibly dropped from the touch count.

*Evidence demanded:* Sensitivity analysis of touch counts across the full ticker history as band_width_pct varies. The gray zone should probably scale with band_width.

*Verdict:* At minimum, add a warning log when band_width_pct < some threshold indicating the gray zone may be meaningless. Ideally: scale the gray zone as a function of band_width.

---

**Constant #3: `SIDEWAYS_SLOPE_THRESHOLD = 0.15`**

*What it is:* Line 58. The maximum `|slope_pct_per_day|` to be classified as "sideways."

*Why? (Layer 1):* 0.15%/day seems conservative.
*Why? (Layer 2):* 0.15%/day × 40 bars = 6% cumulative drift. Is a range that drifts 6% in one direction genuinely "consolidating"?
*Why? (Layer 3):* This threshold was validated against SCM.MI where `slope = 0.0087%/day`. That's a *perfectly* flat example. There is no test with a slope of 0.14%/day (just under threshold) to verify the classification makes trading sense at the boundary.

*Production Explosion #1:* A stock in a confirmed bearish phase drifts at -0.14%/day during a 40-bar "consolidation" window. `classify_trend` returns `is_sideways=True`. The breakout setup is assessed as valid. The "breakout" is a continuation of the downtrend misclassified as a range.
*Production Explosion #2:* If the benchmark (FTSEMIB.MI) is in a strong trend, a stock with zero absolute drift shows non-zero relative slope. The threshold of 0.15 may have been calibrated only during quiet benchmark periods.

*Evidence demanded:* Distribution of `slope_pct_per_day` across all historical consolidation windows in the dataset, with the 0.15 threshold marked. Is 0.15 the 75th percentile of "real consolidations"? The 90th? Nobody knows.

*Verdict:* Must be in `config.json`. Add a boundary test at 0.14 and 0.16.

---

**Constant #4: `MIN_TREND_BARS = 5`**

*What it is:* Line 59. The minimum bars required for a "reliable" OLS estimate.

*Why? (Layer 1):* 5 bars is the minimum to avoid a degenerate regression.
*Why? (Layer 2):* An OLS fit to 5 noisy financial bars has standard error enormous relative to the 0.15%/day threshold. At typical Borsa Italiana daily volatility (0.5–1.5%), the 95% CI on the slope swamps the threshold.
*Why? (Layer 3):* The comment says "minimum for a reliable OLS estimate." That is a lie. 5 data points does not produce a reliable OLS estimate of anything in a noisy financial series.

*Production Explosion #1:* `assess_range` called on a ticker that just entered consolidation 5 bars ago. OLS on 5 bars gives `slope = 0.14%/day`. The setup is flagged as valid consolidation. The ticker breaks out the next day continuing a trend that never paused.
*Production Explosion #2:* If the zero-run is exactly 5 bars long (brief pause in a trend), `classify_trend` returns without error but with meaningless output, and the range is assessed as if it were a genuine consolidation.

*Evidence demanded:* Monte Carlo simulation showing that OLS slope estimation at 5 bars, given typical Italian equity relative price volatility, produces false "sideways" classification below an acceptable rate.

*Verdict:* Raise to minimum 10. Document the statistical justification.

---

**Constant #5: `COMPRESSION_RANK_THRESHOLD = 25.0`**

*What it is:* Line 61. The percentile below which a range is "historically tight."

*Why? (Layer 1):* Bottom quartile seems reasonable.
*Why? (Layer 2):* Applied uniformly regardless of ticker history length. A new listing with 200 bars has a different distribution than a stock with 2500 bars.
*Why? (Layer 3):* `is_compressed` requires BOTH `slope < 0` (over 40 bars) AND `rank < 25` (over 252 bars). These windows are different lengths and reference different time periods. A range actively narrowing from a historically elevated level would fail the rank test despite being potentially more interesting than a range at a tight level that stopped narrowing.

*Production Explosion #1:* A newly listed stock with 200 bars of history has `band_width_pct_rank` computed on 200 values, not 252. The rank is not comparable to a full-history ticker. `is_compressed=True` means different things for different tickers with no way to detect which case you're in.
*Production Explosion #2:* Rank of 26% (just above threshold): `is_compressed=False` returned. By any practical definition the range is nearly as compressed as 24% would be. Binary threshold on a percentile rank is cliff-edge decision making.

*Evidence demanded:* Show that `rank < 25` produces better out-of-sample breakout predictions than `rank < 30` or `rank < 20`. This has never been backtested.

*Verdict:* Move to config. Add `history_available` and `is_rank_reliable` fields to `VolatilityState`.

---

**Design Felony #1: The `_ols_slope` import alias**

*What it is:* Line 47. `from ta.utils import ols_slope as _ols_slope`.

*Why? (Layer 1):* The underscore prefix signals "private."
*Why? (Layer 2):* The function is imported from a public module. It is not private. The underscore is a lie.
*Why? (Layer 3):* Someone searching for `_ols_slope` will find only the import line. Fifteen minutes wasted per encounter.

*Production Explosion #1:* New engineer assumes `_ols_slope` is defined locally, searches, fails, looks in the wrong module.
*Production Explosion #2:* Linters flagging private member access from outside will produce false alarms that get globally silenced.

*Verdict:* `from ta.utils import ols_slope`. Name it what it is.

---

**Design Felony #2: `assess_range` silently skips unlimited trailing non-zero bars**

*What it is:* Lines 476–478. The backward walk skips any trailing non-zero bars with no upper bound.

*Why? (Layer 1):* Allows calling `assess_range` on the breakout bar itself.
*Why? (Layer 2):* No limit on how many non-zero bars are skipped. 200-bar trend + 5-bar pause = skip 200 bars silently.
*Why? (Layer 3):* The caller catches ValueError and returns `None`. The AI is told "no consolidation = ticker in trend." It is not in a trend. It had a consolidation too brief to analyze. Silent misclassification.

*Production Explosion #1:* A ticker exits a 150-bar trend, consolidates for 4 bars (below MIN_TREND_BARS), breaks out. `assess_range` raises. AI says "no consolidation." The breakout from the 4-bar pause is invisible. Valid signal missed.
*Production Explosion #2:* The backward walk is O(n) Python loop. For 2500-bar histories in batch processing this is a performance cliff.

*Verdict:* Add `max_trend_skip` parameter with a documented default. Log a warning when hit.

---

**Design Felony #3: NaN from zero-range bars silently dropped from touch counting**

*What it is:* Lines 499–502. `rng.where(rng != 0)` produces NaN when `rhi_20 == rlo_20`. NaN is silently skipped by `count_touches`.

*Why? (Layer 1):* Zero-division guarded. NaN handled. Looks correct.
*Why? (Layer 2):* `rhi_20 == rlo_20` signals a trading halt or data error, not a normal gap. Silent skip treats pathological data as ordinary missing data.
*Why? (Layer 3):* If 10 of 40 consolidation bars are halted, touch counts are computed on 30 bars reported as if from 40. No warning. No flag. No way to detect this in the output.

*Production Explosion #1:* Trading halt during consolidation produces NaN for 15 bars. Those bars span 2 resistance touches. Touch count = 0. Range assessed as "poorly-defined." Valid setup missed.
*Production Explosion #2:* After NaN skip, remaining bars fall below MIN_TREND_BARS. ValueError raised. Caller returns None. User told "no consolidation." The ticker was in 40-bar consolidation with a data quality issue.

*Evidence demanded:* Count the frequency of `rng == 0` in the actual parquet. If it occurs once in production, this is a real bug.

*Verdict:* Count and log NaN range_pct values. Raise or return degraded result if NaN count exceeds 10% of window bars.

---

**Control Flow Misdemeanor #1: `.shift()` without RangeIndex guard in `breakout_prior_consolidation_length`**

*What it is:* Lines 257–261. `rbo.shift(1)` and `age.shift(1)` on a Series that may have a DatetimeIndex from the parquet.

*Why? (Layer 1):* `shift(1)` shifts by 1 period. Standard.
*Why? (Layer 2):* On a DatetimeIndex with gaps (holidays, data gaps), `shift(1)` shifts by one calendar unit, not one row. Bars after gaps produce NaN instead of the previous bar's value.
*Why? (Layer 3):* The function says "sorted ascending by date" but does not require RangeIndex. Real parquet data has DatetimeIndex. Italian public holidays create gaps.

*Production Explosion #1:* Italian national holiday creates a gap. `rbo.shift(1)` produces NaN for the post-holiday bar. `is_breakout_flip` evaluates to False. A real breakout flip is missed.
*Production Explosion #2:* `age.shift(1)` produces NaN for the same bar. Consolidation length reported as NaN at the flip bar.

*Verdict:* Add `.reset_index(drop=True)` before operating. Or assert `isinstance(rbo.index, pd.RangeIndex)`.

---

**Control Flow Misdemeanor #2: `measure_volatility_compression` uses `iloc[-history_bars:]` without short-history guard**

*What it is:* Lines 369–370. `bw.iloc[-252:]` silently returns fewer bars for new listings.

*Why? (Layer 1):* Python slicing is permissive with short series.
*Why? (Layer 2):* Percentile rank computed on 100 bars is a different metric than on 252 bars, but both are compared to the same threshold (25.0).
*Why? (Layer 3):* A stock with 5 bars of history passes `MIN_TREND_BARS` check and computes `band_width_pct_rank` on 5 values. The rank is quantized to 20pp steps. `is_compressed` on 5 bars of history is noise dressed as signal.

*Production Explosion #1:* New listing with 200 bars: rank computed on 200 observations, not 252. Not comparable to full-history tickers. The AI receives `is_compressed=True` or `False` with no indication that the signal is based on 200 bars.
*Production Explosion #2:* 5-bar history stock: rank can only be 20%, 40%, 60%, 80%, or 100%. `is_compressed=True` requires rank=20% — the absolute minimum of 5 bars. This is nearly random.

*Verdict:* Add `history_available: int` and `is_rank_reliable: bool` to `VolatilityState`. The AI must know when it's operating on thin history.

---

## The Sentence

**Culpability Score: 6/10**

**Crime Counts:**
- Assumptions (unproven): 5
- Magic constants (unjustified): 5
- Design felonies: 3
- Control flow misdemeanors: 2
- Missing error handling: 2
- Security violations: 0

**Most Dangerous Offense:** `SIDEWAYS_SLOPE_THRESHOLD = 0.15` combined with the unlimited trailing non-zero bar skip in `assess_range`. Together they silently classify a brief pause in an active trend as a valid sideways consolidation — feeding directly into the AI system prompt as a high-quality trading signal, without any warning.

**Rehabilitation Plan (most damning first):**
1. `[ ]` Move `SIDEWAYS_SLOPE_THRESHOLD`, `TOUCH_HI_THRESHOLD`, `TOUCH_LO_THRESHOLD`, `COMPRESSION_RANK_THRESHOLD` to `config.json`.
2. `[ ]` Add `max_trend_skip` to `assess_range` and log a warning when the backward walk skips more than N non-zero bars.
3. `[ ]` Add `history_available` and `is_rank_reliable` to `VolatilityState`.
4. `[ ]` Guard `breakout_prior_consolidation_length` against non-RangeIndex with `.reset_index(drop=True)`.
5. `[ ]` Count and log NaN `range_pct` values in `assess_range`.
6. `[ ]` Raise `MIN_TREND_BARS` to 10 with statistical justification.
7. `[ ]` Remove the `_ols_slope` alias.

**Final Statement:**
This code was written by someone who knows what they're doing and then stopped caring just before the finish line. The architecture is sound. The tests are real. The docstrings are not lies. But every threshold is a wish, every constant is untested at its boundary, and the silent degradation paths — short history, data gaps, brief non-zero skips — are the exact scenarios that occur in production on Italian small-caps and are the exact scenarios that will produce a confident-sounding AI analysis built on sand. The code deserves to continue existing. It does not deserve to remain unchanged.
