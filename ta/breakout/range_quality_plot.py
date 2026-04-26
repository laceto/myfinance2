"""
range_quality_plot.py — Visualisation of range-quality indicators for a single ticker.

Three stacked panels:
  1. Price (rclose) + rhi_20 / rlo_20 bands, rbo_20 background shading,
     vertical lines at resistance/support touch entries.
  2. range_position_pct with threshold lines (85 / 65 / 35 / 15) and touch markers.
  3. band_width_pct with OLS trend line over the analysis window.

Usage (notebook):
    from ta.breakout.range_quality_plot import plot_range_quality
    fig = plot_range_quality('STMMI.MI')
    plt.show()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ta.breakout.range_quality import (
    BOUNCE_THRESHOLD,
    DEFAULT_MAX_GAP_BARS,
    RETREAT_THRESHOLD,
    TOUCH_HI_THRESHOLD,
    TOUCH_LO_THRESHOLD,
    RangeQualityConfig,
    VolatilityState,
    assess_range,
    measure_volatility_compression,
)
from ta.utils import ols_slope

_DEFAULT_DATA_PATH = Path("data/results/it/analysis_results.parquet")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_ticker(ticker: str, data_path: Path) -> pd.DataFrame:
    """Load, deduplicate, and sort one ticker from analysis_results.parquet."""
    df = pd.read_parquet(data_path)
    df = df[df["symbol"] == ticker].copy()
    if df.empty:
        raise ValueError(f"Ticker '{ticker}' not found in {data_path}.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    df = df.reset_index(drop=True)
    df["rbo_20"] = df["rbo_20"].fillna(0).astype(np.int8)
    return df


# ---------------------------------------------------------------------------
# State machine helpers
# ---------------------------------------------------------------------------


def _touch_entries(
    range_pct: pd.Series,
    hi_thresh: float = TOUCH_HI_THRESHOLD,
    lo_thresh: float = TOUCH_LO_THRESHOLD,
    retreat: float = RETREAT_THRESHOLD,
    bounce: float = BOUNCE_THRESHOLD,
    max_gap_bars: int = DEFAULT_MAX_GAP_BARS,
) -> tuple[list[int], list[int]]:
    """
    Run the touch state machine and return the bar indices where new touches start.

    Mirrors count_touches() exactly — same gray-zone logic and NaN-gap reset.
    Returns (resistance_entry_indices, support_entry_indices).
    """
    res_entries: list[int] = []
    sup_entries: list[int] = []
    in_res = in_sup = False
    consec_nan = 0

    for i, val in enumerate(range_pct):
        if pd.isna(val):
            consec_nan += 1
            if consec_nan >= max_gap_bars:
                in_res = in_sup = False
            continue
        consec_nan = 0

        if not in_res:
            if val >= hi_thresh:
                in_res = True
                res_entries.append(i)
        else:
            if val < retreat:
                in_res = False

        if not in_sup:
            if val <= lo_thresh:
                in_sup = True
                sup_entries.append(i)
        else:
            if val > bounce:
                in_sup = False

    return res_entries, sup_entries


def _find_consolidation_window(rbo_arr: np.ndarray) -> tuple[int, int]:
    """
    Return (start, end) row indices of the most recent zero-run.

    Mirrors assess_range's vectorised zero-run finder. Returns (0, -1) if
    no consolidation window exists (caller should catch ValueError from assess_range
    before calling this).
    """
    zero_mask = rbo_arr == 0
    zero_positions = np.flatnonzero(zero_mask)
    if zero_positions.size == 0:
        raise ValueError("No consolidation window (rbo_20 == 0) found.")
    end = int(zero_positions[-1])
    nonzero_before = np.flatnonzero(~zero_mask[:end])
    start = int(nonzero_before[-1]) + 1 if nonzero_before.size > 0 else 0
    return start, end


# ---------------------------------------------------------------------------
# Axes helpers
# ---------------------------------------------------------------------------


def _shade_rbo(ax: plt.Axes, rbo_20: pd.Series) -> None:
    """Shade background by rbo_20: green=+1 (bullish), red=-1 (bearish), none=0."""
    for i in range(len(rbo_20) - 1):
        val = rbo_20.iloc[i]
        if val > 0:
            ax.axvspan(i, i + 1, color="limegreen", alpha=0.07, lw=0)
        elif val < 0:
            ax.axvspan(i, i + 1, color="salmon", alpha=0.07, lw=0)


def _shade_analysis_window(ax: plt.Axes, start: int, end: int, n: int) -> None:
    """Highlight the capped analysis window used by assess_range."""
    s = max(0, start)
    e = min(n - 1, end)
    if s < e:
        ax.axvspan(s, e, color="gold", alpha=0.10, lw=0)


def _set_date_ticks(ax: plt.Axes, dates: pd.Series, n_ticks: int = 12) -> None:
    n = len(dates)
    step = max(1, n // n_ticks)
    positions = list(range(0, n, step))
    labels = [dates.iloc[i].strftime("%b %Y") for i in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)


def _annotate_setup(
    ax: plt.Axes,
    setup,           # RangeSetup
    vol: VolatilityState,
) -> None:
    """Text box in the top-right corner of the price panel."""
    compressed = "YES" if vol.is_compressed else "no"
    sideways = "YES" if setup.is_sideways else "no"
    rank_flag = "" if vol.is_rank_reliable else " (*)"
    text = (
        f"Resistance touches : {setup.n_resistance_touches}\n"
        f"Support touches    : {setup.n_support_touches}\n"
        f"Sideways           : {sideways}  (slope {setup.slope_pct_per_day:+.4f} %/day)\n"
        f"Consolidation bars : {setup.consolidation_bars}\n"
        f"Band width         : {setup.band_width_pct:.2f}%\n"
        f"BW rank            : {vol.band_width_pct_rank:.1f}p{rank_flag}  compressed: {compressed}"
    )
    ax.text(
        0.99, 0.97, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="#aaaaaa"),
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def plot_range_quality(
    ticker: str,
    plot_bars: int = 252,
    window_bars: int = 40,
    data_path: Path | str | None = None,
    config: RangeQualityConfig | None = None,
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Plot range-quality indicators for a single ticker.

    Args:
        ticker:      Ticker symbol, e.g. 'STMMI.MI'.
        plot_bars:   Trailing bars to display (default 252 ≈ 1 trading year).
        window_bars: Consolidation window length passed to assess_range (default 40).
        data_path:   Path to analysis_results.parquet. Defaults to project data path.
        config:      RangeQualityConfig for threshold overrides.
        figsize:     Matplotlib figure size.

    Returns:
        matplotlib Figure. Call plt.show() or fig.savefig() after.

    Raises:
        ValueError: if ticker not found or no consolidation window exists.

    Panels:
        1. rclose + rhi_20/rlo_20 bands.  Background: green=bullish breakout,
           red=bearish breakout.  Gold band = analysis window.  Orange vertical
           lines = resistance touch entries; blue = support touch entries.
        2. range_position_pct with threshold lines at 85 / 65 / 35 / 15.
           Orange ▲ = resistance touch entry; blue ▼ = support touch entry.
        3. band_width_pct with OLS trend line over the analysis window
           (green = narrowing / compressed; red = expanding).
    """
    if data_path is None:
        data_path = _DEFAULT_DATA_PATH
    if config is None:
        config = RangeQualityConfig()

    df = _load_ticker(ticker, Path(data_path))

    # Run on full history for accurate results.
    setup = assess_range(df, window_bars=window_bars, config=config)
    vol_state = measure_volatility_compression(
        df,
        window_bars=window_bars,
        rank_threshold=config.compression_rank_threshold,
    )

    # Find consolidation window bounds in full-history coordinates.
    rbo_arr = df["rbo_20"].to_numpy(dtype=np.int8)
    con_start, con_end = _find_consolidation_window(rbo_arr)
    con_capped_start = max(con_start, con_end - window_bars + 1)

    # Slice to the visible view.
    view = df.iloc[-plot_bars:].copy().reset_index(drop=True)
    offset = len(df) - len(view)  # view index i → df index (i + offset)

    dates = view["date"]
    rclose = view["rclose"]
    rhi_20 = view["rhi_20"]
    rlo_20 = view["rlo_20"]
    rbo_20 = view["rbo_20"]
    n = len(view)

    # range_position_pct over the view
    rng = rhi_20 - rlo_20
    range_pct = ((rclose - rlo_20) / rng.where(rng.abs() > 1e-8) * 100).reset_index(drop=True)

    # band_width_pct over the view
    rclose_safe = rclose.where(rclose.abs() > 1e-8)
    bw = ((rhi_20 - rlo_20) / rclose_safe * 100).reset_index(drop=True)

    # Touch entries within the view
    res_entries, sup_entries = _touch_entries(
        range_pct,
        hi_thresh=config.touch_hi_threshold,
        lo_thresh=config.touch_lo_threshold,
        retreat=config.retreat_threshold,
        bounce=config.bounce_threshold,
        max_gap_bars=config.max_gap_bars,
    )

    # Consolidation window in view coordinates
    win_start_v = con_capped_start - offset
    win_end_v = con_end - offset

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, (ax_price, ax_rng, ax_bw) = plt.subplots(
        3, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1.5]},
    )

    x = np.arange(n)

    # -----------------------------------------------------------------------
    # Panel 1 — Price
    # -----------------------------------------------------------------------
    _shade_rbo(ax_price, rbo_20)
    _shade_analysis_window(ax_price, win_start_v, win_end_v, n)

    ax_price.fill_between(x, rlo_20, rhi_20, color="gray", alpha=0.06)
    ax_price.plot(x, rhi_20, color="firebrick", lw=1.0, ls="--", alpha=0.75, label="rhi_20")
    ax_price.plot(x, rlo_20, color="steelblue", lw=1.0, ls="--", alpha=0.75, label="rlo_20")
    ax_price.plot(x, rclose, color="black", lw=1.5, label="rclose")

    for i in res_entries:
        ax_price.axvline(i, color="darkorange", lw=0.9, alpha=0.55, ls=":")
    for i in sup_entries:
        ax_price.axvline(i, color="royalblue", lw=0.9, alpha=0.55, ls=":")

    _annotate_setup(ax_price, setup, vol_state)
    ax_price.set_ylabel("relative price")
    ax_price.set_title(
        f"{ticker} — range quality  (last {plot_bars} bars, analysis window={window_bars})",
        fontsize=11,
    )
    ax_price.legend(loc="upper left", fontsize=8)

    # -----------------------------------------------------------------------
    # Panel 2 — range_position_pct
    # -----------------------------------------------------------------------
    _shade_rbo(ax_rng, rbo_20)
    _shade_analysis_window(ax_rng, win_start_v, win_end_v, n)

    # Gray zones
    ax_rng.axhspan(
        config.retreat_threshold, config.touch_hi_threshold,
        color="salmon", alpha=0.10,
    )
    ax_rng.axhspan(
        config.touch_lo_threshold, config.bounce_threshold,
        color="cornflowerblue", alpha=0.10,
    )

    # Threshold lines
    thresholds = [
        (config.touch_hi_threshold, "firebrick",   f"resist entry {config.touch_hi_threshold:.0f}%"),
        (config.retreat_threshold,  "darkorange",  f"resist exit  {config.retreat_threshold:.0f}%"),
        (config.bounce_threshold,   "deepskyblue", f"support exit {config.bounce_threshold:.0f}%"),
        (config.touch_lo_threshold, "steelblue",   f"support entry {config.touch_lo_threshold:.0f}%"),
    ]
    for y, color, label in thresholds:
        ax_rng.axhline(y, color=color, lw=0.9, ls="--", alpha=0.85, label=label)

    ax_rng.plot(x, range_pct, color="dimgray", lw=1.2, label="range_pct")

    for i in res_entries:
        if 0 <= i < n and not pd.isna(range_pct.iloc[i]):
            ax_rng.scatter(i, range_pct.iloc[i], color="darkorange", zorder=5, s=55, marker="^")
    for i in sup_entries:
        if 0 <= i < n and not pd.isna(range_pct.iloc[i]):
            ax_rng.scatter(i, range_pct.iloc[i], color="royalblue", zorder=5, s=55, marker="v")

    ax_rng.set_ylabel("range_pct (%)")
    ax_rng.legend(loc="upper left", fontsize=7, ncol=2)

    # -----------------------------------------------------------------------
    # Panel 3 — band_width_pct
    # -----------------------------------------------------------------------
    _shade_analysis_window(ax_bw, win_start_v, win_end_v, n)
    ax_bw.plot(x, bw, color="mediumpurple", lw=1.2, label="band_width_pct")

    # OLS trend line over the last window_bars of the view
    bw_clean = bw.dropna()
    if len(bw_clean) >= window_bars:
        bw_win = bw_clean.iloc[-window_bars:].to_numpy(dtype=float)
        slope = ols_slope(bw_win)
        mid_x = (len(bw_win) - 1) / 2
        intercept = bw_win.mean() - slope * mid_x
        x_start = n - window_bars
        trend_y = intercept + slope * np.arange(len(bw_win))
        ols_color = "green" if slope < 0 else "red"
        direction = "narrowing" if slope < 0 else "expanding"
        ax_bw.plot(
            np.arange(x_start, x_start + len(bw_win)),
            trend_y,
            color=ols_color, lw=1.6, ls="--",
            label=f"OLS {slope:+.4f} %/bar ({direction})",
        )

    # Rank annotation
    rank_note = f"rank {vol_state.band_width_pct_rank:.1f}p"
    if not vol_state.is_rank_reliable:
        rank_note += f" ({vol_state.history_available}d)"
    ax_bw.annotate(
        rank_note,
        xy=(n - 1, bw.iloc[-1]),
        xytext=(-5, 10),
        textcoords="offset points",
        fontsize=8,
        color="mediumpurple",
        ha="right",
    )

    ax_bw.set_ylabel("band_width\n(%)")
    ax_bw.legend(loc="upper left", fontsize=7)

    _set_date_ticks(ax_bw, dates)
    fig.tight_layout()
    return fig
