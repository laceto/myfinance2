"""
returns_calculator.py — Per-ticker return calculation module.

Computes lookback-window returns for both absolute close price and
relative close (rclose = ticker / FTSEMIB.MI benchmark) from the
analysis_results.parquet long-format DataFrame.

Two return types are produced for each (column, window) pair:

  ret_{col}_{N}d
      Simple point-to-point return:
          (P_last - P_{last-N}) / P_{last-N}

  cumret_{col}_{N}d
      Geometrically compounded return via log returns:
          1. Compute daily log returns:  ln(P_t / P_{t-1})
          2. Sum the last N log returns: Σ ln(P_i / P_{i-1})  i = last-N+1 … last
          3. Transform back:             exp(Σ) - 1

      The telescoping property guarantees:
          Σ ln(P_i/P_{i-1}) = ln(P_last / P_{last-N})
      so the result equals the simple return for a continuous series.
      The distinction matters when daily returns are summed arithmetically
      instead of geometrically: three +1% days compound to 3.0301%, not 3%.

Invariants:
    - Data is sorted ascending by date before any calculation.
    - Window >= len(series): returns NaN (not a crash or silent wrong value).
    - Base price == 0 / log(0): returns NaN (guards against inf/-inf).
"""

import math
import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LOOKBACK_WINDOWS: list[int] = [3, 5, 7, 10, 15, 30, 45, 60]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_returns(
    df: pd.DataFrame,
    col: str,
    windows: list[int],
) -> dict[str, float]:
    """
    Compute lookback returns for a single column across multiple windows.

    The function operates on the *last* available bar as "today".
    All prices are taken from a date-sorted view of the DataFrame,
    so row order in the caller does not matter.

    Args:
        df:      Per-ticker DataFrame. Must have a ``date`` column and
                 the column named by ``col``.
        col:     Name of the price column (``'close'`` or ``'rclose'``).
        windows: List of lookback windows in trading days (e.g. [3, 5, 60]).

    Returns:
        Dict mapping ``'ret_{col}_{N}d'`` → float.
        NaN when the window exceeds available history or the base price is zero.

    Raises:
        KeyError: If ``col`` is not present in ``df``.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame. Available: {list(df.columns)}")

    results: dict[str, float] = {}
    prefix = f"ret_{col}"

    if df.empty:
        return {f"{prefix}_{w}d": float("nan") for w in windows}

    # Sort ascending so iloc[-1] is always the most recent bar.
    prices: pd.Series = df.sort_values("date")[col].reset_index(drop=True)
    n = len(prices)
    price_last: float = float(prices.iloc[-1])

    for w in windows:
        key = f"{prefix}_{w}d"
        lookback_idx = n - 1 - w  # index of the base bar

        if lookback_idx < 0:
            # Insufficient history for this window.
            results[key] = float("nan")
            continue

        price_base: float = float(prices.iloc[lookback_idx])

        if price_base == 0.0 or math.isnan(price_base):
            # Guard against division by zero or NaN propagation.
            results[key] = float("nan")
            continue

        results[key] = (price_last - price_base) / price_base

    return results


def compute_cumulative_returns(
    df: pd.DataFrame,
    col: str,
    windows: list[int],
) -> dict[str, float]:
    """
    Compute geometrically compounded returns via log returns.

    Algorithm per window N:
        1. daily_log_ret[t] = ln(P[t] / P[t-1])          (length n-1 valid values)
        2. cumulative_log    = sum(daily_log_ret[-N:])     (last N daily log returns)
        3. result            = exp(cumulative_log) - 1

    The telescoping sum property means:
        sum(ln(P[i]/P[i-1])) = ln(P_last / P_{last-N})
    so for a contiguous series the result equals the simple return.
    The log route makes compounding explicit and guards against
    naive arithmetic summation of daily returns.

    Args:
        df:      Per-ticker DataFrame. Must have a ``date`` column and
                 the column named by ``col``.
        col:     Name of the price column (``'close'`` or ``'rclose'``).
        windows: List of lookback windows in trading days (e.g. [3, 5, 60]).

    Returns:
        Dict mapping ``'cumret_{col}_{N}d'`` → float.
        NaN when the window exceeds available history or any log return in the
        window is non-finite (price ≤ 0 or missing).

    Raises:
        KeyError: If ``col`` is not present in ``df``.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame. Available: {list(df.columns)}")

    prefix = f"cumret_{col}"

    if df.empty:
        return {f"{prefix}_{w}d": float("nan") for w in windows}

    # Sort ascending; compute daily log returns as a numpy array.
    prices: np.ndarray = (
        df.sort_values("date")[col].reset_index(drop=True).to_numpy(dtype=float)
    )
    n = len(prices)

    # daily_log_rets[i] = ln(prices[i+1] / prices[i]), length n-1.
    # Using numpy to catch divide-by-zero and log(0) → -inf silently.
    with np.errstate(divide="ignore", invalid="ignore"):
        daily_log_rets: np.ndarray = np.log(prices[1:] / prices[:-1])

    results: dict[str, float] = {}

    for w in windows:
        key = f"{prefix}_{w}d"

        # Need at least w+1 bars to form w daily log returns.
        if n < w + 1:
            results[key] = float("nan")
            continue

        # Last w elements of daily_log_rets correspond to the last w bars.
        window_log_rets = daily_log_rets[n - 1 - w : n - 1]

        if not np.all(np.isfinite(window_log_rets)):
            # Catches log(0) = -inf and log(negative) = nan.
            results[key] = float("nan")
            continue

        results[key] = math.exp(float(window_log_rets.sum())) - 1

    return results


# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------


def build_returns_dashboard(
    data_dict: dict[str, pd.DataFrame],
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Build a wide-format returns dashboard — one row per ticker.

    For each ticker the function computes returns on both ``close``
    (absolute EUR price) and ``rclose`` (price relative to the
    FTSEMIB.MI benchmark) across all requested lookback windows.

    Columns in the output DataFrame:
        ticker             — Yahoo Finance symbol (e.g. "AVIO.MI")
        last_date          — Last available trading date for this ticker
        close              — Last absolute close price
        rclose             — Last relative close value
        ret_close_3d       — 3-day simple absolute return (and so on for each window)
        …
        ret_rclose_3d      — 3-day simple relative return (and so on for each window)
        …
        cumret_close_3d    — 3-day compounded absolute return via log (and so on)
        …
        cumret_rclose_3d   — 3-day compounded relative return via log (and so on)
        …

    Args:
        data_dict: Mapping of ticker → per-ticker DataFrame
                   (as produced by loading analysis_results.parquet).
        windows:   Lookback windows in trading days.
                   Defaults to LOOKBACK_WINDOWS = [3, 5, 7, 10, 15, 30, 45, 60].

    Returns:
        Wide DataFrame sorted by ticker (ascending), ready for Excel export.

    Raises:
        ValueError: If data_dict is empty.
    """
    if not data_dict:
        raise ValueError("data_dict is empty — nothing to compute.")

    if windows is None:
        windows = LOOKBACK_WINDOWS

    rows: list[dict] = []

    for ticker, df in data_dict.items():
        if df.empty:
            log.warning("Ticker %s has no data rows — skipping.", ticker)
            continue

        sorted_df = df.sort_values("date")
        last_row = sorted_df.iloc[-1]

        row: dict = {
            "ticker": ticker,
            "last_date": last_row["date"],
            "close": float(last_row["close"]) if "close" in df.columns else float("nan"),
            "rclose": float(last_row["rclose"]) if "rclose" in df.columns else float("nan"),
        }

        # Absolute price — simple and cumulative returns
        if "close" in df.columns:
            row.update(compute_returns(df, col="close", windows=windows))
            row.update(compute_cumulative_returns(df, col="close", windows=windows))
        else:
            log.warning("Ticker %s missing 'close' column.", ticker)
            for w in windows:
                row[f"ret_close_{w}d"] = float("nan")
                row[f"cumret_close_{w}d"] = float("nan")

        # Relative price — simple and cumulative returns
        if "rclose" in df.columns:
            row.update(compute_returns(df, col="rclose", windows=windows))
            row.update(compute_cumulative_returns(df, col="rclose", windows=windows))
        else:
            log.warning("Ticker %s missing 'rclose' column.", ticker)
            for w in windows:
                row[f"ret_rclose_{w}d"] = float("nan")
                row[f"cumret_rclose_{w}d"] = float("nan")

        rows.append(row)

    if not rows:
        raise ValueError("No rows were produced. All tickers were skipped (empty data).")

    dashboard = pd.DataFrame(rows)

    # Canonical column order: metadata | simple returns | cumulative returns
    ret_close_cols = [f"ret_close_{w}d" for w in windows]
    ret_rclose_cols = [f"ret_rclose_{w}d" for w in windows]
    cumret_close_cols = [f"cumret_close_{w}d" for w in windows]
    cumret_rclose_cols = [f"cumret_rclose_{w}d" for w in windows]
    ordered_cols = (
        ["ticker", "last_date", "close", "rclose"]
        + ret_close_cols
        + ret_rclose_cols
        + cumret_close_cols
        + cumret_rclose_cols
    )
    # Keep any unexpected columns at the end
    extra_cols = [c for c in dashboard.columns if c not in ordered_cols]
    dashboard = dashboard[ordered_cols + extra_cols]

    return dashboard.sort_values("ticker").reset_index(drop=True)
