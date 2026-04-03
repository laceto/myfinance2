"""
utils.py — Shared mathematical utilities for the ta package.

Functions here are low-level helpers with no domain knowledge.
They are imported by ta.breakout and ta.ma modules to avoid duplication.

Public API
----------
ols_slope(values)          — OLS slope only (backward-compatible).
ols_slope_r2(values)       — OLS slope + R² (prefer this when noise detection matters).
"""

from __future__ import annotations

import numpy as np


def ols_slope(values: np.ndarray) -> float:
    """
    Return the OLS linear regression slope of `values` vs a zero-based integer index.

    Computes slope = cov(x, y) / var(x) where x = [0, 1, ..., n-1].

    Args:
        values: 1-D numpy array of floats. Must have at least 2 elements.

    Returns:
        Slope as float. Returns 0.0 for a constant series (zero variance in x = impossible,
        but zero variance in y → slope = 0 naturally; single-element series → 0.0).
    """
    slope, _ = ols_slope_r2(values)
    return slope


def ols_slope_r2(values: np.ndarray) -> tuple[float, float]:
    """
    Return (slope, R²) of the OLS linear regression of `values` vs a zero-based index.

    R² measures the fraction of variance explained by the linear trend:
      - R² < 0.3  → slope is dominated by noise; treat as informational only.
      - R² ≥ 0.7  → strong linear trend in the window.

    Computes slope = cov(x, y) / var(x) where x = [0, 1, ..., n-1].

    Args:
        values: 1-D numpy array of floats. Must have at least 2 elements.

    Returns:
        (slope, r2) as floats.
        Returns (0.0, 0.0) for series with fewer than 2 elements or zero total variance.

    Example:
        >>> ols_slope_r2(np.array([1.0, 2.0, 3.0, 4.0]))
        (1.0, 1.0)   # perfect linear trend
        >>> ols_slope_r2(np.array([2.0, 2.0, 2.0]))
        (0.0, 0.0)   # constant series
    """
    n = len(values)
    if n < 2:
        return 0.0, 0.0

    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()

    ss_x = float(np.sum((x - x_mean) ** 2))
    if ss_x == 0.0:
        return 0.0, 0.0

    slope     = float(np.sum((x - x_mean) * (values - y_mean)) / ss_x)
    intercept = y_mean - slope * x_mean
    y_pred    = slope * x + intercept

    ss_res = float(np.sum((values - y_pred) ** 2))
    ss_tot = float(np.sum((values - y_mean) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return slope, max(0.0, r2)  # clamp: numerical noise can push r2 slightly negative
