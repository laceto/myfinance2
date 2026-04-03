"""
utils.py — Shared mathematical utilities for the ta package.

Functions here are low-level helpers with no domain knowledge.
They are imported by ta.breakout and ta.ma modules to avoid duplication.
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
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0.0:
        return 0.0
    return float(np.sum((x - x_mean) * (values - y_mean)) / denom)
