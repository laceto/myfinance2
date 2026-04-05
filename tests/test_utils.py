"""
test_utils.py — Unit tests for ta.utils shared mathematical helpers.

Coverage
--------
ols_slope_r2:
  - known linear series → slope=1.0, R²=1.0
  - constant series     → slope=0.0, R²=0.0
  - single element      → (0.0, 0.0)
  - NaN in input        → (0.0, 0.0)  [defensive guard]
  - partial NaN input   → (0.0, 0.0)
  - R² clamped to [0,1] (no negative R² from numerical noise)

ols_slope (thin wrapper):
  - delegates to ols_slope_r2; verified for one known case
"""

from __future__ import annotations

import numpy as np
import pytest

from ta.utils import ols_slope, ols_slope_r2


class TestOlsSlopeR2:

    def test_perfect_linear_slope_and_r2(self):
        """[1,2,3,4] → slope=1.0, R²=1.0 (perfect fit)."""
        slope, r2 = ols_slope_r2(np.array([1.0, 2.0, 3.0, 4.0]))
        assert slope == pytest.approx(1.0, abs=1e-9)
        assert r2    == pytest.approx(1.0, abs=1e-9)

    def test_constant_series_slope_zero_r2_zero(self):
        """Flat series: slope=0, R²=0 (no variance to explain)."""
        slope, r2 = ols_slope_r2(np.full(10, 5.0))
        assert slope == pytest.approx(0.0, abs=1e-9)
        assert r2    == pytest.approx(0.0, abs=1e-9)

    def test_single_element_returns_zeros(self):
        """Fewer than 2 elements: undefined OLS; return safe zero."""
        slope, r2 = ols_slope_r2(np.array([3.14]))
        assert slope == 0.0
        assert r2    == 0.0

    def test_empty_array_returns_zeros(self):
        slope, r2 = ols_slope_r2(np.array([], dtype=float))
        assert slope == 0.0
        assert r2    == 0.0

    def test_r2_is_clamped_to_non_negative(self):
        """Numerical noise cannot push R² below 0."""
        # Nearly-flat noisy series: tiny residuals but slope ≈ 0.
        rng  = np.random.default_rng(0)
        vals = rng.normal(0, 1e-15, 20)   # near-zero noise around 0
        _, r2 = ols_slope_r2(vals)
        assert r2 >= 0.0

    def test_all_nan_returns_zeros(self):
        """
        All-NaN input must return (0.0, 0.0), not raise or return NaN.
        NaN can reach ols_slope_r2 from callers that do not strip NaN
        before calling (e.g. a rolling window with missing data).
        Propagating NaN would silently break boolean comparisons downstream:
          `adx_slope >= 0.0` with NaN → False (no warning, wrong classification).
        """
        slope, r2 = ols_slope_r2(np.array([np.nan, np.nan, np.nan]))
        assert slope == 0.0
        assert r2    == 0.0

    def test_partial_nan_returns_zeros(self):
        """Any NaN in the input must return (0.0, 0.0)."""
        arr = np.array([1.0, 2.0, np.nan, 4.0])
        slope, r2 = ols_slope_r2(arr)
        assert slope == 0.0
        assert r2    == 0.0

    def test_negative_slope(self):
        """Decreasing series → slope < 0."""
        slope, _ = ols_slope_r2(np.array([4.0, 3.0, 2.0, 1.0]))
        assert slope == pytest.approx(-1.0, abs=1e-9)

    def test_r2_between_zero_and_one(self):
        """R² is always in [0, 1] for any non-trivial input."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            vals = rng.standard_normal(30)
            _, r2 = ols_slope_r2(vals)
            assert 0.0 <= r2 <= 1.0, f"R²={r2} out of [0, 1]"


class TestOlsSlope:
    def test_delegates_to_r2(self):
        """ols_slope must equal the slope component of ols_slope_r2."""
        arr   = np.array([2.0, 4.0, 6.0, 8.0])
        slope = ols_slope(arr)
        expected_slope, _ = ols_slope_r2(arr)
        assert slope == pytest.approx(expected_slope, abs=1e-9)
