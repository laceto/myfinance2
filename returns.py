"""
Returns calculation module for trading strategies.

This module provides the ReturnsCalculator class for calculating various
return metrics including daily changes, cumulative P&L, simple returns,
log returns, and cumulative returns from trading signals.

Classes:
    ReturnsCalculator: Main class for calculating returns from OHLC data and signals.

Typical usage:
    calc = ReturnsCalculator(ohlc_df)
    result = calc.get_returns(df, 'my_signal')
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from joblib import Parallel, delayed
import logging

# Module-level logger (best practice)
logger = logging.getLogger(__name__)


class ReturnsCalculator:
    """
    Calculates returns and equity curves for trading strategies based on OHLC data and regime signals.

    Supports both absolute and relative OHLC columns, controlled by the `relative` flag.

    Args:
        ohlc_stock: DataFrame containing OHLC and signal columns
        open_col: Base name of the open column (default: 'open')
        high_col: Base name of the high column (default: 'high')
        low_col: Base name of the low column (default: 'low')
        close_col: Base name of the close column (default: 'close')
        relative_prefix: Prefix used for relative columns (default: 'r')
        logger: Optional logger instance (defaults to module logger)

    Raises:
        KeyError: If any required OHLC columns are missing
    """

    def __init__(
            self,
            ohlc_stock: pd.DataFrame,
            open_col: str = "open",
            high_col: str = "high",
            low_col: str = "low",
            close_col: str = "close",
            relative_prefix: str = "r",
            logger: Optional[logging.Logger] = None,
        ):
        self.ohlc_stock = ohlc_stock

        self._base_cols = (open_col, high_col, low_col, close_col)
        self._relative_prefix = relative_prefix

        # Set logger: prefer injected one, fall back to module logger
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Pre-build and validate column mappings
        self._col_mappings: Dict[bool, Tuple[str, str, str, str]] = {
            False: self._base_cols,
            True: tuple(f"{relative_prefix}{base}" for base in self._base_cols),
        }

        # Fail-fast column validation
        for relative, cols in self._col_mappings.items():
            missing = [c for c in cols if c not in ohlc_stock.columns]
            if missing:
                self.logger.error(
                    "Missing OHLC columns for relative=%s: %s. Available: %s",
                    relative, missing, list(ohlc_stock.columns)
                )
                raise KeyError(
                    f"Missing OHLC columns for relative={relative}: {missing}\n"
                    f"Available: {list(ohlc_stock.columns)}\n"
                    f"Expected base: {self._base_cols}"
                )

        self.logger.debug("ReturnsCalculator initialized – %d rows", len(ohlc_stock))

    def _get_ohlc_columns(self, relative: bool = False) -> Tuple[str, str, str, str]:
        """Return pre-validated OHLC column names based on relative flag."""
        return self._col_mappings[relative]

    def get_returns(
        self,
        df: pd.DataFrame,
        signal: str,
        relative: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate returns, cumulative PL, log returns, etc. for a single signal.

        Args:
            df: DataFrame with OHLC and signal column.
            signal: Name of the signal column (e.g., 'bo_5', 'sma_200').
            relative: If True, use relative-prefixed OHLC columns.
            inplace: If True, modify df in place; else return a new copy.

        Returns:
            DataFrame with additional columns:
                - '<signal>_chg1D': Daily price change based on lagged signal.
                - '<signal>_chg1D_fx': Duplicate of daily price change (legacy).
                - '<signal>_PL_cum': Cumulative sum of daily price changes.
                - '<signal>_PL_cum_fx': Duplicate of cumulative changes (legacy).
                - '<signal>_returns': Daily percentage returns.
                - '<signal>_log_returns': Daily log returns.
                - '<signal>_cumul': Cumulative returns from log returns.

        Raises:
            ValueError: If DataFrame is empty or has fewer than 2 rows.
            KeyError: If signal column is missing.
            ValueError: If close or signal columns are not numeric.
        """
        # Validate empty DataFrame
        if df.empty:
            self.logger.error("get_returns called with empty DataFrame")
            raise ValueError("Input DataFrame is empty.")

        # Validate minimum rows for meaningful calculation
        if len(df) < 2:
            self.logger.error(
                "DataFrame has %d rows, need at least 2 for returns calculation",
                len(df)
            )
            raise ValueError(
                f"DataFrame must have at least 2 rows for returns calculation, "
                f"got {len(df)}"
            )

        _o, _h, _l, _c = self._get_ohlc_columns(relative=relative)

        if signal not in df.columns:
            self.logger.error(
                "Signal column '%s' not found. Available: %s",
                signal, list(df.columns)
            )
            raise KeyError(f"Signal column '{signal}' not found in DataFrame.")

        required_cols = [_c, signal]
        if not all(np.issubdtype(df[col].dtype, np.number) for col in required_cols):
            self.logger.error(
                "Non-numeric data in required columns: close=%s, signal=%s",
                df[_c].dtype, df[signal].dtype
            )
            raise ValueError("Close and signal columns must be numeric.")

        result_df = df if inplace else df.copy()

        signal_filled = result_df[signal].fillna(0)

        close_prices = result_df[_c]
        price_diff = close_prices.diff()
        lagged_signal = signal_filled.shift()

        chg1D = price_diff * lagged_signal
        pct_returns = close_prices.pct_change() * lagged_signal

        # Log returns calculation with edge case handling
        # Clip extreme returns to prevent log1p(-1) = -inf
        pct_change_values = close_prices.pct_change()
        pct_change_clipped = pct_change_values.clip(lower=-0.9999)
        log_returns = np.log1p(pct_change_clipped) * lagged_signal

        new_columns = {
            f"{signal}_chg1D": chg1D,
            f"{signal}_chg1D_fx": chg1D,           # legacy
            f"{signal}_PL_cum": chg1D.cumsum(),
            f"{signal}_PL_cum_fx": chg1D.cumsum(),
            f"{signal}_returns": pct_returns,
            f"{signal}_log_returns": log_returns,
            f"{signal}_cumul": np.exp(log_returns.cumsum()) - 1,
        }

        if inplace:
            result_df[signal] = signal_filled
            # Note: DataFrame.assign() does NOT have an inplace parameter
            # Use direct column assignment instead
            for col_name, col_data in new_columns.items():
                result_df[col_name] = col_data
        else:
            result_df = result_df.assign(**{signal: signal_filled, **new_columns})

        self.logger.debug("Computed returns for signal '%s' (rows: %d)", signal, len(result_df))

        return result_df

    def get_returns_multiple(
        self,
        df: pd.DataFrame,
        signals: List[str],
        relative: bool = False,
        n_jobs: int = -1,
        verbose: bool = True,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Compute returns for multiple signals in parallel using joblib.

        Args:
            df: DataFrame with OHLC and signal columns.
            signals: List of signal column names to process.
            relative: If True, use relative-prefixed OHLC columns.
            n_jobs: Number of parallel jobs (-1 = all cores).
            verbose: If True and logger level allows, show progress.
            inplace: If True, modify df in place.

        Returns:
            DataFrame with original columns plus all new signal columns.

        Raises:
            KeyError: If any signal columns are missing.
        """
        if not signals:
            self.logger.info("No signals provided — returning input DataFrame")
            return df if inplace else df.copy()

        missing_signals = [s for s in signals if s not in df.columns]
        if missing_signals:
            self.logger.error(
                "Missing signal columns: %s. Available: %s",
                missing_signals, list(df.columns)
            )
            raise KeyError(f"Missing signal columns: {missing_signals}")

        def _compute_one_signal(sig: str) -> Dict[str, pd.Series]:
            close_col = self._get_ohlc_columns(relative)[3]
            # Minimal slice — helps memory in parallel workers
            working_df = df[[close_col, sig]].copy()

            result = self.get_returns(
                df=working_df,
                signal=sig,
                relative=relative,
                inplace=False,
            )

            prefix = f"{sig}_"
            new_cols = {c: result[c] for c in result.columns if c.startswith(prefix)}
            return new_cols

        # Parallel execution
        parallel = Parallel(n_jobs=n_jobs, verbose=0)

        if verbose and self.logger.isEnabledFor(logging.INFO):
            self.logger.info("Computing returns for %d signals (parallel, n_jobs=%s)",
                            len(signals), n_jobs if n_jobs > 0 else "all cores")

            results = []
            total = len(signals)
            for i, res in enumerate(parallel(delayed(_compute_one_signal)(sig) for sig in signals), 1):
                results.append(res)
                if i % max(1, total // 20) == 0 or i == total:
                    self.logger.info("Progress: %d/%d signals processed", i, total)
        else:
            results = parallel(delayed(_compute_one_signal)(sig) for sig in signals)

        # Merge results
        all_new_columns = {}
        for res in results:
            all_new_columns.update(res)

        if inplace:
            # Note: DataFrame.assign() does NOT have an inplace parameter
            for col_name, col_data in all_new_columns.items():
                df[col_name] = col_data
            result_df = df
        else:
            result_df = df.assign(**all_new_columns)

        self.logger.info("Finished computing %d signals → %d new columns added",
                         len(signals), len(all_new_columns))

        return result_df
