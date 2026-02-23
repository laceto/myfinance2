import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool, cpu_count
from functools import partial

def _process_single_combination(combo, df, direction_col, allow_flips, 
                                require_regime_alignment, verbose):
    """
    Worker function to process a single signal combination.
    Must be defined at module level for pickling compatibility.
    
    Parameters:
    -----------
    combo : dict
        Combination dictionary with 'entry', 'exit', 'name' keys
    df : pd.DataFrame
        Input DataFrame with all signals
    direction_col : str
        Direction filter column
    allow_flips : bool
        Allow position flips
    require_regime_alignment : bool
        Require regime alignment
    verbose : bool
        Print detailed logs
        
    Returns:
    --------
    dict
        Result dictionary with statistics and metadata
    """
    try:
        # Import here to avoid issues with multiprocessing
        from algoshort.combiner import HybridSignalCombiner
        
        # Create a working copy
        df_test = df.copy()
        
        # Initialize combiner
        combiner = HybridSignalCombiner(
            direction_col=direction_col,
            entry_col=combo['entry'],
            exit_col=combo['exit'],
            verbose=verbose
        )
        
        # Combine signals
        output_col = combo['name']
        df_test = combiner.combine_signals(
            df_test,
            output_col=output_col,
            allow_flips=allow_flips,
            require_regime_alignment=require_regime_alignment
        )
        
        # Add metadata
        df_test = combiner.add_signal_metadata(df_test, output_col)
        
        # Get trade summary
        summary = combiner.get_trade_summary(df_test, output_col)
        
        # Return result with combined signal column
        result = {
            'combination_name': combo['name'],
            'entry_signal': combo['entry'],
            'exit_signal': combo['exit'],
            'direction_signal': direction_col,
            'output_column': output_col,
            
            # Trade statistics
            'total_trades': summary['total_entries'],
            'long_trades': summary['entry_long_count'],
            'short_trades': summary['entry_short_count'],
            'long_to_short_flips': summary['flip_long_to_short_count'],
            'short_to_long_flips': summary['flip_short_to_long_count'],
            
            # Position distribution
            'long_bars': summary['long_bars'],
            'short_bars': summary['short_bars'],
            'flat_bars': summary['flat_bars'],
            'long_pct': summary['long_pct'],
            'short_pct': summary['short_pct'],
            'flat_pct': summary['flat_pct'],
            
            # Average holding periods
            'avg_bars_per_long_trade': summary['avg_bars_per_long_trade'],
            'avg_bars_per_short_trade': summary['avg_bars_per_short_trade'],
            
            # Include the combined signal column for later addition to main df
            'combined_signal': df_test[output_col].copy(),
            
            'success': True,
            'error': None
        }
        
        return result
        
    except Exception as e:
        # Return error result
        return {
            'combination_name': combo['name'],
            'entry_signal': combo['entry'],
            'exit_signal': combo['exit'],
            'direction_signal': direction_col,
            'output_column': combo['name'],
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'long_to_short_flips': 0,
            'short_to_long_flips': 0,
            'long_bars': 0,
            'short_bars': 0,
            'flat_bars': 0,
            'long_pct': 0,
            'short_pct': 0,
            'flat_pct': 0,
            'avg_bars_per_long_trade': 0,
            'avg_bars_per_short_trade': 0,
            'combined_signal': None,
            'success': False,
            'error': str(e)
        }

def _process_stock_all_combinations(args):
    """
    Worker function: process all signal combinations serially for a single stock.
    Must be defined at module level for multiprocessing pickling compatibility.

    Parameters
    ----------
    args : tuple
        Packed as (ticker, df, combos, direction_col, allow_flips,
                   require_regime_alignment, verbose)

    Returns
    -------
    tuple[str, list[dict]]
        (ticker, list of result dicts — one per combination, each tagged
        with a 'ticker' key)
    """
    ticker, df, combos, direction_col, allow_flips, require_regime_alignment, verbose = args

    results = []
    for combo in combos:
        result = _process_single_combination(
            combo=combo,
            df=df,
            direction_col=direction_col,
            allow_flips=allow_flips,
            require_regime_alignment=require_regime_alignment,
            verbose=verbose,
        )
        result['ticker'] = ticker
        results.append(result)

    return ticker, results


class SignalGridSearch:
    """
    Grid search over all combinations of entry and exit signals.
    Direction is always 'rrg' (or user-specified).
    """
    
    def __init__(self, df, available_signals, direction_col='rrg'):
        """
        Initialize grid search.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with all signal columns
        available_signals : list
            List of signal column names to use for entry/exit combinations
            Example: ['rbo_50', 'rtt_5010', 'rsma_2050', 'rema_2050']
        direction_col : str, default='rrg'
            Direction filter column (fixed, not included in grid)
        """
        self.df = df.copy()
        self.direction_col = direction_col
        self.available_signals = available_signals
        self.results = None
        
        # Validate that all signals exist in dataframe
        self._validate_signals()
    
    def _validate_signals(self):
        """
        Validate that all specified signals exist in the dataframe.
        
        Raises:
        -------
        ValueError
            If any signal column is missing from dataframe
        """
        missing_signals = []
        
        # Check direction column
        if self.direction_col not in self.df.columns:
            raise ValueError(f"Direction column '{self.direction_col}' not found in dataframe")
        
        # Check available signals
        for sig in self.available_signals:
            if sig not in self.df.columns:
                missing_signals.append(sig)
        
        if missing_signals:
            raise ValueError(
                f"The following signal columns are missing from dataframe: {missing_signals}\n"
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Check if direction column is in available signals (it shouldn't be)
        if self.direction_col in self.available_signals:
            raise ValueError(
                f"Direction column '{self.direction_col}' should not be in available_signals list. "
                f"It is automatically used as the direction filter."
            )
        
        print(f"✓ Validation passed: All {len(self.available_signals)} signals found in dataframe")
    
    def get_available_signals(self):
        """
        Get the list of available signals for entry/exit.
        
        Returns:
        --------
        list
            List of signal column names available for entry/exit
        """
        return self.available_signals
    
    def generate_grid(self):
        """
        Generate all combinations of entry and exit signals.
        
        Creates combinations where:
        - Entry signal: any available signal
        - Exit signal: any available signal (can be same or different)
        
        Returns:
        --------
        list of dict
            Each dict: {'entry': str, 'exit': str, 'name': str}
        """
        signals = self.get_available_signals()
        
        print(f"\n{'='*70}")
        print(f"GRID GENERATION")
        print(f"{'='*70}")
        print(f"Direction column: {self.direction_col}")
        print(f"Available signals for entry/exit: {len(signals)}")
        print(f"Signals: {signals}")
        
        # Generate all combinations (including same signal for entry and exit)
        combinations = []
        for entry_sig in signals:
            for exit_sig in signals:
                combinations.append({
                    'entry': entry_sig,
                    'exit': exit_sig,
                    'name': f'{entry_sig}__{exit_sig}'
                    # 'name': f'entry_{entry_sig}__exit_{exit_sig}'
                })
        
        print(f"\nTotal combinations to test: {len(combinations)}")
        print(f"  = {len(signals)} entry signals × {len(signals)} exit signals")
        
        return combinations
    
    def run_grid_search(self, 
                       allow_flips=True, 
                       require_regime_alignment=True,
                    #    next_day_execution=True,
                       verbose=False):
        """
        Run HybridSignalCombiner on all signal combinations.
        
        Parameters:
        -----------
        allow_flips : bool, default=True
            Allow direct position flips
        require_regime_alignment : bool, default=True
            Require entries to align with direction signal
        next_day_execution : bool, default=True
            Use next-day execution (realistic)
        verbose : bool, default=False
            Print progress for each combination
            
        Returns:
        --------
        pd.DataFrame
            Results with combined signals for each combination
        """
        # from HybridSignalCombiner import HybridSignalCombiner
        
        # Generate grid
        grid = self.generate_grid()
        
        print(f"\n{'='*70}")
        print(f"RUNNING GRID SEARCH")
        print(f"{'='*70}")
        print(f"Direction column: {self.direction_col}")
        print(f"Allow flips: {allow_flips}")
        print(f"Require regime alignment: {require_regime_alignment}")
        # print(f"Next-day execution: {next_day_execution}")
        print(f"\nProcessing {len(grid)} combinations...")
        
        results = []
        
        # Process each combination
        for combo in tqdm(grid, desc="Testing combinations"):
            try:
                # Create a working copy of the dataframe
                df_test = self.df.copy()
                
                # Initialize combiner for this combination
                combiner = HybridSignalCombiner(
                    direction_col=self.direction_col,
                    entry_col=combo['entry'],
                    exit_col=combo['exit'],
                    # next_day_execution=next_day_execution,
                    verbose=verbose
                )
                
                # Combine signals
                output_col = combo['name']
                df_test = combiner.combine_signals(
                    df_test,
                    output_col=output_col,
                    allow_flips=allow_flips,
                    require_regime_alignment=require_regime_alignment
                )
                
                # Add metadata
                df_test = combiner.add_signal_metadata(df_test, output_col)
                
                # Get trade summary
                summary = combiner.get_trade_summary(df_test, output_col)
                
                # Store the combined signal column
                self.df[output_col] = df_test[output_col]
                
                # Store results
                result = {
                    'combination_name': combo['name'],
                    'entry_signal': combo['entry'],
                    'exit_signal': combo['exit'],
                    'direction_signal': self.direction_col,
                    'output_column': output_col,
                    
                    # Trade statistics
                    'total_trades': summary['total_entries'],
                    'long_trades': summary['entry_long_count'],
                    'short_trades': summary['entry_short_count'],
                    'long_to_short_flips': summary['flip_long_to_short_count'],
                    'short_to_long_flips': summary['flip_short_to_long_count'],
                    
                    # Position distribution
                    'long_bars': summary['long_bars'],
                    'short_bars': summary['short_bars'],
                    'flat_bars': summary['flat_bars'],
                    'long_pct': summary['long_pct'],
                    'short_pct': summary['short_pct'],
                    'flat_pct': summary['flat_pct'],
                    
                    # Average holding periods
                    'avg_bars_per_long_trade': summary['avg_bars_per_long_trade'],
                    'avg_bars_per_short_trade': summary['avg_bars_per_short_trade'],
                    
                    'success': True,
                    'error': None
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"\nError with {combo['name']}: {str(e)}")
                results.append({
                    'combination_name': combo['name'],
                    'entry_signal': combo['entry'],
                    'exit_signal': combo['exit'],
                    'direction_signal': self.direction_col,
                    'output_column': combo['name'],
                    'total_trades': 0,
                    'long_trades': 0,
                    'short_trades': 0,
                    'long_to_short_flips': 0,
                    'short_to_long_flips': 0,
                    'long_bars': 0,
                    'short_bars': 0,
                    'flat_bars': 0,
                    'long_pct': 0,
                    'short_pct': 0,
                    'flat_pct': 0,
                    'avg_bars_per_long_trade': 0,
                    'avg_bars_per_short_trade': 0,
                    'success': False,
                    'error': str(e)
                })
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        
        print(f"\n{'='*70}")
        print(f"GRID SEARCH COMPLETE")
        print(f"{'='*70}")
        print(f"Successful combinations: {self.results['success'].sum()}")
        print(f"Failed combinations: {(~self.results['success']).sum()}")
        
        return self.results
    
    def get_results_summary(self):
        """
        Get summary statistics of the grid search results.
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        if self.results is None:
            raise ValueError("Run grid_search first using run_grid_search()")
        
        # Filter successful runs
        successful = self.results[self.results['success'] == True]
        
        if len(successful) == 0:
            return pd.Series({'error': 'No successful combinations'})
        
        summary = {
            'total_combinations': len(self.results),
            'successful_combinations': len(successful),
            'failed_combinations': len(self.results) - len(successful),
            
            'avg_total_trades': successful['total_trades'].mean(),
            'min_total_trades': successful['total_trades'].min(),
            'max_total_trades': successful['total_trades'].max(),
            
            'avg_long_pct': successful['long_pct'].mean(),
            'avg_short_pct': successful['short_pct'].mean(),
            'avg_flat_pct': successful['flat_pct'].mean(),
            
            'most_active_combo': successful.loc[successful['total_trades'].idxmax(), 'combination_name'],
            'least_active_combo': successful.loc[successful['total_trades'].idxmin(), 'combination_name'],
        }
        
        return pd.Series(summary)
    
    def filter_combinations(self, min_trades=10, max_flat_pct=80):
        """
        Filter combinations based on criteria.
        
        Parameters:
        -----------
        min_trades : int, default=10
            Minimum number of total trades
        max_flat_pct : float, default=80
            Maximum percentage of time in flat position
            
        Returns:
        --------
        pd.DataFrame
            Filtered results
        """
        if self.results is None:
            raise ValueError("Run grid_search first")
        
        filtered = self.results[
            (self.results['success'] == True) &
            (self.results['total_trades'] >= min_trades) &
            (self.results['flat_pct'] <= max_flat_pct)
        ]
        
        print(f"Filtered combinations: {len(filtered)} out of {len(self.results)}")
        return filtered
    
    def export_results(self, filename='grid_search_results.csv'):
        """Export results to CSV."""
        if self.results is None:
            raise ValueError("Run grid_search first")
        
        self.results.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
    
    def export_dataframe(self, filename='df_with_all_signals.csv'):
        """Export dataframe with all combined signal columns."""
        self.df.to_csv(filename, index=False)
        print(f"DataFrame with all signals exported to {filename}")
    
    def get_signal_columns(self):
        """
        Get list of all generated signal column names.
        
        Returns:
        --------
        list
            List of signal column names created by grid search
        """
        if self.results is None:
            raise ValueError("Run grid_search first")
        
        return self.results[self.results['success'] == True]['output_column'].tolist()

    def run_grid_search_parallel(self,
                                allow_flips=True,
                                require_regime_alignment=True,
                                verbose=False,
                                n_jobs=-1,
                                backend='multiprocessing',
                                batch_size=None):
        """
        Run HybridSignalCombiner on all signal combinations in parallel.
        
        This method parallelizes the grid search across multiple CPU cores,
        significantly reducing computation time for large parameter spaces.
        Each signal combination is processed independently, making the problem
        embarrassingly parallel.
        
        Parameters:
        -----------
        allow_flips : bool, default=True
            Allow direct position flips (long to short or vice versa)
        require_regime_alignment : bool, default=True
            Require entries to align with direction signal
        verbose : bool, default=False
            Print progress for each combination (not recommended with parallel execution)
        n_jobs : int, default=-1
            Number of parallel jobs to run:
            - -1: Use all available CPU cores
            - 1: Sequential execution (same as original run_grid_search)
            - n: Use n CPU cores
        backend : str, default='multiprocessing'
            Parallelization backend:
            - 'multiprocessing': Use Python's multiprocessing.Pool
            - 'joblib': Use joblib.Parallel (better error handling, requires joblib)
        batch_size : int, default=None
            Number of combinations to process per batch (helps with memory management)
            - None: Automatically determined
            - n: Process n combinations at a time
            
        Returns:
        --------
        pd.DataFrame
            Results with combined signals for each combination
            
        Performance Notes:
        ------------------
        - Expected speedup: ~(n_cores - 1)x on CPU-bound tasks
        - Memory usage: Scales with n_jobs (each worker holds a copy of df)
        - For large DataFrames, consider reducing n_jobs or using batch_size
        - Progress tracking is approximate with multiprocessing
        
        Examples:
        ---------
        # Use all cores
        results = searcher.run_grid_search_parallel(n_jobs=-1)
        
        # Use 4 cores
        results = searcher.run_grid_search_parallel(n_jobs=4)
        
        # Use joblib backend with batching
        results = searcher.run_grid_search_parallel(
            n_jobs=-1,
            backend='joblib',
            batch_size=10
        )
        """
        # Generate grid
        grid = self.generate_grid()
        
        # Determine number of jobs
        if n_jobs == -1:
            n_jobs = cpu_count()
        elif n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or positive integer, got {n_jobs}")
        
        # If n_jobs=1, fall back to sequential
        if n_jobs == 1:
            print("n_jobs=1: Using sequential execution")
            return self.run_grid_search(
                allow_flips=allow_flips,
                require_regime_alignment=require_regime_alignment,
                verbose=verbose
            )
        
        # Print header
        print(f"\n{'='*70}")
        print(f"RUNNING PARALLEL GRID SEARCH")
        print(f"{'='*70}")
        print(f"Direction column: {self.direction_col}")
        print(f"Allow flips: {allow_flips}")
        print(f"Require regime alignment: {require_regime_alignment}")
        print(f"Backend: {backend}")
        print(f"Parallel jobs: {n_jobs} cores")
        print(f"\nProcessing {len(grid)} combinations in parallel...")
        
        # Execute based on backend
        if backend == 'multiprocessing':
            results = self._run_multiprocessing(
                grid, allow_flips, require_regime_alignment, 
                verbose, n_jobs, batch_size
            )
        elif backend == 'joblib':
            results = self._run_joblib(
                grid, allow_flips, require_regime_alignment,
                verbose, n_jobs, batch_size
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'multiprocessing' or 'joblib'")
        
        # Add combined signals to main dataframe
        print("\nAdding combined signal columns to main dataframe...")
        for result in results:
            if result['success'] and result['combined_signal'] is not None:
                self.df[result['output_column']] = result['combined_signal']
        
        # Remove combined_signal from results (no longer needed)
        for result in results:
            if 'combined_signal' in result:
                del result['combined_signal']
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"GRID SEARCH COMPLETE")
        print(f"{'='*70}")
        print(f"Successful combinations: {self.results['success'].sum()}")
        print(f"Failed combinations: {(~self.results['success']).sum()}")
        
        return self.results


    def _run_multiprocessing(self, grid, allow_flips, require_regime_alignment,
                            verbose, n_jobs, batch_size):
        """
        Execute grid search using multiprocessing.Pool.
        
        Parameters:
        -----------
        grid : list
            List of combination dictionaries
        allow_flips : bool
            Allow position flips
        require_regime_alignment : bool
            Require regime alignment
        verbose : bool
            Print detailed logs
        n_jobs : int
            Number of parallel jobs
        batch_size : int or None
            Batch size for processing
            
        Returns:
        --------
        list
            List of result dictionaries
        """
        # Create partial function with fixed parameters
        worker_func = partial(
            _process_single_combination,
            df=self.df,
            direction_col=self.direction_col,
            allow_flips=allow_flips,
            require_regime_alignment=require_regime_alignment,
            verbose=verbose
        )
        
        # Determine batch size
        if batch_size is None:
            batch_size = max(1, len(grid) // n_jobs)
        
        # Process with multiprocessing
        with Pool(processes=n_jobs) as pool:
            # Use imap for progress tracking
            results = list(tqdm(
                pool.imap(worker_func, grid, chunksize=batch_size),
                total=len(grid),
                desc="Testing combinations"
            ))
        
        return results


    def _run_joblib(self, grid, allow_flips, require_regime_alignment,
                    verbose, n_jobs, batch_size):
        """
        Execute grid search using joblib.Parallel.
        
        Requires: pip install joblib
        
        Parameters:
        -----------
        grid : list
            List of combination dictionaries
        allow_flips : bool
            Allow position flips
        require_regime_alignment : bool
            Require regime alignment
        verbose : bool
            Print detailed logs
        n_jobs : int
            Number of parallel jobs
        batch_size : int or None
            Batch size for processing
            
        Returns:
        --------
        list
            List of result dictionaries
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            raise ImportError(
                "joblib backend requires joblib. Install with: pip install joblib"
            )
        
        # Determine batch size
        if batch_size is None:
            batch_size = max(1, len(grid) // n_jobs)
        
        # Process with joblib
        results = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=10)(
            delayed(_process_single_combination)(
                combo=combo,
                df=self.df,
                direction_col=self.direction_col,
                allow_flips=allow_flips,
                require_regime_alignment=require_regime_alignment,
                verbose=verbose
            )
            for combo in tqdm(grid, desc="Testing combinations")
        )
        
        return results





    def run_grid_search_parallel_over_stocks(
        self,
        stock_dfs: dict,
        allow_flips: bool = True,
        require_regime_alignment: bool = True,
        verbose: bool = False,
        n_jobs: int = -1,
        add_signals_to_dfs: bool = False,
    ) -> dict:
        """
        Run HybridSignalCombiner on all signal combinations for each stock in parallel.

        Inverts the parallelism axis of run_grid_search_parallel: workers are
        distributed over stocks rather than signal combinations. Each worker
        receives only one stock's DataFrame, giving dramatically lower per-worker
        memory usage compared to sending the full multi-stock DataFrame to every
        process.

        Parameters
        ----------
        stock_dfs : dict[str, pd.DataFrame]
            Mapping of {ticker: ohlc_DataFrame}. Each DataFrame must contain the
            columns expected by HybridSignalCombiner (direction_col and all
            entry/exit signal columns generated by generate_grid).
        allow_flips : bool, default=True
            Allow direct position flips (long-to-short or vice versa).
        require_regime_alignment : bool, default=True
            Require entries to align with the direction signal.
        verbose : bool, default=False
            Print per-combination progress inside each worker. Not recommended
            with many parallel processes.
        n_jobs : int, default=-1
            Number of parallel worker processes.
            -1 uses all available CPU cores; automatically capped at len(stock_dfs).
            1 runs sequentially without spawning a Pool (useful for testing and
            single-core environments).
        add_signals_to_dfs : bool, default=False
            If True, write the computed combined-signal columns back into the
            DataFrames in stock_dfs (modifies the dict values in-place).

        Returns
        -------
        dict[str, pd.DataFrame]
            {ticker: results_df} where each results_df mirrors the schema returned
            by run_grid_search (one row per signal combination) plus a 'ticker'
            column.

        Raises
        ------
        ValueError
            If stock_dfs is empty, the signal grid is empty, or n_jobs is invalid.

        Performance Notes
        -----------------
        - Per-worker memory: O(one_stock_df) vs O(full_multi_stock_df) in the
          signal-parallel variant --- typically 100-1000x smaller per worker.
        - Best when: many stocks, moderate number of signal combinations.
        - If you have very few stocks (<= n_cores) but thousands of combinations,
          prefer run_grid_search_parallel instead.

        Examples
        --------
        stock_data = {'ENI.MI': df_eni, 'ENEL.MI': df_enel, 'ISP.MI': df_isp}
        per_stock_results = searcher.run_grid_search_parallel_over_stocks(
            stock_dfs=stock_data,
            n_jobs=4,
            add_signals_to_dfs=True,
        )
        results_eni = per_stock_results['ENI.MI']
        """
        if not stock_dfs:
            raise ValueError("stock_dfs is empty -- provide at least one stock DataFrame.")

        grid = self.generate_grid()
        if not grid:
            raise ValueError("Signal grid is empty -- no combinations to evaluate.")

        if n_jobs == -1:
            n_jobs = cpu_count()
        elif n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or a positive integer, got {n_jobs}.")

        n_stocks = len(stock_dfs)
        n_combos = len(grid)
        # Cap workers at stock count -- no benefit spawning more workers than tasks
        effective_jobs = min(n_jobs, n_stocks)

        print(f"\n{'='*70}")
        print(f"RUNNING GRID SEARCH -- PARALLEL OVER {n_stocks} STOCKS")
        print(f"{'='*70}")
        print(f"Direction column:         {self.direction_col}")
        print(f"Signal combinations:      {n_combos}")
        print(f"Allow flips:              {allow_flips}")
        print(f"Require regime alignment: {require_regime_alignment}")
        print(f"Parallel jobs:            {effective_jobs} / {cpu_count()} cores")
        print(
            f"\nProcessing {n_stocks} stocks x {n_combos} combinations "
            f"({n_stocks * n_combos} total tasks)..."
        )

        stock_results: dict = {}

        def _store(ticker, results):
            if add_signals_to_dfs:
                for result in results:
                    if result['success'] and result['combined_signal'] is not None:
                        stock_dfs[ticker][result['output_column']] = result['combined_signal']
            for result in results:
                result.pop('combined_signal', None)
            stock_results[ticker] = pd.DataFrame(results)

        if effective_jobs == 1:
            # Sequential path -- avoids Pool overhead for single-stock/single-core runs
            # and works in environments where multiprocessing is restricted (e.g. pytest).
            for ticker, df in tqdm(stock_dfs.items(), desc="Stocks processed", total=n_stocks):
                _, results = _process_stock_all_combinations(
                    (ticker, df, grid, self.direction_col, allow_flips,
                     require_regime_alignment, verbose)
                )
                _store(ticker, results)
        else:
            # Pack into single-arg tuples required for pool.imap pickling
            worker_args = [
                (ticker, df, grid, self.direction_col, allow_flips,
                 require_regime_alignment, verbose)
                for ticker, df in stock_dfs.items()
            ]
            with Pool(processes=effective_jobs) as pool:
                for ticker, results in tqdm(
                    pool.imap_unordered(_process_stock_all_combinations, worker_args),
                    total=n_stocks,
                    desc="Stocks processed",
                ):
                    _store(ticker, results)

        total_combos = n_stocks * n_combos
        total_success = sum(df_r['success'].sum() for df_r in stock_results.values())

        print(f"\n{'='*70}")
        print(f"GRID SEARCH COMPLETE")
        print(f"{'='*70}")
        print(f"Stocks processed:       {n_stocks}")
        print(f"Combinations per stock: {n_combos}")
        print(f"Successful:             {total_success} / {total_combos}")
        print(f"Failed:                 {total_combos - total_success} / {total_combos}")

        return stock_results


class HybridSignalCombiner:
    """
    Combines multiple regime signals using a hybrid, non-mutually exclusive approach.
    Fully supports both LONG and SHORT positions with verified logic.
    """
    
    def __init__(self, direction_col='floor_ceiling', 
                 entry_col='range_breakout', 
                 exit_col='ma_crossover',
                 verbose=False):
        """
        Initialize the hybrid signal combiner.
        
        Parameters:
        -----------
        direction_col : str
            Column name for direction/regime filter (determines long/short bias)
            Values: 1 (bullish), 0 (neutral), -1 (bearish)
        entry_col : str
            Column name for entry timing signal (triggers position entry)
            Values: 1 (long entry), 0 (no entry), -1 (short entry)
        exit_col : str
            Column name for exit timing signal (triggers position exit)
            Values: 1 (bullish/exit short), 0 (no exit), -1 (bearish/exit long)
        verbose : bool, default=False
            If True, prints detailed trade logic
        """
        self.direction_col = direction_col
        self.entry_col = entry_col
        self.exit_col = exit_col
        self.verbose = verbose
    
    def combine_signals(self, df, output_col='hybrid_signal', 
                       allow_flips=True, require_regime_alignment=True):
        """
        Combine three signals using hybrid logic with full long/short support.
        
        Logic Flow:
        -----------
        FLAT (position = 0):
          - Enter LONG (→1): entry=1 AND (direction=1 OR direction=0 if not strict)
          - Enter SHORT (→-1): entry=-1 AND (direction=-1 OR direction=0 if not strict)
        
        LONG (position = 1):
          - Exit to FLAT (→0): exit=-1 OR direction=-1
          - Flip to SHORT (→-1): allow_flips=True AND entry=-1 AND direction=-1
        
        SHORT (position = -1):
          - Exit to FLAT (→0): exit=1 OR direction=1
          - Flip to LONG (→1): allow_flips=True AND entry=1 AND direction=1
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with signal columns (values must be -1, 0, or 1)
        output_col : str, default='hybrid_signal'
            Name for the combined output signal column
        allow_flips : bool, default=True
            If True, allows flipping from long to short (or vice versa) directly
            If False, must exit to flat before entering opposite direction
        require_regime_alignment : bool, default=True
            If True, entries must align with direction signal (strict filtering)
            If False, can enter in neutral regime (direction=0)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added hybrid_signal column
        """
        df[output_col] = 0
        current_position = 0
        
        for i in range(1, len(df)):
            direction = df.at[i, self.direction_col]
            entry = df.at[i, self.entry_col]
            exit_signal = df.at[i, self.exit_col]
            
            prev_position = current_position
            
            # ========================================
            # STATE: FLAT (looking for entries)
            # ========================================
            if current_position == 0:
                
                # LONG ENTRY LOGIC
                if entry == 1:
                    if require_regime_alignment:
                        # Strict: Only enter long if direction is bullish
                        if direction == 1:
                            current_position = 1
                            if self.verbose:
                                print(f"Bar {i}: ENTER LONG (entry=1, direction=1)")
                    else:
                        # Loose: Enter long unless direction is explicitly bearish
                        if direction != -1:
                            current_position = 1
                            if self.verbose:
                                print(f"Bar {i}: ENTER LONG (entry=1, direction={direction})")
                
                # SHORT ENTRY LOGIC
                elif entry == -1:
                    if require_regime_alignment:
                        # Strict: Only enter short if direction is bearish
                        if direction == -1:
                            current_position = -1
                            if self.verbose:
                                print(f"Bar {i}: ENTER SHORT (entry=-1, direction=-1)")
                    else:
                        # Loose: Enter short unless direction is explicitly bullish
                        if direction != 1:
                            current_position = -1
                            if self.verbose:
                                print(f"Bar {i}: ENTER SHORT (entry=-1, direction={direction})")
            
            # ========================================
            # STATE: LONG (managing long position)
            # ========================================
            elif current_position == 1:
                
                # EXIT LONG LOGIC
                # Exit on: (1) Bearish exit signal, OR (2) Bearish regime
                if exit_signal == -1:
                    current_position = 0
                    if self.verbose:
                        print(f"Bar {i}: EXIT LONG on exit signal (exit=-1)")
                
                elif direction == -1:
                    # Bearish regime: exit or flip
                    if allow_flips and entry == -1:
                        # FLIP: Long → Short
                        current_position = -1
                        if self.verbose:
                            print(f"Bar {i}: FLIP LONG→SHORT (direction=-1, entry=-1)")
                    else:
                        # EXIT: Just close long position
                        current_position = 0
                        if self.verbose:
                            print(f"Bar {i}: EXIT LONG on regime change (direction=-1)")
                
                # FLIP LONG → SHORT (even if regime is neutral/bullish)
                elif allow_flips and entry == -1 and not require_regime_alignment:
                    current_position = -1
                    if self.verbose:
                        print(f"Bar {i}: FLIP LONG→SHORT on entry signal (entry=-1)")
            
            # ========================================
            # STATE: SHORT (managing short position)
            # ========================================
            elif current_position == -1:
                
                # EXIT SHORT LOGIC
                # Exit on: (1) Bullish exit signal, OR (2) Bullish regime
                if exit_signal == 1:
                    current_position = 0
                    if self.verbose:
                        print(f"Bar {i}: EXIT SHORT on exit signal (exit=1)")
                
                elif direction == 1:
                    # Bullish regime: exit or flip
                    if allow_flips and entry == 1:
                        # FLIP: Short → Long
                        current_position = 1
                        if self.verbose:
                            print(f"Bar {i}: FLIP SHORT→LONG (direction=1, entry=1)")
                    else:
                        # EXIT: Just close short position
                        current_position = 0
                        if self.verbose:
                            print(f"Bar {i}: EXIT SHORT on regime change (direction=1)")
                
                # FLIP SHORT → LONG (even if regime is neutral/bearish)
                elif allow_flips and entry == 1 and not require_regime_alignment:
                    current_position = 1
                    if self.verbose:
                        print(f"Bar {i}: FLIP SHORT→LONG on entry signal (entry=1)")
            
            df.at[i, output_col] = current_position
        
        return df
    
    def validate_signals(self, df):
        """
        Validate that signal columns have correct values.
        
        Returns:
        --------
        dict
            Validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        for col in [self.direction_col, self.entry_col, self.exit_col]:
            if col not in df.columns:
                validation['valid'] = False
                validation['errors'].append(f"Missing column: {col}")
                continue
            
            unique_vals = df[col].unique()
            invalid_vals = [v for v in unique_vals if v not in [-1, 0, 1]]
            
            if invalid_vals:
                validation['valid'] = False
                validation['errors'].append(
                    f"Column '{col}' has invalid values: {invalid_vals}. Must be -1, 0, or 1."
                )
            
            # Check for NaN values
            if df[col].isna().any():
                validation['warnings'].append(f"Column '{col}' contains NaN values")
        
        return validation
    
    def add_signal_metadata(self, df, output_col='hybrid_signal'):
        """
        Add detailed metadata columns showing trade logic.
        
        Adds:
        -----
        - position_changed: Boolean, did position change?
        - trade_type: 'entry_long', 'entry_short', 'exit_long', 'exit_short', 
                      'flip_long_to_short', 'flip_short_to_long', 'hold', or 'flat'
        - bars_in_position: Running count of bars in current position
        - position_direction: 'long', 'short', or 'flat'
        """
        df['position_changed'] = df[output_col] != df[output_col].shift(1)
        df['trade_type'] = 'hold'
        df['bars_in_position'] = 0
        df['position_direction'] = 'flat'
        
        bars_count = 0
        
        for i in range(1, len(df)):
            prev_pos = df.at[i-1, output_col]
            curr_pos = df.at[i, output_col]
            
            # Determine trade type
            if prev_pos == 0 and curr_pos == 1:
                df.at[i, 'trade_type'] = 'entry_long'
                bars_count = 0
            elif prev_pos == 0 and curr_pos == -1:
                df.at[i, 'trade_type'] = 'entry_short'
                bars_count = 0
            elif prev_pos == 1 and curr_pos == 0:
                df.at[i, 'trade_type'] = 'exit_long'
                bars_count = 0
            elif prev_pos == -1 and curr_pos == 0:
                df.at[i, 'trade_type'] = 'exit_short'
                bars_count = 0
            elif prev_pos == 1 and curr_pos == -1:
                df.at[i, 'trade_type'] = 'flip_long_to_short'
                bars_count = 0
            elif prev_pos == -1 and curr_pos == 1:
                df.at[i, 'trade_type'] = 'flip_short_to_long'
                bars_count = 0
            elif curr_pos == 0:
                df.at[i, 'trade_type'] = 'flat'
                bars_count = 0
            else:
                df.at[i, 'trade_type'] = 'hold'
                bars_count += 1
            
            # Position direction
            if curr_pos == 1:
                df.at[i, 'position_direction'] = 'long'
            elif curr_pos == -1:
                df.at[i, 'position_direction'] = 'short'
            else:
                df.at[i, 'position_direction'] = 'flat'
            
            df.at[i, 'bars_in_position'] = bars_count
        
        return df
    
    def get_trade_summary(self, df, output_col='hybrid_signal'):
        """
        Generate comprehensive trade statistics for both long and short.
        """
        df = self.add_signal_metadata(df, output_col)
        
        total_bars = len(df)
        long_bars = (df[output_col] == 1).sum()
        short_bars = (df[output_col] == -1).sum()
        flat_bars = (df[output_col] == 0).sum()
        
        # Count trade types
        trade_counts = df['trade_type'].value_counts().to_dict()
        
        summary = {
            # Position distribution
            'total_bars': total_bars,
            'long_bars': long_bars,
            'short_bars': short_bars,
            'flat_bars': flat_bars,
            'long_pct': (long_bars / total_bars * 100) if total_bars > 0 else 0,
            'short_pct': (short_bars / total_bars * 100) if total_bars > 0 else 0,
            'flat_pct': (flat_bars / total_bars * 100) if total_bars > 0 else 0,
            
            # Trade counts by type
            'entry_long_count': trade_counts.get('entry_long', 0),
            'entry_short_count': trade_counts.get('entry_short', 0),
            'exit_long_count': trade_counts.get('exit_long', 0),
            'exit_short_count': trade_counts.get('exit_short', 0),
            'flip_long_to_short_count': trade_counts.get('flip_long_to_short', 0),
            'flip_short_to_long_count': trade_counts.get('flip_short_to_long', 0),
            
            # Aggregate metrics
            'total_entries': (trade_counts.get('entry_long', 0) + 
                            trade_counts.get('entry_short', 0) +
                            trade_counts.get('flip_long_to_short', 0) +
                            trade_counts.get('flip_short_to_long', 0)),
            'total_exits': (trade_counts.get('exit_long', 0) + 
                          trade_counts.get('exit_short', 0)),
            
            # Average holding period
            'avg_bars_per_long_trade': (long_bars / trade_counts.get('entry_long', 1)) 
                                       if trade_counts.get('entry_long', 0) > 0 else 0,
            'avg_bars_per_short_trade': (short_bars / trade_counts.get('entry_short', 1))
                                        if trade_counts.get('entry_short', 0) > 0 else 0,
        }
        
        return summary


