from algoshort.yfinance_handler import YFinanceDataHandler
import logging
import pandas as pd
from pathlib import Path
import time
from datetime import date

from etf_top15 import load_tickers
from etf_universe import ACTIVE_SEED_FILE

logging.basicConfig(
    level=logging.WARNING,              # or DEBUG / WARNING / ERROR
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Daily universe == cold-start universe: both read the active seed
# (etf_universe.ACTIVE_SEED_FILE), so the appended daily bars match the
# backfilled history.
ticker_list = load_tickers(ACTIVE_SEED_FILE)

massive_handler = YFinanceDataHandler(
    cache_dir="data/ohlc/today/etf",  # Cache directory
    enable_logging=True,
    chunk_size=20,                       # Smaller chunks for stability
    log_level=logging.INFO
)

# Get today's date
today = date.today()
tomorrow = today + pd.Timedelta(days=1)

data = massive_handler.download_data(
    symbols=ticker_list,
    use_cache=False,        # Use cache to avoid re-downloading
    threads=True,           # Enable multi-threading,
    start=today,
    end=tomorrow
)

output_dir = Path("./data/ohlc/today/etf")

# Save ALL downloaded data as parquet (best for large datasets)
massive_handler.save_data(
    filepath=str(output_dir / "ohlc_data.parquet"),
    format='parquet',
    multi_symbol_strategy='single_file',
    combine_column=['open', 'high', 'low', 'close', 'volume']
)
