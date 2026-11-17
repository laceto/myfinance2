from algoshort.yfinance_handler import YFinanceDataHandler
import logging
from pathlib import Path
from datetime import date 
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,              # or DEBUG / WARNING / ERROR
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ticker_list = ['TCEHY']

massive_handler = YFinanceDataHandler(
    cache_dir="../data/ohlc/it",  # Cache directory
    enable_logging=True,
    chunk_size=20,                       # Smaller chunks for stability
    log_level=logging.INFO
)

today = date.today()

data = massive_handler.download_data(
    symbols=ticker_list,  # Example ticker
    start='2016-01-01',  # Start date for historical data
    # end='2026-02-20',    # End date for historical data
    end = today,
    # period='5y',           # 5 years of historical data
    interval='1d',         # Daily data
    use_cache=False,        # Use cache to avoid re-downloading
    threads=True           # Enable multi-threading
)

print(data)