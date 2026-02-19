from algoshort.yfinance_handler import YFinanceDataHandler
import logging
import pandas as pd
from pathlib import Path
import time

logging.basicConfig(
    level=logging.WARNING,              # or DEBUG / WARNING / ERROR
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

tickers = pd.read_excel("data/ticker/it/ticker.xlsx")

# from tickers column 'ticker' to a list
ticker_list = tickers['ticker'].tolist()
ticker_list = ticker_list[:10]  # Limit to first 100 tickers for testing
# ticker_list = ['AVIO.MI', 'AZM.MI', 'BAMI.MI', 'BMED.MI']  # Example tickers

massive_handler = YFinanceDataHandler(
    cache_dir="../data/ohlc/historical/it",  # Cache directory
    enable_logging=True,
    chunk_size=20,                       # Smaller chunks for stability
    log_level=logging.INFO
)


data = massive_handler.download_data(
    symbols=ticker_list,
    use_cache=False,        # Use cache to avoid re-downloading
    threads=True,           # Enable multi-threading,
    start='2026-02-18',
    end='2026-02-19'
)

summary = massive_handler.list_available_data()
zero_row_symbols = [symbol for symbol, info in summary.items() if info["rows"] == 0]


data = massive_handler.download_data(
    symbols=zero_row_symbols,
    use_cache=False,        # Use cache to avoid re-downloading
    threads=True,           # Enable multi-threading,
    start='2026-02-18',
    end='2026-02-19'
)

output_dir = Path("./data/ohlc/today/it")

# Save ALL downloaded data as parquet (best for large datasets)
massive_handler.save_data(
    filepath=str(output_dir / "ohlc_data.parquet"),
    format='parquet',
    multi_symbol_strategy='single_file',
    combine_column=['open', 'high', 'low', 'close', 'volume']
)

# read the parquet file to verify it was saved correctly
today_data =pd.read_parquet('./data/ohlc/today/it/ohlc_data.parquet')

# read historical data to verify it was saved correctly
historical_data = pd.read_parquet('./data/ohlc/historical/it/ohlc_data.parquet')

# bind by rows the two dataframes
combined_data = pd.concat([historical_data, today_data], ignore_index=True)
# save the combined dataframe as parquet
combined_data.to_parquet('./data/ohlc/historical/it/ohlc_data.parquet', index=False)
