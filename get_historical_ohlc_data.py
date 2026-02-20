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


tickers = pd.read_excel("data/ticker/it/ticker.xlsx")

ticker_list = tickers['ticker'].tolist()
# ticker_list = ticker_list[:10]  # Limit to first 10 tickers for testing

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
    end='2026-02-20',    # End date for historical data
    # period='5y',           # 5 years of historical data
    interval='1d',         # Daily data
    use_cache=False,        # Use cache to avoid re-downloading
    threads=True           # Enable multi-threading
)

output_dir = Path("./data/ohlc/historical/it")

# Save ALL downloaded data as parquet (best for large datasets)
massive_handler.save_data(
    filepath=str(output_dir / "ohlc_data.parquet"),
    format='parquet',
    multi_symbol_strategy='single_file',
    combine_column=['open', 'high', 'low', 'close', 'volume']
)

summary = massive_handler.list_available_data()
zero_row_symbols = [symbol for symbol, info in summary.items() if info["rows"] == 0]




data = massive_handler.download_data(
    symbols=zero_row_symbols,
    start='2016-01-01',  # Start date for historical data
    end='2026-02-20',    # End date for historical data
    # period='5y',           # 5 years of historical data
    interval='1d',         # Daily data
    use_cache=False,        # Use cache to avoid re-downloading
    threads=True           # Enable multi-threading
)


# Save ALL downloaded data as parquet (best for large datasets)
massive_handler.save_data(
    filepath=str(output_dir / "ohlc_data.parquet"),
    format='parquet',
    multi_symbol_strategy='single_file',
    combine_column=['open', 'high', 'low', 'close', 'volume']
)

df = pd.read_parquet("./data/ohlc/historical/it/ohlc_data.parquet")
df_avio = df[df["symbol"] == "AVIO.MI"]
print(df_avio.tail())