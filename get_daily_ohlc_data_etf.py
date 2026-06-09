from algoshort.yfinance_handler import YFinanceDataHandler
import logging
import pandas as pd
from pathlib import Path
from datetime import date

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

tickers = pd.read_excel("data/ticker/etf/ticker.xlsx")
ticker_list = tickers['ticker'].tolist()

massive_handler = YFinanceDataHandler(
    cache_dir="data/ohlc/today/etf",
    enable_logging=True,
    chunk_size=20,
    log_level=logging.INFO
)

today = date.today()
tomorrow = today + pd.Timedelta(days=1)

data = massive_handler.download_data(
    symbols=ticker_list,
    use_cache=False,
    threads=True,
    start=today,
    end=tomorrow
)

output_dir = Path("./data/ohlc/today/etf")

massive_handler.save_data(
    filepath=str(output_dir / "ohlc_data.parquet"),
    format='parquet',
    multi_symbol_strategy='single_file',
    combine_column=['open', 'high', 'low', 'close', 'volume']
)
