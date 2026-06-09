from algoshort.yfinance_handler import YFinanceDataHandler

import logging
from pathlib import Path
from datetime import date
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

tickers = pd.read_excel("data/ticker/etf/ticker.xlsx")
ticker_list = tickers['ticker'].tolist()

today = date.today()

massive_handler = YFinanceDataHandler(
    cache_dir="../data/ohlc/etf",
    enable_logging=True,
    chunk_size=20,
    log_level=logging.INFO
)

data = massive_handler.download_data(
    symbols=ticker_list,
    start='2016-01-01',
    end=today.isoformat(),
    interval='1d',
    use_cache=False,
    threads=True
)

output_dir = Path("./data/ohlc/historical/etf")

massive_handler.save_data(
    filepath=str(output_dir / "ohlc_data.parquet"),
    format='parquet',
    multi_symbol_strategy='single_file',
    combine_column=['open', 'high', 'low', 'close', 'volume']
)

# Retry any symbols that came back empty on the first pass
summary = massive_handler.list_available_data()
zero_row_symbols = [symbol for symbol, info in summary.items() if info["rows"] == 0]

if zero_row_symbols:
    data = massive_handler.download_data(
        symbols=zero_row_symbols,
        start='2016-01-01',
        end=today.isoformat(),
        interval='1d',
        use_cache=False,
        threads=True
    )

    massive_handler.save_data(
        filepath=str(output_dir / "ohlc_data.parquet"),
        format='parquet',
        multi_symbol_strategy='single_file',
        combine_column=['open', 'high', 'low', 'close', 'volume']
    )
