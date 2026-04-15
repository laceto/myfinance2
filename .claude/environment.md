# Environment Reference — myfinance2

## Setup

```bash
# 1. Create and activate venv (gitignored)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install declared dependencies
pip install -r requirements.txt

# 3. Install the private algoshort wheel (not on PyPI)
pip install algoshort-0.1.0-py3-none-any.whl

# 4. Install extra packages not in requirements.txt
pip install yfinance openpyxl pyarrow joblib
```

> `algoshort-0.1.0-py3-none-any.whl` must be present in the repo root.
> It is a private package — do not attempt to `pip install algoshort` from PyPI.

## Running Scripts

```bash
# Download today's OHLC bar for all Italian market tickers
python get_daily_ohlc_data.py

# Download full historical OHLC data (2016 → present)
python get_historical_ohlc_data.py

# Run stock signal analysis
python analyze_stock.py
```

## Key Paths

| Path | Contents |
|------|---------|
| `data/ticker/it/ticker.xlsx` | Master ticker list |
| `data/ohlc/today/it/ohlc_data.parquet` | Today's OHLC bar |
| `data/ohlc/historical/it/ohlc_data.parquet` | Full historical dataset |
| `config.json` | Strategy parameters (canonical reference) |
| `.github/workflows/` | CI workflow definitions |
