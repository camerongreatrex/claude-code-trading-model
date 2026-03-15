"""
data_pipeline.py

Talks to the outside world: downloads, cleans, and saves raw OHLCV data.
Nothing here does any signal math — that lives in feature_engineering.py.
Run this once to build your dataset, then re-run periodically to refresh it.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path


# Small multi-asset universe for testing. In real research this might be hundreds or thousands of tickers.
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

# Time range for historical data download
START = "2015-01-01"
END = "2025-01-01"

# Directory where raw datasets are stored. Pathlib avoids OS-specific slash issues (Windows "\" vs Mac/Linux "/").
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download function
def download(ticker: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    auto_adjust=True retroactively adjusts prices for:
    • stock splits
    • dividends
    Without this, splits would appear as massive artificial price drops that
    completely break return calculations.
    """
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

    # yfinance sometimes returns multi-level column names like ("Close", "AAPL") instead of just "Close". This flattens the structure.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Keep only OHLCV fields used in most quantitative models
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Remove timezone metadata. Mixing timezone-aware and timezone-naive indices causes pandas merge errors when combining multiple tickers.
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df

# Clean function
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic sanity checks to ensure data integrity. Market data often contains
    small inconsistencies or corrections that can corrupt statistical
    calculations if left untreated.
    """
    # Fill small missing gaps using last known value.
    # IMPORTANT: never interpolate financial prices because that introduces
    # future information into past rows (lookahead bias).
    df = df.ffill().dropna()

    # Remove rows with impossible prices (data feed errors).
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]

    # High must always be >= Low. If not, the row is corrupted.
    df = df[df["High"] >= df["Low"]]

    # Duplicate timestamps sometimes occur due to vendor corrections.
    # keep="last" assumes the latest row contains the corrected value.
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

# Main function
def main():
    # Store each ticker's dataframe so we can later construct matrices
    # across assets (useful for correlation, portfolio construction, etc.)
    all_data = {}
    print("Downloading and processing...\n")

    for ticker in TICKERS:
        # Download and clean the data for the current ticker
        df = download(ticker)
        df = clean(df)

        # Parquet is much faster and smaller than CSV for numeric datasets.
        path = DATA_DIR / f"{ticker}.parquet"         # Save the cleaned data to a Parquet file
        df.to_parquet(path, engine="pyarrow", compression="snappy") # Use PyArrow for faster compression
        print(f"  {ticker}: {len(df)} rows  ->  {path}") # Print the number of rows and the path to the file
        all_data[ticker] = df # Store the cleaned data in the all_data dictionary

    # Multi-asset matrices: rows = trading days, columns = tickers.
    # Standard format for correlation, portfolio optimization, cross-sectional strategies.
    closes = pd.DataFrame({t: d["Close"] for t, d in all_data.items()}).dropna()
    closes.to_parquet(DATA_DIR / "closes_matrix.parquet") # Save the closes matrix to a Parquet file

    # Print the shape and tail of the closes matrix
    print(f"\nClose matrix: {closes.shape} (trading days x tickers)")
    print(closes.tail(3)) # Print the last 3 prices (rows) of the closes matrix 


if __name__ == "__main__":
    main()
