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

START = "2015-01-01"
END = "2025-01-01"

DATA_DIR = Path("data/raw")  # pathlib avoids OS-specific slash issues (Windows "\" vs Mac/Linux "/")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download(ticker: str) -> pd.DataFrame:
    # auto_adjust: retroactively corrects prices for splits/dividends so the series is smooth
    # e.g. without it, Apple's 4:1 split in 2020 would show a fake -75% price drop
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

    # yfinance quirk: sometimes returns columns like ("Close", "AAPL") instead of "Close" — flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # strip timezone — if one ticker is UTC-tagged and another isn't, pandas refuses to merge them
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # ffill: carry the last valid value forward into any gap
    # never interpolate — that averages before and after the gap, using a future value to fill a past one (lookahead bias)
    df = df.ffill().dropna()

    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]  # zero/negative price = data error
    df = df[df["High"] >= df["Low"]]  # high < low is physically impossible — corrupt row

    # keep="last": if a date appears twice, keep the second entry (data providers sometimes send corrections)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def main():
    all_data = {}
    print("Downloading and processing...\n")

    for ticker in TICKERS:
        df = download(ticker)
        df = clean(df)

        path = DATA_DIR / f"{ticker}.parquet"
        df.to_parquet(path, engine="pyarrow", compression="snappy")  # snappy: fast decompress, good for backtesting
        print(f"  {ticker}: {len(df)} rows  ->  {path}")
        all_data[ticker] = df

    # close matrix: rows = trading days, cols = tickers — base for all multi-asset operations
    closes = pd.DataFrame({t: d["Close"] for t, d in all_data.items()}).dropna()
    closes.to_parquet(DATA_DIR / "closes_matrix.parquet")

    print(f"\nClose matrix: {closes.shape}  (trading days x tickers)")
    print(closes.tail(3))


if __name__ == "__main__":
    main()