"""
data_pipeline.py

Downloads, cleans, and saves raw OHLCV data.
Universe deliberately spans uncorrelated asset classes:
  equities (large/small/international), bonds, gold, and sector ETFs.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

# Diversified universe — each asset class responds to different economic drivers.
# This is the fix for the all-tech correlation problem.
TICKERS = {
    "SPY": "equity",       # US large cap — market benchmark
    "IWM": "equity",       # US small cap — different risk profile to large cap
    "EEM": "equity",       # Emerging markets — EM macro, China policy driven
    "TLT": "bond",         # 20yr US Treasury — rate sensitive, often inverse equities
    "GLD": "commodity",    # Gold — fear hedge, inflation hedge
    "XLE": "sector",       # Energy — oil price driven, not Fed policy
    "XLU": "sector",       # Utilities — defensive, rate sensitive
    "XLF": "sector",       # Financials — bank margins, rate cycle driven
}

# Asset class lookup used by signal_generation to apply the right strategy
ASSET_CLASS = TICKERS
TICKER_LIST = list(TICKERS.keys())

START = "2015-01-01"
END   = "2025-01-01"

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # yfinance quirk — flatten column names

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)  # strip timezone to avoid merge issues
    df.index.name = "Date"
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.ffill().dropna()  # forward-fill gaps — never interpolate (lookahead bias)
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df = df[df["High"] >= df["Low"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def main():
    all_data = {}
    print("Downloading and processing...\n")

    for ticker in TICKER_LIST:
        asset_class = ASSET_CLASS[ticker]
        df = download(ticker)
        df = clean(df)

        path = DATA_DIR / f"{ticker}.parquet"
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        print(f"  {ticker} ({asset_class}): {len(df)} rows  ->  {path}")
        all_data[ticker] = df

    closes = pd.DataFrame({t: d["Close"] for t, d in all_data.items()}).dropna()
    closes.to_parquet(DATA_DIR / "closes_matrix.parquet")

    print(f"\nClose matrix: {closes.shape}  (trading days x tickers)")
    print("\nCorrelation matrix (lower = more diversified):")
    returns = closes.pct_change().dropna()
    print(returns.corr().round(2))


if __name__ == "__main__":
    main()