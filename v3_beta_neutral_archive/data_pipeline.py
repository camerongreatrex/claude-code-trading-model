"""
v2/data_pipeline.py
-------------------
Downloads, caches, and serves daily OHLCV price data for the v2 universe
(~450 S&P 500 stocks + 4 ETFs).

Caching strategy
────────────────
  Local parquet files per ticker in data/v2/raw/{TICKER}.parquet.
  On trading days, re-download only if the cached file is stale (last
  modified before today). On weekends/holidays, cache is always fresh.
  Batched yfinance downloads (50 tickers per request) to manage API load.

Outputs
───────
  data/v2/raw/{TICKER}.parquet       — per-ticker daily OHLCV
  data/v2/raw/closes_matrix.parquet  — aligned close prices (T × N)
  data/v2/raw/volume_matrix.parquet  — aligned volumes (T × N)
  data/v2/raw/returns_matrix.parquet — aligned daily log returns (T × N)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

from v2.universe import load_universe, screen_universe, ETF_UNIVERSE

# ── Paths ─────────────────────────────────────────────────────────────────────
V2_RAW_DIR = Path("data/v2/raw")
V2_RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
START = "2010-01-01"
END = "2026-01-01"
BATCH_SIZE = 50
MIN_TRADING_DAYS = 504  # ~2 years


def _is_fresh(path: Path, today: pd.Timestamp) -> bool:
    """Return True if file exists and was modified today (cache hit)."""
    if not path.exists():
        return False
    mtime = pd.Timestamp.fromtimestamp(path.stat().st_mtime).normalize()
    return mtime >= today


def download_batch(tickers: list, start: str = START, end: str = END) -> dict:
    """
    Download OHLCV data for a batch of tickers from yfinance.

    Returns dict of ticker -> DataFrame with [Open, High, Low, Close, Volume].
    """
    ticker_str = " ".join(tickers)
    try:
        data = yf.download(
            ticker_str,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"    Warning: batch download failed: {e}")
        return {}

    if data.empty:
        return {}

    result = {}
    if isinstance(data.columns, pd.MultiIndex):
        available_tickers = data.columns.get_level_values(1).unique()
        for ticker in tickers:
            if ticker in available_tickers:
                try:
                    df = data.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]]
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df = df.dropna(how="all")
                    if len(df) >= MIN_TRADING_DAYS:
                        result[ticker] = df
                except Exception:
                    continue
    else:
        # Single ticker
        if len(tickers) == 1:
            df = data[["Open", "High", "Low", "Close", "Volume"]]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(how="all")
            if len(df) >= MIN_TRADING_DAYS:
                result[tickers[0]] = df

    return result


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filters to raw OHLCV data."""
    df = df.ffill().dropna()
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df = df[df["High"] >= df["Low"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def download_all(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download and cache OHLCV data for the full v2 universe.

    Args:
        force: re-download everything even if cache is fresh.

    Returns:
        (closes_matrix, volume_matrix, returns_matrix) — aligned DataFrames
        with tickers as columns and dates as index.
    """
    # Load or build universe
    try:
        universe = load_universe()
    except FileNotFoundError:
        print("  Universe not found, running initial screening...")
        result = screen_universe(force_refresh=True)
        universe = {
            "stocks": result["stocks"],
            "etfs": result["etfs"],
            "all": result["all"],
        }

    all_tickers = universe["all"]
    today = pd.Timestamp.now().normalize()

    # Determine which tickers need downloading
    if force:
        to_download = all_tickers
    else:
        to_download = []
        for ticker in all_tickers:
            path = V2_RAW_DIR / f"{ticker}.parquet"
            if not _is_fresh(path, today):
                to_download.append(ticker)

    cached_count = len(all_tickers) - len(to_download)
    if cached_count > 0:
        print(f"  Cache hit: {cached_count} tickers already fresh")

    if to_download:
        print(f"  Downloading {len(to_download)} tickers in batches of {BATCH_SIZE}...")
        n_batches = (len(to_download) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(to_download), BATCH_SIZE):
            batch = to_download[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            print(f"    Batch {batch_num}/{n_batches} ({len(batch)} tickers)...")
            batch_data = download_batch(batch)
            for ticker, df in batch_data.items():
                df = clean(df)
                df.to_parquet(V2_RAW_DIR / f"{ticker}.parquet")

    # Build aligned matrices from cached files
    return build_matrices(all_tickers)


def build_matrices(tickers: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build aligned closes, volume, and returns matrices from cached per-ticker parquets.
    """
    if tickers is None:
        universe = load_universe()
        tickers = universe["all"]

    closes = {}
    volumes = {}

    for ticker in tickers:
        path = V2_RAW_DIR / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "Close" in df.columns and len(df) >= MIN_TRADING_DAYS:
                closes[ticker] = df["Close"]
                volumes[ticker] = df["Volume"]

    closes_df = pd.DataFrame(closes).sort_index()
    volume_df = pd.DataFrame(volumes).sort_index()

    # Forward-fill gaps (holidays where some exchanges are closed)
    closes_df = closes_df.ffill()
    volume_df = volume_df.fillna(0)

    # Log returns
    returns_df = np.log(closes_df / closes_df.shift(1))

    # Save matrices
    closes_df.to_parquet(V2_RAW_DIR / "closes_matrix.parquet")
    volume_df.to_parquet(V2_RAW_DIR / "volume_matrix.parquet")
    returns_df.to_parquet(V2_RAW_DIR / "returns_matrix.parquet")

    print(f"  Built matrices: {closes_df.shape[1]} tickers × {closes_df.shape[0]} days")
    return closes_df, volume_df, returns_df


def load_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-built matrices from disk."""
    closes = pd.read_parquet(V2_RAW_DIR / "closes_matrix.parquet")
    volumes = pd.read_parquet(V2_RAW_DIR / "volume_matrix.parquet")
    returns = pd.read_parquet(V2_RAW_DIR / "returns_matrix.parquet")
    return closes, volumes, returns


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 60)
    print("  V2 Data Pipeline")
    print("=" * 60 + "\n")

    force = "--force" in sys.argv
    closes, volumes, returns = download_all(force=force)

    print(f"\n  Tickers with data: {closes.shape[1]}")
    print(f"  Date range: {closes.index[0].date()} to {closes.index[-1].date()}")
    print(f"  Trading days: {closes.shape[0]}")

    # Quick data quality check
    pct_missing = closes.isna().mean().mean() * 100
    print(f"  Missing data: {pct_missing:.2f}%")
