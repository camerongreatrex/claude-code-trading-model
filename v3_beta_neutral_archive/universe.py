"""
v2/universe.py
--------------
Institutional-style universe construction for the beta-neutral model.

Universe: S&P 500 constituents filtered for liquidity and data availability,
plus a small set of ETFs for beta hedging and macro overlay.

Screening rules
───────────────
  1. Pull current S&P 500 membership from Wikipedia
  2. Filter: avg daily dollar volume > $50M over trailing 60 days
  3. Filter: require >= 2 years of clean price history
  4. Re-screen monthly (first trading day of month)

ETF overlay (always included, not part of cross-sectional alpha):
  SPY — beta hedge instrument
  TLT, GLD, UUP — optional macro overlay positions

Output
──────
  data/v2/universe.parquet — current constituent list with metadata
  data/v2/universe_history.parquet — historical screening snapshots
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

# ── Paths ─────────────────────────────────────────────────────────────────────
V2_DATA_DIR = Path("data/v2")
V2_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("data/v2/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_DOLLAR_VOLUME = 50_000_000       # $50M avg daily dollar volume
MIN_HISTORY_DAYS = 504               # ~2 years of trading days
BATCH_SIZE = 50                      # tickers per yfinance batch download
ETF_UNIVERSE = ["SPY", "TLT", "GLD", "UUP"]  # always included for hedging

# S&P 500 tickers that have known yfinance issues (class shares, etc.)
TICKER_FIXES = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}


def fetch_sp500_tickers() -> pd.DataFrame:
    """
    Fetch current S&P 500 constituents from Wikipedia.

    Returns DataFrame with columns: Symbol, Security, GICS Sector, GICS Sub-Industry
    """
    import requests, io
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AlgoTrading/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]

    # Standardise column names
    df = df.rename(columns={
        "Symbol": "symbol",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })

    # Fix ticker symbols for yfinance compatibility
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)

    df = df[["symbol", "name", "sector", "sub_industry"]].copy()
    return df


def _load_cached_constituents() -> pd.DataFrame | None:
    """Load cached constituent list if it exists and is fresh (< 30 days old)."""
    cache_path = CACHE_DIR / "sp500_constituents.parquet"
    if not cache_path.exists():
        return None
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    if (datetime.now() - mtime).days > 30:
        return None
    return pd.read_parquet(cache_path)


def _save_cached_constituents(df: pd.DataFrame):
    df.to_parquet(CACHE_DIR / "sp500_constituents.parquet")


def screen_universe(
    closes_matrix: pd.DataFrame | None = None,
    volume_matrix: pd.DataFrame | None = None,
    force_refresh: bool = False,
) -> dict:
    """
    Screen S&P 500 stocks for the v2 universe.

    Args:
        closes_matrix: pre-loaded close prices (tickers as columns). If None, downloads.
        volume_matrix: pre-loaded volume data. If None, downloads.
        force_refresh: re-download constituents even if cache is fresh.

    Returns:
        dict with keys:
            'stocks': list of stock tickers passing all filters
            'etfs': list of ETF tickers (always SPY, TLT, GLD, UUP)
            'all': stocks + etfs combined
            'sectors': dict mapping ticker -> GICS sector
            'metadata': DataFrame with full screening results
    """
    # 1. Get S&P 500 constituents
    if not force_refresh:
        constituents = _load_cached_constituents()
    else:
        constituents = None

    if constituents is None:
        print("  Fetching S&P 500 constituents from Wikipedia...")
        constituents = fetch_sp500_tickers()
        _save_cached_constituents(constituents)

    all_tickers = constituents["symbol"].tolist()
    sector_map = dict(zip(constituents["symbol"], constituents["sector"]))
    print(f"  S&P 500 raw constituents: {len(all_tickers)}")

    # 2. Download price data if not provided
    if closes_matrix is None or volume_matrix is None:
        print("  Downloading price data for screening (batched)...")
        closes_matrix, volume_matrix = _batch_download_for_screening(all_tickers)

    # 3. Apply filters
    passed = []
    failed_volume = []
    failed_history = []

    for ticker in all_tickers:
        if ticker not in closes_matrix.columns:
            failed_history.append(ticker)
            continue

        prices = closes_matrix[ticker].dropna()

        # History check
        if len(prices) < MIN_HISTORY_DAYS:
            failed_history.append(ticker)
            continue

        # Dollar volume check (trailing 60 days)
        if ticker in volume_matrix.columns:
            vol = volume_matrix[ticker].dropna()
            if len(vol) >= 60:
                dollar_vol = (prices.iloc[-60:] * vol.iloc[-60:]).mean()
                if dollar_vol < MIN_DOLLAR_VOLUME:
                    failed_volume.append(ticker)
                    continue
            else:
                failed_volume.append(ticker)
                continue
        else:
            failed_volume.append(ticker)
            continue

        passed.append(ticker)

    # Remove ETFs from stock list (they're added separately)
    passed = [t for t in passed if t not in ETF_UNIVERSE]

    print(f"  Passed all filters: {len(passed)} stocks")
    print(f"  Failed volume filter: {len(failed_volume)}")
    print(f"  Failed history filter: {len(failed_history)}")

    # Build sector map for passed stocks
    stock_sectors = {t: sector_map.get(t, "Unknown") for t in passed}

    # Build metadata DataFrame
    metadata = constituents[constituents["symbol"].isin(passed)].copy()

    result = {
        "stocks": sorted(passed),
        "etfs": ETF_UNIVERSE,
        "all": sorted(passed) + ETF_UNIVERSE,
        "sectors": stock_sectors,
        "metadata": metadata,
    }

    # Save to disk
    pd.DataFrame({
        "ticker": result["all"],
        "type": ["stock"] * len(passed) + ["etf"] * len(ETF_UNIVERSE),
        "sector": [stock_sectors.get(t, "ETF") for t in result["all"]],
    }).to_parquet(V2_DATA_DIR / "universe.parquet", index=False)

    return result


def _batch_download_for_screening(tickers: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download close prices and volume for all tickers in batches.
    Uses 3 years of data for screening (need 2 years minimum + buffer).
    """
    end = datetime.now()
    start = end - timedelta(days=3 * 365)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    all_closes = {}
    all_volumes = {}

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_str = " ".join(batch)
        print(f"    Downloading batch {i // BATCH_SIZE + 1}/"
              f"{(len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE} "
              f"({len(batch)} tickers)...")

        try:
            data = yf.download(
                batch_str,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                if "Close" in data.columns.get_level_values(0):
                    closes = data["Close"]
                    volumes = data["Volume"]
                else:
                    continue
            else:
                # Single ticker case
                if len(batch) == 1:
                    closes = data[["Close"]].rename(columns={"Close": batch[0]})
                    volumes = data[["Volume"]].rename(columns={"Volume": batch[0]})
                else:
                    continue

            for col in closes.columns:
                all_closes[col] = closes[col]
            for col in volumes.columns:
                all_volumes[col] = volumes[col]

        except Exception as e:
            print(f"    Warning: batch download failed: {e}")
            continue

    closes_df = pd.DataFrame(all_closes)
    volumes_df = pd.DataFrame(all_volumes)

    closes_df.index = pd.to_datetime(closes_df.index).tz_localize(None)
    volumes_df.index = pd.to_datetime(volumes_df.index).tz_localize(None)

    return closes_df, volumes_df


def load_universe() -> dict:
    """Load the cached universe from disk."""
    path = V2_DATA_DIR / "universe.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Universe not yet screened. Run v2/universe.py or screen_universe() first."
        )
    df = pd.read_parquet(path)
    stocks = df[df["type"] == "stock"]["ticker"].tolist()
    etfs = df[df["type"] == "etf"]["ticker"].tolist()
    sectors = dict(zip(df["ticker"], df["sector"]))
    return {
        "stocks": stocks,
        "etfs": etfs,
        "all": stocks + etfs,
        "sectors": sectors,
    }


def get_sector_map() -> dict:
    """Return ticker -> GICS sector mapping for the current universe."""
    universe = load_universe()
    return universe["sectors"]


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 60)
    print("  V2 Universe Screening")
    print("=" * 60 + "\n")

    result = screen_universe(force_refresh=True)

    print(f"\n  Final universe: {len(result['stocks'])} stocks + {len(result['etfs'])} ETFs")
    print(f"  Total instruments: {len(result['all'])}")

    # Sector breakdown
    sector_counts = pd.Series(result["sectors"]).value_counts()
    print(f"\n  Sector breakdown:")
    for sector, count in sector_counts.items():
        print(f"    {sector:<30} {count:>3}")

    print(f"\n  Universe saved to {V2_DATA_DIR / 'universe.parquet'}")
