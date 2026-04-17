"""
shared/famafrench.py
--------------------
Download and cache Fama-French factor data from Ken French's data library.

Provides daily and monthly 5-factor + momentum data for:
  - Factor attribution (Mkt-RF, SMB, HML, RMW, CMA, Mom)
  - Extended historical validation (industry portfolios back to 1963)

Data source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Uses CSV zip files — no API key required.
"""

import io
import zipfile
import numpy as np
import pandas as pd
import requests
from pathlib import Path

CACHE_DIR = Path("data/macro/famafrench")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

# Dataset names → zip file names
DATASETS = {
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "ff5_monthly": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "mom_monthly": "F-F_Momentum_Factor_CSV.zip",
    "industries_monthly": "10_Industry_Portfolios_CSV.zip",
}


def _download_and_extract(dataset_key: str) -> str:
    """Download a FF zip file and return the CSV text content."""
    filename = DATASETS[dataset_key]
    url = f"{BASE_URL}/{filename}"
    print(f"  Downloading {filename}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        return z.read(csv_name).decode("utf-8")


def _parse_ff5(text: str, freq: str) -> pd.DataFrame:
    """Parse Fama-French 5-factor CSV text into a DataFrame."""
    lines = text.strip().split("\n")

    # Find the header row (first row with "Mkt-RF")
    start_idx = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not find header row in FF5 data")

    # Read until blank line or non-numeric data (annual factors section)
    data_lines = [lines[start_idx]]  # header
    for line in lines[start_idx + 1:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        data_lines.append(stripped)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df.columns = [c.strip() for c in df.columns]

    # Parse date column (first unnamed column)
    date_col = df.columns[0]
    if freq == "daily":
        df["Date"] = pd.to_datetime(df[date_col], format="%Y%m%d")
    else:
        df["Date"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m")
        df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)

    df = df.set_index("Date").drop(columns=[date_col])
    df = df.apply(pd.to_numeric, errors="coerce")
    # Convert from percentage to decimal
    df = df / 100.0
    return df


def _parse_momentum(text: str, freq: str) -> pd.DataFrame:
    """Parse Fama-French momentum factor CSV."""
    lines = text.strip().split("\n")

    # Find the CSV header row: ",Mom" or "Date,Mom" pattern
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(",Mom") or stripped.startswith(",  Mom"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not find header row in momentum data")

    data_lines = [lines[start_idx]]
    for line in lines[start_idx + 1:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        data_lines.append(stripped)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df.columns = [c.strip() for c in df.columns]

    # First column is unnamed (date), rename it
    date_col = df.columns[0]
    if freq == "daily":
        df["Date"] = pd.to_datetime(df[date_col], format="%Y%m%d")
    else:
        df["Date"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m")
        df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)

    df = df.set_index("Date").drop(columns=[date_col])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df / 100.0
    df.columns = ["Mom"]
    return df


def _parse_industries(text: str) -> pd.DataFrame:
    """Parse Fama-French 10-industry portfolio monthly returns."""
    lines = text.strip().split("\n")

    # Find the value-weighted returns section
    start_idx = None
    for i, line in enumerate(lines):
        if "NoDur" in line and "Durbl" in line:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not find header in industry data")

    data_lines = [lines[start_idx]]
    for line in lines[start_idx + 1:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        data_lines.append(stripped)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df.columns = [c.strip() for c in df.columns]

    date_col = df.columns[0]
    df["Date"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m")
    df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)
    df = df.set_index("Date").drop(columns=[date_col])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df / 100.0
    return df


def get_ff5_factors(freq: str = "monthly", use_cache: bool = True) -> pd.DataFrame:
    """
    Get Fama-French 5 factors + Momentum as a single DataFrame.

    Returns columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    Values in decimal (0.01 = 1%).

    Args:
        freq: "daily" or "monthly"
        use_cache: if True, use cached parquet if available
    """
    cache_path = CACHE_DIR / f"ff5_mom_{freq}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Loaded cached FF5+Mom ({freq}): {len(df):,} obs")
        return df

    # Download and parse
    ff5_text = _download_and_extract(f"ff5_{freq}")
    ff5 = _parse_ff5(ff5_text, freq)

    mom_text = _download_and_extract(f"mom_{freq}")
    mom = _parse_momentum(mom_text, freq)

    df = ff5.join(mom, how="inner")
    df = df.dropna()
    df.index.name = "Date"

    # Cache
    df.to_parquet(cache_path)
    print(f"  Cached FF5+Mom ({freq}): {len(df):,} obs, "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


def get_industry_portfolios(use_cache: bool = True) -> pd.DataFrame:
    """
    Get Fama-French 10-industry portfolio monthly returns.
    For extended historical validation (back to 1926).
    """
    cache_path = CACHE_DIR / "industries_monthly.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Loaded cached industries: {len(df):,} obs")
        return df

    text = _download_and_extract("industries_monthly")
    df = _parse_industries(text)
    df = df.dropna()
    df.index.name = "Date"

    df.to_parquet(cache_path)
    print(f"  Cached industries: {len(df):,} obs, "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


if __name__ == "__main__":
    print("Downloading Fama-French data...\n")
    ff5_m = get_ff5_factors("monthly", use_cache=False)
    print(f"\nMonthly factors: {ff5_m.columns.tolist()}")
    print(ff5_m.tail())

    ff5_d = get_ff5_factors("daily", use_cache=False)
    print(f"\nDaily factors: {len(ff5_d):,} observations")

    ind = get_industry_portfolios(use_cache=False)
    print(f"\nIndustry portfolios: {ind.columns.tolist()}")
    print(f"  Range: {ind.index.min().date()} → {ind.index.max().date()}")
