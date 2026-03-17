"""
Fetches VIX, VIX9D, and yield curve data.
Produces a daily macro regime DataFrame consumed by signal_generation.py
and portfolio.py.

Install if needed: pip install pandas-datareader
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

try:
    from pandas_datareader import data as pdr
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("pandas-datareader not installed — pip install pandas-datareader")

DATA_DIR  = Path("data/raw")
MACRO_DIR = Path("data/macro")
MACRO_DIR.mkdir(parents=True, exist_ok=True)

START = "2015-01-01"
END   = "2025-01-01"


def fetch_vix() -> pd.DataFrame:
    """
    VIX: 30-day expected S&P volatility from options. Real-time fear gauge.
    <15 = complacency, 15-20 = normal, 20-30 = anxious, >30 = fear, >40 = panic.
    VIX9D: 9-day version — spikes faster, detects acute short-term stress.
    """
    print("  Fetching VIX...")
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)
    vix = vix[["Close"]].rename(columns={"Close": "vix"})
    vix.index = pd.to_datetime(vix.index).tz_localize(None)

    print("  Fetching VIX9D...")
    try:
        vix9d = yf.download("^VIX9D", start=START, end=END, auto_adjust=True, progress=False)
        if isinstance(vix9d.columns, pd.MultiIndex):
            vix9d.columns = vix9d.columns.droplevel(1)
        vix9d = vix9d[["Close"]].rename(columns={"Close": "vix9d"})
        vix9d.index = pd.to_datetime(vix9d.index).tz_localize(None)
        vix = vix.join(vix9d, how="left")
    except Exception:
        vix["vix9d"] = vix["vix"]

    vix["vix9d"] = vix["vix9d"].fillna(vix["vix"])
    return vix


def fetch_yield_curve() -> pd.DataFrame:
    """
    10Y-2Y Treasury spread from FRED. The most reliable recession indicator.
    >1.0 = steep/expansionary, 0-1 = flattening/late cycle, <0 = inverted/recession risk.
    """
    if FRED_AVAILABLE:
        print("  Fetching yield curve from FRED...")
        try:
            spread = pdr.DataReader("T10Y2Y", "fred", start=START, end=END)
            spread.index = pd.to_datetime(spread.index).tz_localize(None)
            spread = spread.rename(columns={"T10Y2Y": "yield_curve"})
            return spread.ffill()
        except Exception as e:
            print(f"  FRED failed ({e}), using VIX proxy")

    # Do NOT use VIX to proxy the yield curve — that creates circular correlation
    # (macro_score would then be 2/3 VIX-driven instead of 1/3).
    # Return a neutral flat curve (0 spread) so macro_score degrades gracefully.
    print("  FRED unavailable — yield curve set to neutral (0). Install pandas-datareader for real data.")
    vix = fetch_vix()
    proxy = pd.DataFrame(index=vix.index)
    proxy["yield_curve"] = 0.0
    return proxy


def compute_macro_features(vix: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    macro = vix.join(curve, how="outer").ffill().dropna()

    # VIX regime thresholds: industry-standard, not fitted to this dataset
    macro["vix_calm"]   = (macro["vix"] < 20).astype(int)
    macro["vix_fear"]   = (macro["vix"] > 30).astype(int)

    # rolling z-score: how extreme is today's VIX vs recent 60 days?
    vix_roll = macro["vix"].rolling(60, min_periods=20)
    macro["vix_zscore"] = (macro["vix"] - vix_roll.mean()) / vix_roll.std()

    # term structure: VIX9D/VIX > 1 = backwardation = acute near-term stress
    macro["vix_term_ratio"]    = macro["vix9d"] / macro["vix"]
    macro["vol_backwardation"] = (macro["vix_term_ratio"] > 1.0).astype(int)

    # yield curve regimes
    macro["curve_inverted"] = (macro["yield_curve"] < 0).astype(int)
    macro["curve_steep"]    = (macro["yield_curve"] > 1).astype(int)
    macro["curve_momentum"] = macro["yield_curve"].diff(20)  # steepening or flattening?

    # composite macro score -1 to +1
    calm_score  =  macro["vix_calm"] - macro["vix_fear"]
    curve_score =  macro["curve_steep"] - macro["curve_inverted"]
    term_score  = -macro["vol_backwardation"]

    macro["macro_score"] = (calm_score + curve_score + term_score) / 3

    # position size multiplier for portfolio.py: 0.5x (fear) to 1.2x (calm)
    macro["size_multiplier"] = (1.0 + 0.2 * macro["macro_score"]).clip(0.5, 1.2)

    return macro


def main():
    print("Fetching macro data...\n")
    vix   = fetch_vix()
    curve = fetch_yield_curve()
    macro = compute_macro_features(vix, curve)

    out = MACRO_DIR / "macro_features.parquet"
    macro.to_parquet(out, engine="pyarrow", compression="snappy")

    print(f"\nMacro features: {macro.shape}  ->  {out}")
    print(f"VIX calm  : {macro['vix_calm'].mean()*100:.1f}% of days")
    print(f"VIX fear  : {macro['vix_fear'].mean()*100:.1f}% of days")
    print(f"Inverted  : {macro['curve_inverted'].mean()*100:.1f}% of days")
    print(f"Steep     : {macro['curve_steep'].mean()*100:.1f}% of days")
    print(f"\nLast 5 rows:")
    print(macro[["vix", "yield_curve", "macro_score", "size_multiplier"]].tail(5).round(3))


if __name__ == "__main__":
    main()