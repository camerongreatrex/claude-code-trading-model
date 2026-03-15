"""
Fetches and processes macro data: VIX, VIX term structure, and yield curve.
Produces a daily macro regime DataFrame that signal_generation.py uses to
modulate signal confidence — not to generate trades directly.

Data sources (all free, no login):
  VIX      : yfinance ticker ^VIX
  VIX9D    : yfinance ticker ^VIX9D
  Yield curve: FRED via pandas_datareader (10Y-2Y spread, series T10Y2Y)

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
    print("pandas-datareader not installed — yield curve will be estimated from VIX.")
    print("Run: pip install pandas-datareader")

DATA_DIR  = Path("data/raw")
MACRO_DIR = Path("data/macro")
MACRO_DIR.mkdir(parents=True, exist_ok=True)

START = "2016-01-01"
END   = "2026-03-14"

# -----------------------------------------------------------------------------
# Fetch raw macro series
# -----------------------------------------------------------------------------

def fetch_vix() -> pd.DataFrame:
    """
    Download VIX (30-day expected volatility) and VIX9D (9-day expected volatility).
    Both are derived from S&P 500 options pricing — no survey, no lag, real-time fear.

    VIX levels:
      < 15  : unusually calm, complacency, momentum works well
      15-20 : normal market
      20-30 : elevated anxiety, caution warranted
      > 30  : fear regime, mean reversion opportunities, shrink position sizes
      > 40  : panic, historically the best long-term entry points, but violent swings
    """
    print("  Fetching VIX...")
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)

    vix = vix[["Close"]].rename(columns={"Close": "vix"})
    vix.index = pd.to_datetime(vix.index).tz_localize(None)

    # VIX9D: 9-day expected volatility — short-term fear sensor
    # When VIX9D > VIX it means fear is more acute right now than over the next month
    # This is called "backwardation" — something scary is happening today specifically
    print("  Fetching VIX9D...")
    try:
        vix9d = yf.download("^VIX9D", start=START, end=END, auto_adjust=True, progress=False)
        if isinstance(vix9d.columns, pd.MultiIndex):
            vix9d.columns = vix9d.columns.droplevel(1)
        vix9d = vix9d[["Close"]].rename(columns={"Close": "vix9d"})
        vix9d.index = pd.to_datetime(vix9d.index).tz_localize(None)
        vix = vix.join(vix9d, how="left")
    except Exception:
        # VIX9D has shorter history — fill with VIX as fallback
        print("  VIX9D unavailable, using VIX as proxy")
        vix["vix9d"] = vix["vix"]

    vix["vix9d"] = vix["vix9d"].fillna(vix["vix"])  # fill any gaps with VIX
    return vix


def fetch_yield_curve() -> pd.DataFrame:
    """
    Fetch the 10Y-2Y Treasury spread from FRED.
    This is the single most watched recession indicator in macro finance.

    Interpretation:
      > 1.0  : steep curve, economy expanding, be aggressive
      0-1.0  : flattening, late cycle, reduce momentum exposure
      < 0    : inverted, recession risk high, defensive posture
      < -0.5 : deeply inverted, historical high recession probability

    Falls back to estimating from VIX if FRED is unavailable.
    """
    if FRED_AVAILABLE:
        print("  Fetching yield curve from FRED...")
        try:
            # T10Y2Y: 10-Year minus 2-Year Treasury yield spread, daily, in percentage points
            spread = pdr.DataReader("T10Y2Y", "fred", start=START, end=END)
            spread.index = pd.to_datetime(spread.index).tz_localize(None)
            spread = spread.rename(columns={"T10Y2Y": "yield_curve"})
            spread = spread.ffill()  # FRED has weekend/holiday gaps — forward fill
            return spread
        except Exception as e:
            print(f"  FRED fetch failed ({e}), using VIX-based estimate")

    # Fallback: rough proxy — high VIX periods historically correlate with flat/inverted curve
    # This is NOT accurate — install pandas-datareader for real yield curve data
    print("  Using VIX-based yield curve proxy (install pandas-datareader for real data)")
    vix = fetch_vix()
    proxy = pd.DataFrame(index=vix.index)
    # Invert and scale VIX to rough curve proxy — purely illustrative
    proxy["yield_curve"] = 1.5 - (vix["vix"] - 15) * 0.05
    proxy["yield_curve"] = proxy["yield_curve"].clip(-1.5, 3.0)
    return proxy

# -----------------------------------------------------------------------------
# Compute macro regime features
# -----------------------------------------------------------------------------

def compute_macro_features(vix: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    """
    Combine VIX and yield curve into a single macro feature DataFrame.
    Each column is a signal modifier — not a trade signal by itself.
    """
    macro = vix.join(curve, how="outer").ffill().dropna()

    # --- VIX regime ---
    # Classify into three zones based on standard practitioner thresholds
    # These thresholds are industry standard — not fitted to this dataset
    macro["vix_regime"] = pd.cut(
        macro["vix"],
        bins=[0, 20, 30, np.inf],
        labels=["calm", "anxious", "fear"],
    )

    # vix_calm: 1 when VIX < 20, 0 otherwise — clean boolean for signal weighting
    macro["vix_calm"]   = (macro["vix"] < 20).astype(int)
    macro["vix_fear"]   = (macro["vix"] > 30).astype(int)

    # vix_zscore: how extreme is today's VIX relative to the last 60 days?
    # +2 means VIX is 2 std devs above its recent average — unusual fear spike
    vix_roll = macro["vix"].rolling(60, min_periods=20)
    macro["vix_zscore"] = (macro["vix"] - vix_roll.mean()) / vix_roll.std()

    # --- VIX term structure ---
    # ratio > 1: short-term fear > long-term fear = backwardation = acute stress today
    # ratio < 1: short-term fear < long-term fear = contango = calm short term
    macro["vix_term_ratio"] = macro["vix9d"] / macro["vix"]
    macro["vol_backwardation"] = (macro["vix_term_ratio"] > 1.0).astype(int)

    # --- Yield curve ---
    # inverted = negative spread = short rates > long rates = recession signal
    macro["curve_inverted"]  = (macro["yield_curve"] < 0).astype(int)
    macro["curve_steep"]     = (macro["yield_curve"] > 1).astype(int)  # > 1% = expansionary

    # curve momentum: is the curve steepening or flattening?
    # steepening (positive) = improving economic outlook
    # flattening (negative) = deteriorating outlook, reduce risk
    macro["curve_momentum"] = macro["yield_curve"].diff(20)  # 20-day change in spread

    # --- Composite macro score ---
    # Combines all macro signals into a single -1 to +1 score
    # +1 = ideal macro environment (calm, steep curve) = be aggressive
    # -1 = bad macro environment (fearful, inverted curve) = be defensive

    # each component scored -1 to +1
    calm_score   =  macro["vix_calm"] - macro["vix_fear"]          # +1 calm, -1 fear
    curve_score  =  macro["curve_steep"] - macro["curve_inverted"]  # +1 steep, -1 inverted
    term_score   = -macro["vol_backwardation"]                       # -1 in backwardation

    macro["macro_score"] = (calm_score + curve_score + term_score) / 3  # normalise to -1/+1

    # --- Position size multiplier ---
    # Converts macro score into a scaling factor for position sizes
    # macro_score of +1 → 1.2x normal size (add 20% in ideal conditions)
    # macro_score of  0 → 1.0x normal size
    # macro_score of -1 → 0.5x normal size (halve positions in bad conditions)
    # This is the key link to portfolio.py — signals stay the same, sizes adapt
    macro["size_multiplier"] = (1.0 + 0.2 * macro["macro_score"]).clip(0.5, 1.2)

    return macro

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Fetching macro data...\n")

    vix   = fetch_vix()
    curve = fetch_yield_curve()
    macro = compute_macro_features(vix, curve)

    out = MACRO_DIR / "macro_features.parquet"
    macro.to_parquet(out, engine="pyarrow", compression="snappy")

    print(f"\nMacro features saved: {macro.shape}  ->  {out}")
    print(f"Date range: {macro.index[0].date()}  ->  {macro.index[-1].date()}")

    print("\nRegime distribution:")
    print(f"  VIX calm  (<20) : {macro['vix_calm'].mean()*100:.1f}% of days")
    print(f"  VIX fear  (>30) : {macro['vix_fear'].mean()*100:.1f}% of days")
    print(f"  Curve inverted  : {macro['curve_inverted'].mean()*100:.1f}% of days")
    print(f"  Curve steep     : {macro['curve_steep'].mean()*100:.1f}% of days")
    print(f"  Vol backwardation: {macro['vol_backwardation'].mean()*100:.1f}% of days")

    print("\nMacro score distribution:")
    print(macro["macro_score"].describe().round(3))

    print("\nSize multiplier distribution:")
    print(macro["size_multiplier"].describe().round(3))

    print("\nLast 5 rows:")
    cols = ["vix", "yield_curve", "vix_term_ratio", "macro_score", "size_multiplier"]
    print(macro[cols].tail(5).round(3))


if __name__ == "__main__":
    main()