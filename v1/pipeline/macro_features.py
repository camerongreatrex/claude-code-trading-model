"""
macro_features.py — market-regime indicators consumed by signal_generation
(vix gating) and portfolio (macro-scaled sizing).

Sources: VIX/VIX9D (yfinance), 10Y-2Y spread (FRED CSV), FRED composite via
fred_features.py.

Output: data/shared/macro/macro_features.parquet with vix, vix9d,
yield_curve, vix_calm/fear, vix_zscore (60d), vix_term_ratio,
vol_backwardation, curve_inverted/steep/momentum (20d), macro_score (-1..+1),
fred_macro_score, size_multiplier (0.50–1.25).

Consumed by: signal_generation.vix_gate/commodity_regime,
portfolio.apply_macro_multiplier.
"""

import io
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pathlib import Path

DATA_DIR  = Path("data/v1/raw")
MACRO_DIR = Path("data/shared/macro")
MACRO_DIR.mkdir(parents=True, exist_ok=True)

START = "2015-01-01"
END   = "2026-01-01"


def fetch_vix() -> pd.DataFrame:
    """Download VIX (30d, fear gauge: <15 calm, 15–20 normal, 20–30 elevated,
    >30 fear, >40 panic) and VIX9D (9d, spikes faster; ratio>1 = backwardation
    = acute near-term stress) from yfinance. Falls back to vix if vix9d fails.
    Returns DataFrame[vix, vix9d], tz-naive index."""
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
    """10Y-2Y Treasury spread from FRED CSV (T10Y2Y). >1 steep, 0–1 flat
    late-cycle, <0 inverted (precedes recession 6–18mo). Direct HTTP GET
    (pandas-datareader needs distutils, removed in 3.12+).
    Fallback: 0.0 neutral series (do NOT proxy from VIX — circular)."""
    print("  Fetching yield curve from FRED...")
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"
        r   = requests.get(url, timeout=15)
        r.raise_for_status()
        df  = pd.read_csv(
            io.StringIO(r.text),
            parse_dates=["observation_date"],
            index_col="observation_date",
        )
        df.columns     = ["yield_curve"]
        df.index.name  = "Date"
        df.index       = pd.to_datetime(df.index).tz_localize(None)
        df             = df.replace(".", np.nan).astype(float)
        df             = df[(df.index >= START) & (df.index <= END)]
        return df.ffill().dropna()
    except Exception as e:
        # Do NOT use VIX to proxy — creates circular correlation.
        # Return neutral (0) so macro_score degrades gracefully.
        print(f"  FRED fetch failed ({e}) — yield curve set to neutral (0).")
        vix   = fetch_vix()
        proxy = pd.DataFrame({"yield_curve": 0.0}, index=vix.index)
        return proxy


def compute_macro_features(vix: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    """Derive macro features from raw VIX and yield-curve data.

    Thresholds (VIX <20, >30, spread <0, >1) are industry-standard, not
    fitted (avoids in-sample bias). size_multiplier scalar 0.5 → range
    0.50–1.25; max-bear cuts exposure to 50% (institutional heuristic).
    Prior 0.2 scalar (0.80–1.20) was too weak."""
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

    # Size multiplier: 0.5 scalar → 0.50–1.25 (calm/steep ~1.25x, neutral 1.0,
    # elevated/inverted ~0.75–0.85, full crisis 0.50). Untuned heuristic.
    macro["size_multiplier"] = (1.0 + 0.5 * macro["macro_score"]).clip(0.50, 1.25)

    # ── FRED macro score (decoupled — works without FRED data) ───────────
    fred_path = MACRO_DIR / "fred_features.parquet"
    if fred_path.exists():
        try:
            fred = pd.read_parquet(fred_path)
            if "fred_macro_score" in fred.columns:
                macro["fred_macro_score"] = (
                    fred["fred_macro_score"]
                    .reindex(macro.index)
                    .ffill()
                    .fillna(0.0)
                )
                print(f"  Merged fred_macro_score from {fred_path}")
        except Exception as e:
            print(f"  FRED features merge skipped: {e}")

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