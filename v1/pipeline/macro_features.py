"""
macro_features.py
-----------------
Fetches market-level regime indicators and engineers derived features.

The macro layer answers one question: "What kind of market environment are
we in right now?"  This context is consumed by signal_generation.py (to
gate signals on extreme VIX days) and portfolio.py (to scale position sizes
up/down based on macro conditions).

Data sources
────────────
  VIX  (^VIX)   — Yahoo Finance, free, no API key
  VIX9D (^VIX9D) — Yahoo Finance, free, no API key
  10Y-2Y spread  — FRED public CSV (no API key, no library required)
  FRED features  — fred_features.py (credit spreads, sentiment, claims, PMI,
                   USD index, breakeven inflation, VIX cross-check)

Output
──────
  data/shared/macro/macro_features.parquet

  Columns:
    vix              — CBOE 30-day implied volatility index
    vix9d            — CBOE 9-day implied volatility (spikes faster)
    yield_curve      — 10Y minus 2Y US Treasury spread (FRED T10Y2Y)
    vix_calm         — 1 when vix < 20 (complacency)
    vix_fear         — 1 when vix > 30 (stress)
    vix_zscore       — rolling 60-day z-score of VIX level
    vix_term_ratio   — vix9d / vix  (>1 = vol backwardation = acute stress)
    vol_backwardation— 1 when vix_term_ratio > 1
    curve_inverted   — 1 when yield_curve < 0 (recession signal)
    curve_steep      — 1 when yield_curve > 1 (strong growth signal)
    curve_momentum   — 20-day change in spread (steepening vs flattening)
    macro_score      — composite -1 to +1 (−1 = full bear, +1 = full bull)
    fred_macro_score — composite -1 to +1 from FRED indicators (if available)
    size_multiplier  — position size scalar for portfolio.py (0.50 to 1.25)

Consumed by
───────────
  signal_generation.py — vix_gate(), commodity_regime()
  portfolio.py         — apply_macro_multiplier()
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
    """
    Download VIX and VIX9D from Yahoo Finance.

    VIX (^VIX): The CBOE 30-day expected S&P 500 volatility derived from
    options prices.  Commonly called the "fear gauge".
      <15  = market complacency / low demand for hedges
      15–20 = normal / baseline
      20–30 = elevated anxiety, some hedging demand
      >30   = fear, forced selling likely
      >40   = panic / crisis conditions

    VIX9D (^VIX9D): The 9-day equivalent.  Because it uses shorter-dated
    options, VIX9D spikes faster than VIX at the onset of a stress event.
    When VIX9D > VIX (backwardation), near-term risk is elevated relative
    to medium-term risk — a useful early warning signal.

    Returns:
        DataFrame with columns [vix, vix9d] indexed by timezone-naive date.
        If VIX9D download fails, vix9d is filled with vix values.
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
    Download the 10Y-2Y Treasury yield spread from the FRED public CSV.

    The 10-year minus 2-year spread is the most widely cited recession
    indicator.  Interpretation:
      > 1.0  = steep yield curve → strong growth expectations, bank NIM favourable
      0–1.0  = flat curve → late-cycle, growth slowing
      < 0.0  = inverted curve → reliable recession predictor (historically
               precedes recession by 6–18 months)

    Uses a direct HTTP GET to the FRED CSV endpoint rather than
    pandas-datareader, which is incompatible with Python 3.12+ (the
    ``distutils`` module it depends on was removed from the standard library).

    Returns:
        DataFrame with column [yield_curve] indexed by timezone-naive date,
        forward-filled and cropped to [START, END].

    Fallback:
        If the FRED download fails (network outage, API change), returns a
        neutral (0.0) proxy so the macro score degrades gracefully rather
        than crashing the pipeline.  Does NOT proxy from VIX to avoid
        introducing circular correlation between the two macro indicators.
    """
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
    """
    Compute all derived macro features from raw VIX and yield curve data.

    Args:
        vix:   DataFrame with columns [vix, vix9d] from fetch_vix().
        curve: DataFrame with column [yield_curve] from fetch_yield_curve().

    Returns:
        Joined DataFrame with all raw and derived macro columns.

    Design notes
    ────────────
    All threshold values (VIX < 20, VIX > 30, spread < 0, spread > 1) are
    published academic / industry standards, not fitted to this dataset.
    Fitting thresholds to historical data would introduce in-sample bias
    and make the features useless for prediction.

    The size_multiplier scalar of 0.5 (giving range 0.50–1.25) was chosen
    so that the "maximum bear" environment (all signals bad) cuts exposure
    to 50% — a standard institutional risk heuristic.  The previous scalar
    of 0.2 only varied from 0.80 to 1.20 and had virtually no effect.
    """
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

    # Position size multiplier for portfolio.py.
    # Previous formula (0.2 scalar) only spanned 0.80–1.20 — virtually no effect.
    # With scalar = 0.5, the range becomes 0.50–1.25 (clipped), giving the macro
    # overlay real teeth in bear conditions:
    #   calm/steep/normal vol  → ~1.25x (more aggressive in favourable env)
    #   neutral                → 1.00x
    #   elevated VIX, inverted → ~0.75-0.85x (meaningful reduction)
    #   full crisis (all bad)  → 0.50x (half exposure)
    # The 0.5 scalar is not tuned to a specific year — it reflects the intuition
    # that "max bear" should produce half-exposure, which is a standard risk
    # management heuristic across systematic funds.
    macro["size_multiplier"] = (1.0 + 0.5 * macro["macro_score"]).clip(0.50, 1.25)

    # ── FRED macro score (from fred_features.py) ─────────────────────────
    # Merge fred_macro_score if the parquet exists.  This keeps the two
    # pipelines decoupled — macro_features.py works fine without FRED data,
    # and fred_features.py can be run independently.
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