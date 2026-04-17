"""
v2/regimes/market_features.py
-----------------------------
Real-time market-implied stress signals for hybrid regime classification.

These are the "fast" signals that reduce the 1-month lag of realized macro
data. Combined with macro_features.py in the hybrid classifier.

Data Sources
────────────
  Via FRED (public CSV, no API key):
    BAMLH0A0HYM2  — HY OAS credit spread (risk-off proxy)
    BAMLC0A0CM    — IG corporate OAS
    T10YIE        — 10Y breakeven inflation (inflation expectations)
    TEDRATE       — TED Spread (interbank stress, discontinued but useful historically)

  Via Yahoo Finance:
    ^VIX          — VIX 30-day implied vol
    ^VIX9D        — VIX 9-day implied vol (faster signal)
    ^VIX3M        — VIX 3-month (for term structure)
    GLD           — Gold price (safe haven demand)
    TLT           — Long Treasury price (flight to quality)
    HYG           — HY bond price (credit risk appetite)

Stress Score
────────────
  The stress_score is a composite -2 to +2 signal:
    > +2 std dev → stress override in risk_model.py
    Combines: HY spread z-score, VIX backwardation, credit/equity divergence

Output
──────
  data/v2/regime_features/market_features.parquet
"""

import io
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pathlib import Path

DATA_DIR = Path("data/v2/regime_features")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# ── FRED series ──────────────────────────────────────────────────────────────
FRED_SERIES = [
    ("BAMLH0A0HYM2", "hy_oas",     "HY OAS Credit Spread"),
    ("BAMLC0A0CM",    "ig_oas",     "IG Corporate OAS"),
    ("T10YIE",        "breakeven",  "10Y Breakeven Inflation"),
    ("TEDRATE",       "ted_spread", "TED Spread (interbank)"),
]

# ── Yahoo Finance series ─────────────────────────────────────────────────────
YF_SERIES = [
    ("^VIX",   "vix"),
    ("^VIX9D", "vix9d"),
    ("^VIX3M", "vix3m"),
]


def _fetch_fred(series_id: str, name: str) -> pd.Series:
    """Fetch a FRED series via public CSV."""
    url = FRED_CSV_URL.format(series=series_id)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(
        io.StringIO(r.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    df.columns = [name]
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.replace(".", np.nan).astype(float)
    df = df.resample("B").last().ffill()
    return df[name].dropna()


def _fetch_yf(tickers: list[tuple[str, str]], start: str = "1990-01-01") -> pd.DataFrame:
    """Fetch Yahoo Finance price series."""
    frames = []
    for yf_ticker, col_name in tickers:
        print(f"  Fetching {yf_ticker}...")
        try:
            data = yf.download(yf_ticker, start=start, progress=False, auto_adjust=True)
            if data.empty:
                print(f"    Empty — skipped")
                continue
            s = data["Close"].squeeze()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.name = col_name
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s.index.name = "Date"
            frames.append(s)
            print(f"    {len(s):,} obs, {s.index.min().date()} -> {s.index.max().date()}")
        except Exception as e:
            print(f"    FAILED ({e}) — skipped")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def fetch_all() -> pd.DataFrame:
    """Fetch all market-implied features from FRED + Yahoo."""
    frames = []

    # FRED series
    for series_id, name, desc in FRED_SERIES:
        print(f"  Fetching {desc} ({series_id})...")
        try:
            s = _fetch_fred(series_id, name)
            if not s.empty:
                frames.append(s)
                print(f"    {len(s):,} obs, {s.index.min().date()} -> {s.index.max().date()}")
        except Exception as e:
            print(f"    FAILED ({e}) — skipped")

    # Yahoo Finance series
    yf_df = _fetch_yf(YF_SERIES)
    if not yf_df.empty:
        for col in yf_df.columns:
            frames.append(yf_df[col].dropna())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).ffill()


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer market-implied features for regime classification.

    Key derived signals:
      vix_term_ratio    — VIX9D / VIX (>1 = backwardation = acute stress)
      vix_term_slope    — VIX3M / VIX (>1 = contango = normal, <1 = inverted)
      hy_oas_zscore     — HY spread z-score (high = stress)
      credit_momentum   — 20-day change in HY OAS (rising = deteriorating)
      flight_to_quality — TLT-SPY relative performance (20-day, positive = risk-off)
    """
    result = pd.DataFrame(index=raw.index)

    # Copy raw levels
    for col in raw.columns:
        result[col] = raw[col]

    # ── VIX term structure ───────────────────────────────────────────────
    if "vix9d" in raw.columns and "vix" in raw.columns:
        result["vix_term_ratio"] = raw["vix9d"] / raw["vix"].replace(0, np.nan)
        result["vix_backwardation"] = (result["vix_term_ratio"] > 1.0).astype(float)

    if "vix3m" in raw.columns and "vix" in raw.columns:
        result["vix_term_slope"] = raw["vix3m"] / raw["vix"].replace(0, np.nan)
        result["vix_contango"] = (result["vix_term_slope"] > 1.0).astype(float)

    # ── VIX features ─────────────────────────────────────────────────────
    if "vix" in raw.columns:
        roll = raw["vix"].rolling(252, min_periods=60)
        result["vix_zscore"] = (raw["vix"] - roll.mean()) / roll.std().replace(0, np.nan)
        result["vix_20d_ma"] = raw["vix"].rolling(20).mean()
        result["vix_above_25"] = (raw["vix"] > 25).astype(float)
        result["vix_above_30"] = (raw["vix"] > 30).astype(float)
        # Realized vs implied vol (using VIX as proxy)
        result["vix_percentile"] = raw["vix"].rolling(252, min_periods=60).rank(pct=True)

    # ── Credit spreads ───────────────────────────────────────────────────
    if "hy_oas" in raw.columns:
        roll_hy = raw["hy_oas"].rolling(1260, min_periods=252)
        result["hy_oas_zscore"] = (raw["hy_oas"] - roll_hy.mean()) / roll_hy.std().replace(0, np.nan)
        result["hy_oas_20d_chg"] = raw["hy_oas"].diff(20)
        result["hy_oas_60d_chg"] = raw["hy_oas"].diff(60)
        result["hy_oas_percentile"] = raw["hy_oas"].rolling(1260, min_periods=252).rank(pct=True)

    if "ig_oas" in raw.columns:
        roll_ig = raw["ig_oas"].rolling(1260, min_periods=252)
        result["ig_oas_zscore"] = (raw["ig_oas"] - roll_ig.mean()) / roll_ig.std().replace(0, np.nan)

    # Credit quality spread (HY - IG)
    if "hy_oas" in raw.columns and "ig_oas" in raw.columns:
        result["credit_quality_spread"] = raw["hy_oas"] - raw["ig_oas"]
        roll_cqs = result["credit_quality_spread"].rolling(1260, min_periods=252)
        result["credit_quality_zscore"] = (
            (result["credit_quality_spread"] - roll_cqs.mean()) /
            roll_cqs.std().replace(0, np.nan)
        )

    # ── Breakeven inflation ──────────────────────────────────────────────
    if "breakeven" in raw.columns:
        roll_be = raw["breakeven"].rolling(1260, min_periods=252)
        result["breakeven_zscore"] = (raw["breakeven"] - roll_be.mean()) / roll_be.std().replace(0, np.nan)
        result["breakeven_20d_chg"] = raw["breakeven"].diff(20)

    return result


def compute_stress_score(features: pd.DataFrame) -> pd.Series:
    """
    Composite stress score — key input to risk_model.py stress override.

    Combines:
      - HY spread z-score (credit stress)
      - VIX z-score (equity vol stress)
      - VIX backwardation (acute stress)
      - Credit quality spread z-score (credit differentiation)

    Output range: roughly -2 to +4, but extreme stress can spike higher.
    Stress override triggers at > 2.0 std dev.
    """
    components = []
    weights = []

    mapping = {
        "hy_oas_zscore":        (1.0, 2.5),   # strongest stress signal
        "vix_zscore":           (1.0, 2.0),
        "vix_backwardation":    (1.0, 1.5),   # binary but informative
        "credit_quality_zscore": (1.0, 1.0),
    }

    for col, (direction, weight) in mapping.items():
        if col in features.columns:
            s = features[col] * direction
            components.append(s)
            weights.append(weight)

    if not components:
        return pd.Series(0.0, index=features.index, name="stress_score")

    weighted = sum(c * w for c, w in zip(components, weights)) / sum(weights)
    return weighted.rename("stress_score")


def build_market_features(use_cache: bool = False) -> pd.DataFrame:
    """
    Full pipeline: fetch → engineer → score → save.
    """
    cache_path = DATA_DIR / "market_features.parquet"
    if use_cache and cache_path.exists():
        print("  Loading cached market features...")
        return pd.read_parquet(cache_path)

    raw = fetch_all()
    if raw.empty:
        raise RuntimeError("No market data fetched — check network")

    features = engineer_features(raw)
    features["stress_score"] = compute_stress_score(features)

    features.to_parquet(cache_path)
    print(f"\n  Saved market features: {features.shape} -> {cache_path}")
    return features


if __name__ == "__main__":
    print("="*60)
    print("  V2 Market-Implied Features")
    print("="*60 + "\n")

    features = build_market_features()

    print(f"\n  Shape: {features.shape}")
    print(f"  Date range: {features.index.min().date()} -> {features.index.max().date()}")

    if "stress_score" in features.columns:
        ss = features["stress_score"].dropna()
        print(f"\n  Stress score statistics:")
        print(f"    Mean:  {ss.mean():.3f}")
        print(f"    Std:   {ss.std():.3f}")
        print(f"    Min:   {ss.min():.3f}")
        print(f"    Max:   {ss.max():.3f}")
        print(f"    >2.0 std: {(ss > 2.0).sum()} days ({(ss > 2.0).mean()*100:.1f}%)")
        print(f"\n  Last 5 rows:")
        key_cols = [c for c in ["vix", "hy_oas", "vix_term_ratio", "stress_score"]
                    if c in features.columns]
        print(features[key_cols].tail().round(3).to_string())
