"""
v2/regimes/macro_features.py
----------------------------
Fetch and engineer realized macro features for regime classification.

These are the "slow" signals — monthly/quarterly economic indicators that
define the fundamental macro state. Combined with market-implied features
(market_features.py) in the hybrid classifier.

FRED Series (all via public CSV, no API key)
─────────────────────────────────────────────
  Growth axis:
    PAYEMS      — Total Nonfarm Payrolls (monthly, leading employment)
    INDPRO      — Industrial Production Index (monthly)
    RSAFS       — Retail Sales (monthly, consumer spending)
    UNRATE      — Unemployment Rate (monthly, lagging but confirms regime)
    ICSA        — Initial Jobless Claims (weekly, leading)

  Inflation axis:
    CPIAUCSL    — CPI All Urban Consumers (monthly)
    PCEPI       — PCE Price Index (monthly, Fed's preferred)
    T10YIE      — 10Y Breakeven Inflation (daily, market-implied)
    PPIFIS      — PPI Final Demand (monthly)

  Yield curve / monetary:
    T10Y2Y      — 10Y-2Y Treasury Spread (daily)
    DFF         — Fed Funds Rate (daily)
    T10Y3M      — 10Y-3M Treasury Spread (daily)

Output
──────
  data/v2/regime_features/macro_features.parquet
"""

import io
import numpy as np
import pandas as pd
import requests
from pathlib import Path

DATA_DIR = Path("data/v2/regime_features")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# (fred_id, short_name, axis, description)
SERIES = [
    # Growth
    ("PAYEMS",   "nfp",           "growth",    "Total Nonfarm Payrolls"),
    ("INDPRO",   "indpro",        "growth",    "Industrial Production Index"),
    ("RSAFS",    "retail_sales",  "growth",    "Retail Sales"),
    ("UNRATE",   "unemp",         "growth",    "Unemployment Rate"),
    ("ICSA",     "claims",        "growth",    "Initial Jobless Claims"),
    # Inflation
    ("CPIAUCSL", "cpi",           "inflation", "CPI All Urban Consumers"),
    ("PCEPI",    "pce",           "inflation", "PCE Price Index"),
    ("T10YIE",   "breakeven_10y", "inflation", "10Y Breakeven Inflation"),
    ("PPIFIS",   "ppi",           "inflation", "PPI Final Demand"),
    # Yield curve / monetary
    ("T10Y2Y",   "curve_10y2y",   "monetary",  "10Y-2Y Treasury Spread"),
    ("DFF",      "fed_funds",     "monetary",  "Fed Funds Rate"),
    ("T10Y3M",   "curve_10y3m",   "monetary",  "10Y-3M Treasury Spread"),
]


def fetch_fred_series(series_id: str, name: str) -> pd.Series:
    """Fetch a FRED series via public CSV endpoint, resample to business daily."""
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


def fetch_all() -> pd.DataFrame:
    """Fetch all macro FRED series."""
    frames = []
    for series_id, name, axis, desc in SERIES:
        print(f"  [{axis:10s}] Fetching {desc} ({series_id})...")
        try:
            s = fetch_fred_series(series_id, name)
            if not s.empty:
                frames.append(s)
                print(f"             {len(s):,} obs, {s.index.min().date()} -> {s.index.max().date()}")
            else:
                print(f"             Empty — skipped")
        except Exception as e:
            print(f"             FAILED ({e}) — skipped")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).ffill()


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer regime-relevant features from raw macro data.

    For each indicator:
      {col}_yoy       — year-over-year % change (for levels: PAYEMS, CPI, etc.)
      {col}_3m_chg    — 3-month change (for rates/spreads)
      {col}_6m_chg    — 6-month change
      {col}_zscore    — rolling 5-year z-score
      {col}_accel     — acceleration (change in the rate of change)
    """
    result = pd.DataFrame(index=raw.index)

    # Indicators that are levels (compute YoY % change)
    level_cols = ["nfp", "indpro", "retail_sales", "cpi", "pce", "ppi"]
    # Indicators that are already rates/spreads (compute changes directly)
    rate_cols = ["unemp", "claims", "breakeven_10y", "curve_10y2y",
                 "fed_funds", "curve_10y3m"]

    for col in raw.columns:
        if col not in result.columns:
            result[col] = raw[col]

        if col in level_cols:
            # Year-over-year % change (252 biz days ~ 1 year)
            result[f"{col}_yoy"] = raw[col].pct_change(252)
            # 3-month % change
            result[f"{col}_3m_chg"] = raw[col].pct_change(63)
        elif col in rate_cols:
            # 3-month change in level
            result[f"{col}_3m_chg"] = raw[col].diff(63)
            # 6-month change
            result[f"{col}_6m_chg"] = raw[col].diff(126)

        # Rolling 5-year z-score (1260 biz days)
        roll = raw[col].rolling(1260, min_periods=252)
        mu, sigma = roll.mean(), roll.std()
        result[f"{col}_zscore"] = (raw[col] - mu) / sigma.replace(0, np.nan)

        # Acceleration: change in 3-month change
        chg_3m = raw[col].diff(63) if col in rate_cols else raw[col].pct_change(63)
        result[f"{col}_accel"] = chg_3m.diff(63)

    return result


def compute_growth_score(features: pd.DataFrame) -> pd.Series:
    """
    Composite growth score from -1 (deep recession) to +1 (strong expansion).

    Uses directional z-scores of growth indicators:
      - NFP growth, IP growth, retail sales growth → positive = bullish
      - Unemployment, claims → positive = bearish (inverted)
    """
    components = []
    weights = []

    mapping = {
        "nfp_yoy":          (+1, 2.0),   # strong signal, double weight
        "indpro_yoy":       (+1, 1.5),
        "retail_sales_yoy": (+1, 1.0),
        "unemp_zscore":     (-1, 1.5),   # inverted: high unemployment = bearish
        "claims_zscore":    (-1, 1.5),   # inverted: high claims = bearish
    }

    for col, (direction, weight) in mapping.items():
        if col in features.columns:
            s = features[col] * direction
            # Clip to avoid extreme outliers dominating
            s = s.clip(-3, 3)
            components.append(s)
            weights.append(weight)

    if not components:
        return pd.Series(0.0, index=features.index, name="growth_score")

    weighted = sum(c * w for c, w in zip(components, weights)) / sum(weights)
    # Normalize to [-1, 1] using tanh
    return np.tanh(weighted).rename("growth_score")


def compute_inflation_score(features: pd.DataFrame) -> pd.Series:
    """
    Composite inflation score from -1 (deflation) to +1 (high inflation).

    Higher = more inflationary pressure.
    """
    components = []
    weights = []

    mapping = {
        "cpi_yoy":           (+1, 2.0),
        "pce_yoy":           (+1, 1.5),
        "ppi_yoy":           (+1, 1.0),
        "breakeven_10y_zscore": (+1, 1.5),
    }

    for col, (direction, weight) in mapping.items():
        if col in features.columns:
            s = features[col] * direction
            s = s.clip(-3, 3)
            components.append(s)
            weights.append(weight)

    if not components:
        return pd.Series(0.0, index=features.index, name="inflation_score")

    weighted = sum(c * w for c, w in zip(components, weights)) / sum(weights)
    return np.tanh(weighted).rename("inflation_score")


def compute_monetary_score(features: pd.DataFrame) -> pd.Series:
    """
    Monetary conditions score: -1 (tight/inverted) to +1 (loose/steep).
    """
    components = []
    weights = []

    mapping = {
        "curve_10y2y":       (+1, 2.0),  # steep curve = loose = bullish
        "curve_10y3m":       (+1, 1.5),
        "fed_funds_zscore":  (-1, 1.0),  # high rates = tight = bearish
    }

    for col, (direction, weight) in mapping.items():
        if col in features.columns:
            s = features[col] * direction
            s = s.clip(-3, 3)
            components.append(s)
            weights.append(weight)

    if not components:
        return pd.Series(0.0, index=features.index, name="monetary_score")

    weighted = sum(c * w for c, w in zip(components, weights)) / sum(weights)
    return np.tanh(weighted).rename("monetary_score")


def build_macro_features(use_cache: bool = False) -> pd.DataFrame:
    """
    Full pipeline: fetch → engineer → score → save.

    Returns DataFrame with raw features + growth_score, inflation_score,
    monetary_score columns.
    """
    cache_path = DATA_DIR / "macro_features.parquet"
    if use_cache and cache_path.exists():
        print("  Loading cached macro features...")
        return pd.read_parquet(cache_path)

    raw = fetch_all()
    if raw.empty:
        raise RuntimeError("No macro data fetched — check network")

    features = engineer_features(raw)
    features["growth_score"] = compute_growth_score(features)
    features["inflation_score"] = compute_inflation_score(features)
    features["monetary_score"] = compute_monetary_score(features)

    features.to_parquet(cache_path)
    print(f"\n  Saved macro features: {features.shape} -> {cache_path}")
    return features


if __name__ == "__main__":
    print("="*60)
    print("  V2 Macro Regime Features")
    print("="*60 + "\n")

    features = build_macro_features()

    print(f"\n  Shape: {features.shape}")
    print(f"  Date range: {features.index.min().date()} -> {features.index.max().date()}")

    scores = features[["growth_score", "inflation_score", "monetary_score"]].dropna()
    print(f"\n  Composite scores (last 5 rows):")
    print(scores.tail().round(3).to_string())

    print(f"\n  Score statistics:")
    print(scores.describe().round(3).to_string())
