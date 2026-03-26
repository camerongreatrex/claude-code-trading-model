"""
fred_features.py
────────────────
Fetches free FRED indicators and engineers daily macro features.

Uses the same direct FRED CSV endpoint as macro_features.fetch_yield_curve() —
no API key, no login, no pandas-datareader required.

Indicators
──────────
  BAMLH0A0HYM2  — ICE BofA US High Yield OAS (credit spread, risk-off signal)
  BAMLC0A0CM    — ICE BofA Investment Grade Corporate OAS
  UMCSENT       — University of Michigan Consumer Sentiment (monthly, ffill)
  ICSA          — Initial Jobless Claims (weekly, leading recession indicator)
  IPMAN         — Industrial Production: Manufacturing (proxy for ISM PMI)
  DTWEXBGS      — US Dollar Broad Trade-Weighted Index
  T10YIE        — 10-Year Breakeven Inflation Rate
  VIXCLS        — VIX daily close (FRED series, cross-check vs yfinance ^VIX)

For each indicator:
  {name}_4wk_chg   — 4-week (20 business day) change
  {name}_zscore    — z-score over rolling 252 days
  {name}_regime    — 1 if above rolling 252-day median, 0 if below

Output
──────
  data/macro/fred_features.parquet

Consumed by
───────────
  macro_features.compute_macro_features() — merged into fred_macro_score
"""

import io
import numpy as np
import pandas as pd
import requests
from pathlib import Path

MACRO_DIR = Path("data/macro")
MACRO_DIR.mkdir(parents=True, exist_ok=True)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# (fred_id, short_name, description)
SERIES = [
    ("BAMLH0A0HYM2", "hy_oas",        "ICE BofA US High Yield OAS"),
    ("BAMLC0A0CM",    "ig_oas",        "ICE BofA IG Corporate OAS"),
    ("UMCSENT",       "sentiment",     "UMich Consumer Sentiment"),
    ("ICSA",          "claims",        "Initial Jobless Claims"),
    ("IPMAN",         "ism_pmi",       "Industrial Production: Manufacturing"),
    ("DTWEXBGS",      "usd_broad",     "USD Broad Trade-Weighted Index"),
    ("T10YIE",        "breakeven_10y", "10Y Breakeven Inflation"),
    ("VIXCLS",        "vix_fred",      "VIX (FRED daily)"),
]

# Direction for composite score: +1 = "above median is bullish",
#                                 -1 = "above median is bearish"
DIRECTION = {
    "hy_oas":        -1,  # wider spreads = risk-off = bearish
    "ig_oas":        -1,  # wider spreads = risk-off = bearish
    "sentiment":     +1,  # higher sentiment = bullish
    "claims":        -1,  # more claims = worse economy = bearish
    "ism_pmi":       +1,  # higher PMI = expansion = bullish
    "usd_broad":     -1,  # stronger USD = headwind for equities
    "breakeven_10y": -1,  # rising inflation expectations = bearish
    "vix_fred":      -1,  # higher VIX = fear = bearish
}


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, name: str) -> pd.Series:
    """
    Fetch a single FRED series via the public CSV endpoint.

    Uses the same URL pattern as macro_features.fetch_yield_curve():
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>

    Returns a business-day-resampled, forward-filled Series named ``name``.
    Weekly/monthly series (ICSA, UMCSENT) are upsampled to daily via ffill.
    """
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
    # Resample to business daily and forward-fill (handles weekly/monthly series)
    df = df.resample("B").last().ffill()
    return df[name].dropna()


def fetch_all() -> pd.DataFrame:
    """Fetch all FRED series and join into a single DataFrame."""
    frames = []
    for series_id, name, desc in SERIES:
        print(f"  Fetching {desc} ({series_id})...")
        try:
            s = fetch_fred_series(series_id, name)
            if not s.empty:
                frames.append(s)
                print(f"    {len(s):,} obs, {s.index.min().date()} → {s.index.max().date()}")
            else:
                print(f"    Empty result — skipped")
        except Exception as e:
            print(f"    FAILED ({e}) — skipped")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).ffill()
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    For each raw indicator compute derived features:
      {col}_4wk_chg  — 20-business-day change (approx 4 calendar weeks)
      {col}_zscore   — rolling 252-day z-score (1 trading year)
      {col}_regime   — 1 if above rolling 252-day median, 0 if below
    """
    result = raw.copy()

    for col in raw.columns:
        # 4-week change
        result[f"{col}_4wk_chg"] = raw[col].diff(20)

        # Rolling 252-day z-score
        roll = raw[col].rolling(252, min_periods=60)
        mu, sigma = roll.mean(), roll.std()
        result[f"{col}_zscore"] = (raw[col] - mu) / sigma

        # Regime dummy: 1 = above rolling 252-day median
        roll_med = raw[col].rolling(252, min_periods=60).median()
        result[f"{col}_regime"] = (raw[col] > roll_med).astype(int)

    return result


def compute_fred_score(features: pd.DataFrame) -> pd.Series:
    """
    Composite fred_macro_score from -1 to +1.

    Each indicator's regime dummy is directionally mapped:
      regime=1 (above median) × direction  → +1 (bullish) or -1 (bearish)
      regime=0 (below median) × -direction → -1 (bearish) or +1 (bullish)

    Averaged across all available indicators, clipped to [-1, +1].
    """
    scores = []
    for col, direction in DIRECTION.items():
        regime_col = f"{col}_regime"
        if regime_col in features.columns:
            # Map regime {0,1} to {-1,+1} then apply direction
            score = direction * (2 * features[regime_col] - 1)
            scores.append(score)

    if not scores:
        return pd.Series(0.0, index=features.index, name="fred_macro_score")

    avg = pd.concat(scores, axis=1).mean(axis=1)
    return avg.clip(-1, 1).rename("fred_macro_score")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Fetching FRED indicators...\n")
    raw = fetch_all()

    if raw.empty:
        print("\nNo FRED data fetched. Check network connection.")
        return

    print(f"\nRaw data: {raw.shape[0]:,} rows × {raw.shape[1]} columns")
    print(f"Date range: {raw.index.min().date()} → {raw.index.max().date()}")

    features = engineer_features(raw)
    features["fred_macro_score"] = compute_fred_score(features)

    out = MACRO_DIR / "fred_features.parquet"
    features.to_parquet(out, engine="pyarrow", compression="snappy")

    print(f"\nFRED features: {features.shape}  →  {out}")
    print(f"\nLast 5 rows (raw levels + score):")
    key_cols = list(raw.columns) + ["fred_macro_score"]
    print(features[key_cols].tail(5).round(3))
    print(f"\nfred_macro_score  range: [{features['fred_macro_score'].min():.2f}, "
          f"{features['fred_macro_score'].max():.2f}]")
    print(f"fred_macro_score  mean:  {features['fred_macro_score'].mean():.3f}")


if __name__ == "__main__":
    main()
