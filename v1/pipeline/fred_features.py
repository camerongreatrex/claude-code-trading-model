"""
fred_features.py — FRED CSV indicators (no API key) → daily macro features.

Indicators: HY OAS, IG OAS, UMich sentiment, jobless claims, IP manufacturing
(PMI proxy), USD broad index, 10Y breakeven, VIX (cross-check).

Per indicator: {name}_4wk_chg (20bd diff), _zscore (252d), _regime (above 252d median).

Output: data/shared/macro/fred_features.parquet → merged into fred_macro_score
by macro_features.compute_macro_features.
"""

import io
import numpy as np
import pandas as pd
import requests
from pathlib import Path

MACRO_DIR = Path("data/shared/macro")
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

# Composite-score direction: +1 above-median bullish, -1 bearish
DIRECTION = {
    "hy_oas":        -1,  # wider spread = risk-off
    "ig_oas":        -1,
    "sentiment":     +1,
    "claims":        -1,  # more = worse economy
    "ism_pmi":       +1,
    "usd_broad":     -1,  # stronger USD = equity headwind
    "breakeven_10y": -1,  # rising inflation expectations
    "vix_fred":      -1,
}


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, name: str) -> pd.Series:
    """Fetch single FRED series (CSV endpoint). Resampled to business-day
    ffill (handles weekly/monthly upsampling for ICSA/UMCSENT)."""
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
    """Per indicator: _4wk_chg (20bd diff), _zscore (252d), _regime (>252d median)."""
    result = raw.copy()

    for col in raw.columns:
        result[f"{col}_4wk_chg"] = raw[col].diff(20)

        roll = raw[col].rolling(252, min_periods=60)
        mu, sigma = roll.mean(), roll.std()
        result[f"{col}_zscore"] = (raw[col] - mu) / sigma

        roll_med = raw[col].rolling(252, min_periods=60).median()
        result[f"{col}_regime"] = (raw[col] > roll_med).astype(int)

    return result


def compute_fred_score(features: pd.DataFrame) -> pd.Series:
    """Composite fred_macro_score [-1,+1]: per-indicator regime {0,1}→{-1,+1}
    × DIRECTION sign, averaged across available indicators."""
    scores = []
    for col, direction in DIRECTION.items():
        regime_col = f"{col}_regime"
        if regime_col in features.columns:
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
