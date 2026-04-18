"""
feature_research.py
-------------------
Information Coefficient (IC) analysis for all engineered features.

What is IC?
───────────
IC (Information Coefficient) is the Spearman rank correlation between a
feature value at time T and the actual forward return at T+N.  It measures
how predictive a feature is — an IC of 0 means the feature is random noise;
an IC of 1.0 means it perfectly predicts the direction of future returns.

Typical quant benchmarks:
  |IC| < 0.02  → weak / noise
  |IC| 0.02–0.05 → marginal predictive power
  |IC| > 0.05  → useful signal (large quant funds often act on IC this small
                  when combined with many factors in a composite model)

IC Information Ratio (IC IR = mean IC / std IC) is the Sharpe ratio of the
IC signal.  Higher = more consistent predictive signal over time.
  IC IR < 0.3  → inconsistent
  IC IR 0.3–0.5 → acceptable
  IC IR > 0.5  → strong / institutional-grade

Method
──────
For each ticker and each feature column:
  1. Compute forward 5-day return at each date T: fwd[T] = Close[T+5]/Close[T] - 1
  2. Convert both feature and fwd_return to percentile ranks (rank IC approximation)
  3. Compute rolling 252-day Pearson correlation of those ranks (≈ rolling Spearman IC)
  4. Pool the IC time series across all tickers to get aggregate statistics

Note on price-level features:
  bb_middle, bb_upper, bb_lower, and obv are cumulative or level features.
  Their IC may appear inflated due to trend autocorrelation rather than
  genuine predictive alpha.  Prefer their normalised variants:
  bb_pct_b, bb_bandwidth, obv_zscore.

Input / output
──────────────
  Reads:  data/features/{TICKER}.parquet  (from feature_engineering.py)
  Writes: data/research/feature_ic.parquet

Usage
─────
  python -m v1.pipeline.feature_research   # run as module
  python pipeline/feature_research.py   # run as script
"""

import sys
from pathlib import Path

# ── Allow running as a script from the repo root ───────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from v1.pipeline.data_pipeline import TICKER_LIST, ASSET_CLASS

# ── Config ─────────────────────────────────────────────────────────────────────
FEATURE_DIR  = Path("data/v1/features")
RESEARCH_DIR = Path("data/v1/research")
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

FWD_DAYS  = 5    # forward return horizon (trading days)
IC_WINDOW = 252  # rolling window for IC computation (1 trading year)

# Raw OHLCV columns — not engineered features, excluded from IC analysis
_OHLCV = {"Open", "High", "Low", "Close", "Volume"}

# Price-level features: their IC can be inflated by trend autocorrelation.
# Flag them in the output rather than removing them — useful for diagnosis.
_LEVEL_FEATURES = {"bb_middle", "bb_upper", "bb_lower", "obv"}


# ── Core computation ───────────────────────────────────────────────────────────

def _load_features(ticker: str) -> "pd.DataFrame | None":
    path = FEATURE_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _rolling_rank_ic(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each engineered feature column in df, compute a rolling 252-day rank IC
    against the 5-day forward return.

    Returns a DataFrame of IC time series (index = dates, columns = features).
    The first IC_WINDOW rows will be NaN (warm-up).
    """
    # Forward 5-day return aligned to signal date T (no look-ahead):
    # fwd[T] = Close[T+5] / Close[T] - 1  →  shift(-FWD_DAYS) after pct_change
    fwd       = df["Close"].pct_change(FWD_DAYS).shift(-FWD_DAYS)
    fwd_rank  = fwd.rank(pct=True)           # percentile rank of forward returns

    feature_cols = [c for c in df.columns if c not in _OHLCV]

    ic_dict = {}
    for feat in feature_cols:
        feat_rank      = df[feat].rank(pct=True)
        ic_dict[feat]  = feat_rank.rolling(IC_WINDOW).corr(fwd_rank)

    return pd.DataFrame(ic_dict, index=df.index)


# ── Aggregation ────────────────────────────────────────────────────────────────

def _aggregate_ic(all_ic: dict) -> pd.DataFrame:
    """
    Pool IC time series across all tickers and compute summary statistics.

    Args:
        all_ic: dict mapping feature_name -> list of IC Series (one per ticker).

    Returns:
        DataFrame with columns: feature, mean_ic, std_ic, ic_ir, pct_positive,
        n_obs, level_feature.  Sorted by ic_ir descending.
    """
    rows = []
    for feat, ic_list in all_ic.items():
        if not ic_list:
            continue
        pooled   = pd.concat(ic_list).dropna()
        if len(pooled) < 10:
            continue
        mean_ic  = pooled.mean()
        std_ic   = pooled.std()
        ic_ir    = mean_ic / std_ic if std_ic > 0 else 0.0
        pct_pos  = (pooled > 0).mean()

        rows.append({
            "feature"      : feat,
            "mean_ic"      : round(mean_ic, 5),
            "std_ic"       : round(std_ic, 5),
            "ic_ir"        : round(ic_ir, 4),
            "pct_positive" : round(pct_pos, 4),
            "n_obs"        : int(len(pooled)),
            "level_feature": feat in _LEVEL_FEATURES,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("ic_ir", ascending=False)
        .reset_index(drop=True)
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Feature IC Analysis")
    print(f"  Forward return horizon : {FWD_DAYS} trading days")
    print(f"  Rolling IC window      : {IC_WINDOW} trading days (1 year)")
    print(f"  Universe               : {len(TICKER_LIST)} tickers")
    print("=" * 70)

    all_ic: dict[str, list] = {}   # feature -> list of IC Series

    for ticker in TICKER_LIST:
        df = _load_features(ticker)
        if df is None:
            print(f"  {ticker:<8} SKIPPED (feature parquet not found — run python run.py first)")
            continue

        ic_df = _rolling_rank_ic(df)
        asset = ASSET_CLASS.get(ticker, "?")
        print(f"  {ticker:<8} ({asset:<14})  {len(df):>5} rows  ->  IC computed for {len(ic_df.columns)} features")

        for feat, series in ic_df.items():
            all_ic.setdefault(feat, []).append(series.dropna())

    if not all_ic:
        print("\nNo data loaded — cannot produce IC table.")
        return

    # Aggregate and save
    results  = _aggregate_ic(all_ic)
    out_path = RESEARCH_DIR / "feature_ic.parquet"
    results.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"\nResults saved  ->  {out_path}")

    # ── Print ranked table ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(
        f"{'#':<4} {'Feature':<22} {'Mean IC':>8} {'Std IC':>8} "
        f"{'IC IR':>8} {'% Pos':>7} {'N':>7}  {'Note'}"
    )
    print("-" * 78)

    for i, row in results.iterrows():
        note = "[level]" if row["level_feature"] else ""
        print(
            f"{i+1:<4} {row['feature']:<22} {row['mean_ic']:>8.4f} {row['std_ic']:>8.4f} "
            f"{row['ic_ir']:>8.4f} {row['pct_positive']:>7.1%} {row['n_obs']:>7}  {note}"
        )

    print("=" * 78)

    # ── Summary ───────────────────────────────────────────────────────────────
    clean = results[~results["level_feature"]]   # exclude potentially inflated features
    n_strong    = (clean["ic_ir"].abs() > 0.5).sum()
    n_moderate  = ((clean["ic_ir"].abs() >= 0.3) & (clean["ic_ir"].abs() <= 0.5)).sum()
    n_weak      = (clean["ic_ir"].abs() < 0.3).sum()

    print(f"\nSummary (excluding {_LEVEL_FEATURES & set(results['feature'])} level features):")
    print(f"  Strong   (|IC IR| > 0.50): {n_strong}")
    print(f"  Moderate (|IC IR| 0.3–0.5): {n_moderate}")
    print(f"  Weak     (|IC IR| < 0.30):  {n_weak}")

    print("\n[level] = price-level feature; IC may be inflated by trend autocorrelation.")
    print("Prefer normalised variants: bb_pct_b, bb_bandwidth, obv_zscore.\n")


if __name__ == "__main__":
    main()
