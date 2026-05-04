"""
feature_research.py — IC (Spearman rank correlation between feature[T] and
fwd return[T+N]) analysis for engineered features.

Benchmarks: |IC|<0.02 noise, 0.02–0.05 marginal, >0.05 useful.
IC IR (= mean/std IC) Sharpe of IC: <0.3 inconsistent, 0.3–0.5 ok, >0.5 strong.

Method: per ticker × feature, rank-IC via 252d rolling Pearson on percentile
ranks of feature vs fwd 5d return; pool across tickers.
Level features (bb_middle/upper/lower, obv): IC may be inflated by trend
autocorrelation — prefer bb_pct_b, bb_bandwidth, obv_zscore.

Reads data/features/{T}.parquet → writes data/research/feature_ic.parquet.
Run: `python -m v1.pipeline.feature_research`.
"""

import sys
from pathlib import Path

# ── Allow running as script from repo root ───────────────────────────────────
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

FWD_DAYS  = 5    # fwd return horizon (days)
IC_WINDOW = 252  # rolling IC window (1y)

# Raw OHLCV — excluded from IC
_OHLCV = {"Open", "High", "Low", "Close", "Volume"}

# Price-level features — IC inflated by trend autocorr; flagged not removed
_LEVEL_FEATURES = {"bb_middle", "bb_upper", "bb_lower", "obv"}


# ── Core computation ───────────────────────────────────────────────────────────

def _load_features(ticker: str) -> "pd.DataFrame | None":
    path = FEATURE_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _rolling_rank_ic(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature rolling 252d rank IC vs 5d fwd return.
    Returns DataFrame[dates × features]; first IC_WINDOW rows NaN."""
    # fwd[T] = Close[T+5]/Close[T]-1 (shift(-FWD_DAYS) after pct_change, no look-ahead)
    fwd       = df["Close"].pct_change(FWD_DAYS).shift(-FWD_DAYS)
    fwd_rank  = fwd.rank(pct=True)

    feature_cols = [c for c in df.columns if c not in _OHLCV]

    ic_dict = {}
    for feat in feature_cols:
        feat_rank      = df[feat].rank(pct=True)
        ic_dict[feat]  = feat_rank.rolling(IC_WINDOW).corr(fwd_rank)

    return pd.DataFrame(ic_dict, index=df.index)


# ── Aggregation ────────────────────────────────────────────────────────────────

def _aggregate_ic(all_ic: dict) -> pd.DataFrame:
    """Pool IC across tickers → summary DataFrame [feature, mean_ic, std_ic,
    ic_ir, pct_positive, n_obs, level_feature], sorted by ic_ir desc."""
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

    all_ic: dict[str, list] = {}

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

    results  = _aggregate_ic(all_ic)
    out_path = RESEARCH_DIR / "feature_ic.parquet"
    results.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"\nResults saved  ->  {out_path}")

    # ── Ranked table ───────────────────────────────────────────────────────
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

    # ── Summary (excludes potentially inflated level features) ─────────────
    clean = results[~results["level_feature"]]
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
