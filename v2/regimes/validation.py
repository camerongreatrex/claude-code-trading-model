"""
v2/regimes/validation.py
------------------------
Validate regime classification against NBER recession dates and compute
accuracy metrics for the Step 7 gate report.

Validation Steps
────────────────
  1. NBER recession recall: % of NBER recession months correctly classified
     as Recession or Late Cycle (pre-recession stress). Target: > 85%.
  2. NBER expansion precision: % of non-recession periods not classified as
     Recession. Penalizes false recession calls.
  3. Random Forest cross-validation: train RF on features → regime labels,
     report accuracy as a sanity check that regimes are learnable from data.
  4. Regime distribution & transition statistics.
  5. Per-regime equity return attribution (do regimes predict SPY returns?).

NBER Recession Dates (hardcoded — authoritative source)
───────────────────────────────────────────────────────
  https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = Path("data/v2/regime_features")

# ── NBER Recession Dates (peak → trough) ─────────────────────────────────────
# Official NBER dates through 2020 recession
NBER_RECESSIONS = [
    ("1948-11-01", "1949-10-01"),
    ("1953-07-01", "1954-05-01"),
    ("1957-08-01", "1958-04-01"),
    ("1960-04-01", "1961-02-01"),
    ("1969-12-01", "1970-11-01"),
    ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]


def get_nber_recession_dates(start: str = "1997-01-01") -> pd.Series:
    """
    Return a daily boolean Series: True during NBER recessions.
    Business-day frequency aligned with our feature data.
    """
    dates = pd.bdate_range(start, pd.Timestamp.today())
    is_recession = pd.Series(False, index=dates, name="nber_recession")

    for peak, trough in NBER_RECESSIONS:
        peak_dt = pd.Timestamp(peak)
        trough_dt = pd.Timestamp(trough)
        if trough_dt >= pd.Timestamp(start):
            mask = (dates >= peak_dt) & (dates <= trough_dt)
            is_recession[mask] = True

    return is_recession


def validate_nber(labels: pd.Series) -> dict:
    """
    Validate regime labels against NBER recession dates.

    Recession recall: what fraction of NBER recession days are classified
    as Recession (2) or Late Cycle (5) — both are "bad" regimes.

    Expansion precision: what fraction of NBER expansion days are NOT
    classified as Recession.
    """
    nber = get_nber_recession_dates(labels.index.min().strftime("%Y-%m-%d"))

    # Align on common dates
    common = labels.index.intersection(nber.index)
    labels_aligned = labels.loc[common]
    nber_aligned = nber.loc[common]

    # NBER recession days
    rec_mask = nber_aligned
    exp_mask = ~nber_aligned

    n_rec_days = rec_mask.sum()
    n_exp_days = exp_mask.sum()

    if n_rec_days == 0:
        return {"error": "No NBER recession days in sample"}

    # Regime 2 = Recession, Regime 5 = Late Cycle
    # "Correctly detected" = classified as Recession OR Late Cycle
    detected_broad = labels_aligned[rec_mask].isin([2, 5])
    recall_broad = detected_broad.mean()

    # Strict recall: only Recession label
    detected_strict = labels_aligned[rec_mask] == 2
    recall_strict = detected_strict.mean()

    # Expansion precision: non-recession days NOT labeled as Recession
    false_recession = (labels_aligned[exp_mask] == 2)
    expansion_precision = 1.0 - false_recession.mean()

    # Per-recession performance
    per_recession = []
    for peak, trough in NBER_RECESSIONS:
        peak_dt = pd.Timestamp(peak)
        trough_dt = pd.Timestamp(trough)
        mask = (common >= peak_dt) & (common <= trough_dt)
        if mask.sum() == 0:
            continue
        rec_labels = labels_aligned[mask]
        detected = rec_labels.isin([2, 5]).mean()
        dominant = rec_labels.value_counts().idxmax()
        per_recession.append({
            "period": f"{peak[:7]} to {trough[:7]}",
            "days": mask.sum(),
            "recall_broad": detected,
            "dominant_regime": dominant,
        })

    return {
        "n_recession_days": int(n_rec_days),
        "n_expansion_days": int(n_exp_days),
        "recall_broad": float(recall_broad),
        "recall_strict": float(recall_strict),
        "expansion_precision": float(expansion_precision),
        "per_recession": per_recession,
    }


def validate_random_forest(features: pd.DataFrame, labels: pd.Series) -> dict:
    """
    Train a Random Forest on features → regime labels using time-series CV.
    Reports accuracy to verify regimes are learnable (sanity check).
    """
    # Select numeric feature columns (exclude raw levels, use derived)
    feature_cols = [c for c in features.columns if any(
        c.endswith(suffix) for suffix in
        ["_yoy", "_3m_chg", "_6m_chg", "_zscore", "_accel", "_percentile",
         "_20d_chg", "_60d_chg", "growth_score", "inflation_score",
         "monetary_score", "stress_score", "vix_term_ratio", "vix_term_slope",
         "vix_backwardation", "vix_contango", "vix_above_25", "vix_above_30",
         "credit_quality_spread"]
    )]

    # Align features and labels
    common = features.index.intersection(labels.index)
    X = features.loc[common, feature_cols].copy()
    y = labels.loc[common].copy()

    # Drop rows with NaN
    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    if len(X) < 500:
        return {"error": f"Only {len(X)} valid samples — insufficient for RF"}

    # Time-series cross-validation (3 splits)
    tscv = TimeSeriesSplit(n_splits=3)
    accuracies = []
    reports = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        reports.append({
            "fold": fold,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": acc,
            "train_range": f"{X_train.index.min().date()} to {X_train.index.max().date()}",
            "test_range": f"{X_test.index.min().date()} to {X_test.index.max().date()}",
        })

    # Feature importance from last fold
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "fold_results": reports,
        "top_features": importances.head(15).to_dict(),
        "n_features": len(feature_cols),
        "n_samples": len(X),
    }


def validate_regime_returns(labels: pd.Series) -> dict:
    """
    Compute per-regime SPY returns to validate economic meaning.

    Good regimes should show:
      Expansion > 0 (strong positive returns)
      Recession < 0 (negative returns)
      Recovery > 0 (strong positive, mean reversion)
      Late Cycle: mixed but lower than Expansion
    """
    print("  Fetching SPY returns for regime attribution...")
    spy = yf.download("SPY", start=labels.index.min().strftime("%Y-%m-%d"),
                       progress=False, auto_adjust=True)
    if spy.empty:
        return {"error": "Could not fetch SPY data"}

    spy_ret = spy["Close"].squeeze().pct_change().dropna()
    spy_ret.index = pd.to_datetime(spy_ret.index).tz_localize(None)
    spy_ret.index.name = "Date"

    common = labels.index.intersection(spy_ret.index)
    labels_aligned = labels.loc[common]
    spy_aligned = spy_ret.loc[common]

    regime_names = {0: "Expansion", 1: "Slowdown", 2: "Recession",
                    3: "Recovery", 4: "Stagflation", 5: "Late Cycle"}

    results = {}
    for regime_id, name in regime_names.items():
        mask = labels_aligned == regime_id
        if mask.sum() == 0:
            continue
        regime_returns = spy_aligned[mask]
        ann_ret = regime_returns.mean() * 252
        ann_vol = regime_returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        results[name] = {
            "days": int(mask.sum()),
            "pct_time": float(mask.mean() * 100),
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
        }

    return results


def run_full_validation() -> dict:
    """Run all validation steps and return combined results."""
    # Load data
    probs = pd.read_parquet(DATA_DIR / "regime_probabilities.parquet")
    labels = pd.read_parquet(DATA_DIR / "regime_labels.parquet")["regime"]

    # Load features for RF
    macro = pd.read_parquet(DATA_DIR / "macro_features.parquet")
    market = pd.read_parquet(DATA_DIR / "market_features.parquet")
    features = macro.join(market, how="outer", rsuffix="_mkt").ffill()
    features = features.loc["1997-01-01":]

    results = {}

    # 1. NBER validation
    print("\n  [1/3] NBER recession validation...")
    results["nber"] = validate_nber(labels)

    # 2. Random Forest
    print("  [2/3] Random Forest cross-validation...")
    results["random_forest"] = validate_random_forest(features, labels)

    # 3. Per-regime returns
    print("  [3/3] Per-regime SPY return attribution...")
    results["regime_returns"] = validate_regime_returns(labels)

    return results


def print_report(results: dict):
    """Print a formatted validation report."""
    print("\n" + "="*70)
    print("  V2 REGIME CLASSIFICATION — VALIDATION REPORT")
    print("="*70)

    # ── NBER ──────────────────────────────────────────────────────────────
    nber = results["nber"]
    if "error" not in nber:
        print(f"\n  NBER RECESSION VALIDATION")
        print(f"  ─────────────────────────")
        print(f"  Recession days in sample:   {nber['n_recession_days']}")
        print(f"  Expansion days in sample:   {nber['n_expansion_days']}")

        recall = nber["recall_broad"]
        gate = "PASS" if recall >= 0.85 else "FAIL"
        print(f"\n  Recession recall (broad):   {recall*100:.1f}%  [{gate}]  (gate: >85%)")
        print(f"  Recession recall (strict):  {nber['recall_strict']*100:.1f}%")
        print(f"  Expansion precision:        {nber['expansion_precision']*100:.1f}%")

        print(f"\n  Per-recession breakdown:")
        for rec in nber["per_recession"]:
            recall_pct = rec["recall_broad"] * 100
            print(f"    {rec['period']:25s}  {rec['days']:4d} days  "
                  f"recall={recall_pct:5.1f}%  dominant={rec['dominant_regime']}")
    else:
        print(f"\n  NBER: {nber['error']}")

    # ── Random Forest ─────────────────────────────────────────────────────
    rf = results["random_forest"]
    if "error" not in rf:
        print(f"\n  RANDOM FOREST VALIDATION")
        print(f"  ────────────────────────")
        print(f"  Features: {rf['n_features']}, Samples: {rf['n_samples']:,}")
        print(f"  Mean accuracy (3-fold TSCV): {rf['mean_accuracy']*100:.1f}%")

        print(f"\n  Fold results:")
        for fold in rf["fold_results"]:
            print(f"    Fold {fold['fold']}: acc={fold['accuracy']*100:.1f}%  "
                  f"train={fold['train_range']}  test={fold['test_range']}")

        print(f"\n  Top 10 features:")
        for feat, imp in list(rf["top_features"].items())[:10]:
            print(f"    {feat:35s} {imp:.4f}")
    else:
        print(f"\n  RF: {rf['error']}")

    # ── Regime Returns ────────────────────────────────────────────────────
    rr = results["regime_returns"]
    if "error" not in rr:
        print(f"\n  PER-REGIME SPY RETURNS")
        print(f"  ──────────────────────")
        print(f"  {'Regime':15s} {'Days':>6s} {'%Time':>6s} {'AnnRet':>8s} {'AnnVol':>8s} {'Sharpe':>7s}")
        print(f"  {'─'*55}")
        for name, stats in sorted(rr.items(), key=lambda x: x[1]["sharpe"], reverse=True):
            print(f"  {name:15s} {stats['days']:6d} {stats['pct_time']:5.1f}% "
                  f"{stats['ann_return']*100:7.1f}% {stats['ann_vol']*100:7.1f}% "
                  f"{stats['sharpe']:6.2f}")
    else:
        print(f"\n  Returns: {rr['error']}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  {'='*55}")
    print(f"  GATE CHECK SUMMARY")
    print(f"  {'='*55}")

    gates = []
    if "error" not in nber:
        nber_pass = nber["recall_broad"] >= 0.85
        gates.append(("NBER recall > 85%", nber_pass, f"{nber['recall_broad']*100:.1f}%"))
    if "error" not in rf:
        rf_pass = rf["mean_accuracy"] >= 0.75
        gates.append(("RF accuracy > 75%", rf_pass, f"{rf['mean_accuracy']*100:.1f}%"))
    if "error" not in rr:
        # Check economic meaning: Expansion > Recession
        exp_sharpe = rr.get("Expansion", {}).get("sharpe", 0)
        rec_sharpe = rr.get("Recession", {}).get("sharpe", 0)
        econ_pass = exp_sharpe > rec_sharpe
        gates.append(("Expansion Sharpe > Recession Sharpe", econ_pass,
                      f"{exp_sharpe:.2f} vs {rec_sharpe:.2f}"))

    all_pass = all(g[1] for g in gates)
    for name, passed, value in gates:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {value}")

    print(f"\n  Overall: {'ALL GATES PASSED — proceed to Phase 8' if all_pass else 'GATE FAILURE — redesign needed'}")

    return all_pass


if __name__ == "__main__":
    print("="*70)
    print("  V2 Regime Validation")
    print("="*70)

    results = run_full_validation()
    passed = print_report(results)
