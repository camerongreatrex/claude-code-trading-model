"""
v2/regimes/classifier.py
------------------------
Hybrid macro regime classifier with continuous probabilities.

Six-Regime Framework (Growth x Inflation x Stress)
───────────────────────────────────────────────────
  1. EXPANSION    — growth ↑, inflation normal, stress low
  2. SLOWDOWN     — growth ↓ (but positive), inflation rising
  3. RECESSION    — growth ↓ (negative), stress elevated
  4. RECOVERY     — growth turning ↑ from trough, stress declining
  5. STAGFLATION  — growth ↓, inflation ↑ simultaneously
  6. LATE_CYCLE   — growth ↑ but decelerating, stress building, curve flat/inverted

Differentiators vs QUANTT (4-regime):
  - Separates "normal bull" (Expansion) from "late-cycle stress bull" (Late Cycle)
  - Separates "recession onset" (Recession) from "early recovery" (Recovery)
  - Continuous probabilities (soft classification) — no hard switches
  - Blends realized macro (slow) with market-implied signals (fast)

Classification Method
─────────────────────
  1. Rule-based scoring: map composite scores to regime membership scores
  2. Gaussian mixture smoothing: fit per-regime score distributions
  3. Output: daily probability vector over 6 regimes (sums to 1.0)

Output
──────
  data/v2/regime_features/regime_probabilities.parquet
  data/v2/regime_features/regime_labels.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from enum import IntEnum

DATA_DIR = Path("data/v2/regime_features")
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Regime(IntEnum):
    EXPANSION   = 0
    SLOWDOWN    = 1
    RECESSION   = 2
    RECOVERY    = 3
    STAGFLATION = 4
    LATE_CYCLE  = 5


REGIME_NAMES = {r: r.name.title().replace("_", " ") for r in Regime}
N_REGIMES = len(Regime)


def _load_features() -> pd.DataFrame:
    """Load and merge macro + market features."""
    macro_path = DATA_DIR / "macro_features.parquet"
    market_path = DATA_DIR / "market_features.parquet"

    macro = pd.read_parquet(macro_path)
    market = pd.read_parquet(market_path)

    # Join on date, forward-fill macro (slower frequency) to daily market dates
    combined = macro.join(market, how="outer", rsuffix="_mkt").ffill()
    return combined


def compute_regime_scores(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw membership scores for each regime based on feature values.

    Each regime gets a score based on how well current conditions match its
    defining characteristics. Higher = more consistent with that regime.

    Returns DataFrame with columns: expansion, slowdown, recession, recovery,
    stagflation, late_cycle (raw scores, not yet probabilities).
    """
    scores = pd.DataFrame(index=features.index)

    g = features.get("growth_score", pd.Series(np.nan, index=features.index)).fillna(0)
    i = features.get("inflation_score", pd.Series(np.nan, index=features.index)).fillna(0)
    m = features.get("monetary_score", pd.Series(np.nan, index=features.index)).fillna(0)
    s = features.get("stress_score", pd.Series(np.nan, index=features.index)).fillna(0)

    # Demean to remove long-run bias (growth_score has positive mean ~0.27)
    g = g - g.expanding(min_periods=252).mean()
    i = i - i.expanding(min_periods=252).mean()
    m = m - m.expanding(min_periods=252).mean()

    # Growth acceleration (positive = accelerating)
    g_accel = g.diff(63).fillna(0)  # ~3 month change in growth score

    # ── EXPANSION: strong growth, moderate inflation, low stress, loose money
    scores["expansion"] = (
        2.5 * g.clip(0, 1)              # growth positive, stronger = better
        + 1.0 * (1 - i.abs())           # moderate inflation (near zero best)
        + 1.5 * (-s).clip(0, 2)         # low stress (must be calm)
        + 0.5 * m.clip(0, 1)            # loose monetary (positive)
    )

    # ── SLOWDOWN: positive but declining growth, rising inflation
    scores["slowdown"] = (
        1.0 * g.clip(-0.2, 0.4)         # still weakly positive
        + 2.5 * (-g_accel).clip(0, 1)   # decelerating growth (key signal)
        + 1.5 * i.clip(0, 1)            # rising inflation
        + 1.0 * (-m).clip(0, 1)         # tightening monetary
    )

    # ── RECESSION: negative growth, elevated stress
    scores["recession"] = (
        3.0 * (-g).clip(0, 1)           # negative growth (must be negative)
        + 2.0 * s.clip(0, 2)            # elevated stress
        + 1.0 * (-m).clip(0, 1)         # tight monetary (inverted curve)
        + 0.5 * (-i).clip(0, 1)         # falling inflation (demand collapse)
    )

    # ── RECOVERY: growth improving from depressed base, stress declining
    # Key: requires BOTH growth acceleration AND prior weakness
    g_3m_chg = g.diff(63).fillna(0)
    s_3m_chg = s.diff(63).fillna(0)
    g_below_avg = (-g).clip(0, 1)       # growth still below average
    scores["recovery"] = (
        2.5 * g_3m_chg.clip(0, 1)       # growth improving
        + 1.5 * g_below_avg             # from depressed base (critical gate)
        + 1.5 * (-s_3m_chg).clip(0, 1)  # stress declining
        + 0.5 * m.clip(0, 1)            # loose monetary (Fed easing)
    )

    # ── STAGFLATION: negative/weak growth + high inflation
    scores["stagflation"] = (
        2.0 * (-g).clip(-0.1, 1)        # weak/negative growth
        + 3.0 * i.clip(0.2, 1)          # high inflation (must be elevated, key gate)
        + 0.5 * s.clip(0, 1)            # some stress
        + 0.5 * (-m).clip(0, 1)         # tight monetary
    )

    # ── LATE CYCLE: positive growth but decelerating, curve flat, stress building
    scores["late_cycle"] = (
        1.0 * g.clip(0, 0.5)            # still positive but moderate
        + 2.0 * (-g_accel).clip(0, 1)   # decelerating
        + 2.0 * s.clip(-0.2, 1.5)       # stress building
        + 2.0 * (-m).clip(0, 1)         # flat/inverted curve (key signal)
        + 0.5 * i.clip(0, 0.5)          # some inflation
    )

    return scores


def scores_to_probabilities(scores: pd.DataFrame, temperature: float = 1.5) -> pd.DataFrame:
    """
    Convert raw regime scores to probability distribution using softmax.

    Temperature controls sharpness:
      - Lower = sharper (more confident, closer to hard classification)
      - Higher = softer (more blended, smoother transitions)
      - 1.5 is calibrated for smooth but meaningful regime differentiation

    Returns DataFrame with same columns, each row summing to 1.0.
    """
    # Softmax with temperature
    exp_scores = np.exp(scores / temperature)
    probs = exp_scores.div(exp_scores.sum(axis=1), axis=0)

    # Handle any NaN rows (early period before features are available)
    probs = probs.fillna(1.0 / N_REGIMES)

    return probs


def classify_regimes(
    features: pd.DataFrame | None = None,
    temperature: float = 1.5,
    use_cache: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run the full classification pipeline.

    Args:
        features: pre-loaded features (if None, loads from disk)
        temperature: softmax temperature for probability conversion
        use_cache: if True, load cached results

    Returns:
        (probabilities, labels)
        - probabilities: DataFrame with 6 columns, each row sums to 1.0
        - labels: Series with the dominant regime label per day
    """
    probs_path = DATA_DIR / "regime_probabilities.parquet"
    labels_path = DATA_DIR / "regime_labels.parquet"

    if use_cache and probs_path.exists() and labels_path.exists():
        probs = pd.read_parquet(probs_path)
        labels = pd.read_parquet(labels_path)["regime"]
        return probs, labels

    if features is None:
        features = _load_features()

    # Drop early rows where features are mostly NaN
    min_date = "1997-01-01"  # HY OAS starts 1996, need 1yr lookback
    features = features.loc[min_date:]

    scores = compute_regime_scores(features)
    probs = scores_to_probabilities(scores, temperature=temperature)

    # Smooth probabilities with 5-day EMA to reduce daily noise
    probs = probs.ewm(span=5).mean()
    # Re-normalize after smoothing
    probs = probs.div(probs.sum(axis=1), axis=0)

    # Dominant regime label
    labels = probs.idxmax(axis=1).map({
        "expansion": Regime.EXPANSION,
        "slowdown": Regime.SLOWDOWN,
        "recession": Regime.RECESSION,
        "recovery": Regime.RECOVERY,
        "stagflation": Regime.STAGFLATION,
        "late_cycle": Regime.LATE_CYCLE,
    }).rename("regime")

    # Save
    probs.to_parquet(probs_path)
    labels.to_frame().to_parquet(labels_path)
    print(f"  Saved regime probabilities: {probs.shape} -> {probs_path}")
    print(f"  Saved regime labels: {len(labels)} -> {labels_path}")

    return probs, labels


def regime_summary(probs: pd.DataFrame, labels: pd.Series) -> dict:
    """Compute summary statistics for regime classification."""
    label_names = labels.map(lambda x: REGIME_NAMES.get(Regime(x), str(x)))

    # Regime distribution
    dist = label_names.value_counts(normalize=True).sort_index()

    # Average probability when dominant
    avg_confidence = {}
    for col in probs.columns:
        mask = probs.idxmax(axis=1) == col
        if mask.sum() > 0:
            avg_confidence[col] = probs.loc[mask, col].mean()

    # Transition matrix (monthly)
    monthly_labels = labels.resample("ME").last().dropna()
    transitions = pd.crosstab(
        monthly_labels.shift(1).dropna().map(lambda x: REGIME_NAMES.get(Regime(int(x)), str(x))),
        monthly_labels.iloc[1:].map(lambda x: REGIME_NAMES.get(Regime(int(x)), str(x))),
        normalize="index",
    )

    return {
        "distribution": dist,
        "avg_confidence": avg_confidence,
        "transitions": transitions,
        "n_transitions": (labels.diff() != 0).sum(),
        "avg_regime_duration_months": len(monthly_labels) / max((monthly_labels.diff() != 0).sum(), 1),
    }


if __name__ == "__main__":
    print("="*60)
    print("  V2 Hybrid Regime Classifier")
    print("="*60 + "\n")

    probs, labels = classify_regimes()

    summary = regime_summary(probs, labels)

    print(f"\n  Date range: {probs.index.min().date()} -> {probs.index.max().date()}")
    print(f"  Total observations: {len(probs):,}")

    print(f"\n  Regime Distribution (% of time):")
    for regime, pct in summary["distribution"].items():
        print(f"    {regime:15s} {pct*100:5.1f}%")

    print(f"\n  Average Confidence (prob when dominant):")
    for regime, conf in summary["avg_confidence"].items():
        print(f"    {regime:15s} {conf:.3f}")

    print(f"\n  Total regime transitions: {summary['n_transitions']}")
    print(f"  Avg regime duration: {summary['avg_regime_duration_months']:.1f} months")

    print(f"\n  Current regime probabilities:")
    latest = probs.iloc[-1]
    for col in sorted(latest.index, key=lambda x: latest[x], reverse=True):
        print(f"    {col:15s} {latest[col]*100:5.1f}%")

    print(f"\n  Transition matrix (monthly):")
    print(summary["transitions"].round(2).to_string())
