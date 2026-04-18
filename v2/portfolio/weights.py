"""
v2/portfolio.py
---------------
Probability-weighted regime blending for portfolio construction.

Core Logic
──────────
  1. Load regime probabilities (6 values summing to 1.0 per day)
  2. For each regime, look up target allocation (regime_allocations.py)
  3. Blend: final_weight[ticker] = sum(prob[regime] * target[regime][ticker])
  4. Apply constraints: single-ETF cap (30%), asset-class cap (60%)
  5. Apply momentum overlay (kill persistent losers)
  6. Normalize to sum to 1.0
  7. Resample to monthly rebalance dates

Output
──────
  data/v2/results/portfolio_weights.parquet  — monthly weights
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from v2.portfolio.regime_allocations import (
    REGIME_ALLOCATIONS, REGIME_NAMES, get_all_tickers,
)
from v2.pipeline.data_pipeline import get_asset_class_map

DATA_DIR = Path("data/v2/results")
DATA_DIR.mkdir(parents=True, exist_ok=True)

REGIME_DIR = Path("data/v2/regime_features")

MAX_SINGLE_ETF = 0.30
MAX_ASSET_CLASS = 0.60


def blend_regime_allocations(probs: pd.Series) -> dict[str, float]:
    """
    Blend regime target allocations weighted by regime probabilities.

    Args:
        probs: Series with index = regime column names
               (expansion, slowdown, recession, recovery, stagflation, late_cycle)

    Returns:
        dict of ticker → blended weight
    """
    regime_name_to_id = {
        "expansion": 0, "slowdown": 1, "recession": 2,
        "recovery": 3, "stagflation": 4, "late_cycle": 5,
    }

    all_tickers = get_all_tickers()
    blended = {t: 0.0 for t in all_tickers}

    for regime_name, regime_id in regime_name_to_id.items():
        prob = probs.get(regime_name, 0.0)
        if prob <= 0 or np.isnan(prob):
            continue
        alloc = REGIME_ALLOCATIONS[regime_id]
        for ticker, weight in alloc.items():
            blended[ticker] += prob * weight

    return blended


def apply_constraints(
    weights: dict[str, float],
    max_single: float = MAX_SINGLE_ETF,
    max_ac: float = MAX_ASSET_CLASS,
    cash_ticker: str = "SHY",
) -> dict[str, float]:
    """
    Enforce position caps and asset-class caps.

    Excess weight is redistributed proportionally to remaining positions,
    with overflow going to cash (SHY).
    """
    ac_map = get_asset_class_map()

    # Pass 1: cap individual ETFs
    excess = 0.0
    for ticker in weights:
        if weights[ticker] > max_single:
            excess += weights[ticker] - max_single
            weights[ticker] = max_single

    # Distribute excess proportionally
    if excess > 0:
        uncapped = {t: w for t, w in weights.items() if w < max_single and w > 0}
        total_uncapped = sum(uncapped.values())
        if total_uncapped > 0:
            for t in uncapped:
                weights[t] += excess * (uncapped[t] / total_uncapped)
        else:
            weights[cash_ticker] = weights.get(cash_ticker, 0) + excess

    # Pass 2: cap asset classes
    for _ in range(3):  # iterate to convergence
        ac_totals = {}
        for ticker, weight in weights.items():
            ac = ac_map.get(ticker, "other")
            ac_totals[ac] = ac_totals.get(ac, 0) + weight

        excess = 0.0
        for ac, total in ac_totals.items():
            if total > max_ac:
                # Scale down all tickers in this asset class
                scale = max_ac / total
                for ticker in weights:
                    if ac_map.get(ticker, "other") == ac:
                        removed = weights[ticker] * (1 - scale)
                        weights[ticker] *= scale
                        excess += removed

        if excess > 0:
            weights[cash_ticker] = weights.get(cash_ticker, 0) + excess

    return weights


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to sum to 1.0, removing zero/negative weights."""
    total = sum(max(0, w) for w in weights.values())
    if total <= 0:
        return {"SHY": 1.0}
    return {t: max(0, w) / total for t, w in weights.items() if w > 1e-6}


def build_portfolio_weights(
    probs: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    apply_momentum: bool = True,
    use_cache: bool = False,
) -> pd.DataFrame:
    """
    Build monthly portfolio weights from regime probabilities.

    Args:
        probs: regime probability DataFrame (daily)
        prices: ETF price DataFrame (for momentum overlay)
        apply_momentum: whether to apply cross-asset momentum filter
        use_cache: load cached results if available

    Returns:
        Monthly-frequency DataFrame of portfolio weights.
    """
    cache_path = DATA_DIR / "portfolio_weights.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    if probs is None:
        probs = pd.read_parquet(REGIME_DIR / "regime_probabilities.parquet")

    # Resample to month-end (rebalance dates)
    monthly_probs = probs.resample("ME").last().dropna()

    all_tickers = get_all_tickers()
    weights_data = []

    for date in monthly_probs.index:
        prob_row = monthly_probs.loc[date]
        blended = blend_regime_allocations(prob_row)
        constrained = apply_constraints(blended.copy())
        normalized = normalize_weights(constrained)

        row = {t: normalized.get(t, 0.0) for t in all_tickers}
        row["Date"] = date
        weights_data.append(row)

    weights_df = pd.DataFrame(weights_data).set_index("Date")

    # Apply momentum overlay if prices available
    if apply_momentum and prices is not None:
        from v2.portfolio.momentum_overlay import apply_momentum_overlay_df
        weights_df = apply_momentum_overlay_df(weights_df, prices)
        # Re-normalize after momentum adjustments
        row_sums = weights_df.sum(axis=1)
        weights_df = weights_df.div(row_sums, axis=0)

    weights_df.to_parquet(cache_path)
    print(f"  Saved portfolio weights: {weights_df.shape} -> {cache_path}")
    return weights_df


if __name__ == "__main__":
    print("="*60)
    print("  V2 Portfolio Construction")
    print("="*60 + "\n")

    weights = build_portfolio_weights(apply_momentum=False)

    print(f"\n  Shape: {weights.shape}")
    print(f"  Date range: {weights.index.min().date()} -> {weights.index.max().date()}")
    print(f"  Months: {len(weights)}")

    # Show latest allocation
    latest = weights.iloc[-1].sort_values(ascending=False)
    print(f"\n  Latest allocation ({weights.index[-1].date()}):")
    for ticker, weight in latest.items():
        if weight > 0.01:
            print(f"    {ticker:5s} {weight:6.1%}")

    # Summary stats
    print(f"\n  Average top-5 holdings concentration: "
          f"{weights.apply(lambda r: r.nlargest(5).sum(), axis=1).mean():.1%}")
    print(f"  Average non-zero positions: "
          f"{(weights > 0.01).sum(axis=1).mean():.0f}")
