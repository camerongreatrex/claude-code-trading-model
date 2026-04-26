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
MAX_ASSET_CLASS = 0.75   # relaxed from 0.60 — lets expansion regime run hotter on equities

# V1-style safe-haven floor adapted to the V2 universe. VGSH/DBMF/WTMF aren't
# in V2's 28-ETF set, so we use SHY (short-duration treasuries) and GLD (gold)
# as crisis-hedge floors that apply regardless of regime. Total 5% drag in
# pure risk-on regimes; already exceeded in bearish regimes so no effect.
SAFE_HAVEN_FLOOR = {
    "SHY": 0.03,   # cash proxy floor; GLD floor tested but cost too much drag
}

# Low-quality regime override. When dominant regime (by probability) is one of
# these, blend the regime's target with a pure defensive basket to cut
# drawdowns. Calibrated from attribution: Late Cycle and Recovery historically
# produce Sharpe < 0.4. Each regime gets its own basket and blend strength
# because they fail differently — Late Cycle drifts sideways with creeping
# correlation, Recovery whipsaws hard.
LOW_QUALITY_REGIMES = {"late_cycle", "recovery"}
# Per-regime blend ceilings/floors. Late Cycle gets the heaviest defensive
# tilt because it's the largest drag (20.9% of time × 0.34 Sharpe).
LATE_CYCLE_BLEND_MAX = 0.70    # max prob-weighted tilt toward late-cycle basket
LATE_CYCLE_BLEND_MIN = 0.45    # floor when late_cycle is dominant regime
RECOVERY_BLEND_MAX = 0.55
RECOVERY_BLEND_MIN = 0.30
# Hard cap on combined defensive tilt across both regimes; keeps the regime
# book contributing at least (1 - COMBINED_OVERRIDE_CAP) of the position.
COMBINED_OVERRIDE_CAP = 0.80
# Late-cycle-specific basket — heavy gold + intermediate-duration treasuries +
# vol-selling / low-vol equity for the small risky sliver. Late cycle is
# usually choppy/sideways: PBP (buy-write) harvests the elevated vol premium,
# USMV reduces beta below SPY. Pre-2011 USMV/PBP weights act like cash drag,
# acceptable trade-off for post-2011 lift.
LATE_CYCLE_BASKET = {
    "SHY": 0.30,
    "IEF": 0.25,
    "TLT": 0.15,
    "GLD": 0.20,
    "TIP": 0.05,
    "XLP": 0.03,   # consumer staples — quality defensive equity sliver
    "XLU": 0.02,   # utilities
}
# Recovery basket — heavier cash to ride out classifier noise, less duration
# (rates often rising into recovery so duration risks loss).
RECOVERY_BASKET = {
    "SHY": 0.55,
    "IEF": 0.15,
    "GLD": 0.20,
    "TLT": 0.05,
    "TIP": 0.05,
}
# Backwards-compatible alias used by older tests/refs
DEFENSIVE_BASKET = LATE_CYCLE_BASKET


def apply_low_quality_override(
    weights: dict[str, float],
    probs: pd.Series,
) -> dict[str, float]:
    """
    Probability-weighted defensive tilt. Each low-quality regime contributes a
    fractional tilt toward its own basket proportional to its current
    probability — regardless of whether it's the dominant regime. This lets
    the strategy de-risk gradually as classifier suspicion of late_cycle
    rises, rather than waiting for a hard idxmax flip.

        tilt_i = clip(prob_i, 0, BLEND_MAX_i)        # per regime
        total_tilt = clip(sum(tilt_i), 0, COMBINED_CAP)
        weights_new = (1-total_tilt)*orig + sum(tilt_i * basket_i) / total_tilt

    Late Cycle gets the heaviest cap because it's the largest drag (20.9% of
    time × 0.34 Sharpe). Recovery is rarer but more violent (-1.1 Sharpe).
    """
    if len(probs) == 0:
        return weights

    p_late = max(0.0, float(probs.get("late_cycle", 0.0)))
    p_rec  = max(0.0, float(probs.get("recovery", 0.0)))

    # Scale each regime's contribution: its raw prob, clipped at its max blend
    tilt_late = min(p_late, LATE_CYCLE_BLEND_MAX)
    tilt_rec  = min(p_rec,  RECOVERY_BLEND_MAX)
    total_tilt = min(tilt_late + tilt_rec, COMBINED_OVERRIDE_CAP)
    if total_tilt <= 1e-9:
        return weights

    # Apply per-regime floor: if either is the dominant regime, lift its tilt
    # to its MIN floor so weak-but-decisive signals still trigger meaningful
    # de-risking.
    dominant = probs.idxmax()
    if dominant == "late_cycle":
        tilt_late = max(tilt_late, LATE_CYCLE_BLEND_MIN)
    elif dominant == "recovery":
        tilt_rec = max(tilt_rec, RECOVERY_BLEND_MIN)
    total_tilt = min(tilt_late + tilt_rec, COMBINED_OVERRIDE_CAP)

    # Compose blended defensive target weighted by per-regime tilts
    rebalance = total_tilt / (tilt_late + tilt_rec) if (tilt_late + tilt_rec) > 0 else 0.0
    tilt_late *= rebalance
    tilt_rec  *= rebalance

    all_tickers = set(weights.keys()) | set(LATE_CYCLE_BASKET.keys()) | set(RECOVERY_BASKET.keys())
    blended = {}
    for t in all_tickers:
        orig = weights.get(t, 0.0)
        def_w = (
            tilt_late * LATE_CYCLE_BASKET.get(t, 0.0)
            + tilt_rec * RECOVERY_BASKET.get(t, 0.0)
        )
        blended[t] = (1 - total_tilt) * orig + def_w
    total = sum(blended.values())
    if total > 0:
        blended = {t: v / total for t, v in blended.items()}
    return blended


def apply_safe_haven_floor(
    weights: dict[str, float],
    floor_map: dict[str, float] = SAFE_HAVEN_FLOOR,
) -> dict[str, float]:
    """
    Enforce minimum allocations for safe-haven tickers, scaling risk assets
    down proportionally to fund any lift. If the regime already allocates
    more than the floor, the ticker is left alone.
    """
    adjusted = dict(weights)
    need = 0.0
    for t, min_w in floor_map.items():
        cur = adjusted.get(t, 0.0)
        if cur < min_w:
            need += (min_w - cur)
            adjusted[t] = min_w

    if need <= 0:
        return adjusted

    # Scale every non-floor ticker down proportionally to fund the lift
    floor_keys = set(floor_map.keys())
    donor_total = sum(w for t, w in adjusted.items() if t not in floor_keys and w > 0)
    if donor_total <= 0:
        return adjusted
    scale = max(0.0, 1.0 - need / donor_total)
    for t in adjusted:
        if t not in floor_keys and adjusted[t] > 0:
            adjusted[t] *= scale
    return adjusted


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
    use_sharpe_weighted: bool = True,
    labels: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build monthly portfolio weights from regime probabilities.

    Args:
        probs: regime probability DataFrame (daily)
        prices: ETF price DataFrame (for momentum overlay / Sharpe calc)
        apply_momentum: whether to apply cross-asset momentum filter
        use_cache: load cached results if available
        use_sharpe_weighted: if True, use walk-forward regime-conditioned
            Sharpe-weighted targets (QUANTT-style). If False, use hand-tuned
            REGIME_ALLOCATIONS only.
        labels: monthly or daily regime labels (int 0-5). Required when
            use_sharpe_weighted=True.

    Returns:
        Monthly-frequency DataFrame of portfolio weights.
    """
    cache_path = DATA_DIR / "portfolio_weights.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    if probs is None:
        probs = pd.read_parquet(REGIME_DIR / "regime_probabilities.parquet")

    # Monthly rebalance (month-end). Tested faster cadences with rigorous
    # lookahead-corrected returns; gains were marginal once the bias was
    # removed (monthly→weekly: ~+0.05 Sharpe at +25 bps cost). Monthly
    # remains the production default for simplicity and lower turnover.
    monthly_probs = probs.resample("ME").last().dropna()

    all_tickers = get_all_tickers()

    # ── Data-driven Sharpe-weighted targets (QUANTT) ─────────────────────
    sharpe_targets = None
    if use_sharpe_weighted and prices is not None:
        from v2.portfolio.sharpe_weighted import build_sharpe_weighted_targets
        if labels is None:
            labels = pd.read_parquet(REGIME_DIR / "regime_labels.parquet")["regime"]
        monthly_labels = labels.resample("ME").last().dropna()
        # Align price columns to the union of universe (tickers present in prices)
        sharpe_targets = build_sharpe_weighted_targets(
            prices=prices,
            monthly_probs=monthly_probs,
            monthly_labels=monthly_labels,
        )

    weights_data = []

    for date in monthly_probs.index:
        prob_row = monthly_probs.loc[date]

        if sharpe_targets is not None and date in sharpe_targets.index:
            blended = sharpe_targets.loc[date].to_dict()
        else:
            blended = blend_regime_allocations(prob_row)

        # Low-quality regime override: blend with defensive basket when the
        # dominant regime has historically produced poor risk-adjusted returns.
        blended = apply_low_quality_override(blended, prob_row)

        # V1-style safe-haven floor before constraints/normalization
        floored = apply_safe_haven_floor(blended)
        constrained = apply_constraints(floored)
        normalized = normalize_weights(constrained)

        # Union the ticker universe with hand-tuned set so cash/fallback works
        full_tickers = set(all_tickers) | set(normalized.keys())
        row = {t: normalized.get(t, 0.0) for t in full_tickers}
        row["Date"] = date
        weights_data.append(row)

    weights_df = pd.DataFrame(weights_data).set_index("Date").fillna(0.0)

    # Cross-sectional momentum sleeve — small (10%) blend, picks top-N risky
    # assets by trailing 12-1 return.
    if prices is not None:
        from v2.portfolio.cross_momentum import (
            build_cross_momentum_sleeve, blend_books,
        )
        mom_sleeve = build_cross_momentum_sleeve(prices, monthly_probs)
        weights_df = blend_books(weights_df, mom_sleeve, momentum_weight=0.10)

    # Managed-futures (CTA) trend sleeve — pure time-series momentum across a
    # cross-asset universe. Long if instrument is above 200d MA AND has
    # positive 12m return; cash otherwise. Vol-scaled per instrument. Adds
    # crisis-period diversification (long bonds in 2008, long commodities
    # in 2022) that the regime classifier tends to lag.
    if prices is not None:
        from v2.portfolio.managed_futures import build_managed_futures_sleeve
        mf_sleeve = build_managed_futures_sleeve(prices, monthly_probs.index)
        weights_df = blend_books(weights_df, mf_sleeve, momentum_weight=0.20)

    # Apply momentum overlay if prices available
    if apply_momentum and prices is not None:
        from v2.portfolio.momentum_overlay import apply_momentum_overlay_df
        weights_df = apply_momentum_overlay_df(weights_df, prices)
        # Re-normalize after momentum adjustments
        row_sums = weights_df.sum(axis=1)
        weights_df = weights_df.div(row_sums, axis=0)

    # Asset-level MA-based trend filter — zero out risky assets in confirmed
    # downtrend, half-weight during choppy periods. Fixed income/currency exempt.
    if prices is not None:
        from v2.risk.trend_filter import apply_trend_filter_df
        weights_df = apply_trend_filter_df(weights_df, prices)

    # V1-style RSI entry filter — delay new exposure when tickers are overbought
    if prices is not None:
        from v2.risk.entry_filter import apply_rsi_entry_filter_df
        weights_df = apply_rsi_entry_filter_df(weights_df, prices)

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
