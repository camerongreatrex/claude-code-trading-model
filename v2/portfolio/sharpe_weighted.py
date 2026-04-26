"""
v2/portfolio/sharpe_weighted.py
------------------------------
Walk-forward, regime-conditioned Sharpe-weighted ETF allocator.

Replaces (or blends with) the hand-tuned REGIME_ALLOCATIONS targets using
QUANTT's data-driven approach: for each rebalance month and each regime r,
compute every ETF's Sharpe ratio using only historical months where the
regime was r, select top-N by Sharpe, weight proportionally with a 0.01
floor.

At each month t, for each ticker i in regime r:
    Ŝ_i,r = r̄^xs_i,r / σ̂^xs_i,r  × √12    (regime-conditioned history < t)

Top-N selection + proportional weights:
    w_i,r = max(Ŝ_i,r, ε) / Σ_{j ∈ TopN} max(Ŝ_j,r, ε)

Blending with probability vector across regimes:
    target_i(t) = Σ_r P_r(t) · w_i,r(t)

When a regime has < MIN_MONTHS of history, the hand-tuned prior from
regime_allocations.py is used instead (cold-start protection).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from v2.portfolio.regime_allocations import (
    REGIME_ALLOCATIONS,
    get_all_tickers as get_prior_tickers,
)
from v2.portfolio.factor_scoring import composite_factor_score
from v2.risk.covariance import hrp_weights_for_subset

# ── Parameters ───────────────────────────────────────────────────────────────
# Per-regime top-N — our deviation from the paper's uniform N=7. Expansion
# gets one extra holding to widen the equity set; other regimes stay at 5.
# Tighter concentration in recession/stagflation tested worse (stagflation
# Sharpe dropped from 1.15 → 0.96 with N=3 vs N=5).
TOP_N_PER_REGIME = {
    "expansion":   6,
    "slowdown":    5,
    "recession":   5,
    "recovery":    5,
    "stagflation": 5,
    "late_cycle":  5,
}
TOP_N = 5                    # fallback if regime not in dict
FLOOR = 0.01                 # weight floor for selected ETFs
MIN_MONTHS_PER_REGIME = 18   # minimum same-regime history before using data-driven
PRIOR_BLEND = 0.15           # 15% hand-tuned prior + 85% data-driven (stability tax)
PRIOR_BLEND_COLD = 1.00      # 100% prior when history < MIN_MONTHS_PER_REGIME
MIN_TICKER_HISTORY = 12      # ticker must have 12 months of data to be considered
PROB_CONCENTRATION = 1.5     # power > 1 sharpens regime prob vector before blending
# "composite" = our multi-factor (Sharpe + momentum + vol penalty) — tested but
# added DD risk on this 18-year window. Exposed as a knob for future research.
# "sharpe" is the production default.
SCORING_MODE = "sharpe"

# Within-regime weighting after top-N selection. "hrp" uses Hierarchical Risk
# Parity over a Ledoit-Wolf-shrunk covariance — splits correlated clusters
# (e.g. QQQ/IWF/XLK) so the regime book stays diversified even when several
# top-Sharpe names are nearly redundant. "score" reverts to the original
# floor-shifted proportional Sharpe weighting.
WITHIN_REGIME_WEIGHTING = "hrp"
HRP_LOOKBACK_DAYS = 252      # 1y daily returns for covariance estimation
HRP_MAX_WEIGHT = 0.40        # per-asset cap inside the top-N HRP solution
HRP_MIN_FLOOR = 0.05         # minimum weight per selected top-N name post-HRP

REGIME_COLS = ["expansion", "slowdown", "recession", "recovery", "stagflation", "late_cycle"]
REGIME_COL_TO_ID = {
    "expansion": 0, "slowdown": 1, "recession": 2,
    "recovery": 3, "stagflation": 4, "late_cycle": 5,
}


def _prior_weights(regime_col: str, universe: list[str]) -> dict[str, float]:
    """Fetch hand-tuned prior for a regime, restricted to universe."""
    regime_id = REGIME_COL_TO_ID[regime_col]
    prior = REGIME_ALLOCATIONS[regime_id]
    w = {t: prior.get(t, 0.0) for t in universe}
    total = sum(w.values())
    if total > 0:
        w = {t: v / total for t, v in w.items()}
    return w


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end prices → monthly simple returns."""
    monthly_px = prices.resample("ME").last()
    return monthly_px.pct_change()


def regime_conditioned_sharpe(
    monthly_returns: pd.DataFrame,
    monthly_labels: pd.Series,
    regime_col: str,
    as_of: pd.Timestamp,
    min_obs: int = 12,
) -> pd.Series:
    """
    Per-ticker annualized Sharpe ratio using only past months where regime
    == regime_col (strict inequality: data < as_of).

    Returns NaN for tickers with < min_obs observations in that regime.
    """
    regime_id = REGIME_COL_TO_ID[regime_col]
    # Intersect indices so labels and returns align on the same month-ends
    common = monthly_returns.index.intersection(monthly_labels.index)
    if len(common) == 0:
        return pd.Series(np.nan, index=monthly_returns.columns)

    lbl = monthly_labels.loc[common]
    ret = monthly_returns.loc[common]
    mask = (lbl.index < as_of) & (lbl == regime_id)

    if int(mask.sum()) < min_obs:
        return pd.Series(np.nan, index=monthly_returns.columns)

    subset = ret.loc[mask.values]
    mu = subset.mean()
    sigma = subset.std()
    sharpe = (mu / sigma.replace(0, np.nan)) * np.sqrt(12)

    # Enforce per-ticker minimum observations (handles tickers with short history)
    valid_count = subset.count()
    sharpe = sharpe.where(valid_count >= min_obs, np.nan)
    return sharpe


def top_n_weights(
    scores: pd.Series,
    top_n: int = TOP_N,
    floor: float = FLOOR,
) -> dict[str, float]:
    """
    Select top-N ETFs by composite score and allocate proportionally.

    Scores can be negative (z-score composite). We shift so the minimum
    top-N score maps to `floor`, then normalise. This preserves relative
    ranking while guaranteeing every selected asset gets at least the
    floor allocation.
    """
    valid = scores.dropna()
    if valid.empty:
        return {}

    top = valid.nlargest(top_n)
    # Shift so the min score in the top-N sits exactly at `floor`
    shift = floor - top.min() if top.min() < floor else 0.0
    adjusted = top + shift
    adjusted = adjusted.clip(lower=floor)
    total = adjusted.sum()
    if total <= 0:
        w = 1.0 / len(top)
        return {t: w for t in top.index}
    return {t: float(v / total) for t, v in adjusted.items()}


def top_n_weights_hrp(
    scores: pd.Series,
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    top_n: int = TOP_N,
    floor: float = HRP_MIN_FLOOR,
    lookback_days: int = HRP_LOOKBACK_DAYS,
    max_weight: float = HRP_MAX_WEIGHT,
) -> dict[str, float]:
    """
    Select top-N by score, then weight via HRP over a Ledoit-Wolf covariance.

    Falls back to floor-shifted proportional weighting if HRP yields a
    degenerate solution. A small per-name floor keeps every selected ticker
    materially in the book — pure HRP can otherwise zero out a name in a
    tightly correlated cluster.
    """
    valid = scores.dropna()
    if valid.empty:
        return {}
    top = valid.nlargest(top_n)
    tickers = list(top.index)

    hrp = hrp_weights_for_subset(
        prices, tickers, as_of,
        lookback_days=lookback_days,
        max_weight=max_weight,
    )
    if not hrp or sum(hrp.values()) <= 1e-9:
        return top_n_weights(scores, top_n=top_n, floor=floor)

    # Apply per-name floor: lift any held name below `floor` up to it,
    # take the lift proportionally from above-floor names.
    held = [t for t in tickers if hrp.get(t, 0.0) > 1e-9]
    if not held:
        return top_n_weights(scores, top_n=top_n, floor=floor)

    w = {t: hrp.get(t, 0.0) for t in held}
    need = sum(max(0.0, floor - v) for v in w.values())
    if need > 0:
        donor_total = sum(v for v in w.values() if v > floor)
        if donor_total > 0:
            scale = max(0.0, 1.0 - need / donor_total)
            for t in w:
                if w[t] > floor:
                    w[t] *= scale
                else:
                    w[t] = floor
    s = sum(w.values())
    if s <= 0:
        return top_n_weights(scores, top_n=top_n, floor=floor)
    return {t: v / s for t, v in w.items()}


def build_per_regime_targets(
    prices: pd.DataFrame,
    monthly_labels: pd.Series,
    as_of: pd.Timestamp,
    top_n: int | None = None,
    floor: float = FLOOR,
    min_months: int = MIN_MONTHS_PER_REGIME,
    prior_blend: float = PRIOR_BLEND,
    prior_blend_cold: float = PRIOR_BLEND_COLD,
    scoring_mode: str = SCORING_MODE,
    within_regime_weighting: str = WITHIN_REGIME_WEIGHTING,
) -> dict[str, dict[str, float]]:
    """
    Compute data-driven target allocations for each regime as of a given date.

    For each regime:
      - If past history has ≥ min_months of same-regime observations:
            target = (1 - prior_blend) * top-N_sharpe_weights + prior_blend * hand_tuned_prior
      - Else: target = hand_tuned_prior (cold start)

    Returns: dict mapping regime_col → {ticker: weight}
    """
    monthly_returns = compute_monthly_returns(prices)
    # Keep only months strictly before as_of (walk-forward guarantee)
    monthly_returns = monthly_returns.loc[monthly_returns.index < as_of]
    labels_trimmed = monthly_labels.loc[monthly_labels.index < as_of]
    # Align on shared month-ends
    common = monthly_returns.index.intersection(labels_trimmed.index)
    monthly_returns = monthly_returns.loc[common]
    labels_trimmed = labels_trimmed.loc[common]

    universe = list(prices.columns)

    targets: dict[str, dict[str, float]] = {}

    for regime_col in REGIME_COLS:
        regime_id = REGIME_COL_TO_ID[regime_col]
        n_obs = int((labels_trimmed == regime_id).sum())
        prior = _prior_weights(regime_col, universe)
        regime_top_n = top_n if top_n is not None else TOP_N_PER_REGIME.get(regime_col, TOP_N)

        if n_obs < min_months:
            targets[regime_col] = prior
            continue

        sharpes = regime_conditioned_sharpe(
            monthly_returns, labels_trimmed, regime_col, as_of,
            min_obs=MIN_TICKER_HISTORY,
        )
        # Restrict to the hand-tuned prior's tickers for this regime — this
        # bakes in asset-class bias and prevents Sharpe from selecting
        # all-bond top-N during post-2008 bond bull periods.
        eligible = [t for t, w in prior.items() if w > 0]
        if scoring_mode == "composite":
            scores = composite_factor_score(
                sharpe=sharpes,
                prices=prices,
                as_of=as_of,
                eligible=eligible if eligible else None,
            )
        else:
            scores = sharpes
            if eligible:
                scores = scores.loc[scores.index.intersection(eligible)]
        if within_regime_weighting == "hrp":
            data_weights = top_n_weights_hrp(
                scores, prices=prices, as_of=as_of, top_n=regime_top_n,
            )
        else:
            data_weights = top_n_weights(scores, top_n=regime_top_n, floor=floor)

        if not data_weights:
            targets[regime_col] = prior
            continue

        # Blend data-driven with prior
        blended: dict[str, float] = {}
        for t in universe:
            blended[t] = (1 - prior_blend) * data_weights.get(t, 0.0) \
                         + prior_blend * prior.get(t, 0.0)
        # Normalize
        total = sum(blended.values())
        if total > 0:
            blended = {t: v / total for t, v in blended.items()}
        targets[regime_col] = blended

    return targets


def _dynamic_concentration(max_prob: float) -> float:
    """
    Map the max regime probability to a sharpening exponent.
        max_prob ≤ 0.35  → 1.0  (no sharpening; regime vote is mixed)
        max_prob ≥ 0.60  → 1.8  (moderate sharpening; one regime dominates)
        linear interpolation between
    """
    if max_prob <= 0.35:
        return 1.0
    if max_prob >= 0.60:
        return 1.8
    return 1.0 + (max_prob - 0.35) / (0.60 - 0.35) * (1.8 - 1.0)


def blend_with_probabilities(
    per_regime_targets: dict[str, dict[str, float]],
    probs: pd.Series,
    concentration: float | str = "auto",
) -> dict[str, float]:
    """
    Blend per-regime targets using the probability vector for a given date.

    If `concentration="auto"`, the sharpening exponent is chosen dynamically
    from the max regime probability (see _dynamic_concentration): decisive
    regime signals get concentrated blends, mixed signals stay diversified.
    """
    all_tickers: set[str] = set()
    for tgt in per_regime_targets.values():
        all_tickers.update(tgt.keys())

    p_raw = {r: max(0.0, float(probs.get(r, 0.0))) for r in per_regime_targets.keys()}
    if concentration == "auto":
        max_p = max(p_raw.values()) if p_raw else 0.0
        c = _dynamic_concentration(max_p)
    else:
        c = float(concentration)
    p_pow = {r: v ** c for r, v in p_raw.items()}
    denom = sum(p_pow.values())
    if denom <= 0:
        p_sharp = {r: 1.0 / len(p_pow) for r in p_pow}
    else:
        p_sharp = {r: v / denom for r, v in p_pow.items()}

    blended = {t: 0.0 for t in all_tickers}
    for regime_col, tgt in per_regime_targets.items():
        p = p_sharp.get(regime_col, 0.0)
        if p <= 0:
            continue
        for t, w in tgt.items():
            blended[t] = blended.get(t, 0.0) + p * w

    total = sum(blended.values())
    if total > 0:
        blended = {t: v / total for t, v in blended.items()}
    return blended


def build_sharpe_weighted_targets(
    prices: pd.DataFrame,
    monthly_probs: pd.DataFrame,
    monthly_labels: pd.Series,
    top_n: int | None = None,
    floor: float = FLOOR,
    min_months: int = MIN_MONTHS_PER_REGIME,
    prior_blend: float = PRIOR_BLEND,
    scoring_mode: str = SCORING_MODE,
    within_regime_weighting: str = WITHIN_REGIME_WEIGHTING,
) -> pd.DataFrame:
    """
    Walk-forward: for each rebalance month, produce a blended target vector
    across the universe using probability-weighted, regime-conditioned
    Sharpe-weighted allocations.

    Args:
        prices: DAILY price DataFrame (columns = tickers)
        monthly_probs: MONTHLY regime probabilities (columns = regime names)
        monthly_labels: MONTHLY regime label series (int values 0-5)

    Returns:
        DataFrame indexed by month-end dates, columns = tickers, rows sum to 1.
    """
    universe = list(prices.columns)
    out_rows = []

    for date in monthly_probs.index:
        per_regime = build_per_regime_targets(
            prices=prices,
            monthly_labels=monthly_labels,
            as_of=date,
            top_n=top_n,
            floor=floor,
            min_months=min_months,
            prior_blend=prior_blend,
            scoring_mode=scoring_mode,
            within_regime_weighting=within_regime_weighting,
        )
        probs_row = monthly_probs.loc[date]
        blended = blend_with_probabilities(per_regime, probs_row)
        row = {t: blended.get(t, 0.0) for t in universe}
        row["Date"] = date
        out_rows.append(row)

    df = pd.DataFrame(out_rows).set_index("Date")
    return df
