"""
optimizer.py
────────────
Mean-variance optimizer that maximises Information Ratio (active return / active
risk vs SPY) subject to portfolio constraints.

Objective
─────────
  max  w^T α − λ w^T Σ w
  s.t. constraints below

  where:
    α    = expected return vector from IC-weighted ensemble signal
    Σ    = Ledoit-Wolf annualised covariance from risk_model.estimate_covariance()
    λ    = risk aversion calibrated so unconstrained portfolio vol ≈ 10%

Constraints
───────────
  1. Long-only for equities, sectors, and stocks  (w_i ≥ 0)
  2. Two-sided for bonds and commodities           (w_i ∈ [-0.20, +0.20])
  3. Max single position: 20% of capital           (|w_i| ≤ 0.20)
  4. Max sector concentration: 40% in any asset class
  5. Min position: 0.5% if signal > 0              (avoids rounding to zero)
  6. Turnover penalty: 0.5 × Σ|w_t − w_{t−1}| × tc  added to objective
  7. Gross exposure ≤ 100% (no leverage)

Solver
──────
  scipy.optimize.minimize with SLSQP — no new packages.

Rebalance
─────────
  Daily: optimizer runs on each day's signals and trailing covariance.

Input / output
──────────────
  Consumed by portfolio.py via optimizer_sizes()
  Adds "ir_optimized" method to all_methods registry
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional

from .risk_model import estimate_covariance
from .data_pipeline import ASSET_CLASS


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_POSITION_PCT      = 0.20      # single-position cap
MAX_SECTOR_PCT        = 0.40      # max weight in any one asset class
MIN_POSITION_PCT      = 0.005     # 0.5% minimum if signal > 0
TARGET_VOL            = 0.10      # 10% annual portfolio vol target
TRANSACTION_COST      = 0.001     # 10 bps round-trip
TURNOVER_PENALTY_MULT = 0.5       # penalty multiplier on turnover
COV_WINDOW            = 126       # 6 months lookback for covariance
REBALANCE_FREQ        = 1         # daily (optimizer runs every day)
ALPHA_LOOKBACK        = 63        # 3-month rolling return for alpha estimate

# Asset classes that may take short positions
TWO_SIDED_CLASSES = {"bond", "commodity"}


# ── Alpha estimation ─────────────────────────────────────────────────────────

def _estimate_alpha(
    ensemble_signals: pd.Series,
    returns: pd.DataFrame,
    tickers: list,
    lookback: int = ALPHA_LOOKBACK,
) -> np.ndarray:
    """
    Expected return vector from ensemble signals scaled by trailing realised vol.

    alpha_i = ensemble_signal_i × trailing_vol_i × sqrt(252)

    The ensemble signal is already an IC-weighted conviction score in [-1, +1].
    Scaling by trailing vol converts the normalised signal into return-space
    units so that the optimizer's quadratic objective is well-conditioned.

    Args:
        ensemble_signals: Series of ensemble signal values for each ticker.
        returns:          Full returns DataFrame.
        tickers:          List of active ticker names.
        lookback:         Rolling window for vol estimation.

    Returns:
        Alpha vector (N,) in annualised return units.
    """
    alpha = np.zeros(len(tickers))
    for i, t in enumerate(tickers):
        sig = float(ensemble_signals.get(t, 0.0))
        if t in returns.columns:
            vol = returns[t].iloc[-lookback:].std() * np.sqrt(252)
            vol = max(vol, 0.01)  # floor at 1% to avoid div-by-zero
        else:
            vol = 0.15  # default 15% if missing
        alpha[i] = sig * vol
    return alpha


# ── Risk aversion calibration ────────────────────────────────────────────────

def _calibrate_lambda(
    alpha: np.ndarray,
    cov: np.ndarray,
    target_vol: float = TARGET_VOL,
) -> float:
    """
    Calibrate risk aversion λ so the unconstrained optimal portfolio has
    approximately target_vol annual volatility.

    Unconstrained solution: w* = (1/2λ) Σ^{-1} α
    Portfolio vol:          σ_p = sqrt(w*^T Σ w*)

    Solving:  λ = sqrt(α^T Σ^{-1} α) / (2 × target_vol)

    Falls back to λ = 5.0 if the algebra is degenerate.
    """
    try:
        cov_inv = np.linalg.pinv(cov, rcond=1e-10)
        quad = float(alpha @ cov_inv @ alpha)
        if quad <= 0:
            return 5.0
        return np.sqrt(quad) / (2.0 * target_vol)
    except np.linalg.LinAlgError:
        return 5.0


# ── Optimisation core ────────────────────────────────────────────────────────

def _build_bounds(tickers: list) -> list:
    """
    Per-ticker weight bounds based on asset class.

    Equities/sectors/stocks: [0, MAX_POSITION_PCT]  (long-only)
    Bonds/commodities:       [-MAX_POSITION_PCT, MAX_POSITION_PCT]  (two-sided)
    """
    bounds = []
    for t in tickers:
        ac = ASSET_CLASS.get(t, "equity_index")
        if ac in TWO_SIDED_CLASSES:
            bounds.append((-MAX_POSITION_PCT, MAX_POSITION_PCT))
        else:
            bounds.append((0.0, MAX_POSITION_PCT))
    return bounds


def _sector_concentration_constraints(tickers: list) -> list:
    """
    Build inequality constraints so no single asset class exceeds MAX_SECTOR_PCT.

    Returns list of constraint dicts for scipy SLSQP.
    Each constraint: MAX_SECTOR_PCT - sum(|w_i| for i in class) >= 0
    """
    # Group ticker indices by asset class
    class_groups: dict = {}
    for i, t in enumerate(tickers):
        ac = ASSET_CLASS.get(t, "equity_index")
        class_groups.setdefault(ac, []).append(i)

    constraints = []
    for ac, indices in class_groups.items():
        # Constraint: MAX_SECTOR_PCT - sum(|w_i|) >= 0
        def make_fn(idx_list):
            def fn(w):
                return MAX_SECTOR_PCT - sum(abs(w[j]) for j in idx_list)
            return fn
        constraints.append({
            "type": "ineq",
            "fun": make_fn(indices),
        })

    return constraints


def _gross_exposure_constraint():
    """Total gross exposure <= 1.0 (no leverage)."""
    return {
        "type": "ineq",
        "fun": lambda w: 1.0 - np.sum(np.abs(w)),
    }


def optimize_weights(
    alpha: np.ndarray,
    cov: np.ndarray,
    tickers: list,
    prev_weights: Optional[np.ndarray] = None,
    ensemble_signals: Optional[pd.Series] = None,
) -> np.ndarray:
    """
    Solve the mean-variance optimisation problem with constraints.

    max  w^T α − λ w^T Σ w − turnover_penalty
    s.t. bounds, sector concentration, gross exposure

    After optimisation, enforces min_position: any weight below MIN_POSITION_PCT
    for a ticker with positive signal is bumped to MIN_POSITION_PCT (then
    renormalised if needed).

    Args:
        alpha:             Expected return vector (N,).
        cov:               Annualised covariance matrix (N × N).
        tickers:           List of ticker names (N,).
        prev_weights:      Previous period weights for turnover penalty.
        ensemble_signals:  Ensemble signal Series for min-position enforcement.

    Returns:
        Optimal weight vector (N,), satisfying all constraints.
    """
    n = len(tickers)
    if n == 0:
        return np.array([])

    lam = _calibrate_lambda(alpha, cov)

    if prev_weights is None:
        prev_weights = np.zeros(n)

    # Objective: negative because scipy minimises
    def objective(w):
        ret = w @ alpha
        risk = w @ cov @ w
        turnover = TURNOVER_PENALTY_MULT * np.sum(np.abs(w - prev_weights)) * TRANSACTION_COST
        return -(ret - lam * risk) + turnover

    def gradient(w):
        grad_ret = -alpha
        grad_risk = 2.0 * lam * (cov @ w)
        grad_turnover = TURNOVER_PENALTY_MULT * TRANSACTION_COST * np.sign(w - prev_weights)
        return grad_ret + grad_risk + grad_turnover

    bounds = _build_bounds(tickers)
    constraints = _sector_concentration_constraints(tickers)
    constraints.append(_gross_exposure_constraint())

    # Initial guess: previous weights (warm start) or equal weight among
    # tickers with positive alpha
    if np.any(prev_weights != 0):
        w0 = prev_weights.copy()
    else:
        pos_alpha = alpha > 0
        if pos_alpha.any():
            w0 = np.where(pos_alpha, 1.0 / pos_alpha.sum(), 0.0)
            # Clip to bounds
            for i, (lo, hi) in enumerate(bounds):
                w0[i] = np.clip(w0[i], lo, hi)
            # Scale to satisfy gross constraint
            gross = np.sum(np.abs(w0))
            if gross > 1.0:
                w0 /= gross
        else:
            w0 = np.zeros(n)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds",
            category=RuntimeWarning,
        )
        result = minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10},
        )

    w_opt = result.x

    # Enforce minimum position: if signal > 0 and weight is positive but
    # below MIN_POSITION_PCT, bump it up.  This prevents rounding to zero.
    if ensemble_signals is not None:
        for i, t in enumerate(tickers):
            sig = float(ensemble_signals.get(t, 0.0))
            if sig > 0 and 0 < w_opt[i] < MIN_POSITION_PCT:
                w_opt[i] = MIN_POSITION_PCT

    # Final projection: re-clip bounds and gross exposure
    for i, (lo, hi) in enumerate(bounds):
        w_opt[i] = np.clip(w_opt[i], lo, hi)

    gross = np.sum(np.abs(w_opt))
    if gross > 1.0:
        w_opt /= gross

    return w_opt


# ── Portfolio-level entry point ──────────────────────────────────────────────

def optimizer_sizes(
    signals: pd.DataFrame,
    ensemble_signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    """
    Mean-variance optimised position sizing using IC-weighted ensemble signals.

    Runs the optimizer daily on the ensemble signal vector and trailing
    Ledoit-Wolf covariance to produce dollar position sizes that maximise
    the portfolio's information ratio subject to constraints.

    This is a NEW sizing method alongside existing ones — it does NOT
    replace atr_sizes, ensemble_sizes, or any other function.

    Args:
        signals:          Binary signal DataFrame (T × N) — used to determine
                          which tickers are active (signal != 0) on each day.
        ensemble_signals: Continuous [-1, +1] ensemble signal DataFrame (T × N).
        features:         Dict[ticker -> feature DataFrame] with Close prices.
        returns:          Daily log-return DataFrame (T × N).
        capital:          Total capital in dollars.

    Returns:
        Dollar position size DataFrame (T × N).
    """
    tickers = [t for t in signals.columns if t in returns.columns]
    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

    prev_weights = np.zeros(len(tickers))
    cov_cache = None
    cov_tickers_cache = None

    for i in range(COV_WINDOW, len(signals)):
        # Recompute covariance periodically (every REBALANCE_FREQ days)
        if cov_cache is None or i % REBALANCE_FREQ == 0:
            ret_window = returns.iloc[max(0, i - COV_WINDOW):i][tickers]
            # Drop tickers with >10% missing
            ret_window = ret_window.dropna(
                thresh=int(len(ret_window) * 0.90), axis=1
            )
            available = ret_window.columns.tolist()
            if len(available) >= 2 and len(ret_window) >= 20:
                cov_cache = estimate_covariance(ret_window)
                cov_tickers_cache = available

        if cov_cache is None or cov_tickers_cache is None:
            continue

        # Determine active tickers: must have a signal AND be in the cov universe
        sig_today = signals.iloc[i]
        ens_today = ensemble_signals.iloc[i]
        active = [
            t for t in cov_tickers_cache
            if sig_today.get(t, 0) != 0
        ]

        if len(active) < 2:
            # Too few active tickers for meaningful optimisation
            prev_weights = np.zeros(len(tickers))
            continue

        # Build sub-matrices for active tickers
        active_idx = [cov_tickers_cache.index(t) for t in active]
        sub_cov = cov_cache[np.ix_(active_idx, active_idx)]
        sub_ens = ens_today.reindex(active).fillna(0.0)

        # Estimate alpha
        alpha = _estimate_alpha(
            sub_ens, returns.iloc[max(0, i - ALPHA_LOOKBACK):i], active
        )

        # Map previous weights to the active subset
        prev_w_sub = np.array([
            prev_weights[tickers.index(t)] if t in tickers else 0.0
            for t in active
        ])

        # Optimise
        w_opt = optimize_weights(
            alpha, sub_cov, active,
            prev_weights=prev_w_sub,
            ensemble_signals=sub_ens,
        )

        # Zero out numerical noise (weights < 0.1% of capital)
        noise_threshold = 0.001
        w_opt[np.abs(w_opt) < noise_threshold] = 0.0

        # Convert weights to dollar sizes
        for j, t in enumerate(active):
            sizes.iloc[i, sizes.columns.get_loc(t)] = w_opt[j] * capital

        # Update prev_weights (full universe)
        full_w = np.zeros(len(tickers))
        for j, t in enumerate(active):
            if t in tickers:
                full_w[tickers.index(t)] = w_opt[j]
        prev_weights = full_w

    return sizes


# ── Signal-gated minimum variance weights ────────────────────────────────────

def minimum_variance_gated_weights(
    active_tickers: list,
    cov_matrix: np.ndarray,
    ensemble_scores: dict,
    asset_classes: dict,
    max_position: float = 0.20,
    max_sector_pct: float = 0.40,
    min_position: float = 0.005,
    ensemble_tilt_max: float = 0.20,
) -> dict:
    """
    Minimum variance portfolio weights with post-solve ensemble tilt.

    Two-phase approach
    ──────────────────
    Phase 1 — Pure minimum variance (no alpha vector):
      min  w^T Σ w
      s.t. sum(w) = 1, w_i ∈ [min_position, max_position],
           sector concentration ≤ max_sector_pct

    This avoids the alpha estimation error that plagued ir_optimized.
    MV weights work best in calm regimes where the covariance estimate
    is most accurate — exactly the regime where rp_regime_aware is weakest.

    Phase 2 — Ensemble tilt (AFTER optimization, not inside):
      w_tilted_i = w_solved_i × (1 + ensemble_tilt_max × score_i)
      Then renormalize and re-clip.

    The tilt adjusts weights by at most ±20% of their MV-optimal value.
    This is deliberately small — the conviction signal modulates allocation
    without overriding the variance-minimizing structure.

    Args:
        active_tickers:    List of ticker names in the active set.
        cov_matrix:        Annualised covariance matrix (N × N) for active
                           tickers, in same order as active_tickers.
        ensemble_scores:   {ticker: float in [-1, +1]} — IC-weighted ensemble
                           conviction from signal_generation.ensemble_signal().
        asset_classes:     {ticker: asset_class_string} for sector constraints.
        max_position:      Maximum weight per ticker (default 0.20).
        max_sector_pct:    Maximum total weight in any one asset class (default 0.40).
        min_position:      Minimum weight per ticker (default 0.005 = 0.5%).
        ensemble_tilt_max: Maximum tilt factor (default 0.20 = ±20%).

    Returns:
        Dict {ticker: weight} summing to 1.0.
        Falls back to equal weights on solver failure.
    """
    n = len(active_tickers)
    if n == 0:
        return {}
    if n == 1:
        return {active_tickers[0]: 1.0}

    # ── Phase 1: solve minimum variance ──────────────────────────────────

    # Objective: minimize w^T Σ w
    def objective(w):
        return w @ cov_matrix @ w

    def gradient(w):
        return 2.0 * (cov_matrix @ w)

    # Bounds: all-long, [min_position, max_position]
    bounds = [(min_position, max_position)] * n

    # Constraints: sum(w) == 1 + sector concentration
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Sector concentration constraints
    class_groups: dict = {}
    for i, t in enumerate(active_tickers):
        ac = asset_classes.get(t, "equity_index")
        class_groups.setdefault(ac, []).append(i)

    for ac, indices in class_groups.items():
        def make_fn(idx_list):
            def fn(w):
                return max_sector_pct - sum(w[j] for j in idx_list)
            return fn
        constraints.append({"type": "ineq", "fun": make_fn(indices)})

    # Initial guess: inverse-vol (same warm start as risk_parity_weights)
    vols = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))
    w0 = 1.0 / vols
    w0 /= w0.sum()
    w0 = np.clip(w0, min_position, max_position)
    w0 /= w0.sum()

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds",
                category=RuntimeWarning,
            )
            result = minimize(
                objective,
                w0,
                jac=gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-10},
            )
        if result.success or result.fun < objective(w0):
            w_solved = result.x
        else:
            warnings.warn(
                f"MV solver did not converge for {n} tickers; using inv-vol fallback"
            )
            w_solved = w0
    except Exception:
        warnings.warn(
            f"MV solver failed for {n} tickers; using equal-weight fallback"
        )
        w_solved = np.full(n, 1.0 / n)
        w_solved = np.clip(w_solved, min_position, max_position)
        w_solved /= w_solved.sum()

    # ── Phase 2: ensemble tilt ───────────────────────────────────────────
    tilted = np.empty(n)
    for i, t in enumerate(active_tickers):
        score = ensemble_scores.get(t, 0.0)
        tilted[i] = w_solved[i] * (1.0 + ensemble_tilt_max * score)

    # Ensure all positive (tilt can't make weight negative since
    # min solved weight is min_position=0.005 and max tilt is 0.80×)
    tilted = np.maximum(tilted, 1e-8)
    tilted /= tilted.sum()

    # Re-clip to bounds and renormalize
    tilted = np.clip(tilted, min_position, max_position)
    tilted /= tilted.sum()

    return {t: float(tilted[i]) for i, t in enumerate(active_tickers)}
