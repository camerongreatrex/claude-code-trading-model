"""
risk_model.py — covariance estimation + risk-based weighting (Ledoit-Wolf,
ERC risk parity, max-diversification, HRP). T/N≈6 (126d/21 assets) → shrinkage
intensity 0.3–0.6. ERC: w_i*(Σw)_i = σ_p²/N (Maillard et al. 2010). Max-div:
DR = (w^T σ)/sqrt(w^T Σ w), analytic long-only w* ∝ Σ^{-1} σ (Choueifaty 2008).
Consumed by portfolio.py.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from sklearn.covariance import LedoitWolf as _LedoitWolf
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _ledoit_wolf_analytical(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward scaled identity (RBLW / Chen et al. 2010).
    Pure-NumPy fallback equivalent to sklearn's LedoitWolf with target (tr(S)/p)·I.
    Returns shrunk daily covariance (N × N) from centred return matrix X (T × N)."""
    n, p = X.shape
    # Biased sample cov (1/n, sklearn convention)
    S     = (X.T @ X) / n
    tr_S  = np.trace(S)
    tr_S2 = np.trace(S @ S)

    # RBLW shrinkage intensity
    denom = (n + 2) * (tr_S2 - tr_S ** 2 / p)
    if abs(denom) < 1e-15:
        return S   # degenerate

    rho = min(1.0, ((n - 2) / n * tr_S2 + tr_S ** 2) / denom)

    # Blend toward (tr(S)/p)·I
    mu = tr_S / p
    return (1.0 - rho) * S + rho * mu * np.eye(p)


# ── Covariance estimation ──────────────────────────────────────────────────────

def estimate_covariance(
    returns: pd.DataFrame,
    method: Literal["ledoit_wolf", "sample"] = "ledoit_wolf",
) -> np.ndarray:
    """Annualised covariance from daily log-returns (T × N).
    method="ledoit_wolf" (Σ̂ = (1-α)·S + α·(tr(S)/N)·I) or "sample" (MLE).
    NaN cols dropped; remaining NaN cells zero-filled before fit."""
    clean = returns.dropna(axis=1, how="all").ffill().fillna(0)
    X     = clean.values - clean.values.mean(axis=0)  # centre

    if method == "ledoit_wolf":
        if _HAS_SKLEARN:
            lw  = _LedoitWolf(assume_centered=True)
            lw.fit(X)
            cov = lw.covariance_
        else:
            # Pure-NumPy RBLW fallback
            cov = _ledoit_wolf_analytical(X)
    else:
        # MLE sample cov
        cov = (X.T @ X) / len(X)
        if cov.ndim == 0:              # single-asset edge case
            cov = np.array([[float(cov)]])

    return cov * 252                   # annualise (×252)


# ── Regime-conditional covariance ─────────────────────────────────────────────

def regime_conditional_covariance(
    returns_df: pd.DataFrame,
    regimes_series: pd.Series,
    current_regime: str,
    min_obs: int = 63,
    fallback_window: int = 126,
) -> Tuple[np.ndarray, int]:
    """Covariance using only returns from current_regime; falls back to last
    fallback_window rows when filtered sample < min_obs (default 63 ≈ 3mo).
    Captures regime-shift in correlations (e.g. ~0.20 bull_calm → ~0.45+ bear_stress).
    Returns (annualised cov N×N, n_regime_obs)."""
    aligned = regimes_series.reindex(returns_df.index)
    mask = aligned == current_regime
    regime_returns = returns_df.loc[mask]

    if len(regime_returns) >= min_obs:
        cov = estimate_covariance(regime_returns)
        return cov, len(regime_returns)

    # Fallback: insufficient regime-specific observations
    tail = returns_df.iloc[-fallback_window:]
    cov = estimate_covariance(tail)
    return cov, len(tail)


# ── Portfolio risk ─────────────────────────────────────────────────────────────

def portfolio_risk(weights: np.ndarray, cov: np.ndarray) -> float:
    """σ_p = sqrt(w^T Σ w). Weights and cov must share units (annualised).
    Clipped to 0 for flat portfolios."""
    w   = np.asarray(weights, dtype=float)
    var = w @ cov @ w
    return float(np.sqrt(max(var, 0.0)))


# ── Risk parity (ERC) ──────────────────────────────────────────────────────────

def risk_parity_weights(
    cov: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-8,
    max_weight: float = 0.30,
) -> np.ndarray:
    """ERC via fixed-point: w_i ← 1/(Cov@w)_i, renormalise (Maillard 2010).
    Init inverse-vol. Returns weights ≥ 0, sum=1, ≤ max_weight (default 30%)."""
    n   = cov.shape[0]
    sig = np.sqrt(np.maximum(np.diag(cov), 1e-12))

    # Inverse-vol init (exact ERC for uncorrelated; near-ERC otherwise)
    w = 1.0 / sig
    w /= w.sum()

    # Bounds [min_w, max_weight]: min_w = 1/(3n), max default 30%
    min_w = 1.0 / (3.0 * n)

    for _ in range(max_iter):
        mrc   = cov @ w                          # marginal risk contributions
        w_new = 1.0 / np.maximum(mrc, 1e-12)     # fixed-point update
        w_new /= w_new.sum()

        # Clip-and-renormalise; up to 20 passes
        for _ in range(20):
            w_new = np.clip(w_new, min_w, max_weight)
            w_new /= w_new.sum()
            if w_new.max() <= max_weight + 1e-9 and w_new.min() >= min_w - 1e-9:
                break

        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new

    return w


# ── Max diversification ────────────────────────────────────────────────────────

def max_diversification_weights(
    cov: np.ndarray,
    max_weight: float = 0.40,
) -> np.ndarray:
    """Max-Diversification (Choueifaty & Coignard 2008): DR = (w^T σ)/sqrt(w^T Σ w).
    Long-only analytic: w* ∝ Σ^{-1} σ, clip negatives, normalise.
    Falls back to inverse-vol if pseudo-inverse degenerate. Default cap 40%."""
    sig = np.sqrt(np.maximum(np.diag(cov), 1e-12))

    try:
        cov_inv = np.linalg.pinv(cov, rcond=1e-10)
        w       = cov_inv @ sig
    except np.linalg.LinAlgError:
        w = 1.0 / sig  # fallback: inverse-vol

    # Long-only: clip negatives
    w = np.clip(w, 0.0, None)
    if w.sum() < 1e-10:
        w = 1.0 / sig   # fallback

    w /= w.sum()

    # Cap and renormalise
    if max_weight < 1.0:
        w = np.clip(w, 0.0, max_weight)
        total = w.sum()
        if total > 1e-10:
            w /= total

    return w


# ── Hierarchical Risk Parity ───────────────────────────────────────────────────

def hrp_weights(cov: np.ndarray, max_weight: float = 0.30) -> np.ndarray:
    """Hierarchical Risk Parity (López de Prado 2016): (1) d_ij=sqrt(0.5(1-ρ_ij)),
    (2) single-linkage clustering for quasi-diagonal leaf order,
    (3) recursive bisection allocating inversely to sub-cluster variance.
    No matrix inversion → robust at small T/N. Falls back to inv-vol if scipy missing."""
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except ImportError:
        # Inverse-vol fallback
        sig = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        w   = 1.0 / sig
        w  /= w.sum()
        return w

    # ── Step 1: Correlation → distance ──────────────────────────────────────
    sig  = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(sig, sig)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # LdP (2016) eq. 4.1: corr ∈ [-1,1] → dist ∈ [0,1]
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)

    # ── Step 2: Single-linkage clustering ──────────────────────────────────
    condensed = squareform(dist, checks=False)
    link      = linkage(condensed, method="single")
    sort_ix   = list(leaves_list(link))               # quasi-diagonal order

    # ── Step 3: Recursive bisection ─────────────────────────────────────────
    w = np.ones(n)   # multiplicative accumulation

    def _cluster_var(items: list) -> float:
        """Equal-weight sub-portfolio variance."""
        sub   = np.array(items, dtype=int)
        eq_w  = np.ones(len(sub)) / len(sub)
        return float(eq_w @ cov[np.ix_(sub, sub)] @ eq_w)

    def _bisect(items: list) -> None:
        if len(items) <= 1:
            return
        mid   = len(items) // 2
        left  = items[:mid]
        right = items[mid:]

        var_l = _cluster_var(left)
        var_r = _cluster_var(right)
        total = var_l + var_r

        if total < 1e-15:
            return   # degenerate

        # alpha = left fraction (right gets 1 - alpha)
        alpha = 1.0 - var_l / total

        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= (1.0 - alpha)

        _bisect(left)
        _bisect(right)

    _bisect(sort_ix)

    w_sum = w.sum()
    if w_sum > 1e-10:
        w /= w_sum

    # ── Cap + iterative redistribution ──────────────────────────────────────
    if max_weight < 1.0:
        for _ in range(20):
            w = np.clip(w, 0.0, max_weight)
            s = w.sum()
            if s > 1e-10:
                w /= s
            if w.max() <= max_weight + 1e-9:
                break

    return w


# ── Standalone analysis ────────────────────────────────────────────────────────

def main():
    """Print shrinkage diagnostics and weighting-scheme comparison on last 126d."""
    from .data_pipeline import TICKER_LIST, ASSET_CLASS

    FEATURE_DIR = Path("data/v1/features")

    returns = pd.DataFrame()
    for ticker in TICKER_LIST:
        path = FEATURE_DIR / f"{ticker}.parquet"
        if path.exists():
            returns[ticker] = pd.read_parquet(path)["log_return"]

    returns = returns.dropna()
    window  = returns.iloc[-126:]   # last 126 trading days

    print("=" * 70)
    print(f"Risk Model Diagnostics  ({window.index[0].date()} to {window.index[-1].date()})")
    print(f"  N assets = {window.shape[1]}   T observations = {len(window)}   T/N = {len(window)/window.shape[1]:.1f}")
    print("=" * 70)

    # ── Covariance diagnostics ───────────────────────────────────────────────
    cov_lw = estimate_covariance(window, method="ledoit_wolf")
    cov_sc = estimate_covariance(window, method="sample")

    eig_lw = np.linalg.eigvalsh(cov_lw)
    eig_sc = np.linalg.eigvalsh(cov_sc)

    print(f"\nLedoit-Wolf shrinkage covariance:")
    print(f"  Condition number : {eig_lw[-1]/max(eig_lw[0],1e-15):.1f}")
    print(f"  Eigenvalue range : {eig_lw[0]:.4f} – {eig_lw[-1]:.4f}")

    print(f"\nSample covariance (MLE):")
    print(f"  Condition number : {eig_sc[-1]/max(eig_sc[0],1e-15):.1f}")
    print(f"  Eigenvalue range : {eig_sc[0]:.4f} – {eig_sc[-1]:.4f}")

    X_centered = window.values - window.values.mean(axis=0)
    if _HAS_SKLEARN:
        alpha = _LedoitWolf(assume_centered=True).fit(X_centered).shrinkage_
    else:
        n, p  = X_centered.shape
        S     = (X_centered.T @ X_centered) / n
        tr_S  = np.trace(S); tr_S2 = np.trace(S @ S)
        denom = (n + 2) * (tr_S2 - tr_S ** 2 / p)
        alpha = min(1.0, ((n - 2) / n * tr_S2 + tr_S ** 2) / denom) if abs(denom) > 1e-15 else 0
    print(f"  Shrinkage intensity (alpha): {alpha:.3f}  ({'sklearn' if _HAS_SKLEARN else 'RBLW analytical'})")

    # ── Weight comparison ────────────────────────────────────────────────────
    w_rp  = risk_parity_weights(cov_lw)
    w_md  = max_diversification_weights(cov_lw)
    vols  = np.sqrt(np.diag(cov_lw))
    w_iv  = 1.0 / vols;  w_iv /= w_iv.sum()

    print(f"\n{'Ticker':<8} {'InvVol':>8} {'RiskParity':>12} {'MaxDiversif':>12} {'Ann.Vol':>10}")
    print("-" * 54)
    for i, t in enumerate(window.columns):
        print(f"{t:<8} {w_iv[i]:>8.3f} {w_rp[i]:>12.3f} {w_md[i]:>12.3f} {vols[i]:>9.1%}")

    print(f"\nPortfolio risk (ann. vol):")
    print(f"  Equal weight  : {portfolio_risk(np.ones(len(w_rp))/len(w_rp), cov_lw):.1%}")
    print(f"  Inverse-vol   : {portfolio_risk(w_iv,  cov_lw):.1%}")
    print(f"  Risk parity   : {portfolio_risk(w_rp,  cov_lw):.1%}")
    print(f"  Max diversif. : {portfolio_risk(w_md,  cov_lw):.1%}")

    dr_iv = (w_iv @ vols) / portfolio_risk(w_iv,  cov_lw)
    dr_rp = (w_rp @ vols) / portfolio_risk(w_rp,  cov_lw)
    dr_md = (w_md @ vols) / portfolio_risk(w_md,  cov_lw)
    print(f"\nDiversification Ratio:")
    print(f"  Inverse-vol   : {dr_iv:.3f}")
    print(f"  Risk parity   : {dr_rp:.3f}")
    print(f"  Max diversif. : {dr_md:.3f}  (maximised)")


if __name__ == "__main__":
    main()
