"""
risk_model.py
-------------
Covariance estimation and risk-based portfolio weighting.

Why Ledoit-Wolf?
────────────────
For N=21 assets on a 126-day window the observation-to-asset ratio is only 6.
Raw sample covariance is known to be badly conditioned in this regime: its
extreme eigenvalues are biased (largest too large, smallest too small), making
portfolio optimisation that inverts the matrix numerically unstable.

Ledoit-Wolf shrinkage analytically estimates how much to pull eigenvalues
toward their grand mean, trading a small bias for a large reduction in
estimation variance.  For T/N ≈ 6 the shrinkage intensity is typically
0.3–0.6, meaning the final matrix is a 40–70% sample / 60–30% target blend.

Risk parity (ERC)
─────────────────
Equal Risk Contribution (Maillard, Roncalli & Teiletche, 2010):
  Find weights w such that each asset contributes equally to total variance:
    w_i * (Σ w)_i = σ_p² / N  for all i

Solved via fixed-point iteration:
  w_i^{k+1} = 1 / (Σ w^k)_i   then renormalise
Initialised with inverse-vol weights (already near-ERC for low-correlation portfolios).

Max diversification
───────────────────
Maximise the Diversification Ratio (Choueifaty & Coignard, 2008):
  DR = (w^T σ) / sqrt(w^T Σ w)
Analytic solution (long-only): w* ∝ Σ^{-1} σ

Input / output
──────────────
  Consumed by portfolio.py: estimate_covariance, portfolio_risk,
  risk_parity_weights, max_diversification_weights
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
    """
    Ledoit-Wolf shrinkage toward scaled identity — pure NumPy implementation.

    Uses the Oracle Approximating Shrinkage (OAS/RBLW) formula from
    Chen, Wiesel, Eldar & Hero (2010), which is analytically equivalent to
    sklearn's LedoitWolf for the spherical target Σ_target = (tr(S)/p) × I.

    For T/N ≈ 6 (our 126-day / 21-asset case) the shrinkage intensity is
    typically 0.3–0.6, pulling extreme eigenvalues toward the mean and
    reducing the condition number by 10–100×.

    Args:
        X: Centred return matrix (T × N).

    Returns:
        Shrunk daily covariance matrix (N × N).
    """
    n, p = X.shape
    # Biased sample covariance (1/n denominator, consistent with sklearn)
    S     = (X.T @ X) / n
    tr_S  = np.trace(S)
    tr_S2 = np.trace(S @ S)

    # RBLW shrinkage intensity (analytically optimal for spherical target)
    denom = (n + 2) * (tr_S2 - tr_S ** 2 / p)
    if abs(denom) < 1e-15:
        return S   # degenerate: return sample covariance unchanged

    rho = min(1.0, ((n - 2) / n * tr_S2 + tr_S ** 2) / denom)

    # Shrunk estimator: blend sample cov toward (tr(S)/p) × I
    mu = tr_S / p
    return (1.0 - rho) * S + rho * mu * np.eye(p)


# ── Covariance estimation ──────────────────────────────────────────────────────

def estimate_covariance(
    returns: pd.DataFrame,
    method: Literal["ledoit_wolf", "sample"] = "ledoit_wolf",
) -> np.ndarray:
    """
    Estimate the annualised covariance matrix from daily log-returns.

    Ledoit-Wolf shrinkage stabilises the estimate when the observation-to-asset
    ratio (T/N) is small.  For T=126, N=21 → T/N ≈ 6, which is the regime
    where shrinkage gives the largest benefit over raw sample covariance.

    The shrinkage target is a scaled identity matrix (spherical shrinkage):
        Σ̂ = (1-α) × S + α × (trace(S)/N) × I
    where α is estimated analytically by the Ledoit-Wolf formula.

    Args:
        returns: Daily log-returns DataFrame (T × N).  Any columns with all-NaN
                 are dropped; remaining NaN cells are zero-filled before fitting.
        method:  "ledoit_wolf" (default, requires scikit-learn) or "sample"
                 (raw MLE covariance, may be poorly conditioned for small T/N).

    Returns:
        Annualised covariance matrix as a numpy array (N × N).
        Positive semi-definite by construction (Ledoit-Wolf or sample cov × 252).
    """
    clean = returns.dropna(axis=1, how="all").ffill().fillna(0)
    X     = clean.values - clean.values.mean(axis=0)  # centre (T, N)

    if method == "ledoit_wolf":
        if _HAS_SKLEARN:
            lw  = _LedoitWolf(assume_centered=True)
            lw.fit(X)
            cov = lw.covariance_        # daily covariance (N × N)
        else:
            # Pure-NumPy fallback: RBLW analytical Ledoit-Wolf
            cov = _ledoit_wolf_analytical(X)
    else:
        # MLE sample covariance
        cov = (X.T @ X) / len(X)
        if cov.ndim == 0:              # single-asset edge case
            cov = np.array([[float(cov)]])

    return cov * 252                   # annualise: Var[annual] = 252 × Var[daily]


# ── Regime-conditional covariance ─────────────────────────────────────────────

def regime_conditional_covariance(
    returns_df: pd.DataFrame,
    regimes_series: pd.Series,
    current_regime: str,
    min_obs: int = 63,
    fallback_window: int = 126,
) -> Tuple[np.ndarray, int]:
    """
    Estimate covariance using only returns from a specific market regime.

    Why regime-conditional?
    ───────────────────────
    The correlation diagnostic shows average pairwise correlation spikes from
    ~0.20 in bull_calm to ~0.45+ in bear_stress.  Standard risk parity uses a
    rolling window that mixes regimes: during a stress period the recent window
    still contains bull_calm days that pull correlations DOWN, making the
    portfolio appear more diversified than it actually is.

    By filtering returns to the current regime's historical observations, the
    covariance matrix reflects the correlation structure that actually prevails
    NOW — not a blended average of different market states.

    Algorithm
    ─────────
    1. Filter returns_df to rows where regimes_series == current_regime.
    2. If filtered count >= min_obs (default 63 ≈ 3 months): fit Ledoit-Wolf
       on those observations only.
    3. Else: fall back to standard estimate_covariance() on the last
       fallback_window (default 126) rows of returns_df.

    The fallback guarantees a valid covariance matrix even for rare regimes
    (e.g. bear_stress may only have 40 days in a bull market period).

    Args:
        returns_df:     Daily log-returns DataFrame (T × N).
        regimes_series: Series of regime label strings ('bull_calm',
                        'bull_stress', 'bear_calm', 'bear_stress') aligned
                        to returns_df.index.  Produced by
                        regime_analysis.label_regimes().
        current_regime: Which regime to condition on (e.g. 'bear_stress').
        min_obs:        Minimum observations required to use the regime-
                        filtered sample (default 63 = ~3 months).
        fallback_window: Number of trailing rows to use when the regime
                         sample is too small (default 126 = 6 months).

    Returns:
        Tuple of (cov_matrix, n_regime_obs):
          cov_matrix:   Annualised covariance matrix (N × N), positive
                        semi-definite by construction (Ledoit-Wolf).
          n_regime_obs: Number of regime-matched observations used.
                        If < min_obs, the fallback was used and this value
                        equals fallback_window (or len(returns_df) if shorter).
    """
    aligned = regimes_series.reindex(returns_df.index)
    mask = aligned == current_regime
    regime_returns = returns_df.loc[mask]

    if len(regime_returns) >= min_obs:
        cov = estimate_covariance(regime_returns)
        return cov, len(regime_returns)

    # Fallback: not enough regime-specific observations
    tail = returns_df.iloc[-fallback_window:]
    cov = estimate_covariance(tail)
    return cov, len(tail)


# ── Portfolio risk ─────────────────────────────────────────────────────────────

def portfolio_risk(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Annualised portfolio volatility from weights and covariance matrix.

    σ_p = sqrt(w^T Σ w)

    Args:
        weights: Portfolio weight vector (N,).  Need not sum to 1; can be
                 dollar-weight fractions (dollar_i / total_capital).
        cov:     Annualised covariance matrix (N × N) from estimate_covariance().
                 Must be in the same units as weights (annual if weights are
                 annual fractions, daily if daily, etc.).

    Returns:
        Annualised portfolio volatility as a float (e.g., 0.12 = 12%).
        Clipped to 0 — returns 0 for a flat (all-zero weight) portfolio.
    """
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
    """
    Equal Risk Contribution weights via fixed-point iteration.

    At convergence: w_i * (Cov @ w)_i = constant for all i.
    Each asset's marginal risk contribution to total portfolio variance
    equals 1/N of the total variance.

    Algorithm (Maillard et al., 2010 fixed-point):
        Initialise: w_i = 1 / sigma_i  (inverse-vol, close to ERC for low-corr)
        Iterate:    w_i <- 1 / (Cov @ w)_i,  then renormalise
        Converges to ERC for any positive-definite covariance matrix.

    The iteration exploits the fixed-point property of the ERC solution:
        At the optimum:  w_i ∝ 1 / (Cov @ w)_i  (each weight is proportional
        to the reciprocal of its own marginal risk contribution).

    Args:
        cov:      Covariance matrix (N × N).  May be daily or annualised —
                  the result is invariant to scalar scaling of the matrix.
        max_iter:   Maximum iterations (default 300).  Typically converges in
                    20-50 iterations for well-conditioned matrices.
        tol:        Convergence threshold on maximum weight change (default 1e-8).
        max_weight: Per-asset weight cap (default 0.30 = 30%).  When correlations
                    are extreme the unconstrained ERC solution can degenerate to a
                    handful of assets; the cap keeps the portfolio diversified.
                    After capping, weights are renormalised to sum to 1.

    Returns:
        Weight vector (N,): non-negative, sums to 1, each ≤ max_weight.
    """
    n   = cov.shape[0]
    sig = np.sqrt(np.maximum(np.diag(cov), 1e-12))

    # Inverse-vol initialisation: already the ERC solution for uncorrelated assets.
    # For correlated portfolios it is close enough for fast convergence.
    w = 1.0 / sig
    w /= w.sum()

    # Weight bounds: [min_w, max_weight].
    # min_w = 1/(3n): no asset below 1/3 of equal weight.
    # max_weight: no single asset above 30% (default).
    # Bounds are enforced by clipping AFTER normalisation and iterating
    # the redistribution until all weights satisfy the constraints.
    min_w = 1.0 / (3.0 * n)

    for _ in range(max_iter):
        mrc   = cov @ w                          # marginal risk contributions
        w_new = 1.0 / np.maximum(mrc, 1e-12)    # fixed-point update
        w_new /= w_new.sum()                     # normalise to sum=1

        # Project onto [min_w, max_weight]: clip after norm so bounds hold
        # post-normalisation.  Up to 20 redistribute passes converge quickly.
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
    """
    Weights that maximise the portfolio Diversification Ratio.

    Diversification Ratio (Choueifaty & Coignard, 2008):
        DR = (w^T sigma) / sqrt(w^T Cov w)
           = weighted-average asset vol / portfolio vol

    Maximising DR finds the portfolio that uses diversification most
    efficiently — it reaps the most risk reduction from combining assets.

    Analytic long-only solution:
        w* ∝ Cov^{-1} sigma  (then clip negatives and normalise)
    where sigma_i = sqrt(Cov_ii) is the individual asset volatility.

    Intuition: assets with low marginal contribution to portfolio risk
    (low row sum in Cov^{-1}) get larger weights.  Assets that are highly
    correlated with the rest of the portfolio get smaller weights.

    Args:
        cov:        Covariance matrix (N × N).  May be daily or annualised.
        max_weight: Maximum single-asset weight (default 0.40 = 40%).
                    Prevents one asset from dominating when the pseudo-inverse
                    assigns extreme weights.

    Returns:
        Weight vector (N,): non-negative, sums to 1.
        Falls back to inverse-vol if the pseudo-inverse is degenerate.
    """
    sig = np.sqrt(np.maximum(np.diag(cov), 1e-12))

    try:
        cov_inv = np.linalg.pinv(cov, rcond=1e-10)
        w       = cov_inv @ sig
    except np.linalg.LinAlgError:
        w = 1.0 / sig  # fallback: inverse-vol

    # Long-only constraint: clip negatives (implied shorts disallowed)
    w = np.clip(w, 0.0, None)
    if w.sum() < 1e-10:
        w = 1.0 / sig   # fallback if all weights are non-positive

    w /= w.sum()

    # Per-asset cap, then renormalise
    if max_weight < 1.0:
        w = np.clip(w, 0.0, max_weight)
        total = w.sum()
        if total > 1e-10:
            w /= total

    return w


# ── Hierarchical Risk Parity ───────────────────────────────────────────────────

def hrp_weights(cov: np.ndarray, max_weight: float = 0.30) -> np.ndarray:
    """
    Hierarchical Risk Parity (HRP) weights via López de Prado (2016).

    Three-step algorithm:
      1. Correlation → distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))
      2. Hierarchical clustering (single linkage) → quasi-diagonalised leaf order
      3. Recursive bisection: split each cluster, allocate capital inversely
         proportional to sub-cluster variance until every asset is a leaf.

    Why HRP over ERC (risk_parity_weights)?
    ─────────────────────────────────────────
    ERC solves a fixed-point iteration using Cov @ w in the denominator.
    For small T/N (≈ 6 in our 126-day / 21-asset case) even Ledoit-Wolf
    shrinkage leaves small eigenvalues that can make marginal risk contributions
    numerically unstable.

    HRP bypasses matrix inversion entirely:
      • Tree structure comes from pairwise distances (robust to noise)
      • Variance estimates use only diagonal elements of covariance sub-blocks
        (single number per sub-cluster — no inversion required)
    Result: weights are numerically well-behaved for any N, T regime.

    Reference: López de Prado (2016), "Building Diversified Portfolios That
    Outperform Out of Sample", Journal of Portfolio Management 42(4).

    Args:
        cov:        Covariance matrix (N × N).  May be daily or annualised —
                    scalar scaling cancels inside the bisection ratio.
        max_weight: Per-asset weight cap (default 0.30 = 30%).

    Returns:
        Weight vector (N,): non-negative, sums to 1, each ≤ max_weight.
        Falls back to inverse-vol weights if scipy is unavailable.
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except ImportError:
        # Graceful degradation: inverse-vol (same as ERC initialisation)
        sig = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        w   = 1.0 / sig
        w  /= w.sum()
        return w

    # ── Step 1: Correlation → distance ──────────────────────────────────────
    sig  = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(sig, sig)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # LdP (2016) eq. 4.1 — maps corr ∈ [-1,1] to distance ∈ [0,1]
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)

    # ── Step 2: Hierarchical clustering → ordered leaf sequence ────────────
    condensed = squareform(dist, checks=False)
    link      = linkage(condensed, method="single")   # single linkage (LdP)
    sort_ix   = list(leaves_list(link))               # quasi-diagonalised order

    # ── Step 3: Recursive bisection ─────────────────────────────────────────
    w = np.ones(n)   # weights accumulate multiplicatively through bisections

    def _cluster_var(items: list) -> float:
        """Equal-weight variance of a sub-portfolio (uses diagonal + off-diag)."""
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
            return   # degenerate — skip, weights unchanged

        # alpha = fraction allocated to left sub-cluster
        # (right sub-cluster gets 1 - alpha)
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

    # ── Per-asset cap, iterative redistribution ──────────────────────────────
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
    """
    Print covariance shrinkage diagnostics and compare weighting schemes
    using the most recent 126 days of returns from data/features/.
    """
    from .data_pipeline import TICKER_LIST, ASSET_CLASS

    FEATURE_DIR = Path("data/features")

    returns = pd.DataFrame()
    for ticker in TICKER_LIST:
        path = FEATURE_DIR / f"{ticker}.parquet"
        if path.exists():
            returns[ticker] = pd.read_parquet(path)["log_return"]

    returns = returns.dropna()
    window  = returns.iloc[-126:]   # most recent 126 trading days

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
