"""
v2/risk/covariance.py
---------------------
Shrunk covariance estimation and Hierarchical Risk Parity weighting.

Ported from V1 (v1/risk/risk_model.py) for use inside the regime-conditioned
top-N allocator. Inside each regime, V2 picks N best-Sharpe ETFs and
proportionally weights them — but two top-N names can be 0.85+ correlated
(QQQ + IWF + XLK), so the apparent diversification is illusory. HRP splits
correlated clusters via a distance-tree, allocating less to redundant names
without inverting the covariance matrix.

Used by sharpe_weighted.top_n_weights as a drop-in for the proportional
score-based weighting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.covariance import LedoitWolf as _LedoitWolf
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _ledoit_wolf_analytical(X: np.ndarray) -> np.ndarray:
    """RBLW analytical Ledoit-Wolf shrinkage toward scaled identity."""
    n, p = X.shape
    S = (X.T @ X) / n
    tr_S = np.trace(S)
    tr_S2 = np.trace(S @ S)
    denom = (n + 2) * (tr_S2 - tr_S ** 2 / p)
    if abs(denom) < 1e-15:
        return S
    rho = min(1.0, ((n - 2) / n * tr_S2 + tr_S ** 2) / denom)
    mu = tr_S / p
    return (1.0 - rho) * S + rho * mu * np.eye(p)


def estimate_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Annualised Ledoit-Wolf shrunk covariance from daily returns."""
    clean = returns.dropna(axis=1, how="all").ffill().fillna(0)
    if clean.shape[1] == 0 or clean.shape[0] < 5:
        return np.eye(max(clean.shape[1], 1))
    X = clean.values - clean.values.mean(axis=0)
    if _HAS_SKLEARN:
        lw = _LedoitWolf(assume_centered=True)
        lw.fit(X)
        cov = lw.covariance_
    else:
        cov = _ledoit_wolf_analytical(X)
    return cov * 252


def hrp_weights(cov: np.ndarray, max_weight: float = 0.40) -> np.ndarray:
    """
    Hierarchical Risk Parity (López de Prado 2016).

    1. Correlation -> distance: d_ij = sqrt(0.5 * (1 - rho_ij))
    2. Single-linkage clustering -> quasi-diagonalised leaf order
    3. Recursive bisection: split cluster, allocate inversely to sub-variance.

    No matrix inversion. Robust to small T/N. Falls back to inverse-vol if
    scipy is missing or the matrix is degenerate.
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    sig = np.sqrt(np.maximum(np.diag(cov), 1e-12))

    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except ImportError:
        w = 1.0 / sig
        return w / w.sum()

    corr = cov / np.outer(sig, sig)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)

    try:
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")
        sort_ix = list(leaves_list(link))
    except (ValueError, IndexError):
        w = 1.0 / sig
        return w / w.sum()

    w = np.ones(n)

    def _cluster_var(items: list) -> float:
        sub = np.array(items, dtype=int)
        eq = np.ones(len(sub)) / len(sub)
        return float(eq @ cov[np.ix_(sub, sub)] @ eq)

    def _bisect(items: list) -> None:
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left, right = items[:mid], items[mid:]
        var_l = _cluster_var(left)
        var_r = _cluster_var(right)
        total = var_l + var_r
        if total < 1e-15:
            return
        alpha = 1.0 - var_l / total
        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= (1.0 - alpha)
        _bisect(left)
        _bisect(right)

    _bisect(sort_ix)

    s = w.sum()
    if s > 1e-10:
        w /= s

    if max_weight < 1.0:
        for _ in range(20):
            w = np.clip(w, 0.0, max_weight)
            s = w.sum()
            if s > 1e-10:
                w /= s
            if w.max() <= max_weight + 1e-9:
                break
    return w


def hrp_weights_for_subset(
    prices: pd.DataFrame,
    tickers: list[str],
    as_of: pd.Timestamp,
    lookback_days: int = 252,
    max_weight: float = 0.40,
) -> dict[str, float]:
    """
    Convenience wrapper: take a price panel + ticker subset + as-of date,
    return {ticker: hrp_weight} using a Ledoit-Wolf-shrunk covariance from
    the trailing `lookback_days` of daily returns strictly before as_of.

    Falls back to equal-weight if the subset is empty or has insufficient
    overlapping history.
    """
    if not tickers:
        return {}
    cols = [t for t in tickers if t in prices.columns]
    if not cols:
        return {}
    if len(cols) == 1:
        return {cols[0]: 1.0}

    px = prices[cols].loc[prices.index < as_of]
    if len(px) < 60:
        w = 1.0 / len(cols)
        return {t: w for t in cols}

    rets = px.tail(lookback_days).pct_change().dropna(how="all").fillna(0.0)
    # Drop tickers with no variance in the window (bad data / new ETF)
    nonzero = rets.std() > 1e-8
    rets = rets.loc[:, nonzero[nonzero].index]
    if rets.shape[1] == 0:
        w = 1.0 / len(cols)
        return {t: w for t in cols}
    if rets.shape[1] == 1:
        out = {t: 0.0 for t in cols}
        out[rets.columns[0]] = 1.0
        return out

    cov = estimate_covariance(rets)
    w = hrp_weights(cov, max_weight=max_weight)
    out = {t: 0.0 for t in cols}
    for t, wt in zip(rets.columns, w):
        out[t] = float(wt)
    return out
