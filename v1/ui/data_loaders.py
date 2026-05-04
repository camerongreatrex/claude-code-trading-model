# ui/data_loaders.py
# Cached loaders and Monte Carlo bootstrap.

import io
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_portfolio_curves() -> pd.DataFrame:
    """Load equity curves for all sizing methods (date-indexed). Empty if parquet missing."""
    path = Path("data/v1/results/portfolio_comparison.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data
def load_ticker_curves() -> dict:
    """Load per-ticker backtest curves (*_curves.parquet, excluding portfolio).
    Returns dict[ticker → DataFrame]; used by chart_asset_sharpe()."""
    curves = {}
    for f in Path("data/v1/results").glob("*_curves.parquet"):
        name = f.stem.replace("_curves", "")
        if name == "portfolio":
            continue
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        curves[name] = df
    return curves

@st.cache_data
def load_walk_forward() -> pd.DataFrame:
    """Walk-forward OOS results (equal-weight regime), 3y train / 1y test.
    Columns [period, sharpe]; empty if parquet missing."""
    path = Path("data/v1/results/walk_forward_regime.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_walk_forward_atr() -> pd.DataFrame:
    """Walk-forward OOS results (ATR+PCA+macro). Columns [period, sharpe]; empty if missing."""
    path = Path("data/v1/results/walk_forward_atr_pca.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_correlation_diagnostic() -> pd.DataFrame:
    path = Path("data/v1/research/correlation_diagnostic.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_regime_correlation() -> pd.DataFrame:
    path = Path("data/v1/research/correlation_diagnostic_regime_corr.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_dead_weight() -> pd.DataFrame:
    path = Path("data/v1/research/correlation_diagnostic_dead_weight.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_oos_selection() -> pd.DataFrame:
    """IS vs OOS Sharpe table (method, is_sharpe, oos_sharpe). Highest oos_sharpe is selected."""
    path = Path("data/v1/results/oos_selection.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_macro() -> pd.DataFrame:
    """Macro features for ATR+PCA+macro strategy (vix, yield_curve, pca scores).
    Used by chart_macro_overlay(); empty if parquet missing."""
    path = Path("data/shared/macro/macro_features.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def load_fred_features() -> pd.DataFrame:
    """FRED features (levels, z-scores, regime dummies, fred_macro_score). Empty if missing."""
    path = Path("data/shared/macro/fred_features.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


# ── Monte Carlo (block bootstrap) ────────────────────────────────────────────
@st.cache_data
def run_monte_carlo(returns_bytes: bytes, n_paths: int = 1000,
                    block_size: int = 21, seed: int = 42) -> np.ndarray:
    """Block bootstrap: resample 21-day blocks with replacement (preserves
    short-term autocorr). Returns (n_paths × n_days) curves starting at 1.0."""
    ret = pd.read_parquet(io.BytesIO(returns_bytes)).values.ravel()
    n   = len(ret)
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n))
    for i in range(n_paths):
        sampled: list = []
        while len(sampled) < n:
            s = rng.integers(0, max(1, n - block_size))
            sampled.extend(ret[s: s + block_size].tolist())
        paths[i] = sampled[:n]
    return np.cumprod(1 + paths, axis=1)
