# ui/data_loaders.py
# Cached data-loading functions and Monte Carlo simulation.

import io
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_portfolio_curves() -> pd.DataFrame:
    """
    Load equity curves for all portfolio sizing methods from disk.

    Cached — only re-reads from disk when the TTL expires (Streamlit default
    session TTL; no explicit TTL is set so the cache lives for the session).

    Returns:
        DataFrame with columns [equal_weight, atr_sized, atr_pca_macro,
        eq_dd_control, vol_target, buy_hold] indexed by date.
        Returns empty DataFrame if the parquet is not found.
    """
    path = Path("data/results/portfolio_comparison.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data
def load_ticker_curves() -> dict:
    """
    Load per-ticker backtest equity curves from disk.

    Scans data/results/ for files matching *_curves.parquet, skipping
    the combined portfolio file.  Used by chart_asset_sharpe() to
    compare per-asset Strategy vs Buy & Hold Sharpe ratios.

    Returns:
        Dict mapping ticker name (str) → DataFrame with at least columns
        [regime, buy_hold] indexed by date.
        Returns empty dict if no matching parquets are found.
    """
    curves = {}
    for f in Path("data/results").glob("*_curves.parquet"):
        name = f.stem.replace("_curves", "")
        if name == "portfolio":
            continue
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        curves[name] = df
    return curves

@st.cache_data
def load_walk_forward() -> pd.DataFrame:
    """
    Load walk-forward OOS results for the equal-weight regime strategy.

    Cached — only re-reads from disk when the session TTL expires.

    Returns:
        DataFrame with columns [period, sharpe] where each row represents
        one walk-forward test window (3-year train / 1-year test).
        Returns empty DataFrame if the parquet is not found.
    """
    path = Path("data/results/walk_forward_regime.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_walk_forward_atr() -> pd.DataFrame:
    """
    Load walk-forward OOS results for the ATR+PCA+macro sizing method.

    Cached — only re-reads from disk when the session TTL expires.

    Returns:
        DataFrame with columns [period, sharpe] for the ATR+PCA+macro
        walk-forward test windows.
        Returns empty DataFrame if the parquet is not found.
    """
    path = Path("data/results/walk_forward_atr_pca.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_correlation_diagnostic() -> pd.DataFrame:
    path = Path("data/research/correlation_diagnostic.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_regime_correlation() -> pd.DataFrame:
    path = Path("data/research/correlation_diagnostic_regime_corr.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_dead_weight() -> pd.DataFrame:
    path = Path("data/research/correlation_diagnostic_dead_weight.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_oos_selection() -> pd.DataFrame:
    """
    Load the IS vs OOS Sharpe comparison table used to select the best
    portfolio sizing method.

    Cached — only re-reads from disk when the session TTL expires.

    Returns:
        DataFrame with columns [method, is_sharpe, oos_sharpe].
        The method with the highest oos_sharpe is highlighted as selected.
        Returns empty DataFrame if the parquet is not found.
    """
    path = Path("data/results/oos_selection.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_macro() -> pd.DataFrame:
    """
    Load precomputed macro features used in the ATR+PCA+macro strategy.

    Cached — only re-reads from disk when the session TTL expires.

    Returns:
        DataFrame indexed by date with columns including [vix, yield_curve]
        (and potentially others such as pca scores).  Used by
        chart_macro_overlay() to display VIX and yield-curve panels
        alongside the portfolio equity curve.
        Returns empty DataFrame if the parquet is not found.
    """
    path = Path("data/macro/macro_features.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def load_fred_features() -> pd.DataFrame:
    """
    Load precomputed FRED indicator features from fred_features.parquet.

    Returns DataFrame with raw levels, z-scores, regime dummies, and
    fred_macro_score.  Returns empty DataFrame if the file doesn't exist.
    """
    path = Path("data/macro/fred_features.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


# ── Monte Carlo (block bootstrap) ────────────────────────────────────────────
@st.cache_data
def run_monte_carlo(returns_bytes: bytes, n_paths: int = 1000,
                    block_size: int = 21, seed: int = 42) -> np.ndarray:
    """
    Block bootstrap — resample 21-day blocks with replacement to preserve
    short-term autocorrelation structure. Returns array (n_paths × n_days)
    of equity curves normalised to start at 1.0.
    """
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
