# v2/ui/data_loaders.py
# Cached data-loading functions for the v2 macro regime rotation dashboard.

import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


RESULTS_DIR = Path("data/v2/results")
REGIME_DIR  = Path("data/v2/regime_features")
PAPER_DIR   = Path("data/v2/paper_trading")


@st.cache_data
def load_equity_curves() -> pd.DataFrame:
    """strategy_gross / strategy_net / spy equity curves."""
    path = RESULTS_DIR / "equity_curves.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def load_net_returns() -> pd.Series:
    path = RESULTS_DIR / "net_returns.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.iloc[:, 0]


@st.cache_data
def load_portfolio_weights() -> pd.DataFrame:
    path = RESULTS_DIR / "portfolio_weights.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def load_walk_forward() -> pd.DataFrame:
    path = RESULTS_DIR / "walk_forward_results.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_regime_probabilities() -> pd.DataFrame:
    path = REGIME_DIR / "regime_probabilities.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def load_regime_labels() -> pd.Series:
    path = REGIME_DIR / "regime_labels.parquet"
    if not path.exists():
        return pd.Series(dtype=int)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df["regime"]


@st.cache_data
def load_macro_features() -> pd.DataFrame:
    path = REGIME_DIR / "macro_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def load_market_features() -> pd.DataFrame:
    path = REGIME_DIR / "market_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def load_etf_prices() -> pd.DataFrame:
    path = RESULTS_DIR / "etf_prices.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def load_paper_state() -> dict:
    path = PAPER_DIR / "state.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_paper_history() -> pd.DataFrame:
    path = PAPER_DIR / "history.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"]) if "date" in pd.read_csv(path, nrows=0).columns else pd.read_csv(path)
    return df


@st.cache_data
def load_paper_trades() -> pd.DataFrame:
    path = PAPER_DIR / "trades.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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
