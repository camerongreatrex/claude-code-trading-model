"""
v2/portfolio.py
---------------
Dollar-neutral long/short portfolio construction with beta hedge and
sector constraints.

Construction rules
──────────────────
  1. Long top decile (~45 stocks), equal-weighted within long book
  2. Short bottom decile (~45 stocks), equal-weighted within short book
  3. Beta hedge: compute net portfolio beta to SPY, overlay SPY position
     to bring net beta to zero
  4. Sector constraint: max ±5% net per GICS sector
  5. Position cap: no single stock > 3% of portfolio

Exposure targets
────────────────
  Gross: 100% (50% long + 50% short) — scale if no leverage available
  Net market: 0% (±0.05 beta tolerance)
  Net sector: max ±5% per GICS sector
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import V2_FEATURES

V2_RESULTS_DIR = Path("data/v2/results")
V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CAPITAL = 100_000
SELECTION_FRACTION = 0.10       # top/bottom 10% (decile) for long/short
MAX_POSITION_PCT = 0.03         # 3% max per individual position
MAX_SECTOR_NET = 0.05           # ±5% max net sector exposure
BETA_TOLERANCE = 0.05           # acceptable |net beta| before hedging
LONG_GROSS_TARGET = 0.50        # 50% of capital on long side
SHORT_GROSS_TARGET = 0.50       # 50% of capital on short side

# Long-extension mode constants
LONG_EXT_LONG_TARGET = 1.30     # 130% long
LONG_EXT_SPY_HEDGE = 0.30       # 30% short SPY


def compute_rolling_betas(
    returns: pd.DataFrame,
    spy_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling OLS beta of each stock to SPY.

    beta_i = cov(r_i, r_SPY) / var(r_SPY) over trailing `window` days.
    """
    spy_var = spy_returns.rolling(window).var()
    betas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for ticker in returns.columns:
        if ticker == "SPY":
            betas[ticker] = 1.0
            continue
        cov = returns[ticker].rolling(window).cov(spy_returns)
        betas[ticker] = cov / spy_var

    return betas


def construct_portfolio(
    composite_scores: pd.DataFrame,
    returns: pd.DataFrame,
    sectors: dict,
    date: pd.Timestamp,
    betas: pd.DataFrame | None = None,
    spy_returns: pd.Series | None = None,
    gross_scale: float = 1.0,
) -> dict:
    """
    Construct a dollar-neutral portfolio for a single rebalance date.

    Routes to long-extension mode if V2_FEATURES['long_extension_mode'] is True.

    Args:
        composite_scores: T × N composite z-scores
        returns: T × N daily return matrix
        sectors: dict ticker -> GICS sector
        date: rebalance date
        betas: pre-computed rolling betas (optional)
        spy_returns: SPY daily return series (for beta computation if betas not provided)
        gross_scale: scale factor for gross exposure (from vol targeting)

    Returns:
        dict with keys:
            'long_weights': {ticker: weight} — positive weights
            'short_weights': {ticker: weight} — negative weights
            'spy_hedge': float — SPY overlay weight (positive = long SPY, negative = short)
            'net_beta': float — portfolio net beta after hedge
            'gross_exposure': float
            'net_exposure': float
            'sector_net': {sector: net_weight}
    """
    if V2_FEATURES.get("long_extension_mode"):
        return _construct_long_extension(
            composite_scores, returns, sectors, date,
            betas=betas, spy_returns=spy_returns, gross_scale=gross_scale,
        )

    if date not in composite_scores.index:
        return _empty_portfolio()

    scores = composite_scores.loc[date].dropna()
    # Exclude ETFs from the cross-sectional ranking
    stock_scores = scores.drop(["SPY", "TLT", "GLD", "UUP"], errors="ignore")

    n_stocks = len(stock_scores)
    if n_stocks < 20:
        return _empty_portfolio()

    n_select = max(5, int(n_stocks * SELECTION_FRACTION))

    # Rank: highest score = long, lowest = short
    ranked = stock_scores.sort_values(ascending=False)
    long_tickers = ranked.head(n_select).index.tolist()
    short_tickers = ranked.tail(n_select).index.tolist()

    # Equal weight within each book
    long_weight = (LONG_GROSS_TARGET * gross_scale) / n_select
    short_weight = (SHORT_GROSS_TARGET * gross_scale) / n_select

    # Cap individual positions
    long_weight = min(long_weight, MAX_POSITION_PCT)
    short_weight = min(short_weight, MAX_POSITION_PCT)

    long_weights = {t: long_weight for t in long_tickers}
    short_weights = {t: -short_weight for t in short_tickers}

    # Sector constraint: clip net sector exposure to ±MAX_SECTOR_NET
    sector_limit = 0.02 if V2_FEATURES.get("tightened_sector_constraint") else MAX_SECTOR_NET
    long_weights, short_weights = _apply_sector_constraints(
        long_weights, short_weights, sectors, max_net=sector_limit,
    )

    # Compute beta hedge
    net_beta = 0.0
    spy_hedge = 0.0

    if betas is not None and date in betas.index:
        stock_betas = betas.loc[date]
        long_beta = sum(long_weights.get(t, 0) * stock_betas.get(t, 1.0) for t in long_weights)
        short_beta = sum(short_weights.get(t, 0) * stock_betas.get(t, 1.0) for t in short_weights)
        net_beta = long_beta + short_beta

        if abs(net_beta) > BETA_TOLERANCE:
            # Overlay SPY to zero out beta
            spy_hedge = -net_beta
            net_beta = 0.0

    # Compute exposures
    gross_long = sum(long_weights.values())
    gross_short = sum(abs(v) for v in short_weights.values())
    gross_exposure = gross_long + gross_short + abs(spy_hedge)
    net_exposure = gross_long - gross_short + spy_hedge

    # Sector net exposures
    sector_net = _compute_sector_net(long_weights, short_weights, sectors)

    return {
        "long_weights": long_weights,
        "short_weights": short_weights,
        "long_tickers": long_tickers,
        "short_tickers": short_tickers,
        "spy_hedge": spy_hedge,
        "net_beta": net_beta,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "sector_net": sector_net,
        "n_long": len(long_weights),
        "n_short": len(short_weights),
    }


def _construct_long_extension(
    composite_scores: pd.DataFrame,
    returns: pd.DataFrame,
    sectors: dict,
    date: pd.Timestamp,
    betas: pd.DataFrame | None = None,
    spy_returns: pd.Series | None = None,
    gross_scale: float = 1.0,
) -> dict:
    """
    Long-extension portfolio: long top decile at 130%, short SPY at 30%.
    No individual stock shorts — eliminates borrow costs and squeeze risk.
    """
    if date not in composite_scores.index:
        return _empty_portfolio()

    scores = composite_scores.loc[date].dropna()
    stock_scores = scores.drop(["SPY", "TLT", "GLD", "UUP"], errors="ignore")

    n_stocks = len(stock_scores)
    if n_stocks < 20:
        return _empty_portfolio()

    n_select = max(5, int(n_stocks * SELECTION_FRACTION))

    ranked = stock_scores.sort_values(ascending=False)
    long_tickers = ranked.head(n_select).index.tolist()

    # Equal weight: 130% gross long across top decile
    long_weight = (LONG_EXT_LONG_TARGET * gross_scale) / n_select
    long_weight = min(long_weight, MAX_POSITION_PCT * 2)  # relax cap slightly for 130%

    long_weights = {t: long_weight for t in long_tickers}

    # SPY hedge: short 30% of capital in SPY
    # Fine-tune with beta: compute portfolio beta and set SPY hedge to neutralize
    spy_hedge = -(LONG_EXT_SPY_HEDGE * gross_scale)

    if betas is not None and date in betas.index:
        stock_betas = betas.loc[date]
        long_beta = sum(long_weights.get(t, 0) * stock_betas.get(t, 1.0) for t in long_weights)
        # SPY beta = 1.0, so to neutralize: spy_hedge = -long_beta
        spy_hedge = -long_beta

    net_beta = 0.0  # by construction

    gross_long = sum(long_weights.values())
    gross_exposure = gross_long + abs(spy_hedge)
    net_exposure = gross_long + spy_hedge

    sector_net = {}
    for t, w in long_weights.items():
        s = sectors.get(t, "Unknown")
        sector_net[s] = sector_net.get(s, 0) + w

    return {
        "long_weights": long_weights,
        "short_weights": {},  # no individual shorts
        "long_tickers": long_tickers,
        "short_tickers": [],
        "spy_hedge": spy_hedge,
        "net_beta": net_beta,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "sector_net": sector_net,
        "n_long": len(long_weights),
        "n_short": 0,
    }


def _apply_sector_constraints(
    long_weights: dict,
    short_weights: dict,
    sectors: dict,
    max_net: float | None = None,
) -> tuple[dict, dict]:
    """
    Clip net sector exposure to ±max_net.

    If a sector's net weight exceeds the limit, scale down the overweight
    side proportionally.
    """
    limit = max_net if max_net is not None else MAX_SECTOR_NET

    sector_long = {}
    sector_short = {}

    for t, w in long_weights.items():
        s = sectors.get(t, "Unknown")
        sector_long[s] = sector_long.get(s, 0) + w

    for t, w in short_weights.items():
        s = sectors.get(t, "Unknown")
        sector_short[s] = sector_short.get(s, 0) + w  # w is negative

    all_sectors = set(sector_long) | set(sector_short)
    for sector in all_sectors:
        net = sector_long.get(sector, 0) + sector_short.get(sector, 0)
        if abs(net) > limit:
            # Scale down the overweight side
            if net > limit:
                # Too much long net — reduce long weights in this sector
                excess = net - limit
                sector_tickers = [t for t in long_weights if sectors.get(t) == sector]
                total_long = sum(long_weights[t] for t in sector_tickers)
                if total_long > 0:
                    scale = max(0, 1 - excess / total_long)
                    for t in sector_tickers:
                        long_weights[t] *= scale
            elif net < -limit:
                # Too much short net — reduce short weights (make less negative)
                excess = abs(net) - limit
                sector_tickers = [t for t in short_weights if sectors.get(t) == sector]
                total_short = sum(abs(short_weights[t]) for t in sector_tickers)
                if total_short > 0:
                    scale = max(0, 1 - excess / total_short)
                    for t in sector_tickers:
                        short_weights[t] *= scale

    return long_weights, short_weights


def _compute_sector_net(long_weights: dict, short_weights: dict, sectors: dict) -> dict:
    """Compute net weight per sector."""
    sector_net = {}
    for t, w in {**long_weights, **short_weights}.items():
        s = sectors.get(t, "Unknown")
        sector_net[s] = sector_net.get(s, 0) + w
    return sector_net


def _empty_portfolio() -> dict:
    return {
        "long_weights": {},
        "short_weights": {},
        "long_tickers": [],
        "short_tickers": [],
        "spy_hedge": 0.0,
        "net_beta": 0.0,
        "gross_exposure": 0.0,
        "net_exposure": 0.0,
        "sector_net": {},
        "n_long": 0,
        "n_short": 0,
    }


def build_portfolio_history(
    composite_scores: pd.DataFrame,
    returns: pd.DataFrame,
    sectors: dict,
    betas: pd.DataFrame | None = None,
    spy_returns: pd.Series | None = None,
    gross_scale_series: pd.Series | None = None,
) -> list[tuple[pd.Timestamp, dict]]:
    """
    Build portfolio at each weekly rebalance date.

    Returns list of (date, portfolio_dict) tuples.
    """
    # Get Friday dates from composite scores
    rebalance_dates = composite_scores.index[composite_scores.index.dayofweek == 4]

    # If scores were forward-filled to daily, get unique Friday values
    if len(rebalance_dates) == 0:
        all_dates = composite_scores.index
        rebalance_dates = all_dates[all_dates.dayofweek == 4]

    history = []
    for date in rebalance_dates:
        gross_scale = 1.0
        if gross_scale_series is not None and date in gross_scale_series.index:
            gross_scale = gross_scale_series.loc[date]

        portfolio = construct_portfolio(
            composite_scores, returns, sectors, date,
            betas=betas, spy_returns=spy_returns, gross_scale=gross_scale,
        )
        history.append((date, portfolio))

    return history
