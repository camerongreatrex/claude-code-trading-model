"""
v2/signals/momentum_overlay.py
------------------------------
Cross-asset trend filter / momentum overlay.

Kills persistent underperformers from the regime-based allocation.
Uses time-series momentum (absolute returns) not cross-sectional.

Methodology
───────────
  For each ETF:
    1. Compute trailing 12-month return (skipping most recent month)
    2. Compute trailing 1-month, 3-month, 6-month returns
    3. Composite momentum score = weighted average of lookbacks
    4. If composite < 0 AND 1-month < 0 (confirming trend):
       → penalize weight by 50%
    5. If composite < -10% AND all lookbacks negative:
       → kill weight entirely (set to 0)
    6. Redistribute removed weight to SHY (cash proxy)

This prevents the regime model from holding deeply trending-down
assets just because the regime says to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum_scores(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite momentum score for each ETF.

    Returns DataFrame with same shape as prices, values are momentum scores.
    Positive = trending up, negative = trending down.
    """
    # Returns over various lookbacks (skip most recent 21 days for 12m)
    ret_1m = prices.pct_change(21)
    ret_3m = prices.pct_change(63)
    ret_6m = prices.pct_change(126)
    ret_12m = prices.shift(21).pct_change(252 - 21)  # 12m skip recent month

    # Weighted composite: favor medium-term (3m, 6m)
    composite = (
        0.15 * ret_1m.fillna(0) +
        0.25 * ret_3m.fillna(0) +
        0.35 * ret_6m.fillna(0) +
        0.25 * ret_12m.fillna(0)
    )

    return composite


def apply_momentum_overlay(
    weights: dict[str, float],
    momentum_scores: pd.Series,
    returns_1m: pd.Series,
    cash_ticker: str = "SHY",
) -> dict[str, float]:
    """
    Apply momentum filter to a single-day allocation.

    Args:
        weights: ticker → weight (from regime blending)
        momentum_scores: ticker → composite momentum score for this date
        returns_1m: ticker → trailing 1-month return for this date
        cash_ticker: where to redirect removed weight

    Returns:
        Adjusted weights dict.
    """
    adjusted = {}
    removed_weight = 0.0

    for ticker, weight in weights.items():
        if ticker == cash_ticker or weight <= 0:
            adjusted[ticker] = weight
            continue

        mom = momentum_scores.get(ticker, 0.0)
        ret1m = returns_1m.get(ticker, 0.0)

        if np.isnan(mom):
            mom = 0.0
        if np.isnan(ret1m):
            ret1m = 0.0

        # Strong negative trend: kill entirely
        if mom < -0.10 and ret1m < 0:
            removed_weight += weight
            adjusted[ticker] = 0.0
        # Mild negative trend: penalize by 50%
        elif mom < 0 and ret1m < 0:
            penalty = weight * 0.5
            removed_weight += penalty
            adjusted[ticker] = weight - penalty
        else:
            adjusted[ticker] = weight

    # Redirect to cash
    adjusted[cash_ticker] = adjusted.get(cash_ticker, 0.0) + removed_weight

    return adjusted


def apply_momentum_overlay_df(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    cash_ticker: str = "SHY",
) -> pd.DataFrame:
    """
    Apply momentum overlay to a full time-series of weights.

    Args:
        weights_df: DatetimeIndex × tickers, each row sums to 1.0
        prices: DatetimeIndex × tickers, adjusted close prices
        cash_ticker: where to redirect removed weight

    Returns:
        Adjusted weights DataFrame.
    """
    momentum = compute_momentum_scores(prices)
    ret_1m = prices.pct_change(21)

    result = weights_df.copy()

    for date in weights_df.index:
        if date not in momentum.index:
            continue

        row_weights = weights_df.loc[date].to_dict()
        mom_scores = momentum.loc[date]
        r1m = ret_1m.loc[date] if date in ret_1m.index else pd.Series(dtype=float)

        adjusted = apply_momentum_overlay(row_weights, mom_scores, r1m, cash_ticker)

        for ticker, weight in adjusted.items():
            if ticker in result.columns:
                result.loc[date, ticker] = weight

    return result
