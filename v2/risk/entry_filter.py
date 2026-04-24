"""
v2/risk/entry_filter.py
-----------------------
RSI entry filter for regime transitions, ported from V1.

V1 uses RSI_14 > 70 as a "skip new long entries" filter at the ticker level.
In V2 we have a regime-rotation system that shifts allocations on every
monthly rebalance, which can demand entry into an ETF that's just had a
blowoff rally. This filter delays new exposure for one month when:

  1. The ticker's weight is going UP (new or increased allocation), AND
  2. RSI_14 on that ticker is > 70 on the rebalance date.

When both conditions hit, the incremental exposure is held back to cash
(SHY) for the month; the rest of the position is sized as targeted. This
is a soft delay, not a blocker — next rebalance the filter re-evaluates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RSI_LOOKBACK = 14
RSI_THRESHOLD = 75.0   # looser than V1's 70 — monthly rebalance already filters whipsaws
CASH_TICKER = "SHY"


def rsi(close: pd.Series, period: int = RSI_LOOKBACK) -> pd.Series:
    """Classic Wilder RSI over `period` bars."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def apply_rsi_entry_filter(
    target_weights: dict[str, float],
    prev_weights: dict[str, float],
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    rsi_threshold: float = RSI_THRESHOLD,
    cash_ticker: str = CASH_TICKER,
) -> dict[str, float]:
    """
    For any ticker whose target weight is HIGHER than its previous weight AND
    whose RSI_14 at `as_of` is above `rsi_threshold`, cap the increase at the
    previous weight and redirect the blocked increment to cash.
    """
    adjusted = dict(target_weights)
    blocked = 0.0

    for ticker, new_w in target_weights.items():
        if ticker == cash_ticker or ticker not in prices.columns:
            continue
        prev_w = prev_weights.get(ticker, 0.0)
        if new_w <= prev_w:
            continue  # reducing or holding — no filter applies

        px = prices[ticker].loc[prices.index <= as_of]
        if len(px) < RSI_LOOKBACK + 2:
            continue
        rsi_at = rsi(px).iloc[-1]
        if not np.isfinite(rsi_at) or rsi_at <= rsi_threshold:
            continue

        # Cap the increase; the blocked increment goes to cash
        increment = new_w - prev_w
        adjusted[ticker] = prev_w
        blocked += increment

    if blocked > 0:
        adjusted[cash_ticker] = adjusted.get(cash_ticker, 0.0) + blocked
    return adjusted


def apply_rsi_entry_filter_df(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    rsi_threshold: float = RSI_THRESHOLD,
    cash_ticker: str = CASH_TICKER,
) -> pd.DataFrame:
    """Apply the filter to an entire monthly weights time-series in sequence."""
    out = weights_df.copy().astype(float)
    prev = {t: 0.0 for t in weights_df.columns}

    for date in weights_df.index:
        tgt = weights_df.loc[date].to_dict()
        filtered = apply_rsi_entry_filter(
            target_weights=tgt,
            prev_weights=prev,
            prices=prices,
            as_of=date,
            rsi_threshold=rsi_threshold,
            cash_ticker=cash_ticker,
        )
        for t, w in filtered.items():
            if t in out.columns:
                out.loc[date, t] = w
        prev = filtered

    # Renormalise rows
    row_sums = out.sum(axis=1).replace(0, np.nan)
    out = out.div(row_sums, axis=0).fillna(0.0)
    return out
