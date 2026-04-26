"""
v2/risk/trend_filter.py
-----------------------
Asset-level trend filter applied after regime allocation.

For each risky asset (equity, sector, real_asset, commodity) we check the
closing price against its 200d and 100d simple moving averages:

    px > MA200                  →  full weight
    MA100 < px < MA200          →  50% weight (choppy / topping)
    px < MA100 AND px < MA200   →  0% weight (confirmed downtrend)

Removed weight is redirected to SHY (cash proxy). Fixed-income and
currency ETFs are exempt because they typically deliver positive returns
even below long-MA (carry + roll).
"""

from __future__ import annotations

import pandas as pd

from v2.pipeline.data_pipeline import get_asset_class_map

# Asset classes where trend filter applies. Fixed income + currency are exempt.
TREND_FILTER_CLASSES = {"equity", "sector", "real_asset", "commodity"}

MA_LONG = 200
MA_SHORT = 100
CHOPPY_MULT = 0.5
DOWNTREND_MULT = 0.0
CASH_TICKER = "SHY"


def compute_ma_state(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return DataFrames for 'above_long', 'above_short' (same shape as prices)."""
    ma_long = prices.rolling(MA_LONG, min_periods=MA_LONG // 2).mean()
    ma_short = prices.rolling(MA_SHORT, min_periods=MA_SHORT // 2).mean()
    return {
        "above_long": prices > ma_long,
        "above_short": prices > ma_short,
    }


def apply_trend_filter_df(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    cash_ticker: str = CASH_TICKER,
) -> pd.DataFrame:
    """
    Apply trend filter to monthly weights. Weights for trend-filtered assets
    are scaled by:
        1.0   if px > MA200
        0.5   if MA100 < px < MA200
        0.0   if px < MA100 AND px < MA200

    Removed weight flows to SHY.
    """
    ac_map = get_asset_class_map()
    risky = {t for t in weights_df.columns if ac_map.get(t) in TREND_FILTER_CLASSES}
    if not risky:
        return weights_df

    state = compute_ma_state(prices)
    above_long = state["above_long"]
    above_short = state["above_short"]

    # Reindex to weights_df dates using latest-known state at or before each date
    above_long_w = above_long.reindex(weights_df.index, method="ffill").fillna(False)
    above_short_w = above_short.reindex(weights_df.index, method="ffill").fillna(False)

    result = weights_df.copy().astype(float)

    for ticker in risky:
        if ticker not in result.columns:
            continue
        if ticker not in above_long_w.columns:
            continue

        long_ok = above_long_w[ticker]
        short_ok = above_short_w[ticker]

        # Default multiplier = 1.0
        mult = pd.Series(1.0, index=result.index)
        # Choppy: above short MA but below long MA
        choppy = (~long_ok) & short_ok
        mult.loc[choppy] = CHOPPY_MULT
        # Downtrend: below both
        downtrend = (~long_ok) & (~short_ok)
        mult.loc[downtrend] = DOWNTREND_MULT

        original = result[ticker].copy()
        result[ticker] = original * mult
        removed = original - result[ticker]
        # Redirect removed weight to cash
        if cash_ticker in result.columns:
            result[cash_ticker] = result[cash_ticker] + removed

    # Renormalize to sum to 1.0
    row_sums = result.sum(axis=1).replace(0, 1.0)
    result = result.div(row_sums, axis=0)
    return result
