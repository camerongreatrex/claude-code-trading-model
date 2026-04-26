"""
v2/portfolio/cross_momentum.py
------------------------------
Cross-sectional momentum sleeve for blending with the regime book.

Every rebalance month, rank the risky-asset universe (equity + sector +
real_asset + commodity) by 12-1 momentum (trailing 12m return skipping
last month). Allocate equal weights to the top N names; park the rest in
SHY. The sleeve is intentionally simple — its value is its low
correlation to the macro-regime book, not standalone performance.

Usage: blend with regime weights at a user-set ratio in weights.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from v2.pipeline.data_pipeline import get_asset_class_map

RISKY_CLASSES = {"equity", "sector", "real_asset", "commodity", "factor"}
LOOKBACK = 252   # ~12 months of trading days
SKIP = 21        # skip most recent month
TOP_N = 4
VOL_LOOKBACK = 63   # 3m daily vol for inverse-vol weighting
INVERSE_VOL_FLOOR = 0.05   # min realized vol (annualized) to avoid div-by-zero
CASH_TICKER = "SHY"


def compute_momentum_rank(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Per-ticker 12-1 momentum as of `as_of`. NaN if insufficient history."""
    px = prices.loc[prices.index < as_of]
    if len(px) < LOOKBACK + 2:
        return pd.Series(np.nan, index=prices.columns)
    end_target = as_of - pd.Timedelta(days=SKIP)
    start_target = as_of - pd.Timedelta(days=LOOKBACK)
    end_idx = px.index[px.index <= end_target]
    start_idx = px.index[px.index <= start_target]
    if len(end_idx) == 0 or len(start_idx) == 0:
        return pd.Series(np.nan, index=prices.columns)
    return (px.loc[end_idx[-1]] / px.loc[start_idx[-1]]) - 1.0


def compute_realized_vol(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Per-ticker annualized realized vol over VOL_LOOKBACK days < as_of."""
    px = prices.loc[prices.index < as_of]
    if len(px) < VOL_LOOKBACK + 2:
        return pd.Series(np.nan, index=prices.columns)
    rets = px.tail(VOL_LOOKBACK + 1).pct_change().dropna(how="all")
    return rets.std() * np.sqrt(252)


def build_cross_momentum_sleeve(
    prices: pd.DataFrame,
    monthly_probs: pd.DataFrame,
    top_n: int = TOP_N,
    cash_ticker: str = CASH_TICKER,
) -> pd.DataFrame:
    """
    Walk-forward monthly cross-sectional momentum portfolio with inverse-vol
    weighting. Equal-vol-budgeted (not equal-dollar) selection prevents a
    single high-vol winner from dominating the sleeve's risk.

    Returns a DataFrame aligned with monthly_probs.index (month-end dates),
    columns = tickers from `prices`, each row sums to 1.0.
    """
    ac_map = get_asset_class_map()
    risky = [t for t in prices.columns if ac_map.get(t) in RISKY_CLASSES]

    rows = []
    universe = list(prices.columns)
    for date in monthly_probs.index:
        mom = compute_momentum_rank(prices, date).reindex(risky)
        mom = mom.dropna()
        if mom.empty:
            row = {t: 0.0 for t in universe}
            row[cash_ticker] = 1.0
        else:
            # Only buy positive-momentum names to avoid forcing longs in bear markets
            positive = mom[mom > 0]
            if positive.empty:
                row = {t: 0.0 for t in universe}
                row[cash_ticker] = 1.0
            else:
                top = positive.nlargest(min(top_n, len(positive)))
                # Inverse-vol weighting: w_i ∝ 1/σ_i, normalized to sum 1.
                # Falls back to equal-weight if vols unavailable.
                vols = compute_realized_vol(prices, date).reindex(top.index)
                vols = vols.where(vols > INVERSE_VOL_FLOOR, INVERSE_VOL_FLOOR)
                inv = 1.0 / vols
                if inv.isna().all() or inv.sum() <= 0:
                    weights_each = pd.Series(1.0 / len(top), index=top.index)
                else:
                    inv = inv.fillna(inv.mean())
                    weights_each = inv / inv.sum()
                row = {t: 0.0 for t in universe}
                for t in top.index:
                    row[t] = float(weights_each[t])
        row["Date"] = date
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Date").fillna(0.0)
    # Normalize (handles floating point)
    row_sums = df.sum(axis=1).replace(0, 1.0)
    df = df.div(row_sums, axis=0)
    return df


def blend_books(
    regime_df: pd.DataFrame,
    momentum_df: pd.DataFrame,
    momentum_weight: float,
) -> pd.DataFrame:
    """Linear blend of two weight books. Union of tickers; NaN → 0."""
    all_cols = regime_df.columns.union(momentum_df.columns)
    r = regime_df.reindex(columns=all_cols, fill_value=0.0)
    m = momentum_df.reindex(columns=all_cols, fill_value=0.0)
    common = r.index.intersection(m.index)
    blended = (1 - momentum_weight) * r.loc[common] + momentum_weight * m.loc[common]
    # Renormalize (should already sum to ~1.0)
    row_sums = blended.sum(axis=1).replace(0, 1.0)
    blended = blended.div(row_sums, axis=0)
    return blended
