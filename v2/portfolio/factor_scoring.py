"""
v2/portfolio/factor_scoring.py
------------------------------
Multi-factor composite scoring for regime-conditioned ETF selection.

This is our own deviation from QUANTT's pure regime-conditioned Sharpe.
We combine four signals, cross-sectionally z-scored inside each regime's
eligible universe:

    composite = w_s · z(sharpe_regime)
              + w_m12 · z(mom_12_1)
              + w_m6  · z(mom_6_1)
              - w_v   · z(vol_63d)

where the Sharpe term is regime-conditioned history (< as_of, regime==r)
and the momentum / vol terms are current (computed from daily prices as
of the rebalance date, independent of regime).

Rationale
─────────
    - regime_conditioned_sharpe tells us "this asset historically worked
      in this regime" — a slow-moving, regime-structural signal.
    - momentum_12_1 / momentum_6_1 are "this asset has a live tailwind"
      — fast-moving signals unconditional on regime.
    - vol_63d penalises assets whose risk has spiked right now, even if
      they screened well historically.

Combining them = the strategy reacts both to macro regime and to live
price action, instead of only historical regime-conditional behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Factor weights (sum to 1.0 ignoring the vol subtraction) ─────────────────
FACTOR_WEIGHTS = {
    "sharpe": 0.55,    # regime-conditioned historical Sharpe (dominant signal)
    "mom_12_1": 0.20,  # 12-month return skipping last month (classic momentum)
    "mom_6_1": 0.10,   # 6-month return skipping last month (faster momentum)
    "vol_pen": 0.15,   # subtracted — penalise high current volatility
}

MOM_LONG_LOOKBACK = 252   # ~12 months
MOM_SHORT_LOOKBACK = 126  # ~6 months
MOM_SKIP = 21             # skip most recent month (classic 12-1)
VOL_LOOKBACK = 63         # ~3 months of daily vol


def _z_score(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, NaN-safe. Returns zeros if no dispersion."""
    s = s.astype(float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def momentum_score(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = MOM_LONG_LOOKBACK,
    skip: int = MOM_SKIP,
) -> pd.Series:
    """
    Per-ticker return from (as_of − lookback) to (as_of − skip).

    Uses the latest available price at or before each target date so that
    month-end rebalance dates always get a value even when the exact day
    isn't a trading day.
    """
    px = prices.loc[prices.index < as_of]
    if px.empty:
        return pd.Series(np.nan, index=prices.columns)

    end_target = as_of - pd.Timedelta(days=skip)
    start_target = as_of - pd.Timedelta(days=lookback)

    end_idx = px.index[px.index <= end_target]
    start_idx = px.index[px.index <= start_target]
    if len(end_idx) == 0 or len(start_idx) == 0:
        return pd.Series(np.nan, index=prices.columns)

    end_px = px.loc[end_idx[-1]]
    start_px = px.loc[start_idx[-1]]
    return (end_px / start_px) - 1.0


def vol_score(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = VOL_LOOKBACK,
) -> pd.Series:
    """Annualised stdev of daily returns over the trailing `lookback` days."""
    px = prices.loc[prices.index < as_of]
    if len(px) < lookback + 2:
        return pd.Series(np.nan, index=prices.columns)
    tail = px.iloc[-(lookback + 1):]
    rets = tail.pct_change(fill_method=None).dropna(how="all")
    return rets.std() * np.sqrt(252)


def composite_factor_score(
    sharpe: pd.Series,
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    eligible: list[str] | None = None,
    weights: dict[str, float] = FACTOR_WEIGHTS,
) -> pd.Series:
    """
    Cross-sectionally z-scored composite of four factors, restricted to the
    eligible universe for a regime.

    Returns a Series indexed by ticker. Tickers with no Sharpe estimate
    (NaN) are dropped entirely — a missing regime-conditional history
    means we don't rank the asset in that regime at all.
    """
    if eligible is not None:
        sharpe = sharpe.reindex(eligible)

    valid = sharpe.dropna().index
    if len(valid) == 0:
        return pd.Series(dtype=float)

    mom_long = momentum_score(prices, as_of, MOM_LONG_LOOKBACK, MOM_SKIP).reindex(valid)
    mom_short = momentum_score(prices, as_of, MOM_SHORT_LOOKBACK, MOM_SKIP).reindex(valid)
    vol = vol_score(prices, as_of, VOL_LOOKBACK).reindex(valid)

    z_sharpe = _z_score(sharpe.loc[valid])
    z_mom_long = _z_score(mom_long)
    z_mom_short = _z_score(mom_short)
    z_vol = _z_score(vol)

    composite = (
        weights["sharpe"] * z_sharpe
        + weights["mom_12_1"] * z_mom_long
        + weights["mom_6_1"] * z_mom_short
        - weights["vol_pen"] * z_vol
    )
    return composite
