"""
v2/portfolio/managed_futures.py
-------------------------------
Time-series-momentum (CTA) sleeve. Each instrument is treated independently:

    long if  price > MA_LONG  AND  trailing_12m_return > 0
    cash otherwise (parked in SHY)

Active longs are equal-weighted; remainder goes to SHY. Each instrument is
also vol-scaled toward TARGET_INSTRUMENT_VOL so a single high-vol name
can't dominate the sleeve.

The sleeve's value is its low correlation with the regime book — trend
following naturally rotates into whatever asset is working (long bonds in
2008, long commodities in 2022, long equities in 2017/2024). Macro
regime classification often lags these turns, so the trend overlay
catches them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Curated instrument universe — broad cross-asset coverage with long history.
# Excludes IWM/IWD/IWF/sector ETFs (already in regime book and correlated).
TREND_UNIVERSE = [
    # Equity
    "SPY", "QQQ", "EFA", "EEM",
    # Bonds
    "TLT", "IEF", "TIP", "LQD", "HYG",
    # Commodities
    "GLD", "SLV", "DBC", "USO",
    # Real assets
    "VNQ",
    # Currency
    "UUP", "FXY",
]

MA_LONG = 200            # 200d simple MA — classic trend filter
MOM_LOOKBACK = 252       # 12m total return
MOM_SKIP = 0             # 12-0 (use full 12m, no skip)
VOL_LOOKBACK = 63        # 3m daily vol for instrument vol-targeting
TARGET_INSTRUMENT_VOL = 0.12   # per-instrument vol target before equal-weighting
INSTRUMENT_VOL_CAP = 2.0       # cap leverage scalar at 2x for any single instrument
INSTRUMENT_VOL_FLOOR = 0.5     # floor at 0.5x
CASH_TICKER = "SHY"


def compute_trend_signal(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    universe: list[str] | None = None,
) -> pd.Series:
    """
    Boolean trend signal per instrument as of `as_of`. True = long, False = cash.
    Uses only data strictly before `as_of` (walk-forward safe).
    """
    if universe is None:
        universe = TREND_UNIVERSE

    px = prices.loc[prices.index < as_of]
    if len(px) < MA_LONG + 2:
        return pd.Series(False, index=universe)

    last = px.iloc[-1]
    ma = px.tail(MA_LONG).mean()
    # 12m return
    if len(px) < MOM_LOOKBACK + 1:
        ret_12m = pd.Series(np.nan, index=px.columns)
    else:
        ret_12m = (px.iloc[-1] / px.iloc[-MOM_LOOKBACK - 1]) - 1.0

    signal = pd.Series(False, index=universe)
    for t in universe:
        if t not in px.columns:
            continue
        p = float(last.get(t, np.nan))
        m = float(ma.get(t, np.nan))
        r = float(ret_12m.get(t, np.nan)) if t in ret_12m.index else np.nan
        if not (np.isfinite(p) and np.isfinite(m) and np.isfinite(r)):
            continue
        signal[t] = (p > m) and (r > 0.0)
    return signal


def compute_instrument_vol_scalar(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    universe: list[str],
) -> pd.Series:
    """Per-instrument scalar = TARGET_INSTRUMENT_VOL / realized vol, clipped."""
    px = prices.loc[prices.index < as_of]
    if len(px) < VOL_LOOKBACK + 2:
        return pd.Series(1.0, index=universe)
    rets = px.tail(VOL_LOOKBACK + 1).pct_change().dropna(how="all")
    realized = rets.std() * np.sqrt(252)

    scalar = pd.Series(1.0, index=universe)
    for t in universe:
        if t not in realized.index:
            continue
        v = float(realized[t])
        if not np.isfinite(v) or v <= 0:
            continue
        s = TARGET_INSTRUMENT_VOL / v
        scalar[t] = float(np.clip(s, INSTRUMENT_VOL_FLOOR, INSTRUMENT_VOL_CAP))
    return scalar


def build_managed_futures_sleeve(
    prices: pd.DataFrame,
    monthly_dates: pd.DatetimeIndex,
    universe: list[str] | None = None,
    cash_ticker: str = CASH_TICKER,
) -> pd.DataFrame:
    """
    Walk-forward monthly trend portfolio.

    Returns DataFrame indexed by monthly_dates, columns = full price-data
    universe (so it can be blended cleanly with the regime book), each row
    sums to 1.0.
    """
    if universe is None:
        universe = TREND_UNIVERSE

    full_universe = list(prices.columns)
    rows = []

    for date in monthly_dates:
        signal = compute_trend_signal(prices, date, universe=universe)
        scalar = compute_instrument_vol_scalar(prices, date, universe)

        actives = [t for t in universe if signal.get(t, False)]
        row = {t: 0.0 for t in full_universe}

        if not actives:
            row[cash_ticker] = 1.0
        else:
            # Equal-weight × per-instrument vol scalar, then renormalize
            raw = {t: scalar[t] for t in actives}
            total = sum(raw.values())
            # Allocate (1 - cash_floor) to actives, with the rest in cash.
            # cash_floor = (n_universe - n_active) / n_universe — partial-conviction floor
            cash_floor = max(0.0, 1.0 - len(actives) / len(universe))
            risky_budget = 1.0 - cash_floor
            for t, raw_w in raw.items():
                row[t] = (raw_w / total) * risky_budget
            row[cash_ticker] = row.get(cash_ticker, 0.0) + cash_floor

        # Normalize defensively
        s = sum(row.values())
        if s > 0:
            row = {t: w / s for t, w in row.items()}
        row["Date"] = date
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Date").fillna(0.0)
    return df
