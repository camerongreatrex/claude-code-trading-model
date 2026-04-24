"""
v2/risk/trailing_stops.py
-------------------------
ATR-based trailing stop overlay, ported from V1's signal_generation.

V2 rebalances monthly, so between rebalances the portfolio holds static
weights. This module lets positions be closed intra-month if price drops
ATR_MULT × ATR below the trailing high since entry — protecting against
acute drawdowns that the monthly cadence can't react to.

The full V1 version supports half-size partial exits and VIX-adaptive
tightening. We keep the core rule (a clean full-exit trailing stop) and
leave the elaborations for later if they prove to add value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ATR_LOOKBACK = 14         # ATR-14 standard
ATR_MULT = 3.0            # V1 default — institutional standard
MIN_HOLD_DAYS = 5         # hold ≥5 trading days before the stop can trigger
CASH_TICKER = "SHY"


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_LOOKBACK) -> pd.Series:
    """Wilder ATR over `period` bars. Falls back to close-based range if HL absent."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def atr_from_close(close: pd.Series, period: int = ATR_LOOKBACK) -> pd.Series:
    """ATR approximation from close-only data (abs daily change, EMA-smoothed)."""
    tr = close.diff().abs()
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def check_trailing_stop(
    entry_date: pd.Timestamp,
    entry_price: float,
    closes: pd.Series,
    atr_series: pd.Series,
    as_of: pd.Timestamp,
    atr_mult: float = ATR_MULT,
    min_hold_days: int = MIN_HOLD_DAYS,
) -> tuple[bool, float | None]:
    """
    Given a position entered on `entry_date` at `entry_price`, check whether
    the trailing stop has triggered by `as_of`.

    Returns
    -------
    (triggered, exit_price)
        triggered=True means the stop was hit at or before `as_of`.
        exit_price is the price on the day the stop first triggered (None if not triggered).
    """
    window = closes.loc[(closes.index >= entry_date) & (closes.index <= as_of)]
    if len(window) < min_hold_days + 1:
        return False, None

    trailing_high = window.cummax()
    # ATR value used for the stop is the ATR at entry (stable width across hold)
    atr_at_entry = atr_series.loc[atr_series.index <= entry_date].iloc[-1] \
        if (atr_series.index <= entry_date).any() else np.nan
    if not np.isfinite(atr_at_entry) or atr_at_entry <= 0:
        return False, None

    stop_level = trailing_high - atr_mult * atr_at_entry
    # Stop can only trigger after the min-hold period
    eligible = window.iloc[min_hold_days:]
    eligible_stop = stop_level.iloc[min_hold_days:]
    triggers = eligible[eligible < eligible_stop]
    if triggers.empty:
        return False, None
    return True, float(triggers.iloc[0])


def apply_monthly_trailing_stops(
    weights_monthly: pd.DataFrame,
    prices_daily: pd.DataFrame,
    atr_mult: float = ATR_MULT,
    min_hold_days: int = MIN_HOLD_DAYS,
    cash_ticker: str = CASH_TICKER,
) -> pd.DataFrame:
    """
    Expand a monthly weights matrix to daily, zeroing out positions that trip
    their trailing stop between rebalances. Rebalance dates reset the running
    high (treated as new entries). Booted-out weight is parked in `cash_ticker`.

    Returns a DAILY weights DataFrame indexed over `prices_daily.index`.
    """
    daily_index = prices_daily.index
    daily_w = pd.DataFrame(0.0, index=daily_index, columns=weights_monthly.columns)

    # Pre-compute ATR per ticker on close-only data
    atr_map = {t: atr_from_close(prices_daily[t]) for t in prices_daily.columns}

    rebal_dates = weights_monthly.index.sort_values()
    for i, rebal in enumerate(rebal_dates):
        period_end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else daily_index[-1]
        period_slice = daily_index[(daily_index >= rebal) & (daily_index < period_end)]
        if len(period_slice) == 0:
            continue
        wts = weights_monthly.loc[rebal]
        # For each ticker, set weight over the period unless stop trips mid-period
        for ticker in wts.index:
            w = float(wts[ticker])
            if w <= 0 or ticker not in prices_daily.columns:
                continue
            px = prices_daily[ticker].loc[period_slice]
            if len(px) == 0 or not np.isfinite(px.iloc[0]):
                continue
            entry_price = float(px.iloc[0])
            a = atr_map[ticker].loc[period_slice]
            a0 = float(a.iloc[0]) if np.isfinite(a.iloc[0]) else np.nan

            if not np.isfinite(a0) or a0 <= 0:
                daily_w.loc[period_slice, ticker] = w
                continue

            trailing_high = px.cummax()
            stop_level = trailing_high - atr_mult * a0
            triggered = px < stop_level
            # Enforce min-hold
            if min_hold_days > 0:
                mask = pd.Series(False, index=px.index)
                mask.iloc[min_hold_days:] = triggered.iloc[min_hold_days:]
                triggered = mask

            if not triggered.any():
                daily_w.loc[period_slice, ticker] = w
                continue

            first_trip = triggered.idxmax()
            daily_w.loc[period_slice[period_slice < first_trip], ticker] = w
            # Post-trip: send the weight to cash for the rest of the period
            post = period_slice[period_slice >= first_trip]
            daily_w.loc[post, cash_ticker] = daily_w.loc[post, cash_ticker].astype(float) + w

    # Normalise rows to sum to 1 (any rounding)
    row_sums = daily_w.sum(axis=1).replace(0, np.nan)
    daily_w = daily_w.div(row_sums, axis=0).fillna(0.0)
    return daily_w
