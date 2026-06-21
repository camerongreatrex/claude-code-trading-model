"""
execution_metrics.py — Simulate trade activity from a dollar-size matrix.

Used to compare live execution cadences (daily vs signal_only vs monthly)
on gross P&L, fee drag, and trade mix (entries / exits / rebalance drift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from v1.portfolio.portfolio import CAPITAL, portfolio_returns


COMMISSION_PCT = 0.0005  # 5 bps/side — matches paper_trader.py


def count_execution_trades(
    sizes: pd.DataFrame,
    *,
    dead_band_pct: float = 0.015,
) -> pd.DataFrame:
    """
    Classify day-over-day position changes from a size matrix.

    Returns a DataFrame with columns:
      date, entries, exits, rebalances, turnover_dollars, fee_dollars
    """
    if sizes.empty:
        return pd.DataFrame()

    w = sizes.fillna(0.0)
    rows = []
    prev = pd.Series(0.0, index=w.columns)
    band = CAPITAL * dead_band_pct

    for dt in w.index:
        cur = w.loc[dt]
        entries = exits = rebal = 0
        turnover = 0.0
        for t in w.columns:
            p, c = float(prev.get(t, 0)), float(cur.get(t, 0))
            if p <= 0 and c > band:
                entries += 1
                turnover += c
            elif p > band and c <= 0:
                exits += 1
                turnover += p
            elif p > band and c > band:
                delta = abs(c - p)
                if delta > band:
                    rebal += 1
                    turnover += delta
        fee = turnover * COMMISSION_PCT
        rows.append({
            "date": dt,
            "entries": entries,
            "exits": exits,
            "rebalances": rebal,
            "turnover_dollars": turnover,
            "fee_dollars": fee,
        })
        prev = cur

    return pd.DataFrame(rows).set_index("date")


def returns_with_fees(
    sizes: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    dead_band_pct: float = 0.015,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Gross and net daily returns after estimated commission drag.

    Net return = gross portfolio return − (fees / CAPITAL) on trade days.
    """
    gross = portfolio_returns(sizes, returns)
    activity = count_execution_trades(sizes, dead_band_pct=dead_band_pct)
    if activity.empty:
        return gross, gross.copy(), activity

    fee_ret = activity["fee_dollars"] / CAPITAL
    fee_ret = fee_ret.reindex(gross.index).fillna(0.0)
    net = gross - fee_ret
    return gross, net, activity


def summarize_execution(
    gross: pd.Series,
    net: pd.Series,
    activity: pd.DataFrame,
    label: str,
) -> dict:
    """Aggregate execution-quality stats for one variant."""
    g = gross.dropna()
    n = len(g)
    if n < 5:
        return {"label": label}

    years = n / 252
    ann_g = float((1 + g).prod() ** (252 / n) - 1) * 100
    ann_n = float((1 + net.dropna()).prod() ** (252 / n) - 1) * 100 if len(net.dropna()) >= 5 else ann_g

    total_trades = int(activity["entries"].sum() + activity["exits"].sum()
                       + activity["rebalances"].sum()) if not activity.empty else 0
    entries = int(activity["entries"].sum()) if not activity.empty else 0
    exits = int(activity["exits"].sum()) if not activity.empty else 0
    rebal = int(activity["rebalances"].sum()) if not activity.empty else 0
    fees = float(activity["fee_dollars"].sum()) if not activity.empty else 0.0
    turnover = float(activity["turnover_dollars"].sum()) if not activity.empty else 0.0

    rebal_pct = (rebal / total_trades * 100) if total_trades else 0.0

    return {
        "label": label,
        "ann_return_gross_pct": round(ann_g, 2),
        "ann_return_net_pct": round(ann_n, 2),
        "fee_drag_pp": round(ann_g - ann_n, 2),
        "total_fees_usd": round(fees, 0),
        "turnover_x": round(turnover / CAPITAL, 2),
        "trades_per_year": round(total_trades / max(years, 0.1), 1),
        "entries": entries,
        "exits": exits,
        "rebalances": rebal,
        "rebalance_pct": round(rebal_pct, 1),
        "n_days": n,
    }
