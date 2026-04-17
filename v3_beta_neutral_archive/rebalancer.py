"""
v2/rebalancer.py
----------------
Turnover-managed weekly rebalancing for the beta-neutral model.

Computes target portfolio, diffs against current holdings, generates ordered
trade list, and enforces the 25% weekly turnover cap.

Turnover priority (when cap would be exceeded):
  1. Close positions that dropped out of top/bottom decile entirely
  2. Add new entries to top/bottom decile
  3. Adjust existing position sizes

Trade ordering: all sells first (frees cash), then buys.

Output
──────
  data/v2/trades/trade_log.parquet  — full trade history
  data/v2/trades/weekly_turnover.parquet — turnover by rebalance date
"""

import numpy as np
import pandas as pd
from pathlib import Path

V2_TRADE_DIR = Path("data/v2/trades")
V2_TRADE_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_WEEKLY_TURNOVER = 0.25  # max 25% of gross exposure per week
COMMISSION_PCT = 0.0005     # 0.05% per side (10bps round-trip)


def compute_turnover(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> float:
    """
    Compute turnover as sum of absolute weight changes / 2.
    (Dividing by 2 because a buy and corresponding sell is one trade.)
    """
    all_tickers = set(current_weights) | set(target_weights)
    total_change = sum(
        abs(target_weights.get(t, 0) - current_weights.get(t, 0))
        for t in all_tickers
    )
    return total_change / 2


def generate_trades(
    current_weights: dict[str, float],
    target_portfolio: dict,
    sectors: dict,
    gross_cap: float | None = None,
) -> tuple[list[dict], dict[str, float]]:
    """
    Generate a turnover-capped trade list to move from current to target weights.

    Args:
        current_weights: {ticker: weight} current positions (positive = long, negative = short)
        target_portfolio: portfolio dict from portfolio.construct_portfolio()
        sectors: ticker -> sector mapping
        gross_cap: if set, absolute turnover cap (otherwise MAX_WEEKLY_TURNOVER)

    Returns:
        (trade_list, new_weights)
        trade_list: list of trade dicts with keys: ticker, side, weight_change,
                    old_weight, new_weight, signal_score, sector, reason
        new_weights: resulting position weights after trades
    """
    cap = gross_cap or MAX_WEEKLY_TURNOVER

    # Build target weight dict (long = positive, short = negative)
    target_weights = {}
    for t, w in target_portfolio["long_weights"].items():
        target_weights[t] = w
    for t, w in target_portfolio["short_weights"].items():
        target_weights[t] = w
    if target_portfolio["spy_hedge"] != 0:
        spy_current = target_weights.get("SPY", 0)
        target_weights["SPY"] = spy_current + target_portfolio["spy_hedge"]

    # Compute all desired changes
    all_tickers = set(current_weights) | set(target_weights)
    changes = []
    for ticker in all_tickers:
        old_w = current_weights.get(ticker, 0)
        new_w = target_weights.get(ticker, 0)
        delta = new_w - old_w
        if abs(delta) < 1e-6:
            continue

        # Classify trade reason
        was_in = abs(old_w) > 1e-6
        will_be_in = abs(new_w) > 1e-6

        if not was_in and will_be_in:
            reason = "new_entry"
            priority = 2
        elif was_in and not will_be_in:
            reason = "exit"
            priority = 1  # highest priority
        else:
            reason = "reweight"
            priority = 3  # lowest priority

        side = "BUY" if delta > 0 else "SELL"

        changes.append({
            "ticker": ticker,
            "side": side,
            "weight_change": abs(delta),
            "delta": delta,
            "old_weight": old_w,
            "new_weight": new_w,
            "sector": sectors.get(ticker, "Unknown"),
            "reason": reason,
            "priority": priority,
        })

    # Sort by priority (exits first), then by size of change (largest first)
    changes.sort(key=lambda x: (x["priority"], -x["weight_change"]))

    # Apply turnover cap
    cumulative_turnover = 0.0
    accepted_trades = []
    for trade in changes:
        new_turnover = cumulative_turnover + trade["weight_change"]
        if new_turnover > cap and cumulative_turnover > 0:
            # Try partial fill
            remaining = cap - cumulative_turnover
            if remaining > 0.001:
                trade = trade.copy()
                fraction = remaining / trade["weight_change"]
                trade["weight_change"] = remaining
                trade["delta"] = trade["delta"] * fraction
                trade["new_weight"] = trade["old_weight"] + trade["delta"]
                accepted_trades.append(trade)
                cumulative_turnover = cap
            break
        accepted_trades.append(trade)
        cumulative_turnover = new_turnover

    # Build new weights
    new_weights = current_weights.copy()
    for trade in accepted_trades:
        new_weights[trade["ticker"]] = trade["new_weight"]

    # Remove zero positions
    new_weights = {t: w for t, w in new_weights.items() if abs(w) > 1e-6}

    # Order: sells first, then buys
    sells = [t for t in accepted_trades if t["side"] == "SELL"]
    buys = [t for t in accepted_trades if t["side"] == "BUY"]
    ordered_trades = sells + buys

    return ordered_trades, new_weights


def simulate_rebalances(
    portfolio_history: list[tuple[pd.Timestamp, dict]],
    sectors: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Simulate a series of weekly rebalances with turnover management.

    Args:
        portfolio_history: list of (date, portfolio_dict) from build_portfolio_history
        sectors: ticker -> sector mapping

    Returns:
        (weights_history, all_trades)
        weights_history: DataFrame with tickers as columns, dates as index
        all_trades: list of all trade dicts with date added
    """
    current_weights = {}
    all_trades = []
    weights_snapshots = []
    turnover_log = []

    for date, target in portfolio_history:
        trades, new_weights = generate_trades(
            current_weights, target, sectors
        )

        # Log trades
        for trade in trades:
            trade["date"] = date
            all_trades.append(trade)

        # Log turnover
        turnover = compute_turnover(current_weights, new_weights)
        turnover_log.append({"date": date, "turnover": turnover})

        current_weights = new_weights
        weights_snapshots.append({"date": date, **current_weights})

    # Build weights history DataFrame
    if weights_snapshots:
        weights_df = pd.DataFrame(weights_snapshots).set_index("date").fillna(0)
    else:
        weights_df = pd.DataFrame()

    # Save trade log
    if all_trades:
        trade_df = pd.DataFrame(all_trades)
        trade_df.to_parquet(V2_TRADE_DIR / "trade_log.parquet", index=False)

    if turnover_log:
        turnover_df = pd.DataFrame(turnover_log).set_index("date")
        turnover_df.to_parquet(V2_TRADE_DIR / "weekly_turnover.parquet")

    return weights_df, all_trades


def compute_transaction_costs(trades: list[dict]) -> float:
    """Compute total transaction costs for a list of trades."""
    return sum(abs(t["weight_change"]) * COMMISSION_PCT for t in trades)
