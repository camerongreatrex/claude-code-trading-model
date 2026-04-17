"""
v2/paper_trader.py
------------------
Paper trading engine for the macro regime rotation strategy.

Designed for Interactive Brokers integration:
  - Monthly rebalance (first business day of each month)
  - Long-only ETF portfolio (~28 instruments, highly liquid)
  - Outputs IB-compatible order list (ticker, action, quantity, order_type)
  - Tracks portfolio state in JSON, history in CSV

Usage
─────
  python -m v2.paper_trader init         — initialize with $100K
  python -m v2.paper_trader rebalance    — run monthly rebalance
  python -m v2.paper_trader status       — show current positions
  python -m v2.paper_trader orders       — generate IB order list (no execution)
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

from v2.universe import get_tickers, get_asset_class_map
from v2.regimes.macro_features import build_macro_features
from v2.regimes.market_features import build_market_features
from v2.regimes.classifier import classify_regimes
from v2.portfolio import build_portfolio_weights, blend_regime_allocations, apply_constraints, normalize_weights
from v2.risk_model import (
    compute_vol_scalar, STRESS_THRESHOLD, STRESS_DEFENSIVE,
    STRESS_TICKERS, STRESS_WEIGHTS,
)

STATE_DIR = Path("data/v2/paper_trading")
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = STATE_DIR / "state.json"
HISTORY_FILE = STATE_DIR / "history.csv"
TRADES_FILE = STATE_DIR / "trades.csv"
ORDERS_FILE = STATE_DIR / "pending_orders.csv"

INITIAL_CAPITAL = 100_000


# ── State Management ─────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load portfolio state from JSON."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    """Save portfolio state to JSON."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_history() -> pd.DataFrame:
    """Load portfolio value history."""
    if HISTORY_FILE.exists():
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame()


def load_trades() -> pd.DataFrame:
    """Load trade log."""
    if TRADES_FILE.exists():
        return pd.read_csv(TRADES_FILE)
    return pd.DataFrame()


def _append_csv(path: Path, row: dict):
    """Append a row to a CSV file, creating headers if needed."""
    df = pd.DataFrame([row])
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


# ── Initialization ────────────────────────────────────────────────────────────

def init_portfolio(capital: float = INITIAL_CAPITAL):
    """Initialize a fresh paper trading portfolio."""
    state = {
        "initial_capital": capital,
        "cash": capital,
        "positions": {},  # ticker → {"shares": int, "avg_price": float}
        "portfolio_value": capital,
        "last_rebalance": None,
        "created_at": datetime.now().isoformat(),
    }
    save_state(state)
    _append_csv(HISTORY_FILE, {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "portfolio_value": capital,
        "cash": capital,
        "n_positions": 0,
    })
    print(f"  Initialized paper portfolio: ${capital:,.0f}")
    return state


# ── Price Fetching ────────────────────────────────────────────────────────────

def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch current prices for all tickers."""
    prices = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if not data.empty:
                close = data["Close"].squeeze()
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                prices[ticker] = float(close.iloc[-1])
        except Exception:
            continue
    return prices


# ── Target Weight Computation ─────────────────────────────────────────────────

def compute_target_weights() -> dict[str, float]:
    """
    Run the full regime pipeline and return current target weights.

    Steps:
      1. Fetch/update macro features
      2. Fetch/update market features
      3. Classify current regime
      4. Blend allocations by regime probabilities
      5. Apply constraints
    """
    print("  Updating macro features...")
    macro = build_macro_features(use_cache=False)

    print("  Updating market features...")
    market = build_market_features(use_cache=False)

    print("  Classifying regime...")
    probs, labels = classify_regimes()

    # Get latest regime probabilities
    latest_probs = probs.iloc[-1]
    print(f"  Current regime: {latest_probs.idxmax()} ({latest_probs.max():.0%})")

    # Blend allocations
    blended = blend_regime_allocations(latest_probs)
    constrained = apply_constraints(blended.copy())
    normalized = normalize_weights(constrained)

    # Check for stress override
    if "stress_score" in market.columns:
        stress = market["stress_score"].iloc[-1]
        print(f"  Stress score: {stress:.2f}")
        if stress > STRESS_THRESHOLD:
            print(f"  STRESS OVERRIDE ACTIVE — forcing {STRESS_DEFENSIVE:.0%} defensive")
            remaining = 1.0 - STRESS_DEFENSIVE
            for ticker in normalized:
                normalized[ticker] *= remaining
            for ticker, weight in zip(STRESS_TICKERS, STRESS_WEIGHTS):
                normalized[ticker] = normalized.get(ticker, 0) + STRESS_DEFENSIVE * weight
            total = sum(normalized.values())
            normalized = {t: w / total for t, w in normalized.items()}

    return normalized


# ── Order Generation (IB-compatible) ─────────────────────────────────────────

def generate_orders(
    target_weights: dict[str, float],
    current_prices: dict[str, float],
    state: dict,
) -> list[dict]:
    """
    Generate IB-compatible order list.

    Each order: {ticker, action (BUY/SELL), quantity, order_type, notional}

    Only generates orders where the position change exceeds 0.5% of portfolio
    to avoid excessive small trades.
    """
    portfolio_value = state["portfolio_value"]
    positions = state.get("positions", {})
    min_trade_pct = 0.005  # 0.5% minimum trade threshold

    orders = []

    # All tickers (union of current positions and targets)
    all_tickers = set(target_weights.keys()) | set(positions.keys())

    for ticker in sorted(all_tickers):
        target_weight = target_weights.get(ticker, 0)
        price = current_prices.get(ticker)
        if price is None or price <= 0:
            continue

        target_shares = int((target_weight * portfolio_value) / price)
        current_shares = positions.get(ticker, {}).get("shares", 0)
        delta = target_shares - current_shares

        if delta == 0:
            continue

        # Check minimum trade threshold
        trade_notional = abs(delta * price)
        trade_pct = trade_notional / portfolio_value
        if trade_pct < min_trade_pct:
            continue

        orders.append({
            "ticker": ticker,
            "action": "BUY" if delta > 0 else "SELL",
            "quantity": abs(delta),
            "order_type": "MKT",  # market order for liquid ETFs
            "notional": round(trade_notional, 2),
            "target_weight": round(target_weight * 100, 1),
            "current_shares": current_shares,
            "target_shares": target_shares,
        })

    # Sort: sells first (free up cash), then buys
    orders.sort(key=lambda x: (x["action"] == "BUY", -x["notional"]))

    return orders


# ── Rebalance Execution (paper) ───────────────────────────────────────────────

def execute_rebalance():
    """Run a full paper rebalance."""
    state = load_state()
    if not state:
        print("  No portfolio state. Run 'init' first.")
        return

    # Get target weights
    target_weights = compute_target_weights()

    # Get current prices
    tickers = list(set(get_tickers()) | set(state.get("positions", {}).keys()))
    print(f"\n  Fetching prices for {len(tickers)} tickers...")
    prices = get_current_prices(tickers)

    # Update portfolio value
    positions = state.get("positions", {})
    position_value = sum(
        pos["shares"] * prices.get(ticker, pos.get("avg_price", 0))
        for ticker, pos in positions.items()
    )
    state["portfolio_value"] = state["cash"] + position_value

    # Generate orders
    orders = generate_orders(target_weights, prices, state)

    if not orders:
        print("  No rebalance needed — positions within threshold.")
        state["last_rebalance"] = datetime.now().isoformat()
        save_state(state)
        return

    # Execute orders (paper)
    print(f"\n  Executing {len(orders)} orders...")
    for order in orders:
        ticker = order["ticker"]
        price = prices[ticker]
        shares = order["quantity"]

        if order["action"] == "SELL":
            # Reduce/close position
            pos = positions.get(ticker, {"shares": 0, "avg_price": 0})
            sell_shares = min(shares, pos["shares"])
            proceeds = sell_shares * price
            state["cash"] += proceeds
            pos["shares"] -= sell_shares
            if pos["shares"] <= 0:
                positions.pop(ticker, None)
            else:
                positions[ticker] = pos
        else:
            # Buy
            cost = shares * price
            if cost > state["cash"]:
                shares = int(state["cash"] / price)
                cost = shares * price
            if shares <= 0:
                continue
            state["cash"] -= cost
            if ticker in positions:
                old = positions[ticker]
                total_shares = old["shares"] + shares
                avg_price = (old["shares"] * old["avg_price"] + cost) / total_shares
                positions[ticker] = {"shares": total_shares, "avg_price": round(avg_price, 4)}
            else:
                positions[ticker] = {"shares": shares, "avg_price": round(price, 4)}

        # Log trade
        _append_csv(TRADES_FILE, {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "action": order["action"],
            "shares": shares,
            "price": round(price, 4),
            "notional": round(shares * price, 2),
            "target_weight": order["target_weight"],
        })

        print(f"    {order['action']:4s} {shares:5d} {ticker:5s} @ ${price:8.2f} = ${shares*price:10,.2f}")

    # Update state
    state["positions"] = positions
    position_value = sum(
        pos["shares"] * prices.get(ticker, pos.get("avg_price", 0))
        for ticker, pos in positions.items()
    )
    state["portfolio_value"] = state["cash"] + position_value
    state["last_rebalance"] = datetime.now().isoformat()
    save_state(state)

    # Log history
    _append_csv(HISTORY_FILE, {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "portfolio_value": round(state["portfolio_value"], 2),
        "cash": round(state["cash"], 2),
        "n_positions": len(positions),
    })

    pnl = state["portfolio_value"] - state["initial_capital"]
    print(f"\n  Portfolio value: ${state['portfolio_value']:,.2f}  (P&L: ${pnl:+,.2f})")
    print(f"  Cash: ${state['cash']:,.2f}")
    print(f"  Positions: {len(positions)}")


# ── Display Functions ─────────────────────────────────────────────────────────

def show_status():
    """Print current portfolio status."""
    state = load_state()
    if not state:
        print("  No portfolio. Run 'init' first.")
        return

    positions = state.get("positions", {})
    print(f"\n  Portfolio Value: ${state['portfolio_value']:,.2f}")
    print(f"  Cash:           ${state['cash']:,.2f}")
    print(f"  Positions:      {len(positions)}")
    print(f"  Last Rebalance: {state.get('last_rebalance', 'Never')}")

    if positions:
        print(f"\n  {'Ticker':6s} {'Shares':>7s} {'Avg Price':>10s} {'Value':>12s} {'Weight':>7s}")
        print(f"  {'─'*50}")

        total = state["portfolio_value"]
        for ticker, pos in sorted(positions.items(), key=lambda x: -x[1]["shares"] * x[1]["avg_price"]):
            value = pos["shares"] * pos["avg_price"]
            weight = value / total if total > 0 else 0
            print(f"  {ticker:6s} {pos['shares']:7d} ${pos['avg_price']:9.2f} ${value:11,.2f} {weight:6.1%}")


def show_orders():
    """Generate and display IB-compatible orders without executing."""
    state = load_state()
    if not state:
        print("  No portfolio. Run 'init' first.")
        return

    target_weights = compute_target_weights()
    tickers = list(set(get_tickers()) | set(state.get("positions", {}).keys()))
    prices = get_current_prices(tickers)

    # Update portfolio value
    positions = state.get("positions", {})
    position_value = sum(
        pos["shares"] * prices.get(ticker, pos.get("avg_price", 0))
        for ticker, pos in positions.items()
    )
    state["portfolio_value"] = state["cash"] + position_value

    orders = generate_orders(target_weights, prices, state)

    if not orders:
        print("  No orders needed — portfolio within rebalance threshold.")
        return

    print(f"\n  IB Order List ({len(orders)} orders)")
    print(f"  {'Action':6s} {'Qty':>6s} {'Ticker':6s} {'Type':5s} {'Notional':>12s} {'Target':>7s}")
    print(f"  {'─'*50}")
    total_notional = 0
    for o in orders:
        print(f"  {o['action']:6s} {o['quantity']:6d} {o['ticker']:6s} {o['order_type']:5s} "
              f"${o['notional']:11,.2f} {o['target_weight']:6.1f}%")
        total_notional += o["notional"]

    print(f"\n  Total turnover: ${total_notional:,.2f} ({total_notional/state['portfolio_value']*100:.1f}% of portfolio)")

    # Save orders to CSV for IB import
    orders_df = pd.DataFrame(orders)
    orders_df.to_csv(ORDERS_FILE, index=False)
    print(f"  Orders saved to {ORDERS_FILE}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m v2.paper_trader {init|rebalance|status|orders}")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "init":
        capital = float(sys.argv[2]) if len(sys.argv) > 2 else INITIAL_CAPITAL
        init_portfolio(capital)
    elif cmd == "rebalance":
        execute_rebalance()
    elif cmd == "status":
        show_status()
    elif cmd == "orders":
        show_orders()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m v2.paper_trader {init|rebalance|status|orders}")
        sys.exit(1)
