"""
v2/paper_trader.py
------------------
Paper trading engine for the beta-neutral cross-sectional model.

Tracks long and short books separately, computes daily P&L, and logs
positions with signal scores and sector information.

State files (data/v2/paper_trading/)
────────────────────────────────────
  state.json       — cash, long_positions, short_positions, spy_hedge, portfolio_value
  trades.csv       — append-only trade log
  history.csv      — daily snapshot: portfolio value, gross/net exposure, beta, etc.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

from v2.universe import load_universe
from v2.signal_generation import generate_signals
from v2.portfolio import construct_portfolio, compute_rolling_betas, CAPITAL
from v2.rebalancer import generate_trades
from v2.risk_model import compute_vol_scale
from v2.costs import (
    commission_cost, slippage_cost, borrow_rate, daily_borrow_cost,
    get_market_cap,
)

# ── Constants ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = float(CAPITAL)
ET_ZONE = ZoneInfo("America/New_York")

PT_DIR = Path("data/v2/paper_trading")
PT_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = PT_DIR / "state.json"
TRADES_FILE = PT_DIR / "trades.csv"
HISTORY_FILE = PT_DIR / "history.csv"


# ── State I/O ─────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    with open(STATE_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def load_trades() -> pd.DataFrame:
    if not TRADES_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(TRADES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _append_csv(path: Path, record: dict):
    row = pd.DataFrame([record])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def _today_et() -> str:
    return str(datetime.now(ET_ZONE).date())


# ── Portfolio operations ──────────────────────────────────────────────────────

def init_portfolio():
    """Initialize a fresh v2 paper trading portfolio."""
    state = {
        "model_version": "v2",
        "cash": INITIAL_CAPITAL,
        "long_positions": {},    # {ticker: {shares, entry_price, entry_date, signal_score, sector}}
        "short_positions": {},   # {ticker: {shares, entry_price, entry_date, signal_score, sector}}
        "spy_hedge_shares": 0,
        "portfolio_value": INITIAL_CAPITAL,
        "initial_capital": INITIAL_CAPITAL,
        "initialized_date": _today_et(),
        "last_eod_date": None,
    }
    save_state(state)
    print(f"  V2 paper portfolio initialized with ${INITIAL_CAPITAL:,.0f}")
    return state


def compute_portfolio_value(state: dict) -> float:
    """Compute current portfolio value from positions + cash."""
    value = state["cash"]

    # Long positions
    all_tickers = list(state.get("long_positions", {}).keys()) + \
                  list(state.get("short_positions", {}).keys())

    if not all_tickers:
        return value

    # Fetch current prices
    prices = _fetch_current_prices(all_tickers)

    for ticker, pos in state.get("long_positions", {}).items():
        price = prices.get(ticker)
        if price:
            value += pos["shares"] * price

    for ticker, pos in state.get("short_positions", {}).items():
        price = prices.get(ticker)
        if price:
            # Short P&L: profit when price goes down
            entry_value = pos["shares"] * pos["entry_price"]
            current_value = pos["shares"] * price
            value += entry_value - current_value  # short profit/loss

    return value


def _fetch_current_prices(tickers: list) -> dict:
    """Fetch current prices for a list of tickers."""
    prices = {}
    if not tickers:
        return prices

    ticker_str = " ".join(tickers[:50])  # batch limit
    try:
        data = yf.download(ticker_str, period="5d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"]
        else:
            closes = data[["Close"]].rename(columns={"Close": tickers[0]})

        for t in tickers:
            if t in closes.columns:
                last = closes[t].dropna()
                if len(last) > 0:
                    prices[t] = float(last.iloc[-1])
    except Exception as e:
        print(f"  Warning: price fetch failed: {e}")

    return prices


def end_of_day_update():
    """
    Run end-of-day update: rebalance if Friday, update portfolio value,
    log daily snapshot.
    """
    state = load_state()
    if not state:
        print("  V2 paper portfolio not initialized. Run 'init' first.")
        return

    today = _today_et()
    if state.get("last_eod_date") == today:
        print(f"  V2 EOD already run for {today}")
        return

    # Update portfolio value
    portfolio_value = compute_portfolio_value(state)

    # Accrue daily short borrow costs
    short_positions = state.get("short_positions", {})
    if short_positions:
        prices = _fetch_current_prices(list(short_positions.keys()))
        total_borrow = 0.0
        for ticker, pos in short_positions.items():
            price = prices.get(ticker, pos["entry_price"])
            short_notional = pos["shares"] * price
            rate = borrow_rate(ticker, get_market_cap(ticker))
            total_borrow += daily_borrow_cost(short_notional, rate)
        state["cash"] -= total_borrow
        if total_borrow > 0:
            print(f"  Borrow cost accrued: ${total_borrow:.2f}")

    # Check if today is Friday (rebalance day)
    today_dt = datetime.now(ET_ZONE).date()
    is_friday = today_dt.weekday() == 4

    if is_friday:
        print("  Friday — running weekly rebalance...")
        _run_rebalance(state, portfolio_value)
        # Re-compute after rebalance
        portfolio_value = compute_portfolio_value(state)

    # Compute exposure metrics
    n_long = len(state.get("long_positions", {}))
    n_short = len(state.get("short_positions", {}))
    gross_exposure = 0
    net_exposure = 0

    prices = _fetch_current_prices(
        list(state.get("long_positions", {}).keys()) +
        list(state.get("short_positions", {}).keys())
    )

    long_value = sum(
        pos["shares"] * prices.get(t, pos["entry_price"])
        for t, pos in state.get("long_positions", {}).items()
    )
    short_value = sum(
        pos["shares"] * prices.get(t, pos["entry_price"])
        for t, pos in state.get("short_positions", {}).items()
    )

    if portfolio_value > 0:
        gross_exposure = (long_value + short_value) / portfolio_value
        net_exposure = (long_value - short_value) / portfolio_value

    # Update state
    state["portfolio_value"] = portfolio_value
    state["last_eod_date"] = today
    save_state(state)

    # Log daily snapshot
    prev = load_history()
    prev_value = prev["portfolio_value"].iloc[-1] if len(prev) > 0 else INITIAL_CAPITAL
    daily_return = (portfolio_value / prev_value - 1) if prev_value > 0 else 0

    _append_csv(HISTORY_FILE, {
        "date": today,
        "portfolio_value": round(portfolio_value, 2),
        "cash": round(state["cash"], 2),
        "n_long": n_long,
        "n_short": n_short,
        "gross_exposure": round(gross_exposure, 4),
        "net_exposure": round(net_exposure, 4),
        "long_value": round(long_value, 2),
        "short_value": round(short_value, 2),
        "daily_return": round(daily_return, 6),
    })

    print(f"  V2 EOD: ${portfolio_value:,.2f} ({daily_return:+.2%}) | "
          f"Long: {n_long} | Short: {n_short} | "
          f"Gross: {gross_exposure:.1%} | Net: {net_exposure:.1%}")


def _run_rebalance(state: dict, portfolio_value: float):
    """Execute weekly rebalance based on current signals."""
    try:
        universe = load_universe()
    except FileNotFoundError:
        print("  Warning: universe not found, skipping rebalance")
        return

    # Fetch recent price data for signal computation
    all_tickers = universe["all"]
    print(f"  Fetching data for {len(all_tickers)} tickers...")

    ticker_str = " ".join(all_tickers[:50])
    closes_dict = {}
    for i in range(0, len(all_tickers), 50):
        batch = all_tickers[i:i + 50]
        try:
            data = yf.download(
                " ".join(batch), period="2y", auto_adjust=True, progress=False
            )
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    if t in data["Close"].columns:
                        closes_dict[t] = data["Close"][t]
            elif len(batch) == 1:
                closes_dict[batch[0]] = data["Close"]
        except Exception:
            continue

    if len(closes_dict) < 50:
        print(f"  Warning: only got data for {len(closes_dict)} tickers, skipping")
        return

    closes = pd.DataFrame(closes_dict).ffill()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    returns = np.log(closes / closes.shift(1))

    # Generate signals
    composites, ranks = generate_signals(
        closes, returns,
        fundamentals=(pd.Series(dtype=float), pd.Series(dtype=float), 0.0),
        sectors=universe.get("sectors"),
    )

    if composites.empty:
        print("  Warning: no signals generated")
        return

    # Get the latest signal date
    latest_date = composites.index[-1]

    # Compute betas
    spy_returns = returns["SPY"] if "SPY" in returns.columns else None
    betas = compute_rolling_betas(returns, spy_returns) if spy_returns is not None else None

    # Construct target portfolio
    target = construct_portfolio(
        composites, returns, universe.get("sectors", {}), latest_date,
        betas=betas, spy_returns=spy_returns,
    )

    if not target["long_weights"]:
        print("  Warning: empty target portfolio")
        return

    # Convert current positions to weight dict
    current_weights = {}
    for t, pos in state.get("long_positions", {}).items():
        price = pos["entry_price"]
        current_weights[t] = pos["shares"] * price / portfolio_value
    for t, pos in state.get("short_positions", {}).items():
        price = pos["entry_price"]
        current_weights[t] = -pos["shares"] * price / portfolio_value

    # Generate trades
    trades, new_weights = generate_trades(
        current_weights, target, universe.get("sectors", {})
    )

    # Execute trades
    for trade in trades:
        _execute_trade(state, trade, portfolio_value)

    print(f"  Executed {len(trades)} trades")


def _execute_trade(state: dict, trade: dict, portfolio_value: float):
    """Execute a single trade (buy/sell) and update state."""
    ticker = trade["ticker"]
    side = trade["side"]
    weight = abs(trade.get("new_weight", 0))
    old_weight = abs(trade.get("old_weight", 0))

    # Fetch current price
    prices = _fetch_current_prices([ticker])
    price = prices.get(ticker)
    if not price:
        return

    target_value = weight * portfolio_value
    shares = int(target_value / price) if price > 0 else 0

    if shares == 0 and weight > 0.001:
        return

    # Determine if this is a long or short position
    new_weight = trade.get("new_weight", 0)
    is_long = new_weight > 0
    is_short = new_weight < 0
    is_exit = abs(new_weight) < 0.001

    long_positions = state.get("long_positions", {})
    short_positions = state.get("short_positions", {})

    # Close existing position if side is changing or exiting
    if ticker in long_positions and (is_short or is_exit):
        pos = long_positions.pop(ticker)
        sell_value = pos["shares"] * price
        comm = commission_cost(pos["shares"], price)
        slip = slippage_cost(price) * pos["shares"]
        state["cash"] += sell_value - comm - slip
        _log_trade(ticker, "SELL", pos["shares"], price, trade)

    if ticker in short_positions and (is_long or is_exit):
        pos = short_positions.pop(ticker)
        # Close short: buy back at current price
        pnl = pos["shares"] * (pos["entry_price"] - price)
        comm = commission_cost(pos["shares"], price)
        slip = slippage_cost(price) * pos["shares"]
        state["cash"] += pos["shares"] * pos["entry_price"] + pnl - comm - slip
        _log_trade(ticker, "COVER", pos["shares"], price, trade)

    # Open new position
    if not is_exit and shares > 0:
        comm = commission_cost(shares, price)
        slip = slippage_cost(price) * shares
        if is_long:
            cost = shares * price + comm + slip
            if cost <= state["cash"]:
                state["cash"] -= cost
                long_positions[ticker] = {
                    "shares": shares,
                    "entry_price": price,
                    "entry_date": _today_et(),
                    "signal_score": float(trade.get("signal_score", 0)),
                    "sector": trade.get("sector", "Unknown"),
                }
                _log_trade(ticker, "BUY", shares, price, trade)
        elif is_short:
            # Short: receive cash for borrowed shares minus costs
            state["cash"] += shares * price - comm - slip
            short_positions[ticker] = {
                "shares": shares,
                "entry_price": price,
                "entry_date": _today_et(),
                "signal_score": float(trade.get("signal_score", 0)),
                "sector": trade.get("sector", "Unknown"),
            }
            _log_trade(ticker, "SHORT", shares, price, trade)

    state["long_positions"] = long_positions
    state["short_positions"] = short_positions


def _log_trade(ticker: str, action: str, shares: int, price: float, trade: dict):
    """Append trade to the trade log."""
    _append_csv(TRADES_FILE, {
        "date": _today_et(),
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "price": round(price, 2),
        "value": round(shares * price, 2),
        "sector": trade.get("sector", ""),
        "reason": trade.get("reason", ""),
    })


def status():
    """Print current portfolio status."""
    state = load_state()
    if not state:
        print("  V2 paper portfolio not initialized.")
        return

    portfolio_value = compute_portfolio_value(state)
    pnl = portfolio_value - state["initial_capital"]
    pnl_pct = pnl / state["initial_capital"]

    n_long = len(state.get("long_positions", {}))
    n_short = len(state.get("short_positions", {}))

    print(f"\n  V2 Paper Portfolio Status")
    print(f"  {'=' * 40}")
    print(f"  Portfolio Value:  ${portfolio_value:>12,.2f}")
    print(f"  Total P&L:        ${pnl:>12,.2f} ({pnl_pct:+.2%})")
    print(f"  Cash:             ${state['cash']:>12,.2f}")
    print(f"  Long Positions:   {n_long:>12}")
    print(f"  Short Positions:  {n_short:>12}")
    print(f"  Initialized:      {state.get('initialized_date', 'N/A')}")
    print(f"  Last EOD:         {state.get('last_eod_date', 'N/A')}")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python -m v2.paper_trader [init|run|status]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "init":
        init_portfolio()
    elif cmd == "run":
        end_of_day_update()
    elif cmd == "status":
        status()
    else:
        print(f"Unknown command: {cmd}")
