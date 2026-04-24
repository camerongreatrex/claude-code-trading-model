"""
v2/scripts/paper_trader.py
--------------------------
Live paper trading engine for the V2 macro regime rotation strategy.

Cadence
───────
  Monthly rebalance — target weights come from the latest row of
  data/v2/results/portfolio_weights.parquet. On the first trading day
  of a new month we sell exits and buy entries to match those weights.

  Between rebalances — positions are checked daily against an ATR-14
  trailing stop (3× ATR). Stopped tickers are moved to cash (SHY) until
  the next monthly rebalance re-evaluates.

State (data/v2/paper_trading/)
──────────────────────────────
  state.json  — V1-compatible schema: cash, positions{ticker: {shares,
                entry_price, entry_date, cost_basis, last_close}},
                initial_capital, initialized_date, last_eod_date,
                last_rebalance, portfolio_value, strategy.
  trades.csv  — date, ticker, action, shares, price, value, commission,
                pnl, reason
  history.csv — date, portfolio_value, cash, invested, n_positions,
                daily_return

The dashboard's `load_paper_state / load_paper_history / load_paper_trades`
helpers read these files directly, so the schema must match V1's.

Usage
─────
  python -m v2.scripts.paper_trader init    — first-run: $100k, enter targets
  python -m v2.scripts.paper_trader run     — EOD update (trailing stops +
                                              monthly rebalance if due)
  python -m v2.scripts.paper_trader status  — print current state
  python -m v2.scripts.paper_trader orders  — dry-run: show rebalance orders
"""

from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from pathlib import Path
from zoneinfo import ZoneInfo

from v2.pipeline.data_pipeline import get_tickers
from v2.risk.trailing_stops import atr_from_close, ATR_MULT, MIN_HOLD_DAYS

# ── Constants ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
COMMISSION_PCT  = 0.0005          # 5 bps/side — matches backtester
ET_ZONE         = ZoneInfo("America/New_York")
CASH_TICKER     = "SHY"
MIN_TRADE_USD   = 200.0           # skip orders below this notional
REBAL_THRESHOLD = 0.005           # 0.5% of portfolio — ignore drift below this

STRATEGY_KEY    = "macro_regime_rotation"

PT_DIR        = Path("data/v2/paper_trading")
STATE_FILE    = PT_DIR / "state.json"
TRADES_FILE   = PT_DIR / "trades.csv"
HISTORY_FILE  = PT_DIR / "history.csv"
ORDERS_FILE   = PT_DIR / "pending_orders.csv"
WEIGHTS_FILE  = Path("data/v2/results/portfolio_weights.parquet")
PT_DIR.mkdir(parents=True, exist_ok=True)


# ── Time helpers ─────────────────────────────────────────────────────────────

def _today_et() -> str:
    return str(datetime.now(ET_ZONE).date())


def _is_trading_day(d: date | None = None) -> bool:
    if d is None:
        d = datetime.now(ET_ZONE).date()
    return d.weekday() < 5


def _is_new_month(last_rebalance: str | None, today: str) -> bool:
    """True if `today` falls in a later calendar month than `last_rebalance`."""
    if not last_rebalance:
        return True
    prev = pd.Timestamp(last_rebalance)
    cur  = pd.Timestamp(today)
    return (cur.year, cur.month) != (prev.year, prev.month) and cur >= prev


# ── State I/O ────────────────────────────────────────────────────────────────

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
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _append_csv(path: Path, record: dict):
    row = pd.DataFrame([record])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


# ── yfinance fetchers ────────────────────────────────────────────────────────

def _fetch_daily(ticker: str, lookback_days: int = 260) -> pd.DataFrame:
    """Download ~1 year of OHLC daily bars (enough for ATR-14 warm-up)."""
    end   = datetime.today() + timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False,
        multi_level_index=False,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df


def _fetch_closes_batch(tickers: list[str], lookback_days: int = 260) -> dict[str, pd.Series]:
    """Batch-fetch close series for a list of tickers. Skips failures silently."""
    out: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            df = _fetch_daily(t, lookback_days)
            if df.empty or "Close" not in df.columns:
                continue
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            out[t] = s.dropna()
        except Exception:
            continue
    return out


def _latest_prices(closes: dict[str, pd.Series]) -> dict[str, float]:
    return {t: float(s.iloc[-1]) for t, s in closes.items() if len(s)}


# ── Target weights ───────────────────────────────────────────────────────────

def load_target_weights() -> pd.Series:
    """Latest row of portfolio_weights.parquet (monthly, already V2-constructed)."""
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"{WEIGHTS_FILE} missing. Run the V2 pipeline first "
            "(v2/scripts/run_v2.py) to produce portfolio weights."
        )
    w = pd.read_parquet(WEIGHTS_FILE)
    latest = w.iloc[-1].astype(float)
    latest = latest[latest > 1e-6]
    total = latest.sum()
    if total > 0:
        latest = latest / total
    return latest


# ── Portfolio valuation ──────────────────────────────────────────────────────

def _portfolio_value(state: dict, prices: dict[str, float]) -> float:
    return state["cash"] + sum(
        pos["shares"] * prices.get(t, pos.get("last_close", pos["entry_price"]))
        for t, pos in state["positions"].items()
    )


# ── Trade execution ──────────────────────────────────────────────────────────

def _buy(state: dict, ticker: str, price: float, dollar_amount: float,
         reason: str, trade_date: str) -> dict:
    """
    Enter or add to a position. For add-to, we weight-average entry_price
    (matches how V1 treats top-ups) so cost_basis stays consistent.
    """
    if dollar_amount < MIN_TRADE_USD:
        return state
    commission = dollar_amount * COMMISSION_PCT
    total_cost = dollar_amount + commission
    if total_cost > state["cash"] * 1.01:
        dollar_amount = max(0.0, state["cash"] * 0.99 - commission)
        if dollar_amount < MIN_TRADE_USD:
            return state
        commission = dollar_amount * COMMISSION_PCT
        total_cost = dollar_amount + commission

    shares = dollar_amount / price
    state["cash"] -= total_cost

    if ticker in state["positions"]:
        pos = state["positions"][ticker]
        total_shares = pos["shares"] + shares
        new_cost = pos.get("cost_basis", pos["shares"] * pos["entry_price"]) + dollar_amount
        pos["shares"]      = total_shares
        pos["entry_price"] = new_cost / total_shares
        pos["cost_basis"]  = new_cost
        pos["last_close"]  = price
    else:
        state["positions"][ticker] = {
            "shares"     : shares,
            "entry_price": price,
            "entry_date" : trade_date,
            "cost_basis" : dollar_amount,
            "last_close" : price,
        }

    _append_csv(TRADES_FILE, {
        "date": trade_date, "ticker": ticker, "action": "BUY",
        "shares": round(shares, 6), "price": round(price, 4),
        "value": round(dollar_amount, 2), "commission": round(commission, 2),
        "pnl": "", "reason": reason,
    })
    print(f"  BUY  {ticker:<6} {shares:10.4f} sh @ ${price:8.2f}  "
          f"(${dollar_amount:>10,.0f})  [{reason}]")
    return state


def _sell(state: dict, ticker: str, price: float, reason: str, trade_date: str,
          shares_to_sell: float | None = None) -> dict:
    """Full sale by default. If `shares_to_sell` is given, partial sale."""
    if ticker not in state["positions"]:
        return state
    pos = state["positions"][ticker]
    full = shares_to_sell is None or shares_to_sell >= pos["shares"] - 1e-9
    n = pos["shares"] if full else float(shares_to_sell)
    if n <= 0:
        return state

    proceeds   = n * price
    commission = proceeds * COMMISSION_PCT
    net        = proceeds - commission
    cb_frac    = pos.get("cost_basis", n * pos["entry_price"]) * (n / pos["shares"])
    pnl        = net - cb_frac
    state["cash"] += net

    if full:
        state["positions"].pop(ticker, None)
    else:
        pos["shares"]    -= n
        pos["cost_basis"] = pos.get("cost_basis", 0.0) - cb_frac
        pos["last_close"] = price

    _append_csv(TRADES_FILE, {
        "date": trade_date, "ticker": ticker, "action": "SELL",
        "shares": round(n, 6), "price": round(price, 4),
        "value": round(proceeds, 2), "commission": round(commission, 2),
        "pnl": round(pnl, 2), "reason": reason,
    })
    pnl_s = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
    print(f"  SELL {ticker:<6} {n:10.4f} sh @ ${price:8.2f}  "
          f"P&L {pnl_s:>10}  [{reason}]")
    return state


# ── Trailing-stop check (run daily between rebalances) ───────────────────────

def _check_trailing_stops(state: dict, closes: dict[str, pd.Series],
                          prices: dict[str, float], today_str: str) -> dict:
    """
    For each open position, compute the trailing-high since entry and the
    ATR-at-entry × ATR_MULT stop level. If today's close is below the stop
    AND the position has been held at least MIN_HOLD_DAYS, exit to cash.

    Uses close-only ATR (matches v2.risk.trailing_stops.atr_from_close).
    """
    for ticker in list(state["positions"]):
        if ticker == CASH_TICKER:
            continue  # never stop-out the cash sleeve
        if ticker not in closes:
            continue

        pos      = state["positions"][ticker]
        s        = closes[ticker]
        entry_dt = pd.Timestamp(pos["entry_date"])
        window   = s.loc[s.index >= entry_dt]
        if len(window) < MIN_HOLD_DAYS + 1:
            continue

        # ATR at entry — stable stop width across the hold.
        atr_series = atr_from_close(s)
        atr_at_entry = atr_series.loc[atr_series.index <= entry_dt]
        if atr_at_entry.empty or not np.isfinite(atr_at_entry.iloc[-1]) \
                or atr_at_entry.iloc[-1] <= 0:
            continue
        atr0 = float(atr_at_entry.iloc[-1])

        trailing_high = float(window.cummax().iloc[-1])
        stop_level    = trailing_high - ATR_MULT * atr0
        today_px      = prices.get(ticker, float(window.iloc[-1]))

        if today_px < stop_level:
            state = _sell(state, ticker, today_px,
                          reason="atr_trailing_stop", trade_date=today_str)
    return state


# ── Rebalance ────────────────────────────────────────────────────────────────

def _rebalance_to_targets(state: dict, targets: pd.Series,
                          prices: dict[str, float], today_str: str) -> dict:
    """
    Bring positions into line with `targets` (weights summing to ~1).

    Steps (V1 order: sells first to free cash, then buys):
      1. For every non-CASH ticker, compute delta = target_value - current_value.
         If |delta| below REBAL_THRESHOLD × portfolio_value, skip (save fees).
      2. SELL any excess (full exit if target weight is 0).
      3. BUY any deficit.
      4. Cash ticker (SHY) is treated as the residual and not explicitly traded
         here — SHY buys/sells arise naturally when target weight differs from
         held position. We route the normal delta calc through it too.
    """
    pv = _portfolio_value(state, prices)
    min_delta = REBAL_THRESHOLD * pv
    all_tickers = set(targets.index) | set(state["positions"].keys())

    # Build delta table
    deltas: list[tuple[str, float, float]] = []  # (ticker, delta_usd, price)
    for t in all_tickers:
        price = prices.get(t)
        if price is None or price <= 0:
            continue
        tgt_w   = float(targets.get(t, 0.0))
        tgt_val = tgt_w * pv
        cur_val = state["positions"].get(t, {}).get("shares", 0.0) * price
        delta   = tgt_val - cur_val
        if abs(delta) < min_delta:
            continue
        deltas.append((t, delta, price))

    # Sells first (negative delta = reduce / exit)
    for t, d, px in sorted(deltas, key=lambda x: x[1]):
        if d >= 0:
            break
        pos = state["positions"].get(t)
        if pos is None:
            continue
        # If target is 0 → full exit; else partial
        tgt_w = float(targets.get(t, 0.0))
        if tgt_w <= 0:
            state = _sell(state, t, px, reason="rebalance_exit",
                          trade_date=today_str)
        else:
            shares_to_sell = min(pos["shares"], abs(d) / px)
            state = _sell(state, t, px, reason="rebalance_trim",
                          trade_date=today_str,
                          shares_to_sell=shares_to_sell)

    # Then buys
    for t, d, px in sorted(deltas, key=lambda x: -x[1]):
        if d <= 0:
            continue
        state = _buy(state, t, px, dollar_amount=min(d, state["cash"] * 0.995),
                     reason="rebalance_entry", trade_date=today_str)

    return state


# ── Initialisation ───────────────────────────────────────────────────────────

def init_portfolio():
    """Create a fresh $100k paper portfolio at the latest target weights."""
    if STATE_FILE.exists():
        print(f"Already initialised. Delete {STATE_FILE} to reset.")
        return

    today_str = _today_et()
    print(f"\nInitialising V2 paper portfolio  ({today_str})")
    print(f"Strategy: {STRATEGY_KEY}\n")

    targets = load_target_weights()
    print(f"Loaded target weights for {len(targets)} tickers "
          f"(sum={targets.sum():.4f}).")

    tickers_to_fetch = sorted(set(targets.index) | {CASH_TICKER})
    print(f"Fetching prices for {len(tickers_to_fetch)} tickers...\n")
    closes = _fetch_closes_batch(tickers_to_fetch)
    prices = _latest_prices(closes)

    missing = [t for t in targets.index if t not in prices]
    if missing:
        print(f"  [warn] no price for {missing} — dropping from targets.")
        targets = targets.drop(missing)
        targets = targets / targets.sum()

    state = {
        "cash"            : INITIAL_CAPITAL,
        "positions"       : {},
        "initial_capital" : INITIAL_CAPITAL,
        "initialized_date": today_str,
        "last_eod_date"   : today_str,
        "last_rebalance"  : today_str,
        "portfolio_value" : INITIAL_CAPITAL,
        "strategy"        : STRATEGY_KEY,
    }

    state = _rebalance_to_targets(state, targets, prices, today_str)

    pv = _portfolio_value(state, prices)
    state["portfolio_value"] = pv
    save_state(state)

    _append_csv(HISTORY_FILE, {
        "date": today_str,
        "portfolio_value": round(pv, 2),
        "cash": round(state["cash"], 2),
        "invested": round(pv - state["cash"], 2),
        "n_positions": len(state["positions"]),
        "daily_return": 0.0,
    })

    print(f"\n{'='*52}")
    print(f"  Positions   : {len(state['positions'])}")
    print(f"  Invested    : ${pv - state['cash']:>10,.2f}")
    print(f"  Cash        : ${state['cash']:>10,.2f}")
    print(f"  Total value : ${pv:>10,.2f}")
    print(f"{'='*52}")
    print(f"\nState -> {STATE_FILE}")


# ── End-of-day update ────────────────────────────────────────────────────────

def end_of_day_update():
    """
    Daily update. On the first trading day of a new month, rebalance to
    the latest portfolio_weights targets. Otherwise, check trailing stops
    on all open positions, then snapshot.
    """
    state = load_state()
    if not state:
        print("No portfolio. Run 'init' first.")
        return

    today_str = _today_et()
    if not _is_trading_day():
        print(f"Today is {datetime.now(ET_ZONE).strftime('%A')} — markets closed, skipping.")
        return
    if state.get("last_eod_date") == today_str:
        print(f"EOD already ran for {today_str} — skipping.")
        return

    prev_value = state.get("portfolio_value", INITIAL_CAPITAL)
    print(f"\nEnd-of-day update  ({today_str})\n")

    # Decide whether to rebalance
    do_rebalance = _is_new_month(state.get("last_rebalance"), today_str)

    # Fetch closes for held + target universe
    if do_rebalance:
        targets = load_target_weights()
        universe = sorted(set(targets.index) | set(state["positions"].keys()) | {CASH_TICKER})
    else:
        targets = None
        universe = sorted(set(state["positions"].keys()) | {CASH_TICKER})

    print(f"Fetching prices for {len(universe)} tickers...\n")
    closes = _fetch_closes_batch(universe)
    prices = _latest_prices(closes)

    # Daily trailing-stop check (runs every day, including rebalance day —
    # a mid-month stop-out reduces the size of the rebalance buy anyway)
    state = _check_trailing_stops(state, closes, prices, today_str)

    if do_rebalance and targets is not None:
        print(f"\n  [rebalance] New month — aligning to latest target weights.")
        # Drop targets we couldn't price
        targets = targets[[t for t in targets.index if t in prices]]
        if targets.sum() > 0:
            targets = targets / targets.sum()
            state = _rebalance_to_targets(state, targets, prices, today_str)
            state["last_rebalance"] = today_str

    # Refresh last_close on every held position for the dashboard
    for t, pos in state["positions"].items():
        if t in prices:
            pos["last_close"] = prices[t]

    pv        = _portfolio_value(state, prices)
    daily_ret = (pv / prev_value - 1) if prev_value > 0 else 0.0
    total_ret = (pv / INITIAL_CAPITAL - 1) * 100

    state["portfolio_value"] = pv
    state["last_eod_date"]   = today_str
    save_state(state)

    _append_csv(HISTORY_FILE, {
        "date": today_str,
        "portfolio_value": round(pv, 2),
        "cash": round(state["cash"], 2),
        "invested": round(pv - state["cash"], 2),
        "n_positions": len(state["positions"]),
        "daily_return": round(daily_ret * 100, 4),
    })

    print(f"\n{'='*52}")
    print(f"  Portfolio value : ${pv:>10,.2f}")
    print(f"  Cash            : ${state['cash']:>10,.2f}")
    print(f"  Positions       : {len(state['positions'])}")
    print(f"  Daily return    : {daily_ret*100:>+8.2f}%")
    print(f"  Total return    : {total_ret:>+8.2f}%")
    print(f"  Last rebalance  : {state.get('last_rebalance')}")
    print(f"{'='*52}")


# ── Status ───────────────────────────────────────────────────────────────────

def show_status():
    state = load_state()
    if not state:
        print("No portfolio. Run 'init' first.")
        return

    positions = state.get("positions", {})
    pv        = state.get("portfolio_value", INITIAL_CAPITAL)
    cash      = state.get("cash", 0.0)
    total_ret = (pv / state.get("initial_capital", INITIAL_CAPITAL) - 1) * 100

    print(f"\nV2 Paper Portfolio")
    print(f"{'─'*52}")
    print(f"  Portfolio Value : ${pv:>10,.2f}  ({total_ret:+.2f}%)")
    print(f"  Cash            : ${cash:>10,.2f}")
    print(f"  Positions       : {len(positions)}")
    print(f"  Last EOD        : {state.get('last_eod_date', 'Never')}")
    print(f"  Last Rebalance  : {state.get('last_rebalance', 'Never')}")

    if positions:
        print(f"\n  {'Ticker':6s} {'Shares':>10s} {'Entry':>9s} {'Last':>9s} "
              f"{'Value':>12s} {'Weight':>7s} {'P&L%':>7s}")
        print(f"  {'─'*70}")
        rows = []
        for t, p in positions.items():
            px    = p.get("last_close", p["entry_price"])
            value = p["shares"] * px
            rows.append((t, p, value, px))
        for t, p, value, px in sorted(rows, key=lambda r: -r[2]):
            weight = value / pv if pv > 0 else 0
            pnl_pct = (px / p["entry_price"] - 1) * 100
            print(f"  {t:6s} {p['shares']:>10.4f} ${p['entry_price']:>8.2f} "
                  f"${px:>8.2f} ${value:>11,.2f} {weight:>6.1%} {pnl_pct:>+6.2f}%")


# ── Orders preview ───────────────────────────────────────────────────────────

def show_orders():
    """Compute what a rebalance would do right now without executing."""
    state = load_state()
    if not state:
        print("No portfolio. Run 'init' first.")
        return

    today_str = _today_et()
    targets   = load_target_weights()
    universe  = sorted(set(targets.index) | set(state["positions"].keys()) | {CASH_TICKER})
    closes    = _fetch_closes_batch(universe)
    prices    = _latest_prices(closes)
    pv        = _portfolio_value(state, prices)

    targets = targets[[t for t in targets.index if t in prices]]
    targets = targets / targets.sum()

    rows = []
    for t in sorted(set(targets.index) | set(state["positions"].keys())):
        px    = prices.get(t)
        if px is None or px <= 0:
            continue
        tgt_w = float(targets.get(t, 0.0))
        cur_v = state["positions"].get(t, {}).get("shares", 0.0) * px
        tgt_v = tgt_w * pv
        delta = tgt_v - cur_v
        if abs(delta) < REBAL_THRESHOLD * pv:
            continue
        rows.append({
            "ticker"     : t,
            "action"     : "BUY" if delta > 0 else "SELL",
            "shares"     : round(abs(delta) / px, 4),
            "price"      : round(px, 2),
            "notional"   : round(abs(delta), 2),
            "target_pct" : round(tgt_w * 100, 2),
            "current_pct": round(cur_v / pv * 100, 2) if pv > 0 else 0.0,
        })

    if not rows:
        print("No orders — portfolio within rebalance threshold.")
        return

    # Sells first, then buys (free cash before deploying)
    rows.sort(key=lambda r: (r["action"] == "BUY", -r["notional"]))
    df = pd.DataFrame(rows)
    df.to_csv(ORDERS_FILE, index=False)

    print(f"\nRebalance preview  ({today_str})  — {len(rows)} orders")
    print(f"{'─'*72}")
    print(f"  {'Action':6s} {'Ticker':6s} {'Shares':>10s} {'Price':>9s} "
          f"{'Notional':>12s} {'Cur%':>7s} {'Tgt%':>7s}")
    for r in rows:
        print(f"  {r['action']:6s} {r['ticker']:6s} {r['shares']:>10.4f} "
              f"${r['price']:>8.2f} ${r['notional']:>11,.2f} "
              f"{r['current_pct']:>6.2f}% {r['target_pct']:>6.2f}%")
    turnover = sum(r["notional"] for r in rows)
    print(f"\n  Total turnover: ${turnover:,.2f} ({turnover/pv*100:.1f}% of PV)")
    print(f"  Orders saved to {ORDERS_FILE}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _usage():
    print("Usage: python -m v2.scripts.paper_trader {init|run|status|orders}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage(); sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == "init":
        init_portfolio()
    elif cmd == "run":
        end_of_day_update()
    elif cmd == "status":
        show_status()
    elif cmd == "orders":
        show_orders()
    else:
        print(f"Unknown command: {cmd}")
        _usage(); sys.exit(1)
