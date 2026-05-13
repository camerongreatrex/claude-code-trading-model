"""
One-shot init for V4N-F 3-month test:
  1. Wipe data/v1/paper_trading_v4nf_3mo/{state.json, history.csv, trades.csv}
  2. Compute V4N-F target dollars using live signals
  3. Buy each name at today's OPEN price (yfinance 1d bar)
  4. Mark portfolio to today's CLOSE → day-1 history row records O→C return
  5. Save state with last_eod_date = today

Run with: PT_INSTANCE=v4nf_3mo PYTHONPATH=. python v1/scripts/_init_v4nf_at_open.py
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

assert os.environ.get("PT_INSTANCE") == "v4nf_3mo", \
    "Must set PT_INSTANCE=v4nf_3mo before running this script."

from v1.scripts.paper_trader import (  # noqa: E402
    PT_DIR, INITIAL_CAPITAL, compute_live_signals, _compute_topn_v2_targets,
    save_state, _append_csv, HISTORY_FILE, TRADES_FILE,
)

STATE_FP   = PT_DIR / "state.json"
HISTORY_FP = PT_DIR / "history.csv"
TRADES_FP  = PT_DIR / "trades.csv"

# ── 1. Wipe old artefacts ─────────────────────────────────────────────────────
for fp in (STATE_FP, HISTORY_FP, TRADES_FP):
    if fp.exists():
        fp.unlink()
        print(f"  removed {fp.name}")

# ── 2. Compute V4N-F target dollars (frozen sizer via PT_INSTANCE routing) ───
print("\nComputing live signals + V4N-F targets...")
signals = compute_live_signals()
targets = _compute_topn_v2_targets(INITIAL_CAPITAL, signals)
print(f"  → {len(targets)} positions, target gross ${sum(targets.values()):,.0f}")

# ── 3. Fetch today's OPEN and CLOSE per ticker ────────────────────────────────
print("\nFetching today's OPEN and CLOSE per ticker...")
ohlc = {}
for t in targets:
    try:
        h = yf.Ticker(t).history(period="2d", auto_adjust=False)
        if h.empty:
            print(f"  {t}: no data — skipping")
            continue
        last = h.iloc[-1]
        ohlc[t] = {"open": float(last["Open"]),
                   "close": float(last["Close"]),
                   "date": str(h.index[-1].date())}
    except Exception as e:
        print(f"  {t}: fetch failed ({e}) — skipping")

# ── 4. Buy at OPEN, build state ──────────────────────────────────────────────
today = str(date.today())
state = {
    "cash"            : INITIAL_CAPITAL,
    "positions"       : {},
    "initial_capital" : INITIAL_CAPITAL,
    "initialized_date": today,
    "last_eod_date"   : today,
    "portfolio_value" : INITIAL_CAPITAL,
    "strategy"        : "top11_adx22_momt_ac55_cap1",
}

print(f"\nBuying at today's OPEN ({today}):")
for t, dollar_target in sorted(targets.items(), key=lambda x: -x[1]):
    if t not in ohlc:
        continue
    open_px = ohlc[t]["open"]
    if open_px <= 0:
        continue
    shares  = dollar_target / open_px
    cost    = shares * open_px
    state["cash"] -= cost
    state["positions"][t] = {
        "shares"      : shares,
        "entry_price" : open_px,
        "cost_basis"  : cost,
        "last_close"  : ohlc[t]["close"],
        "entry_date"  : today,
    }
    _append_csv(TRADES_FP, {
        "date"      : today, "ticker": t, "action": "BUY",
        "shares"    : round(shares, 6), "price": round(open_px, 2),
        "value"     : round(cost, 2),
        "commission": 0.0,
        "pnl"       : 0.0,
        "reason"    : "init_v4nf_at_open",
    })
    print(f"  BUY  {t:<6} {shares:>10.4f} sh @ ${open_px:>7.2f}  (${cost:>9,.0f})")

# ── 5. Mark to today's CLOSE; write history with day-1 O→C return ────────────
pv_close = state["cash"] + sum(
    p["shares"] * ohlc[t]["close"] for t, p in state["positions"].items()
)
day1_ret = pv_close / INITIAL_CAPITAL - 1

_append_csv(HISTORY_FP, {
    "date"            : today,
    "portfolio_value" : round(pv_close, 2),
    "cash"            : round(state["cash"], 2),
    "invested"        : round(pv_close - state["cash"], 2),
    "n_positions"     : len(state["positions"]),
    "daily_return"    : round(day1_ret * 100, 4),
})

state["portfolio_value"] = pv_close
save_state(state)

print(f"\n{'='*60}")
print(f"  Positions    : {len(state['positions'])}")
print(f"  Cash         : ${state['cash']:>11,.2f}")
print(f"  Invested @ O : ${INITIAL_CAPITAL - state['cash']:>11,.2f}")
print(f"  Mark @ Close : ${pv_close:>11,.2f}")
print(f"  Day-1 return : {day1_ret*100:>+11.4f}%   (open → close)")
print(f"{'='*60}")
print(f"\nState written to {STATE_FP}")
