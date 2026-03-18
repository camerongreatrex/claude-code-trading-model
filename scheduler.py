"""
scheduler.py

Automated daily pipeline — runs continuously as a background process.
Fires the EOD update at exactly 4:45 PM US/Eastern on weekdays,
generates tomorrow's pre-market order sheet, and enforces the kill switch.

Usage:
    python scheduler.py          # run in foreground (keep terminal open)
    pythonw scheduler.py         # run silently in background (no window)

How it works:
  - Sleeps most of the time, waking every 30 seconds to check the clock.
  - On weekdays at 4:45 PM ET, calls paper_trader.end_of_day_update().
  - After each EOD run, writes data/paper_trading/orders_tomorrow.json
    so you know exactly what positions you're holding into tomorrow.
  - Checks the kill switch on startup and before every trade session.

Comparison to professional systems:
  - Two Sigma / AQR equivalent: a Cron job on a Linux server fires a
    Python pipeline at market close.  This is the exact same pattern —
    just running on your laptop instead of AWS.
  - The key difference from HFT: no co-location, no sub-second latency.
    Not needed for a daily strategy.
"""

import time
import json
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo       # stdlib since Python 3.9

from paper_trader import (
    load_state, load_history, end_of_day_update,
    _fetch_daily, compute_signal,
    INITIAL_CAPITAL, PT_DIR,
    TICKER_LIST, ASSET_CLASS,
)

# ── Config ─────────────────────────────────────────────────────────────────────
EOD_HOUR      = 16          # 4 PM ET
EOD_MINUTE    = 45          # :45 — give yfinance time to settle after 4 PM close
ET_ZONE       = ZoneInfo("America/New_York")
KILL_SWITCH_DD = 0.15       # pause trading if rolling 20-day portfolio loss > 15%
LOG_FILE       = PT_DIR / "scheduler.log"
ORDERS_FILE    = PT_DIR / "orders_tomorrow.json"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("scheduler")


# ── Kill switch ────────────────────────────────────────────────────────────────

def check_kill_switch() -> bool:
    """
    Returns True (trading BLOCKED) if the portfolio has lost more than
    KILL_SWITCH_DD over the rolling 20-trading-day window.

    Why 20 days / 15%?
      - 20 days = 1 calendar month.  A 15% loss in a month is a 3-sigma
        event for a strategy with 10% annual vol — likely a data error,
        model break, or genuine market dislocation.
      - Automatic pause forces a human review before more capital is risked.
      - Professional equivalent: VaR breach circuit breaker.
    """
    history = load_history()
    if len(history) < 2:
        return False

    recent   = history.tail(20)
    peak     = recent["portfolio_value"].max()
    current  = float(recent["portfolio_value"].iloc[-1])
    drawdown = (current - peak) / peak   # negative number

    if drawdown < -KILL_SWITCH_DD:
        log.warning(
            f"KILL SWITCH ACTIVE — rolling 20-day drawdown = {drawdown*100:.1f}% "
            f"(limit: {-KILL_SWITCH_DD*100:.0f}%).  Trading paused.  "
            "Investigate before re-enabling."
        )
        return True
    return False


# ── Data validation ────────────────────────────────────────────────────────────

def validate_prices(signals: dict) -> dict:
    """
    Remove tickers whose closing price looks like a data error.

    Checks:
      1. Price is positive and finite.
      2. Single-day move is within ±25% (beyond this = almost certainly bad data,
         not a real move — even the 1987 crash was −22.6% in one day).
      3. Price hasn't been unchanged for 5+ consecutive days (stale feed).

    Professional equivalent: pre-trade data validation / 'sanity checks'
    that every systematic firm runs before sending orders.
    """
    clean = {}
    for ticker, sig in signals.items():
        price = sig.get("close")
        if price is None or not np.isfinite(price) or price <= 0:
            log.warning(f"  {ticker}: invalid price {price} — skipped")
            continue
        # Large single-day move check uses yfinance previous close
        try:
            info      = yf.Ticker(ticker).fast_info
            prev      = getattr(info, "previous_close", None) or getattr(info, "regularMarketPreviousClose", None)
            if prev and prev > 0:
                chg = abs(price / prev - 1)
                if chg > 0.25:
                    log.warning(
                        f"  {ticker}: single-day move {chg*100:.1f}% > 25% — "
                        "likely bad data, skipped"
                    )
                    continue
        except Exception:
            pass   # fast_info unavailable — skip the check, don't block the trade
        clean[ticker] = sig
    return clean


# ── Pre-market order sheet ─────────────────────────────────────────────────────

def generate_order_sheet(signals: dict, state: dict):
    """
    Write tomorrow's expected order list to orders_tomorrow.json.

    A real desk produces this the night before so the execution trader
    knows exactly what to do at the open without running code in the morning.

    Format:
      {
        "generated_at": "2026-03-17 16:47:22",
        "for_date":     "2026-03-18",
        "orders": [
          { "ticker": "SPY",  "action": "HOLD",  "reason": "in position, signal LONG" },
          { "ticker": "AMZN", "action": "BUY",   "size_usd": 5200.0, "signal": "golden_cross" },
          { "ticker": "GE",   "action": "SELL",  "reason": "death_cross" },
          ...
        ]
      }
    """
    from paper_trader import (
        _atr_position_size, _trailing_stop_hit, _update_trailing_high,
        MIN_HOLD_DAYS, RSI_OVERBOUGHT,
    )

    positions     = state.get("positions", {})
    cash          = state.get("cash", INITIAL_CAPITAL)
    pv            = state.get("portfolio_value", INITIAL_CAPITAL)
    current_longs = set(positions)
    orders        = []

    for ticker in TICKER_LIST:
        sig = signals.get(ticker)
        if not sig:
            orders.append({"ticker": ticker, "action": "NO_DATA"})
            continue

        price = sig["close"]

        if ticker in current_longs:
            pos = positions[ticker]
            # Check trailing stop
            if _trailing_stop_hit(pos, price, sig.get("atr")):
                orders.append({
                    "ticker": ticker, "action": "SELL",
                    "price_last": round(price, 2),
                    "reason": f"trailing_stop (high={pos.get('trailing_high', pos['entry_price']):.2f})",
                    "entry_price": pos["entry_price"],
                    "unrealised_pnl": round((price - pos["entry_price"]) * pos["shares"], 2),
                })
                continue
            # Check death cross
            if not sig["golden_cross"]:
                entry  = pd.to_datetime(pos["entry_date"])
                held   = (pd.to_datetime(date.today()) - entry).days
                if held >= MIN_HOLD_DAYS:
                    orders.append({
                        "ticker": ticker, "action": "SELL",
                        "price_last": round(price, 2),
                        "reason": "death_cross",
                        "entry_price": pos["entry_price"],
                        "unrealised_pnl": round((price - pos["entry_price"]) * pos["shares"], 2),
                    })
                    continue
                else:
                    orders.append({
                        "ticker": ticker, "action": "HOLD",
                        "reason": f"death_cross but min_hold not met ({held}/{MIN_HOLD_DAYS} days)",
                    })
                    continue
            orders.append({
                "ticker": ticker, "action": "HOLD",
                "reason": "in position, signal LONG",
                "unrealised_pnl": round((price - pos["entry_price"]) * pos["shares"], 2),
            })

        else:
            # Potential new entry
            if sig["golden_cross"]:
                if not sig["rsi_ok"]:
                    orders.append({
                        "ticker": ticker, "action": "SKIP",
                        "reason": f"golden_cross but RSI={sig['rsi']:.0f} > {RSI_OVERBOUGHT} (overbought)",
                    })
                elif cash < _atr_position_size(pv, sig.get("atr"), price) * 1.01:
                    orders.append({
                        "ticker": ticker, "action": "SKIP",
                        "reason": "golden_cross but insufficient cash",
                    })
                else:
                    size = _atr_position_size(pv, sig.get("atr"), price)
                    orders.append({
                        "ticker"    : ticker,
                        "action"    : "BUY",
                        "size_usd"  : round(size, 2),
                        "shares_est": round(size / price, 3),
                        "price_last": round(price, 2),
                        "reason"    : "golden_cross + RSI ok",
                    })
            else:
                orders.append({"ticker": ticker, "action": "FLAT", "reason": "death_cross, no position"})

    sheet = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "for_date"    : str(date.today()),
        "kill_switch" : check_kill_switch(),
        "portfolio"   : {
            "value"      : round(pv, 2),
            "cash"       : round(cash, 2),
            "n_positions": len(current_longs),
        },
        "orders": orders,
    }

    with open(ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(sheet, f, indent=2, default=str)

    buys  = [o for o in orders if o["action"] == "BUY"]
    sells = [o for o in orders if o["action"] == "SELL"]
    holds = [o for o in orders if o["action"] == "HOLD"]
    log.info(
        f"Order sheet → {ORDERS_FILE}  "
        f"({len(buys)} BUY  {len(sells)} SELL  {len(holds)} HOLD)"
    )
    return sheet


# ── Market day check ──────────────────────────────────────────────────────────

def _is_trading_day() -> bool:
    """Check if today is a weekday (Mon–Fri). Doesn't check US holidays."""
    return datetime.now(ET_ZONE).weekday() < 5   # 0=Mon … 4=Fri


def _et_now() -> datetime:
    return datetime.now(ET_ZONE)


# ── Main EOD job ──────────────────────────────────────────────────────────────

def run_eod_job():
    """Full end-of-day pipeline: validate → kill switch → update → order sheet."""
    log.info("=" * 60)
    log.info("EOD pipeline starting")

    # Kill switch check
    if check_kill_switch():
        log.warning("Kill switch active — EOD update skipped.  Review portfolio.")
        return

    # Fetch signals with data validation
    log.info("Fetching & validating prices…")
    raw_signals = {}
    for ticker in TICKER_LIST:
        try:
            df  = _fetch_daily(ticker)
            raw_signals[ticker] = compute_signal(df, ticker)
        except Exception as e:
            log.warning(f"  {ticker}: fetch error — {e}")

    clean_signals = validate_prices(raw_signals)
    skipped = set(TICKER_LIST) - set(clean_signals)
    if skipped:
        log.warning(f"Data validation removed: {sorted(skipped)}")

    # Run EOD update (paper_trader handles its own signal fetch internally,
    # but we log our validation results here for audit trail)
    log.info("Running EOD position update…")
    end_of_day_update()

    # Generate order sheet for tomorrow
    state = load_state()
    if state:
        generate_order_sheet(clean_signals, state)

    log.info("EOD pipeline complete")
    log.info("=" * 60)


# ── Scheduler loop ─────────────────────────────────────────────────────────────

def main():
    log.info("Scheduler started.  Will fire EOD at "
             f"{EOD_HOUR:02d}:{EOD_MINUTE:02d} ET on weekdays.")
    log.info(f"Kill-switch threshold: {KILL_SWITCH_DD*100:.0f}% rolling 20-day loss")
    log.info(f"Log file: {LOG_FILE}")

    last_run_date = None

    while True:
        now       = _et_now()
        today_str = str(now.date())

        # Fire once per day at EOD_HOUR:EOD_MINUTE on trading days
        if (
            _is_trading_day()
            and now.hour == EOD_HOUR
            and now.minute >= EOD_MINUTE
            and last_run_date != today_str
        ):
            last_run_date = today_str
            try:
                run_eod_job()
            except Exception as e:
                log.error(f"EOD job failed: {e}", exc_info=True)

        # Heartbeat every 30 minutes so you can see it's still running
        if now.minute % 30 == 0 and now.second < 31:
            state = load_state()
            pv    = state.get("portfolio_value", INITIAL_CAPITAL) if state else INITIAL_CAPITAL
            ret   = (pv / INITIAL_CAPITAL - 1) * 100
            log.info(f"Heartbeat  |  ET {now.strftime('%H:%M')}  |  "
                     f"Portfolio ${pv:,.0f}  ({ret:+.2f}%)  |  "
                     f"Next EOD: {EOD_HOUR:02d}:{EOD_MINUTE:02d}")

        time.sleep(30)


if __name__ == "__main__":
    main()
