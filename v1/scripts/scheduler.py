"""
scheduler.py — Automated EOD pipeline runner.
30-sec loop fires at 4:45 PM ET weekdays: kill-switch → paper-trade update →
validation → order sheet. Wait until 4:45 so yfinance closes settle.
Kill switch: pause trading if rolling 20-day DD > KILL_SWITCH_DD (15%).
Usage: `python scheduler.py` (foreground) or `pythonw scheduler.py` (Windows bg).
"""

import time
import json
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo       # stdlib >= 3.9

from v1.pipeline.data_pipeline import TICKER_LIST, ASSET_CLASS
from v1.scripts.paper_trader import (
    load_state, load_history, end_of_day_update,
    compute_live_signals, _fetch_daily, _atr_size,
    catchup, INITIAL_CAPITAL, PT_DIR,
)

# ── Config ─────────────────────────────────────────────────────────────────────
EOD_HOUR      = 16          # 4 PM ET
EOD_MINUTE    = 45          # 4:45 — let yfinance close settle
ET_ZONE       = ZoneInfo("America/New_York")
KILL_SWITCH_DD = 0.15       # pause if rolling 20-day loss > 15%
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
    """Return True (BLOCK trading) if rolling 20-day DD > KILL_SWITCH_DD (15%).
    20d/15% is ~3-sigma for 10% vol strategy — forces human review."""
    history = load_history()
    if len(history) < 2:
        return False

    recent   = history.tail(20)
    peak     = recent["portfolio_value"].max()
    current  = float(recent["portfolio_value"].iloc[-1])
    drawdown = (current - peak) / peak   # negative

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
    """Drop tickers failing sanity checks: finite positive price, single-day
    move <= 25% (1987 was -22.6%). Failures logged as warnings, not raised."""
    clean = {}
    for ticker, sig in signals.items():
        price = sig.get("close")
        if price is None or not np.isfinite(price) or price <= 0:
            log.warning(f"  {ticker}: invalid price {price} — skipped")
            continue
        # Large single-day move check via yfinance prev close
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
            pass   # fast_info unavailable — skip check, don't block trade
        clean[ticker] = sig
    return clean


# ── Pre-market order sheet ─────────────────────────────────────────────────────

def generate_order_sheet(signals: dict, state: dict):
    """Compute expected orders and persist to orders_tomorrow.json.
    Actions: BUY/SELL/HOLD/FLAT/SKIP (insufficient cash)/NO_DATA.
    Informational only — actual execution happens in end_of_day_update()."""
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

        price  = sig["close"]
        signal = sig["signal"]   # 0/1 — filters applied in signal_generation.py

        if ticker in current_longs:
            pos     = positions[ticker]
            unr_pnl = round((price - pos["entry_price"]) * pos["shares"], 2)
            if signal == 0:
                orders.append({
                    "ticker"        : ticker,
                    "action"        : "SELL",
                    "price_last"    : round(price, 2),
                    "reason"        : "signal_exit (death cross / trailing stop / RSI filter)",
                    "entry_price"   : pos["entry_price"],
                    "unrealised_pnl": unr_pnl,
                })
            else:
                orders.append({
                    "ticker"        : ticker,
                    "action"        : "HOLD",
                    "reason"        : "signal LONG — hold",
                    "unrealised_pnl": unr_pnl,
                })
        else:
            if signal == 1:
                size = _atr_size(pv, sig.get("atr"), price)
                if cash >= size * 1.01:
                    orders.append({
                        "ticker"    : ticker,
                        "action"    : "BUY",
                        "size_usd"  : round(size, 2),
                        "shares_est": round(size / price, 3),
                        "price_last": round(price, 2),
                        "reason"    : "signal_entry",
                    })
                else:
                    orders.append({
                        "ticker": ticker,
                        "action": "SKIP",
                        "reason": f"signal LONG but insufficient cash (need ${size:,.0f}, have ${cash:,.0f})",
                    })
            else:
                orders.append({"ticker": ticker, "action": "FLAT", "reason": "signal FLAT"})

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
    """True if weekday (Mon-Fri). Does not check US market holidays —
    on holidays yfinance returns no new data; future: pandas_market_calendars."""
    return datetime.now(ET_ZONE).weekday() < 5   # 0=Mon..4=Fri


def _et_now() -> datetime:
    return datetime.now(ET_ZONE)


# ── Main EOD job ──────────────────────────────────────────────────────────────

def run_eod_job():
    """Full EOD pipeline: kill-switch -> update -> validate -> order sheet.
    Runs once per trading day at EOD_HOUR:EOD_MINUTE ET."""
    log.info("=" * 60)
    log.info("EOD pipeline starting")

    # Kill switch
    if check_kill_switch():
        log.warning("Kill switch active — EOD update skipped.  Review portfolio.")
        return

    # EOD update — paper_trader fetches signals and executes
    log.info("Running EOD position update...")
    end_of_day_update()

    # Fetch signals again for validation + order sheet
    log.info("Fetching signals for order sheet and data validation...")
    try:
        raw_signals   = compute_live_signals()
        clean_signals = validate_prices(raw_signals)
        skipped       = set(TICKER_LIST) - set(clean_signals)
        if skipped:
            log.warning(f"Data validation flagged tickers: {sorted(skipped)}")

        state = load_state()
        if state:
            generate_order_sheet(clean_signals, state)
    except Exception as e:
        log.error(f"Order sheet generation failed: {e}", exc_info=True)

    log.info("EOD pipeline complete")
    log.info("=" * 60)


# ── Scheduler loop ─────────────────────────────────────────────────────────────

def main():
    """Scheduler entry point. Runs indefinitely; logs a heartbeat every 30 min."""
    log.info("Scheduler started.  Will fire EOD at "
             f"{EOD_HOUR:02d}:{EOD_MINUTE:02d} ET on weekdays.")
    log.info(f"Kill-switch threshold: {KILL_SWITCH_DD*100:.0f}% rolling 20-day loss")
    log.info(f"Log file: {LOG_FILE}")

    # Catch up on missed trading days; runs synchronously before main loop.
    try:
        n = catchup()
        if n:
            log.info(f"Catch-up complete — {n} missed day(s) replayed.")
        else:
            log.info("Catch-up: portfolio is up to date.")
    except Exception as e:
        log.error(f"Catch-up failed: {e}", exc_info=True)

    last_run_date = None

    while True:
        now       = _et_now()
        today_str = str(now.date())

        # Fire once per day at EOD_HOUR:EOD_MINUTE on weekdays
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

        # Heartbeat every 30 min
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
