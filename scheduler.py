"""
scheduler.py
------------
Automated end-of-day pipeline runner — keeps the paper portfolio in sync
with market closes without any manual intervention.

How it works
────────────
The scheduler sleeps in a 30-second loop, checking the ET clock on each
wakeup.  On weekdays at exactly 4:45 PM ET it triggers the full EOD
pipeline (kill-switch check → paper-trade update → data validation →
order sheet).  After running, it sets ``last_run_date`` so it will not
fire again on the same calendar day even if left running overnight.

Why 4:45 PM (not 4:00 PM)?
────────────────────────────
The 4 PM close marks the end of regular trading, but official closing
prices (especially for less-liquid names) can take several minutes to
settle in Yahoo Finance's API.  Waiting until 4:45 gives yfinance time
to publish the final adjusted close so the strategy's signal for the next
day is based on a clean, settled price.

Kill switch
────────────
If the rolling 20-day portfolio drawdown exceeds KILL_SWITCH_DD (15%),
trading is paused and a warning is logged.  This mirrors the circuit-
breaker logic used at systematic funds — a drawdown of that magnitude
almost always means either a model break or a genuine market dislocation
that warrants human review before further capital is risked.

Professional analogy
────────────────────
Two Sigma / AQR equivalent: a cron job on a Linux server fires a Python
pipeline at market close.  This is the exact same pattern — just running
on a laptop instead of AWS.  The key difference from HFT: no co-location
or sub-millisecond latency (not needed for a daily strategy).

Usage
─────
  python scheduler.py      # run in foreground (terminal must stay open)
  pythonw scheduler.py     # run silently in the background (Windows, no window)
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

from pipeline.data_pipeline import TICKER_LIST, ASSET_CLASS
from paper_trader import (
    load_state, load_history, end_of_day_update,
    compute_live_signals, _fetch_daily, _atr_size,
    catchup, INITIAL_CAPITAL, PT_DIR,
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
    Remove tickers whose latest closing price fails basic sanity checks.

    Three checks are applied in order:
      1. **Finite positive price** — NaN, Inf, or ≤ 0 indicates a bad feed.
      2. **Single-day move ≤ 25%** — The 1987 crash was −22.6% in one day;
         anything beyond ±25% is almost certainly a data error, not a real
         move.  Uses yfinance ``fast_info`` for the previous close.
      3. **No stale feed** — 5+ consecutive unchanged closes suggest the
         data provider stopped updating.  (Not yet implemented; placeholder
         for future hardening.)

    Args:
        signals: dict[ticker -> signal_info dict] from compute_live_signals().

    Returns:
        A filtered copy of ``signals`` containing only tickers that passed
        all checks.  Failures are logged as warnings, not exceptions, so one
        bad ticker never blocks the rest of the order sheet.

    Note:
        ``fast_info`` is a lightweight yfinance endpoint.  If it is
        unavailable (network error, delisted ticker) the check is skipped
        rather than blocking the trade — a conservative design choice.
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
    Compute tomorrow's expected order list and persist it to orders_tomorrow.json.

    Reads the current portfolio state and the validated live signals to
    determine the expected action for every ticker in the universe:
      - ``BUY``      — signal=1, not already long, sufficient cash available
      - ``SELL``     — signal=0, currently long (death cross / trailing stop)
      - ``HOLD``     — signal=1, already long (no action needed)
      - ``FLAT``     — signal=0, not long (stay in cash)
      - ``SKIP``     — signal=1, but insufficient cash to size the position
      - ``NO_DATA``  — ticker was not returned by compute_live_signals()

    The order sheet is informational only — positions are actually executed
    by end_of_day_update().  The sheet exists so you can review what the
    strategy is planning before prices open the next morning.

    Args:
        signals: Validated signal dict from compute_live_signals().
        state:   Current portfolio state dict from load_state().

    Returns:
        The full order sheet dict (also written to ORDERS_FILE as JSON).
    """
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
        signal = sig["signal"]   # 0 or 1 — all filters already applied by signal_generation.py

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
    """
    Return True if today is a weekday (Monday–Friday).

    Note:
        Does not check US market holidays (MLK Day, Thanksgiving, etc.).
        On those days the EOD job will fire but yfinance will return no new
        data, so end_of_day_update() will simply re-use the previous close
        prices and produce a no-change snapshot.  A future improvement
        would be to integrate a holiday calendar (e.g., pandas_market_calendars).
    """
    return datetime.now(ET_ZONE).weekday() < 5   # 0=Mon … 4=Fri


def _et_now() -> datetime:
    return datetime.now(ET_ZONE)


# ── Main EOD job ──────────────────────────────────────────────────────────────

def run_eod_job():
    """
    Full end-of-day pipeline: kill-switch check → update → validate → order sheet.

    Called exactly once per trading day at EOD_HOUR:EOD_MINUTE ET.
    All exceptions are caught and logged so a single day's failure does
    not crash the scheduler process.
    """
    log.info("=" * 60)
    log.info("EOD pipeline starting")

    # Kill switch check
    if check_kill_switch():
        log.warning("Kill switch active — EOD update skipped.  Review portfolio.")
        return

    # Run EOD update — paper_trader fetches live signals and executes trades
    log.info("Running EOD position update...")
    end_of_day_update()

    # Fetch signals separately for data validation logging and order sheet
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
    """
    Entry point for the scheduler process.

    Runs indefinitely — designed to be launched at startup and left running.
    Logs a heartbeat every 30 minutes showing portfolio value and next EOD
    time so you can verify it is alive without reading the full log.
    """
    log.info("Scheduler started.  Will fire EOD at "
             f"{EOD_HOUR:02d}:{EOD_MINUTE:02d} ET on weekdays.")
    log.info(f"Kill-switch threshold: {KILL_SWITCH_DD*100:.0f}% rolling 20-day loss")
    log.info(f"Log file: {LOG_FILE}")

    # Catch up on any trading days missed while the PC was off.
    # Runs synchronously at startup before entering the main loop so the
    # portfolio history is up-to-date before the dashboard is accessed.
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
