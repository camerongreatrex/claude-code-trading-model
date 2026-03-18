"""
paper_trader.py

Pure execution engine.  ALL strategy logic lives in signal_generation.py
and portfolio.py — this file only answers the questions:
  "What does the strategy want?"  → compute_live_signals()
  "Execute those decisions."      → _buy(), _sell()

The three improvements (RSI entry filter, ATR trailing stop, min hold period)
are in signal_generation.py and therefore apply to the honest backtest too.
Paper trader runs the SAME code as the backtest — no divergence.

Usage:
    python paper_trader.py init    # one-time setup at today's close
    python paper_trader.py run     # end-of-day update (run after 4:30 PM ET)
    python paper_trader.py status  # print current state
"""

import json
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from pathlib import Path

from data_pipeline import TICKER_LIST, ASSET_CLASS
from feature_engineering import engineer
from signal_generation import generate, load_macro
from portfolio import atr_sizes, apply_macro_multiplier, CAPITAL, RISK_PER_TRADE, MAX_POSITION_PCT

# ── Constants ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = float(CAPITAL)          # same as portfolio.py ($100k)
COMMISSION_PCT  = 0.0005                  # 0.05% per side — matches backtest

PT_DIR       = Path("data/paper_trading")
STATE_FILE   = PT_DIR / "state.json"
TRADES_FILE  = PT_DIR / "trades.csv"
HISTORY_FILE = PT_DIR / "history.csv"
PT_DIR.mkdir(parents=True, exist_ok=True)


# ── State I/O ──────────────────────────────────────────────────────────────────

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


# ── Live data fetcher (yfinance, free, no API key) ────────────────────────────

def _fetch_daily(ticker: str, lookback_days: int = 700) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False,
        multi_level_index=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df


def fetch_intraday_batch(tickers: list) -> dict:
    """Today's 5-minute bars — used by the dashboard intraday chart.

    Uses period="2d" to get more complete data than period="1d" (yfinance
    sometimes cuts off early-session bars with period="1d").

    All data is converted to ET-naive timestamps then clipped to today's
    regular trading hours (09:25–16:05 ET).  The filter MUST run after the
    tz conversion — doing it before caused UTC values to be compared against
    ET thresholds, silently dropping all morning bars.
    """
    now_et    = pd.Timestamp.now(tz='America/New_York')
    today_str = now_et.strftime('%Y-%m-%d')
    mkt_open  = pd.Timestamp(today_str + " 09:25:00")   # naive ET
    mkt_close = pd.Timestamp(today_str + " 16:05:00")   # naive ET

    results = {}
    for ticker in tickers:
        try:
            # Ticker.history() returns tz-aware America/New_York timestamps
            # directly — more complete during live market hours than yf.download()
            raw = yf.Ticker(ticker).history(period="2d", interval="5m")
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            # Convert tz-aware ET → naive ET strings (wall-clock, DST-correct)
            et_idx = raw.index.tz_convert('America/New_York')
            raw.index = pd.DatetimeIndex(et_idx.strftime('%Y-%m-%d %H:%M:%S'))
            # Filter to today's regular trading hours (comparison is naive ET vs naive ET)
            raw = raw[(raw.index >= mkt_open) & (raw.index <= mkt_close)]
            if not raw.empty:
                results[ticker] = raw
        except Exception:
            pass
    return results


# ── Strategy pipeline (same code path as backtest) ────────────────────────────

def compute_live_signals() -> dict:
    """
    Run the full strategy pipeline on live yfinance data.

    Steps (identical to run.py pipeline):
      1. Fetch last 320 days of OHLCV via yfinance
      2. engineer() from feature_engineering.py (all features)
      3. generate() from signal_generation.py
         → includes RSI entry filter, min-hold filter, ATR trailing stop
      4. Extract latest row for each ticker

    Returns dict[ticker -> signal_info dict].
    """
    macro   = load_macro()
    results = {}

    for ticker in TICKER_LIST:
        print(f"  {ticker:<6}", end=" ", flush=True)
        try:
            raw  = _fetch_daily(ticker)
            feat = engineer(raw)
            sig  = generate(feat, ticker, macro)

            if sig.empty:
                print("no signal data")
                continue

            latest_sig  = int(sig["signal_regime"].iloc[-1])
            latest_comp = int(sig["signal_composite"].iloc[-1])
            latest_atr  = float(feat["atr_14"].iloc[-1])
            latest_close= float(feat["Close"].iloc[-1])
            latest_rsi  = float(feat["rsi_14"].iloc[-1])

            results[ticker] = {
                "signal"     : latest_sig,
                "composite"  : latest_comp,
                "close"      : latest_close,
                "atr"        : latest_atr,
                "rsi"        : latest_rsi,
                "date"       : str(feat.index[-1].date()),
            }
            print(f"${latest_close:>8.2f}  {'LONG' if latest_sig else 'FLAT'}")
        except Exception as e:
            print(f"ERROR {e}")

    return results


# ── ATR position sizing (mirrors portfolio.py atr_sizes) ─────────────────────

def _atr_size(pv: float, atr: float, price: float) -> float:
    """Dollar position size matching portfolio.py RISK_PER_TRADE / ATR logic."""
    if not atr or atr <= 0:
        return min(pv / len(TICKER_LIST), pv * MAX_POSITION_PCT)
    dollar_risk = pv * RISK_PER_TRADE
    dollar_pos  = (dollar_risk / atr) * price
    return min(dollar_pos, pv * MAX_POSITION_PCT)


# ── Trade execution ───────────────────────────────────────────────────────────

def _buy(state: dict, ticker: str, price: float,
         dollar_amount: float, reason: str, trade_date: str) -> dict:
    if ticker in state["positions"]:
        return state   # no averaging into existing positions

    commission = dollar_amount * COMMISSION_PCT
    total_cost = dollar_amount + commission

    if total_cost > state["cash"] * 1.01:
        dollar_amount = state["cash"] * 0.97
        commission    = dollar_amount * COMMISSION_PCT
        total_cost    = dollar_amount + commission

    if dollar_amount < 200:
        return state

    shares = dollar_amount / price
    state["cash"] -= total_cost
    state["positions"][ticker] = {
        "shares"    : shares,
        "entry_price": price,
        "entry_date" : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cost_basis" : dollar_amount,
    }
    _append_csv(TRADES_FILE, {
        "date": trade_date, "ticker": ticker, "action": "BUY",
        "shares": round(shares, 6), "price": round(price, 4),
        "value": round(dollar_amount, 2), "commission": round(commission, 2),
        "pnl": "", "reason": reason,
    })
    print(f"  BUY  {ticker:<6}  {shares:.3f} sh @ ${price:.2f}"
          f"  (${dollar_amount:,.0f})  [{reason}]")
    return state


def _sell(state: dict, ticker: str, price: float,
          reason: str, trade_date: str) -> dict:
    if ticker not in state["positions"]:
        return state
    pos        = state["positions"].pop(ticker)
    proceeds   = pos["shares"] * price
    commission = proceeds * COMMISSION_PCT
    net        = proceeds - commission
    pnl        = net - pos["cost_basis"]
    state["cash"] += net
    _append_csv(TRADES_FILE, {
        "date": trade_date, "ticker": ticker, "action": "SELL",
        "shares": round(pos["shares"], 6), "price": round(price, 4),
        "value": round(proceeds, 2), "commission": round(commission, 2),
        "pnl": round(pnl, 2), "reason": reason,
    })
    pnl_s = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
    print(f"  SELL {ticker:<6}  {pos['shares']:.3f} sh @ ${price:.2f}"
          f"  P&L {pnl_s}  [{reason}]")
    return state


def _portfolio_value(state: dict, prices: dict) -> float:
    return state["cash"] + sum(
        pos["shares"] * prices.get(t, pos["entry_price"])
        for t, pos in state["positions"].items()
    )


# ── Init ──────────────────────────────────────────────────────────────────────

def init_positions():
    """
    One-time setup.  Runs the full strategy pipeline on today's data,
    enters positions for all tickers with signal=1, sizes using ATR method.
    """
    if STATE_FILE.exists():
        print(f"Already initialised.  Delete {STATE_FILE} to reset.")
        return

    today_str = str(date.today())
    print(f"Initialising paper portfolio  ({today_str})\n")
    print("Running strategy pipeline on live data...\n")

    signals = compute_live_signals()
    prices  = {t: s["close"] for t, s in signals.items()}

    state = {
        "cash"            : INITIAL_CAPITAL,
        "positions"       : {},
        "initial_capital" : INITIAL_CAPITAL,
        "initialized_date": today_str,
        "last_eod_date"   : today_str,
        "portfolio_value" : INITIAL_CAPITAL,
    }

    longs = [t for t, s in signals.items() if s["signal"] == 1]
    print(f"\n{len(longs)}/{len(signals)} tickers: LONG signal — entering positions...\n")

    # Distribute capital across ALL active signals.
    # ATR sizing can exceed 100% of capital when many tickers are LONG simultaneously,
    # so we cap each position at capital / n_longs to ensure all signals are represented.
    per_position_cap = (INITIAL_CAPITAL * 0.97) / max(len(longs), 1)

    for ticker in longs:
        sig  = signals[ticker]
        size = min(_atr_size(INITIAL_CAPITAL, sig["atr"], sig["close"]), per_position_cap)
        state = _buy(state, ticker, sig["close"], size,
                     reason="init_golden_cross", trade_date=today_str)

    pv = _portfolio_value(state, prices)
    # Store INITIAL_CAPITAL as the baseline so "Portfolio Today" on day 1
    # is measured from $100k, not the post-commission entry value.
    state["portfolio_value"] = INITIAL_CAPITAL

    _append_csv(HISTORY_FILE, {
        "date": today_str, "portfolio_value": round(pv, 2),
        "cash": round(state["cash"], 2),
        "invested": round(pv - state["cash"], 2),
        "n_positions": len(state["positions"]), "daily_return": 0.0,
    })
    save_state(state)

    print(f"\n{'='*52}")
    print(f"  Positions   : {len(state['positions'])}")
    print(f"  Invested    : ${pv - state['cash']:>10,.2f}")
    print(f"  Cash        : ${state['cash']:>10,.2f}")
    print(f"  Total value : ${pv:>10,.2f}")
    print(f"{'='*52}")
    print(f"\nState -> {STATE_FILE}")
    print("Run 'python paper_trader.py run' (or scheduler.py) after each close.")


# ── End-of-day update ─────────────────────────────────────────────────────────

def end_of_day_update():
    """
    Run the strategy pipeline on today's close prices.
    The signal (signal_regime) from generate() already has all improvements
    baked in — this function just executes what the strategy decided.
    """
    state = load_state()
    if not state:
        print("No portfolio found.  Run 'python paper_trader.py init' first.")
        return

    today_str  = str(date.today())
    prev_value = state.get("portfolio_value", INITIAL_CAPITAL)

    print(f"\nEnd-of-day update  ({today_str})\n")
    print("Running strategy pipeline on live data...\n")

    signals = compute_live_signals()
    prices  = {t: s["close"] for t, s in signals.items()}

    # ── Exits: sell anything where signal_regime flipped to 0 ────────────────
    for ticker in list(state["positions"]):
        if ticker not in signals:
            continue
        if signals[ticker]["signal"] == 0:
            # Signal is 0 — could be death cross, trailing stop, or any filter.
            # The strategy layer already decided; execution just acts on it.
            state = _sell(state, ticker, prices[ticker],
                          reason="signal_exit", trade_date=today_str)

    # ── Entries: buy anything where signal_regime is 1 and not already long ──
    current_longs = set(state["positions"])
    pv = _portfolio_value(state, prices)

    for ticker in TICKER_LIST:
        if ticker in current_longs or ticker not in signals:
            continue
        if signals[ticker]["signal"] == 1:
            sig  = signals[ticker]
            size = _atr_size(pv, sig["atr"], sig["close"])
            if state["cash"] >= size * 1.01:
                state = _buy(state, ticker, sig["close"], size,
                             reason="signal_entry", trade_date=today_str)

    # ── Snapshot ──────────────────────────────────────────────────────────────
    pv        = _portfolio_value(state, prices)
    daily_ret = (pv / prev_value - 1) if prev_value > 0 else 0.0
    total_ret = (pv / INITIAL_CAPITAL - 1) * 100

    state["portfolio_value"] = pv
    state["last_eod_date"]   = today_str
    save_state(state)

    _append_csv(HISTORY_FILE, {
        "date": today_str, "portfolio_value": round(pv, 2),
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
    print(f"{'='*52}")


# ── Intraday curve (used by dashboard) ────────────────────────────────────────

def get_intraday_curve() -> tuple:
    """
    Live portfolio OHLC using today's 5-minute bars.
    Returns (DataFrame(index=ET-naive datetime, columns=[open,high,low,close]),
             dict of {ticker: last_price},
             pd.Series spy benchmark normalized to portfolio's starting value,
             float spy_pct_from_prev — SPY % change from yesterday's close).
    spy_pct_from_prev uses yesterday's daily close as the baseline, matching
    how TradingView / Yahoo Finance display the daily % change.
    """
    _empty_spy = pd.Series(dtype=float)
    empty = (pd.DataFrame(columns=["open", "high", "low", "close"]), {}, _empty_spy, None)
    state = load_state()
    if not state or not state.get("positions"):
        return empty

    cash      = state["cash"]
    positions = state["positions"]
    # Always include SPY so the benchmark curve is available even when SPY
    # is not one of the portfolio holdings.
    intraday  = fetch_intraday_batch(list(set(list(positions) + ["SPY"])))

    col_map     = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    all_series  = {key: {} for key in col_map}   # key -> {ticker: weighted series}
    last_prices = {}

    for ticker, pos in positions.items():
        if ticker not in intraday:
            continue
        df = intraday[ticker]
        for key, yf_col in col_map.items():
            col = df[yf_col] if yf_col in df.columns else None
            if col is None:
                continue
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            all_series[key][ticker] = col * pos["shares"]
        # last price per ticker for the positions table
        close_col = df["Close"] if "Close" in df.columns else df.iloc[:, 3]
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        if not close_col.empty:
            last_prices[ticker] = float(close_col.iloc[-1])

    if not all_series["close"]:
        return empty

    # Forward-fill before summing: tickers with a missing latest bar use their
    # previous price rather than contributing 0, which caused artificial portfolio
    # value dips every time one ticker lagged behind the others.
    ohlc_sums = {}
    for key, ticker_series in all_series.items():
        if not ticker_series:
            continue
        combined = pd.concat(list(ticker_series.values()), axis=1)
        combined = combined.ffill()
        ohlc_sums[key] = combined.sum(axis=1)

    if "close" not in ohlc_sums:
        return empty

    result = pd.DataFrame({k: v + cash for k, v in ohlc_sums.items()})
    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        et = result.index.tz_convert('America/New_York')
        result.index = pd.DatetimeIndex(et.strftime('%Y-%m-%d %H:%M:%S'))
    # fetch_intraday_batch already filters to today's regular hours — drop any
    # remaining NaN rows from the ffill/concat alignment step only.
    result = result.dropna(how="all")

    # Sanity check: reject clearly bad yfinance data (intermittent price spikes)
    if not result.empty:
        last_close = float(result["close"].iloc[-1])
        if last_close > INITIAL_CAPITAL * 2.5 or last_close < INITIAL_CAPITAL * 0.1:
            return empty

    today_str = pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d')

    # S&P 500 benchmark: normalize SPY to the portfolio's FIRST intraday bar value
    # so both lines start at the same point at market open and diverge from there.
    # Using prev_pv (yesterday's close) was wrong — the portfolio may gap up/down
    # at open, pushing the SPY line above/below the portfolio artificially.
    spy_curve        = _empty_spy
    spy_pct_from_prev = None   # None = daily fetch failed; dashboard falls back to first-bar %
    portfolio_open = float(result["close"].iloc[0]) if not result.empty else state.get("portfolio_value", INITIAL_CAPITAL)
    if "SPY" in intraday:
        spy_df  = intraday["SPY"]
        spy_col = spy_df["Close"] if "Close" in spy_df.columns else spy_df.iloc[:, 3]
        if isinstance(spy_col, pd.DataFrame):
            spy_col = spy_col.iloc[:, 0]
        # fetch_intraday_batch already clipped to today's regular hours
        spy_today = spy_col
        if not spy_today.empty:
            spy_curve = (spy_today / float(spy_today.iloc[0])) * portfolio_open

        # Compute SPY % from yesterday's official close (matches TradingView/Yahoo daily %).
        # Use period="5d" so we get enough rows, then filter to dates BEFORE today —
        # when the market is open, period="2d" returns an incomplete today row too,
        # causing .iloc[-1] to grab today's price (~same as current) → ~0% change.
        try:
            spy_daily = yf.download("SPY", period="5d", interval="1d",
                                    auto_adjust=True, progress=False)
            if isinstance(spy_daily.columns, pd.MultiIndex):
                spy_daily.columns = spy_daily.columns.droplevel(1)
            spy_dc = spy_daily["Close"] if "Close" in spy_daily.columns else spy_daily.iloc[:, 3]
            if isinstance(spy_dc, pd.DataFrame):
                spy_dc = spy_dc.iloc[:, 0]
            # Normalise index to date-only so comparison with today_str works regardless
            # of whether yfinance returns tz-aware timestamps or plain dates.
            spy_dc.index = pd.to_datetime(spy_dc.index).tz_localize(None).normalize()
            today_ts = pd.Timestamp(today_str)
            prev_rows = spy_dc[spy_dc.index < today_ts]
            if not prev_rows.empty and not spy_today.empty:
                spy_prev_close = float(prev_rows.iloc[-1])   # confirmed yesterday close
                if spy_prev_close > 0:
                    spy_pct_from_prev = (float(spy_today.iloc[-1]) / spy_prev_close - 1) * 100
        except Exception:
            pass

    return result, last_prices, spy_curve, spy_pct_from_prev


# ── Status ─────────────────────────────────────────────────────────────────────

def print_status():
    state = load_state()
    if not state:
        print("No portfolio found.  Run 'python paper_trader.py init'")
        return
    pv        = state.get("portfolio_value", INITIAL_CAPITAL)
    total_ret = (pv / INITIAL_CAPITAL - 1) * 100
    print(f"\nPaper Portfolio  ({state.get('last_eod_date','?')})")
    print(f"{'='*52}")
    print(f"  Portfolio value : ${pv:>10,.2f}")
    print(f"  Cash            : ${state['cash']:>10,.2f}")
    print(f"  Total return    : {total_ret:>+8.2f}%")
    print(f"\n  Open positions ({len(state['positions'])}):")
    for ticker, pos in state["positions"].items():
        print(f"    {ticker:<6}  {pos['shares']:.3f} sh  "
              f"@ ${pos['entry_price']:.2f}  since {pos['entry_date']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {
        "init":   init_positions,
        "run":    end_of_day_update,
        "status": print_status,
    }.get(cmd, lambda: print(f"Unknown: {cmd}\nUsage: python paper_trader.py [init|run|status]"))()
