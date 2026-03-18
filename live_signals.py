"""
live_signals.py

Fetches fresh daily OHLCV data via yfinance (free, no API key required)
and computes today's signal state for the full universe.

The strategy is a daily close strategy — signals update once per day.
This module replicates the core MA-crossover regime logic from
signal_generation.py against the most recent live prices.

Used by dashboard.py for the Live Signals tab.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from data_pipeline import TICKER_LIST, ASSET_CLASS


def fetch_live_data(tickers: list, lookback_days: int = 320) -> dict:
    """
    Download the last `lookback_days` of OHLCV data for each ticker.
    Returns dict[ticker -> DataFrame].  Tickers that fail are silently skipped.
    """
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    results = {}
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            if len(df) >= 50:
                results[ticker] = df
        except Exception as e:
            print(f"  {ticker}: fetch failed — {e}")

    return results


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def _macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    signal_line = (ema12 - ema26).ewm(span=9, adjust=False).mean()
    return (ema12 - ema26) - signal_line


def compute_live_signal(df: pd.DataFrame, ticker: str) -> dict:
    """
    Compute the current MA-crossover signal and supporting indicators.
    Mirrors the logic in signal_generation.generate() for the regime signal.

    Returns a flat dict of scalars for the latest available bar.
    """
    asset_class = ASSET_CLASS[ticker]
    close       = df["Close"]

    # MA window selection matches signal_generation.py
    if asset_class == "sector_etf":
        fast_w, slow_w       = 100, 300
        fast_lbl, slow_lbl   = "MA100", "MA300"
    else:
        fast_w, slow_w       = 50, 200
        fast_lbl, slow_lbl   = "MA50",  "MA200"

    ma_fast = close.rolling(fast_w).mean()
    ma_slow = close.rolling(slow_w).mean()
    rsi     = _rsi(close)
    macd_h  = _macd_hist(close)

    # ── Latest values ────────────────────────────────────────────────────────
    latest_close = float(close.iloc[-1])
    prev_close   = float(close.iloc[-2]) if len(close) > 1 else latest_close

    f = float(ma_fast.iloc[-1]) if not pd.isna(ma_fast.iloc[-1]) else None
    s = float(ma_slow.iloc[-1]) if not pd.isna(ma_slow.iloc[-1]) else None

    # Golden-cross signal (matches signal_generation.py long-only logic)
    if f is not None and s is not None:
        signal    = 1 if f > s else 0
        ma_spread = (f / s - 1) * 100      # positive = fast above slow
    else:
        signal    = 0
        ma_spread = 0.0

    # 20-day and 60-day returns
    ret_20d = (latest_close / float(close.iloc[-21]) - 1) * 100 if len(close) > 21 else 0.0
    ret_60d = (latest_close / float(close.iloc[-61]) - 1) * 100 if len(close) > 61 else 0.0

    # Distance from 52-week high/low
    hi_52w = float(close.tail(252).max())
    lo_52w = float(close.tail(252).min())
    dist_from_high = (latest_close / hi_52w - 1) * 100

    return {
        "ticker"        : ticker,
        "asset_class"   : asset_class.replace("_", " ").title(),
        "price"         : latest_close,
        "day_chg_pct"   : (latest_close / prev_close - 1) * 100,
        "signal"        : signal,
        "signal_label"  : "LONG" if signal == 1 else "FLAT",
        "ma_fast"       : f,
        "ma_slow"       : s,
        "ma_fast_label" : fast_lbl,
        "ma_slow_label" : slow_lbl,
        "ma_spread_pct" : ma_spread,
        "rsi"           : float(rsi.iloc[-1])    if not pd.isna(rsi.iloc[-1])    else 50.0,
        "macd_hist"     : float(macd_h.iloc[-1]) if not pd.isna(macd_h.iloc[-1]) else 0.0,
        "ret_20d_pct"   : ret_20d,
        "ret_60d_pct"   : ret_60d,
        "hi_52w"        : hi_52w,
        "lo_52w"        : lo_52w,
        "dist_from_high": dist_from_high,
        "last_date"     : df.index[-1].date(),
        "volume"        : float(df["Volume"].iloc[-1]),
        "avg_vol_20d"   : float(df["Volume"].tail(20).mean()),
    }


def get_live_signals() -> tuple[pd.DataFrame, str]:
    """
    Fetch live data and return (signals_df, fetch_timestamp).

    signals_df columns:
        Ticker, Asset Class, Price, Day Chg %, Signal, MA Spread %,
        RSI, 20d Ret %, 60d Ret %, Dist from High %, Last Date
    """
    raw     = fetch_live_data(TICKER_LIST, lookback_days=320)
    rows    = []

    for ticker in TICKER_LIST:
        if ticker not in raw:
            continue
        row = compute_live_signal(raw[ticker], ticker)
        rows.append(row)

    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not rows:
        return pd.DataFrame(), ts

    df = pd.DataFrame(rows)

    display = pd.DataFrame({
        "Ticker"          : df["ticker"],
        "Asset Class"     : df["asset_class"],
        "Price"           : df["price"].round(2),
        "Day Chg %"       : df["day_chg_pct"].round(2),
        "Signal"          : df["signal_label"],
        "MA Spread %"     : df["ma_spread_pct"].round(2),
        "RSI"             : df["rsi"].round(1),
        "20d Ret %"       : df["ret_20d_pct"].round(1),
        "60d Ret %"       : df["ret_60d_pct"].round(1),
        "Dist High %"     : df["dist_from_high"].round(1),
        "Last Date"       : df["last_date"].astype(str),
    })

    return display, ts
