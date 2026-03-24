"""
live_signals.py
---------------
Fetches live market data via yfinance and computes today's signal state
for the full trading universe.  Used by the dashboard's "Live Signals" tab.

Why a separate module from signal_generation.py?
────────────────────────────────────────────────
signal_generation.py runs against the full historical dataset stored in
data/features/ parquet files (produced by the offline pipeline).

live_signals.py operates on fresh data downloaded in real time from
yfinance, with no dependency on the parquet files.  This means the
dashboard can show current signal states even if data_pipeline.py has not
been run today.

The signal logic here intentionally mirrors signal_generation.generate()
for the regime signal (MA crossover + regime filter).  The composite score
is not replicated here — the MA signal is what paper_trader.py actually
trades, so it is the most useful live indicator.

Strategy (daily, long-only)
────────────────────────────
Signals update once per day at the official close.  During market hours,
they reflect the most recently available close (which may be yesterday's
if today's bar has not yet settled in yfinance).

Consumed by
───────────
  dashboard.py — Live Signals tab
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from pipeline.data_pipeline import TICKER_LIST, ASSET_CLASS


def fetch_live_data(tickers: list, lookback_days: int = 700) -> dict:
    """
    Download the most recent ``lookback_days`` of daily OHLCV for each ticker.

    Uses yfinance.download() — free, no API key, approximately 15-minute
    delayed data during market hours on the free tier.

    Args:
        tickers:       List of ticker symbols to download.
        lookback_days: Number of calendar days of history to fetch.
                       Default 700 days (~2.8 years) — enough for MA200
                       to warm up (200 trading days ≈ 280 calendar days).

    Returns:
        dict[ticker -> DataFrame].  Tickers that fail (network error,
        delisted, bad symbol) are silently skipped so one bad ticker does
        not block the rest of the universe.
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
    """
    Wilder's RSI using EWM — matches feature_engineering.add_rsi() exactly.

    Keeping this consistent is important: if the signal uses RSI=72 to
    decide to hold vs. exit, the live dashboard should show the same 72,
    not a slightly different number from a different RSI implementation.

    Args:
        close:  Close price Series.
        window: Look-back period (default 14).

    Returns:
        RSI Series (0–100).
    """
    delta    = close.diff()
    gains    = delta.clip(lower=0)
    losses   = delta.clip(upper=0).abs()
    avg_gain = gains.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = losses.ewm(com=window - 1, min_periods=window).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_hist(close: pd.Series) -> pd.Series:
    """
    MACD histogram (12/26/9) — matches feature_engineering.add_macd() exactly.

    Args:
        close: Close price Series.

    Returns:
        MACD histogram Series.  Positive = bullish momentum, negative = bearish.
    """
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    signal_line = (ema12 - ema26).ewm(span=9, adjust=False).mean()
    return (ema12 - ema26) - signal_line


def compute_live_signal(df: pd.DataFrame, ticker: str) -> dict:
    """
    Compute the current signal state and supporting indicators for one ticker.

    Mirrors the regime signal logic from signal_generation.generate():
      - MA50/200 for most assets; MA100/300 for sector ETFs
      - Golden cross (fast MA > slow MA) = LONG signal = 1
      - Otherwise = FLAT = 0

    Note: The full post-processors (RSI entry filter, min-hold, trailing stop)
    from signal_generation.py are NOT replicated here because they require
    per-bar state tracking over the full history.  The live dashboard shows
    the raw golden-cross signal — the post-processors only change the signal
    on edge cases (overbought entries, short holds, stop hits).

    Args:
        df:     OHLCV DataFrame with ≥ slow_w rows from fetch_live_data().
        ticker: Ticker symbol string (used for asset-class routing).

    Returns:
        Flat dict of scalars for the latest available bar:
          ticker, asset_class, price, day_chg_pct, signal, signal_label,
          ma_fast, ma_slow, ma_fast_label, ma_slow_label, ma_spread_pct,
          rsi, macd_hist, ret_20d_pct, ret_60d_pct, hi_52w, lo_52w,
          dist_from_high, last_date, volume, avg_vol_20d.
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

    # Donchian breakout + squeeze detection
    donchian_high  = df["High"].rolling(20).max()
    donchian_low   = df["Low"].rolling(20).min()
    donchian_width = (donchian_high - donchian_low) / close.replace(0, np.nan)
    width_rank     = donchian_width.rolling(252, min_periods=63).rank(pct=True)
    squeeze        = (width_rank < 0.20)
    breakout       = (close >= donchian_high)
    recent_sq      = squeeze.rolling(5, min_periods=1).max().astype(bool)

    # Volume z-score (if volume available)
    try:
        _vol = df["Volume"].replace(0, np.nan)
        vol_zscore   = ((_vol - _vol.rolling(20).mean()) /
                        _vol.rolling(20).std().replace(0, np.nan)).fillna(0)
        vol_confirm  = bool(vol_zscore.iloc[-1] > 0.5)
    except Exception:
        vol_confirm = False

    # Fast MA20/50 for liquid ETFs
    FAST_TICKERS = {"SPY", "IWM", "TLT", "GLD", "EEM"}
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    if ticker in FAST_TICKERS:
        fast_sig   = int(ma20.iloc[-1] > ma50.iloc[-1]) if not (pd.isna(ma20.iloc[-1]) or pd.isna(ma50.iloc[-1])) else None
        fast_label = "FAST LONG" if fast_sig == 1 else "FAST FLAT"
    else:
        fast_sig   = None
        fast_label = "—"

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
        "breakout"      : int(bool(breakout.iloc[-1])),
        "squeeze"       : int(bool(recent_sq.iloc[-1])),
        "fast_signal"   : fast_sig,
        "fast_label"    : fast_label,
    }


def get_live_signals() -> tuple[pd.DataFrame, str]:
    """
    Fetch live data for the full universe and return a display-ready DataFrame.

    This is the primary entry point called by dashboard.py.  Results are
    cached by Streamlit (via @st.cache_data with a short TTL) to avoid
    fetching the same data on every page interaction.

    Returns:
        Tuple of (display_df, fetch_timestamp):
          display_df: DataFrame with human-readable column names, ready for
                      st.dataframe().  Empty DataFrame if all fetches failed.
          fetch_timestamp: ISO datetime string when the fetch ran.
    """
    raw     = fetch_live_data(TICKER_LIST, lookback_days=700)
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

    # Momentum rank: rank all tickers by 63-day return (pct rank 0–100)
    if "ret_60d_pct" in df.columns:
        mom_rank_pct = df["ret_60d_pct"].rank(pct=True) * 100
    else:
        mom_rank_pct = pd.Series(["—"] * len(df))

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
        "Mom Rank"        : mom_rank_pct.round(0).astype(int).astype(str) + "%" if "ret_60d_pct" in df.columns else "—",
        "Breakout"        : df["breakout"].map(lambda v: "✓" if v else ""),
        "Squeeze"         : df["squeeze"].map(lambda v: "✓" if v else ""),
        "Fast"            : df["fast_label"],
        "Dist High %"     : df["dist_from_high"].round(1),
        "Last Date"       : df["last_date"].astype(str),
    })

    return display, ts
