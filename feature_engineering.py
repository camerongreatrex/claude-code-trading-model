"""
feature_engineering.py

Pure math on clean price data — no downloads, no file paths.
Takes a raw OHLCV DataFrame, returns it enriched with signal columns.
Import these functions anywhere: backtester, live trading, notebooks.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path("data/raw")
FEATURE_DIR = Path("data/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

# Base features — derived directly from OHLCV, used by other functions below
def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # log return: ln(today / yesterday) — additive over time, stable for modelling
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # dollar volume: price x volume — normalises liquidity across cheap vs expensive stocks
    df["dollar_volume"] = df["Close"] * df["Volume"]

    prev = df["Close"].shift(1)

    # true range: max of intraday range, gap up, gap down — captures overnight moves
    df["true_range"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)

    # ATR-14: 14-day average true range — standard volatility ruler for position sizing
    df["atr_14"] = df["true_range"].rolling(14, min_periods=1).mean()

    return df

# Momentum — is the stock trending up or down?
def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # simple lookback returns: how much has price moved over N days?
    df["mom_5"]  = df["Close"].pct_change(5)   # 1-week
    df["mom_20"] = df["Close"].pct_change(20)  # 1-month
    df["mom_60"] = df["Close"].pct_change(60)  # 1-quarter

    return df

# Relative Strength Index (RSI) — is the stock overbought or oversold?
def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()

    delta  = df["Close"].diff()
    gains  = delta.clip(lower=0)
    losses = delta.clip(upper=0).abs()

    # ewm with Wilder's smoothing: recent days weighted more heavily than older ones
    avg_gain = gains.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = losses.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss  # ratio of avg gain to avg loss

    # RSI = 100 - (100 / (1 + RS)) — squishes RS into 0-100
    # >70 overbought (likely pullback), <30 oversold (likely bounce)
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))

    return df

# Moving Average Convergence Divergence (MACD) — is the stock accelerating or decelerating?
def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()

    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()  # short-term trend
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()  # long-term trend

    # MACD line: fast - slow — positive means short-term outpacing long-term (accelerating)
    df["macd_line"]   = ema_fast - ema_slow

    # signal line: smoothed MACD — reduces noise, used as the trigger
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean()

    # histogram: MACD - signal — crossing zero is the actual buy/sell event
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]

    return df

# Mean reversion — is the stock unusually far from where it normally trades?
def add_zscore(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for window in [20, 60]:
        roll = df["Close"].rolling(window)
        # (price - mean) / std: +2 = expensive, -2 = cheap, relative to recent history
        df[f"zscore_{window}"] = (df["Close"] - roll.mean()) / roll.std()

    return df

# Bollinger Bands — is the stock unusually far from where it normally trades?
def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()

    middle = df["Close"].rolling(window).mean()
    std    = df["Close"].rolling(window).std()
    upper  = middle + num_std * std
    lower  = middle - num_std * std

    df["bb_middle"] = middle
    df["bb_upper"]  = upper
    df["bb_lower"]  = lower

    # %B: where is price within the bands? 0=lower band, 0.5=middle, 1=upper, >1=broken out
    df["bb_pct_b"]     = (df["Close"] - lower) / (upper - lower)

    # bandwidth: how wide are the bands relative to price?
    # a squeeze (narrow bands) often precedes a large directional move
    df["bb_bandwidth"] = (upper - lower) / middle

    return df

# Volume — did the market believe the price move?
def add_volume_signals(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()

    vol_roll = df["Volume"].rolling(window)

    # volume z-score: how unusual is today's volume vs recent average?
    # high volume on up day = conviction; high volume on down day = distribution (selling pressure)
    df["volume_zscore"] = (df["Volume"] - vol_roll.mean()) / vol_roll.std()

    # OBV: running total — add volume on up days, subtract on down days
    # np.sign gives +1/-1/0, cumsum builds the running total
    # rising OBV with flat price = quiet accumulation — price usually follows
    df["obv"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()

    # OBV z-score: normalise OBV so it's comparable across tickers and time periods
    obv_roll = df["obv"].rolling(window)
    df["obv_zscore"] = (df["obv"] - obv_roll.mean()) / obv_roll.std()

    return df

# Master function — runs all features in the correct order
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Apply all feature functions to a raw OHLCV DataFrame.
    df = add_base_features(df)
    df = add_momentum(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_zscore(df)
    df = add_bollinger_bands(df)
    df = add_volume_signals(df)
    df = df.dropna()  # drop leading NaNs from rolling windows
    return df

# Main — load raw parquet files, engineer features, save to data/features
def main():
    all_data = {}
    print("Engineering features...\n")
  
    for ticker in TICKERS:
        df = pd.read_parquet(DATA_DIR / f"{ticker}.parquet")
        df = engineer(df)

        out = FEATURE_DIR / f"{ticker}.parquet"
        df.to_parquet(out, engine="pyarrow", compression="snappy")
        print(f"  {ticker}: {len(df)} rows, {len(df.columns)} columns  ->  {out}")

        all_data[ticker] = df

    # returns matrix: used by signal_generation.py and backtester.py
    returns = pd.DataFrame({t: d["log_return"] for t, d in all_data.items()}).dropna()
    returns.to_parquet(FEATURE_DIR / "returns_matrix.parquet")

    print(f"\nReturns matrix: {returns.shape}")

    print("\nAnnualised volatility:")
    for t in TICKERS:
        ann_vol = returns[t].std() * (252 ** 0.5) * 100  # daily standard deviation x sqrt(252 trading days)
        print(f"  {t}: {ann_vol:.1f}%")

    print("\nSample — AAPL last 3 rows:")
    cols = [
        "Close",
        "mom_5", "mom_20", "mom_60",
        "zscore_20", "rsi_14",
        "macd_hist",
        "bb_pct_b", "bb_bandwidth",
        "volume_zscore", "obv_zscore",
    ]
    print(all_data["AAPL"][cols].tail(3).round(4))


if __name__ == "__main__":
    main()