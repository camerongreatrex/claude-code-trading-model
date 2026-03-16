"""

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

# -----------------------------------------------------------------------------
# Base features
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Momentum
# -----------------------------------------------------------------------------

def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # simple lookback returns: how much has price moved over N days?
    df["mom_5"]  = df["Close"].pct_change(5)
    df["mom_20"] = df["Close"].pct_change(20)
    df["mom_60"] = df["Close"].pct_change(60)

    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()

    delta  = df["Close"].diff()
    gains  = delta.clip(lower=0)
    losses = delta.clip(upper=0).abs()

    # Wilder's smoothing: recent days weighted more than older ones
    avg_gain = gains.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = losses.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))  # >70 overbought, <30 oversold

    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()

    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["macd_line"]   = ema_fast - ema_slow        # positive = short-term outpacing long-term
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]  # crossing zero = momentum turning

    return df

# -----------------------------------------------------------------------------
# ADX — trend strength indicator
# -----------------------------------------------------------------------------

def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ADX (Average Directional Index): measures trend STRENGTH, not direction.
    Ranges 0-100. Above 25 = real trend exists. Below 20 = ranging/choppy.

    This fires faster than MA50/200 crossovers because it responds to recent
    price action directly rather than waiting for slow averages to cross.

    Built from two directional movement components:
      +DM: how much did today's high exceed yesterday's high? (upward pressure)
      -DM: how much did today's low fall below yesterday's low? (downward pressure)
    DX = abs(+DI - -DI) / (+DI + -DI) * 100  — how strongly is one side winning?
    ADX = smoothed average of DX over the window
    """
    df   = df.copy()
    high = df["High"]
    low  = df["Low"]
    prev_high = high.shift(1)
    prev_low  = low.shift(1)

    # +DM: today's upward move beyond yesterday's high (0 if no new high)
    plus_dm  = (high - prev_high).clip(lower=0)
    plus_dm[plus_dm < (prev_low - low).clip(lower=0)] = 0  # -DM dominates

    # -DM: today's downward move beyond yesterday's low (0 if no new low)
    minus_dm = (prev_low - low).clip(lower=0)
    minus_dm[minus_dm < (high - prev_high).clip(lower=0)] = 0  # +DM dominates

    # smooth both DM components and ATR over the window
    atr_s      = df["true_range"].ewm(span=window, adjust=False).mean()
    plus_di    = 100 * plus_dm.ewm(span=window, adjust=False).mean()  / atr_s
    minus_di   = 100 * minus_dm.ewm(span=window, adjust=False).mean() / atr_s

    # DX: how strongly is one directional component dominating?
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))

    # ADX: smooth DX — slow-moving, confirms sustained trends rather than one-day spikes
    df["adx"]      = dx.ewm(span=window, adjust=False).mean()
    df["plus_di"]  = plus_di   # +DI > -DI = uptrend
    df["minus_di"] = minus_di  # -DI > +DI = downtrend

    return df

# -----------------------------------------------------------------------------
# Mean reversion
# -----------------------------------------------------------------------------

def add_zscore(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for window in [20, 60]:
        roll = df["Close"].rolling(window)
        # (price - mean) / std: +2 = expensive, -2 = cheap, relative to recent history
        df[f"zscore_{window}"] = (df["Close"] - roll.mean()) / roll.std()

    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()

    middle = df["Close"].rolling(window).mean()
    std    = df["Close"].rolling(window).std()
    upper  = middle + num_std * std
    lower  = middle - num_std * std

    df["bb_middle"]    = middle
    df["bb_upper"]     = upper
    df["bb_lower"]     = lower
    df["bb_pct_b"]     = (df["Close"] - lower) / (upper - lower)  # 0=lower, 1=upper
    df["bb_bandwidth"] = (upper - lower) / middle  # squeeze = big move coming

    return df

# -----------------------------------------------------------------------------
# Volume
# -----------------------------------------------------------------------------

def add_volume_signals(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()

    vol_roll = df["Volume"].rolling(window)

    # volume z-score: how unusual is today's volume vs recent average?
    df["volume_zscore"] = (df["Volume"] - vol_roll.mean()) / vol_roll.std()

    # OBV: add volume on up days, subtract on down days — accumulation tracker
    df["obv"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()

    obv_roll = df["obv"].rolling(window)
    df["obv_zscore"] = (df["obv"] - obv_roll.mean()) / obv_roll.std()

    return df

# -----------------------------------------------------------------------------
# Master function
# -----------------------------------------------------------------------------

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature functions in the correct order."""
    df = add_base_features(df)   # must be first — true_range used by ADX
    df = add_momentum(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_adx(df)             # needs true_range from base features
    df = add_zscore(df)
    df = add_bollinger_bands(df)
    df = add_volume_signals(df)
    df = df.dropna()
    return df

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Engineering features...\n")
    all_data = {}

    for ticker in TICKERS:
        df = pd.read_parquet(DATA_DIR / f"{ticker}.parquet")
        df = engineer(df)

        out = FEATURE_DIR / f"{ticker}.parquet"
        df.to_parquet(out, engine="pyarrow", compression="snappy")
        print(f"  {ticker}: {len(df)} rows, {len(df.columns)} columns  ->  {out}")
        all_data[ticker] = df

    returns = pd.DataFrame({t: d["log_return"] for t, d in all_data.items()}).dropna()
    returns.to_parquet(FEATURE_DIR / "returns_matrix.parquet")

    print(f"\nReturns matrix: {returns.shape}")
    print("\nAnnualised volatility:")
    for t in TICKERS:
        ann_vol = returns[t].std() * (252 ** 0.5) * 100
        print(f"  {t}: {ann_vol:.1f}%")

    print("\nSample — AAPL last 3 rows:")
    cols = ["Close", "mom_20", "zscore_20", "rsi_14", "macd_hist", "adx", "bb_pct_b", "volume_zscore"]
    print(all_data["AAPL"][cols].tail(3).round(4))


if __name__ == "__main__":
    main()