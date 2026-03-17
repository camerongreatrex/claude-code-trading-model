"""
feature_engineering.py

Pure math on clean price data — no downloads, no file paths.
Features are computed identically for all asset classes.
Signal generation uses asset class labels to decide which features to act on.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path("data/raw")
FEATURE_DIR = Path("data/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

from data_pipeline import TICKER_LIST, ASSET_CLASS


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_return"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["dollar_volume"] = df["Close"] * df["Volume"]

    prev = df["Close"].shift(1)
    df["true_range"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)

    df["atr_14"] = df["true_range"].rolling(14, min_periods=1).mean()
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mom_5"]  = df["Close"].pct_change(5)
    df["mom_20"] = df["Close"].pct_change(20)
    df["mom_60"] = df["Close"].pct_change(60)
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df    = df.copy()
    delta = df["Close"].diff()
    gains  = delta.clip(lower=0)
    losses = delta.clip(upper=0).abs()

    avg_gain = gains.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = losses.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))  # >70 overbought, <30 oversold
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["macd_line"]   = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]  # crossing zero = momentum turning
    return df


def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ADX: trend strength 0-100. >25 = real trend. <20 = ranging/choppy.
    Fires faster than MA crossovers — responds to recent price action directly.
    """
    df = df.copy()
    high, low = df["High"], df["Low"]
    prev_high, prev_low = high.shift(1), low.shift(1)

    plus_dm  = (high - prev_high).clip(lower=0)
    plus_dm[plus_dm < (prev_low - low).clip(lower=0)] = 0

    minus_dm = (prev_low - low).clip(lower=0)
    minus_dm[minus_dm < (high - prev_high).clip(lower=0)] = 0

    atr_s    = df["true_range"].ewm(span=window, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=window,  adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(span=window, adjust=False).mean() / atr_s

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    df["adx"]      = dx.ewm(span=window, adjust=False).mean()
    df["plus_di"]  = plus_di   # +DI > -DI = uptrend
    df["minus_di"] = minus_di  # -DI > +DI = downtrend
    return df


def add_zscore(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for window in [20, 60]:
        roll = df["Close"].rolling(window)
        df[f"zscore_{window}"] = (df["Close"] - roll.mean()) / roll.std()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df     = df.copy()
    middle = df["Close"].rolling(window).mean()
    std    = df["Close"].rolling(window).std()
    upper  = middle + num_std * std
    lower  = middle - num_std * std

    df["bb_middle"]    = middle
    df["bb_upper"]     = upper
    df["bb_lower"]     = lower
    df["bb_pct_b"]     = (df["Close"] - lower) / (upper - lower)
    df["bb_bandwidth"] = (upper - lower) / middle
    return df


def add_volume_signals(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df       = df.copy()
    vol_roll = df["Volume"].rolling(window)

    df["volume_zscore"] = (df["Volume"] - vol_roll.mean()) / vol_roll.std()
    df["obv"]           = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()

    obv_roll = df["obv"].rolling(window)
    df["obv_zscore"] = (df["obv"] - obv_roll.mean()) / obv_roll.std()
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all features in the correct order. Order matters — ADX needs true_range."""
    df = add_base_features(df)
    df = add_momentum(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_adx(df)
    df = add_zscore(df)
    df = add_bollinger_bands(df)
    df = add_volume_signals(df)
    df = df.dropna()
    return df


def main():
    print("Engineering features...\n")
    all_data = {}

    for ticker in TICKER_LIST:
        raw_path = DATA_DIR / f"{ticker}.parquet"
        if not raw_path.exists():
            print(f"  {ticker}: raw parquet missing — skipped (run data_pipeline.py first)")
            continue
        df  = pd.read_parquet(raw_path)
        df  = engineer(df)
        out = FEATURE_DIR / f"{ticker}.parquet"
        df.to_parquet(out, engine="pyarrow", compression="snappy")
        print(f"  {ticker}: {len(df)} rows, {len(df.columns)} columns  ->  {out}")
        all_data[ticker] = df

    returns = pd.DataFrame({t: d["log_return"] for t, d in all_data.items()}).dropna()
    returns.to_parquet(FEATURE_DIR / "returns_matrix.parquet")

    print(f"\nReturns matrix: {returns.shape}")
    print("\nAnnualised volatility:")
    for t in returns.columns:
        ann_vol = returns[t].std() * (252 ** 0.5) * 100
        print(f"  {t} ({ASSET_CLASS[t]}): {ann_vol:.1f}%")


if __name__ == "__main__":
    main()