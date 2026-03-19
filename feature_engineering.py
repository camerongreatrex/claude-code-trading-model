"""
feature_engineering.py
-----------------------
Transforms raw OHLCV price data into the numerical features that the signal
generation layer uses to make trading decisions.

Design principle
────────────────
This module is pure mathematics — no downloads, no file I/O during the
transformation, no asset-class logic.  Features are computed identically
for every ticker.  It is the job of signal_generation.py to decide which
features to act on for which asset class.

Features produced (per ticker)
──────────────────────────────
  log_return       — daily log return (lognormal, additive over time)
  dollar_volume    — close × volume (liquidity proxy)
  true_range       — Wilder's TR: max(H-L, |H-prev_C|, |L-prev_C|)
  atr_14           — 14-day average true range (volatility measure)
  mom_5/20/60      — 5-, 20-, 60-day price momentum (% change)
  rsi_14           — Wilder's Relative Strength Index, 14-period EWM
  macd_line        — MACD line (EMA12 − EMA26)
  macd_signal      — Signal line (EMA9 of MACD line)
  macd_hist        — MACD histogram (line − signal; positive = bullish momentum)
  adx              — Average Directional Index: trend strength 0–100
  plus_di/minus_di — Directional Indicators (+DI > −DI = uptrend)
  zscore_20/60     — Price z-score over 20- and 60-day rolling windows
  bb_middle/upper/lower — Bollinger Bands (20-day, 2 std)
  bb_pct_b         — %B: 0 = at lower band, 1 = at upper band, >1 = outside
  bb_bandwidth     — Band width / middle (volatility expansion indicator)
  volume_zscore    — Volume z-score over 20-day window
  obv              — On-Balance Volume (cumulative volume-weighted direction)
  obv_zscore       — OBV z-score over 20-day window

Input / output
──────────────
  Reads:   data/raw/{TICKER}.parquet
  Writes:  data/features/{TICKER}.parquet
           data/features/returns_matrix.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path("data/raw")
FEATURE_DIR = Path("data/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

from data_pipeline import TICKER_LIST, ASSET_CLASS


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute foundational features that other indicators depend on.

    Must be called first in the engineer() pipeline — ATR and log_return
    are prerequisites for ADX, RSI, and MACD calculations.

    Args:
        df: Raw OHLCV DataFrame with columns [Open, High, Low, Close, Volume].

    Returns:
        Copy of df with added columns: log_return, dollar_volume,
        true_range, atr_14.

    Notes:
        True Range = max(High−Low, |High−prev_Close|, |Low−prev_Close|).
        This accounts for overnight gaps that a simple High−Low would miss.
        ATR is the 14-day rolling mean of TR (Wilder's original definition).
    """
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
    """
    Simple price momentum: percentage change over N days.

    Args:
        df: DataFrame with column [Close].

    Returns:
        Copy of df with columns: mom_5, mom_20, mom_60.

    Notes:
        Momentum is measured over 5, 20, and 60 trading days (~1 week,
        1 month, 3 months).  These windows capture short-, medium-, and
        longer-term trend persistence.  Momentum signals in signal_generation.py
        require all three to agree before generating a signal (three-way
        confirmation reduces false positives).
    """
    df = df.copy()
    df["mom_5"]  = df["Close"].pct_change(5)
    df["mom_20"] = df["Close"].pct_change(20)
    df["mom_60"] = df["Close"].pct_change(60)
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Wilder's Relative Strength Index using exponential moving averages.

    Args:
        df:     DataFrame with column [Close].
        window: Look-back period (default 14 — Wilder's original 1978 value).

    Returns:
        Copy of df with column: rsi_{window}.

    Interpretation:
        >70 = overbought (buying pressure has run too far, reversion risk)
        <30 = oversold  (selling pressure has run too far, recovery likely)
        50  = neutral (equal average gains and losses)

    Implementation note:
        Uses EWM with com=window-1 (equivalent to Wilder's smoothing factor
        α = 1/window).  The standard ``rolling().mean()`` would give a
        simple moving average RSI — technically different from Wilder's
        original and widely used by institutional platforms.
    """
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
    """
    Moving Average Convergence / Divergence indicator.

    Args:
        df:     DataFrame with column [Close].
        fast:   Fast EMA period (default 12).
        slow:   Slow EMA period (default 26).
        signal: Signal line smoothing period (default 9).

    Returns:
        Copy of df with columns: macd_line, macd_signal, macd_hist.

    Interpretation:
        macd_hist > 0  = fast EMA above slow EMA = bullish momentum
        macd_hist < 0  = fast EMA below slow EMA = bearish momentum
        macd_hist crossing zero = momentum reversal (used as confirmation
        in signal_generation.momentum_rule())
    """
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["macd_line"]   = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]  # crossing zero = momentum turning
    return df


def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average Directional Index — measures TREND STRENGTH, not direction.

    Args:
        df:     DataFrame with columns [High, Low, Close] and true_range
                (added by add_base_features()).
        window: Smoothing period (default 14).

    Returns:
        Copy of df with columns: adx, plus_di, minus_di.

    Interpretation:
        ADX < 20   = ranging / choppy market — trend signals are unreliable
        ADX 20–25  = developing trend
        ADX > 25   = confirmed trend — momentum signals are meaningful
        ADX > 40   = strong trend (less common, often precedes exhaustion)

        +DI > −DI  = upward pressure dominant
        −DI > +DI  = downward pressure dominant

    Notes:
        ADX is used in signal_generation.py as a regime filter — MA crossover
        signals are only acted on when ADX confirms a real trend exists.
        Without this filter, MA crossovers in choppy markets fire frequently
        and almost always result in whipsaws (costly round-trip trades).
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
    """
    Rolling price z-score: how many standard deviations from the rolling mean.

    Args:
        df: DataFrame with column [Close].

    Returns:
        Copy of df with columns: zscore_20, zscore_60.

    Use in strategy:
        Used by mean_reversion_rule() in signal_generation.py.
        A very negative z-score (e.g., −2.0) means price is far below its
        recent average — a candidate for mean reversion long entry.
        A very positive z-score (e.g., +2.0) flags potential overextension.
    """
    df = df.copy()
    for window in [20, 60]:
        roll = df["Close"].rolling(window)
        df[f"zscore_{window}"] = (df["Close"] - roll.mean()) / roll.std()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands: a volatility envelope around a simple moving average.

    Args:
        df:      DataFrame with column [Close].
        window:  Rolling window for the middle band (default 20).
        num_std: Number of standard deviations for the upper/lower bands (default 2).

    Returns:
        Copy of df with columns: bb_middle, bb_upper, bb_lower,
        bb_pct_b, bb_bandwidth.

    Interpretation:
        bb_pct_b = 0   → price at the lower band (oversold / mean-reversion long)
        bb_pct_b = 0.5 → price at the middle band (neutral)
        bb_pct_b = 1   → price at the upper band (overbought / mean-reversion short)
        bb_pct_b > 1   → price outside the upper band (strong momentum breakout)

        bb_bandwidth rising → volatility expanding (often precedes a big move)
        bb_bandwidth falling → volatility contracting (squeeze, then breakout)
    """
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
    """
    Volume-based features for flow confirmation.

    Args:
        df:     DataFrame with columns [Close, Volume].
        window: Rolling window for z-score computation (default 20).

    Returns:
        Copy of df with columns: volume_zscore, obv, obv_zscore.

    Notes:
        volume_zscore: How unusual is today's volume vs the recent average?
            High volume on an up day = institutional buying (bullish).
            High volume on a down day = institutional selling (bearish).

        OBV (On-Balance Volume): Adds volume on up days, subtracts on down days.
            Rising OBV while price is flat = accumulation (smart money buying).
            Falling OBV while price is flat = distribution (smart money selling).
            OBV divergence from price is a leading indicator of reversals.

        obv_zscore: Normalised OBV used as a flow-strength signal in
            compute_scores() in signal_generation.py.
    """
    df       = df.copy()
    vol_roll = df["Volume"].rolling(window)

    df["volume_zscore"] = (df["Volume"] - vol_roll.mean()) / vol_roll.std()
    df["obv"]           = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()

    obv_roll = df["obv"].rolling(window)
    df["obv_zscore"] = (df["obv"] - obv_roll.mean()) / obv_roll.std()
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature transformations in the correct dependency order.

    Args:
        df: Raw OHLCV DataFrame from data_pipeline.py.

    Returns:
        Feature-enriched DataFrame with all columns from each add_*
        function.  Rows with any NaN are dropped — this removes the
        warm-up period at the start of history where rolling windows
        do not yet have enough data.

    Order matters:
        add_base_features() MUST run first — it produces true_range and
        log_return which are required by add_adx() and add_rsi().
    """
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