"""
Takes feature-engineered data and produces trade signals.
Output is a DataFrame of scores and discrete signals (+1 long, -1 short, 0 flat)
for each ticker on each day.

Two approaches built here:
  1. Rule-based  — explicit thresholds, easy to interpret and debug
  2. Score-based — continuous composite score, more robust in live trading
"""

import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_DIR = Path("data/features")
SIGNAL_DIR = Path("data/signals")
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

# Rule-based signals — each function returns a Series of +1, -1, or 0 for every row.
# Long when mom_20 and mom_60 both positive and MACD confirms; short when both negative. Agreement filters noise.
def momentum_rule(df: pd.DataFrame) -> pd.Series:
    """Return +1 long, -1 short, or 0 when momentum and MACD agree."""
    long = (df["mom_20"] > 0) & (df["mom_60"] > 0) & (df["macd_hist"] > 0)
    short = (df["mom_20"] < 0) & (df["mom_60"] < 0) & (df["macd_hist"] < 0)

    # np.select: checks conditions in order, returns the matching value, defaults to 0.
    return pd.Series(
        np.select([long, short], [1, -1], default=0),
        index=df.index,
        name="signal_momentum",
    )


# Long when zscore_20 < -1.5 and RSI < 35 (oversold); short when zscore > 1.5 and RSI > 65. Two signals reduce false positives.
def mean_reversion_rule(df: pd.DataFrame) -> pd.Series:
    """Return +1 long, -1 short, or 0 when zscore and RSI agree on cheap/expensive."""
    long = (df["zscore_20"] < -1.5) & (df["rsi_14"] < 35)
    short = (df["zscore_20"] > 1.5) & (df["rsi_14"] > 65)

    return pd.Series(
        np.select([long, short], [1, -1], default=0),
        index=df.index,
        name="signal_mean_rev",
    )


# Volume z-score > threshold = busier than normal. High volume + signal = more reliable.
def volume_filter(df: pd.DataFrame, threshold: float = 1.0) -> pd.Series:
    """Return +1 when volume is elevated; use to confirm other signals, not standalone."""
    active = (df["volume_zscore"] > threshold).astype(int)
    return active.rename("volume_filter")


# Score-based signals — continuous scores per indicator, then normalised and combined (closer to real quant models).
# Why normalise? Each indicator has a different scale — RSI 0–100, zscore ~-3 to +3, mom_20 a small decimal.
# We use rolling z-score: (value - rolling_mean) / rolling_std so each has mean~0, std~1 and contributes equally.
def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw indicators into normalised scores, then combine into momentum, mean-reversion, and composite."""
    df = df.copy()
    scores = pd.DataFrame(index=df.index)

    # Momentum: each normalised over 60-day rolling window; higher raw value = more bullish.
    scores["s_mom20"] = _roll_zscore(df["mom_20"], 60)
    scores["s_mom60"] = _roll_zscore(df["mom_60"], 60)
    scores["s_macd"] = _roll_zscore(df["macd_hist"], 60)

    # Mean reversion: flipped so high score = cheap relative to history = bullish.
    scores["s_zscore"] = -_roll_zscore(df["zscore_20"], 60)
    scores["s_rsi"] = -_roll_zscore(df["rsi_14"], 60)
    scores["s_bb"] = -_roll_zscore(df["bb_pct_b"], 60)

    # Volume: not flipped — high OBV z-score = accumulation = bullish.
    scores["s_obv"] = _roll_zscore(df["obv_zscore"], 60)

    # Composite: 0.5/0.5 momentum vs mean-rev; volume adds up to 20% boost. Tune in backtesting.
    mom_cols = ["s_mom20", "s_mom60", "s_macd"]
    scores["score_momentum"] = scores[mom_cols].mean(axis=1)
    rev_cols = ["s_zscore", "s_rsi", "s_bb"]
    scores["score_mean_rev"] = scores[rev_cols].mean(axis=1)
    scores["score_composite"] = (
        0.5 * scores["score_momentum"] + 0.5 * scores["score_mean_rev"]
    ) * (1 + 0.2 * scores["s_obv"])

    return scores


# Result has mean~0, std~1 in each window so indicators are comparable.
def _roll_zscore(series: pd.Series, window: int) -> pd.Series:
    """Normalise series with rolling z-score: (value - rolling_mean) / rolling_std."""
    roll = series.rolling(window, min_periods=window // 2)
    return (series - roll.mean()) / roll.std()

# Convert continuous score to discrete +1 / 0 / -1. Tighter thresholds = fewer, higher-conviction trades.
def scores_to_signal(score: pd.Series, long_thresh: float = 0.5, short_thresh: float = -0.5) -> pd.Series:
    """Convert continuous score to discrete +1 / 0 / -1. Tighter thresholds = fewer, higher-conviction trades."""
    signal = pd.Series(0, index=score.index)
    signal[score > long_thresh] = 1
    signal[score < short_thresh] = -1
    return signal


# Main — generate all signals for every ticker and save.
def generate(df: pd.DataFrame) -> pd.DataFrame:
    """Run all signal logic on a single ticker's feature DataFrame."""
    out = df[["Close", "log_return"]].copy()

    # Rule-based signals.
    out["signal_momentum"] = momentum_rule(df)
    out["signal_mean_rev"] = mean_reversion_rule(df)
    out["volume_filter"] = volume_filter(df)

    # Score-based: compute scores then discrete signal from composite.
    scores = compute_scores(df)
    out = pd.concat([out, scores], axis=1)
    out["signal_composite"] = scores_to_signal(scores["score_composite"])

    return out.dropna()

# Main function — load feature data, generate signals per ticker, save parquet and composite matrix.
def main():
    """Load feature data, generate signals per ticker, save parquet and composite matrix."""
    all_signals = {}
    print("Generating signals...\n")

    for ticker in TICKERS:
        df = pd.read_parquet(FEATURE_DIR / f"{ticker}.parquet")
        sig = generate(df)

        out = SIGNAL_DIR / f"{ticker}.parquet"
        sig.to_parquet(out, engine="pyarrow", compression="snappy")

        # How often is each signal active?
        n = len(sig)
        mom_long  = (sig["signal_momentum"] ==  1).sum()
        mom_short = (sig["signal_momentum"] == -1).sum()
        rev_long  = (sig["signal_mean_rev"] ==  1).sum()
        rev_short = (sig["signal_mean_rev"] == -1).sum()
        comp_long = (sig["signal_composite"] ==  1).sum()
        comp_short = (sig["signal_composite"] == -1).sum()

        print(f"  {ticker} ({n} days)")
        print(f"    momentum  : {mom_long} long  {mom_short} short  ({(mom_long+mom_short)/n*100:.1f}% active)")
        print(f"    mean_rev  : {rev_long} long  {rev_short} short  ({(rev_long+rev_short)/n*100:.1f}% active)")
        print(f"    composite : {comp_long} long  {comp_short} short  ({(comp_long+comp_short)/n*100:.1f}% active)")
        print()

        all_signals[ticker] = sig

    # Composite signal matrix: one column per ticker, used by backtester.
    comp = pd.DataFrame({t: s["signal_composite"] for t, s in all_signals.items()}).dropna()
    comp.to_parquet(SIGNAL_DIR / "composite_signals.parquet")
    print(f"Signal matrix saved: {comp.shape}")

    print("\nSample — AAPL last 5 rows:")
    cols = ["Close", "signal_momentum", "signal_mean_rev", "score_composite", "signal_composite"]
    print(all_signals["AAPL"][cols].tail(5).round(3))


if __name__ == "__main__":
    main()