"""
Takes feature-engineered data and produces trade signals.
Output is a DataFrame of scores and discrete signals (+1 long, -1 short, 0 flat)
for each ticker on each day.

Two approaches built here:
  1. Rule-based  — explicit thresholds, easy to interpret and debug
  2. Score-based — continuous composite score, more robust in live trading

v3: macro regime layer added — VIX and yield curve modulate signal confidence
    and threshold tightness before any trade decision is made.
"""

import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_DIR = Path("data/features")
SIGNAL_DIR  = Path("data/signals")
MACRO_DIR   = Path("data/macro")
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

# -----------------------------------------------------------------------------
# Macro regime loader
# -----------------------------------------------------------------------------

def load_macro() -> pd.DataFrame:
    """
    Load pre-computed macro features from macro_features.py.
    Returns a DataFrame indexed by date with columns like macro_score,
    size_multiplier, vix_fear, curve_inverted, etc.
    If macro data is missing, returns an empty DataFrame and signals proceed
    without macro adjustment — degrades gracefully.
    """
    path = MACRO_DIR / "macro_features.parquet"
    if not path.exists():
        print("  WARNING: macro_features.parquet not found — run macro_features.py first")
        print("  Proceeding without macro adjustment (signals will use neutral weights)")
        return pd.DataFrame()
    return pd.read_parquet(path)

# -----------------------------------------------------------------------------
# Regime filter (price-based)
# -----------------------------------------------------------------------------

def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each day as trending (1) or mean-reverting (0).
    MA50 > MA200 = uptrend = momentum regime.
    MA50 < MA200 = range-bound or falling = reversion regime.
    Prevents mean-reverting into a structural trend (NVDA problem)
    and momentum-chasing a flat index (SPY problem).
    """
    df    = df.copy()
    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()
    df["regime"] = (ma50 > ma200).astype(int)  # 1 = trending, 0 = ranging
    return df

# -----------------------------------------------------------------------------
# Rule-based signals
# -----------------------------------------------------------------------------

# Long when mom_20 and mom_60 both positive and MACD confirms; short when all three negative.
# Requiring three signals to agree filters out short-lived noise.
def momentum_rule(df: pd.DataFrame) -> pd.Series:
    long  = (df["mom_20"] > 0) & (df["mom_60"] > 0) & (df["macd_hist"] > 0)
    short = (df["mom_20"] < 0) & (df["mom_60"] < 0) & (df["macd_hist"] < 0)
    return pd.Series(
        np.select([long, short], [1, -1], default=0),
        index=df.index,
        name="signal_momentum",
    )


# Long when zscore_20 < -1.5 and RSI < 35 (oversold); short when zscore > 1.5 and RSI > 65.
# Two independent signals must agree — reduces false positives.
def mean_reversion_rule(df: pd.DataFrame) -> pd.Series:
    long  = (df["zscore_20"] < -1.5) & (df["rsi_14"] < 35)
    short = (df["zscore_20"] >  1.5) & (df["rsi_14"] > 65)
    return pd.Series(
        np.select([long, short], [1, -1], default=0),
        index=df.index,
        name="signal_mean_rev",
    )


# Volume z-score > threshold = busier than normal. High volume + signal = more reliable.
def volume_filter(df: pd.DataFrame, threshold: float = 1.0) -> pd.Series:
    return (df["volume_zscore"] > threshold).astype(int).rename("volume_filter")


def regime_switched_signal(df: pd.DataFrame) -> pd.Series:
    """
    Apply momentum in trending regimes, mean reversion in ranging ones.
    regime=1 (MA50 > MA200): ride the trend.
    regime=0 (MA50 < MA200): fade the extremes.
    """
    mom = momentum_rule(df)
    rev = mean_reversion_rule(df)
    return pd.Series(
        np.where(df["regime"] == 1, mom, rev),
        index=df.index,
        name="signal_regime",
    )

# -----------------------------------------------------------------------------
# Score-based signals
# -----------------------------------------------------------------------------

# Why normalise? Each indicator has a different scale — RSI 0-100, zscore ~-3 to +3, mom_20 a small decimal.
# Rolling z-score: (value - rolling_mean) / rolling_std puts each on mean~0 std~1 so they contribute equally.
def compute_scores(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw indicators into normalised scores, combine with regime-aware
    weights, then apply macro adjustment.

    The macro layer works as a confidence multiplier:
      - In calm low-VIX environments, scores are amplified slightly
      - In fear/inverted-curve environments, scores are dampened
      - Vol backwardation (acute stress) tightens thresholds automatically
        because a lower score means fewer signals cross the threshold
    """
    df     = df.copy()
    scores = pd.DataFrame(index=df.index)

    # momentum indicators: higher raw value = more bullish = positive score
    scores["s_mom20"] = _roll_zscore(df["mom_20"],    60)
    scores["s_mom60"] = _roll_zscore(df["mom_60"],    60)
    scores["s_macd"]  = _roll_zscore(df["macd_hist"], 60)

    # mean reversion: flipped — high zscore_20 = expensive = bearish, so negate
    scores["s_zscore"] = -_roll_zscore(df["zscore_20"], 60)
    scores["s_rsi"]    = -_roll_zscore(df["rsi_14"],    60)
    scores["s_bb"]     = -_roll_zscore(df["bb_pct_b"],  60)

    # volume: not flipped — high OBV z-score = accumulation = bullish
    scores["s_obv"] = _roll_zscore(df["obv_zscore"], 60)

    scores["score_momentum"] = scores[["s_mom20", "s_mom60", "s_macd"]].mean(axis=1)
    scores["score_mean_rev"] = scores[["s_zscore", "s_rsi", "s_bb"]].mean(axis=1)

    # price-regime weights: trending days lean 80% momentum, ranging days lean 80% reversion
    mom_weight = df["regime"] * 0.6 + 0.2   # 0.8 when trending, 0.2 when ranging
    rev_weight = 1 - mom_weight              # 0.2 when trending, 0.8 when ranging

    raw_composite = (
        mom_weight * scores["score_momentum"] +
        rev_weight * scores["score_mean_rev"]
    ) * (1 + 0.2 * scores["s_obv"])  # volume boosts signal strength by up to 20%

    # --- macro adjustment ---
    # Align macro to price dates — macro has different calendar (includes non-trading days)
    if not macro.empty:
        macro_aligned = macro["macro_score"].reindex(df.index, method="ffill")

        # In fear regimes, dampen the composite score — fewer signals cross the threshold
        # In calm regimes, allow full score through
        # This is NOT adding a new source of alpha — it's reducing risk in bad environments
        vix_fear_aligned = macro["vix_fear"].reindex(df.index, method="ffill").fillna(0)
        curve_inv_aligned = macro["curve_inverted"].reindex(df.index, method="ffill").fillna(0)

        # dampening factor: 1.0 normally, 0.7 when VIX > 30, 0.6 when curve inverted
        # both together = 0.7 * 0.6 = 0.42 — strongly suppressed
        dampen = (1.0
                  - 0.3 * vix_fear_aligned      # fear: reduce score 30%
                  - 0.2 * curve_inv_aligned)     # inverted curve: reduce score 20%

        scores["macro_score"]    = macro_aligned.values
        scores["score_composite"] = raw_composite * dampen

    else:
        # no macro data — use raw composite unchanged
        scores["macro_score"]     = 0.0
        scores["score_composite"] = raw_composite

    return scores


def _roll_zscore(series: pd.Series, window: int) -> pd.Series:
    """Normalise with rolling z-score: (value - rolling_mean) / rolling_std."""
    roll = series.rolling(window, min_periods=window // 2)
    return (series - roll.mean()) / roll.std()


# Tighter thresholds = fewer, higher-conviction trades. Tune these after seeing % active in output.
def scores_to_signal(score: pd.Series, long_thresh: float = 0.5, short_thresh: float = -0.5) -> pd.Series:
    signal = pd.Series(0, index=score.index)
    signal[score >  long_thresh]  =  1
    signal[score < short_thresh]  = -1
    return signal

# -----------------------------------------------------------------------------
# Master generate function
# -----------------------------------------------------------------------------

def generate(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    df  = add_regime(df)
    out = df[["Close", "log_return", "regime"]].copy()

    out["signal_momentum"] = momentum_rule(df)
    out["signal_mean_rev"] = mean_reversion_rule(df)
    out["signal_regime"]   = regime_switched_signal(df)
    out["volume_filter"]   = volume_filter(df)

    scores = compute_scores(df, macro)
    out    = pd.concat([out, scores], axis=1)
    out["signal_composite"] = scores_to_signal(scores["score_composite"])

    return out.dropna()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Generating signals...\n")

    macro       = load_macro()
    all_signals = {}

    for ticker in TICKERS:
        df  = pd.read_parquet(FEATURE_DIR / f"{ticker}.parquet")
        sig = generate(df, macro)

        out = SIGNAL_DIR / f"{ticker}.parquet"
        sig.to_parquet(out, engine="pyarrow", compression="snappy")

        n = len(sig)
        print(f"  {ticker} ({n} days)  |  trending {sig['regime'].mean()*100:.0f}% of time")
        for name in ["signal_momentum", "signal_mean_rev", "signal_regime", "signal_composite"]:
            long  = (sig[name] ==  1).sum()
            short = (sig[name] == -1).sum()
            print(f"    {name:<22}: {long:>4} long  {short:>4} short  ({(long+short)/n*100:.1f}% active)")
        print()

        all_signals[ticker] = sig

    comp   = pd.DataFrame({t: s["signal_composite"] for t, s in all_signals.items()}).dropna()
    regime = pd.DataFrame({t: s["signal_regime"]    for t, s in all_signals.items()}).dropna()

    comp.to_parquet(SIGNAL_DIR   / "composite_signals.parquet")
    regime.to_parquet(SIGNAL_DIR / "regime_signals.parquet")

    print(f"Signal matrices saved: {comp.shape}")
    print("\nSample — AAPL last 5 rows:")
    cols = ["Close", "regime", "signal_regime", "macro_score", "score_composite", "signal_composite"]
    print(all_signals["AAPL"][cols].tail(5).round(3))


if __name__ == "__main__":
    main()