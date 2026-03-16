"""

Takes feature-engineered data and produces trade signals.
Output is a DataFrame of scores and discrete signals (+1 long, -1 short, 0 flat)
for each ticker on each day.

v4 changes:
  - Macro dampening removed from here — it now lives only in portfolio.py
    (was double-dampening before, killing good signals)
  - ADX added as secondary regime confirmation — faster than MA50/200 alone
  - VIX spike gate added — sits out extreme fear days entirely
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
# Macro loader
# -----------------------------------------------------------------------------

def load_macro() -> pd.DataFrame:
    path = MACRO_DIR / "macro_features.parquet"
    if not path.exists():
        print("  WARNING: macro_features.parquet not found — run macro_features.py first")
        return pd.DataFrame()
    return pd.read_parquet(path)

# -----------------------------------------------------------------------------
# Regime filter — price based
# -----------------------------------------------------------------------------

def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-layer regime classification: MA crossover + ADX confirmation.

    Layer 1 — MA50/200 crossover: slow, reliable trend direction indicator.
    Layer 2 — ADX confirmation: fast, measures whether a real trend exists.

    trending = MA50 > MA200 AND ADX > 25 (confirmed uptrend)
    ranging  = MA50 < MA200 OR  ADX < 20 (no confirmed trend)

    The AND condition is the fix from v3: MA50 > MA200 alone fires too early
    and keeps you in "momentum mode" during weak, choppy uptrends where
    mean reversion would actually work better. ADX filters those out.
    """
    df    = df.copy()
    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()

    ma_trend  = (ma50 > ma200)          # slow trend direction
    adx_trend = (df["adx"] > 25)        # fast trend strength confirmation

    # regime=1 only when BOTH agree — reduces false momentum signals in choppy markets
    df["regime"] = (ma_trend & adx_trend).astype(int)
    return df

# -----------------------------------------------------------------------------
# VIX spike gate
# -----------------------------------------------------------------------------

def vix_gate(macro: pd.DataFrame, index: pd.Index) -> pd.Series:
    """
    Returns 0 on days when VIX z-score > 2.5 (extreme fear spike).
    Returns 1 on all other days.

    On extreme VIX spike days, all signals are suppressed — the model sits flat.
    This costs roughly 2% of trading days but eliminates the worst crash-day losses
    where every signal breaks down because fear, not fundamentals, drives prices.

    This is NOT a macro dampening of scores — it's a hard gate. Signal is 0.
    Applied as a multiplier: signal × gate = 0 on spike days, signal × 1 otherwise.
    """
    if macro.empty:
        return pd.Series(1, index=index)  # no macro data — gate always open

    vix_z = macro["vix_zscore"].reindex(index, method="ffill").fillna(0)
    gate  = (vix_z < 2.5).astype(int)  # 0 on extreme spike days, 1 otherwise
    return gate

# -----------------------------------------------------------------------------
# Rule-based signals
# -----------------------------------------------------------------------------

def momentum_rule(df: pd.DataFrame) -> pd.Series:
    # long when mom_20, mom_60, and MACD all agree upward — three-way confirmation
    long  = (df["mom_20"] > 0) & (df["mom_60"] > 0) & (df["macd_hist"] > 0)
    short = (df["mom_20"] < 0) & (df["mom_60"] < 0) & (df["macd_hist"] < 0)
    return pd.Series(np.select([long, short], [1, -1], default=0),
                     index=df.index, name="signal_momentum")


def mean_reversion_rule(df: pd.DataFrame) -> pd.Series:
    # long when zscore and RSI both say oversold — two independent signals agreeing
    long  = (df["zscore_20"] < -1.5) & (df["rsi_14"] < 35)
    short = (df["zscore_20"] >  1.5) & (df["rsi_14"] > 65)
    return pd.Series(np.select([long, short], [1, -1], default=0),
                     index=df.index, name="signal_mean_rev")


def volume_filter(df: pd.DataFrame, threshold: float = 1.0) -> pd.Series:
    # elevated volume = market conviction; used to confirm, not trade standalone
    return (df["volume_zscore"] > threshold).astype(int).rename("volume_filter")


def regime_switched_signal(df: pd.DataFrame) -> pd.Series:
    """
    Route to momentum when ADX-confirmed trend exists, reversion otherwise.
    The stricter regime definition (MA + ADX) reduces time spent in momentum
    mode during choppy markets, which was the main source of bad trades.
    """
    mom = momentum_rule(df)
    rev = mean_reversion_rule(df)
    return pd.Series(
        np.where(df["regime"] == 1, mom, rev),
        index=df.index,
        name="signal_regime",
    )

# -----------------------------------------------------------------------------
# Score-based signals — no macro dampening here (moved to portfolio.py)
# -----------------------------------------------------------------------------

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalised composite score from all price-based indicators.
    Macro adjustment intentionally removed — applied once in portfolio.py only.
    """
    scores = pd.DataFrame(index=df.index)

    # momentum: higher = more bullish
    scores["s_mom20"] = _roll_zscore(df["mom_20"],    60)
    scores["s_mom60"] = _roll_zscore(df["mom_60"],    60)
    scores["s_macd"]  = _roll_zscore(df["macd_hist"], 60)

    # mean reversion: flipped — high zscore = expensive = bearish
    scores["s_zscore"] = -_roll_zscore(df["zscore_20"], 60)
    scores["s_rsi"]    = -_roll_zscore(df["rsi_14"],    60)
    scores["s_bb"]     = -_roll_zscore(df["bb_pct_b"],  60)

    # ADX score: high ADX = strong trend = confidence in momentum signal
    # not flipped — high ADX is good for momentum, neutral for reversion
    scores["s_adx"] = _roll_zscore(df["adx"], 60)

    # volume: not flipped — high OBV = accumulation = bullish
    scores["s_obv"] = _roll_zscore(df["obv_zscore"], 60)

    scores["score_momentum"] = scores[["s_mom20", "s_mom60", "s_macd"]].mean(axis=1)
    scores["score_mean_rev"] = scores[["s_zscore", "s_rsi", "s_bb"]].mean(axis=1)

    # regime-aware weights with ADX modulating momentum confidence
    # trending (regime=1): 80% momentum, boosted further when ADX is strong
    # ranging  (regime=0): 80% mean reversion
    mom_weight = df["regime"] * 0.6 + 0.2
    rev_weight = 1 - mom_weight

    # ADX boost: when ADX is high, momentum score gets extra weight
    # clip to avoid extreme values on outlier ADX days
    adx_boost = (1 + 0.15 * scores["s_adx"].clip(-2, 2)) * df["regime"]

    scores["score_composite"] = (
        (mom_weight * scores["score_momentum"] * (1 + adx_boost * 0.1)) +
        (rev_weight * scores["score_mean_rev"])
    ) * (1 + 0.15 * scores["s_obv"])

    return scores


def _roll_zscore(series: pd.Series, window: int) -> pd.Series:
    """Normalise with rolling z-score: (value - rolling_mean) / rolling_std."""
    roll = series.rolling(window, min_periods=window // 2)
    return (series - roll.mean()) / roll.std()


def scores_to_signal(score: pd.Series, long_thresh: float = 0.5,
                     short_thresh: float = -0.5) -> pd.Series:
    signal = pd.Series(0, index=score.index)
    signal[score >  long_thresh]  =  1
    signal[score < short_thresh]  = -1
    return signal

# -----------------------------------------------------------------------------
# Master generate
# -----------------------------------------------------------------------------

def generate(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    df  = add_regime(df)
    out = df[["Close", "log_return", "regime"]].copy()

    gate = vix_gate(macro, df.index)  # 0 on extreme VIX days, 1 otherwise

    out["signal_momentum"] = momentum_rule(df)   * gate
    out["signal_mean_rev"] = mean_reversion_rule(df) * gate
    out["signal_regime"]   = regime_switched_signal(df) * gate
    out["volume_filter"]   = volume_filter(df)

    scores = compute_scores(df)
    out    = pd.concat([out, scores], axis=1)

    # apply gate to composite too — no trades on extreme fear days
    out["signal_composite"] = scores_to_signal(scores["score_composite"]) * gate

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
        gated = (sig["signal_regime"] == 0).sum()
        print(f"  {ticker} ({n} days)  |  trending {sig['regime'].mean()*100:.0f}%  |  gated {gated} days")
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
    cols = ["Close", "regime", "signal_regime", "score_composite", "signal_composite"]
    print(all_signals["AAPL"][cols].tail(5).round(3))


if __name__ == "__main__":
    main()