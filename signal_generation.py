"""
signal_generation.py

Asset-class-aware signal generation.
Different assets need different strategies:
  equity  — MA+ADX regime switch between momentum and mean reversion
  bond    — momentum only (TLT trends with rate direction for months/years)
  commodity — momentum in fear regimes, mean reversion in calm regimes
  sector  — same as equity but slower MA windows (sector trends are slower)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from data_pipeline import TICKER_LIST, ASSET_CLASS

FEATURE_DIR = Path("data/features")
SIGNAL_DIR  = Path("data/signals")
MACRO_DIR   = Path("data/macro")
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)


def load_macro() -> pd.DataFrame:
    path = MACRO_DIR / "macro_features.parquet"
    if not path.exists():
        print("  WARNING: macro_features.parquet not found — run macro_features.py first")
        return pd.DataFrame()
    return pd.read_parquet(path)

# -----------------------------------------------------------------------------
# Regime filters — one per asset class
# -----------------------------------------------------------------------------

def equity_regime(df: pd.DataFrame) -> pd.Series:
    """
    MA50/200 + ADX confirmation for equities.
    trending = MA50 > MA200 AND ADX > 25 (both must agree).
    ADX prevents false trending signals in choppy uptrends.
    """
    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()
    return ((ma50 > ma200) & (df["adx"] > 25)).astype(int)


def sector_regime(df: pd.DataFrame) -> pd.Series:
    """
    Slower MA windows for sectors — sector rotations take longer than individual stocks.
    MA100/300 reduces whipsawing on sector ETFs that move more slowly.
    """
    ma100 = df["Close"].rolling(100).mean()
    ma300 = df["Close"].rolling(300).mean()
    return ((ma100 > ma300) & (df["adx"] > 20)).astype(int)


def bond_regime(df: pd.DataFrame) -> pd.Series:
    """
    Bonds (TLT) are momentum-only — they trend with rate direction for months or years.
    Mean reversion on bonds is dangerous: a rate-hike cycle can push TLT down 40%.
    regime=1 always here means the momentum signal always fires, never reverts.
    The momentum signal itself will be negative when TLT is falling.
    """
    # bonds always get momentum treatment — never mean reversion
    return pd.Series(1, index=df.index)


def commodity_regime(df: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
    """
    Gold (GLD) trends during fear, mean-reverts during calm.
    Fear regime (VIX > 25): momentum — gold can trend for months during crises.
    Calm regime (VIX < 20): mean reversion — gold oscillates around fair value.
    """
    if macro.empty:
        return equity_regime(df)  # fall back to equity regime if no macro

    vix = macro["vix"].reindex(df.index, method="ffill").fillna(20)
    # 1 = use momentum (fear), 0 = use mean reversion (calm)
    return (vix > 25).astype(int)

# -----------------------------------------------------------------------------
# Signal rules
# -----------------------------------------------------------------------------

def momentum_rule(df: pd.DataFrame) -> pd.Series:
    # three-way confirmation: mom_20, mom_60, and MACD must all agree
    long  = (df["mom_20"] > 0) & (df["mom_60"] > 0) & (df["macd_hist"] > 0)
    short = (df["mom_20"] < 0) & (df["mom_60"] < 0) & (df["macd_hist"] < 0)
    return pd.Series(np.select([long, short], [1, -1], default=0), index=df.index)


def mean_reversion_rule(df: pd.DataFrame) -> pd.Series:
    # two independent signals must agree: zscore and RSI both extreme in same direction
    long  = (df["zscore_20"] < -1.5) & (df["rsi_14"] < 35)
    short = (df["zscore_20"] >  1.5) & (df["rsi_14"] > 65)
    return pd.Series(np.select([long, short], [1, -1], default=0), index=df.index)


def volume_filter(df: pd.DataFrame, threshold: float = 1.0) -> pd.Series:
    # elevated volume = market conviction behind the move
    return (df["volume_zscore"] > threshold).astype(int).rename("volume_filter")

# -----------------------------------------------------------------------------
# VIX gate — sits flat on extreme fear spike days
# -----------------------------------------------------------------------------

def vix_gate(macro: pd.DataFrame, index: pd.Index) -> pd.Series:
    """
    Hard gate: 0 on days when VIX z-score > 2.5 (extreme fear spike, ~top 2% of days).
    On these days every technical signal breaks down — fear, not fundamentals, drives prices.
    Sitting flat on these days costs almost nothing in normal returns but cuts crash losses.
    """
    if macro.empty:
        return pd.Series(1, index=index)
    vix_z = macro["vix_zscore"].reindex(index, method="ffill").fillna(0)
    return (vix_z < 2.5).astype(int)

# -----------------------------------------------------------------------------
# Score computation
# -----------------------------------------------------------------------------

def compute_scores(df: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """
    Normalised composite score. Regime-aware weighting:
      regime=1 → 80% momentum weight
      regime=0 → 80% mean reversion weight
    No macro dampening here — applied once in portfolio.py.
    """
    scores = pd.DataFrame(index=df.index)

    scores["s_mom20"] = _roll_zscore(df["mom_20"],    60)
    scores["s_mom60"] = _roll_zscore(df["mom_60"],    60)
    scores["s_macd"]  = _roll_zscore(df["macd_hist"], 60)
    scores["s_adx"]   = _roll_zscore(df["adx"],       60)

    scores["s_zscore"] = -_roll_zscore(df["zscore_20"], 60)
    scores["s_rsi"]    = -_roll_zscore(df["rsi_14"],    60)
    scores["s_bb"]     = -_roll_zscore(df["bb_pct_b"],  60)

    scores["s_obv"] = _roll_zscore(df["obv_zscore"], 60)

    scores["score_momentum"] = scores[["s_mom20", "s_mom60", "s_macd"]].mean(axis=1)
    scores["score_mean_rev"] = scores[["s_zscore", "s_rsi", "s_bb"]].mean(axis=1)

    mom_weight = regime * 0.6 + 0.2   # 0.8 trending, 0.2 ranging
    rev_weight = 1 - mom_weight

    adx_boost = (1 + 0.15 * scores["s_adx"].clip(-2, 2)) * regime

    scores["score_composite"] = (
        (mom_weight * scores["score_momentum"] * (1 + adx_boost * 0.1)) +
        (rev_weight * scores["score_mean_rev"])
    ) * (1 + 0.15 * scores["s_obv"])

    return scores


def _roll_zscore(series: pd.Series, window: int) -> pd.Series:
    roll = series.rolling(window, min_periods=window // 2)
    return (series - roll.mean()) / roll.std()


def scores_to_signal(score: pd.Series, long_thresh: float = 0.5,
                     short_thresh: float = -0.5) -> pd.Series:
    signal = pd.Series(0, index=score.index)
    signal[score >  long_thresh]  =  1
    signal[score < short_thresh]  = -1
    return signal

# -----------------------------------------------------------------------------
# Master generate — asset-class-aware routing
# -----------------------------------------------------------------------------

def generate(df: pd.DataFrame, ticker: str, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Route each ticker to its appropriate regime and signal logic.
    Same features computed for all assets — only the strategy applied differs.
    """
    asset_class = ASSET_CLASS[ticker]
    out  = df[["Close", "log_return"]].copy()
    gate = vix_gate(macro, df.index)

    # choose regime based on asset class
    if asset_class == "equity":
        regime = equity_regime(df)
    elif asset_class == "sector":
        regime = sector_regime(df)
    elif asset_class == "bond":
        regime = bond_regime(df)
    elif asset_class == "commodity":
        regime = commodity_regime(df, macro)
    else:
        regime = equity_regime(df)  # default

    out["regime"]       = regime
    out["asset_class"]  = asset_class
    out["volume_filter"] = volume_filter(df)

    mom = momentum_rule(df)
    rev = mean_reversion_rule(df)

    # regime-switched signal — bonds always get momentum, others switch
    out["signal_regime"] = (
        pd.Series(np.where(regime == 1, mom, rev), index=df.index) * gate
    )

    # score-based composite
    scores = compute_scores(df, regime)
    out    = pd.concat([out, scores], axis=1)
    out["signal_composite"] = scores_to_signal(scores["score_composite"]) * gate

    return out.dropna()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Generating signals...\n")
    macro       = load_macro()
    all_signals = {}

    for ticker in TICKER_LIST:
        df  = pd.read_parquet(FEATURE_DIR / f"{ticker}.parquet")
        sig = generate(df, ticker, macro)

        out = SIGNAL_DIR / f"{ticker}.parquet"
        sig.to_parquet(out, engine="pyarrow", compression="snappy")

        n = len(sig)
        print(f"  {ticker} ({ASSET_CLASS[ticker]})  {n} days  |  trending {sig['regime'].mean()*100:.0f}%")
        for name in ["signal_regime", "signal_composite"]:
            l = (sig[name] ==  1).sum()
            s = (sig[name] == -1).sum()
            print(f"    {name:<22}: {l:>4} long  {s:>4} short  ({(l+s)/n*100:.1f}% active)")
        print()

        all_signals[ticker] = sig

    regime    = pd.DataFrame({t: s["signal_regime"]    for t, s in all_signals.items()}).dropna()
    composite = pd.DataFrame({t: s["signal_composite"] for t, s in all_signals.items()}).dropna()

    regime.to_parquet(SIGNAL_DIR    / "regime_signals.parquet")
    composite.to_parquet(SIGNAL_DIR / "composite_signals.parquet")
    print(f"Signal matrices saved: {regime.shape}")


if __name__ == "__main__":
    main()