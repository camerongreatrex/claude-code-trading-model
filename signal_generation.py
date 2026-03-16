"""
signal_generation.py

Asset-class and stock-aware signal generation.

Routing logic:
  equity_index  — MA50/200 + ADX regime switch
  sector_etf    — MA100/300 + ADX (slower, sector rotations take longer)
  stock         — MA50/200 + ADX + volatility filter
                  high-vol stocks (XOM, AMZN) use wider zscore thresholds
  bond          — momentum only, never mean reversion
  commodity     — momentum in fear regime, mean reversion in calm

Individual stocks also get a liquidity gate: if dollar_volume is below
the 20-day average, signals are suppressed — thin days produce noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from data_pipeline import TICKER_LIST, ASSET_CLASS, EQUITY_LIKE

FEATURE_DIR = Path("data/features")
SIGNAL_DIR  = Path("data/signals")
MACRO_DIR   = Path("data/macro")
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

# stocks with structurally higher volatility — widen mean reversion thresholds
# so we don't fade every normal daily swing as if it were an extreme
HIGH_VOL_STOCKS = {"XOM", "AMZN", "GS"}

# defensive stocks tend to mean-revert strongly — tighten thresholds for faster signals
DEFENSIVE_STOCKS = {"JNJ", "COST", "NEE", "BRK-B"}


def load_macro() -> pd.DataFrame:
    path = MACRO_DIR / "macro_features.parquet"
    if not path.exists():
        print("  WARNING: macro_features.parquet not found — run macro_features.py first")
        return pd.DataFrame()
    return pd.read_parquet(path)

# -----------------------------------------------------------------------------
# Regime filters
# -----------------------------------------------------------------------------

def equity_index_regime(df: pd.DataFrame) -> pd.Series:
    """MA50/200 + ADX > 25. Standard regime filter for broad indices."""
    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()
    return ((ma50 > ma200) & (df["adx"] > 25)).astype(int)


def sector_regime(df: pd.DataFrame) -> pd.Series:
    """
    Slower MA100/300 for sector ETFs — sector rotations develop over months not weeks.
    Tighter ADX threshold (20 vs 25) because sectors trend less strongly than indices.
    """
    ma100 = df["Close"].rolling(100).mean()
    ma300 = df["Close"].rolling(300).mean()
    return ((ma100 > ma300) & (df["adx"] > 20)).astype(int)


def stock_regime(df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Individual stocks: MA50/200 + ADX, same as equity index.
    Defensive stocks (JNJ, COST) spend more time in mean-reversion mode —
    their businesses are stable so prices revert to fair value quickly.
    High-vol stocks (XOM, AMZN) need stronger ADX confirmation before we
    call it a trend, because they have more noise in their price action.
    """
    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()

    # high-vol stocks require stronger trend confirmation before going momentum
    adx_threshold = 30 if ticker in HIGH_VOL_STOCKS else 25

    return ((ma50 > ma200) & (df["adx"] > adx_threshold)).astype(int)


def bond_regime(df: pd.DataFrame) -> pd.Series:
    """Bonds always get momentum treatment. Mean reversion on TLT is dangerous."""
    return pd.Series(1, index=df.index)


def commodity_regime(df: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
    """
    Gold: momentum in fear (VIX > 25), mean reversion in calm.
    Fear = gold trends for months as a safe haven.
    Calm = gold oscillates around real-rate fair value.
    """
    if macro.empty:
        return equity_index_regime(df)
    vix = macro["vix"].reindex(df.index, method="ffill").fillna(20)
    return (vix > 25).astype(int)  # 1 = fear = use momentum

# -----------------------------------------------------------------------------
# Liquidity gate for individual stocks
# -----------------------------------------------------------------------------

def liquidity_gate(df: pd.DataFrame) -> pd.Series:
    """
    Suppress signals on days when dollar volume is below its 20-day average.
    Thin trading days produce unreliable signals — spreads widen and price
    moves are more likely to reverse. Only matters for individual stocks.
    """
    # shift(1): today's volume must exceed yesterday's 20-day average — no lookahead
    avg_dv = df["dollar_volume"].rolling(20).mean().shift(1)
    return (df["dollar_volume"] >= avg_dv).astype(int)

# -----------------------------------------------------------------------------
# VIX spike gate
# -----------------------------------------------------------------------------

def vix_gate(macro: pd.DataFrame, index: pd.Index) -> pd.Series:
    """
    Hard gate: 0 on extreme VIX spike days (z-score > 2.5, ~top 2% of days).
    Fear drives prices on these days — technical signals break down.
    """
    if macro.empty:
        return pd.Series(1, index=index)
    vix_z = macro["vix_zscore"].reindex(index, method="ffill").fillna(0)
    return (vix_z < 2.5).astype(int)

# -----------------------------------------------------------------------------
# Signal rules — with ticker-aware thresholds
# -----------------------------------------------------------------------------

def momentum_rule(df: pd.DataFrame) -> pd.Series:
    """Three-way confirmation: mom_20, mom_60, and MACD all agree."""
    long  = (df["mom_20"] > 0) & (df["mom_60"] > 0) & (df["macd_hist"] > 0)
    short = (df["mom_20"] < 0) & (df["mom_60"] < 0) & (df["macd_hist"] < 0)
    return pd.Series(np.select([long, short], [1, -1], default=0), index=df.index)


def mean_reversion_rule(df: pd.DataFrame, ticker: str = "") -> pd.Series:
    """
    Two independent signals agree on extreme: zscore and RSI.

    High-vol stocks (XOM, AMZN, GS): wider thresholds (2.0 / 1.5)
      because normal daily swings are large — we only want extreme dislocations.

    Defensive stocks (JNJ, COST, NEE, BRK-B): tighter thresholds (1.2 / 30)
      because their stable businesses mean even moderate dislocations revert quickly.

    Default: 1.5 zscore / 35 RSI
    """
    if ticker in HIGH_VOL_STOCKS:
        z_thresh, rsi_lo, rsi_hi = 2.0, 30, 70
    elif ticker in DEFENSIVE_STOCKS:
        z_thresh, rsi_lo, rsi_hi = 1.2, 38, 62
    else:
        z_thresh, rsi_lo, rsi_hi = 1.5, 35, 65

    long  = (df["zscore_20"] < -z_thresh) & (df["rsi_14"] < rsi_lo)
    short = (df["zscore_20"] >  z_thresh) & (df["rsi_14"] > rsi_hi)
    return pd.Series(np.select([long, short], [1, -1], default=0), index=df.index)


def volume_filter(df: pd.DataFrame, threshold: float = 1.0) -> pd.Series:
    return (df["volume_zscore"] > threshold).astype(int).rename("volume_filter")

# -----------------------------------------------------------------------------
# Score computation
# -----------------------------------------------------------------------------

def compute_scores(df: pd.DataFrame, regime: pd.Series, ticker: str = "") -> pd.DataFrame:
    """
    Normalised composite score with regime-aware weighting.
    No macro dampening — applied once in portfolio.py.
    """
    scores = pd.DataFrame(index=df.index)

    scores["s_mom20"] = _roll_zscore(df["mom_20"],    60)
    scores["s_mom60"] = _roll_zscore(df["mom_60"],    60)
    scores["s_macd"]  = _roll_zscore(df["macd_hist"], 60)
    scores["s_adx"]   = _roll_zscore(df["adx"],       60)

    scores["s_zscore"] = -_roll_zscore(df["zscore_20"], 60)
    scores["s_rsi"]    = -_roll_zscore(df["rsi_14"],    60)
    scores["s_bb"]     = -_roll_zscore(df["bb_pct_b"],  60)
    scores["s_obv"]    =  _roll_zscore(df["obv_zscore"], 60)

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
# Master generate — routes each ticker to the right logic
# -----------------------------------------------------------------------------

def generate(df: pd.DataFrame, ticker: str, macro: pd.DataFrame) -> pd.DataFrame:
    asset_class = ASSET_CLASS[ticker]
    out  = df[["Close", "log_return"]].copy()
    gate = vix_gate(macro, df.index)

    # asset-class-aware regime selection
    if asset_class == "equity_index":
        regime = equity_index_regime(df)
    elif asset_class == "sector_etf":
        regime = sector_regime(df)
    elif asset_class == "stock":
        regime = stock_regime(df, ticker)
    elif asset_class == "bond":
        regime = bond_regime(df)
    elif asset_class == "commodity":
        regime = commodity_regime(df, macro)
    else:
        regime = equity_index_regime(df)

    out["regime"]      = regime
    out["asset_class"] = asset_class

    mom = momentum_rule(df)
    rev = mean_reversion_rule(df, ticker)  # ticker-aware thresholds

    # Two separate gates:
    # - regime_gate  : VIX only. MA crossover is a multi-week position signal;
    #   filtering on daily volume would block ~60% of days and starve it of returns.
    # - composite_gate: VIX + liquidity. Composite scores are day-level timing signals
    #   where below-average volume is genuinely a warning of unreliable price action.
    if asset_class == "stock":
        liq_gate       = liquidity_gate(df)
        regime_gate    = gate               # regime  : VIX gate only
        composite_gate = gate * liq_gate    # composite: both gates
    else:
        regime_gate    = gate
        composite_gate = gate

    out["volume_filter"] = volume_filter(df)

    # ── Revised signal routing ──────────────────────────────────────────────
    if asset_class in {"equity_index", "sector_etf", "stock", "commodity"}:
        # Long-only: equities and gold have structural upward drift over the long run.
        # Equity risk premium + central bank gold buying make both assets net-long.
        # MA golden cross (MA50 > MA200) avoids noisy short-term momentum flips.
        # No shorts — being short these in secular uptrends consistently loses and
        # ignores structural demand. In flat/down regimes: be in cash, not short.
        ma50  = df["Close"].rolling(50).mean()
        ma200 = df["Close"].rolling(200).mean()
        signal_r = (ma50 > ma200).astype(int) * regime_gate

    else:
        # Bonds: two-sided momentum. Rate cycles genuinely go both ways.
        signal_r = pd.Series(np.where(regime == 1, mom, rev), index=df.index) * regime_gate

    out["signal_regime"] = signal_r

    scores = compute_scores(df, regime, ticker)
    out    = pd.concat([out, scores], axis=1)

    raw_composite = scores_to_signal(scores["score_composite"])
    if asset_class in {"equity_index", "sector_etf", "stock", "commodity"}:
        # Long-only for equity and commodity: convert short signals to flat (cash).
        # The mean-reversion score components are already positive when the
        # asset is oversold, so they act as dip-buying signals — compatible
        # with a long-only mandate.
        raw_composite = raw_composite.clip(lower=0)
    out["signal_composite"] = raw_composite * composite_gate

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

        n  = len(sig)
        ac = ASSET_CLASS[ticker]
        print(f"  {ticker:<6} ({ac:<14})  {n} days  |  "
              f"trending {sig['regime'].mean()*100:.0f}%")

        for name in ["signal_regime", "signal_composite"]:
            l = (sig[name] ==  1).sum()
            s = (sig[name] == -1).sum()
            print(f"    {name:<22}: {l:>4} long  {s:>4} short  "
                  f"({(l+s)/n*100:.1f}% active)")
        print()

        all_signals[ticker] = sig

    regime    = pd.DataFrame({t: s["signal_regime"]    for t, s in all_signals.items()}).dropna()
    composite = pd.DataFrame({t: s["signal_composite"] for t, s in all_signals.items()}).dropna()

    regime.to_parquet(SIGNAL_DIR    / "regime_signals.parquet")
    composite.to_parquet(SIGNAL_DIR / "composite_signals.parquet")
    print(f"Signal matrices saved: {regime.shape}")


if __name__ == "__main__":
    main()