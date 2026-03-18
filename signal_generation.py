"""
signal_generation.py

Asset-class and stock-aware signal generation.

Routing logic:
  equity_index  — MA50/200 + ADX regime switch
  sector_etf    — MA100/300 + ADX (slower, sector rotations take longer)
  stock         — MA50/200 + ADX + volatility filter
                  thresholds adapt to each asset's trailing realized vol
  bond          — momentum only, never mean reversion
  commodity     — momentum in fear regime, mean reversion in calm

Individual stocks also get a liquidity gate: if dollar_volume is below
the 20-day average, signals are suppressed — thin days produce noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from data_pipeline import TICKER_LIST, ASSET_CLASS

# ── Strategy improvement constants ────────────────────────────────────────────
# These are principled, not curve-fitted to the historical dataset.
RSI_ENTRY_THRESH  = 70    # Wilder's original overbought level (1978). Skip new longs
                           # when RSI > 70 — the golden cross has already run most of
                           # its initial move and entry risk/reward is unfavourable.
MIN_HOLD_DAYS     = 5     # 1 trading week. Hold at least this long before a death
                           # cross can close the position. Prevents paying two round-
                           # trip commissions on the same week's whipsaw.
ATR_TRAILING_MULT = 3.0   # Exit if price drops 3× ATR below trailing high since entry.
                           # 3× is a published institutional standard (gives room to
                           # breathe while protecting against structural deterioration)., EQUITY_LIKE

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


def stock_regime(df: pd.DataFrame) -> pd.Series:
    """
    Individual stocks: MA50/200 + ADX with a data-driven ADX threshold.
    More volatile stocks have noisier price action and require stronger trend
    confirmation — computed from trailing 60-day realized vol, not hardcoded
    per-ticker. Removes ticker-specific overfitting.
    """
    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()

    # Trailing 60-day realized vol (annualized) drives the ADX bar.
    # Higher vol → higher ADX required before calling it a trend.
    trailing_vol = df["log_return"].rolling(60, min_periods=20).std() * np.sqrt(252)
    trailing_vol = trailing_vol.fillna(
        df["log_return"].expanding(min_periods=10).std() * np.sqrt(252)
    ).fillna(0.20)
    adx_threshold = (20 + (trailing_vol * 50).astype(int)).clip(20, 35)

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


def mean_reversion_rule(df: pd.DataFrame) -> pd.Series:
    """
    Two independent signals agree on extreme: zscore and RSI.

    Thresholds are computed dynamically from the asset's own trailing 252-day
    realized volatility — volatile assets get wider bands automatically so we
    only fade genuine dislocations, not routine daily noise. No ticker-specific
    hardcoding; new tickers inherit correct thresholds without any code changes.

      z_thresh = 1.0 + realized_vol  clipped to [1.0, 2.5]
      rsi_lo   = 30 + int(realized_vol * 20)  clipped to [28, 40]
      rsi_hi   = 100 - rsi_lo
    """
    # Trailing 252-day realized vol (annualized) — no lookahead.
    # Fall back to expanding vol for the first ~63 days of history.
    trailing_rvol = df["log_return"].rolling(252, min_periods=63).std() * np.sqrt(252)
    trailing_rvol = trailing_rvol.fillna(
        df["log_return"].expanding(min_periods=10).std() * np.sqrt(252)
    ).fillna(0.20)

    z_thresh = (1.0 + trailing_rvol).clip(1.0, 2.5)
    rsi_lo   = (30 + (trailing_rvol * 20).astype(int)).clip(28, 40)
    rsi_hi   = 100 - rsi_lo

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

    # Structural design weights — NOT optimized per-ticker or per-period.
    # 0.6/0.2 reflects the strategic choice to weight momentum more heavily
    # in trending regimes and mean-reversion in ranging regimes. Changing
    # these would require full framework re-evaluation, not per-asset tuning.
    mom_weight = regime * 0.6 + 0.2   # 0.8 in trend, 0.2 in range
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
# -----------------------------------------------------------------------------
# Signal post-processors — applied after base regime signal is computed
# These are principled risk/quality filters, not parameter-optimised rules.
# -----------------------------------------------------------------------------

def apply_rsi_entry_filter(signal: pd.Series, rsi: pd.Series) -> pd.Series:
    """
    Block new LONG entries when RSI_14 is above RSI_ENTRY_THRESH (70).
    Once a position is open it is NOT closed by this rule — only entry is gated.

    Why this improves quality without overfitting:
      A golden cross that fires at RSI=80 means the fast MA crossed the slow MA
      after a sustained run-up.  The crossover edge comes from catching early
      trend moves, not from chasing extended ones.  Wilder's 70 threshold is a
      published standard from 1978 — it was not chosen by looking at this data.
    """
    result      = signal.copy().astype(float)
    in_position = False

    for i in range(len(result)):
        val = int(result.iloc[i])
        if not in_position:
            if val == 1:
                if float(rsi.iloc[i]) > RSI_ENTRY_THRESH:
                    result.iloc[i] = 0          # block overbought entry
                else:
                    in_position = True
        else:
            if val == 0:
                in_position = False

    return result.astype(int)


def apply_min_hold_filter(signal: pd.Series) -> pd.Series:
    """
    Once long, require MIN_HOLD_DAYS (5) before a death-cross exit fires.
    Prevents same-week whipsaws where a golden/death cross pair triggers within
    days and costs two round-trip commissions for negligible directional move.

    5 trading days = 1 calendar week — the smallest natural holding unit.
    Not tuned to this dataset.
    """
    result      = signal.copy().astype(float)
    in_position = False
    days_held   = 0

    for i in range(len(result)):
        val = int(result.iloc[i])
        if not in_position:
            if val == 1:
                in_position = True
                days_held   = 1
        else:
            days_held += 1
            if val == 0:
                if days_held <= MIN_HOLD_DAYS:
                    result.iloc[i] = 1          # hold open — min hold not met yet
                else:
                    in_position = False
                    days_held   = 0

    return result.astype(int)


def apply_trailing_stop_signal(signal: pd.Series,
                                close: pd.Series,
                                atr: pd.Series) -> pd.Series:
    """
    Override LONG→FLAT if price drops ATR_TRAILING_MULT × ATR below the
    trailing high since position entry.  Falls back to 20% fixed stop if ATR
    is unavailable.

    Why 3× ATR and not a fixed %:
      ATR-scaled stops adapt to each asset's own volatility.  A 3× stop on SPY
      (~$15) is very different from 3× on a volatile individual stock (~$30).
      Fixed-% stops over-fire on volatile names and under-protect on calm ones.
      3× is a standard institutional parameter (documented in elder/schwager).
      It was not chosen by optimising against this backtest.

    Applies only to long-only assets — bonds with short signals need separate logic.
    """
    result      = signal.copy().astype(float)
    in_pos      = False
    trail_high  = 0.0

    for i in range(len(result)):
        price       = float(close.iloc[i])
        current_atr = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else None
        val         = int(result.iloc[i])

        if not in_pos:
            if val == 1:
                in_pos     = True
                trail_high = price
        else:
            if price > trail_high:
                trail_high = price

            stop = (trail_high - ATR_TRAILING_MULT * current_atr
                    if current_atr and current_atr > 0
                    else trail_high * 0.80)

            if price < stop:
                result.iloc[i] = 0              # trailing stop fires
                in_pos         = False
                trail_high     = 0.0
            elif val == 0:
                in_pos     = False              # normal death-cross exit
                trail_high = 0.0

    return result.astype(int)


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
        regime = stock_regime(df)
    elif asset_class == "bond":
        regime = bond_regime(df)
    elif asset_class == "commodity":
        regime = commodity_regime(df, macro)
    else:
        regime = equity_index_regime(df)

    out["regime"]      = regime
    out["asset_class"] = asset_class

    mom = momentum_rule(df)
    rev = mean_reversion_rule(df)  # thresholds adapt to trailing realized vol

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

        # ── Principled signal improvements (not curve-fitted) ───────────────
        # Applied in this order: entry quality → exit discipline → stop loss.
        # All three use standard financial constants (RSI_ENTRY_THRESH=70,
        # MIN_HOLD_DAYS=5, ATR_TRAILING_MULT=3.0) — none chosen to fit this data.
        signal_r = apply_rsi_entry_filter(signal_r, df["rsi_14"])
        signal_r = apply_min_hold_filter(signal_r)
        signal_r = apply_trailing_stop_signal(signal_r, df["Close"], df["atr_14"])
        signal_r = signal_r * regime_gate  # re-apply gate after post-processing

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
        feat_path = FEATURE_DIR / f"{ticker}.parquet"
        if not feat_path.exists():
            print(f"  {ticker}: feature parquet missing — skipped")
            continue
        df  = pd.read_parquet(feat_path)
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