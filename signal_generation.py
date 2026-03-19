"""
signal_generation.py
--------------------
Asset-class-aware trading signal generation.

This module turns technical features into actionable 0/1 signals.  Every
design decision here has a principled justification — nothing was tuned by
searching over parameter grids on the historical data.

Signal types produced
─────────────────────
  signal_regime     — Primary MA-crossover signal (long-only for most assets).
                      Golden cross (MA50 > MA200) = 1, otherwise 0.
                      Post-processed by RSI entry filter, min-hold filter,
                      and ATR trailing stop.
  signal_composite  — Richer score combining momentum and mean-reversion
                      sub-scores with regime-aware weighting.  Long-only
                      for equity/commodity, two-sided for bonds.

Asset-class routing
───────────────────
  equity_index  → MA50/200 + ADX > 25 regime filter
  sector_etf    → MA100/300 + ADX > 20 (slower: sector rotations take months)
  stock         → MA50/200 + ADX > dynamic threshold (higher vol → higher bar)
  bond          → Two-sided momentum (rates go both ways; mean reversion is risky)
  commodity     → Momentum in fear (VIX > 25), mean reversion in calm

Post-processors applied to signal_regime (principled, not curve-fitted)
────────────────────────────────────────────────────────────────────────
  1. RSI entry filter  — Block new longs when RSI_14 > 70 (overbought at entry)
  2. Min-hold filter   — Stay long for at least MIN_HOLD_DAYS=5 to prevent
                         whipsaw round-trips
  3. ATR trailing stop — Exit if price falls ATR_TRAILING_MULT × ATR below
                         the trailing high since entry

Output
──────
  data/signals/{TICKER}.parquet      — per-ticker signal DataFrame
  data/signals/regime_signals.parquet    — matrix of signal_regime values
  data/signals/composite_signals.parquet — matrix of signal_composite values

Consumed by
───────────
  backtester.py   — simulates P&L against these signals
  portfolio.py    — uses signal matrices for sizing and walk-forward
  paper_trader.py — runs generate() on live data for EOD execution
  live_signals.py — replicates regime logic for the dashboard's Live Signals tab
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
                           # breathe while protecting against structural deterioration).

FEATURE_DIR = Path("data/features")
SIGNAL_DIR  = Path("data/signals")
MACRO_DIR   = Path("data/macro")
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)


def load_macro() -> pd.DataFrame:
    """
    Load macro features from disk.  Returns empty DataFrame if not found.

    The macro file is produced by macro_features.py.  If it is missing,
    downstream functions degrade gracefully:
      - vix_gate() returns all-pass (Series of 1s)
      - commodity_regime() falls back to equity_index_regime()
    """
    path = MACRO_DIR / "macro_features.parquet"
    if not path.exists():
        print("  WARNING: macro_features.parquet not found — run macro_features.py first")
        return pd.DataFrame()
    return pd.read_parquet(path)

# -----------------------------------------------------------------------------
# Regime filters
# -----------------------------------------------------------------------------

def equity_index_regime(df: pd.DataFrame) -> pd.Series:
    """
    Regime filter for broad equity indices (SPY, IWM, EEM).

    Returns 1 (trending = act on momentum signals) when:
      - MA50 > MA200   (golden cross — medium-term trend above long-term trend)
      - ADX > 25       (confirmed trend, not a choppy range)

    Returns 0 otherwise (ranging = suppress signals, stay in cash).

    Args:
        df: Feature DataFrame with columns [Close, adx].

    Returns:
        Binary Series (0 or 1) aligned to df.index.
    """
    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()
    return ((ma50 > ma200) & (df["adx"] > 25)).astype(int)


def sector_regime(df: pd.DataFrame) -> pd.Series:
    """
    Regime filter for sector ETFs (XLE, XLU, XLF).

    Uses slower MA100/300 instead of MA50/200 because sector rotations
    develop over months, not weeks.  A sector that just crossed MA50/200
    may already be halfway through its move; MA100/300 catches confirmed
    multi-month trends.

    ADX threshold is 20 (vs 25 for indices) because sectors trend less
    strongly than broad indices — requiring ADX > 25 would suppress too
    many valid signals.

    Args:
        df: Feature DataFrame with columns [Close, adx].

    Returns:
        Binary Series (0 or 1) aligned to df.index.
    """
    ma100 = df["Close"].rolling(100).mean()
    ma300 = df["Close"].rolling(300).mean()
    return ((ma100 > ma300) & (df["adx"] > 20)).astype(int)


def stock_regime(df: pd.DataFrame) -> pd.Series:
    """
    Regime filter for individual stocks with volatility-adaptive ADX threshold.

    Individual stocks are more volatile than indices, so price action is
    noisier.  A fixed ADX threshold (like the 25 used for indices) would
    either fire too often on low-vol stocks or suppress too many signals on
    high-vol stocks.

    Solution: compute the ADX threshold from each stock's own trailing
    60-day realized volatility.  More volatile stocks require a higher ADX
    reading before a trend is confirmed.

      adx_threshold = 20 + int(trailing_vol × 50)  clipped to [20, 35]

    Example:
      COST (low vol ~15%):  ADX threshold ≈ 27
      NVDA (high vol ~60%): ADX threshold ≈ 35 (capped)

    Args:
        df: Feature DataFrame with columns [Close, adx, log_return].

    Returns:
        Binary Series (0 or 1) aligned to df.index.
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
    """
    Bonds (TLT) always use momentum logic — never mean reversion.

    Interest rate cycles genuinely trend in both directions for years at a
    time (2000–2020 rates fell, 2022 they rose sharply).  Mean-reverting
    TLT during a rising-rate regime is dangerous: rates can keep rising
    for 18+ months.  The regime Series is permanently 1 so that
    signal_generation.generate() always routes TLT to momentum_rule().

    Args:
        df: Feature DataFrame (used only for index alignment).

    Returns:
        Series of 1s aligned to df.index.
    """
    return pd.Series(1, index=df.index)


def commodity_regime(df: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
    """
    Gold (GLD) regime: momentum in fear, mean reversion in calm.

    In a fear environment (VIX > 25), gold trends as a safe-haven flow.
    Investors pile into gold for months — mean-reverting against that flow
    is dangerous.

    In a calm environment (VIX < 25), gold oscillates around its real-rate
    fair value.  The fear premium is absent and mean reversion works.

    Returns 1 (momentum) when VIX > 25, 0 (mean reversion) otherwise.

    Args:
        df:    Feature DataFrame (used for index alignment).
        macro: Macro DataFrame with column [vix].

    Returns:
        Binary Series (0 or 1) aligned to df.index.
        Falls back to equity_index_regime(df) if macro is empty.
    """
    if macro.empty:
        return equity_index_regime(df)
    vix = macro["vix"].reindex(df.index).ffill().fillna(20)
    return (vix > 25).astype(int)  # 1 = fear = use momentum

# -----------------------------------------------------------------------------
# Liquidity gate for individual stocks
# -----------------------------------------------------------------------------

def liquidity_gate(df: pd.DataFrame) -> pd.Series:
    """
    Suppress signals on days when dollar volume is below its 20-day average.

    Thin trading days produce unreliable signals: spreads widen, price
    moves are more likely to reverse, and bid-ask slippage is higher.
    Applied only to individual stocks (not ETFs or indices).

    Uses shift(1) on the rolling average so today's volume must exceed
    YESTERDAY's 20-day average — no look-ahead bias.

    Args:
        df: Feature DataFrame with column [dollar_volume].

    Returns:
        Binary Series (0 or 1).  1 = liquid (allow signal), 0 = thin (suppress).
    """
    # shift(1): today's volume must exceed yesterday's 20-day average — no lookahead
    avg_dv = df["dollar_volume"].rolling(20).mean().shift(1)
    return (df["dollar_volume"] >= avg_dv).astype(int)

# -----------------------------------------------------------------------------
# VIX spike gate
# -----------------------------------------------------------------------------

def vix_gate(macro: pd.DataFrame, index: pd.Index) -> pd.Series:
    """
    Hard gate: suppress all signals on extreme VIX spike days.

    A VIX z-score above 2.5 places the day in the top ~2% of historical
    fear readings.  On such days, technical price signals break down — all
    assets sell off together, correlations spike to 1, and trend/momentum
    logic is essentially noise.

    This is NOT a directional signal — it does not predict the VIX will
    fall.  It simply says "this is not an environment for technical trading;
    stand aside."

    Args:
        macro: Macro DataFrame with column [vix_zscore].
        index: DatetimeIndex to align the gate Series to.

    Returns:
        Binary Series (0 or 1).  0 on spike days (block), 1 otherwise (pass).
    """
    if macro.empty:
        return pd.Series(1, index=index)
    vix_z = macro["vix_zscore"].reindex(index).ffill().fillna(0)
    return (vix_z < 2.5).astype(int)

# -----------------------------------------------------------------------------
# Signal rules — with ticker-aware thresholds
# -----------------------------------------------------------------------------

def momentum_rule(df: pd.DataFrame) -> pd.Series:
    """
    Three-way momentum confirmation signal.

    Requires agreement from three independent momentum indicators:
      - mom_20 > 0   (20-day return positive — medium-term momentum)
      - mom_60 > 0   (60-day return positive — longer-term momentum)
      - macd_hist > 0 (MACD histogram positive — short-term momentum turning up)

    Three-way agreement reduces false signals compared to any single indicator.
    All three disagreeing on direction produces a 0 (flat) signal.

    Returns:
        Series of {-1, 0, 1}.  Long (+1) or short (−1) only when all three agree.
    """
    long  = (df["mom_20"] > 0) & (df["mom_60"] > 0) & (df["macd_hist"] > 0)
    short = (df["mom_20"] < 0) & (df["mom_60"] < 0) & (df["macd_hist"] < 0)
    return pd.Series(np.select([long, short], [1, -1], default=0), index=df.index)


def mean_reversion_rule(df: pd.DataFrame) -> pd.Series:
    """
    Volatility-adaptive mean reversion: buy oversold, sell overbought.

    Two independent signals must agree before generating a reversion trade:
      - Price z-score is extreme (far from rolling mean)
      - RSI confirms the same directional extreme

    Both thresholds scale with each asset's own trailing 252-day realized
    volatility — more volatile assets naturally have wider price swings
    that are NOT dislocations, so they require larger extremes before a
    reversion signal fires.  This eliminates the need for per-ticker
    parameter tuning.

      z_thresh = (1.0 + realized_vol)  clipped to [1.0, 2.5]
      rsi_lo   = (30 + int(realized_vol × 20))  clipped to [28, 40]
      rsi_hi   = 100 − rsi_lo

    Example (realized_vol = 0.20):
      z_thresh = 1.2,  rsi_lo = 34,  rsi_hi = 66

    Args:
        df: Feature DataFrame with [zscore_20, rsi_14, log_return].

    Returns:
        Series of {-1, 0, 1}.
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
    Build a normalised composite score combining momentum and mean-reversion.

    Each raw feature is first rolled-z-scored (cross-sectionally standardised
    within its own history) so features with different scales contribute equally.

    Sub-scores:
      score_momentum = mean(s_mom20, s_mom60, s_macd)
      score_mean_rev = mean(s_zscore, s_rsi, s_bb)   — inverted (negative = oversold)
      score_composite = regime-weighted blend × OBV amplifier

    Regime-aware blending:
      In a trending market (regime=1): 80% momentum, 20% mean reversion.
      In a ranging market (regime=0):  20% momentum, 80% mean reversion.

    Structural weights (0.6/0.2) are NOT per-ticker optimised — they reflect
    the strategic decision to trust trend signals in trending markets and
    reversion signals in ranging ones.

    Args:
        df:     Feature DataFrame.
        regime: Binary regime Series from one of the regime functions.
        ticker: Ticker string (used for logging only; does not affect scores).

    Returns:
        DataFrame of all sub-scores and the composite score.
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
    """
    Rolling z-score of a series: (value − rolling_mean) / rolling_std.

    Args:
        series: Input pandas Series.
        window: Rolling window length.  min_periods = window // 2 so the
                early warm-up period is included (not dropped entirely).

    Returns:
        Series of z-scores.  Values outside ±3 indicate extreme readings.
    """
    roll = series.rolling(window, min_periods=window // 2)
    return (series - roll.mean()) / roll.std()


def scores_to_signal(score: pd.Series, long_thresh: float = 0.5,
                     short_thresh: float = -0.5) -> pd.Series:
    """
    Threshold a composite score Series into a {-1, 0, 1} signal.

    Args:
        score:        Composite score Series (unbounded float).
        long_thresh:  Score above this generates a LONG signal (default 0.5).
        short_thresh: Score below this generates a SHORT signal (default -0.5).

    Returns:
        Signal Series: +1 (long), -1 (short), 0 (flat / cash).
    """
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
    Block new LONG entries when RSI_14 > RSI_ENTRY_THRESH at the time of entry.

    The golden cross identifies trend direction.  The RSI gate ensures we
    only enter early in the trend, not after a sustained run-up where the
    entry risk/reward is poor.  Wilder's original overbought level of 70
    (published 1978) is used — not fitted to this data.

    State machine: tracks whether we are currently in a position so the
    filter only blocks NEW entries (not ongoing positions, which have
    already made money from the initial move).

    Args:
        signal: Binary signal Series (0 or 1) BEFORE the RSI filter.
        rsi:    RSI-14 Series aligned to signal.index.

    Returns:
        Filtered signal Series.  Same index and dtype (int) as input.
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
    Hold a long position for at least MIN_HOLD_DAYS trading days before exiting.

    MA crossovers can produce same-week golden/death-cross pairs (whipsaws)
    where the MA crosses up on Monday, then back down on Friday.  Without
    this filter, that costs two round-trip commissions for near-zero net move.
    MIN_HOLD_DAYS=5 (one trading week) is the natural minimum unit — any
    shorter and the strategy is HFT, not systematic medium-term.

    State machine: counts the number of consecutive days in the position.
    Exit signals received before MIN_HOLD_DAYS are converted to holds (1s).

    Args:
        signal: Binary signal Series (0 or 1) BEFORE the min-hold filter.

    Returns:
        Filtered signal Series.  Same index and dtype (int) as input.
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
    Close a long position if price falls ATR_TRAILING_MULT × ATR below the
    trailing high since entry.

    ATR-scaled stops adapt to each asset's volatility automatically.  A 3×
    stop on SPY (low vol) is tighter in dollar terms than 3× on NVDA (high
    vol), which is the correct behaviour — NVDA has wider natural swings
    that should not trigger a stop.

    ATR_TRAILING_MULT = 3.0 is an institutional standard (referenced in
    Elder, Schwager).  It was not chosen by optimising against this backtest.

    Fallback: if ATR data is unavailable, uses a fixed 20% below the
    trailing high.

    Args:
        signal: Binary signal Series (0 or 1) after min-hold filter.
        close:  Close price Series aligned to signal.index.
        atr:    ATR-14 Series aligned to signal.index.

    Returns:
        Filtered signal Series.  Same index and dtype (int) as input.
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
    """
    Master signal generator — routes each ticker to the correct logic and
    applies all post-processors.

    Args:
        df:     Feature DataFrame from feature_engineering.engineer().
        ticker: Ticker symbol string (used for asset-class routing).
        macro:  Macro DataFrame from load_macro().

    Returns:
        DataFrame with columns:
          Close, log_return, regime, asset_class,
          volume_filter, signal_regime, signal_composite,
          s_mom20, s_mom60, s_macd, s_adx, s_zscore, s_rsi, s_bb, s_obv,
          score_momentum, score_mean_rev, score_composite.
        Rows with NaN in any column are dropped.
    """
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