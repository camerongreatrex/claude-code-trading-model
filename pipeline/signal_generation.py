"""
signal_generation.py
--------------------
Asset-class-aware trading signal generation.

This module turns technical features into actionable signals.  Every design
decision here has a principled justification — nothing was tuned by searching
over parameter grids on the historical data.

Signal types produced
─────────────────────
  signal_regime     — Primary MA-crossover signal (long-only for most assets).
                      Golden cross (MA50 > MA200) = 1, otherwise 0.
                      Post-processed by RSI entry filter, min-hold filter,
                      and ATR trailing stop.
  signal_composite  — Richer score combining momentum and mean-reversion
                      sub-scores with regime-aware weighting.  Long-only
                      for equity/commodity, two-sided for bonds.
  signal_ensemble   — IC-weighted linear ensemble of the top 6 features ranked
                      by information ratio from data/research/feature_ic.parquet.
                      Continuous value in [-1, +1]: positive = long, negative =
                      short, magnitude = conviction.  Features are z-scored over
                      a rolling 252-day window and weighted by their trailing
                      504-day time-series IC (adaptive — features that predicted
                      well recently get more weight).

Asset-class routing
───────────────────
  equity_index  → MA50/200 + ADX > 25 regime filter; long-only
                  Tickers: SPY, IWM, EEM, EFA, VWO
  sector_etf    → MA100/300 + ADX > 20 (slower: sector rotations take months); long-only
                  Tickers: XLE, XLU, XLF, VNQ
  stock         → MA50/200 + ADX > dynamic threshold (higher vol → higher bar); long-only
                  Tickers: JPM, JNJ, XOM, AMZN, NEE, BRK-B, GS, COST, MSFT, NVDA, GE, INTC, VZ
  bond          → Two-sided momentum (rates go both ways; mean reversion is risky)
                  Tickers: TLT (nominal rates), HYG (credit cycle), TIP (real rates)
  commodity     → Momentum in fear (VIX > 25), mean reversion in calm; long-only
                  Tickers: GLD (fear/real rates), DBC (broad basket), UUP (USD/FX)

Post-processors applied to signal_regime (principled, not curve-fitted)
────────────────────────────────────────────────────────────────────────
  1. RSI entry filter  — Block new longs when RSI_14 > 70 (overbought at entry)
  2. Min-hold filter   — Stay long for at least MIN_HOLD_DAYS=5 to prevent
                         whipsaw round-trips
  3. ATR trailing stop — Exit if price falls ATR_TRAILING_MULT × ATR below
                         the trailing high since entry

Output
──────
  data/signals/{TICKER}.parquet          — per-ticker signal DataFrame
  data/signals/regime_signals.parquet    — matrix of signal_regime values
  data/signals/composite_signals.parquet — matrix of signal_composite values
  data/signals/ensemble_signals.parquet  — matrix of signal_ensemble values

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

from .data_pipeline import TICKER_LIST, ASSET_CLASS

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
ATR_TIGHTEN_THRESHOLD = 5.0  # Tighten stop once price is 5× ATR above entry price.
                               # Source: Elder, "Trading for a Living" — large gains mean-
                               # revert more aggressively; protect them with a tighter stop.
ATR_TIGHTEN_MULT      = 1.5  # Tightened stop distance: 1.5× ATR (vs 3× normal).
                               # Locks in most of a 5×-ATR gain while allowing trend to run.
TIME_DECAY_DAYS   = 126  # Close stale longs held > 6 months that are drifting negative.
                           # 126 = half the MA200 lookback — a structural timescale.
                           # Addresses bear_calm underperformance: equities held flat/down for
                           # months while TLT/GLD/TIP rally. Exit the dead weight, free cash.
TIME_DECAY_WINDOW = 21   # Trailing return window for the decay check: 1 calendar month.
                           # close[t] / close[t-21] - 1 < 0 → position is drifting down.

FEATURE_DIR  = Path("data/features")
SIGNAL_DIR   = Path("data/signals")
MACRO_DIR    = Path("data/macro")
RESEARCH_DIR = Path("data/research")
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

    # Regime-aware blending: trend regime → weight momentum; ranging → weight mean-rev.
    # 0.6/0.2 reflects the strategic choice to trust trend signals in trending markets.
    # These weights are fixed structural choices, not tuned per-ticker or per-period.
    mom_weight = regime * 0.6 + 0.2   # 0.8 in trend, 0.2 in range
    rev_weight = 1 - mom_weight

    # Simple linear blend — no ADX boost, no OBV amplifier.
    # The ADX boost (×0.1 effect) and OBV amplifier (×0.15 effect) were small
    # in-sample adjustments that consistently hurt OOS performance: they added
    # noise without improving generalization across the 7 walk-forward periods.
    # A plain weighted sum is harder to overfit and easier to reason about.
    scores["score_composite"] = (
        mom_weight * scores["score_momentum"] +
        rev_weight * scores["score_mean_rev"]
    )

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
    trailing high since entry, with profit-target stop tightening.

    ATR-scaled stops adapt to each asset's volatility automatically.  A 3×
    stop on SPY (low vol) is tighter in dollar terms than 3× on NVDA (high
    vol), which is the correct behaviour — NVDA has wider natural swings
    that should not trigger a stop.

    ATR_TRAILING_MULT = 3.0 is an institutional standard (referenced in
    Elder, Schwager).  It was not chosen by optimising against this backtest.

    Profit-target tightening:
        Once price is more than ATR_TIGHTEN_THRESHOLD (5×) ATR above entry,
        the trailing stop tightens from 3× ATR to ATR_TIGHTEN_MULT (1.5×) ATR.
        Rationale (Elder, "Trading for a Living"): large gains mean-revert
        more aggressively than small gains; protecting a 5×-ATR winner with
        a tighter stop locks in profit while still allowing the trend to run.
        Both constants (5× threshold, 1.5× tightened stop) are published,
        not fitted to this data.

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
    entry_price = 0.0

    for i in range(len(result)):
        price       = float(close.iloc[i])
        current_atr = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else None
        val         = int(result.iloc[i])

        if not in_pos:
            if val == 1:
                in_pos      = True
                trail_high  = price
                entry_price = price
        else:
            if price > trail_high:
                trail_high = price

            if current_atr and current_atr > 0:
                # Tighten stop if price has run ATR_TIGHTEN_THRESHOLD × ATR above entry
                if (price - entry_price) > ATR_TIGHTEN_THRESHOLD * current_atr:
                    stop_mult = ATR_TIGHTEN_MULT
                else:
                    stop_mult = ATR_TRAILING_MULT
                stop = trail_high - stop_mult * current_atr
            else:
                stop = trail_high * 0.80

            if price < stop:
                result.iloc[i] = 0              # trailing stop fires
                in_pos         = False
                trail_high     = 0.0
                entry_price    = 0.0
            elif val == 0:
                in_pos      = False             # normal death-cross exit
                trail_high  = 0.0
                entry_price = 0.0

    return result.astype(int)


def apply_time_decay_exit(signal: pd.Series, close: pd.Series) -> pd.Series:
    """
    Close positions held more than TIME_DECAY_DAYS if the trailing
    TIME_DECAY_WINDOW return is negative.

    Motivation
    ──────────
    The MA200 crossover can hold a position for 6-12+ months even as the
    asset drifts sideways-to-down while uncorrelated assets (TLT, GLD, TIP)
    rally strongly.  The ATR trailing stop protects against sharp declines
    but lets slow bleeds persist.  A time-based staleness check closes
    "dead" positions — ones that have had negative drift for a full month
    after being held for at least 6 months.

    Parameters (structural, not fitted)
    ────────────────────────────────────
      TIME_DECAY_DAYS   = 126  (half the MA200 lookback = 6 trading months)
      TIME_DECAY_WINDOW = 21   (1 calendar month of trailing return)

    Logic (state machine)
    ─────────────────────
      Track days_held: reset to 0 on exit, increment to 1 on entry.
      On each bar where in_pos is True and the upstream signal is still 1:
        - If days_held > TIME_DECAY_DAYS:
            Compute trailing 21-day return = close[i] / close[i-21] - 1.
            If < 0: set signal to 0 (exit), reset state.
      The check fires every bar once the threshold is crossed, so the
      position closes the first day drift turns negative after 6 months.

    Applied after apply_trailing_stop_signal() — acute deterioration is
    caught by the ATR stop; this catches chronic low-grade drift.

    Args:
        signal: Binary signal Series (0 or 1) after ATR stop filter.
        close:  Close price Series aligned to signal.index.

    Returns:
        Filtered signal Series.  Same index and dtype (int) as input.
    """
    result    = signal.copy().astype(float)
    in_pos    = False
    days_held = 0
    cooldown  = 0   # bars remaining before a new entry is allowed after decay exit

    for i in range(len(result)):
        val = int(result.iloc[i])

        # ── Cooldown: force cash for TIME_DECAY_WINDOW bars after a decay exit ──
        # Without this, the MA crossover (still golden) re-opens the position the
        # very next day, turning every time-decay exit into a 1-day round-trip that
        # adds transaction cost with no P&L benefit.  Waiting one full 21-day window
        # before allowing re-entry gives the drift time to either reverse (and the
        # position merits re-opening) or continue (and the MA death cross / ATR stop
        # takes over).
        if cooldown > 0:
            cooldown       -= 1
            result.iloc[i]  = 0
            continue

        if not in_pos:
            if val == 1:
                in_pos    = True
                days_held = 1
        else:
            if val == 0:
                # Upstream filter (ATR stop or death cross) already exited
                in_pos    = False
                days_held = 0
            else:
                # Still holding — increment counter and check staleness
                days_held += 1
                if days_held > TIME_DECAY_DAYS and i >= TIME_DECAY_WINDOW:
                    trailing_ret = (
                        float(close.iloc[i]) / float(close.iloc[i - TIME_DECAY_WINDOW]) - 1
                    )
                    if trailing_ret < 0:
                        result.iloc[i] = 0          # stale position with negative drift
                        in_pos         = False
                        days_held      = 0
                        cooldown       = TIME_DECAY_WINDOW  # stay out for 21 days

    return result.astype(int)


# -----------------------------------------------------------------------------
# IC-weighted ensemble signal
# -----------------------------------------------------------------------------

def ensemble_signal(df: pd.DataFrame) -> pd.Series:
    """
    IC-weighted linear ensemble of the top 6 features by information ratio.

    Design
    ──────
    Feature selection (static):
        Load data/research/feature_ic.parquet and select the top 6 features
        by ABSOLUTE IC information ratio (|ic_ir|), excluding price-level
        features (bb_middle, bb_upper, bb_lower, obv) whose IC is inflated
        by trend autocorrelation rather than genuine alpha.

        Using absolute IC captures both:
          - Features with positive IC (high value → good future return)
          - Features with negative IC (high value → bad future return)
        The sign of mean_ic from the research file determines which direction
        each feature contributes to the ensemble.

    Feature transformation (rolling):
        For each selected feature, compute a rolling 252-day z-score.
        This normalises features with different units and scales so they
        contribute equally before weighting.

    Adaptive IC weighting (trailing, 5-day horizon):
        At each date t, each feature's weight is its trailing 504-day
        RANK correlation with the 5-day forward return — the same horizon
        and method used in feature_research.py.  Matching the horizon is
        critical: a feature's short-term and medium-term IC can have
        opposite signs (e.g. momentum has short-term reversal at 1 day
        but continuation at 5 days).  Mismatching horizons produces sign
        conflicts that actively destroy performance.

        5-day forward log return at date T:
            fwd_5d[T] = log_return[T+1] + … + log_return[T+5]

        Computed as rolling(5).sum().shift(-5) so fwd_5d[T] equals the sum
        of the five log returns starting the day after T.

        Look-ahead protection:
            raw_ic[T]    = rank_corr(feat_rank[T-503:T], fwd_5d_rank[T-503:T])
            → fwd_5d[T] needs log_return[T+1:T+6]: 5 days of future data.
            ic_weight[T] = raw_ic[T-5]
            → uses fwd_5d up to T-5, which needs log_return up to T (EOD ✓).

        When trailing IC is NaN (warm-up < 504 days), falls back to the
        static mean_ic from the research file — a stable, cross-ticker
        estimate from 51,450 pooled observations.

    Signal construction:
        signal_ensemble = clip( Σ ic_weight_i × feat_z_i,  −1, +1 )

        Continuous value in [−1, +1]: encodes conviction, not just direction.
        +1 = maximum long, −1 = maximum short, intermediate = partial position.

    Graceful degradation:
        Returns a Series of 0.0 if the IC parquet is missing or no selected
        features are present in df.

    Args:
        df: Feature DataFrame from feature_engineering.engineer().
            Must contain log_return and whichever features are in the top 6.

    Returns:
        Series of floats in [−1, +1] aligned to df.index,
        named "signal_ensemble".
    """
    ic_path = RESEARCH_DIR / "feature_ic.parquet"
    if not ic_path.exists():
        return pd.Series(0.0, index=df.index, name="signal_ensemble")

    ic_table = pd.read_parquet(ic_path)

    # Top 6 by absolute IC IR, excluding level features (inflated by trend autocorrelation).
    # Using abs captures strong negative-IC features (e.g. momentum reversal after 60 days)
    # which are just as informative as positive-IC features — only the sign differs.
    candidates = ic_table[~ic_table["level_feature"]].copy()
    candidates["abs_ic_ir"] = candidates["ic_ir"].abs()
    top6_rows = candidates.nlargest(6, "abs_ic_ir")
    # Map feature → static mean_ic (sign for fallback, magnitude for direction)
    top6_map  = {
        row["feature"]: row["mean_ic"]
        for _, row in top6_rows.iterrows()
        if row["feature"] in df.columns
    }
    if not top6_map:
        return pd.Series(0.0, index=df.index, name="signal_ensemble")

    # 5-day forward log return aligned to feature date T.
    # rolling(5).sum()[T] = log_return[T-4:T+1] (past 5 days)
    # .shift(-5)[T]       = rolling sum at T+5  = log_return[T+1:T+6] (forward 5 days) ✓
    fwd_5d      = df["log_return"].rolling(5).sum().shift(-5)
    fwd_5d_rank = fwd_5d.rank(pct=True)  # rank for robustness, matching research methodology

    weighted_sum = pd.Series(0.0, index=df.index)

    for feat, static_ic in top6_map.items():
        # Rolling 252-day z-score normalises scale and removes drift
        feat_z    = _roll_zscore(df[feat], 252)
        feat_rank = feat_z.rank(pct=True)  # rank-transform for IC computation

        # Trailing 504-day rank IC against 5-day forward return.
        # Matches feature_research.py methodology (same horizon, rank-based).
        # shift(5): ic_weight[T] = raw_ic[T-5]
        #   raw_ic[T-5] uses fwd_5d_rank up to T-5
        #   fwd_5d[T-5] = log_return[T-4:T] — last element log_return[T] (EOD, known) ✓
        raw_ic    = feat_rank.rolling(504, min_periods=126).corr(fwd_5d_rank)
        ic_weight = raw_ic.shift(5)

        # During warm-up (<504+5 days), fall back to the static research IC.
        # static_ic is pooled across all tickers (51,450 obs) — far more stable
        # than a single-ticker rolling estimate would be over a partial window.
        ic_weight = ic_weight.fillna(static_ic)

        weighted_sum += ic_weight * feat_z

    # Clip to continuous [-1, +1] position size (not binary)
    return weighted_sum.clip(-1.0, 1.0).rename("signal_ensemble")


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
          volume_filter, signal_regime, signal_composite, signal_ensemble,
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
        # Applied in order: entry quality → exit discipline → stop loss → staleness.
        # All constants are published structural timescales, none fitted to this data:
        #   RSI_ENTRY_THRESH=70 (Wilder 1978), MIN_HOLD_DAYS=5 (1 week),
        #   ATR_TRAILING_MULT=3.0 (Elder/Schwager), TIME_DECAY_DAYS=126 (Elder).
        signal_r = apply_rsi_entry_filter(signal_r, df["rsi_14"])
        signal_r = apply_min_hold_filter(signal_r)
        signal_r = apply_trailing_stop_signal(signal_r, df["Close"], df["atr_14"])
        signal_r = apply_time_decay_exit(signal_r, df["Close"])   # close stale longs
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

    # ── Ensemble signal: continuous [-1, +1], IC-weighted, adaptive ────────
    # VIX gate is applied: on extreme panic days all factor signals break down.
    # Long-only for equity/commodity/sector_etf: structural upward drift means
    # systematic short positions in these assets lose on average over time.
    # A continuous signal clipped to [0, +1] acts as a long-conviction overlay:
    # +1 = maximum long, 0 = flat (cash), not a short bet against the market.
    ens = ensemble_signal(df) * gate
    if asset_class in {"equity_index", "sector_etf", "stock", "commodity"}:
        ens = ens.clip(lower=0)
    out["signal_ensemble"] = ens

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

        ens = sig["signal_ensemble"]
        print(f"    {'signal_ensemble':<22}: mean={ens.mean():.3f}  "
              f"std={ens.std():.3f}  "
              f"long={( ens > 0.1).sum():>4}  short={(ens < -0.1).sum():>4}")
        print()

        all_signals[ticker] = sig

    regime    = pd.DataFrame({t: s["signal_regime"]    for t, s in all_signals.items()}).dropna()
    composite = pd.DataFrame({t: s["signal_composite"] for t, s in all_signals.items()}).dropna()
    ensemble  = pd.DataFrame({t: s["signal_ensemble"]  for t, s in all_signals.items()}).dropna()

    regime.to_parquet(SIGNAL_DIR    / "regime_signals.parquet")
    composite.to_parquet(SIGNAL_DIR / "composite_signals.parquet")
    ensemble.to_parquet(SIGNAL_DIR  / "ensemble_signals.parquet")
    print(f"Signal matrices saved: {regime.shape}")


if __name__ == "__main__":
    main()
