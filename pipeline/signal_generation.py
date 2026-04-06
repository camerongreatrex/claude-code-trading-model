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
  commodity     → Momentum in fear (VIX > 25), mean reversion in calm; TWO-SIDED
                  Tickers: GLD (fear/real rates), DBC (broad basket), UUP (USD/FX)
                  Short side enabled: gold/DBC trend down in rising real-rate regimes

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

import json
import numpy as np
import pandas as pd
from pathlib import Path

from .data_pipeline import TICKER_LIST, ASSET_CLASS, HEDGE_MAP

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

# 5 most liquid ETFs by AUM (published fact). MA20/50 fast overlay applied
# only to these — individual stocks are excluded (MA20/50 whipsaws on stocks).
FAST_SIGNAL_TICKERS = {"SPY", "IWM", "TLT", "GLD", "EEM"}

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

def vix_position_scalar(macro: pd.DataFrame, index: pd.Index) -> pd.Series:
    """
    Continuous VIX-based position size scalar in [0.30, 1.0].

    Returns 1.0 in normal conditions, smoothly reduces to 0.30
    at extreme VIX. Uses CBOE published VIX regime breakpoints.
    Applied to sizing only — does not affect signal direction.
    """
    if macro.empty:
        return pd.Series(1.0, index=index)
    vix = macro["vix"].reindex(index).ffill().fillna(20)
    xp  = [0,   15,   20,   25,   30,   35,  100]
    fp  = [1.0, 1.0,  1.0,  0.70, 0.50, 0.35, 0.30]
    scalar = pd.Series(
        np.interp(vix.values, xp, fp),
        index=index,
        name="vix_position_scalar",
    )
    return scalar


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
    Block new entries at RSI extremes: longs when RSI > 70 (overbought),
    shorts when RSI < 30 (oversold — symmetric rule for short side).

    State machine tracks current position direction (0 = flat, +1 = long,
    -1 = short) so the filter only blocks NEW entries, not ongoing positions.

    Args:
        signal: Signal Series {-1, 0, 1} BEFORE the RSI filter.
        rsi:    RSI-14 Series aligned to signal.index.

    Returns:
        Filtered signal Series.  Same index and dtype (int) as input.
    """
    result       = signal.copy().astype(float)
    position_dir = 0   # 0 = flat, +1 = long, -1 = short
    rsi_oversold = 100 - RSI_ENTRY_THRESH   # symmetric short-entry gate: 30

    for i in range(len(result)):
        val = int(result.iloc[i])
        if position_dir == 0:
            if val == 1:
                if float(rsi.iloc[i]) > RSI_ENTRY_THRESH:
                    result.iloc[i] = 0          # block overbought long entry
                else:
                    position_dir = 1
            elif val == -1:
                if float(rsi.iloc[i]) < rsi_oversold:
                    result.iloc[i] = 0          # block oversold short entry
                else:
                    position_dir = -1
        else:
            if val == 0:
                position_dir = 0

    return result.astype(int)


def apply_min_hold_filter(signal: pd.Series, min_hold: int = None) -> pd.Series:
    """
    Hold any position (long or short) for at least min_hold days before exiting.

    State machine tracks current direction (+1/-1) so the min-hold applies
    symmetrically to both sides.  Early exit signals are converted to
    "hold current direction" rather than flat.

    Args:
        signal:   Signal Series {-1, 0, 1} BEFORE the min-hold filter.
        min_hold: Override for minimum hold days. Defaults to MIN_HOLD_DAYS (5).
                  Pass min_hold=3 for the faster MA20/50 overlay.

    Returns:
        Filtered signal Series.  Same index and dtype (int) as input.
    """
    hold_days    = min_hold if min_hold is not None else MIN_HOLD_DAYS
    result       = signal.copy().astype(float)
    position_dir = 0   # 0 = flat, +1 = long, -1 = short
    days_held    = 0

    for i in range(len(result)):
        val = int(result.iloc[i])
        if position_dir == 0:
            if val != 0:
                position_dir = val
                days_held    = 1
        else:
            days_held += 1
            if val == 0:
                if days_held <= hold_days:
                    result.iloc[i] = position_dir   # hold current direction
                else:
                    position_dir = 0
                    days_held    = 0

    return result.astype(int)


def apply_trailing_stop_signal(signal: pd.Series,
                                close: pd.Series,
                                atr: pd.Series,
                                atr_mult: float = None,
                                macro: pd.DataFrame = None) -> pd.Series:
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

    VIX-adaptive stop tightening (when macro is provided):
        In bull_stress (VIX 20–25): tighten to 2.5× ATR (trail_mult × 0.83).
        In elevated stress (VIX > 25): tighten to 2.0× ATR (trail_mult × 0.67,
        min 1.5×).  Locks in more profit during VIX 20–30 transitions to bear.
        VIX breakpoints (20, 25) are CBOE published regime boundaries — not
        fitted to this dataset.  Only applies to the trailing stop multiplier;
        the profit-target tighten (ATR_TIGHTEN_MULT) is unchanged.

    Fallback: if ATR data is unavailable, uses a fixed 20% below the
    trailing high.

    Args:
        signal:   Binary signal Series (0 or 1) after min-hold filter.
        close:    Close price Series aligned to signal.index.
        atr:      ATR-14 Series aligned to signal.index.
        atr_mult: Override for trailing stop ATR multiplier. Defaults to
                  ATR_TRAILING_MULT (3.0). Pass atr_mult=2.0 for the faster
                  MA20/50 overlay where tighter stops suit the shorter timeframe.
        macro:    Optional macro DataFrame with 'vix' column.  When provided,
                  enables VIX-adaptive stop tightening in elevated-VIX regimes.

    Returns:
        Filtered signal Series.  Same index and dtype (int) as input.
    """
    trail_mult   = atr_mult if atr_mult is not None else ATR_TRAILING_MULT
    result       = signal.copy().astype(float)
    position_dir = 0   # 0 = flat, +1 = long, -1 = short
    trail_high   = 0.0
    trail_low    = 0.0
    entry_price  = 0.0

    # VIX-adaptive stop: precompute aligned VIX series if macro is provided
    vix_series = None
    if macro is not None and not macro.empty and "vix" in macro.columns:
        vix_series = macro["vix"].reindex(signal.index).ffill().fillna(20.0)

    for i in range(len(result)):
        price       = float(close.iloc[i])
        current_atr = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else None
        val         = int(result.iloc[i])

        # VIX-adaptive trailing stop multiplier (only affects the trailing stop,
        # not the profit-target tighten which stays at ATR_TIGHTEN_MULT = 1.5×).
        if vix_series is not None:
            current_vix = float(vix_series.iloc[i])
            if current_vix > 25:
                effective_mult = max(trail_mult * 0.67, 1.5)  # ≈ 2.0× at 3.0 base
            elif current_vix > 20:
                effective_mult = trail_mult * 0.83              # ≈ 2.5× at 3.0 base
            else:
                effective_mult = trail_mult
        else:
            effective_mult = trail_mult

        if position_dir == 0:
            if val != 0:
                position_dir = val
                trail_high   = price
                trail_low    = price
                entry_price  = price
        elif position_dir == 1:
            # Long: trail the high, stop below it
            if price > trail_high:
                trail_high = price

            if current_atr and current_atr > 0:
                if (price - entry_price) > ATR_TIGHTEN_THRESHOLD * current_atr:
                    stop_mult = ATR_TIGHTEN_MULT
                else:
                    stop_mult = effective_mult
                stop = trail_high - stop_mult * current_atr
            else:
                stop = trail_high * 0.80

            if price < stop:
                result.iloc[i] = 0              # trailing stop fires (long)
                position_dir   = 0
                trail_high     = 0.0
                entry_price    = 0.0
            elif val == 0:
                position_dir = 0                # normal exit
                trail_high   = 0.0
                entry_price  = 0.0

        else:  # position_dir == -1 (short)
            # Short: trail the low, stop above it
            if price < trail_low:
                trail_low = price

            if current_atr and current_atr > 0:
                # Tighten stop once price has fallen ATR_TIGHTEN_THRESHOLD × ATR below entry
                if (entry_price - price) > ATR_TIGHTEN_THRESHOLD * current_atr:
                    stop_mult = ATR_TIGHTEN_MULT
                else:
                    stop_mult = effective_mult
                stop = trail_low + stop_mult * current_atr
            else:
                stop = trail_low * 1.20

            if price > stop:
                result.iloc[i] = 0              # trailing stop fires (short)
                position_dir   = 0
                trail_low      = 0.0
                entry_price    = 0.0
            elif val == 0:
                position_dir = 0                # normal exit
                trail_low    = 0.0
                entry_price  = 0.0

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
    result       = signal.copy().astype(float)
    position_dir = 0   # 0 = flat, +1 = long, -1 = short
    days_held    = 0
    cooldown     = 0   # bars remaining before a new entry is allowed after decay exit

    for i in range(len(result)):
        val = int(result.iloc[i])

        # ── Cooldown: force cash for TIME_DECAY_WINDOW bars after a decay exit ──
        if cooldown > 0:
            cooldown       -= 1
            result.iloc[i]  = 0
            continue

        if position_dir == 0:
            if val != 0:
                position_dir = val
                days_held    = 1
        else:
            if val == 0:
                # Upstream filter already exited
                position_dir = 0
                days_held    = 0
            else:
                # Still holding — increment counter and check staleness
                days_held += 1
                if days_held > TIME_DECAY_DAYS and i >= TIME_DECAY_WINDOW:
                    trailing_ret = (
                        float(close.iloc[i]) / float(close.iloc[i - TIME_DECAY_WINDOW]) - 1
                    )
                    # Long: negative drift = stale. Short: positive drift = stale.
                    stale = (position_dir == 1 and trailing_ret < 0) or \
                            (position_dir == -1 and trailing_ret > 0)
                    if stale:
                        result.iloc[i] = 0
                        position_dir   = 0
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
# Earnings blackout filter
# -----------------------------------------------------------------------------

def apply_earnings_blackout(
    signal: pd.Series,
    earnings_dates: list,
    blackout_before: int = 2,
    blackout_after: int = 1,
) -> pd.Series:
    """
    Set signal to 0 (flat) during a window around each quarterly earnings date.

    For each earnings date, the strategy goes flat for:
      blackout_before  trading days before the announcement
      blackout_after   trading days after the announcement

    Default 2-before / 1-after = 3-day window per event (4 per year = ~12 days).
    This is a risk-management filter, not an alpha signal.  The expected return
    from holding through earnings is ~0% but variance is huge (3-8% gap).
    Removing variance at ~0 expected cost improves Sharpe through the denominator.

    Args:
        signal:          Per-ticker signal Series (DatetimeIndex).
        earnings_dates:  List of ISO-format date strings ("YYYY-MM-DD").
        blackout_before: Trading days to go flat BEFORE the announcement.
        blackout_after:  Trading days to go flat ON and AFTER the announcement.

    Returns:
        Copy of signal with blackout windows zeroed out.
    """
    if not earnings_dates:
        return signal

    result = signal.copy()
    idx    = result.index  # DatetimeIndex of trading days

    for date_str in earnings_dates:
        edate = pd.Timestamp(date_str)
        # Find nearest trading day (earnings sometimes fall on weekends / market-closed days)
        pos = idx.get_indexer([edate], method="nearest")[0]
        if pos < 0:
            continue
        start = max(0, pos - blackout_before)
        end   = min(len(result) - 1, pos + blackout_after)
        result.iloc[start : end + 1] = 0

    return result


# -----------------------------------------------------------------------------
# Master generate — routes each ticker to the right logic
# -----------------------------------------------------------------------------

def fast_signal(df: pd.DataFrame, ticker: str, macro: pd.DataFrame) -> pd.Series:
    """
    MA20/50 short-term trend overlay for the 5 most liquid ETFs.

    Faster than the MA50/200 regime signal — captures trend initiations
    earlier.  Layered ON TOP of the slow signal as an average, so it never
    fully overrides the slow regime filter.

    For equity_index tickers (SPY, IWM, EEM): long-only (same as MA50/200).
    For TLT (bond):  two-sided — rate trends go both ways.
    For GLD (commodity): two-sided — gold trends both directions.

    Post-processors use tighter settings for the faster timeframe:
      min_hold=3   (3 days vs 5 for MA50/200 — quicker reaction)
      atr_mult=2.0 (2× ATR stop vs 3× — tighter stop for shorter-lived trends)

    RSI entry filter unchanged — same overbought/oversold thresholds.
    VIX gate unchanged — same regime filter as all other signals.

    Args:
        df:     Feature DataFrame from feature_engineering.engineer().
        ticker: Ticker symbol.
        macro:  Macro DataFrame from load_macro().

    Returns:
        Signal Series.  Equity_index: {0, 1}.  TLT/GLD: {-1, 0, 1}.
    """
    gate       = vix_gate(macro, df.index)
    ma20       = df["Close"].rolling(20).mean()
    ma50       = df["Close"].rolling(50).mean()
    asset_class = ASSET_CLASS[ticker]

    if asset_class == "equity_index":
        # Long-only: no structural reason to short broad equity indices.
        fast_sig = (ma20 > ma50).astype(int) * gate
    else:
        # Two-sided for TLT (bond) and GLD (commodity): trends run both ways.
        fast_sig = pd.Series(
            np.where(ma20 > ma50, 1, -1), index=df.index
        ).astype(int) * gate

    fast_sig = apply_rsi_entry_filter(fast_sig, df["rsi_14"])
    fast_sig = apply_min_hold_filter(fast_sig, min_hold=3)
    fast_sig = apply_trailing_stop_signal(fast_sig, df["Close"], df["atr_14"], atr_mult=2.0)
    fast_sig = fast_sig * gate
    return fast_sig


def breakout_entry_signal(df: pd.DataFrame, regime_signal: pd.Series) -> pd.Series:
    """
    Momentum breakout entry: price closing at a new 20-day high, emerging from
    a prior volatility squeeze, with above-average volume confirmation.

    Entry conditions (ALL required):
      (a) regime_signal == 1  — MA50/200 golden cross is active (no counter-trend breakouts)
      (b) breakout_20  == 1  — close >= 20-day Donchian high (price leaving consolidation)
      (c) squeeze within last 5 days — breakout from compression, not grinding continuation
      (d) volume_zscore > 0.5 — mild volume confirmation (>~31% threshold, published Lo & Wang 2000)

    Once entered, held until regime_signal exits (death cross) or trailing stop fires.
    This makes it an entry TIMING improvement on the baseline MA crossover — same exit logic.

    References:
      Donchian channel breakout: Turtle Traders (1983).
      Squeeze breakout concept:  Bollinger (2001).
      Volume confirmation:       Lo and Wang (2000, JFE).
    """
    breakout       = df["breakout_20"]
    recent_squeeze = df["squeeze"].rolling(5, min_periods=1).max()
    vol_confirm    = (df["volume_zscore"] > 0.5).astype(int)

    entry = (
        (regime_signal == 1) &
        (breakout == 1) &
        (recent_squeeze == 1) &
        (vol_confirm == 1)
    )

    result = pd.Series(0, index=df.index, dtype=int)
    in_pos = False
    for i in range(len(result)):
        if not in_pos:
            if entry.iloc[i]:
                in_pos         = True
                result.iloc[i] = 1
        else:
            if regime_signal.iloc[i] == 1:
                result.iloc[i] = 1   # hold while regime intact
            else:
                in_pos = False       # regime exited

    return apply_trailing_stop_signal(result, df["Close"], df["atr_14"])


def oversold_bounce_signal(df: pd.DataFrame, regime_signal: pd.Series) -> pd.Series:
    """
    Mean-reversion dip-buy within an established uptrend.

    Entry conditions (ALL required):
      (a) regime_signal == 1   — MA50/200 uptrend active (never catch falling knives)
      (b) rsi_14 < 35          — oversold (slightly above Wilder's standard 30,
                                  capturing ~8% of uptrend days vs ~3% for RSI<30;
                                  structural choice, not optimised on this dataset)
      (c) bb_pct_b < 0.10      — price at or below lower Bollinger Band
                                  (statistically extreme pullback; Bollinger 2001 standard)
      (d) Close > MA200        — uptrend structure intact after the dip
                                  (retracement not reversal)

    Exit conditions:
      rsi_14 > 60              — bounce played out (momentum recovered to midpoint)
      OR regime_signal == 0    — death cross (regime fully exited)
      OR trailing stop fires

    References:
      RSI levels:    Wilder (1978).
      Bollinger %B:  Bollinger (2001).
    """
    ma200 = df["Close"].rolling(200).mean()

    entry = (
        (regime_signal == 1) &
        (df["rsi_14"] < 35) &
        (df["bb_pct_b"] < 0.10) &
        (df["Close"] > ma200)
    )

    result = pd.Series(0, index=df.index, dtype=int)
    in_pos = False
    for i in range(len(result)):
        if not in_pos:
            if entry.iloc[i]:
                in_pos         = True
                result.iloc[i] = 1
        else:
            if regime_signal.iloc[i] == 0:
                in_pos = False          # regime exited
            elif df["rsi_14"].iloc[i] > 60:
                in_pos = False          # bounce played out
                result.iloc[i] = 0
            else:
                result.iloc[i] = 1      # still in bounce trade

    return apply_trailing_stop_signal(result, df["Close"], df["atr_14"])


# ── Cross-asset entry timing cache and helpers ────────────────────────────────

_CROSS_ASSET_CACHE = None


def _load_cross_asset() -> pd.DataFrame:
    """
    Load cross_asset_features.parquet once and cache at module level.

    generate() is called per-ticker (37 tickers), so caching avoids
    reading the same parquet 37 times per pipeline run.
    """
    global _CROSS_ASSET_CACHE
    if _CROSS_ASSET_CACHE is not None:
        return _CROSS_ASSET_CACHE
    _ca_path = Path("data/signals/cross_asset_features.parquet")
    if _ca_path.exists():
        try:
            _CROSS_ASSET_CACHE = pd.read_parquet(_ca_path)
            _CROSS_ASSET_CACHE.index = pd.to_datetime(_CROSS_ASSET_CACHE.index)
        except Exception:
            _CROSS_ASSET_CACHE = pd.DataFrame()
    else:
        _CROSS_ASSET_CACHE = pd.DataFrame()
    return _CROSS_ASSET_CACHE


def cross_asset_entry_boost(df: pd.DataFrame, regime_signal: pd.Series,
                             cross_asset_df: pd.DataFrame = None) -> pd.Series:
    """
    Improve bull_calm entry timing by detecting favourable cross-asset conditions.

    Entry conditions (ALL required):
      (a) regime_signal == 0 but MA50 is within 1% of MA200 from below
          (about to cross — pre-position before the golden cross fires)
      (b) Cross-asset environment is supportive:
          - tlt_spy_divergence < 0  (bonds underperforming stocks = risk-on)
          - bond_equity_beta  < -0.1 (normal negative correlation = no crisis)
          - hy_ig_ratio_zscore > -1.0 (credit not stressed)

    Once entered, held until regime_signal confirms (MA50 crosses MA200)
    or 10 days pass without confirmation (timeout — false pre-signal).

    This is a pure ENTRY timing signal: it adds earlier entries in bull_calm
    uptrends and never overrides exits.  The position is handed off to the
    main regime signal once the golden cross confirms.

    References:
      Entry pre-positioning: Asness et al. (2013) "Value and Momentum Everywhere".
      TLT/SPY divergence as risk-on proxy: Ilmanen (2011) "Expected Returns".
      Bond-equity beta sign flip during crises: Baele et al. (2010).

    Args:
        df:             Feature DataFrame with 'Close' column.
        regime_signal:  The raw MA50/200 trend signal (1 = golden cross active).
        cross_asset_df: Cross-asset features DataFrame.  If None or empty,
                        returns an all-zero Series (graceful degradation).

    Returns:
        Series of {0, 1} aligned to df.index.  1 = pre-position long.
    """
    if cross_asset_df is None or cross_asset_df.empty:
        return pd.Series(0, index=df.index, dtype=int)

    # Align cross-asset features to this ticker's index
    ca = cross_asset_df.reindex(df.index).ffill()

    # Required columns — return zero signal if any missing
    needed = ["tlt_spy_divergence", "bond_equity_beta", "hy_ig_ratio_zscore"]
    if not all(c in ca.columns for c in needed):
        return pd.Series(0, index=df.index, dtype=int)

    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()

    # "Almost golden cross": MA50 is within 1% of MA200 from below
    ma_gap   = (ma50 / ma200 - 1)
    near_cross = (ma_gap > -0.01) & (ma_gap < 0)

    # Cross-asset risk-on environment
    risk_on = (
        (ca["tlt_spy_divergence"] < 0) &
        (ca["bond_equity_beta"]   < -0.1) &
        (ca["hy_ig_ratio_zscore"] > -1.0)
    )

    entry = near_cross & risk_on & (regime_signal == 0)

    result      = pd.Series(0, index=df.index, dtype=int)
    in_pos      = False
    days_waiting = 0

    for i in range(len(result)):
        if not in_pos:
            if entry.iloc[i]:
                in_pos        = True
                days_waiting  = 1
                result.iloc[i] = 1
        else:
            if regime_signal.iloc[i] == 1:
                # Golden cross confirmed — hand off to main signal
                in_pos        = False
                result.iloc[i] = 0   # main signal takes over
            elif days_waiting >= 10:
                # Timeout — cross didn't confirm
                in_pos        = False
                result.iloc[i] = 0
            else:
                days_waiting  += 1
                result.iloc[i] = 1

    return result


def trend_continuation_reentry(
    df: pd.DataFrame,
    signal_r: pd.Series,
    raw_ma_regime: pd.Series,
    macro: pd.DataFrame,
) -> pd.Series:
    """
    Fast re-entry after trailing stop exits during confirmed uptrends.

    After a trailing stop exit, allow re-entry if ALL of these conditions
    are met within 15 trading days (3 weeks):
      (a) raw_ma_regime == 1 — slow MA50/200 trend still intact
      (b) Close > MA50 — price recovered above fast moving average
      (c) RSI < 65 — not re-entering at an overbought extreme
      (d) VIX z-score < 1.5 — not during a fear spike

    This function only ADDs entries (0→1), never removes them (1→0).
    Only fires after trailing stop exits, not death cross exits.
    """
    ma50 = df["Close"].rolling(50).mean()

    if not macro.empty and "vix_zscore" in macro.columns:
        vix_z = macro["vix_zscore"].reindex(df.index).ffill().fillna(0)
        reentry_gate = (vix_z < 1.0).astype(int)
    else:
        reentry_gate = vix_gate(macro, df.index)

    result = signal_r.copy()
    bars_since_stop_exit = -1   # -1 = not in re-entry window
    REENTRY_WINDOW = 15

    for i in range(1, len(result)):
        current = int(result.iloc[i])
        prev    = int(result.iloc[i - 1])

        # Detect trailing stop exit: signal went 1→0 while MA regime is still 1
        # (death cross exits have raw_ma_regime == 0 → don't trigger this)
        if prev == 1 and current == 0 and int(raw_ma_regime.iloc[i]) == 1:
            bars_since_stop_exit = 0

        if 0 <= bars_since_stop_exit < REENTRY_WINDOW:
            bars_since_stop_exit += 1

            if current == 0:
                price    = float(df["Close"].iloc[i])
                ma50_val = float(ma50.iloc[i]) if not pd.isna(ma50.iloc[i]) else 0.0
                rsi_val  = float(df["rsi_14"].iloc[i]) if not pd.isna(df["rsi_14"].iloc[i]) else 50.0
                regime_ok = int(raw_ma_regime.iloc[i]) == 1
                gate_ok   = int(reentry_gate.iloc[i]) == 1

                if (regime_ok and gate_ok and
                        price > ma50_val and ma50_val > 0 and
                        rsi_val < 60):
                    result.iloc[i] = 1
                    bars_since_stop_exit = -1  # re-entry complete

        elif bars_since_stop_exit >= REENTRY_WINDOW:
            bars_since_stop_exit = -1  # window expired

    return result


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
          signal_multi,
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
    if asset_class in {"equity_index", "sector_etf", "stock"}:
        # Long-only: equities have structural upward drift (equity risk premium).
        # MA golden cross (MA50 > MA200) avoids noisy short-term momentum flips.
        # In flat/down regimes: be in cash, not short.
        ma50  = df["Close"].rolling(50).mean()
        ma200 = df["Close"].rolling(200).mean()
        # raw_ma_regime: the unfiltered MA50/200 trend signal (VIX gate only).
        # Kept separate so breakout_entry_signal and oversold_bounce_signal
        # can use it as a regime gate — they should fire whenever the trend is
        # structurally up (MA50 > MA200), even if the trailing stop or RSI
        # filter has temporarily put signal_r to 0.
        raw_ma_regime = ((ma50 > ma200).astype(int) * regime_gate).astype(int)
        signal_r = raw_ma_regime.copy()

        # ── Principled signal improvements (not curve-fitted) ───────────────
        signal_r = apply_rsi_entry_filter(signal_r, df["rsi_14"])
        signal_r = apply_min_hold_filter(signal_r)
        signal_r = apply_trailing_stop_signal(signal_r, df["Close"], df["atr_14"], macro=macro)
        signal_r = apply_time_decay_exit(signal_r, df["Close"])
        signal_r = signal_r * regime_gate  # re-apply gate after post-processing

        # Fast re-entry after trailing stop exits in confirmed uptrends
        signal_r = trend_continuation_reentry(df, signal_r, raw_ma_regime, macro)

    elif asset_class == "commodity":
        # Two-sided MA50/200 crossover. Commodities have no structural upward
        # drift (no equity risk premium), so death-cross periods are shorted
        # rather than held as cash. MA50/200 generates ~1300 long + ~1200 short
        # days vs momentum_rule()'s 3-way consensus which barely fires.
        # commodity_regime() is still used for composite score blending below.
        ma50  = df["Close"].rolling(50).mean()
        ma200 = df["Close"].rolling(200).mean()
        signal_r = pd.Series(
            np.where(ma50 > ma200, 1, -1),
            index=df.index,
        ).astype(int) * regime_gate
        signal_r = apply_rsi_entry_filter(signal_r, df["rsi_14"])
        signal_r = apply_min_hold_filter(signal_r)
        signal_r = apply_trailing_stop_signal(signal_r, df["Close"], df["atr_14"])
        signal_r = apply_time_decay_exit(signal_r, df["Close"])
        signal_r = signal_r * regime_gate

    else:
        # Bonds: two-sided momentum. Rate cycles genuinely go both ways for years.
        signal_r = pd.Series(np.where(regime == 1, mom, rev), index=df.index) * regime_gate

    out["signal_regime"] = signal_r

    # ── Multi-signal overlay (equity only) ─────────────────────────────────
    # breakout_entry_signal and oversold_bounce_signal are ADDITIONAL entry
    # timers on top of the MA crossover for equity/sector/stock asset classes.
    # Bonds and commodities use the regime signal directly — their edge is in
    # direction (two-sided trend following), not entry timing.
    if asset_class in {"equity_index", "sector_etf", "stock"}:
        # Use raw_ma_regime (not post-processed signal_r) as the regime gate.
        # This allows breakout/bounce/persistence to activate in windows where
        # signal_r is temporarily flat but the structural trend is still intact.
        breakout_sig = breakout_entry_signal(df, raw_ma_regime)
        bounce_sig   = oversold_bounce_signal(df, raw_ma_regime)

        _multi_sources = [signal_r, breakout_sig, bounce_sig]

        # Cross-asset early entry: pre-position when MA50 is approaching MA200
        # from below and cross-asset conditions signal a risk-on environment.
        _ca_df = _load_cross_asset()
        if not _ca_df.empty:
            ca_boost = cross_asset_entry_boost(df, raw_ma_regime, _ca_df)
            _multi_sources.append(ca_boost)

        # max() → long if ANY source says so. Only adds entries, never exits.
        signal_multi = pd.concat(_multi_sources, axis=1).max(axis=1).astype(int)
    else:
        signal_multi = signal_r   # bonds/commodities: no change

    out["signal_multi"] = signal_multi

    # ── Fast MA20/50 overlay (FAST_SIGNAL_TICKERS only) ─────────────────────
    # For the 5 most liquid ETFs, blend a faster MA20/50 signal with the slow
    # MA50/200 signal as an equal-weight average.  The average lives in [−1,+1]
    # for two-sided tickers or [0,+1] for equity_index — a continuous position
    # fraction rather than a binary.  atr_sizes() handles continuous inputs
    # identically to binary ones (linear scaling).
    if ticker in FAST_SIGNAL_TICKERS:
        fast_sig = fast_signal(df, ticker, macro)
        out["signal_fast_overlay"] = ((signal_r.astype(float) + fast_sig.astype(float)) / 2.0)
        out["signal_multi_fast"]   = ((signal_multi.astype(float) + fast_sig.astype(float)) / 2.0)
    else:
        out["signal_fast_overlay"] = signal_r.astype(float)
        out["signal_multi_fast"]   = signal_multi.astype(float)

    scores = compute_scores(df, regime, ticker)
    out    = pd.concat([out, scores], axis=1)

    raw_composite = scores_to_signal(scores["score_composite"])
    if asset_class in {"equity_index", "sector_etf", "stock"}:
        # Long-only for equities: convert short signals to flat (cash).
        raw_composite = raw_composite.clip(lower=0)
    # bonds and commodities pass through both long and short composite signals
    out["signal_composite"] = raw_composite * composite_gate

    # ── Ensemble signal: continuous [-1, +1], IC-weighted, adaptive ────────
    ens = ensemble_signal(df) * gate
    if asset_class in {"equity_index", "sector_etf", "stock"}:
        # Long-only for equities: clip short side to flat.
        ens = ens.clip(lower=0)
    # bonds and commodities pass through full [-1, +1] range
    out["signal_ensemble"] = ens

    # ── Insider sizing modifier ──────────────────────────────────────────
    # Placeholder: 1.0 = no adjustment. Upstream logic (e.g. insider_signals
    # pipeline) can set to 1.3 for tickers with recent insider buying.
    # Consumed by portfolio.py to scale position sizes.
    out["insider_size_mult"] = 1.0

    # ── Carry signal (orthogonal alpha — bond/commodity curve slope) ─────
    # carry_signal.py runs AFTER signal_generation in the pipeline, so
    # carry_signals.parquet may not exist on the very first run.
    # On subsequent runs the file is present and the column is populated.
    # Equity tickers always receive 0 (carry requires fundamental data).
    _carry_path = SIGNAL_DIR / "carry_signals.parquet"
    if _carry_path.exists():
        try:
            _carry_df = pd.read_parquet(_carry_path)
            if ticker in _carry_df.columns:
                out["signal_carry"] = (
                    _carry_df[ticker].reindex(out.index).ffill().fillna(0.0)
                )
            else:
                out["signal_carry"] = 0.0
        except Exception:
            out["signal_carry"] = 0.0
    else:
        out["signal_carry"] = 0.0

    return out.dropna()

# -----------------------------------------------------------------------------
# Pair-trade signal generation
# -----------------------------------------------------------------------------

def generate_pair_signals(all_signals: dict) -> tuple:
    """
    Derive beta-hedged pair signals for stocks in HEDGE_MAP.

    Logic per stock
    ───────────────
    spread        = log(stock_close / hedge_close)   — relative performance
    spread_MA50   = 50-day MA of spread
    spread_MA200  = 200-day MA of spread
    spread_signal = 1 when spread_MA50 > spread_MA200 (stock outperforming
                    its sector — the pair trade has positive carry/momentum)

    pair_signal       = signal_regime  × spread_signal
    multi_pair_signal = signal_multi   × spread_signal

    A value of 1 means: the stock is in an MA50/200 uptrend (signal_regime)
    AND it is outperforming its sector ETF on a 50/200 spread basis.
    Both conditions must hold simultaneously — if either fails, the pair
    is flat (0).  This avoids longs in stocks that are rising purely
    because the whole sector is rising (no alpha) and avoids longs in
    outperformers that are in structural downtrends.

    For tickers NOT in HEDGE_MAP (ETFs, bonds, commodities, AMZN/etc.):
    pair_signal = signal_regime  (unchanged — no pair filter applied).

    Same MA windows (50/200) as the main equity signal — no new parameters.

    Args:
        all_signals: Dict[ticker -> signal DataFrame] from the main signal loop.
                     Each DataFrame must contain 'signal_regime' and 'signal_multi'.

    Returns:
        (pair_df, multi_pair_df): two DataFrames of shape (dates, tickers),
        with the same column set as the input signals.
    """
    pair_dict       = {}
    multi_pair_dict = {}

    for ticker, sig in all_signals.items():
        if ticker not in HEDGE_MAP:
            # Not a pair-trade candidate — use unfiltered signal as-is.
            pair_dict[ticker]       = sig["signal_regime"]
            multi_pair_dict[ticker] = sig["signal_multi"]
            continue

        hedge = HEDGE_MAP[ticker]
        hedge_path = FEATURE_DIR / f"{hedge}.parquet"

        if not hedge_path.exists():
            # Hedge feature file missing (e.g. SPY not yet engineered).
            # Fall back to unfiltered signal so we don't silently drop the ticker.
            pair_dict[ticker]       = sig["signal_regime"]
            multi_pair_dict[ticker] = sig["signal_multi"]
            continue

        hedge_df    = pd.read_parquet(hedge_path)
        stock_close = sig["Close"]
        hedge_close = hedge_df["Close"].reindex(sig.index).ffill()

        # Spread in log-space: positive drift = stock outperforms sector.
        spread       = np.log((stock_close / hedge_close).replace(0, np.nan))
        spread_ma50  = spread.rolling(50, min_periods=25).mean()
        spread_ma200 = spread.rolling(200, min_periods=100).mean()
        spread_signal = (spread_ma50 > spread_ma200).astype(int)

        # Gate: only enter pair trade when BOTH the stock is in an uptrend
        # AND the stock is outperforming its sector on a 50/200 basis.
        pair_dict[ticker] = (
            sig["signal_regime"] * spread_signal
        ).reindex(sig.index).fillna(0).astype(int)

        multi_pair_dict[ticker] = (
            sig["signal_multi"] * spread_signal
        ).reindex(sig.index).fillna(0).astype(int)

    pair_df       = pd.DataFrame(pair_dict).dropna()
    multi_pair_df = pd.DataFrame(multi_pair_dict).dropna()
    return pair_df, multi_pair_df


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

    regime       = pd.DataFrame({t: s["signal_regime"]        for t, s in all_signals.items()}).dropna()
    composite    = pd.DataFrame({t: s["signal_composite"]     for t, s in all_signals.items()}).dropna()
    ensemble     = pd.DataFrame({t: s["signal_ensemble"]      for t, s in all_signals.items()}).dropna()
    multi        = pd.DataFrame({t: s["signal_multi"]         for t, s in all_signals.items()}).dropna()
    fast_overlay = pd.DataFrame({t: s["signal_fast_overlay"]  for t, s in all_signals.items()}).dropna()
    multi_fast   = pd.DataFrame({t: s["signal_multi_fast"]    for t, s in all_signals.items()}).dropna()

    regime.to_parquet(SIGNAL_DIR        / "regime_signals.parquet")
    composite.to_parquet(SIGNAL_DIR     / "composite_signals.parquet")
    ensemble.to_parquet(SIGNAL_DIR      / "ensemble_signals.parquet")
    multi.to_parquet(SIGNAL_DIR         / "multi_signals.parquet")
    fast_overlay.to_parquet(SIGNAL_DIR  / "fast_overlay_signals.parquet")
    multi_fast.to_parquet(SIGNAL_DIR    / "multi_fast_signals.parquet")
    print(f"Signal matrices saved: {regime.shape}")

    # Save carry signal matrix if carry was loaded into any ticker
    _carry_cols = {t: s["signal_carry"]
                   for t, s in all_signals.items()
                   if "signal_carry" in s.columns}
    if _carry_cols:
        _carry_matrix = pd.DataFrame(_carry_cols).dropna()
        _carry_matrix.to_parquet(SIGNAL_DIR / "carry_signals_matrix.parquet")
        print(f"  Carry signal matrix saved: {_carry_matrix.shape}")

    # Print fast overlay stats for the 5 targeted ETFs
    print("\n  Fast MA20/50 overlay — active fraction vs slow MA50/200:")
    for t in sorted(FAST_SIGNAL_TICKERS):
        if t not in all_signals:
            continue
        s = all_signals[t]
        slow_active = (s["signal_regime"].abs() > 0).mean() * 100
        fast_active = (fast_overlay[t].abs() > 0.01).mean() * 100 if t in fast_overlay.columns else 0
        avg_val     = fast_overlay[t].mean() if t in fast_overlay.columns else 0
        print(f"    {t:<5}: slow_regime {slow_active:.0f}% active  fast_overlay mean={avg_val:.2f}")

    # Print extra stats for the multi-signal overlay (includes re-entry + breakout + bounce)
    print("\n  signal_multi vs signal_regime comparison (equity tickers):")
    total_extra = 0
    for t, s in all_signals.items():
        if ASSET_CLASS[t] in {"equity_index", "sector_etf", "stock"}:
            r = int((s["signal_regime"] == 1).sum())
            m = int((s["signal_multi"]  == 1).sum())
            extra = m - r
            total_extra += extra
            print(f"    {t:<8}: regime {r:>4} long days  multi {m:>4} long days  "
                  f"(+{extra} from breakout/bounce/reentry)")
    print(f"\n  Total extra long days added across equity tickers: +{total_extra}")
    print(f"  (target: 200-600 extra days fills bull_calm cash drag gaps)")

    # ── Pair-trade signal generation ──────────────────────────────────────
    print("\n  Generating pair-trade signals (spread MA50/200 filter)...")
    pair_df, multi_pair_df = generate_pair_signals(all_signals)
    pair_df.to_parquet(SIGNAL_DIR       / "pair_signals.parquet")
    multi_pair_df.to_parquet(SIGNAL_DIR / "multi_pair_signals.parquet")
    print(f"  Pair signal matrices saved: {pair_df.shape}")

    print("\n  signal_pair vs signal_regime comparison (HEDGE_MAP tickers):")
    for t in HEDGE_MAP:
        if t in pair_df.columns and t in regime.columns:
            r  = int((regime[t]   == 1).sum())
            p  = int((pair_df[t]  == 1).sum())
            mp = int((multi_pair_df[t] == 1).sum()) if t in multi_pair_df.columns else 0
            hedge = HEDGE_MAP[t]
            print(f"    {t:<6} -> {hedge:<5}: regime {r:>4}  pair {p:>4}  multi_pair {mp:>4}"
                  f"  (spread filter removed {r-p:>3} days)")

    # ── Earnings blackout filtered signals (stocks only) ──────────────────
    earn_path = Path("data/raw/earnings_dates.json")
    if earn_path.exists():
        with open(earn_path) as fh:
            earnings_map = json.load(fh)

        earn_multi_dict      = {}
        earn_multi_fast_dict = {}
        days_removed         = {}

        for t, s in all_signals.items():
            earn_dates = earnings_map.get(t, []) if ASSET_CLASS[t] == "stock" else []
            base_multi      = s["signal_multi"].astype(int)
            base_multi_fast = s["signal_multi_fast"].astype(float)
            if earn_dates:
                filtered_multi      = apply_earnings_blackout(base_multi,      earn_dates)
                filtered_multi_fast = apply_earnings_blackout(base_multi_fast,  earn_dates)
                days_removed[t] = int((base_multi - filtered_multi).abs().sum())
            else:
                filtered_multi      = base_multi
                filtered_multi_fast = base_multi_fast
            earn_multi_dict[t]      = filtered_multi
            earn_multi_fast_dict[t] = filtered_multi_fast

        earn_multi      = pd.DataFrame(earn_multi_dict).dropna()
        earn_multi_fast = pd.DataFrame(earn_multi_fast_dict).dropna()
        earn_multi.to_parquet(SIGNAL_DIR      / "earn_multi_signals.parquet")
        earn_multi_fast.to_parquet(SIGNAL_DIR / "earn_multi_fast_signals.parquet")
        print(f"\n  Earnings-filtered signal matrices saved: {earn_multi.shape}")
        print("  Trading days removed per stock ticker (2-before/1-after per earnings date):")
        for t, n in sorted(days_removed.items()):
            n_dates = len(earnings_map.get(t, []))
            print(f"    {t:<8} {n:>4} days removed  ({n_dates} earnings dates × ~3 days)")
    else:
        print("\n  NOTE: data/raw/earnings_dates.json not found — skipping earnings filter")


if __name__ == "__main__":
    main()
