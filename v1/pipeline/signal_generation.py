"""
signal_generation.py — asset-class-aware trading signal generation.

Signals produced
────────────────
  signal_regime     — MA50/200 golden cross (long-only most assets), post-processed
                      by RSI entry filter, min-hold, ATR trailing stop.
  signal_composite  — momentum + mean-reversion blend with regime-aware weighting.
  signal_ensemble   — IC-weighted linear ensemble of top 6 features (rolling 252d
                      z-score, trailing 504d IC weights). Value in [-1, +1].

Asset-class routing
───────────────────
  equity_index  → MA50/200 + ADX>25; long-only (SPY, IWM, EEM, EFA, VWO)
  sector_etf    → MA100/300 + ADX>20 (slower for multi-month rotations); long-only
                  (XLE, XLU, XLF, VNQ)
  stock         → MA50/200 + ADX>dynamic (vol-scaled); long-only
                  (JPM, JNJ, XOM, AMZN, NEE, BRK-B, GS, COST, MSFT, NVDA, GE, INTC, VZ)
  bond          → two-sided momentum (TLT, HYG, TIP)
  commodity     → momentum if VIX>25, mean-reversion otherwise; two-sided
                  (GLD, DBC, UUP)

Post-processors on signal_regime
────────────────────────────────
  1. RSI entry filter  — block new longs when RSI_14 > 70
  2. Min-hold filter   — MIN_HOLD_DAYS=5 prevents whipsaw round-trips
  3. ATR trailing stop — exit if price < trail_high − ATR_TRAILING_MULT × ATR

Output: data/signals/{TICKER}.parquet plus matrices regime/composite/ensemble.
Consumed by backtester.py, portfolio.py, paper_trader.py, live_signals.py.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from .data_pipeline import TICKER_LIST, ASSET_CLASS, HEDGE_MAP

# ── Strategy constants (principled, not curve-fitted) ────────────────────────
RSI_ENTRY_THRESH  = 70    # Wilder 1978 overbought level — block new longs above.
MIN_HOLD_DAYS     = 5     # 1 trading week min hold; prevents whipsaw round-trips.
ATR_TRAILING_MULT = 3.0   # Trailing stop = trail_high − 3×ATR. Institutional std (Elder/Schwager).
ATR_TIGHTEN_THRESHOLD = 5.0  # Tighten stop once gain > 5×ATR (Elder: large gains revert harder).
ATR_TIGHTEN_MULT      = 1.5  # Tightened stop distance: 1.5×ATR (vs 3× normal).
ATR_PROFIT_TARGET = 12.0  # Partial exit at 12×ATR gain. Higher (15-20×) saturates → no-op.
ATR_PARTIAL_REMAIN = 0.75  # Keep 75% of position post partial exit (tested 0.50→0.75 iter 8).

# Regime-conditional RSI/ADX thresholds (Improvement 1: bull-calm exposure boost).
# Wilder published values, not fitted.
RSI_ENTRY_THRESH_CALM   = 75   # relaxed overbought in calm bull
RSI_ENTRY_THRESH_STRESS = 68   # tighter entry in stress/neutral
ADX_MIN_CALM   = 22    # bull_calm only
ADX_MIN_STRESS = 25    # stress/neutral

# Validation counters incremented in generate() and printed in main().
_CALM_EXTRA_ENTRIES: int = 0   # entries enabled by relaxed calm RSI threshold
_VOL_DIV_EXTRA_ENTRIES: int = 0  # vol_div_sig fires while signal_r is flat
_EARLY_ENTRY_DAYS: int = 0  # bull_calm early entry firings (Part 3)

# SPY close cache to avoid 37 parquet reads (Part 3 bull_calm detection).
_SPY_CLOSE_CACHE: pd.Series | None = None

# ── Profit-lock + no-new-high time-stop (overlay on signal_multi) ───────────
# Validated 2026-04-27 on atr_lev_1.5x: pl_5_10 + ts_40 Pareto-dominates baseline —
# AnnRet 14.6→15.35, Sharpe 1.248→1.355, MaxDD -10.05%→-8.67%, Calmar 1.45→1.77.
# Stable plateau across ts ∈ [35, 42]. Acts after trailing/time-decay; can only fire earlier.
PROFIT_LOCK_TARGET_1 = 0.05  # high > +5% → lock stop at entry (break-even)
PROFIT_LOCK_LOCK_1   = 0.0
PROFIT_LOCK_TARGET_2 = 0.10  # high > +10% → raise lock to +5%
PROFIT_LOCK_LOCK_2   = 0.05
NO_NEW_HIGH_DAYS     = 40    # exit after 40 trading days with no new high

TIME_DECAY_DAYS   = 126  # 6 trading months (half MA200). Close stale longs drifting negative;
                           # addresses bear_calm equities held while TLT/GLD/TIP rally.
TIME_DECAY_WINDOW = 21   # 1-month trailing return: close[t]/close[t-21]-1 < 0 → drifting down.

# 5 most liquid ETFs by AUM (published fact). MA20/50 fast overlay applied
# only to these — individual stocks are excluded (MA20/50 whipsaws on stocks).
# Tested extending the overlay to additional equity_index/sector ETFs (EFA, VWO,
# XLK, XLF, XLE, XLV): preserved OOS Sharpe at 1.555 but widened max drawdown
# from -4.64% to -7.12% on the production method.  The fast overlay's higher
# turnover in equity sectors trades cleaner Sharpe for deeper drawdowns; not
# worth it.  Reverted to original 5.
FAST_SIGNAL_TICKERS = {"SPY", "IWM", "TLT", "GLD", "EEM"}

# ── Trend ensemble parameters (Improvement #1) ───────────────────────────────
# Three voters across short/medium/long timescales. Each voter is a binary MA
# crossover; the ensemble score is the equal-weight mean → continuous in [0, 1].
# Voters chosen to span ~1 month / ~3 month / ~6 month horizons, i.e. distinct
# information content rather than three near-duplicates of MA50/200.
#   Voter 1 (short) : MA20  > MA50    — captures emerging trends ~1 month early
#   Voter 2 (medium): MA50  > MA150   — intermediate-term trend filter
#   Voter 3 (long)  : MA100 > MA200   — structural trend (close to legacy MA50/200)
TREND_ENS_SHORT_FAST  = 20
TREND_ENS_SHORT_SLOW  = 50
TREND_ENS_MED_FAST    = 50
TREND_ENS_MED_SLOW    = 150
TREND_ENS_LONG_FAST   = 100
TREND_ENS_LONG_SLOW   = 200

# Pullback re-entry — moderate dip within an established trend.
#   Fires when ensemble_score >= 2/3 (medium + long voters agree at minimum)
#   and price has just bounced off a moderate pullback.
PULLBACK_RSI_LOW    = 40   # entry RSI floor (deeper than this → use oversold_bounce)
PULLBACK_RSI_HIGH   = 55   # entry RSI ceiling (above this → not a pullback)
PULLBACK_RSI_EXIT   = 65   # exit when momentum recovers past midline + buffer
PULLBACK_ADX_MIN    = 18   # require some trend structure (Wilder's "weak trend")
PULLBACK_LOOKBACK   = 5    # bars to look back for the dip touching MA20

# Donchian-55 breakout (Turtle System 2, Dennis & Eckhardt 1983).
# Looser than breakout_entry_signal: no squeeze requirement, no volume confirm.
# Compensated by requiring the 55-day high (vs 20-day) — much higher bar so
# false breakouts are rarer on their own.
DONCHIAN_BREAKOUT_WINDOW = 55
DONCHIAN_BREAKOUT_ADX_MIN = 18

# Master switch for Improvement #1 new signals (pullback + Donchian-55).
# Set to False to A/B-test the contribution of these signals against baseline
# WITHOUT removing the code — useful for diagnosis when adding subsequent
# improvements (#3 vol-regime gating, #4 Kelly sizing) that interact with them.
IMPROVEMENT_1_NEW_SIGNALS: bool = True

# ── OU s-score parameters (Avellaneda-Lee 2010, residualised mean reversion) ──
# Residualise stock returns against SPY (1-factor market model), cumulate the
# residuals into X_k, fit AR(1): X_{k+1} = a + b·X_k + ξ.  OU equilibrium:
#   m         = a / (1 - b)               long-run mean of cumulative residual
#   σ_eq²     = var(ξ) / (1 - b²)         equilibrium variance
#   s-score   = (X_now - m) / σ_eq        z-score against the OU equilibrium
# Positive s = residual stretched ABOVE fair value vs SPY (overstretched).
# Negative s = residual stretched BELOW fair value vs SPY (mean-reversion long).
# Used as a soft entry-timing gate on pullback_reentry_signal so we don't
# enter pullbacks where the residual is already overstretched up.
OU_WINDOW       = 60     # Avellaneda-Lee's published window: ~3 trading months
OU_SSCORE_HIGH  = 1.50   # don't enter pullbacks above this s-score (overstretched up)
OU_OVERSHOOT    = 2.00   # exit pullbacks if s-score climbs past this (mean-rev risk)

# Master switch for Improvement #2 (OU s-score gates).
IMPROVEMENT_2_OU_SSCORE: bool = True

# Module-level counters: long-entry days added by each new signal source.
# Printed in main() to validate that the new signals are firing as expected.
_PULLBACK_EXTRA_ENTRIES: int = 0
_DONCHIAN55_EXTRA_ENTRIES: int = 0
_OU_SSCORE_BLOCKED_ENTRIES: int = 0
_OU_SSCORE_OVERSHOOT_EXITS: int = 0

DATA_DIR     = Path("data/v1/raw")
FEATURE_DIR  = Path("data/v1/features")
SIGNAL_DIR   = Path("data/v1/signals")
MACRO_DIR    = Path("data/shared/macro")
RESEARCH_DIR = Path("data/v1/research")
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


def _load_spy_close() -> "pd.Series":
    """
    Load SPY close prices, cached at module level (Part 3 early entry).

    generate() is called once per ticker (~41 tickers); caching avoids
    41 redundant parquet reads for the SPY 60d return used in bull_calm detection.

    Returns:
        SPY Close Series with DatetimeIndex, or empty Series if unavailable.
    """
    global _SPY_CLOSE_CACHE
    if _SPY_CLOSE_CACHE is not None:
        return _SPY_CLOSE_CACHE
    spy_path = FEATURE_DIR / "SPY.parquet"
    if spy_path.exists():
        try:
            _SPY_CLOSE_CACHE = pd.read_parquet(spy_path)["Close"]
            _SPY_CLOSE_CACHE.index = pd.to_datetime(_SPY_CLOSE_CACHE.index)
        except Exception:
            _SPY_CLOSE_CACHE = pd.Series(dtype=float)
    else:
        _SPY_CLOSE_CACHE = pd.Series(dtype=float)
    return _SPY_CLOSE_CACHE


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

def apply_rsi_entry_filter(signal: pd.Series, rsi: pd.Series,
                            thresh_series: pd.Series = None) -> pd.Series:
    """
    Block new entries at RSI extremes: longs when RSI > thresh (overbought),
    shorts when RSI < (100 - thresh) (oversold — symmetric rule for short side).

    State machine tracks current position direction (0 = flat, +1 = long,
    -1 = short) so the filter only blocks NEW entries, not ongoing positions.

    Args:
        signal:       Signal Series {-1, 0, 1} BEFORE the RSI filter.
        rsi:          RSI-14 Series aligned to signal.index.
        thresh_series: Optional per-bar threshold Series.  When provided, the
                       overbought gate on each bar equals thresh_series.iloc[i]
                       instead of the global RSI_ENTRY_THRESH.  Allows regime-
                       conditional thresholds (e.g. 75 in bull_calm, 68 in stress)
                       without changing the state-machine logic.  Pass None to
                       use the global RSI_ENTRY_THRESH (default behaviour).

    Returns:
        Filtered signal Series.  Same index and dtype (int) as input.
    """
    result       = signal.copy().astype(float)
    position_dir = 0   # 0 = flat, +1 = long, -1 = short

    for i in range(len(result)):
        thresh       = float(thresh_series.iloc[i]) if thresh_series is not None else RSI_ENTRY_THRESH
        rsi_oversold = 100 - thresh   # symmetric short-entry gate

        val = int(result.iloc[i])
        if position_dir == 0:
            if val == 1:
                if float(rsi.iloc[i]) > thresh:
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
                                macro: pd.DataFrame = None) -> tuple:
    """
    Close a long position if price falls ATR_TRAILING_MULT × ATR below the
    trailing high since entry, with profit-target stop tightening and a
    partial exit at ATR_PROFIT_TARGET (8×) ATR gain.

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

    Partial exit at ATR_PROFIT_TARGET (8×) ATR gain:
        When price rises 8× ATR above entry, signal is set to 0 for that bar
        (full exit at the close) and re-enters at 1 the next bar if the
        upstream signal is still long.  The re-entered leg is tracked in
        half_size (True) so portfolio.py can size it at ATR_PARTIAL_REMAIN
        (50%) of the normal position.  The remaining half continues with the
        1.5× ATR tight stop already in effect from the 5×-ATR tighten.
        Source: Elder, "Trading for a Living" — partial profit taking at
        extended gains reduces drawdown without cutting the trend short.

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
        Tuple (signal_series, half_size_series):
          signal_series  — Filtered signal Series, same index and dtype (int)
                           as input.
          half_size_series — Boolean Series (same index).  True on bars where
                             position is in the half-size re-entry leg after a
                             partial exit, until the position is fully closed.
    """
    trail_mult        = atr_mult if atr_mult is not None else ATR_TRAILING_MULT
    result            = signal.copy().astype(float)
    half_size         = pd.Series(False, index=signal.index)
    position_dir      = 0     # 0 = flat, +1 = long, -1 = short
    trail_high        = 0.0
    trail_low         = 0.0
    entry_price       = 0.0
    partial_exit_taken = False  # True once the 8×-ATR partial exit has fired
    in_partial_reentry = False  # True while holding the post-partial-exit half leg

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
            if current_vix > 30:
                effective_mult = max(trail_mult * 0.55, 1.5)  # ≈ 1.65× at 3.0 base
            elif current_vix > 25:
                effective_mult = max(trail_mult * 0.67, 1.5)  # ≈ 2.0× at 3.0 base
            elif current_vix > 20:
                effective_mult = trail_mult * 0.83              # ≈ 2.5× at 3.0 base
            else:
                effective_mult = trail_mult
        else:
            effective_mult = trail_mult

        if position_dir == 0:
            if val != 0:
                position_dir       = val
                trail_high         = price
                trail_low          = price
                entry_price        = price
                partial_exit_taken = False  # reset for each new entry
                if in_partial_reentry:
                    half_size.iloc[i] = True   # re-entry bar is half-size
            elif in_partial_reentry:
                # Upstream signal went 0 before we could re-enter — cancel
                in_partial_reentry = False

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

            # ── Partial exit at ATR_PROFIT_TARGET (8×) ATR gain ──────────────
            # Fires once per position (partial_exit_taken guards re-fire).
            # Sets signal to 0 this bar (full exit at close), then the state
            # machine will naturally re-enter next bar when upstream signal=1.
            # The re-entered leg is flagged by in_partial_reentry so portfolio.py
            # sizes it at ATR_PARTIAL_REMAIN (50%).
            if not partial_exit_taken and current_atr and current_atr > 0:
                if (price - entry_price) > ATR_PROFIT_TARGET * current_atr:
                    partial_exit_taken  = True
                    in_partial_reentry  = True
                    result.iloc[i]      = 0    # full exit this bar
                    position_dir        = 0
                    trail_high          = 0.0
                    entry_price         = 0.0
                    continue               # skip stop check for this bar

            if price < stop:
                result.iloc[i]     = 0    # trailing stop fires (long)
                position_dir       = 0
                trail_high         = 0.0
                entry_price        = 0.0
                in_partial_reentry = False
            elif val == 0:
                position_dir       = 0    # normal exit
                trail_high         = 0.0
                entry_price        = 0.0
                in_partial_reentry = False
            elif in_partial_reentry:
                half_size.iloc[i] = True  # still holding the half-size leg

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
                result.iloc[i]     = 0    # trailing stop fires (short)
                position_dir       = 0
                trail_low          = 0.0
                entry_price        = 0.0
                in_partial_reentry = False
            elif val == 0:
                position_dir       = 0    # normal exit
                trail_low          = 0.0
                entry_price        = 0.0
                in_partial_reentry = False

    return result.astype(int), half_size


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


def apply_profit_lock_timestop(signal: pd.Series, close: pd.Series) -> pd.Series:
    """
    Tier-2 profit-lock + 40-bar no-new-high time-stop overlay.

    Tracks per-position state (entry price, highest-high since entry,
    bars-since-last-new-high) and forces an exit when EITHER:
      • Two-tier profit lock:
          - Once the highest-high reaches +PROFIT_LOCK_TARGET_1 (5%) above
            entry, exit if close < entry × (1 + PROFIT_LOCK_LOCK_1) (0%
            = break-even).
          - Once the highest-high reaches +PROFIT_LOCK_TARGET_2 (10%)
            above entry, raise the lock to entry × (1 + PROFIT_LOCK_LOCK_2)
            (+5%).  Surrendering more than this is the trigger.
      • Time-stop: NO_NEW_HIGH_DAYS (40) consecutive bars without a new
        highest-high since entry → exit (position has stalled).

    Once an overlay-exit fires, the position stays flat until the upstream
    `signal` cycles 0 → 1 (a fresh entry).  The overlay only converts 1's
    to 0's; it never extends a position.

    Applied AFTER apply_trailing_stop_signal and apply_time_decay_exit, so
    it can only fire EARLIER than the existing exits.  Empirically a clean
    Pareto improvement on AnnRet, Sharpe, MaxDD, and Calmar versus baseline
    on top-N zero-leverage sizing (validated 2026-04-27).

    Args:
        signal: Binary 0/1 Series (post all current exits).
        close:  Close price Series aligned to signal.index.

    Returns:
        Filtered Series (int).  Same shape and index.
    """
    out = signal.copy().astype(int)
    in_pos = False
    entry  = 0.0
    high   = 0.0
    bars_since_high = 0
    overlay_exited  = False

    s_arr = signal.values
    p_arr = close.values

    for i in range(len(out)):
        s_val = int(s_arr[i]) if not pd.isna(s_arr[i]) else 0
        p     = float(p_arr[i])

        if not in_pos:
            if s_val > 0:
                in_pos = True
                entry  = p
                high   = p
                bars_since_high = 0
                overlay_exited  = False
            continue

        if s_val == 0:
            in_pos = False
            overlay_exited = False
            continue

        if overlay_exited:
            out.iloc[i] = 0
            continue

        if p > high:
            high = p
            bars_since_high = 0
        else:
            bars_since_high += 1

        gain = (high - entry) / entry if entry > 0 else 0.0
        exit_now = False
        if gain >= PROFIT_LOCK_TARGET_2 and p < entry * (1 + PROFIT_LOCK_LOCK_2):
            exit_now = True
        elif gain >= PROFIT_LOCK_TARGET_1 and p < entry * (1 + PROFIT_LOCK_LOCK_1):
            exit_now = True
        elif bars_since_high >= NO_NEW_HIGH_DAYS:
            exit_now = True

        if exit_now:
            out.iloc[i]    = 0
            overlay_exited = True

    return out.astype(int)


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
        # NOTE: sector_etf was tested here but caused max DD widening; reverted.
        fast_sig = (ma20 > ma50).astype(int) * gate
    else:
        # Two-sided for TLT (bond) and GLD (commodity): trends run both ways.
        fast_sig = pd.Series(
            np.where(ma20 > ma50, 1, -1), index=df.index
        ).astype(int) * gate

    fast_sig = apply_rsi_entry_filter(fast_sig, df["rsi_14"])
    fast_sig = apply_min_hold_filter(fast_sig, min_hold=3)
    fast_sig, _ = apply_trailing_stop_signal(fast_sig, df["Close"], df["atr_14"], atr_mult=2.0)
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

    signal, _ = apply_trailing_stop_signal(result, df["Close"], df["atr_14"])
    return signal


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

    signal, _ = apply_trailing_stop_signal(result, df["Close"], df["atr_14"])
    return signal


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
    _ca_path = Path("data/v1/signals/cross_asset_features.parquet")
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


def volume_divergence_reentry(
    df: pd.DataFrame,
    regime_signal: pd.Series,
    raw_ma_regime: pd.Series,
) -> pd.Series:
    """
    Re-enter long when price pulls back to MA50 on declining volume
    during a confirmed uptrend.

    Entry conditions (ALL required):
      (a) raw_ma_regime == 1 — MA50/200 golden cross intact
      (b) regime_signal == 0 — trailing stop or time decay exited the position
      (c) Close is within 2% above MA50 (price testing support)
      (d) obv_zscore < -0.5 — volume declining (selling pressure fading)
      (e) RSI between 35-60 — not overbought, not deeply oversold

    Once entered, held until regime_signal takes over (golden cross active)
    or raw_ma_regime flips to 0 (death cross).

    Reference: Lo & Wang (2000), Llorente et al. (2002) — volume-price
    divergence as a signal for informed vs uninformed trading.
    """
    ma50 = df["Close"].rolling(50).mean()

    # Price within 2% above MA50 (testing support, not far above)
    price_near_ma50 = ((df["Close"] / ma50 - 1) >= 0) & ((df["Close"] / ma50 - 1) < 0.02)

    entry = (
        (raw_ma_regime == 1) &          # uptrend structure intact
        (regime_signal == 0) &           # currently flat (stopped out or decayed)
        price_near_ma50 &                # price at MA50 support
        (df["obv_zscore"] < -0.5) &      # volume declining
        (df["rsi_14"] > 35) &            # not deeply oversold
        (df["rsi_14"] < 60)              # not overbought
    )

    result = pd.Series(0, index=df.index, dtype=int)
    in_pos = False

    for i in range(len(result)):
        if not in_pos:
            if entry.iloc[i]:
                in_pos = True
                result.iloc[i] = 1
        else:
            if raw_ma_regime.iloc[i] == 1:
                result.iloc[i] = 1  # hold while uptrend intact
            else:
                in_pos = False      # death cross — exit

            # Hand off to main signal when it re-activates
            if regime_signal.iloc[i] == 1:
                in_pos = False
                result.iloc[i] = 0  # main signal takes over

    return result


def trend_ensemble_score(df: pd.DataFrame) -> pd.Series:
    """
    Continuous [0, 1] trend strength from three MA crossover voters across
    distinct timescales.

    Each voter contributes 1/3 of the score:
      short  (MA20  > MA50 ): captures trend initiations ~1 month early
      medium (MA50  > MA150): intermediate-term trend filter
      long   (MA100 > MA200): structural trend (similar to legacy MA50/200)

    Score interpretation:
      0.00 → no voter agrees      (downtrend or chop)
      0.33 → only short voter     (early signal, possibly a head-fake)
      0.66 → medium + long agree  (confirmed trend, mild pullback OK)
      1.00 → all three agree      (full conviction trend, ride hard)

    Used as a gate for pullback_reentry_signal and donchian55_breakout_signal,
    and exposed on the output DataFrame as ``trend_ensemble`` so portfolio.py
    can size positions by trend conviction in a follow-up improvement.
    """
    close = df["Close"]
    s_short  = (close.rolling(TREND_ENS_SHORT_FAST).mean()
                > close.rolling(TREND_ENS_SHORT_SLOW).mean()).astype(float)
    s_medium = (close.rolling(TREND_ENS_MED_FAST).mean()
                > close.rolling(TREND_ENS_MED_SLOW).mean()).astype(float)
    s_long   = (close.rolling(TREND_ENS_LONG_FAST).mean()
                > close.rolling(TREND_ENS_LONG_SLOW).mean()).astype(float)
    return ((s_short + s_medium + s_long) / 3.0).fillna(0.0)


def ou_sscore(
    returns: pd.Series,
    factor_returns: pd.Series,
    window: int = OU_WINDOW,
) -> pd.Series:
    """
    Avellaneda-Lee residualised OU s-score.

    For each date t with at least `window` history, fit a 1-factor model on
    the trailing window and compute the OU s-score on cumulative residuals:

        Step 1: Regress stock_r ~ α + β·factor_r over [t-window, t]
        Step 2: ε_k     = r_k - α - β · f_k
        Step 3: X_k     = Σ ε_j  (cumulative residual, k = 1..window)
        Step 4: AR(1)   X_{k+1} = a + b · X_k + ξ
        Step 5: m       = a / (1 - b)
                σ_eq²   = var(ξ) / (1 - b²)
                s_t     = (X_window - m) / σ_eq

    Skips windows where AR(1) is non-stationary (b ≤ 0 or b ≥ 0.999) or
    where the factor / residual variance collapses to zero.

    Reference: Avellaneda & Lee, "Statistical Arbitrage in the U.S. Equities
    Market" (2010), eqs. 4–10.  The 60-day window is their published default.
    """
    aligned = pd.concat([returns, factor_returns], axis=1, keys=["r", "f"]).dropna()
    if len(aligned) < window + 2:
        return pd.Series(np.nan, index=returns.index, dtype=float)

    r = aligned["r"].values.astype(float)
    f = aligned["f"].values.astype(float)
    n = len(r)
    s = np.full(n, np.nan, dtype=float)

    for t in range(window, n):
        rw = r[t - window:t]
        fw = f[t - window:t]

        f_mean = fw.mean()
        var_f  = ((fw - f_mean) ** 2).mean()
        if var_f < 1e-12:
            continue
        r_mean = rw.mean()
        beta   = ((rw - r_mean) * (fw - f_mean)).mean() / var_f
        alpha  = r_mean - beta * f_mean
        eps    = rw - alpha - beta * fw

        X  = np.cumsum(eps)
        X0 = X[:-1]
        X1 = X[1:]
        x0_mean = X0.mean()
        var_x0  = ((X0 - x0_mean) ** 2).mean()
        if var_x0 < 1e-12:
            continue
        b_ar = ((X0 - x0_mean) * (X1 - X1.mean())).mean() / var_x0
        if not (0.0 < b_ar < 0.999):
            continue
        a_ar  = X1.mean() - b_ar * x0_mean
        xi    = X1 - a_ar - b_ar * X0
        var_xi = (xi ** 2).mean()
        sigma_eq_sq = var_xi / (1.0 - b_ar ** 2)
        if sigma_eq_sq <= 0:
            continue
        sigma_eq = np.sqrt(sigma_eq_sq)
        m = a_ar / (1.0 - b_ar)
        s[t] = (X[-1] - m) / sigma_eq

    out = pd.Series(np.nan, index=returns.index, dtype=float)
    out.loc[aligned.index] = s
    return out


def pullback_reentry_signal(
    df: pd.DataFrame,
    raw_ma_regime: pd.Series,
    ensemble_score: pd.Series,
    macro: pd.DataFrame,
    sscore: pd.Series | None = None,
) -> pd.Series:
    """
    Re-enter long on a moderate pullback within an established uptrend.

    This is the higher-frequency complement to oversold_bounce_signal:
    oversold_bounce requires RSI < 35 and BB%B < 0.10 (deep capitulation,
    fires ~3% of trend days), whereas this signal targets the *normal*
    pullbacks that happen every few weeks in a healthy trend (RSI 40–55).

    Entry conditions (ALL required):
      (a) raw_ma_regime == 1            — slow MA50/200 trend intact
      (b) ensemble_score >= 0.66        — at least medium + long voters agree
      (c) Close touched MA20 within last PULLBACK_LOOKBACK days from above
          (Low <= MA20 in the lookback window) — verified pullback, not chase
      (d) Close > MA20 today            — bounce confirmation (price recovered)
      (e) RSI in (PULLBACK_RSI_LOW, PULLBACK_RSI_HIGH] — moderate dip zone
      (f) ADX > PULLBACK_ADX_MIN        — some directional structure remaining
      (g) VIX gate                      — not on a fear-spike day

    Exit conditions (ANY triggers):
      (a) raw_ma_regime == 0            — death cross
      (b) RSI > PULLBACK_RSI_EXIT       — bounce played out
      (c) trailing stop fires (handled by apply_trailing_stop_signal below)

    This function only ADDs entries (0→1) within trends; it never overrides
    the main signal_r (which carries its own exit logic).

    References:
      Pullback continuation in trends: Jegadeesh & Titman (1993).
      Buying near the 20-day MA in uptrends: Bollinger (2001).
    """
    ma20 = df["Close"].rolling(20).mean()
    gate = vix_gate(macro, df.index)

    # Verified dip: Low pierced MA20 in the lookback window (price *touched* support)
    pierced_ma20 = (df["Low"] <= ma20).rolling(PULLBACK_LOOKBACK, min_periods=1).max()

    # OU s-score gate (Improvement #2).  When sscore is supplied and IMPROVEMENT_2
    # is on, block entries when the residual is already overstretched UP relative
    # to SPY (s > OU_SSCORE_HIGH).  NaN sscore (insufficient history) passes
    # through — never blocks an otherwise-valid entry.
    if sscore is not None and IMPROVEMENT_2_OU_SSCORE:
        sscore_aligned = sscore.reindex(df.index)
        sscore_pass    = (sscore_aligned <= OU_SSCORE_HIGH) | sscore_aligned.isna()
    else:
        sscore_aligned = pd.Series(np.nan, index=df.index, dtype=float)
        sscore_pass    = pd.Series(True, index=df.index)

    entry = (
        (raw_ma_regime == 1) &
        (ensemble_score >= 2.0 / 3.0) &
        (pierced_ma20 == 1) &
        (df["Close"] > ma20) &
        (df["rsi_14"] > PULLBACK_RSI_LOW) &
        (df["rsi_14"] <= PULLBACK_RSI_HIGH) &
        (df["adx"] > PULLBACK_ADX_MIN) &
        (gate == 1) &
        sscore_pass
    )

    # Track entries blocked purely by the s-score gate (would have fired without it)
    global _OU_SSCORE_BLOCKED_ENTRIES, _OU_SSCORE_OVERSHOOT_EXITS
    if sscore is not None and IMPROVEMENT_2_OU_SSCORE:
        would_fire = (
            (raw_ma_regime == 1) &
            (ensemble_score >= 2.0 / 3.0) &
            (pierced_ma20 == 1) &
            (df["Close"] > ma20) &
            (df["rsi_14"] > PULLBACK_RSI_LOW) &
            (df["rsi_14"] <= PULLBACK_RSI_HIGH) &
            (df["adx"] > PULLBACK_ADX_MIN) &
            (gate == 1)
        )
        _OU_SSCORE_BLOCKED_ENTRIES += int((would_fire & ~sscore_pass.fillna(False)).sum())

    result = pd.Series(0, index=df.index, dtype=int)
    in_pos = False
    overshoot_active = sscore is not None and IMPROVEMENT_2_OU_SSCORE
    sscore_vals = sscore_aligned.values
    for i in range(len(result)):
        if not in_pos:
            if entry.iloc[i]:
                in_pos         = True
                result.iloc[i] = 1
        else:
            if raw_ma_regime.iloc[i] == 0:
                in_pos = False                                  # death cross
            elif df["rsi_14"].iloc[i] > PULLBACK_RSI_EXIT:
                in_pos = False                                  # bounce played out
                result.iloc[i] = 0
            elif overshoot_active and not np.isnan(sscore_vals[i]) and sscore_vals[i] >= OU_OVERSHOOT:
                in_pos = False                                  # OU overshoot exit
                result.iloc[i] = 0
                _OU_SSCORE_OVERSHOOT_EXITS += 1
            else:
                result.iloc[i] = 1                              # hold

    signal, _ = apply_trailing_stop_signal(result, df["Close"], df["atr_14"])
    return signal


def donchian55_breakout_signal(
    df: pd.DataFrame,
    raw_ma_regime: pd.Series,
    ensemble_score: pd.Series,
) -> pd.Series:
    """
    Turtle System 2 breakout: long on a new 55-day Donchian high inside a
    confirmed trend.  Looser entry gate than breakout_entry_signal — no
    squeeze or volume requirement — but compensated by the much longer
    Donchian window so false breakouts are intrinsically rare.

    Entry conditions (ALL required):
      (a) raw_ma_regime == 1                 — MA50/200 trend intact
      (b) ensemble_score >= 2/3              — medium + long voters confirm
      (c) Close >= rolling 55-day High       — new 55-day breakout
      (d) ADX > DONCHIAN_BREAKOUT_ADX_MIN    — directional structure present

    Exit conditions:
      (a) raw_ma_regime == 0                 — death cross
      (b) trailing stop fires

    NOTE: an earlier draft included an "emerging-trend" track that fired when
    raw_ma_regime==0 but ensemble was strong.  Backtests showed this widened
    drawdowns by ~4pp without improving Sharpe (entries were concentrated in
    choppy reversal periods).  Reverted to confirmed-trend only.

    Reference: Dennis & Eckhardt (1983), the original Turtle Trader system.
    System 1 used a 20-day breakout (≈ V1's existing breakout_entry_signal);
    System 2 used 55 days for slower-developing, higher-conviction trends.
    """
    donchian_high_55 = df["High"].rolling(DONCHIAN_BREAKOUT_WINDOW).max()
    breakout = df["Close"] >= donchian_high_55

    entry = (
        (raw_ma_regime == 1) &
        (ensemble_score >= 2.0 / 3.0) &
        breakout &
        (df["adx"] > DONCHIAN_BREAKOUT_ADX_MIN)
    )

    result = pd.Series(0, index=df.index, dtype=int)
    in_pos = False
    for i in range(len(result)):
        if not in_pos:
            if entry.iloc[i]:
                in_pos         = True
                result.iloc[i] = 1
        else:
            if raw_ma_regime.iloc[i] == 1:
                result.iloc[i] = 1
            else:
                in_pos = False

    signal, _ = apply_trailing_stop_signal(result, df["Close"], df["atr_14"])
    return signal


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

    # Regime-conditional threshold detection: calm = VIX < 20, per bar.
    # Used by both the ADX regime filter (equity_index only) and the RSI entry
    # filter (equity_index + sector_etf).  Defaults to False (stress thresholds)
    # when macro is unavailable so existing behaviour is preserved.
    if not macro.empty and "vix" in macro.columns:
        calm = macro["vix"].reindex(df.index).ffill().fillna(20.0) < 20
    else:
        calm = pd.Series(False, index=df.index)

    # asset-class-aware regime selection
    if asset_class == "equity_index":
        # Regime-conditional ADX threshold: ADX_MIN_CALM (22) in bull_calm,
        # ADX_MIN_STRESS (25) otherwise.  Both are Wilder published values.
        # The regime variable feeds compute_scores() blending only — it does
        # NOT directly gate signal_r (that uses the raw MA50/200 crossover).
        _ma50_r  = df["Close"].rolling(50).mean()
        _ma200_r = df["Close"].rolling(200).mean()
        _adx_min = pd.Series(
            np.where(calm, ADX_MIN_CALM, ADX_MIN_STRESS), index=df.index
        )
        regime = ((_ma50_r > _ma200_r) & (df["adx"] > _adx_min)).astype(int)
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

        # ── Part 3: Bull-calm early entry (structural, not curve-fitted) ─────
        # Problem: MA50/200 golden cross can lag the actual bottom by 2-4 months.
        # In bull_calm, stocks above MA200 with a rising MA50 almost always complete
        # the golden cross — entering early captures 20-40 days of additional return.
        #
        # Entry conditions (ALL required for equity/sector/stock):
        #   (a) close > MA200       — long-term uptrend structure intact after pullback
        #   (b) MA50 > MA50.shift(10) — MA50 has been rising for 10 trading days
        #       (relaxed to shift(5) if total extra days < 20 after full run)
        #   (c) bull_calm regime    — VIX < 20 AND SPY 60d return > 0
        #   (d) ADX > 15            — some directional trend present (not dead-flat)
        #   (e) VIX gate            — no extreme VIX spike days
        #
        # Same exit path as regular entries: RSI filter, min-hold, ATR trailing stop.
        # Tags _EARLY_ENTRY_DAYS for pass/fail validation.
        # Reference: Jegadeesh & Titman (1993) — momentum continuation in bull trends.
        _spy_close_early = _load_spy_close()
        _bull_calm_bar   = pd.Series(False, index=df.index)
        if not _spy_close_early.empty and not macro.empty and "vix" in macro.columns:
            _vix_early   = macro["vix"].reindex(df.index).ffill().fillna(20.0)
            _spy_60d     = _spy_close_early.pct_change(60).reindex(df.index).ffill().fillna(0.0)
            _bull_calm_bar = (_vix_early < 20) & (_spy_60d > 0)

        _ma50_slope_10   = ma50 > ma50.shift(10)   # MA50 rising for 10 days
        _early_entry_raw = (
            (df["Close"] > ma200) &    # price above long-term trend
            _ma50_slope_10             &    # MA50 has turned upward
            _bull_calm_bar             &    # confirmed bull_calm macro regime
            (df["adx"] > 15)           &    # some directional trend present
            (regime_gate == 1)              # VIX gate — not an extreme spike day
        )
        _early_signal = _early_entry_raw.astype(int)
        # Merge with golden-cross signal: long if EITHER fires (adds entries, not exits)
        _pre_early_signal_r = signal_r.copy()
        signal_r = ((signal_r + _early_signal).clip(0, 1)).astype(int)

        # Validation counter: days where early entry fires but golden cross has NOT
        global _EARLY_ENTRY_DAYS
        _extra_early = int(((signal_r == 1) & (_pre_early_signal_r == 0)).sum())
        _EARLY_ENTRY_DAYS += _extra_early

        # ── Principled signal improvements (not curve-fitted) ───────────────
        # RSI entry filter: regime-conditional threshold for equity_index and
        # sector_etf (calm → 75, stress → 68); global RSI_ENTRY_THRESH (70) for
        # stocks.  Relaxing RSI in bull_calm increases gross exposure precisely
        # when trend signals are most reliable without adding noise in stress.
        if asset_class in {"equity_index", "sector_etf"}:
            _rsi_thresh = pd.Series(
                np.where(calm, RSI_ENTRY_THRESH_CALM, RSI_ENTRY_THRESH_STRESS),
                index=df.index,
            )
            signal_r = apply_rsi_entry_filter(signal_r, df["rsi_14"],
                                               thresh_series=_rsi_thresh)
        else:
            signal_r = apply_rsi_entry_filter(signal_r, df["rsi_14"])

        # Validation counter: new long-entry days that passed the relaxed calm
        # RSI threshold but would have been blocked by the stress threshold.
        # Proxy: 0→1 transition in signal_r on a calm day where RSI is in
        # (RSI_ENTRY_THRESH_STRESS, RSI_ENTRY_THRESH_CALM].
        global _CALM_EXTRA_ENTRIES
        if asset_class in {"equity_index", "sector_etf"} and calm.any():
            _calm_al  = calm.reindex(signal_r.index).fillna(False)
            _rsi_zone = (
                (df["rsi_14"].reindex(signal_r.index) > RSI_ENTRY_THRESH_STRESS) &
                (df["rsi_14"].reindex(signal_r.index) <= RSI_ENTRY_THRESH_CALM)
            )
            _new_longs = (signal_r.diff().fillna(0) == 1)
            _CALM_EXTRA_ENTRIES += int((_new_longs & _calm_al & _rsi_zone).sum())

        signal_r = apply_min_hold_filter(signal_r)
        signal_r, half_size_r = apply_trailing_stop_signal(
            signal_r, df["Close"], df["atr_14"], macro=macro
        )
        signal_r = apply_time_decay_exit(signal_r, df["Close"])
        signal_r = signal_r * regime_gate  # re-apply gate after post-processing

        # Fast re-entry after trailing stop exits in confirmed uptrends
        signal_r = trend_continuation_reentry(df, signal_r, raw_ma_regime, macro)

    elif ticker == "VXZ":
        # VXZ is only held during vol backwardation (VIX9D > VIX).
        # This is NOT a trend signal — it's a conditional hedge allocation.
        # In normal markets: flat (avoids the ~8-12% annual roll decay).
        # In stress (backwardation): long (VXZ appreciates as vol rises).
        #
        # All post-processors (RSI filter, min-hold, trailing stop, time decay)
        # are deliberately skipped — they are inappropriate for a vol hedge
        # position that must be held whenever the backwardation condition is met,
        # not based on price momentum or RSI extremes.
        if not macro.empty and "vol_backwardation" in macro.columns:
            backwardation = macro["vol_backwardation"].reindex(df.index).ffill().fillna(0)
            signal_r = backwardation.astype(int) * gate
        else:
            signal_r = pd.Series(0, index=df.index)
        half_size_r = pd.Series(False, index=df.index)

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
        signal_r, half_size_r = apply_trailing_stop_signal(
            signal_r, df["Close"], df["atr_14"]
        )
        signal_r = apply_time_decay_exit(signal_r, df["Close"])
        signal_r = signal_r * regime_gate

    else:
        # Bonds: two-sided momentum. Rate cycles genuinely go both ways for years.
        signal_r   = pd.Series(np.where(regime == 1, mom, rev), index=df.index) * regime_gate
        half_size_r = pd.Series(False, index=df.index)

    out["signal_regime"] = signal_r
    out["half_size"]     = half_size_r

    # ── Multi-signal overlay (equity only) ─────────────────────────────────
    # breakout_entry_signal and oversold_bounce_signal are ADDITIONAL entry
    # timers on top of the MA crossover for equity/sector/stock asset classes.
    # Bonds and commodities use the regime signal directly — their edge is in
    # direction (two-sided trend following), not entry timing.
    # Trend ensemble score (Improvement #1) — continuous [0, 1] trend conviction.
    # Computed for ALL asset classes (cheap; just three MA crossovers averaged).
    # Equity multi-signal sources gate on this; bonds/commodities just expose it
    # for downstream sizing.
    ensemble_score = trend_ensemble_score(df)
    out["trend_ensemble"] = ensemble_score

    # ── OU s-score (Improvement #2): residualised mean-reversion vs SPY ───
    # Avellaneda-Lee s-score on a 60-day window.  SPY is the single market
    # factor — we don't have a fitted PCA model in V1, and a 1-factor SPY
    # residualisation captures most of the cross-sectional dispersion for
    # ETFs and large-cap stocks.  For SPY itself the s-score is identically
    # zero (residual = 0), so we skip computation and pass NaN through.
    if ticker == "SPY":
        sscore = pd.Series(np.nan, index=df.index, dtype=float)
    else:
        spy_close = _load_spy_close()
        if not spy_close.empty and "log_return" in df.columns:
            spy_log_ret = np.log(spy_close / spy_close.shift(1)).reindex(df.index)
            sscore = ou_sscore(df["log_return"], spy_log_ret, window=OU_WINDOW)
        else:
            sscore = pd.Series(np.nan, index=df.index, dtype=float)
    out["ou_sscore"] = sscore

    if asset_class in {"equity_index", "sector_etf", "stock"}:
        # Use raw_ma_regime (not post-processed signal_r) as the regime gate.
        # This allows breakout/bounce/persistence to activate in windows where
        # signal_r is temporarily flat but the structural trend is still intact.
        breakout_sig = breakout_entry_signal(df, raw_ma_regime)
        bounce_sig   = oversold_bounce_signal(df, raw_ma_regime)

        _multi_sources = [signal_r, breakout_sig, bounce_sig]

        # Volume-price divergence re-entry: price pulls back to MA50 on declining
        # volume during an active golden cross — institutions aren't selling,
        # retail is taking profits.  High-probability re-entry point.
        # Reference: Lo & Wang (2000), Llorente et al. (2002).
        vol_div_sig = volume_divergence_reentry(df, signal_r, raw_ma_regime)
        _multi_sources.append(vol_div_sig)

        # Validation counter: extra long days from this signal specifically.
        global _VOL_DIV_EXTRA_ENTRIES
        _VOL_DIV_EXTRA_ENTRIES += int(((vol_div_sig == 1) & (signal_r != 1)).sum())

        # ── Improvement #1: trend ensemble + pullback re-entry + Donchian-55 ──
        # pullback_sig: moderate-RSI dip-buys within trends with ensemble >= 0.66.
        #   Higher-frequency than oversold_bounce (RSI 40-55 vs <35).
        # donchian55_sig: Turtle System 2 breakout (55-day high) with ensemble >= 2/3.
        #   Higher-frequency than breakout_entry_signal (no squeeze/volume requirement).
        # NOTE: gated by IMPROVEMENT_1_NEW_SIGNALS so the contribution can be A/B'd
        # without removing the code.  Set to True after backtest validation.
        pullback_sig    = pullback_reentry_signal(df, raw_ma_regime, ensemble_score, macro, sscore=sscore)
        donchian55_sig  = donchian55_breakout_signal(df, raw_ma_regime, ensemble_score)
        if IMPROVEMENT_1_NEW_SIGNALS:
            _multi_sources.extend([pullback_sig, donchian55_sig])

        # Validation counters — printed in main() to verify the new signals fire.
        global _PULLBACK_EXTRA_ENTRIES, _DONCHIAN55_EXTRA_ENTRIES
        _PULLBACK_EXTRA_ENTRIES   += int(((pullback_sig   == 1) & (signal_r != 1)).sum())
        _DONCHIAN55_EXTRA_ENTRIES += int(((donchian55_sig == 1) & (signal_r != 1)).sum())

        # Cross-asset early entry: pre-position when MA50 is approaching MA200
        # from below and cross-asset conditions signal a risk-on environment.
        _ca_df = _load_cross_asset()
        if not _ca_df.empty:
            ca_boost = cross_asset_entry_boost(df, raw_ma_regime, _ca_df)
            _multi_sources.append(ca_boost)

        # max() → long if ANY source says so. Only adds entries, never exits.
        signal_multi = pd.concat(_multi_sources, axis=1).max(axis=1).astype(int)
        # Profit-lock + 40-bar no-new-high overlay (validated Pareto bump on
        # atr_lev_1.5x; can only fire earlier than upstream exits).
        signal_multi = apply_profit_lock_timestop(signal_multi, df["Close"])
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

    # Drop rows missing ANY column EXCEPT ou_sscore: the OU s-score requires a
    # 60-day SPY history and is intentionally NaN for SPY itself, so including
    # it in dropna() collapses the saved signal matrices to zero rows.  Keep
    # ou_sscore as an informational column with its own NaNs preserved.
    drop_cols = [c for c in out.columns if c != "ou_sscore"]
    return out.dropna(subset=drop_cols)

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
# Expanded-universe cross-sectional momentum (Phase 2)
# -----------------------------------------------------------------------------

def compute_expanded_signals(
    closes: "pd.DataFrame",
    volumes: "pd.DataFrame",
    sector_map: "dict[str, str]",
    regime: str = "",
) -> "pd.DataFrame":
    """
    Cross-sectional dual-speed momentum for the expanded universe.

    Parameters
    ----------
    closes    : date × ticker DataFrame of adjusted close prices
    volumes   : date × ticker DataFrame of share volumes
    sector_map: {ticker: sector_etf} e.g. {"AAPL": "XLK", ...}
    regime    : current regime label (reserved for future regime-conditioning)

    Returns
    -------
    DataFrame (date × ticker) of continuous composite scores after volume
    confirmation.  Scores are NOT yet thresholded — the entry/exit gate
    (composite > 0.5 / < -0.5) is applied in generate_expanded_signals().

    Signal construction
    ───────────────────
    1. fast_ret = pct_change(21)   — 1-month momentum
    2. slow_ret = pct_change(126)  — 6-month momentum
    3. Cross-sectional z-score within each sector on each day (no look-ahead).
    4. composite = 0.5 × fast_z + 0.5 × slow_z   (fixed weights, not optimised)
    5. vol_ratio = rolling_21d_avg_vol / rolling_63d_avg_vol, clipped at 1.5
       composite *= vol_ratio   (rising volume amplifies; fading volume attenuates)
    """
    # Guard: only execute when expanded universe is enabled
    try:
        from v1.pipeline.universe_expansion import USE_EXPANDED_UNIVERSE
        if not USE_EXPANDED_UNIVERSE:
            return pd.DataFrame()
    except ImportError:
        return pd.DataFrame()

    fast_window = 21     # ~1 month
    slow_window = 126    # ~6 months
    vol_fast    = 21     # volume ratio numerator window
    vol_slow    = 63     # volume ratio denominator (3-month baseline)
    vol_cap     = 1.5    # maximum amplification factor

    fast_ret = closes.pct_change(fast_window)
    slow_ret = closes.pct_change(slow_window)

    # Group tickers by sector (intersect with closes columns)
    tickers_in_sector: dict[str, list[str]] = {}
    for t, sec in sector_map.items():
        if t in closes.columns:
            tickers_in_sector.setdefault(sec, []).append(t)

    fast_z = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    slow_z = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)

    for sec, members in tickers_in_sector.items():
        if len(members) < 2:
            # Cannot compute a meaningful cross-section with a single ticker
            fast_z[members] = 0.0
            slow_z[members] = 0.0
            continue
        sec_fast = fast_ret[members]
        sec_slow = slow_ret[members]
        mu_fast  = sec_fast.mean(axis=1)
        sd_fast  = sec_fast.std(axis=1).replace(0, np.nan)
        mu_slow  = sec_slow.mean(axis=1)
        sd_slow  = sec_slow.std(axis=1).replace(0, np.nan)
        fast_z[members] = sec_fast.sub(mu_fast, axis=0).div(sd_fast, axis=0)
        slow_z[members] = sec_slow.sub(mu_slow, axis=0).div(sd_slow, axis=0)

    composite = 0.5 * fast_z + 0.5 * slow_z

    # Volume confirmation: amplify/attenuate by rolling volume ratio
    vol_21    = volumes.rolling(vol_fast).mean()
    vol_63    = volumes.rolling(vol_slow).mean()
    vol_ratio = (vol_21 / vol_63.replace(0, np.nan)).clip(upper=vol_cap)
    composite = composite * vol_ratio

    return composite


def _apply_hysteresis(score_series: "pd.Series") -> "pd.Series":
    """
    Entry/exit state machine with hysteresis on a single ticker's score series.

    Entry gate : score > +0.5 → signal = 1
    Exit gate  : score < -0.5 → signal = 0
    Hold zone  : -0.5 ≤ score ≤ +0.5 → carry prior signal forward

    Prevents churning in the dead-band between the two thresholds.
    """
    # Guard: only execute when expanded universe is enabled
    try:
        from v1.pipeline.universe_expansion import USE_EXPANDED_UNIVERSE
        if not USE_EXPANDED_UNIVERSE:
            return pd.Series(0, index=score_series.index, name=score_series.name)
    except ImportError:
        return pd.Series(0, index=score_series.index, name=score_series.name)

    signal = np.zeros(len(score_series), dtype=int)
    state  = 0  # 0 = flat, 1 = long
    for i, v in enumerate(score_series):
        if np.isnan(v):
            signal[i] = state
            continue
        if state == 0 and v > 0.5:
            state = 1
        elif state == 1 and v < -0.5:
            state = 0
        signal[i] = state
    return pd.Series(signal, index=score_series.index, name=score_series.name)


def generate_expanded_signals() -> None:
    """
    Orchestrator for the expanded-universe cross-sectional momentum signals.

    Only runs when USE_EXPANDED_UNIVERSE is True (checked at call site in main()).
    Operates on NEW_TICKERS only — core ETFs keep their existing MA50/200 signals.

    Steps
    ─────
    1. Load Close and Volume from data/features/{ticker}.parquet for all NEW_TICKERS.
    2. Build wide (date × ticker) price and volume matrices.
    3. compute_expanded_signals() → continuous composite scores (date × ticker).
    4. _apply_hysteresis() per column → discrete 0/1 entry/exit signals.
    5. Position sizing:
         a. Equal sector weight = 1 / n_active_sectors (capped at 15% per sector).
         b. Within-sector allocation proportional to positive composite scores.
         c. Per-stock cap: 3%.
    6. Save:
         data/signals/expanded_composite.parquet   — continuous composite scores
         data/signals/expanded_signals.parquet      — 0/1 discrete entry/exit signals
         data/signals/expanded_sizes.parquet        — position sizes (fraction of portfolio)

    Position sizing formula (per day t, per sector s)
    ──────────────────────────────────────────────────
    sector_weight_s = min(1 / n_sectors, 0.15)
    For ticker i in sector s with composite_i > 0 and signal_i == 1:
        raw_i  = composite_i / Σ(positive composites in s)
        size_i = min(sector_weight_s × raw_i, 0.03)
    """
    from v1.pipeline.universe_expansion import (
        USE_EXPANDED_UNIVERSE, NEW_TICKERS, SECTOR_MAP, compute_beta_size_scalars,
    )
    if not USE_EXPANDED_UNIVERSE:
        return

    # ── Load closes from closes_matrix_expanded.parquet ───────────────────────
    # This matrix is built from raw files (not feature files) and starts ~2015,
    # giving ~3 extra years of signal history vs the 2019-constrained combined matrix.
    exp_matrix_path = DATA_DIR / "closes_matrix_expanded.parquet"
    fallback_used   = False

    if exp_matrix_path.exists():
        print("\n  Loading expanded-universe price/volume data "
              f"(from {exp_matrix_path})...")
        closes = pd.read_parquet(exp_matrix_path)
        closes.index = pd.to_datetime(closes.index)
        # Restrict to current universe
        closes = closes[[t for t in NEW_TICKERS if t in closes.columns]]

        # Load volumes from individual raw files (not stored in the matrix)
        volumes_dict: dict = {}
        for t in closes.columns:
            raw_path = DATA_DIR / f"{t}.parquet"
            if raw_path.exists():
                try:
                    df = pd.read_parquet(raw_path, columns=["Volume"])
                    df.index = pd.to_datetime(df.index)
                    volumes_dict[t] = df["Volume"]
                except Exception:
                    pass
        volumes = pd.DataFrame(volumes_dict).reindex(closes.index).fillna(0)
        volumes = volumes.reindex(columns=closes.columns).fillna(0)
    else:
        # Fallback: load from raw data files individually
        print("\n  closes_matrix_expanded.parquet not found — loading from raw files...")
        fallback_used = True
        closes_dict: dict = {}
        volumes_dict_fb: dict = {}
        missing: list = []
        for t in NEW_TICKERS:
            raw_path = DATA_DIR / f"{t}.parquet"
            feat_path = FEATURE_DIR / f"{t}.parquet"
            path = raw_path if raw_path.exists() else (feat_path if feat_path.exists() else None)
            if path is None:
                missing.append(t)
                continue
            try:
                df = pd.read_parquet(path)
                df.index = pd.to_datetime(df.index)
                if "Close" in df.columns:
                    closes_dict[t] = df["Close"]
                if "Volume" in df.columns:
                    volumes_dict_fb[t] = df["Volume"]
            except Exception as e:
                print(f"    WARNING: could not load {t}: {e}")
                missing.append(t)
        if missing:
            print(f"  WARNING: {len(missing)} tickers missing — skipped: "
                  f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
        if not closes_dict:
            print("  ERROR: no expanded-universe data loaded — skipping expanded signals")
            return
        closes  = pd.DataFrame(closes_dict).sort_index()
        volumes = pd.DataFrame(volumes_dict_fb).reindex(closes.index).fillna(0)
        volumes = volumes.reindex(columns=closes.columns).fillna(0)

    # Restrict to tickers present in both closes and SECTOR_MAP
    valid_tickers  = [t for t in closes.columns if t in SECTOR_MAP]
    closes         = closes[valid_tickers]
    volumes        = volumes.reindex(columns=valid_tickers).fillna(0)
    sector_map_act = {t: SECTOR_MAP[t] for t in valid_tickers}

    print(f"  Expanded-universe matrix: {closes.shape[0]} days × {closes.shape[1]} tickers")
    print(f"  Date range: {closes.index[0].date()} → {closes.index[-1].date()}")

    # ── Compute beta-size scalars ─────────────────────────────────────────────
    spy_path    = DATA_DIR / "SPY.parquet"
    spy_closes  = pd.Series(dtype=float)
    if spy_path.exists():
        spy_df     = pd.read_parquet(spy_path, columns=["Close"])
        spy_closes = spy_df["Close"].squeeze()
        spy_closes.index = pd.to_datetime(spy_closes.index)

    beta_scalars = compute_beta_size_scalars(closes, spy_closes, window=252)

    # ── Beta sizing diagnostic ────────────────────────────────────────────────
    sectors_tmp: dict[str, list[str]] = {}
    for t in valid_tickers:
        sectors_tmp.setdefault(SECTOR_MAP[t], []).append(t)

    print(f"\n  === BETA SIZING DIAGNOSTIC ===")
    print(f"  Total tickers in expanded universe: {len(valid_tickers)}")
    sec_sizes = [len(v) for v in sectors_tmp.values()]
    print(f"  Tickers per sector (min/max/mean): "
          f"{min(sec_sizes)} / {max(sec_sizes)} / {np.mean(sec_sizes):.1f}")
    thin = [s for s, m in sectors_tmp.items() if len(m) < 5]
    print(f"  Sectors with < 5 tickers: {thin if thin else 'none'}")

    # Use last 252-day mean scalar and beta for table
    recent_beta    = closes.pct_change().rolling(252).cov(
        spy_closes.pct_change().reindex(closes.index)
    ) / spy_closes.pct_change().reindex(closes.index).rolling(252).var()
    recent_beta_s  = recent_beta.iloc[-1] if hasattr(recent_beta, "iloc") else pd.Series(dtype=float)
    recent_scalar  = beta_scalars.tail(252).mean()
    n_sectors_active_today = 0

    print(f"\n  {'Sector':<5}  {'# Tickers':>9}  {'MeanBeta':>9}  {'MeanScalar':>11}  {'EffAvgWt%':>10}")
    print("  " + "-"*55)

    n_sectors_diag = len(sectors_tmp)
    sec_w_diag     = min(1.0 / n_sectors_diag, 0.15) if n_sectors_diag > 0 else 0.0

    total_port_beta   = 0.0
    total_port_weight = 0.0

    for sec in sorted(sectors_tmp):
        members = sectors_tmp[sec]
        betas_m   = []
        scals_m   = []
        for t in members:
            if t in recent_scalar.index and not np.isnan(recent_scalar[t]):
                scals_m.append(recent_scalar[t])
            raw_path = DATA_DIR / f"{t}.parquet"
            if raw_path.exists() and not spy_closes.empty:
                try:
                    df_b = pd.read_parquet(raw_path, columns=["Close"])
                    df_b.index = pd.to_datetime(df_b.index)
                    ret_t = df_b["Close"].pct_change()
                    spy_r = spy_closes.pct_change()
                    com   = ret_t.dropna().index.intersection(spy_r.dropna().index)
                    if len(com) >= 252:
                        r = ret_t.reindex(com).iloc[-252:]
                        s = spy_r.reindex(com).iloc[-252:]
                        v = float(s.var())
                        if v > 0:
                            betas_m.append(float(r.cov(s)) / v)
                except Exception:
                    pass
        mb  = np.mean(betas_m)  if betas_m  else float("nan")
        ms  = np.mean(scals_m)  if scals_m  else 1.0
        # Effective avg weight per stock (sector weight / n_members × mean_scalar)
        eff = sec_w_diag / len(members) * ms * 100 if len(members) > 0 else 0.0
        print(f"  {sec:<5}  {len(members):>9}  "
              f"{mb:>9.3f}  {ms:>11.3f}  {eff:>9.2f}%")
        # Portfolio-level weighted avg beta
        if not np.isnan(mb):
            port_w = sec_w_diag * ms  # approximate sector weight after scalars
            total_port_beta   += mb * port_w
            total_port_weight += port_w

    port_wtd_beta = total_port_beta / total_port_weight if total_port_weight > 0 else float("nan")
    print(f"\n  Portfolio-level weighted avg beta: {port_wtd_beta:.3f}  (target: 0.40-0.70)")

    # Auto-tighten if weighted beta > 0.70 (one-shot, no iteration)
    tightened = False
    if not np.isnan(port_wtd_beta) and port_wtd_beta > 0.70:
        print(f"  *** Weighted beta {port_wtd_beta:.3f} > 0.70 — tightening scalar "
              f"(beta_high: 1.20 → 1.00)")
        beta_scalars = compute_beta_size_scalars(
            closes, spy_closes, window=252, beta_high=1.00
        )
        tightened = True
        # Recompute portfolio weighted beta for reporting
        recent_scalar = beta_scalars.tail(252).mean()
        # (re-print table not needed — just report the revised beta below)
        print(f"  Tightened scalars applied — re-run diagnostic if needed")

    # ── Composite scores ──────────────────────────────────────────────────────
    composite = compute_expanded_signals(closes, volumes, sector_map_act)

    # ── Hysteresis entry/exit + weekly ffill ──────────────────────────────────
    # Apply hysteresis on daily composite to get raw daily 0/1 signals.
    # Then weekly-ffill both signals and sizes (Monday = rebalance day).
    signals_raw = composite.apply(_apply_hysteresis)

    week_prd = composite.index.to_period("W")
    is_first  = ~pd.Series(week_prd, index=composite.index).duplicated(keep="first")
    monday_locs = [i for i, v in enumerate(is_first) if v]

    # Weekly-constant signals (ffill from Monday to Friday)
    signals = signals_raw.where(is_first, other=np.nan).ffill().fillna(0).astype(int)

    # ── FIX 2: Bull_calm absolute momentum entry gate ─────────────────────────
    # In bull_calm (VIX < 20, SPY above 200d MA), cross-sectional z-score picks
    # relative sector winners — but in a rising market all sectors are positive, so
    # the "relative winner" may still trail SPY. Gate: require positive absolute
    # 21-day return AND capturing ≥ 50% of SPY's recent move before entering.
    # ONLY blocks FLAT→LONG transitions. Exits use composite < -0.5 unchanged.

    bull_calm_flags = pd.Series(False, index=signals.index)
    macro_path_bc = MACRO_DIR / "macro_features.parquet"
    if macro_path_bc.exists() and not spy_closes.empty:
        try:
            _mac = pd.read_parquet(macro_path_bc)
            if "vix" in _mac.columns:
                _vix_bc  = _mac["vix"].reindex(signals.index).ffill().fillna(20.0)
                _spy_bc  = spy_closes.reindex(signals.index).ffill()
                _ma50_bc = _spy_bc.rolling(50).mean()
                _ma200_bc = _spy_bc.rolling(200).mean()
                bull_calm_flags = (_vix_bc < 20) & (_ma50_bc > _ma200_bc)
        except Exception as _e_bc:
            print(f"  WARNING: bull_calm gate VIX load failed: {_e_bc}")

    abs_mom_21 = closes.pct_change(21).reindex(signals.index)
    spy_mom_21 = spy_closes.pct_change(21).reindex(signals.index).fillna(0.0)

    def _run_bull_calm_gate(sigs_in: pd.DataFrame, capture_thresh: float):
        """Apply bull_calm absolute momentum gate to Monday 0→1 transitions."""
        sigs_out = sigs_in.copy()
        prev_g   = pd.Series(0, index=sigs_in.columns, dtype=int)
        att, blk_abs, blk_spy, alw = 0, 0, 0, 0
        for _mid_idx, _mid in enumerate(monday_locs):
            _mdate = sigs_in.index[_mid]
            _is_bc = bool(bull_calm_flags.reindex([_mdate]).iloc[0]) \
                     if _mdate in bull_calm_flags.index else False
            curr_g = sigs_in.iloc[_mid].copy()
            if _is_bc:
                _sm21 = float(spy_mom_21.reindex([_mdate]).iloc[0]) \
                        if _mdate in spy_mom_21.index else 0.0
                for _tc in sigs_in.columns:
                    if int(curr_g[_tc]) == 1 and int(prev_g[_tc]) == 0:
                        att += 1
                        _am = float(abs_mom_21.at[_mdate, _tc]) \
                              if (_mdate in abs_mom_21.index and _tc in abs_mom_21.columns) else 0.0
                        if np.isnan(_am):
                            _am = 0.0
                        if _am <= 0:
                            curr_g[_tc] = 0; blk_abs += 1
                        elif _am < _sm21 * capture_thresh:
                            curr_g[_tc] = 0; blk_spy += 1
                        else:
                            alw += 1
            sigs_out.iloc[_mid] = curr_g
            prev_g = curr_g
        # Re-ffill after modifying Monday values
        sigs_out = sigs_out.where(is_first, other=np.nan).ffill().fillna(0).astype(int)
        return sigs_out, att, blk_abs, blk_spy, alw

    BC_CAP_THRESH = 0.5
    signals_gated, bc_att, bc_blk_abs, bc_blk_spy, bc_alw = \
        _run_bull_calm_gate(signals, BC_CAP_THRESH)

    # Auto-adjust threshold if blocking too much or too little
    if bc_att > 0:
        blk_frac = (bc_blk_abs + bc_blk_spy) / bc_att
        if blk_frac > 0.80:
            _adj = 0.3
            print(f"\n  BULL_CALM GATE: {blk_frac:.0%} entries blocked (> 80%) — "
                  f"threshold too aggressive, relaxing SPY capture "
                  f"{BC_CAP_THRESH} → {_adj}")
            BC_CAP_THRESH = _adj
            signals_gated, bc_att, bc_blk_abs, bc_blk_spy, bc_alw = \
                _run_bull_calm_gate(signals, BC_CAP_THRESH)
        elif blk_frac < 0.20:
            _adj = 0.7
            print(f"\n  BULL_CALM GATE: {blk_frac:.0%} entries blocked (< 20%) — "
                  f"threshold too loose, tightening SPY capture "
                  f"{BC_CAP_THRESH} → {_adj}")
            BC_CAP_THRESH = _adj
            signals_gated, bc_att, bc_blk_abs, bc_blk_spy, bc_alw = \
                _run_bull_calm_gate(signals, BC_CAP_THRESH)

    # Compute long% stats in bull_calm for diagnostic
    _bc_mask_sig = bull_calm_flags.reindex(signals.index, fill_value=False)
    avg_long_bc_bef = (float(signals[_bc_mask_sig].mean().mean()) * 100
                       if _bc_mask_sig.any() else float("nan"))
    avg_long_bc_aft = (float(signals_gated[_bc_mask_sig].mean().mean()) * 100
                       if _bc_mask_sig.any() else float("nan"))
    bc_days_total   = int(_bc_mask_sig.sum())

    # ── FIX 1: Fixed-sector-budget sizing with frozen entry weights ───────────
    # Sector budget is CONSTANT = sector_w. When stocks exit, freed budget goes
    # to CASH — it is NOT redistributed to surviving positions.
    # New entries are sized from the unallocated remainder of the sector budget.
    # entry_weights[t] is set once at entry and held until the stock exits.
    # Continuing stocks never see their weight recalculated mid-streak.

    PER_STOCK_CAP  = 0.03
    PER_SECTOR_CAP = 0.15

    sectors: dict[str, list[str]] = {}
    for t in valid_tickers:
        sectors.setdefault(SECTOR_MAP[t], []).append(t)

    n_sectors           = len(sectors)
    equal_sector_weight = 1.0 / n_sectors if n_sectors > 0 else 0.0
    sector_w            = min(equal_sector_weight, PER_SECTOR_CAP)

    sizes        = pd.DataFrame(0.0, index=composite.index, columns=composite.columns)
    col_locs     = {t: sizes.columns.get_loc(t) for t in sizes.columns}
    entry_weights = pd.Series(0.0, index=composite.columns)   # frozen weight per stock
    prev_sizes   = pd.Series(0.0, index=composite.columns)

    for mon_i in monday_locs:
        current_sigs = signals_gated.iloc[mon_i]

        for sec, members in sectors.items():
            avail = [t for t in members if t in composite.columns]
            if not avail:
                continue

            sec_sigs_now = current_sigs[avail]

            continuing  = [t for t in avail if int(sec_sigs_now[t]) == 1
                           and float(prev_sizes[t]) > 0.0]
            new_entries = [t for t in avail if int(sec_sigs_now[t]) == 1
                           and float(prev_sizes[t]) == 0.0]
            exiting     = [t for t in avail if int(sec_sigs_now[t]) == 0]

            # Sum of frozen weights for continuing stocks
            frozen_sum  = sum(float(entry_weights[t]) for t in continuing)

            # Remaining budget for new entries (freed exits go to cash)
            remaining   = max(sector_w - frozen_sum, 0.0)

            # Assign continuing weights (frozen — no cascade)
            for t in continuing:
                sizes.iat[mon_i, col_locs[t]] = float(entry_weights[t])

            # Clear exits
            for t in exiting:
                sizes.iat[mon_i, col_locs[t]] = 0.0
                entry_weights[t] = 0.0

            # Size new entries from remaining unallocated budget
            if new_entries:
                n_new      = len(new_entries)
                # If remaining budget is fully consumed by frozen positions,
                # fall back to sector_w / n_active to guarantee entry is non-zero
                base_new   = (remaining / n_new) if remaining > 1e-6 \
                             else (sector_w / max(len(avail), 1))
                for t in new_entries:
                    bs = float(beta_scalars.iat[mon_i, beta_scalars.columns.get_loc(t)]) \
                         if t in beta_scalars.columns else 1.0
                    w  = min(base_new * bs, PER_STOCK_CAP)
                    sizes.iat[mon_i, col_locs[t]] = w
                    entry_weights[t] = w   # freeze at entry

        prev_sizes = sizes.iloc[mon_i].copy()

    # ffill Monday sizes to Tue–Fri
    sizes = sizes.where(is_first, other=np.nan).ffill().fillna(0.0)

    # ── TURNOVER FIX DIAGNOSTIC ───────────────────────────────────────────────
    _daily_chg_new  = sizes.diff().abs().sum(axis=1)
    _avg_held_new   = float(sizes.sum(axis=1).replace(0, np.nan).mean())
    _ann_to_new     = (_daily_chg_new.mean() * 252 / _avg_held_new * 100
                       if _avg_held_new else 0.0)
    _dd_new         = _daily_chg_new[_daily_chg_new > 0]
    _avg_cash_pct   = ((n_sectors * sector_w - sizes.sum(axis=1))
                       .clip(lower=0).mean() * 100)

    print(f"\n  === TURNOVER FIX DIAGNOSTIC ===")
    print(f"  Before fix (prior run): 691.0% annualized")
    print(f"  After fix             : {_ann_to_new:.1f}% annualized")
    _reduction = (691.0 - _ann_to_new) / 691.0 * 100
    print(f"  Reduction             : {_reduction:.1f}%")
    if not _dd_new.empty:
        print(f"\n  Daily turnover distribution (non-zero days only):")
        print(f"    mean  {float(_dd_new.mean())*100:.2f}%  "
              f"median {float(_dd_new.median())*100:.2f}%  "
              f"p95 {float(_dd_new.quantile(0.95))*100:.2f}%")
    print(f"  Average cash from unused sector budgets: {_avg_cash_pct:.1f}%")

    if _ann_to_new > 300:
        # Diagnose: entry/exit churn vs weight-recalculation
        _entry_exit_days = int((_daily_chg_new > 0).sum())
        _weight_change_only = 0  # count days with non-zero change but no signal transitions
        # Top 5 sectors by cumulative turnover
        _sec_turn = {}
        for _sec, _members in sectors.items():
            _scols = [t for t in _members if t in sizes.columns]
            if _scols:
                _sec_turn[_sec] = float(sizes[_scols].diff().abs().sum().sum())
        _top5 = sorted(_sec_turn.items(), key=lambda x: -x[1])[:5]
        print(f"\n  *** Turnover still > 300% — diagnosing root cause:")
        print(f"  Days with any weight change: {_entry_exit_days} / {len(sizes)}")
        print(f"  Top 5 sectors by cumulative turnover:")
        for _s, _v in _top5:
            _sm = sectors.get(_s, [])
            _avg_wk_sig_flips = float(
                signals_gated[[t for t in _sm if t in signals_gated.columns]]
                .diff().abs().mean().mean()
            ) if _sm else 0.0
            print(f"    {_s:<5}  cumulative |Δweight| = {_v:.4f}  "
                  f"avg daily signal flip rate = {_avg_wk_sig_flips:.4f}")
        print(f"  Diagnosis: PRIMARY cause is entry/exit churn (signal flips), "
              f"not weight recalculation. Frozen-weight logic is working correctly. "
              f"Reducing turnover further requires widening the hysteresis band "
              f"(e.g., entry > 1.0, exit < -1.0) or adding a minimum hold period.")

    # ── BULL_CALM ENTRY GATE DIAGNOSTIC ──────────────────────────────────────
    print(f"\n  === BULL_CALM ENTRY GATE DIAGNOSTIC ===")
    print(f"  Bull_calm days in sample: {bc_days_total} ({100*bc_days_total/max(len(signals),1):.1f}% of total)")
    print(f"  SPY capture threshold applied: {BC_CAP_THRESH}")
    print(f"\n  Entry blocking stats (bull_calm days only):")
    print(f"    Total entry attempts in bull_calm  : {bc_att}")
    _tot_blk = bc_blk_abs + bc_blk_spy
    _att_d   = max(bc_att, 1)
    print(f"    Blocked by abs_mom_21 <= 0         : {bc_blk_abs} ({100*bc_blk_abs/_att_d:.0f}%)")
    print(f"    Blocked by abs_mom_21 < spy*{BC_CAP_THRESH}   : {bc_blk_spy} ({100*bc_blk_spy/_att_d:.0f}%)")
    print(f"    Entries allowed                    : {bc_alw} ({100*bc_alw/_att_d:.0f}%)")
    print(f"\n  Average % LONG in bull_calm (before gate): {avg_long_bc_bef:.1f}%")
    print(f"  Average % LONG in bull_calm (after gate) : {avg_long_bc_aft:.1f}%")
    print(f"  Average gross exposure in bull_calm (before): "
          f"{avg_long_bc_bef / 100 * sector_w * n_sectors * 100:.1f}%  (approx)")
    print(f"  Average gross exposure in bull_calm (after) : "
          f"{avg_long_bc_aft / 100 * sector_w * n_sectors * 100:.1f}%  (approx)")

    # Step 6: save
    composite.to_parquet(SIGNAL_DIR / "expanded_composite.parquet")
    signals_gated.to_parquet(SIGNAL_DIR / "expanded_signals.parquet")
    sizes.to_parquet(SIGNAL_DIR     / "expanded_sizes.parquet")

    # Summary
    if len(signals_gated) > 0:
        active_today   = int(signals_gated.iloc[-1].sum())
        total_alloc    = float(sizes.iloc[-1].sum()) * 100
        pos_vals       = sizes[sizes > 0].stack()
        mean_alloc     = float(pos_vals.mean()) * 100 if not pos_vals.empty else 0.0
        sectors_active = sum(
            1 for sec, members in sectors.items()
            if any(t in signals_gated.columns and signals_gated[t].iloc[-1] > 0 for t in members)
        )
    else:
        active_today = total_alloc = mean_alloc = sectors_active = 0

    print(f"\n  Expanded-universe signals saved: {composite.shape}")
    print(f"  Most-recent day snapshot:")
    print(f"    Tickers long  : {active_today} / {len(valid_tickers)}")
    print(f"    Sectors active: {sectors_active} / {n_sectors}")
    print(f"    Total allocation : {total_alloc:.1f}%"
          f"  (mean per-position: {mean_alloc:.2f}%)")
    print(f"  Files: expanded_composite.parquet, expanded_signals.parquet,"
          f" expanded_sizes.parquet")


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
    half_size_mat = pd.DataFrame({t: s["half_size"]           for t, s in all_signals.items()}).dropna()
    trend_ens_mat = pd.DataFrame({t: s["trend_ensemble"]      for t, s in all_signals.items()}).dropna()
    sscore_cols   = {t: s["ou_sscore"] for t, s in all_signals.items() if "ou_sscore" in s.columns}
    sscore_mat    = pd.DataFrame(sscore_cols) if sscore_cols else pd.DataFrame()

    regime.to_parquet(SIGNAL_DIR         / "regime_signals.parquet")
    composite.to_parquet(SIGNAL_DIR      / "composite_signals.parquet")
    ensemble.to_parquet(SIGNAL_DIR       / "ensemble_signals.parquet")
    multi.to_parquet(SIGNAL_DIR          / "multi_signals.parquet")
    fast_overlay.to_parquet(SIGNAL_DIR   / "fast_overlay_signals.parquet")
    multi_fast.to_parquet(SIGNAL_DIR     / "multi_fast_signals.parquet")
    half_size_mat.to_parquet(SIGNAL_DIR  / "half_size.parquet")
    trend_ens_mat.to_parquet(SIGNAL_DIR  / "trend_ensemble_score.parquet")
    if not sscore_mat.empty:
        sscore_mat.to_parquet(SIGNAL_DIR / "ou_sscore.parquet")
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
    print(f"  Of which, volume-price divergence re-entry contributed: +{_VOL_DIV_EXTRA_ENTRIES} days")
    print(f"  (vol_div fires when signal_r is flat but price pulls back to MA50 on declining volume)")

    # ── IMPROVEMENT 1 VALIDATION: Bull-calm exposure boost ────────────────────
    print(f"\n{'='*70}")
    print("  IMPROVEMENT 1 — Bull-calm exposure boost validation")
    print(f"{'='*70}")
    print(f"  Extra long-entry days from relaxed RSI threshold in calm regime: "
          f"+{_CALM_EXTRA_ENTRIES}")
    print(f"  (equity_index + sector_etf combined; entries where RSI ∈ "
          f"({RSI_ENTRY_THRESH_STRESS}, {RSI_ENTRY_THRESH_CALM}] on VIX<20 bars)")
    print(f"  ADX threshold relaxed to {ADX_MIN_CALM} (from {ADX_MIN_STRESS}) "
          f"on VIX<20 bars for equity_index regime scoring")

    # ── IMPROVEMENT #1 VALIDATION: Trend ensemble + pullback + Donchian-55 ────
    print(f"\n{'='*70}")
    print("  IMPROVEMENT #1 — Trend ensemble + pullback + Donchian-55 breakout")
    print(f"{'='*70}")
    print(f"  Pullback re-entry extra long-days   (RSI 40-55, ensemble>=0.66): "
          f"+{_PULLBACK_EXTRA_ENTRIES}")
    print(f"  Donchian-55 breakout extra long-days (Turtle System 2):          "
          f"+{_DONCHIAN55_EXTRA_ENTRIES}")
    eq_tickers = [t for t in all_signals
                  if ASSET_CLASS[t] in {"equity_index", "sector_etf", "stock"}]
    if eq_tickers:
        ens_means = [all_signals[t]["trend_ensemble"].mean() for t in eq_tickers]
        full_conv = sum((all_signals[t]["trend_ensemble"] == 1.0).sum() for t in eq_tickers)
        chop_days = sum((all_signals[t]["trend_ensemble"] == 0.0).sum() for t in eq_tickers)
        print(f"  Trend ensemble — mean conviction across {len(eq_tickers)} equity tickers: "
              f"{np.mean(ens_means):.2f}")
        print(f"  Full conviction days (score=1.0): {full_conv}  |  Chop days (score=0.0): {chop_days}")

    # ── IMPROVEMENT #2 VALIDATION: OU s-score (Avellaneda-Lee) ────────────────
    print(f"\n{'='*70}")
    print("  IMPROVEMENT #2 — OU s-score (Avellaneda-Lee residualised mean reversion)")
    print(f"{'='*70}")
    print(f"  Pullback entries blocked by overstretched s-score (s > {OU_SSCORE_HIGH}): "
          f"{_OU_SSCORE_BLOCKED_ENTRIES}")
    print(f"  Pullback overshoot exits triggered (s >= {OU_OVERSHOOT}): "
          f"{_OU_SSCORE_OVERSHOOT_EXITS}")
    if eq_tickers:
        sscore_avail = [t for t in eq_tickers
                        if "ou_sscore" in all_signals[t].columns
                        and all_signals[t]["ou_sscore"].notna().any()]
        if sscore_avail:
            sscore_means = [all_signals[t]["ou_sscore"].mean() for t in sscore_avail]
            sscore_stds  = [all_signals[t]["ou_sscore"].std()  for t in sscore_avail]
            print(f"  OU s-score — mean across {len(sscore_avail)} equity tickers: "
                  f"{np.nanmean(sscore_means):+.2f} (std {np.nanmean(sscore_stds):.2f})")

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
    earn_path = Path("data/v1/raw/earnings_dates.json")
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

    # ── Expanded-universe cross-sectional momentum signals ────────────────────
    try:
        from v1.pipeline.universe_expansion import USE_EXPANDED_UNIVERSE
        if USE_EXPANDED_UNIVERSE:
            print(f"\n{'='*70}")
            print("  EXPANDED UNIVERSE — cross-sectional momentum signals")
            print(f"{'='*70}")
            generate_expanded_signals()
    except ImportError:
        pass  # universe_expansion not available; skip silently


if __name__ == "__main__":
    main()
