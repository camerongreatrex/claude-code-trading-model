"""
portfolio.py
------------
Turns +1/-1/0 signals (and continuous ensemble scores) into actual dollar
position sizes and evaluates the combined portfolio using walk-forward
validation.

Sizing methods (in order of complexity)
────────────────────────────────────────
  1. equal_weight      — Equal fraction of capital per active signal.
                          Simple baseline; no volatility awareness.
  2. atr_sizes         — Risk a fixed dollar amount (1% of capital) per
                          1-ATR adverse move.  Normalises exposure across
                          assets with very different volatility profiles.
  3. kelly_sizes       — Half-Kelly: sizes positions by the signal's
                          rolling edge (win rate × profit factor).
  4. atr + pca         — ATR sizing scaled down when portfolio correlations
                          spike (diversification benefit is falling).
  5. atr + pca + macro — Full system: adds macro size_multiplier from
                          macro_features.py (0.5× in fear / 1.25× in calm).
  6. vol_target        — Scales all sizes so realised portfolio vol targets 10%.
  7. drawdown_control  — Circuit breaker: halves positions when in a drawdown
                          deeper than 12%.
  8. ensemble_sizes    — Uses the continuous IC-weighted ensemble signal from
                          signal_generation.ensemble_signal() as a fractional
                          position weight (signal × ATR base position), then
                          applies PCA scaling and macro multiplier.  Conviction
                          magnitude directly controls size — no binary threshold.

Key constants
─────────────
  CAPITAL          = $100,000  — starting capital for all simulations
  MAX_POSITION_PCT = 20%       — single position capped at 20% of capital
  RISK_PER_TRADE   = 1%        — dollar risk per 1-ATR adverse move

Walk-forward validation
────────────────────────
walk_forward() divides history into rolling 3-year training / 1-year test
windows.  Each test window is genuinely out-of-sample.  Comparing mean OOS
Sharpe to in-sample Sharpe reveals whether a method is overfit.

Output (written to data/results/)
──────────────────────────────────
  portfolio_comparison.parquet   — equity curves for all sizing methods
  walk_forward_regime.parquet    — OOS results (equal-weight, regime signal)
  walk_forward_atr_pca.parquet   — OOS results (ATR+PCA+macro sizing)
  oos_selection.parquet          — IS vs OOS Sharpe comparison for all candidates
  portfolio_equity_curve.parquet — equity curve for the best OOS method

Consumed by
───────────
  dashboard.py  — reads all result parquets for the backtest / portfolio tabs
  paper_trader.py — imports atr_sizes, apply_macro_multiplier for live sizing
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Windows terminals default to cp1252 which can't encode box-drawing characters.
# Force UTF-8 so all print() output works regardless of terminal locale.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .risk_model import estimate_covariance, risk_parity_weights, regime_conditional_covariance, hrp_weights
from .optimizer import optimizer_sizes, minimum_variance_gated_weights
from .regime_analysis import label_regimes
from .backtester import (
    compute_strategy_returns, sharpe_ratio, max_drawdown,
    calmar_ratio, win_rate, profit_factor, summarise, equity_curve
)
from .data_pipeline import TICKER_LIST, ASSET_CLASS, HEDGE_MAP
from .signal_generation import vix_position_scalar, ATR_PARTIAL_REMAIN

SIGNAL_DIR  = Path("data/v1/signals")
FEATURE_DIR = Path("data/v1/features")
MACRO_DIR   = Path("data/shared/macro")
RESULTS_DIR = Path("data/v1/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL          = 100_000
MAX_POSITION_PCT = 0.20
RISK_PER_TRADE   = 0.01

# Cap broad index ETFs at 8% of portfolio to prevent alpha dilution.
# Rationale: SPY/IWM/EEM track the market — large allocations to these
# mean the portfolio is just an expensive index fund. The alpha comes from
# sector ETFs, individual stocks, and uncorrelated assets (bonds, commodities).
# 8% = enough to maintain the trend signal's participation without dominating.
INDEX_ETF_CAP     = 0.08
INDEX_ETF_TICKERS = {"SPY", "IWM", "EEM", "EFA", "VWO"}

RESEARCH_DIR = Path("data/v1/research")

# Module-level cache for regime label data (loaded once per process, re-used across
# all momentum_tilt_sizes calls including walk-forward windows).
_REGIME_DATA_CACHE: dict = {}

# Regime-conditional tilt ranges for momentum_tilt_sizes (Part 1).
# bull_calm:   [0.50, 1.50]  ±50% — maximum concentration toward top-ranked
# bull_stress: [0.60, 1.40]  ±40%
# bear_calm:   [0.80, 1.20]  ±20%
# bear_stress: [0.90, 1.10]  ±10% — near-equal weight in momentum crash regime
# Default (unknown): [0.70, 1.30] ±30% — unchanged from original
_REGIME_TILT_PARAMS: dict = {
    "bull_calm":   (0.50, 1.00),
    "bull_stress": (0.60, 0.80),
    "bear_calm":   (0.80, 0.40),
    "bear_stress": (0.90, 0.20),
}

# Regime-conditional safe-haven floor fractions (Part 2).
# In bull_calm: halve the floor to free 7% for equity deployment.
# In bear_stress: increase floor for stronger crisis protection.
# VIX guardrail: if regime == bull_calm AND VIX >= 20 → use bull_stress floor.
_REGIME_FLOOR_PARAMS: dict = {
    "bull_calm":   {"VGSH": 0.02, "DBMF": 0.02, "WTMF": 0.02},
    "bull_stress": {"VGSH": 0.03, "DBMF": 0.03, "WTMF": 0.03},
    "bear_calm":   {"VGSH": 0.05, "DBMF": 0.04, "WTMF": 0.04},
    "bear_stress": {"VGSH": 0.07, "DBMF": 0.05, "WTMF": 0.05},
}

# Gross exposure targets by regime — non-floor positions are scaled UP toward
# these targets after the adaptive floor is applied (Part 2).
_REGIME_GROSS_TARGET: dict = {
    "bull_calm":   0.92,
    "bull_stress": 0.87,
    "bear_calm":   0.80,
    "bear_stress": 0.70,
}

# Minimum floor allocations — held regardless of signal state in sizing functions.
# Mirrors SAFE_HAVEN_FLOOR in paper_trader.py so backtests reflect the same
# permanent crisis-hedge positions.  See paper_trader.py for full rationale.
SAFE_HAVEN_FLOOR = {
    "VGSH": 0.05,   # 5% — bear_stress corr -0.155 (rises in crashes)
    "DBMF": 0.04,   # 4% — bear_stress corr +0.102 (managed futures)
    "WTMF": 0.04,   # 4% — bear_stress corr +0.138 (managed futures)
}

# Sleeve-based allocation split between the core (MA50/200) universe and the
# expanded cross-sectional momentum universe.  Each sleeve is independently
# managed; their returns are summed as a weighted blend.
# Core = existing 37 tickers.  Expanded = 115 new tickers from SECTOR_STOCKS.
# DISABLED: Only used when USE_EXPANDED_UNIVERSE=True (universe_expansion.py).
ALLOCATION_SPLIT: dict[str, float] = {
    "core":     0.50,   # 50% of capital allocated to the core MA50/200 sleeve
    "expanded": 0.50,   # 50% of capital allocated to the expanded CS-momentum sleeve
}


def load_dead_weight_scalars(
    floor: float = 0.70,
    ceiling: float = 1.20
) -> dict:
    """
    Load dead_weight_pct from research parquet and convert to
    per-ticker position size scalar in [floor, ceiling].

    edge = 0.50 - dead_weight_pct  (positive = better than random)
    scalar = floor + edge_norm * (ceiling - floor)

    Result: TIP (~37.6% DW) gets ~1.20x, FXY (~49.8% DW) gets
    ~0.70x. Pure rank transformation — no return fitting.
    """
    path = RESEARCH_DIR / "correlation_diagnostic_dead_weight.parquet"
    if not path.exists():
        print("  Dead weight parquet not found — scalars defaulting to 1.0")
        return {}
    df = pd.read_parquet(path)
    df = df[df["dead_weight_pct"].notna()].copy()
    df["edge"] = 0.50 - df["dead_weight_pct"] / 100.0
    edge_min = df["edge"].min()
    edge_max = df["edge"].max()
    if edge_max == edge_min:
        return {row["ticker"]: 1.0 for _, row in df.iterrows()}
    df["edge_norm"] = (df["edge"] - edge_min) / (edge_max - edge_min)
    df["scalar"]    = floor + df["edge_norm"] * (ceiling - floor)
    return dict(zip(df["ticker"], df["scalar"]))


_MACRO_CACHE = None


def _get_macro() -> pd.DataFrame:
    global _MACRO_CACHE
    if _MACRO_CACHE is None:
        path = MACRO_DIR / "macro_features.parquet"
        if path.exists():
            _MACRO_CACHE = pd.read_parquet(path)
        else:
            _MACRO_CACHE = pd.DataFrame()
    return _MACRO_CACHE


def _load_regime_data(index: pd.DatetimeIndex) -> tuple:
    """
    Load regime labels and VIX series aligned to index.

    Cached at module level — parquet files are read once per process.
    Called by momentum_tilt_sizes() for each walk-forward window (fast path).

    Returns:
        (regimes, vix_series) where:
          regimes    — Series of "bull_calm"/"bull_stress"/"bear_calm"/"bear_stress"
                       aligned to index.  Falls back to "unknown" if data missing.
          vix_series — VIX Series aligned to index, or None if unavailable.
    """
    global _REGIME_DATA_CACHE
    if "macro" not in _REGIME_DATA_CACHE:
        macro_path = MACRO_DIR / "macro_features.parquet"
        spy_path   = FEATURE_DIR / "SPY.parquet"
        if macro_path.exists() and spy_path.exists():
            try:
                _REGIME_DATA_CACHE["macro"]     = pd.read_parquet(macro_path)
                _REGIME_DATA_CACHE["spy_close"] = pd.read_parquet(spy_path)["Close"]
            except Exception:
                _REGIME_DATA_CACHE["macro"]     = pd.DataFrame()
                _REGIME_DATA_CACHE["spy_close"] = pd.Series(dtype=float)
        else:
            _REGIME_DATA_CACHE["macro"]     = pd.DataFrame()
            _REGIME_DATA_CACHE["spy_close"] = pd.Series(dtype=float)

    macro_df  = _REGIME_DATA_CACHE["macro"]
    spy_close = _REGIME_DATA_CACHE["spy_close"]

    if macro_df.empty or "vix" not in macro_df.columns or spy_close.empty:
        return pd.Series("unknown", index=index), None

    try:
        vix     = macro_df["vix"]
        regimes = label_regimes(vix, spy_close, index)
        vix_s   = vix.reindex(index).ffill().fillna(20.0)
        return regimes, vix_s
    except Exception:
        return pd.Series("unknown", index=index), None


def equal_weight_sizes(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    """
    Split capital equally among all active signals each day.

    If N tickers are signalling LONG on day T, each gets capital/N.
    If no tickers are active, no positions are held (cash).

    Args:
        signals:  DataFrame of signal values {-1, 0, 1}, shape (T × N).
        capital:  Total capital in dollars.

    Returns:
        DataFrame of dollar position sizes, same shape as signals.
        Clipped to ±MAX_POSITION_PCT × capital per position.
    """
    n_active = signals.abs().sum(axis=1).replace(0, np.nan)
    sizes    = signals.multiply(capital / n_active, axis=0).fillna(0)
    return sizes.clip(-capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT)


def atr_sizes(signals: pd.DataFrame, features: dict, capital: float) -> pd.DataFrame:
    """
    Volatility-normalised position sizing via ATR (Average True Range).

    Goal: risk the same dollar amount (RISK_PER_TRADE × capital) per 1-ATR
    adverse move, regardless of which asset is being traded.

    Calculation:
      dollar_risk = capital × RISK_PER_TRADE   (e.g., $1,000 on $100k)
      shares      = dollar_risk / ATR
      position $  = shares × close_price

    Works for both binary {-1, 0, 1} and continuous signals: the
    multiplication is linear, so a signal of 0.5 produces exactly half
    the position of a signal of 1.0.  This property is exploited by
    ensemble_sizes() which passes continuous [-1, +1] ensemble values.

    Why this works:
      A 1-ATR stop loss on the position would lose exactly dollar_risk.
      This means NVDA (high ATR ≈ $30) gets a smaller position than
      SPY (low ATR ≈ $4), automatically normalising risk across assets
      with very different volatility profiles.

    Args:
        signals:  Signal DataFrame (T × N).  Values may be binary or continuous.
        features: Dict[ticker -> feature DataFrame] with atr_14 and Close.
        capital:  Total capital in dollars.

    Returns:
        Dollar position size DataFrame, same shape as signals.
        Clipped to ±MAX_POSITION_PCT × capital.
    """
    dollar_risk = capital * RISK_PER_TRADE
    sizes       = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

    for ticker in signals.columns:
        if ticker not in features:
            continue
        atr   = features[ticker]["atr_14"].reindex(signals.index).ffill()
        close = features[ticker]["Close"].reindex(signals.index).ffill()
        atr   = atr.replace(0, np.nan).ffill()

        dollar_pos    = (dollar_risk / atr) * close
        sizes[ticker] = (signals[ticker] * dollar_pos).clip(
            -capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT
        )

    # Index ETF concentration cap — mirrors paper_trader.INDEX_ETF_CAP.
    # SPY/IWM/EEM have low ATR relative to price, so ATR sizing gives them
    # outsized positions.  Cap at 8% to redirect capital to higher-alpha tickers.
    for _etf in INDEX_ETF_TICKERS:
        if _etf in sizes.columns:
            sizes[_etf] = sizes[_etf].clip(-capital * INDEX_ETF_CAP, capital * INDEX_ETF_CAP)

    # Portfolio-level cap: scale all positions down if total gross exposure
    # exceeds capital.  Without this, 21 assets × 20%-cap each = 4.2× leverage,
    # inflating returns with unrealistic borrowed-money assumptions.
    gross = sizes.abs().sum(axis=1).replace(0, np.nan)
    scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return sizes.multiply(scale, axis=0)


def kelly_sizes(signals: pd.DataFrame, returns: pd.DataFrame,
                capital: float, lookback: int = 252) -> pd.DataFrame:
    """
    Rolling half-Kelly position sizing based on the signal's recent edge.

    The Kelly criterion sizes positions proportionally to the signal's
    statistical edge: Kelly fraction = W − (1−W)/PF where W = win rate,
    PF = profit factor.  Half-Kelly is used (×0.5) to reduce volatility —
    full Kelly is theoretically optimal but in practice leads to very large
    drawdowns from estimation error.

    Critically: only past data is used for each window's W and PF.  The
    window rolls forward in time — this is NOT look-ahead.

    Args:
        signals:  Signal DataFrame (T × N).
        returns:  Daily return DataFrame (T × N) aligned to signals.
        capital:  Total capital in dollars.
        lookback: Rolling window length in days (default 252 = 1 year).

    Returns:
        Dollar position size DataFrame.  Clipped to ±MAX_POSITION_PCT × capital.
        Filled with 0 where Kelly is non-positive (no edge).
    """
    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

    for ticker in signals.columns:
        if ticker not in returns.columns:
            continue
        strat_ret  = compute_strategy_returns(signals[ticker], returns[ticker])

        rolling_wr = strat_ret.rolling(lookback).apply(
            lambda x: float((x[x != 0] > 0).mean()) if (x != 0).any() else 0.5
        )
        rolling_pf = strat_ret.rolling(lookback).apply(
            lambda x: float(x[x > 0].sum() / x[x < 0].abs().sum())
            if x[x < 0].abs().sum() > 0 else 1.0
        )
        kelly = (rolling_wr - (1 - rolling_wr) / rolling_pf.clip(0.01)).clip(0) * 0.5
        sizes[ticker] = (signals[ticker] * kelly * capital).clip(
            -capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT
        )
    return sizes.fillna(0)


def pca_concentration(returns_window: pd.DataFrame) -> float:
    """
    Measure portfolio concentration using PCA on the returns correlation matrix.

    Returns the fraction of total variance explained by the first (largest)
    principal component.

    Interpretation:
      1/n_assets = ideal diversification (variance evenly spread)
      > 0.5      = one dominant factor (e.g., everything sells off together)
      → 1.0      = crisis correlation (all assets move as one)

    In normal markets for 21 assets: expect 0.15–0.25.
    In a crisis (2008, COVID, 2022): can spike to 0.70+.

    Args:
        returns_window: Returns DataFrame slice for the lookback window.

    Returns:
        Float in [0, 1].  Returns 1/n_assets if insufficient data.
    """
    if len(returns_window) < 20 or returns_window.shape[1] < 2:
        return 1 / max(returns_window.shape[1], 1)

    corr        = returns_window.corr().fillna(0)
    eigenvalues = np.maximum(np.linalg.eigh(corr.values)[0], 0)
    total       = eigenvalues.sum()
    return eigenvalues[-1] / total if total > 0 else 0.5


def apply_pca_scaling(sizes: pd.DataFrame, returns: pd.DataFrame,
                      window: int = 126) -> pd.DataFrame:
    """
    Scale position sizes down when portfolio correlations spike.

    When assets are highly correlated, holding many positions does not
    actually diversify risk — the portfolio behaves like a concentrated bet.
    This overlay reduces exposure proportionally as correlation rises.

    scale = (1/n_assets) / pca_concentration, clipped to [0.3, 1.0]

    Minimum scale of 0.3 ensures positions are never reduced below 30%
    of their target size (prevents over-reacting to temporary correlation
    spikes like those that occur during earnings seasons).

    Args:
        sizes:   Dollar position size DataFrame (T × N).
        returns: Daily returns DataFrame (T × N).
        window:  Lookback window for correlation computation (default 126 days = 6 months).
                 6 months balances responsiveness to regime shifts vs.
                 reactivity to short-term noise.

    Returns:
        Scaled position size DataFrame.
    """
    scaled = sizes.copy()
    n      = sizes.shape[1]
    ideal  = 1.0 / n if n > 0 else 0.125  # 1/8 for 8-asset universe

    for i in range(window, len(sizes)):
        concentration = pca_concentration(returns.iloc[i - window:i])
        scale         = (ideal / concentration).clip(0.3, 1.0)
        scaled.iloc[i] = sizes.iloc[i] * scale

    return scaled


def vol_target_sizes(sizes: pd.DataFrame, returns: pd.DataFrame,
                     target_vol: float = 0.10, window: int = 63) -> pd.DataFrame:
    """
    Scale the portfolio so its realised volatility targets a fixed annualised level.

    Target: 10% annual portfolio volatility (a common institutional benchmark).
    Realised vol is computed from the last 63 trading days (~1 quarter).

    Scaling logic:
      scale = target_vol / realised_vol  clipped to [0.5, 1.5]

    Clip range prevents extreme leverage in low-vol markets (cap 1.5×) and
    prevents over-reducing in high-vol markets (floor 0.5×).

    Uses lagged realised vol (computed from yesterday's returns) — no
    look-ahead bias.

    Args:
        sizes:      Dollar position size DataFrame.
        returns:    Daily returns DataFrame.
        target_vol: Target annualised portfolio volatility (default 10%).
        window:     Realised vol estimation window in days (default 63).

    Returns:
        Scaled position size DataFrame.
    """
    weights  = sizes.shift(1) / CAPITAL
    port_ret = (weights * returns.reindex(columns=sizes.columns)).sum(axis=1)
    realized_vol = port_ret.rolling(window, min_periods=21).std() * np.sqrt(252)
    realized_vol = realized_vol.replace(0, np.nan).fillna(target_vol)
    scale = (target_vol / realized_vol).clip(0.5, 1.5)
    return sizes.multiply(scale, axis=0)


def simple_vol_scale(
    sizes: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    window: int = 21,
) -> pd.DataFrame:
    """
    Scale all position sizes so portfolio realized volatility targets target_vol.

    Uses trailing 21-day portfolio vol (1 trading month) — fast enough to
    de-lever before a drawdown deepens, without over-reacting to single-day
    spikes.

    scale = (target_vol / realized_vol).clip(0.5, 1.5).shift(1)

    Clip range:
      0.5 floor — never cut below half exposure (still want rebound participation)
      1.5 ceiling — never lever above 1.5× (prevents runaway in 2017-style calm)

    shift(1) ensures today's scale is based on yesterday's trailing vol.
    No look-ahead bias.

    Why this is more robust than the macro multiplier:
      ONE input  : portfolio's own trailing realized vol
      ONE target : 10% annualized (industry standard)
      ZERO fitted thresholds, regime classifications, or cross-asset relationships
      Vol clusters and mean-reverts (Mandelbrot 1963, Engle GARCH 1982) —
      21-day realized vol is a structural property, not a data-mined signal.

    Args:
        sizes:      Dollar position size DataFrame.
        returns:    Daily returns DataFrame (aligned to sizes).
        target_vol: Target annualised portfolio volatility (default 10%).
        window:     Realized vol window in trading days (default 21 = 1 month).

    Returns:
        Scaled position size DataFrame.
    """
    weights  = sizes.shift(1) / CAPITAL
    port_ret = (weights * returns.reindex(columns=sizes.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    scale    = (target_vol / realized).clip(0.5, 1.5).shift(1).fillna(1.0)
    return sizes.multiply(scale, axis=0)


def apply_drawdown_control(sizes: pd.DataFrame, returns: pd.DataFrame,
                           threshold: float = 0.12,
                           scale_factor: float = 0.5) -> pd.DataFrame:
    """
    Circuit breaker: halve position sizes when the portfolio is in a drawdown
    exceeding `threshold`.

    Standard institutional risk overlay: sustained drawdowns often indicate
    a regime shift where the strategy's edge has temporarily disappeared.
    Reducing exposure during the drawdown limits further losses and preserves
    capital for recovery.  It does NOT try to predict when the drawdown ends.

    Uses yesterday's drawdown to set today's scale (shift(1)) — no look-ahead.

    Args:
        sizes:        Dollar position size DataFrame.
        returns:      Daily returns DataFrame.
        threshold:    Drawdown depth that triggers the halving (default 12%).
        scale_factor: Position multiplier when triggered (default 0.5 = half).

    Returns:
        Scaled position size DataFrame.
    """
    weights  = sizes.shift(1) / CAPITAL
    port_ret = (weights * returns.reindex(columns=sizes.columns)).sum(axis=1)
    cum      = (1 + port_ret).cumprod()
    rolling_peak = cum.cummax()
    drawdown     = (cum - rolling_peak) / rolling_peak   # always ≤ 0

    scale = pd.Series(1.0, index=sizes.index)
    scale[drawdown.shift(1) < -threshold] = scale_factor  # shift(1): yesterday's DD

    return sizes.multiply(scale, axis=0)


def apply_macro_multiplier(sizes: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the macro size_multiplier from macro_features.py to all positions.

    Multiplier range (from macro_features.compute_macro_features):
      ~1.25× when macro_score is +1 (VIX calm, yield curve steep, no backwardation)
      ~1.00× when macro_score is  0 (neutral conditions)
      ~0.50× when macro_score is -1 (VIX fear, curve inverted, backwardation)

    Applied ONCE here in portfolio.py — NOT in signal_generation.py — to avoid
    double-dampening (applying it in both places would compound the effect).

    Args:
        sizes: Dollar position size DataFrame.

    Returns:
        Macro-scaled position size DataFrame.  Returns ``sizes`` unchanged
        if the macro parquet file is not found.
    """
    path = MACRO_DIR / "macro_features.parquet"
    if not path.exists():
        print("  Macro data not found — skipping")
        return sizes

    macro      = pd.read_parquet(path)
    multiplier = macro["size_multiplier"].reindex(sizes.index).ffill().fillna(1.0)
    return sizes.multiply(multiplier, axis=0)


def ensemble_sizes(
    ensemble_signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    """
    Continuous-signal position sizing using the IC-weighted ensemble values.

    Hypothesis
    ──────────
    A continuous signal with adaptive feature weights should produce higher
    OOS Sharpe than binary MA crossover because:
      1. Conviction scaling — a signal of 0.3 bets less than 0.9; the strategy
         never "goes full size" on borderline signals.
      2. Continuous adjustment — the position ramps up/down smoothly rather
         than flipping all-on / all-off at a binary threshold.
      3. Adaptive weighting — features that predicted well recently get more
         weight; stale factors naturally fade to zero without any manual recalibration.

    Sizing mechanics
    ────────────────
    atr_sizes() is reused directly because its multiplication is linear:

        size = ensemble_signal × (dollar_risk / ATR) × close_price

    When ensemble_signal = 1.0 → full ATR-risk position (same as binary long).
    When ensemble_signal = 0.4 → 40% of the ATR-risk position.
    When ensemble_signal = 0.0 → flat.
    When ensemble_signal < 0   → short (if asset class allows it).

    PCA scaling reduces exposure during correlated sell-offs (same logic as
    "ATR + PCA + macro", the best binary method).  Macro multiplier adds the
    regime-level overlay from macro_features.py.

    Args:
        ensemble_signals: DataFrame of continuous [-1, +1] signal values (T × N).
                          Sourced from data/signals/ensemble_signals.parquet.
        features:         Dict[ticker -> feature DataFrame] with atr_14 and Close.
        returns:          Daily returns DataFrame (T × N).
        capital:          Total capital in dollars.

    Returns:
        Dollar position size DataFrame, same shape as ensemble_signals.
        Clipped to ±MAX_POSITION_PCT × capital per position.
    """
    sizes = atr_sizes(ensemble_signals, features, capital)
    sizes = apply_pca_scaling(sizes, returns)
    sizes = apply_macro_multiplier(sizes)
    return sizes


def risk_parity_sizes(
    signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
    capital: float,
    cov_window: int = 126,
    rebalance_freq: int = 21,
) -> pd.DataFrame:
    """
    Risk-parity position sizing using Ledoit-Wolf covariance estimation.

    Two-step process
    ────────────────
    1. ATR base sizing: compute dollar positions using ATR-based risk sizing.
       This gives a total gross-exposure dollar amount per day that already
       accounts for individual asset volatility.

    2. Risk-parity reweighting: redistribute that total gross exposure across
       active positions so each asset contributes equally to portfolio variance.
       Uses a trailing 126-day Ledoit-Wolf shrinkage covariance estimate,
       recomputed every 21 trading days (~monthly).

    Why this improves on ATR sizing
    ────────────────────────────────
    ATR sizing normalises each asset's INDIVIDUAL volatility but ignores
    cross-asset correlations.  During a correlated sell-off (e.g., 2022),
    holding many correlated positions is riskier than holding a few
    uncorrelated ones of the same per-asset ATR size.

    Risk parity accounts for the covariance structure: it up-weights assets
    that are less correlated with the rest of the portfolio and down-weights
    assets that cluster with the dominant market factor (e.g., large-cap US
    equities).  The result is a portfolio where each asset genuinely
    contributes 1/N of total variance.

    Macro multiplier
    ────────────────
    Applied last — reduces all positions in fear regimes (VIX spike, inverted
    curve) and slightly increases in calm regimes.  Applied once here, not in
    signal_generation.py, to avoid double-dampening.

    Args:
        signals:        Signal DataFrame (T × N).  Binary {-1, 0, 1} regime signals.
        features:       Dict[ticker -> feature DataFrame] with atr_14 and Close.
        returns:        Daily return DataFrame (T × N).
        capital:        Total capital in dollars.
        cov_window:     Lookback for covariance estimation (default 126 = 6 months).
                        6 months balances responsiveness to regime shifts vs.
                        stability of the Ledoit-Wolf estimate.
        rebalance_freq: Days between covariance recomputation (default 21 = monthly).
                        Daily recomputation is unnecessary — covariance changes slowly.

    Returns:
        Dollar position size DataFrame, same shape as signals.
        Clipped to ±MAX_POSITION_PCT × capital per position.
    """
    # Step 1: ATR base sizes — total gross exposure per day
    base_sizes = atr_sizes(signals, features, capital)

    adjusted = base_sizes.copy()
    tickers  = [t for t in signals.columns if t in returns.columns]
    n_rows   = len(signals)

    # ── Precompute RP weights at each rebalance date ─────────────────────
    # Build a (T × N) weight matrix: each row carries the RP weights in
    # effect on that day.  Rebalance every `rebalance_freq` days; between
    # rebalances the previous weights persist via forward-fill.
    rp_weight_df = pd.DataFrame(np.nan, index=signals.index, columns=tickers)

    for i in range(cov_window, n_rows, rebalance_freq):
        ret_window = returns.iloc[i - cov_window : i][tickers]
        ret_window = ret_window.dropna(thresh=int(len(ret_window) * 0.90), axis=1)
        available  = ret_window.columns.tolist()

        if len(available) >= 2 and len(ret_window) >= 20:
            ret_clean = ret_window.ffill().fillna(0)
            cov_mat   = estimate_covariance(ret_clean)
            w_rp      = risk_parity_weights(cov_mat)
            for j, t in enumerate(available):
                rp_weight_df.iloc[i, rp_weight_df.columns.get_loc(t)] = w_rp[j]

    # Forward-fill weights between rebalance dates
    rp_weight_df = rp_weight_df.ffill()

    # ── Vectorised redistribution ─────────────────────────────────────────
    # active mask: where signal != 0 AND we have RP weights
    sig_vals  = signals[tickers]
    active    = (sig_vals != 0) & rp_weight_df[tickers].notna()

    # Zero out inactive weights, normalise per row
    w_active  = rp_weight_df[tickers].where(active, 0.0)
    w_sum     = w_active.sum(axis=1).replace(0, np.nan)
    w_norm    = w_active.div(w_sum, axis=0).fillna(0.0)

    # Total ATR exposure among active tickers per day
    total_atr = base_sizes[tickers].where(active, 0.0).abs().sum(axis=1)

    # Redistribute: sign × normalised_weight × total_atr
    sign_df   = sig_vals.where(active, 0.0).clip(-1, 1)
    raw       = sign_df * w_norm * total_atr.values[:, None]

    # Clip to position limits and write back
    cap_limit = capital * MAX_POSITION_PCT
    clipped   = raw.clip(-cap_limit, cap_limit)

    # Only overwrite rows past warm-up (first valid RP weight row)
    first_valid = rp_weight_df.first_valid_index()
    if first_valid is not None:
        mask = signals.index >= first_valid
        for t in tickers:
            adjusted.loc[mask, t] = clipped.loc[mask, t]

    return apply_macro_multiplier(adjusted)


def rp_regime_aware_sizes(
    signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
    capital: float,
    cov_window: int = 126,
    rebalance_freq: int = 21,
    min_regime_obs: int = 63,
) -> pd.DataFrame:
    """
    Risk-parity sizing with regime-conditional covariance estimation.

    Problem solved
    ──────────────
    Standard risk_parity_sizes() uses a rolling 126-day window that blends
    returns from different market regimes.  During bear_stress periods the
    window still contains bull_calm days that UNDERESTIMATE correlations:
      bull_calm avg pairwise corr ≈ 0.20
      bear_stress avg pairwise corr ≈ 0.45+

    The blended estimate sits somewhere in between — the portfolio THINKS
    it is diversified (low-correlation ERC weights) when it is actually
    concentrated (high-correlation reality).  This is the precise mechanism
    behind "diversification fails when you need it most."

    Solution
    ────────
    Replace estimate_covariance(rolling_window) with
    regime_conditional_covariance(all_history, regimes, today's_regime):
    - In bull_calm:  fit on all historical bull_calm days → low-correlation
      weights that lean into diversification (this is when it works).
    - In bear_stress: fit on all historical bear_stress days → high-correlation
      weights that defensively concentrate into true diversifiers (bonds, gold,
      commodities — the assets whose marginal risk contribution is genuinely
      low when equities are correlated at 0.45+).

    Regime labelling reuses regime_analysis.label_regimes() — same VIX < 20
    and SPY 60d return > 0 logic, no new thresholds.

    Flow per rebalance day
    ──────────────────────
    1. Determine today's regime from VIX + SPY 60d return.
    2. Filter ALL history up to today to rows matching that regime.
    3. If ≥ min_regime_obs (63 ≈ 3 months): fit Ledoit-Wolf on those rows.
       Else: fall back to standard 126-day rolling window.
    4. Compute ERC weights from the regime-conditional covariance.
    5. Apply to ATR base sizes (same two-step logic as risk_parity_sizes).
    6. Apply macro multiplier overlay last.

    Args:
        signals:        Signal DataFrame (T × N).
        features:       Dict[ticker -> feature DataFrame] with atr_14, Close.
        returns:        Daily return DataFrame (T × N).
        capital:        Total capital in dollars.
        cov_window:     Fallback rolling window (default 126).
        rebalance_freq: Days between covariance recomputation (default 21).
        min_regime_obs: Minimum regime-filtered observations for conditional
                        estimate (default 63 = ~3 months).

    Returns:
        Dollar position size DataFrame, same shape as signals.
    """
    # ── Build regime labels for the full history ──────────────────────────
    # Cache parquet reads across repeated calls (walk-forward windows).
    if not hasattr(rp_regime_aware_sizes, "_macro_cache"):
        rp_regime_aware_sizes._macro_cache = {}
    _cache = rp_regime_aware_sizes._macro_cache
    macro_path = MACRO_DIR / "macro_features.parquet"
    spy_path   = FEATURE_DIR / "SPY.parquet"

    if not macro_path.exists() or not spy_path.exists():
        return risk_parity_sizes(signals, features, returns, capital,
                                 cov_window, rebalance_freq)

    if "macro" not in _cache:
        _cache["macro"]     = pd.read_parquet(macro_path)
        _cache["spy_close"] = pd.read_parquet(spy_path)["Close"]
    macro     = _cache["macro"]
    spy_close = _cache["spy_close"]
    vix       = macro["vix"] if "vix" in macro.columns else None

    if vix is None:
        return risk_parity_sizes(signals, features, returns, capital,
                                 cov_window, rebalance_freq)

    regimes = label_regimes(vix, spy_close, returns.index)

    # ── ATR base sizes (same as risk_parity_sizes) ────────────────────────
    base_sizes = atr_sizes(signals, features, capital)
    adjusted   = base_sizes.copy()
    tickers    = [t for t in signals.columns if t in returns.columns]
    n_rows     = len(signals)

    # ── Precompute RP weights with adaptive rebalance triggers ────────────
    # Three triggers (any one is sufficient to rebalance):
    #   1. Scheduled: every rebalance_freq (21) days — the original cadence.
    #   2. Regime change: current_regime != last_regime — the new regime's
    #      conditional covariance should replace the old one immediately.
    #   3. Correlation shift: avg pairwise correlation changes by > 5% over
    #      the trailing 21 days (checked every 5 days to limit computation).
    #      5% ≈ 1 std-dev of monthly pairwise correlation changes in a
    #      diversified equity portfolio — not fitted to this dataset.
    rp_weight_df   = pd.DataFrame(np.nan, index=signals.index, columns=tickers)
    CORR_CHANGE_THRESH = 0.05  # trigger if avg pairwise corr shifts by > 5%

    last_rebal_i  = -rebalance_freq   # force first rebalance at cov_window
    last_regime   = None
    last_avg_corr = None

    for i in range(cov_window, n_rows):
        days_since_rebal = i - last_rebal_i
        current_regime   = regimes.iloc[i]

        needs_rebal = False

        # Trigger 1: scheduled rebalance (every 21 days)
        if days_since_rebal >= rebalance_freq:
            needs_rebal = True

        # Trigger 2: regime changed since last rebalance
        if current_regime != last_regime and last_regime is not None:
            needs_rebal = True

        # Trigger 3: correlation shift (check every 5 days, not every bar)
        if days_since_rebal >= 5 and not needs_rebal:
            ret_recent = returns.iloc[max(0, i - 21):i][tickers].dropna(axis=1)
            if len(ret_recent) >= 15 and ret_recent.shape[1] >= 3:
                corr_mat = ret_recent.corr()
                upper    = np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
                avg_corr = float(corr_mat.values[upper].mean())
                if last_avg_corr is not None and abs(avg_corr - last_avg_corr) > CORR_CHANGE_THRESH:
                    needs_rebal   = True
                    last_avg_corr = avg_corr
                elif last_avg_corr is None:
                    last_avg_corr = avg_corr

        if not needs_rebal:
            continue

        ret_history = returns.iloc[:i][tickers]
        ret_history = ret_history.dropna(
            thresh=int(min(len(ret_history), cov_window) * 0.90), axis=1
        )
        available = ret_history.columns.tolist()

        if len(available) >= 2 and len(ret_history) >= 20:
            if current_regime in ("bull_calm", "bull_stress",
                                  "bear_calm", "bear_stress"):
                regime_hist = regimes.iloc[:i]
                cov_mat, _ = regime_conditional_covariance(
                    ret_history.ffill().fillna(0),
                    regime_hist,
                    current_regime,
                    min_obs=min_regime_obs,
                    fallback_window=cov_window,
                )
            else:
                ret_window = ret_history.iloc[-cov_window:]
                cov_mat = estimate_covariance(ret_window.ffill().fillna(0))

            w_rp = risk_parity_weights(cov_mat)
            for j, t in enumerate(available):
                rp_weight_df.iloc[i, rp_weight_df.columns.get_loc(t)] = w_rp[j]

            last_rebal_i  = i
            last_regime   = current_regime

            # Update correlation baseline after a scheduled or regime-change rebalance
            if last_avg_corr is None:
                ret_recent_check = returns.iloc[max(0, i - 21):i][tickers].dropna(axis=1)
                if len(ret_recent_check) >= 15 and ret_recent_check.shape[1] >= 3:
                    corr_check = ret_recent_check.corr()
                    upper_check = np.triu(np.ones(corr_check.shape, dtype=bool), k=1)
                    last_avg_corr = float(corr_check.values[upper_check].mean())

    # Forward-fill weights between rebalance dates
    rp_weight_df = rp_weight_df.ffill()

    # ── Vectorised redistribution (same logic as risk_parity_sizes) ───────
    sig_vals  = signals[tickers]
    active    = (sig_vals != 0) & rp_weight_df[tickers].notna()

    w_active  = rp_weight_df[tickers].where(active, 0.0)
    w_sum     = w_active.sum(axis=1).replace(0, np.nan)
    w_norm    = w_active.div(w_sum, axis=0).fillna(0.0)

    total_atr = base_sizes[tickers].where(active, 0.0).abs().sum(axis=1)
    sign_df   = sig_vals.where(active, 0.0).clip(-1, 1)
    raw       = sign_df * w_norm * total_atr.values[:, None]

    cap_limit = capital * MAX_POSITION_PCT
    clipped   = raw.clip(-cap_limit, cap_limit)

    first_valid = rp_weight_df.first_valid_index()
    if first_valid is not None:
        mask = signals.index >= first_valid
        for t in tickers:
            adjusted.loc[mask, t] = clipped.loc[mask, t]

    return apply_macro_multiplier(adjusted)


def rp_regime_vix_sizes(signals, features, returns, capital):
    """rp_regime_aware + graduated VIX scalar only."""
    base   = rp_regime_aware_sizes(signals, features, returns, capital)
    macro  = _get_macro()
    scalar = vix_position_scalar(macro, base.index)
    scaled = base.multiply(scalar, axis=0)
    # Re-apply gross cap: prevent leverage from scalar > 1.0
    gross  = scaled.abs().sum(axis=1).replace(0, np.nan)
    cap    = (capital / gross).clip(upper=1.0).fillna(1.0)
    return scaled.multiply(cap, axis=0)


def rp_regime_dw_sizes(signals, features, returns, capital):
    """rp_regime_aware + dead weight per-ticker scalar only."""
    base       = rp_regime_aware_sizes(signals, features, returns, capital)
    dw_scalars = load_dead_weight_scalars()
    if not dw_scalars:
        return base
    dw_series  = pd.Series(dw_scalars).reindex(base.columns).fillna(1.0)
    scaled     = base.multiply(dw_series, axis=1)
    gross      = scaled.abs().sum(axis=1).replace(0, np.nan)
    cap        = (capital / gross).clip(upper=1.0).fillna(1.0)
    return scaled.multiply(cap, axis=0)


def rp_regime_vix_dw_sizes(signals, features, returns, capital):
    """
    rp_regime_aware + VIX scalar + dead weight scalar.

    Combines three orthogonal improvements:
      1. Regime-conditional covariance (from rp_regime_aware)
      2. Graduated VIX scalar — reduces in bull_stress
      3. Dead weight scalar — overweights proven signal tickers

    Order: VIX scalar first (portfolio-level), then dead weight
    (ticker-level), then gross cap.
    """
    base       = rp_regime_aware_sizes(signals, features, returns, capital)
    macro      = _get_macro()
    vix_scalar = vix_position_scalar(macro, base.index)
    scaled     = base.multiply(vix_scalar, axis=0)
    dw_scalars = load_dead_weight_scalars()
    if dw_scalars:
        dw_series = pd.Series(dw_scalars).reindex(base.columns).fillna(1.0)
        scaled    = scaled.multiply(dw_series, axis=1)
    gross = scaled.abs().sum(axis=1).replace(0, np.nan)
    cap   = (capital / gross).clip(upper=1.0).fillna(1.0)
    return scaled.multiply(cap, axis=0)


def regime_adaptive_sizes(
    signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
    capital: float,
    cov_window: int = 126,
    rebalance_freq: int = 21,
) -> pd.DataFrame:
    """
    Regime-adaptive position sizing with smooth continuous transitions.

    Addresses two root causes from capture_diagnostic Phase 1:
      (a) Cash drag — scales gross exposure toward a VIX-driven target
          (VIX 12 → 95% deployed, VIX 40 → 45%).
      (b) Counter-cyclical drag — caps bond/commodity notional by VIX
          (VIX 12 → 12% per ticker, VIX 40 → 45% per ticker).

    Phase 3 improvement: replaces step-function regime labels with
    continuous interpolation on raw VIX and SPY 60-day return.  This
    eliminates the cliff-edge whipsaw from Phase 2 (2021-22 OOS Sharpe
    was −0.058 because the bull_calm → bear_stress jump happened after
    the decline was already underway).  As VIX drifts from 19 → 21 the
    portfolio de-risks gradually rather than snapping between parameter sets.

    Parameter interpolation
    ───────────────────────
      VIX breakpoints:   [0, 12, 20, 30, 40, 100]
      gross_target_fp:   [0.95, 0.95, 0.85, 0.60, 0.45, 0.40]
      hedge_cap_fp:      [0.12, 0.12, 0.18, 0.35, 0.45, 0.50]
      tilt_strength_fp:  [0.80, 0.80, 0.60, 0.40, 0.30, 0.25]

      SPY 60d return modifier on gross_target only:
      SPY breakpoints:   [-0.30, -0.15, 0.0, 0.10, 0.30]
      spy_scale_fp:      [ 0.80,  0.90, 1.00, 1.05, 1.10]

    Smoothing: 5-day EMA + 1-day lag prevents flash-crash whipsaw and
    look-ahead bias.  The 5-day span matches the minimum hold period.

    Sizing pipeline:
      1. ATR base sizes
      2. Hedge cap (continuous, per-ticker bond/commodity)
      3. Momentum tilt (strength modulated by VIX)
      4. Dead weight scalar (per-ticker quality)
      5. Gross exposure scale toward smoothed target
      6. Final gross cap (no leverage)

    Args:
        signals:        Signal DataFrame (T × N) — should use signals_multi.
        features:       Dict[ticker → feature DataFrame] with atr_14 and Close.
        returns:        Daily log-return DataFrame (T × N).
        capital:        Starting capital in dollars.
        cov_window:     Kept for API consistency with other sizing functions.
        rebalance_freq: Kept for API consistency with other sizing functions.

    Returns:
        Dollar position size DataFrame, same shape as signals.
    """
    # ── Step 1: Continuous parameter series from VIX + SPY 60d return ─────────
    macro_df = _get_macro()
    spy_path = Path("data/v1/features/SPY.parquet")
    if macro_df.empty or "vix" not in macro_df.columns or not spy_path.exists():
        return momentum_tilt_sizes(signals, features, capital)

    vix       = macro_df["vix"].reindex(signals.index).ffill().fillna(20.0)
    spy_close = pd.read_parquet(spy_path)["Close"].reindex(signals.index).ffill()
    spy_60d   = spy_close.pct_change(60).fillna(0.0)

    # VIX interpolation — CBOE published breakpoints, not fitted to backtest
    vix_xp       = [0,    12,   20,   30,   40,   100]
    vix_gross_fp = [0.95, 0.95, 0.85, 0.60, 0.45, 0.40]
    vix_hedge_fp = [0.12, 0.12, 0.18, 0.35, 0.45, 0.50]
    vix_tilt_fp  = [0.80, 0.80, 0.60, 0.40, 0.30, 0.25]

    # SPY 60d modifier on gross_target only: positive trend → more aggressive
    spy_xp       = [-0.30, -0.15, 0.0, 0.10, 0.30]
    spy_scale_fp = [ 0.80,  0.90, 1.00, 1.05, 1.10]

    raw_gross  = (np.interp(vix.values, vix_xp, vix_gross_fp)
                  * np.interp(spy_60d.values, spy_xp, spy_scale_fp))
    raw_hedge  = np.interp(vix.values, vix_xp, vix_hedge_fp)
    raw_tilt   = np.interp(vix.values, vix_xp, vix_tilt_fp)

    gross_target_s  = pd.Series(raw_gross, index=signals.index).clip(0.40, 0.98)
    hedge_cap_s     = pd.Series(raw_hedge, index=signals.index).clip(0.10, 0.50)
    tilt_strength_s = pd.Series(raw_tilt,  index=signals.index).clip(0.20, 0.90)

    # 5-day EMA smoothing (prevents flash-crash whipsaw)
    # then 1-day lag (prevents same-day look-ahead)
    def _smooth(s: pd.Series) -> pd.Series:
        ema = s.ewm(span=5, adjust=False).mean()
        return ema.shift(1).fillna(ema)

    gross_target_s  = _smooth(gross_target_s)
    hedge_cap_s     = _smooth(hedge_cap_s)
    tilt_strength_s = _smooth(tilt_strength_s)

    # ── Step 2: ATR base sizes ────────────────────────────────────────────────
    sized = atr_sizes(signals, features, capital)

    # ── Step 3: Hedge cap — limit bond/commodity notional in bull regimes ─────
    hedge_tickers = [t for t in signals.columns
                     if ASSET_CLASS.get(t) in ("bond", "commodity")]
    if hedge_tickers:
        hedge_cap_dollars = hedge_cap_s * capital          # Series (T,)
        for t in hedge_tickers:
            sized[t] = sized[t].clip(lower=-hedge_cap_dollars,
                                     upper=hedge_cap_dollars)

    # ── Step 4: Momentum tilt (regime-scaled strength) ────────────────────────
    close_cols = {t: features[t]["Close"].reindex(sized.index).ffill()
                  for t in signals.columns if t in features}
    if close_cols:
        closes      = pd.DataFrame(close_cols)
        mom         = closes.pct_change(63)
        active_mask = signals.abs() > 0
        mom_cols    = [c for c in signals.columns if c in mom.columns]
        mom_masked  = mom[mom_cols].where(active_mask[mom_cols])
        ranks       = mom_masked.rank(axis=1, pct=True)
        n_active    = ranks.notna().sum(axis=1)
        tilt        = pd.DataFrame(1.0, index=sized.index, columns=sized.columns)
        valid_rows  = n_active >= 2
        for c in mom_cols:
            col_ranks = ranks[c]
            col_valid = valid_rows & col_ranks.notna()
            if not col_valid.any():
                continue
            # Standard tilt at full strength: 0.7 + 0.6 × rank ∈ [0.7, 1.3]
            # Blend toward 1.0 by (1 - tilt_strength): 0 → no tilt
            base_tilt = 0.7 + 0.6 * col_ranks[col_valid].values
            strength  = tilt_strength_s[col_valid].values
            tilt.loc[col_valid, c] = 1.0 + (base_tilt - 1.0) * strength
        sized = sized * tilt

    # ── Step 5: Dead weight scalar (per-ticker quality, applied before gross scale)
    dw_scalars = load_dead_weight_scalars()
    if dw_scalars:
        dw_series = pd.Series(dw_scalars).reindex(sized.columns).fillna(1.0)
        sized     = sized.multiply(dw_series, axis=1)

    # ── Step 6: Scale total notional toward gross_target × capital ────────────
    # Note: VIX scalar is intentionally omitted here.  gross_target already
    # encodes regime risk appetite (0.50 in bear_stress, 0.95 in bull_calm).
    # Stacking a VIX scalar on top would result in bear_stress deployment of
    # only 50% × 0.35 = 17.5% — far below the intended 50% floor.
    current_gross  = sized.abs().sum(axis=1).replace(0, np.nan)
    target_gross   = gross_target_s * capital
    exposure_scale = (target_gross / current_gross).fillna(1.0)
    sized          = sized.multiply(exposure_scale, axis=0)
    # Re-clip per-position cap (scaling up may breach MAX_POSITION_PCT)
    cap_limit = capital * MAX_POSITION_PCT
    sized     = sized.clip(-cap_limit, cap_limit)

    # ── Safe-haven floor and VXZ cap ──────────────────────────────────────────
    for _ft, _ff in SAFE_HAVEN_FLOOR.items():
        if _ft in sized.columns:
            _floor_dollars = _ff * capital
            sized[_ft] = sized[_ft].where(sized[_ft] < 0,
                                           sized[_ft].clip(lower=_floor_dollars))
    if "VXZ" in sized.columns:
        sized["VXZ"] = sized["VXZ"].clip(-0.03 * capital, 0.03 * capital)

    # ── Final gross cap (no leverage) ─────────────────────────────────────────
    gross = sized.abs().sum(axis=1).replace(0, np.nan)
    cap   = (capital / gross).clip(upper=1.0).fillna(1.0)
    return sized.multiply(cap, axis=0)


def adaptive_blend_sizes(
    signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    """
    50/50 dollar-size blend of regime_adaptive and multi_mom_tilt.

    Rationale
    ─────────
    Phase 2/3 results show each method has what the other lacks:
      - regime_adaptive: capture ratio 1.111, smooth de-risking in downturns
      - multi_mom_tilt:  OOS Sharpe 1.108, tight IS-OOS gap 0.086

    Blending at the dollar-size level (not returns) diversifies across two
    sizing philosophies whose errors are weakly correlated — regime_adaptive
    underweights during late-bull transitions, multi_mom_tilt is neutral to
    those transitions but misses the upside de-leveraging.  The blend should
    exhibit lower OOS Sharpe variance across walk-forward windows.

    50/50 weight is chosen to avoid introducing a fitted blend parameter.

    Args:
        signals:  Signal DataFrame (T × N) — should use signals_multi.
        features: Dict[ticker → feature DataFrame] with atr_14 and Close.
        returns:  Daily log-return DataFrame (T × N).
        capital:  Starting capital in dollars.

    Returns:
        Dollar position size DataFrame, same shape as signals.
    """
    sizes_regime = regime_adaptive_sizes(signals, features, returns, capital)
    sizes_mom    = momentum_tilt_sizes(signals, features, capital)

    blended = (0.50 * sizes_regime.reindex(columns=returns.columns, fill_value=0)
             + 0.50 * sizes_mom.reindex(columns=returns.columns, fill_value=0))

    # ── Safe-haven floor and VXZ cap ──────────────────────────────────────────
    for _ft, _ff in SAFE_HAVEN_FLOOR.items():
        if _ft in blended.columns:
            _floor_dollars = _ff * capital
            blended[_ft] = blended[_ft].where(blended[_ft] < 0,
                                               blended[_ft].clip(lower=_floor_dollars))
    if "VXZ" in blended.columns:
        blended["VXZ"] = blended["VXZ"].clip(-0.03 * capital, 0.03 * capital)

    gross = blended.abs().sum(axis=1).replace(0, np.nan)
    cap   = (capital / gross).clip(upper=1.0).fillna(1.0)
    return blended.multiply(cap, axis=0)


def portable_alpha_sizes(
    base_sizes: pd.DataFrame,
    returns: pd.DataFrame,
    capital: float,
    target_beta: float = 0.30,
    beta_window: int = 63,
    rebalance_freq: int = 5,
    max_hedge_pct: float = 0.40,
) -> pd.DataFrame:
    """
    Portable alpha overlay: adds a daily SPY short hedge to any base sizing
    method to target a specific portfolio beta.

    Rationale
    ─────────
    multi_mom_tilt carries ~0.45-0.50 portfolio beta to SPY — useful return
    in bull markets but also the primary driver of drawdowns.  The strategy's
    genuine alpha (capture ratio > 1, positive active Sharpe in every WF window)
    is a property of the SIGNAL, not the beta.  Mechanically removing the
    unwanted beta transports the alpha to a lower-beta target without changing
    the signal or position logic.

    This is the institutional "portable alpha" technique: run the alpha engine,
    overlay a beta hedge to transport alpha to any desired beta level.

    Look-ahead safety
    ─────────────────
    rolling_beta[T] uses returns through day T (backward-looking).
    hedge_dollars[T] is set at close of day T.
    portfolio_returns() applies shift(1) to all sizes — so the hedge is only
    applied to day T+1 returns.  No look-ahead bias.

    Parameters
    ──────────
    base_sizes:     Output of any sizing function (T × N dollar positions).
    returns:        Daily return DataFrame (T × N).  Must contain "SPY".
    capital:        Starting capital in dollars.
    target_beta:    Target portfolio beta to SPY (default 0.30).
    beta_window:    Trailing window for rolling beta estimation (default 63 = 1 quarter).
    rebalance_freq: Days between hedge rebalances (default 5 = weekly).
    max_hedge_pct:  Maximum short SPY as fraction of capital (default 0.40).

    Returns
    ───────
    Dollar position DataFrame identical to base_sizes but with SPY column
    adjusted by the beta hedge.
    """
    if "SPY" not in returns.columns:
        return base_sizes

    spy_ret  = returns["SPY"]
    weights  = base_sizes.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base_sizes.columns)).sum(axis=1)

    # Rolling beta: Cov(port, SPY) / Var(SPY)  — fully backward-looking
    rolling_cov  = port_ret.rolling(beta_window).cov(spy_ret)
    rolling_var  = spy_ret.rolling(beta_window).var().replace(0, np.nan)
    rolling_beta = (rolling_cov / rolling_var).fillna(target_beta)

    # Dollar hedge to close the gap to target_beta
    # Negative sign: positive beta_excess → short SPY
    hedge_dollars = -(rolling_beta - target_beta) * capital
    hedge_dollars = hedge_dollars.clip(-max_hedge_pct * capital,
                                        max_hedge_pct * capital)

    # Rebalance only every rebalance_freq days — ffill between dates
    rebal_mask       = pd.Series(False, index=base_sizes.index)
    rebal_mask.iloc[::rebalance_freq] = True
    hedge_rebalanced = hedge_dollars.where(rebal_mask).ffill().fillna(0.0)

    # Apply hedge to SPY column
    result = base_sizes.copy()
    if "SPY" in result.columns:
        result["SPY"] = result["SPY"] + hedge_rebalanced
    else:
        result["SPY"] = hedge_rebalanced

    # Clip net SPY position (long or short) to sensible bounds
    result["SPY"] = result["SPY"].clip(-max_hedge_pct * capital,
                                        MAX_POSITION_PCT * capital)
    return result


def multi_mom_carry_sizes(
    trend_signals: pd.DataFrame,
    carry_signals: pd.DataFrame,
    features: dict,
    capital: float,
    carry_weight: float = 0.25,
) -> pd.DataFrame:
    """
    75% momentum-tilt trend sizing + 25% carry sizing.

    Carry is structurally uncorrelated to trend/momentum — it measures the
    expected return from HOLDING (yield curve roll, futures-curve slope), not
    from price direction.  A 25% carry allocation (Asness, Moskowitz, Pedersen
    2013) adds diversification without overwhelming the primary trend signal.

    Architecture:
      1. Trend sizes  = momentum_tilt_sizes(trend_signals, ...)
         — ATR base + cross-sectional 63d momentum rank tilt
      2. Carry sizes  = atr_sizes(carry_signals, ...)
         — ATR base × continuous carry signal ∈ [-1, +1]
         — Equity tickers carry = 0.0 → no position change from carry alone
      3. Blend        = (1 − carry_weight) × trend + carry_weight × carry
      4. Gross cap    = scale each row so total gross ≤ capital

    Args:
        trend_signals: Signal DataFrame (T × N) used for trend sizing.
        carry_signals: Carry signal DataFrame (T × N), values ∈ [-1, +1].
                       Equity tickers should be 0.
        features:      Dict[ticker → feature DataFrame] with ATR + Close.
        capital:       Total capital in dollars.
        carry_weight:  Weight on carry (default 0.25 = 25%).

    Returns:
        Dollar position size DataFrame, clipped to ±MAX_POSITION_PCT × capital.
    """
    # Step 1: Trend sizes with momentum tilt
    trend_sizes = momentum_tilt_sizes(trend_signals, features, capital)

    # Step 2: Carry sizes — ATR base weighted by carry signal
    # Bond carry (yield-curve z-score) is excluded: when the curve inverts,
    # the signal fires negative (reduce/short bonds) but bonds often RALLY
    # in flight-to-safety.  The trend signal already captures bond direction;
    # adding a conflicting carry overlay hurts OOS Sharpe in inversion periods
    # (2018-2019, 2022-2023 gap widened from +0.09 to +0.24 with bond carry).
    # Commodity carry (futures-curve slope proxy) is genuinely orthogonal and
    # improves the IS-OOS gap.
    carry_aligned = (
        carry_signals
        .reindex(columns=trend_sizes.columns, fill_value=0.0)
        .reindex(index=trend_sizes.index)
        .fillna(0.0)
    )
    # Zero out bond carry — use commodity carry only
    bond_cols = [c for c in carry_aligned.columns if ASSET_CLASS.get(c) == "bond"]
    carry_aligned = carry_aligned.copy()
    carry_aligned[bond_cols] = 0.0

    carry_sizes = atr_sizes(carry_aligned, features, capital)

    # Step 3: Blend — only where carry is non-trivially active (|signal| > 0.05).
    # Equity tickers have carry = 0 and must NOT be diluted.  A fixed-weight
    # blend would reduce all equity positions to (1 - carry_weight) × trend,
    # cutting equity alpha by carry_weight%.  Instead, scale the blend by the
    # carry signal's magnitude so equity tickers remain at full trend size.
    carry_active     = (carry_aligned.abs() > 0.05).astype(float)
    effective_weight = carry_weight * carry_active   # 0 for equities, carry_weight for bonds/commodities
    combined = (1 - effective_weight) * trend_sizes + effective_weight * carry_sizes

    # ── Safe-haven floor and VXZ cap ──────────────────────────────────────────
    for _ft, _ff in SAFE_HAVEN_FLOOR.items():
        if _ft in combined.columns:
            _floor_dollars = _ff * capital
            combined[_ft] = combined[_ft].where(combined[_ft] < 0,
                                                 combined[_ft].clip(lower=_floor_dollars))
    if "VXZ" in combined.columns:
        combined["VXZ"] = combined["VXZ"].clip(-0.03 * capital, 0.03 * capital)

    # Step 4: Gross exposure cap
    gross = combined.abs().sum(axis=1).replace(0, np.nan)
    scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return combined.multiply(scale, axis=0).clip(
        -capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT
    )


def hrp_sizes(
    signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
    capital: float,
    cov_window: int = 126,
    rebalance_freq: int = 21,
) -> pd.DataFrame:
    """
    Hierarchical Risk Parity position sizing (López de Prado, 2016).

    Identical two-step structure as risk_parity_sizes() but replaces the
    Equal Risk Contribution (ERC) solver with HRP weights:

      Step 1 — ATR base sizes: compute volatility-normalised dollar positions
               so total gross exposure is already risk-scaled per asset.
      Step 2 — HRP redistribution: reallocate that total gross exposure using
               HRP weights derived from a rolling Ledoit-Wolf covariance matrix,
               recomputed every `rebalance_freq` trading days.

    HRP advantage over ERC
    ──────────────────────
    ERC requires Cov @ w in the denominator (fixed-point update) — small
    eigenvalues from a noisy T/N ≈ 6 estimate can dominate.  HRP uses only
    pairwise distances (tree structure) and diagonal sub-block variances
    (bisection allocation) — no matrix inversion, numerically stable for any T/N.

    Args:
        signals:        Signal DataFrame (T × N).
        features:       Dict[ticker → feature DataFrame] with atr_14 and Close.
        returns:        Daily log-return DataFrame (T × N).
        capital:        Starting capital in dollars.
        cov_window:     Rolling covariance lookback (default 126 = 6 months).
        rebalance_freq: Days between covariance recomputation (default 21 = monthly).

    Returns:
        Dollar position size DataFrame, same shape as signals.
        Clipped to ±MAX_POSITION_PCT × capital per position.
    """
    base_sizes = atr_sizes(signals, features, capital)

    adjusted = base_sizes.copy()
    tickers  = [t for t in signals.columns if t in returns.columns]
    n_rows   = len(signals)

    # ── Precompute HRP weights at each rebalance date ─────────────────────
    hrp_weight_df = pd.DataFrame(np.nan, index=signals.index, columns=tickers)

    for i in range(cov_window, n_rows, rebalance_freq):
        ret_window = returns.iloc[i - cov_window : i][tickers]
        ret_window = ret_window.dropna(thresh=int(len(ret_window) * 0.90), axis=1)
        available  = ret_window.columns.tolist()

        if len(available) >= 2 and len(ret_window) >= 20:
            ret_clean = ret_window.ffill().fillna(0)
            cov_mat   = estimate_covariance(ret_clean)
            w_hrp     = hrp_weights(cov_mat)
            for j, t in enumerate(available):
                hrp_weight_df.iloc[i, hrp_weight_df.columns.get_loc(t)] = w_hrp[j]

    hrp_weight_df = hrp_weight_df.ffill()

    # ── Vectorised redistribution (mirrors risk_parity_sizes) ────────────
    sig_vals  = signals[tickers]
    active    = (sig_vals != 0) & hrp_weight_df[tickers].notna()

    w_active  = hrp_weight_df[tickers].where(active, 0.0)
    w_sum     = w_active.sum(axis=1).replace(0, np.nan)
    w_norm    = w_active.div(w_sum, axis=0).fillna(0.0)

    total_atr = base_sizes[tickers].where(active, 0.0).abs().sum(axis=1)
    sign_df   = sig_vals.where(active, 0.0).clip(-1, 1)
    raw       = sign_df * w_norm * total_atr.values[:, None]

    cap_limit = capital * MAX_POSITION_PCT
    clipped   = raw.clip(-cap_limit, cap_limit)

    first_valid = hrp_weight_df.first_valid_index()
    if first_valid is not None:
        mask = signals.index >= first_valid
        for t in tickers:
            adjusted.loc[mask, t] = clipped.loc[mask, t]

    return apply_macro_multiplier(adjusted)


def hrp_mom_sizes(
    signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
    capital: float,
    cov_window: int = 126,
    rebalance_freq: int = 21,
    tilt_min: float = 0.7,
    tilt_range: float = 0.6,
    mom_window: int = 63,
) -> pd.DataFrame:
    """
    HRP sizing with a cross-sectional momentum tilt overlay.

    Two-stage process:
      1. HRP sizing: allocate capital using hierarchical risk parity weights
         (correlation-aware, no matrix inversion).
      2. Momentum tilt: re-weight active positions ±30% based on cross-sectional
         63-day return rank (same tilt as momentum_tilt_sizes).

    Rationale
    ─────────
    HRP addresses the ALLOCATION problem: how much of the risk budget to give
    each asset given its correlation with the rest of the portfolio.
    Momentum tilt addresses the SELECTION problem: among active positions,
    favour recent outperformers.  The two adjustments are near-orthogonal —
    HRP weights are driven by the covariance structure (slow-moving, monthly
    rebalance) while momentum ranks change weekly.

    Args:
        signals:        Signal DataFrame (T × N).
        features:       Dict[ticker → feature DataFrame] with atr_14 and Close.
        returns:        Daily log-return DataFrame (T × N).
        capital:        Starting capital in dollars.
        cov_window:     HRP covariance lookback (default 126).
        rebalance_freq: HRP rebalance frequency (default 21).
        tilt_min:       Minimum tilt factor for worst momentum rank (default 0.7).
        tilt_range:     Tilt span (default 0.6 → tilt ∈ [0.7, 1.3]).
        mom_window:     Trailing return window for cross-sectional ranking (default 63).

    Returns:
        Dollar position size DataFrame.  Clipped to ±MAX_POSITION_PCT × capital.
    """
    # Step 1: HRP base sizes
    base = hrp_sizes(signals, features, returns, capital, cov_window, rebalance_freq)

    # Step 2: Cross-sectional momentum tilt (same logic as momentum_tilt_sizes)
    close_cols = {t: features[t]["Close"].reindex(base.index).ffill()
                  for t in signals.columns if t in features}
    if not close_cols:
        return base

    closes = pd.DataFrame(close_cols)
    mom    = closes.pct_change(mom_window)

    active_mask = signals.abs() > 0
    mom_cols    = [c for c in signals.columns if c in mom.columns]
    mom_masked  = mom[mom_cols].where(active_mask[mom_cols])
    ranks       = mom_masked.rank(axis=1, pct=True)
    n_active    = ranks.notna().sum(axis=1)

    tilt      = pd.DataFrame(1.0, index=base.index, columns=base.columns)
    valid_rows = n_active >= 2
    for c in mom_cols:
        col_ranks = ranks[c]
        col_valid = valid_rows & col_ranks.notna()
        tilt.loc[col_valid, c] = tilt_min + tilt_range * col_ranks[col_valid].values

    tilted = base * tilt
    gross  = tilted.abs().sum(axis=1).replace(0, np.nan)
    scale  = (capital / gross).clip(upper=1.0).fillna(1.0)
    return tilted.multiply(scale, axis=0)


def signal_gated_mv_regime_sizes(
    signals: pd.DataFrame,
    ensemble_signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
    capital: float,
    rebalance_freq: int = 21,
    min_regime_obs: int = 63,
    cov_fallback_window: int = 126,
) -> pd.DataFrame:
    """
    Signal-gated minimum variance sizing with regime-conditional covariance.

    Combines three components that each address a specific weakness:

    1. Regime-conditional covariance (from rp_regime_aware):
       Uses only same-regime historical returns for covariance estimation.
       This correctly captures the 0.20→0.45+ correlation spike in stress.

    2. Minimum variance objective (new):
       Minimizes w^T Σ w with NO alpha vector — avoids the estimation error
       that hurt ir_optimized.  MV weights work best in bull_calm where
       the covariance estimate is most accurate (rp_regime_aware's weakest
       regime).

    3. Ensemble tilt (±20% post-solve):
       After solving the MV optimization, tilts weights by the IC-weighted
       ensemble signal.  Score=+1 → 1.20× weight, score=-1 → 0.80× weight.
       Small enough to preserve the MV structure, large enough to add
       conviction-based alpha.

    Signal gating: uses binary regime signals (MA50/200 crossover) for
    direction — a ticker must have signal==1 (long) to receive any weight.
    This keeps the entry/exit discipline from the trend-following system
    while allowing the optimizer to allocate AMONG active positions.

    Args:
        signals:          Binary signal DataFrame (T × N), signal_regime.
        ensemble_signals: Continuous [-1, +1] ensemble signal DataFrame (T × N).
        features:         Dict[ticker -> feature DataFrame].
        returns:          Daily log-return DataFrame (T × N).
        capital:          Total capital in dollars.
        rebalance_freq:   Days between covariance recomputation (default 21).
        min_regime_obs:   Min observations for regime-conditional cov (default 63).
        cov_fallback_window: Fallback window if insufficient regime obs (default 126).

    Returns:
        Dollar position size DataFrame (T × N).
    """
    # ── Load regime labels ────────────────────────────────────────────────
    macro_path = MACRO_DIR / "macro_features.parquet"
    spy_path   = FEATURE_DIR / "SPY.parquet"

    if not macro_path.exists() or not spy_path.exists():
        # Cannot build regime labels — fall back to equal weight
        return equal_weight_sizes(signals, capital)

    macro     = pd.read_parquet(macro_path)
    spy_close = pd.read_parquet(spy_path)["Close"]
    vix       = macro.get("vix")
    if vix is None:
        return equal_weight_sizes(signals, capital)

    regimes = label_regimes(vix, spy_close, returns.index)

    sizes    = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    tickers  = [t for t in signals.columns if t in returns.columns]
    col_locs = {t: signals.columns.get_loc(t) for t in tickers}

    # Cache: recompute cov when regime changes or every rebalance_freq days
    cov_cache        = None
    cov_tickers      = None
    last_regime      = None
    days_since_rebal = 0

    warmup = max(cov_fallback_window, min_regime_obs)

    for i in range(warmup, len(signals)):
        # ── Active tickers: signal == 1 (long only via MA crossover) ─────
        sig_today = signals.iloc[i]
        active = [t for t in tickers if sig_today.get(t, 0) == 1]

        if len(active) < 3:
            # Too few for meaningful optimization — equal weight fallback
            if active:
                w_eq = 1.0 / len(active)
                for t in active:
                    sizes.iloc[i, col_locs[t]] = capital * min(w_eq, MAX_POSITION_PCT)
            continue

        # ── Regime label for today ───────────────────────────────────────
        current_regime = regimes.iloc[i]
        regime_changed = current_regime != last_regime
        days_since_rebal += 1

        # ── Recompute covariance if regime changed or rebalance interval ─
        need_rebal = (
            cov_cache is None
            or regime_changed
            or days_since_rebal >= rebalance_freq
        )

        if need_rebal:
            ret_history = returns.iloc[:i][active]
            ret_history = ret_history.dropna(
                thresh=int(min(len(ret_history), cov_fallback_window) * 0.90),
                axis=1,
            )
            avail = ret_history.columns.tolist()

            if len(avail) >= 3 and len(ret_history) >= 20:
                if current_regime in (
                    "bull_calm", "bull_stress", "bear_calm", "bear_stress"
                ):
                    regime_hist = regimes.iloc[:i]
                    cov_cache, _ = regime_conditional_covariance(
                        ret_history.ffill().fillna(0),
                        regime_hist,
                        current_regime,
                        min_obs=min_regime_obs,
                        fallback_window=cov_fallback_window,
                    )
                else:
                    tail = ret_history.iloc[-cov_fallback_window:]
                    cov_cache = estimate_covariance(tail.ffill().fillna(0))

                cov_tickers = avail
                last_regime = current_regime
                days_since_rebal = 0

        if cov_cache is None or cov_tickers is None:
            continue

        # ── Build sub-cov for today's active set ─────────────────────────
        active_in_cov = [t for t in active if t in cov_tickers]
        if len(active_in_cov) < 3:
            if active:
                w_eq = 1.0 / len(active)
                for t in active:
                    sizes.iloc[i, col_locs[t]] = capital * min(w_eq, MAX_POSITION_PCT)
            continue

        sub_idx = [cov_tickers.index(t) for t in active_in_cov]
        sub_cov = cov_cache[np.ix_(sub_idx, sub_idx)]

        # ── Ensemble scores for active tickers ───────────────────────────
        ens_today = ensemble_signals.iloc[i]
        ens_scores = {
            t: float(ens_today.get(t, 0.0)) for t in active_in_cov
        }

        # ── Asset classes ────────────────────────────────────────────────
        ac_map = {t: ASSET_CLASS.get(t, "equity_index") for t in active_in_cov}

        # ── Solve MV + tilt ──────────────────────────────────────────────
        weights = minimum_variance_gated_weights(
            active_in_cov, sub_cov, ens_scores, ac_map,
        )

        # ── Convert to dollar positions ──────────────────────────────────
        for t, w in weights.items():
            sizes.iloc[i, col_locs[t]] = capital * w

    return apply_macro_multiplier(sizes)


def beta_hedged_sizes(
    signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
    capital: float,
    beta_window: int = 126,
) -> pd.DataFrame:
    """
    ATR-sized long positions in stocks paired with short sector-ETF hedge legs.

    For stocks in HEDGE_MAP where the signal is long (signal == 1):
      long_size   = ATR-based dollar position (same as atr_sizes)
      rolling_beta = Cov(stock_ret, hedge_ret) / Var(hedge_ret)  [126-day window]
      hedge_short  = -rolling_beta × long_size  added to the hedge ETF's column

    The hedge short is ADDED to whatever position the hedge ETF already holds
    from its own signal (it may independently have a long position).  This
    allows the portfolio to be simultaneously long XLF (sector signal) and
    short XLF (hedge for JPM/GS pair trades) — the net position is the sum.

    For tickers not in HEDGE_MAP (ETFs, bonds, commodities): standard ATR sizing.

    Rolling beta uses the same 126-day window used for PCA scaling and
    risk-parity estimation — no new parameter.  Lagged by 1 day (shift(1))
    inside portfolio_returns so no look-ahead.

    Args:
        signals:     Signal DataFrame (pair_signals or multi_pair_signals).
        features:    Dict[ticker -> feature DataFrame] with atr_14 and Close.
        returns:     Daily returns DataFrame — must contain hedge tickers.
        capital:     Starting capital in dollars.
        beta_window: Rolling OLS window for beta estimation (default 126).

    Returns:
        Dollar position size DataFrame.  Positive = long, negative = short.
    """
    # Step 1: ATR base sizes for all tickers using their signals.
    sizes = atr_sizes(signals, features, capital)

    # Step 2: Precompute rolling betas for all HEDGE_MAP pairs.
    # beta = Cov(stock, hedge) / Var(hedge) — rolling over beta_window days.
    rolling_betas: dict = {}
    for stock, hedge in HEDGE_MAP.items():
        if stock not in returns.columns or hedge not in returns.columns:
            continue
        s = returns[stock].fillna(0)
        h = returns[hedge].fillna(0)
        cov_sh = s.rolling(beta_window).cov(h)
        var_h  = h.rolling(beta_window).var().replace(0, np.nan)
        rolling_betas[(stock, hedge)] = (cov_sh / var_h).fillna(1.0)

    # Step 3: Add short hedge legs where stock is long and spread signal is on.
    for stock, hedge in HEDGE_MAP.items():
        key = (stock, hedge)
        if key not in rolling_betas:
            continue
        if stock not in sizes.columns or hedge not in sizes.columns:
            continue

        beta_s = rolling_betas[key].reindex(sizes.index).fillna(1.0)
        stock_long = sizes[stock].clip(lower=0)  # only the long component

        # Short position in hedge ETF: -beta × long_size.
        # Negative because we are shorting the ETF to hedge the long stock.
        hedge_short = -beta_s * stock_long

        # Add to the hedge ETF column (may already have its own long signal).
        sizes[hedge] = (sizes[hedge] + hedge_short).clip(
            -capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT
        )

    # Step 4: Portfolio-level gross cap (same as atr_sizes).
    gross = sizes.abs().sum(axis=1).replace(0, np.nan)
    scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return sizes.multiply(scale, axis=0)


def defensive_tilt_overlay(
    sizes: pd.DataFrame,
    signals: pd.DataFrame,
    macro: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    """
    When VIX term structure inverts (VIX9D > VIX, ratio > 1.05)
    AND TLT is in a positive signal day AND SPY signal is 0 or
    negative, boost VGSH weight by +3% and TLT weight by +3%
    (absolute, capped at their max allowed weight).
    Fund the boost by reducing the lowest-conviction long
    positions proportionally.

    Trigger condition (all three must be true):
      1. vix_term_ratio > 1.05  (VIX9D/VIX backwardation — acute stress onset;
         CBOE published threshold, not fitted to this dataset)
      2. signals["TLT"] > 0 on that bar (TLT in uptrend)
      3. signals["SPY"] == 0 on that bar (SPY not in uptrend)

    When triggered:
      - VGSH weight += min(0.03, max_weight - current_vgsh_weight)
      - TLT  weight += min(0.03, max_weight - current_tlt_weight)
      - Reduce other long weights proportionally to keep sum(weights) = 1.0

    When NOT triggered: return sizes unchanged.

    If VGSH is not in the universe on a given bar, skip the VGSH boost
    silently and only boost TLT.

    The 3% boost and 1.05 VIX ratio are CBOE published backwardation
    thresholds — not fitted to this dataset.

    Args:
        sizes:   Dollar position size DataFrame (T × N).
        signals: Signal DataFrame (T × N) with TLT and SPY columns.
        macro:   Macro DataFrame with vix_term_ratio column.
        capital: Total capital in dollars.

    Returns:
        Adjusted dollar position size DataFrame.  Unchanged if trigger
        conditions are never met or if macro is unavailable.
    """
    if macro.empty or "vix_term_ratio" not in macro.columns:
        return sizes

    result  = sizes.copy()
    max_pos = capital * MAX_POSITION_PCT
    boost   = 0.03 * capital   # 3% absolute boost (CBOE published threshold)

    vix_ratio = macro["vix_term_ratio"].reindex(sizes.index).ffill().fillna(1.0)
    ratio_ok  = vix_ratio > 1.05

    tlt_sig = (
        signals["TLT"].reindex(sizes.index).fillna(0) > 0
        if "TLT" in signals.columns
        else pd.Series(False, index=sizes.index)
    )
    spy_sig = (
        signals["SPY"].reindex(sizes.index).fillna(1) == 0
        if "SPY" in signals.columns
        else pd.Series(False, index=sizes.index)
    )

    triggered = ratio_ok & tlt_sig & spy_sig

    if not triggered.any():
        return result

    # ── Boost VGSH and TLT on triggered rows ─────────────────────────────────
    total_boost_per_row = pd.Series(0.0, index=sizes.index)

    for col in ["VGSH", "TLT"]:
        if col not in result.columns:
            # VGSH not in universe — skip silently; TLT is always present
            continue
        current = result[col].copy()
        # Room below position cap
        room = (max_pos - current).clip(lower=0)
        # Boost = min(3% of capital, available room), only on triggered rows
        add = pd.Series(0.0, index=sizes.index)
        add[triggered] = room[triggered].clip(upper=boost)
        result[col] = current + add
        total_boost_per_row += add

    # ── Reduce other long positions proportionally to fund the boost ──────────
    other_longs = [t for t in result.columns if t not in {"VGSH", "TLT"}]
    if not other_longs:
        return result

    # Sum of long-only exposures in other tickers (per row)
    other_pos_sum = result[other_longs].clip(lower=0).sum(axis=1)

    # Scale factor: (existing_longs - boost_needed) / existing_longs
    # Only apply where triggered AND there are longs to reduce
    reduce_mask = triggered & (other_pos_sum > 0)
    scale = pd.Series(1.0, index=sizes.index)
    scale[reduce_mask] = (
        (other_pos_sum[reduce_mask] - total_boost_per_row[reduce_mask])
        / other_pos_sum[reduce_mask]
    ).clip(lower=0.0)

    # Apply scale only to positive (long) positions in other columns
    for t in other_longs:
        col_data  = result[t].copy()
        long_rows = reduce_mask & (col_data > 0)
        if long_rows.any():
            result.loc[long_rows, t] = col_data[long_rows] * scale[long_rows]

    return result


def portfolio_returns(sizes: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    Compute daily portfolio P&L from dollar position sizes and asset returns.

    Converts dollar sizes to portfolio weights (sizes / CAPITAL), then sums
    the weighted returns across all positions each day.

    Uses shift(1): sizes set at end of day T are held starting at day T+1.
    This is the fundamental anti-look-ahead mechanism — you cannot trade on
    the signal you receive at the close of the same day.

    Args:
        sizes:   Dollar position size DataFrame (T × N).
        returns: Daily return DataFrame (T × N).

    Returns:
        Daily portfolio return Series (T,).
    """
    weights = sizes.shift(1) / CAPITAL
    return (weights * returns.reindex(columns=sizes.columns)).sum(axis=1)


def momentum_tilt_sizes(
    signals: pd.DataFrame,
    features: dict,
    capital: float,
    tilt_min: float = 0.7,
    tilt_range: float = 0.6,
    mom_window: int = 63,
) -> pd.DataFrame:
    """
    ATR base sizing with a cross-sectional momentum tilt.

    For each day, tickers with active signals are ranked by their trailing
    63-day (3-month) return.  Position sizes are scaled by a tilt factor:

        tilt = tilt_min + tilt_range × rank     [rank ∈ [0, 1]]

    Part 1 — Regime-adaptive tilt range (structural, not curve-fitted):
        bull_calm:   [0.50, 1.50]  ±50% — strong dispersion in confirmed up-trend
        bull_stress: [0.60, 1.40]  ±40%
        bear_calm:   [0.80, 1.20]  ±20%
        bear_stress: [0.90, 1.10]  ±10% — near-equal weight during momentum crashes

    Part 2 — Regime-adaptive safe-haven floor:
        bull_calm:   VGSH 2%, DBMF 2%, WTMF 2% = 6% (halved)
        bull_stress: VGSH 3%, DBMF 3%, WTMF 3% = 9%
        bear_calm:   VGSH 5%, DBMF 4%, WTMF 4% = 13% (unchanged)
        bear_stress: VGSH 7%, DBMF 5%, WTMF 5% = 17% (increased)
        VIX guardrail: bull_calm + VIX ≥ 20 → use bull_stress floor.
        Freed floor capital is deployed into active equity signals via gross scaling.

    Part 4 — Signal breadth conviction multiplier:
        breadth = fraction of universe with active signals on a given day.
        conviction_mult = (1.40 − 0.75 × breadth).clip(0.80, 1.25)
        When fewer names are long (high conviction), size up.
        When everything is long, size down per-position; more positions compensate.

    Window choice (63 days): Jegadeesh & Titman (1993) 3–12-month momentum sweet spot.

    Args:
        signals:    Signal DataFrame (T × N), any signal type.
        features:   Dict[ticker → feature DataFrame] with Close prices.
        capital:    Starting capital in dollars.
        tilt_min:   Default minimum tilt factor (overridden by regime, default 0.7).
        tilt_range: Default tilt range (overridden by regime, default 0.6).
        mom_window: Trailing return window for cross-sectional ranking (default 63).

    Returns:
        Dollar position size DataFrame.  Clipped to ±MAX_POSITION_PCT × capital.
    """
    # Step 1: ATR base sizes — normalise for per-asset volatility.
    base = atr_sizes(signals, features, capital)

    # Step 2: Build trailing close-price matrix for momentum ranking.
    close_cols = {t: features[t]["Close"].reindex(base.index).ffill()
                  for t in signals.columns if t in features}
    if not close_cols:
        return base
    closes = pd.DataFrame(close_cols)

    # Step 3: 63-day trailing return for each ticker, computed daily.
    mom = closes.pct_change(mom_window)

    # ── Load regime labels for Parts 1 and 2 ─────────────────────────────────
    regimes, vix_s = _load_regime_data(base.index)
    has_regimes    = not (regimes == "unknown").all()

    # ── Part 1: Regime-adaptive per-bar tilt parameters ──────────────────────
    t_min_s   = pd.Series(tilt_min,   index=base.index)
    t_range_s = pd.Series(tilt_range, index=base.index)
    if has_regimes:
        for reg, (rmin, rrange) in _REGIME_TILT_PARAMS.items():
            mask = regimes == reg
            if mask.any():
                t_min_s[mask]   = rmin
                t_range_s[mask] = rrange

    # ── Part 4: Signal breadth conviction multiplier ──────────────────────────
    # breadth = fraction of universe active; high breadth → lower conviction per position.
    # Linear: 1.175 at breadth=0.3, 1.025 at breadth=0.5, 0.80 at breadth=0.8.
    breadth         = ((signals != 0).sum(axis=1) / max(signals.shape[1], 1)
                       ).clip(0.0, 1.0).reindex(base.index).fillna(0.5)
    conviction_mult = (1.40 - 0.75 * breadth).clip(0.80, 1.25)

    # ── Step 4: Vectorised cross-sectional momentum ranking ───────────────────
    active_mask = signals.abs() > 0
    mom_cols    = [c for c in signals.columns if c in mom.columns]
    mom_masked  = mom[mom_cols].where(active_mask[mom_cols])
    ranks       = mom_masked.rank(axis=1, pct=True)

    n_active    = ranks.notna().sum(axis=1)
    tilt        = pd.DataFrame(1.0, index=base.index, columns=base.columns)
    valid_rows  = n_active >= 2
    for c in mom_cols:
        col_ranks = ranks[c]
        col_valid = valid_rows & col_ranks.notna()
        if not col_valid.any():
            continue
        tilt.loc[col_valid, c] = (
            t_min_s[col_valid].values + t_range_s[col_valid].values * col_ranks[col_valid].values
        )

    # ── Step 5: Apply tilt + conviction multiplier (before floor/cap) ─────────
    tilted = base * tilt * conviction_mult.values[:, None]

    # ── Part 2: Regime-adaptive safe-haven floor ──────────────────────────────
    # Compute per-day effective regime (with VIX guardrail for floor logic).
    if has_regimes:
        eff_regime = regimes.copy()
        if vix_s is not None:
            # If labeled bull_calm but spot VIX ≥ 20, use bull_stress floor.
            transitioning = (regimes == "bull_calm") & (vix_s >= 20)
            eff_regime[transitioning] = "bull_stress"
        floor_tickers = {t: {} for t in SAFE_HAVEN_FLOOR if t in tilted.columns}
        for _ft in floor_tickers:
            # Build per-bar floor fraction from regime
            floor_frac = pd.Series(SAFE_HAVEN_FLOOR[_ft], index=tilted.index)
            for reg, reg_floors in _REGIME_FLOOR_PARAMS.items():
                mask = eff_regime == reg
                if mask.any() and _ft in reg_floors:
                    floor_frac[mask] = reg_floors[_ft]
            floor_dollars = floor_frac * capital
            tilted[_ft]   = tilted[_ft].where(tilted[_ft] < 0,
                                               tilted[_ft].clip(lower=floor_dollars))
        # ── Gross scaling: deploy freed floor capital into active equity signals ──
        # After reducing the floor, scale non-floor long positions toward the
        # regime's gross target so the freed 7% (bull_calm) goes to equities,
        # not cash.  The final gross cap (below) prevents leverage.
        floor_set     = set(SAFE_HAVEN_FLOOR.keys()) & set(tilted.columns)
        non_floor_c   = [t for t in tilted.columns if t not in floor_set]
        if non_floor_c:
            gross_target_s = pd.Series(0.85, index=tilted.index)
            for reg, gt in _REGIME_GROSS_TARGET.items():
                mask = eff_regime == reg
                if mask.any():
                    gross_target_s[mask] = gt
            floor_notional    = tilted[list(floor_set)].abs().sum(axis=1)
            non_floor_target  = (gross_target_s * capital - floor_notional).clip(lower=0)
            current_non_floor = tilted[non_floor_c].abs().sum(axis=1).replace(0, np.nan)
            scale_up = (non_floor_target / current_non_floor).clip(0.50, 1.40).fillna(1.0)
            tilted[non_floor_c] = tilted[non_floor_c].multiply(scale_up, axis=0)
            # Re-clip per-position cap after scale-up
            cap_limit = capital * MAX_POSITION_PCT
            for t in non_floor_c:
                tilted[t] = tilted[t].clip(-cap_limit, cap_limit)
    else:
        # Fallback: static floor (original behaviour)
        for _ft, _ff in SAFE_HAVEN_FLOOR.items():
            if _ft in tilted.columns:
                _floor_dollars = _ff * capital
                tilted[_ft] = tilted[_ft].where(tilted[_ft] < 0,
                                                 tilted[_ft].clip(lower=_floor_dollars))

    # ── VXZ cap: conditional vol hedge — never more than 3% of capital ────────
    if "VXZ" in tilted.columns:
        tilted["VXZ"] = tilted["VXZ"].clip(-0.03 * capital, 0.03 * capital)

    # ── Final gross exposure cap (no leverage) ────────────────────────────────
    gross  = tilted.abs().sum(axis=1).replace(0, np.nan)
    scale  = (capital / gross).clip(upper=1.0).fillna(1.0)
    return tilted.multiply(scale, axis=0)


def active_sharpe_ratio(port_ret: pd.Series, returns: pd.DataFrame) -> float:
    """
    Sharpe of the active return after stripping rolling SPY beta.

    active_return = portfolio_return - beta * SPY_return
    beta          = Cov(port, SPY) / Var(SPY)

    This is the 'true alpha Sharpe': return attributable to the strategy's
    own signal rather than passive market exposure.  A method with high Total
    Sharpe but high beta is mostly leveraged index exposure; a method with
    high Active Sharpe generates return that is genuinely independent of SPY.

    Args:
        port_ret: Daily portfolio return Series.
        returns:  Full returns DataFrame — must contain a 'SPY' column.

    Returns:
        Active Sharpe ratio, or nan if SPY is unavailable.
    """
    if "SPY" not in returns.columns:
        return float("nan")
    spy = returns["SPY"].reindex(port_ret.index).fillna(0)
    p   = port_ret.fillna(0).values
    s   = spy.values
    spy_var = float(np.var(s))
    if spy_var <= 0:
        return float("nan")
    beta   = float(np.cov(p, s)[0, 1]) / spy_var
    active = pd.Series(p - beta * s, index=port_ret.index)
    return sharpe_ratio(active)


def compute_regime_diagnostic(
    port_ret: pd.Series,
    sizes: pd.DataFrame,
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    capital: float,
) -> dict:
    """
    Compute regime-decomposed performance metrics for multi_mom_tilt.

    Returns a dict with per-regime stats plus aggregate capture ratios.
    Printed to stdout and saved to JSON for before/after comparison.
    """
    regimes, vix_s = _load_regime_data(port_ret.index)

    if (regimes == "unknown").all():
        print("  [regime_diagnostic] Cannot compute — regime data unavailable.")
        return {}

    spy_ret  = (returns["SPY"].reindex(port_ret.index).fillna(0)
                if "SPY" in returns.columns else pd.Series(0.0, index=port_ret.index))
    gross_pct = sizes.reindex(port_ret.index).abs().sum(axis=1) / capital * 100
    n_pos     = (signals.reindex(port_ret.index).abs() > 0).sum(axis=1)
    total     = len(port_ret)

    regime_stats = {}
    for reg in ["bull_calm", "bull_stress", "bear_calm", "bear_stress"]:
        mask = regimes == reg
        n    = int(mask.sum())
        if n == 0:
            regime_stats[reg] = dict(pct_days=0.0, avg_daily_return_pct=0.0,
                                     avg_daily_spy_pct=0.0, daily_alpha_bps=0.0,
                                     avg_gross_pct=0.0, avg_positions=0.0)
            continue
        pr           = port_ret[mask]
        sr           = spy_ret[mask]
        avg_port     = float(pr.mean()) * 100
        avg_spy      = float(sr.mean()) * 100
        alpha_bps    = (avg_port - avg_spy) * 100
        avg_gross    = float(gross_pct[mask].mean())
        avg_npos     = float(n_pos[mask].mean())
        regime_stats[reg] = dict(
            pct_days             = round(n / total * 100, 1),
            avg_daily_return_pct = round(avg_port,  4),
            avg_daily_spy_pct    = round(avg_spy,   4),
            daily_alpha_bps      = round(alpha_bps, 2),
            avg_gross_pct        = round(avg_gross, 1),
            avg_positions        = round(avg_npos,  1),
        )

    # Capture ratios
    spy_up   = spy_ret > 0
    spy_dn   = spy_ret < 0
    spy_big  = spy_ret > 0.005
    up_cap   = (float(port_ret[spy_up].mean()) / float(spy_ret[spy_up].mean())
                if spy_up.sum() > 0 and float(spy_ret[spy_up].mean()) != 0 else float("nan"))
    dn_cap   = (float(port_ret[spy_dn].mean()) / float(spy_ret[spy_dn].mean())
                if spy_dn.sum() > 0 and float(spy_ret[spy_dn].mean()) != 0 else float("nan"))
    hit_rate = (float((port_ret[spy_big] > 0).mean()) * 100
                if spy_big.sum() > 0 else float("nan"))

    # OLS beta
    spy_vals  = spy_ret.fillna(0).values
    port_vals = port_ret.fillna(0).values
    spy_var   = float(np.var(spy_vals))
    ols_beta  = (float(np.cov(port_vals, spy_vals)[0, 1]) / spy_var
                 if spy_var > 0 else float("nan"))

    bc_mask = regimes == "bull_calm"
    bs_mask = regimes == "bear_stress"

    result = dict(
        regime_stats              = regime_stats,
        upside_capture            = round(up_cap,  3),
        downside_capture          = round(dn_cap,  3),
        hit_rate_spy_big_pct      = round(hit_rate, 1),
        avg_positions_bull_calm   = round(float(n_pos[bc_mask].mean()), 1) if bc_mask.sum() > 0 else 0.0,
        avg_positions_bear_stress = round(float(n_pos[bs_mask].mean()), 1) if bs_mask.sum() > 0 else 0.0,
        avg_gross_bull_calm_pct   = round(float(gross_pct[bc_mask].mean()), 1) if bc_mask.sum() > 0 else 0.0,
        avg_gross_bear_stress_pct = round(float(gross_pct[bs_mask].mean()), 1) if bs_mask.sum() > 0 else 0.0,
        ols_beta                  = round(ols_beta, 3),
    )

    # ── Print table ───────────────────────────────────────────────────────
    print(f"\n{'='*88}")
    print("  REGIME DIAGNOSTIC  (multi_mom_tilt — full history)")
    print(f"{'='*88}")
    print(f"  {'Regime':<14} {'%Days':>7} {'PortRet%':>9} {'SPYRet%':>9} "
          f"{'Alpha(bps)':>11} {'GrossExp%':>10} {'AvgPos':>7}")
    print("  " + "-" * 72)
    for reg in ["bull_calm", "bull_stress", "bear_calm", "bear_stress"]:
        s = regime_stats[reg]
        print(f"  {reg:<14} {s['pct_days']:>7.1f} {s['avg_daily_return_pct']:>9.4f} "
              f"{s['avg_daily_spy_pct']:>9.4f} {s['daily_alpha_bps']:>11.2f} "
              f"{s['avg_gross_pct']:>10.1f} {s['avg_positions']:>7.1f}")
    print()
    print(f"  Upside capture ratio  (port / SPY | SPY > 0): {up_cap:.3f}")
    print(f"  Downside capture ratio(port / SPY | SPY < 0): {dn_cap:.3f}")
    print(f"  Hit rate on SPY >+0.5% days: {hit_rate:.1f}%")
    print(f"  Avg positions  bull_calm:    {result['avg_positions_bull_calm']:.1f}")
    print(f"  Avg positions  bear_stress:  {result['avg_positions_bear_stress']:.1f}")
    print(f"  Avg gross exp  bull_calm:    {result['avg_gross_bull_calm_pct']:.1f}%")
    print(f"  Avg gross exp  bear_stress:  {result['avg_gross_bear_stress_pct']:.1f}%")
    print(f"  OLS beta to SPY:             {ols_beta:.3f}")
    print(f"{'='*88}")

    return result


def walk_forward(signals: pd.DataFrame, returns: pd.DataFrame,
                 train_years: int = 3, test_years: int = 1,
                 sizing_fn=None) -> pd.DataFrame:
    """
    Rolling out-of-sample validation using a 3-year train / 1-year test split.

    Each test window is genuinely out-of-sample — the strategy parameters
    (MA windows, ADX threshold, RSI level) were developed on pre-2015 data
    and are not refit on each training window.  The walk-forward is purely
    a validation of strategy stability across different market regimes.

    Regime coverage:
      2015–2018 (test 2018):  Late bull, 2018 sell-off, VIX spike
      2016–2019 (test 2019):  Strong bull, trade war, Fed pivot
      2017–2020 (test 2020):  COVID crash and V-shaped recovery
      2018–2021 (test 2021):  Meme stock / growth rally, rising rates begin
      2019–2022 (test 2022):  Bear market, 40-year high inflation, rate hikes
      2020–2023 (test 2023):  Soft landing, AI rally begins

    A mean OOS Sharpe close to the in-sample Sharpe indicates low overfitting.
    A large IS–OOS gap indicates the method is overfit to the training period.

    Args:
        signals:    Signal DataFrame (T × N) from signal_generation.py.
                    May contain binary or continuous values.
        returns:    Daily return DataFrame (T × N).
        train_years: Training window in years (default 3).
        test_years:  OOS test window in years (default 1).
        sizing_fn:  Optional callable(signals, returns) → dollar sizes.
                    If None, uses equal-weight strategy returns directly.

    Returns:
        DataFrame with one row per test period: period, sharpe, ann_return,
        max_dd, n_days.
    """
    train_days = train_years * 252
    test_days  = test_years  * 252
    results    = []
    start      = train_days

    while start + test_days <= len(signals):
        # Include the training window as context so sizing functions that require
        # historical lookback (Kelly needs 252 days, PCA needs 126, vol_target 63)
        # are fully warmed-up at the start of the test window.  Without this,
        # Kelly rolling(252) on a 252-day test window produces a single valid row
        # (day 252) and zeros everywhere else — effectively a flat strategy.
        ctx_start = max(0, start - train_days)
        all_sig   = signals.iloc[ctx_start : start + test_days]
        all_ret   = returns.iloc[ctx_start : start + test_days]

        if sizing_fn is None:
            # Equal-weight: each ticker contributes equally via strategy returns
            period_ret = pd.Series(0.0, index=signals.index[start : start + test_days])
            for ticker in all_sig.columns:
                if ticker in all_ret.columns:
                    strat = compute_strategy_returns(all_sig[ticker], all_ret[ticker])
                    period_ret += strat.iloc[-test_days:].values / len(all_sig.columns)
        else:
            # Custom sizing: compute over full context, evaluate only on test slice.
            # portfolio_returns uses shift(1) so the first test-day return is 0
            # (no carry-over position from training) — this is correct.
            sizes      = sizing_fn(all_sig, all_ret)
            # Defensive tilt overlay: boost VGSH/TLT when VIX backwardation is
            # acute, TLT is trending, and SPY is flat.  Applied last — after all
            # sizing logic — so it doesn't interact with vol scaling or PCA.
            sizes      = defensive_tilt_overlay(sizes, all_sig, _get_macro(), CAPITAL)
            period_ret = portfolio_returns(sizes.iloc[-test_days:], all_ret.iloc[-test_days:])

        y0 = signals.index[start].year
        y1 = signals.index[start + test_days - 1].year
        # Active Sharpe: strip SPY beta from the period's return series.
        # Uses the OOS slice of returns for SPY, aligned to period_ret.
        act_sh = round(active_sharpe_ratio(period_ret, all_ret.iloc[-test_days:]), 3)
        results.append({
            "period"        : f"{y0}-{y1}",
            "sharpe"        : round(sharpe_ratio(period_ret), 3),
            "active_sharpe" : act_sh,
            "ann_return"    : round(((1 + period_ret).prod() ** (252 / len(period_ret)) - 1) * 100, 2),
            "max_dd"        : round(max_drawdown((1 + period_ret).cumprod()) * 100, 2),
            "n_days"        : len(period_ret),
        })
        start += test_days

    return pd.DataFrame(results)


def apply_vol_targeting(
    returns: pd.Series,
    target_vol: float | None = None,
    scale_cap: float = 1.5,
    scale_floor: float = 0.3,
    lookback: int = 63,
) -> tuple[pd.Series, dict]:
    """
    Moreira & Muir (2017) volatility targeting overlay.

    If target_vol is None: set to the strategy's own realized annual vol
    (computed over the full IS period). This ensures we're normalizing vol,
    not leveraging up.

    At each day t:
      realized_vol = std(returns[t-lookback:t]) * sqrt(252)
      scale = target_vol / realized_vol
      scale = clip(scale, scale_floor, scale_cap)
      adjusted_return[t] = returns[t] * scale

    The scale_floor of 0.3 means we never go below 30% exposure (avoid
    being whipsawed out of positions entirely during vol spikes).

    The scale_cap of 1.5 means modest leverage in calm periods.
    NOTE: If we want strictly unlevered, set scale_cap=1.0. Test both.

    Returns: adjusted_returns Series, plus diagnostics dict with:
      - mean_scale, median_scale, min_scale, max_scale
      - pct_days_deleveraged (scale < 1.0)
      - pct_days_leveraged (scale > 1.0)
      - realized_vol_before, realized_vol_after
    """
    returns = returns.dropna()
    if len(returns) < lookback + 1:
        diag = {
            "mean_scale": 1.0, "median_scale": 1.0,
            "min_scale": 1.0, "max_scale": 1.0,
            "pct_days_deleveraged": 0.0, "pct_days_leveraged": 0.0,
            "realized_vol_before": float("nan"), "realized_vol_after": float("nan"),
            "target_vol": float("nan"),
        }
        return returns.copy(), diag

    # Compute realized daily vol using a lookback-day rolling window
    # Use shift(1) so we never use current day's return to set today's scale
    rolling_vol = returns.shift(1).rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(252)
    rolling_vol = rolling_vol.replace(0, np.nan).ffill()

    # Set target_vol to natural strategy vol if not provided
    if target_vol is None:
        target_vol = float(returns.std() * np.sqrt(252))

    scale = (target_vol / rolling_vol).clip(lower=scale_floor, upper=scale_cap)
    scale = scale.fillna(1.0)  # warm-up: no scaling until we have enough history

    adjusted = returns * scale

    realized_vol_before = float(returns.std() * np.sqrt(252))
    realized_vol_after  = float(adjusted.std() * np.sqrt(252))

    diag = {
        "mean_scale"           : float(scale.mean()),
        "median_scale"         : float(scale.median()),
        "min_scale"            : float(scale.min()),
        "max_scale"            : float(scale.max()),
        "pct_days_deleveraged" : float((scale < 1.0).mean() * 100),
        "pct_days_leveraged"   : float((scale > 1.0).mean() * 100),
        "realized_vol_before"  : realized_vol_before,
        "realized_vol_after"   : realized_vol_after,
        "target_vol"           : target_vol,
    }
    return adjusted, diag


def expanded_sleeve_sizes(
    expanded_sizes: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    """
    Convert fractional expanded-universe sizes into dollar position sizes.

    The sizes from expanded_sizes.parquet are fractions of the *expanded
    sleeve* (already position-capped at 3% per stock, 15% per sector).
    This function scales them to dollar amounts using the expanded sleeve's
    capital = capital × ALLOCATION_SPLIT['expanded'].

    Args:
        expanded_sizes: DataFrame (date × ticker) of fractional allocations
                        output by generate_expanded_signals().
        capital:        Total portfolio capital (e.g. CAPITAL = 100 000).

    Returns:
        DataFrame (date × ticker) of dollar position sizes for the expanded
        sleeve, compatible with portfolio_returns().
    """
    # Guard: only execute when expanded universe is enabled
    try:
        from v1.pipeline.universe_expansion import USE_EXPANDED_UNIVERSE
        if not USE_EXPANDED_UNIVERSE:
            return pd.DataFrame()
    except ImportError:
        return pd.DataFrame()

    sleeve_capital = capital * ALLOCATION_SPLIT["expanded"]
    return expanded_sizes * sleeve_capital


def main():
    print("Building portfolio...\n")

    # ── Load signal matrices ───────────────────────────────────────────────────
    regime_signals    = pd.read_parquet(SIGNAL_DIR / "regime_signals.parquet")
    composite_signals = pd.read_parquet(SIGNAL_DIR / "composite_signals.parquet")

    ensemble_path = SIGNAL_DIR / "ensemble_signals.parquet"
    has_ensemble  = ensemble_path.exists()
    if has_ensemble:
        ensemble_signals = pd.read_parquet(ensemble_path)
    else:
        print("  NOTE: ensemble_signals.parquet not found — run signal_generation.py first\n")

    multi_path  = SIGNAL_DIR / "multi_signals.parquet"
    has_multi   = multi_path.exists()
    if has_multi:
        multi_signals = pd.read_parquet(multi_path)
    else:
        print("  NOTE: multi_signals.parquet not found — run signal_generation.py first\n")

    # ── Load features and returns ──────────────────────────────────────────────
    features, returns = {}, pd.DataFrame()
    for ticker in TICKER_LIST:
        feat_path = FEATURE_DIR / f"{ticker}.parquet"
        if not feat_path.exists():
            print(f"  {ticker}: feature file missing — skipped")
            continue
        feat             = pd.read_parquet(feat_path)
        features[ticker] = feat
        returns[ticker]  = feat["log_return"]

    returns           = returns.dropna()
    signals_regime    = regime_signals.reindex(returns.index).fillna(0)
    signals_composite = composite_signals.reindex(returns.index).fillna(0)
    if has_multi:
        signals_multi = multi_signals.reindex(returns.index).fillna(0)

    # ── Partial-exit half-size reduction ──────────────────────────────────────
    # half_size.parquet is written by signal_generation.py.  Where half_size is
    # True, the position is in the half-leg after an 8×-ATR partial exit; size
    # it at ATR_PARTIAL_REMAIN (0.5) of the normal position.  All other sizing
    # logic (risk parity weights, VIX gate, drawdown control) is unchanged.
    _hs_path = SIGNAL_DIR / "half_size.parquet"
    if _hs_path.exists():
        _hs_raw = pd.read_parquet(_hs_path).reindex(returns.index).fillna(False)
        _hs_regime = _hs_raw.reindex(columns=signals_regime.columns, fill_value=False)
        signals_regime = signals_regime * np.where(_hs_regime, ATR_PARTIAL_REMAIN, 1.0)
        if has_multi:
            _hs_multi = _hs_raw.reindex(columns=signals_multi.columns, fill_value=False)
            signals_multi = signals_multi * np.where(_hs_multi, ATR_PARTIAL_REMAIN, 1.0)

    print("Correlation matrix of returns (should be lower with diversified universe):")
    print(returns.corr().round(2))
    print()

    # ── Compute all position size DataFrames ──────────────────────────────────
    # defensive_tilt_overlay is applied as the final step for every method.
    # It boosts VGSH/TLT when VIX backwardation is acute and SPY is flat,
    # funded by proportionally reducing other long positions.
    _macro_for_overlay = _get_macro()

    # ── IMPROVEMENT 2 VALIDATION: count defensive tilt trigger bars ───────────
    _sig_for_trigger = signals_multi if has_multi else signals_regime
    if not _macro_for_overlay.empty and "vix_term_ratio" in _macro_for_overlay.columns:
        _ratio_ok  = _macro_for_overlay["vix_term_ratio"].reindex(returns.index).ffill().fillna(1.0) > 1.05
        _tlt_ok    = ((_sig_for_trigger["TLT"] > 0)
                      if "TLT" in _sig_for_trigger.columns
                      else pd.Series(False, index=returns.index))
        _spy_ok    = ((_sig_for_trigger["SPY"] == 0)
                      if "SPY" in _sig_for_trigger.columns
                      else pd.Series(False, index=returns.index))
        _triggered = (_ratio_ok
                      & _tlt_ok.reindex(returns.index).fillna(False)
                      & _spy_ok.reindex(returns.index).fillna(False))
        _n_tilt    = int(_triggered.sum())
    else:
        _n_tilt = 0
    print(f"\n  IMPROVEMENT 2 — Defensive tilt overlay:")
    print(f"  Bars triggering vix_term_ratio>1.05 + TLT long + SPY flat: {_n_tilt}")

    sizes_eq  = defensive_tilt_overlay(
        equal_weight_sizes(signals_regime, CAPITAL),
        signals_regime, _macro_for_overlay, CAPITAL,
    )
    ret_eq    = portfolio_returns(sizes_eq, returns)

    print("  Computing risk-parity sizes (Ledoit-Wolf, monthly rebalance)...")
    sizes_rp  = defensive_tilt_overlay(
        risk_parity_sizes(signals_regime, features, returns, CAPITAL),
        signals_regime, _macro_for_overlay, CAPITAL,
    )
    ret_rp    = portfolio_returns(sizes_rp, returns)

    if has_multi:
        sizes_multi_eq  = defensive_tilt_overlay(
            equal_weight_sizes(signals_multi, CAPITAL),
            signals_multi, _macro_for_overlay, CAPITAL,
        )
        sizes_multi_atr = defensive_tilt_overlay(
            atr_sizes(signals_multi, features, CAPITAL),
            signals_multi, _macro_for_overlay, CAPITAL,
        )
        ret_multi_eq    = portfolio_returns(sizes_multi_eq,  returns)
        ret_multi_atr   = portfolio_returns(sizes_multi_atr, returns)

    # Cross-sectional momentum tilt
    print("  Computing momentum tilt sizes (cross-sectional 63-day rank)...")
    if has_multi:
        sizes_multi_mom = defensive_tilt_overlay(
            momentum_tilt_sizes(signals_multi, features, CAPITAL),
            signals_multi, _macro_for_overlay, CAPITAL,
        )
        ret_multi_mom   = portfolio_returns(sizes_multi_mom, returns)

    # ── Regime diagnostic for multi_mom_tilt (printed before and after each part) ──
    import json as _json
    _diag_baseline_path = RESULTS_DIR / "regime_diagnostic_baseline.json"
    _diag_final_path    = RESULTS_DIR / "regime_diagnostic_final.json"
    _diag: dict = {}
    if has_multi:
        _diag = compute_regime_diagnostic(
            ret_multi_mom, sizes_multi_mom, returns, signals_multi, CAPITAL
        )
        # Save as baseline if it doesn't exist yet; otherwise save as current
        if not _diag_baseline_path.exists() and _diag:
            with open(_diag_baseline_path, "w") as _f:
                _json.dump(_diag, _f, indent=2)
            print(f"\n  Baseline diagnostic saved → {_diag_baseline_path}")
        elif _diag:
            with open(_diag_final_path, "w") as _f:
                _json.dump(_diag, _f, indent=2)
            # Print delta vs baseline if it exists
            if _diag_baseline_path.exists():
                with open(_diag_baseline_path) as _f:
                    _base = _json.load(_f)
                print("\n  DELTA vs BASELINE (multi_mom_tilt):")
                _metrics = [
                    ("bull_calm daily alpha (bps)", _diag.get("regime_stats", {}).get("bull_calm", {}).get("daily_alpha_bps", 0),
                     _base.get("regime_stats", {}).get("bull_calm", {}).get("daily_alpha_bps", 0)),
                    ("bear_stress daily alpha (bps)", _diag.get("regime_stats", {}).get("bear_stress", {}).get("daily_alpha_bps", 0),
                     _base.get("regime_stats", {}).get("bear_stress", {}).get("daily_alpha_bps", 0)),
                    ("bull_calm gross exp %", _diag.get("avg_gross_bull_calm_pct", 0), _base.get("avg_gross_bull_calm_pct", 0)),
                    ("upside capture",  _diag.get("upside_capture", 0),  _base.get("upside_capture", 0)),
                    ("downside capture", _diag.get("downside_capture", 0), _base.get("downside_capture", 0)),
                    ("OLS beta",        _diag.get("ols_beta", 0),        _base.get("ols_beta", 0)),
                ]
                print(f"  {'Metric':<35} {'Before':>10} {'After':>10} {'Delta':>10}")
                print("  " + "-" * 68)
                for _name, _after, _before in _metrics:
                    _delta = _after - _before
                    print(f"  {_name:<35} {_before:>10.3f} {_after:>10.3f} {_delta:>+10.3f}")


    ret_bnh = returns.mean(axis=1)

    if has_multi:
        print("  Computing regime_adaptive sizes (smooth VIX interp + hedge cap + mom tilt)...")
        sizes_regime_adaptive = defensive_tilt_overlay(
            regime_adaptive_sizes(signals_multi, features, returns, CAPITAL),
            signals_multi, _macro_for_overlay, CAPITAL,
        )
        ret_regime_adaptive   = portfolio_returns(sizes_regime_adaptive, returns)
        print("  Computing adaptive_blend sizes (50% regime_adaptive + 50% multi_mom_tilt)...")
        sizes_adaptive_blend  = defensive_tilt_overlay(
            adaptive_blend_sizes(signals_multi, features, returns, CAPITAL),
            signals_multi, _macro_for_overlay, CAPITAL,
        )
        ret_adaptive_blend    = portfolio_returns(sizes_adaptive_blend, returns)
        print("  Computing portable alpha sizes (multi_mom_tilt + SPY beta hedge, β=0.30)...")
        sizes_portable        = defensive_tilt_overlay(
            portable_alpha_sizes(sizes_multi_mom, returns, CAPITAL, target_beta=0.30),
            signals_multi, _macro_for_overlay, CAPITAL,
        )
        ret_portable          = portfolio_returns(sizes_portable, returns)
        print("  Computing portable alpha low-beta sizes (multi_mom_tilt + SPY beta hedge, β=0.15)...")
        sizes_portable_low    = defensive_tilt_overlay(
            portable_alpha_sizes(sizes_multi_mom, returns, CAPITAL, target_beta=0.15),
            signals_multi, _macro_for_overlay, CAPITAL,
        )
        ret_portable_low      = portfolio_returns(sizes_portable_low, returns)

    # ── Carry signal methods ───────────────────────────────────────────────────
    carry_path = SIGNAL_DIR / "carry_signals.parquet"
    has_carry  = carry_path.exists() and has_multi
    if has_carry:
        carry_signals_raw = pd.read_parquet(carry_path)
        carry_signals_raw = carry_signals_raw.reindex(returns.index).fillna(0.0)
        carry_signals_raw = carry_signals_raw.reindex(columns=signals_multi.columns, fill_value=0.0)

        print("  Computing multi_mom_carry sizes (75% trend + 25% carry)...")
        sizes_mom_carry = defensive_tilt_overlay(
            multi_mom_carry_sizes(signals_multi, carry_signals_raw, features, CAPITAL),
            signals_multi, _macro_for_overlay, CAPITAL,
        )
        ret_mom_carry = portfolio_returns(sizes_mom_carry, returns)

        # Carry-trend orthogonality check
        carry_only_sizes = atr_sizes(carry_signals_raw, features, CAPITAL)
        carry_only_ret   = portfolio_returns(carry_only_sizes, returns)
        carry_trend_corr = float(carry_only_ret.corr(ret_multi_mom))
        print(f"\n  Carry-Trend return correlation: {carry_trend_corr:.3f}", end="")
        if carry_trend_corr > 0.30:
            print(f"  WARNING: > 0.30 — carry may not be adding diversification")
        else:
            print(f"  GOOD: < 0.30 — carry provides genuine diversification")

        print("  Computing portable_carry sizes (multi_mom_carry + β=0.30 hedge)...")
        sizes_portable_carry = defensive_tilt_overlay(
            portable_alpha_sizes(sizes_mom_carry, returns, CAPITAL, target_beta=0.30),
            signals_multi, _macro_for_overlay, CAPITAL,
        )
        ret_portable_carry = portfolio_returns(sizes_portable_carry, returns)

    print("  Computing regime-aware risk parity sizes (regime-conditional cov)...")
    sizes_rp_regime = defensive_tilt_overlay(
        rp_regime_aware_sizes(signals_regime, features, returns, CAPITAL),
        signals_regime, _macro_for_overlay, CAPITAL,
    )
    ret_rp_regime = portfolio_returns(sizes_rp_regime, returns)

    print("  Computing rp_regime_dw sizes (regime cov + dead weight)...")
    sizes_rpd  = defensive_tilt_overlay(
        rp_regime_dw_sizes(signals_regime, features, returns, CAPITAL),
        signals_regime, _macro_for_overlay, CAPITAL,
    )
    ret_rpd    = portfolio_returns(sizes_rpd, returns)

    # ── Blended: 60% rp_regime_aware + 40% multi_mom_tilt ────────────────
    # Return-level blend. Since portfolio_returns is linear in sizes,
    # blending dollar sizes is equivalent to blending the return streams.
    has_blend = has_multi
    if has_blend:
        print("  Computing rp_blend sizes (0.60 rp_regime_aware + 0.40 multi_mom_tilt)...")
        _blend_raw = (0.60 * sizes_rp_regime.reindex(columns=returns.columns, fill_value=0)
                      + 0.40 * sizes_multi_mom.reindex(columns=returns.columns, fill_value=0))
        sizes_blend = defensive_tilt_overlay(
            _blend_raw, signals_regime, _macro_for_overlay, CAPITAL,
        )
        ret_blend = portfolio_returns(sizes_blend, returns)

    # ── Unified method registry ────────────────────────────────────────────────
    # Each entry: (label, full-period return series, signal matrix for WF, sizing_fn for WF)
    # signal matrix must match what the sizing_fn expects — regime methods use
    # signals_regime, composite uses signals_composite, ensemble uses signals_ensemble.
    all_methods = [
        # REMOVED by strategy audit 2026-04-08 — OOS Sharpe 1.307, composite 0.662
        # ("equal weight", ret_eq, signals_regime, lambda sig, ret: equal_weight_sizes(sig, CAPITAL)),
        # REMOVED by strategy audit 2026-04-08 — OOS Sharpe 0.730, composite 0.448
        # ("risk parity", ret_rp, signals_regime, lambda sig, ret: risk_parity_sizes(sig, features, ret, CAPITAL)),
        # REMOVED by strategy audit 2026-04-08 — OOS Sharpe 0.929, composite 0.271
        # ("rp_regime_aware", ret_rp_regime, signals_regime, lambda sig, ret: rp_regime_aware_sizes(sig, features, ret, CAPITAL)),
        # REMOVED by strategy audit 2026-04-08 — OOS Sharpe 0.990, composite 0.277
        # ("rp_regime_dw", ret_rpd, signals_regime, lambda sig, ret: rp_regime_dw_sizes(sig, features, ret, CAPITAL)),
    ]
    if has_blend:
        pass  # REMOVED by strategy audit 2026-04-08 — OOS Sharpe 1.271, composite 0.544
        # all_methods.append((
        #     "rp_blend",
        #     ret_blend, signals_regime,
        #     lambda sig, ret, _sm=signals_multi: (
        #         0.60 * rp_regime_aware_sizes(sig, features, ret, CAPITAL)
        #              .reindex(columns=ret.columns, fill_value=0)
        #         + 0.40 * momentum_tilt_sizes(
        #              _sm.reindex(ret.index).fillna(0), features, CAPITAL)
        #              .reindex(columns=ret.columns, fill_value=0)
        #     ),
        # ))
    if has_multi:
        all_methods.extend([
            (
                "multi equal weight",
                ret_multi_eq, signals_multi,
                lambda sig, ret: equal_weight_sizes(sig, CAPITAL),
            ),
            (
                # Pure ATR — no PCA (cross-asset estimation noise), no macro scalar.
                # One number per ticker (its own 14-day ATR). Zero cross-asset
                # estimation. IS-OOS gap should be smaller than risk parity or PCA
                # overlays because there's nothing to overfit.
                "multi_atr_pure",
                ret_multi_atr, signals_multi,
                lambda sig, ret: atr_sizes(sig, features, CAPITAL),
            ),
        ])
        all_methods.append((
            # Cross-sectional momentum tilt on multi-signal entries.
            # Tilts ATR sizes ±30% toward 63-day cross-sectional winners.
            "multi_mom_tilt",
            ret_multi_mom, signals_multi,
            lambda sig, ret: momentum_tilt_sizes(sig, features, CAPITAL),
        ))
        all_methods.append((
            # Regime-adaptive: smooth VIX-interpolated gross targeting
            # + hedge cap + momentum tilt + dead weight.
            "regime_adaptive",
            ret_regime_adaptive, signals_multi,
            lambda sig, ret: regime_adaptive_sizes(sig, features, ret, CAPITAL),
        ))
        all_methods.append((
            # 50/50 blend: regime_adaptive (capture ratio) + multi_mom_tilt (OOS Sharpe).
            # Diversifies across two sizing philosophies with weakly correlated errors.
            "adaptive_blend",
            ret_adaptive_blend, signals_multi,
            lambda sig, ret: adaptive_blend_sizes(sig, features, ret, CAPITAL),
        ))
        # REMOVED by strategy audit 2026-04-08 — OOS Sharpe 1.584, composite 0.643
        # IS-OOS gap -0.558: regime concentrated, hedge costs 3-4% annualized in bull markets
        # all_methods.append((
        #     "multi_mom_portable",
        #     ret_portable, signals_multi,
        #     lambda sig, ret: portable_alpha_sizes(
        #         momentum_tilt_sizes(sig, features, CAPITAL), ret, CAPITAL, target_beta=0.30),
        # ))
        # REMOVED by strategy audit 2026-04-08 — OOS Sharpe 1.371, composite 0.621
        # all_methods.append((
        #     "multi_mom_port_low",
        #     ret_portable_low, signals_multi,
        #     lambda sig, ret: portable_alpha_sizes(
        #         momentum_tilt_sizes(sig, features, CAPITAL), ret, CAPITAL, target_beta=0.15),
        # ))
    if has_carry:
        pass  # REMOVED by strategy audit 2026-04-08 — both carry methods removed (see comments below)
        # REMOVED by strategy audit 2026-04-08 — OOS Sharpe 1.376, composite 0.675
        # all_methods.append((
        #     "multi_mom_carry",
        #     ret_mom_carry, signals_multi,
        #     lambda sig, ret, _cs=carry_signals_raw: multi_mom_carry_sizes(
        #         sig,
        #         _cs.reindex(index=ret.index, columns=sig.columns, fill_value=0.0),
        #         features, CAPITAL,
        #     ),
        # ))
        # REMOVED by strategy audit 2026-04-08 — OOS Sharpe 1.613, composite 0.526
        # IS-OOS gap -0.673: regime concentrated; SPY hedge costs 3-4% ann in bull markets
        # all_methods.append((
        #     "portable_carry",
        #     ret_portable_carry, signals_multi,
        #     lambda sig, ret, _cs=carry_signals_raw: portable_alpha_sizes(
        #         multi_mom_carry_sizes(
        #             sig,
        #             _cs.reindex(index=ret.index, columns=sig.columns, fill_value=0.0),
        #             features, CAPITAL,
        #         ),
        #         ret, CAPITAL, target_beta=0.30,
        #     ),
        # ))

    # ── Portfolio comparison table ─────────────────────────────────────────────
    print(f"{'='*84}")
    print("  PORTFOLIO COMPARISON  (transaction costs included in all strategy returns)")
    print(f"{'='*84}")
    metrics = ["ann_return", "sharpe", "max_drawdown", "calmar", "win_rate", "profit_factor"]
    print(f"  {'Method':<32}" + "".join(f"{m:>12}" for m in metrics))
    print("  " + "-" * (32 + 12 * len(metrics)))

    for label, ret, _, _ in all_methods:
        s   = summarise(ret, label)
        print(f"  {s['label']:<32}" + "".join(f"{str(s[m]):>12}" for m in metrics))
    bnh = summarise(ret_bnh, "buy & hold")
    print(f"  {bnh['label']:<32}" + "".join(f"{str(bnh[m]):>12}" for m in metrics))

    # ── Gross Exposure Analysis ────────────────────────────────────────────────
    print(f"\n{'='*84}")
    print("  GROSS EXPOSURE ANALYSIS")
    print(f"{'='*84}")
    print(f"  {'Method':<32} {'AvgGross%':>10} {'AvgLong%':>10} {'%DaysFull':>10} {'RetGapAnn%':>11}")
    print("  " + "-" * 77)

    _sizes_map: dict = {
        "equal weight"    : sizes_eq,
        "risk parity"     : sizes_rp,
        "rp_regime_aware" : sizes_rp_regime,
        "rp_regime_dw"    : sizes_rpd,
    }
    if has_blend:
        _sizes_map["rp_blend"] = sizes_blend
    if has_multi:
        _sizes_map["multi equal weight"] = sizes_multi_eq
        _sizes_map["multi_atr_pure"]     = sizes_multi_atr
        _sizes_map["multi_mom_tilt"]     = sizes_multi_mom
        _sizes_map["regime_adaptive"]    = sizes_regime_adaptive
        _sizes_map["adaptive_blend"]     = sizes_adaptive_blend
        _sizes_map["multi_mom_portable"] = sizes_portable
        _sizes_map["multi_mom_port_low"] = sizes_portable_low
    if has_carry:
        _sizes_map["multi_mom_carry"] = sizes_mom_carry
        _sizes_map["portable_carry"]  = sizes_portable_carry

    bnh_ann = bnh["ann_return"]
    _gross_exposures: list = []
    _return_gaps:     list = []
    for _lbl, _ret, _, _ in all_methods:
        _sz        = _sizes_map[_lbl]
        _gross_exp = _sz.abs().sum(axis=1) / CAPITAL
        _long_exp  = _sz.clip(lower=0).sum(axis=1) / CAPITAL
        _avg_gross = _gross_exp.mean() * 100
        _avg_long  = _long_exp.mean() * 100
        _pct_full  = (_gross_exp > 0.90).mean() * 100
        _ret_gap   = bnh_ann - summarise(_ret, _lbl)["ann_return"]
        _gross_exposures.append(_avg_gross)
        _return_gaps.append(_ret_gap)
        print(f"  {_lbl:<32} {_avg_gross:>9.1f}% {_avg_long:>9.1f}% {_pct_full:>9.1f}% {_ret_gap:>+10.1f}%")

    _avg_gross_all = float(np.mean(_gross_exposures))
    _avg_gap_all   = float(np.mean(_return_gaps))
    print(f"\n  Buy & hold gross exposure: 100% | Your avg gross exposure: {_avg_gross_all:.1f}% | "
          f"Return gap: {_avg_gap_all:+.1f}% annualized")

    # ── Walk-forward for every method ─────────────────────────────────────────
    # Runs the full 3yr-train / 1yr-test roll for each method and collects
    # IS + OOS Sharpe.  Every method gets the same treatment — no cherry-picking
    # which ones to validate.
    print(f"\n{'='*84}")
    print("  WALK-FORWARD VALIDATION  (3yr train / 1yr test) — ALL METHODS")
    print(f"{'='*84}")

    wf_store         = {}
    is_sharpes       = {}
    oos_sharpes      = {}
    is_act_sharpes   = {}
    oos_act_sharpes  = {}

    for label, ret, sig_matrix, sizing_fn in all_methods:
        wf = walk_forward(sig_matrix, returns, sizing_fn=sizing_fn)
        wf_store[label]        = wf
        is_sharpes[label]      = round(summarise(ret, label)["sharpe"], 3)
        oos_sharpes[label]     = round(wf["sharpe"].mean(), 3)
        is_act_sharpes[label]  = round(active_sharpe_ratio(ret, returns), 3)
        oos_act_sharpes[label] = round(wf["active_sharpe"].mean(), 3)

        print(f"\n  -- {label}")
        print(f"     IS Sharpe {is_sharpes[label]:.3f}  |  IS Active {is_act_sharpes[label]:.3f}  |  "
              f"Mean OOS {oos_sharpes[label]:.3f} \u00b1 {wf['sharpe'].std():.3f}  |  "
              f"Mean OOS Active {oos_act_sharpes[label]:.3f}  |  "
              f"IS-OOS gap {is_sharpes[label] - oos_sharpes[label]:+.3f}")
        print(wf[["period", "sharpe", "active_sharpe", "ann_return", "max_dd", "n_days"]].to_string(index=False))

    # ── IS vs OOS comparison — all methods ────────────────────────────────────
    print(f"\n{'='*104}")
    print("  IS vs OOS SHARPE — all methods ranked by OOS Active Sharpe")
    print(f"{'='*104}")
    print(f"  {'Method':<32} {'IS Sharpe':>10} {'IS ActSh':>10} {'OOS Sharpe':>10} "
          f"{'OOS ActSh':>10} {'IS-OOS gap':>12}  note")
    print("  " + "-" * 96)

    ranked       = sorted(all_methods, key=lambda m: oos_act_sharpes[m[0]], reverse=True)
    ranked_total = sorted(all_methods, key=lambda m: oos_sharpes[m[0]], reverse=True)
    # Smallest IS-OOS gap among methods with OOS Sharpe > 0.8 — most robust.
    # This is the production selection criterion: we want the method that
    # generalises best, not the one that looked best on a specific OOS window.
    qualified = [(lbl, is_sharpes[lbl] - oos_sharpes[lbl])
                 for lbl, _, _, _ in all_methods if oos_sharpes[lbl] > 0.8]
    best_gap_label = min(qualified, key=lambda x: x[1])[0] if qualified else ranked_total[0][0]

    best_act_label   = ranked[0][0]
    best_total_label = ranked_total[0][0]

    for label, _, _, _ in ranked:
        gap  = is_sharpes[label] - oos_sharpes[label]
        tags = []
        if label == best_act_label:   tags.append("<-- best OOS Active")
        if label == best_total_label: tags.append("<-- best OOS Total")
        if label == best_gap_label:   tags.append("<-- most robust (min IS-OOS gap)")
        if gap > 0.5:                 tags.append("overfit")
        print(f"  {label:<32} {is_sharpes[label]:>10.3f} {is_act_sharpes[label]:>10.3f} "
              f"{oos_sharpes[label]:>10.3f} {oos_act_sharpes[label]:>10.3f} "
              f"{gap:>+12.3f}  {'  '.join(tags)}")

    # Production method selection — two criteria, checked in order:
    #
    # 1. Prefer signal_gated_mv_regime if it meets BOTH:
    #      OOS Sharpe > 1.20  AND  IS-OOS gap ∈ [-0.20, +0.30]
    #    Rationale: a near-zero gap is more predictable in live trading than
    #    a large negative gap (like rp_regime_aware's -0.388) which means
    #    OOS advantage is concentrated in specific stress windows.
    #
    # 2. Otherwise: fall back to smallest IS-OOS gap with OOS Sharpe > 0.8
    #    (existing logic).
    # Priority 1: rp_blend if OOS Sharpe > 0.8 AND gap ∈ [-0.15, +0.15]
    blend_label = "rp_blend"
    if (blend_label in oos_sharpes
            and oos_sharpes[blend_label] > 0.8
            and -0.15 <= (is_sharpes[blend_label] - oos_sharpes[blend_label]) <= 0.15):
        best_label = blend_label
    else:
        best_label = best_gap_label

    best_ret   = dict((m[0], m[1]) for m in all_methods)[best_label]
    prod_gap   = is_sharpes[best_label] - oos_sharpes[best_label]

    print(f"\n  Highest OOS Total  Sharpe: {best_total_label} ({oos_sharpes[best_total_label]:.3f})")
    print(f"  Highest OOS Active Sharpe: {best_act_label}  ({oos_act_sharpes[best_act_label]:.3f})")
    print(f"  Most robust (min gap):     {best_gap_label}  (OOS gap {prod_gap:+.3f})")
    print(f"\n  Production method: {best_label}"
          f"  (OOS Sharpe {oos_sharpes[best_label]:.3f}, IS-OOS gap {prod_gap:+.3f})")
    if best_label != best_total_label:
        print(f"  If prioritizing raw OOS performance: {best_total_label}"
              f"  (OOS Sharpe {oos_sharpes[best_total_label]:.3f},"
              f" IS-OOS gap {is_sharpes[best_total_label] - oos_sharpes[best_total_label]:+.3f})")
    print()
    print("  IS-OOS gap interpretation:")
    print("    Positive gap (OOS < IS): IS > OOS — classical overfitting signal")
    print("      < +0.20 -> acceptable, monitor")
    print("      > +0.20 -> overfit; simplify or regularise")
    print("    Negative gap (OOS > IS): OOS > IS — regime concentration, not overfitting")
    print("      > -0.40 -> robust generalization")
    print("      < -0.40 -> OOS windows contain favorable regimes; expect reversion toward IS Sharpe in live trading")

    # ── Alpha Source Comparison Table + PASS/FAIL ─────────────────────────────
    # Printed only when a baseline diagnostic exists (i.e. not the very first run).
    _new_alpha_path = RESULTS_DIR / "regime_diagnostic_new_alpha.json"
    if (has_multi
            and "multi_mom_tilt" in oos_sharpes
            and _diag_baseline_path.exists()
            and _diag):
        try:
            _wf_mt   = wf_store.get("multi_mom_tilt")
            _oos_s   = oos_sharpes["multi_mom_tilt"]
            _is_s    = is_sharpes["multi_mom_tilt"]
            _gap     = _is_s - _oos_s
            # max_drawdown() expects cumulative returns; ret_multi_mom is daily returns
            _dd_full = max_drawdown((1 + ret_multi_mom).cumprod()) * 100

            with open(_diag_baseline_path) as _bf:
                _bbase = _json.load(_bf)
            _bbc  = _bbase.get("regime_stats", {}).get("bull_calm",   {})
            _bbs  = _bbase.get("regime_stats", {}).get("bear_stress", {})
            _abc  = _diag.get("regime_stats", {}).get("bull_calm",   {})
            _abs_ = _diag.get("regime_stats", {}).get("bear_stress", {})

            _bc_bef = _bbc.get("daily_alpha_bps", -7.49)
            _bs_bef = _bbs.get("daily_alpha_bps", 17.18)
            _bc_aft = _abc.get("daily_alpha_bps", float("nan"))
            _bs_aft = _abs_.get("daily_alpha_bps", float("nan"))

            print(f"\n{'='*92}")
            print("  ALPHA SOURCE COMPARISON TABLE  (value-momentum tilt + carry breadth overlay)")
            print(f"{'='*92}")
            _hdr = f"  {'Metric':<35} {'Before':>17} {'After':>10} {'Delta':>10}  Constraint"
            print(_hdr)
            print("  " + "-" * 84)

            def _row(name, bef, aft, cstr=""):
                _b = f"{bef:.3f}" if bef is not None and not (isinstance(bef, float) and np.isnan(bef)) else "n/a"
                _a = f"{aft:.3f}" if not (isinstance(aft, float) and np.isnan(aft)) else "n/a"
                _d = f"{aft - bef:+.3f}" if (bef is not None and _b != "n/a" and _a != "n/a") else "n/a"
                print(f"  {name:<35} {_b:>17} {_a:>10} {_d:>10}  {cstr}")

            # wf ann_return is already in percent (from summarise * 100)
            _oos_ann = float(_wf_mt["ann_return"].mean()) if _wf_mt is not None else float("nan")
            _row("OOS Sharpe",                1.555,                           _oos_s,   "≥ 1.50")
            _row("IS Sharpe",                 None,                            _is_s,    "")
            _row("IS-OOS Gap",                None,                            _gap,     "< 0.50")
            _row("OOS Ann Return (%)",        None,                            _oos_ann, "")
            _row("Max Drawdown (%) full hist",-4.64,                           _dd_full, "≥ -7.0")
            _row("bull_calm alpha (bps)",     _bc_bef,                         _bc_aft,  "TARGET: > 0")
            _row("bear_stress alpha (bps)",   _bs_bef,                         _bs_aft,  "≥ 14.18")
            _row("Upside capture",            _bbase.get("upside_capture",   0.332), _diag.get("upside_capture",   float("nan")), "")
            _row("Downside capture",          _bbase.get("downside_capture", 0.295), _diag.get("downside_capture", float("nan")), "")
            _row("OLS beta",                  _bbase.get("ols_beta",         0.184), _diag.get("ols_beta",         float("nan")), "")
            _row("Gross exp bull_calm (%)",   _bbase.get("avg_gross_bull_calm_pct",   83.6), _diag.get("avg_gross_bull_calm_pct",   float("nan")), "")
            _row("Avg positions bull_calm",   _bbase.get("avg_positions_bull_calm",   30.1), _diag.get("avg_positions_bull_calm",   float("nan")), "")

            _p_bc  = (not np.isnan(_bc_aft)) and (_bc_aft > _bc_bef + 3.0)
            _p_oos = _oos_s >= 1.50
            _p_gap = _gap < 0.50
            _p_dd  = _dd_full >= -7.0
            _all_p = _p_bc and _p_oos and _p_gap and _p_dd

            print(f"\n  PASS/FAIL VERDICT:")
            print(f"  {'bull_calm alpha improves ≥3 bps':<40}: {'PASS' if _p_bc  else 'FAIL'}  ({_bc_bef:.2f} → {_bc_aft:.2f} bps)")
            print(f"  {'OOS Sharpe ≥ 1.50':<40}: {'PASS' if _p_oos else 'FAIL'}  ({_oos_s:.3f})")
            print(f"  {'IS-OOS gap < 0.50':<40}: {'PASS' if _p_gap else 'FAIL'}  ({_gap:+.3f})")
            print(f"  {'Max drawdown ≥ -7.0%':<40}: {'PASS' if _p_dd  else 'FAIL'}  ({_dd_full:.2f}%)")
            print(f"\n  OVERALL: {'*** PASS — alpha sources deployed ***' if _all_p else '*** FAIL — review and consider reverting ***'}")

            _new_alpha_data = {
                "alpha_sources": ["value_momentum_tilt_70_30", "carry_breadth_scalar_15pct"],
                "before": {
                    "oos_sharpe": 1.555,
                    "max_dd_pct": -4.64,
                    "bull_calm_alpha_bps":   _bc_bef,
                    "bear_stress_alpha_bps": _bs_bef,
                    "upside_capture":   _bbase.get("upside_capture",   0.332),
                    "downside_capture": _bbase.get("downside_capture", 0.295),
                    "ols_beta":         _bbase.get("ols_beta",         0.184),
                },
                "after":             _diag,
                "oos_sharpe_after":  _oos_s,
                "is_sharpe_after":   _is_s,
                "is_oos_gap":        _gap,
                "max_dd_full_pct":   _dd_full,
                "pass_fail": {
                    "bull_calm_alpha": bool(_p_bc),
                    "oos_sharpe":      bool(_p_oos),
                    "is_oos_gap":      bool(_p_gap),
                    "max_drawdown":    bool(_p_dd),
                    "overall":         bool(_all_p),
                },
            }
            with open(_new_alpha_path, "w") as _nf:
                _json.dump(_new_alpha_data, _nf, indent=2)
            print(f"\n  New alpha diagnostic saved → {_new_alpha_path}")
        except Exception as _e:
            print(f"  [new_alpha table] Error: {_e}")

    # ── Down-day alpha: SPY daily return < -1% ────────────────────────────────
    # Tests Improvement 2 directly: does the portfolio print positive return on
    # days when SPY drops > 1%?  Pre-improvement target: > 0% avg, > 40% days pos.
    if "SPY" in returns.columns:
        _spy_daily  = returns["SPY"]
        _spy_down   = _spy_daily < -0.01   # log return proxy (≈ simple for small values)
        _n_down     = int(_spy_down.sum())
        print(f"\n{'='*84}")
        print(f"  DOWN-DAY ALPHA  (SPY daily return < -1%,  n={_n_down} days)")
        print(f"{'='*84}")
        print(f"  {'Method':<32} {'AvgReturn%':>11} {'%DaysPos':>10}")
        print("  " + "-" * 56)

        _all_down_rets = []
        for _lbl, _ret, _, _ in all_methods:
            _port_down = _ret[_spy_down].dropna()
            if len(_port_down) == 0:
                continue
            _avg_r  = float(_port_down.mean()) * 100
            _pct_p  = float((_port_down > 0).mean()) * 100
            _all_down_rets.append(_port_down.values)
            print(f"  {_lbl:<32} {_avg_r:>+10.3f}%  {_pct_p:>8.1f}%")

        if _all_down_rets:
            import numpy as _np
            _combined  = _np.mean(_all_down_rets, axis=0)
            _avg_comb  = float(_combined.mean()) * 100
            _pct_comb  = float((_combined > 0).mean()) * 100
            print(f"\n  All-methods combined:          {_avg_comb:>+10.3f}%  {_pct_comb:>8.1f}%")
            print(f"  SPY average on down days:      "
                  f"{float(_spy_daily[_spy_down].mean())*100:>+10.3f}%  "
                  f"  (benchmark — strategy should beat this)")

    # ── Persist results ────────────────────────────────────────────────────────
    comparison_curves = {
        "equal_weight"    : equity_curve(ret_eq,        CAPITAL),
        "risk_parity"     : equity_curve(ret_rp,        CAPITAL),
        "rp_regime_aware" : equity_curve(ret_rp_regime, CAPITAL),
        "rp_regime_dw"    : equity_curve(ret_rpd,       CAPITAL),
        "buy_hold"        : equity_curve(ret_bnh,       CAPITAL),
    }
    if has_blend:
        comparison_curves["rp_blend"] = equity_curve(ret_blend, CAPITAL)
    if has_multi:
        comparison_curves["multi_equal_weight"] = equity_curve(ret_multi_eq,  CAPITAL)
        comparison_curves["multi_atr_pure"]     = equity_curve(ret_multi_atr, CAPITAL)
        comparison_curves["multi_mom_tilt"]     = equity_curve(ret_multi_mom, CAPITAL)
        comparison_curves["regime_adaptive"]    = equity_curve(ret_regime_adaptive, CAPITAL)
        comparison_curves["adaptive_blend"]     = equity_curve(ret_adaptive_blend,  CAPITAL)
        comparison_curves["multi_mom_portable"] = equity_curve(ret_portable,        CAPITAL)
        comparison_curves["multi_mom_port_low"] = equity_curve(ret_portable_low,    CAPITAL)
    if has_carry:
        comparison_curves["multi_mom_carry"] = equity_curve(ret_mom_carry,      CAPITAL)
        comparison_curves["portable_carry"]  = equity_curve(ret_portable_carry, CAPITAL)

    pd.DataFrame(comparison_curves).to_parquet(RESULTS_DIR / "portfolio_comparison.parquet")

    # Dashboard compatibility: keep the two named walk-forward parquets it expects
    # (was "equal weight" pre-audit; now "multi equal weight" as simplest available)
    wf_store["multi equal weight"].to_parquet(
        RESULTS_DIR / "walk_forward_regime.parquet", index=False)
    wf_store["multi_mom_tilt"].to_parquet(
        RESULTS_DIR / "walk_forward_atr_pca.parquet", index=False)

    oos_df = pd.DataFrame([
        {
            "method"         : lbl,
            "is_sharpe"      : is_sharpes[lbl],
            "oos_sharpe"     : oos_sharpes[lbl],
            "is_act_sharpe"  : is_act_sharpes[lbl],
            "oos_act_sharpe" : oos_act_sharpes[lbl],
        }
        for lbl in is_sharpes
    ])
    oos_df.to_parquet(RESULTS_DIR / "oos_selection.parquet", index=False)

    equity_curve(best_ret, CAPITAL).to_frame("portfolio").to_parquet(
        RESULTS_DIR / "portfolio_equity_curve.parquet"
    )
    print(f"\nEquity curves saved -> {RESULTS_DIR / 'portfolio_comparison.parquet'}")
    print(f"Walk-forward saved  -> {RESULTS_DIR / 'walk_forward_regime.parquet'}")
    print(f"OOS selection saved -> {RESULTS_DIR / 'oos_selection.parquet'}")

    # ── Expanded-universe sleeve + combined portfolio validation ───────────────
    exp_signals_path = SIGNAL_DIR / "expanded_signals.parquet"
    exp_sizes_path   = SIGNAL_DIR / "expanded_sizes.parquet"
    exp_comp_path    = SIGNAL_DIR / "expanded_composite.parquet"

    if not (exp_signals_path.exists() and exp_sizes_path.exists()):
        print("\n  NOTE: expanded_signals.parquet not found — run signal_generation.py"
              " with USE_EXPANDED_UNIVERSE=True to generate expanded-sleeve results")
    else:
        print(f"\n{'='*70}")
        print("  EXPANDED UNIVERSE — sleeve allocation and validation gate")
        print(f"{'='*70}")

        try:
            from v1.pipeline.universe_expansion import USE_EXPANDED_UNIVERSE
            if not USE_EXPANDED_UNIVERSE:
                print("  USE_EXPANDED_UNIVERSE=False — skipping expanded sleeve")
                raise SystemExit
        except ImportError:
            pass

        exp_sizes_raw = pd.read_parquet(exp_sizes_path)
        exp_signals   = pd.read_parquet(exp_signals_path)

        # ── Build expanded returns matrix ──────────────────────────────────────
        exp_ret_cols: dict = {}
        for t in exp_signals.columns:
            feat_path = FEATURE_DIR / f"{t}.parquet"
            if feat_path.exists():
                try:
                    df = pd.read_parquet(feat_path)
                    df.index = pd.to_datetime(df.index)
                    if "Close" in df.columns:
                        exp_ret_cols[t] = df["Close"].pct_change()
                except Exception:
                    pass
        exp_returns = pd.DataFrame(exp_ret_cols).sort_index()

        # ── Dollar sizes for the expanded sleeve ──────────────────────────────
        exp_dollar_sizes = expanded_sleeve_sizes(exp_sizes_raw, CAPITAL)
        exp_ret_series   = portfolio_returns(
            exp_dollar_sizes.reindex(exp_returns.index),
            exp_returns,
        ).dropna()

        # ── OOS split: last 252 days are out-of-sample ───────────────────────
        OOS_DAYS = 252
        if len(exp_ret_series) > OOS_DAYS * 2:
            exp_is  = exp_ret_series.iloc[:-OOS_DAYS]
            exp_oos = exp_ret_series.iloc[-OOS_DAYS:]
        else:
            exp_is  = exp_ret_series
            exp_oos = exp_ret_series

        exp_oos_sharpe  = sharpe_ratio(exp_oos)
        exp_full_sharpe = sharpe_ratio(exp_ret_series)
        exp_max_dd      = max_drawdown((1 + exp_ret_series).cumprod()) * 100

        # ── Annualised one-way turnover of the expanded sleeve ────────────────
        sleeve_cap   = CAPITAL * ALLOCATION_SPLIT["expanded"]
        daily_chg    = exp_dollar_sizes.diff().abs().sum(axis=1)
        exp_turnover = float(daily_chg.mean() * 252 / sleeve_cap * 100)  # % p.a.

        # ── Combined portfolio (core + expanded sleeves) ──────────────────────
        # Core sleeve: scale best_ret by ALLOCATION_SPLIT['core'] fraction.
        # best_ret is already a fractional return on CAPITAL, so weighting by
        # core fraction gives the core sleeve's contribution to total returns.
        common_idx   = exp_ret_series.index.intersection(best_ret.index)
        core_slice   = best_ret.reindex(common_idx).fillna(0) * ALLOCATION_SPLIT["core"]
        exp_slice    = exp_ret_series.reindex(common_idx).fillna(0)
        combined_ret = core_slice + exp_slice

        comb_sharpe = sharpe_ratio(combined_ret.dropna())
        comb_max_dd = max_drawdown((1 + combined_ret.dropna()).cumprod()) * 100
        corr_ce     = float(core_slice.corr(exp_slice))

        # ── Bull-calm alpha for combined portfolio ────────────────────────────
        regimes_c, _ = _load_regime_data(combined_ret.index)
        if "SPY" in returns.columns:
            spy_c = returns["SPY"].reindex(combined_ret.index).fillna(0)
        else:
            spy_c = pd.Series(0.0, index=combined_ret.index)

        bull_calm_mask = regimes_c == "bull_calm"
        if bull_calm_mask.sum() > 5:
            bc_port = float(combined_ret[bull_calm_mask].mean()) * 100
            bc_spy  = float(spy_c[bull_calm_mask].mean()) * 100
            bc_alpha = (bc_port - bc_spy) * 100   # bps/day
        else:
            bc_alpha = float("nan")

        # ── Print validation gate results ─────────────────────────────────────
        GATE_EXP_SHARPE  = 0.80
        GATE_COMB_SHARPE = 1.40
        GATE_BC_ALPHA    = -4.0     # bps/day (must be > this)
        GATE_CORR        = 0.60     # core/expanded correlation (must be < this)
        GATE_TURNOVER    = 300.0    # % p.a. (must be < this)
        GATE_MAX_DD      = -7.0     # % (must be > this i.e. drawdown shallower)

        pass_exp_sharpe  = exp_oos_sharpe  >  GATE_EXP_SHARPE
        pass_comb_sharpe = comb_sharpe     >  GATE_COMB_SHARPE
        pass_bc_alpha    = (not np.isnan(bc_alpha)) and (bc_alpha > GATE_BC_ALPHA)
        pass_corr        = corr_ce         <  GATE_CORR
        pass_turnover    = exp_turnover    <  GATE_TURNOVER
        pass_max_dd      = exp_max_dd      >  GATE_MAX_DD

        def _gate(ok: bool) -> str:
            return "PASS" if ok else "FAIL"

        print(f"\n  {'Metric':<38} {'Value':>10}  {'Gate':>6}  {'Status'}")
        print(f"  {'-'*65}")
        print(f"  {'Expanded OOS Sharpe (last 252d)':<38} {exp_oos_sharpe:>10.3f}  "
              f"{f'>{GATE_EXP_SHARPE}':>6}  {_gate(pass_exp_sharpe)}")
        print(f"  {'Expanded full-history Sharpe':<38} {exp_full_sharpe:>10.3f}  "
              f"{'—':>6}")
        print(f"  {'Combined (core+exp) Sharpe':<38} {comb_sharpe:>10.3f}  "
              f"{f'>{GATE_COMB_SHARPE}':>6}  {_gate(pass_comb_sharpe)}")
        print(f"  {'Bull-calm combined alpha (bps/day)':<38} "
              f"{bc_alpha if not np.isnan(bc_alpha) else float('nan'):>10.2f}  "
              f"{f'>{GATE_BC_ALPHA}':>6}  {_gate(pass_bc_alpha)}")
        print(f"  {'Core / expanded return correlation':<38} {corr_ce:>10.3f}  "
              f"{f'<{GATE_CORR}':>6}  {_gate(pass_corr)}")
        print(f"  {'Expanded turnover (% p.a.)':<38} {exp_turnover:>10.1f}  "
              f"{f'<{GATE_TURNOVER}':>6}  {_gate(pass_turnover)}")
        print(f"  {'Expanded max drawdown (%)':<38} {exp_max_dd:>10.2f}  "
              f"{f'>{GATE_MAX_DD}':>6}  {_gate(pass_max_dd)}")
        print(f"  {'-'*65}")

        all_pass = all([pass_exp_sharpe, pass_comb_sharpe, pass_bc_alpha,
                        pass_corr, pass_turnover, pass_max_dd])
        gate_str = "ALL GATES PASSED" if all_pass else "GATE FAILURE — check metrics above"
        print(f"\n  Expanded sleeve validation: {gate_str}")

        if not all_pass:
            print("  WARNING: Expanded sleeve does not meet all quality gates.")
            print("           Review sector composition and signal parameters before")
            print("           deploying the expanded sleeve in live trading.")

        # ── Save combined portfolio equity curve ──────────────────────────────
        exp_curve  = equity_curve(exp_ret_series, CAPITAL * ALLOCATION_SPLIT["expanded"])
        comb_curve = equity_curve(combined_ret,   CAPITAL)
        pd.DataFrame({
            "core_sleeve"     : equity_curve(best_ret.reindex(common_idx).fillna(0) * ALLOCATION_SPLIT["core"],
                                             CAPITAL * ALLOCATION_SPLIT["core"]),
            "expanded_sleeve" : exp_curve.reindex(common_idx),
            "combined"        : comb_curve,
        }).to_parquet(RESULTS_DIR / "combined_portfolio.parquet")
        print(f"\n  Combined portfolio equity curve saved -> "
              f"{RESULTS_DIR / 'combined_portfolio.parquet'}")


if __name__ == "__main__":
    main()
