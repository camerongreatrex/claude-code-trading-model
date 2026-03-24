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

from .risk_model import estimate_covariance, risk_parity_weights
from .backtester import (
    compute_strategy_returns, sharpe_ratio, max_drawdown,
    calmar_ratio, win_rate, profit_factor, summarise, equity_curve
)
from .data_pipeline import TICKER_LIST, ASSET_CLASS, HEDGE_MAP

SIGNAL_DIR  = Path("data/signals")
FEATURE_DIR = Path("data/features")
MACRO_DIR   = Path("data/macro")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL          = 100_000
MAX_POSITION_PCT = 0.20
RISK_PER_TRADE   = 0.01


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
    col_locs = {t: signals.columns.get_loc(t) for t in tickers}

    # Cache risk-parity weights for the full universe; update monthly.
    # Weights are indexed by ticker name so they survive active-set changes.
    rp_weights_cache: dict = {}   # {ticker: weight}

    for i in range(cov_window, len(signals)):
        # Monthly rebalance: refit covariance on the most recent cov_window days.
        # We compute weights for the full universe regardless of which assets
        # are active today — the active mask is applied below at signal time.
        if (i - cov_window) % rebalance_freq == 0:
            ret_window = returns.iloc[i - cov_window : i][tickers]
            # Drop any ticker with >10% missing returns (illiquid / just listed)
            ret_window = ret_window.dropna(thresh=int(len(ret_window) * 0.90), axis=1)
            available  = ret_window.columns.tolist()

            if len(available) >= 2 and len(ret_window) >= 20:
                ret_clean = ret_window.ffill().fillna(0)
                cov       = estimate_covariance(ret_clean)
                w_rp      = risk_parity_weights(cov)
                rp_weights_cache = {t: float(w_rp[j]) for j, t in enumerate(available)}
            # else: keep previous cache (or empty → fall through to ATR below)

        if not rp_weights_cache:
            continue  # warm-up: not enough history yet, keep ATR sizes

        sig_today = signals.iloc[i]
        active    = [t for t in tickers if sig_today[t] != 0 and t in rp_weights_cache]
        if not active:
            continue

        # Total gross ATR exposure today (preserve the overall size level)
        total_atr = base_sizes.iloc[i][[t for t in active]].abs().sum()
        if total_atr <= 0:
            continue

        # Renormalise risk-parity weights among today's active assets
        w_sum = sum(rp_weights_cache[t] for t in active)
        if w_sum <= 0:
            continue

        # Redistribute ATR total using risk-parity proportions
        for ticker in active:
            w_norm = rp_weights_cache[ticker] / w_sum
            sign   = float(sig_today[ticker])
            adjusted.iloc[i, col_locs[ticker]] = np.clip(
                sign * w_norm * total_atr,
                -capital * MAX_POSITION_PCT,
                capital * MAX_POSITION_PCT,
            )

    return apply_macro_multiplier(adjusted)


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
             = 0.70 at rank 0.0  (worst 3-month performer  → 30% underweight)
             = 1.00 at rank 0.5  (median performer         → neutral weight)
             = 1.30 at rank 1.0  (best 3-month performer   → 30% overweight)

    The rank is computed cross-sectionally across ONLY the tickers that
    have an active signal on that date (signal ≠ 0).  Flat tickers are
    excluded — their rank would be meaningless and would dilute the signal.

    The ±30% tilt (tilt_min=0.7, tilt_range=0.6) is deliberately conservative.
    Most quant equity funds use ±50–100% momentum tilts; ±30% preserves
    diversification while capturing cross-sectional momentum alpha.

    Window choice (63 days):
        Jegadeesh and Titman (1993) showed that 3–12-month momentum
        is strongest and most persistent.  63 trading days (≈3 months)
        is the published sweet spot: shorter captures mean-reversion,
        longer captures reversal.  This window is not fitted to this data.

    Args:
        signals:    Signal DataFrame (T × N), any signal type.
        features:   Dict[ticker → feature DataFrame] with Close prices.
        capital:    Starting capital in dollars.
        tilt_min:   Minimum tilt factor (default 0.7 = worst-performer weight).
        tilt_range: Tilt range (default 0.6 → tilt spans [0.7, 1.3]).
        mom_window: Trailing return window for cross-sectional ranking (default 63).

    Returns:
        Dollar position size DataFrame.  Clipped to ±MAX_POSITION_PCT × capital.
    """
    # Step 1: ATR base sizes — normalise for per-asset volatility.
    base = atr_sizes(signals, features, capital)

    # Step 2: Build trailing close-price matrix for momentum ranking.
    # Only include tickers that have feature data and appear in signals.
    close_cols = {t: features[t]["Close"].reindex(base.index).ffill()
                  for t in signals.columns if t in features}
    if not close_cols:
        return base
    closes = pd.DataFrame(close_cols)

    # Step 3: 63-day trailing return for each ticker, computed daily.
    mom = closes.pct_change(mom_window)   # (T × N), backward-looking only

    # Step 4: On each day, rank ONLY active tickers (signal ≠ 0) by momentum.
    # Tickers with no active signal get a neutral tilt of 1.0 (base size).
    tilt = pd.DataFrame(1.0, index=base.index, columns=base.columns)

    for date in base.index:
        active_mask = signals.loc[date].abs() > 0
        active_cols = [c for c in active_mask.index if active_mask[c] and c in mom.columns]
        if len(active_cols) < 2:
            continue   # need ≥ 2 active tickers to form a meaningful rank

        mom_row = mom.loc[date, active_cols].dropna()
        if len(mom_row) < 2:
            continue

        # Percentile rank within the active set for this date.
        ranks = mom_row.rank(pct=True)   # 0 = worst, 1 = best
        for col, rank_val in ranks.items():
            tilt.loc[date, col] = tilt_min + tilt_range * rank_val

    # Step 5: Apply tilt and re-apply gross exposure cap.
    tilted = base * tilt
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

    fast_path  = SIGNAL_DIR / "fast_overlay_signals.parquet"
    has_fast   = fast_path.exists()
    if has_fast:
        fast_overlay_signals = pd.read_parquet(fast_path)
        mfast_path           = SIGNAL_DIR / "multi_fast_signals.parquet"
        multi_fast_signals   = (pd.read_parquet(mfast_path)
                                if mfast_path.exists() else fast_overlay_signals)
    else:
        print("  NOTE: fast_overlay_signals.parquet not found — run signal_generation.py first\n")

    pair_path  = SIGNAL_DIR / "pair_signals.parquet"
    has_pair   = pair_path.exists()
    if has_pair:
        pair_signals      = pd.read_parquet(pair_path)
        multi_pair_path   = SIGNAL_DIR / "multi_pair_signals.parquet"
        multi_pair_signals = (pd.read_parquet(multi_pair_path)
                              if multi_pair_path.exists() else pair_signals)
    else:
        print("  NOTE: pair_signals.parquet not found — run signal_generation.py first\n")

    earn_multi_path  = SIGNAL_DIR / "earn_multi_signals.parquet"
    has_earn         = earn_multi_path.exists()
    if has_earn:
        earn_multi_signals      = pd.read_parquet(earn_multi_path)
        earn_mfast_path         = SIGNAL_DIR / "earn_multi_fast_signals.parquet"
        earn_multi_fast_signals = (pd.read_parquet(earn_mfast_path)
                                   if earn_mfast_path.exists() else earn_multi_signals)
    else:
        print("  NOTE: earn_multi_signals.parquet not found — run signal_generation.py first\n")

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
    if has_fast:
        signals_fast_overlay = fast_overlay_signals.reindex(returns.index).fillna(0)
        signals_multi_fast   = multi_fast_signals.reindex(returns.index).fillna(0)
    if has_pair:
        signals_pair       = pair_signals.reindex(returns.index).fillna(0)
        signals_multi_pair = multi_pair_signals.reindex(returns.index).fillna(0)
    if has_earn:
        signals_earn_multi      = earn_multi_signals.reindex(returns.index).fillna(0)
        signals_earn_multi_fast = earn_multi_fast_signals.reindex(returns.index).fillna(0)

    print("Correlation matrix of returns (should be lower with diversified universe):")
    print(returns.corr().round(2))
    print()

    # ── Compute all position size DataFrames ──────────────────────────────────
    sizes_eq        = equal_weight_sizes(signals_regime, CAPITAL)
    sizes_atr       = atr_sizes(signals_regime, features, CAPITAL)
    sizes_kelly     = kelly_sizes(signals_regime, returns, CAPITAL)
    sizes_atr_pca   = apply_pca_scaling(sizes_atr, returns)
    sizes_final     = apply_macro_multiplier(sizes_atr_pca)
    sizes_vol       = vol_target_sizes(sizes_final, returns)
    sizes_dd        = apply_drawdown_control(sizes_eq, returns)

    sizes_comp_atr   = atr_sizes(signals_composite, features, CAPITAL)
    sizes_comp_pca   = apply_pca_scaling(sizes_comp_atr, returns)
    sizes_comp_macro = apply_macro_multiplier(sizes_comp_pca)
    sizes_comp_vol   = vol_target_sizes(sizes_comp_macro, returns)

    print("  Computing risk-parity sizes (Ledoit-Wolf, monthly rebalance)...")
    sizes_rp = risk_parity_sizes(signals_regime, features, returns, CAPITAL)

    # rp_macro: risk-parity base sizing (already includes one macro pass inside
    # risk_parity_sizes) with a second macro multiplier applied on top.
    # This gives a more aggressive regime overlay — deeper size reduction in fear
    # (0.5 × 0.5 = 0.25×) and a modest boost in calm (1.25 × 1.25 ≈ 1.56×).
    print("  Computing rp_macro sizes (risk parity + double macro overlay)...")
    sizes_rp_macro = apply_macro_multiplier(sizes_rp)

    ret_eq       = portfolio_returns(sizes_eq,       returns)
    ret_atr      = portfolio_returns(sizes_atr,      returns)
    ret_kelly    = portfolio_returns(sizes_kelly,     returns)
    ret_atr_pca  = portfolio_returns(sizes_atr_pca,  returns)
    ret_final    = portfolio_returns(sizes_final,     returns)
    ret_vol      = portfolio_returns(sizes_vol,       returns)
    ret_dd       = portfolio_returns(sizes_dd,        returns)
    ret_comp_vol = portfolio_returns(sizes_comp_vol,  returns)
    ret_rp       = portfolio_returns(sizes_rp,        returns)
    ret_rp_macro = portfolio_returns(sizes_rp_macro,  returns)

    if has_multi:
        sizes_multi_eq    = equal_weight_sizes(signals_multi, CAPITAL)
        sizes_multi_atr   = atr_sizes(signals_multi, features, CAPITAL)
        sizes_multi_macro = apply_macro_multiplier(sizes_multi_atr)
        ret_multi_eq      = portfolio_returns(sizes_multi_eq,    returns)
        ret_multi_atr     = portfolio_returns(sizes_multi_atr,   returns)
        ret_multi_macro   = portfolio_returns(sizes_multi_macro, returns)

    if has_fast:
        sizes_fast_atr       = atr_sizes(signals_fast_overlay, features, CAPITAL)
        sizes_multi_fast_atr = atr_sizes(signals_multi_fast,   features, CAPITAL)
        ret_fast_atr         = portfolio_returns(sizes_fast_atr,       returns)
        ret_multi_fast_atr   = portfolio_returns(sizes_multi_fast_atr, returns)

    # Cross-sectional momentum tilt — computed over multi and multi_fast signals.
    # The per-date ranking loop is O(T × N) and runs ~2s for 2500 days × 37 tickers.
    print("  Computing momentum tilt sizes (cross-sectional 63-day rank)...")
    if has_multi:
        sizes_multi_mom       = momentum_tilt_sizes(signals_multi, features, CAPITAL)
        ret_multi_mom         = portfolio_returns(sizes_multi_mom, returns)
    if has_fast:
        sizes_multi_fast_mom  = momentum_tilt_sizes(signals_multi_fast, features, CAPITAL)
        ret_multi_fast_mom    = portfolio_returns(sizes_multi_fast_mom, returns)

    # ── Vol-scaled sizing (21-day realized-vol targeting) ─────────────────────
    # simple_vol_scale: ONE input (own trailing vol), ONE target (10%), ZERO fitted
    # thresholds. Scales down when portfolio vol spikes, scales up when calm.
    # More robust than macro multiplier (no cross-asset relationships to overfit).
    if has_fast:
        sizes_multi_fast_atr_vol = simple_vol_scale(sizes_multi_fast_atr, returns)
        ret_multi_fast_atr_vol   = portfolio_returns(sizes_multi_fast_atr_vol, returns)
        sizes_multi_fast_mom_vol = simple_vol_scale(sizes_multi_fast_mom, returns)
        ret_multi_fast_mom_vol   = portfolio_returns(sizes_multi_fast_mom_vol, returns)

    if has_pair:
        print("  Computing beta-hedged pair sizes (ATR long + rolling-beta short hedge)...")
        sizes_pair       = beta_hedged_sizes(signals_pair,       features, returns, CAPITAL)
        sizes_multi_pair = beta_hedged_sizes(signals_multi_pair, features, returns, CAPITAL)
        ret_pair         = portfolio_returns(sizes_pair,       returns)
        ret_multi_pair   = portfolio_returns(sizes_multi_pair, returns)

    ret_bnh      = returns.mean(axis=1)

    if has_ensemble:
        signals_ensemble = ensemble_signals.reindex(returns.index).fillna(0)
        sizes_ens        = ensemble_sizes(signals_ensemble, features, returns, CAPITAL)
        ret_ens          = portfolio_returns(sizes_ens, returns)

    # ── Unified method registry ────────────────────────────────────────────────
    # Each entry: (label, full-period return series, signal matrix for WF, sizing_fn for WF)
    # signal matrix must match what the sizing_fn expects — regime methods use
    # signals_regime, composite uses signals_composite, ensemble uses signals_ensemble.
    all_methods = [
        (
            "equal weight",
            ret_eq, signals_regime,
            lambda sig, ret: equal_weight_sizes(sig, CAPITAL),
        ),
        (
            "ATR sized",
            ret_atr, signals_regime,
            lambda sig, ret: atr_sizes(sig, features, CAPITAL),
        ),
        (
            "half-Kelly",
            ret_kelly, signals_regime,
            lambda sig, ret: kelly_sizes(sig, ret, CAPITAL),
        ),
        (
            "ATR + PCA",
            ret_atr_pca, signals_regime,
            lambda sig, ret: apply_pca_scaling(atr_sizes(sig, features, CAPITAL), ret),
        ),
        (
            "ATR + PCA + macro",
            ret_final, signals_regime,
            lambda sig, ret: apply_macro_multiplier(
                apply_pca_scaling(atr_sizes(sig, features, CAPITAL), ret)),
        ),
        (
            "equal wt + DD control",
            ret_dd, signals_regime,
            lambda sig, ret: apply_drawdown_control(equal_weight_sizes(sig, CAPITAL), ret),
        ),
        (
            "regime + vol target",
            ret_vol, signals_regime,
            lambda sig, ret: vol_target_sizes(
                apply_macro_multiplier(
                    apply_pca_scaling(atr_sizes(sig, features, CAPITAL), ret)), ret),
        ),
        (
            "composite + vol target",
            ret_comp_vol, signals_composite,
            lambda sig, ret: vol_target_sizes(
                apply_macro_multiplier(
                    apply_pca_scaling(atr_sizes(sig, features, CAPITAL), ret)), ret),
        ),
        (
            "risk parity",
            ret_rp, signals_regime,
            lambda sig, ret: risk_parity_sizes(sig, features, ret, CAPITAL),
        ),
        (
            "rp_macro",
            ret_rp_macro, signals_regime,
            lambda sig, ret: apply_macro_multiplier(
                risk_parity_sizes(sig, features, ret, CAPITAL)),
        ),
    ]
    if has_ensemble:
        all_methods.append((
            "ensemble + ATR + PCA + macro",
            ret_ens, signals_ensemble,
            lambda sig, ret: ensemble_sizes(sig, features, ret, CAPITAL),
        ))
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
            (
                # ATR + macro multiplier only. Macro (VIX/yield-curve) uses published
                # thresholds, not fitted parameters — the one overlay worth testing.
                "multi_atr_macro",
                ret_multi_macro, signals_multi,
                lambda sig, ret: apply_macro_multiplier(atr_sizes(sig, features, CAPITAL)),
            ),
        ])
    if has_fast:
        all_methods.extend([
            (
                # Fast MA20/50 overlay blended with slow MA50/200 for 5 liquid ETFs.
                # Continuous signal in [0,1] or [-1,1] — atr_sizes handles this.
                "fast_atr",
                ret_fast_atr, signals_fast_overlay,
                lambda sig, ret: atr_sizes(sig, features, CAPITAL),
            ),
            (
                # Same but starting from multi-signal (breakout + bounce) entries.
                "multi_fast_atr",
                ret_multi_fast_atr, signals_multi_fast,
                lambda sig, ret: atr_sizes(sig, features, CAPITAL),
            ),
        ])
    if has_multi:
        all_methods.append((
            # Cross-sectional momentum tilt on multi-signal entries.
            # Tilts ATR sizes ±30% toward 63-day cross-sectional winners.
            "multi_mom_tilt",
            ret_multi_mom, signals_multi,
            lambda sig, ret: momentum_tilt_sizes(sig, features, CAPITAL),
        ))
    if has_fast:
        all_methods.extend([
            (
                # Same tilt applied to the multi_fast signal (MA20/50 overlay + breakout/bounce).
                "multi_fast_mom_tilt",
                ret_multi_fast_mom, signals_multi_fast,
                lambda sig, ret: momentum_tilt_sizes(sig, features, CAPITAL),
            ),
            (
                # Vol-scaled multi_fast_atr: 21-day realized-vol targeting at 10%.
                # scale = (0.10 / realized_vol).clip(0.5, 1.5).shift(1)
                # ONE parameter (target_vol), structural bounds, zero cross-asset estimation.
                "multi_fast_atr_vol",
                ret_multi_fast_atr_vol, signals_multi_fast,
                lambda sig, ret: simple_vol_scale(
                    atr_sizes(sig, features, CAPITAL), ret),
            ),
            (
                # Vol-scaled momentum-tilt: combines cross-sectional rank tilt with
                # portfolio-level vol targeting. Should improve Calmar via lower DD.
                "multi_fast_mom_vol",
                ret_multi_fast_mom_vol, signals_multi_fast,
                lambda sig, ret: simple_vol_scale(
                    momentum_tilt_sizes(sig, features, CAPITAL), ret),
            ),
        ])
    if has_pair:
        all_methods.extend([
            (
                # Beta-hedged pair trade: long stock + short sector ETF.
                # Uses regime signal filtered by spread outperformance.
                # Should reduce SPY correlation and improve bull_calm alpha.
                "pair_atr",
                ret_pair, signals_pair,
                lambda sig, ret: beta_hedged_sizes(sig, features, ret, CAPITAL),
            ),
            (
                # Same but using multi-signal entries (breakout + bounce) filtered
                # by spread outperformance — best-case combination.
                "multi_pair_atr",
                ret_multi_pair, signals_multi_pair,
                lambda sig, ret: beta_hedged_sizes(sig, features, ret, CAPITAL),
            ),
        ])

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
              f"Mean OOS {oos_sharpes[label]:.3f}  |  Mean OOS Active {oos_act_sharpes[label]:.3f}  |  "
              f"Std OOS {wf['sharpe'].std():.3f}  |  "
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

    # Production method: smallest IS-OOS gap with OOS Sharpe > 0.8.
    # Rationale: a method with OOS 0.95 / gap 0.20 is more trustworthy than
    # OOS 1.02 / gap 0.40 — the lower-gap method will likely degrade less
    # when the market regime shifts.
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
    print("    < 0.3  -> robust, generalises well")
    print("    0.3-0.6 -> moderate overfitting, acceptable")
    print("    > 0.6  -> overfit; simplify or add regularisation")

    # ── Persist results ────────────────────────────────────────────────────────
    comparison_curves = {
        "equal_weight"  : equity_curve(ret_eq,       CAPITAL),
        "atr_sized"     : equity_curve(ret_atr,       CAPITAL),
        "atr_pca_macro" : equity_curve(ret_final,     CAPITAL),
        "eq_dd_control" : equity_curve(ret_dd,        CAPITAL),
        "vol_target"    : equity_curve(ret_vol,       CAPITAL),
        "risk_parity"   : equity_curve(ret_rp,        CAPITAL),
        "rp_macro"      : equity_curve(ret_rp_macro,  CAPITAL),
        "buy_hold"      : equity_curve(ret_bnh,       CAPITAL),
    }
    if has_ensemble:
        comparison_curves["ensemble_atr_pca_macro"] = equity_curve(ret_ens, CAPITAL)
    if has_multi:
        comparison_curves["multi_equal_weight"] = equity_curve(ret_multi_eq,    CAPITAL)
        comparison_curves["multi_atr_pure"]     = equity_curve(ret_multi_atr,   CAPITAL)
        comparison_curves["multi_atr_macro"]    = equity_curve(ret_multi_macro, CAPITAL)
    if has_fast:
        comparison_curves["fast_atr"]       = equity_curve(ret_fast_atr,       CAPITAL)
        comparison_curves["multi_fast_atr"] = equity_curve(ret_multi_fast_atr, CAPITAL)
    if has_pair:
        comparison_curves["pair_atr"]       = equity_curve(ret_pair,       CAPITAL)
        comparison_curves["multi_pair_atr"] = equity_curve(ret_multi_pair, CAPITAL)
    if has_multi:
        comparison_curves["multi_mom_tilt"]      = equity_curve(ret_multi_mom,      CAPITAL)
    if has_fast:
        comparison_curves["multi_fast_mom_tilt"] = equity_curve(ret_multi_fast_mom, CAPITAL)
        comparison_curves["multi_fast_atr_vol"]  = equity_curve(ret_multi_fast_atr_vol, CAPITAL)
        comparison_curves["multi_fast_mom_vol"]  = equity_curve(ret_multi_fast_mom_vol, CAPITAL)

    pd.DataFrame(comparison_curves).to_parquet(RESULTS_DIR / "portfolio_comparison.parquet")

    # Dashboard compatibility: keep the two named walk-forward parquets it expects
    wf_store["equal weight"].to_parquet(
        RESULTS_DIR / "walk_forward_regime.parquet", index=False)
    wf_store["ATR + PCA + macro"].to_parquet(
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


if __name__ == "__main__":
    main()
