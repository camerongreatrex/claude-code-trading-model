"""
v2/risk_model.py
----------------
Risk management for the beta-neutral model.

No trailing stops. The system rebalances weekly — stops on individual names
in a 45-stock portfolio just add whipsaw and transaction costs.

Portfolio-level risk controls
─────────────────────────────
  1. Volatility targeting: target 10% annualized portfolio vol
     scale_factor = target_vol / realized_20d_vol, clamp [0.5, 1.5]
  2. Drawdown circuit breaker: if DD > -10% from peak, cut gross by 50%
     for 2 weeks, then re-enter at full size
  3. Single-stock cap: no position > 3% of portfolio
  4. Correlation monitoring: flag if avg pairwise corr in long book > 0.6
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_VOL = 0.10               # 10% annualized
VOL_LOOKBACK = 20               # 20-day realized vol
VOL_SCALE_MIN = 0.50
VOL_SCALE_MAX = 1.50
DRAWDOWN_THRESHOLD = -0.10      # -10% drawdown triggers circuit breaker
DRAWDOWN_RECOVERY_WEEKS = 2     # reduce exposure for 2 weeks
DRAWDOWN_REDUCTION = 0.50       # cut gross by 50% during breaker
MAX_POSITION_PCT = 0.03         # 3% max per position
CORR_WARNING_THRESHOLD = 0.60   # flag if avg pairwise corr > 0.6


def compute_vol_scale(
    portfolio_returns: pd.Series,
    date: pd.Timestamp,
) -> float:
    """
    Compute gross exposure scale factor for volatility targeting.

    scale = target_vol / realized_20d_vol, clamped to [0.5, 1.5]
    """
    if date not in portfolio_returns.index:
        return 1.0

    loc = portfolio_returns.index.get_loc(date)
    if loc < VOL_LOOKBACK:
        return 1.0

    recent = portfolio_returns.iloc[max(0, loc - VOL_LOOKBACK):loc + 1]
    realized_vol = recent.std() * np.sqrt(252)

    if realized_vol < 0.001:
        return VOL_SCALE_MAX

    scale = TARGET_VOL / realized_vol
    return np.clip(scale, VOL_SCALE_MIN, VOL_SCALE_MAX)


def compute_vol_scale_series(portfolio_returns: pd.Series) -> pd.Series:
    """Compute vol scaling factor for every date in the series."""
    rolling_vol = portfolio_returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)
    scale = TARGET_VOL / rolling_vol
    scale = scale.clip(VOL_SCALE_MIN, VOL_SCALE_MAX)
    scale = scale.fillna(1.0)
    return scale


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Compute drawdown series from equity curve."""
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    return dd


def check_circuit_breaker(
    equity_curve: pd.Series,
    date: pd.Timestamp,
) -> tuple[bool, float]:
    """
    Check if drawdown circuit breaker should be active.

    Args:
        equity_curve: portfolio equity curve up to current date
        date: current date

    Returns:
        (is_active, scale_factor)
        is_active: True if circuit breaker is triggered
        scale_factor: 0.5 if active, 1.0 if not
    """
    if date not in equity_curve.index:
        return False, 1.0

    dd = compute_drawdown(equity_curve)
    current_dd = dd.loc[date]

    if current_dd < DRAWDOWN_THRESHOLD:
        return True, DRAWDOWN_REDUCTION

    # Check if we're still in recovery period (breaker was active recently)
    loc = equity_curve.index.get_loc(date)
    recovery_days = DRAWDOWN_RECOVERY_WEEKS * 5  # trading days
    if loc >= recovery_days:
        recent_dd = dd.iloc[loc - recovery_days:loc + 1]
        if (recent_dd < DRAWDOWN_THRESHOLD).any():
            return True, DRAWDOWN_REDUCTION

    return False, 1.0


def compute_circuit_breaker_series(equity_curve: pd.Series) -> pd.Series:
    """Compute circuit breaker scale factor for every date."""
    dd = compute_drawdown(equity_curve)
    recovery_days = DRAWDOWN_RECOVERY_WEEKS * 5

    breaker_active = pd.Series(False, index=equity_curve.index)
    for i, date in enumerate(equity_curve.index):
        if dd.iloc[i] < DRAWDOWN_THRESHOLD:
            # Activate for current date + recovery_days
            end_idx = min(i + recovery_days, len(equity_curve) - 1)
            breaker_active.iloc[i:end_idx + 1] = True

    scale = pd.Series(1.0, index=equity_curve.index)
    scale[breaker_active] = DRAWDOWN_REDUCTION
    return scale


def check_correlation_warning(
    returns: pd.DataFrame,
    long_tickers: list,
    date: pd.Timestamp,
    window: int = 60,
) -> tuple[bool, float]:
    """
    Check if average pairwise correlation in long book is dangerously high.

    Returns (is_warning, avg_corr)
    """
    if len(long_tickers) < 3:
        return False, 0.0

    available = [t for t in long_tickers if t in returns.columns]
    if len(available) < 3:
        return False, 0.0

    loc = returns.index.get_loc(date) if date in returns.index else -1
    if loc < window:
        return False, 0.0

    recent = returns[available].iloc[loc - window:loc + 1]
    corr_matrix = recent.corr()

    # Average off-diagonal correlation
    n = len(available)
    mask = ~np.eye(n, dtype=bool)
    avg_corr = corr_matrix.values[mask].mean()

    return avg_corr > CORR_WARNING_THRESHOLD, avg_corr


def apply_risk_controls(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
) -> pd.Series:
    """
    Compute combined risk scale factor (vol targeting × circuit breaker).

    Returns a Series of scale factors indexed by date.
    """
    vol_scale = compute_vol_scale_series(portfolio_returns)
    breaker_scale = compute_circuit_breaker_series(equity_curve)
    combined = vol_scale * breaker_scale
    return combined.clip(VOL_SCALE_MIN * DRAWDOWN_REDUCTION, VOL_SCALE_MAX)
