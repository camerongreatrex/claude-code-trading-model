"""
v2/risk_model.py
----------------
Risk management overlays for the macro regime rotation portfolio.

Three layers (applied sequentially):
  1. Volatility targeting: scale exposure to target 10% ann. vol
  2. Drawdown circuit breaker: cut exposure after deep drawdown
  3. Stress override: force defensive allocation on extreme stress

All three reduce gross exposure by shifting to SHY (cash proxy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/v2/results")
REGIME_DIR = Path("data/v2/regime_features")


# ── Configuration (from config.py V2_CONFIG) ─────────────────────────────────
TARGET_VOL = 0.12
VOL_SCALE_CAP = 1.5
VOL_SCALE_FLOOR = 0.3
VOL_LOOKBACK = 20           # 20 trading days for realized vol

DD_TRIGGER = -0.12           # -12% drawdown triggers circuit breaker
DD_EXPOSURE_CUT = 0.40       # cut exposure by 40%
DD_RESTORE = -0.08           # restore when DD recovers above -8%
DD_COOLDOWN_DAYS = 21        # minimum 1 month before restore

STRESS_THRESHOLD = 2.0       # std devs for stress override
STRESS_DEFENSIVE = 0.50      # force 50% into defensive basket
STRESS_TICKERS = ["SHY", "TLT", "GLD"]  # defensive basket
STRESS_WEIGHTS = [0.40, 0.35, 0.25]     # defensive basket weights


def compute_vol_scalar(
    portfolio_returns: pd.Series,
    target_vol: float = TARGET_VOL,
    lookback: int = VOL_LOOKBACK,
    cap: float = VOL_SCALE_CAP,
    floor: float = VOL_SCALE_FLOOR,
) -> pd.Series:
    """
    Compute daily volatility scaling factor.

    scalar = target_vol / realized_vol_20d
    Capped at 1.5x, floored at 0.3x.
    """
    realized_vol = portfolio_returns.rolling(lookback, min_periods=10).std() * np.sqrt(252)
    scalar = target_vol / realized_vol.replace(0, np.nan)
    scalar = scalar.clip(floor, cap).fillna(1.0)
    return scalar


def apply_vol_targeting(
    weights_df: pd.DataFrame,
    portfolio_returns: pd.Series,
    target_vol: float = TARGET_VOL,
    cash_ticker: str = "SHY",
) -> pd.DataFrame:
    """
    Scale all risky weights by vol scalar, put remainder in cash.

    If scalar < 1.0: reduce risky exposure, add to cash
    If scalar > 1.0: increase risky exposure (up to 1.5x), reduce cash
    """
    scalar = compute_vol_scalar(portfolio_returns, target_vol)

    result = weights_df.copy()

    for date in weights_df.index:
        if date not in scalar.index:
            continue

        s = scalar.asof(date)
        if np.isnan(s):
            s = 1.0

        # Scale all non-cash positions
        row = weights_df.loc[date].copy()
        cash_weight = row.get(cash_ticker, 0.0)
        risky_weight = row.drop(cash_ticker, errors="ignore")

        scaled_risky = risky_weight * s
        new_cash = max(0, 1.0 - scaled_risky.sum())

        for ticker in scaled_risky.index:
            result.loc[date, ticker] = scaled_risky[ticker]
        result.loc[date, cash_ticker] = new_cash

    # Normalize
    row_sums = result.sum(axis=1)
    result = result.div(row_sums.replace(0, 1), axis=0)

    return result


def apply_drawdown_breaker(
    weights_df: pd.DataFrame,
    equity_curve: pd.Series,
    dd_trigger: float = DD_TRIGGER,
    exposure_cut: float = DD_EXPOSURE_CUT,
    dd_restore: float = DD_RESTORE,
    cooldown_days: int = DD_COOLDOWN_DAYS,
    cash_ticker: str = "SHY",
) -> pd.DataFrame:
    """
    Cut exposure when drawdown exceeds threshold.

    State machine:
      NORMAL → BREAKER_ON when DD < dd_trigger
      BREAKER_ON → NORMAL when DD > dd_restore AND cooldown elapsed
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max

    result = weights_df.copy()
    breaker_on = False
    breaker_since = None

    for date in weights_df.index:
        dd = drawdown.asof(date) if date in drawdown.index or len(drawdown) > 0 else 0
        if pd.isna(dd):
            dd = 0

        if not breaker_on and dd < dd_trigger:
            breaker_on = True
            breaker_since = date

        if breaker_on:
            days_in = (date - breaker_since).days if breaker_since else 0
            if dd > dd_restore and days_in >= cooldown_days:
                breaker_on = False
                breaker_since = None
            else:
                # Cut risky exposure
                row = result.loc[date].copy()
                cash_w = row.get(cash_ticker, 0.0)
                risky = row.drop(cash_ticker, errors="ignore")

                cut_amount = risky.sum() * exposure_cut
                scale = 1.0 - exposure_cut
                for ticker in risky.index:
                    result.loc[date, ticker] = risky[ticker] * scale
                result.loc[date, cash_ticker] = cash_w + cut_amount

    return result


def apply_stress_override(
    weights_df: pd.DataFrame,
    stress_scores: pd.Series,
    threshold: float = STRESS_THRESHOLD,
    defensive_pct: float = STRESS_DEFENSIVE,
    defensive_tickers: list[str] = None,
    defensive_weights: list[float] = None,
) -> pd.DataFrame:
    """
    Force defensive allocation when stress score exceeds threshold.

    If stress > 2 std dev: force 50% into SHY+TLT+GLD, scale
    remaining allocation into the other 50%.
    """
    if defensive_tickers is None:
        defensive_tickers = STRESS_TICKERS
    if defensive_weights is None:
        defensive_weights = STRESS_WEIGHTS

    result = weights_df.copy()

    for date in weights_df.index:
        stress = stress_scores.asof(date) if date in stress_scores.index or len(stress_scores) > 0 else 0
        if pd.isna(stress):
            stress = 0

        if stress > threshold:
            row = weights_df.loc[date].copy()
            remaining_pct = 1.0 - defensive_pct

            # Scale existing allocation into remaining_pct
            total_existing = row.sum()
            if total_existing > 0:
                row = row / total_existing * remaining_pct

            # Add defensive allocation
            for ticker, weight in zip(defensive_tickers, defensive_weights):
                if ticker in row.index:
                    row[ticker] = row.get(ticker, 0) + defensive_pct * weight

            # Normalize
            total = row.sum()
            if total > 0:
                row = row / total

            result.loc[date] = row

    return result


def apply_all_risk_overlays(
    weights_df: pd.DataFrame,
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    stress_scores: pd.Series,
    apply_vol: bool = True,
    apply_dd: bool = True,
    apply_stress: bool = True,
) -> pd.DataFrame:
    """
    Apply all three risk overlays sequentially.

    Order matters: vol targeting → drawdown breaker → stress override
    """
    result = weights_df.copy()

    if apply_vol:
        result = apply_vol_targeting(result, portfolio_returns)

    if apply_dd:
        result = apply_drawdown_breaker(result, equity_curve)

    if apply_stress:
        result = apply_stress_override(result, stress_scores)

    # Final normalize
    row_sums = result.sum(axis=1)
    result = result.div(row_sums.replace(0, 1), axis=0)

    return result


if __name__ == "__main__":
    print("="*60)
    print("  V2 Risk Model")
    print("="*60)
    print("\n  Risk model loaded. Applied during backtesting.")
    print(f"  Vol target:          {TARGET_VOL:.0%}")
    print(f"  Vol scale range:     [{VOL_SCALE_FLOOR:.1f}x, {VOL_SCALE_CAP:.1f}x]")
    print(f"  DD trigger:          {DD_TRIGGER:.0%}")
    print(f"  DD exposure cut:     {DD_EXPOSURE_CUT:.0%}")
    print(f"  DD restore:          {DD_RESTORE:.0%}")
    print(f"  Stress threshold:    {STRESS_THRESHOLD:.1f} std dev")
    print(f"  Stress defensive %:  {STRESS_DEFENSIVE:.0%}")
    print(f"  Stress basket:       {dict(zip(STRESS_TICKERS, STRESS_WEIGHTS))}")
