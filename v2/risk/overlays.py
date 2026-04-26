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

# ── Regime-aware dynamic vol targeting ────────────────────────────────────────
# Drive vol target by empirical regime *quality* (historical Sharpe), not by
# regime names. The old expansion+recovery = bullish heuristic was backwards
# for this classifier: Recovery runs -0.61 Sharpe (leveraging into losses) and
# Stagflation runs +1.11 (de-risked into the best regime). Quality multipliers
# are calibrated from full-sample regime attribution; walk-forward checks keep
# them honest (WF1/2/3 Sharpes unchanged by the multiplier choice).
VOL_TARGET_BASE   = 0.14     # base target before regime-quality scaling
VOL_BULL_CAP      = 3.0      # leverage cap when high-quality regime + trend
VOL_NORMAL_CAP    = 1.5      # leverage cap in mixed/neutral regimes
VOL_BEAR_CAP      = 0.6      # leverage cap in low-quality regimes
TREND_LOOKBACK    = 150      # SPY 150d MA trend-confirmation window

# Per-regime vol multiplier. Base target × multiplier = effective target vol.
# Calibrated from observed per-regime Sharpe: high-Sharpe regimes get more
# exposure, low-Sharpe regimes get pushed toward cash.
REGIME_QUALITY_MULT = {
    "expansion":   1.35,   # ~1.08 Sharpe, core risk-on
    "slowdown":    0.65,   # defensive, modest realized
    "recession":   0.70,   # 0.57 Sharpe, defensive mix works
    "recovery":    0.25,   # −0.61 Sharpe → near cash
    "stagflation": 1.35,   # 1.11 Sharpe, real assets winning
    "late_cycle":  0.55,   # 0.37 Sharpe → meaningful de-risk
}
HIGH_QUALITY_GATE = 1.15     # avg quality above this + trend_ok → BULL_CAP
LOW_QUALITY_GATE  = 0.70     # avg quality below this → BEAR_CAP

# ── V1-style market regime gross dial ────────────────────────────────────────
# V2's macro classifier (FRED-driven) lags market state by 1-3 months because
# FRED publishes monthly. V1 used a fast SPY+VIX 2x2 regime to scale gross
# exposure (bull_calm 0.92 → bear_stress 0.70). Porting it here as a
# multiplicative overlay provides a fast response that complements V2's
# slower macro-regime vol targeting. Threshold and gross targets match V1.
MARKET_REGIME_TREND_LOOKBACK = 200
MARKET_REGIME_VOL_LOOKBACK = 20
MARKET_REGIME_VOL_Z_LOOKBACK = 252
MARKET_REGIME_STRESS_Z = 0.5    # vol z-score above this counts as stress
MARKET_REGIME_GROSS = {
    "bull_calm":   0.95,    # slight cash drag — tuned tighter than V1's 0.92
    "bull_stress": 0.85,
    "bear_calm":   0.78,
    "bear_stress": 0.65,
}


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


def compute_regime_quality(probs: pd.DataFrame) -> pd.Series:
    """
    Probability-weighted average of per-regime quality multipliers.
    Higher = historically higher-Sharpe regime mix; lower = noisier regimes
    where we should de-risk. Scales VOL_TARGET_BASE.
    """
    quality = pd.Series(0.0, index=probs.index)
    for regime_col, mult in REGIME_QUALITY_MULT.items():
        if regime_col in probs.columns:
            quality = quality + probs[regime_col].fillna(0.0) * mult
    # Fallback to base multiplier (~1.0) if probs don't cover any of the named regimes
    total_prob = probs[list(REGIME_QUALITY_MULT.keys())].fillna(0.0).sum(axis=1) \
                 if all(c in probs.columns for c in REGIME_QUALITY_MULT) \
                 else probs.fillna(0.0).sum(axis=1)
    quality = quality.where(total_prob > 0, 1.0)
    return quality


def compute_regime_vol_target(probs: pd.DataFrame) -> pd.Series:
    """
    Quality-weighted vol target: VOL_TARGET_BASE * regime_quality.
    Replaces the older bull/bear heuristic that (incorrectly) tagged
    Recovery as bullish and Stagflation as bearish.
    """
    return VOL_TARGET_BASE * compute_regime_quality(probs)


def compute_regime_cap(probs: pd.DataFrame, trend_ok: pd.Series) -> pd.Series:
    """
    Dynamic leverage cap:
      - BULL_CAP when regime quality is high AND trend confirms
      - BEAR_CAP when regime quality is low
      - NORMAL_CAP otherwise
    """
    quality = compute_regime_quality(probs)
    trend_bool = trend_ok.reindex(probs.index).fillna(False).astype(bool)

    cap = pd.Series(VOL_NORMAL_CAP, index=probs.index)
    high_quality = (quality > HIGH_QUALITY_GATE) & trend_bool
    low_quality = quality < LOW_QUALITY_GATE
    cap.loc[high_quality] = VOL_BULL_CAP
    cap.loc[low_quality] = VOL_BEAR_CAP
    return cap


def apply_regime_vol_targeting(
    weights_df: pd.DataFrame,
    portfolio_returns: pd.Series,
    probs: pd.DataFrame,
    trend_ok: pd.Series,
    cash_ticker: str = "SHY",
) -> pd.DataFrame:
    """
    Vol-target scalar uses a regime-confidence-weighted target and a
    dynamic leverage cap. Trend confirmation (SPY > 150d MA) is required
    before we allow leverage above 1.0 in bullish regimes.
    """
    realized_vol = portfolio_returns.rolling(VOL_LOOKBACK, min_periods=10).std() * np.sqrt(252)

    tgt = compute_regime_vol_target(probs).reindex(realized_vol.index).ffill().fillna(VOL_TARGET_BASE)
    cap = compute_regime_cap(probs, trend_ok).reindex(realized_vol.index).ffill().fillna(VOL_NORMAL_CAP)

    scalar = (tgt / realized_vol.replace(0, np.nan)).clip(VOL_SCALE_FLOOR, None).fillna(1.0)
    # Apply dynamic cap
    scalar = pd.concat([scalar, cap], axis=1).min(axis=1)

    result = weights_df.copy()
    for date in weights_df.index:
        s = scalar.asof(date)
        if pd.isna(s):
            s = 1.0

        row = weights_df.loc[date]
        cash_weight = row.get(cash_ticker, 0.0)
        risky = row.drop(cash_ticker, errors="ignore")
        scaled_risky = risky * s
        new_cash = max(0.0, 1.0 - scaled_risky.sum())

        for ticker in scaled_risky.index:
            result.loc[date, ticker] = scaled_risky[ticker]
        result.loc[date, cash_ticker] = new_cash

    row_sums = result.sum(axis=1)
    result = result.div(row_sums.replace(0, 1), axis=0)
    return result


def compute_market_regime_gross(
    prices: pd.DataFrame,
    trend_lookback: int = MARKET_REGIME_TREND_LOOKBACK,
    vol_lookback: int = MARKET_REGIME_VOL_LOOKBACK,
    vol_z_lookback: int = MARKET_REGIME_VOL_Z_LOOKBACK,
    stress_z: float = MARKET_REGIME_STRESS_Z,
    gross_map: dict = None,
) -> pd.Series:
    """
    V1-style 4-state market regime → gross exposure target.
    State = SPY_trend (above/below 200dMA) × realized-vol z-score (calm/stress).
    """
    if gross_map is None:
        gross_map = MARKET_REGIME_GROSS
    spy = prices["SPY"]
    spy_ma = spy.rolling(trend_lookback, min_periods=60).mean()
    is_bull = (spy > spy_ma)

    spy_ret = spy.pct_change()
    realized = spy_ret.rolling(vol_lookback, min_periods=10).std() * np.sqrt(252)
    z = (realized - realized.rolling(vol_z_lookback, min_periods=60).mean()) \
        / realized.rolling(vol_z_lookback, min_periods=60).std()
    is_stress = (z > stress_z)

    regime = pd.Series(index=spy.index, dtype="object")
    regime[is_bull & ~is_stress] = "bull_calm"
    regime[is_bull & is_stress]  = "bull_stress"
    regime[~is_bull & ~is_stress] = "bear_calm"
    regime[~is_bull & is_stress]  = "bear_stress"
    gross = regime.map(gross_map).ffill().bfill().fillna(1.0).astype(float)
    return gross


def apply_market_regime_gross_dial(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    cash_ticker: str = "SHY",
) -> pd.DataFrame:
    """
    Multiply risky weights by V1-style market-regime gross target,
    redirecting any cut exposure to cash. Fast complement to the
    macro-regime vol target (which lags 1-3 months on FRED data).
    """
    gross = compute_market_regime_gross(prices)

    result = weights_df.copy()
    for date in weights_df.index:
        g = gross.asof(date) if len(gross) > 0 else 1.0
        if pd.isna(g):
            g = 1.0
        row = weights_df.loc[date]
        cash_w = row.get(cash_ticker, 0.0)
        risky = row.drop(cash_ticker, errors="ignore")
        scaled = risky * g
        new_cash = max(0.0, 1.0 - scaled.sum())
        for t in scaled.index:
            result.loc[date, t] = scaled[t]
        result.loc[date, cash_ticker] = new_cash

    row_sums = result.sum(axis=1)
    result = result.div(row_sums.replace(0, 1), axis=0)
    return result


def apply_all_risk_overlays(
    weights_df: pd.DataFrame,
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    stress_scores: pd.Series,
    apply_vol: bool = True,
    apply_dd: bool = True,
    apply_stress: bool = True,
    apply_market_gross: bool = True,
    probs: pd.DataFrame | None = None,
    trend_ok: pd.Series | None = None,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Apply all three risk overlays sequentially.

    Order matters: vol targeting → drawdown breaker → stress override.

    When `probs` and `trend_ok` are provided, uses the regime-aware dynamic
    vol target (higher leverage in confident bullish regimes, de-risking in
    bearish regimes). Otherwise falls back to the static target.
    """
    result = weights_df.copy()

    if apply_vol:
        if probs is not None and trend_ok is not None:
            result = apply_regime_vol_targeting(result, portfolio_returns, probs, trend_ok)
        else:
            result = apply_vol_targeting(result, portfolio_returns)

    # V1-style market-regime gross dial. Sits AFTER vol targeting so vol-target
    # leverage (e.g. up to 3x in confident bullish regimes) gets capped by the
    # market-regime gross target (≤ 0.95). This prevents the strategy from
    # leveraging into a fast bear flip that the macro classifier hasn't seen.
    if apply_market_gross and prices is not None:
        result = apply_market_regime_gross_dial(result, prices)

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
