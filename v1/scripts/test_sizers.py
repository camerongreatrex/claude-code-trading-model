"""
test_sizers.py
--------------
Comprehensive sweep of position sizing methods under MAX_POSITION_PCT=0.10
($10k per-name cap on $100k capital).

Tested:
  - atr_pure                    : production baseline (1/ATR vol-normalisation)
  - half_kelly                  : Kelly W/PF × 0.5
  - quarter_kelly               : Kelly × 0.25
  - atr_kelly_70_30             : 70/30 atr-kelly blend
  - atr_kelly_50_50             : 50/50 blend
  - inverse_vol_60d             : size ∝ 1/vol_60d
  - inverse_vol_20d             : size ∝ 1/vol_20d (faster)
  - garch_inverse_vol           : size ∝ 1/GARCH_cond_vol
  - trend_strength              : size ∝ adx (above ADX_MIN), atr-baseline
  - signal_strength_kelly       : Kelly × signal_value (continuous)
  - vol_target_atr              : atr_pure + portfolio vol-target overlay (12% ann)
  - garch_kelly_combined        : Kelly × (1/GARCH_vol)
  - sharpe_weighted_atr         : atr × rolling Sharpe rank
  - momentum_quality_atr        : atr × (recent mom × 1/recent_vol)
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes, kelly_sizes, defensive_tilt_overlay,
    portfolio_returns, summarise, MAX_POSITION_PCT, CAPITAL,
    compute_strategy_returns,
    _load_garch_vol_matrix,
)


def _clip(sz: pd.DataFrame) -> pd.DataFrame:
    return sz.clip(-CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT)


def raw_atr_sizes(signals, features, capital, max_gross=None):
    """ATR sizes WITHOUT the 100%-gross cap.  Optional max_gross caps total
    gross at max_gross × capital (e.g. 2.0 = 200%)."""
    from v1.portfolio.portfolio import RISK_PER_TRADE, INDEX_ETF_TICKERS, INDEX_ETF_CAP
    dollar_risk = capital * RISK_PER_TRADE
    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for ticker in signals.columns:
        if ticker not in features:
            continue
        atr   = features[ticker]["atr_14"].reindex(signals.index).ffill()
        close = features[ticker]["Close"].reindex(signals.index).ffill()
        atr   = atr.replace(0, np.nan).ffill()
        dollar_pos = (dollar_risk / atr) * close
        sizes[ticker] = (signals[ticker] * dollar_pos).clip(
            -capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT
        )
    for etf in INDEX_ETF_TICKERS:
        if etf in sizes.columns:
            sizes[etf] = sizes[etf].clip(-capital * INDEX_ETF_CAP, capital * INDEX_ETF_CAP)
    if max_gross is not None:
        gross = sizes.abs().sum(axis=1).replace(0, np.nan)
        scale = (max_gross * capital / gross).clip(upper=1.0).fillna(1.0)
        sizes = sizes.multiply(scale, axis=0)
    return sizes


def kelly_sizes_scaled(signals, returns, capital, scale=0.5, lookback=252):
    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for ticker in signals.columns:
        if ticker not in returns.columns:
            continue
        strat_ret = compute_strategy_returns(signals[ticker], returns[ticker])
        rolling_wr = strat_ret.rolling(lookback).apply(
            lambda x: float((x[x != 0] > 0).mean()) if (x != 0).any() else 0.5
        )
        rolling_pf = strat_ret.rolling(lookback).apply(
            lambda x: float(x[x > 0].sum() / x[x < 0].abs().sum())
            if x[x < 0].abs().sum() > 0 else 1.0
        )
        kelly = (rolling_wr - (1 - rolling_wr) / rolling_pf.clip(0.01)).clip(0) * scale
        sizes[ticker] = (signals[ticker] * kelly * capital).clip(
            -capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT
        )
    return sizes.fillna(0)


def inverse_vol_sizes(signals, returns, capital, window=60, target_pct=0.04):
    """Size = signal × min(target / realized_vol, MAX_POSITION_PCT) × capital.

    target_pct = target risk allocation per active position (4% of capital).
    realized_vol = rolling ann stdev of daily log returns.
    """
    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for ticker in signals.columns:
        if ticker not in returns.columns:
            continue
        vol = returns[ticker].rolling(window).std() * np.sqrt(252)
        vol = vol.replace(0, np.nan).ffill().bfill().fillna(0.20)
        # 4% target × (0.20 / vol)  -> low-vol assets get up to cap, high-vol get less
        weight = (target_pct * (0.20 / vol)).clip(upper=MAX_POSITION_PCT)
        sizes[ticker] = signals[ticker] * weight * capital
    return _clip(sizes).fillna(0)


def garch_inverse_vol_sizes(signals, capital, target_pct=0.04):
    """Size ∝ 1/GARCH_cond_vol, clipped to MAX_POSITION_PCT."""
    garch = _load_garch_vol_matrix()
    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    if garch.empty:
        return sizes
    garch = garch.reindex(signals.index).ffill().bfill().fillna(0.20)
    for ticker in signals.columns:
        if ticker not in garch.columns:
            sizes[ticker] = signals[ticker] * 0.04 * capital  # fallback eq-weight 4%
            continue
        v = garch[ticker].replace(0, np.nan).ffill().bfill().fillna(0.20)
        weight = (target_pct * (0.20 / v)).clip(upper=MAX_POSITION_PCT)
        sizes[ticker] = signals[ticker] * weight * capital
    return _clip(sizes).fillna(0)


def trend_strength_sizes(signals, features, capital, adx_floor=20, adx_cap=50):
    """ATR-base × ADX/30 normalisation.  Stronger trend = larger position."""
    base = atr_sizes(signals, features, capital)
    out  = base.copy()
    for t in signals.columns:
        if t not in features:
            continue
        adx = features[t]["adx"].reindex(signals.index).ffill().fillna(20.0)
        # Map ADX in [adx_floor, adx_cap] -> [0.6, 1.4]
        scale = 0.6 + 0.8 * ((adx.clip(adx_floor, adx_cap) - adx_floor) / (adx_cap - adx_floor))
        out[t] = base[t] * scale
    return _clip(out)


def signal_strength_kelly(signals, returns, capital, scale=0.5, lookback=252):
    """Kelly fraction × |signal_value| — assumes signal in [-1, 1].
    Same as kelly_sizes_scaled because that already multiplies by signal."""
    return kelly_sizes_scaled(signals, returns, capital, scale, lookback)


def garch_kelly_combined(signals, returns, capital, scale=0.5, lookback=252,
                         target_vol=0.20):
    """Kelly fraction × (target_vol / GARCH_cond_vol) — vol-conditional Kelly."""
    base = kelly_sizes_scaled(signals, returns, capital, scale, lookback)
    garch = _load_garch_vol_matrix()
    if garch.empty:
        return base
    garch = garch.reindex(signals.index).ffill().bfill().fillna(0.20)
    out = base.copy()
    for t in base.columns:
        if t in garch.columns:
            v = garch[t].replace(0, np.nan).ffill().bfill().fillna(0.20)
            scaler = (target_vol / v).clip(0.3, 1.5)
            out[t] = base[t] * scaler
    return _clip(out)


def sharpe_weighted_atr(signals, features, returns, capital, lookback=126):
    """ATR sizing × rolling 6-month per-ticker Sharpe (clipped to [0.5, 1.5])."""
    base = atr_sizes(signals, features, capital)
    out  = base.copy()
    for t in signals.columns:
        if t not in returns.columns:
            continue
        strat_ret = signals[t].shift(1) * returns[t]
        rolling_mean = strat_ret.rolling(lookback).mean() * 252
        rolling_std  = strat_ret.rolling(lookback).std() * np.sqrt(252)
        sharpe = (rolling_mean / rolling_std.replace(0, np.nan)).fillna(0)
        # Map sharpe (typically -2 to +2) -> [0.5, 1.5] tilt
        scale = (1.0 + 0.25 * sharpe.clip(-2, 2)).clip(0.5, 1.5)
        out[t] = base[t] * scale
    return _clip(out)


def momentum_quality_atr(signals, features, returns, capital, mom_window=63):
    """ATR sizing × (mom_63 / vol_63) — Sharpe-of-recent-returns tilt."""
    base = atr_sizes(signals, features, capital)
    out  = base.copy()
    for t in signals.columns:
        if t not in returns.columns:
            continue
        mom = returns[t].rolling(mom_window).sum()
        vol = returns[t].rolling(mom_window).std()
        ratio = (mom / vol.replace(0, np.nan)).fillna(0)
        # Cross-sectional rank -> [0.6, 1.4]
        rank = ratio.rolling(252).apply(
            lambda x: (x.iloc[-1] > x).mean() if not x.empty else 0.5
        ).fillna(0.5)
        scale = 0.6 + 0.8 * rank
        out[t] = base[t] * scale
    return _clip(out)


def vol_target_atr(signals, features, returns, capital, target_vol=0.10, window=63):
    """ATR sizing + portfolio-level vol target (10% annualised)."""
    base = atr_sizes(signals, features, capital)
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    scaler = (target_vol / realized).clip(0.3, 2.0).shift(1).fillna(1.0)
    return _clip(base.multiply(scaler, axis=0))


def leveraged_atr_voltgt(signals, features, returns, capital,
                         leverage=2.0, vol_cap=0.13, window=63):
    """Leveraged ATR with a vol cap.  Most days: full leverage.  Stress days:
    scale DOWN to keep ann vol <= vol_cap.  No scale-up on calm days (asymmetric).
    """
    base = _clip(leverage * atr_sizes(signals, features, capital))
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(vol_cap)
    # Asymmetric: only scale down (never up).  This caps DD without adding leverage.
    scaler = (vol_cap / realized).clip(upper=1.0).shift(1).fillna(1.0)
    return _clip(base.multiply(scaler, axis=0))


def main() -> None:
    SIGNAL_DIR  = Path("data/v1/signals")
    FEATURE_DIR = Path("data/v1/features")
    MACRO_DIR   = Path("data/shared/macro")

    signals_raw = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    features: dict = {}
    returns = pd.DataFrame()
    for t in signals_raw.columns:
        p = FEATURE_DIR / f"{t}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            features[t] = f
            returns[t] = f["log_return"]

    returns = returns.dropna()
    signals = signals_raw.reindex(returns.index).fillna(0)

    macro_path = MACRO_DIR / "macro_features.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else pd.DataFrame()

    # ── Build sizers ───────────────────────────────────────────────────────
    print("Building sizing matrices...")
    sz_atr  = atr_sizes(signals, features, CAPITAL)
    sz_half = kelly_sizes_scaled(signals, returns, CAPITAL, 0.5)
    sz_full = kelly_sizes_scaled(signals, returns, CAPITAL, 1.0)
    sz_qtr  = kelly_sizes_scaled(signals, returns, CAPITAL, 0.25)

    variants: dict[str, pd.DataFrame] = {
        "atr_pure":              sz_atr,
        "quarter_kelly":         sz_qtr,
        "half_kelly":            sz_half,
        "full_kelly":            sz_full,
        "atr_70/kelly_30":       _clip(0.7 * sz_atr + 0.3 * sz_half),
        "atr_50/kelly_50":       _clip(0.5 * sz_atr + 0.5 * sz_half),
        "atr_30/kelly_70":       _clip(0.3 * sz_atr + 0.7 * sz_half),
        "inverse_vol_60d":       inverse_vol_sizes(signals, returns, CAPITAL, 60),
        "inverse_vol_20d":       inverse_vol_sizes(signals, returns, CAPITAL, 20),
        "garch_inverse_vol":     garch_inverse_vol_sizes(signals, CAPITAL),
        "trend_strength":        trend_strength_sizes(signals, features, CAPITAL),
        "garch_kelly_combined":  garch_kelly_combined(signals, returns, CAPITAL, 0.5),
        "sharpe_weighted_atr":   sharpe_weighted_atr(signals, features, returns, CAPITAL),
        "momentum_quality_atr":  momentum_quality_atr(signals, features, returns, CAPITAL),
        "vol_target_atr":        vol_target_atr(signals, features, returns, CAPITAL),
        "vol_target_atr_12":     vol_target_atr(signals, features, returns, CAPITAL, target_vol=0.12),
        "vol_target_atr_15":     vol_target_atr(signals, features, returns, CAPITAL, target_vol=0.15),
        # leverage atr_pure then re-clip
        "atr_pure_x1.3":         _clip(1.3 * sz_atr),
        "atr_pure_x1.5":         _clip(1.5 * sz_atr),
        "atr_pure_x2.0":         _clip(2.0 * sz_atr),
        "atr_pure_x2.5":         _clip(2.5 * sz_atr),
        "atr_pure_x3.0":         _clip(3.0 * sz_atr),
        "vol_target_atr_18":     vol_target_atr(signals, features, returns, CAPITAL, target_vol=0.18),
        "vol_target_atr_20":     vol_target_atr(signals, features, returns, CAPITAL, target_vol=0.20),
        # Leveraged ATR + asymmetric vol cap (cuts in stress, doesn't add leverage on calm)
        "atr_x1.5_volcap10":     leveraged_atr_voltgt(signals, features, returns, CAPITAL, 1.5, 0.10),
        "atr_x2_volcap12":       leveraged_atr_voltgt(signals, features, returns, CAPITAL, 2.0, 0.12),
        "atr_x2_volcap10":       leveraged_atr_voltgt(signals, features, returns, CAPITAL, 2.0, 0.10),
        "atr_x2.5_volcap13":     leveraged_atr_voltgt(signals, features, returns, CAPITAL, 2.5, 0.13),
        "atr_x3_volcap14":       leveraged_atr_voltgt(signals, features, returns, CAPITAL, 3.0, 0.14),
        # Raw atr (no 100%-gross cap), various max_gross levels
        "atr_raw_unbounded":     raw_atr_sizes(signals, features, CAPITAL),
        "atr_raw_g150":          raw_atr_sizes(signals, features, CAPITAL, 1.50),
        "atr_raw_g200":          raw_atr_sizes(signals, features, CAPITAL, 2.00),
        "atr_raw_g250":          raw_atr_sizes(signals, features, CAPITAL, 2.50),
        # vol-target on the 70/30 blend
        "vol_target_blend_70_30": vol_target_atr(
            signals, features, returns, CAPITAL, target_vol=0.10
        ).pipe(lambda b: _clip(0.7 * b + 0.3 * sz_half)),
        # Sharpe-weighted Kelly: combine sharpe_weighted_atr × half_kelly weights
        "sharpe_x_kelly":        _clip(
            0.5 * sharpe_weighted_atr(signals, features, returns, CAPITAL)
            + 0.5 * sz_half
        ),
    }

    print(f"\n{'Method':<24} {'AnnRet%':>9} {'Sharpe':>8} {'MaxDD%':>8} "
          f"{'Calmar':>8} {'MaxPos$':>10}")
    print("-" * 78)
    rows = []
    for name, sz in variants.items():
        sz_overlay = defensive_tilt_overlay(sz, signals, macro, CAPITAL)
        ret = portfolio_returns(sz_overlay, returns)
        s = summarise(ret, name)
        max_pos = sz_overlay.abs().max().max()
        rows.append((name, s["ann_return"], s["sharpe"], s["max_drawdown"],
                     s["calmar"], max_pos))
        print(f"{s['label']:<24} {s['ann_return']:>9} {s['sharpe']:>8} "
              f"{s['max_drawdown']:>8} {s['calmar']:>8} ${max_pos:>9,.0f}")

    print("\nRanked by Sharpe:")
    rows.sort(key=lambda r: -float(r[2]))
    for r in rows:
        print(f"  {r[0]:<24} {r[1]:>7}  Sharpe {r[2]:>6}  DD {r[3]:>6}")
    print("\nRanked by AnnRet:")
    rows.sort(key=lambda r: -float(r[1]))
    for r in rows:
        print(f"  {r[0]:<24} {r[1]:>7}  Sharpe {r[2]:>6}  DD {r[3]:>6}")


if __name__ == "__main__":
    main()
