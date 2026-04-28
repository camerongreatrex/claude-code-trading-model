"""
test_expanded_universe.py
-------------------------
Test whether adding the 114 expansion S&P 500 stocks lifts performance
when combined with the new vol-target blend sizer (blend_a25_vt14 @1.5x).

Two configs:
  CORE-ONLY     — 42-ticker universe, current production state.
  CORE+EXPANDED — 156-ticker universe, signals = core ∪ expansion.

Each is run with both the BASE atr_lev_1.5x sizer and the new
blend_a25_vt14 @1.5x to isolate universe lift vs sizer lift.

Run: python -m v1.scripts.test_expanded_universe
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes, defensive_tilt_overlay,
    portfolio_returns, summarise, MAX_POSITION_PCT, CAPITAL,
    INDEX_ETF_TICKERS, INDEX_ETF_CAP, RISK_PER_TRADE,
)


def _clip(sz):
    return sz.clip(-CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT)


def atr_sizes_local(signals, features, capital):
    """Mirrors portfolio.atr_sizes but accepts signals with names not in features."""
    dollar_risk = capital * RISK_PER_TRADE
    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
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
    for _etf in INDEX_ETF_TICKERS:
        if _etf in sizes.columns:
            sizes[_etf] = sizes[_etf].clip(-capital * INDEX_ETF_CAP, capital * INDEX_ETF_CAP)
    gross = sizes.abs().sum(axis=1).replace(0, np.nan)
    scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return sizes.multiply(scale, axis=0)


def vol_target_atr(signals, features, returns, capital,
                   target_vol=0.14, scale_max=1.5, window=63):
    base = atr_sizes_local(signals, features, capital)
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    scaler = (target_vol / realized).clip(0.3, scale_max).shift(1).fillna(1.0)
    return _clip(base.multiply(scaler, axis=0))


def blend_a25_vt14_x15(signals, features, returns, macro):
    sz_atr   = atr_sizes_local(signals, features, CAPITAL)
    sz_vt    = vol_target_atr(signals, features, returns, CAPITAL,
                              target_vol=0.14, scale_max=1.5)
    sz_blend = _clip(0.25 * sz_vt + 0.75 * sz_atr)
    sz       = _clip(sz_blend * 1.5)
    return defensive_tilt_overlay(sz, signals, macro, CAPITAL)


def base_atr_lev15(signals, features, macro):
    sz = _clip(atr_sizes_local(signals, features, CAPITAL) * 1.5)
    return defensive_tilt_overlay(sz, signals, macro, CAPITAL)


def load_features_returns(tickers):
    """Load features + log_returns for the given tickers from data/v1/features/"""
    features: dict = {}
    returns = pd.DataFrame()
    feat_dir = Path("data/v1/features")
    for t in tickers:
        p = feat_dir / f"{t}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            features[t] = f
            returns[t]  = f["log_return"]
    return features, returns


def main() -> None:
    SIGNAL_DIR  = Path("data/v1/signals")
    MACRO_DIR   = Path("data/shared/macro")

    print("Loading core signals + expansion signals...")
    core_signals = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    exp_signals  = pd.read_parquet(SIGNAL_DIR / "expanded_signals.parquet")

    common_idx = core_signals.index.intersection(exp_signals.index)
    print(f"Common date range: {common_idx.min().date()} to {common_idx.max().date()} "
          f"({len(common_idx)} days)")

    # Build combined signals: core takes precedence (overlay applied),
    # add expansion stocks where available
    all_tickers = sorted(set(core_signals.columns) | set(exp_signals.columns))
    core_set    = set(core_signals.columns)
    exp_only    = [t for t in exp_signals.columns if t not in core_set]
    print(f"Core: {len(core_set)} tickers; expansion-only: {len(exp_only)}; total: {len(all_tickers)}")

    combined = pd.DataFrame(0, index=common_idx, columns=all_tickers, dtype=int)
    for t in core_set:
        combined[t] = core_signals[t].reindex(common_idx).fillna(0).astype(int)
    for t in exp_only:
        combined[t] = exp_signals[t].reindex(common_idx).fillna(0).astype(int)

    # Load all features (core + expansion)
    features_all, returns_all = load_features_returns(all_tickers)
    returns_all = returns_all.reindex(common_idx).dropna(how="all")

    # Restrict signals to date range we have returns for
    common_idx2 = combined.index.intersection(returns_all.index)
    combined    = combined.reindex(common_idx2).fillna(0)
    returns_all = returns_all.reindex(common_idx2).fillna(0)

    # Core-only frames
    core_aligned = core_signals.reindex(common_idx2).fillna(0).astype(int)
    features_core, returns_core = load_features_returns(core_signals.columns.tolist())
    returns_core = returns_core.reindex(common_idx2).fillna(0)

    macro_path = MACRO_DIR / "macro_features.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else pd.DataFrame()

    print(f"\nFinal universe sizes — core: {len(core_aligned.columns)} tickers, "
          f"combined: {combined.shape[1]} tickers")
    print(f"Mean active signals/day — core: "
          f"{(core_aligned != 0).sum(axis=1).mean():.1f}; combined: "
          f"{(combined != 0).sum(axis=1).mean():.1f}")

    print("\n" + "=" * 96)
    print("EXPANDED UNIVERSE TEST — same date range, same overlay-applied core signals")
    print("=" * 96)
    print(f"{'config':<48} {'AnnRet%':>8} {'Sharpe':>7} {'MaxDD%':>8} "
          f"{'Calmar':>7} {'Gross':>6}")
    print("-" * 96)

    rows = []

    # 1. CORE-ONLY × atr_lev_1.5x (the current production)
    sz = base_atr_lev15(core_aligned, features_core, macro)
    ret = portfolio_returns(sz, returns_core)
    s = summarise(ret, "core × atr_lev_1.5x (PROD)")
    g = float(sz.abs().sum(axis=1).mean()) / CAPITAL
    rows.append((s["label"], s, g))
    print(f"{s['label']:<48} {s['ann_return']:>8} {s['sharpe']:>7} "
          f"{s['max_drawdown']:>8} {s['calmar']:>7} {g:>6.2f}")

    # 2. CORE-ONLY × blend_a25_vt14 @1.5x (new winner from prior tests)
    sz = blend_a25_vt14_x15(core_aligned, features_core, returns_core, macro)
    ret = portfolio_returns(sz, returns_core)
    s = summarise(ret, "core × blend_a25_vt14 @1.5x")
    g = float(sz.abs().sum(axis=1).mean()) / CAPITAL
    rows.append((s["label"], s, g))
    print(f"{s['label']:<48} {s['ann_return']:>8} {s['sharpe']:>7} "
          f"{s['max_drawdown']:>8} {s['calmar']:>7} {g:>6.2f}")

    # 3. COMBINED × atr_lev_1.5x
    sz = base_atr_lev15(combined, features_all, macro)
    ret = portfolio_returns(sz, returns_all)
    s = summarise(ret, "combined × atr_lev_1.5x")
    g = float(sz.abs().sum(axis=1).mean()) / CAPITAL
    rows.append((s["label"], s, g))
    print(f"{s['label']:<48} {s['ann_return']:>8} {s['sharpe']:>7} "
          f"{s['max_drawdown']:>8} {s['calmar']:>7} {g:>6.2f}")

    # 4. COMBINED × blend_a25_vt14 @1.5x
    sz = blend_a25_vt14_x15(combined, features_all, returns_all, macro)
    ret = portfolio_returns(sz, returns_all)
    s = summarise(ret, "combined × blend_a25_vt14 @1.5x")
    g = float(sz.abs().sum(axis=1).mean()) / CAPITAL
    rows.append((s["label"], s, g))
    print(f"{s['label']:<48} {s['ann_return']:>8} {s['sharpe']:>7} "
          f"{s['max_drawdown']:>8} {s['calmar']:>7} {g:>6.2f}")


if __name__ == "__main__":
    main()
