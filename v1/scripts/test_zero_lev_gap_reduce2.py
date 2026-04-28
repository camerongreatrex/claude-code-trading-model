"""
test_zero_lev_gap_reduce2.py
----------------------------
Round 4: build on the small wins from gap_reduce v1.
  - top11_vt14sm25_x1.5_adx20  → 20.48% (best so far, +0.27pp over prod)
  - top11_vt14sm25_x1.5_momt   → 20.43% with better DD (-6.51 vs -6.95)

This round stacks them, sweeps ADX threshold, adds VIX-gated leverage
(cut size when VIX > X — protects 2022 and reduces OOS regime-dependence),
adds drawdown-aware sizing, and tests an asset-class-quota overlay.

Run: python -m v1.scripts.test_zero_lev_gap_reduce2
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes,
    portfolio_returns, MAX_POSITION_PCT, CAPITAL,
    walk_forward,
)
from v1.pipeline.backtester import sharpe_ratio, max_drawdown
from v1.pipeline.data_pipeline import ASSET_CLASS


def _clip(sz):
    return sz.clip(-CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT)


def _gross_cap(sz, max_gross=1.0):
    gross = sz.abs().sum(axis=1).replace(0, np.nan)
    scale = (max_gross * CAPITAL / gross).clip(upper=1.0).fillna(1.0)
    return sz.multiply(scale, axis=0)


def vol_target_atr(signals, features, returns, capital,
                   target_vol=0.14, scale_max=2.5, window=63):
    base = atr_sizes(signals, features, capital)
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    scaler = (target_vol / realized).clip(0.3, scale_max).shift(1).fillna(1.0)
    return _clip(base.multiply(scaler, axis=0))


def adx_matrix(signals, features):
    m = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for t in signals.columns:
        if t in features and "adx" in features[t].columns:
            m[t] = features[t]["adx"].reindex(signals.index).ffill().fillna(0.0)
    return m


def mom_matrix(signals, features, window=63):
    m = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for t in signals.columns:
        if t in features and "Close" in features[t].columns:
            c = features[t]["Close"].reindex(signals.index).ffill()
            m[t] = (c / c.shift(window) - 1).fillna(0.0)
    return m


def top_n(sz_base, conviction, n):
    rank = conviction.rank(axis=1, ascending=False, method="first")
    keep = (rank <= n).astype(float)
    return _clip(sz_base * keep)


def adx_filter(sz_base, adx_m, threshold=20.0):
    keep = (adx_m >= threshold).astype(float)
    return _clip(sz_base * keep)


def momentum_tilt(sz, mom_m, lo=0.7, hi=1.3):
    """Within active longs, scale by 63d momentum percentile rank."""
    r = mom_m.where(sz > 0).rank(axis=1, pct=True).fillna(0.5)
    return sz.multiply(lo + (hi - lo) * r)


def vix_gate(sz, vix, threshold=25.0, floor=0.6):
    """When VIX > threshold, scale sizes by `floor` (default 60%)."""
    v = vix.reindex(sz.index).ffill().fillna(15.0)
    scale = pd.Series(np.where(v > threshold, floor, 1.0), index=sz.index)
    return sz.multiply(scale, axis=0)


def dd_aware(sz, returns, dd_trigger=-0.04, scale_floor=0.5, lookback=21):
    """Cut size when rolling 21d portfolio return < dd_trigger.  Approximate
    DD-trigger: when realised drawdown over `lookback` days exceeds `dd_trigger`,
    scale gross by `scale_floor`."""
    weights = sz.shift(1) / CAPITAL
    port_ret = (weights * returns.reindex(columns=sz.columns)).sum(axis=1)
    eq = (1 + port_ret).cumprod()
    rolling_max = eq.rolling(lookback, min_periods=5).max()
    rolling_dd = (eq / rolling_max - 1)
    scale = pd.Series(np.where(rolling_dd < dd_trigger, scale_floor, 1.0),
                      index=sz.index).shift(1).fillna(1.0)
    return sz.multiply(scale, axis=0)


def asset_class_quota(sz, max_pct=0.55):
    """Cap any single asset class at max_pct of gross.  Reallocates excess
    proportionally within the same class."""
    out = sz.copy()
    by_class = {}
    for t in sz.columns:
        ac = ASSET_CLASS.get(t, "other")
        by_class.setdefault(ac, []).append(t)

    for ac, tickers in by_class.items():
        cls_sz = out[tickers].abs().sum(axis=1)
        gross = out.abs().sum(axis=1).replace(0, np.nan)
        cls_pct = cls_sz / gross
        over = cls_pct > max_pct
        if over.any():
            scale = pd.Series(1.0, index=sz.index)
            scale[over] = (max_pct * gross[over] / cls_sz[over]).fillna(1.0)
            for t in tickers:
                out[t] = out[t] * scale
    return _clip(out)


# ── sizing_fn factory ─────────────────────────────────────────────────────────
def make_fn(features, n=11, target_vol=0.14, scale_max=2.5, lev_x=1.5,
            vt_window=63, adx_thr=0.0, mom_tilt=False, vix_thr=None, vix=None,
            dd_thr=None, ac_quota=None):
    def fn(signals, returns):
        adx_m = adx_matrix(signals, features)
        conviction = adx_m * signals.abs()
        sz = vol_target_atr(signals, features, returns, CAPITAL,
                            target_vol=target_vol, scale_max=scale_max,
                            window=vt_window)
        if adx_thr > 0:
            sz = adx_filter(sz, adx_m, threshold=adx_thr)
        sz = top_n(sz, conviction, n)
        if mom_tilt:
            mom_m = mom_matrix(signals, features, 63)
            sz = momentum_tilt(sz, mom_m)
        sz = _clip(sz * lev_x)
        if vix_thr is not None and vix is not None:
            sz = vix_gate(sz, vix, threshold=vix_thr, floor=0.65)
        if dd_thr is not None:
            sz = dd_aware(sz, returns, dd_trigger=dd_thr)
        if ac_quota is not None:
            sz = asset_class_quota(sz, max_pct=ac_quota)
        return _gross_cap(sz, max_gross=1.0)
    return fn


def is_metrics(sizing_fn, signals, returns):
    sizes = sizing_fn(signals, returns)
    ret = portfolio_returns(sizes, returns)
    return sharpe_ratio(ret), float(((1 + ret).prod() ** (252 / len(ret)) - 1) * 100), \
           float(max_drawdown((1 + ret).cumprod()) * 100)


def main() -> None:
    SIGNAL_DIR  = Path("data/v1/signals")
    FEATURE_DIR = Path("data/v1/features")
    MACRO_DIR   = Path("data/shared/macro")

    print("Loading signals + features...")
    signals_raw = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    features: dict = {}
    returns = pd.DataFrame()
    for t in signals_raw.columns:
        p = FEATURE_DIR / f"{t}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            features[t] = f
            returns[t]  = f["log_return"]
    returns = returns.dropna()
    signals = signals_raw.reindex(returns.index).fillna(0)
    macro   = pd.read_parquet(MACRO_DIR / "macro_features.parquet")
    vix     = macro["vix"]
    print(f"Universe: {len(signals.columns)} tickers, {len(signals)} days")

    cands = {
        # Baseline — current production
        "PROD top11_vt14sm25_x1.5":           make_fn(features),
        # Round 1 wins
        "PROD + adx20":                        make_fn(features, adx_thr=20),
        "PROD + momt":                         make_fn(features, mom_tilt=True),
        # Stack the two round-1 wins
        "PROD + adx20 + momt":                 make_fn(features, adx_thr=20, mom_tilt=True),
        # ADX threshold sweep on PROD
        "PROD + adx15":                        make_fn(features, adx_thr=15),
        "PROD + adx18":                        make_fn(features, adx_thr=18),
        "PROD + adx22":                        make_fn(features, adx_thr=22),
        "PROD + adx25":                        make_fn(features, adx_thr=25),
        # ADX threshold sweep + momt
        "PROD + adx18 + momt":                 make_fn(features, adx_thr=18, mom_tilt=True),
        "PROD + adx22 + momt":                 make_fn(features, adx_thr=22, mom_tilt=True),
        "PROD + adx25 + momt":                 make_fn(features, adx_thr=25, mom_tilt=True),
        # VIX gate (cut to 65% when VIX > X)
        "PROD + vix25":                        make_fn(features, vix_thr=25, vix=vix),
        "PROD + vix28":                        make_fn(features, vix_thr=28, vix=vix),
        "PROD + vix30":                        make_fn(features, vix_thr=30, vix=vix),
        "PROD + adx20 + vix28":                make_fn(features, adx_thr=20, vix_thr=28, vix=vix),
        "PROD + adx20 + momt + vix28":         make_fn(features, adx_thr=20, mom_tilt=True, vix_thr=28, vix=vix),
        # DD-aware sizing
        "PROD + dd-4pct":                      make_fn(features, dd_thr=-0.04),
        "PROD + dd-5pct":                      make_fn(features, dd_thr=-0.05),
        "PROD + adx20 + momt + dd-5pct":       make_fn(features, adx_thr=20, mom_tilt=True, dd_thr=-0.05),
        # Asset-class quota (no class > 55%)
        "PROD + ac55":                         make_fn(features, ac_quota=0.55),
        "PROD + ac60":                         make_fn(features, ac_quota=0.60),
        "PROD + adx20 + momt + ac55":          make_fn(features, adx_thr=20, mom_tilt=True, ac_quota=0.55),
        # Full stack
        "FULL adx20 momt vix28 ac60":          make_fn(features, adx_thr=20, mom_tilt=True, vix_thr=28, vix=vix, ac_quota=0.60),
        "FULL adx20 momt vix28 dd5":           make_fn(features, adx_thr=20, mom_tilt=True, vix_thr=28, vix=vix, dd_thr=-0.05),
        # Top-12 variants — slight diversification bump on best stack
        "top12 + adx20 + momt":                make_fn(features, n=12, adx_thr=20, mom_tilt=True),
        "top12 + adx20":                       make_fn(features, n=12, adx_thr=20),
        # Top-10 — tighter conviction with adx + momt
        "top10 + adx20 + momt":                make_fn(features, n=10, adx_thr=20, mom_tilt=True),
        "top10 + adx20":                       make_fn(features, n=10, adx_thr=20),
    }

    print(f"\nRunning {len(cands)} candidates...")
    print("=" * 130)

    summary = []
    for name, fn in cands.items():
        try:
            is_sh, is_ar, is_dd = is_metrics(fn, signals, returns)
            wf = walk_forward(signals, returns, sizing_fn=fn)
        except Exception as e:
            print(f"{name:<40}  FAILED: {e}")
            continue
        oos_sh   = wf["sharpe"].mean()
        oos_act  = wf["active_sharpe"].mean()
        oos_ret  = wf["ann_return"].mean()
        oos_dd   = wf["max_dd"].min()
        gap      = oos_sh - is_sh
        summary.append({
            "name": name, "is_sh": is_sh, "is_ret": is_ar, "is_dd": is_dd,
            "oos_sh": oos_sh, "oos_act": oos_act, "oos_ret": oos_ret,
            "oos_dd": oos_dd, "gap": gap,
        })

    df = pd.DataFrame(summary)

    print("\n" + "=" * 130)
    print("SORTED BY OOS AnnRet (DD > -10% only)")
    print("=" * 130)
    df_safe = df[df["oos_dd"] > -10.0].sort_values("oos_ret", ascending=False)
    print(f"{'name':<40} {'IS_Sh':>7} {'IS_Ret':>8} {'OOS_Sh':>7} {'OOS_Act':>8} "
          f"{'OOS_Ret':>8} {'OOS_DD':>8} {'gap':>7}")
    for _, r in df_safe.iterrows():
        flag = " ★" if (r["oos_ret"] > 20.21 and r["oos_dd"] > -7.0) else ""
        print(f"{r['name']:<40} {r['is_sh']:>7.3f} {r['is_ret']:>8.2f} "
              f"{r['oos_sh']:>7.3f} {r['oos_act']:>8.3f} "
              f"{r['oos_ret']:>8.2f} {r['oos_dd']:>8.2f} {r['gap']:>+7.3f}{flag}")

    print("\nKEY:  ★ = beats production (OOS_Ret > 20.21% AND OOS_DD > -7%)")


if __name__ == "__main__":
    main()
