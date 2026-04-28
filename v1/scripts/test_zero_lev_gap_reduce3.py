"""
test_zero_lev_gap_reduce3.py
----------------------------
Round 5: focused sweep around the round-2 winner `PROD + adx22 + momt`
(20.93% OOS / 2.306 Sh / -6.51 DD).

Goals:
  - Push AnnRet higher
  - Tighten gap a bit more
  - Keep DD < -7% comfortably

Axes:
  - ADX threshold 20-25 with momt (which level optimal?)
  - momt strength (0.6-1.4 vs 0.7-1.3 vs 0.8-1.2)
  - momt window (42, 63, 90 days)
  - Stack ac55 + vix28 with adx22 + momt
  - Try lev_x 1.6, 1.7 (top-N + cap absorbs anyway, may free more headroom)

Run: python -m v1.scripts.test_zero_lev_gap_reduce3
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
    r = mom_m.where(sz > 0).rank(axis=1, pct=True).fillna(0.5)
    return sz.multiply(lo + (hi - lo) * r)


def vix_gate(sz, vix, threshold=25.0, floor=0.65):
    v = vix.reindex(sz.index).ffill().fillna(15.0)
    scale = pd.Series(np.where(v > threshold, floor, 1.0), index=sz.index)
    return sz.multiply(scale, axis=0)


def asset_class_quota(sz, max_pct=0.55):
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


# ── factory ───────────────────────────────────────────────────────────────────
def make_fn(features, n=11, target_vol=0.14, scale_max=2.5, lev_x=1.5,
            adx_thr=22, mom_tilt=True, mom_window=63, mom_lo=0.7, mom_hi=1.3,
            vix_thr=None, vix=None, ac_quota=None):
    def fn(signals, returns):
        adx_m = adx_matrix(signals, features)
        conviction = adx_m * signals.abs()
        sz = vol_target_atr(signals, features, returns, CAPITAL,
                            target_vol=target_vol, scale_max=scale_max)
        if adx_thr > 0:
            sz = adx_filter(sz, adx_m, threshold=adx_thr)
        sz = top_n(sz, conviction, n)
        if mom_tilt:
            mom_m = mom_matrix(signals, features, mom_window)
            sz = momentum_tilt(sz, mom_m, lo=mom_lo, hi=mom_hi)
        sz = _clip(sz * lev_x)
        if vix_thr is not None and vix is not None:
            sz = vix_gate(sz, vix, threshold=vix_thr, floor=0.65)
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
    macro = pd.read_parquet(MACRO_DIR / "macro_features.parquet")
    vix = macro["vix"]
    print(f"Universe: {len(signals.columns)} tickers, {len(signals)} days")

    cands = {
        # Reference
        "PROD top11_vt14sm25_x1.5":           make_fn(features, adx_thr=0, mom_tilt=False),
        # Round-2 winner
        "WIN adx22+momt":                      make_fn(features, adx_thr=22),

        # ADX threshold sweep around 22 with momt
        "adx20+momt":                          make_fn(features, adx_thr=20),
        "adx21+momt":                          make_fn(features, adx_thr=21),
        "adx23+momt":                          make_fn(features, adx_thr=23),
        "adx24+momt":                          make_fn(features, adx_thr=24),
        "adx25+momt":                          make_fn(features, adx_thr=25),

        # Momt strength variations on adx22
        "adx22+momt(0.6/1.4)":                 make_fn(features, adx_thr=22, mom_lo=0.6, mom_hi=1.4),
        "adx22+momt(0.5/1.5)":                 make_fn(features, adx_thr=22, mom_lo=0.5, mom_hi=1.5),
        "adx22+momt(0.8/1.2)":                 make_fn(features, adx_thr=22, mom_lo=0.8, mom_hi=1.2),
        "adx22+momt(0.6/1.3)":                 make_fn(features, adx_thr=22, mom_lo=0.6, mom_hi=1.3),

        # Momt window variations on adx22
        "adx22+momt_w42":                      make_fn(features, adx_thr=22, mom_window=42),
        "adx22+momt_w90":                      make_fn(features, adx_thr=22, mom_window=90),
        "adx22+momt_w126":                     make_fn(features, adx_thr=22, mom_window=126),

        # Stack ac quotas / vix on adx22+momt
        "adx22+momt+ac50":                     make_fn(features, adx_thr=22, ac_quota=0.50),
        "adx22+momt+ac55":                     make_fn(features, adx_thr=22, ac_quota=0.55),
        "adx22+momt+ac60":                     make_fn(features, adx_thr=22, ac_quota=0.60),
        "adx22+momt+vix25":                    make_fn(features, adx_thr=22, vix_thr=25, vix=vix),
        "adx22+momt+vix28":                    make_fn(features, adx_thr=22, vix_thr=28, vix=vix),
        "adx22+momt+vix30":                    make_fn(features, adx_thr=22, vix_thr=30, vix=vix),

        # Triple stacks
        "adx22+momt+ac55+vix28":               make_fn(features, adx_thr=22, vix_thr=28, vix=vix, ac_quota=0.55),
        "adx22+momt+ac55+vix25":               make_fn(features, adx_thr=22, vix_thr=25, vix=vix, ac_quota=0.55),

        # Best-of-best with stronger momt
        "adx22+momt(0.6/1.4)+ac55":            make_fn(features, adx_thr=22, mom_lo=0.6, mom_hi=1.4, ac_quota=0.55),
        "adx22+momt(0.6/1.4)+ac55+vix28":      make_fn(features, adx_thr=22, mom_lo=0.6, mom_hi=1.4, vix_thr=28, vix=vix, ac_quota=0.55),
        "adx22+momt(0.5/1.5)+ac55":            make_fn(features, adx_thr=22, mom_lo=0.5, mom_hi=1.5, ac_quota=0.55),

        # Try larger lev_x with cap absorbing
        "adx22+momt_x1.7":                     make_fn(features, adx_thr=22, lev_x=1.7),
        "adx22+momt_x1.8":                     make_fn(features, adx_thr=22, lev_x=1.8),
        "adx22+momt_x2.0":                     make_fn(features, adx_thr=22, lev_x=2.0),
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
    df_safe = df[df["oos_dd"] > -10.0].sort_values("oos_ret", ascending=False)

    print("\n" + "=" * 130)
    print("SORTED BY OOS AnnRet (DD > -10%)")
    print("=" * 130)
    print(f"{'name':<40} {'IS_Sh':>7} {'IS_Ret':>8} {'OOS_Sh':>7} {'OOS_Act':>8} "
          f"{'OOS_Ret':>8} {'OOS_DD':>8} {'gap':>7}")
    for _, r in df_safe.iterrows():
        flag = " ★" if (r["oos_ret"] > 20.93) else ("  +" if r["oos_ret"] > 20.50 else "")
        print(f"{r['name']:<40} {r['is_sh']:>7.3f} {r['is_ret']:>8.2f} "
              f"{r['oos_sh']:>7.3f} {r['oos_act']:>8.3f} "
              f"{r['oos_ret']:>8.2f} {r['oos_dd']:>8.2f} {r['gap']:>+7.3f}{flag}")

    print("\nKEY: ★ = beats round-2 winner adx22+momt (20.93%);  + = > 20.50%")


if __name__ == "__main__":
    main()
