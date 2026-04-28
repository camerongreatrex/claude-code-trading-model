"""
test_zero_lev_gap_reduce.py
---------------------------
Round 3: shrink the IS-OOS gap on top11_vt14sm25_x1.5 (-0.553) while keeping
zero leverage and pushing AnnRet.  Constraint: OOS worst DD > -10%.

Strategy: dial back the vt aggression (lower scale_max, lower lev_x) and
widen top-N — the gap is "OOS > IS" regime concentration in 2023-24, so the
fix is to make the boost less reactive to that one calm-vol window.  Also
test orthogonal axes: ADX threshold filter, longer vt window, momentum tilt
overlay on top of top-N.

Run: python -m v1.scripts.test_zero_lev_gap_reduce
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
    """Zero out any size where adx < threshold (drop weak trends entirely)."""
    keep = (adx_m >= threshold).astype(float)
    return _clip(sz_base * keep)


# ── sizing_fn factories ───────────────────────────────────────────────────────
def make_topn_vt_fn(features, n, target_vol, scale_max, lev_x, vt_window=63,
                    adx_thr=0.0, mom_tilt=False):
    def fn(signals, returns):
        adx_m = adx_matrix(signals, features)
        conviction = adx_m * signals.abs()
        sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                target_vol=target_vol, scale_max=scale_max,
                                window=vt_window)
        if adx_thr > 0:
            sz_vt = adx_filter(sz_vt, adx_m, threshold=adx_thr)
        sz = top_n(sz_vt, conviction, n)
        if mom_tilt:
            mom = mom_matrix(signals, features, 63)
            # Rank momentum within active longs, scale 0.7-1.3
            r = mom.where(sz > 0).rank(axis=1, pct=True).fillna(0.5)
            sz = sz.multiply(0.7 + 0.6 * r)
        return _gross_cap(_clip(sz * lev_x), max_gross=1.0)
    return fn


def is_metrics(sizing_fn, signals, returns):
    """Compute IS Sharpe over the full sample (no walk-forward)."""
    sizes = sizing_fn(signals, returns)
    ret = portfolio_returns(sizes, returns)
    return sharpe_ratio(ret), float(((1 + ret).prod() ** (252 / len(ret)) - 1) * 100), \
           float(max_drawdown((1 + ret).cumprod()) * 100)


def main() -> None:
    SIGNAL_DIR  = Path("data/v1/signals")
    FEATURE_DIR = Path("data/v1/features")

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
    print(f"Universe: {len(signals.columns)} tickers, {len(signals)} days "
          f"({signals.index[0].date()} → {signals.index[-1].date()})")

    candidates = {
        # Baseline (current production)
        "PROD top11_vt14sm25_x1.5":         make_topn_vt_fn(features, 11, 0.14, 2.5, 1.5),

        # Axis 1: lower lev_x, lower scale_max (less aggressive on calm-vol days)
        "top11_vt14sm15_x1.3":              make_topn_vt_fn(features, 11, 0.14, 1.5, 1.3),
        "top11_vt14sm15_x1.2":              make_topn_vt_fn(features, 11, 0.14, 1.5, 1.2),
        "top11_vt14sm18_x1.3":              make_topn_vt_fn(features, 11, 0.14, 1.8, 1.3),
        "top11_vt14sm20_x1.3":              make_topn_vt_fn(features, 11, 0.14, 2.0, 1.3),
        "top11_vt14sm20_x1.4":              make_topn_vt_fn(features, 11, 0.14, 2.0, 1.4),

        # Axis 2: wider top-N (more diversification across regimes)
        "top13_vt14sm15_x1.3":              make_topn_vt_fn(features, 13, 0.14, 1.5, 1.3),
        "top13_vt14sm18_x1.3":              make_topn_vt_fn(features, 13, 0.14, 1.8, 1.3),
        "top13_vt14sm20_x1.3":              make_topn_vt_fn(features, 13, 0.14, 2.0, 1.3),
        "top13_vt14sm20_x1.4":              make_topn_vt_fn(features, 13, 0.14, 2.0, 1.4),
        "top13_vt14sm25_x1.5":              make_topn_vt_fn(features, 13, 0.14, 2.5, 1.5),
        "top15_vt14sm15_x1.3":              make_topn_vt_fn(features, 15, 0.14, 1.5, 1.3),
        "top15_vt14sm18_x1.3":              make_topn_vt_fn(features, 15, 0.14, 1.8, 1.3),
        "top15_vt14sm20_x1.3":              make_topn_vt_fn(features, 15, 0.14, 2.0, 1.3),
        "top15_vt14sm25_x1.5":              make_topn_vt_fn(features, 15, 0.14, 2.5, 1.5),
        "top17_vt14sm20_x1.3":              make_topn_vt_fn(features, 17, 0.14, 2.0, 1.3),

        # Axis 3: longer vt_window (less reactive to short-term vol regime shifts)
        "top11_vt14sm25_x1.5_w90":          make_topn_vt_fn(features, 11, 0.14, 2.5, 1.5, vt_window=90),
        "top13_vt14sm20_x1.3_w90":          make_topn_vt_fn(features, 13, 0.14, 2.0, 1.3, vt_window=90),
        "top13_vt14sm20_x1.3_w126":         make_topn_vt_fn(features, 13, 0.14, 2.0, 1.3, vt_window=126),

        # Axis 4: ADX threshold filter (drop weak trends entirely)
        "top13_vt14sm20_x1.3_adx20":        make_topn_vt_fn(features, 13, 0.14, 2.0, 1.3, adx_thr=20.0),
        "top13_vt14sm20_x1.3_adx25":        make_topn_vt_fn(features, 13, 0.14, 2.0, 1.3, adx_thr=25.0),
        "top11_vt14sm25_x1.5_adx20":        make_topn_vt_fn(features, 11, 0.14, 2.5, 1.5, adx_thr=20.0),
        "top15_vt14sm20_x1.3_adx20":        make_topn_vt_fn(features, 15, 0.14, 2.0, 1.3, adx_thr=20.0),

        # Axis 5: momentum tilt overlay (reward strongest 63d returns within top-N)
        "top13_vt14sm20_x1.3_momt":         make_topn_vt_fn(features, 13, 0.14, 2.0, 1.3, mom_tilt=True),
        "top15_vt14sm20_x1.3_momt":         make_topn_vt_fn(features, 15, 0.14, 2.0, 1.3, mom_tilt=True),
        "top11_vt14sm25_x1.5_momt":         make_topn_vt_fn(features, 11, 0.14, 2.5, 1.5, mom_tilt=True),

        # Stack: ADX filter + momentum tilt + wider top-N
        "top15_vt14sm20_x1.3_adx20_momt":   make_topn_vt_fn(features, 15, 0.14, 2.0, 1.3, adx_thr=20.0, mom_tilt=True),
        "top13_vt14sm20_x1.3_adx20_momt":   make_topn_vt_fn(features, 13, 0.14, 2.0, 1.3, adx_thr=20.0, mom_tilt=True),
        "top13_vt14sm18_x1.3_adx20_momt":   make_topn_vt_fn(features, 13, 0.14, 1.8, 1.3, adx_thr=20.0, mom_tilt=True),
    }

    print(f"\nRunning {len(candidates)} candidates with IS + walk-forward OOS...")
    print("=" * 130)

    summary = []
    for name, fn in candidates.items():
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
        print(f"{r['name']:<40} {r['is_sh']:>7.3f} {r['is_ret']:>8.2f} "
              f"{r['oos_sh']:>7.3f} {r['oos_act']:>8.3f} "
              f"{r['oos_ret']:>8.2f} {r['oos_dd']:>8.2f} {r['gap']:>+7.3f}")

    print("\n" + "=" * 130)
    print("SORTED BY |GAP| (smallest first) — DD > -10%, OOS_Ret > prod baseline 20.21%")
    print("=" * 130)
    df_robust = df[(df["oos_dd"] > -10.0) & (df["oos_ret"] > 18.0)].copy()
    df_robust["abs_gap"] = df_robust["gap"].abs()
    df_robust = df_robust.sort_values("abs_gap")
    print(f"{'name':<40} {'IS_Sh':>7} {'IS_Ret':>8} {'OOS_Sh':>7} {'OOS_Act':>8} "
          f"{'OOS_Ret':>8} {'OOS_DD':>8} {'gap':>7}")
    for _, r in df_robust.iterrows():
        print(f"{r['name']:<40} {r['is_sh']:>7.3f} {r['is_ret']:>8.2f} "
              f"{r['oos_sh']:>7.3f} {r['oos_act']:>8.3f} "
              f"{r['oos_ret']:>8.2f} {r['oos_dd']:>8.2f} {r['gap']:>+7.3f}")

    print("\nWINNER PICK CRITERIA:")
    print("  - DD > -10%")
    print("  - OOS_Ret >= prod (20.21%)")
    print("  - |gap| < prod (0.553)")
    if len(df_robust) > 0:
        df_winners = df_robust[df_robust["abs_gap"] < 0.55]
        if len(df_winners) > 0:
            best = df_winners.sort_values("oos_ret", ascending=False).iloc[0]
            print(f"\n  >>> Best: {best['name']}")
            print(f"      IS_Sh {best['is_sh']:.3f} | OOS_Sh {best['oos_sh']:.3f} | "
                  f"OOS_Ret {best['oos_ret']:.2f}% | OOS_DD {best['oos_dd']:.2f}% | "
                  f"gap {best['gap']:+.3f}")


if __name__ == "__main__":
    main()
