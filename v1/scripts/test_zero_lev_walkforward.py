"""
test_zero_lev_walkforward.py
----------------------------
OOS walk-forward validation of the top 1.0x-leverage candidates from
test_zero_lev_push2.py.  Confirms IS Pareto improvement is not overfit.

walk_forward already applies defensive_tilt_overlay AFTER sizing_fn, so the
sizing_fn returns capped sizes (gross ≤ 1.0).  The overlay only reallocates
between VGSH/TLT and other longs — gross stays at 1.0.

Run: python -m v1.scripts.test_zero_lev_walkforward
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes, kelly_sizes,
    portfolio_returns, summarise, MAX_POSITION_PCT, CAPITAL,
    walk_forward,
)


def _clip(sz):
    return sz.clip(-CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT)


def _gross_cap(sz, max_gross=1.0):
    gross = sz.abs().sum(axis=1).replace(0, np.nan)
    scale = (max_gross * CAPITAL / gross).clip(upper=1.0).fillna(1.0)
    return sz.multiply(scale, axis=0)


def vol_target_atr(signals, features, returns, capital,
                   target_vol=0.14, scale_max=3.0, window=63):
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


def top_n(sz_base, conviction, n):
    rank = conviction.rank(axis=1, ascending=False, method="first")
    keep = (rank <= n).astype(float)
    return _clip(sz_base * keep)


# ── sizing_fn factories ───────────────────────────────────────────────────────
def make_topn_vt_fn(features, n, target_vol, scale_max, lev_x):
    def fn(signals, returns):
        adx_m = adx_matrix(signals, features)
        conviction = adx_m * signals.abs()
        sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                target_vol=target_vol, scale_max=scale_max)
        sz = top_n(sz_vt, conviction, n)
        return _gross_cap(_clip(sz * lev_x), max_gross=1.0)
    return fn


def make_atr_capped_fn(features, lev_x):
    def fn(signals, returns):
        sz = _clip(atr_sizes(signals, features, CAPITAL) * lev_x)
        return _gross_cap(sz, max_gross=1.0)
    return fn


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

    candidates = {
        # Baselines
        "BASE atr @1.0x":                    make_atr_capped_fn(features, 1.0),
        "BASE atr_lev_1.5x (cap@1.0)":       make_atr_capped_fn(features, 1.5),
        # Top concentration candidates from push2
        "top10_vt14sm3_x2.5":                make_topn_vt_fn(features, 10, 0.14, 3.0, 2.5),
        "top10_vt20sm3_x1.5":                make_topn_vt_fn(features, 10, 0.20, 3.0, 1.5),
        "top11_vt14sm25_x1.5":               make_topn_vt_fn(features, 11, 0.14, 2.5, 1.5),
        "top12_vt14sm3_x2.5":                make_topn_vt_fn(features, 12, 0.14, 3.0, 2.5),
        "top12_vt20sm3_x1.5":                make_topn_vt_fn(features, 12, 0.20, 3.0, 1.5),
        "top12_vt18sm3_x1.5":                make_topn_vt_fn(features, 12, 0.18, 3.0, 1.5),
        "top12_vt14sm3_x1.5":                make_topn_vt_fn(features, 12, 0.14, 3.0, 1.5),
        "top10_vt18sm3_x1.5":                make_topn_vt_fn(features, 10, 0.18, 3.0, 1.5),
        # Conservative (best Calmar / lower DD)
        "top9_vt14sm25_x1.5":                make_topn_vt_fn(features, 9, 0.14, 2.5, 1.5),
    }

    print("\n" + "=" * 116)
    print("WALK-FORWARD OOS — 3y train / 1y test, 2018-2025 windows  (gross_cap = 1.0)")
    print("=" * 116)

    summary = []
    for name, fn in candidates.items():
        try:
            wf = walk_forward(signals, returns, sizing_fn=fn)
        except Exception as e:
            print(f"{name:<32}  FAILED: {e}")
            continue
        mean_sh   = wf["sharpe"].mean()
        mean_ret  = wf["ann_return"].mean()
        worst_dd  = wf["max_dd"].min()
        active_sh = wf["active_sharpe"].mean()
        summary.append((name, mean_sh, mean_ret, worst_dd, active_sh))
        print(f"\n  {name}")
        print(f"    {'period':<12} {'sharpe':>8} {'act_sh':>8} {'ann_ret':>8} {'max_dd':>8}")
        for _, row in wf.iterrows():
            print(f"    {row['period']:<12} {row['sharpe']:>8.3f} {row['active_sharpe']:>8.3f} "
                  f"{row['ann_return']:>8.2f} {row['max_dd']:>8.2f}")
        print(f"    {'MEAN':<12} {mean_sh:>8.3f} {active_sh:>8.3f} {mean_ret:>8.2f} {worst_dd:>8.2f} (worst)")

    print("\n" + "=" * 116)
    print("OOS SUMMARY — sorted by mean OOS AnnRet")
    print("=" * 116)
    print(f"{'name':<36} {'mean_sh':>10} {'act_sh':>10} {'mean_ret':>10} {'worst_dd':>10}")
    for n, sh, r, dd, act in sorted(summary, key=lambda x: -x[2]):
        flag = " ★" if dd > -12.0 else ""
        print(f"{n:<36} {sh:>10.3f} {act:>10.3f} {r:>10.2f} {dd:>10.2f}{flag}")


if __name__ == "__main__":
    main()
