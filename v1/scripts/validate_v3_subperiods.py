"""
Validate V3 production (CAND-A + 12% 4-asset sleeve) for sub-period stability.
Universe starts 2019-08. Checks: (1) 1y rolling walk-forward (3 windows),
(2) 6mo rolling (~6 windows), (3) year-by-year OOS slice, (4) vs baseline V2
(no sleeve, cap10) on same windows. Overfit signal: OOS collapse / divergence.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import v1.portfolio.portfolio as _pp
from v1.portfolio.portfolio import (
    top_n_adx_momt_ac_sizes,
    diversifier_sleeve_overlay,
    portfolio_returns,
    defensive_tilt_overlay,
    _get_macro,
    SIGNAL_DIR,
    FEATURE_DIR,
    CAPITAL,
)
from v1.pipeline.data_pipeline import TICKER_LIST
from v1.pipeline.signal_generation import ATR_PARTIAL_REMAIN
from v1.pipeline.backtester import sharpe_ratio, max_drawdown


def sortino(ret):
    dd = ret[ret < 0]
    if len(dd) < 2:
        return float("nan")
    s = dd.std() * np.sqrt(252)
    return float(ret.mean() * 252 / s) if s > 0 else float("nan")


def metrics(r):
    if len(r) < 30:
        return dict(sharpe=float("nan"), ann=float("nan"), mdd=float("nan"),
                    calmar=float("nan"), sortino=float("nan"))
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    mdd = max_drawdown((1 + r).cumprod())
    return dict(
        sharpe=sharpe_ratio(r), ann=ann, mdd=mdd,
        calmar=ann / abs(mdd) if mdd < 0 else float("nan"),
        sortino=sortino(r),
        n=len(r),
    )


def load_inputs():
    multi = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    feats: dict = {}
    rets = pd.DataFrame()
    for t in TICKER_LIST:
        fp = FEATURE_DIR / f"{t}.parquet"
        if not fp.exists():
            continue
        f = pd.read_parquet(fp)
        feats[t] = f
        rets[t] = f["log_return"]
    rets = rets.dropna()
    sig = multi.reindex(rets.index).fillna(0.0)
    hs_path = SIGNAL_DIR / "half_size.parquet"
    if hs_path.exists():
        hs = pd.read_parquet(hs_path).reindex(rets.index).fillna(False)
        hs = hs.reindex(columns=sig.columns, fill_value=False)
        sig = sig * np.where(hs, ATR_PARTIAL_REMAIN, 1.0)
    return sig, feats, rets, _get_macro()


def make_sizer(use_sleeve: bool, cap_pct: float):
    base_kw = dict(
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )

    def sizer(sig, feats, rets, macro):
        old_cap = _pp.MAX_POSITION_PCT
        _pp.MAX_POSITION_PCT = cap_pct
        try:
            sizes = top_n_adx_momt_ac_sizes(sig, feats, rets, CAPITAL, **base_kw)
            sizes = defensive_tilt_overlay(sizes, sig, macro, CAPITAL)
            if use_sleeve:
                sizes = diversifier_sleeve_overlay(
                    sizes, CAPITAL,
                    sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
                    sleeve_pct=0.12,
                )
        finally:
            _pp.MAX_POSITION_PCT = old_cap
        return sizes

    return sizer


def walkforward(sig, feats, rets, macro, sizer, train_days, test_days):
    out = []
    start = train_days
    while start + test_days <= len(sig):
        ctx_start = max(0, start - train_days)
        s = sig.iloc[ctx_start: start + test_days]
        r = rets.iloc[ctx_start: start + test_days]
        sizes = sizer(s, feats, r, macro)
        period = portfolio_returns(sizes.iloc[-test_days:], r.iloc[-test_days:]).dropna()
        m = metrics(period)
        m["period"] = (f"{sig.index[start].date()}→"
                        f"{sig.index[start+test_days-1].date()}")
        out.append(m)
        start += test_days
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in "
          f"{time.time()-t0:.1f}s\n")

    # Full-period series for both, sliced for sub-period analysis
    sizer_v3   = make_sizer(use_sleeve=True,  cap_pct=0.12)
    sizer_v2   = make_sizer(use_sleeve=False, cap_pct=0.10)

    sizes_v3 = sizer_v3(sig, feats, rets, macro)
    sizes_v2 = sizer_v2(sig, feats, rets, macro)
    pr_v3 = portfolio_returns(sizes_v3, rets).dropna()
    pr_v2 = portfolio_returns(sizes_v2, rets).dropna()

    # ── Calendar-year breakdown ──
    print("=" * 96)
    print("Calendar-year metrics (full series, no walk-forward refit)")
    print("=" * 96)
    print(f"{'Year':<6} {'V2 (cap10, no sleeve)':<46} {'V3 (cap12 + 12% sleeve)':<46}")
    print(f"{'':6} {'Sh':>6} {'Ann':>7} {'MaxDD':>7} {'Cal':>5} {'Sort':>6}  "
          f"{'Sh':>6} {'Ann':>7} {'MaxDD':>7} {'Cal':>5} {'Sort':>6}")
    for yr in sorted(set(pr_v3.index.year)):
        v2y = pr_v2[pr_v2.index.year == yr]
        v3y = pr_v3[pr_v3.index.year == yr]
        if len(v3y) < 30 or len(v2y) < 30:
            continue
        m2 = metrics(v2y)
        m3 = metrics(v3y)
        print(f"{yr:<6} "
              f"{m2['sharpe']:>6.2f} {m2['ann']*100:>6.2f}% {m2['mdd']*100:>6.2f}% "
              f"{m2['calmar']:>5.2f} {m2['sortino']:>6.2f}  "
              f"{m3['sharpe']:>6.2f} {m3['ann']*100:>6.2f}% {m3['mdd']*100:>6.2f}% "
              f"{m3['calmar']:>5.2f} {m3['sortino']:>6.2f}")

    # ── Walk-forward 1y rolling (3 windows; 22-23, 23-24, 24-25) ──
    print("\n" + "=" * 96)
    print("Walk-forward 1-year rolling (3-yr train / 1-yr test, true OOS refit)")
    print("=" * 96)
    print(f"{'Period':<26} {'V2':<32} {'V3':<32}")
    print(f"{'':26} {'Sh':>6} {'Ann':>7} {'MaxDD':>7} {'Cal':>5}  "
          f"{'Sh':>6} {'Ann':>7} {'MaxDD':>7} {'Cal':>5}")
    wf_v2 = walkforward(sig, feats, rets, macro, sizer_v2, 756, 252)
    wf_v3 = walkforward(sig, feats, rets, macro, sizer_v3, 756, 252)
    for i in range(min(len(wf_v2), len(wf_v3))):
        a, b = wf_v2.iloc[i], wf_v3.iloc[i]
        print(f"{a['period']:<26} "
              f"{a['sharpe']:>6.2f} {a['ann']*100:>6.2f}% {a['mdd']*100:>6.2f}% {a['calmar']:>5.2f}  "
              f"{b['sharpe']:>6.2f} {b['ann']*100:>6.2f}% {b['mdd']*100:>6.2f}% {b['calmar']:>5.2f}")
    print(f"{'MEAN':<26} "
          f"{wf_v2['sharpe'].mean():>6.2f} {wf_v2['ann'].mean()*100:>6.2f}% "
          f"{wf_v2['mdd'].mean()*100:>6.2f}% {wf_v2['calmar'].mean():>5.2f}  "
          f"{wf_v3['sharpe'].mean():>6.2f} {wf_v3['ann'].mean()*100:>6.2f}% "
          f"{wf_v3['mdd'].mean()*100:>6.2f}% {wf_v3['calmar'].mean():>5.2f}")

    # ── Walk-forward 6mo rolling (~6 windows; finer stability check) ──
    print("\n" + "=" * 96)
    print("Walk-forward 6-month rolling (3-yr train / 6-mo test) — finer stability check")
    print("=" * 96)
    wf_v2_6m = walkforward(sig, feats, rets, macro, sizer_v2, 756, 126)
    wf_v3_6m = walkforward(sig, feats, rets, macro, sizer_v3, 756, 126)
    print(f"{'Period':<26} {'V2 Sh':>6} {'V2 Ann':>8}  {'V3 Sh':>6} {'V3 Ann':>8}")
    for i in range(min(len(wf_v2_6m), len(wf_v3_6m))):
        a, b = wf_v2_6m.iloc[i], wf_v3_6m.iloc[i]
        print(f"{a['period']:<26} "
              f"{a['sharpe']:>6.2f} {a['ann']*100:>7.2f}%  "
              f"{b['sharpe']:>6.2f} {b['ann']*100:>7.2f}%")
    print(f"{'MEAN':<26} "
          f"{wf_v2_6m['sharpe'].mean():>6.2f} {wf_v2_6m['ann'].mean()*100:>7.2f}%  "
          f"{wf_v3_6m['sharpe'].mean():>6.2f} {wf_v3_6m['ann'].mean()*100:>7.2f}%")
    print(f"{'STD':<26} "
          f"{wf_v2_6m['sharpe'].std():>6.2f} {wf_v2_6m['ann'].std()*100:>7.2f}%  "
          f"{wf_v3_6m['sharpe'].std():>6.2f} {wf_v3_6m['ann'].std()*100:>7.2f}%")
    print(f"{'WORST':<26} "
          f"{wf_v2_6m['sharpe'].min():>6.2f} {wf_v2_6m['ann'].min()*100:>7.2f}%  "
          f"{wf_v3_6m['sharpe'].min():>6.2f} {wf_v3_6m['ann'].min()*100:>7.2f}%")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
