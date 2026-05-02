"""
V4-noML candidates — combine the three non-ML winners into stacked candidates,
then walk-forward validate.

Single-overlay winners on V3 (full-period OOS):
  - Profit-taking (lb=10 sig=1.5 scl=0.7)         : Sh 2.48 (+0.15), Ann 19.25%, DD -5.69%
  - Backwardation overlay (mul=0.5)                : Sh 2.42, Ann 18.07%, DD -4.08%, Cal 4.42
  - Conditional vol-carry (fz=1.5 mul=0.5 roc=5d)  : Sh 2.39, Ann 19.27%, DD -4.57%, Cal 4.22

Stacked candidates:
  V4N-A  =  V3 + PT                             (Sharpe-max, minimal DD change)
  V4N-B  =  V3 + PT + Cond-VC                   (Sharpe + balanced DD)
  V4N-C  =  V3 + PT + Backwardation             (Sharpe + max DD relief)
  V4N-D  =  V3 + PT + Cond-VC + Backwardation   (full stack)

Validation:
  - Full-period OOS metrics
  - Calendar-year breakdown
  - 1-yr walk-forward (3 windows, true OOS)
  - 6-month walk-forward stability (~6 windows)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import v1.portfolio.portfolio as _pp
from v1.portfolio.portfolio import (
    top_n_adx_momt_ac_sizes,
    diversifier_sleeve_overlay,
    portfolio_returns,
    defensive_tilt_overlay,
    _get_macro,
    SIGNAL_DIR, FEATURE_DIR, CAPITAL,
)
from v1.pipeline.data_pipeline import TICKER_LIST
from v1.pipeline.signal_generation import ATR_PARTIAL_REMAIN
from v1.pipeline.backtester import sharpe_ratio, max_drawdown

OOS_WARMUP = 756


def metrics(r):
    if len(r) < 30:
        return dict(sharpe=float("nan"), ann=float("nan"), mdd=float("nan"),
                    calmar=float("nan"), vol=float("nan"), n=len(r))
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    mdd = max_drawdown((1 + r).cumprod())
    return dict(
        sharpe=sharpe_ratio(r), ann=ann, mdd=mdd,
        calmar=ann / abs(mdd) if mdd < 0 else float("nan"),
        vol=r.std() * np.sqrt(252), n=len(r),
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


def make_v3_sizes(sig, feats, rets, macro):
    base_kw = dict(
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )
    s = top_n_adx_momt_ac_sizes(sig, feats, rets, CAPITAL, **base_kw)
    s = defensive_tilt_overlay(s, sig, macro, CAPITAL)
    s = diversifier_sleeve_overlay(
        s, CAPITAL,
        sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
        sleeve_pct=0.12,
    )
    return s


def zero_lev_clip(sizes, capital=CAPITAL):
    g = sizes.abs().sum(axis=1).replace(0, np.nan)
    sc = (capital / g).clip(upper=1.0).fillna(1.0)
    return sizes.multiply(sc, axis=0)


def profit_take(sizes, rets, lookback=10, sigma_thresh=1.5, scale=0.7):
    out = sizes.copy()
    for t in sizes.columns:
        if t not in rets.columns:
            continue
        cum = rets[t].rolling(lookback).sum()
        std = rets[t].rolling(lookback).std() * np.sqrt(lookback)
        z = (cum / std.replace(0, np.nan)).reindex(out.index).fillna(0)
        bad = (z >= sigma_thresh) & (out[t] > 0)
        col = out[t].values.astype(float)
        col = np.where(bad.values, col * scale, col)
        out[t] = col
    return zero_lev_clip(out)


def cond_vol_carry(sizes, macro, fear_z=1.5, roc_days=5, fear_mult=0.5):
    if "vix_zscore" not in macro.columns or "vix" not in macro.columns:
        return sizes
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    roc = macro["vix"].reindex(sizes.index).ffill().diff(roc_days)
    mult = pd.Series(1.0, index=sizes.index)
    mask = (z >= fear_z) & (roc > 0)
    mult[mask] = fear_mult
    return sizes.multiply(mult, axis=0)


def backwardation_overlay(sizes, macro, mult=0.5):
    if "vol_backwardation" not in macro.columns:
        return sizes
    flag = macro["vol_backwardation"].reindex(sizes.index).ffill().fillna(0).astype(bool)
    m = pd.Series(1.0, index=sizes.index)
    m[flag.values] = mult
    return sizes.multiply(m, axis=0)


# ---- V4-noML candidate sizers ----
def make_v4n_a(sig, feats, rets, macro):
    s = make_v3_sizes(sig, feats, rets, macro)
    return profit_take(s, rets)


def make_v4n_b(sig, feats, rets, macro):
    s = make_v3_sizes(sig, feats, rets, macro)
    s = profit_take(s, rets)
    s = cond_vol_carry(s, macro)
    return s


def make_v4n_c(sig, feats, rets, macro):
    s = make_v3_sizes(sig, feats, rets, macro)
    s = profit_take(s, rets)
    s = backwardation_overlay(s, macro)
    return s


def make_v4n_d(sig, feats, rets, macro):
    s = make_v3_sizes(sig, feats, rets, macro)
    s = profit_take(s, rets)
    s = cond_vol_carry(s, macro)
    s = backwardation_overlay(s, macro)
    return s


def walkforward_eval(sig, feats, rets, macro, sizer, train_days, test_days):
    out = []
    start = train_days
    while start + test_days <= len(sig):
        ctx_start = max(0, start - train_days)
        s = sig.iloc[ctx_start: start + test_days]
        r = rets.iloc[ctx_start: start + test_days]
        sizes = sizer(s, feats, r, macro)
        period = portfolio_returns(sizes.iloc[-test_days:], r.iloc[-test_days:]).dropna()
        m = metrics(period)
        m["period"] = (f"{sig.index[start].date()}->"
                        f"{sig.index[start+test_days-1].date()}")
        out.append(m)
        start += test_days
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in {time.time()-t0:.1f}s\n")

    sizers = [
        ("V3 baseline",           make_v3_sizes),
        ("V4N-A: V3 + PT",        make_v4n_a),
        ("V4N-B: V3 + PT + cVC",  make_v4n_b),
        ("V4N-C: V3 + PT + BW",   make_v4n_c),
        ("V4N-D: full stack",     make_v4n_d),
    ]

    # Full period OOS
    print("=" * 92)
    print("FULL-PERIOD OOS METRICS (post 756d warmup)")
    print("=" * 92)
    print(f"{'Variant':<32} {'Sh':>6} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    sizes_cache = {}
    for name, sz in sizers:
        s = sz(sig, feats, rets, macro)
        sizes_cache[name] = s
        pr = portfolio_returns(s, rets).dropna()
        m = metrics(pr.iloc[OOS_WARMUP:])
        print(f"{name:<32} {m['sharpe']:>6.2f} "
              f"{m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # Calendar-year breakdown
    print("\n" + "=" * 92)
    print("CALENDAR-YEAR BREAKDOWN (Sharpe / Ann / DD)")
    print("=" * 92)
    pr_cache = {n: portfolio_returns(s, rets).dropna() for n, s in sizes_cache.items()}
    print(f"{'Year':<6} " + "".join(f"{n[:14]:<22}" for n, _ in sizers))
    print(f"{'':6} " + "".join(f"{'Sh':>5} {'Ann':>7} {'DD':>7}  " for _ in sizers))
    for yr in sorted(set(pr_cache["V3 baseline"].index.year)):
        if all(len(pr_cache[n][pr_cache[n].index.year == yr]) >= 30 for n, _ in sizers):
            row = f"{yr:<6} "
            for name, _ in sizers:
                pr = pr_cache[name][pr_cache[name].index.year == yr]
                m = metrics(pr)
                row += (f"{m['sharpe']:>5.2f} {m['ann']*100:>6.2f}% "
                        f"{m['mdd']*100:>6.2f}%  ")
            print(row)

    # 1-yr walk-forward
    print("\n" + "=" * 100)
    print("1-YEAR WALK-FORWARD (3-yr train / 1-yr test)")
    print("=" * 100)
    wf = {n: walkforward_eval(sig, feats, rets, macro, sz, 756, 252)
          for n, sz in sizers}
    print(f"{'Period':<26} " + "".join(f"{n[:14]:<22}" for n, _ in sizers))
    print(f"{'':26} " + "".join(f"{'Sh':>5} {'Ann':>7} {'DD':>7}  " for _ in sizers))
    n_pers = min(len(w) for w in wf.values())
    for i in range(n_pers):
        row = f"{wf['V3 baseline'].iloc[i]['period']:<26} "
        for name, _ in sizers:
            r = wf[name].iloc[i]
            row += f"{r['sharpe']:>5.2f} {r['ann']*100:>6.2f}% {r['mdd']*100:>6.2f}%  "
        print(row)
    row = f"{'MEAN':<26} "
    for name, _ in sizers:
        w = wf[name]
        row += (f"{w['sharpe'].mean():>5.2f} {w['ann'].mean()*100:>6.2f}% "
                f"{w['mdd'].mean()*100:>6.2f}%  ")
    print(row)
    row = f"{'WORST':<26} "
    for name, _ in sizers:
        w = wf[name]
        row += (f"{w['sharpe'].min():>5.2f} {w['ann'].min()*100:>6.2f}% "
                f"{w['mdd'].min()*100:>6.2f}%  ")
    print(row)

    # 6-month walk-forward stability
    print("\n" + "=" * 92)
    print("6-MONTH WALK-FORWARD STABILITY (3-yr train / 6-mo test)")
    print("=" * 92)
    wf6 = {n: walkforward_eval(sig, feats, rets, macro, sz, 756, 126)
           for n, sz in sizers}
    print(f"{'Stat':<8} " + "".join(f"{'Sh':>6}|{'Ann':>7} " for _ in sizers))
    for stat, fn in [("MEAN", "mean"), ("STD", "std"),
                     ("WORST", "min"), ("BEST", "max")]:
        row = f"{stat:<8} "
        for name, _ in sizers:
            w = wf6[name]
            sh = getattr(w["sharpe"], fn)()
            an = getattr(w["ann"], fn)() * 100
            row += f"{sh:>6.2f}|{an:>6.2f}% "
        print(row)

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
