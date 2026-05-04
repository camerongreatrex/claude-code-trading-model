"""
V4 candidate construction & walk-forward validation.
V4A = V3 + ML filter (HistGBM P(5d+)<0.40 skip).
V4B = V3 + ML filter + vol-carry (calm=0->1.0, fear=+2->0.5).
Reports full-period OOS, calendar-year, 1-yr walk-forward (3 windows), 6-mo stability.
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
from sklearn.ensemble import HistGradientBoostingClassifier

OOS_WARMUP = 756

PER_TICKER_FEATS = [
    "mom_5", "mom_20", "mom_60", "rsi_14", "adx", "plus_di", "minus_di",
    "macd_hist", "zscore_20", "zscore_60", "bb_pct_b", "volume_zscore",
]
MACRO_FEATS = ["vix_zscore", "yield_curve", "vix_term_ratio", "curve_momentum"]


def metrics(r):
    if len(r) < 30:
        return dict(sharpe=float("nan"), ann=float("nan"), mdd=float("nan"),
                    calmar=float("nan"), vol=float("nan"), n=len(r))
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    mdd = max_drawdown((1 + r).cumprod())
    return dict(
        sharpe=sharpe_ratio(r), ann=ann, mdd=mdd,
        calmar=ann / abs(mdd) if mdd < 0 else float("nan"),
        vol=r.std() * np.sqrt(252),
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


def vol_carry_overlay(sizes, macro, calm_z=0.0, fear_z=2.0,
                      calm_mult=1.0, fear_mult=0.5):
    if "vix_zscore" not in macro.columns:
        return sizes
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    span = max(fear_z - calm_z, 1e-6)
    raw = calm_mult + (fear_mult - calm_mult) * ((z - calm_z) / span).clip(0, 1)
    mult = raw.clip(lower=fear_mult, upper=calm_mult)
    return sizes.multiply(mult, axis=0)


def build_position_panel(sizes_v3, feats, macro, rets, fwd=5):
    macro_aligned = macro.reindex(rets.index).ffill()
    macro_view = macro_aligned[[c for c in MACRO_FEATS if c in macro_aligned.columns]]
    parts = []
    for t in sizes_v3.columns:
        if t not in feats or t not in rets.columns:
            continue
        df = feats[t]
        cols = [c for c in PER_TICKER_FEATS if c in df.columns]
        if not cols:
            continue
        x = df[cols].copy()
        if "atr_14" in df.columns and "Close" in df.columns:
            x["atr_pct"] = (df["atr_14"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
        x = x.shift(1)
        x = x.join(macro_view, how="left")
        held = sizes_v3[t].reindex(x.index) > 0
        x = x[held]
        if len(x) == 0:
            continue
        fwd_ret = rets[t].rolling(fwd).sum().shift(-fwd)
        x["target"] = (fwd_ret.reindex(x.index) > 0).astype(int)
        x["fwd_ret"] = fwd_ret.reindex(x.index)
        x["ticker"] = t
        x = x.dropna(subset=["target"])
        parts.append(x)
    panel = pd.concat(parts, axis=0).sort_index()
    panel.index.name = "date"
    return panel


def walkforward_predict(panel, train_days=504, retrain_freq=63, min_train=252):
    feature_cols = [c for c in panel.columns
                     if c not in ("target", "fwd_ret", "ticker")]
    panel = panel.sort_index()
    dates = panel.index.unique().sort_values()
    preds = pd.Series(np.nan, index=panel.index, dtype=float)
    last_train_idx = -10**9
    model = None
    for i, d in enumerate(dates):
        if i < min_train:
            continue
        if i - last_train_idx >= retrain_freq or model is None:
            ts_pos = max(0, i - train_days)
            train_dates = dates[ts_pos: i]
            train = panel.loc[panel.index.isin(train_dates)]
            X = train[feature_cols].values
            y = train["target"].values
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            if mask.sum() < 100 or len(np.unique(y[mask])) < 2:
                continue
            model = HistGradientBoostingClassifier(
                max_depth=3, max_iter=80, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=1.0,
                random_state=42,
            )
            model.fit(X[mask], y[mask])
            last_train_idx = i
        today = panel.loc[panel.index == d]
        Xt = today[feature_cols].values
        mask = ~np.isnan(Xt).any(axis=1)
        if mask.sum() == 0:
            continue
        p = np.full(len(today), np.nan)
        p[mask] = model.predict_proba(Xt[mask])[:, 1]
        preds.loc[panel.index == d] = p
    return preds


def apply_ml_filter(sizes_v3, panel, preds, threshold=0.40):
    out = sizes_v3.copy()
    panel = panel.copy()
    panel["pred"] = preds.values
    flagged = panel[panel["pred"].notna() & (panel["pred"] < threshold)]
    for d, row in flagged.iterrows():
        t = row["ticker"]
        if t in out.columns:
            out.loc[d, t] = 0.0
    return out


def make_v4a(sig, feats, rets, macro):
    """V3 + ML filter (skip thr=0.40)."""
    sv3 = make_v3_sizes(sig, feats, rets, macro)
    panel = build_position_panel(sv3, feats, macro, rets, fwd=5)
    preds = walkforward_predict(panel)
    return apply_ml_filter(sv3, panel, preds, threshold=0.40)


def make_v4b(sig, feats, rets, macro):
    """V3 + ML filter + vol-carry."""
    s = make_v4a(sig, feats, rets, macro)
    s = vol_carry_overlay(s, macro, calm_z=0.0, fear_z=2.0,
                          calm_mult=1.0, fear_mult=0.5)
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

    # Full-period OOS
    sv3 = make_v3_sizes(sig, feats, rets, macro)
    sv4a = make_v4a(sig, feats, rets, macro)
    sv4b = make_v4b(sig, feats, rets, macro)
    pr_v3  = portfolio_returns(sv3, rets).dropna()
    pr_v4a = portfolio_returns(sv4a, rets).dropna()
    pr_v4b = portfolio_returns(sv4b, rets).dropna()
    m_v3  = metrics(pr_v3.iloc[OOS_WARMUP:])
    m_v4a = metrics(pr_v4a.iloc[OOS_WARMUP:])
    m_v4b = metrics(pr_v4b.iloc[OOS_WARMUP:])

    print("=" * 88)
    print("FULL-PERIOD OOS METRICS (post 756d warmup)")
    print("=" * 88)
    print(f"{'Variant':<48} {'Sh':>6} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    for name, m in [("V3 baseline", m_v3),
                    ("V4A = V3 + ML filter", m_v4a),
                    ("V4B = V3 + ML filter + vol-carry", m_v4b)]:
        print(f"{name:<48} {m['sharpe']:>6.2f} "
              f"{m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # Calendar-year breakdown
    print("\n" + "=" * 88)
    print("CALENDAR-YEAR BREAKDOWN")
    print("=" * 88)
    print(f"{'Year':<6} {'V3':<28} {'V4A (ML)':<28} {'V4B (ML+VC)':<28}")
    print(f"{'':6} {'Sh':>6} {'Ann':>7} {'DD':>7}  "
          f"{'Sh':>6} {'Ann':>7} {'DD':>7}  "
          f"{'Sh':>6} {'Ann':>7} {'DD':>7}")
    for yr in sorted(set(pr_v3.index.year)):
        s3 = pr_v3[pr_v3.index.year == yr]
        sa = pr_v4a[pr_v4a.index.year == yr]
        sb = pr_v4b[pr_v4b.index.year == yr]
        if min(len(s3), len(sa), len(sb)) < 30:
            continue
        m3, ma, mb = metrics(s3), metrics(sa), metrics(sb)
        print(f"{yr:<6} "
              f"{m3['sharpe']:>6.2f} {m3['ann']*100:>6.2f}% {m3['mdd']*100:>6.2f}%  "
              f"{ma['sharpe']:>6.2f} {ma['ann']*100:>6.2f}% {ma['mdd']*100:>6.2f}%  "
              f"{mb['sharpe']:>6.2f} {mb['ann']*100:>6.2f}% {mb['mdd']*100:>6.2f}%")

    # 1-yr walk-forward (true OOS refit)
    print("\n" + "=" * 88)
    print("1-YEAR WALK-FORWARD (3-yr train / 1-yr test, true OOS refit incl. ML)")
    print("=" * 88)
    print(f"{'Period':<26} {'V3':<22} {'V4A':<22} {'V4B':<22}")
    print(f"{'':26} {'Sh':>6} {'Ann':>7} {'DD':>7}  "
          f"{'Sh':>6} {'Ann':>7} {'DD':>7}  "
          f"{'Sh':>6} {'Ann':>7} {'DD':>7}")
    wf3  = walkforward_eval(sig, feats, rets, macro, make_v3_sizes,  756, 252)
    wfa  = walkforward_eval(sig, feats, rets, macro, make_v4a,       756, 252)
    wfb  = walkforward_eval(sig, feats, rets, macro, make_v4b,       756, 252)
    n = min(len(wf3), len(wfa), len(wfb))
    for i in range(n):
        a, b, c = wf3.iloc[i], wfa.iloc[i], wfb.iloc[i]
        print(f"{a['period']:<26} "
              f"{a['sharpe']:>6.2f} {a['ann']*100:>6.2f}% {a['mdd']*100:>6.2f}%  "
              f"{b['sharpe']:>6.2f} {b['ann']*100:>6.2f}% {b['mdd']*100:>6.2f}%  "
              f"{c['sharpe']:>6.2f} {c['ann']*100:>6.2f}% {c['mdd']*100:>6.2f}%")
    print(f"{'MEAN':<26} "
          f"{wf3['sharpe'].mean():>6.2f} {wf3['ann'].mean()*100:>6.2f}% "
          f"{wf3['mdd'].mean()*100:>6.2f}%  "
          f"{wfa['sharpe'].mean():>6.2f} {wfa['ann'].mean()*100:>6.2f}% "
          f"{wfa['mdd'].mean()*100:>6.2f}%  "
          f"{wfb['sharpe'].mean():>6.2f} {wfb['ann'].mean()*100:>6.2f}% "
          f"{wfb['mdd'].mean()*100:>6.2f}%")
    print(f"{'WORST':<26} "
          f"{wf3['sharpe'].min():>6.2f} {wf3['ann'].min()*100:>6.2f}% "
          f"{wf3['mdd'].min()*100:>6.2f}%  "
          f"{wfa['sharpe'].min():>6.2f} {wfa['ann'].min()*100:>6.2f}% "
          f"{wfa['mdd'].min()*100:>6.2f}%  "
          f"{wfb['sharpe'].min():>6.2f} {wfb['ann'].min()*100:>6.2f}% "
          f"{wfb['mdd'].min()*100:>6.2f}%")

    # 6-month walk-forward stability
    print("\n" + "=" * 88)
    print("6-MONTH WALK-FORWARD STABILITY (3-yr train / 6-mo test)")
    print("=" * 88)
    wf3_6  = walkforward_eval(sig, feats, rets, macro, make_v3_sizes, 756, 126)
    wfa_6  = walkforward_eval(sig, feats, rets, macro, make_v4a,      756, 126)
    wfb_6  = walkforward_eval(sig, feats, rets, macro, make_v4b,      756, 126)
    print(f"{'Stat':<10} {'V3 Sh':>7} {'V4A Sh':>7} {'V4B Sh':>7}  "
          f"{'V3 Ann':>8} {'V4A Ann':>8} {'V4B Ann':>8}")
    for stat, fn in [("MEAN", "mean"), ("STD", "std"),
                     ("WORST", "min"), ("BEST", "max")]:
        f3 = getattr(wf3_6["sharpe"], fn)()
        fa = getattr(wfa_6["sharpe"], fn)()
        fb = getattr(wfb_6["sharpe"], fn)()
        a3 = getattr(wf3_6["ann"], fn)() * 100
        aa = getattr(wfa_6["ann"], fn)() * 100
        ab = getattr(wfb_6["ann"], fn)() * 100
        print(f"{stat:<10} {f3:>7.2f} {fa:>7.2f} {fb:>7.2f}  "
              f"{a3:>7.2f}% {aa:>7.2f}% {ab:>7.2f}%")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
