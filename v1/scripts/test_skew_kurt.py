"""
Skew/kurtosis tail-asymmetry sleeve on top of V4N-F base.

Hypothesis: Names with negatively skewed return distributions (long left tail,
short right tail) earn a "lottery-ticket discount" — investors over-pay for
positively skewed names, under-pay for negative-skew names → forward-return
edge by going LONG NEGATIVE-SKEW (the academic skewness anomaly).

Counter-hypothesis: Within a trending universe, *positive* skew + low kurtosis
identifies clean uptrends (consistent up-days) — going LONG POSITIVE-SKEW may
also work as a quality/momentum filter.

Both signs tested.

Signals:
  * skew_w  — rolling skewness of returns (window 21/63/126)
  * kurt_w  — rolling kurtosis (excess) — high kurt = fat tails (avoid?)
  * downside_dev — std of negative returns only (lower = smoother climb)
  * upside_capture — sum positive returns / sum |returns|

Three-phase test (P1 standalone, P2 shrink+renorm, P3 dollar-additive).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v1.portfolio.portfolio import (
    top_n_adx_momt_ac_sizes,
    diversifier_sleeve_overlay,
    defensive_tilt_overlay,
    profit_take_overlay,
    cond_vol_carry_overlay,
    portfolio_returns,
    _get_macro,
    SIGNAL_DIR, FEATURE_DIR, CAPITAL,
)
from v1.pipeline.data_pipeline import TICKERS, TICKER_LIST
from v1.pipeline.signal_generation import ATR_PARTIAL_REMAIN
from v1.pipeline.backtester import sharpe_ratio, max_drawdown

OOS_WARMUP     = 756
RISK_PER_TRADE = 0.005


def metrics(r: pd.Series) -> dict:
    if len(r) < 30:
        return {}
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    mdd = max_drawdown((1 + r).cumprod())
    return dict(
        sharpe=sharpe_ratio(r), ann=ann, mdd=mdd,
        calmar=ann / abs(mdd) if mdd < 0 else float("nan"),
        vol=r.std() * np.sqrt(252),
    )


def load_inputs():
    multi = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    feats: dict = {}
    rets  = pd.DataFrame()
    for t in TICKER_LIST:
        fp = FEATURE_DIR / f"{t}.parquet"
        if not fp.exists():
            continue
        f = pd.read_parquet(fp)
        feats[t] = f
        rets[t]  = f["log_return"]
    rets = rets.dropna()
    sig  = multi.reindex(rets.index).fillna(0.0)
    hs_path = SIGNAL_DIR / "half_size.parquet"
    if hs_path.exists():
        hs = pd.read_parquet(hs_path).reindex(rets.index).fillna(False)
        hs = hs.reindex(columns=sig.columns, fill_value=False)
        sig = sig * np.where(hs, ATR_PARTIAL_REMAIN, 1.0)
    return sig, feats, rets, _get_macro()


def build_v4nf_base(sig, feats, rets):
    base = top_n_adx_momt_ac_sizes(
        sig, feats, rets, CAPITAL,
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )
    base = defensive_tilt_overlay(base, sig, _get_macro(), CAPITAL)
    base = diversifier_sleeve_overlay(
        base, CAPITAL,
        sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
        sleeve_pct=0.12,
    )
    base = profit_take_overlay(base, rets, lookback=10, sigma_thresh=1.5,
                                  scale=0.7, max_gross=1.0, capital=CAPITAL)
    base = cond_vol_carry_overlay(base, _get_macro(),
                                     fear_z=1.5, roc_days=5, fear_mult=0.5)
    return base


def build_signal_panel(rets: pd.DataFrame, signal_name: str,
                        window: int = 63) -> pd.DataFrame:
    out = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for t in rets.columns:
        r = rets[t]
        if signal_name == "skew":
            out[t] = r.rolling(window).skew()
        elif signal_name == "kurt":
            out[t] = r.rolling(window).kurt()
        elif signal_name == "downside_dev":
            neg = r.where(r < 0, 0.0)
            out[t] = neg.rolling(window).std()
        elif signal_name == "upside_capture":
            pos = r.where(r > 0, 0.0).rolling(window).sum()
            absr = r.abs().rolling(window).sum().replace(0, np.nan)
            out[t] = pos / absr
        else:
            raise ValueError(signal_name)
    return out


def _atr_dollar_size(feat, idx, capital, cap_per_name=0.05):
    atr = feat["atr_14"].reindex(idx).ffill()
    cl  = feat["Close"].reindex(idx).ffill().replace(0, np.nan).ffill()
    sz  = (capital * RISK_PER_TRADE / atr.replace(0, np.nan) * cl).fillna(0.0)
    return sz.clip(upper=capital * cap_per_name)


def tail_sleeve(rets: pd.DataFrame, feats: dict, panel: pd.DataFrame,
                  candidates: list, sign: int = +1,
                  top_k: int = 5, sleeve_pct: float = 0.10,
                  cap_per_name: float = 0.05,
                  sizing: str = "atr") -> pd.DataFrame:
    """sign=+1: long highest-rank, sign=-1: long lowest-rank (negative-skew anomaly)."""
    cands = [t for t in candidates if t in panel.columns]
    sub = panel[cands] * sign
    ranks = sub.rank(axis=1, ascending=False, method="first")
    long_mask = ranks <= top_k

    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    for t in cands:
        if sizing == "atr":
            sz = _atr_dollar_size(feats[t], rets.index, CAPITAL, cap_per_name)
        else:
            sz = pd.Series(CAPITAL * cap_per_name, index=rets.index)
        out[t] = (sz * long_mask[t].astype(float)).reindex(rets.index).fillna(0.0)

    sleeve_gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = sleeve_pct * CAPITAL
    scale  = (target / sleeve_gross).clip(upper=1.0).fillna(0.0)
    return out.multiply(scale, axis=0)


def _layer_long_sleeve(longs, sleeve, sleeve_pct, capital=CAPITAL):
    out = longs * (1.0 - sleeve_pct)
    out = out.add(sleeve, fill_value=0.0)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


def _layer_dollar_additive(longs, sleeve, max_gross_x=1.10, capital=CAPITAL):
    out = longs.add(sleeve, fill_value=0.0)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    ceil = max_gross_x * capital
    cap_scale = (ceil / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


def report(name, sizes, rets, base_dd=None, base_ann=None, base_pr=None):
    pr = portfolio_returns(sizes, rets).dropna()
    pr_oos = pr.iloc[OOS_WARMUP:] if len(pr) > OOS_WARMUP else pr
    m = metrics(pr_oos)
    if not m:
        print(f"{name:<70} insufficient data")
        return None
    flag = ""
    if base_dd  is not None and m["mdd"] >= base_dd:  flag += " *DD"
    if base_ann is not None and m["ann"] >= base_ann: flag += " *Ann"
    corr = ""
    if base_pr is not None:
        common = pr_oos.index.intersection(base_pr.index)
        if len(common) > 30:
            c = pr_oos.reindex(common).corr(base_pr.reindex(common))
            corr = f"  ρ={c:+.2f}"
    print(f"{name:<70} {m['sharpe']:>5.2f} {m['ann']*100:>6.2f}% "
          f"{m['mdd']*100:>6.2f}% {m['calmar']:>5.2f} {m['vol']*100:>5.1f}%{flag}{corr}")
    return m | {"name": name, "pr": pr_oos}


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in {time.time()-t0:.1f}s")

    stocks  = [t for t, c in TICKERS.items() if c == "stock"      and t in rets.columns]
    sectors = [t for t, c in TICKERS.items() if c == "sector_etf" and t in rets.columns]
    full    = stocks + sectors

    print(f"Universe: {len(stocks)} stocks, {len(sectors)} sectors")

    print("\nBuilding signal panels...")
    panels = {}
    for sname in ("skew", "kurt", "downside_dev", "upside_capture"):
        for w in (21, 63, 126):
            p = build_signal_panel(rets, sname, window=w)
            panels[f"{sname}_w{w}"] = p
    print(f"  {len(panels)} panels built")

    base = build_v4nf_base(sig, feats, rets)
    base_pr = portfolio_returns(base, rets).dropna()
    base_pr_oos = base_pr.iloc[OOS_WARMUP:] if len(base_pr) > OOS_WARMUP else base_pr

    print(f"\n{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    base_m = report("V4N base (no tail sleeve)", base, rets)
    if base_m is None:
        return
    base_dd, base_ann, base_sh = base_m["mdd"], base_m["ann"], base_m["sharpe"]
    results = [base_m]

    # ── Phase 1: STANDALONE ──────────────────────────────────────────────────
    print("\n--- Phase 1. Tail sleeve STANDALONE (both signs) ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    p1_keys = [f"{s}_w63" for s in ("skew", "kurt", "downside_dev", "upside_capture")]
    for univ_name, univ in [("stocks", stocks), ("full", full)]:
        for pkey in p1_keys:
            for sn in (+1, -1):
                for k in (3, 5, 8):
                    sl = tail_sleeve(rets, feats, panels[pkey], univ,
                                       sign=sn, top_k=k,
                                       sleeve_pct=1.0, cap_per_name=0.20,
                                       sizing="atr")
                    sn_tag = "+1" if sn == +1 else "-1"
                    m = report(f"P1: {univ_name} {pkey} sgn{sn_tag} k{k}",
                                sl, rets)
                    if m: results.append(m | {"_phase": "p1"})

    # ── Phase 2: LAYERED shrink+renorm ──────────────────────────────────────
    print("\n--- Phase 2. Layered onto V4N base (shrink + 1.0× cap) ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    for univ_name, univ in [("stocks", stocks), ("full", full)]:
        for pkey, panel in panels.items():
            for sn in (+1, -1):
                for k in (3, 5):
                    for sp in (0.05, 0.08, 0.12):
                        sl = tail_sleeve(rets, feats, panel, univ,
                                           sign=sn, top_k=k,
                                           sleeve_pct=sp, cap_per_name=0.04,
                                           sizing="atr")
                        combined = _layer_long_sleeve(base, sl, sp)
                        sn_tag = "+1" if sn == +1 else "-1"
                        m = report(f"P2: {univ_name} {pkey} sgn{sn_tag} k{k} sp{sp}",
                                    combined, rets, base_dd, base_ann, base_pr_oos)
                        if m: results.append(m | {"_phase": "p2"})

    # ── Phase 3: DOLLAR-ADDITIVE ────────────────────────────────────────────
    print("\n--- Phase 3. Dollar-additive layering ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    for univ_name, univ in [("stocks", stocks), ("full", full)]:
        for pkey, panel in panels.items():
            for sn in (+1, -1):
                for k in (3, 5):
                    for sp in (0.05, 0.08):
                        for mg in (1.05, 1.10):
                            sl = tail_sleeve(rets, feats, panel, univ,
                                               sign=sn, top_k=k,
                                               sleeve_pct=sp, cap_per_name=0.04,
                                               sizing="atr")
                            combined = _layer_dollar_additive(base, sl, max_gross_x=mg)
                            sn_tag = "+1" if sn == +1 else "-1"
                            m = report(f"P3: {univ_name} {pkey} sgn{sn_tag} k{k} sp{sp} g{mg}",
                                        combined, rets, base_dd, base_ann, base_pr_oos)
                            if m: results.append(m | {"_phase": "p3"})

    # ── Leaderboards ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"V4N base: Sh {base_sh:.2f}  Ann {base_ann*100:.2f}%  "
          f"DD {base_dd*100:.2f}%  Cal {base_m['calmar']:.2f}")
    print('='*100)

    p1 = [r for r in results if r.get("_phase") == "p1"]
    p2 = [r for r in results if r.get("_phase") == "p2"]
    p3 = [r for r in results if r.get("_phase") == "p3"]

    print("\nMost ORTHOGONAL P1 sleeves (lowest |corr|):")
    for r in p1:
        common = r["pr"].index.intersection(base_pr_oos.index)
        r["_corr"] = (r["pr"].reindex(common).corr(base_pr_oos.reindex(common))
                      if len(common) > 30 else float("nan"))
    by_o = sorted([r for r in p1 if not np.isnan(r["_corr"])], key=lambda x: abs(x["_corr"]))
    for r in by_o[:10]:
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={r['_corr']:+.3f}")

    print("\nTop standalone P1 by Sharpe:")
    for r in sorted(p1, key=lambda x: -x["sharpe"])[:10]:
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={r['_corr']:+.3f}")

    print("\nTop 15 LAYERED (P2) by Sharpe:")
    for r in sorted(p2, key=lambda x: -x["sharpe"])[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.2f}")

    print("\nP2 variants beating BOTH baseline AnnRet AND DD:")
    pareto2 = [r for r in p2 if r["ann"] >= base_ann and r["mdd"] >= base_dd]
    pareto2.sort(key=lambda x: -x["sharpe"])
    if not pareto2:
        print("  (none)")
    for r in pareto2[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.2f}")

    print("\nTop 15 DOLLAR-ADDITIVE (P3) by Sharpe:")
    for r in sorted(p3, key=lambda x: -x["sharpe"])[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.2f}")

    print("\nP3 variants beating BOTH baseline AnnRet AND DD:")
    pareto3 = [r for r in p3 if r["ann"] >= base_ann and r["mdd"] >= base_dd]
    pareto3.sort(key=lambda x: -x["sharpe"])
    if not pareto3:
        print("  (none)")
    for r in pareto3[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.2f}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
