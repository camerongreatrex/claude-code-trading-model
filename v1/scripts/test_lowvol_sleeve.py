"""
Low-volatility anomaly sleeve on top of V4N-F base.

Hypothesis: long the LOWEST-volatility decile of the candidate universe.  This
is the most-documented academic anomaly outside momentum and is structurally
orthogonal to V4N's high-beta high-momentum trend picks: low-vol names tend
to be defensive, mature, low-cyclicality businesses (utilities, staples,
healthcare) — a completely different character profile.

Mechanic:
  * Each day, rank candidates by trailing realised vol (std of log_return over
    VOL_WINDOW days).
  * Long top_k LOWEST-vol names (= lowest decile if k = N/10).
  * Equal-dollar OR inverse-vol-weighted sizing, scaled to sleeve_pct.
  * Two layer modes: P2 shrink+renorm 1.0× cap, P3 dollar-additive 1.05/1.10×.

Note: low-vol effect is documented to be strongest in stocks; less so in
sectors and bonds.  Run all three universes for transparency.
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

OOS_WARMUP = 756


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


def realized_vol(rets: pd.DataFrame, window: int) -> pd.DataFrame:
    return rets.rolling(window).std() * np.sqrt(252)


def lowvol_sleeve(rets: pd.DataFrame, vol: pd.DataFrame, candidates: list,
                    top_k: int = 8,
                    sleeve_pct: float = 0.10,
                    weighting: str = "eq",        # "eq" | "inv_vol"
                    cap_per_name: float = 0.04,
                    min_history_days: int = 252) -> pd.DataFrame:
    """
    Long-only lowest-vol cross-sectional sleeve.
    Each day rank `candidates` by trailing vol (low first), pick top_k.
    """
    cands = [t for t in candidates if t in rets.columns]
    v = vol[cands].copy()
    # require enough history per name to avoid early-life illusory low vol
    enough = (rets[cands].rolling(min_history_days).count() >= min_history_days)
    v = v.where(enough)

    ranks = v.rank(axis=1, ascending=True, method="first")
    keep  = ranks <= top_k

    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    if weighting == "inv_vol":
        inv = (1.0 / v).where(keep, 0.0).fillna(0.0)
        norm = inv.sum(axis=1).replace(0, np.nan)
        wts = inv.div(norm, axis=0).fillna(0.0)
    else:  # equal
        kept = keep.astype(float)
        norm = kept.sum(axis=1).replace(0, np.nan)
        wts  = kept.div(norm, axis=0).fillna(0.0)

    sizes = wts * (sleeve_pct * CAPITAL)
    cap   = CAPITAL * cap_per_name
    sizes = sizes.clip(upper=cap)
    # rescale after cap so total == sleeve_pct again (cap usually non-binding)
    g = sizes.sum(axis=1).replace(0, np.nan)
    target = sleeve_pct * CAPITAL
    scale  = (target / g).clip(upper=1.0).fillna(0.0)
    out.loc[:, sizes.columns] = sizes.multiply(scale, axis=0)
    return out


def _layer_long_sleeve(longs: pd.DataFrame, sleeve: pd.DataFrame,
                         sleeve_pct: float, capital: float = CAPITAL) -> pd.DataFrame:
    out = longs * (1.0 - sleeve_pct)
    out = out.add(sleeve, fill_value=0.0)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


def _layer_dollar_additive(longs: pd.DataFrame, sleeve: pd.DataFrame,
                             max_gross_x: float = 1.10,
                             capital: float = CAPITAL) -> pd.DataFrame:
    out   = longs.add(sleeve, fill_value=0.0)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    ceil  = max_gross_x * capital
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

    stocks  = [t for t, c in TICKERS.items() if c == "stock"        and t in rets.columns]
    sectors = [t for t, c in TICKERS.items() if c == "sector_etf"   and t in rets.columns]
    indexes = [t for t, c in TICKERS.items() if c == "equity_index" and t in rets.columns]
    full    = stocks + sectors + indexes
    print(f"Universe: {len(stocks)} stocks, {len(sectors)} sectors, {len(indexes)} indexes")

    # Pre-compute realised-vol panels
    vol_60  = realized_vol(rets, 60)
    vol_90  = realized_vol(rets, 90)
    vol_252 = realized_vol(rets, 252)
    vol_panels = [("v60", vol_60), ("v90", vol_90), ("v252", vol_252)]

    base = build_v4nf_base(sig, feats, rets)
    base_pr = portfolio_returns(base, rets).dropna()
    base_pr_oos = base_pr.iloc[OOS_WARMUP:] if len(base_pr) > OOS_WARMUP else base_pr

    print(f"\n{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    base_m = report("V4N base (no low-vol sleeve)", base, rets)
    if base_m is None:
        print("Base failed; aborting.")
        return
    base_dd, base_ann, base_sh = base_m["mdd"], base_m["ann"], base_m["sharpe"]
    results = [base_m]

    # ── Phase 1: STANDALONE — gauge raw alpha + orthogonality ─────────────────
    print("\n--- Phase 1. Low-vol sleeve STANDALONE (sleeve_pct=1.0 so it IS the portfolio) ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    for univ_name, univ in [("full", full), ("stocks", stocks), ("sectors", sectors)]:
        for vol_name, vol in vol_panels:
            for k in (5, 8, 12):
                if k > len(univ):
                    continue
                for w in ("eq", "inv_vol"):
                    sl = lowvol_sleeve(rets, vol, univ,
                                          top_k=k, sleeve_pct=1.0,
                                          weighting=w, cap_per_name=0.20)
                    m = report(f"P1: {univ_name} {vol_name} k{k} {w}", sl, rets)
                    if m: results.append(m | {"_phase": "p1"})

    # ── Phase 2: LAYERED with shrink+renorm ──────────────────────────────────
    print("\n--- Phase 2. Layered onto V4N base (shrink + 1.0× cap) ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    grid2 = []
    for univ_name, univ in [("full", full), ("stocks", stocks)]:
        for vol_name, vol in vol_panels:
            for k in (5, 8, 12):
                if k > len(univ):
                    continue
                for sp in (0.05, 0.08, 0.12):
                    for w in ("eq", "inv_vol"):
                        grid2.append((univ_name, univ, vol_name, vol, k, sp, w))

    for (uname, univ, vname, vol, k, sp, w) in grid2:
        sl = lowvol_sleeve(rets, vol, univ,
                              top_k=k, sleeve_pct=sp,
                              weighting=w, cap_per_name=0.04)
        combined = _layer_long_sleeve(base, sl, sp)
        m = report(f"P2: {uname} {vname} k{k} sp{sp} {w}",
                    combined, rets, base_dd, base_ann, base_pr_oos)
        if m: results.append(m | {"_phase": "p2"})

    # ── Phase 3: DOLLAR-ADDITIVE ──────────────────────────────────────────────
    print("\n--- Phase 3. Dollar-additive layering ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    grid3 = []
    for univ_name, univ in [("full", full), ("stocks", stocks)]:
        for vol_name, vol in vol_panels:
            for k in (5, 8, 12):
                if k > len(univ):
                    continue
                for sp in (0.05, 0.08, 0.12):
                    for mg in (1.05, 1.10):
                        grid3.append((univ_name, univ, vol_name, vol, k, sp, mg))

    for (uname, univ, vname, vol, k, sp, mg) in grid3:
        sl = lowvol_sleeve(rets, vol, univ,
                              top_k=k, sleeve_pct=sp,
                              weighting="eq", cap_per_name=0.04)
        combined = _layer_dollar_additive(base, sl, max_gross_x=mg)
        m = report(f"P3: {uname} {vname} k{k} sp{sp} g{mg}",
                    combined, rets, base_dd, base_ann, base_pr_oos)
        if m: results.append(m | {"_phase": "p3"})

    # ── Leaderboards ──────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"V4N base: Sh {base_sh:.2f}  Ann {base_ann*100:.2f}%  "
          f"DD {base_dd*100:.2f}%  Cal {base_m['calmar']:.2f}")
    print('='*100)

    p1 = [r for r in results if r.get("_phase") == "p1"]
    p2 = [r for r in results if r.get("_phase") == "p2"]
    p3 = [r for r in results if r.get("_phase") == "p3"]

    print("\nMost ORTHOGONAL standalone P1 sleeves (lowest |corr| to V4N):")
    for r in p1:
        common = r["pr"].index.intersection(base_pr_oos.index)
        r["_corr"] = (r["pr"].reindex(common).corr(base_pr_oos.reindex(common))
                      if len(common) > 30 else float("nan"))
    by_ortho = sorted([r for r in p1 if not np.isnan(r["_corr"])],
                       key=lambda x: abs(x["_corr"]))
    for r in by_ortho[:12]:
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={r['_corr']:+.3f}")

    print("\nTop standalone P1 by Sharpe (raw alpha quality):")
    for r in sorted(p1, key=lambda x: -x["sharpe"])[:10]:
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={r['_corr']:+.3f}")

    print("\nTop 15 LAYERED (P2, shrink+renorm) by Sharpe:")
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
