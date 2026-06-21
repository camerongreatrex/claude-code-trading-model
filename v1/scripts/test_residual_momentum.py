"""
Cross-sectional residual momentum sleeve on top of V4N-F base.

Hypothesis: V4N is heavily beta-loaded (trend follows market direction).  If we
strip market beta from each name's returns and rank what's *left* (the
idiosyncratic component), we get a signal that's structurally orthogonal to
V4N's beta-tilted trend.  Long the top-K idiosyncratic winners, optionally
short the bottom-K losers.

Mechanic:
  * Rolling 60d regression: T.r = alpha + beta * SPY.r + resid.
  * residual_mom[T,t] = sum of resid over trailing LOOKBACK days (default 21).
  * Each day: cross-sectionally rank, long top-K by resid_mom.
  * Equal-dollar (or ATR) sizing, scaled to sleeve_pct of capital.
  * Layer onto V4N base in two flavors:
      P2 = shrink-and-renorm (gross cap 1.0×)  ← will likely collapse to ρ≈1
      P3 = dollar-additive   (gross cap 1.05–1.10×) ← preserves orthogonality

Walk-forward: skip first OOS_WARMUP=756 days for headline metrics.
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
BENCHMARK      = "SPY"


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


# ── Residual computation ────────────────────────────────────────────────────
def compute_residuals(rets: pd.DataFrame, bench: str = BENCHMARK,
                       beta_window: int = 60) -> pd.DataFrame:
    """Per-ticker rolling regression of returns on benchmark.  Returns the
    residual series (idiosyncratic return) for each ticker."""
    if bench not in rets.columns:
        raise ValueError(f"Benchmark {bench} missing from returns")
    spy = rets[bench]
    spy_mean = spy.rolling(beta_window).mean()
    spy_var  = spy.rolling(beta_window).var()

    out = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for t in rets.columns:
        r = rets[t]
        cov   = r.rolling(beta_window).cov(spy)
        beta  = cov / spy_var.replace(0, np.nan)
        alpha = r.rolling(beta_window).mean() - beta * spy_mean
        out[t] = r - alpha - beta * spy
    return out


def _atr_dollar_size(feat, idx, capital, cap_per_name=0.05):
    atr = feat["atr_14"].reindex(idx).ffill()
    cl  = feat["Close"].reindex(idx).ffill().replace(0, np.nan).ffill()
    sz  = (capital * RISK_PER_TRADE / atr.replace(0, np.nan) * cl).fillna(0.0)
    return sz.clip(upper=capital * cap_per_name)


# ── Sleeve construction ─────────────────────────────────────────────────────
def residual_mom_sleeve(rets: pd.DataFrame, feats: dict, residuals: pd.DataFrame,
                          candidates: list,
                          lookback: int = 21,
                          top_k: int = 5,
                          short_bottom_k: int = 0,
                          sleeve_pct: float = 0.10,
                          cap_per_name: float = 0.05,
                          sizing: str = "atr",
                          min_resid_mom: float | None = None) -> pd.DataFrame:
    """
    Cross-sectional residual momentum sleeve.

    Each day rank candidates by trailing `lookback`-day residual return.
    Long top_k; optionally short bottom_k (set 0 for long-only / zero-leverage).
    Sizing: 'atr' = vol-targeted per name; 'eq' = equal-dollar slot.
    `min_resid_mom`: optional floor (only enter if rank momentum > 0 etc).
    """
    cands = [t for t in candidates if t in rets.columns]
    rmom  = residuals[cands].rolling(lookback).sum()

    # Cross-sectional rank per day; rank 1 = highest residual momentum
    ranks = rmom.rank(axis=1, ascending=False, method="first")
    long_mask  = ranks <= top_k
    short_mask = pd.DataFrame(False, index=ranks.index, columns=ranks.columns)
    if short_bottom_k > 0:
        short_ranks = rmom.rank(axis=1, ascending=True, method="first")
        short_mask  = short_ranks <= short_bottom_k

    if min_resid_mom is not None:
        long_mask  = long_mask  & (rmom > min_resid_mom)
        short_mask = short_mask & (rmom < -min_resid_mom)

    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    for t in cands:
        if sizing == "atr":
            sz = _atr_dollar_size(feats[t], rets.index, CAPITAL, cap_per_name)
        else:  # equal-dollar
            sz = pd.Series(CAPITAL * cap_per_name, index=rets.index)
        out[t] = (sz * long_mask[t].astype(float)
                  - sz * short_mask[t].astype(float)).reindex(rets.index).fillna(0.0)

    sleeve_gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = sleeve_pct * CAPITAL
    scale  = (target / sleeve_gross).clip(upper=1.0).fillna(0.0)
    return out.multiply(scale, axis=0)


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

    print(f"\nComputing rolling-60d residuals vs {BENCHMARK}...")
    t1 = time.time()
    resid60  = compute_residuals(rets, BENCHMARK, beta_window=60)
    resid120 = compute_residuals(rets, BENCHMARK, beta_window=120)
    print(f"  done in {time.time()-t1:.1f}s")

    base = build_v4nf_base(sig, feats, rets)
    base_pr = portfolio_returns(base, rets).dropna()
    base_pr_oos = base_pr.iloc[OOS_WARMUP:] if len(base_pr) > OOS_WARMUP else base_pr

    print(f"\n{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    base_m = report("V4N base (no resid-mom sleeve)", base, rets)
    if base_m is None:
        print("Base failed; aborting.")
        return
    base_dd, base_ann, base_sh = base_m["mdd"], base_m["ann"], base_m["sharpe"]
    results = [base_m]

    # ── Phase 1: STANDALONE — gauge raw alpha + orthogonality ─────────────────
    print("\n--- Phase 1. Residual-momentum sleeve STANDALONE ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    for univ_name, univ in [("full", full), ("stocks", stocks), ("sectors", sectors)]:
        for resid_name, resid in [("r60", resid60), ("r120", resid120)]:
            for lb in (10, 21, 63):
                for k in (3, 5, 8):
                    sl = residual_mom_sleeve(rets, feats, resid, univ,
                                                lookback=lb, top_k=k,
                                                short_bottom_k=0,
                                                sleeve_pct=1.0,
                                                cap_per_name=0.20,
                                                sizing="atr")
                    m = report(f"P1: {univ_name} {resid_name} lb{lb} k{k} long-only",
                                sl, rets)
                    if m: results.append(m | {"_phase": "p1"})

    # Long/short variants on stocks (universe big enough for both legs)
    print("\n--- Phase 1b. Long/short residual sleeve (stocks only) ---")
    for resid_name, resid in [("r60", resid60), ("r120", resid120)]:
        for lb in (21, 63):
            for k in (3, 5):
                sl = residual_mom_sleeve(rets, feats, resid, stocks,
                                            lookback=lb, top_k=k,
                                            short_bottom_k=k,
                                            sleeve_pct=1.0,
                                            cap_per_name=0.10,
                                            sizing="atr")
                m = report(f"P1b: stocks {resid_name} lb{lb} k{k} L/S",
                            sl, rets)
                if m: results.append(m | {"_phase": "p1b"})

    # ── Phase 2: LAYERED with shrink+renorm (gross cap 1.0×) ─────────────────
    print("\n--- Phase 2. Layered onto V4N base (shrink + 1.0× cap) ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    grid2 = []
    for univ_name, univ in [("full", full), ("stocks", stocks)]:
        for resid_name, resid in [("r60", resid60), ("r120", resid120)]:
            for lb in (10, 21, 63):
                for k in (3, 5):
                    for sp in (0.05, 0.08, 0.12):
                        grid2.append((univ_name, univ, resid_name, resid, lb, k, sp))

    for (uname, univ, rname, resid, lb, k, sp) in grid2:
        sl = residual_mom_sleeve(rets, feats, resid, univ,
                                    lookback=lb, top_k=k, short_bottom_k=0,
                                    sleeve_pct=sp, cap_per_name=0.04,
                                    sizing="atr")
        combined = _layer_long_sleeve(base, sl, sp)
        m = report(f"P2: {uname} {rname} lb{lb} k{k} sp{sp}",
                    combined, rets, base_dd, base_ann, base_pr_oos)
        if m: results.append(m | {"_phase": "p2"})

    # ── Phase 3: DOLLAR-ADDITIVE layered (gross cap 1.05/1.10×) ──────────────
    print("\n--- Phase 3. Dollar-additive layering (no shrink) ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    grid3 = []
    for univ_name, univ in [("full", full), ("stocks", stocks), ("sectors", sectors)]:
        for resid_name, resid in [("r60", resid60), ("r120", resid120)]:
            for lb in (10, 21, 63):
                for k in (3, 5):
                    for sp in (0.05, 0.08, 0.12):
                        for mg in (1.05, 1.10):
                            grid3.append((univ_name, univ, resid_name, resid, lb, k, sp, mg))

    for (uname, univ, rname, resid, lb, k, sp, mg) in grid3:
        sl = residual_mom_sleeve(rets, feats, resid, univ,
                                    lookback=lb, top_k=k, short_bottom_k=0,
                                    sleeve_pct=sp, cap_per_name=0.04,
                                    sizing="atr")
        combined = _layer_dollar_additive(base, sl, max_gross_x=mg)
        m = report(f"P3: {uname} {rname} lb{lb} k{k} sp{sp} g{mg}",
                    combined, rets, base_dd, base_ann, base_pr_oos)
        if m: results.append(m | {"_phase": "p3"})

    # ── Leaderboards ──────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"V4N base: Sh {base_sh:.2f}  Ann {base_ann*100:.2f}%  "
          f"DD {base_dd*100:.2f}%  Cal {base_m['calmar']:.2f}")
    print('='*100)

    p1  = [r for r in results if r.get("_phase") in ("p1", "p1b")]
    p2  = [r for r in results if r.get("_phase") == "p2"]
    p3  = [r for r in results if r.get("_phase") == "p3"]

    print("\nMost ORTHOGONAL standalone P1/P1b sleeves (lowest |corr| to V4N):")
    for r in p1:
        common = r["pr"].index.intersection(base_pr_oos.index)
        r["_corr"] = (r["pr"].reindex(common).corr(base_pr_oos.reindex(common))
                      if len(common) > 30 else float("nan"))
    by_ortho = sorted([r for r in p1 if not np.isnan(r["_corr"])],
                       key=lambda x: abs(x["_corr"]))
    for r in by_ortho[:12]:
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={r['_corr']:+.3f}")

    print("\nTop standalone P1/P1b by Sharpe (raw alpha quality):")
    by_sh1 = sorted(p1, key=lambda x: -x["sharpe"])
    for r in by_sh1[:10]:
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={r['_corr']:+.3f}")

    print("\nTop 15 LAYERED (P2, shrink+renorm) by Sharpe:")
    by_sh2 = sorted(p2, key=lambda x: -x["sharpe"])
    for r in by_sh2[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.2f}")

    print("\nP2 variants beating BOTH baseline AnnRet AND DD:")
    pareto2 = [r for r in p2 if r["ann"] >= base_ann and r["mdd"] >= base_dd]
    pareto2.sort(key=lambda x: -x["sharpe"])
    if not pareto2:
        print("  (none)")
    for r in pareto2[:10]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<70} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.2f}")

    print("\nTop 15 DOLLAR-ADDITIVE (P3) by Sharpe:")
    by_sh3 = sorted(p3, key=lambda x: -x["sharpe"])
    for r in by_sh3[:15]:
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
