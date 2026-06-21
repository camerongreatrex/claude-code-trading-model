"""
Mean-reversion sleeve research on top of V4N-F production stack.

Goal: add a small (5-10%) long-only sleeve that buys oversold names betting on
short-term rebound.  V4N-F is pure trend-following — mean reversion is a
genuinely orthogonal signal class (entries fire when trend-followers sell).

Sleeve mechanic:
  * Each day, scan candidate universe for names with mom_5 <= mom_thresh
    (and optionally rsi_14 <= rsi_thresh, or Close < bb_lower).
  * Pick the N most-oversold (most negative mom_5).
  * ATR-size each, scale total to sleeve_pct of capital.
  * Hold for hold_days (continued daily-recompute approximates this).
  * Layer onto V4N-F base: longs *= (1-sleeve_pct), then add MR sleeve.
  * Re-cap combined gross at 1.0× capital (zero-leverage preserved).

Orthogonality check: correlation of MR-sleeve daily PnL vs V4N-F daily PnL.
A genuine orthogonal signal should be near-zero correlated.

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


# ── V4N-F base reproduction (mirrors paper_trader sizer) ────────────────────
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
    # NB: omits asym_vol_boost / fear_topRS / accel_kicker / bull_sleeve_swap
    # — those live in paper_trader._compute_topn_v2_targets and aren't exposed
    # as a clean DataFrame overlay here.  Base used as proxy for V4N stack.
    return base


def _atr_dollar_size(feat, idx, capital, cap_per_name=0.04):
    atr = feat["atr_14"].reindex(idx).ffill()
    cl  = feat["Close"].reindex(idx).ffill().replace(0, np.nan).ffill()
    sz  = (capital * RISK_PER_TRADE / atr.replace(0, np.nan) * cl).fillna(0.0)
    return sz.clip(upper=capital * cap_per_name)


# ── Mean-reversion sleeve ──────────────────────────────────────────────────
def mean_rev_sleeve(rets: pd.DataFrame, feats: dict, candidates: list,
                     mom_thresh: float = -0.05,
                     rsi_thresh: float | None = 35.0,
                     sleeve_pct: float = 0.08,
                     max_names: int = 5,
                     cap_per_name: float = 0.03,
                     hold_days: int = 5) -> pd.DataFrame:
    """
    Long-only oversold sleeve.

    Entry: mom_5 <= mom_thresh AND (rsi_thresh is None OR rsi_14 <= rsi_thresh).
    Hold:  rolling-window — name stays active hold_days after entry.
    Pick:  top max_names by most-negative mom_5 each day.
    Size:  ATR-based, then total-scaled to sleeve_pct of capital.
    """
    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    cands = [t for t in candidates
             if t in feats and t in rets.columns
             and "mom_5" in feats[t].columns
             and "rsi_14" in feats[t].columns]
    if not cands:
        return out

    entries = {}
    mom_by_t = {}
    for t in cands:
        m5  = feats[t]["mom_5"].reindex(rets.index).ffill()
        rsi = feats[t]["rsi_14"].reindex(rets.index).ffill()
        cond = (m5 <= mom_thresh)
        if rsi_thresh is not None:
            cond = cond & (rsi <= rsi_thresh)
        entries[t]  = cond.fillna(False)
        mom_by_t[t] = m5

    entry_df = pd.DataFrame(entries)
    # active = entry triggered any time in last hold_days bars
    active = entry_df.rolling(hold_days, min_periods=1).sum() > 0

    if max_names is not None and max_names < len(cands):
        mom_df = pd.DataFrame(mom_by_t)
        # Mask non-active to +inf so they sort last; rank ascending picks most-negative
        masked = mom_df.where(active, np.inf)
        ranks  = masked.rank(axis=1, method="first")
        keep   = ranks <= max_names
        active = active & keep

    for t in cands:
        sz = _atr_dollar_size(feats[t], rets.index, CAPITAL, cap_per_name)
        out[t] = (sz * active[t].astype(float)).reindex(rets.index).fillna(0.0)

    sleeve_gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = sleeve_pct * CAPITAL
    scale = (target / sleeve_gross).clip(upper=1.0).fillna(0.0)
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
    """Dollar-additive: do NOT shrink longs.  Add sleeve on top, only cap gross
    at max_gross_x × capital (default 1.10× — small leverage allowance preserves
    sleeve orthogonality instead of collapsing back to base via renorm)."""
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
        print(f"{name:<60} insufficient data")
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
    print(f"{name:<60} {m['sharpe']:>5.2f} {m['ann']*100:>6.2f}% "
          f"{m['mdd']*100:>6.2f}% {m['calmar']:>5.2f} {m['vol']*100:>5.1f}%{flag}{corr}")
    return m | {"name": name, "pr": pr_oos}


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in "
          f"{time.time()-t0:.1f}s\n")

    stocks  = [t for t, c in TICKERS.items() if c == "stock"      and t in rets.columns]
    sectors = [t for t, c in TICKERS.items() if c == "sector_etf" and t in rets.columns]
    indexes = [t for t, c in TICKERS.items() if c == "equity_index" and t in rets.columns]
    full    = stocks + sectors + indexes

    print(f"Universe: {len(stocks)} stocks, {len(sectors)} sectors, "
          f"{len(indexes)} indexes\n")

    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")

    base = build_v4nf_base(sig, feats, rets)
    base_pr = portfolio_returns(base, rets).dropna()
    base_pr_oos = base_pr.iloc[OOS_WARMUP:] if len(base_pr) > OOS_WARMUP else base_pr
    base_m = report("V4N base (no MR sleeve)", base, rets)
    if base_m is None:
        print("Base failed; aborting.")
        return
    base_dd, base_ann = base_m["mdd"], base_m["ann"]
    base_sh = base_m["sharpe"]
    results = [base_m]

    # ── Phase 1: orthogonality probe — pure MR sleeve standalone ─────────────
    print("\n--- Phase 1. MR sleeve STANDALONE (gauge raw alpha + Sharpe) ---")
    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    for univ_name, univ in [("full", full), ("stocks", stocks), ("sectors", sectors)]:
        for mt in (-0.04, -0.06, -0.08):
            for rt in (None, 30.0, 35.0):
                sl = mean_rev_sleeve(rets, feats, univ,
                                       mom_thresh=mt, rsi_thresh=rt,
                                       sleeve_pct=1.0, max_names=5,
                                       cap_per_name=0.10, hold_days=5)
                rt_lbl = f"r{int(rt)}" if rt else "noR"
                m = report(f"P1: {univ_name} m{mt} {rt_lbl}", sl, rets)
                if m: results.append(m | {"_phase": "p1"})

    # ── Phase 2: layered onto V4N-F base ────────────────────────────────────
    print("\n--- Phase 2. MR sleeve LAYERED (small) onto V4N base ---")
    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    grid = []
    for univ_name, univ in [("full", full), ("stocks", stocks)]:
        for mt in (-0.04, -0.05, -0.06, -0.08):
            for rt in (None, 30.0, 35.0, 40.0):
                for sp in (0.05, 0.08, 0.10):
                    for mx in (3, 5):
                        for hd in (3, 5, 8):
                            grid.append((univ_name, univ, mt, rt, sp, mx, hd))

    for (uname, univ, mt, rt, sp, mx, hd) in grid:
        sl = mean_rev_sleeve(rets, feats, univ,
                               mom_thresh=mt, rsi_thresh=rt,
                               sleeve_pct=sp, max_names=mx,
                               cap_per_name=0.03, hold_days=hd)
        combined = _layer_long_sleeve(base, sl, sp)
        rt_lbl = f"r{int(rt)}" if rt else "noR"
        m = report(f"P2: {uname} m{mt} {rt_lbl} sp{sp} mx{mx} hd{hd}",
                    combined, rets, base_dd, base_ann, base_pr_oos)
        if m: results.append(m | {"_phase": "p2"})

    # ── Phase 3: dollar-additive layering (no shrink, soft 1.10× cap) ────────
    print("\n--- Phase 3. MR sleeve DOLLAR-ADDITIVE onto V4N base "
          "(no long-shrink, max_gross 1.05x/1.10x) ---")
    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    grid3 = []
    for univ_name, univ in [("full", full), ("stocks", stocks), ("sectors", sectors)]:
        for mt in (-0.04, -0.06, -0.08):
            for rt in (None, 30.0, 35.0):
                for sp in (0.05, 0.08, 0.10):
                    for mx in (3, 5):
                        for hd in (3, 5, 8):
                            for mg in (1.05, 1.10):
                                grid3.append((univ_name, univ, mt, rt, sp, mx, hd, mg))

    for (uname, univ, mt, rt, sp, mx, hd, mg) in grid3:
        sl = mean_rev_sleeve(rets, feats, univ,
                               mom_thresh=mt, rsi_thresh=rt,
                               sleeve_pct=sp, max_names=mx,
                               cap_per_name=0.03, hold_days=hd)
        combined = _layer_dollar_additive(base, sl, max_gross_x=mg)
        rt_lbl = f"r{int(rt)}" if rt else "noR"
        m = report(f"P3: {uname} m{mt} {rt_lbl} sp{sp} mx{mx} hd{hd} g{mg}",
                    combined, rets, base_dd, base_ann, base_pr_oos)
        if m: results.append(m | {"_phase": "p3"})

    # ── Leaderboards ─────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"V4N base: Sh {base_sh:.2f}  Ann {base_ann*100:.2f}%  "
          f"DD {base_dd*100:.2f}%  Cal {base_m['calmar']:.2f}")
    print('='*90)

    p2 = [r for r in results if r.get("_phase") == "p2"]

    print("\nTop 15 LAYERED variants by Sharpe:")
    by_sh = sorted(p2, key=lambda x: -x["sharpe"])
    for r in by_sh[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index)) if base_pr_oos is not None else float("nan")
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.2f}")

    print("\nVariants beating BOTH baseline AnnRet AND baseline DD:")
    pareto = [r for r in p2 if r["ann"] >= base_ann and r["mdd"] >= base_dd]
    pareto.sort(key=lambda x: -x["sharpe"])
    if not pareto:
        print("  (none — MR sleeve always costs DD or AnnRet)")
    for r in pareto[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.2f}")

    p3 = [r for r in results if r.get("_phase") == "p3"]

    print("\nTop 15 DOLLAR-ADDITIVE (P3) variants by Sharpe:")
    by_sh3 = sorted(p3, key=lambda x: -x["sharpe"])
    for r in by_sh3[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.2f}")

    print("\nP3 variants beating BOTH baseline AnnRet AND baseline DD:")
    pareto3 = [r for r in p3 if r["ann"] >= base_ann and r["mdd"] >= base_dd]
    pareto3.sort(key=lambda x: -x["sharpe"])
    if not pareto3:
        print("  (none — dollar-additive MR sleeve still no Pareto win)")
    for r in pareto3[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.2f}")

    print("\nMost ORTHOGONAL standalone sleeves (lowest |corr| to V4N base):")
    p1 = [r for r in results if r.get("_phase") == "p1"]
    for r in p1:
        common = r["pr"].index.intersection(base_pr_oos.index)
        r["_corr"] = r["pr"].reindex(common).corr(base_pr_oos.reindex(common)) if len(common) > 30 else float("nan")
    by_ortho = sorted([r for r in p1 if not np.isnan(r["_corr"])],
                       key=lambda x: abs(x["_corr"]))
    for r in by_ortho[:10]:
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={r['_corr']:+.3f}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
