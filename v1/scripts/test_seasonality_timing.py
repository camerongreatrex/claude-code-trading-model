"""
Calendar / seasonality timing overlays on V4N-F base.

Hypothesis: certain calendar windows have well-documented return premia or
penalties that are *structurally orthogonal* to any return-based signal (zero
correlation by construction with the input).  Scaling V4N base exposure up/
down by date should add expected return without adding return-driven beta.

Effects tested:
  * TOM  (turn-of-month) — last K + first K trading days of each month
                            historically over-perform ; equity exposure UP
  * DOW  (day-of-week)   — Mondays/Tuesdays historically positive,
                            Wednesdays mediocre, Fridays mixed
  * HW   (Halloween)     — Nov–Apr historically beats May–Oct
  * JAN  (January)       — first ~10 trading days of Jan
  * YE   (Year-end)      — Santa rally last 5 days of Dec + first 2 of Jan
  * FOMC — proxied by 3rd-Wed-of-month +/- 1d (real FOMC dates absent)

Mechanic per overlay:
  scale_factor[t] ∈ [scale_lo, scale_hi].  base[t] *= scale_factor[t], then
  re-cap at 1.0× gross (zero leverage preserved).

Standalone test: each overlay alone.
Layered combinations: pairs/triples of complementary overlays.
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


# ── Calendar masks ────────────────────────────────────────────────────────
def trading_day_in_month(idx: pd.DatetimeIndex) -> pd.Series:
    """1-indexed trading-day-of-month."""
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).cumcount() + 1


def trading_days_per_month(idx: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(idx, index=idx)
    counts = s.groupby([idx.year, idx.month]).transform("count")
    return counts


def make_tom_scale(idx, k: int = 4, scale_hi: float = 1.10,
                    scale_lo: float = 0.95) -> pd.Series:
    """Turn-of-month: last k + first k trading days = scale_hi, else scale_lo."""
    tdom   = trading_day_in_month(idx)
    n_dom  = trading_days_per_month(idx)
    is_tom = (tdom <= k) | (tdom > n_dom - k)
    return pd.Series(np.where(is_tom, scale_hi, scale_lo), index=idx)


def make_dow_scale(idx, mon_tue: float = 1.05, wed: float = 0.97,
                    thu_fri: float = 1.00) -> pd.Series:
    dow = pd.Series(idx.dayofweek, index=idx)
    s   = pd.Series(thu_fri, index=idx, dtype=float)
    s[dow.isin([0, 1])] = mon_tue
    s[dow == 2]         = wed
    return s


def make_halloween_scale(idx, nov_apr: float = 1.05,
                          may_oct: float = 0.95) -> pd.Series:
    month = pd.Series(idx.month, index=idx)
    s     = pd.Series(may_oct, index=idx, dtype=float)
    s[month.isin([11, 12, 1, 2, 3, 4])] = nov_apr
    return s


def make_january_scale(idx, jan_first10: float = 1.10,
                        other: float = 1.00) -> pd.Series:
    tdom  = trading_day_in_month(idx)
    month = pd.Series(idx.month, index=idx)
    s     = pd.Series(other, index=idx, dtype=float)
    s[(month == 1) & (tdom <= 10)] = jan_first10
    return s


def make_year_end_scale(idx, santa: float = 1.10, other: float = 1.00) -> pd.Series:
    tdom  = trading_day_in_month(idx)
    n_dom = trading_days_per_month(idx)
    month = pd.Series(idx.month, index=idx)
    s     = pd.Series(other, index=idx, dtype=float)
    last5_dec = (month == 12) & (tdom > n_dom - 5)
    first2_jan = (month == 1)  & (tdom <= 2)
    s[last5_dec | first2_jan] = santa
    return s


def make_fomc_proxy_scale(idx, drift: float = 1.05, other: float = 1.00) -> pd.Series:
    """Pre-FOMC drift proxy: 3rd Wednesday of month +/- 1 trading day.
    Real FOMC dates not in our data; the 3rd-Wed cycle approximates roughly
    8/year (overshoots — FOMC is 8/yr, this is 12/yr); treat as best-effort."""
    s = pd.Series(other, index=idx, dtype=float)
    by_month = pd.Series(idx, index=idx).groupby([idx.year, idx.month])
    # find 3rd Wednesday in each calendar month present in idx
    fomc_targets = set()
    for (_, _), group in by_month:
        wed = group[group.dt.dayofweek == 2]
        if len(wed) >= 3:
            fomc_targets.add(wed.iloc[2])
    fomc_set = pd.DatetimeIndex(sorted(fomc_targets))
    # drift window: 1 trading day before through FOMC day itself
    pos = idx.get_indexer(fomc_set)
    pos = pos[pos >= 1]
    drift_idx = idx[pos - 1].union(idx[pos])
    s.loc[drift_idx] = drift
    return s


def apply_scale(base: pd.DataFrame, scale: pd.Series,
                  capital: float = CAPITAL) -> pd.DataFrame:
    out   = base.multiply(scale.reindex(base.index).fillna(1.0), axis=0)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
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
            corr = f"  ρ={c:+.3f}"
    print(f"{name:<60} {m['sharpe']:>5.2f} {m['ann']*100:>6.2f}% "
          f"{m['mdd']*100:>6.2f}% {m['calmar']:>5.2f} {m['vol']*100:>5.1f}%{flag}{corr}")
    return m | {"name": name, "pr": pr_oos}


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in {time.time()-t0:.1f}s")

    base = build_v4nf_base(sig, feats, rets)
    base_pr = portfolio_returns(base, rets).dropna()
    base_pr_oos = base_pr.iloc[OOS_WARMUP:] if len(base_pr) > OOS_WARMUP else base_pr
    idx = base.index

    print(f"\n{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    base_m = report("V4N base (no seasonality)", base, rets)
    if base_m is None:
        print("Base failed; aborting.")
        return
    base_dd, base_ann, base_sh = base_m["mdd"], base_m["ann"], base_m["sharpe"]
    results = [base_m]

    # ── Phase 1: each overlay alone, modest scale ────────────────────────────
    print("\n--- Phase 1. Single overlay (modest, ±5–10%) ---")
    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    overlays_solo = [
        ("TOM_k4_h1.10_l0.95",  make_tom_scale(idx, 4, 1.10, 0.95)),
        ("TOM_k3_h1.08_l0.97",  make_tom_scale(idx, 3, 1.08, 0.97)),
        ("TOM_k5_h1.10_l0.95",  make_tom_scale(idx, 5, 1.10, 0.95)),
        ("DOW_mt1.05_w0.97",    make_dow_scale(idx, 1.05, 0.97, 1.00)),
        ("DOW_mt1.08_w0.95",    make_dow_scale(idx, 1.08, 0.95, 1.00)),
        ("HW_n1.05_m0.95",      make_halloween_scale(idx, 1.05, 0.95)),
        ("HW_n1.08_m0.92",      make_halloween_scale(idx, 1.08, 0.92)),
        ("JAN_first10_1.10",    make_january_scale(idx, 1.10, 1.00)),
        ("YE_santa_1.10",       make_year_end_scale(idx, 1.10, 1.00)),
        ("FOMC_proxy_1.05",     make_fomc_proxy_scale(idx, 1.05, 1.00)),
        ("FOMC_proxy_1.08",     make_fomc_proxy_scale(idx, 1.08, 1.00)),
    ]
    p1_results = []
    for name, scale in overlays_solo:
        sized = apply_scale(base, scale)
        m = report(f"P1: {name}", sized, rets, base_dd, base_ann, base_pr_oos)
        if m: p1_results.append(m | {"_phase": "p1", "_scale": scale})

    # ── Phase 2: pairs (TOM × DOW × HW combos) ───────────────────────────────
    print("\n--- Phase 2. Combined overlays (multiply scales) ---")
    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    combos = []
    tom_options = [("TOMk4", make_tom_scale(idx, 4, 1.10, 0.95)),
                    ("TOMk3", make_tom_scale(idx, 3, 1.08, 0.97))]
    dow_options = [("DOWmt", make_dow_scale(idx, 1.05, 0.97, 1.00)),
                    ("DOWoff", pd.Series(1.0, index=idx))]
    hw_options  = [("HWlite", make_halloween_scale(idx, 1.05, 0.95)),
                    ("HWoff",  pd.Series(1.0, index=idx))]
    ye_options  = [("YE10",  make_year_end_scale(idx, 1.10, 1.00)),
                    ("YEoff", pd.Series(1.0, index=idx))]
    p2_results = []
    for tn, ts in tom_options:
        for dn, ds in dow_options:
            for hn, hs in hw_options:
                for yn, ys in ye_options:
                    if all(x.endswith("off") for x in (dn, hn, yn)):
                        continue   # already covered in P1
                    combo = ts * ds * hs * ys
                    sized = apply_scale(base, combo)
                    name = f"P2: {tn}+{dn}+{hn}+{yn}"
                    m = report(name, sized, rets, base_dd, base_ann, base_pr_oos)
                    if m: p2_results.append(m | {"_phase": "p2"})

    # ── Phase 3: aggressive turn-of-month (bigger swing) ─────────────────────
    print("\n--- Phase 3. Aggressive TOM amplification ---")
    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    p3_results = []
    for k in (3, 4, 5):
        for hi in (1.12, 1.15, 1.20):
            for lo in (0.90, 0.85, 0.80):
                scale = make_tom_scale(idx, k, hi, lo)
                sized = apply_scale(base, scale)
                m = report(f"P3: TOM k{k} hi{hi} lo{lo}", sized, rets,
                            base_dd, base_ann, base_pr_oos)
                if m: p3_results.append(m | {"_phase": "p3"})

    # ── Leaderboards ──────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"V4N base: Sh {base_sh:.2f}  Ann {base_ann*100:.2f}%  "
          f"DD {base_dd*100:.2f}%  Cal {base_m['calmar']:.2f}")
    print('='*100)

    all_p = p1_results + p2_results + p3_results

    print("\nTop 15 by Sharpe (all phases):")
    for r in sorted(all_p, key=lambda x: -x["sharpe"])[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.3f}")

    print("\nVariants beating BOTH baseline AnnRet AND DD:")
    pareto = [r for r in all_p if r["ann"] >= base_ann and r["mdd"] >= base_dd]
    pareto.sort(key=lambda x: -x["sharpe"])
    if not pareto:
        print("  (none)")
    for r in pareto[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.3f}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
