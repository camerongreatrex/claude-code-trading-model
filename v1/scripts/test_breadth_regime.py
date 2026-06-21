"""
Breadth & cross-sectional dispersion regime overlays on V4N-F base.

Hypothesis: V4N is a trend-following stack — it works better when MANY names
are trending (broad market) and when residual dispersion is HIGH (idiosyncratic
edge available).  Conversely, narrow rallies (few leaders) and low dispersion
(everything correlated) make trend picks fragile.

Distinct from VIX/fear gating (cond_vol_carry_overlay) because breadth and
dispersion measure cross-sectional structure, not implied vol level.

Signals:
  * BREADTH(w)  = fraction of stock universe with mom_w > 0
  * DISP(w)     = cross-sectional std of trailing-w log returns
  * HIGHFRAC(w) = fraction trading within 5% of trailing-w high

Overlay mode: scale_factor = mapping from signal value → [scale_lo, scale_hi].
base[t] *= scale[t], then re-cap gross at 1.0× (zero leverage preserved).

Test phases:
  P1 — single signal, two thresholds × three windows
  P2 — combine top P1 signals (multiply scales)
  P3 — continuous z-score-based scaling (smoother than threshold)
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


# ── Breadth & dispersion signals ─────────────────────────────────────────────
def breadth(rets: pd.DataFrame, feats: dict, candidates: list,
              window: int = 60) -> pd.Series:
    """Fraction of `candidates` whose mom_w > 0 each day (mom from features)."""
    cols = [t for t in candidates if t in feats and f"mom_{window}" in feats[t].columns]
    if not cols:
        return pd.Series(np.nan, index=rets.index)
    df = pd.DataFrame({t: feats[t][f"mom_{window}"].reindex(rets.index)
                        for t in cols})
    return (df > 0).mean(axis=1)


def dispersion(rets: pd.DataFrame, candidates: list, window: int = 20) -> pd.Series:
    """Cross-sectional std of trailing-window cumulative returns."""
    cols = [t for t in candidates if t in rets.columns]
    cum = rets[cols].rolling(window).sum()
    return cum.std(axis=1)


def near_high_frac(rets: pd.DataFrame, feats: dict, candidates: list,
                     window: int = 60, pct: float = 0.05) -> pd.Series:
    """Fraction of candidates with Close within `pct` of trailing-window high."""
    cols = [t for t in candidates if t in feats and "Close" in feats[t].columns]
    out  = pd.DataFrame(index=rets.index, columns=cols, dtype=float)
    for t in cols:
        cl   = feats[t]["Close"].reindex(rets.index).ffill()
        high = cl.rolling(window).max()
        out[t] = (cl >= high * (1 - pct)).astype(float)
    return out.mean(axis=1)


# ── Scale-mapping helpers ────────────────────────────────────────────────────
def threshold_scale(sig: pd.Series, hi_thresh: float, lo_thresh: float,
                     scale_hi: float = 1.10, scale_mid: float = 1.00,
                     scale_lo: float = 0.85) -> pd.Series:
    out = pd.Series(scale_mid, index=sig.index, dtype=float)
    out[sig >= hi_thresh] = scale_hi
    out[sig <= lo_thresh] = scale_lo
    return out


def zscore_scale(sig: pd.Series, window: int = 252,
                   amp: float = 0.10, clip_z: float = 2.0) -> pd.Series:
    """Continuous: scale = 1 + amp * clip(z, ±clip_z) / clip_z (z>0 → up, <0 → down)."""
    mu = sig.rolling(window).mean()
    sd = sig.rolling(window).std()
    z  = ((sig - mu) / sd.replace(0, np.nan)).clip(-clip_z, clip_z)
    return (1.0 + amp * (z / clip_z)).fillna(1.0)


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

    stocks  = [t for t, c in TICKERS.items() if c == "stock"      and t in rets.columns]
    sectors = [t for t, c in TICKERS.items() if c == "sector_etf" and t in rets.columns]
    full    = stocks + sectors
    print(f"Universe (breadth): {len(stocks)} stocks, {len(sectors)} sectors")

    base = build_v4nf_base(sig, feats, rets)
    base_pr = portfolio_returns(base, rets).dropna()
    base_pr_oos = base_pr.iloc[OOS_WARMUP:] if len(base_pr) > OOS_WARMUP else base_pr

    # Pre-compute signals
    print("\nComputing breadth / dispersion signals...")
    sigs = {
        "br20_stk":  breadth(rets, feats, stocks, 20),
        "br60_stk":  breadth(rets, feats, stocks, 60),
        "br60_full": breadth(rets, feats, full,   60),
        "disp20":    dispersion(rets, stocks, 20),
        "disp60":    dispersion(rets, stocks, 60),
        "nh60":      near_high_frac(rets, feats, stocks, 60, 0.05),
        "nh20":      near_high_frac(rets, feats, stocks, 20, 0.03),
    }

    print(f"\n{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    base_m = report("V4N base (no breadth overlay)", base, rets)
    if base_m is None:
        return
    base_dd, base_ann, base_sh = base_m["mdd"], base_m["ann"], base_m["sharpe"]
    results = [base_m]

    # ── Phase 1: threshold-based, single signal ──────────────────────────────
    print("\n--- Phase 1. Threshold scale, single signal ---")
    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    p1 = []
    for name, s in sigs.items():
        # Quantile thresholds for absolute calibration
        for hi_q, lo_q in [(0.70, 0.30), (0.80, 0.20), (0.90, 0.10)]:
            hi_t = s.quantile(hi_q); lo_t = s.quantile(lo_q)
            for hi, lo in [(1.05, 0.95), (1.08, 0.92), (1.10, 0.90), (1.10, 0.80)]:
                scale = threshold_scale(s, hi_t, lo_t, hi, 1.0, lo)
                sized = apply_scale(base, scale)
                lbl = f"P1: {name} q{int(hi_q*100)}/{int(lo_q*100)} hi{hi}/lo{lo}"
                m = report(lbl, sized, rets, base_dd, base_ann, base_pr_oos)
                if m: p1.append(m | {"_phase": "p1"})

    # ── Phase 2: continuous z-score scale ────────────────────────────────────
    print("\n--- Phase 2. Continuous z-score scale ---")
    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    p2 = []
    for name, s in sigs.items():
        for w in (126, 252, 504):
            for amp in (0.05, 0.10, 0.15):
                scale = zscore_scale(s, window=w, amp=amp, clip_z=2.0)
                sized = apply_scale(base, scale)
                lbl = f"P2: {name} z(w={w}) amp{amp}"
                m = report(lbl, sized, rets, base_dd, base_ann, base_pr_oos)
                if m: p2.append(m | {"_phase": "p2"})

    # ── Phase 3: combine top 2 signals ────────────────────────────────────────
    print("\n--- Phase 3. Combine breadth + dispersion (multiplicative) ---")
    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    p3 = []
    for br_w in (20, 60):
        for disp_w in (20, 60):
            br = breadth(rets, feats, stocks, br_w)
            dp = dispersion(rets, stocks, disp_w)
            for amp_b, amp_d in [(0.08, 0.08), (0.10, 0.05), (0.05, 0.10)]:
                s_br = zscore_scale(br, 252, amp_b)
                s_dp = zscore_scale(dp, 252, amp_d)
                combined = s_br * s_dp
                sized = apply_scale(base, combined)
                lbl = f"P3: br{br_w}/disp{disp_w} amp{amp_b}+{amp_d}"
                m = report(lbl, sized, rets, base_dd, base_ann, base_pr_oos)
                if m: p3.append(m | {"_phase": "p3"})

    # ── Leaderboards ──────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"V4N base: Sh {base_sh:.2f}  Ann {base_ann*100:.2f}%  "
          f"DD {base_dd*100:.2f}%  Cal {base_m['calmar']:.2f}")
    print('='*100)

    all_r = p1 + p2 + p3
    print("\nTop 15 by Sharpe (all phases):")
    for r in sorted(all_r, key=lambda x: -x["sharpe"])[:15]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.3f}")

    print("\nVariants beating BOTH baseline AnnRet AND DD:")
    pareto = [r for r in all_r if r["ann"] >= base_ann and r["mdd"] >= base_dd]
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
