"""
Cross-asset macro regime overlays on V4N-F base.

Hypothesis: existing V4N base only consumes VIX-based fear (cond_vol_carry).
There are several other cross-asset signals already in our macro/feature
panels that may carry independent regime info — credit spreads, yield-curve
divergence, gold/dollar dynamics, bond-equity correlation.

Mechanic: each signal → continuous z-score scale → multiply base exposure,
re-cap gross at 1.0×.  Same overlay machinery as breadth_regime test.

Signals tested (all from existing macro/feature panel):
  TLT_SPY_DIV   — flight-to-quality (bond-equity divergence)
  HY_IG_Z       — high-yield/investment-grade credit stress
  CRED_EQ_CORR  — credit-equity correlation lag (regime shift)
  BOND_EQ_BETA  — bond-equity beta (correlation breakdown)
  COMM_USD      — commodity-dollar signal
  VIX_TERM_Z    — vol term-structure (contango vs backwardation)
  GLD_DBC_RATIO — gold/commodities ratio (risk-off proxy)
  HYG_TLT_RATIO — credit spread proxy (risk-on)
  UUP_MOM       — dollar momentum (risk-off)

Sign convention: positive z → SCALE UP (bullish for trend), negative → DOWN.
For some signals the natural sign is inverted (e.g. high HY_IG_Z = stress = bearish);
test both signs to find the right direction.
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


def zscore_scale(sig: pd.Series, window: int = 252,
                   amp: float = 0.10, clip_z: float = 2.0,
                   sign: float = +1.0) -> pd.Series:
    """sign=+1 → high z scales UP; sign=-1 → high z scales DOWN."""
    mu = sig.rolling(window).mean()
    sd = sig.rolling(window).std()
    z  = ((sig - mu) / sd.replace(0, np.nan)).clip(-clip_z, clip_z)
    return (1.0 + sign * amp * (z / clip_z)).fillna(1.0)


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
    print(f"Macro panel columns: {list(macro.columns)[:10]}{'...' if len(macro.columns)>10 else ''}")

    base = build_v4nf_base(sig, feats, rets)
    base_pr = portfolio_returns(base, rets).dropna()
    base_pr_oos = base_pr.iloc[OOS_WARMUP:] if len(base_pr) > OOS_WARMUP else base_pr

    # ── Build cross-asset signals from macro panel + asset prices ─────────────
    sigs = {}

    # From SPY's feature panel (it has all macro features merged in)
    spy_f = feats.get("SPY")
    if spy_f is not None:
        for col in ("tlt_spy_divergence", "hy_ig_ratio_zscore",
                    "credit_equity_corr_lag1", "bond_equity_beta",
                    "commodity_dollar_signal", "vix_term_zscore"):
            if col in spy_f.columns:
                sigs[col] = spy_f[col].reindex(rets.index)

    # Direct cross-asset ratios (use rolling smoothing to make stable signals)
    def _ratio_mom(num, den, mom_w=20):
        if num not in feats or den not in feats: return None
        n = feats[num]["Close"].reindex(rets.index).ffill()
        d = feats[den]["Close"].reindex(rets.index).ffill()
        r = (n / d).replace([np.inf, -np.inf], np.nan).ffill()
        return r.pct_change(mom_w)

    # Risk-off ratios
    if "GLD" in feats and "DBC" in feats:
        sigs["GLD_DBC_mom20"] = _ratio_mom("GLD", "DBC", 20)
    if "HYG" in feats and "TLT" in feats:
        sigs["HYG_TLT_mom20"] = _ratio_mom("HYG", "TLT", 20)
    if "TLT" in feats and "SPY" in feats:
        sigs["TLT_SPY_mom20"] = _ratio_mom("TLT", "SPY", 20)
    if "UUP" in feats:
        sigs["UUP_mom20"] = feats["UUP"]["Close"].reindex(rets.index).ffill().pct_change(20)
    if "DBC" in feats:
        sigs["DBC_mom60"] = feats["DBC"]["Close"].reindex(rets.index).ffill().pct_change(60)

    sigs = {k: v for k, v in sigs.items() if v is not None and v.notna().any()}
    print(f"\nBuilt {len(sigs)} cross-asset signals: {list(sigs.keys())}")

    print(f"\n{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    base_m = report("V4N base (no cross-asset overlay)", base, rets)
    if base_m is None:
        return
    base_dd, base_ann, base_sh = base_m["mdd"], base_m["ann"], base_m["sharpe"]
    results = [base_m]

    # ── Phase 1: each signal alone, both signs, several amps ─────────────────
    print("\n--- Phase 1. Single signal, both sign conventions ---")
    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    p1 = []
    for name, s in sigs.items():
        for sign in (+1, -1):
            for w in (126, 252):
                for amp in (0.05, 0.10, 0.15):
                    scale = zscore_scale(s, window=w, amp=amp, sign=sign)
                    sized = apply_scale(base, scale)
                    sgn = "+" if sign > 0 else "-"
                    lbl = f"P1: {name} sgn{sgn} w{w} amp{amp}"
                    m = report(lbl, sized, rets, base_dd, base_ann, base_pr_oos)
                    if m: p1.append(m | {"_phase": "p1", "_sig": name})

    # ── Leaderboard ──────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"V4N base: Sh {base_sh:.2f}  Ann {base_ann*100:.2f}%  "
          f"DD {base_dd*100:.2f}%  Cal {base_m['calmar']:.2f}")
    print('='*100)

    print("\nTop 20 by Sharpe (P1):")
    for r in sorted(p1, key=lambda x: -x["sharpe"])[:20]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.3f}")

    print("\nVariants beating BOTH baseline AnnRet AND DD:")
    pareto = [r for r in p1 if r["ann"] >= base_ann and r["mdd"] >= base_dd]
    pareto.sort(key=lambda x: -x["sharpe"])
    if not pareto:
        print("  (none)")
    for r in pareto[:20]:
        c = r["pr"].corr(base_pr_oos.reindex(r["pr"].index))
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  ρ={c:+.3f}")

    # ── Phase 2: top-2 signal combos ──────────────────────────────────────────
    if pareto:
        print("\n--- Phase 2. Combine top Pareto signals ---")
        print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
        # take top-3 distinct signals
        seen = set(); top_sigs = []
        for r in pareto:
            if r["_sig"] not in seen:
                seen.add(r["_sig"]); top_sigs.append(r)
            if len(top_sigs) >= 4: break
        for i, ra in enumerate(top_sigs):
            for rb in top_sigs[i+1:]:
                # rebuild scales from saved-pr's variant name not possible — use base settings
                # take the "best" hyperparams of each (assume their P1 line)
                a_parts = ra["name"].split()
                b_parts = rb["name"].split()
                # parse "P1: name sgn+ w252 amp0.1" — simple heuristic
                def _parse(p):
                    name = p[1]
                    sign = +1 if p[2].endswith("+") else -1
                    w = int(p[3].lstrip("w"))
                    amp = float(p[4].lstrip("amp"))
                    return name, sign, w, amp
                an, asg, aw, aamp = _parse(a_parts)
                bn, bsg, bw, bamp = _parse(b_parts)
                sa = zscore_scale(sigs[an], window=aw, amp=aamp, sign=asg)
                sb = zscore_scale(sigs[bn], window=bw, amp=bamp, sign=bsg)
                sized = apply_scale(base, sa * sb)
                lbl = f"P2: ({an},{asg:+d},{aamp})×({bn},{bsg:+d},{bamp})"
                m = report(lbl, sized, rets, base_dd, base_ann, base_pr_oos)

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
