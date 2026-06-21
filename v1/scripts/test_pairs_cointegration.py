"""
Sector-pair cointegration / ratio-reversion sleeve on top of V4N-F base.

Hypothesis: V4N base is long-only and beta-loaded (S&P-correlated).  A
market-neutral sector-pair mean-reversion sleeve — long underperformer / short
outperformer when their ratio diverges from its rolling mean — should be
*structurally* uncorrelated with V4N's directional alpha and pay carry from
spread reversion.

Pairs (chosen for economic linkage):
  XLK / XLF   — tech vs banks (rate sensitivity)
  XLE / XLU   — oil vs utilities (commodity vs defensive)
  XLY / XLP   — discretionary vs staples (cyclical pair)
  XLV / XLI   — health vs industrial
  XLB / XLRE  — materials vs real estate
  IWM / SPY   — small-cap vs large-cap (size factor)
  QQQ / SPY   — tech-tilt vs broad

Mechanic per pair:
  ratio  = px_a / px_b
  z      = (ratio - rolling_mean(z_window)) / rolling_std(z_window)
  enter long-A short-B  if z < -entry_z   (A undervalued vs B)
  enter short-A long-B  if z > +entry_z
  exit at |z| < exit_z (mean reversion)
  position size: dollar-neutral via ATR per leg

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

PAIRS = [
    ("XLK", "XLF"),
    ("XLE", "XLU"),
    ("XLY", "XLP"),
    ("XLV", "XLI"),
    ("XLB", "XLRE"),
    ("IWM", "SPY"),
    ("QQQ", "SPY"),
]


def metrics(r):
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


def _atr_dollar_size(feat, idx, capital, cap_per_name=0.05):
    atr = feat["atr_14"].reindex(idx).ffill()
    cl  = feat["Close"].reindex(idx).ffill().replace(0, np.nan).ffill()
    sz  = (capital * RISK_PER_TRADE / atr.replace(0, np.nan) * cl).fillna(0.0)
    return sz.clip(upper=capital * cap_per_name)


def pair_signal(feats: dict, rets: pd.DataFrame, a: str, b: str,
                  z_window: int = 60) -> pd.Series:
    """Returns z-scored ratio: positive z = A overvalued vs B; negative = undervalued."""
    pa = feats[a]["Close"].reindex(rets.index).ffill()
    pb = feats[b]["Close"].reindex(rets.index).ffill().replace(0, np.nan)
    ratio = np.log(pa / pb)
    mu  = ratio.rolling(z_window).mean()
    sd  = ratio.rolling(z_window).std().replace(0, np.nan)
    return (ratio - mu) / sd


def pairs_sleeve(rets: pd.DataFrame, feats: dict,
                   pairs: list, z_window: int = 60,
                   entry_z: float = 2.0, exit_z: float = 0.5,
                   cap_per_name: float = 0.04,
                   sleeve_pct: float = 0.10,
                   sizing: str = "atr") -> pd.DataFrame:
    """Long underperformer / short outperformer per pair, hold until z reverts."""
    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)

    for a, b in pairs:
        if a not in feats or b not in feats:
            continue
        z = pair_signal(feats, rets, a, b, z_window)

        # State: -1 = short-A long-B  (z > +entry, A overvalued)
        #         0 = flat
        #        +1 = long-A short-B  (z < -entry, A undervalued)
        state = pd.Series(0, index=z.index, dtype=int)
        cur = 0
        for i, ts in enumerate(z.index):
            v = z.iloc[i]
            if np.isnan(v):
                state.iloc[i] = cur
                continue
            if cur == 0:
                if v >= entry_z:
                    cur = -1
                elif v <= -entry_z:
                    cur = +1
            else:
                if abs(v) <= exit_z:
                    cur = 0
                elif (cur == +1 and v >= entry_z) or (cur == -1 and v <= -entry_z):
                    cur *= -1  # flip on opposite extreme
            state.iloc[i] = cur

        if sizing == "atr":
            sz_a = _atr_dollar_size(feats[a], rets.index, CAPITAL, cap_per_name)
            sz_b = _atr_dollar_size(feats[b], rets.index, CAPITAL, cap_per_name)
        else:
            sz_a = pd.Series(CAPITAL * cap_per_name, index=rets.index)
            sz_b = pd.Series(CAPITAL * cap_per_name, index=rets.index)

        out[a] = out[a].add(sz_a * state, fill_value=0.0)
        out[b] = out[b].add(-sz_b * state, fill_value=0.0)

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

    valid_pairs = [(a, b) for a, b in PAIRS if a in feats and b in feats]
    print(f"Valid pairs ({len(valid_pairs)}): {valid_pairs}")

    base = build_v4nf_base(sig, feats, rets)
    base_pr = portfolio_returns(base, rets).dropna()
    base_pr_oos = base_pr.iloc[OOS_WARMUP:] if len(base_pr) > OOS_WARMUP else base_pr

    print(f"\n{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    base_m = report("V4N base (no pairs sleeve)", base, rets)
    if base_m is None:
        return
    base_dd, base_ann, base_sh = base_m["mdd"], base_m["ann"], base_m["sharpe"]
    results = [base_m]

    # ── Phase 1: STANDALONE ──────────────────────────────────────────────────
    print("\n--- Phase 1. Pairs sleeve STANDALONE (all pairs combined) ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    for zw in (60, 90, 126):
        for ez in (1.5, 2.0, 2.5):
            for xz in (0.0, 0.5):
                sl = pairs_sleeve(rets, feats, valid_pairs,
                                    z_window=zw, entry_z=ez, exit_z=xz,
                                    cap_per_name=0.10, sleeve_pct=1.0)
                m = report(f"P1: zw{zw} ez{ez} xz{xz}", sl, rets)
                if m: results.append(m | {"_phase": "p1"})

    # ── Phase 2: LAYERED shrink+renorm ──────────────────────────────────────
    print("\n--- Phase 2. Layered onto V4N base (shrink + 1.0× cap) ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    for zw in (60, 90, 126):
        for ez in (1.5, 2.0, 2.5):
            for xz in (0.0, 0.5):
                for sp in (0.05, 0.08, 0.12):
                    sl = pairs_sleeve(rets, feats, valid_pairs,
                                        z_window=zw, entry_z=ez, exit_z=xz,
                                        cap_per_name=0.04, sleeve_pct=sp)
                    combined = _layer_long_sleeve(base, sl, sp)
                    m = report(f"P2: zw{zw} ez{ez} xz{xz} sp{sp}",
                                combined, rets, base_dd, base_ann, base_pr_oos)
                    if m: results.append(m | {"_phase": "p2"})

    # ── Phase 3: DOLLAR-ADDITIVE ────────────────────────────────────────────
    print("\n--- Phase 3. Dollar-additive layering ---")
    print(f"{'Variant':<70} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    for zw in (60, 90, 126):
        for ez in (1.5, 2.0, 2.5):
            for xz in (0.0, 0.5):
                for sp in (0.05, 0.08):
                    for mg in (1.05, 1.10):
                        sl = pairs_sleeve(rets, feats, valid_pairs,
                                            z_window=zw, entry_z=ez, exit_z=xz,
                                            cap_per_name=0.04, sleeve_pct=sp)
                        combined = _layer_dollar_additive(base, sl, max_gross_x=mg)
                        m = report(f"P3: zw{zw} ez{ez} xz{xz} sp{sp} g{mg}",
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
