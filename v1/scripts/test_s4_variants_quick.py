"""
Quick S4 variants sweep: confirm winner before wiring into paper_trader.

Variants tested:
  S4 alone on BASE (no O)         — clean attribution
  S4 on O (asym 1.15/-0.5/0.9/1.0) — current candidate
  S4 on O + tiny H short hedge     — convexity bonus?
  S4 on O2 (more aggressive asym)  — push the calm side harder
  S4 with k in (3, 5, 7) × fz in (0.75, 1.0, 1.25)

Anti-overfit gate: bull windows >= O - 0.3pp, ≥ 2 stress windows beat O + 0.5pp.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v1.scripts.test_smart_shorts import (
    load_inputs, build_v3, apply_v4nb_overlays,
    asym_vol_boost, vix_spike_hedge, _layer_short,
    metrics, OOS_WARMUP,
)
from v1.scripts.test_bear_alpha import (
    fear_topRS_concentration, STRESS_WINDOWS, BULL_WINDOWS, slice_stats,
)
from v1.portfolio.portfolio import portfolio_returns, CAPITAL


def evaluate(name, sizes, rets):
    pr = portfolio_returns(sizes, rets).dropna()
    out = {"name": name, "pr": pr}
    out["oos"] = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
    for label, s, e in STRESS_WINDOWS + BULL_WINDOWS:
        out[label] = slice_stats(pr, s, e)
    return out


def fmt(m):
    if not m: return "  n/a"
    return f"{m['ann']*100:>5.1f}%"


def print_row(r):
    oos = r.get("oos", {})
    print(f"{r['name']:<48} Sh{oos.get('sharpe', 0):>5.2f} "
          f"A{oos.get('ann', 0)*100:>6.2f}% DD{oos.get('mdd', 0)*100:>6.2f}%  "
          + " ".join(f"{w[0][:8]:>9}={fmt(r.get(w[0]))}"
                    for w in STRESS_WINDOWS + BULL_WINDOWS))


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days in {time.time()-t0:.1f}s")

    base_v3 = build_v3(sig, feats, rets)
    base = apply_v4nb_overlays(base_v3, rets, macro)
    o_base = asym_vol_boost(base, macro, calm_boost=1.15,
                              calm_z=-0.5, fear_cut=0.9, fear_z=1.0)
    o2_base = asym_vol_boost(base, macro, calm_boost=1.20,
                               calm_z=-0.5, fear_cut=0.9, fear_z=1.0)

    rows = []
    rows.append(evaluate("V4N-B BASE", base, rets))
    rows.append(evaluate("O baseline (asym 1.15/0.9)", o_base, rets))
    rows.append(evaluate("O2 baseline (asym 1.20/0.9)", o2_base, rets))

    # S4 on each base
    for k in (3, 5):
        for fz in (0.75, 1.0, 1.25):
            for base_name, base_sizes in [("BASE", base), ("O", o_base), ("O2", o2_base)]:
                s = fear_topRS_concentration(base_sizes, feats, macro, top_k=k, fear_z=fz)
                rows.append(evaluate(f"S4-{base_name} k={k} fz={fz}", s, rets))

    # S4 on O + tiny H hedge
    print("\n=== S4-O + tiny H short hedge sweeps ===")
    for k in (3, 5):
        for sp_h in (0.02, 0.04):
            for h_z in (1.0, 1.5):
                sh = vix_spike_hedge(rets, feats, macro, hedges=("SPY", "QQQ"),
                                       z_thresh=h_z, short_pct=sp_h)
                s4_o = fear_topRS_concentration(o_base, feats, macro, top_k=k, fear_z=1.0)
                combined = _layer_short(s4_o, sh, sp_h)
                rows.append(evaluate(f"S4-O k={k} + H z>{h_z} sp={sp_h}", combined, rets))

    # Print all
    print(f"\n=== ALL VARIANTS (per-window AnnRet %) ===")
    for r in rows:
        print_row(r)

    # Anti-overfit gate vs O baseline
    o = rows[1]
    o_bull = {w[0]: o[w[0]]["ann"] if o[w[0]] else None for w in BULL_WINDOWS}
    o_stress = {w[0]: o[w[0]]["ann"] if o[w[0]] else None for w in STRESS_WINDOWS}

    survivors = []
    for r in rows[3:]:   # skip the 3 baselines
        bull_ok = True
        for w in BULL_WINDOWS:
            r_ann = r[w[0]]["ann"] if r[w[0]] else 0
            o_ann = o_bull.get(w[0]) or 0
            if r_ann < o_ann - 0.003:
                bull_ok = False
                break
        improvements = 0
        for w in STRESS_WINDOWS:
            r_ann = r[w[0]]["ann"] if r[w[0]] else 0
            o_ann = o_stress.get(w[0]) or 0
            if r_ann > o_ann + 0.005:
                improvements += 1
        if bull_ok and improvements >= 2:
            survivors.append((r, improvements))

    survivors.sort(key=lambda x: (-x[1], -x[0]["oos"]["ann"]))
    print(f"\n=== Anti-overfit gate survivors (bull>=O-0.3pp, >=2 stress beat O+0.5pp) ===")
    for r, imp in survivors[:10]:
        oos = r["oos"]
        print(f"  +{imp} stress | {r['name']:<48} OOS Sh {oos['sharpe']:.2f} "
              f"Ann {oos['ann']*100:.2f}% DD {oos['mdd']*100:.2f}% "
              f"Cal {oos['calmar']:.2f}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
