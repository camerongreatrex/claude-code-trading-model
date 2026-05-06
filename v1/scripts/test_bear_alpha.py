"""
Bear-regime alpha research — push AnnRet in stress windows without overfitting.

Premise: V4N-B and its tweaks (D/O/Q) all underperform their bull-period averages
in stress (COVID crash, Bear 2022).  Try overlays that ADD return when fear is
high (not just defend), then verify they don't hurt bull periods.

Families:
  S1. Defensive-long tilt   — in fear, overweight TLT/GLD/UUP/XLP/XLU sleeve
  S2. VXZ long-vol hedge    — long VXZ (mid-term VIX futures) when vix_z > thr
  S3. Vol-spike mean rev    — at extreme vix_z, long SPY 3-5d (panic bottom bet)
  S4. Top-RS concentration  — in fear, double-weight top-3 longs (cull weak)
  S5. Defensive sector pair — long XLP+XLU+XLV, short XLY in fear
  S6. Best combo            — stack winners on top of O baseline

Selection guard: only mark a variant as a winner if it improves AT LEAST 2 of 3
stress windows (COVID, Bear 2022 H1, H2 2022) AND doesn't drop bull-period
AnnRet (2021, 2023, 2024, 2025) by more than 0.3pp anywhere.

This selection rule is the anti-overfit gate.
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
    asym_vol_boost, _atr_dollar_size, RISK_PER_TRADE,
    metrics, OOS_WARMUP,
)
from v1.portfolio.portfolio import portfolio_returns, CAPITAL


STRESS_WINDOWS = [
    ("COVID crash",   "2020-02-20", "2020-04-30"),
    ("Bear 2022 H1",  "2022-01-01", "2022-06-30"),
    ("H2 2022 tail",  "2022-08-01", "2022-12-31"),
]
BULL_WINDOWS = [
    ("Bull 2021",     "2021-01-01", "2021-12-31"),
    ("2023 recovery", "2023-01-01", "2023-12-31"),
    ("2024 bull",     "2024-01-01", "2024-12-31"),
    ("2025 YTD",      "2025-01-01", "2025-12-31"),
]


def slice_stats(pr, start, end):
    s = pr.loc[start:end].dropna()
    if len(s) < 20:
        return None
    return metrics(s)


# ── S1. Fear-regime defensive long tilt ────────────────────────────────────
def fear_defensive_tilt(sizes, macro, defensives=("TLT", "GLD", "UUP", "XLP", "XLU"),
                          tilt_pct=0.10, fear_z=1.0, capital=CAPITAL):
    """When vix_z >= fear_z, scale longs by (1-tilt_pct) and add equal-$ defensive
    longs totalling tilt_pct * capital.  Combined gross capped at 1×."""
    if "vix_zscore" not in macro.columns:
        return sizes
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    fear = (z >= fear_z).astype(float)

    out = sizes.copy()
    # Reduce all positions in fear days by tilt_pct
    long_scale = 1.0 - (tilt_pct * fear)
    out = out.multiply(long_scale, axis=0)

    # Add defensive longs
    n_def = len(defensives)
    each = (tilt_pct * capital) / n_def
    for t in defensives:
        if t not in out.columns:
            continue
        out[t] = out[t].add(each * fear, fill_value=0.0)

    # Cap gross
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


# ── S2. VXZ long-vol hedge ──────────────────────────────────────────────────
def vxz_long_vol_hedge(sizes, feats, macro, sleeve_pct=0.05,
                         z_thresh=1.0, capital=CAPITAL):
    """Long VXZ (mid-term VIX futures) when vix_z >= z_thresh.  Captures
    realized convexity in fear without needing broker shorts."""
    if "VXZ" not in feats or "vix_zscore" not in macro.columns:
        return sizes
    out = sizes.copy()
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    fear = (z >= z_thresh).astype(float)

    sz = _atr_dollar_size(feats["VXZ"], sizes.index, capital, cap_per_name=0.05)
    target = sleeve_pct * capital
    vxz_pos = (sz * fear).clip(upper=target)

    if "VXZ" in out.columns:
        # Reduce other positions by sleeve_pct on fear days
        long_scale = 1.0 - (sleeve_pct * fear)
        out = out.multiply(long_scale, axis=0)
        out["VXZ"] = out["VXZ"].add(vxz_pos, fill_value=0.0)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


# ── S3. Extreme-fear SPY mean reversion ────────────────────────────────────
def panic_meanrev_long(sizes, feats, macro, hedge="SPY",
                         z_panic=2.0, hold_days=5, sleeve_pct=0.10,
                         capital=CAPITAL):
    """At extreme vix_z spike, long SPY for N days (panic bottom bet)."""
    if hedge not in feats or "vix_zscore" not in macro.columns:
        return sizes
    out = sizes.copy()
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    spike = (z >= z_panic) & (z.shift(1) < z_panic)   # rising-edge
    # Hold long for hold_days after spike
    active = spike.astype(float)
    for k in range(1, hold_days):
        active = active + spike.shift(k).fillna(0).astype(float)
    active = (active > 0).astype(float)

    sz = _atr_dollar_size(feats[hedge], sizes.index, capital, cap_per_name=0.10)
    target = sleeve_pct * capital
    add = (sz * active).clip(upper=target)
    if hedge in out.columns:
        long_scale = 1.0 - (sleeve_pct * active)
        out = out.multiply(long_scale, axis=0)
        out[hedge] = out[hedge].add(add, fill_value=0.0)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


# ── S4-smooth. Frozen top-K book between rebal days ────────────────────────
def fear_topRS_smoothed(sizes, feats, macro, top_k=3, fear_z=1.0,
                          rebal_freq="W-FRI", capital=CAPITAL):
    """Regime-switch weekly/monthly: on rebal days, if vix_z >= fear_z, FREEZE
    book to top-K (by 63d RS) equal-$ until next rebal day.  Otherwise use
    V4N-B sizing.  Total long $ = sum of V4N-B longs at the rebal day (gross
    preserved)."""
    if "vix_zscore" not in macro.columns:
        return sizes
    out = sizes.copy()
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()

    rs = pd.DataFrame(index=sizes.index, columns=sizes.columns, dtype=float)
    for t in sizes.columns:
        if t in feats and "log_return" in feats[t].columns:
            rs[t] = feats[t]["log_return"].reindex(sizes.index).rolling(63).sum()
    rs = rs.shift(1)

    rebal_days = sizes.index.to_series().groupby(pd.Grouper(freq=rebal_freq)).max().dropna().values
    rebal_set = set(pd.to_datetime(rebal_days))

    frozen_book = None   # dict {ticker: dollar} or None for normal V4N-B
    for d in sizes.index:
        if d in rebal_set:
            if z.loc[d] >= fear_z:
                row = sizes.loc[d]
                long_pos = row[row > 0]
                if len(long_pos) >= top_k:
                    rs_row = rs.loc[d, long_pos.index].dropna()
                    if len(rs_row) >= top_k:
                        keep = rs_row.nlargest(top_k).index.tolist()
                        total_long = long_pos.sum()
                        each = total_long / top_k
                        # Preserve bond/diversifier sleeve (sizes < 0 OR sleeve names)
                        sleeve_book = row.copy()
                        sleeve_book[long_pos.index] = 0.0   # zero out longs
                        for t in keep:
                            sleeve_book[t] = each
                        frozen_book = sleeve_book
                    else:
                        frozen_book = None
                else:
                    frozen_book = None
            else:
                frozen_book = None
        if frozen_book is not None:
            out.loc[d, frozen_book.index] = frozen_book.values
    return out


# ── S4. Fear-regime top-RS concentration ───────────────────────────────────
def fear_topRS_concentration(sizes, feats, macro, top_k=5, fear_z=1.0,
                                capital=CAPITAL):
    """In fear, drop bottom-half of long positions and redistribute to top-K
    by 63d relative strength.  Survivors lead recoveries."""
    if "vix_zscore" not in macro.columns:
        return sizes
    out = sizes.copy()
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()

    # Pre-compute 63d returns matrix from feats[t]['log_return']
    rs = pd.DataFrame(index=sizes.index, columns=sizes.columns, dtype=float)
    for t in sizes.columns:
        if t in feats and "log_return" in feats[t].columns:
            rs[t] = feats[t]["log_return"].reindex(sizes.index).rolling(63).sum()
    rs = rs.shift(1)   # use yesterday's RS for today's decision

    rebal_days = sizes.index[(z >= fear_z)]
    for d in rebal_days:
        row = out.loc[d]
        long_pos = row[row > 0]
        if len(long_pos) < top_k:
            continue
        # Filter to longs with valid RS
        rs_row = rs.loc[d, long_pos.index].dropna()
        if len(rs_row) < top_k:
            continue
        keep = rs_row.nlargest(top_k).index
        drop = [t for t in long_pos.index if t not in keep]
        # Redistribute dropped notional equally to top-K
        dropped_notional = long_pos.loc[drop].sum()
        each = dropped_notional / top_k
        for t in keep:
            out.loc[d, t] = out.loc[d, t] + each
        for t in drop:
            out.loc[d, t] = 0.0

    return out


# ── S5. Defensive vs cyclical sector pair (fear-only) ──────────────────────
def fear_sector_pair(sizes, feats, macro,
                      longs=("XLP", "XLU", "XLV"),
                      shorts=("XLY", "XLF"),
                      sleeve_pct=0.06, fear_z=1.0, capital=CAPITAL):
    """In fear, long defensives + short cyclicals (beta-neutral pair).
    Note: needs broker shorts.  Available alternative: long-only XLP/XLU/XLV."""
    if "vix_zscore" not in macro.columns:
        return sizes
    out = sizes.copy()
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    fear = (z >= fear_z).astype(float)

    long_each  = (sleeve_pct * capital) / len(longs)
    short_each = (sleeve_pct * capital) / len(shorts)

    # Reduce general longs by 2*sleeve_pct in fear (long+short total exposure)
    long_scale = 1.0 - (2 * sleeve_pct * fear)
    out = out.multiply(long_scale, axis=0)
    for t in longs:
        if t in out.columns:
            out[t] = out[t].add(long_each * fear, fill_value=0.0)
    for t in shorts:
        if t in out.columns:
            out[t] = out[t].add(-short_each * fear, fill_value=0.0)

    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


# ── S5b. Long-only defensive sector tilt (no broker shorts) ────────────────
def fear_defensive_sector_long(sizes, feats, macro,
                                  longs=("XLP", "XLU", "XLV"),
                                  sleeve_pct=0.08, fear_z=1.0, capital=CAPITAL):
    if "vix_zscore" not in macro.columns:
        return sizes
    out = sizes.copy()
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    fear = (z >= fear_z).astype(float)
    each = (sleeve_pct * capital) / len(longs)

    long_scale = 1.0 - (sleeve_pct * fear)
    out = out.multiply(long_scale, axis=0)
    for t in longs:
        if t in out.columns:
            out[t] = out[t].add(each * fear, fill_value=0.0)

    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


def evaluate(name, sizes, rets):
    """Return per-window stats dict for name."""
    pr = portfolio_returns(sizes, rets).dropna()
    out = {"name": name, "pr": pr}
    out["oos"] = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
    for label, s, e in STRESS_WINDOWS + BULL_WINDOWS:
        out[label] = slice_stats(pr, s, e)
    return out


def fmt_window(m, attr="ann"):
    if not m: return "  n/a "
    v = m.get(attr, 0)
    return f"{v*100:>5.1f}"


def print_table(rows, label):
    headers = ["OOS_Sh", "OOS_Ann", "OOS_DD"] + \
              [w[0] for w in STRESS_WINDOWS] + [w[0] for w in BULL_WINDOWS]
    print(f"\n=== {label} (AnnRet % per window) ===")
    print(f"{'Variant':<48} {'OOS Sh':>7} {'OOS Ann':>8} {'OOS DD':>8}  ", end="")
    for w in STRESS_WINDOWS + BULL_WINDOWS:
        print(f"{w[0][:11]:>12}", end=" ")
    print()
    for r in rows:
        oos = r.get("oos", {})
        print(f"{r['name']:<48} {oos.get('sharpe', 0):>7.2f} "
              f"{oos.get('ann', 0)*100:>7.2f}% {oos.get('mdd', 0)*100:>7.2f}%  ", end="")
        for w in STRESS_WINDOWS + BULL_WINDOWS:
            print(f"{fmt_window(r.get(w[0])):>12}", end=" ")
        print()


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {rets.index.min().date()} → "
          f"{rets.index.max().date()} in {time.time()-t0:.1f}s")

    # Build base + O-winner
    base_v3 = build_v3(sig, feats, rets)
    base_v4nb = apply_v4nb_overlays(base_v3, rets, macro)
    o_base = asym_vol_boost(base_v4nb, macro, calm_boost=1.15,
                              calm_z=-0.5, fear_cut=0.9, fear_z=1.0)

    rows = []
    rows.append(evaluate("V4N-B baseline", base_v4nb, rets))
    rows.append(evaluate("O winner (asym 1.15/-0.5/0.9/1.0)", o_base, rets))

    # ── S1. Defensive-long tilt ──────────────────────────────────────────────
    print("\n--- S1. Defensive-long tilt sweeps ---")
    for fz in (0.5, 1.0, 1.5):
        for tp in (0.06, 0.10, 0.15):
            s = fear_defensive_tilt(o_base, macro, fear_z=fz, tilt_pct=tp)
            rows.append(evaluate(f"S1: O + def-tilt fz={fz} tp={tp}", s, rets))

    # ── S2. VXZ long-vol hedge ──────────────────────────────────────────────
    print("--- S2. VXZ long-vol hedge sweeps ---")
    for zt in (0.5, 1.0, 1.5):
        for sp in (0.03, 0.05, 0.08):
            s = vxz_long_vol_hedge(o_base, feats, macro, z_thresh=zt, sleeve_pct=sp)
            rows.append(evaluate(f"S2: O + VXZ z>{zt} sp={sp}", s, rets))

    # ── S3. Panic mean-reversion ────────────────────────────────────────────
    print("--- S3. Panic mean-reversion long SPY ---")
    for zp in (1.5, 2.0, 2.5):
        for hd in (3, 5, 7):
            for sp in (0.05, 0.10):
                s = panic_meanrev_long(o_base, feats, macro,
                                          z_panic=zp, hold_days=hd, sleeve_pct=sp)
                rows.append(evaluate(f"S3: O + panic z>{zp} hold={hd} sp={sp}", s, rets))

    # ── S4. Top-RS concentration in fear ────────────────────────────────────
    print("--- S4. Top-RS concentration in fear ---")
    s4_results = []
    for k in (3, 5, 7):
        for fz in (0.5, 1.0, 1.5, 2.0):
            s = fear_topRS_concentration(o_base, feats, macro, top_k=k, fear_z=fz)
            r = evaluate(f"S4: O + topRS k={k} fz={fz}", s, rets)
            rows.append(r)
            s4_results.append(r)

    # ── S4-smooth: weekly/monthly rebal versions ────────────────────────────
    print("--- S4-smooth: weekly/monthly rebal (low turnover) ---")
    s4s_results = []
    for k in (3, 5):
        for fz in (0.75, 1.0, 1.25):
            for rf in ("W-FRI", "M"):
                s = fear_topRS_smoothed(o_base, feats, macro,
                                          top_k=k, fear_z=fz, rebal_freq=rf)
                r = evaluate(f"S4s: O + topRS k={k} fz={fz} {rf}", s, rets)
                rows.append(r)
                s4s_results.append((r, s))

    # Turnover for each smoothed variant — REAL day-over-day churn
    def real_turnover(sizes_df):
        daily_change = sizes_df.diff().abs().sum(axis=1)   # $ traded each day
        return daily_change

    print("\n=== S4-smooth REAL turnover (day-over-day) ===")
    o_dc = real_turnover(o_base)
    o_ann = (o_dc.sum() * 252 / len(o_base)) / CAPITAL
    print(f"  O baseline ann_turnover: {o_ann*100:.0f}% of capital")
    for r, s in s4s_results:
        dc = real_turnover(s)
        ann_t = (dc.sum() * 252 / len(s)) / CAPITAL
        s_pr = portfolio_returns(s, rets).dropna()
        # Cost = daily $ traded * cost rate / capital, applied to next day return
        cd5  = (dc / CAPITAL * 0.0005).reindex(s_pr.index).fillna(0.0)
        cd15 = (dc / CAPITAL * 0.0015).reindex(s_pr.index).fillna(0.0)
        m_raw = metrics(s_pr.iloc[OOS_WARMUP:])
        m5 = metrics((s_pr - cd5).iloc[OOS_WARMUP:])
        m15 = metrics((s_pr - cd15).iloc[OOS_WARMUP:])
        print(f"  {r['name']:<48} ann_turn={ann_t*100:>5.0f}%  "
              f"raw: Sh{m_raw['sharpe']:.2f}/A{m_raw['ann']*100:.1f}%  "
              f"5bps: Sh{m5['sharpe']:.2f}/A{m5['ann']*100:.1f}%  "
              f"15bps: Sh{m15['sharpe']:.2f}/A{m15['ann']*100:.1f}%")

    # S4 on V4N-B BASE (no O) — cleaner attribution
    print("--- S4 on BASE (no O) ---")
    s4_base_results = []
    for k in (3, 5, 7):
        for fz in (0.75, 1.0, 1.25):
            s = fear_topRS_concentration(base_v4nb, feats, macro, top_k=k, fear_z=fz)
            r = evaluate(f"S4-base: BASE + topRS k={k} fz={fz}", s, rets)
            rows.append(r)
            s4_base_results.append(r)

    # Turnover audit: how many rebalance days, position changes
    def turnover_stats(sizes_overlay, sizes_orig):
        diff = (sizes_overlay - sizes_orig).abs().sum(axis=1)
        n_rebal = (diff > 1.0).sum()
        avg_turn = diff[diff > 1.0].mean()
        annual_turn = (diff.sum() * 252 / len(sizes_orig)) / CAPITAL
        return n_rebal, avg_turn, annual_turn

    # Real turnover for unsmoothed S4 winner
    s4_winner_sizes = fear_topRS_concentration(o_base, feats, macro, top_k=3, fear_z=1.0)
    print(f"\n=== S4 unsmoothed winner (k=3 fz=1.0) REAL turnover ===")
    s4_dc = s4_winner_sizes.diff().abs().sum(axis=1)
    ann_s4 = (s4_dc.sum() * 252 / len(o_base)) / CAPITAL
    s4_pr = portfolio_returns(s4_winner_sizes, rets).dropna()
    m_s4_raw = metrics(s4_pr.iloc[OOS_WARMUP:])
    cd5_s4 = (s4_dc / CAPITAL * 0.0005).reindex(s4_pr.index).fillna(0.0)
    cd15_s4 = (s4_dc / CAPITAL * 0.0015).reindex(s4_pr.index).fillna(0.0)
    m_s4_5 = metrics((s4_pr - cd5_s4).iloc[OOS_WARMUP:])
    m_s4_15 = metrics((s4_pr - cd15_s4).iloc[OOS_WARMUP:])
    print(f"  ann_turnover: {ann_s4*100:.0f}% of capital")
    print(f"  raw : Sh {m_s4_raw['sharpe']:.2f}  Ann {m_s4_raw['ann']*100:.2f}%  DD {m_s4_raw['mdd']*100:.2f}%")
    print(f"  5bps: Sh {m_s4_5['sharpe']:.2f}  Ann {m_s4_5['ann']*100:.2f}%  DD {m_s4_5['mdd']*100:.2f}%")
    print(f"  15bps: Sh {m_s4_15['sharpe']:.2f}  Ann {m_s4_15['ann']*100:.2f}%  DD {m_s4_15['mdd']*100:.2f}%")

    # Show ALL S4 variants for inspection
    print("\n=== ALL S4 variants (audit) ===")
    print_table([rows[0], rows[1]] + s4_results + s4_base_results, "S4 audit")

    # ── S5b. Long-only defensive sectors ────────────────────────────────────
    print("--- S5b. Defensive sector long tilt (no shorts) ---")
    for fz in (0.5, 1.0, 1.5):
        for sp in (0.05, 0.08, 0.12):
            s = fear_defensive_sector_long(o_base, feats, macro,
                                              fear_z=fz, sleeve_pct=sp)
            rows.append(evaluate(f"S5b: O + def-sec fz={fz} sp={sp}", s, rets))

    # ── S6. Best combos (apply best from each family on top of O) ────────────
    # We'll filter winners after the per-family runs.

    # ── Anti-overfit gate ───────────────────────────────────────────────────
    print("\n=== Anti-overfit screen ===")
    base = rows[0]
    o = rows[1]
    o_bull = {w[0]: o[w[0]]["ann"] if o[w[0]] else None for w in BULL_WINDOWS}
    o_stress = {w[0]: o[w[0]]["ann"] if o[w[0]] else None for w in STRESS_WINDOWS}

    survivors = []
    for r in rows[2:]:
        # Bull guard: no bull window may drop > 0.3pp AnnRet vs O
        bull_ok = True
        for w in BULL_WINDOWS:
            r_ann = r[w[0]]["ann"] if r[w[0]] else 0
            o_ann = o_bull.get(w[0]) or 0
            if r_ann < o_ann - 0.003:   # 0.3pp threshold
                bull_ok = False
                break
        # Stress lift: at least 2 of 3 stress windows must beat O by > 0.5pp
        improvements = 0
        for w in STRESS_WINDOWS:
            r_ann = r[w[0]]["ann"] if r[w[0]] else 0
            o_ann = o_stress.get(w[0]) or 0
            if r_ann > o_ann + 0.005:
                improvements += 1
        if bull_ok and improvements >= 2:
            survivors.append((r, improvements))

    survivors.sort(key=lambda x: (-x[1], -x[0]["oos"]["ann"]))
    print(f"\nFound {len(survivors)} variants passing anti-overfit gate "
          f"(bull≥O-0.3pp ALL windows AND ≥2 stress windows beat O+0.5pp):")
    for r, imp in survivors[:15]:
        print(f"  +{imp} stress | {r['name']:<50} OOS Sh "
              f"{r['oos']['sharpe']:.2f} Ann {r['oos']['ann']*100:.2f}% "
              f"DD {r['oos']['mdd']*100:.2f}%")

    print_table([base, o] + [r for r, _ in survivors[:8]], "Top survivors vs base/O")

    # Stack top-2 survivors on top of O for combo test
    if len(survivors) >= 2:
        print("\n--- S6. Stack top-2 survivors on O ---")
        best_a, _ = survivors[0]
        best_b, _ = survivors[1]
        # Identify family from name
        # We need to re-build the sizes; simplest: re-run apply chain
        # Skip if combined stacking is not straightforward.
        # Instead: stack S1 and S2 manually if both present
        s1_winners = [r for r, _ in survivors if r["name"].startswith("S1:")]
        s2_winners = [r for r, _ in survivors if r["name"].startswith("S2:")]
        s5_winners = [r for r, _ in survivors if r["name"].startswith("S5b:")]
        if s1_winners and s2_winners:
            # Re-build: O + S1 + S2 stacked
            s1_p = s1_winners[0]["name"]
            s2_p = s2_winners[0]["name"]
            print(f"  Trying stack: {s1_p} + {s2_p}")
            # Parse sleeve params (lazy: just use defaults of best individual)
            # Skip explicit re-parse; just compose S1 then S2
            o_with_s1 = o_base
            for tok in s1_winners[0]["name"].split():
                if "fz=" in tok: fz_s1 = float(tok.split("=")[1])
                if "tp=" in tok: tp_s1 = float(tok.split("=")[1])
            o_with_s1 = fear_defensive_tilt(o_base, macro,
                                                fear_z=fz_s1, tilt_pct=tp_s1)
            for tok in s2_winners[0]["name"].split():
                if tok.startswith("z>"): zt_s2 = float(tok[2:])
                if tok.startswith("sp="): sp_s2 = float(tok.split("=")[1])
            o_stacked = vxz_long_vol_hedge(o_with_s1, feats, macro,
                                              z_thresh=zt_s2, sleeve_pct=sp_s2)
            stacked_row = evaluate(f"S6: O + S1 + S2 stack", o_stacked, rets)
            print_table([base, o, s1_winners[0], s2_winners[0], stacked_row],
                          "S6 stack vs components")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
