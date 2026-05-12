"""
AnnRet lift on top of V4N-D production stack.

V4N-D baseline (current production):
  V3 sizer (top11+adx22+momt+ac55) + defensive_tilt + 12% sleeve
  + profit_take + cond_vol_carry + asym_vol_boost(O) + fear_topRS(S4 k=3)
  → OOS Sh 2.52 / Ann 23.18% / DD -4.20% / Cal 5.52

Goal: push AnnRet > 24% with DD <= -4.5%, Sharpe >= 2.45.
Anti-overfit gate: bull windows >= base-0.3pp AND >=2 stress windows beat base.

Families tested as last-step overlays on V4N-D:
  L. calm_topRS_concentration  — symmetric of S4 in calm regime
  P. push_calm_boost           — replace V4N-D's 1.15 asym with 1.18/1.20/1.25
  A. accel_kicker              — within longs, tilt to names where 21d/63d RS > thresh
  G. calm_gross_floor_lift     — raise gross floor 0.95→0.98 in calm regime
  C. combos of the above winners
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
    asym_vol_boost_overlay,
    fear_topRS_concentration_overlay,
    portfolio_returns,
    _get_macro,
    SIGNAL_DIR, FEATURE_DIR, CAPITAL,
)
from v1.pipeline.data_pipeline import TICKER_LIST
from v1.pipeline.signal_generation import ATR_PARTIAL_REMAIN
from v1.pipeline.backtester import sharpe_ratio, max_drawdown

OOS_WARMUP = 756


# ── Inputs / metrics ────────────────────────────────────────────────────────
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


# ── V4N-D base builder ──────────────────────────────────────────────────────
def build_v4nd(sig, feats, rets, macro,
                calm_boost=1.15, calm_z=-0.5,
                fear_cut=0.9, fear_z_o=1.0,
                top_k=3, fear_z_s4=1.0, rs_window=63):
    """Build full V4N-D production sizes (matches portfolio.py main path)."""
    s = top_n_adx_momt_ac_sizes(
        sig, feats, rets, CAPITAL,
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )
    s = defensive_tilt_overlay(s, sig, macro, CAPITAL)
    s = diversifier_sleeve_overlay(
        s, CAPITAL,
        sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
        sleeve_pct=0.12,
    )
    s = profit_take_overlay(s, rets, lookback=10, sigma_thresh=1.5,
                              scale=0.7, max_gross=1.0, capital=CAPITAL)
    s = cond_vol_carry_overlay(s, macro, fear_z=1.5, roc_days=5, fear_mult=0.5)
    s = asym_vol_boost_overlay(s, macro,
                                  calm_boost=calm_boost, calm_z=calm_z,
                                  fear_cut=fear_cut, fear_z=fear_z_o)
    s = fear_topRS_concentration_overlay(s, feats, macro,
                                            top_k=top_k, fear_z=fear_z_s4,
                                            rs_window=rs_window)
    return s


# ── Family L: calm-regime top-RS concentration ──────────────────────────────
def calm_topRS_concentration(sizes: pd.DataFrame, features: dict,
                              macro: pd.DataFrame, top_k: int = 5,
                              calm_z: float = -0.5,
                              rs_window: int = 63) -> pd.DataFrame:
    """In calm regime, drop bottom longs and concentrate notional into top-K."""
    if macro is None or "vix_zscore" not in macro.columns:
        return sizes
    out = sizes.copy()
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    rs = pd.DataFrame(index=sizes.index, columns=sizes.columns, dtype=float)
    for t in sizes.columns:
        if t in features and "log_return" in features[t].columns:
            rs[t] = features[t]["log_return"].reindex(sizes.index).rolling(rs_window).sum()
    rs = rs.shift(1)
    calm_days = sizes.index[(z <= calm_z)]
    for d in calm_days:
        row = out.loc[d]
        long_pos = row[row > 0]
        if len(long_pos) < top_k:
            continue
        rs_row = rs.loc[d, long_pos.index].dropna()
        if len(rs_row) < top_k:
            continue
        keep = rs_row.nlargest(top_k).index
        drop = [t for t in long_pos.index if t not in keep]
        dropped_notional = long_pos.loc[drop].sum()
        each = dropped_notional / top_k
        for t in keep:
            out.loc[d, t] = out.loc[d, t] + each
        for t in drop:
            out.loc[d, t] = 0.0
    return out


# ── Family A: acceleration kicker (21d vs 63d RS ratio) ─────────────────────
def accel_kicker(sizes: pd.DataFrame, features: dict,
                  accel_thresh: float = 1.3, accel_boost: float = 1.20,
                  short_w: int = 21, long_w: int = 63,
                  capital: float = CAPITAL) -> pd.DataFrame:
    """
    Within active longs, boost names where short-window RS rank > long-window
    RS rank by `accel_thresh` (acceleration). Renormalize gross to preserve.
    """
    out = sizes.copy()
    rs_s = pd.DataFrame(index=sizes.index, columns=sizes.columns, dtype=float)
    rs_l = pd.DataFrame(index=sizes.index, columns=sizes.columns, dtype=float)
    for t in sizes.columns:
        if t in features and "log_return" in features[t].columns:
            r = features[t]["log_return"].reindex(sizes.index)
            rs_s[t] = r.rolling(short_w).sum()
            rs_l[t] = r.rolling(long_w).sum()
    rs_s = rs_s.shift(1); rs_l = rs_l.shift(1)
    # Avoid divide-by-zero; convert to ratio of (1+rs)
    ratio = (1 + rs_s) / (1 + rs_l)

    for d in sizes.index:
        row = out.loc[d]
        long_pos = row[row > 0]
        if long_pos.empty:
            continue
        r_row = ratio.loc[d, long_pos.index].dropna()
        if r_row.empty:
            continue
        accel = r_row[r_row >= accel_thresh].index
        if len(accel) == 0:
            continue
        gross_before = long_pos.sum()
        for t in accel:
            out.loc[d, t] = out.loc[d, t] * accel_boost
        # Renormalize longs to preserve gross
        new_long = out.loc[d, long_pos.index]
        scale = gross_before / new_long.sum() if new_long.sum() > 0 else 1.0
        out.loc[d, long_pos.index] = new_long * scale
    return out


# ── Family G: calm-regime gross-floor lift (post-overlays) ──────────────────
def calm_gross_floor_lift(sizes: pd.DataFrame, macro: pd.DataFrame,
                            calm_z: float = -0.5,
                            calm_floor: float = 0.98,
                            max_gross: float = 1.0,
                            capital: float = CAPITAL) -> pd.DataFrame:
    """
    On calm days where current gross < calm_floor*capital, scale UP to that
    floor.  Capped at max_gross*capital.  Acts AFTER all other overlays.
    """
    if macro is None or "vix_zscore" not in macro.columns:
        return sizes
    out = sizes.copy()
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = calm_floor * capital
    scale = pd.Series(1.0, index=sizes.index)
    calm_mask = (z <= calm_z) & (gross < target)
    scale[calm_mask] = (target / gross[calm_mask]).clip(upper=max_gross * capital / gross[calm_mask])
    out = out.multiply(scale, axis=0)
    # Final hard cap
    g = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = ((max_gross * capital) / g).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


# ── Reporter ────────────────────────────────────────────────────────────────
def report(name, sizes, rets, base_dd=None, base_ann=None):
    pr = portfolio_returns(sizes, rets).dropna()
    m = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
    if not m:
        print(f"{name:<60} insufficient data")
        return None
    flag = ""
    if base_dd is not None and m["mdd"] >= base_dd - 0.003:
        flag += " *"   # within 0.3pp of baseline DD
    if base_ann is not None and m["ann"] > base_ann:
        flag += "+"
    print(f"{name:<60} {m['sharpe']:>5.2f} {m['ann']*100:>6.2f}% "
          f"{m['mdd']*100:>6.2f}% {m['calmar']:>5.2f} {m['vol']*100:>5.1f}%{flag}")
    return m | {"name": name, "returns": pr}


# ── Window-robustness gate ──────────────────────────────────────────────────
def window_check(pr: pd.Series, base_pr: pd.Series, label: str) -> dict:
    """Returns dict of per-window AnnRet diffs (variant - base) in pp."""
    windows = [
        ("Pre-COVID 19-08→20-02",     "2019-08-07", "2020-02-19"),
        ("COVID crash 20-02→20-04",   "2020-02-20", "2020-04-30"),
        ("COVID rec  20 H2",          "2020-05-01", "2020-12-31"),
        ("Bull 2021",                  "2021-01-01", "2021-12-31"),
        ("Bear 22 H1",                 "2022-01-01", "2022-06-30"),
        ("OOS H2 22 (bear tail)",     "2022-08-01", "2022-12-31"),
        ("2023 recovery",              "2023-01-01", "2023-12-31"),
        ("2024 bull",                  "2024-01-01", "2024-12-31"),
        ("2025 H1",                    "2025-01-01", "2025-06-30"),
        ("2025 H2",                    "2025-07-01", "2025-12-31"),
    ]
    diffs = {}
    print(f"\n  Window robustness for {label}:")
    print(f"    {'Window':<28}  {'base Ann':>9}  {'var Ann':>9}  {'Δ pp':>6}")
    for lab, s, e in windows:
        b = pr.loc[s:e].dropna()
        a = base_pr.loc[s:e].dropna()
        if len(b) < 20 or len(a) < 20:
            continue
        b_ann = (1 + b).prod() ** (252 / len(b)) - 1
        a_ann = (1 + a).prod() ** (252 / len(a)) - 1
        d = (b_ann - a_ann) * 100
        diffs[lab] = d
        flag = "✓" if d > 0 else " "
        print(f"    {lab:<28}  {a_ann*100:>8.2f}%  {b_ann*100:>8.2f}%  {d:>+5.2f} {flag}")
    return diffs


# ── Main sweep ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in "
          f"{time.time()-t0:.1f}s\n")

    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    results = []

    # ── V4N-D baseline ──────────────────────────────────────────────────────
    base_sizes = build_v4nd(sig, feats, rets, macro)
    base_m = report("V4N-D baseline (production)", base_sizes, rets)
    results.append(base_m)
    base_dd, base_ann, base_pr = base_m["mdd"], base_m["ann"], base_m["returns"]

    # ── L. calm_topRS_concentration ─────────────────────────────────────────
    print("\n--- L. Calm-regime Top-RS concentration (symmetric to S4) ---")
    for tk in (3, 5, 7):
        for cz in (-0.5, -1.0):
            v = calm_topRS_concentration(base_sizes, feats, macro,
                                          top_k=tk, calm_z=cz)
            m = report(f"L: top_k={tk} calm_z={cz}", v, rets, base_dd, base_ann)
            if m: results.append(m)

    # ── P. Push calm_boost ──────────────────────────────────────────────────
    print("\n--- P. Push calm_boost on V4N-D's asym overlay ---")
    for cb in (1.18, 1.20, 1.25, 1.30):
        for cz in (-0.5, -1.0):
            v = build_v4nd(sig, feats, rets, macro,
                            calm_boost=cb, calm_z=cz)
            m = report(f"P: calm_boost={cb} calm_z={cz}", v, rets, base_dd, base_ann)
            if m: results.append(m)

    # ── A. Acceleration kicker ──────────────────────────────────────────────
    print("\n--- A. Acceleration kicker (21d/63d RS ratio tilt) ---")
    for at in (1.00, 1.02, 1.05, 1.10):
        for ab in (1.25, 1.35, 1.50, 1.75):
            v = accel_kicker(base_sizes, feats,
                              accel_thresh=at, accel_boost=ab)
            m = report(f"A: accel_thresh={at} accel_boost={ab}", v, rets, base_dd, base_ann)
            if m: results.append(m)

    # ── A2: shorter/longer horizon for accel signal ────────────────────────
    print("\n--- A2. Accel-kicker horizon variants (best A: thresh=1.05, boost=1.35) ---")
    for sw, lw in [(10, 63), (21, 63), (21, 126), (10, 42), (5, 21)]:
        v = accel_kicker(base_sizes, feats,
                          accel_thresh=1.05, accel_boost=1.35,
                          short_w=sw, long_w=lw)
        m = report(f"A2: sw={sw} lw={lw}", v, rets, base_dd, base_ann)
        if m: results.append(m)

    # ── X. Combos: best A + low-DD L ───────────────────────────────────────
    print("\n--- X. Combos: best-A then low-DD L ---")
    a_best = accel_kicker(base_sizes, feats,
                            accel_thresh=1.05, accel_boost=1.35)
    for tk, cz in [(7, -0.5), (5, -0.5), (3, -1.0)]:
        v = calm_topRS_concentration(a_best, feats, macro,
                                      top_k=tk, calm_z=cz)
        m = report(f"X: A(1.05,1.35) + L(k={tk}, cz={cz})", v, rets, base_dd, base_ann)
        if m: results.append(m)

    # ── G. Calm-regime gross-floor lift ────────────────────────────────────
    print("\n--- G. Calm-regime gross-floor lift (raise floor in calm) ---")
    for cf in (0.97, 0.98, 0.99, 1.00):
        for cz in (-0.5, -1.0):
            v = calm_gross_floor_lift(base_sizes, macro,
                                       calm_z=cz, calm_floor=cf)
            m = report(f"G: calm_floor={cf} calm_z={cz}", v, rets, base_dd, base_ann)
            if m: results.append(m)

    # ── Leaderboards ────────────────────────────────────────────────────────
    print(f"\n{'='*80}\nBaseline V4N-D: Sh {base_m['sharpe']:.2f}  "
          f"Ann {base_m['ann']*100:.2f}%  DD {base_m['mdd']*100:.2f}%  "
          f"Cal {base_m['calmar']:.2f}\n{'='*80}")

    print("\nTop 10 by AnnRet (DD <= V4N-D - 0.3pp tolerance):")
    safe = [r for r in results if r and r["mdd"] >= base_dd - 0.003 and r["ann"] > base_ann]
    safe.sort(key=lambda x: -x["ann"])
    for r in safe[:10]:
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    print("\nTop 5 by Sharpe (DD <= V4N-D):")
    by_sh = [r for r in results if r and r["mdd"] >= base_dd - 0.003]
    by_sh.sort(key=lambda x: -x["sharpe"])
    for r in by_sh[:5]:
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    # ── Window-robustness on top-3 by AnnRet ────────────────────────────────
    print(f"\n{'='*80}\nWINDOW ROBUSTNESS for top-3 AnnRet candidates\n{'='*80}")
    for r in safe[:3]:
        diffs = window_check(r["returns"], base_pr, r["name"])
        bull_loss = sum(1 for k, v in diffs.items()
                        if any(b in k for b in ("Bull 2021","2024","2023","2025"))
                        and v < -0.3)
        stress_wins = sum(1 for k, v in diffs.items()
                          if any(b in k for b in ("COVID","Bear","H2 22"))
                          and v > 0.5)
        verdict = ("PASS" if bull_loss == 0 and stress_wins >= 2
                   else "MARGINAL" if bull_loss <= 1 and stress_wins >= 1
                   else "FAIL")
        print(f"  → {verdict}  (bull losses {bull_loss}, stress wins {stress_wins})")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
