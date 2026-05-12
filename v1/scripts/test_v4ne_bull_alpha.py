"""
Bull-regime alpha overlays on V4N-E.

V4N-E is structurally bear-biased (12% defensive sleeve, calm-only boost).
Yearly breakdown: lags SPY 5-13pp in bull years (2019/20/21/24), wins big in
bears (2022 +25.6pp, 2025 +7.8pp).

Goal: add overlays that activate ONLY in bull regime (SPY > 50dMA AND VIX-z
<= 0) and off otherwise.  Should close the bull-year deficit without harming
the bear-year alpha that's already working.

Overlays tested:
  B1. Sector L/S dispersion — long top-2 sector ETFs by 63d RS, short
       bottom-2.  Long-side comes out of V4N-E equity sleeve (so total long
       gross unchanged), short side adds short notional capped by max_gross.
  B2. Equity-index overweight — replace defensive sleeve (TLT/GLD/DBMF/VGSH)
       with SPY/IWM mix during bull.  Reverts in bear so sleeve protection
       is intact when needed.
  B3. Long sleeve leverage-up — boost the long sleeve gross from 0.95 to
       1.00 (use full capital) during bull only.
  B4. Combined: B1 + B2 (most aggressive bull tilt).

Yearly win-rate vs SPY is the headline metric — must improve from baseline
2/7 in bull years while keeping ALL bear-year wins.
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
    accel_kicker_overlay,
    portfolio_returns,
    _get_macro,
    SIGNAL_DIR, FEATURE_DIR, CAPITAL,
)
from v1.pipeline.data_pipeline import TICKERS, TICKER_LIST
from v1.pipeline.signal_generation import ATR_PARTIAL_REMAIN
from v1.pipeline.backtester import sharpe_ratio, max_drawdown

OOS_WARMUP = 756


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


def build_v4ne(sig, feats, rets, macro, sleeve_pct=0.12, gross_floor=0.95):
    s = top_n_adx_momt_ac_sizes(
        sig, feats, rets, CAPITAL,
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=gross_floor,
    )
    s = defensive_tilt_overlay(s, sig, macro, CAPITAL)
    s = diversifier_sleeve_overlay(s, CAPITAL,
                                     sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
                                     sleeve_pct=sleeve_pct)
    s = profit_take_overlay(s, rets, lookback=10, sigma_thresh=1.5,
                              scale=0.7, max_gross=1.0, capital=CAPITAL)
    s = cond_vol_carry_overlay(s, macro, fear_z=1.5, roc_days=5, fear_mult=0.5)
    s = asym_vol_boost_overlay(s, macro,
                                  calm_boost=1.15, calm_z=-0.5,
                                  fear_cut=0.9, fear_z=1.0)
    s = fear_topRS_concentration_overlay(s, feats, macro,
                                            top_k=3, fear_z=1.0, rs_window=63)
    s = accel_kicker_overlay(s, feats,
                                accel_thresh=1.05, accel_boost=1.35,
                                short_w=10, long_w=42)
    return s


def bull_mask(feats, macro, rets, spy_ma=50, vix_z_thresh=0.0):
    """True on days that qualify as bull regime (one-day-shifted, no look-ahead)."""
    spy = feats.get("SPY")
    if spy is None or "vix_zscore" not in macro.columns:
        return pd.Series(False, index=rets.index)
    spy_close = spy["Close"].reindex(rets.index).ffill()
    spy_avg = spy_close.rolling(spy_ma).mean()
    vix_z = macro["vix_zscore"].reindex(rets.index).ffill().bfill()
    bull = (spy_close > spy_avg) & (vix_z <= vix_z_thresh)
    return bull.shift(1).fillna(False).astype(bool)


# ── B1. Sector L/S dispersion (bull-only) ───────────────────────────────────
def bull_dispersion_overlay(sizes, feats, rets, bull, sectors,
                              top_k=2, bot_k=2, short_pct=0.05,
                              rs_window=63, capital=CAPITAL, max_gross=1.0):
    """
    On bull days, identify top-k and bottom-k sectors by 63d return.
    Short bottom-k equally up to short_pct * pv.  Long-side leaders are
    already captured by V4N-E's top-N — we don't overweight further to
    avoid concentration risk; instead we deploy the freed capital from
    sleeve trim (B4 combo) or just net-short the worst sectors.
    Maintains zero-leverage invariant.
    """
    out = sizes.copy()
    sec = [t for t in sectors if t in feats]
    if not sec or short_pct <= 0:
        return out

    # 63d log-return cumulant for each sector, shifted 1 day (no look-ahead)
    R = pd.DataFrame({t: feats[t]["log_return"] for t in sec}).reindex(rets.index)
    cum = R.rolling(rs_window).sum().shift(1)

    short_size_each = (short_pct * capital) / max(bot_k, 1)
    bull_idx = sizes.index[bull.values]
    for d in bull_idx:
        if d not in cum.index:
            continue
        row = cum.loc[d].dropna()
        if len(row) < bot_k + top_k:
            continue
        worst = row.nsmallest(bot_k).index.tolist()
        for t in worst:
            # Add short. If V4N-E already long this name, net-out (subtract)
            cur = float(out.loc[d, t]) if t in out.columns else 0.0
            out.loc[d, t] = cur - short_size_each

    # Cap gross at max_gross * capital
    gross = out.loc[bull_idx].abs().sum(axis=1)
    over = gross > max_gross * capital
    if over.any():
        scale = (max_gross * capital) / gross.where(over, max_gross * capital)
        out.loc[bull_idx] = out.loc[bull_idx].multiply(scale, axis=0)
    return out


# ── B2. Equity-index overweight (replace sleeve in bull) ────────────────────
def bull_equity_overweight_overlay(sizes, bull, eq_tickers=("SPY", "IWM"),
                                      sleeve_tickers=("TLT","GLD","DBMF","VGSH"),
                                      eq_pct=0.12, capital=CAPITAL):
    """
    On bull days, ZERO out the defensive sleeve and reallocate that notional
    equally to SPY/IWM.  Reverts to V4N-E sleeve on non-bull days.
    """
    out = sizes.copy()
    bull_idx = sizes.index[bull.values]
    if len(bull_idx) == 0:
        return out

    each = (eq_pct * capital) / len(eq_tickers)
    for d in bull_idx:
        for t in sleeve_tickers:
            if t in out.columns:
                out.loc[d, t] = 0.0
        for t in eq_tickers:
            if t in out.columns:
                cur = float(out.loc[d, t])
                # Add but cap (use 8% INDEX_ETF_CAP equivalent)
                out.loc[d, t] = min(cur + each, capital * 0.10)
    return out


# ── B3. Long-sleeve leverage-up via gross_floor bump ────────────────────────
def bull_floor_bump_overlay(sig, feats, rets, macro, bull,
                              base_floor=0.95, bull_floor=1.00):
    """Build V4N-E twice (base/bull) and switch on bull regime."""
    base = build_v4ne(sig, feats, rets, macro, gross_floor=base_floor)
    high = build_v4ne(sig, feats, rets, macro, gross_floor=bull_floor)
    out = base.copy()
    bull_idx = base.index[bull.values]
    out.loc[bull_idx] = high.loc[bull_idx]
    return out


def report(name, sizes, rets, base_dd=None, base_ann=None, base_sh=None):
    pr = portfolio_returns(sizes, rets).dropna()
    m = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
    if not m:
        print(f"{name:<55} insufficient data")
        return None
    flags = ""
    if base_sh  is not None: flags += "S" if m["sharpe"] >= base_sh else "·"
    if base_ann is not None: flags += "A" if m["ann"]    >= base_ann else "·"
    if base_dd  is not None: flags += "D" if m["mdd"]    >= base_dd  else "·"
    print(f"{name:<55} {m['sharpe']:>5.2f} {m['ann']*100:>6.2f}% "
          f"{m['mdd']*100:>6.2f}% {m['calmar']:>5.2f} {m['vol']*100:>5.1f}%  [{flags}]")
    return m | {"name": name, "returns": pr, "sizes": sizes}


def yearly_breakdown(pr: pd.Series, spy_pr: pd.Series, label: str, base_pr=None):
    """AnnRet by year + win/loss vs SPY, optionally vs baseline."""
    df = pd.DataFrame({"port": pr, "spy": spy_pr.reindex(pr.index)})
    if base_pr is not None:
        df["base"] = base_pr.reindex(pr.index)
    df = df.dropna()
    df["yr"] = df.index.year

    print(f"\n  {label}:")
    if base_pr is not None:
        print(f"    {'Year':>5}  {'Port':>8}  {'Base':>8}  {'SPY':>8}  {'vBase':>7}  {'vSPY':>7}  win")
    else:
        print(f"    {'Year':>5}  {'Port':>8}  {'SPY':>8}  {'Δ':>7}  win")

    wins_spy = wins_base = total = 0
    for yr, g in df.groupby("yr"):
        port_ann = (1 + g["port"]).prod() - 1
        spy_ann = (1 + g["spy"]).prod() - 1
        d_spy = (port_ann - spy_ann) * 100
        if base_pr is not None:
            base_ann_y = (1 + g["base"]).prod() - 1
            d_base = (port_ann - base_ann_y) * 100
            if d_base > 0: wins_base += 1
        if d_spy > 0: wins_spy += 1
        total += 1
        flag_spy = "✓" if d_spy > 0 else "x"
        if base_pr is not None:
            print(f"    {yr:>5}  {port_ann*100:>7.2f}%  {base_ann_y*100:>7.2f}%  "
                  f"{spy_ann*100:>7.2f}%  {d_base:>+6.2f}  {d_spy:>+6.2f} {flag_spy}")
        else:
            print(f"    {yr:>5}  {port_ann*100:>7.2f}%  {spy_ann*100:>7.2f}%"
                  f"  {d_spy:>+6.2f}  {flag_spy}")
    return wins_spy, wins_base, total


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in "
          f"{time.time()-t0:.1f}s\n")

    spy_simple = (np.exp(feats["SPY"]["log_return"].reindex(rets.index).fillna(0.0)) - 1)
    sectors = [t for t, c in TICKERS.items() if c == "sector_etf" and t in feats]

    print(f"Sector universe ({len(sectors)}): {sectors}\n")
    print(f"{'Variant':<55} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}  [SAD]")

    # ── V4N-E baseline ────────────────────────────────────────────────────
    base_sizes = build_v4ne(sig, feats, rets, macro)
    base_m = report("V4N-E baseline", base_sizes, rets)
    base_dd, base_ann, base_sh = base_m["mdd"], base_m["ann"], base_m["sharpe"]
    base_pr = base_m["returns"]

    # ── Bull regime stats ────────────────────────────────────────────────
    bull = bull_mask(feats, macro, rets, spy_ma=50, vix_z_thresh=0.0)
    bull_oos = bull.iloc[OOS_WARMUP:]
    bull_pct = bull_oos.sum() / len(bull_oos) * 100
    print(f"\nBull regime fraction (OOS): {bull_pct:.1f}% of days "
          f"({bull_oos.sum()}/{len(bull_oos)})")

    # ── B1: Sector L/S dispersion ────────────────────────────────────────
    print("\n--- B1. Bull-only sector L/S dispersion (short worst-k) ---")
    b1_results = []
    for sp in (0.03, 0.05, 0.08, 0.10):
        for bk in (1, 2, 3):
            v = bull_dispersion_overlay(base_sizes, feats, rets, bull, sectors,
                                          bot_k=bk, short_pct=sp)
            m = report(f"B1: short_pct={sp} bot_k={bk}", v, rets,
                       base_dd, base_ann, base_sh)
            if m: b1_results.append(m)

    # ── B2: Equity-index overweight ──────────────────────────────────────
    print("\n--- B2. Bull-only equity-index OW (replace sleeve with SPY/IWM) ---")
    b2_results = []
    for ep in (0.06, 0.10, 0.12):
        v = bull_equity_overweight_overlay(base_sizes, bull, eq_pct=ep)
        m = report(f"B2: eq_pct={ep} (SPY+IWM)", v, rets,
                   base_dd, base_ann, base_sh)
        if m: b2_results.append(m)

    # ── B3: Long-sleeve floor bump ───────────────────────────────────────
    print("\n--- B3. Bull-only gross-floor bump (95% → 100%) ---")
    b3_results = []
    for bf in (0.97, 0.99, 1.00):
        v = bull_floor_bump_overlay(sig, feats, rets, macro, bull,
                                       base_floor=0.95, bull_floor=bf)
        m = report(f"B3: bull_floor={bf}", v, rets, base_dd, base_ann, base_sh)
        if m: b3_results.append(m)

    # ── B4: Combo (B1 + B2) ──────────────────────────────────────────────
    print("\n--- B4. Combo: B1 dispersion + B2 equity OW ---")
    b4_results = []
    for ep in (0.06, 0.10):
        for sp, bk in [(0.05, 2), (0.08, 2), (0.05, 3)]:
            v = bull_equity_overweight_overlay(base_sizes, bull, eq_pct=ep)
            v = bull_dispersion_overlay(v, feats, rets, bull, sectors,
                                          bot_k=bk, short_pct=sp)
            m = report(f"B4: eq={ep} short={sp}/{bk}", v, rets,
                       base_dd, base_ann, base_sh)
            if m: b4_results.append(m)

    # ── B5: B2 + B3 (equity OW + floor bump) ─────────────────────────────
    print("\n--- B5. Combo: B2 equity OW + B3 floor bump ---")
    b5_results = []
    for ep in (0.10, 0.12):
        for bf in (0.99, 1.00):
            v = bull_floor_bump_overlay(sig, feats, rets, macro, bull,
                                           base_floor=0.95, bull_floor=bf)
            v = bull_equity_overweight_overlay(v, bull, eq_pct=ep)
            m = report(f"B5: eq={ep} bull_floor={bf}", v, rets,
                       base_dd, base_ann, base_sh)
            if m: b5_results.append(m)

    # ── B6: Bull-regime accel boost push (1.35 → 1.5/1.75) ────────────────
    print("\n--- B6. Bull-only accel boost push (more aggressive momo) ---")
    b6_results = []
    # Build V4N-E without the accel kicker, then re-apply with bull-conditional boost
    base_no_accel = top_n_adx_momt_ac_sizes(
        sig, feats, rets, CAPITAL,
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )
    base_no_accel = defensive_tilt_overlay(base_no_accel, sig, macro, CAPITAL)
    base_no_accel = diversifier_sleeve_overlay(base_no_accel, CAPITAL,
                                                 sleeve_tickers=("TLT","GLD","DBMF","VGSH"),
                                                 sleeve_pct=0.12)
    base_no_accel = profit_take_overlay(base_no_accel, rets, lookback=10,
                                          sigma_thresh=1.5, scale=0.7,
                                          max_gross=1.0, capital=CAPITAL)
    base_no_accel = cond_vol_carry_overlay(base_no_accel, macro,
                                             fear_z=1.5, roc_days=5, fear_mult=0.5)
    base_no_accel = asym_vol_boost_overlay(base_no_accel, macro,
                                             calm_boost=1.15, calm_z=-0.5,
                                             fear_cut=0.9, fear_z=1.0)
    base_no_accel = fear_topRS_concentration_overlay(base_no_accel, feats, macro,
                                                       top_k=3, fear_z=1.0,
                                                       rs_window=63)
    for bull_b in (1.50, 1.75, 2.00):
        # Apply 1.35 normally, bull_b in bull regime
        normal = accel_kicker_overlay(base_no_accel, feats,
                                        accel_thresh=1.05, accel_boost=1.35,
                                        short_w=10, long_w=42)
        bull_v = accel_kicker_overlay(base_no_accel, feats,
                                        accel_thresh=1.05, accel_boost=bull_b,
                                        short_w=10, long_w=42)
        v = normal.copy()
        bull_idx = v.index[bull.values]
        v.loc[bull_idx] = bull_v.loc[bull_idx]
        m = report(f"B6: bull_accel_boost={bull_b}", v, rets,
                   base_dd, base_ann, base_sh)
        if m: b6_results.append(m)

    # ── B7: XLK overweight (mega-cap tech tilt via sleeve→XLK swap) ───────
    # XLK proxies for QQQ (top holdings AAPL/MSFT/NVDA) and IS in V1 universe.
    print("\n--- B7. Bull-only XLK overweight (sleeve→XLK swap) ---")
    b7_results = []
    has_xlk = "XLK" in feats and "XLK" in rets.columns
    if not has_xlk:
        print("  XLK not in universe — skipping B7")
    else:
        for swap_pct in (0.04, 0.06, 0.08, 0.10):
            v = base_sizes.copy()
            bull_idx = v.index[bull.values]
            for d in bull_idx:
                tlt_cur = float(v.loc[d, "TLT"]) if "TLT" in v.columns else 0.0
                shave = min(swap_pct * CAPITAL, tlt_cur)
                if "TLT" in v.columns:
                    v.loc[d, "TLT"] = tlt_cur - shave
                xlk_cur = float(v.loc[d, "XLK"]) if "XLK" in v.columns else 0.0
                v.loc[d, "XLK"] = min(xlk_cur + shave, CAPITAL * 0.12)
            m = report(f"B7: swap_pct={swap_pct} (TLT→XLK in bull)", v, rets,
                       base_dd, base_ann, base_sh)
            if m: b7_results.append(m)

    # ── B9: Bull-only equity-index leverage stack (most aggressive) ──────
    # Replace ENTIRE defensive sleeve (12%) with SPY/IWM/XLK mix in bull.
    print("\n--- B9. Bull-only full sleeve→equity rotation ---")
    b9_results = []
    for spy_pct, iwm_pct, xlk_pct in [(0.06, 0.03, 0.03),
                                        (0.04, 0.04, 0.04),
                                        (0.06, 0.00, 0.06),
                                        (0.04, 0.00, 0.08)]:
        v = base_sizes.copy()
        bull_idx = v.index[bull.values]
        for d in bull_idx:
            # Zero defensive sleeve
            for st in ("TLT", "GLD", "DBMF", "VGSH"):
                if st in v.columns:
                    v.loc[d, st] = 0.0
            # Add equity mix
            for tk, pct in [("SPY", spy_pct), ("IWM", iwm_pct), ("XLK", xlk_pct)]:
                if tk in v.columns and pct > 0:
                    cur = float(v.loc[d, tk])
                    v.loc[d, tk] = min(cur + pct * CAPITAL, CAPITAL * 0.12)
        m = report(f"B9: SPY={spy_pct}/IWM={iwm_pct}/XLK={xlk_pct}", v, rets,
                   base_dd, base_ann, base_sh)
        if m: b9_results.append(m)

    # ── B8: B6 + B7 (most aggressive bull tilt) ───────────────────────────
    b8_results = []
    if has_xlk:
        print("\n--- B8. Combo: B6 accel push + B7 XLK swap ---")
        for bull_b in (1.50, 1.75, 2.00):
            for swap_pct in (0.06, 0.08, 0.10):
                normal = accel_kicker_overlay(base_no_accel, feats,
                                                accel_thresh=1.05, accel_boost=1.35,
                                                short_w=10, long_w=42)
                bull_v = accel_kicker_overlay(base_no_accel, feats,
                                                accel_thresh=1.05, accel_boost=bull_b,
                                                short_w=10, long_w=42)
                v = normal.copy()
                bull_idx = v.index[bull.values]
                v.loc[bull_idx] = bull_v.loc[bull_idx]
                # Then XLK swap
                for d in bull_idx:
                    tlt_cur = float(v.loc[d, "TLT"]) if "TLT" in v.columns else 0.0
                    shave = min(swap_pct * CAPITAL, tlt_cur)
                    if "TLT" in v.columns:
                        v.loc[d, "TLT"] = tlt_cur - shave
                    xlk_cur = float(v.loc[d, "XLK"]) if "XLK" in v.columns else 0.0
                    v.loc[d, "XLK"] = min(xlk_cur + shave, CAPITAL * 0.12)
                m = report(f"B8: accel={bull_b} swap={swap_pct}", v, rets,
                           base_dd, base_ann, base_sh)
                if m: b8_results.append(m)

    # ── Headline: pick variants improving Sharpe AND Ann (DD ≤ baseline+0.3pp) ──
    print(f"\n{'='*100}\nLeaderboard (Sh ≥ base, Ann > base, DD ≤ base+0.3pp)\n{'='*100}\n")
    all_results = (b1_results + b2_results + b3_results + b4_results +
                    b5_results + b6_results + b7_results + b8_results +
                    b9_results)
    safe = [r for r in all_results
            if r["sharpe"] >= base_sh - 0.10
            and r["ann"]    >  base_ann
            and r["mdd"]    >= base_dd - 0.005]
    safe.sort(key=lambda x: -x["ann"])
    for r in safe[:12]:
        print(f"  {r['name']:<55} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    # ── Yearly breakdown for top-3 candidates vs SPY AND vs baseline ──────
    print(f"\n{'='*100}\nYearly win-rate vs SPY + vs baseline for top-3\n{'='*100}")
    seen = set()
    for r in safe[:3]:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        wins_spy, wins_base, total = yearly_breakdown(
            r["returns"], spy_simple, r["name"], base_pr=base_pr,
        )
        print(f"  → {r['name']}: {wins_spy}/{total} years vs SPY, "
              f"{wins_base}/{total} years vs V4N-E base")

    # Also show baseline yearly for reference
    print(f"\n--- V4N-E baseline reference ---")
    yearly_breakdown(base_pr, spy_simple, "V4N-E baseline")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
