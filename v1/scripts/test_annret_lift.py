"""
AnnRet-lift sweep on top of V4N-B production stack.

Goal: keep AnnRet >=19% while DD stays <=-5%, ideally Sharpe >2.4.

V4N-B baseline (production):
  V3 sizer (top11+adx22+momt+ac55) + defensive_tilt + 12% sleeve diversifier
  + profit_take_overlay + cond_vol_carry_overlay.

Tests four families layered onto that base:
  A. Knob tuning  : gross_floor, target_vol, lev_x, mom_window
  B. Bond/cmdty shorts: allow signal=-1 to short bonds & commodities
  C. Sector shorts replace sleeve : drop diversifier, conditional sector shorts
  D. Vol-conditional leverage     : bump lev_x in calm-VIX regimes

Reports leaders by (a) Sharpe, (b) AnnRet with DD <= V4N-B baseline DD.
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
from v1.pipeline.data_pipeline import TICKERS, TICKER_LIST, ASSET_CLASS
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
    """Load multi_signals + raw signals (with -1 for bonds/cmdty short)."""
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


def build_v3(sig, feats, rets, **kw):
    """V3 base + defensive tilt + 12% sleeve."""
    base_kw = dict(
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )
    base_kw.update(kw)
    s = top_n_adx_momt_ac_sizes(sig, feats, rets, CAPITAL, **base_kw)
    s = defensive_tilt_overlay(s, sig, _get_macro(), CAPITAL)
    s = diversifier_sleeve_overlay(
        s, CAPITAL,
        sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
        sleeve_pct=0.12,
    )
    return s


def apply_v4nb_overlays(sizes, rets, macro):
    """Production overlays: profit_take + cond_vol_carry."""
    out = profit_take_overlay(sizes, rets, lookback=10, sigma_thresh=1.5,
                                 scale=0.7, max_gross=1.0, capital=CAPITAL)
    out = cond_vol_carry_overlay(out, macro,
                                   fear_z=1.5, roc_days=5, fear_mult=0.5)
    return out


def add_short_legs(sizes: pd.DataFrame, sig: pd.DataFrame, rets: pd.DataFrame,
                    feats: dict, capital: float,
                    short_classes=("bond", "commodity"),
                    short_pct: float = 0.10,
                    cap_per_name: float = 0.05,
                    require_adx: float = 22.0) -> pd.DataFrame:
    """
    Allow signal=-1 in `short_classes` to short — sized by ATR like longs,
    capped per-name and as % of pv.  Distinct from sleeve overlay because
    sizes are integrated into the base book (not added on top).
    """
    out = sizes.copy()
    eligible = [t for t in sig.columns
                if ASSET_CLASS.get(t, "") in short_classes
                and t in feats]

    # ATR matrix for sizing
    atr_m = pd.DataFrame(np.nan, index=sig.index, columns=eligible)
    close_m = pd.DataFrame(np.nan, index=sig.index, columns=eligible)
    adx_m   = pd.DataFrame(0.0, index=sig.index, columns=eligible)
    for t in eligible:
        f = feats[t]
        if "atr_14" in f.columns:
            atr_m[t] = f["atr_14"].reindex(sig.index).ffill()
        if "Close" in f.columns:
            close_m[t] = f["Close"].reindex(sig.index).ffill()
        if "adx" in f.columns:
            adx_m[t] = f["adx"].reindex(sig.index).ffill().fillna(0.0)

    risk_per_trade = 0.005   # mirror RISK_PER_TRADE
    base = pd.DataFrame(0.0, index=sig.index, columns=out.columns)
    for t in eligible:
        s_t   = sig.get(t, pd.Series(0.0, index=sig.index)).reindex(sig.index).fillna(0.0)
        atr_t = atr_m[t].fillna(0.0)
        cl_t  = close_m[t].replace(0, np.nan).ffill().fillna(1.0)
        adx_t = adx_m[t]
        sz    = (capital * risk_per_trade / atr_t.replace(0, np.nan) * cl_t).fillna(0.0)
        sz    = sz.clip(upper=capital * cap_per_name)
        active = (s_t < 0) & (adx_t >= require_adx)
        if t in base.columns:
            base[t] = (-sz * active.astype(float)).reindex(sig.index)

    # Scale total short notional to short_pct of capital
    short_gross = base.abs().sum(axis=1).replace(0, np.nan)
    target      = short_pct * capital
    scale       = (target / short_gross).clip(upper=1.0).fillna(0.0)
    base        = base.multiply(scale, axis=0)

    # Reduce longs by short_pct so combined gross stays at max_gross
    out = out * (1.0 - short_pct)
    out = out.add(base, fill_value=0.0)
    # Final zero-leverage clip
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


def add_vol_lev_boost(sizes: pd.DataFrame, macro: pd.DataFrame,
                        boost: float = 1.10, calm_z: float = -0.5,
                        max_gross: float = 1.0, capital: float = CAPITAL) -> pd.DataFrame:
    """
    When vix_zscore <= calm_z, scale sizes UP by `boost`.  Capped at max_gross.
    """
    if "vix_zscore" not in macro.columns:
        return sizes
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    scale = pd.Series(1.0, index=sizes.index)
    scale[z <= calm_z] = boost
    out = sizes.multiply(scale, axis=0)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = ((max_gross * capital) / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


def report(name, sizes, rets, baseline_dd=None):
    pr = portfolio_returns(sizes, rets).dropna()
    m = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
    if not m:
        print(f"{name:<60} insufficient data")
        return None
    flag = ""
    if baseline_dd is not None and m["mdd"] >= baseline_dd:
        flag = " *"   # DD as good or better than baseline
    print(f"{name:<60} {m['sharpe']:>5.2f} {m['ann']*100:>6.2f}% "
          f"{m['mdd']*100:>6.2f}% {m['calmar']:>5.2f} {m['vol']*100:>5.1f}%{flag}")
    return m | {"name": name}


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in "
          f"{time.time()-t0:.1f}s\n")

    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    results = []

    # ── V4N-B baseline (production stack) ───────────────────────────────────
    base_v3 = build_v3(sig, feats, rets)
    base_v4nb = apply_v4nb_overlays(base_v3, rets, macro)
    base_m = report("V4N-B baseline (V3 + profit_take + cond_vol_carry)",
                     base_v4nb, rets)
    results.append(base_m)
    base_dd = base_m["mdd"]

    # ── A. Knob tuning ──────────────────────────────────────────────────────
    print("\n--- A. Knob tuning ---")
    for gf in (0.95, 0.97, 0.99, 1.00):
        for tv in (0.14, 0.16, 0.18):
            for lev in (1.5, 1.7, 1.9):
                v3 = build_v3(sig, feats, rets,
                                gross_floor=gf, target_vol=tv, lev_x=lev)
                full = apply_v4nb_overlays(v3, rets, macro)
                m = report(f"A: gf={gf} tv={tv} lev={lev}", full, rets, base_dd)
                if m: results.append(m)

    print("\n--- A2. mom_window sweep on best gf/tv/lev ---")
    # Pick a couple promising knob combos to vary mom_window on
    for gf, tv, lev in [(0.97, 0.16, 1.7), (0.99, 0.14, 1.5),
                          (0.97, 0.14, 1.7), (1.00, 0.14, 1.5)]:
        for mw in (63, 126, 252):
            v3 = build_v3(sig, feats, rets,
                            gross_floor=gf, target_vol=tv, lev_x=lev,
                            mom_window=mw)
            full = apply_v4nb_overlays(v3, rets, macro)
            m = report(f"A2: gf={gf} tv={tv} lev={lev} mw={mw}", full, rets, base_dd)
            if m: results.append(m)

    # ── B. Integrated bond/cmdty shorts on V4N-B baseline ────────────────────
    print("\n--- B. Integrated bond/cmdty shorts ---")
    for sp in (0.05, 0.10, 0.15):
        for adxr in (18.0, 22.0, 28.0):
            withshort = add_short_legs(base_v4nb, sig, rets, feats, CAPITAL,
                                         short_classes=("bond", "commodity"),
                                         short_pct=sp, require_adx=adxr)
            m = report(f"B: shorts sp={sp} adx={adxr}", withshort, rets, base_dd)
            if m: results.append(m)

    # ── C. Replace 12% sleeve with conditional sector shorts ─────────────────
    # (build V3 base WITHOUT the sleeve overlay, then add sector shorts)
    print("\n--- C. Sector shorts replace sleeve diversifier ---")

    def build_v3_no_sleeve(**kw):
        base_kw = dict(
            top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
            lev_x=1.5, max_gross=1.0,
            adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
            ac_quota=0.55, gross_floor=0.95,
        )
        base_kw.update(kw)
        s = top_n_adx_momt_ac_sizes(sig, feats, rets, CAPITAL, **base_kw)
        s = defensive_tilt_overlay(s, sig, macro, CAPITAL)
        return s

    sectors = [t for t, c in TICKERS.items()
               if c == "sector_etf" and t in rets.columns]

    def conditional_sector_shorts(sleeve_pct, adx_t=22.0, lookback=63,
                                    bottom_q=0.25, max_shorts=3,
                                    rebal="W-FRI"):
        # Re-use build_conditional_short_sizes inline here
        R = rets[sectors]
        cum = R.rolling(lookback).sum().shift(5)
        adx_m = pd.DataFrame(0.0, index=R.index, columns=sectors)
        for t in sectors:
            if t in feats and "adx" in feats[t].columns:
                adx_m[t] = feats[t]["adx"].reindex(R.index).ffill().fillna(0.0)
        sg = sig.reindex(columns=sectors, fill_value=0.0).reindex(R.index).fillna(0.0)
        sz = pd.DataFrame(0.0, index=R.index, columns=rets.columns)
        days = R.index.to_series().groupby(pd.Grouper(freq=rebal)).max().dropna()
        for d in days:
            if d not in cum.index:
                continue
            row = cum.loc[d].dropna()
            elig = [t for t in row.index
                    if sg.loc[d, t] == 0 and adx_m.loc[d, t] >= adx_t]
            if not elig:
                continue
            qv = row.quantile(bottom_q)
            bot = [t for t in elig if row[t] <= qv]
            if not bot:
                continue
            worst = sorted(bot, key=lambda x: row[x])[:max_shorts]
            each = (sleeve_pct * CAPITAL) / len(worst)
            for t in worst:
                sz.loc[d, t] = -each
        return sz.replace(0.0, np.nan).ffill().fillna(0.0)

    for sp in (0.08, 0.12, 0.15):
        v3ns = build_v3_no_sleeve()
        ss   = conditional_sector_shorts(sleeve_pct=sp)
        v3ns = v3ns * (1.0 - sp)
        combined = v3ns.add(ss, fill_value=0.0)
        gross = combined.abs().sum(axis=1).replace(0, np.nan)
        cap_scale = (CAPITAL / gross).clip(upper=1.0).fillna(1.0)
        combined = combined.multiply(cap_scale, axis=0)
        full = apply_v4nb_overlays(combined, rets, macro)
        m = report(f"C: sector-shorts replace sleeve, sp={sp}", full, rets, base_dd)
        if m: results.append(m)

    # ── D. Vol-conditional lev boost on V4N-B baseline ───────────────────────
    print("\n--- D. Vol-conditional leverage boost in calm regimes ---")
    for boost in (1.05, 1.10, 1.15, 1.20):
        for cz in (-0.5, -1.0, -1.5):
            boosted = add_vol_lev_boost(base_v4nb, macro,
                                          boost=boost, calm_z=cz)
            m = report(f"D: boost={boost} calm_z={cz}", boosted, rets, base_dd)
            if m: results.append(m)

    # ── Combinations: best-of-A + B + D ──────────────────────────────────────
    print("\n--- E. Combos (best knob settings + shorts + boost) ---")
    # Take top-3 by AnnRet (with DD <= baseline) from A
    a_results = [r for r in results if r and r["name"].startswith("A:")
                  and r["mdd"] >= base_dd]
    a_results.sort(key=lambda x: -x["ann"])
    top_a = a_results[:3]

    for ar in top_a:
        # Re-parse the knob string
        parts = ar["name"].split()
        kw = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=")
                kw[{"gf":"gross_floor","tv":"target_vol","lev":"lev_x"}.get(k,k)] = float(v)
        v3 = build_v3(sig, feats, rets, **kw)
        full = apply_v4nb_overlays(v3, rets, macro)
        for boost in (1.10, 1.15):
            for cz in (-0.5, -1.0):
                combo = add_vol_lev_boost(full, macro, boost=boost, calm_z=cz)
                m = report(
                    f"E: {ar['name'][3:]} + boost={boost} cz={cz}",
                    combo, rets, base_dd
                )
                if m: results.append(m)

    # ── Leaderboards ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}\nBaseline V4N-B: Sharpe {base_m['sharpe']:.2f}  "
          f"AnnRet {base_m['ann']*100:.2f}%  DD {base_m['mdd']*100:.2f}%  "
          f"Calmar {base_m['calmar']:.2f}\n{'='*80}")

    print("\nTop 10 by AnnRet (DD <= V4N-B baseline):")
    safe = [r for r in results if r and r["mdd"] >= base_dd]
    safe.sort(key=lambda x: -x["ann"])
    for r in safe[:10]:
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    print("\nTop 10 by Sharpe overall:")
    by_sh = [r for r in results if r]
    by_sh.sort(key=lambda x: -x["sharpe"])
    for r in by_sh[:10]:
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    print("\nTop 10 by Calmar (DD <= -3%):")
    by_cal = [r for r in results if r and r["mdd"] <= -0.03]
    by_cal.sort(key=lambda x: -x["calmar"])
    for r in by_cal[:10]:
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
