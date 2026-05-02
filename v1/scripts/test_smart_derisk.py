"""
Non-ML smart-de-risking sweep.  Goal: capture V4B-like Calmar / DD lift
without ML and without bleeding AnnRet (plain vol-carry costs ~2pp Ann).

Overlays tested on top of V3 (top11_adx22_momt_ac55_cap1 + 12% sleeve):

  1. CONDITIONAL VOL-CARRY — scale down only when VIX-z is BOTH high AND
     rising 5d.  Avoids cutting after the storm.
  2. BACKWARDATION OVERLAY — scale down when vol_backwardation flag is on
     (vix9d > vix, ~23% of days).  Hard regime signal.
  3. YIELD-CURVE OVERLAY — scale down when curve_inverted (recession bias).
  4. PORTFOLIO-DD DE-RISK — track running DD; cut gross to X% if DD exceeds
     trigger; restore on recovery.
  5. ADAPTIVE SLEEVE — sleeve_pct grows in fear regime (more defensive).
  6. CALM-REGIME CAP BOOST — raise per-name cap from 0.12 to 0.14 in calm
     regimes only (push AnnRet on safe days).
  7. LOWER GROSS_FLOOR — test 0.85, 0.90 (smaller risk budget; cleaner DD).
  8. PROFIT-TAKING — scale individual position 0.7x when up >2σ over 20d.

For each overlay we report OOS Sh/Ann/MaxDD/Calmar.  Zero-leverage cap
enforced after every overlay.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import v1.portfolio.portfolio as _pp
from v1.portfolio.portfolio import (
    top_n_adx_momt_ac_sizes,
    diversifier_sleeve_overlay,
    portfolio_returns,
    defensive_tilt_overlay,
    _get_macro,
    SIGNAL_DIR, FEATURE_DIR, CAPITAL,
)
from v1.pipeline.data_pipeline import TICKER_LIST
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


def make_v3_sizer(cap_pct: float = 0.12, gross_floor: float = 0.95,
                   sleeve_pct: float = 0.12):
    base_kw = dict(
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=gross_floor,
    )

    def sizer(sig, feats, rets, macro):
        old = _pp.MAX_POSITION_PCT
        _pp.MAX_POSITION_PCT = cap_pct
        try:
            s = top_n_adx_momt_ac_sizes(sig, feats, rets, CAPITAL, **base_kw)
            s = defensive_tilt_overlay(s, sig, macro, CAPITAL)
            s = diversifier_sleeve_overlay(
                s, CAPITAL,
                sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
                sleeve_pct=sleeve_pct,
            )
        finally:
            _pp.MAX_POSITION_PCT = old
        return s

    return sizer


def zero_lev_clip(sizes, capital=CAPITAL):
    g = sizes.abs().sum(axis=1).replace(0, np.nan)
    sc = (capital / g).clip(upper=1.0).fillna(1.0)
    return sizes.multiply(sc, axis=0)


# -----------------------------------------------------------------------
# Overlays
# -----------------------------------------------------------------------
def cond_vol_carry(sizes, macro, fear_z=1.5, roc_days=5,
                    fear_mult=0.5):
    """Scale down only when vix_zscore >= fear_z AND VIX rising over roc_days."""
    if "vix_zscore" not in macro.columns or "vix" not in macro.columns:
        return sizes
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    roc = macro["vix"].reindex(sizes.index).ffill().diff(roc_days)
    mult = pd.Series(1.0, index=sizes.index)
    mask = (z >= fear_z) & (roc > 0)
    mult[mask] = fear_mult
    return sizes.multiply(mult, axis=0)


def backwardation_overlay(sizes, macro, mult=0.7):
    """Scale down by `mult` whenever vix9d > vix (vol backwardation)."""
    if "vol_backwardation" not in macro.columns:
        return sizes
    flag = macro["vol_backwardation"].reindex(sizes.index).ffill().fillna(0).astype(bool)
    m = pd.Series(1.0, index=sizes.index)
    m[flag.values] = mult
    return sizes.multiply(m, axis=0)


def curve_overlay(sizes, macro, mult=0.7):
    """Scale down by `mult` when curve_inverted."""
    if "curve_inverted" not in macro.columns:
        return sizes
    flag = macro["curve_inverted"].reindex(sizes.index).ffill().fillna(0).astype(bool)
    m = pd.Series(1.0, index=sizes.index)
    m[flag.values] = mult
    return sizes.multiply(m, axis=0)


def dd_derisk(sizes, rets, dd_trigger=-0.04, derisk_mult=0.6,
               restore_dd=-0.02, capital=CAPITAL):
    """
    Walk forward through the strategy's equity curve.  Each day:
      - compute current strategy DD from running peak
      - if DD <= dd_trigger : scale tomorrow's sizes by derisk_mult
      - stay derisked until DD recovers above restore_dd
    Avoids look-ahead by computing DD with cumulative T-1 returns.
    """
    pr = portfolio_returns(sizes, rets).dropna()
    eq = (1 + pr).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    state = pd.Series(1.0, index=pr.index)
    derisked = False
    for d, ddv in dd.items():
        if not derisked and ddv <= dd_trigger:
            derisked = True
        elif derisked and ddv >= restore_dd:
            derisked = False
        state.loc[d] = derisk_mult if derisked else 1.0
    # Apply state to NEXT day's sizes (no look-ahead)
    state_lag = state.shift(1).reindex(sizes.index).fillna(1.0)
    return sizes.multiply(state_lag, axis=0)


def adaptive_sleeve(sig, feats, rets, macro, base_pct=0.10, fear_pct=0.20,
                     fear_z=1.0, capital=CAPITAL):
    """
    Build V3 with daily-varying sleeve: base_pct in calm, fear_pct in fear
    (vix_zscore >= fear_z).  Implementation: build two V3s with different
    sleeve sizes, blend per-day by regime mask.
    """
    sizer_calm = make_v3_sizer(sleeve_pct=base_pct)
    sizer_fear = make_v3_sizer(sleeve_pct=fear_pct)
    sc = sizer_calm(sig, feats, rets, macro)
    sf = sizer_fear(sig, feats, rets, macro)
    z = macro["vix_zscore"].reindex(sc.index).ffill().bfill()
    fear_mask = (z >= fear_z).astype(float)
    out = sc.multiply(1 - fear_mask, axis=0).add(
            sf.multiply(fear_mask, axis=0), fill_value=0.0)
    return out


def calm_cap_boost(sig, feats, rets, macro, calm_z=-0.5,
                    calm_cap=0.14, normal_cap=0.12):
    """
    Use calm_cap when vix_zscore <= calm_z, else normal_cap.  Build two V3s
    with different caps and blend.  In calm regimes higher cap may capture
    more upside without breaching zero-lev.
    """
    sizer_normal = make_v3_sizer(cap_pct=normal_cap)
    sizer_calm   = make_v3_sizer(cap_pct=calm_cap)
    sn = sizer_normal(sig, feats, rets, macro)
    sc = sizer_calm(sig, feats, rets, macro)
    z = macro["vix_zscore"].reindex(sn.index).ffill().bfill()
    calm_mask = (z <= calm_z).astype(float)
    out = sn.multiply(1 - calm_mask, axis=0).add(
            sc.multiply(calm_mask, axis=0), fill_value=0.0)
    return zero_lev_clip(out)


def profit_take(sizes, rets, lookback=20, sigma_thresh=2.0, scale=0.7):
    """
    For each long position, if the name's 20d cum log-return divided by 20d
    rolling std exceeds sigma_thresh, scale that day's position by `scale`
    (take half off the table on parabolic moves).
    """
    out = sizes.copy()
    for t in sizes.columns:
        if t not in rets.columns:
            continue
        cum = rets[t].rolling(lookback).sum()
        std = rets[t].rolling(lookback).std() * np.sqrt(lookback)
        z = (cum / std.replace(0, np.nan)).reindex(out.index).fillna(0)
        bad = (z >= sigma_thresh) & (out[t] > 0)
        col = out[t].values.astype(float)
        col = np.where(bad.values, col * scale, col)
        out[t] = col
    return zero_lev_clip(out)


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------
def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in {time.time()-t0:.1f}s")

    sizer_v3 = make_v3_sizer()
    sizes_v3 = sizer_v3(sig, feats, rets, macro)
    pr_v3 = portfolio_returns(sizes_v3, rets).dropna()
    m_v3 = metrics(pr_v3.iloc[OOS_WARMUP:])

    # Reference: plain vol-carry (calm=0->1.0 fear=+2->0.5)
    def vc(sizes, m):
        if "vix_zscore" not in m.columns:
            return sizes
        z = m["vix_zscore"].reindex(sizes.index).ffill().bfill()
        span = 2.0
        raw = 1.0 + (0.5 - 1.0) * ((z - 0.0) / span).clip(0, 1)
        mu = raw.clip(lower=0.5, upper=1.0)
        return sizes.multiply(mu, axis=0)

    pr_vc = portfolio_returns(vc(sizes_v3, macro), rets).dropna()
    m_vc = metrics(pr_vc.iloc[OOS_WARMUP:])

    print("\n" + "=" * 92)
    print("REFERENCE: V3 baseline & V3 + plain vol-carry (target to beat on Ann/DD)")
    print("=" * 92)
    print(f"{'Variant':<60} {'Sh':>6} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    for n, m in [("V3 baseline", m_v3),
                 ("V3 + plain vol-carry", m_vc)]:
        print(f"{n:<60} {m['sharpe']:>6.2f} "
              f"{m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")
    print("-" * 92)

    # 1. Conditional vol-carry
    print("\n[1] CONDITIONAL VOL-CARRY (scale down only when vix-z high AND rising)")
    for fz, roc, fm in [(1.0, 5, 0.6), (1.5, 5, 0.5), (1.5, 10, 0.5),
                         (1.0, 5, 0.4), (2.0, 5, 0.4)]:
        s = cond_vol_carry(sizes_v3, macro, fear_z=fz, roc_days=roc, fear_mult=fm)
        m = metrics(portfolio_returns(s, rets).dropna().iloc[OOS_WARMUP:])
        print(f"V3 + cvc fz={fz} roc={roc}d mul={fm:<48}".rstrip().ljust(60),
              f"{m['sharpe']:>6.2f} {m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # 2. Backwardation overlay
    print("\n[2] BACKWARDATION OVERLAY (scale down when vix9d > vix)")
    for mu in [0.5, 0.6, 0.7, 0.8]:
        s = backwardation_overlay(sizes_v3, macro, mult=mu)
        m = metrics(portfolio_returns(s, rets).dropna().iloc[OOS_WARMUP:])
        print(f"V3 + bw mul={mu}".ljust(60),
              f"{m['sharpe']:>6.2f} {m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # 3. Yield-curve overlay
    print("\n[3] YIELD-CURVE OVERLAY (scale down when curve_inverted)")
    for mu in [0.6, 0.7, 0.8, 0.9]:
        s = curve_overlay(sizes_v3, macro, mult=mu)
        m = metrics(portfolio_returns(s, rets).dropna().iloc[OOS_WARMUP:])
        print(f"V3 + yc mul={mu}".ljust(60),
              f"{m['sharpe']:>6.2f} {m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # 4. Portfolio DD de-risk
    print("\n[4] PORTFOLIO-DD DE-RISK (cut gross when running DD breaches threshold)")
    for trig, dm, rest in [(-0.03, 0.6, -0.015), (-0.03, 0.7, -0.015),
                            (-0.04, 0.6, -0.02),  (-0.04, 0.5, -0.02),
                            (-0.05, 0.5, -0.025)]:
        s = dd_derisk(sizes_v3, rets, dd_trigger=trig, derisk_mult=dm, restore_dd=rest)
        m = metrics(portfolio_returns(s, rets).dropna().iloc[OOS_WARMUP:])
        print(f"V3 + dd trig={trig} mul={dm} rest={rest}".ljust(60),
              f"{m['sharpe']:>6.2f} {m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # 5. Adaptive sleeve
    print("\n[5] ADAPTIVE SLEEVE (base_pct in calm, fear_pct in fear)")
    for bp, fp, fz in [(0.10, 0.20, 1.0), (0.08, 0.20, 1.0), (0.10, 0.25, 1.5),
                        (0.12, 0.20, 1.0), (0.08, 0.18, 0.5)]:
        s = adaptive_sleeve(sig, feats, rets, macro, base_pct=bp, fear_pct=fp, fear_z=fz)
        m = metrics(portfolio_returns(s, rets).dropna().iloc[OOS_WARMUP:])
        print(f"V3 + as base={bp} fear={fp} fz={fz}".ljust(60),
              f"{m['sharpe']:>6.2f} {m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # 6. Calm-regime cap boost
    print("\n[6] CALM-REGIME CAP BOOST (cap=0.14 in calm, 0.12 normal)")
    for cz, cc in [(-0.5, 0.13), (-0.5, 0.14), (-1.0, 0.14),
                    (0.0, 0.13), (0.0, 0.14)]:
        s = calm_cap_boost(sig, feats, rets, macro, calm_z=cz, calm_cap=cc)
        m = metrics(portfolio_returns(s, rets).dropna().iloc[OOS_WARMUP:])
        print(f"V3 + ccb cz={cz} cap={cc}".ljust(60),
              f"{m['sharpe']:>6.2f} {m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # 7. Lower gross_floor
    print("\n[7] LOWER GROSS_FLOOR (smaller risk budget on calm days)")
    for gf in [0.85, 0.90, 0.95]:
        sizer = make_v3_sizer(gross_floor=gf)
        s = sizer(sig, feats, rets, macro)
        m = metrics(portfolio_returns(s, rets).dropna().iloc[OOS_WARMUP:])
        print(f"V3 gross_floor={gf}".ljust(60),
              f"{m['sharpe']:>6.2f} {m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # 8. Profit taking
    print("\n[8] PROFIT-TAKING (scale to 0.7 if 20d z >= sigma_thresh)")
    for lb, st, sc in [(20, 1.5, 0.7), (20, 2.0, 0.7), (20, 2.5, 0.7),
                        (10, 1.5, 0.7), (10, 2.0, 0.5)]:
        s = profit_take(sizes_v3, rets, lookback=lb, sigma_thresh=st, scale=sc)
        m = metrics(portfolio_returns(s, rets).dropna().iloc[OOS_WARMUP:])
        print(f"V3 + pt lb={lb} sig={st} scl={sc}".ljust(60),
              f"{m['sharpe']:>6.2f} {m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
