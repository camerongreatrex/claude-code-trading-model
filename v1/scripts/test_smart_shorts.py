"""
Smart-shorts research on top of V4N-B production stack.

Goal: integrate shorts that DON'T kill AnnRet.  Prior naive shorts (always-on
bond/cmdty, conditional sector) all dropped AnnRet 2-4pp because they fight
structural equity premium.  Try:

  F. Bear-regime equity shorts  : short SPY/QQQ only after N consecutive
                                   days of signal_multi==0 (confirmed bear)
  G. Death-cross shorts          : short ticker when SMA50 < SMA200
  H. VIX-spike tail hedge        : short SPY only when vix_zscore > +1
  I. RSI-overbought breakdown    : short when rsi_14 > 70 and short-term
                                   momentum (mom_5) turns negative
  J. Donchian breakdown          : short when Close < donchian_low_20
  K. Combos                      : Family D vol-boost + best smart short

All shorts are sized small (<=10% sleeve), beta-allocated from the long book
(longs reduced by sleeve_pct), combined gross capped at 1.0× capital.

Walk-forward: skip first OOS_WARMUP=756 days for headline metrics.
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
RISK_PER_TRADE = 0.005


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


def build_v3(sig, feats, rets, **kw):
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
    out = profit_take_overlay(sizes, rets, lookback=10, sigma_thresh=1.5,
                                 scale=0.7, max_gross=1.0, capital=CAPITAL)
    out = cond_vol_carry_overlay(out, macro,
                                   fear_z=1.5, roc_days=5, fear_mult=0.5)
    return out


def add_vol_lev_boost(sizes, macro, boost=1.15, calm_z=-0.5,
                       max_gross=1.0, capital=CAPITAL):
    if "vix_zscore" not in macro.columns:
        return sizes
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    scale = pd.Series(1.0, index=sizes.index)
    scale[z <= calm_z] = boost
    out = sizes.multiply(scale, axis=0)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = ((max_gross * capital) / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


def _atr_dollar_size(feat, idx, capital, cap_per_name=0.05):
    """Standard ATR-based dollar size for a name (mirrors live sizer)."""
    atr = feat["atr_14"].reindex(idx).ffill()
    cl  = feat["Close"].reindex(idx).ffill().replace(0, np.nan).ffill()
    sz  = (capital * RISK_PER_TRADE / atr.replace(0, np.nan) * cl).fillna(0.0)
    return sz.clip(upper=capital * cap_per_name)


def _layer_short(longs: pd.DataFrame, short_dollars: pd.DataFrame,
                  short_pct: float, capital: float = CAPITAL) -> pd.DataFrame:
    """Reduce longs by short_pct, add short_dollars, cap combined gross at 1×."""
    out = longs * (1.0 - short_pct)
    out = out.add(short_dollars, fill_value=0.0)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


# ── F. Bear-regime equity shorts ────────────────────────────────────────────
def bear_regime_shorts(rets, feats, sig, hedges=("SPY", "QQQ"),
                        consec_days=5, short_pct=0.08, cap_per_name=0.05):
    """Short hedges only when their signal has been 0 for N consecutive days."""
    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    for t in hedges:
        if t not in feats or t not in sig.columns:
            continue
        s_t = sig[t].reindex(rets.index).fillna(0.0)
        # Trigger after consec_days of zeros (rolling sum of (s==0) >= N)
        zero = (s_t == 0).astype(int)
        active = zero.rolling(consec_days).sum() >= consec_days
        sz = _atr_dollar_size(feats[t], rets.index, CAPITAL, cap_per_name)
        if t in out.columns:
            out[t] = (-sz * active.astype(float)).reindex(rets.index).fillna(0.0)
    # Scale total short notional to short_pct
    short_gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = short_pct * CAPITAL
    scale = (target / short_gross).clip(upper=1.0).fillna(0.0)
    return out.multiply(scale, axis=0)


# ── G. Death-cross shorts ───────────────────────────────────────────────────
def death_cross_shorts(rets, feats, candidates, short_pct=0.08, max_shorts=3,
                        cap_per_name=0.05):
    """Short candidate when SMA50(Close) < SMA200(Close) AND Close < SMA200."""
    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    cands = [t for t in candidates if t in feats and t in rets.columns]
    triggers = {}
    for t in cands:
        cl = feats[t]["Close"].reindex(rets.index).ffill()
        sma50  = cl.rolling(50).mean()
        sma200 = cl.rolling(200).mean()
        active = (sma50 < sma200) & (cl < sma200)
        triggers[t] = active.fillna(False)

    # Each day: rank active candidates by how negative SMA50/SMA200 ratio is,
    # take top max_shorts (most-broken trends)
    trig_df = pd.DataFrame(triggers)
    for t in cands:
        cl = feats[t]["Close"].reindex(rets.index).ffill()
        sma200 = cl.rolling(200).mean().replace(0, np.nan)
        sz = _atr_dollar_size(feats[t], rets.index, CAPITAL, cap_per_name)
        # Distance below 200d (more negative = worse)
        dist = (cl / sma200) - 1.0
        if t in out.columns:
            out[t] = (-sz * trig_df[t].astype(float)).reindex(rets.index).fillna(0.0)

    # Cap to max_shorts per day (keep N most negative-distance)
    if max_shorts < len(cands):
        for d in out.index:
            row = out.loc[d]
            actives = [c for c in cands if row.get(c, 0) < 0]
            if len(actives) > max_shorts:
                # Keep ones with lowest Close/SMA200 ratio
                ratios = {c: feats[c]["Close"].get(d, np.nan) /
                              feats[c]["Close"].rolling(200).mean().get(d, np.nan)
                          if d in feats[c].index else 1.0
                          for c in actives}
                worst = sorted(actives, key=lambda c: ratios.get(c, 1.0))[:max_shorts]
                drop = [c for c in actives if c not in worst]
                out.loc[d, drop] = 0.0

    short_gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = short_pct * CAPITAL
    scale = (target / short_gross).clip(upper=1.0).fillna(0.0)
    return out.multiply(scale, axis=0)


# ── H. VIX-spike tail hedge ─────────────────────────────────────────────────
def vix_spike_hedge(rets, feats, macro, hedges=("SPY", "QQQ"),
                     z_thresh=1.0, short_pct=0.08, cap_per_name=0.05):
    """Short SPY/QQQ when vix_zscore exceeds z_thresh."""
    if "vix_zscore" not in macro.columns:
        return pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    z = macro["vix_zscore"].reindex(rets.index).ffill().bfill()
    active = z >= z_thresh

    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    for t in hedges:
        if t not in feats or t not in rets.columns:
            continue
        sz = _atr_dollar_size(feats[t], rets.index, CAPITAL, cap_per_name)
        if t in out.columns:
            out[t] = (-sz * active.astype(float)).reindex(rets.index).fillna(0.0)

    short_gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = short_pct * CAPITAL
    scale = (target / short_gross).clip(upper=1.0).fillna(0.0)
    return out.multiply(scale, axis=0)


# ── L. Proportional VIX-spike hedge (size scales with z magnitude) ───────────
def vix_proportional_hedge(rets, feats, macro, hedges=("SPY", "QQQ"),
                             z_floor=0.5, z_full=2.0, max_short_pct=0.06,
                             cap_per_name=0.05):
    """Linearly scale short notional from 0 at z=z_floor to max_short_pct at z=z_full."""
    if "vix_zscore" not in macro.columns:
        return pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    z = macro["vix_zscore"].reindex(rets.index).ffill().bfill()
    intensity = ((z - z_floor) / (z_full - z_floor)).clip(0.0, 1.0)
    target_pct = intensity * max_short_pct

    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    for t in hedges:
        if t not in feats or t not in rets.columns:
            continue
        sz = _atr_dollar_size(feats[t], rets.index, CAPITAL, cap_per_name)
        if t in out.columns:
            out[t] = (-sz * (target_pct > 0).astype(float)).reindex(rets.index).fillna(0.0)
    short_gross = out.abs().sum(axis=1).replace(0, np.nan)
    target_dollars = target_pct * CAPITAL
    scale = (target_dollars / short_gross).clip(upper=1.0).fillna(0.0)
    return out.multiply(scale, axis=0)


# ── M. VIX term-structure backwardation hedge ───────────────────────────────
def vix_term_hedge(rets, feats, macro, hedges=("SPY", "QQQ"),
                    term_z_thresh=1.0, short_pct=0.04, cap_per_name=0.05):
    """Short when vix_term_zscore (vix9d/vix backwardation) > thresh — strong fear."""
    col = "vix_term_zscore" if "vix_term_zscore" in macro.columns else None
    if col is None:
        # macro doesn't have it; try synthesizing from ratio
        if "vix_term_ratio" in macro.columns:
            r = macro["vix_term_ratio"].reindex(rets.index).ffill().bfill()
            z_synth = (r - r.rolling(252, min_periods=60).mean()) / \
                        r.rolling(252, min_periods=60).std()
        else:
            return pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    else:
        z_synth = macro[col].reindex(rets.index).ffill().bfill()
    active = z_synth >= term_z_thresh
    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    for t in hedges:
        if t not in feats or t not in rets.columns:
            continue
        sz = _atr_dollar_size(feats[t], rets.index, CAPITAL, cap_per_name)
        if t in out.columns:
            out[t] = (-sz * active.astype(float)).reindex(rets.index).fillna(0.0)
    short_gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = short_pct * CAPITAL
    scale = (target / short_gross).clip(upper=1.0).fillna(0.0)
    return out.multiply(scale, axis=0)


# ── N. Macro-gated breakdown shorts (J + macro fear filter) ─────────────────
def gated_breakdown_shorts(rets, feats, macro, candidates,
                             vix_z_min=0.0, short_pct=0.08, cap_per_name=0.05):
    """Donchian-low breakdowns ONLY when vix_zscore >= vix_z_min (fear gate)."""
    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    if "vix_zscore" not in macro.columns:
        return out
    z = macro["vix_zscore"].reindex(rets.index).ffill().bfill()
    gate = (z >= vix_z_min).astype(float)
    cands = [t for t in candidates if t in feats and t in rets.columns
             and "donchian_low_20" in feats[t].columns]
    for t in cands:
        cl  = feats[t]["Close"].reindex(rets.index).ffill()
        dlo = feats[t]["donchian_low_20"].reindex(rets.index).ffill()
        active = (cl <= dlo).astype(float) * gate
        sz = _atr_dollar_size(feats[t], rets.index, CAPITAL, cap_per_name)
        if t in out.columns:
            out[t] = (-sz * active).reindex(rets.index).fillna(0.0)
    short_gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = short_pct * CAPITAL
    scale = (target / short_gross).clip(upper=1.0).fillna(0.0)
    return out.multiply(scale, axis=0)


# ── O. Asymmetric vol boost — boost in calm + de-boost in fear ──────────────
def asym_vol_boost(sizes, macro, calm_boost=1.15, calm_z=-0.5,
                     fear_cut=0.85, fear_z=1.0, max_gross=1.0, capital=CAPITAL):
    """Scale UP in calm, DOWN in fear (proactive risk-off)."""
    if "vix_zscore" not in macro.columns:
        return sizes
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    scale = pd.Series(1.0, index=sizes.index)
    scale[z <= calm_z] = calm_boost
    scale[z >= fear_z] = fear_cut
    out = sizes.multiply(scale, axis=0)
    gross = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = ((max_gross * capital) / gross).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


# ── I. RSI-overbought breakdown shorts ──────────────────────────────────────
def rsi_breakdown_shorts(rets, feats, candidates, rsi_thresh=70.0,
                          short_pct=0.06, max_shorts=2, cap_per_name=0.04):
    """Short when rsi_14 > thresh AND mom_5 < 0 (overbought + losing steam)."""
    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    cands = [t for t in candidates if t in feats and t in rets.columns
             and "rsi_14" in feats[t].columns and "mom_5" in feats[t].columns]
    for t in cands:
        rsi = feats[t]["rsi_14"].reindex(rets.index).ffill()
        m5  = feats[t]["mom_5"].reindex(rets.index).ffill()
        active = (rsi > rsi_thresh) & (m5 < 0)
        sz = _atr_dollar_size(feats[t], rets.index, CAPITAL, cap_per_name)
        if t in out.columns:
            out[t] = (-sz * active.astype(float)).reindex(rets.index).fillna(0.0)

    short_gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = short_pct * CAPITAL
    scale = (target / short_gross).clip(upper=1.0).fillna(0.0)
    return out.multiply(scale, axis=0)


# ── J. Donchian breakdown shorts ────────────────────────────────────────────
def donchian_breakdown_shorts(rets, feats, candidates, short_pct=0.08,
                                max_shorts=3, cap_per_name=0.05):
    """Short when Close < donchian_low_20 (confirmed downside breakout)."""
    out = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    cands = [t for t in candidates if t in feats and t in rets.columns
             and "donchian_low_20" in feats[t].columns]
    for t in cands:
        cl  = feats[t]["Close"].reindex(rets.index).ffill()
        dlo = feats[t]["donchian_low_20"].reindex(rets.index).ffill()
        active = cl <= dlo
        sz = _atr_dollar_size(feats[t], rets.index, CAPITAL, cap_per_name)
        if t in out.columns:
            out[t] = (-sz * active.astype(float)).reindex(rets.index).fillna(0.0)

    short_gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = short_pct * CAPITAL
    scale = (target / short_gross).clip(upper=1.0).fillna(0.0)
    return out.multiply(scale, axis=0)


def report(name, sizes, rets, baseline_dd=None, baseline_ann=None):
    pr = portfolio_returns(sizes, rets).dropna()
    m = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
    if not m:
        print(f"{name:<60} insufficient data")
        return None
    flag = ""
    if baseline_dd is not None and m["mdd"] >= baseline_dd:
        flag += " *DD"
    if baseline_ann is not None and m["ann"] >= baseline_ann:
        flag += " *AnnRet"
    print(f"{name:<60} {m['sharpe']:>5.2f} {m['ann']*100:>6.2f}% "
          f"{m['mdd']*100:>6.2f}% {m['calmar']:>5.2f} {m['vol']*100:>5.1f}%{flag}")
    return m | {"name": name}


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in "
          f"{time.time()-t0:.1f}s\n")

    stocks = [t for t, c in TICKERS.items() if c == "stock" and t in rets.columns]
    sectors = [t for t, c in TICKERS.items() if c == "sector_etf" and t in rets.columns]
    full_universe = stocks + sectors

    print(f"{'Variant':<60} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    results = []

    # Baseline V4N-B
    base_v3 = build_v3(sig, feats, rets)
    base_v4nb = apply_v4nb_overlays(base_v3, rets, macro)
    base_m = report("V4N-B baseline", base_v4nb, rets)
    results.append(base_m)
    base_dd, base_ann = base_m["mdd"], base_m["ann"]

    # Family D winner reference (boost=1.15, calm_z=-0.5)
    d_best = add_vol_lev_boost(base_v4nb, macro, boost=1.15, calm_z=-0.5)
    d_m = report("D winner: boost=1.15 calm_z=-0.5", d_best, rets, base_dd, base_ann)
    results.append(d_m)
    d_dd, d_ann = d_m["mdd"], d_m["ann"]

    # ── F. Bear-regime equity shorts ─────────────────────────────────────────
    print("\n--- F. Bear-regime equity shorts (signal=0 for N days) ---")
    for nd in (3, 5, 8, 12):
        for sp in (0.04, 0.06, 0.08, 0.10):
            sh = bear_regime_shorts(rets, feats, sig,
                                      hedges=("SPY", "QQQ"),
                                      consec_days=nd, short_pct=sp)
            combined = _layer_short(base_v4nb, sh, sp)
            m = report(f"F: spy/qqq nd={nd} sp={sp}", combined, rets, base_dd, base_ann)
            if m: results.append(m)

    # ── G. Death-cross shorts ────────────────────────────────────────────────
    print("\n--- G. Death-cross shorts (SMA50<SMA200 & Close<SMA200) ---")
    for univ_name, univ in [("sectors", sectors), ("full22", full_universe)]:
        for sp in (0.05, 0.08, 0.12):
            for ms in (2, 3, 5):
                sh = death_cross_shorts(rets, feats, univ,
                                          short_pct=sp, max_shorts=ms)
                combined = _layer_short(base_v4nb, sh, sp)
                m = report(f"G: {univ_name} sp={sp} ms={ms}", combined, rets, base_dd, base_ann)
                if m: results.append(m)

    # ── H. VIX-spike tail hedge ──────────────────────────────────────────────
    print("\n--- H. VIX-spike tail hedge (vix_z > thresh) ---")
    for zt in (0.5, 1.0, 1.5, 2.0):
        for sp in (0.04, 0.06, 0.08, 0.10):
            sh = vix_spike_hedge(rets, feats, macro,
                                   hedges=("SPY", "QQQ"),
                                   z_thresh=zt, short_pct=sp)
            combined = _layer_short(base_v4nb, sh, sp)
            m = report(f"H: vix_z>{zt} sp={sp}", combined, rets, base_dd, base_ann)
            if m: results.append(m)

    # ── I. RSI-overbought breakdown shorts ───────────────────────────────────
    print("\n--- I. RSI-overbought breakdown (rsi>thresh & mom_5<0) ---")
    for rt in (65.0, 70.0, 75.0):
        for sp in (0.04, 0.06, 0.08):
            sh = rsi_breakdown_shorts(rets, feats, full_universe,
                                        rsi_thresh=rt, short_pct=sp)
            combined = _layer_short(base_v4nb, sh, sp)
            m = report(f"I: rsi>{rt} sp={sp}", combined, rets, base_dd, base_ann)
            if m: results.append(m)

    # ── J. Donchian breakdown shorts ─────────────────────────────────────────
    print("\n--- J. Donchian breakdown shorts (Close < donchian_low_20) ---")
    for univ_name, univ in [("sectors", sectors), ("full22", full_universe)]:
        for sp in (0.05, 0.08, 0.10):
            sh = donchian_breakdown_shorts(rets, feats, univ, short_pct=sp)
            combined = _layer_short(base_v4nb, sh, sp)
            m = report(f"J: {univ_name} donch sp={sp}", combined, rets, base_dd, base_ann)
            if m: results.append(m)

    # ── K. Combos: Family D vol-boost + best smart short ─────────────────────
    print("\n--- K. Combos: D-boost + best-of-family short ---")
    # Run D-boost variants alone first to confirm
    for boost in (1.10, 1.15, 1.20):
        for cz in (-0.3, -0.5, -1.0):
            d = add_vol_lev_boost(base_v4nb, macro, boost=boost, calm_z=cz)
            m = report(f"K-D: boost={boost} cz={cz}", d, rets, base_dd, base_ann)
            if m: results.append(m)

    # Pick top short candidates from F/H (likely the cleanest hedges) and combine
    # with D-boost.  Strategy: D adds AnnRet in calm; short adds DD insurance in stress.
    print("\n--- K2. D-boost + F-bear-shorts combo ---")
    for boost in (1.10, 1.15):
        for cz in (-0.5,):
            d = add_vol_lev_boost(base_v4nb, macro, boost=boost, calm_z=cz)
            for nd in (5, 8):
                for sp in (0.05, 0.08):
                    sh = bear_regime_shorts(rets, feats, sig,
                                              hedges=("SPY", "QQQ"),
                                              consec_days=nd, short_pct=sp)
                    combined = _layer_short(d, sh, sp)
                    m = report(f"K2: D{boost}/{cz} + F nd={nd} sp={sp}",
                                combined, rets, base_dd, base_ann)
                    if m: results.append(m)

    print("\n--- K3. D-boost + H-vix-hedge combo ---")
    for boost in (1.10, 1.15):
        for cz in (-0.5,):
            d = add_vol_lev_boost(base_v4nb, macro, boost=boost, calm_z=cz)
            for zt in (1.0, 1.5):
                for sp in (0.04, 0.06, 0.08):
                    sh = vix_spike_hedge(rets, feats, macro,
                                           hedges=("SPY", "QQQ"),
                                           z_thresh=zt, short_pct=sp)
                    combined = _layer_short(d, sh, sp)
                    m = report(f"K3: D{boost}/{cz} + H z>{zt} sp={sp}",
                                combined, rets, base_dd, base_ann)
                    if m: results.append(m)

    # ── K4. Refined H sweep — tiny short_pct + multi-hedge ───────────────────
    print("\n--- K4. Refined H sweep (tiny sp, +IWM hedge) ---")
    for hedges in [("SPY",), ("SPY", "QQQ"), ("SPY", "QQQ", "IWM")]:
        for zt in (0.5, 1.0):
            for sp in (0.02, 0.03, 0.04):
                sh = vix_spike_hedge(rets, feats, macro, hedges=hedges,
                                       z_thresh=zt, short_pct=sp)
                combined = _layer_short(base_v4nb, sh, sp)
                hname = "+".join(hedges)
                m = report(f"K4: H {hname} z>{zt} sp={sp}", combined, rets,
                            base_dd, base_ann)
                if m: results.append(m)

    # ── K5. D-boost + tiny H combo ──────────────────────────────────────────
    print("\n--- K5. D-boost + tiny H combo (push Pareto) ---")
    for boost in (1.10, 1.15, 1.20, 1.25):
        for cz in (-0.3, -0.5):
            d = add_vol_lev_boost(base_v4nb, macro, boost=boost, calm_z=cz)
            for zt in (0.5, 1.0):
                for sp in (0.02, 0.03, 0.04):
                    sh = vix_spike_hedge(rets, feats, macro,
                                           hedges=("SPY", "QQQ"),
                                           z_thresh=zt, short_pct=sp)
                    combined = _layer_short(d, sh, sp)
                    m = report(f"K5: D{boost}/{cz} + H z>{zt} sp={sp}",
                                combined, rets, base_dd, base_ann)
                    if m: results.append(m)

    # ── L. Proportional VIX-spike hedge ─────────────────────────────────────
    print("\n--- L. Proportional VIX hedge (size scales w/ z magnitude) ---")
    for zf in (0.0, 0.5):
        for zfu in (1.5, 2.0, 2.5):
            for msp in (0.04, 0.06, 0.08):
                sh = vix_proportional_hedge(rets, feats, macro,
                                              hedges=("SPY", "QQQ"),
                                              z_floor=zf, z_full=zfu,
                                              max_short_pct=msp)
                # Use msp as the layered short_pct (avg long reduction approx)
                combined = _layer_short(base_v4nb, sh, msp)
                m = report(f"L: zf={zf} zfu={zfu} max={msp}", combined, rets,
                            base_dd, base_ann)
                if m: results.append(m)

    # ── M. VIX term-structure hedge ─────────────────────────────────────────
    print("\n--- M. VIX term-structure backwardation hedge ---")
    for tz in (0.5, 1.0, 1.5):
        for sp in (0.03, 0.04, 0.06):
            sh = vix_term_hedge(rets, feats, macro,
                                  term_z_thresh=tz, short_pct=sp)
            combined = _layer_short(base_v4nb, sh, sp)
            m = report(f"M: termz>{tz} sp={sp}", combined, rets, base_dd, base_ann)
            if m: results.append(m)

    # ── N. Macro-gated breakdown shorts ─────────────────────────────────────
    print("\n--- N. Donchian breakdowns gated by VIX fear ---")
    for vz in (0.0, 0.5, 1.0):
        for sp in (0.04, 0.06, 0.08):
            sh = gated_breakdown_shorts(rets, feats, macro, full_universe,
                                          vix_z_min=vz, short_pct=sp)
            combined = _layer_short(base_v4nb, sh, sp)
            m = report(f"N: vz>={vz} sp={sp}", combined, rets, base_dd, base_ann)
            if m: results.append(m)

    # ── O. Asymmetric vol scaling (boost calm + cut fear) ───────────────────
    print("\n--- O. Asymmetric vol scaling ---")
    for cb in (1.10, 1.15, 1.20):
        for fc in (0.70, 0.80, 0.90):
            for fz in (1.0, 1.5):
                a = asym_vol_boost(base_v4nb, macro, calm_boost=cb,
                                     calm_z=-0.5, fear_cut=fc, fear_z=fz)
                m = report(f"O: cb={cb} fc={fc} fz={fz}", a, rets, base_dd, base_ann)
                if m: results.append(m)

    # ── P. Best combos: O + L (asym + proportional hedge) ───────────────────
    print("\n--- P. Asym vol + proportional hedge combo ---")
    for cb in (1.15, 1.20):
        for fc in (0.80, 0.90):
            a = asym_vol_boost(base_v4nb, macro, calm_boost=cb,
                                 calm_z=-0.5, fear_cut=fc, fear_z=1.0)
            for zf in (0.0, 0.5):
                for msp in (0.03, 0.04, 0.06):
                    sh = vix_proportional_hedge(rets, feats, macro,
                                                  hedges=("SPY", "QQQ"),
                                                  z_floor=zf, z_full=2.0,
                                                  max_short_pct=msp)
                    combined = _layer_short(a, sh, msp)
                    m = report(f"P: O(cb={cb},fc={fc}) + L zf={zf} msp={msp}",
                                combined, rets, base_dd, base_ann)
                    if m: results.append(m)

    # ── R. O (asym vol) + tiny H (vix-spike short) — natural combo ──────────
    print("\n--- R. O + H combo (asym scaling + tiny vix-spike short) ---")
    for cb in (1.15, 1.20):
        for fc in (0.85, 0.90):
            o = asym_vol_boost(base_v4nb, macro, calm_boost=cb, calm_z=-0.5,
                                 fear_cut=fc, fear_z=1.0)
            for sp in (0.02, 0.03, 0.04):
                sh = vix_spike_hedge(rets, feats, macro,
                                       hedges=("SPY", "QQQ"),
                                       z_thresh=0.5, short_pct=sp)
                combined = _layer_short(o, sh, sp)
                m = report(f"R: O(cb={cb},fc={fc}) + H sp={sp}",
                            combined, rets, base_dd, base_ann)
                if m: results.append(m)

    # ── Q. Best D+H + tiny term-hedge stack ─────────────────────────────────
    print("\n--- Q. D-boost + H-spike + M-term-hedge stack ---")
    for boost in (1.15, 1.20):
        for cz in (-0.3, -0.5):
            d = add_vol_lev_boost(base_v4nb, macro, boost=boost, calm_z=cz)
            for sp_h in (0.02, 0.03):
                sh_h = vix_spike_hedge(rets, feats, macro,
                                         hedges=("SPY", "QQQ"),
                                         z_thresh=0.5, short_pct=sp_h)
                stage1 = _layer_short(d, sh_h, sp_h)
                for sp_m in (0.02, 0.03):
                    sh_m = vix_term_hedge(rets, feats, macro,
                                            term_z_thresh=1.0, short_pct=sp_m)
                    combined = _layer_short(stage1, sh_m, sp_m)
                    m = report(f"Q: D{boost}/{cz} + H sp={sp_h} + M sp={sp_m}",
                                combined, rets, base_dd, base_ann)
                    if m: results.append(m)

    # ── Leaderboards ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"V4N-B baseline: Sh {base_m['sharpe']:.2f}  Ann {base_m['ann']*100:.2f}%  "
          f"DD {base_m['mdd']*100:.2f}%  Cal {base_m['calmar']:.2f}")
    print(f"D winner ref:   Sh {d_m['sharpe']:.2f}  Ann {d_m['ann']*100:.2f}%  "
          f"DD {d_m['mdd']*100:.2f}%  Cal {d_m['calmar']:.2f}")
    print('='*80)

    print("\nTop 15 by AnnRet (DD <= V4N-B baseline DD):")
    safe = [r for r in results if r and r["mdd"] >= base_dd]
    safe.sort(key=lambda x: -x["ann"])
    for r in safe[:15]:
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    print("\nTop 15 by Calmar (DD <= -3.5%):")
    by_cal = [r for r in results if r and r["mdd"] <= -0.035]
    by_cal.sort(key=lambda x: -x["calmar"])
    for r in by_cal[:15]:
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    print("\nTop 15 by Sharpe:")
    by_sh = [r for r in results if r]
    by_sh.sort(key=lambda x: -x["sharpe"])
    for r in by_sh[:15]:
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    # Beats both baseline AnnRet AND DD
    print("\nVariants beating BOTH baseline AnnRet AND baseline DD:")
    pareto = [r for r in results if r and r["ann"] >= base_ann
              and r["mdd"] >= base_dd
              and r["name"] not in ("V4N-B baseline",)]
    pareto.sort(key=lambda x: -x["sharpe"])
    if not pareto:
        print("  (none — only AnnRet improvements come at DD cost)")
    for r in pareto[:15]:
        print(f"  {r['name']:<60} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
