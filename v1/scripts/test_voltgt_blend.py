"""
test_voltgt_blend.py
--------------------
Two angles for limiting DD on vol-target sizing:

1. Partial blends — α × vt + (1-α) × atr — keeps the calm-day boost but
   dampens stress-day amplification.
2. Quality filters on entries — drop weak ADX or weak slope entries before
   sizing so the booster only deploys on high-conviction trades.
3. Asymmetric vol target — only scale DOWN in stress, never UP (no boost
   on calm) → similar Sharpe with less DD.

Goal: AnnRet >= 18% with MaxDD better than -10% on 1.5x leverage,
       AnnRet >= 15% with MaxDD better than  -8% on 1.0x leverage.

Run: python -m v1.scripts.test_voltgt_blend
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes, defensive_tilt_overlay,
    portfolio_returns, summarise, MAX_POSITION_PCT, CAPITAL,
)


def _clip(sz: pd.DataFrame) -> pd.DataFrame:
    return sz.clip(-CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT)


def vol_target_atr(signals, features, returns, capital,
                   target_vol=0.13, window=63, scale_min=0.3, scale_max=2.5,
                   asymmetric=False):
    """If asymmetric=True, scaler.clip(upper=1.0) — only cuts in stress, never boosts."""
    base = atr_sizes(signals, features, capital)
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    if asymmetric:
        scaler = (target_vol / realized).clip(scale_min, 1.0).shift(1).fillna(1.0)
    else:
        scaler = (target_vol / realized).clip(scale_min, scale_max).shift(1).fillna(1.0)
    return _clip(base.multiply(scaler, axis=0))


def adx_quality_atr(signals, features, capital, adx_min=15.0, adx_full=30.0):
    """Continuous ADX gate: scale 0 → 1 across [adx_min, adx_full]."""
    base = atr_sizes(signals, features, capital)
    out = base.copy()
    for t in signals.columns:
        if t not in features:
            continue
        adx = features[t].get("adx")
        if adx is None:
            continue
        adx = adx.reindex(signals.index).ffill().fillna(0.0)
        scale = ((adx - adx_min) / (adx_full - adx_min)).clip(0, 1)
        out[t] = base[t] * scale
    return _clip(out)


def vol_target_adx_filter(signals, features, returns, capital,
                          target_vol=0.13, scale_max=2.5,
                          adx_min=15.0, adx_full=30.0):
    """Vol-target sizing on ADX-quality-filtered base."""
    quality_base = adx_quality_atr(signals, features, capital, adx_min, adx_full)
    weights = quality_base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=quality_base.columns)).sum(axis=1)
    realized = port_ret.rolling(63, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    scaler = (target_vol / realized).clip(0.3, scale_max).shift(1).fillna(1.0)
    return _clip(quality_base.multiply(scaler, axis=0))


def lev(sz_base, signals, macro, lev_x=1.5):
    sz = _clip(sz_base * lev_x)
    return defensive_tilt_overlay(sz, signals, macro, CAPITAL)


def run(name, sz, returns):
    ret = portfolio_returns(sz, returns)
    s = summarise(ret, name)
    g = float(sz.abs().sum(axis=1).mean()) / CAPITAL
    print(f"{name:<36} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
          f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  gr {g:>4.2f}")
    return s, g


def main() -> None:
    SIGNAL_DIR  = Path("data/v1/signals")
    FEATURE_DIR = Path("data/v1/features")
    MACRO_DIR   = Path("data/shared/macro")

    print("Loading signals + features...")
    signals_raw = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    features: dict = {}
    returns = pd.DataFrame()
    for t in signals_raw.columns:
        p = FEATURE_DIR / f"{t}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            features[t] = f
            returns[t]  = f["log_return"]
    returns = returns.dropna()
    signals = signals_raw.reindex(returns.index).fillna(0)
    macro_path = MACRO_DIR / "macro_features.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else pd.DataFrame()

    sz_atr = atr_sizes(signals, features, CAPITAL)

    rows = []

    print("\n" + "=" * 100)
    print("BASELINE")
    print("=" * 100)
    s, g = run("BASE: atr_lev_1.5x", lev(sz_atr, signals, macro, 1.5), returns)
    rows.append(("BASE: atr_lev_1.5x", s, g))
    s, g = run("BASE: atr @1.0x", lev(sz_atr, signals, macro, 1.0), returns)
    rows.append(("BASE: atr @1.0x", s, g))
    s, g = run("PRIOR-WIN: vt14_sm1.5_x1.5",
                lev(vol_target_atr(signals, features, returns, CAPITAL, 0.14, scale_max=1.5),
                    signals, macro, 1.5),
                returns)
    rows.append(("PRIOR-WIN: vt14_sm1.5_x1.5", s, g))

    print("\n" + "=" * 100)
    print("ASYMMETRIC vol-target (scale DOWN only) — preserves DD, lifts Sharpe")
    print("=" * 100)
    for tv in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]:
        sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                target_vol=tv, asymmetric=True)
        s, g = run(f"asym_vt{int(tv*100)} @1.5x", lev(sz_vt, signals, macro, 1.5), returns)
        rows.append((f"asym_vt{int(tv*100)} @1.5x", s, g))

    print("\n" + "=" * 100)
    print("BLEND atr × vt — α × vt + (1-α) × atr at 1.5x leverage")
    print("=" * 100)
    for alpha in [0.25, 0.35, 0.50, 0.65, 0.75]:
        for tv in [0.12, 0.14]:
            sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                    target_vol=tv, scale_max=1.5)
            sz_blend = _clip(alpha * sz_vt + (1 - alpha) * sz_atr)
            s, g = run(f"blend_a{int(alpha*100)}_vt{int(tv*100)}_sm1.5_x1.5",
                        lev(sz_blend, signals, macro, 1.5), returns)
            rows.append((f"blend_a{int(alpha*100)}_vt{int(tv*100)}_sm1.5_x1.5", s, g))

    print("\n" + "=" * 100)
    print("VOL-TARGET + ADX QUALITY FILTER on entries")
    print("=" * 100)
    for adx_min in [10, 14, 18]:
        for tv in [0.12, 0.14]:
            sz = vol_target_adx_filter(signals, features, returns, CAPITAL,
                                        target_vol=tv, scale_max=1.5,
                                        adx_min=adx_min, adx_full=adx_min+15)
            s, g = run(f"adx{adx_min}_vt{int(tv*100)}_sm1.5_x1.5",
                        lev(sz, signals, macro, 1.5), returns)
            rows.append((f"adx{adx_min}_vt{int(tv*100)}_sm1.5_x1.5", s, g))

    print("\n" + "=" * 100)
    print("BLEND atr × vt at 1.0x leverage — path back to no-margin")
    print("=" * 100)
    for alpha in [0.35, 0.50, 0.65, 0.80]:
        for tv in [0.12, 0.13, 0.14]:
            sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                    target_vol=tv, scale_max=2.0)
            sz_blend = _clip(alpha * sz_vt + (1 - alpha) * sz_atr)
            s, g = run(f"blend_a{int(alpha*100)}_vt{int(tv*100)}_x1.0",
                        lev(sz_blend, signals, macro, 1.0), returns)
            rows.append((f"blend_a{int(alpha*100)}_vt{int(tv*100)}_x1.0", s, g))

    print("\n" + "=" * 100)
    print("PARETO RANKINGS")
    print("=" * 100)
    print("\nTop 10 by Sharpe:")
    for label, s, g in sorted(rows, key=lambda r: -float(r[1]["sharpe"]))[:10]:
        print(f"  {label:<36} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  gr {g:>4.2f}")
    print("\nTop 10 by Calmar:")
    for label, s, g in sorted(rows, key=lambda r: -float(r[1]["calmar"]))[:10]:
        print(f"  {label:<36} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  gr {g:>4.2f}")
    print("\nTop 10 by AnnRet with MaxDD better than -10%:")
    constrained = [r for r in rows if float(r[1]["max_drawdown"]) > -10.0]
    for label, s, g in sorted(constrained, key=lambda r: -float(r[1]["ann_return"]))[:10]:
        print(f"  {label:<36} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  gr {g:>4.2f}")
    print("\nTop 5 by AnnRet with MaxDD better than -9%:")
    very_constrained = [r for r in rows if float(r[1]["max_drawdown"]) > -9.0]
    for label, s, g in sorted(very_constrained, key=lambda r: -float(r[1]["ann_return"]))[:5]:
        print(f"  {label:<36} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  gr {g:>4.2f}")


if __name__ == "__main__":
    main()
