"""
test_voltgt_capped.py
---------------------
Constrains average gross exposure ≤ 1.5x (matching current production cap).
Tests vol-target blends with portfolio-level gross caps to honour the
"no more leverage" requirement while still pushing returns higher.

Run: python -m v1.scripts.test_voltgt_capped
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes, defensive_tilt_overlay,
    portfolio_returns, summarise, MAX_POSITION_PCT, CAPITAL,
)


def _clip(sz):
    return sz.clip(-CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT)


def _gross_cap(sz, max_gross=1.5):
    """Scale down rows whose gross exceeds max_gross × CAPITAL.  Leaves smaller
    rows untouched (asymmetric: cap only)."""
    gross = sz.abs().sum(axis=1).replace(0, np.nan)
    scale = (max_gross * CAPITAL / gross).clip(upper=1.0).fillna(1.0)
    return sz.multiply(scale, axis=0)


def vol_target_atr(signals, features, returns, capital,
                   target_vol=0.13, scale_max=2.5, window=63):
    base = atr_sizes(signals, features, capital)
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    scaler = (target_vol / realized).clip(0.3, scale_max).shift(1).fillna(1.0)
    return _clip(base.multiply(scaler, axis=0))


def lev_capped(sz_base, signals, macro, lev_x=1.5, max_gross=1.5):
    sz = _clip(sz_base * lev_x)
    sz = _gross_cap(sz, max_gross=max_gross)
    return defensive_tilt_overlay(sz, signals, macro, CAPITAL)


def run(name, sz, returns, rows):
    ret = portfolio_returns(sz, returns)
    s = summarise(ret, name)
    g = float(sz.abs().sum(axis=1).mean()) / CAPITAL
    g_max = float(sz.abs().sum(axis=1).max()) / CAPITAL
    rows.append((name, s, g, g_max))
    print(f"{name:<40} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
          f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
          f"gr_avg {g:>4.2f} gr_max {g_max:>4.2f}")


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

    print("\n" + "=" * 110)
    print("BASELINES")
    print("=" * 110)
    sz = _clip(sz_atr * 1.5); sz = defensive_tilt_overlay(sz, signals, macro, CAPITAL)
    run("BASE atr_lev_1.5x (no gross cap)", sz, returns, rows)
    sz = lev_capped(sz_atr, signals, macro, lev_x=1.5, max_gross=1.5)
    run("BASE atr_lev_1.5x (gross_cap_1.5)", sz, returns, rows)

    print("\n" + "=" * 110)
    print("VOL-TARGET BLEND with PORTFOLIO GROSS CAP at 1.5x")
    print("=" * 110)
    for alpha in [0.25, 0.35, 0.50, 0.75, 1.00]:
        for tv in [0.12, 0.14]:
            sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                    target_vol=tv, scale_max=2.5)
            sz_blend = _clip(alpha * sz_vt + (1 - alpha) * sz_atr)
            sz = lev_capped(sz_blend, signals, macro, lev_x=1.5, max_gross=1.5)
            run(f"blend_a{int(alpha*100)}_vt{int(tv*100)}_x1.5_capg1.5",
                sz, returns, rows)

    print("\n" + "=" * 110)
    print("VOL-TARGET BLEND with PORTFOLIO GROSS CAP at 1.4x (extra-tight)")
    print("=" * 110)
    for alpha in [0.50, 1.00]:
        for tv in [0.12, 0.14]:
            sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                    target_vol=tv, scale_max=2.5)
            sz_blend = _clip(alpha * sz_vt + (1 - alpha) * sz_atr)
            sz = lev_capped(sz_blend, signals, macro, lev_x=1.5, max_gross=1.4)
            run(f"blend_a{int(alpha*100)}_vt{int(tv*100)}_x1.5_capg1.4",
                sz, returns, rows)

    print("\n" + "=" * 110)
    print("VOL-TARGET BLEND with PORTFOLIO GROSS CAP at 1.0x (no leverage)")
    print("=" * 110)
    sz = lev_capped(sz_atr, signals, macro, lev_x=1.0, max_gross=1.0)
    run("BASE atr_pure (gross_cap_1.0)", sz, returns, rows)
    for alpha in [0.50, 0.75, 1.00]:
        for tv in [0.12, 0.14]:
            sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                    target_vol=tv, scale_max=2.5)
            sz_blend = _clip(alpha * sz_vt + (1 - alpha) * sz_atr)
            sz = lev_capped(sz_blend, signals, macro, lev_x=1.0, max_gross=1.0)
            run(f"blend_a{int(alpha*100)}_vt{int(tv*100)}_x1.0_capg1.0",
                sz, returns, rows)

    print("\n" + "=" * 110)
    print("RANKINGS")
    print("=" * 110)
    print("\nTop 8 by Sharpe (gross_cap honoured):")
    for label, s, g, gm in sorted(rows, key=lambda r: -float(r[1]["sharpe"]))[:8]:
        print(f"  {label:<40} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
              f"gr_avg {g:>4.2f} gr_max {gm:>4.2f}")
    print("\nTop 8 by AnnRet with gross_avg <= 1.5 AND DD better than -10%:")
    constrained = [r for r in rows if r[2] <= 1.5 and float(r[1]["max_drawdown"]) > -10.0]
    for label, s, g, gm in sorted(constrained, key=lambda r: -float(r[1]["ann_return"]))[:8]:
        print(f"  {label:<40} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
              f"gr_avg {g:>4.2f} gr_max {gm:>4.2f}")
    print("\nTop 5 with gross_avg <= 1.0 (NO LEVERAGE):")
    no_lev = [r for r in rows if r[2] <= 1.05]  # tolerance for noise
    for label, s, g, gm in sorted(no_lev, key=lambda r: -float(r[1]["ann_return"]))[:5]:
        print(f"  {label:<40} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
              f"gr_avg {g:>4.2f} gr_max {gm:>4.2f}")


if __name__ == "__main__":
    main()
