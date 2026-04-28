"""
test_voltgt_pareto.py
---------------------
Vol-target sizing produced very high returns in test_push_returns but with
high DD. Sweep parameter combos to find the best Pareto point under the
constraint:  push AnnRet above current 15.30 prod baseline while keeping
MaxDD <= -10% (or as close to -8.67 as possible).

Also test 1.0x leverage equivalents — the user wants to drop leverage
once returns are high enough.

Run: python -m v1.scripts.test_voltgt_pareto
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
                   target_vol=0.13, window=63, scale_min=0.3, scale_max=2.5):
    base = atr_sizes(signals, features, capital)
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    scaler = (target_vol / realized).clip(scale_min, scale_max).shift(1).fillna(1.0)
    return _clip(base.multiply(scaler, axis=0))


def vol_target_dd_brake(signals, features, returns, capital,
                        target_vol=0.13, window=63, scale_max=2.5,
                        dd_trigger=-0.07, dd_cut=0.50, dd_recover=-0.04):
    """Vol target + drawdown circuit breaker: cut gross by dd_cut when DD <
    dd_trigger; restore when DD recovers above dd_recover."""
    base = vol_target_atr(signals, features, returns, capital,
                          target_vol=target_vol, window=window, scale_max=scale_max)
    # Compute portfolio equity to get rolling DD
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    eq = (1.0 + port_ret).cumprod()
    rolling_max = eq.cummax()
    dd = (eq / rolling_max) - 1.0
    # State machine: cut when dd < trigger, restore when dd > recover
    in_brake = False
    mult = pd.Series(1.0, index=base.index)
    for i in range(len(dd)):
        if not in_brake and dd.iloc[i] < dd_trigger:
            in_brake = True
        elif in_brake and dd.iloc[i] > dd_recover:
            in_brake = False
        mult.iloc[i] = (1.0 - dd_cut) if in_brake else 1.0
    mult = mult.shift(1).fillna(1.0)
    return _clip(base.multiply(mult, axis=0))


def lev(sz_base, signals, macro, lev_x=1.5):
    sz = _clip(sz_base * lev_x)
    return defensive_tilt_overlay(sz, signals, macro, CAPITAL)


def run_variant(name, sz, returns):
    ret = portfolio_returns(sz, returns)
    s   = summarise(ret, name)
    gross = float(sz.abs().sum(axis=1).mean()) / CAPITAL
    return s, gross


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

    # ============== EXP A: VOL-TARGET parameter sweep at 1.5x =============
    print("\n" + "=" * 96)
    print("VOL-TARGET sweep at 1.5x leverage (target_vol × scale_max)")
    print("=" * 96)
    print(f"{'method':<32} {'AnnRet%':>8} {'Sharpe':>7} {'MaxDD%':>8} "
          f"{'Calmar':>7} {'Gross':>6}")
    print("-" * 96)

    rows = []
    # baseline
    sz = lev(sz_atr, signals, macro, 1.5)
    s, g = run_variant("BASE: atr_lev_1.5x", sz, returns)
    print(f"{s['label']:<32} {s['ann_return']:>8} {s['sharpe']:>7} "
          f"{s['max_drawdown']:>8} {s['calmar']:>7} {g:>6.2f}")
    rows.append((s['label'], s, g))

    for tv in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]:
        for sm in [1.5, 2.0, 2.5]:
            sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                    target_vol=tv, scale_max=sm)
            sz = lev(sz_vt, signals, macro, 1.5)
            label = f"vt{int(tv*100)}_sm{sm}_x1.5"
            s, g = run_variant(label, sz, returns)
            rows.append((label, s, g))
            print(f"{label:<32} {s['ann_return']:>8} {s['sharpe']:>7} "
                  f"{s['max_drawdown']:>8} {s['calmar']:>7} {g:>6.2f}")

    # ============== EXP B: VOL-TARGET + DD CIRCUIT BREAKER ================
    print("\n" + "=" * 96)
    print("VOL-TARGET + DRAWDOWN BRAKE at 1.5x — try to cap MaxDD < -10%")
    print("=" * 96)
    print(f"{'method':<32} {'AnnRet%':>8} {'Sharpe':>7} {'MaxDD%':>8} "
          f"{'Calmar':>7} {'Gross':>6}")
    print("-" * 96)

    for (tv, sm, dd_trig, dd_cut, dd_rec) in [
        (0.13, 2.5, -0.07, 0.50, -0.04),
        (0.13, 2.5, -0.06, 0.50, -0.03),
        (0.13, 2.5, -0.07, 0.40, -0.04),
        (0.13, 2.0, -0.07, 0.50, -0.04),
        (0.13, 2.0, -0.06, 0.50, -0.03),
        (0.12, 2.0, -0.07, 0.40, -0.04),
        (0.12, 2.0, -0.06, 0.50, -0.03),
        (0.12, 1.5, -0.06, 0.40, -0.03),
        (0.11, 1.5, -0.06, 0.40, -0.03),
        (0.10, 1.5, -0.06, 0.40, -0.03),
    ]:
        sz_dd = vol_target_dd_brake(signals, features, returns, CAPITAL,
                                     target_vol=tv, scale_max=sm,
                                     dd_trigger=dd_trig, dd_cut=dd_cut,
                                     dd_recover=dd_rec)
        sz = lev(sz_dd, signals, macro, 1.5)
        label = f"vt{int(tv*100)}_sm{sm}_brk{abs(int(dd_trig*100))}_{int(dd_cut*100)}"
        s, g = run_variant(label, sz, returns)
        rows.append((label, s, g))
        print(f"{label:<32} {s['ann_return']:>8} {s['sharpe']:>7} "
              f"{s['max_drawdown']:>8} {s['calmar']:>7} {g:>6.2f}")

    # ============== EXP C: VOL-TARGET at 1.0x leverage =====================
    print("\n" + "=" * 96)
    print("VOL-TARGET at 1.0x leverage — path back to no-margin operation")
    print("=" * 96)
    print(f"{'method':<32} {'AnnRet%':>8} {'Sharpe':>7} {'MaxDD%':>8} "
          f"{'Calmar':>7} {'Gross':>6}")
    print("-" * 96)

    sz_1x = lev(sz_atr, signals, macro, 1.0)
    s, g = run_variant("BASE: atr_pure @1.0x", sz_1x, returns)
    rows.append((s['label'], s, g))
    print(f"{s['label']:<32} {s['ann_return']:>8} {s['sharpe']:>7} "
          f"{s['max_drawdown']:>8} {s['calmar']:>7} {g:>6.2f}")

    for tv in [0.10, 0.11, 0.12, 0.13, 0.14]:
        for sm in [1.5, 2.0]:
            sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                    target_vol=tv, scale_max=sm)
            sz = lev(sz_vt, signals, macro, 1.0)
            label = f"vt{int(tv*100)}_sm{sm}_x1.0"
            s, g = run_variant(label, sz, returns)
            rows.append((label, s, g))
            print(f"{label:<32} {s['ann_return']:>8} {s['sharpe']:>7} "
                  f"{s['max_drawdown']:>8} {s['calmar']:>7} {g:>6.2f}")

    # ============== Summary ===============================================
    print("\n" + "=" * 96)
    print("PARETO SUMMARY — all variants ranked by Sharpe (top 12)")
    print("=" * 96)
    rows_sharpe = sorted(rows, key=lambda r: -float(r[1]["sharpe"]))[:12]
    for label, s, g in rows_sharpe:
        print(f"  {label:<32} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  gr {g:>4.2f}")

    print("\nRanked by Calmar (top 12):")
    rows_cal = sorted(rows, key=lambda r: -float(r[1]["calmar"]))[:12]
    for label, s, g in rows_cal:
        print(f"  {label:<32} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  gr {g:>4.2f}")

    print("\nReturn-maximisers with MaxDD better than -11% (constrained):")
    constrained = [r for r in rows if float(r[1]["max_drawdown"]) > -11.0]
    constrained = sorted(constrained, key=lambda r: -float(r[1]["ann_return"]))[:8]
    for label, s, g in constrained:
        print(f"  {label:<32} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  gr {g:>4.2f}")


if __name__ == "__main__":
    main()
