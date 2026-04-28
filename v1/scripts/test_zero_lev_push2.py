"""
test_zero_lev_push2.py
----------------------
Round 2 — push the concentration lever harder.  Round 1 found top12_vt14
gives 14.14%/-8.97 DD; we have 3pp DD headroom to -12.

Test grid:
  • Tighter top-N (5..12)
  • Stronger vol-target boosts (since cap binds, higher sm just lifts cap-utilisation)
  • Kelly inside top-N
  • Momentum-z inside top-N
  • COMBO stacks: top-N × vt × kelly, top-N × vt × momz

Run: python -m v1.scripts.test_zero_lev_push2
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes, kelly_sizes, defensive_tilt_overlay,
    portfolio_returns, summarise, MAX_POSITION_PCT, CAPITAL,
)


def _clip(sz):
    return sz.clip(-CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT)


def _gross_cap(sz, max_gross=1.0):
    gross = sz.abs().sum(axis=1).replace(0, np.nan)
    scale = (max_gross * CAPITAL / gross).clip(upper=1.0).fillna(1.0)
    return sz.multiply(scale, axis=0)


def vol_target_atr(signals, features, returns, capital,
                   target_vol=0.14, scale_max=2.5, window=63):
    base = atr_sizes(signals, features, capital)
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    scaler = (target_vol / realized).clip(0.3, scale_max).shift(1).fillna(1.0)
    return _clip(base.multiply(scaler, axis=0))


def adx_matrix(signals, features):
    m = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for t in signals.columns:
        if t in features and "adx" in features[t].columns:
            m[t] = features[t]["adx"].reindex(signals.index).ffill().fillna(0.0)
    return m


def top_n_concentrate(sz_base, conviction, top_n=12):
    rank = conviction.rank(axis=1, ascending=False, method="first")
    keep = (rank <= top_n).astype(float)
    return _clip(sz_base * keep)


def run(name, sz, signals, macro, returns, rows):
    sz = defensive_tilt_overlay(sz, signals, macro, CAPITAL)
    sz = _gross_cap(sz, max_gross=1.0)
    ret = portfolio_returns(sz, returns)
    s = summarise(ret, name)
    g = float(sz.abs().sum(axis=1).mean()) / CAPITAL
    g_max = float(sz.abs().sum(axis=1).max()) / CAPITAL
    rows.append((name, s, g, g_max))
    print(f"{name:<48} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
          f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
          f"gr {g:>4.2f}/{g_max:>4.2f}")


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
    adx_m  = adx_matrix(signals, features)
    conviction_adx = adx_m * signals.abs()
    sz_kelly = kelly_sizes(signals, returns, CAPITAL)

    rows = []

    # ── Tighter top-N grid with vt boost ──────────────────────────────────────
    print("\n" + "=" * 116)
    print("TIGHTER top-N × vt grid (target_vol=0.14, scale_max=2.5)")
    print("=" * 116)
    sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                            target_vol=0.14, scale_max=2.5)
    for n in [5, 6, 7, 8, 9, 10, 11, 12]:
        sz = top_n_concentrate(sz_vt, conviction_adx, top_n=n)
        run(f"top{n}_vt14sm25_x1.5", _clip(sz * 1.5), signals, macro, returns, rows)

    # ── HIGHER vt scale_max so cap binds harder ───────────────────────────────
    print("\n" + "=" * 116)
    print("STRONGER vt boost (scale_max 3-5) inside top-N")
    print("=" * 116)
    for n in [8, 10, 12]:
        for sm in [3.0, 4.0, 5.0]:
            sz_vt2 = vol_target_atr(signals, features, returns, CAPITAL,
                                     target_vol=0.14, scale_max=sm)
            sz = top_n_concentrate(sz_vt2, conviction_adx, top_n=n)
            run(f"top{n}_vt14sm{int(sm)}_x1.5", _clip(sz * 1.5), signals, macro, returns, rows)

    # ── HIGHER target_vol with strong scale_max ───────────────────────────────
    print("\n" + "=" * 116)
    print("HIGHER target_vol (16-20) inside top-N (push more risk per name)")
    print("=" * 116)
    for n in [8, 10, 12]:
        for tv in [0.16, 0.18, 0.20]:
            sz_vt2 = vol_target_atr(signals, features, returns, CAPITAL,
                                     target_vol=tv, scale_max=3.0)
            sz = top_n_concentrate(sz_vt2, conviction_adx, top_n=n)
            run(f"top{n}_vt{int(tv*100)}sm3_x1.5", _clip(sz * 1.5),
                signals, macro, returns, rows)

    # ── KELLY inside top-N ────────────────────────────────────────────────────
    print("\n" + "=" * 116)
    print("KELLY inside top-N (pure kelly, atr×kelly blend)")
    print("=" * 116)
    for n in [8, 10, 12, 15]:
        # pure kelly capped to top-N
        sz = top_n_concentrate(sz_kelly, conviction_adx, top_n=n)
        run(f"top{n}_kelly_x1.5", _clip(sz * 1.5), signals, macro, returns, rows)
        # kelly + atr blend inside top-N
        for k_alpha in [0.5, 0.7]:
            sz_kb = _clip(k_alpha * sz_kelly + (1 - k_alpha) * sz_atr)
            sz = top_n_concentrate(sz_kb, conviction_adx, top_n=n)
            run(f"top{n}_kelly_a{int(k_alpha*100)}_atr_x1.5",
                _clip(sz * 1.5), signals, macro, returns, rows)

    # ── COMBO stacks at top-12 (the sweet spot from round 1) ──────────────────
    print("\n" + "=" * 116)
    print("STACKED COMBOS at top-12 — vt × kelly × momz layered together")
    print("=" * 116)
    for n in [10, 12]:
        sz_vt2 = vol_target_atr(signals, features, returns, CAPITAL,
                                 target_vol=0.14, scale_max=3.0)
        # vt + kelly stacked
        for k_alpha in [0.3, 0.5]:
            sz_combo = _clip((1 - k_alpha) * sz_vt2 + k_alpha * sz_kelly)
            sz = top_n_concentrate(sz_combo, conviction_adx, top_n=n)
            run(f"top{n}_vt14ka{int(k_alpha*100)}_x1.5",
                _clip(sz * 1.5), signals, macro, returns, rows)

    # ── Test with leverage 2.0 / 2.5 — cap still binds at 1.0 ─────────────────
    print("\n" + "=" * 116)
    print("Push leverage_x harder (cap still binds at gr_max=1.0)")
    print("=" * 116)
    for n in [8, 10, 12]:
        for lev_x in [1.5, 2.0, 2.5]:
            sz_vt2 = vol_target_atr(signals, features, returns, CAPITAL,
                                     target_vol=0.14, scale_max=3.0)
            sz = top_n_concentrate(sz_vt2, conviction_adx, top_n=n)
            run(f"top{n}_vt14sm3_x{lev_x:.1f}", _clip(sz * lev_x),
                signals, macro, returns, rows)

    # ── RANKINGS ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 116)
    print("RANKINGS — gross_cap = 1.0x, DD budget = -12%")
    print("=" * 116)
    print("\nTop 15 by AnnRet with MaxDD better than -12%:")
    constrained = [r for r in rows if float(r[1]["max_drawdown"]) > -12.0]
    for label, s, g, gm in sorted(constrained, key=lambda r: -float(r[1]["ann_return"]))[:15]:
        print(f"  {label:<48} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
              f"gr {g:>4.2f}/{gm:>4.2f}")
    print("\nTop 10 by Sharpe with MaxDD better than -12%:")
    for label, s, g, gm in sorted(constrained, key=lambda r: -float(r[1]["sharpe"]))[:10]:
        print(f"  {label:<48} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
              f"gr {g:>4.2f}/{gm:>4.2f}")
    print("\nTop 10 by Calmar with MaxDD better than -12%:")
    for label, s, g, gm in sorted(constrained, key=lambda r: -float(r[1]["calmar"]))[:10]:
        print(f"  {label:<48} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
              f"gr {g:>4.2f}/{gm:>4.2f}")


if __name__ == "__main__":
    main()
