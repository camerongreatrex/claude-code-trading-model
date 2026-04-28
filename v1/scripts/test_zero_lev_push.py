"""
test_zero_lev_push.py
---------------------
Zero-leverage push: gross_cap = 1.0x, DD budget = 12%, maximise AnnRet.

At gross_cap=1.0 the lever is REALLOCATION, not amplification.  So the test
matrix focuses on:
  1. Concentration (top-N strongest signals/day)
  2. Conviction weighting (ADX, momentum z-score scales the size)
  3. Vol-target reallocation (boost calm-period weights, gross_cap binds)
  4. Kelly variants (full, half, quarter) blended with atr / vt
  5. Profit-lock parameter variants on top of overlay-applied signals
  6. Defensive overlay on/off comparison

Universe: 42-ticker production set (overlay already applied to multi_signals.parquet).
Capital: $100k.  Per-name cap: $10k.

Run: python -m v1.scripts.test_zero_lev_push
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes, kelly_sizes, defensive_tilt_overlay,
    portfolio_returns, summarise, MAX_POSITION_PCT, CAPITAL,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _clip(sz):
    return sz.clip(-CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT)


def _gross_cap(sz, max_gross=1.0):
    """Scale rows whose gross > max_gross × CAPITAL down to exactly max_gross."""
    gross = sz.abs().sum(axis=1).replace(0, np.nan)
    scale = (max_gross * CAPITAL / gross).clip(upper=1.0).fillna(1.0)
    return sz.multiply(scale, axis=0)


# ── sizers ──────────────────────────────────────────────────────────────────
def vol_target_atr(signals, features, returns, capital,
                   target_vol=0.13, scale_max=2.5, window=63):
    base = atr_sizes(signals, features, capital)
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    scaler = (target_vol / realized).clip(0.3, scale_max).shift(1).fillna(1.0)
    return _clip(base.multiply(scaler, axis=0))


def top_n_concentrate(sz_base, signals, features, top_n=12):
    """Keep only the top-N strongest signals/day (by ADX × |signal|).
    Drops capital onto the remaining names — concentration play."""
    adx_mat = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for t in signals.columns:
        if t in features and "adx" in features[t].columns:
            adx_mat[t] = features[t]["adx"].reindex(signals.index).ffill().fillna(0.0)
    conviction = adx_mat * signals.abs()
    rank = conviction.rank(axis=1, ascending=False, method="first")
    keep = (rank <= top_n).astype(float)
    return _clip(sz_base * keep)


def conviction_weight(sz_base, signals, features, adx_min=10.0, adx_full=35.0):
    """Continuous ADX-conviction scale (0..1) on top of any base sizer."""
    adx_mat = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for t in signals.columns:
        if t in features and "adx" in features[t].columns:
            adx_mat[t] = features[t]["adx"].reindex(signals.index).ffill().fillna(0.0)
    scale = ((adx_mat - adx_min) / (adx_full - adx_min)).clip(0, 1)
    return _clip(sz_base * scale)


def momentum_zscore_weight(sz_base, signals, returns, lookback=63):
    """Scale by 63-day momentum z-score (stronger trend → larger size)."""
    mom = returns.rolling(lookback, min_periods=20).sum()
    mu = mom.rolling(252, min_periods=63).mean()
    sd = mom.rolling(252, min_periods=63).std().replace(0, np.nan)
    z = ((mom - mu) / sd).clip(-2, 2).fillna(0.0)
    # Map z ∈ [-2, 2] → scale ∈ [0.3, 1.5]
    scale = (0.9 + 0.3 * z).clip(0.3, 1.5)
    return _clip(sz_base * scale)


def kelly_atr_blend(signals, features, returns, capital, k_alpha=0.5, kelly_fraction=0.5):
    """Blend kelly + atr.  kelly_fraction scales raw kelly (0.5 = half-Kelly)."""
    sz_atr = atr_sizes(signals, features, capital)
    sz_k = kelly_sizes(signals, returns, capital) * (kelly_fraction / 0.5)  # native is half-K
    return _clip(k_alpha * sz_k + (1 - k_alpha) * sz_atr)


# ── runner ──────────────────────────────────────────────────────────────────
def run(name, sz, signals, macro, returns, rows):
    """Apply defensive overlay + gross cap + portfolio return."""
    sz = defensive_tilt_overlay(sz, signals, macro, CAPITAL)
    sz = _gross_cap(sz, max_gross=1.0)
    ret = portfolio_returns(sz, returns)
    s = summarise(ret, name)
    g = float(sz.abs().sum(axis=1).mean()) / CAPITAL
    g_max = float(sz.abs().sum(axis=1).max()) / CAPITAL
    rows.append((name, s, g, g_max))
    print(f"{name:<46} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
          f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
          f"gr {g:>4.2f}/{g_max:>4.2f}")


def run_no_overlay(name, sz, returns, rows):
    """No defensive overlay + gross cap + portfolio return."""
    sz = _gross_cap(sz, max_gross=1.0)
    ret = portfolio_returns(sz, returns)
    s = summarise(ret, name)
    g = float(sz.abs().sum(axis=1).mean()) / CAPITAL
    g_max = float(sz.abs().sum(axis=1).max()) / CAPITAL
    rows.append((name, s, g, g_max))
    print(f"{name:<46} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
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

    rows = []

    # ── BASELINES ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("BASELINES (gross_cap = 1.0x)")
    print("=" * 110)
    run("BASE atr_pure @1.0x",        sz_atr,         signals, macro, returns, rows)
    run("BASE atr × 1.5 then cap@1.0", _clip(sz_atr * 1.5), signals, macro, returns, rows)
    run("BASE atr × 2.0 then cap@1.0", _clip(sz_atr * 2.0), signals, macro, returns, rows)

    # ── VOL-TARGET REALLOCATION at 1.0 cap ────────────────────────────────────
    print("\n" + "=" * 110)
    print("VOL-TARGET REALLOCATION (cap binds — boost reallocates, doesn't amplify)")
    print("=" * 110)
    for tv in [0.12, 0.14, 0.16, 0.18]:
        for sm in [1.5, 2.0, 3.0]:
            sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                    target_vol=tv, scale_max=sm)
            run(f"vt{int(tv*100)}_sm{sm:.1f}_x1.5_cap",
                _clip(sz_vt * 1.5), signals, macro, returns, rows)

    # ── BLENDS at 1.0 cap (alpha × vt + (1-alpha) × atr) × leverage ───────────
    print("\n" + "=" * 110)
    print("BLEND atr × vt at multiple leverage multipliers, all capped at 1.0")
    print("=" * 110)
    for alpha in [0.50, 0.65, 0.80, 1.00]:
        for tv in [0.13, 0.14, 0.16]:
            for lev_x in [1.5, 2.0, 2.5]:
                sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                        target_vol=tv, scale_max=2.5)
                sz_blend = _clip(alpha * sz_vt + (1 - alpha) * sz_atr)
                run(f"blend_a{int(alpha*100)}_vt{int(tv*100)}_x{lev_x:.1f}_cap",
                    _clip(sz_blend * lev_x), signals, macro, returns, rows)

    # ── CONCENTRATION (top-N strongest signals each day) ──────────────────────
    print("\n" + "=" * 110)
    print("CONCENTRATION — keep only top-N strongest signals per day")
    print("=" * 110)
    for top_n in [8, 10, 12, 15, 20]:
        sz_top = top_n_concentrate(sz_atr, signals, features, top_n=top_n)
        run(f"top{top_n}_atr_x1.5_cap",
            _clip(sz_top * 1.5), signals, macro, returns, rows)
        # top-N + vol-target
        sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                target_vol=0.14, scale_max=2.5)
        sz_top_vt = top_n_concentrate(sz_vt, signals, features, top_n=top_n)
        run(f"top{top_n}_vt14_x1.5_cap",
            _clip(sz_top_vt * 1.5), signals, macro, returns, rows)

    # ── CONVICTION WEIGHTING (continuous ADX scale) ────────────────────────────
    print("\n" + "=" * 110)
    print("CONVICTION WEIGHTING — continuous ADX scale on top of base sizer")
    print("=" * 110)
    for amin, amax in [(10, 30), (12, 35), (15, 40)]:
        sz_cv = conviction_weight(sz_atr, signals, features, adx_min=amin, adx_full=amax)
        run(f"adxconv_{amin}-{amax}_atr_x1.5_cap",
            _clip(sz_cv * 1.5), signals, macro, returns, rows)
        # combine with vt
        sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                target_vol=0.14, scale_max=2.5)
        sz_cv_vt = conviction_weight(sz_vt, signals, features, adx_min=amin, adx_full=amax)
        run(f"adxconv_{amin}-{amax}_vt14_x1.5_cap",
            _clip(sz_cv_vt * 1.5), signals, macro, returns, rows)

    # ── MOMENTUM Z-SCORE weighting ────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("MOMENTUM Z-SCORE weighting — scale by 63d momentum z-score")
    print("=" * 110)
    for lb in [42, 63, 126]:
        sz_mz = momentum_zscore_weight(sz_atr, signals, returns, lookback=lb)
        run(f"momz_lb{lb}_atr_x1.5_cap",
            _clip(sz_mz * 1.5), signals, macro, returns, rows)

    # ── KELLY blends ──────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("KELLY × ATR blends (capped @1.0)")
    print("=" * 110)
    for k_alpha in [0.25, 0.50]:
        for kf in [0.25, 0.50]:
            sz_kb = kelly_atr_blend(signals, features, returns, CAPITAL,
                                     k_alpha=k_alpha, kelly_fraction=kf)
            run(f"kelly_a{int(k_alpha*100)}_kf{int(kf*100)}_x1.5_cap",
                _clip(sz_kb * 1.5), signals, macro, returns, rows)

    # ── COMBO winners — try stacking ──────────────────────────────────────────
    print("\n" + "=" * 110)
    print("STACKED COMBOS — top-N × conviction × vol-target")
    print("=" * 110)
    for top_n in [12, 15]:
        sz_vt = vol_target_atr(signals, features, returns, CAPITAL,
                                target_vol=0.14, scale_max=2.5)
        sz = top_n_concentrate(sz_vt, signals, features, top_n=top_n)
        sz = conviction_weight(sz, signals, features, adx_min=12, adx_full=35)
        run(f"COMBO top{top_n}_vt14_adxcv12-35_x1.5_cap",
            _clip(sz * 1.5), signals, macro, returns, rows)
        sz = top_n_concentrate(sz_atr, signals, features, top_n=top_n)
        sz = conviction_weight(sz, signals, features, adx_min=12, adx_full=35)
        run(f"COMBO top{top_n}_atr_adxcv12-35_x1.5_cap",
            _clip(sz * 1.5), signals, macro, returns, rows)

    # ── No-overlay comparison for top combos (does defensive overlay help at 1.0?) ──
    print("\n" + "=" * 110)
    print("NO-OVERLAY compare on top-N + atr × 1.5 (overlay vs no-overlay)")
    print("=" * 110)
    for top_n in [12, 15]:
        sz_top = top_n_concentrate(sz_atr, signals, features, top_n=top_n)
        run_no_overlay(f"NOOVERLAY top{top_n}_atr_x1.5_cap",
            _clip(sz_top * 1.5), returns, rows)

    # ── RANKINGS ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("RANKINGS — gross_cap = 1.0x, DD budget = -12%")
    print("=" * 110)
    print("\nTop 15 by AnnRet with MaxDD better than -12%:")
    constrained = [r for r in rows if float(r[1]["max_drawdown"]) > -12.0]
    for label, s, g, gm in sorted(constrained, key=lambda r: -float(r[1]["ann_return"]))[:15]:
        print(f"  {label:<46} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
              f"gr {g:>4.2f}/{gm:>4.2f}")
    print("\nTop 10 by Sharpe with MaxDD better than -12%:")
    for label, s, g, gm in sorted(constrained, key=lambda r: -float(r[1]["sharpe"]))[:10]:
        print(f"  {label:<46} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
              f"gr {g:>4.2f}/{gm:>4.2f}")
    print("\nTop 10 by Calmar with MaxDD better than -12%:")
    for label, s, g, gm in sorted(constrained, key=lambda r: -float(r[1]["calmar"]))[:10]:
        print(f"  {label:<46} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
              f"gr {g:>4.2f}/{gm:>4.2f}")
    print("\nStricter — Top 8 by AnnRet with MaxDD better than -10%:")
    very = [r for r in rows if float(r[1]["max_drawdown"]) > -10.0]
    for label, s, g, gm in sorted(very, key=lambda r: -float(r[1]["ann_return"]))[:8]:
        print(f"  {label:<46} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  "
              f"gr {g:>4.2f}/{gm:>4.2f}")


if __name__ == "__main__":
    main()
