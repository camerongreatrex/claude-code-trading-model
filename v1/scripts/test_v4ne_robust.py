"""
Robustness-focused lift on top of V4N-E production stack.

V4N-E: Sh 2.58 / Ann 23.98% / DD -4.23% / Cal 5.67 (3.4y OOS, 100% rolling
252d alpha vs SPY). Goal here is NOT peak Ann — it's regime durability for
a 5y SPY-beat bet. Test overlays that should add robustness (not break
bull-period performance):

  R. Regime-aware sleeve  — sleeve_pct expands when SPY < 200dMA, contracts
                             when SPY > 50dMA + ADX>20 (trend strength).
  T. SPY-trend gross-floor bump — raise gross floor when SPY trend strong.
  E. Equity-index breakout overlay — boost positions in IWM/EFA/EEM when
     they break their 60d high (broad-market breadth confirmation).

Anti-overfit gate is stricter for a 5y bet:
  - bull windows >= V4N-E - 0.2pp (very tight)
  - bear/stress windows >= V4N-E (no degradation)
  - max DD <= V4N-E + 0.2pp
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


# ── V4N-E base builder (current production) ─────────────────────────────────
def build_v4ne(sig, feats, rets, macro,
                sleeve_pct=0.12,
                sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
                gross_floor=0.95,
                accel_thresh=1.05, accel_boost=1.35):
    s = top_n_adx_momt_ac_sizes(
        sig, feats, rets, CAPITAL,
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=gross_floor,
    )
    s = defensive_tilt_overlay(s, sig, macro, CAPITAL)
    s = diversifier_sleeve_overlay(s, CAPITAL,
                                     sleeve_tickers=sleeve_tickers,
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
                                accel_thresh=accel_thresh, accel_boost=accel_boost,
                                short_w=10, long_w=42)
    return s


# ── Family R: regime-aware sleeve ───────────────────────────────────────────
def regime_aware_sleeve(sig, feats, rets, macro,
                          sleeve_low=0.08, sleeve_high=0.18,
                          spy_ma=200,
                          sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH")):
    """
    Build V4N-E but with sleeve_pct that flips:
      - sleeve_high when SPY < spy_ma (defensive)
      - sleeve_low  when SPY > spy_ma (bull, lean toward equities)
    Implemented as two parallel V4N-E builds, then daily switch.
    """
    spy = feats.get("SPY")
    if spy is None:
        return build_v4ne(sig, feats, rets, macro)
    spy_close = spy["Close"].reindex(rets.index).ffill()
    spy_above = spy_close > spy_close.rolling(spy_ma).mean()
    spy_above = spy_above.shift(1).fillna(False)  # no look-ahead

    base_low  = build_v4ne(sig, feats, rets, macro, sleeve_pct=sleeve_low,
                             sleeve_tickers=sleeve_tickers)
    base_high = build_v4ne(sig, feats, rets, macro, sleeve_pct=sleeve_high,
                             sleeve_tickers=sleeve_tickers)
    out = base_low.copy()
    bear_idx = rets.index[~spy_above]
    out.loc[bear_idx] = base_high.loc[bear_idx]
    return out


# ── Family T: SPY-trend gross-floor bump ────────────────────────────────────
def spy_trend_floor(sizes, feats, macro,
                      strong_floor=1.00, base_floor=0.95,
                      spy_short_ma=50, spy_long_ma=200,
                      capital=CAPITAL, max_gross=1.0):
    """
    On days where SPY > 50dMA AND SPY > 200dMA AND VIX-z <= 0 (bull regime),
    scale gross UP to strong_floor*capital if currently below.
    """
    out = sizes.copy()
    spy = feats.get("SPY")
    if spy is None or "vix_zscore" not in macro.columns:
        return out
    spy_close = spy["Close"].reindex(sizes.index).ffill()
    spy_50 = spy_close.rolling(spy_short_ma).mean()
    spy_200 = spy_close.rolling(spy_long_ma).mean()
    vix_z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    bull = ((spy_close > spy_50) & (spy_close > spy_200) & (vix_z <= 0))
    bull = bull.shift(1).fillna(False)

    gross = out.abs().sum(axis=1).replace(0, np.nan)
    target = strong_floor * capital
    scale = pd.Series(1.0, index=sizes.index)
    bull_low = bull & (gross < target)
    scale[bull_low] = (target / gross[bull_low]).clip(upper=max_gross * capital / gross[bull_low])
    out = out.multiply(scale, axis=0)
    g = out.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = ((max_gross * capital) / g).clip(upper=1.0).fillna(1.0)
    return out.multiply(cap_scale, axis=0)


# ── Family E: equity-index breakout boost ───────────────────────────────────
def equity_breakout_boost(sizes, feats,
                            tickers=("IWM", "EFA", "EEM", "SPY"),
                            lookback=60, boost=1.20, capital=CAPITAL,
                            max_gross=1.0):
    """
    For each equity-index name held long, if today's close == 60d high (real
    breakout), boost size by `boost`. Renormalize to preserve gross.
    """
    out = sizes.copy()
    high_flags = pd.DataFrame(False, index=sizes.index, columns=tickers)
    for t in tickers:
        if t not in feats:
            continue
        c = feats[t]["Close"].reindex(sizes.index).ffill()
        roll_max = c.rolling(lookback).max()
        high_flags[t] = (c >= roll_max).shift(1).fillna(False)

    for d in sizes.index:
        row = out.loc[d]
        long_pos = row[row > 0]
        if long_pos.empty:
            continue
        # Names that are long AND breaking out
        elig = [t for t in tickers if t in long_pos.index and high_flags.loc[d, t]]
        if not elig:
            continue
        gross_before = long_pos.sum()
        for t in elig:
            out.loc[d, t] = out.loc[d, t] * boost
        new_long = out.loc[d, long_pos.index]
        scale = gross_before / new_long.sum() if new_long.sum() > 0 else 1.0
        out.loc[d, long_pos.index] = new_long * scale
    return out


def report(name, sizes, rets, base_dd=None, base_ann=None):
    pr = portfolio_returns(sizes, rets).dropna()
    m = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
    if not m:
        print(f"{name:<55} insufficient data")
        return None
    flag = ""
    if base_dd is not None and m["mdd"] >= base_dd - 0.002:
        flag += " *"
    if base_ann is not None and m["ann"] > base_ann:
        flag += "+"
    print(f"{name:<55} {m['sharpe']:>5.2f} {m['ann']*100:>6.2f}% "
          f"{m['mdd']*100:>6.2f}% {m['calmar']:>5.2f} {m['vol']*100:>5.1f}%{flag}")
    return m | {"name": name, "returns": pr}


def window_check(pr, base_pr, label, spy_pr):
    """Per-window AnnRet diff vs base AND vs SPY."""
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
    print(f"\n  Window robustness for {label}:")
    print(f"    {'Window':<26}  {'base Ann':>9}  {'var Ann':>9}  {'Δ vs base':>9}  {'SPY Ann':>9}  {'Δ vs SPY':>9}")
    diffs_b, diffs_s = {}, {}
    for lab, s, e in windows:
        b = pr.loc[s:e].dropna()
        a = base_pr.loc[s:e].dropna()
        sp = spy_pr.loc[s:e].dropna()
        if len(b) < 20 or len(a) < 20:
            continue
        b_ann = (1 + b).prod() ** (252 / len(b)) - 1
        a_ann = (1 + a).prod() ** (252 / len(a)) - 1
        s_ann = (1 + sp).prod() ** (252 / len(sp)) - 1
        d_b = (b_ann - a_ann) * 100
        d_s = (b_ann - s_ann) * 100
        diffs_b[lab] = d_b
        diffs_s[lab] = d_s
        flag_b = "✓" if d_b > 0 else " "
        flag_s = "✓" if d_s > 0 else "x"
        print(f"    {lab:<26}  {a_ann*100:>8.2f}%  {b_ann*100:>8.2f}%  {d_b:>+8.2f}  {s_ann*100:>8.2f}%  {d_s:>+8.2f} {flag_s}")
    return diffs_b, diffs_s


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in "
          f"{time.time()-t0:.1f}s\n")

    # SPY return series for SPY-relative comparison
    spy = feats["SPY"]["log_return"].reindex(rets.index).fillna(0.0)
    spy_simple = (np.exp(spy) - 1)

    print(f"{'Variant':<55} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    results = []

    # ── V4N-E baseline ──────────────────────────────────────────────────────
    base_sizes = build_v4ne(sig, feats, rets, macro)
    base_m = report("V4N-E baseline (production)", base_sizes, rets)
    results.append(base_m)
    base_dd, base_ann, base_pr = base_m["mdd"], base_m["ann"], base_m["returns"]

    # ── R. Regime-aware sleeve ──────────────────────────────────────────────
    print("\n--- R. Regime-aware sleeve (bear: high, bull: low) ---")
    for sl, sh in [(0.08, 0.16), (0.06, 0.18), (0.10, 0.16), (0.05, 0.20),
                    (0.08, 0.20), (0.10, 0.18)]:
        v = regime_aware_sleeve(sig, feats, rets, macro,
                                  sleeve_low=sl, sleeve_high=sh)
        m = report(f"R: sleeve_low={sl} sleeve_high={sh}", v, rets, base_dd, base_ann)
        if m: results.append(m)

    # ── T. SPY-trend gross-floor bump ──────────────────────────────────────
    print("\n--- T. SPY-trend gross-floor bump (bull: lean in fully) ---")
    for sf in (0.97, 0.98, 0.99, 1.00):
        v = spy_trend_floor(base_sizes, feats, macro, strong_floor=sf)
        m = report(f"T: strong_floor={sf}", v, rets, base_dd, base_ann)
        if m: results.append(m)

    # ── E. Equity-index breakout boost ─────────────────────────────────────
    print("\n--- E. Equity-index breakout boost (60d high) ---")
    for lb in (40, 60, 80):
        for bo in (1.10, 1.20, 1.30):
            v = equity_breakout_boost(base_sizes, feats,
                                        lookback=lb, boost=bo)
            m = report(f"E: lookback={lb} boost={bo}", v, rets, base_dd, base_ann)
            if m: results.append(m)

    # ── X. Combos: best R + best T (most natural pair) ─────────────────────
    print("\n--- X. Combos: best R + T ---")
    for sl, sh in [(0.08, 0.16), (0.10, 0.18)]:
        v = regime_aware_sleeve(sig, feats, rets, macro,
                                  sleeve_low=sl, sleeve_high=sh)
        v = spy_trend_floor(v, feats, macro, strong_floor=0.98)
        m = report(f"X: R({sl},{sh}) + T(0.98)", v, rets, base_dd, base_ann)
        if m: results.append(m)

    # ── Leaderboards ───────────────────────────────────────────────────────
    print(f"\n{'='*80}\nBaseline V4N-E: Sh {base_m['sharpe']:.2f}  "
          f"Ann {base_m['ann']*100:.2f}%  DD {base_m['mdd']*100:.2f}%  "
          f"Cal {base_m['calmar']:.2f}\n{'='*80}")

    print("\nTop 8 by AnnRet (DD <= V4N-E + 0.2pp):")
    safe = [r for r in results if r and r["mdd"] >= base_dd - 0.002 and r["ann"] > base_ann]
    safe.sort(key=lambda x: -x["ann"])
    for r in safe[:8]:
        print(f"  {r['name']:<55} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    print("\nTop 5 by Sharpe (any DD):")
    by_sh = [r for r in results if r]
    by_sh.sort(key=lambda x: -x["sharpe"])
    for r in by_sh[:5]:
        print(f"  {r['name']:<55} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}")

    # ── Window-robustness with SPY comparison ──────────────────────────────
    print(f"\n{'='*80}\nWINDOW + SPY ROBUSTNESS for top-3 candidates\n{'='*80}")
    for r in safe[:3]:
        diffs_b, diffs_s = window_check(r["returns"], base_pr, r["name"], spy_simple)
        bull_loss = sum(1 for k, v in diffs_b.items()
                        if any(b in k for b in ("Bull 2021","2024","2023","2025"))
                        and v < -0.2)
        spy_loss = sum(1 for k, v in diffs_s.items() if v < 0)
        print(f"  → bull losses vs base: {bull_loss},  windows lagging SPY: {spy_loss}/{len(diffs_s)}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
