"""
Window-robustness backtest for V4N-B and top candidates.

Computes each candidate's daily returns ONCE on the full series, then slices
across multiple windows (calendar years, regime windows, rolling 252d).
Reports Sh/Ann/DD/Cal per window per strategy, plus the distribution of rolling
1y stats so we can see how stable each variant is across time.

Candidates tested:
  base : V4N-B baseline (V3 + profit_take + cond_vol_carry)
  D    : V4N-B + vol_lev_boost(boost=1.15, calm_z=-0.5)
  O    : V4N-B + asym_vol(cb=1.15, fc=0.9, fz=1.0)            ← new winner
  O2   : V4N-B + asym_vol(cb=1.20, fc=0.9, fz=1.5)
  K5   : V4N-B + D + H-vix-spike(z>=0.5, sp=0.02)
  Q    : V4N-B + D + H-spike + M-term-hedge stack

Data span: 2019-08-07 → 2025-12-31 (~6.4y). OOS warmup ~756d → OOS from ~2022-08.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v1.scripts.test_smart_shorts import (
    load_inputs, build_v3, apply_v4nb_overlays,
    add_vol_lev_boost, asym_vol_boost,
    vix_spike_hedge, vix_term_hedge,
    _layer_short, metrics, OOS_WARMUP,
)
from v1.portfolio.portfolio import portfolio_returns, CAPITAL


def fmt(m):
    if not m:
        return "  n/a"
    return (f"Sh {m['sharpe']:>5.2f}  Ann {m['ann']*100:>6.2f}%  "
            f"DD {m['mdd']*100:>6.2f}%  Cal {m['calmar']:>5.2f}")


def slice_metrics(pr: pd.Series, start: str, end: str) -> dict:
    s = pr.loc[start:end].dropna()
    return metrics(s)


def build_candidates(sig, feats, rets, macro):
    """Build returns series for each candidate."""
    base_v3 = build_v3(sig, feats, rets)
    base = apply_v4nb_overlays(base_v3, rets, macro)

    # D: boost only
    d_sizes = add_vol_lev_boost(base, macro, boost=1.15, calm_z=-0.5)

    # O: asym (calm boost + fear cut)
    o_sizes = asym_vol_boost(base, macro, calm_boost=1.15,
                                calm_z=-0.5, fear_cut=0.9, fear_z=1.0)

    # O2: more aggressive asym
    o2_sizes = asym_vol_boost(base, macro, calm_boost=1.20,
                                 calm_z=-0.5, fear_cut=0.9, fear_z=1.5)

    # K5: D + H-spike
    sh_h = vix_spike_hedge(rets, feats, macro,
                              hedges=("SPY", "QQQ"),
                              z_thresh=0.5, short_pct=0.02)
    k5_d = add_vol_lev_boost(base, macro, boost=1.15, calm_z=-0.3)
    k5_sizes = _layer_short(k5_d, sh_h, 0.02)

    # Q: D + H + M (term)
    q_d = add_vol_lev_boost(base, macro, boost=1.15, calm_z=-0.5)
    q1 = _layer_short(q_d, sh_h, 0.02)
    sh_m = vix_term_hedge(rets, feats, macro, term_z_thresh=1.0, short_pct=0.02)
    q_sizes = _layer_short(q1, sh_m, 0.02)

    candidates = {
        "base": base,
        "D":    d_sizes,
        "O":    o_sizes,
        "O2":   o2_sizes,
        "K5":   k5_sizes,
        "Q":    q_sizes,
    }
    returns = {name: portfolio_returns(s, rets).dropna()
               for name, s in candidates.items()}
    return returns


def windowed_table(returns: dict, windows: list):
    """Return DataFrame: rows = windows, cols = (name, metric)."""
    print(f"\n{'Window':<28} {'Days':>5}  ", end="")
    for name in returns:
        print(f"| {name:<10} ", end="")
    print()
    print("-" * (35 + 13 * len(returns)))

    # Per-window line: print each candidate's headline summary
    for label, start, end in windows:
        sub_lengths = [len(pr.loc[start:end].dropna()) for pr in returns.values()]
        n = max(sub_lengths) if sub_lengths else 0
        print(f"{label:<28} {n:>5}  ", end="")
        for name, pr in returns.items():
            m = slice_metrics(pr, start, end)
            if not m:
                print(f"|  insuf.    ", end="")
            else:
                print(f"| Sh{m['sharpe']:4.2f}/A{m['ann']*100:4.1f}", end="")
        print()


def detail_per_window(returns: dict, windows: list):
    """Print full per-window breakdown."""
    for label, start, end in windows:
        print(f"\n── {label}  ({start} → {end}) ──")
        for name, pr in returns.items():
            m = slice_metrics(pr, start, end)
            if not m:
                print(f"  {name:<5}  insufficient data")
                continue
            print(f"  {name:<5}  {fmt(m)}  vol {m['vol']*100:>4.1f}%  "
                  f"({len(pr.loc[start:end].dropna())}d)")


def rolling_distribution(returns: dict, window: int = 252):
    """Compute rolling 252d Sharpe/Ann/DD distribution across full OOS."""
    print(f"\n=== Rolling {window}d distribution (post-OOS warmup) ===")
    print(f"{'Strategy':<6} {'Sh p10':>7} {'Sh p50':>7} {'Sh p90':>7} "
          f"{'Sh min':>7} {'Ann p10':>8} {'Ann p50':>8} {'DD min':>8}")
    for name, pr in returns.items():
        # Rolling Sharpe
        roll_mean = pr.rolling(window).mean() * 252
        roll_std = pr.rolling(window).std() * np.sqrt(252)
        roll_sh = (roll_mean / roll_std).dropna()

        # Rolling annualized return
        roll_ann = ((1 + pr).rolling(window).apply(lambda x: x.prod(), raw=False) ** (252 / window) - 1).dropna()

        # Rolling DD: max DD over each 252d window
        def _mdd(s):
            cum = (1 + s).cumprod()
            return (cum / cum.cummax() - 1).min()
        roll_dd = pr.rolling(window).apply(_mdd, raw=False).dropna()

        # Filter to OOS only
        if len(roll_sh) > OOS_WARMUP:
            roll_sh = roll_sh.iloc[OOS_WARMUP:]
            roll_ann = roll_ann.iloc[OOS_WARMUP:]
            roll_dd = roll_dd.iloc[OOS_WARMUP:]

        print(f"{name:<6} {roll_sh.quantile(0.1):>7.2f} {roll_sh.quantile(0.5):>7.2f} "
              f"{roll_sh.quantile(0.9):>7.2f} {roll_sh.min():>7.2f} "
              f"{roll_ann.quantile(0.1)*100:>7.2f}% {roll_ann.quantile(0.5)*100:>7.2f}% "
              f"{roll_dd.min()*100:>7.2f}%")


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers, "
          f"{rets.index.min().date()} → {rets.index.max().date()} "
          f"in {time.time()-t0:.1f}s")

    returns = build_candidates(sig, feats, rets, macro)

    # Headline OOS metrics (skip first 756 days)
    print(f"\n=== Headline OOS (skip first {OOS_WARMUP} days) ===")
    for name, pr in returns.items():
        m = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
        print(f"  {name:<5}  {fmt(m)}  ({len(pr)-OOS_WARMUP}d)")

    # Calendar-year windows (only OOS years with >180d of data)
    cal_windows = [
        ("Full OOS",                "2022-08-01", "2025-12-31"),
        ("OOS H2 2022 (bear tail)", "2022-08-01", "2022-12-31"),
        ("2023 (recovery)",         "2023-01-01", "2023-12-31"),
        ("2024 (bull)",             "2024-01-01", "2024-12-31"),
        ("2025 YTD",                "2025-01-01", "2025-12-31"),
        ("2025 H1",                 "2025-01-01", "2025-06-30"),
        ("2025 H2",                 "2025-07-01", "2025-12-31"),
    ]

    # In-sample windows (just for sanity — pre-OOS regime context)
    is_windows = [
        ("Pre-COVID 2019-08→2020-02",   "2019-08-07", "2020-02-19"),
        ("COVID crash 2020-02→2020-04", "2020-02-20", "2020-04-30"),
        ("COVID recovery 2020 H2",      "2020-05-01", "2020-12-31"),
        ("Bull 2021",                    "2021-01-01", "2021-12-31"),
        ("Bear 2022 H1",                 "2022-01-01", "2022-06-30"),
    ]

    print("\n=== OOS calendar windows ===")
    detail_per_window(returns, cal_windows)

    print("\n=== In-sample windows (regime context, NOT used for selection) ===")
    detail_per_window(returns, is_windows)

    print("\n=== Compact OOS summary table ===")
    windowed_table(returns, cal_windows)

    rolling_distribution(returns, window=252)

    # Worst 60d window per strategy
    print("\n=== Worst 60d return window (post-OOS) ===")
    for name, pr in returns.items():
        oos = pr.iloc[OOS_WARMUP:] if len(pr) > OOS_WARMUP else pr
        roll60 = (1 + oos).rolling(60).apply(lambda x: x.prod(), raw=False) - 1
        worst = roll60.min()
        worst_d = roll60.idxmin()
        print(f"  {name:<5}  worst 60d {worst*100:>6.2f}% ending {worst_d.date()}")

    # Worst single day
    print("\n=== Worst single OOS day per strategy ===")
    for name, pr in returns.items():
        oos = pr.iloc[OOS_WARMUP:] if len(pr) > OOS_WARMUP else pr
        worst = oos.min()
        worst_d = oos.idxmin()
        print(f"  {name:<5}  worst day {worst*100:>6.2f}% on {worst_d.date()}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
