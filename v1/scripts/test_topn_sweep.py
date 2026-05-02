"""
Sweep variants of `top_n_adx_momt_ac_sizes` (the V1 production sizer) and
report OOS metrics so we can see which knob, if any, beats the pinned
`top11_adx22_momt_ac55_cap1` on Sharpe / AnnRet / MaxDD / Calmar / Sortino
without ever using leverage.

Baseline pinned in v1/config/params.py:
    top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
    lev_x=1.5, max_gross=1.0, adx_threshold=22.0,
    mom_window=63, mom_lo=0.7, mom_hi=1.3, ac_quota=0.55,
    gross_floor=0.95.

Every variant is strictly zero-leverage (max_gross=1.0).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v1.portfolio.portfolio import (  # noqa: E402
    top_n_adx_momt_ac_sizes,
    portfolio_returns,
    defensive_tilt_overlay,
    _get_macro,
    SIGNAL_DIR,
    FEATURE_DIR,
    CAPITAL,
)
from v1.pipeline.data_pipeline import TICKER_LIST  # noqa: E402
from v1.pipeline.signal_generation import ATR_PARTIAL_REMAIN  # noqa: E402
from v1.pipeline.backtester import sharpe_ratio, max_drawdown  # noqa: E402

OOS_WARMUP = 756  # 3 trading years, matches dashboard


def sortino_ratio(ret: pd.Series, target: float = 0.0) -> float:
    excess = ret - target / 252
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    dd_std = downside.std() * np.sqrt(252)
    return float(excess.mean() * 252 / dd_std) if dd_std > 0 else float("nan")


def metrics(ret: pd.Series) -> dict:
    if len(ret) < 60:
        return dict(ann=float("nan"), vol=float("nan"), sharpe=float("nan"),
                    sortino=float("nan"), mdd=float("nan"), calmar=float("nan"))
    ann = (1 + ret).prod() ** (252 / len(ret)) - 1
    vol = ret.std() * np.sqrt(252)
    sh = sharpe_ratio(ret)
    so = sortino_ratio(ret)
    mdd = max_drawdown((1 + ret).cumprod())
    cal = ann / abs(mdd) if mdd < 0 else float("nan")
    return dict(ann=ann, vol=vol, sharpe=sh, sortino=so, mdd=mdd, calmar=cal)


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

    macro = _get_macro()
    return sig, feats, rets, macro


def run_variant(name: str, sig, feats, rets, macro, **kwargs) -> dict:
    base_kwargs = dict(
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )
    base_kwargs.update(kwargs)
    sizes = top_n_adx_momt_ac_sizes(sig, feats, rets, CAPITAL, **base_kwargs)
    sizes = defensive_tilt_overlay(sizes, sig, macro, CAPITAL)
    pr = portfolio_returns(sizes, rets).dropna()

    full = metrics(pr)
    oos = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else dict()
    gross_pct = (sizes.abs().sum(axis=1) / CAPITAL)
    return dict(
        name=name,
        ann_oos=oos.get("ann"), vol_oos=oos.get("vol"),
        sharpe_oos=oos.get("sharpe"), sortino_oos=oos.get("sortino"),
        mdd_oos=oos.get("mdd"), calmar_oos=oos.get("calmar"),
        sharpe_full=full["sharpe"], mdd_full=full["mdd"],
        gross_avg=float(gross_pct.mean()),
        gross_max=float(gross_pct.max()),
        days_over_1x=int((gross_pct > 1.0001).sum()),
    )


def main():
    t0 = time.time()
    print("Loading inputs...")
    sig, feats, rets, macro = load_inputs()
    print(f"  loaded {len(rets)} days, {len(rets.columns)} tickers, "
          f"{len(feats)} features in {time.time()-t0:.1f}s\n")

    # Allow temporary per-name cap override via module attribute monkey-patch
    import v1.portfolio.portfolio as _pp
    _orig_cap = _pp.MAX_POSITION_PCT

    def with_cap(cap_pct: float, fn):
        _pp.MAX_POSITION_PCT = cap_pct
        try:
            return fn()
        finally:
            _pp.MAX_POSITION_PCT = _orig_cap

    variants = [
        ("BASELINE (top11/ac55/mom63/gf95, cap10)", {}),

        # Round-2 stacking: best Calmar driver was top_n=10, best Sharpe driver
        # was ac_quota=0.45-0.50, and gfloor=0.90 gave a small DD improvement.
        # Test all sensible combinations of those three.
        ("top10",                            dict(top_n=10)),
        ("top10 + ac0.45",                   dict(top_n=10, ac_quota=0.45)),
        ("top10 + ac0.50",                   dict(top_n=10, ac_quota=0.50)),
        ("top10 + ac0.45 + gf0.90",          dict(top_n=10, ac_quota=0.45, gross_floor=0.90)),
        ("top10 + ac0.50 + gf0.90",          dict(top_n=10, ac_quota=0.50, gross_floor=0.90)),
        ("top10 + ac0.45 + mom84",           dict(top_n=10, ac_quota=0.45, mom_window=84)),
        ("top10 + ac0.50 + mom84",           dict(top_n=10, ac_quota=0.50, mom_window=84)),
        ("top10 + ac0.45 + tvol0.12",        dict(top_n=10, ac_quota=0.45, target_vol=0.12)),
        ("top10 + ac0.50 + tvol0.12",        dict(top_n=10, ac_quota=0.50, target_vol=0.12)),
        ("top10 + ac0.45 + mb0.6-1.4",       dict(top_n=10, ac_quota=0.45, mom_lo=0.6, mom_hi=1.4)),
        ("top10 + ac0.45 + adx20",           dict(top_n=10, ac_quota=0.45, adx_threshold=20)),
        ("top10 + ac0.45 + adx25",           dict(top_n=10, ac_quota=0.45, adx_threshold=25)),
        ("top10 + ac0.45 + gf1.00",          dict(top_n=10, ac_quota=0.45, gross_floor=1.00)),

        # Also try top_n=11 with ac0.45 (keep current top_n)
        ("top11 + ac0.45",                   dict(ac_quota=0.45)),
        ("top11 + ac0.50",                   dict(ac_quota=0.50)),
        ("top11 + ac0.45 + gf0.90",          dict(ac_quota=0.45, gross_floor=0.90)),
        ("top11 + ac0.45 + mom84",           dict(ac_quota=0.45, mom_window=84)),

        # Edge case: top_n=10 with all-best
        ("top10 + ac0.45 + mom84 + gf0.90",  dict(top_n=10, ac_quota=0.45, mom_window=84, gross_floor=0.90)),
        ("top10 + ac0.50 + mom84 + gf0.90",  dict(top_n=10, ac_quota=0.50, mom_window=84, gross_floor=0.90)),
    ]

    # Per-name cap variants — push gross up by allowing 11% or 12% per name
    # so concentrated top-10 can still deploy near the 95% floor.
    cap_variants = [
        (0.11, "cap11 + top10 + ac0.50 + mom84",   dict(top_n=10, ac_quota=0.50, mom_window=84)),
        (0.11, "cap11 + top10 + ac0.45 + mom84",   dict(top_n=10, ac_quota=0.45, mom_window=84)),
        (0.12, "cap12 + top10 + ac0.50 + mom84",   dict(top_n=10, ac_quota=0.50, mom_window=84)),
        (0.12, "cap12 + top10 + ac0.45 + mom84",   dict(top_n=10, ac_quota=0.45, mom_window=84)),
        (0.11, "cap11 + top11 + baseline",         {}),
        (0.12, "cap12 + top11 + baseline",         {}),
    ]

    rows = []
    for name, kw in variants:
        ts = time.time()
        try:
            r = run_variant(name, sig, feats, rets, macro, **kw)
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            continue
        rows.append(r)
        print(f"  {name:<40} OOS Sh {r['sharpe_oos']:.3f}  "
              f"Ann {r['ann_oos']*100:5.2f}%  "
              f"DD {r['mdd_oos']*100:6.2f}%  "
              f"Cal {r['calmar_oos']:5.2f}  "
              f"Sort {r['sortino_oos']:.2f}  "
              f"gross {r['gross_avg']*100:.1f}%  "
              f"({time.time()-ts:.1f}s)")

    # Per-name cap sweeps (need module monkey-patch)
    for cap_pct, name, kw in cap_variants:
        ts = time.time()
        try:
            r = with_cap(cap_pct,
                lambda: run_variant(name, sig, feats, rets, macro, **kw))
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            continue
        rows.append(r)
        print(f"  {name:<40} OOS Sh {r['sharpe_oos']:.3f}  "
              f"Ann {r['ann_oos']*100:5.2f}%  "
              f"DD {r['mdd_oos']*100:6.2f}%  "
              f"Cal {r['calmar_oos']:5.2f}  "
              f"Sort {r['sortino_oos']:.2f}  "
              f"gross {r['gross_avg']*100:.1f}%  "
              f"({time.time()-ts:.1f}s)")

    df = pd.DataFrame(rows).set_index("name")

    print("\n=== Top 8 by OOS Sharpe ===")
    print(df.sort_values("sharpe_oos", ascending=False).head(8)[
        ["sharpe_oos", "ann_oos", "mdd_oos", "calmar_oos", "sortino_oos", "gross_avg"]
    ].round(4).to_string())

    print("\n=== Top 8 by Calmar (return per drawdown unit) ===")
    print(df.sort_values("calmar_oos", ascending=False).head(8)[
        ["sharpe_oos", "ann_oos", "mdd_oos", "calmar_oos", "sortino_oos", "gross_avg"]
    ].round(4).to_string())

    print("\n=== Top 8 by Min Drawdown (least negative) ===")
    print(df.sort_values("mdd_oos", ascending=False).head(8)[
        ["sharpe_oos", "ann_oos", "mdd_oos", "calmar_oos", "sortino_oos", "gross_avg"]
    ].round(4).to_string())

    out = Path("data/v1/results/topn_sweep.csv")
    df.to_csv(out)
    print(f"\nFull table -> {out}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
