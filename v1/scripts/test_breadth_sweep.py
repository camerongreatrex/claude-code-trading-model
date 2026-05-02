"""
Breadth-focused sweep: keep CAND-A (cap 12%, top11, ac55, mom63) as new
baseline and test variants that increase the *number* of names actually
held each day.  User reported the strategy concentrating in 4 names
(NVDA-driven), causing single-name DD when NVDA pulled back.

Variants tested:
  - Higher top_n (15, 18, 22, 28) — let more names into the pool
  - Lower ADX threshold (15, 18, 20) — fewer names get filtered out
  - Equal-weight overlay (within selected longs, ignore vol-target sizing)
  - "Min positions floor" — if fewer than X names selected, expand selection
  - Diversifier sleeve — small forced equal-weight allocation across uncorrelated
                          assets (TLT, GLD, IWM) on top of momentum picks

We measure:
  - mean # active names (count > $500 position) on each day
  - distribution of #names: median, p25, p75
  - max single-name share of gross
  - all standard return/risk metrics on the OOS slice
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import v1.portfolio.portfolio as _pp
from v1.portfolio.portfolio import (
    top_n_adx_momt_ac_sizes,
    portfolio_returns,
    defensive_tilt_overlay,
    _get_macro,
    SIGNAL_DIR,
    FEATURE_DIR,
    CAPITAL,
)
from v1.pipeline.data_pipeline import TICKER_LIST
from v1.pipeline.signal_generation import ATR_PARTIAL_REMAIN
from v1.pipeline.backtester import sharpe_ratio, max_drawdown

OOS_WARMUP = 756
ACTIVE_THRESHOLD = 500.0  # $500 = "real" position, not floating dust


def sortino_ratio(ret: pd.Series, target: float = 0.0) -> float:
    excess = ret - target / 252
    dd = excess[excess < 0]
    if len(dd) < 2:
        return float("nan")
    s = dd.std() * np.sqrt(252)
    return float(excess.mean() * 252 / s) if s > 0 else float("nan")


def metrics(ret: pd.Series) -> dict:
    if len(ret) < 60:
        return {}
    ann = (1 + ret).prod() ** (252 / len(ret)) - 1
    return dict(
        ann=ann,
        vol=ret.std() * np.sqrt(252),
        sharpe=sharpe_ratio(ret),
        sortino=sortino_ratio(ret),
        mdd=max_drawdown((1 + ret).cumprod()),
        calmar=ann / abs(max_drawdown((1 + ret).cumprod()))
        if max_drawdown((1 + ret).cumprod()) < 0 else float("nan"),
    )


def breadth_stats(sizes: pd.DataFrame) -> dict:
    abs_sz = sizes.abs()
    n_active = (abs_sz > ACTIVE_THRESHOLD).sum(axis=1)
    gross = abs_sz.sum(axis=1).replace(0, np.nan)
    max_share = abs_sz.max(axis=1) / gross
    return dict(
        n_mean=float(n_active.mean()),
        n_median=float(n_active.median()),
        n_p25=float(n_active.quantile(0.25)),
        n_p75=float(n_active.quantile(0.75)),
        n_min=float(n_active.min()),
        max_single_share_p95=float(max_share.quantile(0.95)),
        max_single_share_avg=float(max_share.mean()),
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


def equal_weight_overlay(sizes: pd.DataFrame, top_n: int, capital: float) -> pd.DataFrame:
    """Replace per-name vol-target dollars with equal-weight within active longs.
    Each active name gets capital/top_n (or 1/n_active if more names active)."""
    out = sizes.copy()
    abs_sz = out.abs()
    active_mask = abs_sz > 1.0  # any non-zero size = active
    n_act = active_mask.sum(axis=1).clip(lower=1)
    target_per_name = (capital * 0.95 / n_act.clip(upper=top_n))  # gross 95%
    sign = np.sign(out)
    out = sign.multiply(target_per_name, axis=0).where(active_mask, 0.0)
    return out.clip(-capital * _pp.MAX_POSITION_PCT, capital * _pp.MAX_POSITION_PCT)


def add_diversifier_sleeve(sizes: pd.DataFrame, capital: float,
                           tickers: list, sleeve_pct: float) -> pd.DataFrame:
    """Force long allocation in given tickers as a permanent diversifier sleeve.
    Reduces existing positions proportionally so total gross stays ≤ 1.0×."""
    out = sizes.copy()
    sleeve_dollar = capital * sleeve_pct
    each = sleeve_dollar / max(len(tickers), 1)
    available = [t for t in tickers if t in out.columns]
    if not available:
        return out
    # Scale existing down by (1 - sleeve_pct) to make room
    out = out * (1.0 - sleeve_pct)
    for t in available:
        out[t] = out[t] + each
    return out.clip(-capital * _pp.MAX_POSITION_PCT, capital * _pp.MAX_POSITION_PCT)


def run_variant(name, sig, feats, rets, macro,
                kwargs=None, equal_weight=False,
                diversifier=None, diversifier_pct=0.10):
    kwargs = kwargs or {}
    base_kwargs = dict(
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )
    base_kwargs.update(kwargs)
    sizes = top_n_adx_momt_ac_sizes(sig, feats, rets, CAPITAL, **base_kwargs)
    if equal_weight:
        sizes = equal_weight_overlay(sizes, base_kwargs["top_n"], CAPITAL)
    if diversifier:
        sizes = add_diversifier_sleeve(sizes, CAPITAL, diversifier, diversifier_pct)
    sizes = defensive_tilt_overlay(sizes, sig, macro, CAPITAL)
    pr = portfolio_returns(sizes, rets).dropna()

    # Slice OOS for breadth + perf
    oos_sizes = sizes.iloc[OOS_WARMUP:]
    bs = breadth_stats(oos_sizes)
    m = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else {}
    return dict(name=name, **m, **bs,
                gross_avg=float((oos_sizes.abs().sum(axis=1) / CAPITAL).mean()))


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in {time.time()-t0:.1f}s\n")

    variants = [
        # === Baseline (NEW: CAND-A) ===
        ("CAND-A baseline (cap12,top11)",     dict()),

        # === Higher top_n — let more names in ===
        ("top15",                             dict(top_n=15)),
        ("top18",                             dict(top_n=18)),
        ("top22",                             dict(top_n=22)),
        ("top28 (almost full universe)",      dict(top_n=28)),

        # === Lower ADX threshold — let more names qualify ===
        ("adx18",                             dict(adx_threshold=18)),
        ("adx15",                             dict(adx_threshold=15)),
        ("adx12",                             dict(adx_threshold=12)),
        ("adx0 (no ADX filter)",              dict(adx_threshold=0)),

        # === Bigger pool + lower filter — expect better breadth ===
        ("top18 + adx18",                     dict(top_n=18, adx_threshold=18)),
        ("top22 + adx15",                     dict(top_n=22, adx_threshold=15)),
        ("top28 + adx12",                     dict(top_n=28, adx_threshold=12)),
        ("top28 + adx0",                      dict(top_n=28, adx_threshold=0)),

        # === Pool-expansion + tighter ac_quota — diversification floor ===
        ("top22 + adx15 + ac0.40",            dict(top_n=22, adx_threshold=15, ac_quota=0.40)),
        ("top28 + adx12 + ac0.35",            dict(top_n=28, adx_threshold=12, ac_quota=0.35)),

        # === Equal-weight overlay (kills NVDA dominance via vol-target) ===
        ("EW overlay top15",                  dict(top_n=15)),  # equal_weight set below
        ("EW overlay top22",                  dict(top_n=22)),  # equal_weight set below
        ("EW overlay top28",                  dict(top_n=28)),  # equal_weight set below
    ]

    rows = []
    for name, kw in variants:
        ts = time.time()
        ew = name.startswith("EW overlay")
        try:
            r = run_variant(name, sig, feats, rets, macro, kwargs=kw, equal_weight=ew)
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            continue
        rows.append(r)
        print(f"  {name:<40} Sh {r['sharpe']:.3f}  "
              f"Ann {r['ann']*100:5.2f}%  "
              f"DD {r['mdd']*100:6.2f}%  "
              f"Cal {r['calmar']:5.2f}  "
              f"# {r['n_median']:.0f}/{r['n_p75']:.0f}  "
              f"max1 {r['max_single_share_avg']*100:.0f}%  "
              f"gross {r['gross_avg']*100:.0f}%")

    # Diversifier sleeve variants on top of best breadth picks
    div_variants = [
        # Sleeve size sweep on bonds+gold (uncorrelated to equity momentum)
        ("CAND-A + 5% TLT/GLD",       dict(), ["TLT", "GLD"], 0.05),
        ("CAND-A + 8% TLT/GLD",       dict(), ["TLT", "GLD"], 0.08),
        ("CAND-A + 10% TLT/GLD",      dict(), ["TLT", "GLD"], 0.10),
        ("CAND-A + 15% TLT/GLD",      dict(), ["TLT", "GLD"], 0.15),
        ("CAND-A + 20% TLT/GLD",      dict(), ["TLT", "GLD"], 0.20),
        # Try DBMF (managed futures) — positive carry, low equity correlation
        ("CAND-A + 8% TLT/GLD/DBMF",  dict(), ["TLT", "GLD", "DBMF"], 0.08),
        ("CAND-A + 12% TLT/GLD/DBMF", dict(), ["TLT", "GLD", "DBMF"], 0.12),
        ("CAND-A + 15% TLT/GLD/DBMF/WTMF",
                                       dict(), ["TLT", "GLD", "DBMF", "WTMF"], 0.15),
        # Just managed futures, no bonds
        ("CAND-A + 10% DBMF/WTMF",    dict(), ["DBMF", "WTMF"], 0.10),
        ("CAND-A + 15% DBMF/WTMF",    dict(), ["DBMF", "WTMF"], 0.15),
        # Wider diversifier — bonds, gold, MF, short-bonds
        ("CAND-A + 12% TLT/GLD/DBMF/VGSH",
                                       dict(), ["TLT", "GLD", "DBMF", "VGSH"], 0.12),
        # Equity-beta sleeve (won't help DD but will boost beta exposure on dull days)
        ("CAND-A + 10% SPY/IWM/QQQ",  dict(), ["SPY", "IWM", "QQQ"], 0.10),

        # ── Wide-diversifier permanent sleeves to hit 50% universe target ──
        # 8 ETFs across asset classes — gets us to 11+8 = 19 active names
        ("CAND-A + 16% wide-8",       dict(),
            ["TLT", "GLD", "VGSH", "DBMF", "WTMF", "VWO", "EFA", "TIP"], 0.16),
        ("CAND-A + 20% wide-8",       dict(),
            ["TLT", "GLD", "VGSH", "DBMF", "WTMF", "VWO", "EFA", "TIP"], 0.20),
        # 10 ETFs — gets us to ~21 active = 50% of universe
        ("CAND-A + 20% wide-10",      dict(),
            ["TLT", "GLD", "VGSH", "DBMF", "WTMF", "VWO", "EFA", "TIP", "MUB", "EMB"], 0.20),
        ("CAND-A + 25% wide-10",      dict(),
            ["TLT", "GLD", "VGSH", "DBMF", "WTMF", "VWO", "EFA", "TIP", "MUB", "EMB"], 0.25),
        # Concentrated 4-asset diversifier (best per-name conviction, less breadth)
        ("CAND-A + 12% TLT/GLD/DBMF/VGSH (4-asset)", dict(),
            ["TLT", "GLD", "DBMF", "VGSH"], 0.12),

        # ── Aiming for 50%+ universe (21+ names = 11 momentum + 10+ sleeve) ──
        # 14-asset sleeve: bonds, gold, MF, intl, FX, defensive
        ("CAND-A + 20% wide-14",      dict(),
            ["TLT","GLD","VGSH","DBMF","WTMF","VWO","EFA","TIP","MUB","EMB","IWM","EWJ","FXE","UUP"], 0.20),
        ("CAND-A + 25% wide-14",      dict(),
            ["TLT","GLD","VGSH","DBMF","WTMF","VWO","EFA","TIP","MUB","EMB","IWM","EWJ","FXE","UUP"], 0.25),
        ("CAND-A + 30% wide-14",      dict(),
            ["TLT","GLD","VGSH","DBMF","WTMF","VWO","EFA","TIP","MUB","EMB","IWM","EWJ","FXE","UUP"], 0.30),
    ]
    for name, kw, tickers, pct in div_variants:
        try:
            r = run_variant(name, sig, feats, rets, macro,
                            kwargs=kw, diversifier=tickers, diversifier_pct=pct)
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            continue
        rows.append(r)
        print(f"  {name:<40} Sh {r['sharpe']:.3f}  "
              f"Ann {r['ann']*100:5.2f}%  "
              f"DD {r['mdd']*100:6.2f}%  "
              f"Cal {r['calmar']:5.2f}  "
              f"# {r['n_median']:.0f}/{r['n_p75']:.0f}  "
              f"max1 {r['max_single_share_avg']*100:.0f}%  "
              f"gross {r['gross_avg']*100:.0f}%")

    df = pd.DataFrame(rows).set_index("name")

    print("\n=== Top 8 by Sharpe (must have median ≥ 8 names active) ===")
    qual = df[df["n_median"] >= 8].sort_values("sharpe", ascending=False).head(8)
    print(qual[["sharpe", "ann", "mdd", "calmar", "n_median", "n_p75",
                "max_single_share_avg", "gross_avg"]].round(3).to_string())

    print("\n=== Top 8 by AnnRet (must have median ≥ 8 names active) ===")
    qual = df[df["n_median"] >= 8].sort_values("ann", ascending=False).head(8)
    print(qual[["sharpe", "ann", "mdd", "calmar", "n_median", "n_p75",
                "max_single_share_avg", "gross_avg"]].round(3).to_string())

    print("\n=== Top 8 by Calmar with breadth ≥ 10 names ===")
    qual = df[df["n_median"] >= 10].sort_values("calmar", ascending=False).head(8)
    print(qual[["sharpe", "ann", "mdd", "calmar", "n_median", "n_p75",
                "max_single_share_avg", "gross_avg"]].round(3).to_string())

    out = Path("data/v1/results/breadth_sweep.csv")
    df.to_csv(out)
    print(f"\nSaved -> {out}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
