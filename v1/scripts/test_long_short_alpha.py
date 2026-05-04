"""
Cross-sectional L/S momentum sleeve candidate to layer on V3
(top11_adx22_momt_ac55_cap1 + 12% sleeve). Universe: 13 stocks + 9 sector ETFs.

Daily: 63d return rank → long top-K, short bottom-K, equal-weight per leg,
beta-neutral ($long=$short), sized at sleeve_pct (e.g. 15%).
Tested standalone then layered on V3; combined gross ≤ 1.0× (zero leverage).
Rebal monthly; lookback 63d skipping recent 5d (avoid 1w reversal).
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
    diversifier_sleeve_overlay,
    portfolio_returns,
    defensive_tilt_overlay,
    _get_macro,
    SIGNAL_DIR, FEATURE_DIR, CAPITAL,
)
from v1.pipeline.data_pipeline import TICKERS, TICKER_LIST
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


def build_ls_sizes(rets: pd.DataFrame, capital: float, candidates: list,
                    k: int = 5, lookback: int = 63, skip: int = 5,
                    sleeve_pct: float = 0.15, rebal_freq: str = "M") -> pd.DataFrame:
    """
    Build long/short momentum dollar sizes.  Long top K, short bottom K, equal $.
    Rebalance monthly to avoid daily noise.
    """
    cands = [t for t in candidates if t in rets.columns]
    R = rets[cands]

    # Cum returns over lookback (skip recent 5d to dodge reversal)
    log_ret = R.copy()
    cum = log_ret.rolling(lookback, min_periods=int(lookback * 0.6)).sum().shift(skip)

    # Cross-sectional rank → month-end picks, ffill
    sizes = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    eom = R.index.to_series().groupby(pd.Grouper(freq=rebal_freq)).max()
    for d in eom:
        if d not in cum.index:
            continue
        row = cum.loc[d].dropna()
        if len(row) < 2 * k:
            continue
        ranked = row.sort_values()
        shorts = ranked.index[:k].tolist()
        longs = ranked.index[-k:].tolist()
        each = (sleeve_pct * capital) / k
        for t in longs:
            sizes.loc[d, t] = each
        for t in shorts:
            sizes.loc[d, t] = -each

    # ffill until next rebal, zero before first
    sizes = sizes.replace(0.0, np.nan).ffill().fillna(0.0)
    return sizes


def make_v3_sizer():
    base_kw = dict(
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )

    def sizer(sig, feats, rets, macro):
        s = top_n_adx_momt_ac_sizes(sig, feats, rets, CAPITAL, **base_kw)
        s = defensive_tilt_overlay(s, sig, macro, CAPITAL)
        s = diversifier_sleeve_overlay(
            s, CAPITAL,
            sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
            sleeve_pct=0.12,
        )
        return s

    return sizer


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in "
          f"{time.time()-t0:.1f}s")

    stocks = [t for t, c in TICKERS.items() if c == "stock" and t in rets.columns]
    sector_etfs = [t for t, c in TICKERS.items() if c == "sector_etf" and t in rets.columns]

    print(f"\nL/S universe: {len(stocks)} stocks + {len(sector_etfs)} sector ETFs")

    # ── Standalone L/S sleeve (no V3 base) ──
    print("\n=== STANDALONE L/S momentum (sleeve only, sleeve_pct=1.0 = full capital) ===")
    print(f"{'Variant':<55} {'Sh':>6} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    standalone_variants = [
        ("Stocks-only K=3 lookback=63",        stocks, dict(k=3, lookback=63)),
        ("Stocks-only K=4 lookback=63",        stocks, dict(k=4, lookback=63)),
        ("Stocks-only K=5 lookback=63",        stocks, dict(k=5, lookback=63)),
        ("Stocks-only K=4 lookback=126",       stocks, dict(k=4, lookback=126)),
        ("Sectors-only K=3 lookback=63",       sector_etfs, dict(k=3, lookback=63)),
        ("Sectors-only K=4 lookback=63",       sector_etfs, dict(k=4, lookback=63)),
        ("Sectors-only K=4 lookback=126",      sector_etfs, dict(k=4, lookback=126)),
        ("Combined-22 K=5 lookback=63",        stocks + sector_etfs, dict(k=5, lookback=63)),
        ("Combined-22 K=6 lookback=63",        stocks + sector_etfs, dict(k=6, lookback=63)),
        ("Combined-22 K=5 lookback=126",       stocks + sector_etfs, dict(k=5, lookback=126)),
    ]
    standalone_sizes = {}
    for name, univ, kw in standalone_variants:
        sizes = build_ls_sizes(rets, CAPITAL, univ, sleeve_pct=1.0, **kw)
        pr = portfolio_returns(sizes, rets).dropna()
        m = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
        standalone_sizes[name] = sizes
        print(f"{name:<55} {m.get('sharpe', float('nan')):>6.2f} "
              f"{m.get('ann', 0)*100:>6.2f}% {m.get('mdd', 0)*100:>6.2f}% "
              f"{m.get('calmar', float('nan')):>5.2f} {m.get('vol', 0)*100:>5.1f}%")

    # ── Layered: V3 base + L/S sleeve, varied sleeve_pct ──
    print("\n=== V3 + L/S sleeve combinations (V3 reduced by sleeve_pct, L/S added) ===")
    print(f"{'Variant':<55} {'Sh':>6} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")

    sizer_v3 = make_v3_sizer()
    sizes_v3 = sizer_v3(sig, feats, rets, macro)
    pr_v3 = portfolio_returns(sizes_v3, rets).dropna()
    m_v3 = metrics(pr_v3.iloc[OOS_WARMUP:])
    print(f"{'V3 baseline (no L/S)':<55} {m_v3['sharpe']:>6.2f} "
          f"{m_v3['ann']*100:>6.2f}% {m_v3['mdd']*100:>6.2f}% "
          f"{m_v3['calmar']:>5.2f} {m_v3['vol']*100:>5.1f}%")

    # Layer best 2-3 standalone L/S configs
    best_ls_configs = [
        ("Combined-22 K=5 lookback=126", stocks + sector_etfs, dict(k=5, lookback=126)),
        ("Sectors-only K=4 lookback=126", sector_etfs, dict(k=4, lookback=126)),
        ("Stocks-only K=4 lookback=126",  stocks, dict(k=4, lookback=126)),
    ]
    layered_sleeve_pcts = [0.05, 0.10, 0.15, 0.20]

    for ls_name, univ, ls_kw in best_ls_configs:
        for sp in layered_sleeve_pcts:
            ls_sizes = build_ls_sizes(rets, CAPITAL, univ, sleeve_pct=sp, **ls_kw)
            # V3*(1-sp) + L/S sleeve. L/S beta-neutral (gross=2*sp, net=0);
            # combined gross can exceed 1.0 (e.g. sp=0.15 → 1.15), so cap at 1.0×.
            v3_scaled = sizes_v3 * (1.0 - sp)
            combined = v3_scaled.add(ls_sizes, fill_value=0.0)
            # Zero-lev clip
            gross = combined.abs().sum(axis=1).replace(0, np.nan)
            cap_scale = (CAPITAL / gross).clip(upper=1.0).fillna(1.0)
            combined = combined.multiply(cap_scale, axis=0)

            pr = portfolio_returns(combined, rets).dropna()
            m = metrics(pr.iloc[OOS_WARMUP:])
            print(f"{'V3 + ' + str(int(sp*100)) + '% ' + ls_name:<55} "
                  f"{m['sharpe']:>6.2f} {m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
                  f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
