"""
Conditional sector-short overlay test.

Recipe (more selective than test_long_short_alpha.py's unconditional L/S):
  For each rebal day, short the worst sector ETF(s) IFF ALL hold:
    1. signal_multi == 0    (regime turned flat/down)
    2. adx >= adx_thresh    (trend strength confirmed)
    3. 63d return is in bottom quartile of the sector universe

  Equal-$ across selected shorts; total short notional = sleeve_pct * capital.
  Layered on V3 base by reducing V3 to (1 - sleeve_pct) and adding shorts.
  Combined gross capped at max_gross (zero leverage) so live = backtest.

Run as walk-forward (skip OOS warmup) and report Sh/Ann/DD/Cal vs V3 baseline.
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
    portfolio_returns,
    defensive_tilt_overlay,
    profit_take_overlay,
    cond_vol_carry_overlay,
    asym_vol_boost_overlay,
    fear_topRS_concentration_overlay,
    accel_kicker_overlay,
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


def build_conditional_short_sizes(
    rets: pd.DataFrame, feats: dict, sig: pd.DataFrame, capital: float,
    sectors: list, lookback: int = 63, adx_thresh: float = 22.0,
    bottom_q: float = 0.25, sleeve_pct: float = 0.10,
    rebal_freq: str = "W-FRI", max_shorts: int = 3,
) -> pd.DataFrame:
    """
    Conditional short sleeve.  At each rebal day, candidates are sector ETFs
    where signal_multi == 0 AND adx >= adx_thresh AND 63d return is in the
    bottom `bottom_q` of the sector universe.  Up to `max_shorts` worst by
    63d return are shorted equal-$, totaling sleeve_pct * capital.
    """
    sec = [t for t in sectors if t in rets.columns]
    R = rets[sec]

    # 63d cumulative log return (skip recent 5d for reversal safety)
    cum63 = R.rolling(lookback).sum().shift(5)

    # ADX matrix
    adx = pd.DataFrame(0.0, index=R.index, columns=sec)
    for t in sec:
        if t in feats and "adx" in feats[t].columns:
            adx[t] = feats[t]["adx"].reindex(R.index).ffill().fillna(0.0)

    # Signal matrix (regime flat/down = 0)
    sig_sec = sig.reindex(columns=sec, fill_value=0.0).reindex(R.index).fillna(0.0)

    sizes = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    rebal_days = R.index.to_series().groupby(pd.Grouper(freq=rebal_freq)).max().dropna()

    for d in rebal_days:
        if d not in cum63.index:
            continue
        row_ret = cum63.loc[d].dropna()
        row_adx = adx.loc[d]
        row_sig = sig_sec.loc[d]

        # Eligibility: signal flat AND ADX strong
        eligible = [t for t in row_ret.index
                    if row_sig.get(t, 1.0) == 0
                    and row_adx.get(t, 0.0) >= adx_thresh]
        if not eligible:
            continue

        # Bottom quartile within full sector universe
        q_threshold = row_ret.quantile(bottom_q)
        bottom = [t for t in eligible if row_ret[t] <= q_threshold]
        if not bottom:
            continue

        # Worst N (most negative cum return)
        worst = sorted(bottom, key=lambda t: row_ret[t])[:max_shorts]
        each = (sleeve_pct * capital) / len(worst)
        for t in worst:
            sizes.loc[d, t] = -each

    sizes = sizes.replace(0.0, np.nan).ffill().fillna(0.0)
    return sizes


def make_v4ne_sizer():
    """Full V4N-E production stack — same as paper trading."""
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
        s = profit_take_overlay(s, rets, lookback=10, sigma_thresh=1.5,
                                  scale=0.7, max_gross=1.0, capital=CAPITAL)
        s = cond_vol_carry_overlay(s, macro, fear_z=1.5, roc_days=5, fear_mult=0.5)
        s = asym_vol_boost_overlay(s, macro,
                                      calm_boost=1.15, calm_z=-0.5,
                                      fear_cut=0.9, fear_z=1.0)
        s = fear_topRS_concentration_overlay(s, feats, macro,
                                                top_k=3, fear_z=1.0, rs_window=63)
        s = accel_kicker_overlay(s, feats,
                                    accel_thresh=1.05, accel_boost=1.35,
                                    short_w=10, long_w=42)
        return s

    return sizer


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in "
          f"{time.time()-t0:.1f}s")

    sectors = [t for t, c in TICKERS.items() if c == "sector_etf" and t in rets.columns]
    print(f"Sector universe ({len(sectors)}): {sectors}\n")

    # V3 baseline (production sizer)
    sizer_v3 = make_v4ne_sizer()
    sizes_v3 = sizer_v3(sig, feats, rets, macro)
    pr_v3 = portfolio_returns(sizes_v3, rets).dropna()
    m_v3 = metrics(pr_v3.iloc[OOS_WARMUP:])
    print(f"{'V4N-E baseline (no short overlay)':<58}"
          f" {m_v3['sharpe']:>5.2f} {m_v3['ann']*100:>6.2f}% "
          f"{m_v3['mdd']*100:>6.2f}% {m_v3['calmar']:>5.2f} {m_v3['vol']*100:>5.1f}%")

    print(f"\n{'Variant':<58} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")

    # Sweep adx_thresh × bottom_q × sleeve_pct × max_shorts × rebal_freq
    grid = []
    for adx_t in (22.0, 25.0, 30.0):
        for bq in (0.20, 0.25, 0.33):
            for sp in (0.05, 0.08, 0.12, 0.15):
                for ms in (1, 2, 3):
                    for rf in ("W-FRI", "M"):
                        grid.append((adx_t, bq, sp, ms, rf))

    results = []
    for adx_t, bq, sp, ms, rf in grid:
        sh = build_conditional_short_sizes(
            rets, feats, sig, CAPITAL, sectors,
            adx_thresh=adx_t, bottom_q=bq, sleeve_pct=sp,
            max_shorts=ms, rebal_freq=rf,
        )
        # V3 reduced + shorts added; cap combined gross at 1× capital
        v3_scaled = sizes_v3 * (1.0 - sp)
        combined  = v3_scaled.add(sh, fill_value=0.0)
        gross     = combined.abs().sum(axis=1).replace(0, np.nan)
        cap_scale = (CAPITAL / gross).clip(upper=1.0).fillna(1.0)
        combined  = combined.multiply(cap_scale, axis=0)

        pr = portfolio_returns(combined, rets).dropna()
        m  = metrics(pr.iloc[OOS_WARMUP:])
        if not m:
            continue
        name = f"sp={sp:.2f} adx={adx_t:.0f} bq={bq:.2f} ms={ms} {rf}"
        results.append((m["sharpe"], m["ann"], m["mdd"], m["calmar"], m["vol"], name))

    # Top-15 by Sharpe
    results.sort(key=lambda x: -x[0])
    print("\nTop 15 by Sharpe:")
    for sh, ann, dd, cal, vol, name in results[:15]:
        print(f"{name:<58} {sh:>5.2f} {ann*100:>6.2f}% {dd*100:>6.2f}% {cal:>5.2f} {vol*100:>5.1f}%")

    # Top-15 by AnnRet (with DD < V4N-E baseline)
    print(f"\nTop 15 by AnnRet (DD <= V4N-E baseline {m_v3['mdd']*100:.2f}%):")
    safe = [r for r in results if r[2] >= m_v3["mdd"]]   # DD less negative
    safe.sort(key=lambda x: -x[1])
    for sh, ann, dd, cal, vol, name in safe[:15]:
        print(f"{name:<58} {sh:>5.2f} {ann*100:>6.2f}% {dd*100:>6.2f}% {cal:>5.2f} {vol*100:>5.1f}%")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
