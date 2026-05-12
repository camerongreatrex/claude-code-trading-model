"""
Friction + bull-regime sleeve experiment on V4N-E.

Two attacks on the "loses on up days" problem:
  (a) Weekly rebalance — ffill V4N-E daily sizes within each W-FRI bucket so
      the live book doesn't churn intra-week.  Test with transaction cost
      sweep (0/5/10 bps roundtrip).
  (b) Bull-regime sleeve cut — sleeve_pct drops from 0.12 → 0.06 (or 0.04)
      when SPY > 50dMA AND VIX-z <= 0.  Different from prior R-family which
      used SPY 200dMA (too slow); 50dMA is more reactive.

Combined (a)+(b) is the lead candidate.  Reports win-rate vs SPY by year
and per-window AnnRet diffs to verify the bull-day deficit closes.
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


def build_v4ne(sig, feats, rets, macro, sleeve_pct=0.12):
    s = top_n_adx_momt_ac_sizes(
        sig, feats, rets, CAPITAL,
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )
    s = defensive_tilt_overlay(s, sig, macro, CAPITAL)
    s = diversifier_sleeve_overlay(s, CAPITAL,
                                     sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
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
                                accel_thresh=1.05, accel_boost=1.35,
                                short_w=10, long_w=42)
    return s


def bull_regime_sleeve(sig, feats, rets, macro,
                         sleeve_bull=0.06, sleeve_base=0.12,
                         spy_short_ma=50):
    """Build V4N-E twice (sleeve_bull, sleeve_base) and switch daily on regime."""
    spy = feats.get("SPY")
    vix_z = macro.get("vix_zscore") if macro is not None else None
    if spy is None or vix_z is None:
        return build_v4ne(sig, feats, rets, macro, sleeve_pct=sleeve_base)
    spy_close = spy["Close"].reindex(rets.index).ffill()
    spy_50 = spy_close.rolling(spy_short_ma).mean()
    vix_z_r = vix_z.reindex(rets.index).ffill().bfill()
    bull = ((spy_close > spy_50) & (vix_z_r <= 0)).shift(1).fillna(False).astype(bool)

    base_b = build_v4ne(sig, feats, rets, macro, sleeve_pct=sleeve_bull)
    base_n = build_v4ne(sig, feats, rets, macro, sleeve_pct=sleeve_base)
    out = base_n.copy()
    bull_idx = rets.index[bull.values]
    out.loc[bull_idx] = base_b.loc[bull_idx]
    return out


def weekly_rebalance(sizes: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    """
    Resample sizes to weekly cadence: take the value on each rebal day
    and ffill until next rebal day. Mirrors holding the book steady
    intra-week in live trading.
    """
    rebal_dates = sizes.index.to_series().groupby(pd.Grouper(freq=freq)).max().dropna()
    out = pd.DataFrame(np.nan, index=sizes.index, columns=sizes.columns)
    for d in rebal_dates:
        if d in sizes.index:
            out.loc[d] = sizes.loc[d]
    out = out.ffill().fillna(0.0)
    return out


def apply_costs(sizes: pd.DataFrame, returns: pd.Series, bps_one_way: float) -> pd.Series:
    """
    Apply transaction cost based on turnover.  Each bps is one-way (so a full
    flip from +X to -X costs 2 * bps).  Computes |Δposition_pct| per day
    and subtracts cost from the corresponding day's return.
    """
    if bps_one_way <= 0:
        return returns
    pos_pct = sizes.div(CAPITAL)
    turnover = pos_pct.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (bps_one_way / 10_000.0)
    out = returns.copy()
    out = out.subtract(cost.reindex(out.index).fillna(0.0), axis=0)
    return out


def report(name, sizes, rets, base_dd=None, base_ann=None, base_sh=None, cost_bps=0):
    pr_raw = portfolio_returns(sizes, rets).dropna()
    pr = apply_costs(sizes, pr_raw, cost_bps)
    m = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
    if not m:
        print(f"{name:<55} insufficient data")
        return None
    flags = ""
    if base_sh  is not None: flags += "S" if m["sharpe"] >= base_sh else "·"
    if base_ann is not None: flags += "A" if m["ann"]    >= base_ann else "·"
    if base_dd  is not None: flags += "D" if m["mdd"]    >= base_dd  else "·"
    # Turnover
    pos_pct = sizes.div(CAPITAL)
    turn = pos_pct.diff().abs().sum(axis=1).mean() * 252
    print(f"{name:<55} {m['sharpe']:>5.2f} {m['ann']*100:>6.2f}% "
          f"{m['mdd']*100:>6.2f}% {m['calmar']:>5.2f} {m['vol']*100:>5.1f}% "
          f"trn={turn:>5.1f}x  [{flags}]")
    return m | {"name": name, "returns": pr, "sizes": sizes, "turnover": turn}


def yearly_breakdown(pr: pd.Series, spy_pr: pd.Series, label: str):
    """AnnRet by year + win/loss vs SPY."""
    df = pd.DataFrame({"port": pr, "spy": spy_pr.reindex(pr.index)}).dropna()
    df["yr"] = df.index.year
    out = df.groupby("yr").apply(
        lambda g: pd.Series({
            "port_ann": (1 + g["port"]).prod() - 1,
            "spy_ann":  (1 + g["spy"]).prod() - 1,
        })
    )
    out["diff"] = (out["port_ann"] - out["spy_ann"]) * 100
    print(f"\n  {label}:")
    print(f"    {'Year':>5}  {'Port':>8}  {'SPY':>8}  {'Δ':>7}  win")
    for yr, row in out.iterrows():
        flag = "✓" if row["diff"] > 0 else "x"
        print(f"    {yr:>5}  {row['port_ann']*100:>7.2f}%  {row['spy_ann']*100:>7.2f}%"
              f"  {row['diff']:>+6.2f}  {flag}")
    wins = (out["diff"] > 0).sum()
    return wins, len(out)


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in "
          f"{time.time()-t0:.1f}s\n")

    spy_simple = (np.exp(feats["SPY"]["log_return"].reindex(rets.index).fillna(0.0)) - 1)

    print(f"{'Variant':<55} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6} {'Turn':>6}  [SAD]")

    # ── V4N-E baseline (daily, no costs) ────────────────────────────────────
    base_sizes = build_v4ne(sig, feats, rets, macro)
    base_m = report("V4N-E baseline (daily, 0bps)", base_sizes, rets)
    base_dd, base_ann, base_sh = base_m["mdd"], base_m["ann"], base_m["sharpe"]
    base_pr = base_m["returns"]

    # ── Cost sensitivity on baseline (shows friction headwind) ──────────────
    print("\n--- Cost sensitivity on V4N-E baseline ---")
    for c in (5, 10, 15):
        report(f"V4N-E baseline (daily, {c}bps)", base_sizes, rets,
               base_dd, base_ann, base_sh, cost_bps=c)

    # ── (a) Weekly rebalance ───────────────────────────────────────────────
    print("\n--- (a) Weekly rebalance (W-FRI) ---")
    weekly_sizes = weekly_rebalance(base_sizes, freq="W-FRI")
    for c in (0, 5, 10):
        report(f"Weekly W-FRI ({c}bps)", weekly_sizes, rets,
               base_dd, base_ann, base_sh, cost_bps=c)

    # Bi-weekly for comparison
    biweek_sizes = weekly_rebalance(base_sizes, freq="2W-FRI")
    for c in (0, 5, 10):
        report(f"Bi-weekly 2W-FRI ({c}bps)", biweek_sizes, rets,
               base_dd, base_ann, base_sh, cost_bps=c)

    # ── (b) Bull-regime sleeve cut (SPY > 50dMA AND VIX-z <= 0) ────────────
    print("\n--- (b) Bull-regime sleeve cut (sleeve_bull/sleeve_base) ---")
    bull_results = []
    for sb in (0.04, 0.06, 0.08):
        for sn in (0.10, 0.12):
            v = bull_regime_sleeve(sig, feats, rets, macro,
                                     sleeve_bull=sb, sleeve_base=sn)
            m = report(f"Bull sl={sb}/{sn} (50dMA+VIX-z≤0)", v, rets,
                       base_dd, base_ann, base_sh)
            if m: bull_results.append(m)

    # ── (a)+(b) combo ──────────────────────────────────────────────────────
    print("\n--- (a)+(b) combo: weekly + bull-regime sleeve ---")
    combo_results = []
    for sb in (0.04, 0.06, 0.08):
        v = bull_regime_sleeve(sig, feats, rets, macro,
                                 sleeve_bull=sb, sleeve_base=0.12)
        v_w = weekly_rebalance(v, freq="W-FRI")
        for c in (0, 5, 10):
            m = report(f"Combo sb={sb} W-FRI ({c}bps)", v_w, rets,
                       base_dd, base_ann, base_sh, cost_bps=c)
            if m: combo_results.append(m)

    # ── Headline: best combo at 5bps (realistic friction) ──────────────────
    print(f"\n{'='*100}\nLeaderboard: variants beating V4N-E baseline (Sh+Ann, DD within 0.2pp)\n{'='*100}\n")
    all_results = bull_results + combo_results
    safe = [r for r in all_results
            if r["sharpe"] >= base_sh - 0.05
            and r["ann"]    >= base_ann
            and r["mdd"]    >= base_dd - 0.002]
    safe.sort(key=lambda x: -x["ann"])
    for r in safe[:10]:
        print(f"  {r['name']:<55} Sh {r['sharpe']:.2f}  Ann {r['ann']*100:.2f}%  "
              f"DD {r['mdd']*100:.2f}%  Cal {r['calmar']:.2f}  trn {r['turnover']:.1f}x")

    # ── Yearly breakdown for top-3 to verify "up day" deficit closes ───────
    print(f"\n{'='*100}\nYearly win-rate vs SPY for top-3 candidates\n{'='*100}")
    # Always include baseline + best combo for comparison
    show = []
    if safe:
        show.extend(safe[:2])
    show.append(base_m)
    seen = set()
    for r in show:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        wins, total = yearly_breakdown(r["returns"], spy_simple, r["name"])
        print(f"  → {r['name']}: {wins}/{total} years beating SPY")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
