"""
Slow universe expansion on top of V4N-E production stack.

V4N-E baseline (42 tickers): Sh 2.58 / Ann 23.98% / DD -4.23% / Cal 5.67.

Goal: add small batches of liquid S&P 500 names that *improve all three*
of Sharpe, AnnRet, and (don't worsen) MaxDD.  Reject any batch that hurts
even one metric.

Approach:
  1. Generate signals for each expansion ticker via the same generate()
     function used for core tickers (with ASSET_CLASS monkey-patched).
  2. For each test batch (curated subset of NEW_TICKERS), build a
     combined-universe multi_signals DataFrame and run the full V4N-E
     pipeline (top11+ADX22+momt+AC55 + all overlays).
  3. Compare Sh / Ann / DD / Cal vs the 42-ticker baseline.

Batches tested:
  T2  : top-2 by ADV per sector (~22 names)
  T1  : top-1 by ADV per sector (~11 names) — minimal expansion
  XLK : full XLK sector additions (deepest tech)
  XLV : full XLV sector additions (defensive diversifier)
  XLY : full XLY sector additions (consumer disc — adds TSLA, HD)
  MEGA: AAPL, GOOGL, META, AVGO, LLY, UNH, TSLA, HD only (8 megacaps)
  ALL : full 114-name expansion (for reference; expected to fail)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Patch ASSET_CLASS for expansion tickers BEFORE importing modules that read it
from v1.pipeline.data_pipeline import ASSET_CLASS as _CORE_AC, TICKERS as _CORE_TICKERS
from v1.pipeline.universe_expansion import NEW_TICKERS, SECTOR_MAP, SECTOR_STOCKS

import v1.pipeline.data_pipeline as _dp
import v1.pipeline.signal_generation as _sg

for _t in NEW_TICKERS:
    if _t not in _dp.ASSET_CLASS:
        _dp.ASSET_CLASS[_t] = "stock"
    if _t not in _sg.ASSET_CLASS:
        _sg.ASSET_CLASS[_t] = "stock"

from v1.pipeline.signal_generation import generate, ATR_PARTIAL_REMAIN
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


# ── Generate signals for a single expansion ticker (cached) ─────────────────
_SIG_CACHE: dict[str, pd.Series] = {}


def get_signal_multi(ticker: str, macro: pd.DataFrame) -> pd.Series | None:
    """Return signal_multi series for a ticker (cached, generated on demand)."""
    if ticker in _SIG_CACHE:
        return _SIG_CACHE[ticker]
    feat_path = FEATURE_DIR / f"{ticker}.parquet"
    if not feat_path.exists():
        _SIG_CACHE[ticker] = None
        return None
    try:
        feat = pd.read_parquet(feat_path)
        sig_df = generate(feat, ticker, macro)
        if "signal_multi" not in sig_df.columns:
            _SIG_CACHE[ticker] = None
            return None
        s = sig_df["signal_multi"].astype(float)
        _SIG_CACHE[ticker] = s
        return s
    except Exception as e:
        print(f"  [{ticker}] signal gen failed: {e}")
        _SIG_CACHE[ticker] = None
        return None


# ── Build expanded inputs ────────────────────────────────────────────────────
def load_core_inputs():
    """Same as test_v4nd_lift but returns separately for splice."""
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
    sig_core = multi.reindex(rets.index).fillna(0.0)
    hs_path = SIGNAL_DIR / "half_size.parquet"
    if hs_path.exists():
        hs = pd.read_parquet(hs_path).reindex(rets.index).fillna(False)
        hs = hs.reindex(columns=sig_core.columns, fill_value=False)
        sig_core = sig_core * np.where(hs, ATR_PARTIAL_REMAIN, 1.0)
    return sig_core, feats, rets, _get_macro()


def expanded_inputs(extra_tickers: list[str], sig_core, feats_core, rets_core, macro):
    """Return (sig_exp, feats_exp, rets_exp) with extras spliced in."""
    sig_exp = sig_core.copy()
    feats_exp = dict(feats_core)
    rets_exp = rets_core.copy()
    for t in extra_tickers:
        if t in sig_exp.columns:
            continue
        s = get_signal_multi(t, macro)
        if s is None:
            continue
        # Reindex to base index
        s = s.reindex(rets_exp.index).fillna(0.0)
        sig_exp[t] = s
        feat_path = FEATURE_DIR / f"{t}.parquet"
        if feat_path.exists():
            f = pd.read_parquet(feat_path)
            feats_exp[t] = f
            rets_exp[t] = f["log_return"].reindex(rets_exp.index).fillna(0.0)
    return sig_exp, feats_exp, rets_exp


# ── Full V4N-E pipeline ─────────────────────────────────────────────────────
def build_v4ne(sig, feats, rets, macro):
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
                                     sleeve_pct=0.12)
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


def report(name, sizes, rets, base_dd=None, base_ann=None, base_sh=None):
    pr = portfolio_returns(sizes, rets).dropna()
    m = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
    if not m:
        print(f"{name:<45} insufficient data")
        return None
    flags = ""
    if base_sh  is not None: flags += "S" if m["sharpe"] >= base_sh else "·"
    if base_ann is not None: flags += "A" if m["ann"]    >= base_ann else "·"
    if base_dd  is not None: flags += "D" if m["mdd"]    >= base_dd  else "·"
    print(f"{name:<45} {m['sharpe']:>5.2f} {m['ann']*100:>6.2f}% "
          f"{m['mdd']*100:>6.2f}% {m['calmar']:>5.2f} {m['vol']*100:>5.1f}%  [{flags}]")
    return m | {"name": name, "returns": pr}


# ── Curated batches ─────────────────────────────────────────────────────────
def get_top_per_sector(k: int) -> list[str]:
    """Top-k by SECTOR_STOCKS list order (already ADV-sorted in source)."""
    out = []
    for sec, names in SECTOR_STOCKS.items():
        out.extend(names[:k])
    return out


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    sig_core, feats_core, rets_core, macro = load_core_inputs()
    print(f"Loaded {len(rets_core)} days, {len(rets_core.columns)} core tickers in "
          f"{time.time()-t0:.1f}s\n")

    # ── V4N-E baseline ──────────────────────────────────────────────────────
    print(f"{'Variant':<45} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}  [SAD]")
    base_sizes = build_v4ne(sig_core, feats_core, rets_core, macro)
    base_m = report("V4N-E baseline (42 core)", base_sizes, rets_core)
    base_dd, base_ann, base_sh = base_m["mdd"], base_m["ann"], base_m["sharpe"]
    base_pr = base_m["returns"]

    # ── Pre-generate signals for all candidates (warm cache) ────────────────
    print(f"\nGenerating signals for {len(NEW_TICKERS)} expansion tickers...")
    sig_t = time.time()
    ok = 0
    for t in NEW_TICKERS:
        if get_signal_multi(t, macro) is not None:
            ok += 1
    print(f"  {ok}/{len(NEW_TICKERS)} signals generated in {time.time()-sig_t:.1f}s\n")

    print(f"{'Variant':<45} {'Sh':>5} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}  [SAD]")
    results = [base_m]

    # ── Curated batches ─────────────────────────────────────────────────────
    batches = {
        "MEGA-8 (top conviction megacaps)":
            ["AAPL", "GOOGL", "META", "AVGO", "LLY", "UNH", "TSLA", "HD"],
        "MEGA-12 (megacaps + leaders)":
            ["AAPL", "GOOGL", "META", "AVGO", "LLY", "UNH", "TSLA", "HD",
             "ORCL", "AMD", "NFLX", "WMT"],
        "T1 (top-1/sector by ADV)":
            get_top_per_sector(1),
        "T2 (top-2/sector by ADV)":
            get_top_per_sector(2),
        "T3 (top-3/sector by ADV)":
            get_top_per_sector(3),
        "XLK only (tech depth)":
            SECTOR_STOCKS["XLK"],
        "XLV only (healthcare diversifier)":
            SECTOR_STOCKS["XLV"],
        "XLY only (consumer disc — TSLA/HD)":
            SECTOR_STOCKS["XLY"],
        "DEFENSIVE (XLP+XLU+XLRE)":
            SECTOR_STOCKS["XLP"] + SECTOR_STOCKS["XLU"] + SECTOR_STOCKS["XLRE"],
        "ALL 114 (full expansion)":
            list(NEW_TICKERS),
    }

    for name, extras in batches.items():
        sig_e, feats_e, rets_e = expanded_inputs(extras, sig_core, feats_core,
                                                    rets_core, macro)
        sizes = build_v4ne(sig_e, feats_e, rets_e, macro)
        m = report(f"{name:<45}", sizes, rets_e,
                    base_dd=base_dd, base_ann=base_ann, base_sh=base_sh)
        if m:
            results.append(m)

    # ── Leaderboard: only batches passing ALL THREE gates ────────────────────
    print(f"\n{'='*80}")
    print(f"Baseline:  Sh {base_sh:.2f}  Ann {base_ann*100:.2f}%  DD {base_dd*100:.2f}%")
    print(f"Pass gate: Sh >= base AND Ann >= base AND DD >= base (no degradation)")
    print(f"{'='*80}\n")
    pass_all = [r for r in results
                if r["name"] != "V4N-E baseline (42 core)"
                and r["sharpe"] >= base_sh
                and r["ann"]    >= base_ann
                and r["mdd"]    >= base_dd]
    pass_all.sort(key=lambda x: -x["ann"])
    if not pass_all:
        print("  NONE — no batch improves all three metrics simultaneously.\n")
    else:
        print(f"  Passing batches ({len(pass_all)}):")
        for r in pass_all:
            print(f"    {r['name']:<45} Sh {r['sharpe']:.2f} (+{r['sharpe']-base_sh:+.2f})  "
                  f"Ann {r['ann']*100:.2f}% (+{(r['ann']-base_ann)*100:+.2f}pp)  "
                  f"DD {r['mdd']*100:.2f}% ({(r['mdd']-base_dd)*100:+.2f}pp)")

    # ── Soft-pass: improves Sh + Ann, DD within 0.2pp ───────────────────────
    soft = [r for r in results
            if r["name"] != "V4N-E baseline (42 core)"
            and r["sharpe"] >= base_sh
            and r["ann"]    >= base_ann
            and r["mdd"]    >= base_dd - 0.002]
    soft = [r for r in soft if r not in pass_all]
    if soft:
        print(f"\n  Soft-pass (Sh+Ann pass, DD within 0.2pp):")
        soft.sort(key=lambda x: -x["ann"])
        for r in soft[:5]:
            print(f"    {r['name']:<45} Sh {r['sharpe']:.2f}  "
                  f"Ann {r['ann']*100:.2f}%  DD {r['mdd']*100:.2f}%  "
                  f"(DD Δ {(r['mdd']-base_dd)*100:+.2f}pp)")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
