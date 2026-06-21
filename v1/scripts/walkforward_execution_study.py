"""
Walk-forward study focused on EXECUTION QUALITY (not Sharpe tuning).

Compares V4N-F stack under:
  - rebalance cadence: daily | signal_only | monthly
  - top_n: 11 vs 14 (on available universe)

Reports OOS gross/net return, fee drag, trades/year, % rebalance churn.

Run:
  PYTHONPATH=. python v1/scripts/walkforward_execution_study.py

After adding tickers to data_pipeline.py, refresh data first:
  python run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v1.config.params import LIVE_TOP_N
from v1.pipeline.data_pipeline import TICKER_LIST
from v1.pipeline.signal_generation import ATR_PARTIAL_REMAIN
from v1.portfolio.portfolio import (
    CAPITAL,
    top_n_adx_momt_ac_sizes,
    diversifier_sleeve_overlay,
    profit_take_overlay,
    cond_vol_carry_overlay,
    asym_vol_boost_overlay,
    fear_topRS_concentration_overlay,
    accel_kicker_overlay,
    bull_sleeve_swap_overlay,
    apply_live_cadence_to_sizes,
    defensive_tilt_overlay,
    _get_macro,
    sharpe_ratio,
    max_drawdown,
    SIGNAL_DIR,
    FEATURE_DIR,
)
from v1.portfolio.execution_metrics import (
    returns_with_fees,
    summarize_execution,
)

RESULTS_PATH = Path("data/v1/results/walkforward_execution_study.json")
TRAIN_YEARS, TEST_YEARS = 3, 1


def load_panel() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Load multi_signals + per-ticker features/returns (tickers with parquets)."""
    multi = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    feats: dict = {}
    rets = pd.DataFrame()
    available = []
    for t in TICKER_LIST:
        fp = FEATURE_DIR / f"{t}.parquet"
        if not fp.exists():
            continue
        f = pd.read_parquet(fp)
        feats[t] = f
        rets[t] = f["log_return"]
        available.append(t)
    rets = rets.dropna(how="all")
    sig = multi.reindex(columns=rets.columns, fill_value=0.0).reindex(rets.index).fillna(0.0)
    hs_path = SIGNAL_DIR / "half_size.parquet"
    if hs_path.exists():
        hs = pd.read_parquet(hs_path).reindex(rets.index).fillna(False)
        hs = hs.reindex(columns=sig.columns, fill_value=False)
        sig = sig * np.where(hs, ATR_PARTIAL_REMAIN, 1.0)
    macro = _get_macro().reindex(rets.index).ffill()
    print(f"  Universe: {len(available)}/{len(TICKER_LIST)} tickers with features")
    missing = [t for t in TICKER_LIST if t not in available]
    if missing:
        print(f"  Missing (run `python run.py`): {', '.join(missing)}")
    return sig, feats, rets, macro


def build_v4nf_sizes(
    sig: pd.DataFrame,
    feats: dict,
    rets: pd.DataFrame,
    macro: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    sz = top_n_adx_momt_ac_sizes(
        sig, feats, rets, CAPITAL,
        top_n=top_n, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0, adx_threshold=22.0,
        mom_lo=0.7, mom_hi=1.3, ac_quota=0.55, gross_floor=0.95,
    )
    sz = diversifier_sleeve_overlay(
        sz, CAPITAL, sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"), sleeve_pct=0.12,
    )
    sz = profit_take_overlay(sz, rets, lookback=10, sigma_thresh=1.5, scale=0.7)
    sz = cond_vol_carry_overlay(sz, macro, fear_z=1.5, roc_days=5, fear_mult=0.5)
    sz = asym_vol_boost_overlay(sz, macro, calm_boost=1.15, calm_z=-0.5,
                              fear_cut=0.9, fear_z=1.0)
    sz = fear_topRS_concentration_overlay(sz, feats, macro, top_k=3, fear_z=1.0)
    sz = accel_kicker_overlay(sz, feats, accel_thresh=1.05, accel_boost=1.35)
    sz = bull_sleeve_swap_overlay(sz, macro, CAPITAL, src="TLT", dst="XLK", swap_pct=0.10)
    return sz


def walkforward_oos(
    sig: pd.DataFrame,
    feats: dict,
    rets: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    top_n: int,
    cadence: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling 3y/1y OOS with execution metrics per window."""
    train_days = TRAIN_YEARS * 252
    test_days = TEST_YEARS * 252
    wf_rows = []
    period_labels = []
    start = train_days

    while start + test_days <= len(sig):
        ctx_start = max(0, start - train_days)
        all_sig = sig.iloc[ctx_start: start + test_days]
        all_ret = rets.iloc[ctx_start: start + test_days]
        all_macro = macro.iloc[ctx_start: start + test_days]

        base = build_v4nf_sizes(all_sig, feats, all_ret, all_macro, top_n)
        base = defensive_tilt_overlay(base, all_sig, all_macro, CAPITAL)
        sizes = apply_live_cadence_to_sizes(
            base, all_sig, cadence=cadence, rebalance_every=21,
        )
        test_sz = sizes.iloc[-test_days:]
        test_ret = all_ret.iloc[-test_days:]

        gross, net, activity = returns_with_fees(test_sz, test_ret)
        gross = gross.dropna()
        net = net.dropna()

        y0 = sig.index[start].year
        y1 = sig.index[start + test_days - 1].year
        period = f"{y0}-{y1}"
        period_labels.append(period)

        summ = summarize_execution(gross, net, activity, period)
        summ["period"] = period
        summ["sharpe_gross"] = round(float(sharpe_ratio(gross)), 3)
        summ["sharpe_net"] = round(float(sharpe_ratio(net)), 3)
        summ["max_dd_pct"] = round(
            float(max_drawdown((1 + gross).cumprod()) * 100), 2,
        )
        wf_rows.append(summ)
        start += test_days

    wf = pd.DataFrame(wf_rows)
    return wf, pd.DataFrame()  # second df reserved


def aggregate_wf(wf: pd.DataFrame) -> dict:
    if wf.empty:
        return {}
    num_cols = [
        "ann_return_gross_pct", "ann_return_net_pct", "fee_drag_pp",
        "trades_per_year", "rebalance_pct", "sharpe_gross", "sharpe_net",
        "max_dd_pct", "turnover_x",
    ]
    out = {}
    for c in num_cols:
        if c in wf.columns:
            out[f"mean_{c}"] = round(float(wf[c].mean()), 3)
            out[f"min_{c}"] = round(float(wf[c].min()), 3)
    out["total_entries"] = int(wf["entries"].sum())
    out["total_exits"] = int(wf["exits"].sum())
    out["total_rebalances"] = int(wf["rebalances"].sum())
    out["n_windows"] = len(wf)
    return out


def full_sample_summary(
    sig: pd.DataFrame,
    feats: dict,
    rets: pd.DataFrame,
    macro: pd.DataFrame,
    top_n: int,
    cadence: str,
) -> dict:
    base = build_v4nf_sizes(sig, feats, rets, macro, top_n)
    base = defensive_tilt_overlay(base, sig, macro, CAPITAL)
    sizes = apply_live_cadence_to_sizes(base, sig, cadence=cadence, rebalance_every=21)
    gross, net, activity = returns_with_fees(sizes, rets)
    return summarize_execution(
        gross.dropna(), net.dropna(), activity,
        f"full_{cadence}_top{top_n}",
    )


def main():
    t0 = time.time()
    print("Walk-forward execution study (3y train / 1y test)")
    print("=" * 72)
    sig, feats, rets, macro = load_panel()
    print(f"  {len(rets)} trading days  ({rets.index[0].date()} → {rets.index[-1].date()})\n")

    variants = []
    for top_n in (11, LIVE_TOP_N):
        for cadence in ("daily", "signal_only", "monthly"):
            variants.append((top_n, cadence))

    all_results = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "variants": {}}

    for top_n, cadence in variants:
        key = f"top{top_n}_{cadence}"
        print(f"\n--- {key} ---")
        wf, _ = walkforward_oos(sig, feats, rets, macro, top_n=top_n, cadence=cadence)
        agg = aggregate_wf(wf)
        full = full_sample_summary(sig, feats, rets, macro, top_n, cadence)
        all_results["variants"][key] = {
            "walk_forward_windows": wf.to_dict(orient="records"),
            "oos_aggregate": agg,
            "full_sample": full,
        }
        if agg:
            print(
                f"  OOS mean net {agg.get('mean_ann_return_net_pct', 0):+.1f}%  "
                f"gross {agg.get('mean_ann_return_gross_pct', 0):+.1f}%  "
                f"fee drag {agg.get('mean_fee_drag_pp', 0):.2f}pp  "
                f"trades/yr {agg.get('mean_trades_per_year', 0):.0f}  "
                f"rebal% {agg.get('mean_rebalance_pct', 0):.0f}%  "
                f"Sh(net) {agg.get('mean_sharpe_net', 0):.2f}"
            )

    # Rank by net return then low rebalance %
    ranking = []
    for key, v in all_results["variants"].items():
        a = v.get("oos_aggregate", {})
        ranking.append({
            "variant": key,
            "net_ann_pct": a.get("mean_ann_return_net_pct", -999),
            "gross_ann_pct": a.get("mean_ann_return_gross_pct", -999),
            "fee_drag_pp": a.get("mean_fee_drag_pp", 999),
            "trades_per_year": a.get("mean_trades_per_year", 999),
            "rebalance_pct": a.get("mean_rebalance_pct", 999),
            "sharpe_net": a.get("mean_sharpe_net", 0),
        })
    ranking = sorted(
        ranking,
        key=lambda x: (-x["net_ann_pct"], x["rebalance_pct"], x["fee_drag_pp"]),
    )
    all_results["ranking"] = ranking
    all_results["recommendation"] = ranking[0] if ranking else None

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 72)
    print("RANKING (OOS net return, then lower rebalance %)")
    print("-" * 72)
    for i, r in enumerate(ranking, 1):
        print(
            f"  {i}. {r['variant']:<22}  net {r['net_ann_pct']:+.1f}%  "
            f"fee {r['fee_drag_pp']:.2f}pp  trades/yr {r['trades_per_year']:.0f}  "
            f"rebal {r['rebalance_pct']:.0f}%  Sh {r['sharpe_net']:.2f}"
        )
    if ranking:
        best = ranking[0]["variant"]
        print(f"\nRecommended live config: {best}")
        print(f"Set LIVE_REBALANCE_MODE / LIVE_TOP_N in v1/config/params.py accordingly.")
    print(f"\nSaved → {RESULTS_PATH}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
