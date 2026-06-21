"""
Compare backtest P&L under different live rebalance cadences.

  daily        — implicit in raw top-N size matrix (weight changes every day)
  signal_only  — hold until drop from top-N or signal=0 (matches new live default)
  monthly      — full rebalance every 21 trading days

Run: PYTHONPATH=. python v1/scripts/test_rebalance_cadence.py
Requires: data/v1/features/*.parquet and signal_multi in pipeline outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from v1.config.params import LIVE_TOP_N, V1_PRODUCTION_METHOD
from v1.pipeline.data_pipeline import TICKER_LIST, ASSET_CLASS
from v1.pipeline.signal_generation import load_macro
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
    portfolio_returns,
    apply_live_cadence_to_sizes,
    sharpe_ratio,
)
from v1.pipeline.feature_engineering import engineer
from v1.pipeline.signal_generation import generate

FEATURE_DIR = Path("data/v1/features")
START = "2018-01-01"


def _load_panel():
    features, returns, signals = {}, {}, None
    for t in TICKER_LIST:
        p = FEATURE_DIR / f"{t}.parquet"
        if not p.exists():
            continue
        raw = pd.read_parquet(p)
        raw.index = pd.to_datetime(raw.index)
        raw = raw[raw.index >= START]
        if len(raw) < 252:
            continue
        feat = engineer(raw)
        sig = generate(feat, load_macro())
        features[t] = feat
        returns[t] = feat["Close"].pct_change()
        if signals is None:
            signals = pd.DataFrame(index=feat.index)
        signals[t] = sig["signal_multi"]
    returns = pd.DataFrame(returns).dropna(how="all")
    signals = signals.reindex(returns.index).fillna(0)
    return features, returns, signals


def _build_v4nf_sizes(features, returns, signals, top_n: int) -> pd.DataFrame:
    sz = top_n_adx_momt_ac_sizes(
        signals, features, returns, CAPITAL,
        top_n=top_n, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0, adx_threshold=22.0,
        mom_lo=0.7, mom_hi=1.3, ac_quota=0.55,
    )
    macro = load_macro().reindex(returns.index).ffill()
    sz = diversifier_sleeve_overlay(sz, CAPITAL, sleeve_pct=0.12)
    sz = profit_take_overlay(sz, returns, lookback=10, sigma_thresh=1.5, scale=0.7)
    sz = cond_vol_carry_overlay(sz, macro, fear_z=1.5, roc_days=5, fear_mult=0.5)
    sz = asym_vol_boost_overlay(sz, macro, calm_boost=1.15, calm_z=-0.5,
                              fear_cut=0.9, fear_z=1.0)
    sz = fear_topRS_concentration_overlay(sz, features, macro, top_k=3, fear_z=1.0)
    sz = accel_kicker_overlay(sz, features, accel_thresh=1.05, accel_boost=1.35)
    sz = bull_sleeve_swap_overlay(sz, macro, CAPITAL, src="TLT", dst="XLK", swap_pct=0.10)
    return sz


def _summarize(ret: pd.Series, label: str) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {"label": label}
    ann = float(ret.mean() * 252)
    sh  = float(sharpe_ratio(ret))
    dd  = float(((1 + ret).cumprod() / (1 + ret).cumprod().cummax() - 1).min())
    turnover = float((ret != 0).mean())
    return {
        "label": label, "ann_ret": ann, "sharpe": sh, "max_dd": dd,
        "avg_daily_turnover": turnover,
    }


def main():
    print(f"Loading panel ({len(TICKER_LIST)} tickers in universe)...")
    features, returns, signals = _load_panel()
    print(f"  {len(features)} tickers with features, {len(returns)} days\n")

    for top_n in (11, LIVE_TOP_N):
        print(f"=== top_n={top_n} ({V1_PRODUCTION_METHOD}) ===")
        base = _build_v4nf_sizes(features, returns, signals, top_n)
        rows = []
        for cadence in ("daily", "signal_only", "monthly"):
            sz = apply_live_cadence_to_sizes(
                base, signals, cadence=cadence, rebalance_every=21,
            )
            ret = portfolio_returns(sz, returns)
            rows.append(_summarize(ret, cadence))
        df = pd.DataFrame(rows).set_index("label")
        print(df.to_string(float_format=lambda x: f"{x:+.4f}"))
        print()


if __name__ == "__main__":
    main()
