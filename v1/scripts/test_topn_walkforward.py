"""
Walk-forward (3y train / 1y test, rolling) on top sweep candidates to confirm
regime-stable wins (not just aggregate OOS).
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


def walk_forward(sig, feats, rets, macro, kwargs, train_years=3, test_years=1):
    train_days = train_years * 252
    test_days = test_years * 252
    out = []
    start = train_days
    while start + test_days <= len(sig):
        ctx_start = max(0, start - train_days)
        all_sig = sig.iloc[ctx_start: start + test_days]
        all_ret = rets.iloc[ctx_start: start + test_days]
        sizes = top_n_adx_momt_ac_sizes(all_sig, feats, all_ret, CAPITAL, **kwargs)
        sizes = defensive_tilt_overlay(sizes, all_sig, macro, CAPITAL)
        period = portfolio_returns(sizes.iloc[-test_days:], all_ret.iloc[-test_days:]).dropna()
        ann = (1 + period).prod() ** (252 / len(period)) - 1
        sh = sharpe_ratio(period)
        mdd = max_drawdown((1 + period).cumprod())
        y0 = sig.index[start].year
        y1 = sig.index[start + test_days - 1].year
        out.append(dict(period=f"{y0}-{y1}", sharpe=sh, ann=ann, mdd=mdd, n=len(period)))
        start += test_days
    return pd.DataFrame(out)


def run(name, kwargs, sig, feats, rets, macro, cap_pct=0.10):
    _pp.MAX_POSITION_PCT = cap_pct
    try:
        df = walk_forward(sig, feats, rets, macro, kwargs)
    finally:
        _pp.MAX_POSITION_PCT = 0.10
    print(f"\n=== {name}  (cap={cap_pct*100:.0f}%) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  mean Sh {df['sharpe'].mean():.3f}  mean Ann {df['ann'].mean()*100:.2f}%  "
          f"worst Sh {df['sharpe'].min():.3f}  worst DD {df['mdd'].min()*100:.2f}%")
    return df


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in {time.time()-t0:.1f}s")

    base = dict(top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
                lev_x=1.5, max_gross=1.0, adx_threshold=22.0,
                mom_window=63, mom_lo=0.7, mom_hi=1.3,
                ac_quota=0.55, gross_floor=0.95)

    candidates = [
        ("BASELINE  cap10 + top11 + ac55 + mom63", base, 0.10),
        ("CAND-A    cap12 + top11 + ac55 + mom63", base, 0.12),
        ("CAND-B    cap12 + top10 + ac45 + mom84",
            {**base, "top_n": 10, "ac_quota": 0.45, "mom_window": 84}, 0.12),
        ("CAND-C    cap10 + top10 + ac45 + mom84 + gf0.90",
            {**base, "top_n": 10, "ac_quota": 0.45, "mom_window": 84, "gross_floor": 0.90}, 0.10),
        ("CAND-D    cap12 + top10 + ac50 + mom84",
            {**base, "top_n": 10, "ac_quota": 0.50, "mom_window": 84}, 0.12),
    ]

    for name, kw, cap in candidates:
        run(name, kw, sig, feats, rets, macro, cap)

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
