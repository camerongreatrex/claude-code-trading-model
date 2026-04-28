"""
test_voltgt_walkforward.py
--------------------------
Walk-forward validation of the top vol-target blend candidates from
test_voltgt_blend.py.  Confirms the IS Pareto improvement is not overfit.

Each candidate runs through 6 OOS test windows (2018-2023) using the
existing walk_forward harness with sizing_fn injected.

Run: python -m v1.scripts.test_voltgt_walkforward
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes, defensive_tilt_overlay,
    portfolio_returns, summarise, MAX_POSITION_PCT, CAPITAL,
    walk_forward,
)


def _clip(sz):
    return sz.clip(-CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT)


def vol_target_atr(signals, features, returns, capital,
                   target_vol=0.13, window=63, scale_min=0.3, scale_max=2.5):
    base = atr_sizes(signals, features, capital)
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    scaler = (target_vol / realized).clip(scale_min, scale_max).shift(1).fillna(1.0)
    return _clip(base.multiply(scaler, axis=0))


def make_blend_sizing_fn(features, alpha, target_vol, scale_max, lev_x):
    """Curry a (signals, returns) → sizes function for walk_forward."""
    def fn(signals, returns):
        sz_atr = atr_sizes(signals, features, CAPITAL)
        sz_vt  = vol_target_atr(signals, features, returns, CAPITAL,
                                target_vol=target_vol, scale_max=scale_max)
        sz_blend = _clip(alpha * sz_vt + (1 - alpha) * sz_atr)
        return _clip(sz_blend * lev_x)
    return fn


def make_atr_sizing_fn(features, lev_x):
    def fn(signals, returns):
        return _clip(atr_sizes(signals, features, CAPITAL) * lev_x)
    return fn


def main() -> None:
    SIGNAL_DIR  = Path("data/v1/signals")
    FEATURE_DIR = Path("data/v1/features")
    MACRO_DIR   = Path("data/shared/macro")

    print("Loading signals + features...")
    signals_raw = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    features: dict = {}
    returns = pd.DataFrame()
    for t in signals_raw.columns:
        p = FEATURE_DIR / f"{t}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            features[t] = f
            returns[t]  = f["log_return"]
    returns = returns.dropna()
    signals = signals_raw.reindex(returns.index).fillna(0)

    candidates = {
        "BASE atr_lev_1.5x":           make_atr_sizing_fn(features, 1.5),
        "BASE atr @1.0x":              make_atr_sizing_fn(features, 1.0),
        "blend_a25_vt12 @1.5x":        make_blend_sizing_fn(features, 0.25, 0.12, 1.5, 1.5),
        "blend_a25_vt14 @1.5x":        make_blend_sizing_fn(features, 0.25, 0.14, 1.5, 1.5),
        "blend_a35_vt14 @1.5x":        make_blend_sizing_fn(features, 0.35, 0.14, 1.5, 1.5),
        "blend_a50_vt14 @1.5x":        make_blend_sizing_fn(features, 0.50, 0.14, 1.5, 1.5),
        "blend_a75_vt14 @1.5x":        make_blend_sizing_fn(features, 0.75, 0.14, 1.5, 1.5),
        "blend_a80_vt13 @1.0x":        make_blend_sizing_fn(features, 0.80, 0.13, 2.0, 1.0),
        "blend_a65_vt14 @1.0x":        make_blend_sizing_fn(features, 0.65, 0.14, 2.0, 1.0),
    }

    print("\n" + "=" * 110)
    print("WALK-FORWARD OOS — 3y train / 1y test, 2018-2023 windows")
    print("=" * 110)

    for name, fn in candidates.items():
        try:
            wf = walk_forward(signals, returns, sizing_fn=fn)
        except Exception as e:
            print(f"{name:<32}  FAILED: {e}")
            continue
        mean_sh   = wf["sharpe"].mean()
        std_sh    = wf["sharpe"].std()
        mean_ret  = wf["ann_return"].mean()
        worst_dd  = wf["max_dd"].min()
        active_sh = wf["active_sharpe"].mean()
        print(f"\n  {name}")
        print(f"    {'period':<12} {'sharpe':>8} {'act_sh':>8} {'ann_ret':>8} {'max_dd':>8}")
        for _, row in wf.iterrows():
            print(f"    {row['period']:<12} {row['sharpe']:>8.3f} {row['active_sharpe']:>8.3f} "
                  f"{row['ann_return']:>8.2f} {row['max_dd']:>8.2f}")
        print(f"    {'MEAN':<12} {mean_sh:>8.3f} {active_sh:>8.3f} {mean_ret:>8.2f} {worst_dd:>8.2f} (worst)")


if __name__ == "__main__":
    main()
