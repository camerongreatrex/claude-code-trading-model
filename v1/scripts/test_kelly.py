"""
test_kelly.py
-------------
Standalone comparison of Kelly variants vs production atr_sizes under the
$10k per-name cap (MAX_POSITION_PCT=0.10).

Half-Kelly (default in kelly_sizes()): fraction × 0.5
Full-Kelly: same logic without the 0.5 dampener
Quarter-Kelly: × 0.25

Reports ann return / Sharpe / max DD on each, and the max single-name
position notional reached over the full backtest.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes, kelly_sizes, defensive_tilt_overlay,
    portfolio_returns, summarise, MAX_POSITION_PCT, CAPITAL,
    compute_strategy_returns,
)


def kelly_sizes_scaled(signals: pd.DataFrame, returns: pd.DataFrame,
                        capital: float, scale: float, lookback: int = 252) -> pd.DataFrame:
    """Same as kelly_sizes() but with a configurable Kelly fraction multiplier."""
    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for ticker in signals.columns:
        if ticker not in returns.columns:
            continue
        strat_ret = compute_strategy_returns(signals[ticker], returns[ticker])
        rolling_wr = strat_ret.rolling(lookback).apply(
            lambda x: float((x[x != 0] > 0).mean()) if (x != 0).any() else 0.5
        )
        rolling_pf = strat_ret.rolling(lookback).apply(
            lambda x: float(x[x > 0].sum() / x[x < 0].abs().sum())
            if x[x < 0].abs().sum() > 0 else 1.0
        )
        kelly = (rolling_wr - (1 - rolling_wr) / rolling_pf.clip(0.01)).clip(0) * scale
        sizes[ticker] = (signals[ticker] * kelly * capital).clip(
            -capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT
        )
    return sizes.fillna(0)


def main() -> None:
    SIGNAL_DIR  = Path("data/v1/signals")
    FEATURE_DIR = Path("data/v1/features")
    MACRO_DIR   = Path("data/shared/macro")

    signals_raw = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    features: dict = {}
    returns = pd.DataFrame()
    for t in signals_raw.columns:
        p = FEATURE_DIR / f"{t}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            features[t] = f
            returns[t] = f["log_return"]

    returns = returns.dropna()    # match production pipeline
    signals = signals_raw.reindex(returns.index).fillna(0)

    macro_path = MACRO_DIR / "macro_features.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else pd.DataFrame()

    # ── Build size variants ────────────────────────────────────────────────
    variants: dict[str, pd.DataFrame] = {}
    print("Building size matrices...")

    sz_atr = atr_sizes(signals, features, CAPITAL)
    variants["atr_pure (production)"] = sz_atr

    sz_half = kelly_sizes_scaled(signals, returns, CAPITAL, scale=0.5)
    variants["half_kelly"] = sz_half

    sz_full = kelly_sizes_scaled(signals, returns, CAPITAL, scale=1.0)
    variants["full_kelly"] = sz_full

    sz_qtr = kelly_sizes_scaled(signals, returns, CAPITAL, scale=0.25)
    variants["quarter_kelly"] = sz_qtr

    # 70/30 atr/half-kelly blend — keep atr's Sharpe but borrow Kelly's return lift.
    # Re-clip after blending in case scale-up pushed any name above $10k.
    sz_blend_70 = (0.7 * sz_atr + 0.3 * sz_half).clip(
        -CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT
    )
    variants["atr_70_kelly_30"] = sz_blend_70

    sz_blend_50 = (0.5 * sz_atr + 0.5 * sz_half).clip(
        -CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT
    )
    variants["atr_50_kelly_50"] = sz_blend_50

    # ── Apply defensive overlay + portfolio gross rescale + compute returns
    print("\n{:<28} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
        "Method", "AnnRet%", "Sharpe", "MaxDD%", "Calmar", "MaxPos$"
    ))
    print("-" * 86)

    for name, sz in variants.items():
        sz_overlay = defensive_tilt_overlay(sz, signals, macro, CAPITAL)
        ret = portfolio_returns(sz_overlay, returns)
        s = summarise(ret, name)
        max_pos = sz_overlay.abs().max().max()
        print(f"{s['label']:<28} {s['ann_return']:>10} {s['sharpe']:>10} "
              f"{s['max_drawdown']:>10} {s['calmar']:>10} ${max_pos:>11,.0f}")


if __name__ == "__main__":
    main()
