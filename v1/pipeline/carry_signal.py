"""
carry_signal.py — daily carry signals (return from HOLDING, uncorrelated
to momentum).

Signals: bond carry = 60d z-score of 10Y-2Y spread (applied to TLT/HYG/TIP/BWX);
commodity carry = 21d ret − (252d ret)/12 → futures-curve slope proxy
(GLD/DBC/UUP/DBA/FXE/FXY); equity carry = 0 (yfinance dividend data unreliable,
noisy proxy hurts more than helps).

Output: data/signals/carry_signals.parquet — all TICKER_LIST cols, values
[-1,+1], shifted(1) so yesterday's carry sizes today (no look-ahead).
Consumed by portfolio.multi_mom_carry_sizes/portable_carry, and signal_generation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .data_pipeline import TICKER_LIST, ASSET_CLASS

FEATURE_DIR = Path("data/v1/features")
MACRO_DIR   = Path("data/shared/macro")
SIGNAL_DIR  = Path("data/v1/signals")

BOND_TICKERS      = [t for t in TICKER_LIST if ASSET_CLASS.get(t) == "bond"]
COMMODITY_TICKERS = [t for t in TICKER_LIST if ASSET_CLASS.get(t) == "commodity"]


def bond_carry(macro_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Bond carry from 10Y-2Y slope: 60d z-score, clip ±2, /2 → [-1,+1],
    shift(1). Same signal for all bond tickers (uniform rate environment)."""
    if "yield_curve" not in macro_df.columns:
        return pd.DataFrame(0.0, index=index, columns=BOND_TICKERS)

    yc = macro_df["yield_curve"].reindex(index).ffill()

    # Rolling 60-day z-score
    mu     = yc.rolling(60, min_periods=20).mean()
    sd     = yc.rolling(60, min_periods=20).std().replace(0, np.nan)
    zscore = ((yc - mu) / sd).fillna(0.0)

    # Normalize to [-1, +1]
    signal = (zscore.clip(-2, 2) / 2).shift(1).fillna(0.0)

    return pd.DataFrame({t: signal for t in BOND_TICKERS})


def commodity_carry(features_dict: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Commodity carry: 21d_ret − (252d_ret/12) proxies futures-curve slope
    (>0 backwardation, <0 contango). 252d z-score, clip ±2, /2 → [-1,+1]."""
    result = {}
    for ticker in COMMODITY_TICKERS:
        if ticker not in features_dict or "Close" not in features_dict[ticker].columns:
            result[ticker] = pd.Series(0.0, index=index)
            continue

        close = features_dict[ticker]["Close"].reindex(index).ffill()

        ret21  = close.pct_change(21)
        ret252 = close.pct_change(252)
        ret252_monthly = ret252 / 12   # annualised → monthly equivalent

        carry_proxy = ret21 - ret252_monthly

        # Z-score over 252 days (1 year of data)
        mu     = carry_proxy.rolling(252, min_periods=60).mean()
        sd     = carry_proxy.rolling(252, min_periods=60).std().replace(0, np.nan)
        zscore = ((carry_proxy - mu) / sd).fillna(0.0)

        signal = (zscore.clip(-2, 2) / 2).shift(1).fillna(0.0)
        result[ticker] = signal

    return pd.DataFrame(result)


def compute_all_carry():
    """Orchestrate full-universe carry compute → carry_signals.parquet.
    Equity tickers filled 0; index aligned to macro_features."""
    # ── 1. Macro features ─────────────────────────────────────────────────────
    macro_path = MACRO_DIR / "macro_features.parquet"
    if not macro_path.exists():
        raise FileNotFoundError(
            f"{macro_path} not found. Run 'python run.py macro' first."
        )
    macro_df = pd.read_parquet(macro_path)
    macro_df.index = pd.to_datetime(macro_df.index)
    index = macro_df.index

    # ── 2. Per-ticker features ─────────────────────────────────────────────────
    features_dict: dict[str, pd.DataFrame] = {}
    for ticker in COMMODITY_TICKERS:
        fp = FEATURE_DIR / f"{ticker}.parquet"
        if fp.exists():
            try:
                df = pd.read_parquet(fp)
                df.index = pd.to_datetime(df.index)
                features_dict[ticker] = df
            except Exception as e:
                print(f"  [warn] Could not load features for {ticker}: {e}")

    # ── 3. Compute carry signals ───────────────────────────────────────────────
    print("  Computing bond carry (10Y-2Y yield curve z-score)...")
    bond_df = bond_carry(macro_df, index)

    print("  Computing commodity carry (21d vs 252d return differential)...")
    commodity_df = commodity_carry(features_dict, index)

    # ── 4. Combine and fill equity with 0 ─────────────────────────────────────
    carry_df = pd.concat([bond_df, commodity_df], axis=1)
    for ticker in TICKER_LIST:
        if ticker not in carry_df.columns:
            carry_df[ticker] = 0.0

    # Reorder to match TICKER_LIST
    carry_df = carry_df.reindex(columns=TICKER_LIST, fill_value=0.0)

    # ── 5. Save ────────────────────────────────────────────────────────────────
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SIGNAL_DIR / "carry_signals.parquet"
    carry_df.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"\n  Carry signals saved → {out_path}  {carry_df.shape}")

    # ── 6. Summary statistics ──────────────────────────────────────────────────
    print("\n  Carry signal summary (non-zero tickers):")
    print(f"  {'Ticker':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Non-zero%':>10}")
    print("  " + "-" * 54)
    for ticker in BOND_TICKERS + COMMODITY_TICKERS:
        if ticker not in carry_df.columns:
            continue
        s        = carry_df[ticker].dropna()
        nonzero  = (s.abs() > 0.01).mean() * 100
        print(f"  {ticker:<8} {s.mean():>+8.3f} {s.std():>8.3f} "
              f"{s.min():>8.3f} {s.max():>8.3f} {nonzero:>9.1f}%")

    # Yield curve context for the bond signal
    if "yield_curve" in macro_df.columns:
        yc_recent = macro_df["yield_curve"].iloc[-1]
        print(f"\n  Current yield curve (10Y-2Y): {yc_recent:+.2f}%  "
              f"({'inverted' if yc_recent < 0 else 'steep' if yc_recent > 1 else 'flat'})")
        bond_recent = carry_df[BOND_TICKERS[0]].iloc[-1] if BOND_TICKERS else 0
        print(f"  Current bond carry signal: {bond_recent:+.3f}")

    return carry_df


if __name__ == "__main__":
    compute_all_carry()
