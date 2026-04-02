"""
carry_signal.py
───────────────
Compute daily carry signals for all universe tickers.

Carry = expected return from HOLDING, not from price direction.
Structurally uncorrelated to momentum/trend signals.

Signal types:
  Bond carry     — rolling 60-day z-score of the 10Y-2Y yield curve spread.
                   Steep curve → positive carry (hold bonds earns roll-down).
                   Inverted curve → negative carry (hold bonds loses carry).
                   Applied equally to TLT, HYG, TIP, BWX — all share the
                   same rate environment.

  Commodity carry — short-term (21d) return minus monthly-equivalent of the
                    long-term (252d) return.  Approximates the futures curve
                    slope (backwardation = positive carry, contango = negative).
                    Tickers: GLD, DBC, UUP, DBA, FXE, FXY.

  Equity carry   — returns 0.0 for all equity tickers.  Dividend yield requires
                   fundamental data that is not reliably available from OHLCV
                   (yfinance ex-dates are retroactively adjusted, biasing any
                   returns-based proxy).  A noisy equity carry signal hurts more
                   than it helps, so we omit it.

Output: data/signals/carry_signals.parquet
  Columns = all TICKER_LIST tickers
  Index   = same DatetimeIndex as macro_features
  Values  = [-1, +1] continuous signal (0 for equity tickers)

shift(1) is applied so yesterday's carry is used to size today's position.
No look-ahead bias.

Consumed by:
  portfolio.py       — multi_mom_carry_sizes(), portable_carry
  signal_generation  — per-ticker signal_carry column in *.parquet files
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .data_pipeline import TICKER_LIST, ASSET_CLASS

FEATURE_DIR = Path("data/features")
MACRO_DIR   = Path("data/macro")
SIGNAL_DIR  = Path("data/signals")

BOND_TICKERS      = [t for t in TICKER_LIST if ASSET_CLASS.get(t) == "bond"]
COMMODITY_TICKERS = [t for t in TICKER_LIST if ASSET_CLASS.get(t) == "commodity"]


def bond_carry(macro_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Bond carry signal from the 10Y-2Y yield curve slope.

    A steep yield curve (spread > rolling average) means bonds earn positive
    carry from rolling down the curve.  An inverted curve means negative carry.

    Signal construction:
      1. Reindex yield_curve to the common date index
      2. Compute rolling 60-day z-score
      3. Clip at ±2 and divide by 2 → values in [-1, +1]
      4. shift(1) — use yesterday's curve to size today

    The same signal is applied to ALL bond tickers because the rate environment
    drives all bond carry uniformly (TLT, HYG, TIP, BWX all respond to the
    steepness of the curve, even if magnitudes differ).

    Args:
        macro_df: DataFrame from macro_features.parquet with a 'yield_curve' column.
        index:    DatetimeIndex to reindex the output to.

    Returns:
        DataFrame (len(index) × 4) with one column per bond ticker, values ∈ [-1, +1].
    """
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
    """
    Commodity carry signal from the 21-day vs 252-day return differential.

    For physical commodity ETFs, the futures-curve slope determines whether
    the roll is a cost (contango) or an income (backwardation).  We proxy this
    using price data:

      carry_proxy = 21d_return − (252d_return / 12)

    If the 21-day return exceeds the monthly-equivalent of the annual return,
    the near end of the curve is elevated → backwardation → positive carry.
    The reverse signals contango → negative carry.

    The proxy is z-scored over a 252-day window to normalise across assets with
    different return volatilities.

    Args:
        features_dict: Dict[ticker → feature DataFrame] with Close column.
        index:         DatetimeIndex to align the output to.

    Returns:
        DataFrame (len(index) × 6) with one column per commodity ticker.
    """
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
    """
    Orchestrate carry signal computation for the full universe and save output.

    Steps:
      1. Load macro_features.parquet (requires macro_features.py to have run)
      2. Load per-ticker feature parquets for Close prices
      3. Compute bond_carry() and commodity_carry()
      4. Fill all equity tickers with 0.0 (no carry signal)
      5. Save to data/signals/carry_signals.parquet
      6. Print summary statistics

    The output index aligns to macro_features — the longest common history.
    This matches the index used by portfolio.py for backtesting.
    """
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
