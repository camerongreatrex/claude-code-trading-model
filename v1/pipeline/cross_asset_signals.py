"""
cross_asset_signals.py
──────────────────────
Compute daily cross-asset lead-lag features that exploit documented
information transmission between asset classes.

These are NOT price momentum — they are relative-value and spread signals
designed to capture structural lead-lag relationships:

  1. credit_equity_spread   — HYG–SPY rolling correlation, lagged 1-3 days.
                              When HYG leads SPY lower, predicts equity weakness.
  2. tlt_spy_divergence     — TLT 5d return minus SPY 5d return.
                              Positive = institutional flight-to-quality.
  3. vix_term_structure_signal — (VIX9D − VIX) 60-day z-score.
                              Spikes predict elevated short-term vol for 5-10 days.
  4. commodity_dollar_signal — DBC 5d return minus UUP 5d return.
                              Positive = reflation. Negative = deflation/$ strength.
  5. bond_equity_rotation   — 20-day rolling beta of TLT to SPY.
                              β < −0.3 = normal negative correlation.
                              β > 0 = correlation breakdown (crisis or reflation).
  6. hy_ig_spread           — HYG/LQD price ratio, 252-day z-score.
                              Widening (falling ratio) precedes equity sell-offs.

No look-ahead: all features use shift(1) before output.

Data sources
────────────
  data/features/{SPY,HYG,TLT,DBC,UUP}.parquet  — log_return, Close
  data/shared/macro/macro_features.parquet              — vix, vix9d
  yfinance (LQD only — not in the main universe)

Output
──────
  data/signals/cross_asset_features.parquet
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

FEATURE_DIR = Path("data/v1/features")
MACRO_DIR   = Path("data/shared/macro")
SIGNAL_DIR  = Path("data/v1/signals")
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

# Date range matching the main pipeline
START = "2015-01-01"
END   = "2026-01-01"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_feature(ticker: str) -> pd.DataFrame:
    """Load a per-ticker feature parquet."""
    path = FEATURE_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature file missing: {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def _load_returns(tickers: list) -> pd.DataFrame:
    """Load log returns for a list of tickers, aligned on common dates."""
    frames = {}
    for t in tickers:
        df = _load_feature(t)
        if "log_return" in df.columns:
            frames[t] = df["log_return"]
    return pd.DataFrame(frames).dropna()


def _load_closes(tickers: list) -> pd.DataFrame:
    """Load close prices for a list of tickers, aligned on common dates."""
    frames = {}
    for t in tickers:
        df = _load_feature(t)
        if "Close" in df.columns:
            frames[t] = df["Close"]
    return pd.DataFrame(frames).dropna()


def _fetch_lqd() -> pd.Series:
    """
    Download LQD (iShares Investment Grade Corporate Bond ETF) close prices
    from yfinance.  LQD is not in the main universe so we fetch it here.
    """
    print("  Fetching LQD from yfinance...")
    df = yf.download("LQD", start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "LQD"
    return s


def _load_macro() -> pd.DataFrame:
    """Load macro features (VIX, VIX9D)."""
    path = MACRO_DIR / "macro_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Macro features missing: {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


# ── Feature computation ──────────────────────────────────────────────────────

def credit_equity_spread(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 5-day correlation between HYG and SPY returns, lagged 1-3 days.

    When HYG leads SPY lower (correlation breaks down or becomes very
    negative with a lag), this predicts equity weakness 2-3 days ahead.
    Credit markets often price in stress before equity markets react.

    Returns three columns:
      credit_equity_corr_lag1  — 5-day rolling corr shifted 1 day
      credit_equity_corr_lag2  — shifted 2 days
      credit_equity_corr_lag3  — shifted 3 days
    """
    hyg = returns["HYG"]
    spy = returns["SPY"]
    corr = hyg.rolling(5, min_periods=4).corr(spy)
    return pd.DataFrame({
        "credit_equity_corr_lag1": corr.shift(1),
        "credit_equity_corr_lag2": corr.shift(2),
        "credit_equity_corr_lag3": corr.shift(3),
    })


def tlt_spy_divergence(returns: pd.DataFrame) -> pd.DataFrame:
    """
    TLT 5-day return minus SPY 5-day return (flight-to-quality signal).

    When institutional money rotates from equities into long-duration
    Treasuries, TLT outperforms SPY over a trailing window.  A positive
    divergence indicates risk-off positioning that often precedes
    further equity weakness.

    Returns:
      tlt_spy_divergence — TLT 5d return minus SPY 5d return, shifted 1 day
    """
    tlt_5d = returns["TLT"].rolling(5, min_periods=4).sum()
    spy_5d = returns["SPY"].rolling(5, min_periods=4).sum()
    div = tlt_5d - spy_5d
    return pd.DataFrame({"tlt_spy_divergence": div.shift(1)})


def vix_term_structure_signal(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score of (VIX9D − VIX) over a rolling 60-day window.

    VIX9D > VIX (backwardation) means near-term implied vol exceeds
    medium-term — a sign of acute stress.  The z-score normalises this
    spread so that extreme readings (|z| > 2) flag regime changes.

    Extends the binary vol_backwardation flag already in macro_features.py
    with a continuous, mean-reverting signal.

    Returns:
      vix_term_zscore — 60-day z-score of (VIX9D − VIX), shifted 1 day
    """
    spread = macro["vix9d"] - macro["vix"]
    roll = spread.rolling(60, min_periods=20)
    zscore = (spread - roll.mean()) / roll.std()
    return pd.DataFrame({"vix_term_zscore": zscore.shift(1)})


def commodity_dollar_signal(returns: pd.DataFrame) -> pd.DataFrame:
    """
    DBC 5-day return minus UUP 5-day return (commodity vs dollar carry).

    Positive = reflation trade (commodities rising, dollar weakening).
    Negative = deflation / dollar strength.

    This spread captures the macro reflationary vs deflationary impulse
    without being exposed to any single commodity or currency.

    Returns:
      commodity_dollar_signal — DBC 5d return minus UUP 5d return, shifted 1 day
    """
    dbc_5d = returns["DBC"].rolling(5, min_periods=4).sum()
    uup_5d = returns["UUP"].rolling(5, min_periods=4).sum()
    sig = dbc_5d - uup_5d
    return pd.DataFrame({"commodity_dollar_signal": sig.shift(1)})


def bond_equity_rotation(returns: pd.DataFrame) -> pd.DataFrame:
    """
    20-day rolling OLS beta of TLT returns to SPY returns.

    Beta < −0.3 = normal structural negative correlation (bonds rally
                  when equities sell off — the traditional hedge).
    Beta ≈  0   = correlation breakdown, diversification failing.
    Beta >  0   = positive correlation (crisis or strong reflation).

    Regime shifts in this beta are among the most powerful predictors of
    portfolio-level drawdowns because they indicate that the bond hedge
    is no longer working.

    Returns:
      bond_equity_beta — 20-day rolling beta, shifted 1 day
    """
    tlt = returns["TLT"]
    spy = returns["SPY"]
    cov = tlt.rolling(20, min_periods=15).cov(spy)
    var = spy.rolling(20, min_periods=15).var()
    beta = cov / var
    return pd.DataFrame({"bond_equity_beta": beta.shift(1)})


def hy_ig_spread(closes: pd.DataFrame, lqd_close: pd.Series) -> pd.DataFrame:
    """
    HYG / LQD price ratio normalised to a 252-day z-score.

    A falling ratio means high-yield bonds are underperforming
    investment-grade bonds — credit markets are pricing in higher
    default risk.  This typically precedes equity drawdowns by days
    to weeks.

    The z-score normalises the ratio so that extreme readings
    (z < −1.5) flag periods of acute credit stress.

    Requires LQD (iShares Investment Grade Corporate Bond ETF)
    to be fetched separately since it's not in the main universe.

    Returns:
      hy_ig_ratio_zscore — 252-day z-score of HYG/LQD ratio, shifted 1 day
    """
    hyg = closes["HYG"]
    # Align LQD to the same index
    lqd = lqd_close.reindex(hyg.index).ffill()
    ratio = hyg / lqd
    roll = ratio.rolling(252, min_periods=60)
    zscore = (ratio - roll.mean()) / roll.std()
    return pd.DataFrame({"hy_ig_ratio_zscore": zscore.shift(1)})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Computing cross-asset lead-lag features...\n")

    # ── Load data ─────────────────────────────────────────────────────────
    needed_tickers = ["SPY", "HYG", "TLT", "DBC", "UUP"]
    print("  Loading returns...")
    returns = _load_returns(needed_tickers)
    print(f"    {returns.shape[0]:,} trading days, {list(returns.columns)}")

    print("  Loading closes...")
    closes = _load_closes(needed_tickers)

    lqd_close = _fetch_lqd()
    print(f"    LQD: {len(lqd_close):,} obs, {lqd_close.index.min().date()} → "
          f"{lqd_close.index.max().date()}")

    macro = _load_macro()
    print(f"    Macro: {macro.shape[0]:,} days")

    # ── Compute features ──────────────────────────────────────────────────
    print("\n  Computing features...")

    feat1 = credit_equity_spread(returns)
    print(f"    credit_equity_spread:     {feat1.columns.tolist()}")

    feat2 = tlt_spy_divergence(returns)
    print(f"    tlt_spy_divergence:       {feat2.columns.tolist()}")

    feat3 = vix_term_structure_signal(macro)
    print(f"    vix_term_structure_signal: {feat3.columns.tolist()}")

    feat4 = commodity_dollar_signal(returns)
    print(f"    commodity_dollar_signal:  {feat4.columns.tolist()}")

    feat5 = bond_equity_rotation(returns)
    print(f"    bond_equity_rotation:     {feat5.columns.tolist()}")

    feat6 = hy_ig_spread(closes, lqd_close)
    print(f"    hy_ig_spread:             {feat6.columns.tolist()}")

    # ── Merge all features on common index ────────────────────────────────
    all_features = pd.concat(
        [feat1, feat2, feat3, feat4, feat5, feat6],
        axis=1,
    )
    # Align to trading days present in returns (the tightest common index)
    all_features = all_features.reindex(returns.index)
    all_features = all_features.dropna(how="all")

    # ── Save ──────────────────────────────────────────────────────────────
    out = SIGNAL_DIR / "cross_asset_features.parquet"
    all_features.to_parquet(out, engine="pyarrow", compression="snappy")

    print(f"\n  Output: {all_features.shape}  →  {out}")
    print(f"  Date range: {all_features.index.min().date()} → "
          f"{all_features.index.max().date()}")
    print(f"  Columns: {list(all_features.columns)}")
    print(f"\n  Last 3 rows:")
    print(all_features.tail(3).round(4).to_string())

    # ── Summary statistics ────────────────────────────────────────────────
    print(f"\n  Non-null counts:")
    for col in all_features.columns:
        nn = all_features[col].notna().sum()
        print(f"    {col:30s}  {nn:>5,} / {len(all_features):,}")


if __name__ == "__main__":
    main()
