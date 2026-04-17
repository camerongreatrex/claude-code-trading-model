"""
regime_analysis.py
------------------
Decomposes equal-weight strategy performance by market regime.

The key question: does the strategy add value in ALL four regimes, or only
in bull_calm?  A strategy that outperforms only when VIX < 20 and markets
are rising is not an alpha strategy — it is leveraged beta with extra steps.
A real edge should show up in bear and stress regimes, or at minimum should
protect capital when the simple buy-and-hold bleeds.

Regime labels (applied to each trading day)
───────────────────────────────────────────
  bull_calm   : VIX < 20  AND  SPY 60d return > 0   (rising market, low fear)
  bull_stress : VIX >= 20 AND  SPY 60d return > 0   (rising market, elevated fear)
  bear_calm   : VIX < 20  AND  SPY 60d return <= 0  (falling/flat market, low fear)
  bear_stress : VIX >= 20 AND  SPY 60d return <= 0  (falling/flat, high fear)

Metrics per regime
──────────────────
  strategy_sharpe   — annualised Sharpe of equal-weight strategy returns
  strategy_ann_ret  — annualised return of equal-weight strategy
  bnh_sharpe        — buy-and-hold (equal-weight, fully invested) Sharpe
  bnh_ann_ret       — buy-and-hold annualised return
  alpha_sharpe      — strategy_sharpe - bnh_sharpe (how much Sharpe is added)
  n_days            — number of trading days in this regime
  pct_days          — % of total sample in this regime
  avg_turnover      — mean fraction of portfolio positions changing per day
                      (signal flips / N tickers; 0 = no rebalance, 1 = full turnover)

Data sources
────────────
  Reads: data/macro/macro_features.parquet   (VIX)
         data/features/SPY.parquet            (Close for 60d return)
         data/signals/regime_signals.parquet  (signal matrix, T x N)
         data/features/{TICKER}.parquet       (log_return columns)

  Writes: data/research/regime_decomposition.parquet

Usage
─────
  python -m pipeline.regime_analysis
  python pipeline/regime_analysis.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from pipeline.data_pipeline import TICKER_LIST

# ── Paths ──────────────────────────────────────────────────────────────────────
MACRO_DIR    = Path("data/macro")
SIGNAL_DIR   = Path("data/signals")
FEATURE_DIR  = Path("data/features")
RESEARCH_DIR = Path("data/research")
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

# ── Regime thresholds ──────────────────────────────────────────────────────────
VIX_STRESS_THRESH  = 20    # VIX >= 20 = stressed environment
SPY_LOOKBACK       = 60    # trading days for SPY trend classification

# Ordered for display
REGIME_ORDER = ["bull_calm", "bull_stress", "bear_calm", "bear_stress"]

REGIME_LABELS = {
    "bull_calm"  : "Bull / Calm   (VIX<20, SPY 60d>0)",
    "bull_stress": "Bull / Stress (VIX>=20, SPY 60d>0)",
    "bear_calm"  : "Bear / Calm   (VIX<20,  SPY 60d<=0)",
    "bear_stress": "Bear / Stress (VIX>=20, SPY 60d<=0)",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_macro() -> pd.DataFrame:
    path = MACRO_DIR / "macro_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Macro features not found at {path}. Run macro_features.py first.")
    return pd.read_parquet(path)


def _load_signals() -> pd.DataFrame:
    path = SIGNAL_DIR / "regime_signals.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Regime signals not found at {path}. Run signal_generation.py first.")
    return pd.read_parquet(path)


def _load_returns(tickers: list) -> pd.DataFrame:
    """Load log_return columns for all tickers, return aligned DataFrame."""
    series = {}
    for t in tickers:
        path = FEATURE_DIR / f"{t}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["log_return"])
            series[t] = df["log_return"]
    if not series:
        raise FileNotFoundError("No feature parquets found. Run feature_engineering.py first.")
    return pd.DataFrame(series)


def _load_spy_close() -> pd.Series:
    path = FEATURE_DIR / "SPY.parquet"
    if not path.exists():
        raise FileNotFoundError("SPY feature parquet not found.")
    return pd.read_parquet(path, columns=["Close"])["Close"]


# ── Regime labelling ───────────────────────────────────────────────────────────

def label_regimes(
    vix: pd.Series,
    spy_close: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Label each date in index as one of four market regimes.

    VIX is aligned from macro features.  SPY 60d return is computed from
    adjusted close prices — no look-ahead (pct_change(60) uses past 60 days).

    Returns:
        Series of strings ('bull_calm', 'bull_stress', 'bear_calm',
        'bear_stress') aligned to index.  NaN on warm-up dates before
        SPY has 60 days of history.
    """
    # SPY 60-day return: price[t] / price[t-60] - 1 (purely historical)
    spy_60d = spy_close.pct_change(SPY_LOOKBACK)

    # Align VIX and SPY return to the working index
    vix_aligned    = vix.reindex(index).ffill()
    spy_60d_aligned = spy_60d.reindex(index).ffill()

    stressed   = vix_aligned >= VIX_STRESS_THRESH   # True = VIX >= 20
    bull_trend = spy_60d_aligned > 0                 # True = SPY up over 60d

    conditions = [
        (~stressed) &  bull_trend,   # bull_calm
         stressed   &  bull_trend,   # bull_stress
        (~stressed) & ~bull_trend,   # bear_calm
         stressed   & ~bull_trend,   # bear_stress
    ]
    # np.select requires a consistent dtype across choices and default.
    # Use a string sentinel; days that don't match any condition (warm-up
    # before SPY has 60 days of history) remain "unlabelled" and are
    # excluded from regime analysis by value-match in decompose().
    labels = pd.Series(
        np.select(conditions, REGIME_ORDER, default="unlabelled"),
        index=index,
        name="regime",
    )
    return labels


# ── Performance metrics ────────────────────────────────────────────────────────

def _sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio. Returns 0.0 if std is zero or series is empty."""
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def _ann_return(returns: pd.Series) -> float:
    """Annualised return from daily log returns."""
    if len(returns) == 0:
        return 0.0
    return float(returns.mean() * 252)


def _turnover(signals: pd.DataFrame, mask: pd.Series) -> float:
    """
    Mean fraction of positions changing per day within the regime.

    For each day t in the regime, count how many tickers have
    signal[t] != signal[t-1].  Divide by N tickers.
    Average across regime days.

    Returns 0 if no days or no signal columns available.
    """
    if signals.empty or mask.sum() == 0:
        return 0.0
    # Signal changes: 1 where signal flipped vs previous day, 0 otherwise
    flipped    = (signals.diff().abs() > 0).astype(float)
    regime_days = flipped.loc[mask]
    if regime_days.empty:
        return 0.0
    # Mean fraction of tickers turning over per day
    return float(regime_days.mean(axis=1).mean())


# ── Core decomposition ─────────────────────────────────────────────────────────

def decompose(
    strat_ret  : pd.Series,
    bnh_ret    : pd.Series,
    signals    : pd.DataFrame,
    regimes    : pd.Series,
) -> pd.DataFrame:
    """
    Compute per-regime performance metrics.

    Args:
        strat_ret : daily equal-weight strategy log returns
        bnh_ret   : daily equal-weight buy-and-hold log returns
        signals   : (T x N) integer signal DataFrame (for turnover)
        regimes   : Series of regime labels aligned to strat_ret.index

    Returns:
        DataFrame with one row per regime and metric columns.
    """
    total_days = len(strat_ret)
    rows = []

    for regime in REGIME_ORDER:
        mask_full = (regimes == regime)
        mask      = mask_full.reindex(strat_ret.index).fillna(False)

        sr = strat_ret[mask]
        br = bnh_ret.reindex(strat_ret.index)[mask]

        n    = int(mask.sum())
        pct  = n / total_days * 100 if total_days > 0 else 0.0

        # Turnover: align signals to strat_ret index before masking
        sig_aligned = signals.reindex(strat_ret.index)
        turn = _turnover(sig_aligned, mask)

        s_sharpe   = _sharpe(sr)
        s_ann_ret  = _ann_return(sr)
        b_sharpe   = _sharpe(br)
        b_ann_ret  = _ann_return(br)
        alpha      = s_sharpe - b_sharpe

        rows.append({
            "regime"          : regime,
            "n_days"          : n,
            "pct_days"        : round(pct, 1),
            "strategy_sharpe" : round(s_sharpe, 3),
            "strategy_ann_ret": round(s_ann_ret * 100, 2),
            "bnh_sharpe"      : round(b_sharpe, 3),
            "bnh_ann_ret"     : round(b_ann_ret * 100, 2),
            "alpha_sharpe"    : round(alpha, 3),
            "avg_turnover"    : round(turn, 4),
        })

    return pd.DataFrame(rows)


# ── Print table ────────────────────────────────────────────────────────────────

def _print_table(df: pd.DataFrame) -> None:
    W = 90
    print("\n" + "=" * W)
    print("  REGIME DECOMPOSITION  --  Equal-Weight Strategy vs Buy-and-Hold")
    print("=" * W)
    print(
        f"  {'Regime':<32} {'Days':>5} {'%Tot':>5}  "
        f"{'Strat Sharpe':>12} {'Strat Ret%':>10}  "
        f"{'B&H Sharpe':>10} {'B&H Ret%':>8}  "
        f"{'Alpha Sh':>8} {'Turnover':>8}"
    )
    print("  " + "-" * (W - 2))

    for _, row in df.iterrows():
        regime  = row["regime"]
        label   = REGIME_LABELS.get(regime, regime)

        # Flag regimes where strategy meaningfully underperforms or B&H
        alpha_flag = ""
        if row["alpha_sharpe"] < -0.1:
            alpha_flag = " <-- underperforms B&H"
        elif row["strategy_sharpe"] < 0.0:
            alpha_flag = " <-- negative Sharpe"

        print(
            f"  {label:<32} {row['n_days']:>5} {row['pct_days']:>4.1f}%  "
            f"{row['strategy_sharpe']:>12.3f} {row['strategy_ann_ret']:>9.1f}%  "
            f"{row['bnh_sharpe']:>10.3f} {row['bnh_ann_ret']:>7.1f}%  "
            f"{row['alpha_sharpe']:>+8.3f} {row['avg_turnover']:>7.1%}"
            f"{alpha_flag}"
        )

    print("  " + "-" * (W - 2))

    # Overall row
    all_strat = df["strategy_sharpe"].mean()
    all_bnh   = df["bnh_sharpe"].mean()
    all_alpha = df["alpha_sharpe"].mean()
    print(
        f"  {'ALL REGIMES (simple mean)':<32} {df['n_days'].sum():>5}  100%  "
        f"{all_strat:>12.3f} {'':>10}  "
        f"{all_bnh:>10.3f} {'':>8}  "
        f"{all_alpha:>+8.3f}"
    )
    print("=" * W)

    # Interpretation
    print("\n  Interpretation:")
    n_positive_alpha = (df["alpha_sharpe"] > 0).sum()
    n_positive_strat = (df["strategy_sharpe"] > 0).sum()
    n_regimes        = len(df)

    print(f"    Strategy adds Sharpe alpha in {n_positive_alpha}/{n_regimes} regimes.")
    print(f"    Strategy has positive Sharpe in {n_positive_strat}/{n_regimes} regimes.")

    bear_stress = df[df["regime"] == "bear_stress"]
    if len(bear_stress):
        bs = bear_stress.iloc[0]
        bnh_dd_proxy = bs["bnh_ann_ret"]
        strat_proxy  = bs["strategy_ann_ret"]
        if strat_proxy > bnh_dd_proxy:
            print(f"    Bear/Stress: strategy ({strat_proxy:+.1f}% ann) outperforms B&H ({bnh_dd_proxy:+.1f}% ann) -- good crash protection.")
        else:
            diff = bnh_dd_proxy - strat_proxy
            print(f"    Bear/Stress: strategy ({strat_proxy:+.1f}% ann) lags B&H ({bnh_dd_proxy:+.1f}% ann) by {diff:.1f}% ann.")

    bull_calm = df[df["regime"] == "bull_calm"]
    if len(bull_calm):
        bc = bull_calm.iloc[0]
        pct = bc["pct_days"]
        sh  = bc["strategy_sharpe"]
        print(
            f"    Bull/Calm is {pct:.0f}% of all days (Sharpe {sh:.3f}). "
            + ("Strategy is NOT purely a bull-calm play." if n_positive_strat >= 3
               else "Strategy relies heavily on bull-calm conditions.")
        )
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  Regime Performance Decomposition")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    macro     = _load_macro()
    signals   = _load_signals()
    spy_close = _load_spy_close()

    tickers = [t for t in TICKER_LIST if t in signals.columns]
    returns = _load_returns(tickers)

    # Align everything to the signals index (shortest common window)
    idx = signals.index
    returns  = returns.reindex(idx)
    spy_close = spy_close.reindex(idx).ffill()

    print(f"  Signals matrix   : {signals.shape}")
    print(f"  Returns matrix   : {returns.shape}")
    print(f"  SPY close series : {spy_close.shape[0]} rows")
    print(f"  Macro features   : {macro.shape}")

    # ── Compute strategy and B&H returns ──────────────────────────────────────
    # Strategy: position on day t-1 earns log_return on day t
    # Equal weight across all tickers in the signal matrix
    strat_matrix = returns[tickers].multiply(signals[tickers].shift(1).fillna(0))
    strat_ret    = strat_matrix.mean(axis=1).dropna()

    # Buy-and-hold: equal-weight, fully invested every day
    bnh_ret = returns[tickers].mean(axis=1)

    # ── Label regimes ──────────────────────────────────────────────────────────
    regimes = label_regimes(macro["vix"], spy_close, idx)

    # Print regime distribution
    regime_counts = regimes.value_counts()
    total = len(regimes.dropna())
    print(f"\nRegime distribution ({total} labelled days):")
    for r in REGIME_ORDER:
        n   = regime_counts.get(r, 0)
        pct = n / total * 100 if total > 0 else 0
        print(f"  {REGIME_LABELS[r]:<42}  {n:>4} days  ({pct:.1f}%)")

    # ── Decompose ──────────────────────────────────────────────────────────────
    print("\nComputing regime metrics...")
    decomp = decompose(strat_ret, bnh_ret, signals, regimes)

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = RESEARCH_DIR / "regime_decomposition.parquet"
    decomp.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Results saved -> {out_path}")

    # ── Print ──────────────────────────────────────────────────────────────────
    _print_table(decomp)


if __name__ == "__main__":
    main()
