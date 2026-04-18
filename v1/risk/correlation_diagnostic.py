"""
correlation_diagnostic.py
--------------------------
Read-only diagnostic: regime decomposition baseline + beta decomposition +
regime correlation + dead weight ticker analysis.

This module does NOT modify any pipeline data.  It reads existing outputs and
prints diagnostics to help understand where edge comes from (true alpha vs
beta-driven returns) and which tickers are dragging performance.

Usage
─────
  python -m v1.pipeline.correlation_diagnostic
  python run.py diagnostic
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from v1.regimes.analysis import (
    label_regimes,
    decompose,
    _print_table,
    _sharpe,
    _ann_return,
    REGIME_ORDER,
    REGIME_LABELS,
    _load_macro,
    _load_signals,
    _load_spy_close,
)
from v1.pipeline.data_pipeline import TICKER_LIST

# ── Paths ───────────────────────────────────────────────────────────────────────
FEATURE_DIR  = Path("data/v1/features")
RESULTS_DIR  = Path("data/v1/results")
RESEARCH_DIR = Path("data/v1/research")
SIGNAL_DIR   = Path("data/v1/signals")
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ────────────────────────────────────────────────────────────────

def _load_portfolio() -> pd.DataFrame:
    path = RESULTS_DIR / "portfolio_comparison.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Portfolio comparison not found at {path}. Run portfolio.py first.")
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df


def _load_returns_matrix() -> pd.DataFrame:
    path = FEATURE_DIR / "returns_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Returns matrix not found at {path}. Run feature_engineering.py first.")
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df


# ── Beta Decomposition ──────────────────────────────────────────────────────────

def _rolling_beta_corr(
    port_ret: pd.Series,
    spy_ret: pd.Series,
    window: int = 60,
) -> tuple:
    """Return (rolling_beta, rolling_corr) Series aligned to port_ret.index."""
    aligned   = pd.concat([port_ret, spy_ret], axis=1).dropna()
    p         = aligned.iloc[:, 0]
    s         = aligned.iloc[:, 1]
    roll_cov  = p.rolling(window).cov(s)
    roll_var  = s.rolling(window).var()
    roll_beta = roll_cov / roll_var
    roll_corr = p.rolling(window).corr(s)
    return roll_beta, roll_corr


def beta_decomposition(
    portfolio_df: pd.DataFrame,
    returns_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each sizing method in portfolio_comparison compute:
      total_sharpe, OLS beta to SPY, avg rolling 60-day beta,
      avg rolling 60-day correlation, active return Sharpe.

    Portfolio equity curves are converted to daily pct returns.
    SPY daily log returns come from returns_matrix (small-return approx holds).
    """
    # Daily pct returns from equity curves
    port_returns = portfolio_df.pct_change().dropna()

    spy_ret = returns_matrix["SPY"].reindex(port_returns.index)

    methods = [c for c in port_returns.columns if c != "buy_hold"]

    rows = []
    for method in methods:
        pret = port_returns[method].dropna()
        spy_aligned = spy_ret.reindex(pret.index)

        common = pd.concat([pret, spy_aligned], axis=1).dropna()
        if len(common) < 60:
            continue
        p = common.iloc[:, 0]
        s = common.iloc[:, 1]

        # OLS beta (full period)
        ols_beta = float(p.cov(s) / s.var())

        # Rolling 60-day beta and correlation
        roll_beta, roll_corr = _rolling_beta_corr(pret, spy_aligned, window=60)
        avg_roll_beta = float(roll_beta.mean())
        avg_roll_corr = float(roll_corr.mean())

        # Active return: strip market beta
        active_ret    = p - ols_beta * s
        total_sharpe  = _sharpe(p)
        active_sharpe = _sharpe(active_ret)

        rows.append({
            "method"        : method,
            "total_sharpe"  : round(total_sharpe, 3),
            "beta"          : round(ols_beta, 3),
            "avg_roll_beta" : round(avg_roll_beta, 3),
            "avg_corr"      : round(avg_roll_corr, 3),
            "active_sharpe" : round(active_sharpe, 3),
        })

    return pd.DataFrame(rows)


def _print_beta_table(df: pd.DataFrame) -> None:
    W = 90
    print("\n" + "=" * W)
    print("  BETA DECOMPOSITION  --  True Alpha Sharpe After Stripping SPY Beta")
    print("=" * W)
    print(
        f"  {'Method':<26} {'Total Sharpe':>12} {'OLS Beta':>8} "
        f"{'Roll Beta':>9} {'Roll Corr':>9} {'Active Sharpe':>13}"
    )
    print("  " + "-" * (W - 2))

    for _, row in df.iterrows():
        flag = ""
        if row["active_sharpe"] < 0.1:
            flag = "  <-- minimal true alpha"
        elif row["active_sharpe"] > 0.5:
            flag = "  <-- strong alpha"

        print(
            f"  {row['method']:<26} {row['total_sharpe']:>12.3f} "
            f"{row['beta']:>8.3f} {row['avg_roll_beta']:>9.3f} "
            f"{row['avg_corr']:>9.3f} {row['active_sharpe']:>13.3f}{flag}"
        )

    print("=" * W)

    best = df.loc[df["active_sharpe"].idxmax()]
    worst = df.loc[df["active_sharpe"].idxmin()]
    print(f"\n  Highest active Sharpe: {best['method']}  ({best['active_sharpe']:.3f})")
    print(f"  Lowest  active Sharpe: {worst['method']}  ({worst['active_sharpe']:.3f})")
    avg_beta = df["beta"].mean()
    print(f"  Average beta across methods: {avg_beta:.3f}")
    if avg_beta > 0.7:
        print("  --> Portfolio is largely a beta play; active Sharpe reveals the true edge.")
    print()


# ── Regime Correlation ──────────────────────────────────────────────────────────

def regime_correlation(
    returns_matrix: pd.DataFrame,
    regimes: pd.Series,
    portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each of the 4 regimes compute:
      - Average pairwise correlation of all universe tickers
      - Equal-weight portfolio beta to SPY within the regime
      - Annualised active return within the regime (after beta-strip)
    """
    port_returns = portfolio_df.pct_change().dropna()
    port_col = "equal_weight" if "equal_weight" in port_returns.columns else port_returns.columns[0]
    port_ret = port_returns[port_col]

    spy_ret = returns_matrix["SPY"]

    rows = []
    for regime in REGIME_ORDER:
        mask = regimes == regime

        # ── Avg pairwise correlation of universe tickers ──────────────────────
        ret_mask    = mask.reindex(returns_matrix.index).fillna(False).infer_objects(copy=False).astype(bool)
        regime_rets = returns_matrix.loc[ret_mask]
        n_regime    = int(ret_mask.sum())

        if n_regime < 10:
            rows.append({
                "regime": regime, "n_days": n_regime,
                "avg_corr": np.nan, "portfolio_beta": np.nan,
                "active_return_ann_pct": np.nan,
            })
            continue

        corr_mat = regime_rets.corr()
        upper    = np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
        avg_corr = float(corr_mat.where(upper).stack().mean())

        # ── Portfolio beta and active return within regime ────────────────────
        port_mask = mask.reindex(port_ret.index).fillna(False).infer_objects(copy=False).astype(bool)
        p_regime  = port_ret[port_mask]
        s_regime  = spy_ret.reindex(p_regime.index)

        common = pd.concat([p_regime, s_regime], axis=1).dropna()
        if len(common) < 5:
            beta       = np.nan
            active_ann = np.nan
        else:
            p = common.iloc[:, 0]
            s = common.iloc[:, 1]
            beta       = float(p.cov(s) / s.var())
            active_ret = p - beta * s
            active_ann = float(_ann_return(active_ret) * 100)

        rows.append({
            "regime"               : regime,
            "n_days"               : n_regime,
            "avg_corr"             : round(avg_corr, 4),
            "portfolio_beta"       : round(beta, 3) if not np.isnan(beta) else np.nan,
            "active_return_ann_pct": round(active_ann, 2) if not np.isnan(active_ann) else np.nan,
        })

    return pd.DataFrame(rows)


def _print_regime_corr_table(df: pd.DataFrame) -> None:
    W = 82
    print("\n" + "=" * W)
    print("  REGIME CORRELATION  --  Diversification and Beta by Market Regime")
    print("=" * W)
    print(
        f"  {'Regime':<36} {'N Days':>6} {'Avg Corr':>8} "
        f"{'Port Beta':>9} {'Active Ret%':>11}"
    )
    print("  " + "-" * (W - 2))

    for _, row in df.iterrows():
        label    = REGIME_LABELS.get(row["regime"], row["regime"])
        corr_str = f"{row['avg_corr']:.4f}" if pd.notna(row["avg_corr"]) else "   N/A"
        beta_str = f"{row['portfolio_beta']:.3f}" if pd.notna(row["portfolio_beta"]) else "   N/A"
        act_str  = f"{row['active_return_ann_pct']:+.2f}%" if pd.notna(row["active_return_ann_pct"]) else "   N/A"

        # Flag regimes where correlations spike (diversification collapses)
        flag = ""
        if pd.notna(row["avg_corr"]) and row["avg_corr"] > 0.6:
            flag = "  <-- corr spike: diversification collapses"

        print(
            f"  {label:<36} {row['n_days']:>6} {corr_str:>8} "
            f"{beta_str:>9} {act_str:>11}{flag}"
        )

    print("=" * W)

    valid = df[df["avg_corr"].notna()]
    if len(valid):
        high_regime = valid.loc[valid["avg_corr"].idxmax(), "regime"]
        low_regime  = valid.loc[valid["avg_corr"].idxmin(), "regime"]
        print(f"\n  Correlation is highest in {high_regime} — beta hedging most valuable here.")
        print(f"  Correlation is lowest  in {low_regime}  — genuine diversification available.")
    print()


# ── Dead Weight Analysis ────────────────────────────────────────────────────────

def dead_weight_analysis(
    signals: pd.DataFrame,
    returns_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each ticker: fraction of days where the regime signal == 1 (long)
    AND the ticker's daily return was negative (holding a loser).

    Ranked descending by dead_weight_pct.
    A perfect signal would yield 0%.  A random signal yields ~50%.
    Tickers above 50% are net drags on signal quality.
    """
    tickers = [t for t in signals.columns if t in returns_matrix.columns]

    rows = []
    for ticker in tickers:
        sig  = signals[ticker].reindex(returns_matrix.index)
        ret  = returns_matrix[ticker]

        long_mask  = sig == 1
        total_long = int(long_mask.sum())

        if total_long == 0:
            rows.append({
                "ticker"              : ticker,
                "long_days"           : 0,
                "dead_weight_days"    : 0,
                "dead_weight_pct"     : np.nan,
                "avg_loss_on_dead_days": np.nan,
            })
            continue

        dead      = long_mask & (ret < 0)
        dead_days = int(dead.sum())
        dead_pct  = dead_days / total_long * 100
        avg_loss  = float(ret[dead].mean() * 100) if dead_days > 0 else 0.0

        rows.append({
            "ticker"              : ticker,
            "long_days"           : total_long,
            "dead_weight_days"    : dead_days,
            "dead_weight_pct"     : round(dead_pct, 2),
            "avg_loss_on_dead_days": round(avg_loss, 4),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("dead_weight_pct", ascending=False)
        .reset_index(drop=True)
    )
    return df


def _print_dead_weight_table(df: pd.DataFrame) -> None:
    W = 76
    print("\n" + "=" * W)
    print("  DEAD WEIGHT ANALYSIS  --  Long Signal with Negative Return Days")
    print("  (random signal => ~50% dead weight; below 50% = edge)")
    print("=" * W)
    print(
        f"  {'Ticker':<8} {'Long Days':>9} {'Dead Days':>9} "
        f"{'Dead Wt%':>8} {'Avg Loss%':>9}"
    )
    print("  " + "-" * (W - 2))

    for _, row in df.iterrows():
        if pd.isna(row["dead_weight_pct"]):
            print(f"  {row['ticker']:<8} {'no long signals':>38}")
            continue

        flag = ""
        if row["dead_weight_pct"] > 52:
            flag = "  <-- above random"
        elif row["dead_weight_pct"] < 44:
            flag = "  <-- strong edge"

        avg_loss_str = f"{row['avg_loss_on_dead_days']:>8.4f}%" if row["dead_weight_days"] > 0 else "       N/A"

        print(
            f"  {row['ticker']:<8} {row['long_days']:>9} {row['dead_weight_days']:>9} "
            f"{row['dead_weight_pct']:>7.1f}%{avg_loss_str}{flag}"
        )

    print("=" * W)

    valid = df[df["dead_weight_pct"].notna()]
    if len(valid):
        avg_dw  = valid["dead_weight_pct"].mean()
        n_above = (valid["dead_weight_pct"] > 50).sum()
        print(f"\n  Average dead weight: {avg_dw:.1f}%  ({n_above}/{len(valid)} tickers above random 50%)")
    print()


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  Correlation Diagnostic  --  Regime Baseline Analysis")
    print("  (read-only — no pipeline files modified)")
    print("=" * 70)

    # ── Load shared data ───────────────────────────────────────────────────────
    print("\nLoading data...")
    macro          = _load_macro()
    signals        = _load_signals()
    spy_close      = _load_spy_close()
    portfolio_df   = _load_portfolio()
    returns_matrix = _load_returns_matrix()

    # Filter tickers to those present in both signals and returns_matrix
    tickers = [t for t in TICKER_LIST if t in signals.columns and t in returns_matrix.columns]

    # Build common index (signals window) and align
    idx               = signals.index
    returns_aligned   = returns_matrix.reindex(idx)
    spy_close_aligned = spy_close.reindex(idx).ffill()

    print(f"  Signals matrix   : {signals.shape}")
    print(f"  Returns matrix   : {returns_matrix.shape}  (raw)")
    print(f"  Portfolio curves : {portfolio_df.shape}")
    print(f"  Tickers in both  : {len(tickers)}")

    # ── Label regimes (shared across all analyses) ─────────────────────────────
    regimes = label_regimes(macro["vix"], spy_close_aligned, idx)

    # ── 1. Regime Decomposition ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [1/4]  Regime Decomposition  (existing regime_analysis output)")
    print("─" * 70)

    strat_matrix = returns_aligned[tickers].multiply(signals[tickers].shift(1).fillna(0))
    strat_ret    = strat_matrix.mean(axis=1).dropna()
    bnh_ret      = returns_aligned[tickers].mean(axis=1)

    decomp = decompose(strat_ret, bnh_ret, signals, regimes)
    _print_table(decomp)

    # ── 2. Beta Decomposition ──────────────────────────────────────────────────
    print("─" * 70)
    print("  [2/4]  Beta Decomposition  (true alpha Sharpe after SPY beta strip)")
    print("─" * 70)

    beta_df = beta_decomposition(portfolio_df, returns_matrix)
    _print_beta_table(beta_df)

    # ── 3. Regime Correlation ──────────────────────────────────────────────────
    print("─" * 70)
    print("  [3/4]  Regime Correlation  (avg pairwise corr by market regime)")
    print("─" * 70)

    regime_corr_df = regime_correlation(returns_matrix, regimes, portfolio_df)
    _print_regime_corr_table(regime_corr_df)

    # ── 4. Dead Weight ─────────────────────────────────────────────────────────
    print("─" * 70)
    print("  [4/4]  Dead Weight  (long signal on down days, ranked by ticker)")
    print("─" * 70)

    dw_df = dead_weight_analysis(signals, returns_matrix)
    _print_dead_weight_table(dw_df)

    # ── Save ───────────────────────────────────────────────────────────────────
    # Primary output: beta decomposition (most actionable table)
    # Secondary:      regime correlation + dead weight
    out_beta    = RESEARCH_DIR / "correlation_diagnostic.parquet"
    out_rc      = RESEARCH_DIR / "correlation_diagnostic_regime_corr.parquet"
    out_dw      = RESEARCH_DIR / "correlation_diagnostic_dead_weight.parquet"

    beta_df.to_parquet(out_beta, engine="pyarrow", compression="snappy", index=False)
    regime_corr_df.to_parquet(out_rc,   engine="pyarrow", compression="snappy", index=False)
    dw_df.to_parquet(out_dw,   engine="pyarrow", compression="snappy", index=False)

    print("Results saved:")
    print(f"  {out_beta}   (beta decomposition)")
    print(f"  {out_rc}  (regime correlation)")
    print(f"  {out_dw}   (dead weight)")
    print()


if __name__ == "__main__":
    main()
