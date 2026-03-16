"""
portfolio.py

Turns +1/-1/0 signals into actual position sizes across the diversified universe.
Walk-forward validation gives honest out-of-sample performance.

Sizing methods:
  1. Equal weight  — baseline
  2. ATR-based     — volatility normalised (primary method)
  3. Half-Kelly    — edge-adjusted
  4. ATR + PCA     — correlation-aware
  5. ATR + PCA + macro — full system
"""

import numpy as np
import pandas as pd
from pathlib import Path
from backtester import (
    compute_strategy_returns, sharpe_ratio, max_drawdown,
    calmar_ratio, win_rate, profit_factor, summarise, equity_curve
)
from data_pipeline import TICKER_LIST, ASSET_CLASS

SIGNAL_DIR  = Path("data/signals")
FEATURE_DIR = Path("data/features")
MACRO_DIR   = Path("data/macro")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL          = 100_000
MAX_POSITION_PCT = 0.20
RISK_PER_TRADE   = 0.01


def equal_weight_sizes(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Equal capital split among active signals each day."""
    n_active = signals.abs().sum(axis=1).replace(0, np.nan)
    sizes    = signals.multiply(capital / n_active, axis=0).fillna(0)
    return sizes.clip(-capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT)


def atr_sizes(signals: pd.DataFrame, features: dict, capital: float) -> pd.DataFrame:
    """
    Risk the same dollar amount per trade regardless of asset volatility.
    dollar_risk / ATR = shares. 1-ATR adverse move always costs the same.
    Prevents volatile assets (NVDA, GLD) from dominating portfolio PnL.
    """
    dollar_risk = capital * RISK_PER_TRADE
    sizes       = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

    for ticker in signals.columns:
        if ticker not in features:
            continue
        atr   = features[ticker]["atr_14"].reindex(signals.index, method="ffill").ffill()
        close = features[ticker]["Close"].reindex(signals.index,   method="ffill").ffill()
        atr   = atr.replace(0, np.nan).ffill()

        dollar_pos    = (dollar_risk / atr) * close
        sizes[ticker] = (signals[ticker] * dollar_pos).clip(
            -capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT
        )
    return sizes


def kelly_sizes(signals: pd.DataFrame, returns: pd.DataFrame,
                capital: float, lookback: int = 252) -> pd.DataFrame:
    """
    Rolling half-Kelly sizing: adapts as signal edge changes over time.
    Uses only past data — no lookahead. Bigger when signal working, smaller when not.
    """
    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

    for ticker in signals.columns:
        if ticker not in returns.columns:
            continue
        strat_ret  = compute_strategy_returns(signals[ticker], returns[ticker])

        rolling_wr = strat_ret.rolling(lookback).apply(
            lambda x: float((x[x != 0] > 0).mean()) if (x != 0).any() else 0.5
        )
        rolling_pf = strat_ret.rolling(lookback).apply(
            lambda x: float(x[x > 0].sum() / x[x < 0].abs().sum())
            if x[x < 0].abs().sum() > 0 else 1.0
        )
        kelly = (rolling_wr - (1 - rolling_wr) / rolling_pf.clip(0.01)).clip(0) * 0.5
        sizes[ticker] = (signals[ticker] * kelly * capital).clip(
            -capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT
        )
    return sizes.fillna(0)


def pca_concentration(returns_window: pd.DataFrame) -> float:
    """
    Largest eigenvalue / sum of eigenvalues.
    Measures how much portfolio variance is explained by one shared factor.
    0.2 = ideal diversification for 5 assets. 0.8+ = crisis correlation.
    """
    if len(returns_window) < 20 or returns_window.shape[1] < 2:
        return 1 / max(returns_window.shape[1], 1)

    corr        = returns_window.corr().fillna(0)
    eigenvalues = np.maximum(np.linalg.eigh(corr.values)[0], 0)
    total       = eigenvalues.sum()
    return eigenvalues[-1] / total if total > 0 else 0.5


def apply_pca_scaling(sizes: pd.DataFrame, returns: pd.DataFrame,
                      window: int = 60) -> pd.DataFrame:
    """
    Scale positions down when assets are highly correlated.
    ideal_concentration = 1/n_tickers.
    scale = ideal / actual, clipped 0.3-1.0.
    With a diversified universe, concentration should average lower than before.
    """
    scaled = sizes.copy()
    n      = sizes.shape[1]
    ideal  = 1.0 / n if n > 0 else 0.125  # 1/8 for 8-asset universe

    for i in range(window, len(sizes)):
        concentration = pca_concentration(returns.iloc[i - window:i])
        scale         = (ideal / concentration).clip(0.3, 1.0)
        scaled.iloc[i] = sizes.iloc[i] * scale

    return scaled


def apply_macro_multiplier(sizes: pd.DataFrame) -> pd.DataFrame:
    """
    Apply macro size_multiplier from macro_features.py once here.
    Not applied in signal_generation — single application prevents double-dampening.
    0.5x in fear/inverted environments, 1.2x in calm/steep environments.
    """
    path = MACRO_DIR / "macro_features.parquet"
    if not path.exists():
        print("  Macro data not found — skipping")
        return sizes

    macro      = pd.read_parquet(path)
    multiplier = macro["size_multiplier"].reindex(sizes.index, method="ffill").fillna(1.0)
    return sizes.multiply(multiplier, axis=0)


def portfolio_returns(sizes: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """shift(1): sizes set today, returns realised tomorrow. No lookahead."""
    weights = sizes.shift(1) / CAPITAL
    return (weights * returns.reindex(columns=sizes.columns)).sum(axis=1)


def walk_forward(signals: pd.DataFrame, returns: pd.DataFrame,
                 train_years: int = 3, test_years: int = 1) -> pd.DataFrame:
    """
    Rolling out-of-sample validation. Each test period is genuinely unseen.
    Equal-weight used here to isolate signal quality from sizing effects.
    Mean OOS Sharpe close to in-sample = low overfitting.
    """
    train_days = train_years * 252
    test_days  = test_years  * 252
    results    = []
    start      = train_days

    while start + test_days <= len(signals):
        test_sig   = signals.iloc[start : start + test_days]
        test_ret   = returns.iloc[start : start + test_days]
        period_ret = pd.Series(0.0, index=test_sig.index)

        for ticker in test_sig.columns:
            if ticker in test_ret.columns:
                strat = compute_strategy_returns(test_sig[ticker], test_ret[ticker])
                period_ret += strat / len(test_sig.columns)

        y0, y1 = test_sig.index[0].year, test_sig.index[-1].year
        results.append({
            "period"     : f"{y0}-{y1}",
            "sharpe"     : round(sharpe_ratio(period_ret), 3),
            "ann_return" : round(((1 + period_ret).prod() ** (252 / len(period_ret)) - 1) * 100, 2),
            "max_dd"     : round(max_drawdown((1 + period_ret).cumprod()) * 100, 2),
            "n_days"     : len(period_ret),
        })
        start += test_days

    return pd.DataFrame(results)


def main():
    print("Building portfolio...\n")

    regime_signals = pd.read_parquet(SIGNAL_DIR / "regime_signals.parquet")

    features, returns = {}, pd.DataFrame()
    for ticker in TICKER_LIST:
        feat             = pd.read_parquet(FEATURE_DIR / f"{ticker}.parquet")
        features[ticker] = feat
        returns[ticker]  = feat["log_return"]

    returns = returns.dropna()
    signals = regime_signals.reindex(returns.index).fillna(0)

    print("Correlation matrix of returns (should be lower with diversified universe):")
    print(returns.corr().round(2))
    print()

    sizes_eq      = equal_weight_sizes(signals, CAPITAL)
    sizes_atr     = atr_sizes(signals, features, CAPITAL)
    sizes_kelly   = kelly_sizes(signals, returns, CAPITAL)
    sizes_atr_pca = apply_pca_scaling(sizes_atr, returns)
    sizes_final   = apply_macro_multiplier(sizes_atr_pca)

    ret_eq      = portfolio_returns(sizes_eq,      returns)
    ret_atr     = portfolio_returns(sizes_atr,     returns)
    ret_kelly   = portfolio_returns(sizes_kelly,   returns)
    ret_atr_pca = portfolio_returns(sizes_atr_pca, returns)
    ret_final   = portfolio_returns(sizes_final,   returns)
    ret_bnh     = returns.mean(axis=1)

    print(f"{'='*68}")
    print("  PORTFOLIO COMPARISON")
    print(f"{'='*68}")
    metrics = ["ann_return", "sharpe", "max_drawdown", "calmar", "win_rate", "profit_factor"]
    print(f"  {'Method':<26}" + "".join(f"{m:>14}" for m in metrics))
    print("  " + "-" * (26 + 14 * len(metrics)))

    for label, ret in [
        ("equal weight",      ret_eq),
        ("ATR sized",         ret_atr),
        ("half-Kelly",        ret_kelly),
        ("ATR + PCA",         ret_atr_pca),
        ("ATR + PCA + macro", ret_final),
        ("buy & hold",        ret_bnh),
    ]:
        s   = summarise(ret, label)
        row = f"  {s['label']:<26}" + "".join(f"{str(s[m]):>14}" for m in metrics)
        print(row)

    print(f"\n{'='*68}")
    print("  WALK-FORWARD VALIDATION  (3yr train / 1yr test)")
    print(f"{'='*68}")
    wf = walk_forward(signals, returns)
    print(wf.to_string(index=False))
    print(f"\n  Mean OOS Sharpe : {wf['sharpe'].mean():.3f}")
    print(f"  Std  OOS Sharpe : {wf['sharpe'].std():.3f}")
    print(f"  Worst period    : {wf.loc[wf['sharpe'].idxmin(), 'period']}  ({wf['sharpe'].min():.3f})")
    print(f"  Best  period    : {wf.loc[wf['sharpe'].idxmax(), 'period']}  ({wf['sharpe'].max():.3f})")
    print()
    print("  Mean OOS close to in-sample → low overfitting")
    print("  High std → inconsistent, signals need simplifying")

    equity_curve(ret_final, CAPITAL).to_frame("portfolio").to_parquet(
        RESULTS_DIR / "portfolio_equity_curve.parquet"
    )
    print(f"\nEquity curve saved -> {RESULTS_DIR / 'portfolio_equity_curve.parquet'}")


if __name__ == "__main__":
    main()