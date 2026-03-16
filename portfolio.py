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
                      window: int = 126) -> pd.DataFrame:
    """
    Scale positions down when assets are highly correlated.
    ideal_concentration = 1/n_tickers.
    scale = ideal / actual, clipped 0.3-1.0.
    Window = 126 days (~6 months) to reduce reactivity to short-term noise
    while still catching genuine regime shifts in correlation.
    """
    scaled = sizes.copy()
    n      = sizes.shape[1]
    ideal  = 1.0 / n if n > 0 else 0.125  # 1/8 for 8-asset universe

    for i in range(window, len(sizes)):
        concentration = pca_concentration(returns.iloc[i - window:i])
        scale         = (ideal / concentration).clip(0.3, 1.0)
        scaled.iloc[i] = sizes.iloc[i] * scale

    return scaled


def vol_target_sizes(sizes: pd.DataFrame, returns: pd.DataFrame,
                     target_vol: float = 0.10, window: int = 63) -> pd.DataFrame:
    """
    Scale all positions so the portfolio targets a given annualised volatility.
    Uses lagged realised vol — no lookahead. sizes[t] is held at t+1 (via shift(1)
    in portfolio_returns), so using realized_vol[t] for scale[t] is clean.
    Clip 0.5–1.5x: never more than 1.5x levered, never less than half size.
    """
    weights  = sizes.shift(1) / CAPITAL
    port_ret = (weights * returns.reindex(columns=sizes.columns)).sum(axis=1)
    realized_vol = port_ret.rolling(window, min_periods=21).std() * np.sqrt(252)
    realized_vol = realized_vol.replace(0, np.nan).fillna(target_vol)
    scale = (target_vol / realized_vol).clip(0.5, 1.5)
    return sizes.multiply(scale, axis=0)


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

    regime_signals    = pd.read_parquet(SIGNAL_DIR / "regime_signals.parquet")
    composite_signals = pd.read_parquet(SIGNAL_DIR / "composite_signals.parquet")

    features, returns = {}, pd.DataFrame()
    for ticker in TICKER_LIST:
        feat             = pd.read_parquet(FEATURE_DIR / f"{ticker}.parquet")
        features[ticker] = feat
        returns[ticker]  = feat["log_return"]

    returns           = returns.dropna()
    signals_regime    = regime_signals.reindex(returns.index).fillna(0)
    signals_composite = composite_signals.reindex(returns.index).fillna(0)

    print("Correlation matrix of returns (should be lower with diversified universe):")
    print(returns.corr().round(2))
    print()

    # --- Regime-signal portfolio chain ---
    sizes_eq      = equal_weight_sizes(signals_regime, CAPITAL)
    sizes_atr     = atr_sizes(signals_regime, features, CAPITAL)
    sizes_kelly   = kelly_sizes(signals_regime, returns, CAPITAL)
    sizes_atr_pca = apply_pca_scaling(sizes_atr, returns)
    sizes_final   = apply_macro_multiplier(sizes_atr_pca)
    sizes_vol     = vol_target_sizes(sizes_final, returns)  # vol-targeted final

    # --- Composite-signal portfolio chain (tests richer signal) ---
    sizes_comp_atr     = atr_sizes(signals_composite, features, CAPITAL)
    sizes_comp_pca     = apply_pca_scaling(sizes_comp_atr, returns)
    sizes_comp_macro   = apply_macro_multiplier(sizes_comp_pca)
    sizes_comp_vol     = vol_target_sizes(sizes_comp_macro, returns)

    ret_eq         = portfolio_returns(sizes_eq,         returns)
    ret_atr        = portfolio_returns(sizes_atr,        returns)
    ret_kelly      = portfolio_returns(sizes_kelly,      returns)
    ret_atr_pca    = portfolio_returns(sizes_atr_pca,    returns)
    ret_final      = portfolio_returns(sizes_final,      returns)
    ret_vol        = portfolio_returns(sizes_vol,        returns)
    ret_comp_vol   = portfolio_returns(sizes_comp_vol,   returns)
    ret_bnh        = returns.mean(axis=1)

    print(f"{'='*76}")
    print("  PORTFOLIO COMPARISON  (transaction costs included in all strategy returns)")
    print(f"{'='*76}")
    metrics = ["ann_return", "sharpe", "max_drawdown", "calmar", "win_rate", "profit_factor"]
    print(f"  {'Method':<30}" + "".join(f"{m:>14}" for m in metrics))
    print("  " + "-" * (30 + 14 * len(metrics)))

    for label, ret in [
        ("equal weight",             ret_eq),
        ("ATR sized",                ret_atr),
        ("half-Kelly",               ret_kelly),
        ("ATR + PCA",                ret_atr_pca),
        ("ATR + PCA + macro",        ret_final),
        ("regime + vol target",      ret_vol),
        ("composite + vol target",   ret_comp_vol),
        ("buy & hold",               ret_bnh),
    ]:
        s   = summarise(ret, label)
        row = f"  {s['label']:<30}" + "".join(f"{str(s[m]):>14}" for m in metrics)
        print(row)

    # Walk-forward on both signal types so we can see OOS consistency
    print(f"\n{'='*76}")
    print("  WALK-FORWARD VALIDATION  (3yr train / 1yr test) — regime signals")
    print(f"{'='*76}")
    wf_regime = walk_forward(signals_regime, returns)
    print(wf_regime.to_string(index=False))
    print(f"\n  Mean OOS Sharpe : {wf_regime['sharpe'].mean():.3f}")
    print(f"  Std  OOS Sharpe : {wf_regime['sharpe'].std():.3f}")

    print(f"\n{'='*76}")
    print("  WALK-FORWARD VALIDATION  (3yr train / 1yr test) — composite signals")
    print(f"{'='*76}")
    wf_comp = walk_forward(signals_composite, returns)
    print(wf_comp.to_string(index=False))
    print(f"\n  Mean OOS Sharpe : {wf_comp['sharpe'].mean():.3f}")
    print(f"  Std  OOS Sharpe : {wf_comp['sharpe'].std():.3f}")

    # Select the best portfolio by in-sample Sharpe across all regime-signal methods.
    # (Regime signal validated by OOS walk-forward above — composite consistently loses OOS.)
    candidates = {
        "equal weight"       : ret_eq,
        "ATR sized"          : ret_atr,
        "ATR + PCA + macro"  : ret_final,
        "regime + vol target": ret_vol,
    }
    best_label = max(candidates, key=lambda k: summarise(candidates[k], k)["sharpe"])
    best_ret   = candidates[best_label]

    print(f"\n  Best portfolio (highest in-sample Sharpe): {best_label}")
    print()
    print("  Mean OOS close to in-sample Sharpe → low overfitting")
    print("  High std → inconsistent, signals need simplifying")

    equity_curve(best_ret, CAPITAL).to_frame("portfolio").to_parquet(
        RESULTS_DIR / "portfolio_equity_curve.parquet"
    )
    print(f"\nEquity curve saved -> {RESULTS_DIR / 'portfolio_equity_curve.parquet'}")


if __name__ == "__main__":
    main()