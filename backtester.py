"""
backtester.py

Simulates trading signals against historical returns.
Computes full performance profile per ticker and cross-asset.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from data_pipeline import TICKER_LIST, ASSET_CLASS

SIGNAL_DIR  = Path("data/signals")
FEATURE_DIR = Path("data/features")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_strategy_returns(signals: pd.Series, returns: pd.Series) -> pd.Series:
    # shift(1): signal seen at end of day, trade executed next day — no lookahead
    return signals.shift(1) * returns


def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    rolling_peak = cumulative_returns.cummax()
    drawdown     = (cumulative_returns - rolling_peak) / rolling_peak
    return drawdown.min()


def calmar_ratio(annualised_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return annualised_return / abs(max_dd)


def win_rate(strategy_returns: pd.Series) -> float:
    active = strategy_returns[strategy_returns != 0]
    if len(active) == 0:
        return 0.0
    return (active > 0).sum() / len(active)


def profit_factor(strategy_returns: pd.Series) -> float:
    gains  = strategy_returns[strategy_returns > 0].sum()
    losses = strategy_returns[strategy_returns < 0].abs().sum()
    if losses == 0:
        return np.inf
    return gains / losses


def summarise(strategy_returns: pd.Series, label: str = "") -> dict:
    strategy_returns = strategy_returns.dropna()
    cum_returns  = (1 + strategy_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    n_years      = len(strategy_returns) / 252
    ann_return   = (1 + total_return) ** (1 / n_years) - 1
    max_dd       = max_drawdown(cum_returns)

    return {
        "label"         : label,
        "total_return"  : round(total_return * 100, 2),
        "ann_return"    : round(ann_return   * 100, 2),
        "sharpe"        : round(sharpe_ratio(strategy_returns), 3),
        "max_drawdown"  : round(max_dd       * 100, 2),
        "calmar"        : round(calmar_ratio(ann_return, max_dd), 3),
        "win_rate"      : round(win_rate(strategy_returns)     * 100, 2),
        "profit_factor" : round(profit_factor(strategy_returns), 3),
        "n_trades"      : int((strategy_returns != 0).sum()),
    }


def buy_and_hold(returns: pd.Series) -> pd.Series:
    return returns


def equity_curve(strategy_returns: pd.Series, starting_capital: float = 10_000) -> pd.Series:
    return starting_capital * (1 + strategy_returns).cumprod()


def backtest_ticker(ticker: str) -> dict:
    sig_df  = pd.read_parquet(SIGNAL_DIR  / f"{ticker}.parquet")
    feat_df = pd.read_parquet(FEATURE_DIR / f"{ticker}.parquet")

    df = sig_df.join(feat_df[["log_return"]], how="inner", rsuffix="_feat")
    df = df.dropna(subset=["log_return", "signal_regime", "signal_composite"])

    returns = df["log_return"]

    strat_regime    = compute_strategy_returns(df["signal_regime"],    returns)
    strat_composite = compute_strategy_returns(df["signal_composite"], returns)
    strat_bnh       = buy_and_hold(returns)

    results = {
        "regime"    : summarise(strat_regime,     f"{ticker} regime"),
        "composite" : summarise(strat_composite,  f"{ticker} composite"),
        "buy_hold"  : summarise(strat_bnh,         f"{ticker} buy & hold"),
    }

    curves = pd.DataFrame({
        "regime"    : equity_curve(strat_regime),
        "composite" : equity_curve(strat_composite),
        "buy_hold"  : equity_curve(strat_bnh),
    })
    curves.to_parquet(RESULTS_DIR / f"{ticker}_curves.parquet")
    return results


def print_results(ticker: str, results: dict) -> None:
    asset_class = ASSET_CLASS[ticker]
    print(f"\n{'='*68}")
    print(f"  {ticker}  ({asset_class})")
    print(f"{'='*68}")
    metrics = ["ann_return", "sharpe", "max_drawdown", "calmar", "win_rate", "profit_factor", "n_trades"]
    print(f"  {'Strategy':<26}" + "".join(f"{m:>14}" for m in metrics))
    print("  " + "-" * (26 + 14 * len(metrics)))
    for res in results.values():
        print(f"  {res['label']:<26}" + "".join(f"{str(res[m]):>14}" for m in metrics))


def main():
    print("Running backtests...\n")
    all_results = {}

    for ticker in TICKER_LIST:
        results = backtest_ticker(ticker)
        all_results[ticker] = results
        print_results(ticker, results)

    print(f"\n{'='*68}")
    print("  BEST SHARPE PER TICKER")
    print(f"{'='*68}")
    for ticker, results in all_results.items():
        best = max(results.items(), key=lambda x: x[1]["sharpe"])
        print(f"  {ticker} ({ASSET_CLASS[ticker]}): {best[1]['label']:<28}"
              f"Sharpe {best[1]['sharpe']}  |  Ann. {best[1]['ann_return']}%  |  Max DD {best[1]['max_drawdown']}%")


if __name__ == "__main__":
    main()