"""
Simulates trading the signals from signal_generation.py against historical returns.
Computes a full performance profile: returns, Sharpe, drawdown, win rate, profit factor.

No execution logic lives here — this is pure simulation.
Execution (slippage, order routing) comes later in portfolio.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SIGNAL_DIR  = Path("data/signals")
FEATURE_DIR = Path("data/features")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

# -----------------------------------------------------------------------------
# Core return calculation
# -----------------------------------------------------------------------------

def compute_strategy_returns(signals: pd.Series, returns: pd.Series) -> pd.Series:
    """
    The entire backtesting engine in one line:
      strategy_return = signal_yesterday x return_today

    shift(1): you see the signal at end of day, trade at next open, capture next day's return.
    Without the shift you'd be using today's return to validate today's signal — lookahead bias.
    """
    return signals.shift(1) * returns

# -----------------------------------------------------------------------------
# Performance metrics
# -----------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    """
    Annualised Sharpe: mean / std * sqrt(252).
    Below 1.0 = poor, above 1.5 = good, above 2.0 = excellent.
    Ignores risk-free rate — negligible difference for strategy comparison.
    """
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Largest peak-to-trough loss in the equity curve.
    rolling_peak: highest equity seen so far at each point in time.
    drawdown: how far below that peak you currently are, as a percentage.
    """
    rolling_peak = cumulative_returns.cummax()
    drawdown     = (cumulative_returns - rolling_peak) / rolling_peak
    return drawdown.min()


def calmar_ratio(annualised_return: float, max_dd: float) -> float:
    """Ann. return / abs(max drawdown) — return per unit of drawdown risk."""
    if max_dd == 0:
        return 0.0
    return annualised_return / abs(max_dd)


def win_rate(strategy_returns: pd.Series) -> float:
    """Fraction of active trading days where the strategy made money."""
    active = strategy_returns[strategy_returns != 0]  # ignore flat (signal=0) days
    if len(active) == 0:
        return 0.0
    return (active > 0).sum() / len(active)


def profit_factor(strategy_returns: pd.Series) -> float:
    """
    Total gross profit / total gross loss.
    Above 1.0 = made more than lost. Above 1.5 = solid edge.
    Better than win rate alone — accounts for the size of wins vs losses.
    """
    gains  = strategy_returns[strategy_returns > 0].sum()
    losses = strategy_returns[strategy_returns < 0].abs().sum()
    if losses == 0:
        return np.inf
    return gains / losses


def summarise(strategy_returns: pd.Series, label: str = "") -> dict:
    """Bundle all metrics into one dict for easy comparison across strategies/tickers."""
    strategy_returns = strategy_returns.dropna()
    cum_returns  = (1 + strategy_returns).cumprod()  # compound each day's return
    total_return = cum_returns.iloc[-1] - 1           # final value minus starting value
    n_years      = len(strategy_returns) / 252
    ann_return   = (1 + total_return) ** (1 / n_years) - 1  # CAGR: compound annual growth rate
    max_dd       = max_drawdown(cum_returns)

    return {
        "label"         : label,
        "total_return"  : round(total_return * 100, 2),
        "ann_return"    : round(ann_return   * 100, 2),
        "sharpe"        : round(sharpe_ratio(strategy_returns), 3),
        "max_drawdown"  : round(max_dd       * 100, 2),
        "calmar"        : round(calmar_ratio(ann_return, max_dd), 3),
        "win_rate"      : round(win_rate(strategy_returns)    * 100, 2),
        "profit_factor" : round(profit_factor(strategy_returns), 3),
        "n_trades"      : int((strategy_returns != 0).sum()),
    }

# -----------------------------------------------------------------------------
# Benchmark and equity curve
# -----------------------------------------------------------------------------

def buy_and_hold(returns: pd.Series) -> pd.Series:
    """
    Simplest benchmark: long every day, no signal needed.
    Every strategy must beat this on a risk-adjusted basis —
    otherwise just buy an index fund.
    """
    return returns


def equity_curve(strategy_returns: pd.Series, starting_capital: float = 10_000) -> pd.Series:
    """Dollar equity curve: compounds each day's return from starting capital."""
    return starting_capital * (1 + strategy_returns).cumprod()

# -----------------------------------------------------------------------------
# Single ticker backtest
# -----------------------------------------------------------------------------

def backtest_ticker(ticker: str) -> dict[str, dict]:
    sig_df  = pd.read_parquet(SIGNAL_DIR  / f"{ticker}.parquet")
    feat_df = pd.read_parquet(FEATURE_DIR / f"{ticker}.parquet")

    df = sig_df.join(feat_df[["log_return"]], how="inner", rsuffix="_feat")
    df = df.dropna(subset=["log_return", "signal_composite", "signal_momentum",
                            "signal_mean_rev", "signal_regime"])

    returns = df["log_return"]

    strat_composite = compute_strategy_returns(df["signal_composite"], returns)
    strat_momentum  = compute_strategy_returns(df["signal_momentum"],  returns)
    strat_mean_rev  = compute_strategy_returns(df["signal_mean_rev"],  returns)
    strat_regime    = compute_strategy_returns(df["signal_regime"],    returns)
    strat_bnh       = buy_and_hold(returns)

    results = {
        "composite" : summarise(strat_composite, f"{ticker} composite"),
        "momentum"  : summarise(strat_momentum,  f"{ticker} momentum"),
        "mean_rev"  : summarise(strat_mean_rev,  f"{ticker} mean reversion"),
        "regime"    : summarise(strat_regime,     f"{ticker} regime-switched"),
        "buy_hold"  : summarise(strat_bnh,         f"{ticker} buy & hold"),
    }

    curves = pd.DataFrame({
        "composite" : equity_curve(strat_composite),
        "momentum"  : equity_curve(strat_momentum),
        "mean_rev"  : equity_curve(strat_mean_rev),
        "regime"    : equity_curve(strat_regime),
        "buy_hold"  : equity_curve(strat_bnh),
    })
    curves.to_parquet(RESULTS_DIR / f"{ticker}_curves.parquet")

    return results

# -----------------------------------------------------------------------------
# Pretty print
# -----------------------------------------------------------------------------

def print_results(ticker: str, results: dict[str, dict]) -> None:
    print(f"\n{'='*68}")
    print(f"  {ticker}")
    print(f"{'='*68}")
    metrics = ["ann_return", "sharpe", "max_drawdown", "calmar", "win_rate", "profit_factor", "n_trades"]
    header  = f"  {'Strategy':<26}" + "".join(f"{m:>14}" for m in metrics)
    print(header)
    print("  " + "-" * (26 + 14 * len(metrics)))
    for key, res in results.items():
        row = f"  {res['label']:<26}" + "".join(f"{str(res[m]):>14}" for m in metrics)
        print(row)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Running backtests...\n")
    all_results = {}

    for ticker in TICKERS:
        results = backtest_ticker(ticker)
        all_results[ticker] = results
        print_results(ticker, results)

    print(f"\n{'='*68}")
    print("  BEST SHARPE PER TICKER")
    print(f"{'='*68}")
    for ticker, results in all_results.items():
        best = max(results.items(), key=lambda x: x[1]["sharpe"])
        print(f"  {ticker}: {best[1]['label']:<30} Sharpe {best[1]['sharpe']}  |  Ann. {best[1]['ann_return']}%  |  Max DD {best[1]['max_drawdown']}%")


if __name__ == "__main__":
    main()