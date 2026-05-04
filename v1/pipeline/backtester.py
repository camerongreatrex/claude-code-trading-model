"""
Simulates signals vs historical returns. Signals shifted +1d (observe close T,
trade open T+1). Transaction costs applied per position change. Outputs equity
curves to data/results/. Consumed by portfolio.py and dashboard.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .data_pipeline import TICKER_LIST, ASSET_CLASS

SIGNAL_DIR  = Path("data/v1/signals")
FEATURE_DIR = Path("data/v1/features")
RESULTS_DIR = Path("data/v1/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Transaction cost model: one-way = COST_BY_ASSET_CLASS[class] + SLIPPAGE.
# Spread sized by AUM/liquidity per class; slippage covers open-fill uncertainty.

COST_BY_ASSET_CLASS: dict[str, float] = {
    "equity_index": 0.0002,   # 0.02% — SPY/IWM/EEM; massive liquidity, penny-wide spreads
    "bond"        : 0.0003,   # 0.03% — TLT; liquid but slightly wider than equity index
    "commodity"   : 0.0003,   # 0.03% — GLD; liquid, slight spread vs underlying spot
    "sector_etf"  : 0.0004,   # 0.04% — XLE/XLU/XLF; smaller AUM, wider spreads
    "stock"       : 0.0005,   # 0.05% — individual stocks; idiosyncratic spread + impact
}

SLIPPAGE = 0.0002   # 0.02% per trade — open-price execution uncertainty


def get_cost(ticker: str) -> float:
    """One-way cost = asset-class spread + slippage. Round-trip is 2x."""
    asset_class = ASSET_CLASS.get(ticker, "stock")
    return COST_BY_ASSET_CLASS.get(asset_class, COST_BY_ASSET_CLASS["stock"]) + SLIPPAGE


def compute_strategy_returns(signals: pd.Series, returns: pd.Series,
                              cost: float = 0.0007) -> pd.Series:
    """
    Signal -> returns with shift(1) execution and turnover cost.
    Returns: position * returns - turnover * cost.
    """
    position = signals.shift(1).fillna(0)
    # abs(diff): 1->-1 reversal costs 2 units
    turnover = position.diff().abs().fillna(0)
    return position * returns - turnover * cost


def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Annualised Sharpe (rf=0). Returns 0.0 if std is zero."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Min of (cum - rolling_peak)/rolling_peak. Float <= 0."""
    rolling_peak = cumulative_returns.cummax()
    drawdown     = (cumulative_returns - rolling_peak) / rolling_peak
    return drawdown.min()


def calmar_ratio(annualised_return: float, max_dd: float) -> float:
    """ann_return / |max_dd|. Returns 0.0 if max_dd is 0."""
    if max_dd == 0:
        return 0.0
    return annualised_return / abs(max_dd)


def win_rate(strategy_returns: pd.Series) -> float:
    """Profitable fraction of active (non-zero return) days."""
    active = strategy_returns[strategy_returns != 0]
    if len(active) == 0:
        return 0.0
    return (active > 0).sum() / len(active)


def profit_factor(strategy_returns: pd.Series) -> float:
    """Gross gains / gross losses. Returns np.inf if no losses."""
    gains  = strategy_returns[strategy_returns > 0].sum()
    losses = strategy_returns[strategy_returns < 0].abs().sum()
    if losses == 0:
        return np.inf
    return gains / losses


def summarise(strategy_returns: pd.Series, label: str = "") -> dict:
    """
    Full perf metric dict: label, total_return, ann_return, sharpe,
    max_drawdown, calmar, win_rate, profit_factor, n_trades.
    """
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
    """Passive benchmark: returns input unchanged."""
    return returns


def equity_curve(strategy_returns: pd.Series, starting_capital: float = 10_000) -> pd.Series:
    """Compounds returns from starting_capital (default $10,000)."""
    return starting_capital * (1 + strategy_returns).cumprod()


def backtest_ticker(ticker: str) -> dict:
    """
    Run regime/composite/buy_hold backtest for one ticker.
    Writes {ticker}_curves.parquet. Returns {} if data missing.
    """
    sig_path  = SIGNAL_DIR  / f"{ticker}.parquet"
    feat_path = FEATURE_DIR / f"{ticker}.parquet"
    if not sig_path.exists() or not feat_path.exists():
        print(f"  {ticker}: signal or feature data missing — skipped")
        return {}
    sig_df  = pd.read_parquet(sig_path)
    feat_df = pd.read_parquet(feat_path)

    df = sig_df.join(feat_df[["log_return"]], how="inner", rsuffix="_feat")
    df = df.dropna(subset=["log_return", "signal_regime", "signal_composite"])

    returns = df["log_return"]

    # Per-asset-class cost: avoids over-penalising liquid ETFs.
    cost = get_cost(ticker)
    strat_regime    = compute_strategy_returns(df["signal_regime"],    returns, cost=cost)
    strat_composite = compute_strategy_returns(df["signal_composite"], returns, cost=cost)
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
    """Print formatted comparison table for one ticker."""
    asset_class = ASSET_CLASS[ticker]
    cost        = get_cost(ticker)
    print(f"\n{'='*68}")
    print(f"  {ticker}  ({asset_class})  — cost {cost*100:.3f}% per side / {cost*200:.3f}% round-trip")
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
        if not results:
            continue
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