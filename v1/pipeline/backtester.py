"""
backtester.py
-------------
Simulates trading signals against historical daily returns to produce a
full performance profile for each ticker and the combined portfolio.

Design principle: honest accounting
─────────────────────────────────────
  - Signals are shifted by 1 day: you observe the close of day T but only
    trade at the open of day T+1.  No look-ahead bias.
  - Transaction costs are applied to every position change (0.05% per side).
    A round-trip costs 0.10%.  This prevents the backtest from rewarding
    high-turnover signals that would be destroyed by real spreads and commissions.
  - No slippage model beyond the fixed cost.  Adding a market-impact model
    would be more realistic but adds complexity that is disproportionate for
    a daily strategy with liquid ETFs and large-caps.

Performance metrics produced
─────────────────────────────
  total_return   — cumulative % return over the full period
  ann_return     — annualised return (geometric)
  sharpe         — annualised Sharpe ratio (risk-adjusted return)
  max_drawdown   — worst peak-to-trough loss (%)
  calmar         — ann_return / |max_drawdown| (return per unit of drawdown risk)
  win_rate       — % of active trading days that were profitable
  profit_factor  — gross profits / gross losses (>1 = profitable in aggregate)
  n_trades       — number of days with non-zero strategy return (proxy for activity)

Outputs (written to data/results/)
───────────────────────────────────
  {TICKER}_curves.parquet — equity curves for regime / composite / buy-hold

Consumed by
───────────
  portfolio.py  — imports compute_strategy_returns, summarise, equity_curve
  dashboard.py  — reads equity curve parquets for the backtest charts
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .data_pipeline import TICKER_LIST, ASSET_CLASS

SIGNAL_DIR  = Path("data/v1/signals")
FEATURE_DIR = Path("data/v1/features")
RESULTS_DIR = Path("data/v1/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Transaction cost model
# ──────────────────────
# Costs are split into two components that are summed per trade:
#
#   1. Bid-ask spread (per asset class)
#      The half-spread is what you pay to cross the market.  Each asset class
#      has a different spread based on AUM, liquidity, and market-maker density.
#
#        equity_index ETFs (SPY, IWM, EEM):
#          SPY spread ≈ $0.01 on $560 ≈ 0.002%.  Using 0.02% adds a safety
#          margin for adverse fills and for IWM/EEM which are slightly wider.
#
#        bond / commodity ETFs (TLT, GLD):
#          TLT spread ≈ $0.01 on $90 ≈ 0.011%.  Less AUM than equity index
#          ETFs, slightly wider.  0.03% is a reasonable upper bound.
#
#        sector ETFs (XLE, XLU, XLF):
#          AUM is 5–10× smaller than SPY; spreads run 0.02–0.05%.
#          0.04% is a midpoint.
#
#        individual stocks (JPM, NVDA, AMZN, etc.):
#          Large-caps have tight spreads (~0.01–0.03%) but experience more
#          price impact than ETFs for the same dollar size.  0.05% covers
#          spread + small-cap names like GE/INTC with wider spreads.
#
#   2. Open-price slippage (uniform 0.02%)
#      Signals are observed at close T; trades execute at open T+1.
#      Any pre-market movement, order queue position, or gap open moves
#      your actual fill away from the expected price.  0.02% is a
#      conservative estimate for non-urgent, patient limit-style orders.
#
# Total one-way cost = COST_BY_ASSET_CLASS[asset_class] + SLIPPAGE
# A round-trip costs 2× the one-way total.

COST_BY_ASSET_CLASS: dict[str, float] = {
    "equity_index": 0.0002,   # 0.02% — SPY/IWM/EEM; massive liquidity, penny-wide spreads
    "bond"        : 0.0003,   # 0.03% — TLT; liquid but slightly wider than equity index
    "commodity"   : 0.0003,   # 0.03% — GLD; liquid, slight spread vs underlying spot
    "sector_etf"  : 0.0004,   # 0.04% — XLE/XLU/XLF; smaller AUM, wider spreads
    "stock"       : 0.0005,   # 0.05% — individual stocks; idiosyncratic spread + impact
}

SLIPPAGE = 0.0002   # 0.02% per trade — open-price execution uncertainty


def get_cost(ticker: str) -> float:
    """
    Return the total one-way transaction cost for a given ticker.

    Combines the asset-class-specific spread cost with a fixed slippage
    component that models open-price execution uncertainty.

    Args:
        ticker: Ticker symbol present in ASSET_CLASS.

    Returns:
        One-way cost as a decimal fraction (e.g. 0.0007 = 0.07%).
        Round-trip cost is 2× this value.
    """
    asset_class = ASSET_CLASS.get(ticker, "stock")
    return COST_BY_ASSET_CLASS.get(asset_class, COST_BY_ASSET_CLASS["stock"]) + SLIPPAGE


def compute_strategy_returns(signals: pd.Series, returns: pd.Series,
                              cost: float = 0.0007) -> pd.Series:
    """
    Convert a signal Series into a returns Series, accounting for execution delay
    and transaction costs.

    Execution model:
      Signal observed at close of day T → position held from open of day T+1.
      This is modelled by shift(1): position[T+1] = signal[T].

    Transaction cost:
      Charged on the absolute change in position each day.
        Entry  (0→1): costs 1 unit × cost
        Exit   (1→0): costs 1 unit × cost
        Hold   (1→1): costs 0 (no change)

    Cost breakdown (default 0.07% for stocks):
      Spread component: 0.05% (stocks) — bid-ask half-spread to cross market
      Slippage:         0.02%          — open-price execution uncertainty
      Equity index ETFs use a lower default via get_cost(); pass explicitly.

    Args:
        signals: Signal Series of {-1, 0, 1} values (from signal_generation.py).
        returns: Daily log return Series aligned to signals.index.
        cost:    One-way transaction cost as fraction of notional.
                 Use get_cost(ticker) for the per-asset-class default.

    Returns:
        Strategy daily return Series: (position × returns) − (turnover × cost).
    """
    position = signals.shift(1).fillna(0)
    # abs(diff): 0→1 or 1→0 costs 1 unit; 1→-1 costs 2 units (full reversal)
    turnover = position.diff().abs().fillna(0)
    return position * returns - turnover * cost


def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    """
    Annualised Sharpe ratio: excess return per unit of volatility.

    Assumes the risk-free rate is zero (appropriate for a strategy that is
    frequently in cash — the cash position earns approximately the risk-free
    rate, but this is not modelled here for simplicity).

    Sharpe interpretation:
      < 0.5  = poor risk-adjusted return
      0.5–1  = acceptable
      1–2    = good (most institutional funds target 1+)
      > 2    = exceptional (hedge funds rarely sustain > 2)

    Args:
        returns: Daily strategy return Series.
        periods: Trading days per year (default 252 for US equities).

    Returns:
        Annualised Sharpe ratio (float).  Returns 0.0 if returns have zero std.
    """
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a fraction (negative number).

    Computed as: min((cumulative_returns - rolling_peak) / rolling_peak)

    A drawdown of -0.15 means the portfolio fell 15% below its previous
    all-time high at some point during the period.

    Args:
        cumulative_returns: Cumulative return Series (1.0 = starting value,
                            1.10 = 10% cumulative gain).

    Returns:
        Float ≤ 0.  Multiply by 100 for percentage.
    """
    rolling_peak = cumulative_returns.cummax()
    drawdown     = (cumulative_returns - rolling_peak) / rolling_peak
    return drawdown.min()


def calmar_ratio(annualised_return: float, max_dd: float) -> float:
    """
    Calmar ratio: annualised return divided by the absolute maximum drawdown.

    Measures how much return the strategy earns per unit of drawdown risk.
    Higher is better.

      Calmar > 1  = earns more than 1% per 1% of max drawdown (good)
      Calmar < 0.5 = drawdown risk is high relative to return (poor)

    Widely used by CTA / systematic hedge funds as a risk-adjusted metric
    because max drawdown directly measures investor pain, which Sharpe ratio
    does not (Sharpe is symmetric; drawdown is one-sided downside).

    Args:
        annualised_return: Annualised strategy return (decimal, e.g. 0.12).
        max_dd:            Maximum drawdown (decimal, e.g. -0.15).

    Returns:
        Float.  Returns 0.0 if max_dd is 0 (no drawdown occurred).
    """
    if max_dd == 0:
        return 0.0
    return annualised_return / abs(max_dd)


def win_rate(strategy_returns: pd.Series) -> float:
    """
    Fraction of active trading days where the strategy made money.

    "Active" means days where the strategy return is non-zero (i.e., a
    position was held that day).  Days in cash (return = 0) are excluded
    because they neither win nor lose.

    Args:
        strategy_returns: Daily strategy return Series.

    Returns:
        Float in [0, 1].  0.5 = 50% of active days were profitable.
    """
    active = strategy_returns[strategy_returns != 0]
    if len(active) == 0:
        return 0.0
    return (active > 0).sum() / len(active)


def profit_factor(strategy_returns: pd.Series) -> float:
    """
    Ratio of gross profits to gross losses.

    Profit factor > 1 means the strategy makes more money on winning days
    than it loses on losing days.  This is separate from win_rate: a strategy
    can have win_rate = 40% but profit_factor > 1 if winners are much larger
    than losers (trend-following strategies often look like this).

    Args:
        strategy_returns: Daily strategy return Series.

    Returns:
        Float.  Returns np.inf if there are no losing days.
    """
    gains  = strategy_returns[strategy_returns > 0].sum()
    losses = strategy_returns[strategy_returns < 0].abs().sum()
    if losses == 0:
        return np.inf
    return gains / losses


def summarise(strategy_returns: pd.Series, label: str = "") -> dict:
    """
    Compute the full performance metric suite for a strategy return Series.

    Args:
        strategy_returns: Daily return Series (from compute_strategy_returns).
        label:            Name string included in the output dict for display.

    Returns:
        Dict with keys: label, total_return, ann_return, sharpe,
        max_drawdown, calmar, win_rate, profit_factor, n_trades.
        All numeric values are rounded for clean display.
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
    """
    Trivial buy-and-hold benchmark: always holds, never trades.

    Returns the raw daily log return unchanged.  Used to compare strategy
    performance against passive ownership of the same asset.

    Args:
        returns: Daily log return Series.

    Returns:
        Same Series unchanged.
    """
    return returns


def equity_curve(strategy_returns: pd.Series, starting_capital: float = 10_000) -> pd.Series:
    """
    Convert a daily return Series to a dollar equity curve.

    Compounds each day's return starting from ``starting_capital``.
    Uses (1 + return).cumprod() — standard for log returns that are small,
    approximately equal to the continuous compounding equivalent.

    Args:
        strategy_returns: Daily return Series.
        starting_capital: Starting dollar value (default $10,000).

    Returns:
        Dollar equity curve Series aligned to strategy_returns.index.
    """
    return starting_capital * (1 + strategy_returns).cumprod()


def backtest_ticker(ticker: str) -> dict:
    """
    Run the full backtest for a single ticker and save equity curves.

    Loads signal data and feature data, aligns them on date, then runs
    compute_strategy_returns() for three strategies:
      - regime    (MA-crossover signal with all post-processors)
      - composite (richer score-based signal)
      - buy & hold (passive benchmark for comparison)

    Args:
        ticker: Ticker symbol string.

    Returns:
        Dict with keys ["regime", "composite", "buy_hold"] each mapping to
        a summarise() dict.  Returns empty dict if data files are missing.

    Side effect:
        Writes {ticker}_curves.parquet to RESULTS_DIR.
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

    # Use per-asset-class cost so liquid ETFs (SPY, IWM) aren't over-penalised
    # while individual stocks absorb their wider spreads + open-price slippage.
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
    """
    Print a formatted comparison table for one ticker's backtest results.

    Args:
        ticker:  Ticker symbol string.
        results: Dict from backtest_ticker() containing performance metrics.
    """
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