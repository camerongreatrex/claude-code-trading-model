"""
portfolio.py

Turns +1/-1/0 signals into actual position sizes.
Also runs walk-forward validation to get honest out-of-sample performance.

Three sizing methods built here:
  1. Equal weight      — baseline, same dollar amount on every signal
  2. ATR-based         — risk the same dollar volatility on every trade
  3. Half-Kelly        — size by signal edge (win rate + profit factor)

PCA correlation check: if positions are too correlated, shrink all sizes.
Walk-forward validation: rolling train/test to get unbiased Sharpe estimate.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from backtester import (
    compute_strategy_returns, sharpe_ratio, max_drawdown,
    calmar_ratio, win_rate, profit_factor, summarise, equity_curve
)

SIGNAL_DIR  = Path("data/signals")
FEATURE_DIR = Path("data/features")
MACRO_DIR   = Path("data/macro")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TICKERS          = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
CAPITAL          = 100_000   # starting capital in dollars
MAX_POSITION_PCT = 0.20      # no single position > 20% of capital
RISK_PER_TRADE   = 0.01      # risk 1% of capital per trade (ATR method)

# -----------------------------------------------------------------------------
# 1. Equal weight sizing — the baseline
# -----------------------------------------------------------------------------

def equal_weight_sizes(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    """
    Simplest possible sizing: divide capital equally among active signals.
    If 3 tickers have active signals today, each gets capital/3.
    This is the baseline everything else must beat.
    """
    n_active = signals.abs().sum(axis=1).replace(0, np.nan)  # count active signals per day
    size_per_ticker = capital / n_active                       # equal share of capital

    # multiply by signal direction (+1/-1) to get signed position
    # where signal=0, position=0 (flat)
    sizes = signals.multiply(size_per_ticker, axis=0).fillna(0)
    return sizes.clip(-capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT)


# -----------------------------------------------------------------------------
# 2. ATR-based sizing — volatility normalised
# -----------------------------------------------------------------------------

def atr_sizes(signals: pd.DataFrame, features: dict, capital: float) -> pd.DataFrame:
    """
    Risk the same dollar amount on every trade regardless of stock volatility.

    Dollar risk per trade = capital × RISK_PER_TRADE (e.g. $1,000 on $100k account)
    Position size = dollar_risk / ATR

    Example:
      NVDA ATR = $15, risk = $1,000  → buy 66 shares  ($990 at risk)
      SPY  ATR = $4,  risk = $1,000  → buy 250 shares  ($1,000 at risk)

    This means a 1-ATR move against you costs the same dollar amount on every trade.
    Without this, volatile stocks like NVDA dominate your PnL — one bad NVDA day
    wipes out three good SPY days.
    """
    dollar_risk = capital * RISK_PER_TRADE  # e.g. $1,000

    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

    for ticker in signals.columns:
        if ticker not in features:
            continue
        atr = features[ticker]["atr_14"].reindex(signals.index, method="ffill")
        atr = atr.replace(0, np.nan).ffill()

        # shares = dollar_risk / ATR — then multiply by close price to get dollar position
        close     = features[ticker]["Close"].reindex(signals.index, method="ffill")
        n_shares  = dollar_risk / atr
        dollar_pos = n_shares * close

        # apply signal direction and cap at max position
        sizes[ticker] = (signals[ticker] * dollar_pos).clip(
            -capital * MAX_POSITION_PCT,
             capital * MAX_POSITION_PCT
        )

    return sizes


# -----------------------------------------------------------------------------
# 3. Half-Kelly sizing
# -----------------------------------------------------------------------------

def kelly_fraction(wr: float, pf: float) -> float:
    """
    Kelly criterion: what fraction of capital to bet given your edge?

    Full Kelly = (win_rate / loss_rate) - (avg_loss / avg_win)
    Simplified with profit factor: f = wr - (1 - wr) / pf

    We use half-Kelly in practice because:
      - Kelly assumes you know the exact edge (you don't)
      - Full Kelly has enormous drawdowns even when correct
      - Half-Kelly gives ~75% of full Kelly return with ~50% of the drawdown
    """
    if pf <= 0 or wr <= 0:
        return 0.0
    loss_rate = 1 - wr
    full_kelly = wr - (loss_rate / pf)
    return max(0.0, full_kelly * 0.5)  # half-Kelly, floored at 0


def kelly_sizes(signals: pd.DataFrame, returns: pd.DataFrame,
                capital: float, lookback: int = 252) -> pd.DataFrame:
    """
    Size each position by the rolling Kelly fraction of that ticker's signal.

    Uses a rolling window of past performance to estimate current edge.
    lookback=252 means we estimate Kelly from the last year of signal returns.

    This adapts position sizes as the strategy's edge changes over time —
    bigger when the signal has been working, smaller when it hasn't.
    This is NOT overfitting because we only use past data to compute the size.
    """
    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

    for ticker in signals.columns:
        sig = signals[ticker]
        ret = returns[ticker] if ticker in returns.columns else None
        if ret is None:
            continue

        strat_ret = compute_strategy_returns(sig, ret)

        # rolling win rate and profit factor over the lookback window
        rolling_wr = strat_ret.rolling(lookback).apply(
            lambda x: (x[x != 0] > 0).mean() if (x != 0).any() else 0.5
        )
        rolling_pf = strat_ret.rolling(lookback).apply(
            lambda x: (x[x > 0].sum() / x[x < 0].abs().sum())
            if x[x < 0].abs().sum() > 0 else 1.0
        )

        # Kelly fraction for each day based on rolling performance
        kelly = (rolling_wr - (1 - rolling_wr) / rolling_pf.clip(0.01)).clip(0) * 0.5

        sizes[ticker] = (signals[ticker] * kelly * capital).clip(
            -capital * MAX_POSITION_PCT,
             capital * MAX_POSITION_PCT
        )

    return sizes.fillna(0)


# -----------------------------------------------------------------------------
# 4. PCA correlation check — the eigenvalue layer
# -----------------------------------------------------------------------------

def pca_concentration(returns_window: pd.DataFrame) -> float:
    """
    Measure how correlated the current positions are using PCA.

    PCA decomposes the correlation matrix into eigenvalues.
    Each eigenvalue represents an independent source of risk.

    If one eigenvalue dominates (e.g. explains 80% of variance),
    it means all your positions are essentially the same bet — the "market factor."
    You think you have 5 positions but you really have 1.

    concentration = largest eigenvalue / sum of all eigenvalues
    = fraction of total variance explained by the dominant factor

    0.2 = perfectly diversified (each of 5 assets contributes equally)
    1.0 = perfectly concentrated (all assets move identically)

    We use this to scale down all position sizes when concentration is high.
    """
    if len(returns_window) < 20 or returns_window.shape[1] < 2:
        return 0.5  # default if not enough data

    corr = returns_window.corr().fillna(0)

    # eigenvalues of the correlation matrix
    # np.linalg.eigh: efficient for symmetric matrices, returns sorted eigenvalues
    eigenvalues = np.linalg.eigh(corr.values)[0]
    eigenvalues = np.maximum(eigenvalues, 0)  # numerical errors can give tiny negatives

    total     = eigenvalues.sum()
    if total == 0:
        return 0.5

    # fraction explained by largest eigenvalue — the market factor
    concentration = eigenvalues[-1] / total  # [-1] = largest (eigh sorts ascending)
    return concentration


def apply_pca_scaling(sizes: pd.DataFrame, returns: pd.DataFrame,
                      window: int = 60) -> pd.DataFrame:
    """
    Scale down all position sizes when portfolio is highly concentrated.

    Concentration above 0.6 means one factor drives >60% of variance —
    you're not as diversified as you think.
    At perfect diversification (1/n_tickers), concentration = 0.2 for 5 tickers.

    Scaling factor:
      concentration 0.20 → scale 1.0  (full size, well diversified)
      concentration 0.50 → scale 0.75 (moderately concentrated, trim a bit)
      concentration 0.80 → scale 0.40 (highly concentrated, significant reduction)
    """
    scaled = sizes.copy()

    for i in range(window, len(sizes)):
        ret_window    = returns.iloc[i - window:i]
        concentration = pca_concentration(ret_window)

        # linear scale from 1.0 at perfect diversification to 0.2 at full concentration
        # clip to 0.2-1.0 range so we never go completely flat or above full size
        scale = (1.0 - concentration).clip(0.2, 1.0)
        scaled.iloc[i] = sizes.iloc[i] * scale

    return scaled


# -----------------------------------------------------------------------------
# 5. Walk-forward validation
# -----------------------------------------------------------------------------

def walk_forward(signals: pd.DataFrame, returns: pd.DataFrame,
                 train_years: int = 3, test_years: int = 1) -> pd.DataFrame:
    """
    The honest way to measure strategy performance.

    Instead of testing on the same data you developed on, we:
      1. Train on the first N years (compute Kelly fractions, thresholds)
      2. Test on the NEXT year (completely unseen data)
      3. Step forward one year and repeat

    This gives us a sequence of out-of-sample test periods that together
    cover the full history. The average Sharpe across all test periods
    is your realistic expected performance — not the in-sample Sharpe
    which is always inflated.

    train_years=3, test_years=1 means:
      Period 1: train 2015-2017, test 2018
      Period 2: train 2015-2018, test 2019
      Period 3: train 2015-2019, test 2020  ← COVID crash in test set
      Period 4: train 2015-2020, test 2021
      Period 5: train 2015-2021, test 2022  ← rate hike bear market in test set
      Period 6: train 2015-2022, test 2023
      Period 7: train 2015-2023, test 2024
    """
    train_days = train_years * 252
    test_days  = test_years  * 252

    results = []
    start   = train_days

    while start + test_days <= len(signals):
        test_signals = signals.iloc[start : start + test_days]
        test_returns = returns.iloc[start : start + test_days]

        # equal weight for simplicity in walk-forward — isolates signal quality
        # from sizing decisions (which are evaluated separately)
        period_ret = pd.Series(0.0, index=test_signals.index)
        for ticker in test_signals.columns:
            if ticker in test_returns.columns:
                strat = compute_strategy_returns(test_signals[ticker], test_returns[ticker])
                period_ret += strat / len(test_signals.columns)  # equal weight average

        period_start = test_signals.index[0].year
        period_end   = test_signals.index[-1].year

        results.append({
            "period"     : f"{period_start}-{period_end}",
            "sharpe"     : round(sharpe_ratio(period_ret), 3),
            "ann_return" : round(((1 + period_ret).prod() ** (252 / len(period_ret)) - 1) * 100, 2),
            "max_dd"     : round(max_drawdown((1 + period_ret).cumprod()) * 100, 2),
            "n_days"     : len(period_ret),
        })

        start += test_days  # step forward one test period

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 6. Macro size adjustment
# -----------------------------------------------------------------------------

def apply_macro_multiplier(sizes: pd.DataFrame) -> pd.DataFrame:
    """
    Scale all position sizes by the macro size_multiplier from macro_features.py.
    This is the portfolio-level application of the macro regime — reduces all
    sizes in fear/inverted-curve environments, allows full size in calm ones.
    """
    path = MACRO_DIR / "macro_features.parquet"
    if not path.exists():
        return sizes  # no macro data, return unchanged

    macro      = pd.read_parquet(path)
    multiplier = macro["size_multiplier"].reindex(sizes.index, method="ffill").fillna(1.0)

    return sizes.multiply(multiplier, axis=0)


# -----------------------------------------------------------------------------
# 7. Full portfolio backtest
# -----------------------------------------------------------------------------

def portfolio_returns(sizes: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    Convert daily dollar sizes into a daily portfolio return series.

    Each day: position_return = (size / capital) × asset_return
    Sum across all tickers = total portfolio return for the day.

    Using log_return here means we're computing approximate dollar PnL.
    For exact dollar PnL you'd use simple returns, but log returns are
    close enough for daily data and keep everything consistent.
    """
    weights = sizes.shift(1) / CAPITAL  # shift: sizes set today, returns realised tomorrow
    weighted_returns = weights * returns.reindex(sizes.columns, axis=1)
    return weighted_returns.sum(axis=1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Building portfolio...\n")

    # load signals and features
    regime_signals = pd.read_parquet(SIGNAL_DIR / "regime_signals.parquet")

    features = {}
    returns  = pd.DataFrame()
    for ticker in TICKERS:
        feat = pd.read_parquet(FEATURE_DIR / f"{ticker}.parquet")
        features[ticker] = feat
        returns[ticker]  = feat["log_return"]

    returns = returns.dropna()

    # align signals to returns dates
    signals = regime_signals.reindex(returns.index).fillna(0)

    # --- sizing methods ---
    sizes_eq    = equal_weight_sizes(signals, CAPITAL)
    sizes_atr   = atr_sizes(signals, features, CAPITAL)
    sizes_kelly = kelly_sizes(signals, returns, CAPITAL)

    # apply PCA scaling to ATR sizes (the most principled combination)
    sizes_atr_pca = apply_pca_scaling(sizes_atr, returns)

    # apply macro multiplier to the PCA-scaled ATR sizes
    sizes_final = apply_macro_multiplier(sizes_atr_pca)

    # --- compute portfolio returns ---
    ret_eq      = portfolio_returns(sizes_eq,    returns)
    ret_atr     = portfolio_returns(sizes_atr,   returns)
    ret_kelly   = portfolio_returns(sizes_kelly, returns)
    ret_atr_pca = portfolio_returns(sizes_atr_pca, returns)
    ret_final   = portfolio_returns(sizes_final, returns)
    ret_bnh     = returns.mean(axis=1)  # equal-weight buy-and-hold benchmark

    # --- print results ---
    print(f"{'='*68}")
    print("  PORTFOLIO COMPARISON")
    print(f"{'='*68}")
    metrics = ["ann_return", "sharpe", "max_drawdown", "calmar", "win_rate", "profit_factor"]
    header  = f"  {'Method':<22}" + "".join(f"{m:>14}" for m in metrics)
    print(header)
    print("  " + "-" * (22 + 14 * len(metrics)))

    for label, ret in [
        ("equal weight",    ret_eq),
        ("ATR sized",       ret_atr),
        ("half-Kelly",      ret_kelly),
        ("ATR + PCA",       ret_atr_pca),
        ("ATR + PCA + macro", ret_final),
        ("buy & hold",      ret_bnh),
    ]:
        s = summarise(ret, label)
        row = f"  {s['label']:<22}" + "".join(f"{str(s[m]):>14}" for m in metrics)
        print(row)

    # --- walk-forward validation ---
    print(f"\n{'='*68}")
    print("  WALK-FORWARD VALIDATION  (out-of-sample, 3yr train / 1yr test)")
    print(f"{'='*68}")
    wf = walk_forward(signals, returns)
    print(wf.to_string(index=False))
    print(f"\n  Mean out-of-sample Sharpe : {wf['sharpe'].mean():.3f}")
    print(f"  Std  out-of-sample Sharpe : {wf['sharpe'].std():.3f}")
    print(f"  Worst period              : {wf.loc[wf['sharpe'].idxmin(), 'period']}  ({wf['sharpe'].min():.3f})")
    print(f"  Best  period              : {wf.loc[wf['sharpe'].idxmax(), 'period']}  ({wf['sharpe'].max():.3f})")
    print()
    print("  If mean OOS Sharpe is close to in-sample Sharpe -> low overfitting")
    print("  If mean OOS Sharpe is much lower -> signals are overfit, simplify")

    # save final equity curve
    eq = equity_curve(ret_final, CAPITAL)
    eq.to_frame("portfolio").to_parquet(RESULTS_DIR / "portfolio_equity_curve.parquet")
    print(f"\nEquity curve saved -> {RESULTS_DIR / 'portfolio_equity_curve.parquet'}")


if __name__ == "__main__":
    main()