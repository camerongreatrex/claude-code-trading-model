"""
v2/backtester.py
----------------
Long/short backtester with walk-forward validation for the beta-neutral model.

Simulates the weekly-rebalanced dollar-neutral portfolio and computes full
performance metrics. Walk-forward uses 3-year training / 1-year test windows.

Metrics reported
────────────────
  Sharpe, max drawdown, annualized return, beta to SPY, alpha,
  information ratio, win rate, avg holding period, avg weekly turnover,
  correlation to SPY (daily returns), performance on SPY-up vs SPY-down days.

Critical check: if |corr to SPY| > 0.30, something is wrong with beta neutrality.

Output
──────
  data/v2/results/backtest_equity.parquet      — equity curve
  data/v2/results/backtest_long_short.parquet   — long/short book curves
  data/v2/results/backtest_metrics.json         — summary statistics
  data/v2/results/walk_forward_results.parquet  — OOS results per window
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from v2.signal_generation import generate_signals
from v2.portfolio import (
    construct_portfolio, compute_rolling_betas, build_portfolio_history,
    CAPITAL,
)
from v2.rebalancer import simulate_rebalances, compute_transaction_costs
from v2.risk_model import (
    compute_vol_scale_series, compute_circuit_breaker_series,
    compute_drawdown, TARGET_VOL,
)
from v2.costs import compute_backtest_costs, summarize_costs

V2_RESULTS_DIR = Path("data/v2/results")
V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Walk-forward parameters ───────────────────────────────────────────────────
TRAIN_YEARS = 3
TEST_YEARS = 1
TRADING_DAYS_PER_YEAR = 252


def backtest(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    composite_scores: pd.DataFrame,
    sectors: dict,
    spy_returns: pd.Series | None = None,
    apply_risk_controls: bool = True,
    label: str = "full",
    volumes: pd.DataFrame | None = None,
    market_caps: dict[str, float] | None = None,
) -> dict:
    """
    Run a full backtest of the beta-neutral strategy.

    Args:
        closes: aligned close prices (T × N)
        returns: aligned daily log returns (T × N)
        composite_scores: T × N composite signal z-scores
        sectors: ticker -> GICS sector mapping
        spy_returns: SPY daily returns for beta computation
        apply_risk_controls: apply vol targeting and circuit breaker
        label: label for this backtest run
        volumes: aligned volume matrix (T × N) for cost model
        market_caps: ticker -> market cap for borrow rate tiering

    Returns:
        dict with equity curves, metrics, and trade statistics
    """
    if spy_returns is None and "SPY" in returns.columns:
        spy_returns = returns["SPY"]

    # Compute rolling betas
    print(f"  [{label}] Computing rolling betas...")
    betas = compute_rolling_betas(returns, spy_returns, window=60)

    # Build portfolio at each rebalance date
    print(f"  [{label}] Building portfolio history...")
    portfolio_history = build_portfolio_history(
        composite_scores, returns, sectors,
        betas=betas, spy_returns=spy_returns,
    )

    if not portfolio_history:
        print(f"  [{label}] Warning: no portfolio history generated")
        return _empty_result(label)

    # Simulate rebalances with turnover management
    print(f"  [{label}] Simulating rebalances...")
    weights_df, all_trades = simulate_rebalances(portfolio_history, sectors)

    if weights_df.empty:
        return _empty_result(label)

    # Compute gross daily portfolio returns (before costs)
    print(f"  [{label}] Computing daily returns...")
    gross_returns, long_returns, short_returns = _compute_daily_returns_gross(
        weights_df, returns, spy_returns
    )

    # Compute realistic costs
    print(f"  [{label}] Computing transaction costs...")
    if volumes is None:
        volumes = pd.DataFrame()
    cost_dict = compute_backtest_costs(
        weights_df, closes, volumes,
        market_caps=market_caps, capital=CAPITAL,
    )

    # Net returns = gross returns - total costs
    net_returns = gross_returns - cost_dict["total"].reindex(gross_returns.index).fillna(0)

    # Apply risk controls (vol targeting + circuit breaker)
    if apply_risk_controls and len(net_returns) > 20:
        equity = (1 + net_returns).cumprod() * CAPITAL
        vol_scale = compute_vol_scale_series(net_returns)
        breaker_scale = compute_circuit_breaker_series(equity)
        combined_scale = (vol_scale * breaker_scale).reindex(net_returns.index).fillna(1.0)
        net_returns = net_returns * combined_scale
        gross_returns = gross_returns * combined_scale

    # Compute metrics
    net_equity = (1 + net_returns).cumprod() * CAPITAL
    gross_equity = (1 + gross_returns).cumprod() * CAPITAL

    metrics = _compute_metrics(
        net_returns, long_returns, short_returns,
        spy_returns, net_equity, all_trades, label
    )

    # Add gross vs net comparison
    gross_sharpe = _sharpe(gross_returns)
    n_years = len(net_returns) / 252
    cost_summary = summarize_costs(cost_dict, n_years)
    metrics["gross_sharpe"] = round(gross_sharpe, 3)
    metrics["net_sharpe"] = metrics["sharpe"]
    metrics.update(cost_summary)
    metrics["gross_ann_return"] = round(gross_returns.mean() * 252, 4)
    metrics["net_max_drawdown"] = metrics["max_drawdown"]

    # Save results
    net_equity.to_frame("equity").to_parquet(
        V2_RESULTS_DIR / f"backtest_equity_{label}.parquet"
    )

    long_short_df = pd.DataFrame({
        "long": (1 + long_returns).cumprod() * (CAPITAL / 2),
        "short": (1 + short_returns).cumprod() * (CAPITAL / 2),
        "combined": net_equity,
        "gross": gross_equity,
    })
    long_short_df.to_parquet(V2_RESULTS_DIR / f"backtest_long_short_{label}.parquet")

    with open(V2_RESULTS_DIR / f"backtest_metrics_{label}.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return {
        "equity_curve": net_equity,
        "gross_equity": gross_equity,
        "long_curve": long_short_df["long"],
        "short_curve": long_short_df["short"],
        "portfolio_returns": net_returns,
        "gross_returns": gross_returns,
        "long_returns": long_returns,
        "short_returns": short_returns,
        "metrics": metrics,
        "trades": all_trades,
        "weights": weights_df,
        "costs": cost_dict,
    }


def _compute_daily_returns_gross(
    weights_df: pd.DataFrame,
    returns: pd.DataFrame,
    spy_returns: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute gross daily portfolio returns (before costs) from weekly weight
    snapshots and daily returns. Weights are forward-filled between rebalance dates.
    """
    # Reindex weights to daily frequency (forward-fill)
    daily_dates = returns.index
    weights_daily = weights_df.reindex(daily_dates).ffill().fillna(0)

    # Align columns
    common_tickers = weights_daily.columns.intersection(returns.columns)
    weights_aligned = weights_daily[common_tickers]
    returns_aligned = returns[common_tickers].reindex(weights_daily.index).fillna(0)

    # Portfolio return = sum of weight * return for each position
    # Shift weights by 1 day to avoid look-ahead (observe signal, trade next day)
    weights_shifted = weights_aligned.shift(1).fillna(0)

    daily_returns = (weights_shifted * returns_aligned).sum(axis=1)

    # Separate long and short book returns
    long_weights = weights_shifted.clip(lower=0)
    short_weights = weights_shifted.clip(upper=0)

    long_returns = (long_weights * returns_aligned).sum(axis=1)
    short_returns = (short_weights * returns_aligned).sum(axis=1)

    return daily_returns, long_returns, short_returns


def _sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio."""
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    return ann_ret / ann_vol if ann_vol > 0 else 0.0


def _compute_metrics(
    portfolio_returns: pd.Series,
    long_returns: pd.Series,
    short_returns: pd.Series,
    spy_returns: pd.Series,
    equity_curve: pd.Series,
    trades: list[dict],
    label: str,
) -> dict:
    """Compute comprehensive performance metrics."""
    # Align spy_returns
    spy = spy_returns.reindex(portfolio_returns.index).fillna(0)

    # Basic metrics
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if len(equity_curve) > 0 else 0

    # Drawdown
    dd = compute_drawdown(equity_curve)
    max_dd = dd.min()

    # Beta to SPY
    if len(spy) > 60:
        cov = portfolio_returns.cov(spy)
        var_spy = spy.var()
        beta = cov / var_spy if var_spy > 0 else 0
    else:
        beta = 0

    # Alpha (annualized)
    spy_ann_return = spy.mean() * 252
    alpha = ann_return - beta * spy_ann_return

    # Correlation to SPY
    if len(spy) > 20:
        corr_to_spy = portfolio_returns.corr(spy)
    else:
        corr_to_spy = 0

    # Information ratio (vs zero benchmark since we're market-neutral)
    tracking_error = portfolio_returns.std() * np.sqrt(252)
    info_ratio = ann_return / tracking_error if tracking_error > 0 else 0

    # Win rate (daily)
    active_days = portfolio_returns[portfolio_returns != 0]
    daily_win_rate = (active_days > 0).mean() if len(active_days) > 0 else 0

    # Trade-level win rate
    trade_pnls = _estimate_trade_pnls(trades)
    trade_win_rate = (trade_pnls > 0).mean() if len(trade_pnls) > 0 else 0

    # Performance on SPY up vs down days
    spy_up = spy > 0
    spy_down = spy < 0

    if spy_up.sum() > 0:
        ret_on_spy_up = portfolio_returns[spy_up].mean() * 252
        pct_positive_on_spy_up = (portfolio_returns[spy_up] > 0).mean()
    else:
        ret_on_spy_up = 0
        pct_positive_on_spy_up = 0

    if spy_down.sum() > 0:
        ret_on_spy_down = portfolio_returns[spy_down].mean() * 252
        pct_positive_on_spy_down = (portfolio_returns[spy_down] > 0).mean()
    else:
        ret_on_spy_down = 0
        pct_positive_on_spy_down = 0

    # Long vs short book
    long_ann = long_returns.mean() * 252
    short_ann = short_returns.mean() * 252

    return {
        "label": label,
        "sharpe": round(sharpe, 3),
        "ann_return": round(ann_return, 4),
        "ann_vol": round(ann_vol, 4),
        "total_return": round(total_return, 4),
        "max_drawdown": round(max_dd, 4),
        "beta_to_spy": round(beta, 4),
        "alpha": round(alpha, 4),
        "corr_to_spy": round(corr_to_spy, 4),
        "info_ratio": round(info_ratio, 3),
        "daily_win_rate": round(daily_win_rate, 4),
        "trade_win_rate": round(trade_win_rate, 4),
        "ret_on_spy_up_days": round(ret_on_spy_up, 4),
        "ret_on_spy_down_days": round(ret_on_spy_down, 4),
        "pct_positive_spy_up": round(pct_positive_on_spy_up, 4),
        "pct_positive_spy_down": round(pct_positive_on_spy_down, 4),
        "long_book_ann_return": round(long_ann, 4),
        "short_book_ann_return": round(short_ann, 4),
        "n_trades": len(trades),
        "n_trading_days": len(portfolio_returns),
    }


def _estimate_trade_pnls(trades: list[dict]) -> pd.Series:
    """Estimate per-trade P&L from trade log (rough: based on weight changes)."""
    if not trades:
        return pd.Series(dtype=float)
    pnls = []
    for trade in trades:
        # Rough estimate: positive weight change on winning signal is good
        if trade["side"] == "SELL" and trade.get("old_weight", 0) > 0:
            # Closing a long — mark as PnL based on signal direction
            pnls.append(trade.get("weight_change", 0))
        elif trade["side"] == "BUY" and trade.get("old_weight", 0) < 0:
            # Covering a short
            pnls.append(trade.get("weight_change", 0))
    return pd.Series(pnls) if pnls else pd.Series(dtype=float)


def walk_forward_validation(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    sectors: dict,
    n_windows: int = 3,
    volumes: pd.DataFrame | None = None,
    market_caps: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Walk-forward validation with rolling train/test windows.

    Uses TRAIN_YEARS for in-sample signal generation and TEST_YEARS for
    out-of-sample evaluation. Windows roll forward by TEST_YEARS each step.

    Args:
        closes: full aligned close prices
        returns: full aligned daily returns
        sectors: ticker -> sector mapping
        n_windows: number of OOS windows

    Returns:
        DataFrame with one row per window, columns for all metrics
    """
    spy_returns = returns["SPY"] if "SPY" in returns.columns else None

    total_days = len(closes)
    train_days = TRAIN_YEARS * TRADING_DAYS_PER_YEAR
    test_days = TEST_YEARS * TRADING_DAYS_PER_YEAR

    # Calculate window positions
    results = []
    for i in range(n_windows):
        test_end_idx = total_days - i * test_days
        test_start_idx = test_end_idx - test_days
        train_start_idx = test_start_idx - train_days

        if train_start_idx < 0:
            print(f"  Window {i + 1}: insufficient data, skipping")
            continue

        train_dates = closes.index[train_start_idx:test_start_idx]
        test_dates = closes.index[test_start_idx:test_end_idx]

        print(f"\n  Window {i + 1}/{n_windows}")
        print(f"    Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
        print(f"    Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")

        # Generate signals on training data
        train_closes = closes.loc[train_dates]
        train_returns = returns.loc[train_dates]

        print(f"    Generating IS signals...")
        # Skip fundamentals for walk-forward (too slow, and they're point-in-time)
        is_composites, is_ranks = generate_signals(
            train_closes, train_returns,
            fundamentals=(pd.Series(dtype=float), pd.Series(dtype=float), 0.0),
            sectors=sectors,
        )

        # In-sample backtest
        print(f"    Running IS backtest...")
        train_volumes = volumes.loc[train_dates] if volumes is not None and not volumes.empty else None
        is_result = backtest(
            train_closes, train_returns, is_composites, sectors,
            spy_returns=spy_returns.loc[train_dates] if spy_returns is not None else None,
            label=f"is_w{i + 1}",
            volumes=train_volumes,
            market_caps=market_caps,
        )

        # Generate signals for test period using train-period parameters
        # (the signal is computed fresh on test-period prices, but using the
        # same lookback windows — no re-fitting)
        full_dates = closes.index[train_start_idx:test_end_idx]
        full_closes = closes.loc[full_dates]
        full_returns = returns.loc[full_dates]

        print(f"    Generating OOS signals...")
        oos_composites, oos_ranks = generate_signals(
            full_closes, full_returns,
            fundamentals=(pd.Series(dtype=float), pd.Series(dtype=float), 0.0),
            sectors=sectors,
        )

        # Slice to test period only
        oos_composites_test = oos_composites.reindex(test_dates).ffill()
        test_returns = returns.loc[test_dates]
        test_closes = closes.loc[test_dates]

        print(f"    Running OOS backtest...")
        test_volumes = volumes.loc[test_dates] if volumes is not None and not volumes.empty else None
        oos_result = backtest(
            test_closes, test_returns, oos_composites_test, sectors,
            spy_returns=spy_returns.loc[test_dates] if spy_returns is not None else None,
            label=f"oos_w{i + 1}",
            volumes=test_volumes,
            market_caps=market_caps,
        )

        window_result = {
            "window": i + 1,
            "train_start": str(train_dates[0].date()),
            "train_end": str(train_dates[-1].date()),
            "test_start": str(test_dates[0].date()),
            "test_end": str(test_dates[-1].date()),
            "is_sharpe": is_result["metrics"]["sharpe"],
            "oos_gross_sharpe": oos_result["metrics"].get("gross_sharpe", 0),
            "oos_sharpe": oos_result["metrics"]["sharpe"],
            "is_return": is_result["metrics"]["ann_return"],
            "oos_return": oos_result["metrics"]["ann_return"],
            "oos_gross_return": oos_result["metrics"].get("gross_ann_return", 0),
            "oos_max_dd": oos_result["metrics"]["max_drawdown"],
            "oos_beta": oos_result["metrics"]["beta_to_spy"],
            "oos_corr_spy": oos_result["metrics"]["corr_to_spy"],
            "oos_win_rate": oos_result["metrics"]["daily_win_rate"],
            "oos_pct_pos_spy_up": oos_result["metrics"]["pct_positive_spy_up"],
            "oos_pct_pos_spy_down": oos_result["metrics"]["pct_positive_spy_down"],
            "oos_long_return": oos_result["metrics"]["long_book_ann_return"],
            "oos_short_return": oos_result["metrics"]["short_book_ann_return"],
            "oos_cost_bps": oos_result["metrics"].get("total_bps", 0),
        }
        results.append(window_result)

        # Print OOS summary
        m = oos_result["metrics"]
        print(f"\n    OOS Results:")
        print(f"      Gross Sharpe:     {m.get('gross_sharpe', 0):+.3f}")
        print(f"      Net Sharpe:       {m['sharpe']:+.3f}")
        print(f"      Gross Return:     {m.get('gross_ann_return', 0):+.2%}")
        print(f"      Net Return:       {m['ann_return']:+.2%}")
        print(f"      Cost drag (bps):  {m.get('total_bps', 0):.0f}")
        print(f"        Commission:     {m.get('commission_bps', 0):.0f}")
        print(f"        Slippage:       {m.get('slippage_bps', 0):.0f}")
        print(f"        Borrow:         {m.get('borrow_bps', 0):.0f}")
        print(f"        Impact:         {m.get('impact_bps', 0):.0f}")
        print(f"      Max Drawdown:     {m['max_drawdown']:+.2%}")
        print(f"      Beta to SPY:      {m['beta_to_spy']:+.4f}")
        print(f"      Corr to SPY:      {m['corr_to_spy']:+.4f}")
        print(f"      Win Rate:         {m['daily_win_rate']:.1%}")
        print(f"      % pos SPY up:     {m['pct_positive_spy_up']:.1%}")
        print(f"      % pos SPY down:   {m['pct_positive_spy_down']:.1%}")

    if not results:
        print("\n  ERROR: No walk-forward windows could be computed")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df.to_parquet(V2_RESULTS_DIR / "walk_forward_results.parquet", index=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  Mean OOS Sharpe:       {results_df['oos_sharpe'].mean():+.3f}")
    print(f"  Mean OOS Return:       {results_df['oos_return'].mean():+.2%}")
    print(f"  Mean OOS Max DD:       {results_df['oos_max_dd'].mean():+.2%}")
    print(f"  Mean OOS Beta:         {results_df['oos_beta'].mean():+.4f}")
    print(f"  Mean OOS SPY Corr:     {results_df['oos_corr_spy'].mean():+.4f}")
    print(f"  Mean OOS Win Rate:     {results_df['oos_win_rate'].mean():.1%}")

    # HARD STOP CHECKS
    failed = False
    for _, row in results_df.iterrows():
        w = int(row["window"])
        if row["oos_sharpe"] < 0.2:
            print(f"\n  *** STOP: Window {w} OOS Sharpe = {row['oos_sharpe']:.3f} < 0.2 ***")
            failed = True
        if abs(row["oos_corr_spy"]) > 0.3:
            print(f"\n  *** WARNING: Window {w} SPY correlation = {row['oos_corr_spy']:.3f} > 0.3 ***")
            print(f"      Beta neutrality may be compromised")

    if failed:
        print(f"\n  VALIDATION FAILED — do not proceed to paper trading")
    else:
        print(f"\n  VALIDATION PASSED — safe to proceed to paper trading")

    return results_df


def _empty_result(label: str) -> dict:
    return {
        "equity_curve": pd.Series(dtype=float),
        "long_curve": pd.Series(dtype=float),
        "short_curve": pd.Series(dtype=float),
        "portfolio_returns": pd.Series(dtype=float),
        "long_returns": pd.Series(dtype=float),
        "short_returns": pd.Series(dtype=float),
        "metrics": {
            "label": label, "sharpe": 0, "ann_return": 0, "ann_vol": 0,
            "total_return": 0, "max_drawdown": 0, "beta_to_spy": 0,
            "alpha": 0, "corr_to_spy": 0, "info_ratio": 0,
            "daily_win_rate": 0, "trade_win_rate": 0,
            "ret_on_spy_up_days": 0, "ret_on_spy_down_days": 0,
            "pct_positive_spy_up": 0, "pct_positive_spy_down": 0,
            "long_book_ann_return": 0, "short_book_ann_return": 0,
            "n_trades": 0, "n_trading_days": 0,
        },
        "trades": [],
        "weights": pd.DataFrame(),
    }


if __name__ == "__main__":
    from v2.data_pipeline import download_all, load_matrices
    from v2.universe import load_universe, screen_universe
    from v2.costs import fetch_market_caps

    print("\n" + "=" * 60)
    print("  V2 Backtester — Walk-Forward Validation")
    print("=" * 60 + "\n")

    # Load or build universe
    try:
        universe = load_universe()
    except FileNotFoundError:
        print("  Running universe screening first...")
        screen_universe(force_refresh=True)
        universe = load_universe()

    # Download data
    print("  Loading data...")
    try:
        closes, volumes, returns = load_matrices()
        print(f"  Loaded cached matrices: {closes.shape[1]} tickers × {closes.shape[0]} days")
    except FileNotFoundError:
        print("  Downloading data (this may take 15-20 minutes on first run)...")
        closes, volumes, returns = download_all()

    sectors = universe["sectors"]

    # Fetch market caps for borrow cost tiering
    print("  Loading market caps for cost model...")
    market_caps = fetch_market_caps(universe["stocks"])

    # Run walk-forward validation
    results_df = walk_forward_validation(
        closes, returns, sectors, n_windows=3,
        volumes=volumes, market_caps=market_caps,
    )

    if not results_df.empty:
        # Also run a full-sample backtest for the equity curve
        print(f"\n  Running full-sample backtest...")
        composites, ranks = generate_signals(
            closes, returns,
            fundamentals=(pd.Series(dtype=float), pd.Series(dtype=float), 0.0),
            sectors=sectors,
        )
        full_result = backtest(
            closes, returns, composites, sectors, label="full",
            volumes=volumes, market_caps=market_caps,
        )

        print(f"\n  Full-sample metrics:")
        for key, val in full_result["metrics"].items():
            if isinstance(val, float):
                print(f"    {key:<25} {val:+.4f}")
            else:
                print(f"    {key:<25} {val}")
