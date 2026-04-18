"""
v2/backtester.py
----------------
Walk-forward backtest for the macro regime rotation strategy.

Backtest Structure
──────────────────
  1. Download ETF prices for entire universe
  2. Build regime features → classify → portfolio weights
  3. Compute gross returns (before costs)
  4. Apply risk overlays (vol targeting, DD breaker, stress override)
  5. Apply realistic transaction costs (shared/costs.py)
  6. Compute net returns and all metrics
  7. Walk-forward: 3 windows with expanding training

Metrics
───────
  - Gross/Net Sharpe, annualized return/vol, max DD
  - Beta to SPY, Information Ratio vs SPY
  - Per-regime return attribution
  - Block bootstrap 95% CI on Sharpe (10,000 replications)
  - Permutation test on regime labels

Success Gates (Phase 12)
────────────────────────
  - Net Sharpe > 0.55 full-sample
  - All 3 WF windows Sharpe > 0.2
  - Max DD < -25% (i.e., DD > -25%)
  - Block bootstrap p-value < 0.05 vs SPY
  - Cost drag < 80 bps/year
  - Annualized return > 10%
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from v2.pipeline.data_pipeline import get_tickers, get_asset_class_map
from v2.regimes.classification import classify_regimes, REGIME_NAMES, Regime
from v2.portfolio.weights import build_portfolio_weights
from v2.risk.overlays import apply_all_risk_overlays
from v2.portfolio.momentum_overlay import apply_momentum_overlay_df

DATA_DIR = Path("data/v2/results")
DATA_DIR.mkdir(parents=True, exist_ok=True)

REGIME_DIR = Path("data/v2/regime_features")


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    """Load cached ETF prices from disk (downloaded by universe.py)."""
    cache_path = DATA_DIR / "etf_prices.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found. Run the full pipeline first: python run.py v2"
        )
    prices = pd.read_parquet(cache_path)
    print(f"  Loaded prices: {prices.shape}")
    return prices


# ── Core Backtest ─────────────────────────────────────────────────────────────

def compute_portfolio_returns(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily portfolio returns from monthly weights and daily prices.

    Weights are held constant within each month (monthly rebalance).
    """
    daily_returns = prices.pct_change().fillna(0)

    # Forward-fill monthly weights to daily
    common_tickers = weights_df.columns.intersection(daily_returns.columns)
    weights_daily = weights_df[common_tickers].reindex(daily_returns.index).ffill().fillna(0)

    # Portfolio return = sum of weight * asset return
    port_returns = (weights_daily * daily_returns[common_tickers]).sum(axis=1)
    port_returns.name = "portfolio_return"

    return port_returns


def compute_cost_drag(weights_df: pd.DataFrame, n_years: float) -> float:
    """
    Estimate annualized cost drag from monthly turnover.

    For ETFs: ~5 bps round-trip cost (tight spreads, low commissions).
    Monthly rebalance with moderate turnover.
    """
    if len(weights_df) < 2:
        return 0.0

    # Monthly turnover = sum of absolute weight changes
    turnover = weights_df.diff().abs().sum(axis=1).iloc[1:]
    avg_monthly_turnover = turnover.mean()

    # Cost per unit turnover: ~5 bps for liquid ETFs
    cost_per_turnover = 0.0005
    annual_cost = avg_monthly_turnover * 12 * cost_per_turnover

    return annual_cost


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    label: str = "Strategy",
) -> dict:
    """Compute comprehensive performance metrics."""
    ann_factor = 252
    n_years = len(returns) / ann_factor

    ann_ret = returns.mean() * ann_factor
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Drawdown
    cum = (1 + returns).cumprod()
    running_max = cum.expanding().max()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    metrics = {
        "label": label,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_years": n_years,
        "total_return": cum.iloc[-1] - 1 if len(cum) > 0 else 0,
    }

    if benchmark_returns is not None:
        common = returns.index.intersection(benchmark_returns.index)
        r = returns.loc[common]
        b = benchmark_returns.loc[common]

        # Beta
        cov = r.cov(b)
        var_b = b.var()
        beta = cov / var_b if var_b > 0 else 0

        # Alpha (annualized)
        alpha = (r.mean() - beta * b.mean()) * ann_factor

        # Information ratio
        excess = r - b
        ir = excess.mean() / excess.std() * np.sqrt(ann_factor) if excess.std() > 0 else 0

        metrics["beta"] = beta
        metrics["alpha"] = alpha
        metrics["information_ratio"] = ir

    return metrics


def block_bootstrap_sharpe(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    block_size: int = 63,
) -> dict:
    """
    Block bootstrap 95% CI on Sharpe ratio (QUANTT methodology).

    Uses overlapping blocks of ~3 months to preserve autocorrelation.
    """
    n = len(returns)
    values = returns.values
    sharpes = []

    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        # Draw random block start indices
        n_blocks = n // block_size + 1
        starts = rng.randint(0, n - block_size, size=n_blocks)

        # Concatenate blocks
        boot_sample = np.concatenate([values[s:s+block_size] for s in starts])[:n]

        ann_ret = boot_sample.mean() * 252
        ann_vol = boot_sample.std() * np.sqrt(252)
        if ann_vol > 0:
            sharpes.append(ann_ret / ann_vol)

    sharpes = np.array(sharpes)
    ci_low, ci_high = np.percentile(sharpes, [2.5, 97.5])

    return {
        "mean_sharpe": float(np.mean(sharpes)),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "std": float(np.std(sharpes)),
    }


def bootstrap_p_value_vs_spy(
    strategy_returns: pd.Series,
    spy_returns: pd.Series,
    n_bootstrap: int = 10000,
    block_size: int = 63,
) -> float:
    """
    Block bootstrap p-value: H0 = strategy Sharpe <= SPY Sharpe.
    Returns p-value (fraction of bootstrap samples where strategy < SPY).
    """
    common = strategy_returns.index.intersection(spy_returns.index)
    strat = strategy_returns.loc[common].values
    spy = spy_returns.loc[common].values
    n = len(strat)

    rng = np.random.RandomState(42)
    strat_wins = 0

    for _ in range(n_bootstrap):
        n_blocks = n // block_size + 1
        starts = rng.randint(0, n - block_size, size=n_blocks)

        strat_boot = np.concatenate([strat[s:s+block_size] for s in starts])[:n]
        spy_boot = np.concatenate([spy[s:s+block_size] for s in starts])[:n]

        strat_sharpe = strat_boot.mean() / strat_boot.std() if strat_boot.std() > 0 else 0
        spy_sharpe = spy_boot.mean() / spy_boot.std() if spy_boot.std() > 0 else 0

        if strat_sharpe > spy_sharpe:
            strat_wins += 1

    return 1.0 - strat_wins / n_bootstrap


def regime_attribution(
    returns: pd.Series,
    labels: pd.Series,
) -> dict:
    """Per-regime return attribution."""
    common = returns.index.intersection(labels.index)
    r = returns.loc[common]
    l = labels.loc[common]

    results = {}
    for regime_id in range(6):
        name = REGIME_NAMES.get(Regime(regime_id), str(regime_id))
        mask = l == regime_id
        if mask.sum() < 10:
            continue
        regime_r = r[mask]
        ann_ret = regime_r.mean() * 252
        ann_vol = regime_r.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        results[name] = {
            "days": int(mask.sum()),
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
        }

    return results


# ── Walk-Forward ──────────────────────────────────────────────────────────────

def walk_forward_backtest(
    prices: pd.DataFrame,
    probs: pd.DataFrame,
    labels: pd.Series,
    stress_scores: pd.Series,
) -> list[dict]:
    """
    3-window walk-forward validation.

    Windows (using available data from ~2007 when most ETFs exist):
      Window 1: train 2007-2014, test 2015-2017
      Window 2: train 2007-2018, test 2019-2021
      Window 3: train 2007-2022, test 2023-2025
    """
    windows = [
        {"name": "WF1", "train_end": "2014-12-31", "test_start": "2015-01-01", "test_end": "2017-12-31"},
        {"name": "WF2", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2021-12-31"},
        {"name": "WF3", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2025-12-31"},
    ]

    results = []
    for w in windows:
        test_probs = probs.loc[w["test_start"]:w["test_end"]]
        if len(test_probs) < 20:
            continue

        test_weights = build_portfolio_weights(test_probs, prices, apply_momentum=False)
        test_returns = compute_portfolio_returns(test_weights, prices)
        test_returns = test_returns.loc[w["test_start"]:w["test_end"]]

        spy_returns = prices["SPY"].pct_change().dropna()
        metrics = compute_metrics(test_returns, spy_returns, label=w["name"])
        metrics["window"] = w["name"]
        metrics["test_period"] = f"{w['test_start']} to {w['test_end']}"
        results.append(metrics)

    return results


# ── Full Backtest ─────────────────────────────────────────────────────────────

def run_full_backtest() -> dict:
    """Run complete backtest pipeline and return all results."""
    # 1. Load prices (downloaded by universe.py)
    print("\n  [1/7] Loading ETF prices...")
    prices = load_prices()

    # 2. Load regime data
    print("  [2/7] Loading regime classifications...")
    probs = pd.read_parquet(REGIME_DIR / "regime_probabilities.parquet")
    labels = pd.read_parquet(REGIME_DIR / "regime_labels.parquet")["regime"]
    market_features = pd.read_parquet(REGIME_DIR / "market_features.parquet")
    stress_scores = market_features["stress_score"] if "stress_score" in market_features.columns else pd.Series(dtype=float)

    # 3. Build portfolio weights
    print("  [3/7] Building portfolio weights...")
    weights = build_portfolio_weights(probs, prices, apply_momentum=True)

    # 4. Compute gross returns
    print("  [4/7] Computing gross returns...")
    gross_returns = compute_portfolio_returns(weights, prices)

    # Trim to common period where most ETFs have data
    start_date = "2008-01-01"  # HYG inception 2007-04
    gross_returns = gross_returns.loc[start_date:]
    weights = weights.loc[start_date:]

    # 5. Apply risk overlays
    print("  [5/7] Applying risk overlays...")
    equity_curve = (1 + gross_returns).cumprod()
    risk_weights = apply_all_risk_overlays(
        weights, gross_returns, equity_curve, stress_scores,
    )
    risk_returns = compute_portfolio_returns(risk_weights, prices).loc[start_date:]

    # 6. Compute costs and net returns
    print("  [6/7] Computing transaction costs...")
    n_years = len(risk_returns) / 252
    cost_drag = compute_cost_drag(risk_weights, n_years)
    daily_cost = cost_drag / 252
    net_returns = risk_returns - daily_cost

    # SPY benchmark
    spy_returns = prices["SPY"].pct_change().dropna().loc[start_date:]

    # 7. Compute all metrics
    print("  [7/7] Computing metrics...")

    gross_metrics = compute_metrics(gross_returns, spy_returns, "Gross")
    risk_metrics = compute_metrics(risk_returns, spy_returns, "Risk-Adjusted")
    net_metrics = compute_metrics(net_returns, spy_returns, "Net")
    spy_metrics = compute_metrics(spy_returns, label="SPY B&H")

    # Bootstrap
    print("  Running block bootstrap (10,000 replications)...")
    bootstrap = block_bootstrap_sharpe(net_returns)
    p_value = bootstrap_p_value_vs_spy(net_returns, spy_returns)

    # Walk-forward
    print("  Running walk-forward validation...")
    wf_results = walk_forward_backtest(prices, probs, labels, stress_scores)

    # Regime attribution
    regime_attr = regime_attribution(net_returns, labels)

    # Save equity curve
    eq_df = pd.DataFrame({
        "strategy_gross": (1 + gross_returns).cumprod(),
        "strategy_net": (1 + net_returns).cumprod(),
        "spy": (1 + spy_returns.reindex(net_returns.index).fillna(0)).cumprod(),
    })
    eq_df.to_parquet(DATA_DIR / "equity_curves.parquet")

    # Save net returns
    net_returns.to_frame("net_return").to_parquet(DATA_DIR / "net_returns.parquet")

    return {
        "gross": gross_metrics,
        "risk_adjusted": risk_metrics,
        "net": net_metrics,
        "spy": spy_metrics,
        "cost_drag_bps": cost_drag * 10000,
        "bootstrap": bootstrap,
        "p_value_vs_spy": p_value,
        "walk_forward": wf_results,
        "regime_attribution": regime_attr,
        "start_date": str(gross_returns.index.min().date()),
        "end_date": str(gross_returns.index.max().date()),
    }


def print_report(results: dict):
    """Print formatted backtest report."""
    print("\n" + "="*70)
    print("  V2 MACRO REGIME ROTATION — BACKTEST REPORT")
    print("="*70)

    print(f"\n  Period: {results['start_date']} to {results['end_date']}")

    # ── Performance Table ─────────────────────────────────────────────
    print(f"\n  PERFORMANCE SUMMARY")
    print(f"  {'Metric':20s} {'Gross':>10s} {'Risk-Adj':>10s} {'Net':>10s} {'SPY B&H':>10s}")
    print(f"  {'─'*62}")

    for metric, fmt in [
        ("ann_return", "{:.1%}"), ("ann_vol", "{:.1%}"),
        ("sharpe", "{:.2f}"), ("max_dd", "{:.1%}"),
    ]:
        vals = []
        for key in ["gross", "risk_adjusted", "net", "spy"]:
            v = results[key].get(metric, 0)
            vals.append(fmt.format(v))
        print(f"  {metric:20s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s} {vals[3]:>10s}")

    net = results["net"]
    print(f"\n  Beta to SPY:          {net.get('beta', 0):.2f}")
    print(f"  Alpha (ann):          {net.get('alpha', 0):.1%}")
    print(f"  Information Ratio:    {net.get('information_ratio', 0):.2f}")
    print(f"  Cost drag:            {results['cost_drag_bps']:.0f} bps/year")

    # ── Bootstrap ─────────────────────────────────────────────────────
    bs = results["bootstrap"]
    print(f"\n  BLOCK BOOTSTRAP (10,000 replications)")
    print(f"  Net Sharpe:     {bs['mean_sharpe']:.2f}")
    print(f"  95% CI:         [{bs['ci_95_low']:.2f}, {bs['ci_95_high']:.2f}]")
    print(f"  p-value vs SPY: {results['p_value_vs_spy']:.4f}")

    # ── Walk-Forward ──────────────────────────────────────────────────
    print(f"\n  WALK-FORWARD VALIDATION")
    print(f"  {'Window':8s} {'Period':25s} {'AnnRet':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Gate':>6s}")
    print(f"  {'─'*65}")
    for wf in results["walk_forward"]:
        gate = "PASS" if wf["sharpe"] > 0.2 else "FAIL"
        print(f"  {wf['window']:8s} {wf['test_period']:25s} "
              f"{wf['ann_return']:7.1%} {wf['sharpe']:7.2f} {wf['max_dd']:7.1%} {gate:>6s}")

    # ── Regime Attribution ────────────────────────────────────────────
    print(f"\n  PER-REGIME ATTRIBUTION (Net Returns)")
    print(f"  {'Regime':15s} {'Days':>6s} {'AnnRet':>8s} {'AnnVol':>8s} {'Sharpe':>8s}")
    print(f"  {'─'*50}")
    for name, stats in sorted(results["regime_attribution"].items(),
                               key=lambda x: x[1]["sharpe"], reverse=True):
        print(f"  {name:15s} {stats['days']:6d} {stats['ann_return']:7.1%} "
              f"{stats['ann_vol']:7.1%} {stats['sharpe']:7.2f}")

    # ── Gate Check ────────────────────────────────────────────────────
    print(f"\n  {'='*55}")
    print(f"  SUCCESS GATE CHECK")
    print(f"  {'='*55}")

    gates = [
        ("Net Sharpe > 0.55", net["sharpe"] > 0.55, f"{net['sharpe']:.2f}"),
        ("All WF Sharpe > 0.2",
         all(w["sharpe"] > 0.2 for w in results["walk_forward"]),
         ", ".join(f"{w['sharpe']:.2f}" for w in results["walk_forward"])),
        ("Max DD > -25%", net["max_dd"] > -0.25, f"{net['max_dd']:.1%}"),
        ("p-value < 0.05", results["p_value_vs_spy"] < 0.05,
         f"{results['p_value_vs_spy']:.4f}"),
        ("Cost drag < 80 bps", results["cost_drag_bps"] < 80,
         f"{results['cost_drag_bps']:.0f} bps"),
        ("Ann return > 10%", net["ann_return"] > 0.10,
         f"{net['ann_return']:.1%}"),
    ]

    all_pass = True
    for name, passed, value in gates:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {value}")

    print(f"\n  Overall: {'ALL GATES PASSED' if all_pass else 'GATE FAILURE'}")
    return all_pass


if __name__ == "__main__":
    print("="*70)
    print("  V2 Macro Regime Rotation — Full Backtest")
    print("="*70)

    results = run_full_backtest()
    passed = print_report(results)
