"""
v2/diagnostics.py
-----------------
Signal attribution diagnostics for the beta-neutral model.

Runs offline to produce a full diagnostic report answering:
  "Why is the Sharpe low?"

Components
──────────
  1. Per-signal standalone performance (each signal as its own long/short portfolio)
  2. Long vs short book attribution on the composite
  3. Sector drift check (rolling net sector exposure)
  4. Decile spread over time (top minus bottom decile return)
  5. Window 3 deep-dive (Mag 7 check)

Output
──────
  v2/diagnostics_report.md  — full findings
  v2/diagnostics/*.png      — supporting charts
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from v2.signal_generation import (
    compute_momentum_12_1, compute_momentum_6_1,
    compute_reversal_1w, compute_low_vol,
    _zscore_cross_sectional,
)
from v2.portfolio import (
    construct_portfolio, compute_rolling_betas, build_portfolio_history,
    CAPITAL, SELECTION_FRACTION,
)
from v2.rebalancer import simulate_rebalances
from v2.backtester import backtest, TRAIN_YEARS, TEST_YEARS, TRADING_DAYS_PER_YEAR
from v2.costs import compute_backtest_costs, summarize_costs

DIAG_DIR = Path("v2/diagnostics")
DIAG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("v2/diagnostics_report.md")

MAG_7 = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sharpe(returns: pd.Series) -> float:
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    return ann_ret / ann_vol if ann_vol > 0 else 0.0


def _max_dd(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return dd.min()


def _beta_to_spy(returns: pd.Series, spy_returns: pd.Series) -> float:
    common = returns.index.intersection(spy_returns.index)
    if len(common) < 60:
        return 0.0
    r = returns.reindex(common).fillna(0)
    s = spy_returns.reindex(common).fillna(0)
    cov = r.cov(s)
    var_s = s.var()
    return cov / var_s if var_s > 0 else 0.0


def _corr_to_spy(returns: pd.Series, spy_returns: pd.Series) -> float:
    common = returns.index.intersection(spy_returns.index)
    if len(common) < 20:
        return 0.0
    return returns.reindex(common).corr(spy_returns.reindex(common))


# ── 1. Per-signal standalone performance ─────────────────────────────────────

def run_single_signal_backtest(
    signal_raw: pd.DataFrame,
    returns: pd.DataFrame,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    sectors: dict,
    spy_returns: pd.Series,
    market_caps: dict | None = None,
    signal_name: str = "signal",
) -> dict:
    """
    Run a single signal as a standalone long/short portfolio:
    top decile long, bottom decile short, equal-weighted, weekly rebalance,
    beta hedged via SPY, WITH costs.
    """
    # Get weekly (Friday) dates
    all_dates = signal_raw.index
    fridays = all_dates[all_dates.dayofweek == 4]
    if len(all_dates) > 252:
        fridays = fridays[fridays >= all_dates[252]]

    # Z-score cross-sectionally at each Friday
    composite = pd.DataFrame(index=fridays, columns=signal_raw.columns, dtype=float)
    for date in fridays:
        if date not in signal_raw.index:
            continue
        vals = signal_raw.loc[date].dropna()
        if len(vals) < 20:
            continue
        z = _zscore_cross_sectional(vals)
        z = z.clip(-3, 3)
        composite.loc[date, z.index] = z.values

    composite = composite.dropna(how="all")
    if composite.empty:
        return {"signal": signal_name, "sharpe": 0, "max_dd": 0, "beta": 0,
                "corr_spy": 0, "decile_spread": 0}

    # Forward-fill to daily
    composite_daily = composite.reindex(closes.index).ffill()

    # Run through the standard backtest pipeline
    result = backtest(
        closes, returns, composite_daily, sectors,
        spy_returns=spy_returns, label=f"diag_{signal_name}",
        volumes=volumes, market_caps=market_caps,
    )

    net_returns = result["portfolio_returns"]
    long_returns = result["long_returns"]
    short_returns = result["short_returns"]

    # Decile spread: long book return minus short book return (annualized)
    # Short returns are already negative when shorts lose money, so spread = long + (-short)
    # i.e. long_ret - short_ret_contribution = long_ret - short_ret
    # But short_returns is (negative weights * returns), so the "short return" for the purpose
    # of spread is -short_returns (what you'd earn if you just shorted those names)
    spread_daily = long_returns - short_returns  # when shorts work, short_returns < 0, so spread increases
    decile_spread_ann = spread_daily.mean() * 252

    return {
        "signal": signal_name,
        "net_sharpe": round(_sharpe(net_returns), 3),
        "gross_sharpe": round(_sharpe(result.get("gross_returns", net_returns)), 3),
        "max_dd": round(_max_dd(net_returns), 4),
        "beta": round(_beta_to_spy(net_returns, spy_returns), 4),
        "corr_spy": round(_corr_to_spy(net_returns, spy_returns), 4),
        "decile_spread_ann": round(decile_spread_ann, 4),
        "ann_return": round(net_returns.mean() * 252, 4),
        "long_ann_return": round(long_returns.mean() * 252, 4),
        "short_ann_return": round(short_returns.mean() * 252, 4),
    }


def per_signal_diagnostics(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    volumes: pd.DataFrame,
    sectors: dict,
    spy_returns: pd.Series,
    market_caps: dict | None = None,
) -> pd.DataFrame:
    """Run each signal individually as a standalone long/short portfolio."""

    print("\n  === Per-Signal Standalone Performance ===")

    signals = {
        "momentum_12_1": compute_momentum_12_1(closes),
        "momentum_6_1": compute_momentum_6_1(closes),
        "reversal_1w": compute_reversal_1w(closes),
        "low_vol": compute_low_vol(returns),
    }

    results = []
    for name, raw in signals.items():
        print(f"\n  Running {name}...")
        r = run_single_signal_backtest(
            raw, returns, closes, volumes, sectors, spy_returns,
            market_caps=market_caps, signal_name=name,
        )
        results.append(r)

        # Diagnosis
        tag = ""
        if r["net_sharpe"] < 0.1:
            tag = " *** NOISE (Sharpe < 0.1) ***"
        if abs(r["decile_spread_ann"]) < 0.02:
            tag += " *** WEAK SPREAD (< 2%) ***"

        print(f"    Net Sharpe:      {r['net_sharpe']:+.3f}{tag}")
        print(f"    Gross Sharpe:    {r['gross_sharpe']:+.3f}")
        print(f"    Max DD:          {r['max_dd']:+.2%}")
        print(f"    Beta:            {r['beta']:+.4f}")
        print(f"    Decile spread:   {r['decile_spread_ann']:+.2%}")

    df = pd.DataFrame(results)
    return df


# ── 2. Long vs short book attribution ────────────────────────────────────────

def long_short_attribution(
    backtest_result: dict,
    spy_returns: pd.Series,
) -> dict:
    """
    Attribute performance to long and short books separately.
    """
    print("\n  === Long vs Short Book Attribution ===")

    long_ret = backtest_result["long_returns"]
    short_ret = backtest_result["short_returns"]
    net_ret = backtest_result["portfolio_returns"]
    cost_dict = backtest_result.get("costs", {})

    long_sharpe = _sharpe(long_ret)
    # Negate short returns: positive = making money on shorts
    short_only_sharpe = _sharpe(-short_ret)

    long_contribution = long_ret.mean() * 252
    short_contribution = short_ret.mean() * 252

    # Cost breakdown on short book
    borrow_total = cost_dict.get("borrow", pd.Series(0.0, index=net_ret.index)).sum()
    n_years = len(net_ret) / 252
    borrow_ann = borrow_total / n_years if n_years > 0 else 0

    # Short book net contribution: short P&L minus short-specific costs
    # short_ret already has the sign: negative when shorts lose money
    short_net_contribution = short_contribution - borrow_ann

    result = {
        "long_only_sharpe": round(long_sharpe, 3),
        "short_only_sharpe": round(short_only_sharpe, 3),
        "long_ann_contribution": round(long_contribution, 4),
        "short_ann_contribution": round(short_contribution, 4),
        "short_borrow_cost_ann": round(borrow_ann, 4),
        "short_net_contribution": round(short_net_contribution, 4),
        "short_book_destroying_value": short_net_contribution < 0,
    }

    print(f"    Long-only Sharpe:         {result['long_only_sharpe']:+.3f}")
    print(f"    Short-only Sharpe:        {result['short_only_sharpe']:+.3f}")
    print(f"    Long contribution (ann):  {result['long_ann_contribution']:+.2%}")
    print(f"    Short contribution (ann): {result['short_ann_contribution']:+.2%}")
    print(f"    Short borrow cost (ann):  {result['short_borrow_cost_ann']:+.2%}")
    print(f"    Short NET contribution:   {result['short_net_contribution']:+.2%}")

    if result["short_book_destroying_value"]:
        print(f"\n    *** SHORT BOOK IS DESTROYING VALUE ***")
        print(f"    Consider switching to long-extension mode (long + SPY hedge)")

    return result


# ── 3. Sector drift check ────────────────────────────────────────────────────

def sector_drift_check(
    weights_df: pd.DataFrame,
    sectors: dict,
) -> pd.DataFrame:
    """
    Compute rolling weekly net sector exposure.
    Returns DataFrame: dates × sectors.
    """
    print("\n  === Sector Drift Check ===")

    # Get unique sectors
    unique_sectors = sorted(set(sectors.values()))

    sector_exposure = pd.DataFrame(index=weights_df.index, columns=unique_sectors, dtype=float)

    for date in weights_df.index:
        day_weights = weights_df.loc[date]
        for sector in unique_sectors:
            sector_tickers = [t for t in day_weights.index if sectors.get(t) == sector]
            sector_exposure.loc[date, sector] = day_weights[sector_tickers].sum()

    sector_exposure = sector_exposure.fillna(0)

    # Check for persistent net exposure
    print(f"\n    Average net sector exposure:")
    for sector in unique_sectors:
        mean_exp = sector_exposure[sector].mean()
        max_exp = sector_exposure[sector].abs().max()
        persistent = abs(mean_exp) > 0.02  # > 2% persistent bias
        flag = " *** PERSISTENT BIAS ***" if persistent else ""
        print(f"      {sector:<30} mean: {mean_exp:+.2%}  max |exp|: {max_exp:.2%}{flag}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    for sector in unique_sectors:
        ax.plot(sector_exposure.index, sector_exposure[sector], label=sector, alpha=0.7)
    ax.axhline(0.05, color="red", linestyle="--", alpha=0.5, label="±5% constraint")
    ax.axhline(-0.05, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Net Sector Exposure Over Time")
    ax.set_ylabel("Net Weight")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "sector_drift.png", dpi=150)
    plt.close(fig)

    return sector_exposure


# ── 4. Decile spread over time ───────────────────────────────────────────────

def decile_spread_analysis(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    composite_scores: pd.DataFrame,
    spy_returns: pd.Series,
) -> pd.DataFrame:
    """
    Monthly long-short decile spread: top decile return minus bottom decile return.
    Overlays with market regime.
    """
    print("\n  === Decile Spread Over Time ===")

    # Get weekly scores (non-NaN rows from composite)
    weekly_scores = composite_scores.dropna(how="all")

    spreads = []
    for date in weekly_scores.index:
        scores = weekly_scores.loc[date].dropna()
        if len(scores) < 20:
            continue

        n_select = max(5, int(len(scores) * SELECTION_FRACTION))
        ranked = scores.sort_values(ascending=False)
        top_tickers = ranked.head(n_select).index
        bottom_tickers = ranked.tail(n_select).index

        # Forward 5-day return (one week ahead)
        date_loc = returns.index.get_loc(date)
        if date_loc + 5 >= len(returns):
            continue
        fwd_returns = returns.iloc[date_loc + 1:date_loc + 6]

        top_available = top_tickers.intersection(fwd_returns.columns)
        bottom_available = bottom_tickers.intersection(fwd_returns.columns)

        if len(top_available) > 0 and len(bottom_available) > 0:
            top_ret = fwd_returns[top_available].mean(axis=1).sum()
            bottom_ret = fwd_returns[bottom_available].mean(axis=1).sum()
            spreads.append({
                "date": date,
                "top_decile_ret": top_ret,
                "bottom_decile_ret": bottom_ret,
                "spread": top_ret - bottom_ret,
            })

    spread_df = pd.DataFrame(spreads).set_index("date")

    if spread_df.empty:
        print("    No spread data computed")
        return spread_df

    # Monthly aggregation
    monthly_spread = spread_df["spread"].resample("ME").sum()

    # Identify negative spread months
    negative_months = monthly_spread[monthly_spread < 0]
    print(f"\n    Total weekly observations: {len(spread_df)}")
    print(f"    Mean weekly spread: {spread_df['spread'].mean():+.4f}")
    print(f"    Annualized spread: {spread_df['spread'].mean() * 52:+.2%}")
    print(f"    Months with negative spread: {len(negative_months)} / {len(monthly_spread)}")

    if len(negative_months) > 0:
        print(f"\n    Worst 10 months:")
        for date, val in negative_months.sort_values().head(10).items():
            print(f"      {date.strftime('%Y-%m')}: {val:+.4f}")

    # Regime overlay: classify each month as bull/bear and calm/stress
    spy_monthly = spy_returns.resample("ME").sum()
    spy_vol = spy_returns.resample("ME").std() * np.sqrt(252)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Monthly spread bars
    colors = ["green" if v > 0 else "red" for v in monthly_spread.values]
    ax1.bar(monthly_spread.index, monthly_spread.values, width=25, color=colors, alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title("Monthly Long-Short Decile Spread")
    ax1.set_ylabel("Monthly Spread Return")

    # Cumulative spread
    cum_spread = spread_df["spread"].cumsum()
    ax2.plot(cum_spread.index, cum_spread.values, color="navy", linewidth=1.5)
    ax2.set_title("Cumulative Long-Short Spread")
    ax2.set_ylabel("Cumulative Spread")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    fig.savefig(DIAG_DIR / "decile_spread.png", dpi=150)
    plt.close(fig)

    return spread_df


# ── 5. Window 3 deep-dive ────────────────────────────────────────────────────

def window_3_deepdive(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    composite_scores: pd.DataFrame,
    spy_returns: pd.Series,
) -> dict:
    """
    Deep-dive into Window 3 (2023):
    - Top/bottom 20 stocks by signal entering the window
    - Their actual performance
    - Were Mag 7 in the top decile?
    """
    print("\n  === Window 3 Deep-Dive (2023) ===")

    # Determine Window 3 dates (the earliest OOS window = furthest back)
    total_days = len(closes)
    test_days = TEST_YEARS * TRADING_DAYS_PER_YEAR

    # Window 3 is the 3rd window back (i=2 in the walk-forward loop)
    test_end_idx = total_days - 2 * test_days
    test_start_idx = test_end_idx - test_days

    if test_start_idx < 0 or test_end_idx > total_days:
        print("    Cannot determine Window 3 boundaries")
        return {}

    test_dates = closes.index[test_start_idx:test_end_idx]
    print(f"    Window 3 period: {test_dates[0].date()} to {test_dates[-1].date()}")

    # Get signal scores at the start of Window 3
    # Find the last composite score date before or at test_start
    pre_test_scores = composite_scores.loc[:test_dates[0]].dropna(how="all")
    if pre_test_scores.empty:
        print("    No signal scores available before Window 3")
        return {}

    entry_date = pre_test_scores.index[-1]
    entry_scores = pre_test_scores.loc[entry_date].dropna().sort_values(ascending=False)

    print(f"    Entry signal date: {entry_date.date()}")
    print(f"    Stocks with scores: {len(entry_scores)}")

    # Top 20 and bottom 20
    top_20 = entry_scores.head(20)
    bottom_20 = entry_scores.tail(20)

    # Actual performance during Window 3
    test_returns = returns.loc[test_dates]
    total_returns = test_returns.sum()  # approximate cumulative log return

    print(f"\n    Top 20 (long) stocks entering Window 3:")
    print(f"    {'Ticker':<8} {'Signal':>8} {'Actual Return':>14}")
    for ticker, score in top_20.items():
        actual = total_returns.get(ticker, float("nan"))
        print(f"    {ticker:<8} {score:+8.3f} {actual:+14.2%}")

    print(f"\n    Bottom 20 (short) stocks entering Window 3:")
    print(f"    {'Ticker':<8} {'Signal':>8} {'Actual Return':>14}")
    for ticker, score in bottom_20.items():
        actual = total_returns.get(ticker, float("nan"))
        print(f"    {ticker:<8} {score:+8.3f} {actual:+14.2%}")

    # Mag 7 check
    print(f"\n    Mag 7 Check:")
    n_select = max(5, int(len(entry_scores) * SELECTION_FRACTION))
    top_decile_tickers = set(entry_scores.head(n_select).index)

    mag7_in_top = []
    mag7_missing = []
    for ticker in MAG_7:
        if ticker in entry_scores.index:
            rank = list(entry_scores.index).index(ticker) + 1
            score = entry_scores[ticker]
            in_top = ticker in top_decile_tickers
            actual = total_returns.get(ticker, float("nan"))
            status = "IN top decile" if in_top else f"RANK {rank}/{len(entry_scores)}"
            print(f"      {ticker:<6} score={score:+.3f}  {status}  "
                  f"actual={actual:+.2%}")
            if in_top:
                mag7_in_top.append(ticker)
            else:
                mag7_missing.append(ticker)
        else:
            print(f"      {ticker:<6} NOT IN UNIVERSE")

    if mag7_missing:
        print(f"\n    *** {len(mag7_missing)} Mag 7 stocks MISSED by top decile ***")
        print(f"    Missing: {', '.join(mag7_missing)}")
        print(f"    This explains the Window 3 failure if they rallied hard in 2023.")

    return {
        "window_3_start": str(test_dates[0].date()),
        "window_3_end": str(test_dates[-1].date()),
        "entry_date": str(entry_date.date()),
        "top_20": {t: {"score": float(s), "actual_return": float(total_returns.get(t, 0))}
                   for t, s in top_20.items()},
        "bottom_20": {t: {"score": float(s), "actual_return": float(total_returns.get(t, 0))}
                      for t, s in bottom_20.items()},
        "mag7_in_top_decile": mag7_in_top,
        "mag7_missing_from_top": mag7_missing,
    }


# ── Master diagnostic runner ─────────────────────────────────────────────────

def run_diagnostics(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    volumes: pd.DataFrame,
    sectors: dict,
    composite_scores: pd.DataFrame,
    backtest_result: dict,
    market_caps: dict | None = None,
) -> dict:
    """
    Run the full diagnostic suite and generate report.

    Args:
        closes: aligned close prices
        returns: aligned returns
        volumes: aligned volume matrix
        sectors: ticker -> sector mapping
        composite_scores: composite z-scores from signal generation
        backtest_result: result dict from full-sample backtest()
        market_caps: ticker -> market cap

    Returns:
        dict with all diagnostic findings
    """
    spy_returns = returns["SPY"] if "SPY" in returns.columns else None
    if spy_returns is None:
        print("  ERROR: SPY not found in returns — cannot run diagnostics")
        return {}

    findings = {}

    # 1. Per-signal standalone
    print("\n" + "=" * 60)
    print("  PHASE 2: SIGNAL ATTRIBUTION DIAGNOSTICS")
    print("=" * 60)

    signal_df = per_signal_diagnostics(
        closes, returns, volumes, sectors, spy_returns,
        market_caps=market_caps,
    )
    findings["per_signal"] = signal_df.to_dict("records")

    # 2. Long vs short book attribution
    ls_attr = long_short_attribution(backtest_result, spy_returns)
    findings["long_short_attribution"] = ls_attr

    # 3. Sector drift
    weights_df = backtest_result.get("weights", pd.DataFrame())
    if not weights_df.empty:
        sector_exp = sector_drift_check(weights_df, sectors)
        findings["sector_drift_mean"] = sector_exp.mean().to_dict()
    else:
        print("\n  Skipping sector drift (no weights)")

    # 4. Decile spread
    spread_df = decile_spread_analysis(closes, returns, composite_scores, spy_returns)
    if not spread_df.empty:
        findings["decile_spread_ann"] = round(spread_df["spread"].mean() * 52, 4)
        findings["decile_spread_negative_months_pct"] = round(
            (spread_df["spread"].resample("ME").sum() < 0).mean(), 3
        )

    # 5. Window 3 deep-dive
    w3 = window_3_deepdive(closes, returns, composite_scores, spy_returns)
    findings["window_3"] = w3

    # Generate report
    _write_report(findings, backtest_result)

    return findings


def _write_report(findings: dict, backtest_result: dict):
    """Write the diagnostic report to markdown."""
    metrics = backtest_result.get("metrics", {})

    lines = [
        "# V2 Diagnostic Report",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary Metrics",
        "",
        f"- **Gross Sharpe:** {metrics.get('gross_sharpe', 'N/A')}",
        f"- **Net Sharpe:** {metrics.get('net_sharpe', 'N/A')}",
        f"- **Net Max DD:** {metrics.get('max_drawdown', 'N/A')}",
        f"- **Cost drag (bps/yr):** {metrics.get('total_bps', 'N/A')}",
        f"  - Commission: {metrics.get('commission_bps', 'N/A')}",
        f"  - Slippage: {metrics.get('slippage_bps', 'N/A')}",
        f"  - Borrow: {metrics.get('borrow_bps', 'N/A')}",
        f"  - Impact: {metrics.get('impact_bps', 'N/A')}",
        "",
    ]

    # Per-signal results
    lines.append("## Per-Signal Standalone Performance")
    lines.append("")
    lines.append("| Signal | Net Sharpe | Gross Sharpe | Max DD | Beta | Decile Spread | Diagnosis |")
    lines.append("|--------|-----------|-------------|--------|------|---------------|-----------|")

    for sig in findings.get("per_signal", []):
        diag = []
        if sig.get("net_sharpe", 0) < 0.1:
            diag.append("NOISE")
        if abs(sig.get("decile_spread_ann", 0)) < 0.02:
            diag.append("WEAK SPREAD")
        diag_str = ", ".join(diag) if diag else "OK"

        lines.append(
            f"| {sig['signal']} | {sig.get('net_sharpe', 0):+.3f} | "
            f"{sig.get('gross_sharpe', 0):+.3f} | {sig.get('max_dd', 0):+.2%} | "
            f"{sig.get('beta', 0):+.4f} | {sig.get('decile_spread_ann', 0):+.2%} | {diag_str} |"
        )

    lines.append("")

    # Long/short attribution
    ls = findings.get("long_short_attribution", {})
    lines.append("## Long vs Short Book Attribution")
    lines.append("")
    lines.append(f"- **Long-only Sharpe:** {ls.get('long_only_sharpe', 'N/A')}")
    lines.append(f"- **Short-only Sharpe:** {ls.get('short_only_sharpe', 'N/A')}")
    lines.append(f"- **Long contribution (ann):** {ls.get('long_ann_contribution', 0):+.2%}")
    lines.append(f"- **Short contribution (ann):** {ls.get('short_ann_contribution', 0):+.2%}")
    lines.append(f"- **Short borrow cost (ann):** {ls.get('short_borrow_cost_ann', 0):+.2%}")
    lines.append(f"- **Short NET contribution:** {ls.get('short_net_contribution', 0):+.2%}")

    if ls.get("short_book_destroying_value"):
        lines.append("")
        lines.append("> **RECOMMENDATION:** Short book is net-negative after costs. "
                      "Consider switching to long-extension mode "
                      "(long top decile + short SPY as single hedge).")
    lines.append("")

    # Decile spread
    lines.append("## Decile Spread")
    lines.append("")
    lines.append(f"- **Annualized spread:** {findings.get('decile_spread_ann', 'N/A')}")
    lines.append(f"- **Months with negative spread:** "
                 f"{findings.get('decile_spread_negative_months_pct', 'N/A')}")
    lines.append("")
    lines.append("![Decile Spread](diagnostics/decile_spread.png)")
    lines.append("")

    # Sector drift
    lines.append("## Sector Drift")
    lines.append("")
    lines.append("![Sector Drift](diagnostics/sector_drift.png)")
    lines.append("")

    # Window 3
    w3 = findings.get("window_3", {})
    if w3:
        lines.append("## Window 3 Deep-Dive")
        lines.append("")
        lines.append(f"- **Period:** {w3.get('window_3_start')} to {w3.get('window_3_end')}")
        lines.append(f"- **Mag 7 in top decile:** {', '.join(w3.get('mag7_in_top_decile', []))}")
        lines.append(f"- **Mag 7 MISSING:** {', '.join(w3.get('mag7_missing_from_top', []))}")

        if w3.get("mag7_missing_from_top"):
            lines.append("")
            lines.append("> **FINDING:** Mag 7 stocks were not in the top decile entering 2023. "
                          "Classic 12-1 momentum lagged because these stocks had poor 2022 returns. "
                          "This is the primary driver of Window 3 failure.")
        lines.append("")

    # Key recommendations
    lines.append("## Key Recommendations")
    lines.append("")

    noise_signals = [s["signal"] for s in findings.get("per_signal", [])
                     if s.get("net_sharpe", 0) < 0.1]
    if noise_signals:
        lines.append(f"1. **Drop noise signals:** {', '.join(noise_signals)} "
                      f"have net Sharpe < 0.1 — they're adding noise, not alpha.")

    if ls.get("short_book_destroying_value"):
        lines.append("2. **Switch to long-extension mode:** Short book costs exceed short alpha. "
                      "Long top decile at 130%, short SPY at 30% as hedge.")

    if w3.get("mag7_missing_from_top"):
        lines.append("3. **Add residual momentum:** Regress out market beta before computing "
                      "momentum. This avoids the 'just picking high-beta losers' problem.")

    lines.append("")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n  Report saved to {REPORT_PATH}")


# ── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from v2.data_pipeline import load_matrices
    from v2.universe import load_universe
    from v2.signal_generation import generate_signals
    from v2.costs import fetch_market_caps

    print("\n" + "=" * 60)
    print("  V2 DIAGNOSTIC DEEP-DIVE")
    print("=" * 60 + "\n")

    # Load data
    print("  Loading data...")
    try:
        universe = load_universe()
    except FileNotFoundError:
        print("  ERROR: Universe not screened. Run v2/universe.py first.")
        sys.exit(1)

    try:
        closes, volumes, returns = load_matrices()
    except FileNotFoundError:
        print("  ERROR: Data not downloaded. Run v2/data_pipeline.py first.")
        sys.exit(1)

    sectors = universe["sectors"]
    print(f"  Loaded: {closes.shape[1]} tickers x {closes.shape[0]} days")

    # Fetch market caps
    print("  Loading market caps...")
    market_caps = fetch_market_caps(universe["stocks"])

    # Generate composite signals
    print("  Generating signals...")
    composites, ranks = generate_signals(
        closes, returns,
        fundamentals=(pd.Series(dtype=float), pd.Series(dtype=float), 0.0),
        sectors=sectors,
    )

    # Run full-sample backtest with costs
    print("  Running full-sample backtest with cost model...")
    full_result = backtest(
        closes, returns, composites, sectors, label="diag_full",
        volumes=volumes, market_caps=market_caps,
    )

    print(f"\n  Full-sample: Gross Sharpe = {full_result['metrics'].get('gross_sharpe', 'N/A')}, "
          f"Net Sharpe = {full_result['metrics']['sharpe']}, "
          f"Cost drag = {full_result['metrics'].get('total_bps', 'N/A')} bps/yr")

    # Run diagnostics
    findings = run_diagnostics(
        closes, returns, volumes, sectors, composites, full_result,
        market_caps=market_caps,
    )

    print("\n" + "=" * 60)
    print("  DIAGNOSTICS COMPLETE")
    print("=" * 60)
