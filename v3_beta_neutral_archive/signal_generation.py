"""
v2/signal_generation.py
-----------------------
Cross-sectional signal generation for the beta-neutral model.

Signals are computed weekly across the full ~450 stock universe. All signals
are z-scored cross-sectionally (subtract mean, divide by std across all stocks
that week) so they're comparable.

Signal definitions
──────────────────
  momentum_12_1 : 12-month return skipping most recent month (Jegadeesh-Titman)
  momentum_6_1  : 6-month return skipping most recent month
  reversal_1w   : 5-day return, sign flipped (short-term mean reversion)
  earnings_yield: trailing EPS / price (value signal, higher = cheaper = long)
  roe           : return on equity (quality signal, higher = better = long)
  low_vol       : 60-day realized vol, sign flipped (low-vol anomaly)

Composite score
───────────────
  composite = 0.30 * momentum_12_1
            + 0.20 * momentum_6_1
            + 0.10 * reversal_1w
            + 0.15 * earnings_yield  (if available)
            + 0.10 * roe             (if available)
            + 0.15 * low_vol

If fundamental signals unavailable for >30% of universe, their weights are
redistributed proportionally across momentum and vol signals.

Output
──────
  data/v2/signals/composite_scores.parquet  — T × N matrix of composite z-scores
  data/v2/signals/signal_ranks.parquet      — T × N matrix of rank (1 = best long)
  data/v2/signals/signal_components.parquet  — individual signal components (latest)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

from config import V2_FEATURES

V2_SIGNAL_DIR = Path("data/v2/signals")
V2_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Signal weights ────────────────────────────────────────────────────────────
BASE_WEIGHTS = {
    "momentum_12_1": 0.30,
    "momentum_6_1": 0.20,
    "reversal_1w": 0.10,
    "earnings_yield": 0.15,
    "roe": 0.10,
    "low_vol": 0.15,
}

# Weights when fundamentals unavailable — redistribute to momentum/vol
NO_FUNDAMENTALS_WEIGHTS = {
    "momentum_12_1": 0.35,
    "momentum_6_1": 0.20,
    "reversal_1w": 0.15,
    "low_vol": 0.30,
}

# Weights with residual momentum enabled (redistributes from raw momentum)
RESIDUAL_MOM_WEIGHTS = {
    "residual_momentum": 0.25,
    "momentum_12_1": 0.15,
    "momentum_6_1": 0.15,
    "reversal_1w": 0.15,
    "low_vol": 0.30,
}

RESIDUAL_MOM_FUND_WEIGHTS = {
    "residual_momentum": 0.25,
    "momentum_12_1": 0.15,
    "momentum_6_1": 0.10,
    "reversal_1w": 0.10,
    "earnings_yield": 0.15,
    "roe": 0.10,
    "low_vol": 0.15,
}

# Regime-adjusted weights (bear_stress: boost reversal)
BEAR_STRESS_WEIGHTS = {
    "momentum_12_1": 0.20,
    "momentum_6_1": 0.15,
    "reversal_1w": 0.25,
    "earnings_yield": 0.15,
    "roe": 0.10,
    "low_vol": 0.15,
}

BEAR_STRESS_NO_FUND_WEIGHTS = {
    "momentum_12_1": 0.25,
    "momentum_6_1": 0.20,
    "reversal_1w": 0.30,
    "low_vol": 0.25,
}


def _zscore_cross_sectional(series: pd.Series) -> pd.Series:
    """Z-score a series cross-sectionally (across stocks at one point in time)."""
    mu = series.mean()
    sigma = series.std()
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def compute_momentum_12_1(closes: pd.DataFrame) -> pd.DataFrame:
    """
    12-month return skipping most recent month (standard Jegadesch-Titman momentum).
    lookback = 252 days, skip = 21 days.
    """
    total_return = closes.pct_change(252, fill_method=None)
    recent_return = closes.pct_change(21, fill_method=None)
    mom = total_return - recent_return
    return mom


def compute_momentum_6_1(closes: pd.DataFrame) -> pd.DataFrame:
    """6-month return skipping most recent month."""
    total_return = closes.pct_change(126, fill_method=None)
    recent_return = closes.pct_change(21, fill_method=None)
    mom = total_return - recent_return
    return mom


def compute_reversal_1w(closes: pd.DataFrame) -> pd.DataFrame:
    """5-day return, sign flipped (recent losers bounce)."""
    ret_5d = closes.pct_change(5, fill_method=None)
    return -ret_5d


def compute_low_vol(returns: pd.DataFrame) -> pd.DataFrame:
    """60-day realized volatility, sign flipped (low-vol anomaly)."""
    vol_60d = returns.rolling(60).std() * np.sqrt(252)
    return -vol_60d


def compute_residual_momentum(
    returns: pd.DataFrame,
    spy_returns: pd.Series,
    regression_window: int = 252,
    momentum_window: int = 252,
    skip_window: int = 21,
) -> pd.DataFrame:
    """
    Residual (idiosyncratic) momentum: regress each stock's daily returns on
    SPY over trailing `regression_window` days to get residuals, then compute
    12-1 momentum on the cumulative residual return.

    This isolates stock-specific momentum from market momentum, avoiding the
    "just picking high-beta losers" problem in post-bear-market recoveries.
    """
    tickers = [c for c in returns.columns if c != "SPY"]
    spy = spy_returns.reindex(returns.index).fillna(0)

    residual_mom = pd.DataFrame(index=returns.index, columns=tickers, dtype=float)

    for ticker in tickers:
        stock_ret = returns[ticker].fillna(0)

        # Rolling OLS residuals: r_i = alpha + beta * r_SPY + epsilon
        # Use vectorized rolling cov/var for speed
        rolling_cov = stock_ret.rolling(regression_window, min_periods=60).cov(spy)
        rolling_var = spy.rolling(regression_window, min_periods=60).var()
        rolling_beta = rolling_cov / rolling_var
        rolling_alpha = stock_ret.rolling(regression_window, min_periods=60).mean() - \
                        rolling_beta * spy.rolling(regression_window, min_periods=60).mean()

        # Residual = actual - predicted
        predicted = rolling_alpha + rolling_beta * spy
        residual = stock_ret - predicted

        # Cumulative residual return over momentum_window, skip recent skip_window
        cum_resid_total = residual.rolling(momentum_window, min_periods=momentum_window).sum()
        cum_resid_recent = residual.rolling(skip_window, min_periods=skip_window).sum()
        residual_mom[ticker] = cum_resid_total - cum_resid_recent

    return residual_mom


def _compute_momentum_factor_return(
    mom_12_1: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute weekly momentum factor return: long top-quintile mom, short bottom-quintile.
    Used to detect momentum crashes. Computed at weekly (Friday) frequency for speed,
    then forward-filled to daily.
    """
    common_dates = mom_12_1.index.intersection(returns.index)
    fridays = common_dates[common_dates.dayofweek == 4]
    fridays = fridays[fridays >= common_dates[min(252, len(common_dates) - 1)]]

    factor_returns = pd.Series(np.nan, index=returns.index)

    for date in fridays:
        mom_vals = mom_12_1.loc[date].dropna()
        if len(mom_vals) < 20:
            continue

        n_quintile = max(5, len(mom_vals) // 5)
        ranked = mom_vals.sort_values(ascending=False)
        top = ranked.head(n_quintile).index.intersection(returns.columns)
        bottom = ranked.tail(n_quintile).index.intersection(returns.columns)

        if len(top) > 0 and len(bottom) > 0:
            top_ret = returns.loc[date, top].mean()
            bottom_ret = returns.loc[date, bottom].mean()
            factor_returns.loc[date] = top_ret - bottom_ret

    return factor_returns.ffill().fillna(0)


def fetch_fundamentals(tickers: list) -> tuple[pd.Series, pd.Series, float]:
    """
    Fetch earnings yield and ROE from yfinance .info for all tickers.

    Returns:
        (earnings_yield_series, roe_series, coverage_pct)
        coverage_pct: fraction of tickers with valid data
    """
    ey_data = {}
    roe_data = {}

    print("  Fetching fundamental data...")
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(tickers)} tickers processed...")
        try:
            info = yf.Ticker(ticker).info
            eps = info.get("trailingEps")
            price = info.get("currentPrice") or info.get("previousClose")
            roe_val = info.get("returnOnEquity")

            if eps is not None and price is not None and price > 0:
                ey_data[ticker] = eps / price
            if roe_val is not None:
                roe_data[ticker] = roe_val
        except Exception:
            continue

    ey_series = pd.Series(ey_data)
    roe_series = pd.Series(roe_data)
    coverage = len(ey_data) / len(tickers) if tickers else 0

    print(f"  Fundamental coverage: {coverage:.1%} "
          f"(earnings_yield: {len(ey_data)}, roe: {len(roe_data)})")

    return ey_series, roe_series, coverage


def generate_signals(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    regime: str = "bull_calm",
    fundamentals: tuple | None = None,
    weekly_dates: pd.DatetimeIndex | None = None,
    sectors: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate cross-sectional composite signals for all stocks.

    All signals are z-scored cross-sectionally (across all stocks at each date).

    Args:
        closes: aligned close price matrix (T × N)
        returns: aligned daily log return matrix (T × N)
        regime: current market regime (affects signal weights)
        fundamentals: (earnings_yield, roe, coverage) tuple or None to fetch
        weekly_dates: specific dates to compute signals for
        sectors: dict ticker -> GICS sector (reserved for future use)

    Returns:
        (composite_scores, signal_ranks) — both T × N DataFrames
    """
    tickers = closes.columns.tolist()

    # Determine rebalance dates (Fridays)
    if weekly_dates is None:
        all_dates = closes.index
        weekly_dates = all_dates[all_dates.dayofweek == 4]  # Friday = 4
        if len(all_dates) > 252:
            weekly_dates = weekly_dates[weekly_dates >= all_dates[252]]

    # Compute raw signals (full time series)
    print("  Computing momentum signals...")
    mom_12_1_raw = compute_momentum_12_1(closes)
    mom_6_1_raw = compute_momentum_6_1(closes)
    rev_1w_raw = compute_reversal_1w(closes)
    low_vol_raw = compute_low_vol(returns)

    # Residual momentum (if enabled)
    use_residual_mom = V2_FEATURES.get("residual_momentum", False)
    resid_mom_raw = None
    if use_residual_mom:
        spy_ret = returns["SPY"] if "SPY" in returns.columns else None
        if spy_ret is not None:
            print("  Computing residual momentum...")
            resid_mom_raw = compute_residual_momentum(returns, spy_ret)
        else:
            print("  Warning: SPY not found, skipping residual momentum")
            use_residual_mom = False

    # Fetch or use provided fundamentals
    has_fundamentals = False
    if fundamentals is not None:
        ey_series, roe_series, coverage = fundamentals
        has_fundamentals = coverage >= 0.30
    else:
        try:
            ey_series, roe_series, coverage = fetch_fundamentals(tickers)
            has_fundamentals = coverage >= 0.30
        except Exception as e:
            print(f"  Warning: fundamental data fetch failed: {e}")
            has_fundamentals = False
            ey_series = pd.Series(dtype=float)
            roe_series = pd.Series(dtype=float)

    if not has_fundamentals:
        print("  Fundamental coverage < 30%, using momentum/vol weights only")

    # Select weight scheme
    is_bear_stress = regime == "bear_stress"
    if use_residual_mom:
        if has_fundamentals:
            weights = RESIDUAL_MOM_FUND_WEIGHTS
        else:
            weights = RESIDUAL_MOM_WEIGHTS
    elif has_fundamentals:
        weights = BEAR_STRESS_WEIGHTS if is_bear_stress else BASE_WEIGHTS
    else:
        weights = BEAR_STRESS_NO_FUND_WEIGHTS if is_bear_stress else NO_FUNDAMENTALS_WEIGHTS

    # Compute composite scores at each rebalance date
    composite_scores = pd.DataFrame(index=weekly_dates, columns=tickers, dtype=float)
    signal_ranks = pd.DataFrame(index=weekly_dates, columns=tickers, dtype=float)

    print(f"  Computing composite scores for {len(weekly_dates)} rebalance dates...")
    for date in weekly_dates:
        if date not in mom_12_1_raw.index:
            continue

        # Get signal values at this date
        signals = {}
        signals["momentum_12_1"] = mom_12_1_raw.loc[date].dropna()
        signals["momentum_6_1"] = mom_6_1_raw.loc[date].dropna()
        signals["reversal_1w"] = rev_1w_raw.loc[date].dropna()
        signals["low_vol"] = low_vol_raw.loc[date].dropna()

        if use_residual_mom and resid_mom_raw is not None and date in resid_mom_raw.index:
            signals["residual_momentum"] = resid_mom_raw.loc[date].dropna()

        if has_fundamentals:
            signals["earnings_yield"] = ey_series
            signals["roe"] = roe_series

        # Find common tickers with all required signals
        common = set(tickers)
        for key in weights:
            if key in signals:
                common &= set(signals[key].index)
        common = sorted(common)

        if len(common) < 20:
            continue

        # Z-score each signal cross-sectionally
        z_signals = {}
        for key in weights:
            if key in signals:
                raw = signals[key].reindex(common).astype(float)
                z_signals[key] = _zscore_cross_sectional(raw)

        # Composite = weighted sum of z-scores
        composite = pd.Series(0.0, index=common)
        for key, weight in weights.items():
            if key in z_signals:
                composite += weight * z_signals[key]

        # Winsorise at ±3 to prevent outlier dominance
        composite = composite.clip(-3, 3)

        composite_scores.loc[date, common] = composite.values
        signal_ranks.loc[date, common] = composite.rank(ascending=False)

    # Drop rows that are all NaN
    composite_scores = composite_scores.dropna(how="all")
    signal_ranks = signal_ranks.dropna(how="all")

    # Forward-fill to daily frequency for the backtester
    # (signals hold constant between rebalance dates)
    all_dates = closes.index
    composite_daily = composite_scores.reindex(all_dates).ffill()
    ranks_daily = signal_ranks.reindex(all_dates).ffill()

    # Save
    composite_daily.to_parquet(V2_SIGNAL_DIR / "composite_scores.parquet")
    ranks_daily.to_parquet(V2_SIGNAL_DIR / "signal_ranks.parquet")

    # Save latest signal components for dashboard
    if len(composite_scores) > 0:
        latest_date = composite_scores.index[-1]
        components = pd.DataFrame({
            "momentum_12_1": mom_12_1_raw.loc[latest_date] if latest_date in mom_12_1_raw.index else np.nan,
            "momentum_6_1": mom_6_1_raw.loc[latest_date] if latest_date in mom_6_1_raw.index else np.nan,
            "reversal_1w": rev_1w_raw.loc[latest_date] if latest_date in rev_1w_raw.index else np.nan,
            "low_vol": low_vol_raw.loc[latest_date] if latest_date in low_vol_raw.index else np.nan,
            "composite": composite_scores.iloc[-1],
        })
        if has_fundamentals:
            components["earnings_yield"] = ey_series
            components["roe"] = roe_series
        components.to_parquet(V2_SIGNAL_DIR / "signal_components.parquet")

    print(f"  Signals generated: {composite_daily.shape[1]} tickers × {composite_daily.shape[0]} days")
    print(f"  Rebalance dates: {len(composite_scores)}")

    return composite_daily, ranks_daily


def load_signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-computed signals from disk."""
    composites = pd.read_parquet(V2_SIGNAL_DIR / "composite_scores.parquet")
    ranks = pd.read_parquet(V2_SIGNAL_DIR / "signal_ranks.parquet")
    return composites, ranks


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from v2.data_pipeline import load_matrices

    print("\n" + "=" * 60)
    print("  V2 Signal Generation")
    print("=" * 60 + "\n")

    closes, volumes, returns = load_matrices()
    composites, ranks = generate_signals(closes, returns)

    print(f"\n  Signal matrix shape: {composites.shape}")
    latest = composites.iloc[-1].dropna().sort_values()
    print(f"\n  Top 10 longs (latest):")
    for ticker, score in latest.tail(10).iloc[::-1].items():
        print(f"    {ticker:<8} {score:+.3f}")
    print(f"\n  Top 10 shorts (latest):")
    for ticker, score in latest.head(10).items():
        print(f"    {ticker:<8} {score:+.3f}")
