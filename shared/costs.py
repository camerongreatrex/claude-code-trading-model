"""
v2/costs.py
-----------
Realistic transaction cost model for backtesting and paper trading.

Cost components
───────────────
  1. Commissions: $0.005/share, $1 minimum per trade (IB Pro)
  2. Spread/slippage: 0.5 * bid-ask estimate; wider for low-ADV names
  3. Short borrow: tiered by market cap (25/50/200 bps annual)
  4. Market impact: sqrt-style impact scaled by trade_size / ADV

All costs are applied symmetrically on entry and exit unless noted.
Both backtester and paper_trader import from here for consistency.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from functools import lru_cache

# ── Cache for market cap lookups ─────────────────────────────────────────────
_MCAP_CACHE_PATH = Path("data/v2/cache/market_caps.parquet")
_MCAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
_mcap_cache: dict[str, float] = {}


# ── Commission costs ─────────────────────────────────────────────────────────
COMMISSION_PER_SHARE = 0.005      # $0.005 per share
COMMISSION_MINIMUM = 1.00         # $1 minimum per trade


def commission_cost(shares: int, price: float) -> float:
    """
    IB Pro tier commission: $0.005/share, $1 minimum per trade.
    Applied on each buy and each sell.
    """
    cost = max(shares * COMMISSION_PER_SHARE, COMMISSION_MINIMUM)
    return cost


def commission_cost_pct(shares: int, price: float) -> float:
    """Commission as a fraction of trade notional."""
    notional = shares * price
    if notional <= 0:
        return 0.0
    return commission_cost(shares, price) / notional


# ── Spread / slippage ────────────────────────────────────────────────────────
# Slippage = 0.5 * estimated bid-ask spread
# Liquid S&P names: spread ≈ 0.0005 * price (1 cent on a $20 stock)
# Low-ADV names (< $100M): spread ≈ 0.001 * price

LOW_ADV_THRESHOLD = 100_000_000   # $100M average daily dollar volume


def slippage_cost(price: float, adv: float | None = None) -> float:
    """
    Estimated half-spread slippage per share.
    Applied on both entry and exit (round-trip = 2x this).

    Args:
        price: stock price
        adv: average daily dollar volume (trailing 20 days). If < $100M,
             wider spread assumed.
    """
    if adv is not None and adv < LOW_ADV_THRESHOLD:
        half_spread = 0.001 * price
    else:
        half_spread = 0.0005 * price
    return max(0.01, half_spread)


def slippage_cost_pct(price: float, adv: float | None = None) -> float:
    """Slippage as a fraction of price (one-way)."""
    return slippage_cost(price, adv) / price if price > 0 else 0.0


# ── Short borrow costs ──────────────────────────────────────────────────────
# Tiered by market cap as proxy for borrow difficulty:
#   > $50B  → 25 bps/yr  (mega-cap, easy to borrow)
#   $10-50B → 50 bps/yr  (large-cap, general collateral)
#   < $10B  → 200 bps/yr (mid-cap / high short interest)

BORROW_RATE_MEGA = 0.0025    # 25 bps annual
BORROW_RATE_LARGE = 0.0050   # 50 bps annual
BORROW_RATE_MID = 0.0200     # 200 bps annual

MCAP_MEGA_THRESHOLD = 50e9   # $50B
MCAP_LARGE_THRESHOLD = 10e9  # $10B


def _load_mcap_cache():
    """Load cached market caps from disk."""
    global _mcap_cache
    if _MCAP_CACHE_PATH.exists():
        df = pd.read_parquet(_MCAP_CACHE_PATH)
        _mcap_cache = dict(zip(df["ticker"], df["market_cap"]))


def _save_mcap_cache():
    """Persist market cap cache to disk."""
    if _mcap_cache:
        df = pd.DataFrame(list(_mcap_cache.items()), columns=["ticker", "market_cap"])
        df.to_parquet(_MCAP_CACHE_PATH, index=False)


def get_market_cap(ticker: str) -> float | None:
    """
    Get market cap for a ticker (cached).
    Returns None if lookup fails.
    """
    if not _mcap_cache:
        _load_mcap_cache()

    if ticker in _mcap_cache:
        return _mcap_cache[ticker]

    try:
        info = yf.Ticker(ticker).info
        mcap = info.get("marketCap")
        if mcap is not None:
            _mcap_cache[ticker] = float(mcap)
            return float(mcap)
    except Exception:
        pass

    return None


def fetch_market_caps(tickers: list[str]) -> dict[str, float]:
    """
    Batch-fetch and cache market caps for a list of tickers.
    Only fetches tickers not already cached.
    """
    if not _mcap_cache:
        _load_mcap_cache()

    missing = [t for t in tickers if t not in _mcap_cache]

    if missing:
        print(f"  Fetching market caps for {len(missing)} tickers...")
        for i, ticker in enumerate(missing):
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(missing)}...")
            try:
                info = yf.Ticker(ticker).info
                mcap = info.get("marketCap")
                if mcap is not None:
                    _mcap_cache[ticker] = float(mcap)
            except Exception:
                continue
        _save_mcap_cache()

    return {t: _mcap_cache[t] for t in tickers if t in _mcap_cache}


def borrow_rate(ticker: str, market_cap: float | None = None) -> float:
    """
    Annual borrow rate for shorting a stock, based on market cap tier.

    Returns annual rate as a decimal (e.g. 0.0025 for 25 bps).
    """
    if market_cap is None:
        market_cap = get_market_cap(ticker)

    if market_cap is None:
        return BORROW_RATE_LARGE  # default to GC if unknown

    if market_cap >= MCAP_MEGA_THRESHOLD:
        return BORROW_RATE_MEGA
    elif market_cap >= MCAP_LARGE_THRESHOLD:
        return BORROW_RATE_LARGE
    else:
        return BORROW_RATE_MID


def daily_borrow_cost(short_notional: float, annual_rate: float) -> float:
    """
    Daily borrow cost on a short position.
    Accrued daily: short_notional * (annual_rate / 252).
    """
    return short_notional * (annual_rate / 252)


# ── Market impact ────────────────────────────────────────────────────────────
# impact = 0.1 * (trade_size / ADV) * price, capped at 50 bps
IMPACT_COEFFICIENT = 0.1
IMPACT_CAP_BPS = 50  # 50 bps max


def market_impact_cost(trade_notional: float, adv: float, price: float) -> float:
    """
    Market impact cost per share for a given trade.

    impact = 0.1 * (trade_size / ADV) * price, capped at 50 bps of price.

    Args:
        trade_notional: dollar value of the trade
        adv: average daily dollar volume (trailing 20 days)
        price: stock price

    Returns:
        Impact cost in dollars (total, not per-share).
    """
    if adv <= 0 or price <= 0:
        return 0.0

    impact_per_share = IMPACT_COEFFICIENT * (trade_notional / adv) * price
    cap = (IMPACT_CAP_BPS / 10_000) * price
    impact_per_share = min(impact_per_share, cap)

    shares = trade_notional / price
    return impact_per_share * shares


def market_impact_pct(trade_notional: float, adv: float) -> float:
    """Market impact as a fraction of trade notional."""
    if adv <= 0 or trade_notional <= 0:
        return 0.0
    pct = IMPACT_COEFFICIENT * (trade_notional / adv)
    return min(pct, IMPACT_CAP_BPS / 10_000)


# ── Aggregate cost functions (used by backtester) ────────────────────────────

def total_trade_cost_pct(
    price: float,
    shares: int,
    adv: float | None = None,
    is_short: bool = False,
    market_cap: float | None = None,
) -> float:
    """
    Total one-way trading cost as a fraction of trade notional.
    Includes commission + slippage + market impact.
    (Borrow costs are accrued daily, not at trade time.)

    Args:
        price: stock price
        shares: number of shares traded
        adv: average daily dollar volume
        is_short: True if this is a short sale (no additional cost at trade time,
                  borrow is accrued daily)
        market_cap: for borrow rate lookup (unused here, borrow is daily)
    """
    notional = shares * price
    if notional <= 0:
        return 0.0

    # Commission
    comm = commission_cost_pct(shares, price)

    # Slippage
    slip = slippage_cost_pct(price, adv)

    # Market impact
    impact = market_impact_pct(notional, adv) if adv else 0.0

    return comm + slip + impact


def compute_backtest_costs(
    weights_df: pd.DataFrame,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    market_caps: dict[str, float] | None = None,
    capital: float = 100_000,
) -> dict[str, pd.Series]:
    """
    Compute all cost components for a backtest weight history.

    Returns dict with daily cost series:
        'commission': daily commission drag
        'slippage': daily slippage drag
        'impact': daily market impact drag
        'borrow': daily short borrow cost
        'total': sum of all costs
    All expressed as fractions of portfolio value (return drag).
    """
    daily_dates = weights_df.index
    common_tickers = weights_df.columns.intersection(closes.columns)

    # Compute ADV (20-day trailing average daily dollar volume)
    if not volumes.empty:
        dollar_volume = closes * volumes
        adv_20 = dollar_volume.rolling(20, min_periods=5).mean()
    else:
        adv_20 = pd.DataFrame(index=closes.index, columns=closes.columns, dtype=float)

    # Pre-compute borrow rates per ticker
    if market_caps is None:
        market_caps = {}
    borrow_rates = {}
    for ticker in common_tickers:
        mcap = market_caps.get(ticker)
        borrow_rates[ticker] = borrow_rate(ticker, mcap)

    # Daily weight changes → trade costs
    weight_changes = weights_df[common_tickers].diff().abs()
    # First row: treat as full initial trade
    weight_changes.iloc[0] = weights_df[common_tickers].iloc[0].abs()

    commission_costs = pd.Series(0.0, index=daily_dates)
    slippage_costs = pd.Series(0.0, index=daily_dates)
    impact_costs = pd.Series(0.0, index=daily_dates)
    borrow_costs = pd.Series(0.0, index=daily_dates)

    for date in daily_dates:
        if date not in closes.index:
            continue

        # ── Trade costs (only on days with weight changes) ──
        day_changes = weight_changes.loc[date]
        has_trades = day_changes.sum() > 1e-6

        if has_trades:
            day_commission = 0.0
            day_slippage = 0.0
            day_impact = 0.0

            for ticker in common_tickers:
                delta_w = day_changes.get(ticker, 0)
                if delta_w < 1e-6:
                    continue

                price = closes.loc[date, ticker] if ticker in closes.columns else 0
                if price <= 0 or np.isnan(price):
                    continue

                trade_notional = delta_w * capital
                shares = int(trade_notional / price) if price > 0 else 0
                if shares == 0:
                    continue

                # Commission as return drag
                day_commission += commission_cost(shares, price) / capital

                # Slippage
                adv_val = None
                if ticker in adv_20.columns and date in adv_20.index:
                    adv_val = adv_20.loc[date, ticker]
                    if pd.isna(adv_val):
                        adv_val = None
                day_slippage += slippage_cost(price, adv_val) * shares / capital

                # Market impact
                if adv_val and adv_val > 0:
                    day_impact += market_impact_cost(trade_notional, adv_val, price) / capital

            commission_costs.loc[date] = day_commission
            slippage_costs.loc[date] = day_slippage
            impact_costs.loc[date] = day_impact

        # ── Daily borrow cost on short book ──
        weights_today = weights_df.loc[date, common_tickers] if date in weights_df.index else pd.Series(dtype=float)
        short_weights = weights_today[weights_today < -1e-6]

        day_borrow = 0.0
        for ticker in short_weights.index:
            short_notional = abs(short_weights[ticker]) * capital
            annual_rate = borrow_rates.get(ticker, BORROW_RATE_LARGE)
            day_borrow += daily_borrow_cost(short_notional, annual_rate) / capital
        borrow_costs.loc[date] = day_borrow

    total = commission_costs + slippage_costs + impact_costs + borrow_costs

    return {
        "commission": commission_costs,
        "slippage": slippage_costs,
        "impact": impact_costs,
        "borrow": borrow_costs,
        "total": total,
    }


def summarize_costs(cost_dict: dict[str, pd.Series], n_years: float | None = None) -> dict:
    """
    Summarize cost components into annualized bps.

    Args:
        cost_dict: output from compute_backtest_costs
        n_years: number of years in sample. If None, estimated from series length.

    Returns:
        dict with annualized cost drag in bps for each component.
    """
    if n_years is None:
        n_days = len(cost_dict["total"])
        n_years = n_days / 252

    if n_years <= 0:
        return {k: 0.0 for k in ["commission_bps", "slippage_bps", "impact_bps",
                                   "borrow_bps", "total_bps"]}

    total_comm = cost_dict["commission"].sum()
    total_slip = cost_dict["slippage"].sum()
    total_impact = cost_dict["impact"].sum()
    total_borrow = cost_dict["borrow"].sum()
    total_all = cost_dict["total"].sum()

    return {
        "commission_bps": round(total_comm / n_years * 10_000, 1),
        "slippage_bps": round(total_slip / n_years * 10_000, 1),
        "impact_bps": round(total_impact / n_years * 10_000, 1),
        "borrow_bps": round(total_borrow / n_years * 10_000, 1),
        "total_bps": round(total_all / n_years * 10_000, 1),
    }
