"""
data_pipeline.py
----------------
Downloads, cleans, and persists raw OHLCV price data for the full universe.

Outputs (written to data/raw/)
──────────────────────────────
  {TICKER}.parquet          — daily OHLCV for each ticker
  closes_matrix.parquet     — aligned close prices for all tickers (T × N matrix)

Consumed by
───────────
  feature_engineering.py    — reads individual ticker parquets
  live_signals.py           — uses TICKER_LIST and ASSET_CLASS constants
  signal_generation.py      — uses ASSET_CLASS to route signal logic
  paper_trader.py           — uses TICKER_LIST for the trading universe

Universe design principle
─────────────────────────
Every ticker in the universe should have a DIFFERENT primary economic driver.
If two assets respond to the same macroeconomic shock, one is redundant.
The current mix covers: broad equity (SPY, IWM, EEM), rates / safe-haven
(TLT, GLD), sector rotation (XLE, XLU, XLF), and idiosyncratic company
drivers (JPM, JNJ, XOM, AMZN, NEE, BRK-B, GS, COST, MSFT, NVDA).

Survivorship bias
─────────────────
GE, INTC, and VZ are included as "survivorship-bias anchors".  A 2015
investor would have held these large-cap names.  Excluding them because
they underperformed would inflate backtest returns by ~2–4% p.a. — a
well-documented data-mining trap that overstates strategy performance.

Date range
──────────
START = 2015-01-01.  Chosen because:
  - Covers two full market cycles (2015 correction, 2018 sell-off, COVID, 2022 bear).
  - Enough data (≈2,500 days) for the MA200 and 252-day rolling windows to warm up.
  - Avoids the 2008–2009 crisis which would require modelling a regime not
    relevant to the current market structure.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# -------------------------------------------------------------------------
# Universe design principle:
#   Every entry should have a DIFFERENT primary economic driver.
#   If two assets respond to the same thing, one is redundant.
#
# Asset classes:
#   equity_index  — broad market exposure, benchmark
#   bond          — rate direction driven, often inverse equities
#   commodity     — supply/demand and fear driven
#   sector_etf    — sector rotation exposure
#   stock         — idiosyncratic alpha, individual company drivers
# -------------------------------------------------------------------------

TICKERS = {
    # --- Macro / index ---
    "SPY" : "equity_index",   # US large cap — the benchmark everything is measured against
    "IWM" : "equity_index",   # US small cap — different risk profile, outperforms in early cycle
    "EEM" : "equity_index",   # Emerging markets — China policy, commodity prices, USD strength

    # --- Rates / safe haven ---
    "TLT" : "bond",           # 20yr Treasury — rate direction, inverse equities in risk-off
    "GLD" : "commodity",      # Gold — fear hedge, real rate driven (rises when real rates fall)

    # --- Sector ETFs ---
    "XLE" : "sector_etf",     # Energy — oil supply/demand, NOT Fed policy
    "XLU" : "sector_etf",     # Utilities — defensive, rate sensitive, counter-cyclical
    "XLF" : "sector_etf",     # Financials — yield curve slope, credit cycle

    # --- Individual stocks: each with a distinct economic driver ---
    "JPM" : "stock",          # Financials — interest rate spreads, commercial banking, credit
    "JNJ" : "stock",          # Healthcare — non-cyclical, defensive, FDA/pipeline driven
    "XOM" : "stock",          # Energy — oil major, production decisions, capex cycles
    "AMZN": "stock",          # Cloud/consumer — AWS enterprise spend lags equity market 6-12mo
    "NEE" : "stock",          # Clean energy utilities — rate sensitive + energy policy driven
    "BRK-B": "stock",         # Conglomerate — insurance, railroads, consumer brands, near-uncorrelated
    "GS"  : "stock",          # Investment banking — M&A volumes, capital markets, trading revenue
    "COST": "stock",          # Consumer staples — defensive, membership model, recession resistant

    # --- High-growth tech: included because a 2015 investor would have considered both ---
    # MSFT was a Dow Jones component in 2015 ($45B revenue, #1 enterprise software).
    # NVDA was a $8B GPU/gaming co in 2015 — smaller but liquid and actively traded.
    # Excluding the decade's biggest winners is its own form of survivorship bias.
    "MSFT": "stock",          # Enterprise cloud — Azure, Office 365, AI; different customer from AMZN AWS
    "NVDA": "stock",          # GPU/AI silicon — data center + gaming; driven by AI compute demand

    # --- Survivorship-bias anchors: underperformers 2015-2025 ---
    # A 2015 investor would have held these large-cap names. Excluding them
    # would inflate backtest returns by omitting known losers (survivorship bias).
    "GE"  : "stock",          # Industrial conglomerate — power write-downs, breakup, secular decline
    "INTC": "stock",          # Semiconductor — lost process leadership to TSMC/AMD, share loss
    "VZ"  : "stock",          # Telecom — 5G capex drag, subscriber pressure, near-zero real return
}

# Equity-like assets that can use momentum + mean reversion regime switching
EQUITY_LIKE = {"equity_index", "sector_etf", "stock"}

# Asset class lookup consumed by signal_generation and portfolio
ASSET_CLASS = TICKERS
TICKER_LIST = list(TICKERS.keys())

START = "2015-01-01"
END   = "2026-01-01"

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Minimum trading days required to include a ticker in the universe.
# Tickers below this threshold are skipped with a warning so a mid-period
# delisting or recent IPO degrades gracefully instead of crashing downstream.
MIN_TRADING_DAYS = 1000


def download(ticker: str) -> pd.DataFrame:
    """
    Download adjusted OHLCV data for a single ticker from Yahoo Finance.

    Uses ``auto_adjust=True`` so prices are dividend- and split-adjusted —
    essential for long backtests where unadjusted prices would show
    artificial gaps on ex-dividend dates.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g., "SPY", "BRK-B").

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] indexed by
        timezone-naive date.  Timezone is stripped to avoid merge issues when
        combining tickers downloaded at different times.
    """
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # yfinance quirk — flatten column names

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)  # strip timezone to avoid merge issues
    df.index.name = "Date"
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality filters to raw OHLCV data.

    Filters applied (in order):
      1. Forward-fill NaN gaps (e.g., bank holidays where one exchange is
         closed).  Forward-fill preserves the most recent real price.
         Interpolation is deliberately NOT used — it would look forward and
         introduce look-ahead bias.
      2. Drop rows where any OHLC price is non-positive (bad feed data).
      3. Drop rows where High < Low (impossible candle — corrupt data).
      4. Remove duplicate dates, keeping the last occurrence.
      5. Sort chronologically.

    Args:
        df: Raw OHLCV DataFrame from download().

    Returns:
        Cleaned DataFrame.  May have fewer rows than the input if corrupt
        rows were removed.
    """
    df = df.ffill().dropna()  # forward-fill gaps — never interpolate (lookahead bias)
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df = df[df["High"] >= df["Low"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def main():
    """
    Download and clean all tickers, then print a summary report.

    Also computes and prints the cross-ticker correlation matrix so you
    can quickly verify that the universe is genuinely diversified (target:
    average pairwise correlation < 0.35).
    """
    all_data = {}
    print("Downloading and processing...\n")
    print(f"  {'Ticker':<8} {'Class':<16} {'Rows':<8} {'Driver'}")
    print("  " + "-" * 70)

    driver_notes = {
        "SPY" : "broad US equity market",
        "IWM" : "small cap, early cycle outperformance",
        "EEM" : "China/EM macro, USD strength, commodities",
        "TLT" : "10yr+ Treasury rate direction",
        "GLD" : "real rates, fear, USD weakness",
        "XLE" : "oil supply/demand, OPEC decisions",
        "XLU" : "defensive, regulated utility revenue",
        "XLF" : "yield curve slope, credit cycle",
        "JPM" : "NIM spread, commercial lending, credit quality",
        "JNJ" : "healthcare non-cyclical, FDA pipeline",
        "XOM" : "oil major capex, production volumes",
        "AMZN": "AWS enterprise cloud, consumer discretionary",
        "NEE" : "clean energy policy, PPA contracts, rates",
        "BRK-B":"insurance float, railroads, consumer brands",
        "GS"  : "M&A volumes, IPO market, trading revenue",
        "COST": "membership model, consumer staples, defensive",
        "MSFT": "Azure cloud share gain, Office 365, AI integration (OpenAI partnership)",
        "NVDA": "GPU data center, AI training/inference demand, gaming",
        # survivorship-bias anchors
        "GE"  : "industrial restructuring, power write-downs, long-term decline",
        "INTC": "process node lag vs TSMC/AMD, fab investment overhang",
        "VZ"  : "5G capex drag, subscriber pressure, near-zero real return",
    }

    for ticker in TICKER_LIST:
        df = download(ticker)
        df = clean(df)

        if len(df) < MIN_TRADING_DAYS:
            print(f"  {ticker:<8} {'SKIPPED':<16} {len(df):<8} rows < {MIN_TRADING_DAYS} minimum — excluded from universe")
            continue

        path = DATA_DIR / f"{ticker}.parquet"
        df.to_parquet(path, engine="pyarrow", compression="snappy")

        note = driver_notes.get(ticker, "")
        print(f"  {ticker:<8} {ASSET_CLASS[ticker]:<16} {len(df):<8} {note}")
        all_data[ticker] = df

    closes = pd.DataFrame({t: d["Close"] for t, d in all_data.items()}).dropna()
    closes.to_parquet(DATA_DIR / "closes_matrix.parquet")

    print(f"\nClose matrix: {closes.shape}  (trading days x tickers)")

    # print correlation matrix grouped by asset class so structure is visible
    returns = closes.pct_change().dropna()
    corr    = returns.corr().round(2)
    print("\nCorrelation matrix:")
    print(corr)

    # average pairwise correlation — lower is better
    n    = len(corr)   # use actual matrix size, not TICKER_LIST (some may be skipped)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    avg_corr = corr.values[mask].mean()
    print(f"\nAverage pairwise correlation: {avg_corr:.3f}")
    print("Target: below 0.35 for genuine diversification")


if __name__ == "__main__":
    main()