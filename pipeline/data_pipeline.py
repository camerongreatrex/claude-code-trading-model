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
The current mix covers:
  - Broad equity    : SPY, IWM, EEM, EFA, VWO
  - Rates/safe-haven: TLT, GLD
  - Credit/inflation: HYG (credit cycle), TIP (real rates)
  - Intl bonds      : BWX (ex-US govt, ECB/BOJ divergence from Fed)
  - Commodity basket: DBC (broad commodities), UUP (USD/FX)
  - Agriculture     : DBA (weather/crop supply — ~0 correlation to equities)
  - FX              : FXE (EUR/USD), FXY (JPY/USD safe-haven carry unwind)
  - Sector rotation : XLE, XLU, XLF, VNQ
  - Company drivers : JPM, JNJ, XOM, AMZN, NEE, BRK-B, GS, COST, MSFT, NVDA
  - Survivorship anchors: GE, INTC, VZ

Phase 4 additions (EFA, VWO, HYG, TIP, DBC, UUP, VNQ):
  - EFA/VWO: international diversification reduces US-equity-block concentration
  - HYG: credit cycle signal independent of TLT rate direction
  - TIP: real rate proxy (opposite to TLT nominal duration in inflation regimes)
  - DBC: broad commodity basket with oil/agriculture/metals diversification vs GLD
  - UUP: USD dollar index — inverse to EM/commodities, distinct FX driver
  - VNQ: REITs — rental income driver distinct from XLU utility regulated revenue

Phase 6 additions (DBMF, WTMF) — managed futures (structurally different return driver):
  - CTA trend-following across commodities, rates, FX, equities — zero overlap with MA equity signals
  - DBMF: bear_stress corr +0.102; $3.3B AUM; replicates top 20 CTA funds via Dynamic Beta Engine
  - WTMF: bear_stress corr +0.138; more conservative; lower max DD (-13.2%); history from 2011
  - Both classified "commodity" to get two-sided MA50/200 + commodity regime routing (correct for trend-followers)
  - CTA (inception 2022) skipped until 2027 — insufficient walk-forward OOS windows

Phase 5 additions (BWX, DBA, FXE, FXY) — verified stress diversifiers:
  - Selected after testing 8 candidates; 4 removed because bear_stress corr > 0.60
    (EWJ 0.854, EWZ 0.708, FXI 0.633, EMB 0.638 — false friends that co-move in crises)
  - BWX: bear_stress corr 0.048 — ECB/BOJ rate paths diverge from Fed during US stress
  - DBA: bear_stress corr 0.338 — agriculture supply/weather driver, below universe avg
  - FXE: bear_stress corr 0.048 — EUR safe-haven flows during USD stress episodes
  - FXY: bear_stress corr −0.318 — yen carry unwind is structurally INVERSE to US sell-offs

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

import json
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
    # --- Broad equity indices ---
    "SPY" : "equity_index",   # US large cap — the benchmark everything is measured against
    "IWM" : "equity_index",   # US small cap — different risk profile, outperforms in early cycle
    "EEM" : "equity_index",   # EM (China-heavy) — China policy, commodity prices, USD strength
    "EFA" : "equity_index",   # Developed international ex-US (MSCI EAFE) — EU/Japan/UK macro, USD weakness tailwind
    "VWO" : "equity_index",   # Broad EM (less China-concentrated than EEM) — broader EM macro, commodities

    # --- Rates / safe haven ---
    "TLT" : "bond",           # 20yr Treasury — nominal rate direction, inverse equities in risk-off

    # --- Credit and inflation bonds (distinct drivers from TLT) ---
    "HYG" : "bond",           # High yield corporate — credit cycle proxy, widens in risk-off (spread risk ≠ duration risk)
    "TIP" : "bond",           # TIPS inflation-linked — real rate proxy; rises when inflation > nominal rates

    # --- Commodities ---
    "GLD" : "commodity",      # Gold — fear hedge, real rate driven (rises when real rates fall)
    "DBC" : "commodity",      # Broad commodity basket (oil + agriculture + metals) — diversified supply/demand driver
    "UUP" : "commodity",      # US Dollar Index ETF — FX exposure; inversely correlated to EM and commodities

    # --- Sector ETFs (trading universe + beta-hedge vehicles) ---
    "XLE" : "sector_etf",     # Energy — oil supply/demand, NOT Fed policy
    "XLU" : "sector_etf",     # Utilities — defensive, rate sensitive, counter-cyclical
    "XLF" : "sector_etf",     # Financials — yield curve slope, credit cycle
    "VNQ" : "sector_etf",     # REITs — rental income driver distinct from XLU regulated utility revenue; rate sensitive
    "XLK" : "sector_etf",     # Technology — hedge for MSFT, NVDA, AMZN, INTC (GICS tech sector)
    "XLV" : "sector_etf",     # Healthcare — hedge for JNJ (GICS healthcare sector)
    "XLI" : "sector_etf",     # Industrials — hedge for GE (GICS industrials sector)
    "XLC" : "sector_etf",     # Communication services — hedge for VZ (GICS comm services)
    "XLP" : "sector_etf",     # Consumer staples — hedge for COST (GICS consumer staples)

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

    # --- Phase 5: Stress-uncorrelated assets (bear_stress hedge) ---
    # Pairwise correlation spikes from 0.197 (bull_calm) to 0.446 (bear_stress).
    # These assets have structurally different drivers that stay low-corr in stress.

    # International bonds — divergent rate regimes (bear_stress corr: BWX 0.048)
    "BWX" : "bond",            # International Treasury ex-US — ECB/BOJ rates diverge from Fed path;
                               # near-zero bear_stress correlation (0.048 vs universe avg 0.446)

    # Agriculture — genuinely zero financial-system correlation (bear_stress corr: DBA 0.338)
    "DBA" : "commodity",       # Broad agriculture ETF — weather, crop yields, supply disruption;
                               # published ~0.05 correlation to S&P 500 in ALL regimes including stress

    # Currency carry — structural hedges vs US equity sell-offs
    "FXE" : "commodity",       # Euro/USD — EUR strengthens on hawkish ECB divergence from Fed;
                               # bear_stress corr 0.048 (near zero)
    "FXY" : "commodity",       # Yen/USD — safe-haven carry unwind; NEGATIVE corr to SPY in stress
                               # (-0.318 bear_stress) — genuine portfolio hedge

    # --- Phase 6: Managed futures (structurally different return driver) ---
    # CTA trend-following across commodities, rates, currencies, equities.
    # Zero overlap with the existing MA-crossover equity signal — this is
    # a fundamentally different strategy embedded as an asset allocation.
    "DBMF": "commodity",       # iMGP DBi Managed Futures — replicates top 20 CTA hedge funds;
                               # bear_stress corr +0.102; inception May 2019 (~7 years)
    "WTMF": "commodity",       # WisdomTree Managed Futures — conservative CTA replication;
                               # bear_stress corr +0.138; lower DD (-13.2%) than DBMF; full history from 2011
}

# Equity-like assets that can use momentum + mean reversion regime switching
EQUITY_LIKE = {"equity_index", "sector_etf", "stock"}

# Asset class lookup consumed by signal_generation and portfolio
ASSET_CLASS = TICKERS
TICKER_LIST = list(TICKERS.keys())

# GICS sector-ETF mapping for beta-hedged pair trades.
# Each stock is paired with its sector ETF. Long stock + short sector ETF
# isolates idiosyncratic return (stock outperformance within its sector)
# from the sector/market beta. Factual GICS classification — not fitted.
HEDGE_MAP: dict[str, str] = {
    "MSFT" : "XLK",   # Technology (GICS 45)
    "NVDA" : "XLK",   # Technology (GICS 45)
    "AMZN" : "XLK",   # Technology / Consumer Discretionary — primarily treated as tech
    "INTC" : "XLK",   # Technology (GICS 45)
    "JPM"  : "XLF",   # Financials (GICS 40)
    "GS"   : "XLF",   # Financials (GICS 40)
    "JNJ"  : "XLV",   # Healthcare (GICS 35)
    "XOM"  : "XLE",   # Energy (GICS 10)
    "NEE"  : "XLU",   # Utilities (GICS 55)
    "GE"   : "XLI",   # Industrials (GICS 20)
    "VZ"   : "XLC",   # Communication Services (GICS 50)
    "COST" : "XLP",   # Consumer Staples (GICS 30)
    "BRK-B": "SPY",   # Conglomerate — no single sector ETF; use broad market
}

START = "2015-01-01"
END   = "2026-01-01"

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Minimum trading days required to include a ticker in the universe.
# Tickers below this threshold are skipped with a warning so a mid-period
# delisting or recent IPO degrades gracefully instead of crashing downstream.
MIN_TRADING_DAYS = 1000


def fetch_earnings_dates(ticker: str) -> list:
    """
    Fetch quarterly earnings announcement dates for a stock from yfinance.

    Uses yf.Ticker.get_earnings_dates(limit=50) which reliably returns ~50
    historical + upcoming quarterly dates (12+ years).  Dates are timezone-
    stripped and returned as ISO-format strings ("YYYY-MM-DD").

    Returns an empty list on failure — earnings filtering is optional and the
    strategy works without it.  Only called for asset_class == "stock".
    """
    try:
        tk = yf.Ticker(ticker)
        ed = tk.get_earnings_dates(limit=50)
        if ed is None or ed.empty:
            return []
        dates = pd.to_datetime(ed.index).tz_localize(None)
        # Filter to our backtest window plus a small buffer
        dates = dates[(dates >= START) & (dates <= END)]
        return sorted(d.strftime("%Y-%m-%d") for d in dates)
    except Exception:
        return []


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
        "EFA" : "developed international ex-US (MSCI EAFE): EU/Japan/UK macro, USD weakness tailwind",
        "VWO" : "broad EM (less China-heavy than EEM): EM rates, commodity exporters, USD",
        "TLT" : "10yr+ Treasury nominal rate direction",
        "HYG" : "high yield credit cycle — spread widens in risk-off; distinct from TLT duration risk",
        "TIP" : "TIPS real rate proxy — rises when inflation > nominal rates; distinct from TLT",
        "GLD" : "real rates, fear, USD weakness",
        "DBC" : "broad commodity basket (oil, agriculture, metals): diversified supply/demand",
        "UUP" : "US Dollar Index: FX exposure; inversely correlated to EM and commodities",
        "XLE" : "oil supply/demand, OPEC decisions",
        "XLU" : "defensive, regulated utility revenue",
        "XLF" : "yield curve slope, credit cycle",
        "VNQ" : "REITs: rental income + rate sensitivity; distinct driver from XLU utility revenue",
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
        # sector ETFs added as beta-hedge vehicles for pair trades
        "XLK" : "Technology sector ETF — hedge for MSFT, NVDA, AMZN, INTC (GICS 45)",
        "XLV" : "Healthcare sector ETF — hedge for JNJ (GICS 35)",
        "XLI" : "Industrials sector ETF — hedge for GE (GICS 20)",
        "XLC" : "Communication Services sector ETF — hedge for VZ (GICS 50)",
        "XLP" : "Consumer Staples sector ETF — hedge for COST (GICS 30)",
        # phase 5 stress-uncorrelated additions (bear_stress corr verified < 0.35)
        "BWX" : "International govt bonds ex-US — ECB/BOJ divergence from Fed; bear_stress corr 0.048",
        "DBA" : "Agriculture ETF — weather, crop supply; ~0.05 corr to S&P in all regimes",
        "FXE" : "Euro/USD — ECB/Fed divergence; bear_stress corr 0.048 (near zero)",
        "FXY" : "Yen/USD — carry unwind safe-haven; bear_stress corr −0.318 (negative hedge)",
        # phase 6 managed futures additions
        "DBMF": "iMGP DBi Managed Futures — replicates top 20 CTA hedge funds; bear_stress corr +0.102",
        "WTMF": "WisdomTree Managed Futures — conservative CTA replication; bear_stress corr +0.138; lower DD",
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

    # ── Earnings dates (stocks only) ───────────────────────────────────────────
    # Fetch quarterly earnings announcement dates for individual stocks.
    # Used by signal_generation.py to go flat 2 days before / 1 day after each
    # release — avoids the 3-8% binary coin-flip gap risk.
    # ETFs, bonds, and commodities are skipped (no single-date earnings event).
    stock_tickers = [t for t in all_data if ASSET_CLASS[t] == "stock"]
    earnings_map: dict[str, list] = {}
    print("\nFetching earnings dates (stocks only)...")
    for t in stock_tickers:
        dates = fetch_earnings_dates(t)
        earnings_map[t] = dates
        print(f"  {t:<8} {len(dates):>3} dates  "
              f"({dates[0] if dates else 'n/a'} — {dates[-1] if dates else 'n/a'})")
    earn_path = DATA_DIR / "earnings_dates.json"
    with open(earn_path, "w") as fh:
        json.dump(earnings_map, fh, indent=2)
    print(f"\nEarnings dates saved -> {earn_path}")

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