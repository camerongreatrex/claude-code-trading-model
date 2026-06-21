"""
data_pipeline.py — Downloads, cleans, and persists raw OHLCV for the full universe.

Outputs (data/raw/): {TICKER}.parquet, closes_matrix.parquet (T × N).
Universe design: each ticker has a distinct economic driver to keep avg pairwise corr < 0.35.
Includes survivorship-bias anchors (GE, INTC, VZ); START = 2015-01-01 covers multiple cycles.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# Universe: distinct economic drivers per ticker.
# Asset classes: equity_index, bond, commodity, sector_etf, stock.

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

    # --- Phase 5: Stress-uncorrelated diversifiers ---
    "BWX" : "bond",            # Intl Treasury ex-US; bear_stress corr 0.048
    "DBA" : "commodity",       # Agriculture ETF; ~0.05 corr to S&P
    "FXE" : "commodity",       # EUR/USD; bear_stress corr 0.048
    "FXY" : "commodity",       # JPY/USD; bear_stress corr -0.318 (negative hedge)

    # --- Phase 6: Managed futures (CTA trend-following) ---
    "DBMF": "commodity",       # DBi MF; bear_stress corr +0.102; inception May 2019
    "WTMF": "commodity",       # WisdomTree MF; bear_stress corr +0.138; from 2011

    # --- Phase 7: Screened additions ---
    "VGSH": "bond",            # Short-Term Treasury; avg corr 0.010, bear_stress -0.155
    "MUB" : "bond",            # Municipal Bonds; avg corr 0.244, bear_stress 0.361

    # --- Phase 8: Conditional vol hedge ---
    # VXZ held only during vol backwardation (VIX9D > VIX); mid-term futures have
    # ~60% less roll decay than VIXY. Override applied in signal_generation.py.
    "VXZ" : "commodity",       # iPath VIX Mid-Term Futures ETN

    # --- Phase 9: universe expansion (2026-05) — distinct macro drivers ---
    "EWT" : "equity_index",   # Taiwan — semiconductor supply chain (TSMC)
    "EWY" : "equity_index",   # South Korea — memory/export cycle
    "INDA": "equity_index",   # India — domestic consumption, RBI policy
    "HACK": "equity_index",   # Cybersecurity — secular spend, low cyclicality
    "XBI" : "equity_index",   # Biotech — FDA/pipeline, idiosyncratic
    "COPX": "commodity",       # Copper miners — electrification/industrial cycle
    "IYR" : "equity_index",   # US REITs — rental income (distinct from VNQ)
    "SLV" : "commodity",       # Silver — industrial + monetary demand
    "URA" : "commodity",       # Uranium — nuclear power demand
}

# Equity-like assets — used for momentum + mean reversion regime switching
EQUITY_LIKE = {"equity_index", "sector_etf", "stock"}

ASSET_CLASS = TICKERS
TICKER_LIST = list(TICKERS.keys())

# GICS sector-ETF mapping for beta-hedged pair trades (long stock, short sector ETF).
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

START = "2010-01-01"
END   = "2026-01-01"

DATA_DIR = Path("data/v1/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Min trading days to include a ticker; below threshold tickers are skipped.
MIN_TRADING_DAYS = 1000


def _is_fresh(path: Path, today: pd.Timestamp) -> bool:
    """Return True if the parquet exists and was last written today (cache hit)."""
    if not path.exists():
        return False
    mtime = pd.Timestamp.fromtimestamp(path.stat().st_mtime).normalize()
    return mtime >= today


def fetch_earnings_dates(ticker: str) -> list:
    """Fetch quarterly earnings dates from yfinance (limit=50). Returns ISO strings;
    empty list on failure. Stock-only."""
    try:
        tk = yf.Ticker(ticker)
        ed = tk.get_earnings_dates(limit=50)
        if ed is None or ed.empty:
            return []
        dates = pd.to_datetime(ed.index).tz_localize(None)
        # Filter to backtest window
        dates = dates[(dates >= START) & (dates <= END)]
        return sorted(d.strftime("%Y-%m-%d") for d in dates)
    except Exception:
        return []


def download(ticker: str) -> pd.DataFrame:
    """Download adjusted OHLCV from Yahoo Finance (auto_adjust=True for div/split adj).
    Returns DataFrame indexed by tz-naive date."""
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # yfinance quirk

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Quality filters: ffill gaps, drop non-positive prices, drop High<Low rows,
    dedupe dates (keep last), sort. Never interpolate (lookahead bias)."""
    df = df.ffill().dropna()
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df = df[df["High"] >= df["Low"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def main():
    """Download and clean all tickers; print summary + correlation matrix
    (target avg pairwise corr < 0.35)."""
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
        # phase 7 screened additions (universe_screen.py verified)
        "VGSH": "Short-Term Treasury ETF — near-zero avg corr (0.010), negative bear_stress corr (-0.155); cash substitute",
        "MUB" : "Municipal Bond ETF — state/local credit driver; avg corr 0.244, bear_stress corr 0.361",
        # phase 8 conditional vol hedge
        "VXZ" : "iPath S&P 500 VIX Mid-Term Futures ETN — conditional hedge; held only during vol backwardation (VIX9D > VIX); ~60% less roll decay than VIXY",
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

    # ── Earnings dates (stocks only) ──
    # Used by signal_generation.py to go flat 2d before / 1d after release.
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

    # Correlation matrix grouped by asset class
    returns = closes.pct_change().dropna()
    corr    = returns.corr().round(2)
    print("\nCorrelation matrix:")
    print(corr)

    # Avg pairwise correlation — lower is better
    n    = len(corr)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    avg_corr = corr.values[mask].mean()
    print(f"\nAverage pairwise correlation: {avg_corr:.3f}")
    print("Target: below 0.35 for genuine diversification")

    # ── Expanded universe (opt-in) ──
    # Batch-downloads ~115 S&P 500 stocks; cached by mtime == today.
    try:
        from v1.pipeline.universe_expansion import (
            USE_EXPANDED_UNIVERSE,
            get_expanded_universe,
            SECTOR_STOCKS,
            NEW_TICKERS,
            SECTOR_MAP,
        )
    except ImportError:
        USE_EXPANDED_UNIVERSE = False

    if USE_EXPANDED_UNIVERSE:
        _, expanded_asset_class, _ = get_expanded_universe()
        today = pd.Timestamp.today().normalize()

        print(f"\n{'─' * 72}")
        print(f"  Expanded universe  (USE_EXPANDED_UNIVERSE = True)")
        print(f"  New stocks to download: {len(NEW_TICKERS)}")
        print(f"{'─' * 72}")

        for sector_etf, sector_tickers in SECTOR_STOCKS.items():
            to_download = [
                t for t in sector_tickers
                if not _is_fresh(DATA_DIR / f"{t}.parquet", today)
            ]
            n_cached = len(sector_tickers) - len(to_download)

            if n_cached:
                print(f"\n  {sector_etf}: {n_cached} cached, downloading {len(to_download)}...")
            else:
                print(f"\n  {sector_etf}: downloading {len(to_download)} tickers...")

            if to_download:
                try:
                    batch = yf.download(
                        to_download,
                        start=START,
                        end=END,
                        auto_adjust=True,
                        progress=False,
                        group_by="ticker",
                    )
                except Exception as e:
                    print(f"    Batch download failed: {e}")
                    batch = None

                if batch is not None and not batch.empty:
                    for ticker in to_download:
                        try:
                            if len(to_download) == 1:
                                df_raw = batch
                            else:
                                lvl0 = batch.columns.get_level_values(0)
                                if ticker not in lvl0:
                                    print(f"    {ticker:<8} MISSING in batch response")
                                    continue
                                df_raw = batch[ticker].dropna(how="all")

                            if isinstance(df_raw.columns, pd.MultiIndex):
                                df_raw.columns = df_raw.columns.droplevel(1)

                            df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]]
                            df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)
                            df_raw.index.name = "Date"
                            df_raw = clean(df_raw)

                            if len(df_raw) < MIN_TRADING_DAYS:
                                print(f"    {ticker:<8} SKIP  {len(df_raw)} rows < {MIN_TRADING_DAYS}")
                                continue

                            (DATA_DIR / f"{ticker}.parquet").parent.mkdir(parents=True, exist_ok=True)
                            df_raw.to_parquet(DATA_DIR / f"{ticker}.parquet",
                                              engine="pyarrow", compression="snappy")
                            all_data[ticker] = df_raw
                            print(f"    {ticker:<8} {len(df_raw)} rows  ({sector_etf})")

                        except Exception as e:
                            print(f"    {ticker:<8} ERROR: {e}")

            # Load cached tickers into all_data
            for ticker in sector_tickers:
                if ticker not in all_data:
                    path = DATA_DIR / f"{ticker}.parquet"
                    if path.exists():
                        try:
                            all_data[ticker] = pd.read_parquet(path)
                        except Exception:
                            pass

        # Expanded-only closes; dropna(how='all') preserves full 2015+ history.
        # Core closes_matrix.parquet is NOT rebuilt here — stays core-only.
        exp_closes = pd.DataFrame(
            {t: all_data[t]["Close"] for t in NEW_TICKERS if t in all_data}
        ).dropna(how="all")
        exp_closes.to_parquet(DATA_DIR / "closes_matrix_expanded.parquet")
        print(f"\ncore  closes_matrix.parquet    : {closes.shape}  "
              f"({closes.index[0].date()} to {closes.index[-1].date()})")
        print(f"expanded closes_matrix_expanded: {exp_closes.shape}  "
              f"({exp_closes.index[0].date()} to {exp_closes.index[-1].date()})")

        # Append earnings dates for new stocks (merges existing file)
        new_stock_tickers = [
            t for t in NEW_TICKERS
            if t in all_data and expanded_asset_class.get(t) == "stock"
        ]
        if new_stock_tickers:
            earn_path = DATA_DIR / "earnings_dates.json"
            if earn_path.exists():
                with open(earn_path) as fh:
                    earnings_map = json.load(fh)
            else:
                earnings_map = {}

            print(f"\nFetching earnings dates for {len(new_stock_tickers)} new stocks...")
            for t in new_stock_tickers:
                dates = fetch_earnings_dates(t)
                earnings_map[t] = dates
                print(f"  {t:<8} {len(dates):>3} dates  "
                      f"({dates[0] if dates else 'n/a'} — {dates[-1] if dates else 'n/a'})")
            with open(earn_path, "w") as fh:
                json.dump(earnings_map, fh, indent=2)
            print(f"\nEarnings dates updated -> {earn_path}")

        # ── Validation gate ──
        ref_start     = pd.Timestamp("2015-01-01")
        ref_days      = len(pd.bdate_range(ref_start, pd.Timestamp.today()))
        missing_dl    : list[str] = []
        low_coverage  : list[str] = []
        sector_counts : dict[str, int] = {}

        for ticker in NEW_TICKERS:
            path = DATA_DIR / f"{ticker}.parquet"
            if not path.exists():
                missing_dl.append(ticker)
                continue
            df_t = all_data.get(ticker)
            if df_t is None:
                missing_dl.append(ticker)
                continue
            in_window = df_t.loc[ref_start:]
            cov = len(in_window) / ref_days
            if cov < 0.90:
                low_coverage.append(f"{ticker}({cov:.0%})")
            sec = SECTOR_MAP.get(ticker)
            if sec:
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

        print(f"\n{'─' * 72}")
        print(f"  Validation gate — expanded universe")
        print(f"{'─' * 72}")
        print(f"  {'Sector':<6}  {'Stocks':>6}")
        for sec in sorted(sector_counts):
            print(f"  {sec:<6}  {sector_counts[sec]:>6}")

        n_ok = len(NEW_TICKERS) - len(missing_dl) - len(low_coverage)
        print(f"\n  Tickers with >= 90% coverage since 2015: {n_ok}/{len(NEW_TICKERS)}")
        if missing_dl:
            print(f"  Missing downloads ({len(missing_dl)}): {', '.join(missing_dl)}")
        if low_coverage:
            print(f"  Low coverage (<90%): {', '.join(low_coverage)}")
        if not missing_dl and not low_coverage:
            print(f"  All {len(NEW_TICKERS)} new tickers downloaded successfully")
        print(f"  Total universe: {len(all_data)} tickers  "
              f"(core: {len(TICKER_LIST)}, expansion: {len(NEW_TICKERS)})")
        print(f"{'─' * 72}")


if __name__ == "__main__":
    main()