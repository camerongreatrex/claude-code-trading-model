"""
v2/universe.py
--------------
Cross-asset ETF universe for macro regime rotation.

~30 ETFs spanning equities, fixed income, commodities, currencies, and real
assets.  Selected for: (1) sufficient history (inception pre-2008 preferred),
(2) high AUM / tight spreads, (3) minimal overlap within each asset class.

Output
──────
  data/v2/universe.parquet  — ticker, asset_class, sub_class, description, inception
  data/v2/results/etf_prices.parquet — adjusted close prices for all ETFs
  Prints summary table on run.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path("data/v2")
DATA_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("data/v2/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── ETF Universe Definition ──────────────────────────────────────────────────
# Each tuple: (ticker, asset_class, sub_class, description, approx_inception)

UNIVERSE = [
    # ── US Equities ──────────────────────────────────────────────────────────
    ("SPY",  "equity",    "us_large",     "S&P 500",                          "1993-01-29"),
    ("QQQ",  "equity",    "us_tech",      "Nasdaq 100",                       "1999-03-10"),
    ("IWM",  "equity",    "us_small",     "Russell 2000 Small Cap",           "2000-05-22"),
    ("IWD",  "equity",    "us_value",     "Russell 1000 Value",               "2000-05-22"),
    ("IWF",  "equity",    "us_growth",    "Russell 1000 Growth",              "2000-05-22"),

    # ── International Equities ───────────────────────────────────────────────
    ("EFA",  "equity",    "intl_dev",     "MSCI EAFE (Developed ex-US)",      "2001-08-14"),
    ("EEM",  "equity",    "intl_em",      "MSCI Emerging Markets",            "2003-04-07"),

    # ── US Fixed Income ──────────────────────────────────────────────────────
    ("SHY",  "fixed_income", "us_short",  "1-3Y Treasuries (cash proxy)",     "2002-07-22"),
    ("IEF",  "fixed_income", "us_mid",    "7-10Y Treasuries",                 "2002-07-22"),
    ("TLT",  "fixed_income", "us_long",   "20+Y Treasuries",                  "2002-07-22"),
    ("TIP",  "fixed_income", "us_tips",   "TIPS (inflation-linked bonds)",    "2003-12-04"),
    ("LQD",  "fixed_income", "us_ig",     "Investment Grade Corporate Bonds", "2002-07-22"),
    ("HYG",  "fixed_income", "us_hy",     "High Yield Corporate Bonds",       "2007-04-04"),

    # ── Commodities ──────────────────────────────────────────────────────────
    ("GLD",  "commodity",  "gold",         "Gold",                             "2004-11-18"),
    ("SLV",  "commodity",  "silver",       "Silver",                           "2006-04-28"),
    ("DBC",  "commodity",  "broad",        "Broad Commodities (energy/metals/ag)", "2006-02-03"),
    ("USO",  "commodity",  "oil",          "Crude Oil",                        "2006-04-10"),

    # ── Real Assets ──────────────────────────────────────────────────────────
    ("VNQ",  "real_asset", "us_reit",      "US REITs",                         "2004-09-23"),
    ("VNQI", "real_asset", "intl_reit",    "International REITs",              "2010-11-01"),

    # ── Currencies / Dollar ──────────────────────────────────────────────────
    ("UUP",  "currency",  "usd_long",     "US Dollar Bullish (DXY proxy)",    "2007-02-20"),
    ("FXE",  "currency",  "eur",          "Euro",                             "2005-12-09"),
    ("FXY",  "currency",  "jpy",          "Japanese Yen",                     "2007-02-12"),

    # ── Sector Tilts (for regime-specific overweight) ────────────────────────
    ("XLE",  "sector",    "energy",       "Energy Select SPDR",               "1998-12-16"),
    ("XLU",  "sector",    "utilities",    "Utilities Select SPDR",            "1998-12-16"),
    ("XLK",  "sector",    "technology",   "Technology Select SPDR",           "1998-12-16"),
    ("XLF",  "sector",    "financials",   "Financials Select SPDR",           "1998-12-16"),
    ("XLP",  "sector",    "staples",      "Consumer Staples Select SPDR",     "1998-12-16"),
    ("XLV",  "sector",    "healthcare",   "Healthcare Select SPDR",           "1998-12-16"),

    # ── Yield Enhancement (vol-selling proxy) ────────────────────────────────
    # PBP harvests the volatility risk premium by writing covered calls on the
    # S&P 500 — earns option premium in choppy/sideways markets at the cost of
    # capping upside in big rallies. Best in late_cycle/slowdown.
    ("PBP",  "yield_enh", "buywrite",     "S&P 500 BuyWrite Covered Call",    "2007-12-20"),

    # ── Equity Factor Sleeves ────────────────────────────────────────────────
    # Pair-trading-style factor exposures wrapped in long-only ETFs. MTUM
    # captures momentum (long winners), USMV captures the low-vol anomaly
    # (long low-beta names, structurally lower DD). Only available from
    # ~2013 — Sharpe allocator falls back to priors before then.
    ("MTUM", "factor",    "momentum",     "MSCI USA Momentum Factor",         "2013-04-18"),
    ("USMV", "factor",    "low_vol",      "MSCI USA Min-Vol Factor",          "2011-10-20"),
]


def get_universe() -> pd.DataFrame:
    """Return the ETF universe as a DataFrame."""
    df = pd.DataFrame(UNIVERSE, columns=["ticker", "asset_class", "sub_class",
                                          "description", "inception"])
    df["inception"] = pd.to_datetime(df["inception"])
    return df


def get_tickers() -> list[str]:
    """Return just the ticker list."""
    return [t[0] for t in UNIVERSE]


def get_asset_class_map() -> dict[str, str]:
    """Return ticker → asset_class mapping."""
    return {t[0]: t[1] for t in UNIVERSE}


def get_sub_class_map() -> dict[str, str]:
    """Return ticker → sub_class mapping."""
    return {t[0]: t[2] for t in UNIVERSE}


def download_prices(tickers: list[str], start: str = "1996-01-01") -> pd.DataFrame:
    """Download adjusted close prices for all ETFs and cache to parquet."""
    cache_path = RESULTS_DIR / "etf_prices.parquet"
    if cache_path.exists():
        prices = pd.read_parquet(cache_path)
        print(f"  Loaded cached prices: {prices.shape}")
        # Check if reasonably fresh (within 7 days)
        if (pd.Timestamp.today() - prices.index.max()).days < 7:
            return prices

    print(f"  Downloading prices for {len(tickers)} ETFs...")
    data = yf.download(tickers, start=start, progress=False, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    prices = prices.ffill()

    prices.to_parquet(cache_path)
    print(f"  Saved prices: {prices.shape} -> {cache_path}")
    return prices


def save_universe():
    """Save universe definition to parquet."""
    df = get_universe()
    path = DATA_DIR / "universe.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved universe: {len(df)} ETFs → {path}")
    return df


if __name__ == "__main__":
    df = save_universe()

    print(f"\n{'='*70}")
    print(f"  V2 Cross-Asset ETF Universe: {len(df)} instruments")
    print(f"{'='*70}")

    for ac in df["asset_class"].unique():
        subset = df[df["asset_class"] == ac]
        print(f"\n  {ac.upper()} ({len(subset)})")
        for _, row in subset.iterrows():
            print(f"    {row['ticker']:5s}  {row['description']:45s}  since {row['inception'].date()}")

    print(f"\n  Earliest inception: {df['inception'].min().date()}")
    print(f"  Latest inception:  {df['inception'].max().date()}")
    print(f"  Full overlap from: ~2011 (all ETFs trading)")

    # Download prices as part of universe setup
    print(f"\n  Downloading ETF prices...")
    tickers = get_tickers()
    download_prices(tickers)
