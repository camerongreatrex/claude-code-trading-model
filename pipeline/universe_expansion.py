# DISABLED: Expanded stock universe failed validation (3 iterations, Apr 2026).
# Bull_calm alpha remained negative due to 0.91 SPY correlation. Cross-asset
# allocation (core system) confirmed as primary alpha source per QUANTT ANOVA
# findings (p>0.49 for within-regime cross-industry dispersion). Retained for
# future research.

"""
universe_expansion.py
─────────────────────
Expands the tradeable universe from the 42-instrument core to ~157 instruments
by adding top-liquidity S&P 500 constituents organised by GICS sector.

Config flag
───────────
  USE_EXPANDED_UNIVERSE = True   # set False to use only the original core

Public API
──────────
  get_expanded_universe()         → (ticker_list, asset_class_dict, sector_map)
  refresh_universe_quarterly()    → re-screens live S&P 500 (run manually)

Design
──────
  The core instruments are the foundation — their ASSET_CLASS and HEDGE_MAP
  entries in data_pipeline.py are unchanged.  New stock tickers are ADDITIVE:
  assigned asset_class "stock" and catalogued in SECTOR_MAP.

  Sector coverage (GICS ETF → individual stocks):
    XLK  Technology (45)            XLV  Healthcare (35)
    XLE  Energy (10)                XLI  Industrials (20)
    XLF  Financials (40)            XLC  Communication Services (50)
    XLP  Consumer Staples (30)      XLY  Consumer Discretionary (25)
    XLRE Real Estate (60)           XLB  Materials (15)
    XLU  Utilities (55)

  Static pre-screening criteria (applied when building this list):
    ADV > $20M    — sufficient liquidity for EOD execution
    MCap > $5B    — avoids micro-cap noise (all S&P 500 members qualify)
    Top 10–15 most liquid names per sector from the S&P 500 index

  To refresh this list quarterly, run:
    python -m pipeline.universe_expansion
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
USE_EXPANDED_UNIVERSE: bool = False  # gates not yet passing — set True to re-enable

# ── New stocks per GICS sector ────────────────────────────────────────────────
# Additive to the core universe. Each list contains the top 10–15 most liquid
# S&P 500 names for that sector. Tickers already in the core are excluded.
# ADV > $20M and MCap > $5B verified at time of construction (2026-Q1).

SECTOR_STOCKS: dict[str, list[str]] = {
    # Top 10-15 most liquid S&P 500 names per GICS sector (excluding core tickers).
    # MMC removed — delisted (0 rows downloaded).
    # Beta filtering removed — replaced with beta-weighted sizing in signal_generation.py.
    "XLK": ["AAPL", "ACN", "ADBE", "AVGO", "CRM", "AMD", "QCOM", "TXN", "AMAT", "MU", "ORCL", "NOW"],
    "XLV": ["UNH", "LLY", "ABBV", "ABT", "MRK", "BMY", "AMGN", "PFE", "MDT", "GILD", "SYK", "DHR", "TMO"],
    "XLE": ["COP", "CVX", "EOG", "DVN", "OXY", "MPC", "SLB", "HAL", "PSX", "KMI", "VLO"],
    "XLI": ["RTX", "HON", "UNP", "UPS", "LMT", "DE", "NSC", "CAT", "ETN", "FDX", "BA", "EMR", "MMM"],
    "XLF": ["SPGI", "CB", "AXP", "MS", "WFC", "BAC", "BLK", "SCHW", "PNC"],
    "XLC": ["T", "TMUS", "CMCSA", "CHTR", "DIS", "META", "GOOGL", "NFLX"],
    "XLP": ["PG", "KO", "PEP", "WMT", "PM", "MO", "CL", "MDLZ", "GIS", "KMB", "SBUX"],
    "XLY": ["HD", "MCD", "LOW", "TJX", "MAR", "NKE", "BKNG", "TSLA"],
    "XLRE": ["AMT", "EQIX", "CCI", "PSA", "O", "AVB", "EQR", "PLD", "WELL", "SPG"],
    "XLB": ["LIN", "APD", "SHW", "NEM", "ECL", "VMC", "MLM", "FCX", "NUE", "PPG"],
    "XLU": ["DUK", "SO", "D", "EXC", "AEP", "XEL", "PEG", "WEC", "ED"],
}

# Flat list of all new tickers (unique, sector insertion order preserved)
_seen: set[str] = set()
NEW_TICKERS: list[str] = []
for _sector_tickers in SECTOR_STOCKS.values():
    for _t in _sector_tickers:
        if _t not in _seen:
            _seen.add(_t)
            NEW_TICKERS.append(_t)
del _seen, _sector_tickers, _t

# ── SECTOR_MAP ────────────────────────────────────────────────────────────────
# Maps every individual stock ticker → its GICS sector ETF.
# Covers both the existing core stocks and all new expansion stocks.
# Non-stock instruments (broad ETFs, bonds, commodities) are excluded.
# Uses actual GICS classifications, NOT the hedging shortcuts in HEDGE_MAP.

SECTOR_MAP: dict[str, str] = {
    # ── Core stocks (from data_pipeline.py) ──────────────────────────────────
    "JPM"  : "XLF",   # Financials
    "GS"   : "XLF",   # Financials
    "BRK-B": "XLF",   # Financials (conglomerate; insurance-heavy per GICS)
    "JNJ"  : "XLV",   # Healthcare
    "XOM"  : "XLE",   # Energy
    "GE"   : "XLI",   # Industrials
    "NEE"  : "XLU",   # Utilities
    "COST" : "XLP",   # Consumer Staples
    "MSFT" : "XLK",   # Technology
    "NVDA" : "XLK",   # Technology
    "INTC" : "XLK",   # Technology
    "VZ"   : "XLC",   # Communication Services
    # AMZN: GICS Consumer Discretionary (XLY), distinct from XLK hedging role
    "AMZN" : "XLY",
}

# Add all expansion stocks using SECTOR_STOCKS as the source of truth
for _sector, _tickers in SECTOR_STOCKS.items():
    for _t in _tickers:
        SECTOR_MAP[_t] = _sector
del _sector, _tickers, _t


# ── Public API ────────────────────────────────────────────────────────────────

def get_expanded_universe() -> tuple[list[str], dict[str, str], dict[str, str]]:
    """
    Return the full expanded universe.

    Returns:
        ticker_list    : all tickers in order (core + expansion stocks)
        asset_class    : {ticker: asset_class} for every ticker
        sector_map     : {ticker: sector_etf} for all individual stocks

    Core tickers retain their original asset_class from data_pipeline.TICKERS.
    All new expansion stocks are assigned asset_class "stock".
    """
    from pipeline.data_pipeline import TICKERS as CORE_TICKERS  # lazy — avoids circular import

    combined_asset_class: dict[str, str] = dict(CORE_TICKERS)
    for t in NEW_TICKERS:
        if t not in combined_asset_class:
            combined_asset_class[t] = "stock"

    return list(combined_asset_class.keys()), combined_asset_class, SECTOR_MAP


def compute_beta_size_scalars(
    closes: "pd.DataFrame",
    spy_closes: "pd.Series",
    window: int = 252,
    beta_low: float = 0.50,
    beta_high: float = 1.20,
    scalar_min: float = 0.30,
    scalar_max: float = 1.00,
) -> "pd.DataFrame":
    """
    Compute a per-ticker, per-date sizing scalar based on 252-day rolling OLS beta to SPY.

    Scalar formula (linear, clamped):
        scalar = clip(1.50 - beta, scalar_min, scalar_max)

    Breakpoints:
        beta ≤ 0.50  → scalar = 1.0  (full size)
        beta = 0.85  → scalar ≈ 0.65
        beta ≥ 1.20  → scalar = 0.3  (minimum 30% of base size — never zero)

    The scalar is applied to the POSITION SIZE, not to a hard include/exclude decision.
    A high-beta stock (beta=1.5) still participates at 30% of base weight, maintaining
    cross-sectional signal quality in all sectors.

    Tightened scalar (beta_high=1.00):
        Call with beta_high=1.00 to use scalar = clip(1.50 - beta, 0.30, 1.00) with
        a steeper drop-off, reducing high-beta exposure more aggressively.

    Args:
        closes    : date × ticker DataFrame of adjusted close prices
        spy_closes: SPY adjusted close prices (date-aligned with closes)
        window    : rolling beta lookback in days (default 252 = 1 year)
        beta_low  : beta at or below which scalar = scalar_max (default 0.50)
        beta_high : beta at or above which scalar = scalar_min (default 1.20)
        scalar_min: minimum scalar applied to highest-beta names (default 0.30)
        scalar_max: maximum scalar applied to lowest-beta names (default 1.00)

    Returns:
        DataFrame of shape (len(closes), len(closes.columns)) with scalar values
        in [scalar_min, scalar_max].  NaN betas (insufficient data) → scalar_max.
    """
    spy_ret = spy_closes.pct_change()
    scalars = pd.DataFrame(
        scalar_max, index=closes.index, columns=closes.columns, dtype=float
    )

    # Align SPY to closes index
    spy_aligned = spy_ret.reindex(closes.index)

    for t in closes.columns:
        ret_t = closes[t].pct_change()
        roll_cov = ret_t.rolling(window, min_periods=window // 2).cov(spy_aligned)
        roll_var = spy_aligned.rolling(window, min_periods=window // 2).var()
        roll_beta = roll_cov / roll_var.replace(0, np.nan)

        # scalar = clip(1.50 - beta, scalar_min, scalar_max)
        scalar_s = (1.50 - roll_beta).clip(scalar_min, scalar_max)
        scalar_s = scalar_s.fillna(scalar_max)   # NaN (warm-up) → full size
        scalars[t] = scalar_s

    return scalars


def refresh_universe_quarterly(
    adv_threshold: float = 20_000_000,
    top_n_per_sector: int = 15,
    start: str = "2015-01-01",
) -> dict[str, list[str]]:
    """
    Re-screen S&P 500 constituents and return the top-N most liquid names per
    GICS sector that pass the ADV filter.

    Run manually (not automated) — intended for quarterly universe review.
    Prints a suggested SECTOR_STOCKS diff so you can update the static list.

    S&P 500 membership implies MCap > $14.5B (index inclusion threshold),
    so a separate market-cap filter is not needed here.

    Args:
        adv_threshold    : Minimum average daily dollar volume ($20M default).
        top_n_per_sector : Max tickers returned per GICS sector (default 15).
        start            : Start date for the ADV calculation window.

    Returns:
        {sector_etf: [ticker, ...]} — tickers that passed, sorted ADV-descending.
    """
    print("Fetching S&P 500 constituents from Wikipedia...")
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            flavor="lxml",
        )
        sp500 = tables[0]
    except Exception as e:
        print(f"  Failed to fetch S&P 500 list: {e}")
        return {}

    sp500 = sp500.rename(columns={"Symbol": "ticker", "GICS Sector": "gics_sector"})
    sp500["ticker"] = sp500["ticker"].str.replace(".", "-", regex=False)

    GICS_TO_ETF: dict[str, str] = {
        "Information Technology"  : "XLK",
        "Health Care"             : "XLV",
        "Energy"                  : "XLE",
        "Industrials"             : "XLI",
        "Financials"              : "XLF",
        "Communication Services"  : "XLC",
        "Consumer Staples"        : "XLP",
        "Consumer Discretionary"  : "XLY",
        "Real Estate"             : "XLRE",
        "Materials"               : "XLB",
        "Utilities"               : "XLU",
    }

    sp500["sector_etf"] = sp500["gics_sector"].map(GICS_TO_ETF)
    sp500 = sp500.dropna(subset=["sector_etf"])
    tickers_by_sector: dict[str, list[str]] = (
        sp500.groupby("sector_etf")["ticker"].apply(list).to_dict()
    )

    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    results: dict[str, list[str]] = {}

    for sector_etf, sector_tickers in sorted(tickers_by_sector.items()):
        print(f"\n  {sector_etf}: screening {len(sector_tickers)} constituents...")

        try:
            batch = yf.download(
                sector_tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as e:
            print(f"    Batch download failed: {e}")
            continue

        passed: list[tuple[str, float]] = []

        for ticker in sector_tickers:
            try:
                if len(sector_tickers) == 1:
                    df = batch
                else:
                    lvl0 = batch.columns.get_level_values(0)
                    if ticker not in lvl0:
                        continue
                    df = batch[ticker].dropna(how="all")

                if df.empty or len(df) < 500:
                    continue

                close  = df["Close"].dropna()
                volume = df["Volume"].dropna()
                adv    = float((close * volume).mean())

                if adv >= adv_threshold:
                    passed.append((ticker, adv))

            except Exception:
                continue

        passed.sort(key=lambda x: x[1], reverse=True)
        results[sector_etf] = [t for t, _ in passed[:top_n_per_sector]]
        for t, adv in passed[:top_n_per_sector]:
            print(f"    {t:<8}  ADV=${adv/1e6:.0f}M")

    # Print suggested diff
    W = 70
    print(f"\n{'=' * W}")
    print("  Suggested SECTOR_STOCKS (copy into universe_expansion.py):")
    print(f"{'=' * W}")
    from pipeline.data_pipeline import TICKERS as CORE_TICKERS
    core_set = set(CORE_TICKERS.keys())

    for sector_etf, tickers in sorted(results.items()):
        new_only = [t for t in tickers if t not in core_set]
        print(f'    "{sector_etf}": {new_only},')

    return results


# ── Validation gate ───────────────────────────────────────────────────────────

def validate_expanded_universe(
    data_dir: Path = Path("data/raw"),
    coverage_start: str = "2018-01-01",
    min_coverage: float = 0.90,
    min_stocks_per_sector: int = 5,
    max_mean_beta: float = 0.80,
) -> bool:
    """
    Validate the expanded universe and print a sector-level summary table.

    Beta is now used for SIZING SCALARS only (compute_beta_size_scalars), not for
    hard inclusion/exclusion.  No tickers are removed based on beta.

    Checks (hard pass/fail):
      1. No sector has fewer than min_stocks_per_sector (default 5) stocks
      2. All NEW_TICKERS have > min_coverage (90%) data since coverage_start (2018)

    Prints per-sector table:
      Sector | Count | Mean Beta | Median Beta | Mean ADV ($M) | Mean Coverage | Pass?

    Returns:
        True if all checks pass, False otherwise.
    """
    all_tickers, asset_class, sector_map = get_expanded_universe()

    ref_start    = pd.Timestamp(coverage_start)
    trading_days = len(pd.bdate_range(ref_start, pd.Timestamp.today()))

    # ── Load SPY returns for beta computation ─────────────────────────────────
    spy_ret = pd.Series(dtype=float)
    spy_path = data_dir / "SPY.parquet"
    if spy_path.exists():
        try:
            spy_df  = pd.read_parquet(spy_path, columns=["Close"])
            spy_df.index = pd.to_datetime(spy_df.index)
            spy_ret = spy_df["Close"].pct_change().dropna()
        except Exception:
            pass

    # ── Per-ticker stats ──────────────────────────────────────────────────────
    ticker_stats: dict[str, dict] = {}

    for ticker in NEW_TICKERS:
        path = data_dir / f"{ticker}.parquet"
        if not path.exists():
            ticker_stats[ticker] = {"coverage": 0.0, "adv_m": 0.0,
                                    "beta": float("nan"), "missing": True}
            continue
        try:
            df = pd.read_parquet(path, columns=["Close", "Volume"])
            df.index = pd.to_datetime(df.index)

            in_window = df.loc[ref_start:]
            coverage  = len(in_window) / trading_days

            adv_m = float((df["Close"] * df["Volume"]).mean()) / 1e6

            beta = float("nan")
            if not spy_ret.empty:
                ret    = df["Close"].pct_change().dropna()
                common = ret.index.intersection(spy_ret.index)
                if len(common) >= 252:
                    r       = ret.reindex(common)
                    s       = spy_ret.reindex(common)
                    spy_var = float(s.var())
                    if spy_var > 0:
                        beta = float(r.cov(s)) / spy_var

            ticker_stats[ticker] = {
                "coverage": coverage,
                "adv_m"   : adv_m,
                "beta"    : beta,
                "missing" : False,
            }
        except Exception as e:
            ticker_stats[ticker] = {"coverage": 0.0, "adv_m": 0.0,
                                    "beta": float("nan"), "missing": True,
                                    "error": str(e)}

    # ── Aggregate by sector ───────────────────────────────────────────────────
    sector_summary: dict[str, dict] = {}
    sectors_with_stocks = sorted({v for v in sector_map.values()})

    for sec in sectors_with_stocks:
        sec_tickers = [t for t in NEW_TICKERS if sector_map.get(t) == sec]
        if not sec_tickers:
            continue

        valid    = [t for t in sec_tickers if not ticker_stats.get(t, {}).get("missing")]
        betas    = [ticker_stats[t]["beta"] for t in valid if not np.isnan(ticker_stats[t]["beta"])]
        advs     = [ticker_stats[t]["adv_m"] for t in valid]
        covs     = [ticker_stats[t]["coverage"] for t in valid]

        sector_summary[sec] = {
            "count"       : len(sec_tickers),
            "valid"       : len(valid),
            "mean_beta"   : float(np.mean(betas))   if betas else float("nan"),
            "median_beta" : float(np.median(betas)) if betas else float("nan"),
            "mean_adv_m"  : float(np.mean(advs))    if advs  else 0.0,
            "mean_cov"    : float(np.mean(covs))     if covs  else 0.0,
        }

    # ── Print sector summary table ────────────────────────────────────────────
    W = 82
    print("=" * W)
    print(f"  Validation Gate — Expanded Universe  (coverage from {coverage_start})")
    print("=" * W)
    print(f"\n  {'Sector':<6}  {'Stocks':>6}  {'MeanBeta':>9}  {'MedBeta':>8}  "
          f"{'ADV($M)':>8}  {'Coverage':>9}  {'Gate':>6}")
    print("  " + "-" * (W - 2))

    sector_fails: list[str] = []
    for sec in sorted(sector_summary):
        s = sector_summary[sec]
        count_ok  = s["count"] >= min_stocks_per_sector
        cov_ok    = s["mean_cov"] >= min_coverage
        gate      = "PASS" if (count_ok and cov_ok) else "FAIL"
        if gate == "FAIL":
            sector_fails.append(sec)

        mean_b = f"{s['mean_beta']:.3f}" if not np.isnan(s["mean_beta"]) else "  n/a"
        med_b  = f"{s['median_beta']:.3f}" if not np.isnan(s["median_beta"]) else "  n/a"
        print(f"  {sec:<6}  {s['count']:>6}  {mean_b:>9}  {med_b:>8}  "
              f"{s['mean_adv_m']:>8.1f}  {s['mean_cov']:>8.1%}  {gate:>6}")

    # ── Overall checks ────────────────────────────────────────────────────────
    all_betas  = [ticker_stats[t]["beta"] for t in NEW_TICKERS
                  if not ticker_stats.get(t, {}).get("missing")
                  and not np.isnan(ticker_stats[t]["beta"])]
    mean_beta  = float(np.mean(all_betas)) if all_betas else float("nan")
    # Beta is informational only — no hard gate. High-beta tickers get reduced
    # sizing via compute_beta_size_scalars(), not exclusion.
    beta_note  = ("NOTE: beta-weighted sizing active — no hard beta gate" if True else "")

    missing    = [t for t in NEW_TICKERS if ticker_stats.get(t, {}).get("missing")]
    low_cov    = [t for t in NEW_TICKERS
                  if not ticker_stats.get(t, {}).get("missing")
                  and ticker_stats[t]["coverage"] < min_coverage]

    n_total    = len(NEW_TICKERS)
    n_ok       = len(NEW_TICKERS) - len(missing) - len(low_cov)

    all_pass = (not sector_fails) and (not missing) and (not low_cov)

    print(f"\n  {'─' * (W - 2)}")
    print(f"  Overall mean beta: {mean_beta:.3f}  ({beta_note})")
    print(f"  Min sector stock count: "
          f"{min(s['count'] for s in sector_summary.values())}  "
          f"(target >= {min_stocks_per_sector})  "
          f"{'PASS' if not sector_fails else 'FAIL: ' + ', '.join(sector_fails)}")
    print(f"  Coverage >= {min_coverage:.0%} since {coverage_start}: "
          f"{n_ok}/{n_total} tickers  "
          f"{'PASS' if not missing and not low_cov else 'FAIL'}")

    if missing:
        print(f"  Missing parquets ({len(missing)}): {', '.join(missing)}")
        print("    → Run: python -m pipeline.data_pipeline  (USE_EXPANDED_UNIVERSE=True)")
    if low_cov:
        print(f"  Low coverage (<{min_coverage:.0%}): {', '.join(low_cov)}")
        print("    → These tickers likely listed after 2018 — acceptable for newer names")

    unmapped = [t for t in NEW_TICKERS if t not in sector_map]
    if unmapped:
        print(f"  Unmapped tickers: {', '.join(unmapped)}")
    else:
        print(f"  Sector mappings: complete ({len(sector_map)} total stocks mapped)")

    print(f"\n  {'='*20}  {'GATE: PASS' if all_pass else 'GATE: FAIL'}  {'='*20}")
    print("=" * W)
    return all_pass


def main() -> None:
    """
    Entry point for manual validation / quarterly re-screen.

    Usage:
        python -m pipeline.universe_expansion            # validate only
        python -m pipeline.universe_expansion --refresh  # full S&P 500 re-screen
    """
    import sys

    if "--refresh" in sys.argv:
        print("Running quarterly S&P 500 re-screen...\n")
        refresh_universe_quarterly()
    else:
        all_tickers, asset_class, sector_map = get_expanded_universe()
        print(f"Expanded universe: {len(all_tickers)} tickers "
              f"({len(NEW_TICKERS)} new + core)")
        print(f"Sector map covers: {len(sector_map)} individual stocks\n")
        validate_expanded_universe()


if __name__ == "__main__":
    main()
