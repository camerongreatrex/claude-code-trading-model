# DISABLED: Expanded universe failed validation (Apr 2026). Bull_calm alpha negative
# due to 0.91 SPY correlation. Cross-asset allocation confirmed as primary alpha source.

"""
Expands tradeable universe from 42 core to ~157 instruments via top-liquidity S&P 500
constituents by GICS sector. Set USE_EXPANDED_UNIVERSE=True to enable.
Public API: get_expanded_universe(), refresh_universe_quarterly().
Static screen: ADV>$20M, MCap>$5B, top 10–15 per sector.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
USE_EXPANDED_UNIVERSE: bool = False  # gates not yet passing — set True to re-enable

# ── New stocks per GICS sector ────────────────────────────────────────────────
# Additive to core. Top 10–15 liquid S&P 500 names per sector (ADV>$20M, MCap>$5B,
# 2026-Q1). Core tickers excluded.

SECTOR_STOCKS: dict[str, list[str]] = {
    # MMC removed — delisted. Beta filtering replaced with beta-weighted sizing.
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

# Flat unique ticker list (insertion order preserved)
_seen: set[str] = set()
NEW_TICKERS: list[str] = []
for _sector_tickers in SECTOR_STOCKS.values():
    for _t in _sector_tickers:
        if _t not in _seen:
            _seen.add(_t)
            NEW_TICKERS.append(_t)
del _seen, _sector_tickers, _t

# ── SECTOR_MAP ────────────────────────────────────────────────────────────────
# Stock ticker → GICS sector ETF (real GICS, not HEDGE_MAP shortcuts).
# Excludes broad ETFs, bonds, commodities.

SECTOR_MAP: dict[str, str] = {
    # ── Core stocks ─────────────────────────────────────────────────────────
    "JPM"  : "XLF",
    "GS"   : "XLF",
    "BRK-B": "XLF",   # conglomerate; insurance-heavy per GICS
    "JNJ"  : "XLV",
    "XOM"  : "XLE",
    "GE"   : "XLI",
    "NEE"  : "XLU",
    "COST" : "XLP",
    "MSFT" : "XLK",
    "NVDA" : "XLK",
    "INTC" : "XLK",
    "VZ"   : "XLC",
    "AMZN" : "XLY",   # GICS XLY, distinct from XLK hedging role
}

# Add all expansion stocks using SECTOR_STOCKS as the source of truth
for _sector, _tickers in SECTOR_STOCKS.items():
    for _t in _tickers:
        SECTOR_MAP[_t] = _sector
del _sector, _tickers, _t


# ── Public API ────────────────────────────────────────────────────────────────

def get_expanded_universe() -> tuple[list[str], dict[str, str], dict[str, str]]:
    """
    Return (tickers, asset_class, sector_map). Core tickers keep their original
    asset_class; expansion stocks assigned "stock".
    """
    from v1.pipeline.data_pipeline import TICKERS as CORE_TICKERS  # lazy — avoids circular import

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
    Per-ticker sizing scalar from 252d rolling OLS beta to SPY:
        scalar = clip(1.50 - beta, scalar_min, scalar_max)
    Breakpoints: beta<=0.50 -> 1.0, beta=0.85 -> ~0.65, beta>=1.20 -> 0.30.
    Applied to size (never excludes). NaN betas -> scalar_max.
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
    Re-screen S&P 500: top-N most liquid per GICS sector passing ADV filter.
    Manual quarterly run; prints suggested SECTOR_STOCKS diff. Returns
    {sector_etf: [ticker, ...]} sorted ADV-descending.
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
    from v1.pipeline.data_pipeline import TICKERS as CORE_TICKERS
    core_set = set(CORE_TICKERS.keys())

    for sector_etf, tickers in sorted(results.items()):
        new_only = [t for t in tickers if t not in core_set]
        print(f'    "{sector_etf}": {new_only},')

    return results


# ── Validation gate ───────────────────────────────────────────────────────────

def validate_expanded_universe(
    data_dir: Path = Path("data/v1/raw"),
    coverage_start: str = "2018-01-01",
    min_coverage: float = 0.90,
    min_stocks_per_sector: int = 5,
    max_mean_beta: float = 0.80,
) -> bool:
    """
    Validate expanded universe; print sector summary. Beta is informational only.
    Hard checks: per-sector count >= min_stocks_per_sector (5),
    and per-ticker coverage >= min_coverage (90%) since coverage_start (2018).
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

    # ── Per-ticker stats ─────────────────────────────────────────────────────
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
    # Beta informational only; high-beta gets reduced sizing via compute_beta_size_scalars().
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
        print("    → Run: python -m v1.pipeline.data_pipeline  (USE_EXPANDED_UNIVERSE=True)")
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
    Manual entry point. Use --refresh for full S&P 500 re-screen, else validate-only.
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
