"""
universe_screen.py
──────────────────
Screen candidate tickers for universe expansion.

Criteria for inclusion:
  1. Liquidity: avg daily dollar volume > $50M (tight spreads, reliable data)
  2. History: at least 5 years of daily data (enough for walk-forward validation)
  3. Low correlation: average pairwise correlation with existing universe < 0.35
  4. Bear stress correlation: correlation with SPY during VIX > 20 periods < 0.40
  5. Distinct driver: not redundant with an existing ticker's economic driver

Asset classes screened:
  - International sector ETFs (ex-US country indices)
  - Real assets (timber, infrastructure, water, clean energy)
  - Alternative strategies (merger arb, managed futures, anti-beta)
  - Fixed income niches (munis, floating rate, senior loans, EM bonds)
  - Commodity sub-sectors (uranium, lithium, rare earths, agriculture sub-indices)
  - Currency ETFs beyond FXE/FXY (CHF, AUD, CAD, GBP)

Usage:
  python -m v1.pipeline.universe_screen
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from v1.pipeline.data_pipeline import TICKER_LIST, ASSET_CLASS

# ── Candidate tickers to screen ───────────────────────────────────────────────
# Each tuple: (ticker, proposed_asset_class, description, economic_driver)
CANDIDATES = [
    # International sector ETFs — different regulatory/macro drivers than US sectors
    ("EWG",  "equity_index", "Germany (DAX) ETF",              "EU industrial cycle, ECB policy"),
    ("EWU",  "equity_index", "UK (FTSE) ETF",                  "BOE policy, GBP, commodity-heavy index"),
    ("EWA",  "equity_index", "Australia ETF",                   "Commodity exporter, China demand, RBA"),
    ("EWC",  "equity_index", "Canada ETF",                      "Oil/resource exporter, BOC policy"),
    ("EWT",  "equity_index", "Taiwan ETF",                      "Semiconductor supply chain (TSMC)"),
    ("EWY",  "equity_index", "South Korea ETF",                 "Samsung/SK Hynix, memory cycle"),
    ("INDA", "equity_index", "India ETF",                       "Domestic consumption, RBI policy"),

    # Real assets — inflation hedges with structural demand drivers
    ("WOOD", "commodity",    "Global Timber & Forestry ETF",    "Housing starts, construction, carbon credits"),
    ("IGF",  "equity_index", "Global Infrastructure ETF",       "Government capex, toll roads, utilities"),
    ("PHO",  "commodity",    "Water Resources ETF",             "Water scarcity, infrastructure spending"),
    ("ICLN", "equity_index", "Clean Energy ETF",                "Energy transition policy, IRA subsidies"),

    # Alternative strategy ETFs — structurally different return drivers
    ("MNA",  "commodity",    "Merger Arbitrage ETF",            "M&A spread capture, deal completion rates"),
    ("CTA",  "commodity",    "CTA/Managed Futures ETF",         "Trend-following across all asset classes"),
    ("BTAL", "commodity",    "Anti-Beta ETF (long low-beta, short high-beta)", "Low-vol factor, structural short"),
    ("QAI",  "commodity",    "Hedge Fund Multi-Strategy ETF",   "Multi-strategy hedge fund replication"),

    # Fixed income niches — different rate/credit sensitivities than TLT/HYG/TIP
    ("FLOT", "bond",         "Floating Rate Notes ETF",         "Rising rate beneficiary, near-zero duration"),
    ("BKLN", "bond",         "Senior Loan ETF",                 "Floating rate, senior secured, credit spread"),
    ("MUB",  "bond",         "Municipal Bond ETF",              "Tax-exempt income, state/local credit"),
    ("VGSH", "bond",         "Short-Term Treasury ETF",         "Cash proxy, rate hike protection"),
    ("EMB",  "bond",         "EM Dollar Bond ETF",              "EM sovereign credit, USD strength inverse"),

    # Commodity sub-sectors — distinct supply/demand dynamics
    ("URA",  "commodity",    "Uranium ETF",                     "Nuclear energy demand, supply deficit"),
    ("LIT",  "commodity",    "Lithium & Battery Tech ETF",      "EV demand, battery supply chain"),
    ("REMX", "commodity",    "Rare Earth/Strategic Metals ETF", "China supply dominance, defense demand"),
    ("WEAT", "commodity",    "Wheat ETF",                       "Agricultural weather, geopolitical supply risk"),
    ("CANE", "commodity",    "Sugar ETF",                       "Ethanol mandate, tropical weather"),
    ("SLV",  "commodity",    "Silver ETF",                      "Industrial demand + monetary demand (dual driver)"),
    ("PPLT", "commodity",    "Platinum ETF",                    "Auto catalytic converters, hydrogen economy"),

    # Currency ETFs — carry and macro divergence
    ("FXF",  "commodity",    "Swiss Franc ETF",                 "Ultimate safe haven, SNB policy"),
    ("FXA",  "commodity",    "Australian Dollar ETF",           "Commodity currency, China proxy, RBA carry"),
    ("FXC",  "commodity",    "Canadian Dollar ETF",             "Oil correlation, BOC policy divergence"),
    ("FXB",  "commodity",    "British Pound ETF",               "BOE policy, Brexit aftermath, UK macro"),

    # Additional candidates for $20M threshold re-screen
    ("COPX", "commodity",    "Copper Miners ETF",               "Electrification demand, housing, industrial cycle"),
    ("HACK", "equity_index", "Cybersecurity ETF",               "Secular growth in cyber spending, low cyclicality"),
    ("XBI",  "equity_index", "Biotech ETF (SPDR)",              "FDA pipeline, binary event risk, low macro correlation"),
    ("IYR",  "equity_index", "US Real Estate ETF (iShares)",    "Rental income, rate sensitivity, distinct from VNQ"),
]

FEATURE_DIR = Path("data/v1/features")
MACRO_DIR   = Path("data/shared/macro")
START = "2015-01-01"
END   = "2026-01-01"


def _load_existing_returns() -> pd.DataFrame:
    """Load log returns for all existing universe tickers."""
    returns = {}
    for ticker in TICKER_LIST:
        path = FEATURE_DIR / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "log_return" in df.columns:
                returns[ticker] = df["log_return"]
    return pd.DataFrame(returns).dropna()


def _fetch_candidate(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV for a candidate ticker."""
    try:
        df = yf.download(ticker, start=START, end=END, auto_adjust=True,
                         progress=False, multi_level_index=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        print(f"  {ticker}: download failed — {e}")
        return pd.DataFrame()


def screen_candidate(
    ticker: str,
    candidate_df: pd.DataFrame,
    existing_returns: pd.DataFrame,
    vix: pd.Series = None,
) -> dict:
    """
    Screen one candidate against the existing universe.

    Returns dict with screening metrics, or None if candidate fails
    basic data quality checks.
    """
    if len(candidate_df) < 1000:
        return None

    close  = candidate_df["Close"]
    volume = candidate_df["Volume"]

    # Liquidity check: average daily dollar volume
    avg_dv = float((close * volume).mean())
    if avg_dv < 20_000_000:  # $20M minimum (sufficient for portfolios under $500k)
        return {"ticker": ticker, "status": "FAIL",
                "reason": f"illiquid (${avg_dv/1e6:.0f}M avg daily vol)"}

    # Compute daily log returns
    cand_ret = np.log(close / close.shift(1)).dropna()
    cand_ret.name = ticker

    # Align with existing universe
    common_idx = cand_ret.index.intersection(existing_returns.index)
    if len(common_idx) < 500:
        return {"ticker": ticker, "status": "FAIL",
                "reason": f"insufficient overlap ({len(common_idx)} days)"}

    cand_aligned  = cand_ret.reindex(common_idx)
    exist_aligned = existing_returns.reindex(common_idx)

    # Average pairwise correlation with existing universe
    pairwise_corrs = []
    for col in exist_aligned.columns:
        c = float(cand_aligned.corr(exist_aligned[col]))
        if not np.isnan(c):
            pairwise_corrs.append((col, c))

    if not pairwise_corrs:
        avg_corr = 1.0
        max_corr = 1.0
        most_corr_ticker = ""
    else:
        corr_vals        = [c for _, c in pairwise_corrs]
        avg_corr         = float(np.mean(corr_vals))
        max_idx          = int(np.argmax(corr_vals))
        max_corr         = corr_vals[max_idx]
        most_corr_ticker = pairwise_corrs[max_idx][0]

    # SPY correlation
    spy_corr = float(cand_aligned.corr(exist_aligned["SPY"])) \
               if "SPY" in exist_aligned.columns else 1.0

    # Bear stress correlation (VIX > 20 periods)
    bear_stress_corr = spy_corr  # default if no VIX data
    if vix is not None:
        vix_aligned  = vix.reindex(common_idx).ffill()
        stress_mask  = vix_aligned > 20
        if stress_mask.sum() > 60 and "SPY" in exist_aligned.columns:
            stress_cand = cand_aligned[stress_mask]
            stress_spy  = exist_aligned["SPY"][stress_mask]
            bear_stress_corr = float(stress_cand.corr(stress_spy))

    # Annualized volatility
    ann_vol = float(cand_ret.std() * np.sqrt(252) * 100)

    # Annualized return
    total_ret = float(np.exp(cand_ret.sum()) - 1)
    n_years   = len(cand_ret) / 252
    ann_ret   = float((1 + total_ret) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

    # Max drawdown
    cum    = (1 + close.pct_change().fillna(0)).cumprod()
    peak   = cum.cummax()
    max_dd = float(((cum - peak) / peak).min() * 100)

    # Pass/fail logic
    pass_corr       = avg_corr < 0.35
    pass_stress     = bear_stress_corr < 0.40
    pass_redundancy = max_corr < 0.70

    if pass_corr and pass_stress and pass_redundancy:
        status = "PASS"
    elif pass_corr or pass_stress:
        status = "MARGINAL"
    else:
        status = "FAIL"

    return {
        "ticker"           : ticker,
        "status"           : status,
        "trading_days"     : len(candidate_df),
        "avg_dollar_vol_M" : round(avg_dv / 1e6, 0),
        "ann_return_pct"   : round(ann_ret, 1),
        "ann_vol_pct"      : round(ann_vol, 1),
        "max_dd_pct"       : round(max_dd, 1),
        "avg_corr"         : round(avg_corr, 3),
        "spy_corr"         : round(spy_corr, 3),
        "bear_stress_corr" : round(bear_stress_corr, 3),
        "max_corr"         : round(max_corr, 3),
        "most_corr_with"   : most_corr_ticker,
    }


def main():
    W = 80
    print("=" * W)
    print("  Universe Expansion Screen")
    print(f"  Existing universe: {len(TICKER_LIST)} tickers")
    print(f"  Candidates: {len(CANDIDATES)} tickers")
    print("=" * W)

    print("\nLoading existing universe returns...")
    existing = _load_existing_returns()
    print(f"  {existing.shape[0]} trading days x {existing.shape[1]} tickers")

    # Load VIX for bear stress correlation
    vix = None
    macro_path = MACRO_DIR / "macro_features.parquet"
    if macro_path.exists():
        macro = pd.read_parquet(macro_path)
        if "vix" in macro.columns:
            vix = macro["vix"]

    # Current universe avg pairwise correlation (baseline)
    corr_mat = existing.corr()
    upper    = np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
    baseline_corr = float(corr_mat.values[upper].mean())
    print(f"  Current avg pairwise correlation: {baseline_corr:.3f}")

    print(f"\nScreening {len(CANDIDATES)} candidates...\n")

    results = []
    for ticker, asset_class, description, driver in CANDIDATES:
        if ticker in TICKER_LIST:
            print(f"  {ticker:<6} SKIP — already in universe")
            continue

        print(f"  {ticker:<6} downloading...", end=" ", flush=True)
        df = _fetch_candidate(ticker)
        if df.empty:
            continue

        result = screen_candidate(ticker, df, existing, vix)
        if result is None:
            print("insufficient data")
            continue

        result["asset_class"] = asset_class
        result["description"] = description
        result["driver"]      = driver
        results.append(result)

        icon = {"PASS": "v", "MARGINAL": "~", "FAIL": "x"}.get(result["status"], "?")
        if "avg_corr" in result:
            print(f"{icon} {result['status']:<8} "
                  f"corr={result['avg_corr']:.3f}  "
                  f"stress={result['bear_stress_corr']:.3f}  "
                  f"vol={result['ann_vol_pct']:.0f}%  "
                  f"ret={result['ann_return_pct']:+.1f}%")
        else:
            print(f"{icon} {result['status']:<8} {result.get('reason', '')}")

    if not results:
        print("\nNo candidates screened successfully.")
        return

    df_results = pd.DataFrame(results)

    passed   = df_results[df_results["status"] == "PASS"].sort_values("avg_corr")
    marginal = df_results[df_results["status"] == "MARGINAL"].sort_values("avg_corr")
    failed   = df_results[df_results["status"] == "FAIL"]

    print(f"\n{'=' * W}")
    print(f"  PASSED ({len(passed)} tickers) -- ready for universe addition")
    print(f"{'=' * W}")
    if not passed.empty:
        print(f"  {'Ticker':<8} {'Class':<14} {'Avg Corr':>8} {'Stress':>8} "
              f"{'Max Corr':>8} {'Most Corr':>10} {'Vol%':>6} {'Ret%':>6} {'DD%':>6}")
        print("  " + "-" * (W - 2))
        for _, r in passed.iterrows():
            print(f"  {r['ticker']:<8} {r['asset_class']:<14} {r['avg_corr']:>8.3f} "
                  f"{r['bear_stress_corr']:>8.3f} {r['max_corr']:>8.3f} "
                  f"{r['most_corr_with']:>10} {r['ann_vol_pct']:>5.0f}% "
                  f"{r['ann_return_pct']:>+5.1f}% {r['max_dd_pct']:>5.1f}%")
        print()
        for _, r in passed.iterrows():
            print(f"  {r['ticker']}: {r['description']} -- {r['driver']}")

        # Show what the lower threshold surfaced vs original $50M
        new_from_lower = passed[passed["avg_dollar_vol_M"] < 50]
        if not new_from_lower.empty:
            print(f"\n  NEW from $20M threshold (would have failed at $50M):")
            for _, r in new_from_lower.iterrows():
                print(f"    {r['ticker']:<8} ${r['avg_dollar_vol_M']:.0f}M ADV  "
                      f"corr={r['avg_corr']:.3f}  stress={r['bear_stress_corr']:.3f}  "
                      f"— {r.get('description', '')}")

    print(f"\n{'=' * W}")
    print(f"  MARGINAL ({len(marginal)} tickers) -- worth investigating further")
    print(f"{'=' * W}")
    if not marginal.empty:
        for _, r in marginal.iterrows():
            print(f"  {r['ticker']:<8} corr={r['avg_corr']:.3f}  "
                  f"stress={r['bear_stress_corr']:.3f}  "
                  f"max_corr={r['max_corr']:.3f} ({r['most_corr_with']})  "
                  f"-- {r['description']}")

    print(f"\n  FAILED: {len(failed)} tickers (high correlation or illiquid)")
    if not failed.empty:
        for _, r in failed.iterrows():
            raw_reason = r.get("reason")
            if raw_reason and not (isinstance(raw_reason, float) and np.isnan(raw_reason)):
                reason = raw_reason
            elif "avg_corr" in r and not (isinstance(r["avg_corr"], float) and np.isnan(r["avg_corr"])):
                reason = f"corr={r['avg_corr']:.3f}  stress={r['bear_stress_corr']:.3f}  max_corr={r['max_corr']:.3f} ({r['most_corr_with']})"
            else:
                reason = "no metrics available"
            print(f"    {r['ticker']:<8} {reason}")

    # Projected correlation impact if PASS tickers were added
    if not passed.empty:
        print(f"\n{'=' * W}")
        print(f"  PROJECTED IMPACT")
        print(f"{'=' * W}")
        n_new    = len(passed)
        n_total  = len(TICKER_LIST) + n_new
        old_n    = len(TICKER_LIST)
        old_pairs   = old_n * (old_n - 1) / 2
        new_pairs   = n_new * old_n
        total_pairs = n_total * (n_total - 1) / 2
        new_avg_corr    = passed["avg_corr"].mean()
        projected_corr  = (old_pairs * baseline_corr + new_pairs * new_avg_corr) / total_pairs
        print(f"  Current universe: {old_n} tickers, avg corr {baseline_corr:.3f}")
        print(f"  Adding {n_new} PASS tickers: projected avg corr {projected_corr:.3f}")
        print(f"  Correlation reduction: {baseline_corr - projected_corr:+.3f}")
        print(f"  More tickers with active signals = higher gross exposure = less cash drag")

        # Top 5 by lowest bear_stress_corr (best crash diversifiers)
        top5 = passed.nsmallest(5, "bear_stress_corr")
        print(f"\n  Top crash diversifiers (lowest bear_stress_corr):")
        for _, r in top5.iterrows():
            print(f"    {r['ticker']:<8} stress_corr={r['bear_stress_corr']:.3f}  "
                  f"avg_corr={r['avg_corr']:.3f}  -- {r['description']}")

    # Save results
    out_path = Path("data/v1/research/universe_screen.parquet")
    Path("data/v1/research").mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
