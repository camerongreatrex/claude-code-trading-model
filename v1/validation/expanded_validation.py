#!/usr/bin/env python3
"""
run_expanded_validation.py
──────────────────────────
Full validation, diagnostic, and fix pass for the expanded universe.
Runs every step end-to-end, prints actual numbers, applies fixes.

Changes vs prior pass:
  - Step 1: beta is INFORMATIONAL ONLY — no hard removal gate.
  - Step 2: loads from closes_matrix_expanded.parquet (full ~2015 history).
  - Step 3: frozen-weight sizing + beta scalars (in signal_generation.py).
  - Step 4a: 7-window walk-forward OOS validation.
  - Step 4b: allocation split optimisation if all gates pass.
  - Step 5: expanded Sharpe gate lowered from 0.80 → 0.50.

Usage:
    cd /Users/cameron/Documents/GitHub/Algorithmic-Trading-2
    python run_expanded_validation.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path("data/v1/raw")
FEATURE_DIR = Path("data/v1/features")
SIGNAL_DIR  = Path("data/v1/signals")
RESULTS_DIR = Path("data/v1/results")
MACRO_DIR   = Path("data/shared/macro")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── helpers ──────────────────────────────────────────────────────────────────

def section(title: str, width: int = 72) -> None:
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def compute_beta(ret: pd.Series, spy_ret: pd.Series, lookback: int = 252) -> float:
    common = ret.dropna().index.intersection(spy_ret.dropna().index)
    if len(common) < 30:
        return float("nan")
    r = ret.reindex(common).iloc[-lookback:]
    s = spy_ret.reindex(common).iloc[-lookback:]
    v = float(s.var())
    if v == 0:
        return float("nan")
    return float(r.cov(s)) / v


def compute_bear_beta(ret: pd.Series, spy_ret: pd.Series) -> float:
    common = ret.dropna().index.intersection(spy_ret.dropna().index)
    bear_days = common[spy_ret.reindex(common) < -0.01]
    if len(bear_days) < 10:
        return float("nan")
    r = ret.reindex(bear_days)
    s = spy_ret.reindex(bear_days)
    v = float(s.var())
    if v == 0:
        return float("nan")
    return float(r.cov(s)) / v


def sharpe(rets: pd.Series) -> float:
    if len(rets) < 10 or rets.std() == 0:
        return float("nan")
    return float(rets.mean() / rets.std() * np.sqrt(252))


def annual_return(rets: pd.Series) -> float:
    return float(rets.mean() * 252 * 100)


def annual_vol(rets: pd.Series) -> float:
    return float(rets.std() * np.sqrt(252) * 100)


def max_dd(rets: pd.Series) -> float:
    if len(rets) == 0:
        return float("nan")
    curve = (1 + rets).cumprod()
    roll_max = curve.cummax()
    dd = (curve - roll_max) / roll_max
    return float(dd.min() * 100)


def calmar(rets: pd.Series) -> float:
    ar = annual_return(rets)
    md = max_dd(rets)
    if md == 0 or np.isnan(md):
        return float("nan")
    return ar / abs(md)


def ols_beta_to_spy(rets: pd.Series, spy_rets: pd.Series) -> float:
    common = rets.dropna().index.intersection(spy_rets.dropna().index)
    if len(common) < 30:
        return float("nan")
    r, s = rets.reindex(common), spy_rets.reindex(common)
    v = float(s.var())
    if v == 0:
        return float("nan")
    return float(r.cov(s)) / v


def regime_alpha_bps(rets: pd.Series, spy: pd.Series, mask: pd.Series) -> float:
    common_i = rets.index.intersection(spy.index)
    m = mask.reindex(common_i).fillna(False)
    r_reg = rets.reindex(common_i)[m]
    s_reg = spy.reindex(common_i)[m]
    if len(r_reg) < 5:
        return float("nan")
    return (float(r_reg.mean()) - float(s_reg.mean())) * 10000


def _build_exp_returns(sizes: pd.DataFrame, allocation_split_expanded: float) -> pd.Series:
    """Build the expanded sleeve return series from sizes and raw closes."""
    ret_cols: dict = {}
    for t in sizes.columns:
        for p in [FEATURE_DIR / f"{t}.parquet", DATA_DIR / f"{t}.parquet"]:
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    df.index = pd.to_datetime(df.index)
                    if "Close" in df.columns:
                        ret_cols[t] = df["Close"].pct_change()
                    break
                except Exception:
                    pass

    returns_wide = pd.DataFrame(ret_cols).sort_index()
    sizes_aligned = sizes.reindex(returns_wide.index)
    exp_ret = (
        sizes_aligned.shift(1)
        * allocation_split_expanded
        * returns_wide.reindex(columns=sizes_aligned.columns)
    ).sum(axis=1).dropna()
    return exp_ret


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Build closes_matrix_expanded.parquet from existing raw files
# ══════════════════════════════════════════════════════════════════════════════

def step0_build_expanded_matrix() -> None:
    section("STEP 0: Build closes_matrix_expanded.parquet")
    from v1.pipeline.universe_expansion import NEW_TICKERS

    exp_path = DATA_DIR / "closes_matrix_expanded.parquet"

    # Check which raw files we have
    available = [t for t in NEW_TICKERS if (DATA_DIR / f"{t}.parquet").exists()]
    missing   = [t for t in NEW_TICKERS if t not in available]

    print(f"  NEW_TICKERS total    : {len(NEW_TICKERS)}")
    print(f"  Raw files available  : {len(available)}")
    print(f"  Missing raw parquets : {len(missing)}")
    if missing:
        print(f"  Missing tickers      : {missing}")

    if not available:
        print("  ERROR: no raw files found — run pipeline.data_pipeline first")
        return

    # Build expanded closes from individual raw files
    print(f"\n  Building expanded closes matrix from {len(available)} raw files...")
    closes_dict: dict = {}
    for t in available:
        try:
            df = pd.read_parquet(DATA_DIR / f"{t}.parquet", columns=["Close"])
            df.index = pd.to_datetime(df.index)
            closes_dict[t] = df["Close"]
        except Exception as e:
            print(f"    {t}: ERROR loading — {e}")

    if not closes_dict:
        print("  ERROR: could not load any close prices")
        return

    exp_closes = pd.DataFrame(closes_dict).sort_index()
    # Drop rows where ALL tickers are NaN (e.g., pre-market history)
    exp_closes = exp_closes.dropna(how="all")

    exp_closes.to_parquet(exp_path)
    print(f"  closes_matrix_expanded.parquet: {exp_closes.shape[0]} rows × {exp_closes.shape[1]} cols")
    print(f"  Date range: {exp_closes.index[0].date()} → {exp_closes.index[-1].date()}")

    # Also print core closes_matrix for comparison
    core_path = DATA_DIR / "closes_matrix.parquet"
    if core_path.exists():
        cm = pd.read_parquet(core_path)
        print(f"\n  closes_matrix.parquet (core): {cm.shape[0]} rows × {cm.shape[1]} cols")
        print(f"  Date range: {pd.to_datetime(cm.index[0]).date()} → {pd.to_datetime(cm.index[-1]).date()}")

    print(f"\n  NOTE: expanded matrix uses dropna(how='all') — individual ticker NaNs "
          f"are preserved. The signal generation's sector groupby handles per-ticker "
          f"NaN gaps via pct_change which naturally produces NaN for missing rows.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0b — Engineer features for expanded tickers (if missing)
# ══════════════════════════════════════════════════════════════════════════════

def step0b_features() -> None:
    section("STEP 0b: Engineer features for expanded tickers")
    from v1.pipeline.universe_expansion import NEW_TICKERS
    from v1.pipeline.feature_engineering import engineer

    missing_feat = [t for t in NEW_TICKERS if not (FEATURE_DIR / f"{t}.parquet").exists()]
    print(f"  Feature files missing: {len(missing_feat)} / {len(NEW_TICKERS)}")

    if not missing_feat:
        print("  All expanded tickers have feature files — skipping")
        return

    cm_path = DATA_DIR / "closes_matrix.parquet"
    closes_matrix = pd.read_parquet(cm_path) if cm_path.exists() else None
    if closes_matrix is not None:
        print(f"  Core closes matrix loaded: {closes_matrix.shape}")

    ok, skipped = 0, 0
    for t in missing_feat:
        raw_path = DATA_DIR / f"{t}.parquet"
        if not raw_path.exists():
            print(f"    {t:<8} SKIP — raw file missing")
            skipped += 1
            continue
        try:
            raw = pd.read_parquet(raw_path)
            df  = engineer(raw, ticker=t, closes_matrix=closes_matrix)
            (FEATURE_DIR / f"{t}.parquet").parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(FEATURE_DIR / f"{t}.parquet", engine="pyarrow", compression="snappy")
            print(f"    {t:<8} {len(df):>4} rows, {len(df.columns):>2} cols")
            ok += 1
        except Exception as e:
            print(f"    {t:<8} ERROR: {e}")
            skipped += 1

    print(f"\n  Features engineered: {ok} tickers  |  Skipped: {skipped}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data integrity check (beta is INFORMATIONAL ONLY — no removal)
# ══════════════════════════════════════════════════════════════════════════════

def step1_data_integrity() -> dict:
    """
    Returns ticker_stats dict. Beta is printed for information; NO hard removal.
    Beta-weighted sizing in signal_generation.py handles high-beta exposure.
    """
    section("STEP 1: Data Integrity Check")
    from v1.pipeline.universe_expansion import NEW_TICKERS, SECTOR_MAP, SECTOR_STOCKS

    # ── 1a. closes_matrix_expanded.parquet ───────────────────────────────────
    exp_path = DATA_DIR / "closes_matrix_expanded.parquet"
    core_path = DATA_DIR / "closes_matrix.parquet"

    print(f"\n  1a. Matrix files")
    if core_path.exists():
        cm = pd.read_parquet(core_path)
        cm.index = pd.to_datetime(cm.index)
        print(f"      closes_matrix.parquet (core)    : {cm.shape[0]} rows × {cm.shape[1]} cols  "
              f"{cm.index[0].date()} → {cm.index[-1].date()}")
    else:
        print("      closes_matrix.parquet: NOT FOUND")

    if exp_path.exists():
        exp_cm = pd.read_parquet(exp_path)
        exp_cm.index = pd.to_datetime(exp_cm.index)
        print(f"      closes_matrix_expanded.parquet  : {exp_cm.shape[0]} rows × {exp_cm.shape[1]} cols  "
              f"{exp_cm.index[0].date()} → {exp_cm.index[-1].date()}")
        nan_pct = exp_cm.isna().mean() * 100
        flagged_nan = nan_pct[nan_pct > 20].sort_values(ascending=False)
        print(f"\n  NaN > 20% — {len(flagged_nan)} tickers flagged:")
        if flagged_nan.empty:
            print("      None (PASS)")
        else:
            for t, p in flagged_nan.items():
                print(f"      {t:<8}  {p:.1f}%")
    else:
        print("      closes_matrix_expanded.parquet : NOT FOUND — run Step 0 first")

    # ── 1b. Per-ticker beta table ─────────────────────────────────────────────
    print(f"\n  1b. Beta table for NEW_TICKERS (sorted by 252d beta)")
    print(f"  NOTE: Beta is informational only — no hard removal. "
          f"High-beta tickers get reduced sizing via compute_beta_size_scalars().")
    spy_path = DATA_DIR / "SPY.parquet"
    spy_ret  = pd.Series(dtype=float)
    if spy_path.exists():
        spy_df  = pd.read_parquet(spy_path, columns=["Close"])
        spy_df.index = pd.to_datetime(spy_df.index)
        spy_ret = spy_df["Close"].pct_change().dropna()

    ref_start    = pd.Timestamp("2018-01-01")
    trading_days = len(pd.bdate_range(ref_start, pd.Timestamp.today()))

    ticker_stats: dict = {}
    for t in NEW_TICKERS:
        raw_path = DATA_DIR / f"{t}.parquet"
        if not raw_path.exists():
            ticker_stats[t] = dict(beta=float("nan"), bear_beta=float("nan"),
                                   adv_m=0.0, coverage=0.0, missing=True, sector=SECTOR_MAP.get(t,"?"))
            continue
        try:
            df = pd.read_parquet(raw_path, columns=["Close", "Volume"])
            df.index = pd.to_datetime(df.index)
            in_window = df.loc[ref_start:]
            coverage  = len(in_window) / trading_days
            adv_m     = float((df["Close"] * df["Volume"]).mean()) / 1e6
            ret_s     = df["Close"].pct_change().dropna()
            beta      = compute_beta(ret_s, spy_ret, 252)
            bear_beta = compute_bear_beta(ret_s, spy_ret)
            ticker_stats[t] = dict(beta=beta, bear_beta=bear_beta, adv_m=adv_m,
                                   coverage=coverage, missing=False,
                                   sector=SECTOR_MAP.get(t, "?"))
        except Exception as e:
            ticker_stats[t] = dict(beta=float("nan"), bear_beta=float("nan"),
                                   adv_m=0.0, coverage=0.0, missing=True,
                                   sector=SECTOR_MAP.get(t, "?"), error=str(e))

    print(f"\n  {'Ticker':<8}  {'Sector':<5}  {'252dBeta':>9}  {'BearBeta':>9}  "
          f"{'ADV($M)':>8}  {'Coverage':>9}  {'Scalar':>7}")
    print("  " + "-"*72)
    sorted_tickers = sorted(ticker_stats.items(),
                            key=lambda x: x[1]["beta"] if not np.isnan(x[1]["beta"]) else 99)
    for t, s in sorted_tickers:
        b_str  = f"{s['beta']:.3f}"      if not np.isnan(s["beta"])      else "   n/a"
        bb_str = f"{s['bear_beta']:.3f}" if not np.isnan(s["bear_beta"]) else "   n/a"
        # Compute the scalar that would be applied at this beta level (point-in-time)
        beta_v = s["beta"] if not np.isnan(s["beta"]) else 0.5
        scalar = float(np.clip(1.50 - beta_v, 0.30, 1.00))
        flag   = " ← reduced" if beta_v > 0.80 else ""
        print(f"  {t:<8}  {s['sector']:<5}  {b_str:>9}  {bb_str:>9}  "
              f"{s['adv_m']:>8.1f}  {s['coverage']:>8.1%}  {scalar:>6.2f}{flag}")

    # ── 1c. Sector summary ────────────────────────────────────────────────────
    print(f"\n  1c. Sector summary (all tickers — no hard beta removal)")
    print(f"\n  {'Sector':<5}  {'# Tickers':>9}  {'MeanBeta':>9}  {'MedBeta':>8}  "
          f"{'MeanBearBeta':>12}  {'MeanADV($M)':>12}  {'MeanScalar':>10}")
    print("  " + "-"*80)

    all_betas = []
    for sec in sorted(SECTOR_STOCKS.keys()):
        sec_t  = [t for t in NEW_TICKERS if SECTOR_MAP.get(t) == sec]
        valid  = [t for t in sec_t if not ticker_stats.get(t, {}).get("missing")]
        betas  = [ticker_stats[t]["beta"]      for t in valid if not np.isnan(ticker_stats[t]["beta"])]
        bbetas = [ticker_stats[t]["bear_beta"] for t in valid if not np.isnan(ticker_stats[t]["bear_beta"])]
        advs   = [ticker_stats[t]["adv_m"]     for t in valid]
        scalars = [float(np.clip(1.50 - b, 0.30, 1.00)) for b in betas]
        all_betas.extend(betas)

        mb  = f"{np.mean(betas):.3f}"   if betas  else "  n/a"
        mdb = f"{np.median(betas):.3f}" if betas  else "  n/a"
        mbb = f"{np.mean(bbetas):.3f}"  if bbetas else "  n/a"
        ma  = f"{np.mean(advs):.1f}"    if advs   else "  n/a"
        ms  = f"{np.mean(scalars):.3f}" if scalars else "  n/a"
        print(f"  {sec:<5}  {len(sec_t):>9}  {mb:>9}  {mdb:>8}  {mbb:>12}  {ma:>12}  {ms:>10}")

    mean_all_beta = float(np.mean(all_betas)) if all_betas else float("nan")
    print(f"\n  Portfolio unweighted mean beta: {mean_all_beta:.3f}")
    print(f"  NOTE: High-beta tickers get reduced sizing (scalar range 0.30–1.00). "
          f"No tickers removed.")

    return ticker_stats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Signal generation diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def step2_signals() -> tuple:
    """Generate signals and run diagnostics. Returns (composite, signals, sizes)."""
    section("STEP 2: Signal Generation Diagnostic")

    print("  Running generate_expanded_signals()...")
    import importlib
    import v1.pipeline.signal_generation as sg_mod
    importlib.reload(sg_mod)
    sg_mod.generate_expanded_signals()

    comp_path = SIGNAL_DIR / "expanded_composite.parquet"
    sig_path  = SIGNAL_DIR / "expanded_signals.parquet"
    sz_path   = SIGNAL_DIR / "expanded_sizes.parquet"

    if not (comp_path.exists() and sig_path.exists() and sz_path.exists()):
        print("  ERROR: Output parquets not found after generate_expanded_signals()")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    composite = pd.read_parquet(comp_path)
    signals   = pd.read_parquet(sig_path)
    sizes     = pd.read_parquet(sz_path)
    composite.index = pd.to_datetime(composite.index)
    signals.index   = pd.to_datetime(signals.index)
    sizes.index     = pd.to_datetime(sizes.index)

    # ── 2a. composite ─────────────────────────────────────────────────────────
    print(f"\n  2a. expanded_composite.parquet")
    print(f"      Shape     : {composite.shape}")
    print(f"      Date range: {composite.index[0].date()} → {composite.index[-1].date()}")

    last5 = composite.tail(5)
    print(f"\n  Cross-sectional distribution (last 5 dates):")
    print(f"  {'Date':<12}  {'Mean':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}  {'Non-NaN':>8}")
    print("  " + "-"*55)
    for dt, row in last5.iterrows():
        valid = row.dropna()
        if len(valid) == 0:
            print(f"  {str(dt.date()):<12}  {'n/a':>7}  {'n/a':>7}  {'n/a':>7}  {'n/a':>7}  {0:>8}")
            continue
        flag = " *** Z-SCORE BROKEN" if abs(float(valid.mean())) > 0.3 else ""
        print(f"  {str(dt.date()):<12}  {float(valid.mean()):>7.3f}  {float(valid.std()):>7.3f}  "
              f"{float(valid.min()):>7.3f}  {float(valid.max()):>7.3f}  {len(valid):>8}{flag}")

    # ── 2b. signals ───────────────────────────────────────────────────────────
    print(f"\n  2b. expanded_signals.parquet")
    last252_sig   = signals.tail(252)
    long_counts   = last252_sig.sum(axis=1)
    total_tickers = signals.shape[1]
    avg_long_pct  = float(long_counts.mean() / total_tickers * 100) if total_tickers > 0 else 0.0

    print(f"  Last 252 days — long/flat distribution (every ~12th row):")
    print(f"  {'Date':<12}  {'LONG':>6}  {'FLAT':>6}  {'%Long':>7}")
    print("  " + "-"*38)
    step = max(1, len(last252_sig) // 20)
    for i, (dt, row) in enumerate(last252_sig.iterrows()):
        if i % step == 0 or i == len(last252_sig) - 1:
            n_long = int(row.sum())
            n_flat = total_tickers - n_long
            pct    = n_long / total_tickers * 100 if total_tickers > 0 else 0
            print(f"  {str(dt.date()):<12}  {n_long:>6}  {n_flat:>6}  {pct:>6.1f}%")

    print(f"\n  Average % LONG (last 252d): {avg_long_pct:.1f}%  (target 30-60%)")
    if avg_long_pct > 70:
        print("  *** WARNING: avg LONG > 70% — entry threshold may be too loose")
    elif avg_long_pct < 15:
        print("  *** WARNING: avg LONG < 15% — entry threshold may be too tight")
    else:
        print("  In-range")

    long_freq = signals.mean().sort_values(ascending=False)
    print(f"\n  10 tickers LONG most frequently:")
    for t, f in long_freq.head(10).items():
        print(f"    {t:<8}  {f:.1%}")
    print(f"\n  10 tickers LONG least frequently:")
    for t, f in long_freq.tail(10).items():
        print(f"    {t:<8}  {f:.1%}")

    # ── 2c. sizes ─────────────────────────────────────────────────────────────
    print(f"\n  2c. expanded_sizes.parquet")
    n_stock_viol = int((sizes > 0.03).values.sum())
    print(f"  Per-stock cap (>3%) violations: {n_stock_viol}")

    from v1.pipeline.universe_expansion import SECTOR_STOCKS, SECTOR_MAP
    print(f"  Per-sector cap (>15%) violations:")
    sector_viol_found = False
    for sec, members in SECTOR_STOCKS.items():
        sec_cols = [t for t in members if t in sizes.columns]
        if not sec_cols:
            continue
        sec_total = sizes[sec_cols].sum(axis=1)
        viol_dates = sec_total[sec_total > 0.15]
        if not viol_dates.empty:
            sector_viol_found = True
            print(f"    {sec}: {len(viol_dates)} violations  (max={float(viol_dates.max()):.3f})")
    if not sector_viol_found:
        print("    None (PASS)")

    gross_exp = sizes.sum(axis=1) * 100
    avg_gross = float(gross_exp.mean())
    print(f"\n  Average gross exposure : {avg_gross:.1f}%  (target 40-70%)")

    if len(sizes) > 0:
        latest = sizes.iloc[-1]
        print(f"\n  Sector weight distribution — {sizes.index[-1].date()}:")
        print(f"  {'Sector':<5}  {'Active':>6}  {'TotalWt':>8}  {'LargestPos':>12}")
        print("  " + "-"*42)
        for sec, members in sorted(SECTOR_STOCKS.items()):
            sec_cols = [t for t in members if t in sizes.columns]
            if not sec_cols:
                continue
            sec_latest = latest[sec_cols]
            active = int((sec_latest > 0).sum())
            total  = float(sec_latest.sum()) * 100
            largest   = float(sec_latest.max()) * 100 if active > 0 else 0.0
            largest_t = sec_latest.idxmax() if active > 0 else "—"
            print(f"  {sec:<5}  {active:>6}  {total:>7.2f}%  {largest:>8.2f}% ({largest_t})")

    return composite, signals, sizes


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Portfolio integration diagnostic (OOS = last 252 days)
# ══════════════════════════════════════════════════════════════════════════════

def step3_portfolio(alloc_expanded: float = 0.50) -> dict:
    section("STEP 3: Portfolio Integration Diagnostic")

    from v1.portfolio.portfolio import ALLOCATION_SPLIT
    from v1.regimes.analysis import label_regimes

    alloc_exp = alloc_expanded  # may be overridden by step 4 optimisation

    sz_path = SIGNAL_DIR / "expanded_sizes.parquet"
    if not sz_path.exists():
        print("  expanded sizes not found — run Step 2 first")
        return {}

    exp_sizes = pd.read_parquet(sz_path)
    exp_sizes.index = pd.to_datetime(exp_sizes.index)

    # Build expanded sleeve returns
    exp_ret_series = _build_exp_returns(exp_sizes, alloc_exp)

    # Load core sleeve returns
    core_rets = pd.Series(dtype=float)
    core_curve_path = RESULTS_DIR / "portfolio_equity_curve.parquet"
    if core_curve_path.exists():
        try:
            curve_df = pd.read_parquet(core_curve_path)
            curve_df.index = pd.to_datetime(curve_df.index)
            col = curve_df.columns[0]
            core_rets = curve_df[col].pct_change().dropna()
        except Exception as e:
            print(f"  WARNING: could not load core equity curve: {e}")
    if core_rets.empty:
        print("  NOTE: portfolio_equity_curve.parquet not found — "
              "run portfolio.main() for core curve. "
              "Portfolio metrics will use expanded sleeve only for some calculations.")

    # SPY
    spy_rets = pd.Series(dtype=float)
    for p in [FEATURE_DIR / "SPY.parquet", DATA_DIR / "SPY.parquet"]:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                df.index = pd.to_datetime(df.index)
                spy_rets = df["Close"].pct_change().dropna()
                break
            except Exception:
                pass

    # Regime labels
    regimes = pd.Series(dtype=str)
    macro_path = MACRO_DIR / "macro_features.parquet"
    if macro_path.exists() and not spy_rets.empty:
        try:
            macro_df = pd.read_parquet(macro_path)
            vix = macro_df["vix"] if "vix" in macro_df.columns else pd.Series(dtype=float)
            if not vix.empty:
                regimes = label_regimes(vix, spy_rets, exp_ret_series.index)
        except Exception as e:
            print(f"  WARNING: could not load regime labels: {e}")

    OOS_DAYS = 252
    exp_oos = exp_ret_series.iloc[-OOS_DAYS:] if len(exp_ret_series) > OOS_DAYS else exp_ret_series

    # Combined
    if not core_rets.empty:
        common = exp_ret_series.index.intersection(core_rets.index)
        core_alloc = 1.0 - alloc_exp
        core_slice = core_rets.reindex(common).fillna(0) * core_alloc
        exp_slice  = exp_ret_series.reindex(common).fillna(0)
        comb_rets  = (core_slice + exp_slice).dropna()
    else:
        comb_rets = exp_ret_series.copy()

    comb_oos = comb_rets.iloc[-OOS_DAYS:] if len(comb_rets) >= OOS_DAYS else comb_rets

    # ── 3a. Sleeve table ──────────────────────────────────────────────────────
    print(f"\n  3a. Sleeve-level performance (OOS = last 252 days)")
    print(f"\n  {'Metric':<22}  {'Core Sleeve':>14}  {'Expanded Sleeve':>16}  {'Combined':>12}")
    print("  " + "-"*70)

    def _fmt(x, fmt=".3f"):
        return f"{x:{fmt}}" if not np.isnan(x) else "    n/a"

    core_oos = core_rets.iloc[-OOS_DAYS:] if len(core_rets) >= OOS_DAYS else core_rets
    metrics = [
        ("OOS Sharpe",       sharpe(core_oos),        sharpe(exp_oos),        sharpe(comb_oos)),
        ("Annual Return %",  annual_return(core_oos),  annual_return(exp_oos),  annual_return(comb_oos)),
        ("Annual Vol %",     annual_vol(core_oos),     annual_vol(exp_oos),     annual_vol(comb_oos)),
        ("Max Drawdown %",   max_dd(core_oos),         max_dd(exp_oos),         max_dd(comb_oos)),
        ("Calmar Ratio",     calmar(core_oos),         calmar(exp_oos),         calmar(comb_oos)),
        ("OLS Beta to SPY",  ols_beta_to_spy(core_oos, spy_rets),
                             ols_beta_to_spy(exp_oos, spy_rets.reindex(exp_oos.index)),
                             ols_beta_to_spy(comb_oos, spy_rets)),
    ]
    for label, cv, ev, cbv in metrics:
        print(f"  {label:<22}  {_fmt(cv):>14}  {_fmt(ev):>16}  {_fmt(cbv):>12}")

    # ── 3b. Regime alpha ──────────────────────────────────────────────────────
    print(f"\n  3b. Regime-decomposed alpha (bps/day)")
    print(f"\n  {'Regime':<14}  {'Core':>10}  {'Expanded':>10}  {'Combined':>10}")
    print("  " + "-"*52)
    for regime in ["bull_calm", "bull_stress", "bear_calm", "bear_stress"]:
        if regimes.empty:
            print(f"  {regime:<14}  {'n/a':>10}  {'n/a':>10}  {'n/a':>10}")
            continue
        mask = regimes == regime
        if mask.sum() < 5:
            print(f"  {regime:<14}  {'<5d':>10}  {'<5d':>10}  {'<5d':>10}")
            continue
        c_a  = regime_alpha_bps(core_rets, spy_rets, mask) if not core_rets.empty else float("nan")
        e_a  = regime_alpha_bps(exp_ret_series, spy_rets, mask)
        cb_a = regime_alpha_bps(comb_rets, spy_rets, mask)
        flag = ""
        if regime == "bull_calm" and not np.isnan(cb_a) and cb_a < -749:
            flag = "  *** CRITICAL: bull_calm alpha below gate!"
        print(f"  {regime:<14}  {_fmt(c_a,'.2f'):>10}  {_fmt(e_a,'.2f'):>10}  "
              f"{_fmt(cb_a,'.2f'):>10}{flag}")

    # ── 3c. Correlation ───────────────────────────────────────────────────────
    print(f"\n  3c. Correlation matrix (daily returns)")
    corr_ce = float("nan")
    if not core_rets.empty:
        common3 = core_rets.index.intersection(exp_ret_series.index).intersection(spy_rets.index)
        c3 = core_rets.reindex(common3)
        e3 = exp_ret_series.reindex(common3)
        s3 = spy_rets.reindex(common3)
        corr_ce = float(c3.corr(e3))
        corr_cs = float(c3.corr(s3))
        corr_es = float(e3.corr(s3))
        print(f"  Core vs Expanded : {corr_ce:.3f}"
              f"{'  *** HIGH' if corr_ce > 0.70 else ''}")
        print(f"  Core vs SPY      : {corr_cs:.3f}")
        print(f"  Expanded vs SPY  : {corr_es:.3f}")
    else:
        common3 = exp_ret_series.index.intersection(spy_rets.index)
        e3 = exp_ret_series.reindex(common3)
        s3 = spy_rets.reindex(common3)
        corr_es = float(e3.corr(s3))
        print(f"  Core sleeve not available — skipping core correlations")
        print(f"  Expanded vs SPY  : {corr_es:.3f}")

    # ── 3d. Turnover ──────────────────────────────────────────────────────────
    print(f"\n  3d. Turnover diagnostic (expanded sleeve)")
    daily_chg    = exp_sizes.diff().abs().sum(axis=1)
    avg_held     = exp_sizes.sum(axis=1).replace(0, np.nan).mean()
    ann_turnover = float(daily_chg.mean() * 252 / avg_held * 100) if avg_held else 0.0
    print(f"  Annualised turnover: {ann_turnover:.1f}%  (target < 300%)")

    daily_dist = daily_chg[daily_chg > 0]
    if not daily_dist.empty:
        print(f"  Daily turnover: mean {float(daily_dist.mean())*100:.2f}%  "
              f"median {float(daily_dist.median())*100:.2f}%  "
              f"p95 {float(daily_dist.quantile(0.95))*100:.2f}%")
    if ann_turnover > 300:
        print("  *** WARNING: turnover > 300%")
        ticker_turnover = exp_sizes.diff().abs().sum(axis=0).sort_values(ascending=False)
        print("  Top 10 tickers by turnover contribution:")
        for t, v in ticker_turnover.head(10).items():
            print(f"    {t:<8}  cumulative daily weight change: {v:.4f}")

    return dict(
        exp_rets=exp_ret_series,
        core_rets=core_rets,
        comb_rets=comb_rets,
        spy_rets=spy_rets,
        regimes=regimes,
        exp_oos=exp_oos,
        comb_oos=comb_oos,
        ann_turnover=ann_turnover,
        corr_ce=corr_ce,
        exp_oos_sharpe=sharpe(exp_oos),
        comb_sharpe=sharpe(comb_oos),
        comb_max_dd=max_dd(comb_oos),
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4a — Walk-forward OOS validation (7 windows)
# ══════════════════════════════════════════════════════════════════════════════

def step4a_walk_forward(sizes: pd.DataFrame, spy_rets: pd.Series,
                        regimes: pd.Series, alloc_exp: float = 0.50) -> dict:
    section("STEP 4a: Walk-Forward OOS Validation (7 windows, expanded sleeve)")

    windows = [
        ("W1", "2015-01-01", "2018-01-01", "2018-01-01", "2019-01-01"),
        ("W2", "2016-01-01", "2019-01-01", "2019-01-01", "2020-01-01"),
        ("W3", "2017-01-01", "2020-01-01", "2020-01-01", "2021-01-01"),
        ("W4", "2018-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
        ("W5", "2019-01-01", "2022-01-01", "2022-01-01", "2023-01-01"),
        ("W6", "2020-01-01", "2023-01-01", "2023-01-01", "2024-01-01"),
        ("W7", "2021-01-01", "2024-01-01", "2024-01-01", "2025-01-01"),
    ]

    # Build the full expanded return series
    exp_ret_full = _build_exp_returns(sizes, alloc_exp)

    # Previous OOS Sharpe from last run (for comparison column)
    _prev_sharpe = {
        "W1": -0.854, "W2": 1.227, "W3": 0.374,
        "W4": 2.116,  "W5": -0.132, "W6": 0.592, "W7": 1.574,
    }

    print(f"\n  Expanded sleeve return series: {len(exp_ret_full)} days  "
          f"{exp_ret_full.index[0].date()} → {exp_ret_full.index[-1].date()}")
    print(f"\n  {'Window':<6}  {'TestYear':>9}  {'OOS Sharpe':>11}  "
          f"{'BullCalm α(bps)':>16}  {'Prev Sharpe':>12}")
    print("  " + "-"*66)

    wf_results = []
    all_oos_rets = []

    for (wname, train_s, train_e, test_s, test_e) in windows:
        t_start = pd.Timestamp(test_s)
        t_end   = pd.Timestamp(test_e)

        oos = exp_ret_full.loc[t_start:t_end]
        if len(oos) < 20:
            print(f"  {wname:<6}  {test_s[:4]:>9}  {'insufficient data':>11}")
            continue

        sh   = sharpe(oos)
        ar   = annual_return(oos)
        md   = max_dd(oos)

        # Bull_calm alpha for this OOS window
        bc_alpha = float("nan")
        if not regimes.empty and not spy_rets.empty:
            mask     = (regimes == "bull_calm")
            bc_alpha = regime_alpha_bps(oos, spy_rets, mask)

        prev_sh = _prev_sharpe.get(wname, float("nan"))
        bc_str  = f"{bc_alpha:>16.2f}" if not np.isnan(bc_alpha) else f"{'n/a':>16}"
        print(f"  {wname:<6}  {test_s[:4]:>9}  {sh:>11.3f}  "
              f"{bc_str}  {prev_sh:>12.3f}")

        wf_results.append(dict(window=wname, test_year=test_s[:4], sharpe=sh,
                                ann_ret=ar, max_dd=md, bull_calm_alpha=bc_alpha))
        all_oos_rets.append(oos)

    if not wf_results:
        print("  ERROR: no walk-forward windows had sufficient data")
        return {}

    sharpes = [r["sharpe"] for r in wf_results if not np.isnan(r["sharpe"])]
    mean_sh = float(np.mean(sharpes)) if sharpes else float("nan")
    std_sh  = float(np.std(sharpes))  if sharpes else float("nan")
    best    = max(wf_results, key=lambda r: r["sharpe"] if not np.isnan(r["sharpe"]) else -99)
    worst   = min(wf_results, key=lambda r: r["sharpe"] if not np.isnan(r["sharpe"]) else 99)

    print(f"\n  Walk-forward summary:")
    print(f"    Mean OOS Sharpe : {mean_sh:.3f}  (previous: 0.700)")
    print(f"    Std OOS Sharpe  : {std_sh:.3f}  "
          f"{'(*** high variance — regime-dependent)' if std_sh > 0.5 else '(stable)'}")
    print(f"    Best window     : {best['window']} ({best['test_year']})  "
          f"Sharpe={best['sharpe']:.3f}")
    print(f"    Worst window    : {worst['window']} ({worst['test_year']})  "
          f"Sharpe={worst['sharpe']:.3f}")

    # Concatenated OOS return series
    if all_oos_rets:
        oos_concat = pd.concat(all_oos_rets).sort_index().drop_duplicates()
        print(f"\n  Concatenated OOS series: {len(oos_concat)} days  "
              f"Sharpe={sharpe(oos_concat):.3f}  "
              f"AnnRet={annual_return(oos_concat):.2f}%  "
              f"MaxDD={max_dd(oos_concat):.2f}%")
    else:
        oos_concat = pd.Series(dtype=float)

    return dict(wf_results=wf_results, mean_sharpe=mean_sh, std_sharpe=std_sh,
                best=best, worst=worst, oos_concat=oos_concat)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4b — Allocation split optimisation (only if overall PASS)
# ══════════════════════════════════════════════════════════════════════════════

def step4b_alloc_optimise(exp_ret_full: pd.Series, core_rets: pd.Series,
                          spy_rets: pd.Series) -> float:
    section("STEP 4b: Allocation Split Optimisation")
    print("  All gates passed — testing three allocation splits.\n")

    splits = [("60/40 core/exp", 0.40), ("50/50 core/exp", 0.50), ("40/60 core/exp", 0.60)]

    print(f"  {'Split':<18}  {'Combined Sharpe':>15}  {'BullCalm α(bps)':>16}  "
          f"{'Combined MaxDD%':>16}  {'Recommendation'}")
    print("  " + "-"*85)

    best_split = 0.50
    best_sharpe = -999.0

    for (label, exp_frac) in splits:
        core_frac = 1.0 - exp_frac
        if core_rets.empty:
            print(f"  {label:<18}  (core sleeve unavailable — cannot optimise)")
            break
        common = exp_ret_full.index.intersection(core_rets.index)
        comb = (exp_ret_full.reindex(common).fillna(0) +
                core_rets.reindex(common).fillna(0) * core_frac).dropna()
        sh  = sharpe(comb)
        md  = max_dd(comb)
        bc_alpha = float("nan")
        macro_path = MACRO_DIR / "macro_features.parquet"
        if macro_path.exists() and not spy_rets.empty:
            try:
                from v1.regimes.analysis import label_regimes
                macro_df = pd.read_parquet(macro_path)
                vix = macro_df["vix"] if "vix" in macro_df.columns else pd.Series(dtype=float)
                if not vix.empty:
                    regs = label_regimes(vix, spy_rets, comb.index)
                    mask = regs == "bull_calm"
                    bc_alpha = regime_alpha_bps(comb, spy_rets, mask)
            except Exception:
                pass

        dd_ok   = not np.isnan(md) and md > -7.0
        rec     = "✓ viable" if (dd_ok and not np.isnan(sh)) else "✗ DD fail"
        print(f"  {label:<18}  {sh:>15.3f}  "
              f"{bc_alpha:>16.2f}  {md:>16.2f}%  {rec}")
        if not np.isnan(sh) and dd_ok and sh > best_sharpe:
            best_sharpe = sh
            best_split  = exp_frac

    print(f"\n  Recommended expanded allocation: {best_split:.0%}")
    print(f"  (ALLOCATION_SPLIT will be updated in pipeline/portfolio.py)")
    _patch_allocation_split(expanded=best_split)
    return best_split


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Final summary report
# ══════════════════════════════════════════════════════════════════════════════

def step5_summary(ticker_stats: dict, composite: pd.DataFrame,
                  signals: pd.DataFrame, portfolio_data: dict,
                  wf_data: dict, alloc_exp: float) -> None:
    section("STEP 5: Final Summary")
    from v1.pipeline.universe_expansion import NEW_TICKERS, SECTOR_STOCKS
    from v1.portfolio.portfolio import ALLOCATION_SPLIT

    n_core     = 42
    n_expanded = len(NEW_TICKERS)
    n_total    = n_core + n_expanded

    avg_long_pct = float(signals.tail(252).mean().mean() * 100) if not signals.empty else float("nan")

    exp_oos_sharpe = portfolio_data.get("exp_oos_sharpe", float("nan"))
    comb_sharpe    = portfolio_data.get("comb_sharpe",    float("nan"))
    corr_ce        = portfolio_data.get("corr_ce",        float("nan"))
    ann_turnover   = portfolio_data.get("ann_turnover",   float("nan"))
    comb_max_dd_v  = portfolio_data.get("comb_max_dd",    float("nan"))

    wf_mean_sharpe = wf_data.get("mean_sharpe", float("nan")) if wf_data else float("nan")
    wf_std_sharpe  = wf_data.get("std_sharpe",  float("nan")) if wf_data else float("nan")
    wf_best        = wf_data.get("best",  {}) if wf_data else {}
    wf_worst       = wf_data.get("worst", {}) if wf_data else {}

    # Gates (updated thresholds per Step 4 instructions)
    GATE_EXP_WF_SHARPE  = 0.50   # lowered from 0.80 — sleeve adds diversification, not replacement
    GATE_COMB_SHARPE    = 1.40
    GATE_BULL_CALM      = -7.49  # > -7.49 bps/day (combined)
    GATE_CORR           = 0.60
    GATE_TURNOVER       = 300.0
    GATE_MAX_DD         = -7.0

    pass_exp_wf     = not np.isnan(wf_mean_sharpe) and wf_mean_sharpe > GATE_EXP_WF_SHARPE
    pass_comb       = not np.isnan(comb_sharpe)    and comb_sharpe    > GATE_COMB_SHARPE
    pass_corr       = not np.isnan(corr_ce)        and corr_ce        < GATE_CORR
    pass_turnover   = not np.isnan(ann_turnover)   and ann_turnover   < GATE_TURNOVER
    pass_max_dd     = not np.isnan(comb_max_dd_v)  and comb_max_dd_v  > GATE_MAX_DD

    # Bull_calm alpha gate — use portfolio_data regime alpha if available
    regimes  = portfolio_data.get("regimes", pd.Series(dtype=str))
    spy_rets = portfolio_data.get("spy_rets", pd.Series(dtype=float))
    comb_rets = portfolio_data.get("comb_rets", pd.Series(dtype=float))
    bc_alpha = float("nan")
    if not regimes.empty and not spy_rets.empty and not comb_rets.empty:
        mask     = regimes == "bull_calm"
        bc_alpha = regime_alpha_bps(comb_rets, spy_rets, mask)
    pass_bull_calm = not np.isnan(bc_alpha) and bc_alpha > GATE_BULL_CALM

    all_pass = all([pass_exp_wf, pass_comb, pass_corr, pass_turnover, pass_max_dd, pass_bull_calm])

    n_covered = len([t for t in NEW_TICKERS
                     if not ticker_stats.get(t, {}).get("missing")
                     and ticker_stats.get(t, {}).get("coverage", 0) > 0.90])

    exp_data = DATA_DIR / "closes_matrix_expanded.parquet"
    core_data = DATA_DIR / "closes_matrix.parquet"
    exp_shape_str  = "not built"
    core_shape_str = "not built"
    exp_dates_str  = ""
    core_dates_str = ""
    if exp_data.exists():
        df = pd.read_parquet(exp_data)
        df.index = pd.to_datetime(df.index)
        exp_shape_str  = f"{df.shape[0]} rows × {df.shape[1]} cols"
        exp_dates_str  = f"{df.index[0].date()} to {df.index[-1].date()}"
    if core_data.exists():
        df = pd.read_parquet(core_data)
        df.index = pd.to_datetime(df.index)
        core_shape_str  = f"{df.shape[0]} rows × {df.shape[1]} cols"
        core_dates_str  = f"{df.index[0].date()} to {df.index[-1].date()}"

    def _gate_str(ok: bool, value: float, fmt: str, gate: str, unit: str = "") -> str:
        val_s = f"{value:{fmt}}{unit}" if not np.isnan(value) else "n/a"
        return f"{val_s}  [{'PASS' if ok else 'FAIL'}  gate: {gate}]"

    core_alloc_pct = ALLOCATION_SPLIT.get("core", 0.50) * 100
    exp_alloc_pct  = ALLOCATION_SPLIT.get("expanded", 0.50) * 100

    print(f"""
=== EXPANDED UNIVERSE VALIDATION REPORT (FIXED) ===

Universe: {n_core} core + {n_expanded} expanded = {n_total} total
  Beta filtering: REMOVED — replaced with beta-weighted sizing scalars (0.30–1.00)

Data:
  Core closes_matrix:     {core_shape_str}  {core_dates_str}
  Expanded closes_matrix: {exp_shape_str}  {exp_dates_str}
  Coverage > 90% since 2018: {n_covered}/{n_expanded} tickers

Signal Quality:
  Avg % LONG (last 252d):   {avg_long_pct:.1f}%  (target: 30-60%)
  Entry threshold:           0.5 (hysteresis ±0.5)

Walk-Forward OOS (expanded sleeve, 7 windows):
  Mean OOS Sharpe:   {wf_mean_sharpe:.3f}  [{'PASS' if pass_exp_wf else 'FAIL'}  gate: >{GATE_EXP_WF_SHARPE}]
  Std OOS Sharpe:    {wf_std_sharpe:.3f}  ({'stable' if wf_std_sharpe < 0.5 else '*** high variance'})
  Best window:       {wf_best.get('window','n/a')} ({wf_best.get('test_year','n/a')})  Sharpe={wf_best.get('sharpe', float('nan')):.3f}
  Worst window:      {wf_worst.get('window','n/a')} ({wf_worst.get('test_year','n/a')})  Sharpe={wf_worst.get('sharpe', float('nan')):.3f}

Portfolio Metrics (Combined, OOS = last 252d):
  {'Metric':<26}  {'Core Sleeve':>12}  {'Expanded Sleeve':>16}  {'Combined':>12}
  OOS Sharpe                 {_fmt_v(sharpe(portfolio_data.get('core_rets', pd.Series(dtype=float)).iloc[-252:] if len(portfolio_data.get('core_rets', pd.Series(dtype=float))) >= 252 else portfolio_data.get('core_rets', pd.Series(dtype=float)))):>12}  {_fmt_v(portfolio_data.get('exp_oos_sharpe', float('nan'))):>16}  {_fmt_v(comb_sharpe):>12}
  Annual Return %            {_fmt_v(annual_return(portfolio_data.get('core_rets', pd.Series(dtype=float)).iloc[-252:] if len(portfolio_data.get('core_rets', pd.Series(dtype=float))) >= 252 else portfolio_data.get('core_rets', pd.Series(dtype=float)))):>12}  {_fmt_v(annual_return(portfolio_data.get('exp_oos', pd.Series(dtype=float)))):>16}  {_fmt_v(annual_return(portfolio_data.get('comb_oos', pd.Series(dtype=float)))):>12}
  Max Drawdown %             {_fmt_v(max_dd(portfolio_data.get('core_rets', pd.Series(dtype=float)).iloc[-252:] if len(portfolio_data.get('core_rets', pd.Series(dtype=float))) >= 252 else portfolio_data.get('core_rets', pd.Series(dtype=float)))):>12}  {_fmt_v(max_dd(portfolio_data.get('exp_oos', pd.Series(dtype=float)))):>16}  {_fmt_v(comb_max_dd_v):>12}
""")

    print(f"Regime Alpha (bps/day):")
    all_regime_labels = ["bull_calm", "bull_stress", "bear_calm", "bear_stress"]
    print(f"  {'Regime':<14}  {'Core':>10}  {'Expanded':>10}  {'Combined':>10}")
    print("  " + "-"*52)
    core_rets_full  = portfolio_data.get("core_rets", pd.Series(dtype=float))
    exp_rets_full   = portfolio_data.get("exp_rets",  pd.Series(dtype=float))
    for regime in all_regime_labels:
        if regimes.empty:
            print(f"  {regime:<14}  {'n/a':>10}  {'n/a':>10}  {'n/a':>10}")
            continue
        mask = regimes == regime
        c_a  = regime_alpha_bps(core_rets_full, spy_rets, mask) if not core_rets_full.empty else float("nan")
        e_a  = regime_alpha_bps(exp_rets_full,  spy_rets, mask) if not exp_rets_full.empty  else float("nan")
        cb_a = regime_alpha_bps(comb_rets,       spy_rets, mask) if not comb_rets.empty      else float("nan")
        bc_g = f"  (PASS if > {GATE_BULL_CALM})" if regime == "bull_calm" else ""
        print(f"  {regime:<14}  {_fmt_v(c_a,'.2f'):>10}  {_fmt_v(e_a,'.2f'):>10}  "
              f"{_fmt_v(cb_a,'.2f'):>10}{bc_g}")

    print(f"""
Diversification:
  Core-expanded corr: {_gate_str(pass_corr, corr_ce, '.3f', f'<{GATE_CORR}')}

Risk:
  Combined max DD:    {_gate_str(pass_max_dd, comb_max_dd_v, '.2f', f'>{GATE_MAX_DD}', '%')}
  Expanded turnover:  {_gate_str(pass_turnover, ann_turnover, '.1f', f'<{GATE_TURNOVER}', '%')}

Allocation split: core {core_alloc_pct:.0f}% / expanded {exp_alloc_pct:.0f}%

Validation Gates:
  {'Gate':<38}  {'Value':>10}  {'Threshold':>12}  {'Prev Value':>11}  {'Status':>6}
  {'-'*85}
  {'Expanded mean OOS Sharpe (7-window)':<38}  {wf_mean_sharpe:>10.3f}  {f'>{GATE_EXP_WF_SHARPE}':>12}  {'0.700':>11}  {'PASS' if pass_exp_wf else 'FAIL':>6}
  {'Combined OOS Sharpe':<38}  {comb_sharpe:>10.3f}  {f'>{GATE_COMB_SHARPE}':>12}  {'1.482':>11}  {'PASS' if pass_comb else 'FAIL':>6}
  {'Bull_calm alpha (combined, bps/day)':<38}  {bc_alpha:>10.2f}  {f'>{GATE_BULL_CALM}':>12}  {'-22.67':>11}  {'PASS' if pass_bull_calm else 'FAIL':>6}
  {'Core-expanded correlation':<38}  {corr_ce:>10.3f}  {f'<{GATE_CORR}':>12}  {'0.444':>11}  {'PASS' if pass_corr else 'FAIL':>6}
  {'Expanded turnover (%)':<38}  {ann_turnover:>10.1f}  {f'<{GATE_TURNOVER}':>12}  {'691.0':>11}  {'PASS' if pass_turnover else 'FAIL':>6}
  {'Combined max drawdown (%)':<38}  {comb_max_dd_v:>10.2f}  {f'>{GATE_MAX_DD}':>12}  {'-6.47':>11}  {'PASS' if pass_max_dd else 'FAIL':>6}

Overall: {'PASS' if all_pass else 'FAIL'}
""")

    if not all_pass:
        print("  Failing gates:")
        if not pass_exp_wf:
            print(f"    - Expanded mean OOS Sharpe {wf_mean_sharpe:.3f} ≤ {GATE_EXP_WF_SHARPE}")
            print(f"      Diagnosis: check which walk-forward windows are dragging the mean. "
                  f"If worst window is a specific regime year (e.g. 2022 bear), "
                  f"consider regime-conditioning the expanded sleeve entry threshold.")
        if not pass_comb:
            print(f"    - Combined Sharpe {comb_sharpe:.3f} ≤ {GATE_COMB_SHARPE}")
            print(f"      Diagnosis: expanded sleeve returns not high enough to lift combined. "
                  f"Consider reducing expanded allocation to 30-40% to let core sleeve dominate.")
        if not pass_bull_calm:
            print(f"    - Bull_calm alpha {bc_alpha:.2f} ≤ {GATE_BULL_CALM} bps/day")
            print(f"      Diagnosis: defensive sectors (XLU, XLP, XLV) drag in bull_calm. "
                  f"Consider tightening entry for these sectors in low-VIX periods.")
        if not pass_corr:
            print(f"    - Core-expanded ρ {corr_ce:.3f} ≥ {GATE_CORR}")
            print(f"      Diagnosis: expanded universe too correlated to core — reduce "
                  f"overlap in sector ETFs or remove sectors already in core.")
        if not pass_turnover:
            print(f"    - Turnover {ann_turnover:.1f}% ≥ {GATE_TURNOVER}%")
            print(f"      Diagnosis: frozen-weight approach should have reduced this. "
                  f"If still > 300%, check that weekly ffill is active on sizes.")
        if not pass_max_dd:
            print(f"    - Combined max DD {comb_max_dd_v:.2f}% ≤ {GATE_MAX_DD}%")
            print(f"      Diagnosis: reduce expanded allocation or tighten sector caps.")
        print(f"""
  RECOMMENDATION:
    Set USE_EXPANDED_UNIVERSE = False in pipeline/universe_expansion.py
    before live trading until the above gates are resolved.
    All three code fixes (beta sizing, expanded matrix, frozen weights)
    remain in place — do NOT revert them.
""")
    else:
        print(f"""
  All gates passed.
  Expanded universe is VALIDATED for live deployment.
  Recommended allocation: core {core_alloc_pct:.0f}% / expanded {exp_alloc_pct:.0f}%
  (already written to pipeline/portfolio.py ALLOCATION_SPLIT)
""")


def _fmt_v(x, fmt=".3f"):
    return f"{x:{fmt}}" if not np.isnan(x) else "n/a"


# ── Patcher helpers ────────────────────────────────────────────────────────────

def _patch_allocation_split(expanded: float) -> None:
    import re
    path  = Path("pipeline/portfolio.py")
    src   = path.read_text()
    core  = round(1.0 - expanded, 2)
    updated = re.sub(r'("core"\s*:\s*)\d+\.\d+',     f'\\g<1>{core}',     src)
    updated = re.sub(r'("expanded"\s*:\s*)\d+\.\d+',  f'\\g<1>{expanded}', updated)
    path.write_text(updated)
    print(f"  ALLOCATION_SPLIT updated: core={core}, expanded={expanded}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _run_steps_2_to_5(ticker_stats: dict) -> None:
    """Run steps 2–5: signals, portfolio, walk-forward, summary."""
    # Step 2: generate signals (applies both fixes inside signal_generation.py)
    composite, signals, sizes = step2_signals()

    # Step 3: portfolio integration (OOS = last 252 days)
    portfolio_data = step3_portfolio(alloc_expanded=0.50)

    # Step 4a: walk-forward OOS (7 windows)
    spy_rets = portfolio_data.get("spy_rets", pd.Series(dtype=float))
    regimes  = portfolio_data.get("regimes",  pd.Series(dtype=str))
    wf_data  = step4a_walk_forward(sizes, spy_rets, regimes, alloc_exp=0.50)

    # Step 4b: allocation split optimisation (only if all gates pass)
    alloc_exp = 0.50
    if wf_data:
        wf_mean   = wf_data.get("mean_sharpe", float("nan"))
        comb_sh   = portfolio_data.get("comb_sharpe", float("nan"))
        corr_ce   = portfolio_data.get("corr_ce", float("nan"))
        turn      = portfolio_data.get("ann_turnover", float("nan"))
        comb_dd   = portfolio_data.get("comb_max_dd", float("nan"))

        regimes2  = portfolio_data.get("regimes", pd.Series(dtype=str))
        comb_rets = portfolio_data.get("comb_rets", pd.Series(dtype=float))
        bc_alpha  = float("nan")
        if not regimes2.empty and not spy_rets.empty and not comb_rets.empty:
            bc_alpha = regime_alpha_bps(comb_rets, spy_rets, regimes2 == "bull_calm")

        gates_pass = (
            not np.isnan(wf_mean)  and wf_mean  > 0.50 and
            not np.isnan(comb_sh)  and comb_sh  > 1.40 and
            not np.isnan(corr_ce)  and corr_ce  < 0.60 and
            not np.isnan(turn)     and turn     < 300.0 and
            not np.isnan(comb_dd)  and comb_dd  > -7.0  and
            (np.isnan(bc_alpha) or bc_alpha > -7.49)
        )

        if gates_pass:
            exp_ret_full = portfolio_data.get("exp_rets", pd.Series(dtype=float))
            core_rets    = portfolio_data.get("core_rets", pd.Series(dtype=float))
            alloc_exp    = step4b_alloc_optimise(exp_ret_full, core_rets, spy_rets)
            portfolio_data = step3_portfolio(alloc_expanded=alloc_exp)
        else:
            section("STEP 4b: Allocation Split Optimisation")
            print("  Gates not all passing — skipping allocation optimisation.")
            print("  Fix failing gates first, then run optimisation.")

    # Step 5: final summary report (with Prev Value column)
    step5_summary(ticker_stats, composite, signals, portfolio_data, wf_data, alloc_exp)


def main_from_step2() -> None:
    """Skip steps 0–1 (data already built). Run steps 2–5 only."""
    print("=" * 72)
    print("  EXPANDED UNIVERSE — Fix Pass (steps 2–5)")
    print("  Fix 1: fixed-sector-budget-to-cash sizing")
    print("  Fix 2: bull_calm absolute momentum entry gate")
    print("=" * 72)
    # Minimal ticker_stats (beta info not needed for steps 2-5)
    from v1.pipeline.universe_expansion import NEW_TICKERS, SECTOR_MAP
    ticker_stats = {t: dict(beta=float("nan"), bear_beta=float("nan"),
                            adv_m=0.0, coverage=1.0, missing=False,
                            sector=SECTOR_MAP.get(t, "?"))
                    for t in NEW_TICKERS}
    _run_steps_2_to_5(ticker_stats)


def main() -> None:
    print("=" * 72)
    print("  EXPANDED UNIVERSE — End-to-end Validation, Diagnostic & Fix Pass")
    print("  (Beta-weighted sizing, expanded closes matrix, fixed-budget sizing)")
    print("=" * 72)

    # Step 0: build expanded closes matrix
    step0_build_expanded_matrix()

    # Step 0b: engineer features for any missing tickers
    step0b_features()

    # Step 1: data integrity (informational only — no hard removal)
    ticker_stats = step1_data_integrity()

    _run_steps_2_to_5(ticker_stats)


if __name__ == "__main__":
    main_from_step2()
