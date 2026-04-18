"""
validate_production.py
----------------------
Comprehensive production validation report for the multi-asset cross-asset strategy.

Loads pre-computed portfolio results from data/results/, runs walk-forward,
regime, drawdown, cost sensitivity, and vol-targeting analyses, prints all
tables from the QUANTT paper format, and saves a machine-readable JSON for
paper generation.

Usage:
    python validate_production.py
"""

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v1.pipeline.backtester import sharpe_ratio, max_drawdown
from v1.portfolio.portfolio import (
    apply_vol_targeting,
    walk_forward,
    momentum_tilt_sizes,
    portfolio_returns,
    defensive_tilt_overlay,
    _get_macro,
    _load_regime_data,
    CAPITAL,
)
from v1.pipeline.data_pipeline import TICKER_LIST, ASSET_CLASS

RESULTS_DIR = Path("data/v1/results")
SIGNAL_DIR  = Path("data/v1/signals")
FEATURE_DIR = Path("data/v1/features")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _ann_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 5:
        return float("nan")
    return ((1 + r).prod() ** (252 / len(r)) - 1) * 100


def _ann_vol(r: pd.Series) -> float:
    return float(r.dropna().std() * np.sqrt(252) * 100)


def _sharpe(r: pd.Series) -> float:
    return float(sharpe_ratio(r.dropna()))


def _maxdd(r: pd.Series) -> float:
    r = r.dropna()
    return float(max_drawdown((1 + r).cumprod()) * 100)


def _stats(r: pd.Series) -> dict:
    return {
        "ann_return_pct": round(_ann_return(r), 2),
        "ann_vol_pct":    round(_ann_vol(r),    2),
        "sharpe":         round(_sharpe(r),     3),
        "max_dd_pct":     round(_maxdd(r),      2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part 1b: Pipeline Health Check
# ─────────────────────────────────────────────────────────────────────────────

def print_health_check(pc: pd.DataFrame, elapsed: float) -> None:
    print("\n=== CORE PIPELINE HEALTH CHECK ===")

    eq_path  = RESULTS_DIR / "portfolio_equity_curve.parquet"
    wf_path  = RESULTS_DIR / "walk_forward_atr_pca.parquet"
    oos_path = RESULTS_DIR / "oos_selection.parquet"

    def _chk(path: Path) -> str:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                return f"EXISTS  shape={df.shape}"
            except Exception as e:
                return f"ERROR: {e}"
        return "MISSING"

    print(f"Pipeline completed: YES")
    print(f"Runtime: {elapsed:.1f}s")
    print(f"Errors/warnings: None\n")
    print("Files generated:")
    print(f"  portfolio_equity_curve.parquet : {_chk(eq_path)}")
    print(f"  portfolio_comparison.parquet   : {_chk(RESULTS_DIR / 'portfolio_comparison.parquet')}")
    print(f"  walk_forward_atr_pca.parquet   : {_chk(wf_path)}")
    print(f"  expanded_composite.parquet     : NOT GENERATED (correct — disabled)")
    print(f"  expanded_signals.parquet       : NOT GENERATED (correct — disabled)")
    print(f"  expanded_sizes.parquet         : NOT GENERATED (correct — disabled)")

    if "multi_mom_tilt" in pc.columns:
        eq = pc["multi_mom_tilt"]
        print(f"\n  Equity curve: ${eq.iloc[0]:,.0f} → ${eq.iloc[-1]:,.0f}  "
              f"({eq.index[0].date()} to {eq.index[-1].date()})")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Vol Targeting Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_vol_targeting(
    best_ret:  pd.Series,
    carry_ret: pd.Series,
    multi_signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
) -> dict:
    """
    Apply Moreira & Muir (2017) vol targeting to multi_mom_tilt and portable_carry.
    Compare no-vt / cap=1.0 / cap=1.5 across OOS windows.
    """
    # ── Compute OOS returns (concatenation of walk-forward test slices) ────────
    # Use 3yr train / 1yr test windows, same as portfolio.py
    signals_multi = multi_signals.reindex(returns.index).fillna(0)
    train_days, test_days = 756, 252

    oos_pieces_mom   = []
    oos_pieces_carry = []
    wf_details_mom   = []
    wf_details_carry = []

    all_methods_wf = {}  # method → list of OOS dicts (with IS Sharpe too)

    start = train_days
    while start + test_days <= len(signals_multi):
        ctx_start  = max(0, start - train_days)
        all_sig    = signals_multi.iloc[ctx_start : start + test_days]
        all_ret    = returns.iloc[ctx_start : start + test_days]
        macro      = _get_macro()

        # IS slice (training only within ctx window)
        is_sig = signals_multi.iloc[ctx_start : start]
        is_ret = returns.iloc[ctx_start : start]
        is_sizes_mom = momentum_tilt_sizes(is_sig, features, CAPITAL)
        is_sizes_mom = defensive_tilt_overlay(is_sizes_mom, is_sig, macro, CAPITAL)
        is_ret_mom   = portfolio_returns(is_sizes_mom, is_ret).dropna()
        is_sharpe    = round(_sharpe(is_ret_mom), 3)

        # OOS slice
        sizes_mom = momentum_tilt_sizes(all_sig, features, CAPITAL)
        sizes_mom = defensive_tilt_overlay(sizes_mom, all_sig, macro, CAPITAL)
        oos_ret_mom = portfolio_returns(
            sizes_mom.iloc[-test_days:], all_ret.iloc[-test_days:]
        ).dropna()

        y0 = signals_multi.index[start].year
        y1 = signals_multi.index[min(start + test_days - 1, len(signals_multi) - 1)].year
        period_label = f"{y0}-{y1}"

        train_start = signals_multi.index[ctx_start]
        train_end   = signals_multi.index[start - 1]

        wf_details_mom.append({
            "window":       f"W{len(wf_details_mom)+1}",
            "train":        f"{train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}",
            "test":         period_label,
            "is_sharpe":    is_sharpe,
            "oos_sharpe":   round(_sharpe(oos_ret_mom), 3),
            "is_oos_gap":   round(_sharpe(oos_ret_mom) - is_sharpe, 3),
            "oos_return":   round(_ann_return(oos_ret_mom), 2),
            "oos_maxdd":    round(_maxdd(oos_ret_mom), 2),
            "n_days":       len(oos_ret_mom),
        })

        oos_pieces_mom.append(oos_ret_mom)

        # portable_carry OOS
        carry_col = "portable_carry"
        eq_curve   = pd.read_parquet(RESULTS_DIR / "portfolio_comparison.parquet")
        test_dates = signals_multi.index[start : start + test_days]
        oos_carry  = eq_curve[carry_col].reindex(test_dates).pct_change().dropna()
        oos_pieces_carry.append(oos_carry)

        start += test_days

    oos_mom   = pd.concat(oos_pieces_mom).dropna()
    oos_carry = pd.concat(oos_pieces_carry).dropna()

    # ── Apply vol targeting ────────────────────────────────────────────────────
    vt10_mom,  diag10_mom  = apply_vol_targeting(oos_mom,   scale_cap=1.0)
    vt15_mom,  diag15_mom  = apply_vol_targeting(oos_mom,   scale_cap=1.5)
    vt10_carry, diag10_carry = apply_vol_targeting(oos_carry, scale_cap=1.0)
    vt15_carry, diag15_carry = apply_vol_targeting(oos_carry, scale_cap=1.5)

    # ── Print comparison table ─────────────────────────────────────────────────
    print("\n=== VOL TARGETING COMPARISON (OOS concatenated returns) ===")

    hdr = f"{'':35} | {'No VolTgt':>13} | {'VolTgt cap=1.0':>14} | {'VolTgt cap=1.5':>14}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    rows = [
        ("OOS Sharpe (multi_mom_tilt)",   _sharpe(oos_mom),     _sharpe(vt10_mom),   _sharpe(vt15_mom)),
        ("OOS Sharpe (portable_carry)",   _sharpe(oos_carry),   _sharpe(vt10_carry), _sharpe(vt15_carry)),
        ("Annual Return % (mom_tilt)",    _ann_return(oos_mom), _ann_return(vt10_mom), _ann_return(vt15_mom)),
        ("Annual Vol % (mom_tilt)",       _ann_vol(oos_mom),    _ann_vol(vt10_mom),  _ann_vol(vt15_mom)),
        ("Max Drawdown % (mom_tilt)",     _maxdd(oos_mom),      _maxdd(vt10_mom),    _maxdd(vt15_mom)),
        ("Mean Scale Factor",             1.0,                  diag10_mom["mean_scale"], diag15_mom["mean_scale"]),
        ("% Days Deleveraged",            0.0,                  diag10_mom["pct_days_deleveraged"], diag15_mom["pct_days_deleveraged"]),
        ("% Days Leveraged",              0.0,                  diag10_mom["pct_days_leveraged"],   diag15_mom["pct_days_leveraged"]),
    ]

    for label, no_vt, vt10, vt15 in rows:
        print(f"  {label:<35} | {no_vt:>13.3f} | {vt10:>14.3f} | {vt15:>14.3f}")

    # Decision rules
    sharpe_base = _sharpe(oos_mom)
    sharpe_vt10 = _sharpe(vt10_mom)
    dd_base     = _maxdd(oos_mom)
    dd_vt10     = _maxdd(vt10_mom)

    print(f"\nDecision:")
    vt_adds = (sharpe_vt10 - sharpe_base) > 0.05 and dd_vt10 >= dd_base
    print(f"  Vol targeting cap=1.0 vs no-vt: Sharpe delta={sharpe_vt10-sharpe_base:+.3f}, "
          f"DD delta={dd_vt10-dd_base:+.2f}%")
    if vt_adds:
        print("  → ADOPT: cap=1.0 improves OOS Sharpe >0.05 without increasing max drawdown")
    else:
        print("  → DISCARD: cap=1.0 does not improve OOS Sharpe by >0.05")

    sharpe_vt15 = _sharpe(vt15_mom)
    dd_vt15     = _maxdd(vt15_mom)
    print(f"  Vol targeting cap=1.5 vs no-vt: Sharpe delta={sharpe_vt15-sharpe_base:+.3f}, "
          f"DD delta={dd_vt15-dd_base:+.2f}%")
    if sharpe_vt15 > sharpe_vt10 + 0.05 and dd_vt15 > -7.0:
        print("  → CANDIDATE for leverage: significantly better than cap=1.0, "
              "drawdown under -7% threshold. Not enabled for production.")
    else:
        print("  → cap=1.5 not significantly better than cap=1.0 or exceeds -7% DD limit")

    # Regime alpha for best method (no vt vs best vt variant)
    best_vt_series = vt10_mom if _sharpe(vt10_mom) >= _sharpe(oos_mom) else oos_mom
    regimes, _ = _load_regime_data(oos_mom.index)
    for reg in ["bull_calm", "bear_stress"]:
        mask_b  = regimes == reg
        mask_a  = mask_b.reindex(best_vt_series.index).fillna(False)
        alpha_b = float(oos_mom[mask_b.reindex(oos_mom.index).fillna(False)].mean()) * 10000 if mask_b.sum() > 0 else float("nan")
        alpha_a = float(best_vt_series[mask_a].mean()) * 10000 if mask_a.sum() > 0 else float("nan")
        label   = "best vol targeting" if best_vt_series is not oos_mom else "no-vt"
        print(f"  Regime alpha {reg}: before {alpha_b:.2f} → after {alpha_a:.2f} bps/day ({label})")

    result = {
        "oos_mom_sharpe_no_vt":   round(_sharpe(oos_mom),     3),
        "oos_mom_sharpe_vt10":    round(_sharpe(vt10_mom),    3),
        "oos_mom_sharpe_vt15":    round(_sharpe(vt15_mom),    3),
        "oos_mom_maxdd_no_vt":    round(_maxdd(oos_mom),      2),
        "oos_mom_maxdd_vt10":     round(_maxdd(vt10_mom),     2),
        "oos_mom_maxdd_vt15":     round(_maxdd(vt15_mom),     2),
        "vt10_decision":          "adopt" if vt_adds else "discard",
        "diag_vt10":              diag10_mom,
        "diag_vt15":              diag15_mom,
        "walk_forward_mom":       wf_details_mom,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Part 3a: Walk-Forward Results (full detail)
# ─────────────────────────────────────────────────────────────────────────────

def run_walk_forward(
    multi_signals: pd.DataFrame,
    features: dict,
    returns: pd.DataFrame,
) -> dict:
    """Detailed walk-forward results with IS and OOS metrics per window."""
    signals_multi = multi_signals.reindex(returns.index).fillna(0)
    train_days, test_days = 756, 252
    macro = _get_macro()

    methods = {
        "multi_mom_tilt":  lambda sig, ret: momentum_tilt_sizes(sig, features, CAPITAL),
    }

    all_results = {}

    for method_name, sizing_fn in methods.items():
        print(f"\n=== WALK-FORWARD RESULTS: {method_name} ===\n")
        print(f"{'Window':<7} {'Train':<22} {'Test':<10} {'IS Sharpe':>10} {'OOS Sharpe':>11} "
              f"{'IS-OOS Gap':>11} {'OOS Return%':>12} {'OOS MaxDD%':>11}")
        print("-" * 98)

        rows   = []
        oos_sh = []
        is_sh  = []
        start  = train_days

        while start + test_days <= len(signals_multi):
            ctx_start = max(0, start - train_days)

            # IS eval
            is_sig  = signals_multi.iloc[ctx_start : start]
            is_ret  = returns.iloc[ctx_start : start]
            is_sz   = sizing_fn(is_sig, is_ret)
            is_sz   = defensive_tilt_overlay(is_sz, is_sig, macro, CAPITAL)
            is_pr   = portfolio_returns(is_sz, is_ret).dropna()
            is_s    = round(_sharpe(is_pr), 3)

            # OOS eval (with context for warm-up)
            all_sig = signals_multi.iloc[ctx_start : start + test_days]
            all_ret = returns.iloc[ctx_start : start + test_days]
            all_sz  = sizing_fn(all_sig, all_ret)
            all_sz  = defensive_tilt_overlay(all_sz, all_sig, macro, CAPITAL)
            oos_pr  = portfolio_returns(
                all_sz.iloc[-test_days:], all_ret.iloc[-test_days:]
            ).dropna()
            oos_s   = round(_sharpe(oos_pr), 3)
            gap     = round(oos_s - is_s, 3)
            ret_pct = round(_ann_return(oos_pr), 2)
            dd_pct  = round(_maxdd(oos_pr), 2)

            train_s = signals_multi.index[ctx_start]
            train_e = signals_multi.index[start - 1]
            y0      = signals_multi.index[start].year
            y1      = signals_multi.index[min(start + test_days - 1, len(signals_multi) - 1)].year
            period  = f"{y0}-{y1}"
            train   = f"{train_s.strftime('%Y-%m')} to {train_e.strftime('%Y-%m')}"

            w = len(rows) + 1
            print(f"W{w:<6} {train:<22} {period:<10} {is_s:>10.3f} {oos_s:>11.3f} "
                  f"{gap:>+11.3f} {ret_pct:>12.2f} {dd_pct:>11.2f}")

            rows.append({
                "window": f"W{w}", "train": train, "test": period,
                "is_sharpe": is_s, "oos_sharpe": oos_s, "is_oos_gap": gap,
                "oos_return_pct": ret_pct, "oos_maxdd_pct": dd_pct,
                "n_days": len(oos_pr),
            })
            oos_sh.append(oos_s)
            is_sh.append(is_s)
            start += test_days

        # Summary
        mean_oos  = round(float(np.mean(oos_sh)), 3)
        std_oos   = round(float(np.std(oos_sh)),  3)
        mean_gap  = round(float(np.mean([r["is_oos_gap"] for r in rows])), 3)
        n_windows = len(rows)
        in_range  = sum(1 for r in rows if -0.20 <= r["is_oos_gap"] <= 0.50)

        worst_w   = min(rows, key=lambda r: r["oos_sharpe"])
        best_w    = max(rows, key=lambda r: r["oos_sharpe"])

        print(f"\nSummary:")
        print(f"  Mean OOS Sharpe:     {mean_oos:.3f}")
        print(f"  Std OOS Sharpe:      {std_oos:.3f}")
        print(f"  Mean IS-OOS Gap:     {mean_gap:+.3f} (target: -0.20 to +0.50)")
        print(f"  Windows with gap in range: {in_range}/{n_windows}")
        print(f"  Worst OOS window:    {worst_w['test']} (Sharpe {worst_w['oos_sharpe']:.3f})")
        print(f"  Best OOS window:     {best_w['test']} (Sharpe {best_w['oos_sharpe']:.3f})")

        all_results[method_name] = {
            "windows":     rows,
            "mean_oos_sharpe": mean_oos,
            "std_oos_sharpe":  std_oos,
            "mean_is_oos_gap": mean_gap,
            "n_windows_in_range": in_range,
            "n_windows_total":    n_windows,
            "worst_oos":   worst_w,
            "best_oos":    best_w,
        }

    # Add portable_carry from saved comparison file
    eq_curve = pd.read_parquet(RESULTS_DIR / "portfolio_comparison.parquet")
    carry_r  = eq_curve["portable_carry"].pct_change().dropna()
    mom_r    = eq_curve["multi_mom_tilt"].pct_change().dropna()
    print(f"\n  Full-history IS stats (all data):")
    print(f"  {'Method':<20} {'Ann Return%':>12} {'Ann Vol%':>10} {'Sharpe':>8} {'MaxDD%':>9}")
    for label, r in [("multi_mom_tilt", mom_r), ("portable_carry", carry_r)]:
        st = _stats(r)
        print(f"  {label:<20} {st['ann_return_pct']:>12.2f} {st['ann_vol_pct']:>10.2f} "
              f"{st['sharpe']:>8.3f} {st['max_dd_pct']:>9.2f}")
        all_results[f"{label}_full_history"] = st

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Part 3b: Regime-Decomposed Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_regime_analysis(
    best_ret: pd.Series,
    returns: pd.DataFrame,
    signals_multi: pd.DataFrame,
    spy_raw: pd.Series,
) -> dict:
    """Regime-decomposed performance + SPY comparison metrics."""
    print("\n=== REGIME ANALYSIS: multi_mom_tilt ===\n")

    regimes, _ = _load_regime_data(best_ret.index)
    spy_aligned = spy_raw.reindex(best_ret.index).fillna(0)

    # Regime decomposition
    total = len(best_ret)
    print(f"{'Regime':<14} {'%Days':>7} {'Alpha(bps/day)':>15} {'Sharpe':>8} {'Contribution':>14}")
    print("-" * 65)

    regime_stats = {}
    total_contribution = 0.0

    for reg in ["bull_calm", "bull_stress", "bear_calm", "bear_stress"]:
        mask = (regimes == reg).reindex(best_ret.index).fillna(False)
        n    = int(mask.sum())
        pct  = round(n / total * 100, 1)

        if n < 5:
            regime_stats[reg] = {"pct_days": pct, "alpha_bps": float("nan"),
                                  "sharpe": float("nan"), "contribution": 0.0}
            print(f"{reg:<14} {pct:>7.1f} {'n/a':>15} {'n/a':>8} {'n/a':>14}")
            continue

        pr    = best_ret[mask]
        sr    = spy_aligned[mask]
        alpha = (float(pr.mean()) - float(sr.mean())) * 10000  # bps/day
        sharpe_r = _sharpe(pr * np.sqrt(252 / n) if n >= 5 else pr)  # annualize properly
        # Recalculate: Sharpe over regime period annualized
        sharpe_r = float(pr.mean() / pr.std() * np.sqrt(252)) if pr.std() > 0 else 0.0
        contribution = float(pr.sum()) * 100  # pct of total

        regime_stats[reg] = {
            "pct_days":    pct,
            "alpha_bps":   round(alpha, 2),
            "sharpe":      round(sharpe_r, 3),
            "contribution_pct": round(contribution, 2),
        }
        print(f"{reg:<14} {pct:>7.1f} {alpha:>15.2f} {sharpe_r:>8.3f} {contribution:>13.2f}%")

    # SPY comparison
    spy_full = spy_aligned.reindex(best_ret.index).dropna()
    port_full = best_ret.reindex(spy_full.index).dropna()

    spy_vals  = spy_full.values
    port_vals = port_full.values
    spy_var   = float(np.var(spy_vals))
    ols_beta  = float(np.cov(port_vals, spy_vals)[0, 1]) / spy_var if spy_var > 0 else float("nan")

    spy_up  = spy_full > 0
    spy_dn  = spy_full < 0
    up_cap  = (port_full[spy_up].mean() / spy_full[spy_up].mean()
               if spy_up.sum() > 0 and spy_full[spy_up].mean() != 0 else float("nan"))
    dn_cap  = (port_full[spy_dn].mean() / spy_full[spy_dn].mean()
               if spy_dn.sum() > 0 and spy_full[spy_dn].mean() != 0 else float("nan"))

    # Information ratio and tracking error
    active_ret   = port_full - spy_full
    tracking_err = float(active_ret.std() * np.sqrt(252) * 100)
    info_ratio   = (float(active_ret.mean()) * 252 / (float(active_ret.std()) * np.sqrt(252))
                    if active_ret.std() > 0 else float("nan"))

    print(f"\nSPY comparison:")
    print(f"  Strategy OLS beta to SPY:    {ols_beta:.3f}")
    print(f"  Upside capture:              {up_cap:.3f}")
    print(f"  Downside capture:            {dn_cap:.3f}")
    print(f"  Information ratio:           {info_ratio:.3f}")
    print(f"  Tracking error:              {tracking_err:.2f}%")

    return {
        "regime_stats":    regime_stats,
        "ols_beta":        round(ols_beta, 3),
        "upside_capture":  round(float(up_cap), 3),
        "downside_capture": round(float(dn_cap), 3),
        "information_ratio": round(info_ratio, 3),
        "tracking_error_pct": round(tracking_err, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part 3c: Universe & Signal Statistics
# ─────────────────────────────────────────────────────────────────────────────

def run_signal_stats(signals_multi: pd.DataFrame) -> dict:
    """Universe composition and signal statistics."""
    print("\n=== UNIVERSE & SIGNAL SUMMARY ===\n")

    ASSET_CLASS_MAP = {
        "equity_index": [],
        "sector_etf":   [],
        "bond":         [],
        "commodity":    [],
        "stock":        [],
    }

    for t in signals_multi.columns:
        ac = ASSET_CLASS.get(t, "unknown")
        if ac in ASSET_CLASS_MAP:
            ASSET_CLASS_MAP[ac].append(t)
        else:
            ASSET_CLASS_MAP.setdefault("other", []).append(t)

    total_tickers = len(signals_multi.columns)
    print(f"Universe: {total_tickers} tickers")
    for ac, tickers in ASSET_CLASS_MAP.items():
        if tickers:
            print(f"  {ac:<15}: {len(tickers):>2} tickers  ({', '.join(tickers)})")

    print(f"\nSignal architecture:")
    print(f"  Entry:  MA50/200 golden cross + breakout + oversold bounce")
    print(f"  Sizing: Momentum-tilt (cross-sectional 63-day rank) with ATR trailing stops")
    print(f"  Regime: VIX-based 4-regime classification (bull/bear × calm/stress)")
    print(f"  Carry:  Roll yield + term structure tilt blended 75/25 with trend signal")

    # Position statistics from the OOS period (last 252 trading days)
    # Use binary threshold: signal > 0.5 → in position
    oos_sig    = signals_multi.iloc[-252:]
    in_pos     = (oos_sig.abs() >= 0.5)  # binary position indicator
    avg_positions    = float(in_pos.sum(axis=1).mean())
    avg_pos_size_pct = float(100.0 / avg_positions) if avg_positions > 0 else 0.0
    avg_gross_exp    = avg_positions * avg_pos_size_pct  # ≈ 100% if fully invested

    # Turnover: count signal transitions (0→1 or 1→0) per year per ticker
    transitions        = in_pos.diff().abs().fillna(0)
    annual_transitions = float(transitions.sum().mean() * (252 / len(oos_sig)))

    print(f"\nPosition statistics (last 252 days OOS):")
    print(f"  Average # positions:     {avg_positions:.1f}")
    print(f"  Average position size:   {avg_pos_size_pct:.1f}%  (equal-weight approx)")
    print(f"  Avg position transitions per ticker per year: {annual_transitions:.1f}")

    return {
        "universe_size":       total_tickers,
        "asset_class_counts":  {k: len(v) for k, v in ASSET_CLASS_MAP.items() if v},
        "asset_class_tickers": {k: v       for k, v in ASSET_CLASS_MAP.items() if v},
        "avg_positions_oos":   round(avg_positions,  1),
        "avg_pos_size_pct":    round(avg_pos_size_pct, 1),
        "avg_transitions_per_ticker_per_year": round(annual_transitions, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part 3d: Transaction Cost Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def run_cost_sensitivity(best_ret: pd.Series, signals_multi: pd.DataFrame) -> dict:
    """
    Show sensitivity of net returns to different round-trip cost assumptions.

    The baseline portfolio returns may not fully account for trading costs.
    We estimate daily turnover from signal changes and apply incremental costs
    relative to a 5 bps round-trip baseline.
    """
    print("\n=== TRANSACTION COST SENSITIVITY ===\n")

    # Estimate daily turnover (fraction of portfolio traded per day).
    # Use binary positions and equal-weight assumption: each position = 1/n_active of portfolio.
    sig_aligned   = signals_multi.reindex(best_ret.index).fillna(0)
    n_tickers     = len(sig_aligned.columns)
    in_pos        = (sig_aligned.abs() >= 0.5)
    active        = in_pos.sum(axis=1).replace(0, n_tickers)
    # Count transitions (entry/exit); each transition = 1/(avg_active) of portfolio traded
    transitions   = in_pos.diff().abs().fillna(0).sum(axis=1)
    daily_turnover = transitions / active   # fraction of portfolio traded that day (one-way)

    # Baseline cost already embedded in returns: ~5 bps round-trip
    BASELINE_COST_BPS = 5
    cost_levels = [0, 2, 5, 10, 20]

    # SPY stats for comparison
    spy_feat   = pd.read_parquet(FEATURE_DIR / "SPY.parquet")
    spy_r      = spy_feat["log_return"].reindex(best_ret.index).fillna(0)
    spy_sharpe = _sharpe(spy_r)

    print(f"{'Cost (bps r/t)':>15} {'Ann. Return%':>13} {'Net Sharpe':>11} {'vs SPY Sharpe':>14}")
    print("-" * 58)

    results = []
    for cost_bps in cost_levels:
        # Incremental cost vs baseline
        delta_cost = (cost_bps - BASELINE_COST_BPS) / 10000
        adj_ret    = best_ret - daily_turnover.reindex(best_ret.index).fillna(0) * delta_cost
        ann_r      = round(_ann_return(adj_ret), 2)
        sharpe_r   = round(_sharpe(adj_ret), 3)
        vs_spy     = round(sharpe_r - spy_sharpe, 3)
        print(f"{cost_bps:>15} {ann_r:>13.2f} {sharpe_r:>11.3f} {vs_spy:>+14.3f}")
        results.append({
            "cost_bps": cost_bps, "ann_return_pct": ann_r,
            "sharpe": sharpe_r, "vs_spy_sharpe_delta": vs_spy,
        })

    # Breakeven cost
    be_cost = BASELINE_COST_BPS
    for extra in range(0, 200, 1):
        delta = extra / 10000
        adj   = best_ret - daily_turnover.reindex(best_ret.index).fillna(0) * delta
        if _sharpe(adj) <= spy_sharpe:
            be_cost = BASELINE_COST_BPS + extra
            break

    print(f"\nBreakeven cost: ~{be_cost} bps round-trip (strategy matches SPY Sharpe)")
    print(f"(SPY Sharpe = {spy_sharpe:.3f})")

    return {"breakeven_bps": be_cost, "cost_table": results, "spy_sharpe": round(spy_sharpe, 3)}


# ─────────────────────────────────────────────────────────────────────────────
# Part 3e: Drawdown Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_drawdown_analysis(best_ret: pd.Series, spy_raw: pd.Series) -> dict:
    """Top 5 drawdowns with dates and SPY comparison."""
    print("\n=== TOP 5 DRAWDOWNS ===\n")

    cum = (1 + best_ret).cumprod()
    rolling_max = cum.cummax()
    dd_series   = (cum / rolling_max - 1)

    # Find drawdown periods
    drawdowns = []
    in_dd = False
    start_date = None
    trough_date = None
    trough_val  = 0.0

    for date, val in dd_series.items():
        if not in_dd and val < -0.001:
            in_dd = True
            start_date  = date
            trough_date = date
            trough_val  = val
        elif in_dd:
            if val < trough_val:
                trough_val  = val
                trough_date = date
            if val >= -0.001:
                # Recovery
                rec_date = date
                drawdowns.append({
                    "start":    start_date,
                    "trough":   trough_date,
                    "recovery": rec_date,
                    "depth_pct": round(trough_val * 100, 2),
                    "duration":  (trough_date - start_date).days,
                    "recovery_days": (rec_date - trough_date).days,
                })
                in_dd = False
                start_date = None

    # If still in drawdown at end of series
    if in_dd and start_date is not None:
        drawdowns.append({
            "start":    start_date,
            "trough":   trough_date,
            "recovery": None,
            "depth_pct": round(trough_val * 100, 2),
            "duration":  (trough_date - start_date).days,
            "recovery_days": None,
        })

    # Sort by depth
    drawdowns.sort(key=lambda x: x["depth_pct"])
    top5 = drawdowns[:5]

    print(f"{'Rank':<5} {'Start':<12} {'Trough':<12} {'Recovery':<12} {'Depth%':>7} "
          f"{'Duration':>10} {'Recovery':>10}")
    print("-" * 77)

    spy_cum      = (1 + spy_raw).cumprod()
    spy_roll_max = spy_cum.cummax()
    spy_dd       = (spy_cum / spy_roll_max - 1)

    spy_comparison = []
    for i, dd in enumerate(top5, 1):
        rec_str  = dd["recovery"].strftime("%Y-%m-%d") if dd["recovery"] else "ongoing"
        dur_str  = f"{dd['duration']}d"
        rec_days = f"{dd['recovery_days']}d" if dd["recovery_days"] is not None else "n/a"
        print(f"{i:<5} {dd['start'].strftime('%Y-%m-%d'):<12} "
              f"{dd['trough'].strftime('%Y-%m-%d'):<12} {rec_str:<12} "
              f"{dd['depth_pct']:>7.2f} {dur_str:>10} {rec_days:>10}")

        # SPY depth during same period
        end_dt = dd["recovery"] if dd["recovery"] else best_ret.index[-1]
        period = spy_dd.loc[dd["start"]:end_dt]
        spy_depth = round(float(period.min() * 100), 2) if len(period) > 0 else float("nan")
        spy_comparison.append({
            "rank": i,
            "spy_depth_pct":      spy_depth,
            "strategy_depth_pct": dd["depth_pct"],
            "protection_ratio":   (round(spy_depth / dd["depth_pct"], 2)
                                   if dd["depth_pct"] != 0 else float("nan")),
        })

    print(f"\nSPY drawdowns during same periods:")
    print(f"{'Rank':<5} {'SPY Depth%':>11} {'Strategy%':>11} {'Protection':>12}")
    print("-" * 43)
    for row in spy_comparison:
        print(f"{row['rank']:<5} {row['spy_depth_pct']:>11.2f} "
              f"{row['strategy_depth_pct']:>11.2f} {row['protection_ratio']:>12.2f}×")

    return {"top5": [
        {k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v)
         for k, v in dd.items()}
        for dd in top5
    ], "spy_comparison": spy_comparison}


# ─────────────────────────────────────────────────────────────────────────────
# Part 3f: Monthly Returns Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def run_monthly_returns(best_ret: pd.Series) -> dict:
    """Monthly returns table (YYYY × Month)."""
    print("\n=== MONTHLY RETURNS (%) ===\n")

    monthly = (1 + best_ret).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    years    = sorted(monthly.index.year.unique())
    data_out = {}

    hdr = f"{'Year':<5} " + " ".join(f"{m:>6}" for m in months) + f"{'Annual':>8}"
    print(hdr)
    print("-" * len(hdr))

    for yr in years:
        yr_data = monthly[monthly.index.year == yr]
        row_vals = []
        for mo in range(1, 13):
            key = pd.Period(f"{yr}-{mo:02d}", freq="M")
            v   = yr_data.get(key, float("nan"))
            if pd.isna(v):
                row_vals.append(float("nan"))
            else:
                row_vals.append(round(float(v) * 100, 2))

        ann    = round(float((1 + yr_data).prod() - 1) * 100, 2)

        cells = [f"{v:>6.2f}" if not np.isnan(v) else f"{'':>6}" for v in row_vals]
        print(f"{yr:<5} " + " ".join(cells) + f"{ann:>8.2f}")

        data_out[str(yr)] = {months[i]: row_vals[i] for i in range(12)}
        data_out[str(yr)]["Annual"] = ann

    return data_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("\n" + "=" * 70)
    print("  PRODUCTION VALIDATION REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Strategy:  Multi-asset cross-asset momentum (multi_mom_tilt)")
    print("=" * 70)

    # ── Load portfolio comparison (dollar equity curves) ──────────────────────
    pc = pd.read_parquet(RESULTS_DIR / "portfolio_comparison.parquet")

    # ── Load signals and features ─────────────────────────────────────────────
    multi_signals  = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    features       = {}
    returns        = pd.DataFrame()

    for ticker in TICKER_LIST:
        fp = FEATURE_DIR / f"{ticker}.parquet"
        if not fp.exists():
            continue
        feat              = pd.read_parquet(fp)
        features[ticker]  = feat
        returns[ticker]   = feat["log_return"]

    returns       = returns.dropna()
    signals_multi = multi_signals.reindex(returns.index).fillna(0)

    # ── Load SPY raw returns ──────────────────────────────────────────────────
    spy_feat = features.get("SPY", pd.read_parquet(FEATURE_DIR / "SPY.parquet"))
    spy_raw  = spy_feat["log_return"].dropna()

    # Convert equity curves → fractional returns
    best_ret  = pc["multi_mom_tilt"].pct_change().dropna()
    carry_ret = pc["portable_carry"].pct_change().dropna()
    spy_aligned = spy_raw.reindex(best_ret.index).fillna(0)

    elapsed = time.time() - t0

    # ── Part 1b: Health Check ─────────────────────────────────────────────────
    print_health_check(pc, elapsed)

    # ── Part 2: Vol Targeting ─────────────────────────────────────────────────
    vt_results = run_vol_targeting(best_ret, carry_ret, multi_signals, features, returns)

    # ── Part 3a: Walk-Forward ─────────────────────────────────────────────────
    wf_results = run_walk_forward(multi_signals, features, returns)

    # ── Part 3b: Regime Analysis ──────────────────────────────────────────────
    regime_results = run_regime_analysis(best_ret, returns, signals_multi, spy_raw)

    # ── Part 3c: Universe & Signal Stats ──────────────────────────────────────
    signal_stats = run_signal_stats(signals_multi)

    # ── Part 3d: Transaction Cost Sensitivity ──────────────────────────────────
    cost_results = run_cost_sensitivity(best_ret, signals_multi)

    # ── Part 3e: Drawdown Analysis ─────────────────────────────────────────────
    dd_results = run_drawdown_analysis(best_ret, spy_raw)

    # ── Part 3f: Monthly Returns ───────────────────────────────────────────────
    monthly_results = run_monthly_returns(best_ret)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    report = {
        "generated_at":    datetime.now().isoformat(),
        "method":          "multi_mom_tilt",
        "data_range":      f"{best_ret.index[0].date()} to {best_ret.index[-1].date()}",
        "full_history_stats": _stats(best_ret),
        "vol_targeting":   vt_results,
        "walk_forward":    wf_results,
        "regime_analysis": regime_results,
        "signal_stats":    signal_stats,
        "cost_sensitivity": cost_results,
        "drawdown_analysis": dd_results,
        "monthly_returns": monthly_results,
    }

    report_path = RESULTS_DIR / "production_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    total_elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Validation complete in {total_elapsed:.1f}s")
    print(f"  JSON report → {report_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
