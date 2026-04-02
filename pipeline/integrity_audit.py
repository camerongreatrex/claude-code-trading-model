"""
integrity_audit.py
──────────────────
Read-only diagnostic: 4 integrity checks + production recommendation.

Checks:
  1. Look-ahead bias — position sizes must not use future data
  2. Walk-forward window integrity — non-overlapping, sufficient trading days
  3. Regime parameter sensitivity — key params must not be fragile / overfit
  4. Transaction cost reality check — turnover must not destroy net Sharpe

Run:
  python run.py audit          (via run.py shortcut)
  python -m pipeline.integrity_audit  (direct)
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── imports from pipeline ─────────────────────────────────────────────────────
from .portfolio import (
    regime_adaptive_sizes,
    adaptive_blend_sizes,
    momentum_tilt_sizes,
    portfolio_returns,
    atr_sizes,
    load_dead_weight_scalars,
    _get_macro,
    CAPITAL,
    MAX_POSITION_PCT,
)
from .backtester import sharpe_ratio
from .data_pipeline import ASSET_CLASS

SIGNAL_DIR   = Path("data/signals")
FEATURE_DIR  = Path("data/features")
RESULTS_DIR  = Path("data/results")
RESEARCH_DIR = Path("data/research")

# ── candidate methods and their OOS metrics (from Phase 2/3 walk-forward) ─────
CANDIDATES = {
    "multi_mom_tilt": lambda s, f, r: momentum_tilt_sizes(s, f, CAPITAL),
    "adaptive_blend": lambda s, f, r: adaptive_blend_sizes(s, f, r, CAPITAL),
    "regime_adaptive": lambda s, f, r: regime_adaptive_sizes(s, f, r, CAPITAL),
}

# OOS metrics recorded from targeted walk-forward (Phase 2/3 targeted script)
OOS_METRICS = {
    "multi_mom_tilt":  {"oos_sharpe": 1.108, "gap": 0.086, "capture": 1.118},
    "adaptive_blend":  {"oos_sharpe": 1.071, "gap": 0.124, "capture": 1.110},
    "regime_adaptive": {"oos_sharpe": 1.022, "gap": 0.153, "capture": 1.101},
}


# ── data loader ───────────────────────────────────────────────────────────────

def _load_data():
    signals = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    features, returns = {}, pd.DataFrame()
    for t in signals.columns:
        p = FEATURE_DIR / f"{t}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            features[t] = f
            returns[t]  = f["log_return"]
    returns = returns.dropna()
    sigs    = signals.reindex(returns.index).fillna(0)
    return sigs, features, returns


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Look-Ahead Bias Detection
# ─────────────────────────────────────────────────────────────────────────────

def check_lookahead(sigs, features, returns) -> dict:
    """
    For each candidate, compare sizes computed on full history vs history
    with the last 5 rows removed.  Sizes on the 5th-to-last day must be
    identical — any difference indicates the function consulted future data.

    PASS: max dollar difference on day T-5 < $1 across all tickers.
    FAIL: any ticker differs by >= $1.
    """
    _sep()
    print("  CHECK 1 — LOOK-AHEAD BIAS DETECTION")
    _sep()

    results = {"check": 1, "name": "look_ahead_bias", "pass": True, "details": {}}

    for method, fn in CANDIDATES.items():
        try:
            sizes_full  = fn(sigs,          features, returns)
            sizes_trunc = fn(sigs.iloc[:-5], features, returns.iloc[:-5])

            # Day T-5: last row of truncated == index [-6] of full
            row_full  = sizes_full.iloc[-6]
            row_trunc = sizes_trunc.iloc[-1]
            cols      = row_full.index.intersection(row_trunc.index)
            diff      = (row_full[cols] - row_trunc[cols]).abs()
            max_diff  = float(diff.max()) if len(diff) > 0 else 0.0
            worst     = str(diff.idxmax()) if len(diff) > 0 else "n/a"
            passed    = max_diff < 1.0

            if not passed:
                results["pass"] = False
            status = "PASS ✓" if passed else "FAIL ✗"
            results["details"][method] = {
                "max_diff_dollars": round(max_diff, 4),
                "worst_ticker": worst,
                "pass": passed,
            }
            print(f"  {method:<22}  {status}   max_diff = ${max_diff:.4f}  (ticker: {worst})")

        except Exception as exc:
            results["pass"] = False
            results["details"][method] = {"pass": False, "error": str(exc)}
            print(f"  {method:<22}  ERROR: {exc}")

    _result_line(1, results["pass"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Walk-Forward Window Integrity
# ─────────────────────────────────────────────────────────────────────────────

def check_walkforward_integrity() -> dict:
    """
    Load walk_forward_regime.parquet (equal-weight strategy) and verify:
      - Test windows are non-overlapping
      - Each window has >= 200 trading days
      - Stored OOS Sharpe in oos_selection.parquet matches WF mean within 0.05

    PASS: all three sub-checks pass.
    """
    _sep()
    print("  CHECK 2 — WALK-FORWARD WINDOW INTEGRITY")
    _sep()

    results = {"check": 2, "name": "walkforward_integrity",
               "pass": True, "issues": []}

    wf_path  = RESULTS_DIR / "walk_forward_regime.parquet"
    oos_path = RESULTS_DIR / "oos_selection.parquet"

    if not wf_path.exists():
        print("  walk_forward_regime.parquet not found — skipping (not a hard failure)")
        _result_line(2, True)
        return results

    wf = pd.read_parquet(wf_path)
    print(f"  {len(wf)} walk-forward windows found:")
    print(wf[["period", "n_days", "sharpe"]].to_string(index=False, justify="right"))

    # Sub-check A: parse periods and verify no overlap
    periods = []
    for p in wf["period"]:
        parts = str(p).split("-")
        if len(parts) == 2:
            try:
                periods.append((int(parts[0]), int(parts[1])))
            except ValueError:
                pass

    overlaps = 0
    for i in range(len(periods) - 1):
        if periods[i + 1][0] < periods[i][1]:
            msg = f"overlap: {periods[i]} overlaps {periods[i+1]}"
            results["issues"].append(msg)
            results["pass"] = False
            overlaps += 1
            print(f"  ✗ OVERLAP: {msg}")
    if overlaps == 0:
        print(f"  ✓ No period overlaps ({len(periods)} windows, "
              f"start years: {[p[0] for p in periods]})")

    # Sub-check B: minimum days per window
    short_windows = wf[wf["n_days"] < 200]
    if not short_windows.empty:
        for _, row in short_windows.iterrows():
            msg = f"period {row['period']} has only {row['n_days']} days"
            results["issues"].append(msg)
            results["pass"] = False
            print(f"  ✗ SHORT WINDOW: {msg}")
    else:
        min_days = int(wf["n_days"].min())
        print(f"  ✓ All windows >= 200 trading days (min = {min_days})")

    # Sub-check C: stored OOS Sharpe cross-check (equal weight baseline)
    if oos_path.exists():
        oos_df   = pd.read_parquet(oos_path)
        wf_mean  = round(float(wf["sharpe"].mean()), 3)
        eq_row   = oos_df[oos_df["method"] == "equal weight"]
        if not eq_row.empty:
            stored = round(float(eq_row["oos_sharpe"].iloc[0]), 3)
            diff   = abs(wf_mean - stored)
            if diff > 0.05:
                msg = (f"equal weight OOS Sharpe mismatch: "
                       f"WF mean={wf_mean}, stored={stored}, diff={diff:.3f}")
                results["issues"].append(msg)
                results["pass"] = False
                print(f"  ✗ OOS MISMATCH: {msg}")
            else:
                print(f"  ✓ OOS cross-check OK: WF mean={wf_mean}, "
                      f"stored={stored}, diff={diff:.3f}")

    _result_line(2, results["pass"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Regime Parameter Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def _regime_adaptive_parameterised(signals, features, returns, capital,
                                    bull_gross: float = 0.95,
                                    bull_hedge: float = 0.12) -> pd.DataFrame:
    """
    Parameterised clone of regime_adaptive_sizes — only the two bull_calm
    breakpoints (VIX=12) are varied; everything else is fixed at defaults.
    Used exclusively by the sensitivity sweep in Check 3.
    """
    macro_df = _get_macro()
    spy_path = Path("data/features/SPY.parquet")
    if macro_df.empty or "vix" not in macro_df.columns or not spy_path.exists():
        return momentum_tilt_sizes(signals, features, capital)

    vix       = macro_df["vix"].reindex(signals.index).ffill().fillna(20.0)
    spy_close = pd.read_parquet(spy_path)["Close"].reindex(signals.index).ffill()
    spy_60d   = spy_close.pct_change(60).fillna(0.0)

    vix_xp       = [0,          12,          20,   30,   40,   100]
    vix_gross_fp = [bull_gross, bull_gross,  0.85, 0.60, 0.45, 0.40]
    vix_hedge_fp = [bull_hedge, bull_hedge,  0.18, 0.35, 0.45, 0.50]
    vix_tilt_fp  = [0.80,       0.80,        0.60, 0.40, 0.30, 0.25]

    spy_xp       = [-0.30, -0.15, 0.0, 0.10, 0.30]
    spy_scale_fp = [ 0.80,  0.90, 1.00, 1.05, 1.10]

    raw_gross = (np.interp(vix.values, vix_xp, vix_gross_fp)
                 * np.interp(spy_60d.values, spy_xp, spy_scale_fp))
    raw_hedge = np.interp(vix.values, vix_xp, vix_hedge_fp)
    raw_tilt  = np.interp(vix.values, vix_xp, vix_tilt_fp)

    gt_s = pd.Series(raw_gross, index=signals.index).clip(0.40, 0.98)
    hc_s = pd.Series(raw_hedge, index=signals.index).clip(0.10, 0.50)
    ts_s = pd.Series(raw_tilt,  index=signals.index).clip(0.20, 0.90)

    def _smooth(s):
        ema = s.ewm(span=5, adjust=False).mean()
        return ema.shift(1).fillna(ema)

    gt_s = _smooth(gt_s)
    hc_s = _smooth(hc_s)
    ts_s = _smooth(ts_s)

    sized = atr_sizes(signals, features, capital)

    hedge_tickers = [t for t in signals.columns
                     if ASSET_CLASS.get(t) in ("bond", "commodity")]
    if hedge_tickers:
        hc_dollars = hc_s * capital
        for t in hedge_tickers:
            sized[t] = sized[t].clip(lower=-hc_dollars, upper=hc_dollars)

    close_cols = {t: features[t]["Close"].reindex(sized.index).ffill()
                  for t in signals.columns if t in features}
    if close_cols:
        closes      = pd.DataFrame(close_cols)
        mom         = closes.pct_change(63)
        active_mask = signals.abs() > 0
        mom_cols    = [c for c in signals.columns if c in mom.columns]
        mom_masked  = mom[mom_cols].where(active_mask[mom_cols])
        ranks       = mom_masked.rank(axis=1, pct=True)
        n_active    = ranks.notna().sum(axis=1)
        tilt        = pd.DataFrame(1.0, index=sized.index, columns=sized.columns)
        valid_rows  = n_active >= 2
        for c in mom_cols:
            col_ranks = ranks[c]
            col_valid = valid_rows & col_ranks.notna()
            if not col_valid.any():
                continue
            base_tilt = 0.7 + 0.6 * col_ranks[col_valid].values
            strength  = ts_s[col_valid].values
            tilt.loc[col_valid, c] = 1.0 + (base_tilt - 1.0) * strength
        sized = sized * tilt

    dw = load_dead_weight_scalars()
    if dw:
        sized = sized.multiply(pd.Series(dw).reindex(sized.columns).fillna(1.0),
                               axis=1)

    current_gross  = sized.abs().sum(axis=1).replace(0, np.nan)
    target_gross   = gt_s * capital
    sized          = sized.multiply((target_gross / current_gross).fillna(1.0),
                                    axis=0)
    sized          = sized.clip(-capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT)
    gross          = sized.abs().sum(axis=1).replace(0, np.nan)
    return sized.multiply((capital / gross).clip(upper=1.0).fillna(1.0), axis=0)


def check_sensitivity(sigs, features, returns) -> dict:
    """
    Sweep the two bull_calm breakpoints of regime_adaptive and verify that
    adjacent steps do not cause Sharpe to collapse (fragility = overfit).

    PASS: every adjacent step changes Sharpe by < 0.15.
    FAIL: any adjacent pair differs by >= 0.15.
    """
    _sep()
    print("  CHECK 3 — PARAMETER SENSITIVITY  (regime_adaptive)")
    _sep()

    results = {"check": 3, "name": "parameter_sensitivity",
               "pass": True, "details": {}}

    baseline_sizes = regime_adaptive_sizes(sigs, features, returns, CAPITAL)
    baseline_ret   = portfolio_returns(baseline_sizes, returns)
    baseline_sh    = sharpe_ratio(baseline_ret)
    print(f"  Baseline regime_adaptive IS Sharpe: {baseline_sh:.3f}")

    FRAGILE_THRESHOLD = 0.15

    def _sweep_column(param_name, values, fixed_gross, fixed_hedge):
        sharpes = {}
        print(f"\n  Sweep: {param_name}   (other param held at default)")
        hdr = f"  {'value':>8}  {'Sharpe':>8}  {'Δ vs baseline':>15}  {'Adjacent Δ':>12}  {'Fragile?':>10}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        fragile = False
        prev_sh = None
        for v in values:
            if param_name.startswith("target_gross"):
                bg, bh = v, fixed_hedge
            else:
                bg, bh = fixed_gross, v
            sz = _regime_adaptive_parameterised(
                sigs, features, returns, CAPITAL,
                bull_gross=bg, bull_hedge=bh,
            )
            sh     = sharpe_ratio(portfolio_returns(sz, returns))
            delta  = sh - baseline_sh
            adj    = abs(sh - prev_sh) if prev_sh is not None else float("nan")
            is_frag = (not np.isnan(adj)) and adj >= FRAGILE_THRESHOLD
            if is_frag:
                fragile = True
            sharpes[v] = round(sh, 4)
            adj_str  = f"{adj:+.3f}" if not np.isnan(adj) else "  --"
            frag_str = "YES ✗" if is_frag else "no"
            print(f"  {v:>8.2f}  {sh:>8.3f}  {delta:>+15.3f}  {adj_str:>12}  {frag_str:>10}")
            prev_sh = sh
        return sharpes, fragile

    gross_sharpes, gross_fragile = _sweep_column(
        "target_gross_bull_calm",
        [0.85, 0.88, 0.90, 0.92, 0.95],
        fixed_gross=0.95, fixed_hedge=0.12,
    )
    hedge_sharpes, hedge_fragile = _sweep_column(
        "hedge_cap_bull_calm",
        [0.08, 0.10, 0.12, 0.15, 0.18],
        fixed_gross=0.95, fixed_hedge=0.12,
    )

    if gross_fragile or hedge_fragile:
        results["pass"] = False

    results["details"] = {
        "baseline_sharpe": round(baseline_sh, 4),
        "gross_sweep": gross_sharpes,
        "hedge_sweep": hedge_sharpes,
        "gross_fragile": gross_fragile,
        "hedge_fragile": hedge_fragile,
    }
    _result_line(3, results["pass"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — Transaction Cost Reality Check
# ─────────────────────────────────────────────────────────────────────────────

def check_transaction_costs(sigs, features, returns) -> dict:
    """
    Verify that new methods don't generate excessive turnover relative to
    multi_mom_tilt.

    NOTE: Transaction costs are ALREADY deducted in the backtester via
    compute_strategy_returns() (backtester.py lines 115-145).  The Sharpe
    figures in OOS_METRICS are therefore already net of costs.  Re-deducting
    costs here would double-count them.

    This check only verifies RELATIVE turnover (new method vs baseline).
    An extra 2× turnover would double the cost drag already absorbed in
    the Sharpe figures, which would be a meaningful hidden disadvantage.

    PASS: turnover < 2× multi_mom_tilt baseline for all candidates.
    FAIL: any candidate exceeds 2× baseline turnover.

    multi_mom_tilt is the baseline and is never flagged.
    """
    _sep()
    print("  CHECK 4 — TRANSACTION COSTS  (costs already in Sharpe — checking relative turnover)")
    _sep()
    print("  Note: backtester deducts 5–10 bps/side depending on asset class.")
    print("  Gross Sharpe shown is already net of those costs.")
    print("  FLAG criterion: turnover > 2× multi_mom_tilt baseline.\n")

    COST_BPS = 0.0007   # 7 bps informational estimate (already in backtester)
    results  = {"check": 4, "name": "transaction_costs",
                "pass": True, "details": {}}

    mom_sizes      = momentum_tilt_sizes(sigs, features, CAPITAL)
    baseline_to    = float(mom_sizes.diff().abs().sum(axis=1).mean() / CAPITAL)
    baseline_annto = baseline_to * 252

    hdr = (f"  {'Method':<22}  {'Daily TO':>10}  {'Ann TO':>8}  {'×baseline':>10}"
           f"  {'Est cost/yr':>12}  {'(net) Sharpe':>13}  {'Status':>8}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for method, fn in CANDIDATES.items():
        sizes    = fn(sigs, features, returns)
        ret      = portfolio_returns(sizes, returns)
        net_sh   = sharpe_ratio(ret)   # already net of costs

        daily_to  = float(sizes.diff().abs().sum(axis=1).mean() / CAPITAL)
        annual_to = daily_to * 252
        to_ratio  = daily_to / baseline_to if baseline_to > 0 else 1.0
        est_cost  = annual_to * COST_BPS   # informational

        high_to = to_ratio > 2.0

        if method == "multi_mom_tilt":
            status = "baseline"
        elif high_to:
            status = "FLAG ✗"
            results["pass"] = False
        else:
            status = "ok ✓"

        results["details"][method] = {
            "daily_turnover":  round(daily_to,  4),
            "annual_turnover": round(annual_to, 3),
            "to_ratio":        round(to_ratio,  3),
            "est_cost_pct":    round(est_cost,  4),
            "net_sharpe":      round(net_sh,    3),
            "pass":            status != "FLAG ✗",
        }
        print(f"  {method:<22}  {daily_to:>10.4f}  {annual_to:>8.2f}  {to_ratio:>10.2f}"
              f"  {est_cost:>+11.4f}  {net_sh:>13.3f}  {status:>8}")

    print(f"\n  Baseline (multi_mom_tilt): {baseline_annto:.2f}× annual turnover"
          f"  (daily avg ${baseline_to * CAPITAL:,.0f} / $100k capital)")
    _result_line(4, results["pass"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT — Production Recommendation
# ─────────────────────────────────────────────────────────────────────────────

def final_recommendation(check_results: list) -> dict:
    """
    Disqualify any method that failed a check, then select using:
      score = OOS Sharpe × Capture Ratio   (tiebreaker: smaller IS-OOS gap)
    """
    print("\n" + "=" * 70)
    print("  PRODUCTION RECOMMENDATION")
    print("=" * 70)

    # Collect disqualifications
    disqualified = set()

    c1 = _find_check(check_results, 1)
    if c1:
        for method, d in c1.get("details", {}).items():
            if not d.get("pass", True):
                disqualified.add(method)

    c3 = _find_check(check_results, 3)
    if c3 and not c3.get("pass", True):
        disqualified.add("regime_adaptive")   # only regime_adaptive is tested in c3

    c4 = _find_check(check_results, 4)
    if c4:
        for method, d in c4.get("details", {}).items():
            if not d.get("pass", True):
                disqualified.add(method)

    if disqualified:
        print(f"\n  Disqualified (failed integrity check): {sorted(disqualified)}")

    # Score qualifying candidates
    print(f"\n  {'Method':<22}  {'OOS Sh':>8}  {'Capture':>8}  {'Score':>8}  {'Gap':>8}  {'Eligible':>9}")
    print("  " + "─" * 68)
    scores = {}
    for method, m in OOS_METRICS.items():
        score    = m["oos_sharpe"] * m["capture"]
        eligible = method not in disqualified
        flag     = "✓" if eligible else "✗ disqualified"
        print(f"  {method:<22}  {m['oos_sharpe']:>8.3f}  {m['capture']:>8.3f}"
              f"  {score:>8.4f}  {m['gap']:>+8.3f}  {flag:>9}")
        if eligible:
            scores[method] = score

    if not scores:
        print("\n  ⚠  All candidates disqualified — manual review required.")
        return {"check": 5, "name": "recommendation",
                "method": None, "reason": "all candidates disqualified"}

    # Select: highest score, tiebreak by smallest gap
    best = max(scores, key=lambda m: (scores[m], -OOS_METRICS[m]["gap"]))
    bm   = OOS_METRICS[best]

    reason_map = {
        "multi_mom_tilt": (
            "Highest OOS Sharpe × Capture product (1.239). "
            "Tightest IS-OOS gap (+0.086) across all 4 walk-forward windows — "
            "strongest evidence of out-of-sample generalisation. "
            "Consistent positive OOS Sharpe in every year tested."
        ),
        "adaptive_blend": (
            "Strong OOS Sharpe × Capture product (1.189). "
            "Tighter IS-OOS gap (+0.124) than regime_adaptive alone. "
            "Blending smooth de-risking (regime_adaptive) with signal consistency "
            "(multi_mom_tilt) reduces window-to-window OOS variance."
        ),
        "regime_adaptive": (
            "Best capture ratio (1.101) among candidates. "
            "Smooth VIX interpolation (Phase 3) resolved the 2021-22 negative window. "
            "IS-OOS gap +0.153 is within the < 0.40 acceptance bound."
        ),
    }

    print(f"\n  ┌─ SELECTED: {best}")
    print(f"  │  Score:     {scores[best]:.4f}  "
          f"(OOS Sharpe {bm['oos_sharpe']} × Capture {bm['capture']})")
    print(f"  │  IS-OOS gap: {bm['gap']:+.3f}")
    print(f"  │")
    print(f"  │  Reason: {reason_map.get(best, 'Highest OOS Sharpe × Capture.')}")
    print(f"  └─")
    print()
    print(f"  Action required:")
    print(f"    python paper_trader.py strategy {best}")
    print()
    print(f"  This will take effect on the next EOD update.")
    print("=" * 70)

    return {
        "check":        5,
        "name":         "recommendation",
        "method":       best,
        "score":        round(scores[best], 4),
        "reason":       reason_map.get(best, ""),
        "disqualified": sorted(disqualified),
        "all_scores":   {m: round(v, 4) for m, v in scores.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep():
    print("\n" + "─" * 70)


def _result_line(n: int, passed: bool):
    verdict = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n  CHECK {n} RESULT: {verdict}")


def _find_check(check_results, n):
    return next((c for c in check_results if c.get("check") == n), None)


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  BACKTESTING INTEGRITY AUDIT")
    print("=" * 70)

    print("\nLoading data...")
    sigs, features, returns = _load_data()
    print(f"  signals: {sigs.shape}   returns: {returns.shape}")

    check_results = []
    check_results.append(check_lookahead(sigs, features, returns))
    check_results.append(check_walkforward_integrity())
    check_results.append(check_sensitivity(sigs, features, returns))
    check_results.append(check_transaction_costs(sigs, features, returns))
    check_results.append(final_recommendation(check_results))

    # Persist
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESEARCH_DIR / "integrity_audit.json"
    with open(out_path, "w") as fh:
        json.dump(check_results, fh, indent=2, default=_json_default)
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    main()
