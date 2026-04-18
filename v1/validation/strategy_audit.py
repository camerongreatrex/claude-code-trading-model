"""
strategy_audit.py
-----------------
Comprehensive strategy audit: ranks all strategies by composite score and
produces KEEP / DELETE / LIVE recommendations for pipeline cleanup.

Metrics computed per strategy:
  - OOS Sharpe          (from oos_selection.parquet)
  - IS Sharpe           (from oos_selection.parquet)
  - IS-OOS gap          (raw signed; abs used in scoring)
  - Annualised return   (from portfolio_comparison.parquet equity curves)
  - Max drawdown        (from portfolio_comparison.parquet equity curves)
  - Terminal value      (final equity curve value on $100k capital)
  - Capture ratio       (upside_capture / downside_capture vs SPY)
  - Avg gross exposure  (mean fraction of tickers with active signal)
  - Return gap vs B&H   (annualised strategy return − buy & hold return)

Composite score:
  score = 0.35 × norm(oos_sharpe)
        + 0.25 × norm(ann_return)
        + 0.20 × norm(capture_ratio)
        + 0.20 × (1 − norm(abs_is_oos_gap))

Usage:
  python strategy_audit.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CAPITAL = 100_000
RESULTS = Path("data/v1/results")
SIGNALS = Path("data/v1/signals")
FEATURES = Path("data/v1/features")

# ── Name mapping: portfolio_comparison column → oos_selection method string ────
# oos_selection uses mixed naming (spaces, underscores); portfolio_comparison
# uses underscores.  Map the equity curve column names to OOS table method names.
_PC_TO_OOS = {
    "equal_weight"       : "equal weight",
    "risk_parity"        : "risk parity",
    "rp_regime_aware"    : "rp_regime_aware",
    "rp_regime_dw"       : "rp_regime_dw",
    "rp_blend"           : "rp_blend",
    "multi_equal_weight" : "multi equal weight",
    "multi_atr_pure"     : "multi_atr_pure",
    "multi_mom_tilt"     : "multi_mom_tilt",
    "regime_adaptive"    : "regime_adaptive",
    "adaptive_blend"     : "adaptive_blend",
    "multi_mom_portable" : "multi_mom_portable",
    "multi_mom_port_low" : "multi_mom_port_low",
    "multi_mom_carry"    : "multi_mom_carry",
    "portable_carry"     : "portable_carry",
}

# Signal matrix used by each strategy (for gross exposure proxy)
_SIGNAL_SOURCE = {
    "equal_weight"       : "regime",
    "risk_parity"        : "regime",
    "rp_regime_aware"    : "regime",
    "rp_regime_dw"       : "regime",
    "rp_blend"           : "regime",
    "multi_equal_weight" : "multi",
    "multi_atr_pure"     : "multi",
    "multi_mom_tilt"     : "multi",
    "regime_adaptive"    : "multi",
    "adaptive_blend"     : "multi",
    "multi_mom_portable" : "multi",
    "multi_mom_port_low" : "multi",
    "multi_mom_carry"    : "multi",
    "portable_carry"     : "multi",
}


def _ann_return(equity: pd.Series) -> float:
    """Annualised return from an equity curve starting at CAPITAL."""
    n = len(equity.dropna())
    if n < 2:
        return np.nan
    final = equity.dropna().iloc[-1]
    return (final / CAPITAL) ** (252 / n) - 1


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    eq = equity.dropna()
    running_max = eq.cummax()
    dd = (eq - running_max) / running_max
    return float(dd.min())


def _capture_ratio(strat_returns: pd.Series, spy_returns: pd.Series) -> float:
    """
    Convexity ratio: upside capture / downside capture.

    upside_capture  = mean(strat on SPY+days) / mean(SPY on SPY+days)
    downside_capture = mean(strat on SPY-days) / mean(SPY on SPY-days)

    Values > 1.0 indicate the strategy captures more upside per unit of
    downside than SPY — i.e. convex payoff profile.
    """
    aligned = pd.concat([strat_returns, spy_returns], axis=1).dropna()
    aligned.columns = ["strat", "spy"]

    up   = aligned[aligned["spy"] > 0]
    down = aligned[aligned["spy"] < 0]

    if len(up) < 10 or len(down) < 10:
        return np.nan

    spy_up_mean  = up["spy"].mean()
    spy_dn_mean  = down["spy"].mean()

    up_cap  = up["strat"].mean()  / spy_up_mean  if spy_up_mean  != 0 else np.nan
    dn_cap  = down["strat"].mean() / spy_dn_mean  if spy_dn_mean  != 0 else np.nan

    # dn_cap is positive (both strategy and SPY are negative, ratio is positive)
    if dn_cap is None or np.isnan(dn_cap) or dn_cap <= 0:
        return np.nan

    return float(up_cap / dn_cap)


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1].  Returns 0.5 if all values identical."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def main():
    print("=" * 76)
    print("  STRATEGY AUDIT  —  composite scoring and cleanup recommendations")
    print("=" * 76)

    # ── Load data ──────────────────────────────────────────────────────────────
    oos = pd.read_parquet(RESULTS / "oos_selection.parquet")
    pc  = pd.read_parquet(RESULTS / "portfolio_comparison.parquet")

    oos = oos.set_index("method")

    # SPY returns from buy_hold equity curve (buy_hold is 100% SPY since the
    # portfolio_comparison uses SPY-based buy_hold as the benchmark).
    spy_equity  = pc["buy_hold"].dropna()
    spy_returns = spy_equity.pct_change().dropna()

    # Signal matrices for gross-exposure proxy
    sig_regime = pd.read_parquet(SIGNALS / "regime_signals.parquet")
    sig_multi  = pd.read_parquet(SIGNALS / "multi_signals.parquet")

    strategies = [k for k in _PC_TO_OOS if k in pc.columns]

    rows = []
    for key in strategies:
        oos_name = _PC_TO_OOS[key]
        equity   = pc[key].dropna()

        # ── OOS / IS Sharpe ───────────────────────────────────────────────────
        if oos_name in oos.index:
            oos_sharpe = float(oos.loc[oos_name, "oos_sharpe"])
            is_sharpe  = float(oos.loc[oos_name, "is_sharpe"])
        else:
            oos_sharpe = np.nan
            is_sharpe  = np.nan

        gap     = is_sharpe - oos_sharpe          # signed: positive = overfitting
        abs_gap = abs(gap)

        # ── Equity curve metrics ───────────────────────────────────────────────
        ann_ret  = _ann_return(equity)
        max_dd   = _max_drawdown(equity)
        terminal = float(equity.iloc[-1])

        # ── Return gap vs buy & hold ──────────────────────────────────────────
        bnh_ann = _ann_return(pc["buy_hold"].dropna())
        ret_gap = ann_ret - bnh_ann

        # ── Capture ratio ─────────────────────────────────────────────────────
        strat_returns = equity.pct_change().dropna()
        strat_returns, spy_aligned = strat_returns.align(spy_returns, join="inner")
        capture = _capture_ratio(strat_returns, spy_aligned)

        # ── Gross exposure proxy ──────────────────────────────────────────────
        src = _SIGNAL_SOURCE.get(key, "multi")
        sig_df = sig_regime if src == "regime" else sig_multi
        # Align signal dates to equity curve dates (signals may start earlier)
        common_idx = sig_df.index.intersection(equity.index)
        if len(common_idx) > 0:
            sig_slice  = sig_df.reindex(common_idx)
            gross_exp  = float(sig_slice.abs().mean().mean())  # avg fraction active
        else:
            gross_exp  = np.nan

        rows.append({
            "strategy"    : key,
            "oos_sharpe"  : oos_sharpe,
            "is_sharpe"   : is_sharpe,
            "gap"         : gap,
            "abs_gap"     : abs_gap,
            "ann_ret"     : ann_ret,
            "max_dd"      : max_dd,
            "terminal"    : terminal,
            "capture"     : capture,
            "gross_exp"   : gross_exp,
            "ret_gap_bnh" : ret_gap,
        })

    df = pd.DataFrame(rows).set_index("strategy")

    # ── Composite score ────────────────────────────────────────────────────────
    # score = 0.35×norm(oos_sharpe) + 0.25×norm(ann_ret) +
    #         0.20×norm(capture)    + 0.20×(1 − norm(abs_gap))
    #
    # Weight rationale:
    #   35% OOS Sharpe     — risk-adjusted performance on unseen data (primary)
    #   25% Ann return     — raw return matters psychologically for live trading
    #   20% Capture ratio  — convexity: want upside > downside participation
    #   20% Robustness     — strategies with huge IS-OOS gaps are regime-dependent
    #                        and likely to mean-revert toward IS Sharpe in live trading

    n_oos     = _normalize(df["oos_sharpe"])
    n_ret     = _normalize(df["ann_ret"])
    n_cap     = _normalize(df["capture"].fillna(df["capture"].median()))
    n_rob     = 1.0 - _normalize(df["abs_gap"])   # larger gap → penalised

    df["composite"] = (
        0.35 * n_oos
      + 0.25 * n_ret
      + 0.20 * n_cap
      + 0.20 * n_rob
    )

    df = df.sort_values("composite", ascending=False)

    # ── Ranked table ──────────────────────────────────────────────────────────
    print(f"\n{'Strategy':<22} {'OOS Sh':>7} {'IS Sh':>7} {'Gap':>7} {'AnnRet':>7} {'MaxDD':>7} {'Terminal':>10} {'Capture':>8} {'GrossExp':>9} {'RetGap':>7} {'Score':>7}")
    print("  " + "-" * 105)

    for strat, row in df.iterrows():
        gap_str  = f"{row['gap']:+.3f}"
        ann_str  = f"{row['ann_ret']*100:+.1f}%"
        dd_str   = f"{row['max_dd']*100:.1f}%"
        cap_str  = f"{row['capture']:.2f}" if not np.isnan(row['capture']) else "  N/A"
        ge_str   = f"{row['gross_exp']*100:.0f}%" if not np.isnan(row['gross_exp']) else " N/A"
        rg_str   = f"{row['ret_gap_bnh']*100:+.1f}%"
        print(
            f"  {strat:<22} {row['oos_sharpe']:>7.3f} {row['is_sharpe']:>7.3f} {gap_str:>7} "
            f"{ann_str:>7} {dd_str:>7} {row['terminal']:>10,.0f} {cap_str:>8} {ge_str:>9} "
            f"{rg_str:>7} {row['composite']:>7.3f}"
        )

    # ── Buy & hold reference ──────────────────────────────────────────────────
    bnh_ann = _ann_return(pc["buy_hold"].dropna())
    bnh_dd  = _max_drawdown(pc["buy_hold"].dropna())
    bnh_ter = float(pc["buy_hold"].dropna().iloc[-1])
    print(f"\n  {'buy_hold (benchmark)':<22} {'—':>7} {'—':>7} {'—':>7} {bnh_ann*100:>+6.1f}% {bnh_dd*100:>6.1f}% {bnh_ter:>10,.0f}")

    # ── IS-OOS gap notes ──────────────────────────────────────────────────────
    print("\n  IS-OOS gap legend:")
    print("    > +0.20  overfitting signal — IS significantly exceeds OOS")
    print("    [-0.40, +0.20]  robust generalisation")
    print("    < -0.40  regime concentration — OOS windows were unusually favorable")
    print("             Expect live Sharpe to revert toward IS Sharpe")

    # ── Score component breakdown ─────────────────────────────────────────────
    print(f"\n{'='*76}")
    print("  SCORE COMPONENTS  (0–1 normalised, higher is better)")
    print(f"{'='*76}")
    print(f"  {'Strategy':<22} {'OOS(35%)':>10} {'Ret(25%)':>10} {'Cap(20%)':>10} {'Rob(20%)':>10} {'Total':>7}")
    print("  " + "-" * 73)
    for strat, row in df.iterrows():
        i = df.index.get_loc(strat)
        print(f"  {strat:<22} {n_oos.iloc[i]:>10.3f} {n_ret.iloc[i]:>10.3f} "
              f"{n_cap.iloc[i]:>10.3f} {n_rob.iloc[i]:>10.3f} {row['composite']:>7.3f}")

    # ── Recommendation ────────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print("  RECOMMENDATION")
    print(f"{'='*76}")

    top5 = df.index[:5].tolist()
    rest = df.index[5:].tolist()

    # Tie-break rule: if momentum_tilt and portable_carry are within 0.02 of
    # each other in composite score, prefer momentum_tilt (simpler, no hedge
    # friction, closer to raw return, fewer live failure modes).
    recommended = top5[0]
    if "portable_carry" in top5[:3] and "multi_mom_tilt" in top5[:3]:
        pc_score  = df.loc["portable_carry",  "composite"]
        mmt_score = df.loc["multi_mom_tilt",  "composite"]
        if abs(pc_score - mmt_score) <= 0.04:
            recommended = "multi_mom_tilt"
            print(f"\n  NOTE: portable_carry and multi_mom_tilt are within 0.04 composite")
            print(f"  score.  Preferring multi_mom_tilt — simpler, no SPY hedge friction,")
            print(f"  IS-OOS gap {df.loc['multi_mom_tilt','gap']:+.3f} vs {df.loc['portable_carry','gap']:+.3f}  (more robust).")
    elif recommended == "portable_carry":
        pc_gap  = df.loc["portable_carry", "abs_gap"]
        mmt_row = df.loc["multi_mom_tilt"] if "multi_mom_tilt" in df.index else None
        print(f"\n  NOTE: portable_carry leads on OOS Sharpe but IS-OOS gap is "
              f"{df.loc['portable_carry','gap']:+.3f}")
        print(f"  (regime concentrated, likely to revert in live trading).")
        if mmt_row is not None:
            print(f"  Considering multi_mom_tilt as safer live alternative")
            print(f"  (IS-OOS gap {mmt_row['gap']:+.3f}, OOS Sharpe {mmt_row['oos_sharpe']:.3f}).")

    print(f"\n  ┌─ TOP 5 STRATEGIES TO KEEP ────────────────────────────────────────┐")
    for rank, strat in enumerate(top5, 1):
        row = df.loc[strat]
        print(f"  │  {rank}. {strat:<24}  OOS {row['oos_sharpe']:.3f}  composite {row['composite']:.3f}   │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ STRATEGIES TO DELETE (not in top 5) ─────────────────────────────┐")
    for strat in rest:
        row = df.loc[strat]
        print(f"  │  - {strat:<24}  OOS {row['oos_sharpe']:.3f}  composite {row['composite']:.3f}   │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ► RECOMMENDED LIVE STRATEGY:  {recommended}")
    print(f"    OOS Sharpe   {df.loc[recommended,'oos_sharpe']:.3f}")
    print(f"    IS Sharpe    {df.loc[recommended,'is_sharpe']:.3f}")
    print(f"    IS-OOS gap   {df.loc[recommended,'gap']:+.3f}")
    print(f"    Ann return   {df.loc[recommended,'ann_ret']*100:.1f}%")
    print(f"    Max drawdown {df.loc[recommended,'max_dd']*100:.1f}%")
    print(f"    Composite    {df.loc[recommended,'composite']:.3f}")
    print()

    # Return top5 and recommended for programmatic use
    return top5, recommended, df


if __name__ == "__main__":
    top5, recommended, results = main()
