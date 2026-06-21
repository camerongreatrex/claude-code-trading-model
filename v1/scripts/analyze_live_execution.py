"""
Analyze live paper-trading execution vs backtest cadences.

Uses main + v4nf_3mo trade ledgers and walkforward_execution_study.json to
summarize what hurt live P&L and which no-topup cadence fits medium frequency.

Run: PYTHONPATH=. python v1/scripts/analyze_live_execution.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

STUDY = Path("data/v1/results/walkforward_execution_study.json")
INSTANCES = {
    "main": Path("data/v1/paper_trading"),
    "v4nf_3mo": Path("data/v1/paper_trading_v4nf_3mo"),
}


def _period_return(history: pd.DataFrame, start: str, end: str) -> float | None:
    h = history[(history["date"] >= start) & (history["date"] <= end)]
    if len(h) < 2:
        return None
    return float(h["portfolio_value"].iloc[-1] / h["portfolio_value"].iloc[0] - 1)


def _trade_summary(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["commission"] = pd.to_numeric(t["commission"], errors="coerce").fillna(0)
    t["pnl"] = pd.to_numeric(t["pnl"], errors="coerce").fillna(0)
    return (
        t.groupby("reason")
        .agg(n=("ticker", "count"), comm=("commission", "sum"), pnl=("pnl", "sum"))
        .sort_values("n", ascending=False)
    )


def main() -> None:
    print("=" * 60)
    print("Live execution analysis (3mo test + main paper trading)")
    print("=" * 60)

    for label, pdir in INSTANCES.items():
        hist = pd.read_csv(pdir / "history.csv")
        hist["date"] = pd.to_datetime(hist["date"])
        trades = pd.read_csv(pdir / "trades.csv")
        print(f"\n--- {label} ({hist['date'].min().date()} → {hist['date'].max().date()}) ---")
        tot = hist["portfolio_value"].iloc[-1] / hist["portfolio_value"].iloc[0] - 1
        print(f"  Total return: {tot * 100:+.2f}%")
        print(f"  Commission:   ${pd.to_numeric(trades['commission'], errors='coerce').fillna(0).sum():,.2f}")
        rebal = trades[trades["reason"].str.contains("rebalance", na=False)]
        print(f"  Rebalance trades: {len(rebal)} (${rebal['commission'].astype(float).sum():,.2f} fees)")
        print(_trade_summary(trades).head(8).to_string())

    main_hist = pd.read_csv(INSTANCES["main"] / "history.csv")
    main_hist["date"] = pd.to_datetime(main_hist["date"])
    pre = _period_return(main_hist, "2026-03-23", "2026-05-03")
    post = _period_return(main_hist, "2026-05-04", "2026-06-19")
    print("\n--- Main: pre vs post daily top-up (May 4 switch) ---")
    if pre is not None:
        print(f"  Pre-topup  (signal only):  {pre * 100:+.2f}%")
    if post is not None:
        print(f"  Post-topup (daily rebalance): {post * 100:+.2f}%")

    if STUDY.exists():
        study = json.loads(STUDY.read_text())
        print("\n--- Walk-forward OOS cadence comparison (backtest, fee-aware) ---")
        for key in ("top11_signal_only", "top11_daily", "top11_monthly"):
            agg = study["variants"][key]["oos_aggregate"]
            print(
                f"  {key:22s}  net {agg['mean_ann_return_net_pct']:5.1f}%  "
                f"Sh {agg['mean_sharpe_net']:.2f}  "
                f"trades/yr {agg['mean_trades_per_year']:.0f}  "
                f"rebal {agg['mean_rebalance_pct']:.0f}%"
            )
        rec = study.get("recommendation", {})
        print(f"\n  Recommended live cadence: {rec.get('variant', '?')}")

    print("\n--- Live model (current) ---")
    print("  rank_rotate: enter top-N on signal; rank_exit after min hold; NO top-up")
    print("  Target: ~640 trades/yr (medium), ~20% OOS net ann in backtest")


if __name__ == "__main__":
    main()
