"""
capture_diagnostic.py — read-only upside/downside capture decomposition.
Diagnoses concave (low-beta) profile vs target convex.

Sections: (1) capture ratios per method (up/dn = mean(strat|spy>0)/mean(spy|spy>0)),
(2) cash drag of production method, (3) asset-class up/down-day P&L bps,
(4) per-ticker efficiency = upside_capture × active_pct/100 (bottom-5 = drag),
(5) filter-impact ladder: raw_ma → +RSI → +min_hold → +ATR_stop → +time_decay.

Usage: python -m v1.pipeline.capture_diagnostic | python run.py capture
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from v1.pipeline.data_pipeline import TICKER_LIST, ASSET_CLASS
from v1.regimes.analysis import (
    label_regimes,
    _load_macro,
    _load_spy_close,
    REGIME_ORDER,
    REGIME_LABELS,
)
from v1.pipeline.signal_generation import (
    equity_index_regime,
    sector_regime,
    stock_regime,
    apply_rsi_entry_filter,
    apply_min_hold_filter,
    apply_trailing_stop_signal,
    apply_time_decay_exit,
    vix_gate,
    load_macro as sg_load_macro,
)
from v1.portfolio.portfolio import equal_weight_sizes, CAPITAL

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURE_DIR  = Path("data/v1/features")
RESULTS_DIR  = Path("data/v1/results")
RESEARCH_DIR = Path("data/v1/research")
SIGNAL_DIR   = Path("data/v1/signals")
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_portfolio() -> pd.DataFrame:
    path = RESULTS_DIR / "portfolio_comparison.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Portfolio comparison not found at {path}. Run portfolio.py first.")
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df


def _load_oos_selection() -> pd.DataFrame:
    path = RESULTS_DIR / "oos_selection.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_regime_signals() -> pd.DataFrame:
    path = SIGNAL_DIR / "regime_signals.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Regime signals not found at {path}. Run signal_generation.py first.")
    return pd.read_parquet(path)


def _load_signal_matrix(name: str) -> pd.DataFrame:
    """Load a named signal matrix file; fall back to regime_signals if missing."""
    path = SIGNAL_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return _load_regime_signals()


def _load_returns_matrix() -> pd.DataFrame:
    path = FEATURE_DIR / "returns_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Returns matrix not found at {path}. Run feature_engineering.py first.")
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df


def _production_method(oos_df: pd.DataFrame) -> tuple:
    """Return (method_label, portfolio_col) for best OOS Sharpe. Falls back to equal_weight."""
    if oos_df.empty or "oos_sharpe" not in oos_df.columns or "method" not in oos_df.columns:
        return "equal weight", "equal_weight"
    idx   = oos_df["oos_sharpe"].idxmax()
    label = str(oos_df.loc[idx, "method"])
    col   = label.replace(" ", "_").replace("-", "_")
    return label, col


def _signal_matrix_for_method(col_name: str) -> pd.DataFrame:
    """Signal matrix for a portfolio column: 'multi' methods → multi_signals.parquet, else regime."""
    if "multi" in col_name:
        return _load_signal_matrix("multi_signals")
    return _load_regime_signals()


# ── Section 1: Upside/Downside Capture Ratios ─────────────────────────────────

def compute_capture_ratios(
    portfolio_df: pd.DataFrame,
    spy_ret: pd.Series,
) -> pd.DataFrame:
    """Per method: upside_capture (>1 beats SPY on up days), downside_capture (<1 better),
    capture_ratio = up/dn (>1 = convex)."""
    port_returns = portfolio_df.pct_change().dropna()
    rows = []

    for col in port_returns.columns:
        pret   = port_returns[col]
        common = pd.concat([pret, spy_ret.rename("spy")], axis=1).dropna()
        if len(common) < 60:
            continue

        p, s    = common[col], common["spy"]
        up_mask = s > 0
        dn_mask = s < 0

        mean_spy_up = s[up_mask].mean()
        mean_spy_dn = s[dn_mask].mean()

        up_cap = (p[up_mask].mean() / mean_spy_up
                  if up_mask.sum() > 0 and mean_spy_up != 0 else np.nan)
        dn_cap = (p[dn_mask].mean() / mean_spy_dn
                  if dn_mask.sum() > 0 and mean_spy_dn != 0 else np.nan)
        cap_ratio = (up_cap / dn_cap
                     if not np.isnan(up_cap) and not np.isnan(dn_cap) and dn_cap != 0
                     else np.nan)

        rows.append({
            "method"           : col,
            "upside_capture"   : round(float(up_cap),    3),
            "downside_capture" : round(float(dn_cap),    3),
            "capture_ratio"    : round(float(cap_ratio), 3),
            "n_up_days"        : int(up_mask.sum()),
            "n_down_days"      : int(dn_mask.sum()),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("capture_ratio", ascending=False)
        .reset_index(drop=True)
    )


def _print_capture_table(df: pd.DataFrame) -> None:
    W = 90
    print("\n" + "=" * W)
    print("  SECTION 1: UPSIDE/DOWNSIDE CAPTURE RATIOS  (full backtest period)")
    print("  Capture Ratio >1 = convex (beats B&H on up days AND protects on down)")
    print("  Capture Ratio <1 = concave (lags on up days, reduces draw — low beta)")
    print("=" * W)
    print(
        f"  {'Method':<30} {'Up Cap':>8} {'Dn Cap':>8} {'Cap Ratio':>10}"
        f" {'Up Days':>8} {'Dn Days':>8}"
    )
    print("  " + "-" * (W - 2))

    for _, row in df.iterrows():
        flag = ""
        cr = row["capture_ratio"]
        if np.isnan(cr):
            flag = ""
        elif cr >= 1.20:
            flag = "  <-- strongly convex"
        elif cr >= 1.0:
            flag = "  <-- convex"
        else:
            flag = "  <-- concave (problem)"

        up_s  = f"{row['upside_capture']:>8.3f}"   if not np.isnan(row["upside_capture"])   else "     N/A"
        dn_s  = f"{row['downside_capture']:>8.3f}"  if not np.isnan(row["downside_capture"])  else "     N/A"
        cr_s  = f"{row['capture_ratio']:>10.3f}"    if not np.isnan(row["capture_ratio"])     else "       N/A"

        print(
            f"  {row['method']:<30} {up_s} {dn_s} {cr_s}"
            f" {row['n_up_days']:>8} {row['n_down_days']:>8}{flag}"
        )

    print("=" * W)
    valid = df[df["capture_ratio"].notna()]
    if len(valid):
        best  = valid.loc[valid["capture_ratio"].idxmax()]
        worst = valid.loc[valid["capture_ratio"].idxmin()]
        n_convex = (valid["capture_ratio"] >= 1.0).sum()
        print(f"\n  Best  capture ratio : {best['method']}  ({best['capture_ratio']:.3f})")
        print(f"  Worst capture ratio : {worst['method']}  ({worst['capture_ratio']:.3f})")
        print(f"  Convex methods      : {n_convex} / {len(valid)}")
    print()


# ── Section 2: Cash Drag Analysis ─────────────────────────────────────────────

def compute_cash_drag(
    signals: pd.DataFrame,
    spy_ret: pd.Series,
    regimes: pd.Series,
) -> tuple:
    """Quantify cash-time cost. Returns (summary_df scalar metrics, regime_df per-regime exposure)."""
    tickers = [t for t in signals.columns if t in TICKER_LIST]
    n = len(tickers)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()

    sig       = signals[tickers].reindex(spy_ret.index).fillna(0)
    gross_exp = sig.abs().sum(axis=1) / n          # universe-active fraction
    cash_frac = 1.0 - gross_exp

    spy_aligned = spy_ret.reindex(sig.index).fillna(0)
    daily_drag  = cash_frac * spy_aligned          # missed return/day
    ann_drag    = float(daily_drag.mean() * 252)

    below50     = int((gross_exp < 0.50).sum())
    total       = len(gross_exp)

    summary_rows = [
        {"metric": "Avg Gross Exposure %",
         "value":  f"{gross_exp.mean()*100:.1f}%"},
        {"metric": "Avg Cash Allocation %",
         "value":  f"{cash_frac.mean()*100:.1f}%"},
        {"metric": "Annualised Cash Drag (SPY basis)",
         "value":  f"{ann_drag*100:.2f}%"},
        {"metric": "Days Below 50% Invested",
         "value":  f"{below50} / {total}  ({below50/total*100:.1f}%)"},
        {"metric": "Min Gross Exposure",
         "value":  f"{gross_exp.min()*100:.1f}%"},
        {"metric": "Max Gross Exposure",
         "value":  f"{gross_exp.max()*100:.1f}%"},
    ]

    regime_rows = []
    for regime in REGIME_ORDER:
        mask = (regimes == regime).reindex(gross_exp.index).fillna(False)
        n_days = int(mask.sum())
        if n_days < 5:
            continue
        regime_rows.append({
            "regime"           : regime,
            "avg_exposure_pct" : round(float(gross_exp[mask].mean() * 100), 1),
            "avg_cash_pct"     : round(float(cash_frac[mask].mean() * 100),  1),
            "ann_drag_pct"     : round(float(daily_drag[mask].mean() * 252 * 100), 2),
            "n_days"           : n_days,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(regime_rows)


def _print_cash_drag(summary_df: pd.DataFrame, regime_df: pd.DataFrame,
                     prod_label: str) -> None:
    W = 80
    print("\n" + "=" * W)
    print(f"  SECTION 2: CASH DRAG ANALYSIS  --  {prod_label}")
    print("=" * W)
    for _, row in summary_df.iterrows():
        print(f"  {row['metric']:<42}  {row['value']}")

    if not regime_df.empty:
        print()
        print(f"  {'Regime':<36} {'Avg Exp%':>9} {'Avg Cash%':>10} {'Ann Drag%':>10} {'N Days':>8}")
        print("  " + "-" * (W - 2))
        for _, row in regime_df.iterrows():
            label = REGIME_LABELS.get(row["regime"], row["regime"])
            flag  = (
                "  <-- heavy cash in bull!" if row["avg_cash_pct"] > 60 and "bull" in row["regime"]
                else "  <-- low exposure in fear" if row["avg_cash_pct"] > 50 and "bear" in row["regime"]
                else ""
            )
            print(
                f"  {label:<36} {row['avg_exposure_pct']:>8.1f}%"
                f" {row['avg_cash_pct']:>9.1f}% {row['ann_drag_pct']:>+9.2f}%"
                f" {row['n_days']:>8}{flag}"
            )
    print("=" * W)
    print()


# ── Section 3: Asset Class Contribution ───────────────────────────────────────

def compute_asset_class_contribution(
    signals: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    spy_ret: pd.Series,
) -> pd.DataFrame:
    """Per asset-class annualised P&L bps split SPY-up vs SPY-dn days.
    EW-normalised signals; position[t]=signal[t-1]/n_active[t-1] (no look-ahead)."""
    tickers = [t for t in signals.columns if t in returns_matrix.columns]
    sig = signals[tickers].reindex(returns_matrix.index).fillna(0)
    ret = returns_matrix[tickers].reindex(sig.index)

    # EW portfolio weights per day
    n_active = sig.abs().sum(axis=1).replace(0, np.nan)
    weights  = sig.divide(n_active, axis=0).fillna(0)

    # shift(1): yesterday's weight earns today's return (no look-ahead)
    pos          = weights.shift(1).fillna(0)
    contributions = pos * ret   # daily fractional P&L per ticker

    spy_aligned = spy_ret.reindex(contributions.index).fillna(0)
    up_mask     = spy_aligned > 0
    dn_mask     = spy_aligned < 0

    rows = []
    asset_classes = sorted(set(ASSET_CLASS.get(t, "stock") for t in tickers))
    for ac in asset_classes:
        ac_tickers = [t for t in tickers if ASSET_CLASS.get(t) == ac]
        if not ac_tickers:
            continue
        ac_contrib = contributions[ac_tickers].sum(axis=1)

        # Annualised bps: mean_daily × 252 × 10000
        total_bps  = float(ac_contrib.mean()       * 252 * 10_000)
        up_bps     = float(ac_contrib[up_mask].mean() * 252 * 10_000)
        dn_bps     = float(ac_contrib[dn_mask].mean() * 252 * 10_000)
        avg_weight = float(pos[ac_tickers].abs().sum(axis=1).mean() * 100)  # % portfolio

        rows.append({
            "asset_class"  : ac,
            "tickers"      : ", ".join(ac_tickers),
            "total_bps"    : round(total_bps,  1),
            "up_day_bps"   : round(up_bps,     1),
            "down_day_bps" : round(dn_bps,     1),
            "avg_weight_pct": round(avg_weight, 1),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("total_bps", ascending=False)
        .reset_index(drop=True)
    )


def _print_asset_class_table(df: pd.DataFrame) -> None:
    W = 94
    print("\n" + "=" * W)
    print("  SECTION 3: ASSET CLASS P&L CONTRIBUTION  (annualised basis points)")
    print("  Negative up_day_bps = class DRAGS on rally days  <-- primary culprit")
    print("=" * W)
    print(
        f"  {'Asset Class':<14} {'Total bps':>10} {'Up-Day bps':>11}"
        f" {'Dn-Day bps':>11} {'Avg Wt%':>8}  Tickers"
    )
    print("  " + "-" * (W - 2))

    for _, row in df.iterrows():
        flag = ""
        if row["up_day_bps"] < 0:
            flag = "  <-- RALLY DRAG"
        elif row["up_day_bps"] > 0 and row["down_day_bps"] > 0:
            flag = "  <-- adds on both"
        # Truncate long ticker list
        tk_str = row["tickers"]
        if len(tk_str) > 28:
            tk_str = tk_str[:25] + "..."
        print(
            f"  {row['asset_class']:<14} {row['total_bps']:>10.1f} {row['up_day_bps']:>+11.1f}"
            f" {row['down_day_bps']:>+11.1f} {row['avg_weight_pct']:>7.1f}%  {tk_str}{flag}"
        )

    print("=" * W)
    drag = df[df["up_day_bps"] < 0]
    if len(drag):
        print(f"\n  {len(drag)} class(es) with NEGATIVE up-day contribution: "
              f"{drag['asset_class'].tolist()}")
        print("  --> These classes hold the portfolio back on SPY up-days.")
    print()


# ── Section 4: Per-Ticker Signal Efficiency ───────────────────────────────────

def compute_ticker_efficiency(
    signals: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    spy_ret: pd.Series,
) -> pd.DataFrame:
    """Per ticker: upside_capture vs SPY-up, active_pct, cash_drag_days
    (signal=0 ∧ ret>0), efficiency_score = up_cap × active_pct/100. Ranked desc."""
    tickers = [t for t in signals.columns if t in returns_matrix.columns]
    spy_aligned  = spy_ret.reindex(returns_matrix.index).fillna(0)
    up_mask      = spy_aligned > 0
    mean_spy_up  = float(spy_aligned[up_mask].mean()) if up_mask.sum() > 0 else np.nan

    rows = []
    for ticker in tickers:
        sig = signals[ticker].reindex(returns_matrix.index).fillna(0)
        ret = returns_matrix[ticker]

        # shift(1) so yesterday's signal earns today's return
        strat_ret = sig.shift(1).fillna(0) * ret

        # Per-ticker upside capture vs SPY (not vs own ret)
        if mean_spy_up and mean_spy_up != 0 and up_mask.sum() > 5:
            up_cap = float(strat_ret[up_mask].mean() / mean_spy_up)
        else:
            up_cap = np.nan

        total_days     = len(sig)
        active_days    = int((sig != 0).sum())
        active_pct     = active_days / total_days * 100 if total_days > 0 else 0.0
        cash_drag_days = int(((sig == 0) & (ret > 0)).sum())
        efficiency     = up_cap * (active_pct / 100.0) if not np.isnan(up_cap) else np.nan

        rows.append({
            "ticker"          : ticker,
            "asset_class"     : ASSET_CLASS.get(ticker, "unknown"),
            "upside_capture"  : round(up_cap,          3),
            "active_pct"      : round(active_pct,      1),
            "cash_drag_days"  : cash_drag_days,
            "efficiency_score": round(efficiency,      3),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("efficiency_score", ascending=False)
        .reset_index(drop=True)
    )
    return df


def _print_ticker_efficiency(df: pd.DataFrame) -> None:
    W = 90
    print("\n" + "=" * W)
    print("  SECTION 4: PER-TICKER SIGNAL EFFICIENCY")
    print("  Score = upside_capture × (active_pct/100)  |  Bottom 5 = drag candidates")
    print("=" * W)
    print(
        f"  {'Ticker':<8} {'Class':<14} {'Up Cap':>8} {'Active%':>8}"
        f" {'CashDragDays':>13} {'Score':>8}"
    )
    print("  " + "-" * (W - 2))

    bottom5 = set(
        df[df["efficiency_score"].notna()].tail(5)["ticker"].tolist()
    )

    for _, row in df.iterrows():
        flag   = "  <-- DRAG CANDIDATE" if row["ticker"] in bottom5 else ""
        up_s   = f"{row['upside_capture']:>8.3f}"   if not np.isnan(row["upside_capture"])    else "     N/A"
        sc_s   = f"{row['efficiency_score']:>8.3f}" if not np.isnan(row["efficiency_score"])  else "     N/A"
        print(
            f"  {row['ticker']:<8} {row['asset_class']:<14} {up_s}"
            f" {row['active_pct']:>7.1f}% {row['cash_drag_days']:>13} {sc_s}{flag}"
        )

    print("=" * W)
    print(f"\n  Drag candidates (bottom 5): {sorted(bottom5)}")
    avg_up = df["upside_capture"].dropna().mean()
    print(f"  Average upside capture across all tickers: {avg_up:.3f}")
    print()


# ── Section 5: Filter Impact Analysis ─────────────────────────────────────────

def _raw_ma_signal(df: pd.DataFrame, ticker: str, gate: pd.Series) -> pd.Series:
    """
    Base MA crossover signal with VIX gate only — no post-processors.
    Equity/sector/stock: long-only MA50/200 (or MA100/300 for sector_etf).
    """
    ac = ASSET_CLASS.get(ticker, "stock")
    if ac == "sector_etf":
        ma_fast = df["Close"].rolling(100).mean()
        ma_slow = df["Close"].rolling(300).mean()
    else:
        ma_fast = df["Close"].rolling(50).mean()
        ma_slow = df["Close"].rolling(200).mean()

    return ((ma_fast > ma_slow).astype(int) * gate).astype(int)


def _portfolio_upside_capture(
    signal_matrix: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    spy_ret: pd.Series,
) -> float:
    """Equal-weight portfolio upside capture for a given signal matrix."""
    tickers = [t for t in signal_matrix.columns if t in returns_matrix.columns]
    if not tickers:
        return np.nan

    sig      = signal_matrix[tickers].reindex(returns_matrix.index).fillna(0)
    ret      = returns_matrix[tickers].reindex(sig.index)
    n_active = sig.abs().sum(axis=1).replace(0, np.nan)
    weights  = sig.divide(n_active, axis=0).fillna(0)
    port_ret = (weights.shift(1).fillna(0) * ret).sum(axis=1)

    spy_aligned  = spy_ret.reindex(port_ret.index).fillna(0)
    up_mask      = spy_aligned > 0
    mean_spy_up  = float(spy_aligned[up_mask].mean())

    if mean_spy_up == 0 or up_mask.sum() < 20:
        return np.nan
    return float(port_ret[up_mask].mean() / mean_spy_up)


def compute_filter_impact(
    returns_matrix: pd.DataFrame,
    spy_ret: pd.Series,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure how much each post-processor costs in upside capture.

    Builds 5 progressive filter levels for equity/sector/stock tickers:
      raw_ma     : MA crossover + VIX gate only
      rsi_filter : + RSI entry block (no new longs when RSI > 70)
      min_hold   : + minimum hold 5 days
      atr_stop   : + ATR trailing stop (3× ATR)
      time_decay : + time decay exit (126 days stale)

    Each level produces a signal matrix; upside capture is measured as
    the equal-weight portfolio upside capture over the full backtest period.
    Delta = change vs the previous level (negative = filter costs upside).
    """
    equity_tickers = [
        t for t in TICKER_LIST
        if ASSET_CLASS.get(t) in {"equity_index", "sector_etf", "stock"}
        and (FEATURE_DIR / f"{t}.parquet").exists()
    ]

    LEVELS = ["raw_ma", "rsi_filter", "min_hold", "atr_stop", "time_decay"]
    # Accumulate per-level signal series in a dict: level → {ticker: Series}
    level_signals: dict = {lv: {} for lv in LEVELS}

    print(f"  Building filter-level signals for {len(equity_tickers)} equity tickers...")
    for ticker in equity_tickers:
        df = pd.read_parquet(FEATURE_DIR / f"{ticker}.parquet")
        if not {"Close", "rsi_14", "atr_14"}.issubset(df.columns):
            continue

        gate = vix_gate(macro, df.index)

        # Level 0: raw MA (VIX gate only, no RSI/min-hold/stop/decay)
        s0 = _raw_ma_signal(df, ticker, gate)
        level_signals["raw_ma"][ticker] = s0

        # Level 1: + RSI entry filter
        s1 = apply_rsi_entry_filter(s0.copy(), df["rsi_14"])
        level_signals["rsi_filter"][ticker] = s1

        # Level 2: + min-hold (5 trading days)
        s2 = apply_min_hold_filter(s1.copy())
        level_signals["min_hold"][ticker] = s2

        # Level 3: + ATR trailing stop (3× ATR)
        s3 = apply_trailing_stop_signal(s2.copy(), df["Close"], df["atr_14"])
        level_signals["atr_stop"][ticker] = s3

        # Level 4: + time decay exit (stale positions > 126 days)
        s4 = apply_time_decay_exit(s3.copy(), df["Close"])
        level_signals["time_decay"][ticker] = s4

    # Build DataFrames from the accumulated series
    # Align all tickers to the same index (intersection)
    common_idx = None
    for ticker_dict in level_signals.values():
        for s in ticker_dict.values():
            common_idx = s.index if common_idx is None else common_idx.intersection(s.index)

    rows    = []
    prev_up = None
    for lv in LEVELS:
        td = level_signals[lv]
        if not td:
            continue
        mat = pd.DataFrame(
            {t: s.reindex(common_idx).fillna(0) for t, s in td.items()}
        )
        up_cap = _portfolio_upside_capture(mat, returns_matrix, spy_ret)
        delta  = (up_cap - prev_up) if prev_up is not None else 0.0
        rows.append({
            "filter_level"   : lv,
            "upside_capture" : round(float(up_cap),  3),
            "delta_vs_prev"  : round(float(delta),   3),
        })
        prev_up = up_cap

    return pd.DataFrame(rows)


def _print_filter_impact(df: pd.DataFrame) -> None:
    W = 72
    print("\n" + "=" * W)
    print("  SECTION 5: FILTER IMPACT ANALYSIS  (equity/sector/stock)")
    print("  delta < 0 = filter reduces upside capture  |  delta > 0 = filter adds")
    print("=" * W)
    print(
        f"  {'Filter Level':<20} {'Up Capture':>12} {'Delta':>10}  Note"
    )
    print("  " + "-" * (W - 2))

    for _, row in df.iterrows():
        d = row["delta_vs_prev"]
        if row["filter_level"] == "raw_ma":
            note = "(baseline)"
        elif d <= -0.05:
            note = "<-- MAJOR cost"
        elif d <= -0.02:
            note = "<-- notable cost"
        elif d >= 0.02:
            note = "<-- adds upside"
        else:
            note = ""

        up_s = f"{row['upside_capture']:>12.3f}" if not np.isnan(row["upside_capture"]) else "         N/A"
        d_s  = f"{d:>+10.3f}" if row["filter_level"] != "raw_ma" else "          --"
        print(f"  {row['filter_level']:<20} {up_s} {d_s}  {note}")

    print("=" * W)

    costs = df[df["filter_level"] != "raw_ma"]
    if len(costs):
        worst_i = costs["delta_vs_prev"].idxmin()
        worst   = df.loc[worst_i]
        print(f"\n  Costliest filter for upside: {worst['filter_level']}"
              f"  (delta {worst['delta_vs_prev']:+.3f})")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  Capture Diagnostic  --  Upside/Downside Decomposition")
    print("  (read-only — no pipeline files modified)")
    print("=" * 70)

    # ── Load shared data ────────────────────────────────────────────────────
    print("\nLoading data...")
    macro          = _load_macro()
    spy_close      = _load_spy_close()
    portfolio_df   = _load_portfolio()
    returns_matrix = _load_returns_matrix()
    oos_df         = _load_oos_selection()
    regime_signals = _load_regime_signals()

    prod_label, prod_col = _production_method(oos_df)
    prod_signals = _signal_matrix_for_method(prod_col)

    # Align to regime signals index (backtest window)
    idx               = regime_signals.index
    returns_aligned   = returns_matrix.reindex(idx)
    spy_close_aligned = spy_close.reindex(idx).ffill()
    spy_ret           = returns_aligned["SPY"] if "SPY" in returns_aligned.columns else pd.Series(dtype=float)

    regimes = label_regimes(macro["vix"], spy_close_aligned, idx)

    print(f"  Portfolio curves : {portfolio_df.shape}")
    print(f"  Returns matrix   : {returns_matrix.shape}  (raw)")
    print(f"  Regime signals   : {regime_signals.shape}")
    print(f"  Production method: {prod_label}  (col: {prod_col})")
    print(f"  Date range       : {idx[0].date()} → {idx[-1].date()}")
    print(f"  SPY return rows  : {spy_ret.dropna().shape[0]}")

    # ── Section 1 ────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [1/5]  Upside/Downside Capture Ratios  (all methods)")
    print("─" * 70)
    # Use SPY log returns aligned to portfolio_df index for Section 1
    spy_for_port = returns_matrix["SPY"].reindex(portfolio_df.index) if "SPY" in returns_matrix.columns else pd.Series(dtype=float)
    cap_df = compute_capture_ratios(portfolio_df, spy_for_port)
    _print_capture_table(cap_df)

    # ── Section 2 ────────────────────────────────────────────────────────────
    print("─" * 70)
    print(f"  [2/5]  Cash Drag Analysis  ({prod_label})")
    print("─" * 70)
    drag_summary, drag_regime = compute_cash_drag(
        prod_signals.reindex(idx).fillna(0),
        spy_ret, regimes,
    )
    _print_cash_drag(drag_summary, drag_regime, prod_label)

    # ── Section 3 ────────────────────────────────────────────────────────────
    print("─" * 70)
    print(f"  [3/5]  Asset Class Contribution  ({prod_label})")
    print("─" * 70)
    ac_df = compute_asset_class_contribution(
        prod_signals.reindex(idx).fillna(0),
        returns_aligned, spy_ret,
    )
    _print_asset_class_table(ac_df)

    # ── Section 4 ────────────────────────────────────────────────────────────
    print("─" * 70)
    print("  [4/5]  Per-Ticker Signal Efficiency  (regime signal)")
    print("─" * 70)
    tick_df = compute_ticker_efficiency(
        regime_signals.reindex(idx).fillna(0),
        returns_aligned, spy_ret,
    )
    _print_ticker_efficiency(tick_df)

    # ── Section 5 ────────────────────────────────────────────────────────────
    print("─" * 70)
    print("  [5/5]  Filter Impact Analysis  (equity/sector/stock)")
    print("─" * 70)
    sg_macro   = sg_load_macro()
    filter_df  = compute_filter_impact(returns_aligned, spy_ret, sg_macro)
    _print_filter_impact(filter_df)

    # ── Save parquets ─────────────────────────────────────────────────────────
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    out_cap    = RESEARCH_DIR / "capture_ratios.parquet"
    out_drag   = RESEARCH_DIR / "cash_drag.parquet"
    out_ac     = RESEARCH_DIR / "asset_class_contribution.parquet"
    out_tick   = RESEARCH_DIR / "ticker_efficiency.parquet"
    out_filter = RESEARCH_DIR / "filter_impact.parquet"

    cap_df.to_parquet(   out_cap,    engine="pyarrow", compression="snappy", index=False)
    drag_regime.to_parquet(out_drag, engine="pyarrow", compression="snappy", index=False)
    ac_df.to_parquet(    out_ac,     engine="pyarrow", compression="snappy", index=False)
    tick_df.to_parquet(  out_tick,   engine="pyarrow", compression="snappy", index=False)
    filter_df.to_parquet(out_filter, engine="pyarrow", compression="snappy", index=False)

    print("Results saved:")
    print(f"  {out_cap}      (Section 1 — capture ratios)")
    print(f"  {out_drag}        (Section 2 — cash drag by regime)")
    print(f"  {out_ac}  (Section 3 — asset class contribution)")
    print(f"  {out_tick}    (Section 4 — ticker efficiency)")
    print(f"  {out_filter}      (Section 5 — filter impact)")
    print()


if __name__ == "__main__":
    main()
