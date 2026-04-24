"""
v2/ui/dashboard.py
------------------
Streamlit dashboard for the v2 macro regime rotation system.

Visual identity matches v1 (dark grey tone-on-tone, same palette, same layout),
but the data layer reads from `data/v2/` parquets — regime probabilities,
macro features, walk-forward results and the strategy equity curve.

Run:
  streamlit run v2/ui/dashboard.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from v2.ui.styles import CSS, _layout, PALETTE, LABELS, get_color, get_label
from v2.ui.charts import (
    metrics, metrics_table,
    chart_equity, chart_equity_risk_adjusted, chart_drawdown,
    chart_monte_carlo, chart_mc_histogram,
    chart_walk_forward, chart_monthly_heatmap,
    chart_regime_stack, chart_regime_label_strip, chart_current_allocation,
    chart_macro_overlay, chart_regime_bars,
    chart_paper_portfolio, chart_regime_transitions,
)
from v2.ui.data_loaders import (
    load_equity_curves, load_net_returns, load_portfolio_weights,
    load_walk_forward, load_regime_probabilities, load_regime_labels,
    load_macro_features, load_market_features, load_etf_prices,
    load_paper_state, load_paper_history, load_paper_trades,
    run_monte_carlo,
)
from v2.pipeline.data_pipeline import get_asset_class_map
from v2.portfolio.regime_allocations import REGIME_NAMES


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Greatrex V2 — Macro Regime Rotation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)


# ── Header metrics ───────────────────────────────────────────────────────────
def _delta_vs_spy(stat_val: float, spy_val: float, fmt: str = "{:+.1%}",
                   positive_good: bool = True) -> str:
    diff = stat_val - spy_val
    return fmt.format(diff)


def render_header(eq_df: pd.DataFrame, wf: pd.DataFrame | None = None):
    lcol, rcol = st.columns([3, 1])
    with lcol:
        st.markdown("## 📈 V2 Macro Regime Rotation — Dashboard")
        st.markdown(
            "<div style='color:#888;font-size:.85rem;'>"
            "Monthly rebalance · 28-ETF cross-asset · 6-regime hybrid classifier · "
            "Regime-aware vol targeting"
            "</div>",
            unsafe_allow_html=True,
        )
    with rcol:
        if not eq_df.empty:
            st.markdown(
                f"<div style='text-align:right;color:#666;font-size:.78rem;'>"
                f"Period: {eq_df.index.min().date()} → {eq_df.index.max().date()}<br>"
                f"{len(eq_df):,} trading days"
                f"</div>",
                unsafe_allow_html=True,
            )

    if eq_df.empty or "strategy_net" not in eq_df.columns:
        st.info("No equity curves found yet. Run `python -m v2.pipeline.backtester`.")
        return

    strat_ret = eq_df["strategy_net"].pct_change().dropna()
    spy_ret   = eq_df["spy"].pct_change().dropna() if "spy" in eq_df.columns else pd.Series(dtype=float)
    m = metrics(strat_ret)
    ms = metrics(spy_ret) if not spy_ret.empty else {}

    # OOS Sharpe from walk-forward: mean of OOS windows when available
    oos_sharpe = None
    if wf is not None and not wf.empty and "oos_sharpe" in wf.columns:
        oos_sharpe = float(wf["oos_sharpe"].mean())
    sharpe_disp = oos_sharpe if oos_sharpe is not None else m["sharpe"]

    # Active Sharpe — walk-forward OOS active return (strategy − SPY) Sharpe
    active_sharpe = None
    active_ret = strat_ret - spy_ret.reindex(strat_ret.index).fillna(0) if not spy_ret.empty else None
    if active_ret is not None and not active_ret.empty:
        vol = active_ret.std() * (252 ** 0.5)
        active_sharpe = (active_ret.mean() * 252) / vol if vol > 0 else None

    pf = m.get("profit_factor", float("inf"))

    st.markdown('<div class="section-head">Full-sample out-of-sample performance</div>',
                unsafe_allow_html=True)

    cols = st.columns(9)

    def _delta(val, ref, pct=True):
        d = val - ref
        return f"{d*100:+.1f}%" if pct else f"{d:+.2f}"

    with cols[0]:
        st.metric("Ann. Return", f"{m['ann_r']*100:.1f}%",
                  delta=_delta(m['ann_r'], ms.get('ann_r', 0)),
                  help="CAGR over the backtest. Arrow shows vs SPY buy & hold.")
    with cols[1]:
        st.metric("Sharpe Ratio", f"{sharpe_disp:.2f}",
                  delta=_delta(sharpe_disp, ms.get('sharpe', 0), pct=False),
                  help="OOS walk-forward Sharpe when available; else full-sample. "
                       ">1.0 good, >2.0 exceptional. Arrow vs SPY.")
    with cols[2]:
        st.metric("Volatility", f"{m['vol']*100:.1f}%",
                  delta=_delta(m['vol'], ms.get('vol', 0)), delta_color="inverse",
                  help="Annualised std of daily returns. Lower = smoother ride.")
    with cols[3]:
        st.metric("Max Drawdown", f"{m['max_dd']*100:.1f}%",
                  delta=_delta(m['max_dd'], ms.get('max_dd', 0)),
                  help="Worst peak-to-trough loss. Green arrow = shallower than SPY.")
    with cols[4]:
        st.metric("Calmar Ratio", f"{m['calmar']:.2f}",
                  delta=_delta(m['calmar'], ms.get('calmar', 0), pct=False),
                  help="Annualised return / |max DD|. >1.0 good.")
    with cols[5]:
        st.metric("Win Rate", f"{m['win_rate']*100:.0f}%",
                  help="Percentage of active trading days with positive returns.")
    with cols[6]:
        st.metric("Profit Factor", f"{pf:.2f}" if pf < 100 else "∞",
                  help="Gross profit ÷ gross loss on daily returns. >1.5 good.")
    with cols[7]:
        st.metric("Active Sharpe",
                  f"{active_sharpe:.2f}" if active_sharpe is not None else "—",
                  help="Sharpe of excess returns (strategy − SPY). "
                       ">0.5 = meaningful alpha.")
    with cols[8]:
        st.metric("VaR 95% (1d)", f"{m['var_95']*100:.2f}%",
                  delta=_delta(m['var_95'], ms.get('var_95', 0)), delta_color="inverse",
                  help="Daily 95% historical VaR. Lower = smaller tail risk.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    eq_df = load_equity_curves()
    net_returns = load_net_returns()
    weights_df = load_portfolio_weights()
    wf = load_walk_forward()
    probs = load_regime_probabilities()
    labels = load_regime_labels()
    macro = load_macro_features()
    market = load_market_features()
    prices = load_etf_prices()

    render_header(eq_df, wf)

    tab_port, tab_sig, tab_sp, tab_bt, tab_val, tab_risk = st.tabs([
        "  📈  Portfolio",
        "  📡  Signals",
        "  📊  vs S&P 500",
        "  🔬  Backtest",
        "  🔍  Validation",
        "  ⚠️  Risk",
    ])

    # ── Portfolio tab ────────────────────────────────────────────────────────
    with tab_port:
        if eq_df.empty:
            st.info("No equity curves found.")
        else:
            cols_to_show = [c for c in ["strategy_net", "strategy_gross", "spy"]
                            if c in eq_df.columns]
            st.plotly_chart(
                chart_equity(eq_df, columns=cols_to_show),
                theme=None, use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True},
            )
            st.plotly_chart(
                chart_equity_risk_adjusted(eq_df, columns=cols_to_show),
                theme=None, use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True},
            )
            st.plotly_chart(
                chart_drawdown(eq_df, columns=cols_to_show),
                theme=None, use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True},
            )

            st.markdown('<div class="section-head">Strategy summary</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                metrics_table(eq_df[cols_to_show]),
                use_container_width=True, hide_index=True,
            )

            st.markdown('<div class="section-head">Monthly returns — Strategy (Net)</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                chart_monthly_heatmap(net_returns),
                theme=None, use_container_width=True,
                config={"scrollZoom": False, "displayModeBar": False},
            )

    # ── Signals tab (regime transitions + current allocation) ───────────────
    with tab_sig:
        if probs.empty:
            st.info("No regime probabilities found.")
        else:
            # Current regime card
            latest = probs.iloc[-1]
            dominant = latest.idxmax()
            conf = latest.max() * 100
            c_date = probs.index.max().date()

            st.markdown('<div class="section-head">Current regime</div>',
                        unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Dominant regime", get_label(dominant), delta=f"{conf:.0f}% confidence")
            with k2:
                bull = latest.get("expansion", 0) + latest.get("recovery", 0)
                st.metric("Bull score", f"{bull*100:.0f}%",
                          delta=f"{(bull-0.45)*100:+.0f}pp vs gate")
            with k3:
                bear = latest.get("recession", 0) + latest.get("stagflation", 0)
                st.metric("Bear score", f"{bear*100:.0f}%",
                          delta=f"{(bear-0.45)*100:+.0f}pp vs gate")
            with k4:
                st.metric("As of", str(c_date))

            if not labels.empty:
                st.plotly_chart(
                    chart_regime_transitions(labels, lookback_days=252),
                    theme=None, use_container_width=True,
                    config={"scrollZoom": False, "displayModeBar": False},
                )

            st.plotly_chart(
                chart_regime_stack(probs.loc["2008":]),
                theme=None, use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True},
            )
            if not labels.empty:
                st.plotly_chart(
                    chart_regime_label_strip(labels.loc["2008":]),
                    theme=None, use_container_width=True,
                    config={"scrollZoom": False, "displayModeBar": False},
                )

            # Current allocation
            if not weights_df.empty:
                st.markdown('<div class="section-head">Current ETF allocation</div>',
                            unsafe_allow_html=True)
                ac_map = get_asset_class_map()
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.plotly_chart(
                        chart_current_allocation(weights_df.iloc[-1], ac_map),
                        theme=None, use_container_width=True,
                        config={"scrollZoom": False, "displayModeBar": False},
                    )
                with c2:
                    latest_w = weights_df.iloc[-1]
                    latest_w = latest_w[latest_w > 0.005].sort_values(ascending=False)
                    rows = [{"Ticker": t, "Weight": f"{w*100:.1f}%",
                             "Asset Class": ac_map.get(t, "—")}
                            for t, w in latest_w.items()]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True)

            # Per-regime attribution
            if not net_returns.empty and not labels.empty:
                common = net_returns.index.intersection(labels.index)
                r = net_returns.loc[common]
                l = labels.loc[common]
                regime_ids = {0: "Expansion", 1: "Slowdown", 2: "Recession",
                              3: "Recovery", 4: "Stagflation", 5: "Late Cycle"}
                attribution = {}
                for rid, name in regime_ids.items():
                    mask = l == rid
                    if mask.sum() < 10:
                        continue
                    rr = r[mask]
                    ann_ret = rr.mean() * 252
                    ann_vol = rr.std() * np.sqrt(252)
                    sh = ann_ret / ann_vol if ann_vol > 0 else 0
                    attribution[name] = {
                        "days": int(mask.sum()),
                        "ann_return": ann_ret,
                        "ann_vol": ann_vol,
                        "sharpe": sh,
                    }
                st.markdown('<div class="section-head">Per-regime attribution</div>',
                            unsafe_allow_html=True)
                a1, a2 = st.columns([2, 1])
                with a1:
                    st.plotly_chart(
                        chart_regime_bars(attribution),
                        theme=None, use_container_width=True,
                        config={"displayModeBar": False},
                    )
                with a2:
                    rows = [{
                        "Regime": name,
                        "Days": d["days"],
                        "Ann Return": f"{d['ann_return']*100:+.1f}%",
                        "Ann Vol": f"{d['ann_vol']*100:.1f}%",
                        "Sharpe": f"{d['sharpe']:.2f}",
                    } for name, d in attribution.items()]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True)

    # ── vs S&P 500 tab (live paper trader view) ─────────────────────────────
    with tab_sp:
        state = load_paper_state()
        hist = load_paper_history()
        trades = load_paper_trades()

        if not state:
            st.info("Paper trading not yet initialized. Run "
                    "`python -m v2.scripts.paper_trader init`.")
        else:
            nav = state.get("nav", state.get("portfolio_value",
                                              state.get("equity", 100_000)))
            cash = state.get("cash", 0)
            n_pos = len(state.get("positions", {}))
            last_rebal = state.get("last_rebalance",
                                   state.get("last_eod_date", "—"))
            init_cap = state.get("initial_capital", 100_000)
            total_ret = (nav / init_cap - 1) * 100 if init_cap else 0.0

            p1, p2, p3, p4, p5 = st.columns(5)
            with p1:
                st.metric("NAV", f"${nav:,.0f}",
                          delta=f"{total_ret:+.2f}% vs start")
            with p2:
                st.metric("Cash", f"${cash:,.0f}")
            with p3:
                st.metric("Open positions", n_pos)
            with p4:
                st.metric("Last rebalance", str(last_rebal))
            with p5:
                st.metric("Total trades", len(trades) if not trades.empty else 0)

            # Paper portfolio chart with SPY overlay
            spy_curve = eq_df["spy"] * init_cap if "spy" in eq_df.columns and not eq_df.empty else None
            entry_value = None
            if not hist.empty and "portfolio_value" in hist.columns:
                entry_value = float(hist["portfolio_value"].iloc[0])
            st.plotly_chart(
                chart_paper_portfolio(
                    history_df=hist,
                    trades_df=trades,
                    spy_curve=spy_curve,
                    entry_value=entry_value,
                ),
                theme=None, use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True},
            )

            if state.get("positions"):
                st.markdown('<div class="section-head">Open positions</div>',
                            unsafe_allow_html=True)
                rows = []
                for ticker, pos in state["positions"].items():
                    if isinstance(pos, dict):
                        qty = pos.get("shares", pos.get("qty", 0))
                        last_close = pos.get("last_close", pos.get("price", 0))
                        mval = pos.get("market_value",
                                       qty * last_close if last_close else 0)
                        cost = pos.get("cost_basis", 0)
                        pnl = mval - cost if cost else 0
                        pnl_pct = (pnl / cost * 100) if cost else 0
                        rows.append({
                            "Ticker": ticker,
                            "Shares": f"{qty:,.2f}",
                            "Entry": f"${pos.get('entry_price', 0):,.2f}",
                            "Last": f"${last_close:,.2f}",
                            "Market Value": f"${mval:,.0f}",
                            "Weight": f"{mval / max(nav, 1) * 100:.1f}%",
                            "P&L": f"${pnl:+,.0f} ({pnl_pct:+.1f}%)",
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True)

            if not trades.empty:
                st.markdown('<div class="section-head">Recent trades</div>',
                            unsafe_allow_html=True)
                st.dataframe(trades.tail(30), use_container_width=True,
                             hide_index=True)

    # ── Backtest tab ─────────────────────────────────────────────────────────
    with tab_bt:
        if eq_df.empty:
            st.info("No backtest results.")
        else:
            st.markdown('<div class="section-head">Gross vs Net vs Benchmark</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                chart_equity(eq_df),
                theme=None, use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True},
            )
            st.plotly_chart(
                chart_drawdown(eq_df),
                theme=None, use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True},
            )
            st.markdown('<div class="section-head">All curves — summary</div>',
                        unsafe_allow_html=True)
            st.dataframe(metrics_table(eq_df), use_container_width=True,
                         hide_index=True)

    # ── Validation tab ───────────────────────────────────────────────────────
    with tab_val:
        if wf.empty:
            st.info("No walk-forward results.")
        else:
            st.markdown('<div class="section-head">Walk-forward OOS</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                chart_walk_forward(wf),
                theme=None, use_container_width=True,
                config={"displayModeBar": False},
            )
            # Display walk-forward dataframe
            display_cols = [c for c in ["window", "test_start", "test_end",
                                         "is_sharpe", "oos_sharpe", "oos_return",
                                         "oos_max_drawdown"] if c in wf.columns]
            if display_cols:
                fmt = wf[display_cols].copy()
                for c in fmt.columns:
                    if fmt[c].dtype.kind in "fc":
                        fmt[c] = fmt[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
                st.dataframe(fmt, use_container_width=True, hide_index=True)

    # ── Risk tab ─────────────────────────────────────────────────────────────
    with tab_risk:
        risk_mc, risk_macro = st.tabs(["  🎲  Monte Carlo", "  🌍  Macro Overlay"])

        with risk_mc:
            if net_returns.empty:
                st.info("No net returns found.")
            else:
                buf = io.BytesIO()
                net_returns.to_frame().to_parquet(buf)
                buf.seek(0)

                eq_paths = run_monte_carlo(buf.getvalue(), n_paths=1000)
                actual = (1 + net_returns.values).cumprod()

                st.plotly_chart(
                    chart_monte_carlo(eq_paths, actual),
                    theme=None, use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": True},
                )
                st.plotly_chart(
                    chart_mc_histogram(eq_paths),
                    theme=None, use_container_width=True,
                    config={"displayModeBar": False},
                )

                finals = eq_paths[:, -1]
                p5, p25, p50, p75, p95 = (np.percentile(finals, q)
                                           for q in (5, 25, 50, 75, 95))
                st.markdown('<div class="section-head">Monte Carlo terminal statistics</div>',
                            unsafe_allow_html=True)
                mc = st.columns(7)
                with mc[0]: st.metric("5th pct", f"{(p5-1)*100:.0f}%")
                with mc[1]: st.metric("25th pct", f"{(p25-1)*100:.0f}%")
                with mc[2]: st.metric("Median", f"{(p50-1)*100:.0f}%")
                with mc[3]: st.metric("75th pct", f"{(p75-1)*100:.0f}%")
                with mc[4]: st.metric("95th pct", f"{(p95-1)*100:.0f}%")
                with mc[5]: st.metric("Prob. positive", f"{(finals>1).mean()*100:.0f}%")
                with mc[6]: st.metric("Prob. 2×", f"{(finals>2).mean()*100:.0f}%")

        with risk_macro:
            if market.empty and macro.empty:
                st.info("No macro features found.")
            else:
                merged_market = market.copy()
                if not macro.empty:
                    for col in macro.columns:
                        if col not in merged_market.columns:
                            merged_market[col] = macro[col]
                st.plotly_chart(
                    chart_macro_overlay(eq_df, merged_market),
                    theme=None, use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": True},
                )

                if "vix" in market.columns:
                    st.markdown('<div class="section-head">Macro regime statistics (full history)</div>',
                                unsafe_allow_html=True)
                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.metric("VIX calm days (<20)",
                                  f"{(market['vix'] < 20).mean() * 100:.0f}%")
                    with r2:
                        st.metric("VIX stress days (>30)",
                                  f"{(market['vix'] > 30).mean() * 100:.0f}%")
                    if "hy_oas" in market.columns:
                        with r3:
                            st.metric("Median HY OAS",
                                      f"{market['hy_oas'].median():.2f}%")
                        with r4:
                            st.metric("HY OAS 95% pct",
                                      f"{market['hy_oas'].quantile(0.95):.2f}%")


if __name__ == "__main__":
    main()
