"""
dashboard.py
------------
Streamlit web dashboard for the algorithmic trading system.

To run:
  streamlit run dashboard.py

Architecture overview
─────────────────────
The dashboard is organised into tabs, each backed by cached data-loading
functions.  Expensive operations (reading parquets, fetching yfinance data)
are wrapped in @st.cache_data with a TTL to avoid re-running on every
user interaction.

Sub-package structure (ui/)
────────────────────────────
  ui/styles.py       — CSS constant, _layout() helper, PALETTE, LABELS
  ui/data_loaders.py — @st.cache_data loaders + run_monte_carlo()
  ui/charts.py       — metrics(), metrics_table(), all chart_*() builders

Tabs
────
  Overview       — Backtest equity curves, portfolio comparison table,
                   walk-forward OOS results, IS vs OOS Sharpe comparison
  Portfolio      — ATR+PCA+macro equity curve with B&H benchmark, drawdown,
                   rolling Sharpe, monthly returns heatmap, trade analysis
  Live Signals   — Current MA-crossover signal state for all 21 tickers,
                   refreshed every few minutes from yfinance
  Paper Trading  — Live intraday candlestick chart of the paper portfolio,
                   open positions table, daily/trade P&L breakdown,
                   "Run EOD Update" button for manual end-of-day runs
  vs S&P 500     — Today's intraday portfolio vs SPY benchmark (normalised),
                   historical daily performance comparison

Paper Trading chart details
────────────────────────────
  Historical daily closes → go.Scatter (blue line)
  Today's 5-min bars     → go.Candlestick (green/red candles)
  After close extension  → go.Scatter (blue dotted flat line to now)
  SPY benchmark          → go.Scatter (purple dotted, normalised to same start)
  Now vertical line      → add_shape with orange dotted line
  uirevision="paper_portfolio" — Plotly preserves user zoom/pan across
                                 5-second fragment refreshes.

Key design decisions
─────────────────────
  - Chart x-range right edge = max(now, 16:05) so the chart always shows
    at least through market close, and extends to current time after hours.
  - The flat line after the last candle is a separate Scatter trace (not
    an extra OHLC bar) so it looks clean rather than showing a doji candle.
  - spy_pct_from_prev uses period="5d" daily data filtered to dates before
    today to get the confirmed previous close without the incomplete
    in-progress today row corrupting the calculation.
  - Delta string for portfolio value uses "+" prefix before the number so
    Streamlit can detect the sign for green/red colouring.

Data files consumed (from data/results/ and data/paper_trading/)
────────────────────────────────────────────────────────────────
  portfolio_comparison.parquet   — equity curves for all sizing methods
  walk_forward_regime.parquet    — OOS walk-forward results
  walk_forward_atr_pca.parquet   — OOS results for ATR+PCA+macro
  oos_selection.parquet          — IS vs OOS Sharpe selection table
  portfolio_equity_curve.parquet — best OOS method equity curve
  data/paper_trading/state.json  — current portfolio state
  data/paper_trading/trades.csv  — trade log
  data/paper_trading/history.csv — daily portfolio value snapshots
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from live_signals import get_live_signals
from paper_trader import (
    load_state, load_trades, load_history, catchup,
    get_intraday_curve, end_of_day_update, INITIAL_CAPITAL as PT_INITIAL_CAPITAL,
)
from scheduler import check_kill_switch, ORDERS_FILE
from ui.styles import CSS, _layout, PALETTE, LABELS
from ui.charts import (
    metrics, metrics_table,
    chart_equity, chart_drawdown, chart_monte_carlo, chart_mc_histogram,
    chart_walk_forward, chart_asset_sharpe, chart_macro_overlay,
    chart_monthly_heatmap, chart_ma_spread, chart_paper_portfolio,
)
from ui.data_loaders import (
    load_portfolio_curves, load_ticker_curves, load_walk_forward,
    load_walk_forward_atr, load_oos_selection, load_macro, run_monte_carlo,
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Greatrex Quant Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Fine-grained CSS tweaks (dark grey tone-on-tone) ──────────────────────────
st.markdown(CSS, unsafe_allow_html=True)


# ── Startup catch-up ──────────────────────────────────────────────────────────

@st.cache_resource(ttl=3600, show_spinner=False)
def _startup_catchup() -> int:
    """Replay missed trading days on first dashboard load (once per hour)."""
    return catchup()


# ── Main layout ───────────────────────────────────────────────────────────────
def main():
    """
    Top-level Streamlit entry point — builds the full dashboard layout.

    Called once per page load (or full rerun).  Loads all cached data,
    renders the header and top-level metrics, then delegates each tab's
    content to dedicated chart/fragment functions.

    Tab structure:
      Tab 1 — Equity Curves, drawdown, monthly heatmap, summary table
      Tab 2 — Monte Carlo bootstrap fan chart + histogram
      Tab 3 — Walk-forward OOS validation + per-asset Sharpe comparison
      Tab 4 — Macro overlay (VIX + yield curve) with portfolio curves
      Tab 5 — Paper Trading & Live Signals (auto-refreshing fragment)
      Tab 6 — vs S&P 500 (auto-refreshing fragment)
    """
    # Catch up missed trading days before rendering any tabs.
    # st.cache_resource(ttl=3600) ensures this runs at most once per hour,
    # not on every Streamlit rerun.  The idempotency guard in
    # _end_of_day_update_for_date prevents conflicts with the scheduler.
    with st.spinner("Catching up missed trading days..."):
        n_caught_up = _startup_catchup()
    if n_caught_up and "_catchup_done" not in st.session_state:
        st.cache_data.clear()
        st.session_state["_catchup_done"] = True

    # Load all data first so n_assets is available before header renders
    df_port       = load_portfolio_curves()
    ticker_curves = load_ticker_curves()
    wf            = load_walk_forward()
    wf_atr        = load_walk_forward_atr()
    oos_sel       = load_oos_selection()
    macro         = load_macro()
    n_assets      = len(ticker_curves) if ticker_curves else 20

    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("## 📈 Strategy Performance Dashboard")
        st.markdown(
            f"<span style='color:#666;font-size:.82rem'>"
            f"Multi-asset systematic strategy · 2015–2025 · {n_assets} assets "
            f"(incl. survivorship-bias anchors GE/INTC/WBA/VZ) · "
            f"0.1% round-trip transaction costs · MA50/200 golden-cross signals"
            f"</span>",
            unsafe_allow_html=True,
        )

    if df_port.empty:
        st.error("Run `python run.py portfolio` first to generate portfolio data.")
        return

    # ── Top metrics ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Equal-weight strategy vs buy & hold</div>',
                unsafe_allow_html=True)

    eq_ret  = df_port["equal_weight"].pct_change().dropna() if "equal_weight" in df_port.columns else pd.Series(dtype=float)
    bnh_ret = df_port["buy_hold"].pct_change().dropna()     if "buy_hold"     in df_port.columns else pd.Series(dtype=float)
    m_eq    = metrics(eq_ret)
    m_bnh   = metrics(bnh_ret)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def delta_str(val, ref, pct=True):
        """
        Format the strategy-vs-benchmark delta for a Streamlit st.metric delta arg.

        Args:
            val:  Strategy metric value.
            ref:  Benchmark (Buy & Hold) metric value.
            pct:  If True, format as percentage string with sign; otherwise as
                  a plain signed float with two decimal places.

        Returns:
            String with explicit sign (e.g. "+2.3%" or "-0.15") for Streamlit
            to detect positive/negative and apply green/red colouring.
        """
        d = val - ref
        s = f"{d*100:+.1f}%" if pct else f"{d:+.2f}"
        return s

    # Ann. Return: higher is better → delta_color="normal" (default): green = strategy > B&H
    with c1: st.metric("Ann. Return",  f"{m_eq['ann_r']*100:.1f}%",
                        delta=delta_str(m_eq['ann_r'], m_bnh['ann_r']),
                        help="Compound Annual Growth Rate (CAGR) — the yearly return if capital "
                             "was invested for the full backtest period. Arrow shows vs buy & hold: "
                             "green ↑ = strategy beats B&H, red ↓ = B&H won.")
    # Sharpe: higher is better → delta_color="normal": green = strategy Sharpe > B&H Sharpe
    with c2: st.metric("Sharpe Ratio", f"{m_eq['sharpe']:.2f}",
                        delta=delta_str(m_eq['sharpe'], m_bnh['sharpe'], pct=False),
                        help="Return ÷ volatility — how much return you earned per unit of risk. "
                             ">1.0 is good, >2.0 is exceptional, <0 means you lost money. "
                             "Arrow shows vs buy & hold: green ↑ = better risk-adjusted return than B&H.")
    # Volatility: lower is better → delta_color="inverse": green = strategy vol < B&H vol
    with c3: st.metric("Volatility",   f"{m_eq['vol']*100:.1f}%",
                        delta=delta_str(m_eq['vol'], m_bnh['vol']), delta_color="inverse",
                        help="Annualised standard deviation of daily returns — how wildly returns "
                             "bounce day-to-day. 15% means a typical daily swing of ~1%. "
                             "Arrow shows vs buy & hold: green ↓ = LESS volatile than B&H (good), "
                             "red ↑ = MORE volatile (worse risk profile).")
    # Max Drawdown: both values are negative fractions; a less-negative delta means
    # the strategy had a shallower drawdown → delta_color="normal": green = strategy DD > B&H DD
    # (e.g., strategy -10% vs B&H -20%: delta = +10% → green, which correctly means less loss)
    with c4: st.metric("Max Drawdown", f"{m_eq['max_dd']*100:.1f}%",
                        delta=delta_str(m_eq['max_dd'], m_bnh['max_dd']), delta_color="normal",
                        help="Largest peak-to-trough loss in portfolio history — e.g. −20% means "
                             "the portfolio fell 20% from its highest point before recovering. "
                             "Arrow shows vs buy & hold: green ↑ = SHALLOWER drawdown than B&H (good), "
                             "red ↓ = deeper loss (worse downside protection).")
    with c5: st.metric("Calmar Ratio", f"{m_eq['calmar']:.2f}",
                        help="Annualised return divided by the absolute maximum drawdown. "
                             "Measures how much return you earned relative to the worst loss. "
                             ">1.0 is considered good; institutional target is often >0.5.")
    with c6: st.metric("Win Rate",     f"{m_eq['win_rate']*100:.0f}%",
                        help="Percentage of active trading days where the strategy had a "
                             "positive return. Even a 50% win rate can be very profitable "
                             "if winning days are larger than losing days (see Profit Factor).")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "  📊  Equity Curves",
        "  🎲  Monte Carlo",
        "  🔁  Walk-Forward",
        "  🌍  Macro Overlay",
        "  📡  Live Signals",
        "  📈  vs S&P 500",
    ])

    # ── Tab 1: Equity curves + drawdown + summary table ──────────────────────
    with tab1:
        st.plotly_chart(chart_equity(df_port), theme=None, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})
        st.plotly_chart(chart_drawdown(df_port), theme=None, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

        st.markdown('<div class="section-head">All portfolio methods — summary</div>',
                    unsafe_allow_html=True)
        tbl = metrics_table(df_port)
        st.dataframe(
            tbl.style.map(
                lambda v: "color:#50fa7b" if (isinstance(v, str) and v.startswith("+")) else
                          "color:#ff5555" if (isinstance(v, str) and "-" in v and "%" in v and v != "-0.0%") else "",
            ),
            use_container_width=True, hide_index=True,
        )

        # Monthly returns heatmap
        st.markdown("")
        st.markdown('<div class="section-head">Monthly returns — equal weight strategy</div>',
                    unsafe_allow_html=True)
        if not eq_ret.empty:
            st.plotly_chart(chart_monthly_heatmap(eq_ret), theme=None, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

    # ── Tab 2: Monte Carlo ───────────────────────────────────────────────────
    with tab2:
        st.markdown(
            "<span style='color:#888;font-size:.82rem'>"
            "Block-bootstrap resampling of historical daily returns — preserves the "
            "autocorrelation and fat-tail structure of the actual return distribution. "
            "Each path is an equally-plausible alternative history using the same signal edge."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        col_sl1, col_sl2, _ = st.columns([1, 1, 2])
        with col_sl1:
            n_paths = st.select_slider("Paths", [200, 500, 1000, 2000], value=1000)
        with col_sl2:
            block_sz = st.select_slider("Block size (days)", [5, 10, 21, 42, 63], value=21)

        if eq_ret.empty:
            st.warning("No equal-weight return data available.")
        else:
            with st.spinner("Bootstrapping…"):
                import io
                buf = io.BytesIO()
                eq_ret.to_frame("ret").to_parquet(buf)
                eq_curves = run_monte_carlo(buf.getvalue(), n_paths, block_sz)

            actual_norm = (1 + eq_ret).cumprod().values

            st.plotly_chart(chart_monte_carlo(eq_curves, actual_norm),
                            theme=None, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})
            st.plotly_chart(chart_mc_histogram(eq_curves),
                            theme=None, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

            # MC summary stats
            finals = eq_curves[:, -1]
            p5, p25, p50, p75, p95 = np.percentile(finals, [5, 25, 50, 75, 95])
            st.markdown('<div class="section-head">Monte Carlo terminal statistics</div>',
                        unsafe_allow_html=True)
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1: st.metric("5th pct",  f"{(p5-1)*100:.0f}%",
                                 help="Only 5% of simulated paths ended below this total return. "
                                      "Rough stress-test floor — the near-worst-case outcome "
                                      "based on the historical return distribution.")
            with mc2: st.metric("25th pct", f"{(p25-1)*100:.0f}%",
                                 help="25% of paths ended below this return. The pessimistic "
                                      "quartile — represents a plausible unfavourable scenario.")
            with mc3: st.metric("Median",   f"{(p50-1)*100:.0f}%",
                                 help="50th percentile total return across all simulated paths. "
                                      "The expected central outcome under the historical edge.")
            with mc4: st.metric("75th pct", f"{(p75-1)*100:.0f}%",
                                 help="75% of paths ended below this return. The optimistic "
                                      "quartile — a favourable but plausible outcome.")
            with mc5: st.metric("95th pct", f"{(p95-1)*100:.0f}%",
                                 help="Only 5% of paths exceeded this return. Near the "
                                      "best-case scenario — do not plan around this number.")

            mc6, mc7, _ = st.columns([1, 1, 2])
            with mc6: st.metric("Prob. positive return", f"{(finals>1).mean()*100:.0f}%",
                                 help="Fraction of simulated paths that ended with a positive "
                                      "total return. High % = strategy edge is robust to "
                                      "different orderings of the historical returns.")
            with mc7: st.metric("Prob. 2× capital",      f"{(finals>2).mean()*100:.0f}%",
                                 help="Fraction of paths that doubled the starting capital. "
                                      "A measure of upside potential under block-bootstrap "
                                      "resampling of the observed return stream.")

    # ── Tab 3: Walk-forward + per-asset ──────────────────────────────────────
    with tab3:
        st.markdown(
            "<span style='color:#888;font-size:.82rem'>"
            "<b>Walk-forward validation</b>: train on 3 years, test on the next 1 year (rolling). "
            "Each test period is genuinely unseen data. "
            "Mean OOS Sharpe close to in-sample = low overfitting. "
            "A large IS→OOS gap means the method overfit the training period."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        left, right = st.columns([1, 2])
        with left:
            if wf.empty:
                st.warning("Walk-forward data not found.")
            else:
                st.plotly_chart(chart_walk_forward(wf), theme=None, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})
                mean_s = wf["sharpe"].mean()
                color  = "#50fa7b" if mean_s > 0 else "#ff5555"
                st.markdown(f"""
<div style="background:#252525;border-radius:10px;padding:14px 16px;font-size:.82rem;color:#c0c0c0;line-height:1.8">
  <b>Equal-weight signal OOS</b><br>
  <b>Mean OOS Sharpe</b> <span style="color:{color};font-weight:600">{mean_s:.3f}</span><br>
  <b>Std  OOS Sharpe</b>  {wf['sharpe'].std():.3f}<br>
  <b>Worst period</b>  {wf.loc[wf['sharpe'].idxmin(),'period']}
    (<span style="color:{PALETTE['neg']}">{wf['sharpe'].min():.2f}</span>)<br>
  <b>Best period</b>  {wf.loc[wf['sharpe'].idxmax(),'period']}
    (<span style="color:{PALETTE['pos']}">{wf['sharpe'].max():.2f}</span>)
</div>""", unsafe_allow_html=True)

            if not wf_atr.empty:
                st.markdown("")
                mean_s_atr = wf_atr["sharpe"].mean()
                color_atr  = "#50fa7b" if mean_s_atr > 0 else "#ff5555"
                st.markdown(f"""
<div style="background:#252525;border-radius:10px;padding:14px 16px;font-size:.82rem;color:#c0c0c0;line-height:1.8">
  <b>ATR+PCA+Macro sizing OOS</b><br>
  <b>Mean OOS Sharpe</b> <span style="color:{color_atr};font-weight:600">{mean_s_atr:.3f}</span><br>
  <b>Std  OOS Sharpe</b>  {wf_atr['sharpe'].std():.3f}
</div>""", unsafe_allow_html=True)

        with right:
            if ticker_curves:
                st.plotly_chart(chart_asset_sharpe(ticker_curves),
                                theme=None, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

        # OOS selection comparison table
        if not oos_sel.empty:
            st.markdown("")
            st.markdown(
                '<div class="section-head">Portfolio method selection — in-sample vs out-of-sample Sharpe</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "<span style='color:#888;font-size:.82rem'>"
                "<b>IS Sharpe</b>: full in-sample (2015–2025). "
                "<b>OOS Sharpe</b>: mean Sharpe across walk-forward test windows. "
                "The method with the highest OOS Sharpe is selected — not the highest IS Sharpe. "
                "A large IS→OOS drop signals overfitting of that method."
                "</span>",
                unsafe_allow_html=True,
            )
            display_oos = oos_sel.rename(columns={
                "method"    : "Method",
                "is_sharpe" : "IS Sharpe (full period)",
                "oos_sharpe": "OOS Sharpe (walk-fwd mean)",
            })
            # Highlight the selected (best OOS) row
            best_oos_method = oos_sel.loc[oos_sel["oos_sharpe"].idxmax(), "method"]
            def _highlight_best(row):
                return ["background-color:#1a3a1a" if row["Method"] == best_oos_method
                        else "" for _ in row]
            st.dataframe(
                display_oos.style.apply(_highlight_best, axis=1).format({
                    "IS Sharpe (full period)"   : "{:.3f}",
                    "OOS Sharpe (walk-fwd mean)": "{:.3f}",
                }),
                use_container_width=True, hide_index=True,
            )
            st.markdown(
                f"<span style='color:#50fa7b;font-size:.82rem'>"
                f"✓ Selected method: <b>{best_oos_method}</b></span>",
                unsafe_allow_html=True,
            )

    # ── Tab 4: Macro overlay ─────────────────────────────────────────────────
    with tab4:
        if macro.empty:
            st.warning("Macro data not found. Run `python macro_features.py`.")
        else:
            st.markdown(
                "<span style='color:#888;font-size:.82rem'>"
                "VIX gate (z-score > 2.5) blocks all signals on extreme fear days. "
                "Inverted yield curve reduces position sizes via macro_score multiplier."
                "</span>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(chart_macro_overlay(df_port, macro),
                            theme=None, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

            # Regime statistics
            st.markdown('<div class="section-head">Macro regime statistics (full history)</div>',
                        unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            with r1: st.metric("VIX calm days (<20)",  f"{(macro['vix']<20).mean()*100:.0f}%",
                                 help="Days where VIX < 20 — low fear environment. Signals "
                                      "are not gated; normal position sizing applies.")
            with r2: st.metric("VIX fear days (>30)",  f"{(macro['vix']>30).mean()*100:.0f}%",
                                 help="Days where VIX > 30 — elevated fear. Macro score "
                                      "multiplier reduces position sizes. VIX z-score > 2.5 "
                                      "triggers a hard gate (all signals blocked).")
            with r3: st.metric("Curve inverted (<0)",  f"{(macro['yield_curve']<0).mean()*100:.0f}%",
                                 help="Days where 10Y−2Y Treasury spread < 0 (inverted). "
                                      "Historically precedes recessions. Macro multiplier "
                                      "reduces sizing when the curve is inverted.")
            with r4: st.metric("Curve steep (>1)",     f"{(macro['yield_curve']>1).mean()*100:.0f}%",
                                 help="Days where 10Y−2Y spread > 1% (steep). Historically "
                                      "associated with early-cycle expansion. Macro multiplier "
                                      "slightly increases sizing in steep environments.")

    # ── Tab 5: Paper Trading & Live Signals ──────────────────────────────────
    with tab5:

        @st.fragment(run_every=5)
        def _live_section():
            """
            Streamlit fragment: the full "Paper Trading & Live Signals" tab content.

            Auto-refreshes every 5 seconds via run_every=5.  The fragment boundary
            means only this section re-runs on each tick — the rest of the dashboard
            (other tabs) is not re-evaluated, keeping the page responsive.

            Renders two main sections:

              1. Paper Portfolio
                 - Header row with market open/closed indicator and manual buttons
                   (Refresh Now, Run EOD Update — EOD button disabled during market hours)
                 - Portfolio Overview metrics: live value, total return, realized P&L,
                   cash, open position count
                 - Today's P&L metrics: unrealized and realized breakdown
                 - TradingView-style equity chart (chart_paper_portfolio)
                 - Open positions table (current price, unrealized P&L per ticker)
                 - Recent trades log (last 20 trades)
                 - Kill switch status and tomorrow's order sheet

              2. Live Signals
                 - MA crossover signal state for all universe tickers (cached 5 min)
                 - Summary metrics: long count, flat count, gross exposure, avg RSI
                 - MA spread bar chart
                 - Full universe signal table with colour coding

            uirevision="paper_portfolio" on the chart ensures Plotly preserves the
            user's zoom/pan state across every 5-second data refresh.
            """
            now = pd.Timestamp.now(tz='America/New_York').replace(tzinfo=None)

            # Header row: clock + manual refresh
            hdr_left, hdr_right = st.columns([3, 1])
            with hdr_left:
                market_open = 9 <= now.hour < 16
                clock_color = "#50fa7b" if market_open else "#888"
                st.markdown(
                    f"<span style='color:{clock_color};font-size:.85rem;font-weight:600'>"
                    f"{'◉ MARKET OPEN' if market_open else '○ MARKET CLOSED'}"
                    f"</span>"
                    f"<span style='color:#555;font-size:.78rem'>  ·  "
                    f"Updated: {now.strftime('%Y-%m-%d  %H:%M:%S')}  ·  auto-refresh 5s  ·  data ≈15 min delay (Yahoo Finance)</span>",
                    unsafe_allow_html=True,
                )
            with hdr_right:
                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button("Refresh now", type="secondary"):
                        st.session_state.pop("paper_chart_x_range", None)
                        st.cache_data.clear()
                        st.rerun()
                with btn2:
                    _market_closed = now.hour >= 16 or now.hour < 9
                    if st.button("Run EOD Update", type="primary",
                                 disabled=not _market_closed,
                                 help="Run after 4 PM ET to record today's close, "
                                      "evaluate tomorrow's signals, and update history. "
                                      "Disabled during market hours."):
                        with st.spinner("Running EOD update…"):
                            try:
                                end_of_day_update()
                                st.cache_data.clear()
                                st.success("EOD update complete — history and signals updated.")
                                st.rerun()
                            except Exception as _e:
                                st.error(f"EOD update failed: {_e}")

            # ── Paper Portfolio ───────────────────────────────────────────────
            pt_state   = load_state()
            pt_history = load_history()
            pt_trades  = load_trades()

            if not pt_state:
                st.info(
                    "Paper trading not yet initialised.  Run this command once to start:\n\n"
                    "```\npython paper_trader.py init\n```\n\n"
                    "Then run after each market close:\n\n"
                    "```\npython paper_trader.py run\n```"
                )
            else:
                cash  = pt_state["cash"]
                n_pos = len(pt_state.get("positions", {}))

                # TradingView-style equity chart (intraday live)
                intraday_df, live_prices, spy_curve, _spy_pct = get_intraday_curve()

                prev_pv = pt_state.get("portfolio_value", PT_INITIAL_CAPITAL)

                # ── Open position market values (from fresh per-ticker prices) ─
                tot_invested = sum(pos["cost_basis"] for pos in pt_state.get("positions", {}).values())
                tot_cur_val  = sum(
                    pos["shares"] * (live_prices.get(t) or pos["entry_price"])
                    for t, pos in pt_state.get("positions", {}).items()
                )
                tot_unreal   = tot_cur_val - tot_invested
                tot_chg_pct  = (tot_cur_val / tot_invested - 1) * 100 if tot_invested else 0.0

                # pv from live per-ticker prices (consistent with positions table;
                # avoids the intraday OHLC sum having stale zeros at 5-min boundaries)
                pv        = (tot_cur_val + cash) if live_prices else prev_pv
                daily_ret = (pv / prev_pv - 1) * 100 if prev_pv else 0.0
                total_ret = (pv / PT_INITIAL_CAPITAL - 1) * 100
                daily_pnl_usd = pv - prev_pv

                # ── Realized P&L from closed trades ──────────────────────────
                today_str_filter = now.strftime('%Y-%m-%d')
                if not pt_trades.empty and "pnl" in pt_trades.columns:
                    sells = pt_trades[
                        (pt_trades["action"] == "SELL") &
                        pt_trades["pnl"].notna() &
                        (pt_trades["pnl"].astype(str).str.strip() != "")
                    ]
                    all_realized   = pd.to_numeric(sells["pnl"], errors="coerce").fillna(0).sum()
                    today_sells    = sells[sells["date"].dt.strftime('%Y-%m-%d') == today_str_filter]
                    today_realized = pd.to_numeric(today_sells["pnl"], errors="coerce").fillna(0).sum()
                else:
                    all_realized   = 0.0
                    today_realized = 0.0

                # today_unrealized dollar = change vs prev close position value
                # today_unrealized_pct uses cost_basis denominator (tot_invested)
                # so it matches the positions table unrealized % exactly.
                prev_positions_val   = prev_pv - cash
                today_unrealized     = tot_cur_val - prev_positions_val
                today_unrealized_pct = (tot_cur_val / tot_invested - 1) * 100 if tot_invested else 0.0

                # ── GROUP 1: Portfolio Overview ───────────────────────────────
                st.markdown('<div class="section-head">Portfolio Overview</div>', unsafe_allow_html=True)
                pm1, pm2, pm3, pm4, pm5 = st.columns(5)
                with pm1:
                    st.metric(
                        "Portfolio Value", f"${pv:,.0f}",
                        delta=f"{daily_ret:+.2f}%",
                        delta_color="normal",
                        help="Live value: cash + all open positions marked to last 5-min bar. "
                             "Arrow shows today's dollar change vs yesterday's close — green = up, red = down.",
                    )
                with pm2:
                    st.metric(
                        "Total Return", f"{total_ret:+.2f}%",
                        help="Overall % return vs $100k starting capital since inception. "
                             "Includes both unrealized gains (open positions) and realized gains (closed trades).",
                    )
                with pm3:
                    st.metric(
                        "Realized P&L", f"${all_realized:+,.0f}",
                        delta=f"{today_realized:+,.0f} today" if today_realized != 0 else None,
                        delta_color="normal",
                        help="Total locked-in profit/loss from all closed trades since inception. "
                             "Only increases/decreases when a position is sold. Arrow shows today's closed trades.",
                    )
                with pm4:
                    st.metric(
                        "Cash", f"${cash:,.0f}",
                        help="Uninvested cash. The gap vs $100k starting capital reflects commissions (~0.05% per trade).",
                    )
                with pm5:
                    st.metric("Open Positions", str(n_pos))

                # ── GROUP 2: Daily P&L Breakdown ──────────────────────────────
                st.markdown('<div class="section-head">Today\'s P&L</div>', unsafe_allow_html=True)
                d1, d2 = st.columns(2)
                with d1:
                    st.metric(
                        "Unrealized (today)", f"${today_unrealized:+,.0f}",
                        delta=f"{today_unrealized_pct:+.2f}%",
                        delta_color="normal",
                        help="Change in open position values today vs yesterday's close. "
                             "Not locked in — fluctuates until positions are sold. Arrow = % version.",
                    )
                with d2:
                    st.metric(
                        "Realized (today)", f"${today_realized:+,.0f}",
                        help="P&L from positions actually closed today. Zero if no trades were executed today.",
                    )

                # ── Equity chart ──────────────────────────────────────────────
                # ── x-range management ───────────────────────────────────────
                # The x_range is stored in session_state and only updated when the
                # date rolls over OR the right edge needs to advance.  Passing a
                # constant range to chart_paper_portfolio() (combined with
                # uirevision="paper_portfolio") means Plotly.js treats each 5-second
                # refresh as a data update rather than a full re-render, so user
                # zoom/pan is preserved across refreshes.
                _today = now.strftime('%Y-%m-%d')
                # Right edge tracks current time (or 16:05 if before close),
                # so the flat post-close line is always visible.
                _right = max(now, pd.Timestamp(_today + " 16:05:00"))
                _right_str = _right.strftime('%Y-%m-%dT%H:%M:00')
                _cached = st.session_state.get("paper_chart_x_range", ["", ""])
                if not _cached[0].startswith(_today) or _cached[1] < _right_str:
                    st.session_state["paper_chart_x_range"] = [
                        f"{_today}T09:25:00",
                        _right_str,
                    ]

                st.plotly_chart(
                    chart_paper_portfolio(pt_history, intraday_df, pt_trades,
                                          now=now, entry_value=prev_pv,
                                          x_range=st.session_state["paper_chart_x_range"],
                                          spy_curve=spy_curve),
                    theme=None, use_container_width=True,
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["resetScale2d", "autoScale2d"],
                    },
                    key="paper_portfolio_chart",
                )

                # ── GROUP 3: Open Positions ───────────────────────────────────
                if pt_state.get("positions"):
                    st.markdown('<div class="section-head">Open Positions</div>',
                                unsafe_allow_html=True)

                    t1, t2, t3 = st.columns(3)
                    with t1:
                        st.metric(
                            "Total Invested", f"${tot_invested:,.0f}",
                            help="Sum of cost basis (dollars deployed) across all open positions.",
                        )
                    with t2:
                        st.metric(
                            "Market Value", f"${tot_cur_val:,.0f}",
                            delta=f"{tot_chg_pct:+.2f}%",
                            delta_color="normal",
                            help="Current market value of all open positions (shares × last 5-min bar close). "
                                 "Arrow shows % gain/loss vs cost basis — green = above cost, red = below.",
                        )
                    with t3:
                        st.metric(
                            "Unrealized P&L", f"${tot_unreal:+,.0f}",
                            delta=f"{tot_chg_pct:+.2f}%",
                            delta_color="normal",
                            help="Market Value minus cost basis — the open gain/loss across all positions. "
                                 "Unrealized until sold. Arrow shows % change: green = up, red = down.",
                        )

                    pos_rows = []
                    for ticker, pos in pt_state["positions"].items():
                        cur = live_prices.get(ticker) or pos["entry_price"]
                        cur_val    = pos["shares"] * cur
                        unreal_pnl = cur_val - pos["cost_basis"]
                        unreal_pct = (cur / pos["entry_price"] - 1) * 100
                        pos_rows.append({
                            "Ticker"     : ticker,
                            "Entry Time" : pos.get("entry_date", "—"),
                            "Invested"   : round(pos["cost_basis"], 2),
                            "Entry $"    : round(pos["entry_price"], 2),
                            "Current $"  : round(cur, 2),
                            "Chg %"      : round(unreal_pct, 2),
                            "Unreal P&L" : round(unreal_pnl, 2),
                        })

                    pos_df = pd.DataFrame(pos_rows)

                    def _color_pnl(v):
                        if isinstance(v, (int, float)):
                            if v > 0: return "color:#50fa7b"
                            if v < 0: return "color:#ff5555"
                        return ""

                    st.dataframe(
                        pos_df.style
                        .map(_color_pnl, subset=["Chg %", "Unreal P&L"])
                        .format({"Invested": "${:,.2f}", "Entry $": "${:.2f}",
                                 "Current $": "${:.2f}", "Chg %": "{:+.2f}%",
                                 "Unreal P&L": "${:+,.2f}"}),
                        use_container_width=True, hide_index=True,
                    )


                # ── Recent trades ─────────────────────────────────────────────
                if not pt_trades.empty:
                    st.markdown('<div class="section-head">Recent trades</div>',
                                unsafe_allow_html=True)
                    recent = pt_trades.sort_values("date", ascending=False).head(20)
                    def _color_action(v):
                        if v == "BUY":  return "color:#50fa7b;font-weight:600"
                        if v == "SELL": return "color:#ff5555;font-weight:600"
                        return ""
                    st.dataframe(
                        recent.style.map(_color_action, subset=["action"]),
                        use_container_width=True, hide_index=True,
                    )

                # ── Kill switch + order sheet ─────────────────────────────────
                kill = check_kill_switch()
                ks_c, _ = st.columns([1, 3])
                with ks_c:
                    if kill:
                        st.error("KILL SWITCH ACTIVE — drawdown >15%. Trading paused.")
                    else:
                        st.success("Kill switch: OK")

                if ORDERS_FILE.exists():
                    import json as _json
                    with open(ORDERS_FILE, encoding="utf-8") as _f:
                        sheet = _json.load(_f)
                    st.markdown(
                        f'<div class="section-head">Tomorrow\'s orders '
                        f'— {sheet.get("generated_at","?")}</div>',
                        unsafe_allow_html=True,
                    )
                    orders_df = pd.DataFrame(sheet.get("orders", []))
                    if not orders_df.empty:
                        def _color_order(v):
                            if v == "BUY":  return "color:#50fa7b;font-weight:600"
                            if v == "SELL": return "color:#ff5555;font-weight:600"
                            if v == "HOLD": return "color:#888"
                            if v in ("FLAT", "SKIP", "NO_DATA"): return "color:#555"
                            return ""
                        st.dataframe(
                            orders_df.style.map(_color_order, subset=["action"]),
                            use_container_width=True, hide_index=True,
                        )
                else:
                    st.caption(
                        "No order sheet yet — run `python scheduler.py` "
                        "or `python paper_trader.py run` after 4:45 PM ET."
                    )

                st.caption(
                    f"Init: {pt_state.get('initialized_date','?')}  ·  "
                    f"Last EOD: {pt_state.get('last_eod_date','?')}  ·  "
                    "Filters: RSI<70  ·  3x ATR stop  ·  5-day hold  ·  15% kill switch"
                )

            # ── Live Signals ──────────────────────────────────────────────────
            st.divider()
            st.markdown("### Universe Signal State")
            st.markdown(
                "<span style='color:#888;font-size:.82rem'>"
                "MA crossover logic mirrors the backtest. Signal flips at daily close. "
                "Cached 5 min."
                "</span>",
                unsafe_allow_html=True,
            )

            @st.cache_data(ttl=300, show_spinner=False)
            def _cached_live():
                """
                Fetch live MA-crossover signals for all universe tickers.

                Cached with TTL=300s (5 minutes) so rapid fragment re-runs
                (every 5 seconds) do not hammer the yfinance API on every tick.

                Returns:
                    Tuple of (live_df, fetch_timestamp) from get_live_signals().
                    live_df contains columns [Ticker, Signal, Price, Day Chg %,
                    MA Spread %, RSI, 20d Ret %, 60d Ret %, Dist High %].
                """
                return get_live_signals()

            with st.spinner("Fetching live prices…"):
                live_df, fetch_ts = _cached_live()

            if live_df.empty:
                st.error("Could not fetch live data. Check your internet connection.")
            else:
                n_long  = (live_df["Signal"] == "LONG").sum()
                n_flat  = (live_df["Signal"] == "FLAT").sum()
                exp_pct = n_long / len(live_df) * 100

                lv1, lv2, lv3, lv4 = st.columns(4)
                with lv1: st.metric("Long signals",   str(n_long))
                with lv2: st.metric("Flat (cash)",    str(n_flat))
                with lv3: st.metric("Gross exposure", f"{exp_pct:.0f}%")
                avg_rsi_long = live_df.loc[live_df["Signal"] == "LONG", "RSI"].mean()
                with lv4: st.metric("Avg RSI (longs)", f"{avg_rsi_long:.1f}" if n_long > 0 else "—")

                st.plotly_chart(chart_ma_spread(live_df), theme=None, use_container_width=True,
                                config={"scrollZoom": True, "displayModeBar": True})

                st.markdown('<div class="section-head">Full universe — current signal state</div>',
                            unsafe_allow_html=True)

                def _style_live(row):
                    return (["background-color:#0d2b0d"] * len(row)
                            if row["Signal"] == "LONG" else [""] * len(row))

                def _color_cell(v):
                    if isinstance(v, str) and v == "LONG":  return "color:#50fa7b;font-weight:600"
                    if isinstance(v, str) and v == "FLAT":  return "color:#666"
                    if isinstance(v, (int, float)):
                        if v > 0: return "color:#50fa7b"
                        if v < 0: return "color:#ff5555"
                    return ""

                st.dataframe(
                    live_df.style
                    .apply(_style_live, axis=1)
                    .map(_color_cell, subset=["Signal", "Day Chg %", "MA Spread %",
                                                   "20d Ret %", "60d Ret %", "Dist High %"])
                    .format({"Price": "${:.2f}", "Day Chg %": "{:+.2f}%",
                             "MA Spread %": "{:+.2f}%", "RSI": "{:.1f}",
                             "20d Ret %": "{:+.1f}%", "60d Ret %": "{:+.1f}%",
                             "Dist High %": "{:+.1f}%"}),
                    use_container_width=True, hide_index=True,
                )

        _live_section()

    # ── Tab 6: vs S&P 500 ────────────────────────────────────────────────────
    with tab6:

        @st.cache_data(ttl=300, show_spinner=False)
        def _spy_history(start_date: str):
            """
            Fetch daily SPY closing prices from portfolio inception to today.

            Cached with TTL=300s to avoid repeated yfinance calls on each
            fragment refresh.  Tries yf.download() first; falls back to
            yf.Ticker.history() if the multi-index download variant fails.

            Args:
                start_date: ISO date string (YYYY-MM-DD) for the first day of
                            the paper portfolio, used as the yfinance start param.

            Returns:
                pd.Series of SPY daily closes indexed by tz-naive datetime.
                Returns empty Series if both download attempts fail.
            """
            import yfinance as yf
            try:
                raw = yf.download("SPY", start=start_date, auto_adjust=True,
                                  progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.droplevel(1)
                raw.index = pd.to_datetime(raw.index).tz_localize(None)
                if "Close" in raw.columns and not raw.empty:
                    return raw["Close"].dropna()
            except Exception:
                pass
            # Fallback: Ticker.history()
            try:
                raw = yf.Ticker("SPY").history(start=start_date, auto_adjust=True)
                raw.index = pd.to_datetime(raw.index).tz_localize(None)
                if "Close" in raw.columns and not raw.empty:
                    return raw["Close"].dropna()
            except Exception:
                pass
            return pd.Series(dtype=float)

        @st.fragment(run_every=30)
        def _vs_sp_section():
            """
            Streamlit fragment: the full "vs S&P 500" tab content.

            Renders two sections:
              1. Live — Today vs S&P 500
                 - Portfolio % today (vs yesterday's close)
                 - SPY % today (from yesterday's official close, matching TradingView)
                 - Alpha (portfolio − SPY)
                 - Intraday chart: both lines normalised to same open value, extended
                   to current time with a flat line after the last 5-min bar.

              2. Historical Record vs S&P 500
                 - Daily portfolio returns vs SPY returns since inception
                 - Cumulative return comparison chart
                 - Alpha histogram

            Refreshes automatically as a Streamlit fragment (auto_refresh=True).
            """
            now_et = pd.Timestamp.now(tz='America/New_York').replace(tzinfo=None)

            pt_state   = load_state()
            pt_history = load_history()

            if not pt_state:
                st.info("Paper trading not initialised. Run `python paper_trader.py init` first.")
                return

            # ── Intraday (live) comparison ────────────────────────────────────
            st.markdown('<div class="section-head">Live — Today vs S&P 500</div>',
                        unsafe_allow_html=True)

            intraday_df, live_prices, spy_intraday, spy_pct_from_prev = get_intraday_curve()

            cash    = pt_state["cash"]
            prev_pv = pt_state.get("portfolio_value", PT_INITIAL_CAPITAL)
            tot_invested = sum(pos["cost_basis"] for pos in pt_state.get("positions", {}).values())
            tot_cur_val = sum(
                pos["shares"] * (live_prices.get(t) or pos["entry_price"])
                for t, pos in pt_state.get("positions", {}).items()
            )
            pv = (tot_cur_val + cash) if live_prices else prev_pv

            # Portfolio today % — uses cost_basis denominator to match positions table.
            port_today_pct = (tot_cur_val / tot_invested - 1) * 100 if tot_invested else 0.0

            # SPY % from yesterday's close (matches TradingView / Yahoo Finance display).
            # spy_pct_from_prev is None when the daily fetch failed — fall back to
            # first-bar-of-day calculation (misses the open gap but is always available).
            if spy_pct_from_prev is not None:
                spy_today_pct = spy_pct_from_prev
            elif not spy_intraday.empty:
                spy_today_pct = (float(spy_intraday.iloc[-1]) / float(spy_intraday.iloc[0]) - 1) * 100
            else:
                spy_today_pct = 0.0

            # Last intraday bar timestamp — shows user how fresh the data is
            _last_bar_str = ""
            if not spy_intraday.empty:
                _last_bar = spy_intraday.index[-1]
                _last_bar_str = pd.Timestamp(_last_bar).strftime("%H:%M") if hasattr(_last_bar, "strftime") else str(_last_bar)[-5:]

            alpha_today = port_today_pct - spy_today_pct
            beating     = alpha_today > 0

            c1, c2, c3, c4 = st.columns(4)
            # Portfolio Today: higher is better → delta_color="normal" (default green for positive)
            with c1:
                st.metric("Portfolio Today", f"{port_today_pct:+.2f}%",
                          help="Portfolio's intraday % change vs yesterday's close.")
            # S&P 500 Today: informational, no delta colouring needed
            with c2:
                _spy_help = (
                    "SPY % change from yesterday's close — same baseline as TradingView. "
                    f"Last bar: {_last_bar_str} ET. "
                    "Data source: Yahoo Finance free tier (≈15 min delay). "
                    "Real-time sources (Bloomberg, Polygon) would reduce this gap."
                )
                st.metric("S&P 500 Today", f"{spy_today_pct:+.2f}%", help=_spy_help)
            with c3:
                # Alpha delta_color is dynamic: "normal" when outperforming (positive alpha
                # → green is correct), "inverse" when underperforming so the word
                # "Underperforming" still shows in red rather than green.
                st.metric("Alpha (today)", f"{alpha_today:+.2f}%",
                          delta="Outperforming" if beating else "Underperforming",
                          delta_color="normal" if beating else "inverse",
                          help="Portfolio return minus S&P return today. Positive = beating the index.")
            with c4:
                st.metric("Portfolio Value", f"${pv:,.0f}",
                          help="Live portfolio value (positions × latest 5-min bar + cash).")

            # Intraday chart — both normalized to same starting value
            if not spy_intraday.empty or not intraday_df.empty:
                fig_intra = go.Figure()

                if not intraday_df.empty and "close" in intraday_df.columns:
                    # Extend portfolio line to current time (flat after last bar)
                    _port_x = list(intraday_df.index)
                    _port_y = list(intraday_df["close"])
                    if now_et > pd.Timestamp(_port_x[-1]):
                        _port_x.append(now_et)
                        _port_y.append(_port_y[-1])
                    fig_intra.add_trace(go.Scatter(
                        x=_port_x, y=_port_y,
                        name="My Portfolio", line=dict(color="#4a9eff", width=2),
                        hovertemplate="<b>Portfolio</b><br>%{x|%H:%M}  $%{y:,.0f}<extra></extra>",
                    ))

                if not spy_intraday.empty:
                    # Extend SPY line to current time (flat after last bar)
                    _spy_x = list(spy_intraday.index)
                    _spy_y = list(spy_intraday.values)
                    if now_et > pd.Timestamp(_spy_x[-1]):
                        _spy_x.append(now_et)
                        _spy_y.append(_spy_y[-1])
                    fig_intra.add_trace(go.Scatter(
                        x=_spy_x, y=_spy_y,
                        name="S&P 500 (SPY)", line=dict(color="#ffb86c", width=2, dash="dot"),
                        hovertemplate="<b>S&P 500</b><br>%{x|%H:%M}  $%{y:,.0f}<extra></extra>",
                    ))

                # Now line
                now_str = now_et.strftime('%Y-%m-%dT%H:%M:%S')
                fig_intra.add_shape(type="line", x0=now_str, x1=now_str,
                                    y0=0, y1=1, yref="paper",
                                    line=dict(color="#888", dash="dot", width=1))

                _vs_today = now_et.strftime('%Y-%m-%d')
                _vs_right = max(now_et, pd.Timestamp(_vs_today + " 16:05:00"))
                fig_intra.update_layout(**_layout(
                    height=300, uirevision="vs_spy_intra",
                    dragmode="pan",
                    title=dict(text="Today — Portfolio vs S&P 500 (both normalized to same open value)",
                               font=dict(size=12)),
                    yaxis=dict(tickprefix="$", tickformat=",.0f", autorange=True),
                    xaxis=dict(type="date", tickformat="%H:%M", rangeslider=dict(visible=False),
                               range=[f"{_vs_today}T09:25:00", _vs_right.strftime('%Y-%m-%dT%H:%M:00')]),
                    margin=dict(l=70),
                ))
                st.plotly_chart(fig_intra, theme=None, use_container_width=True,
                                config={"scrollZoom": True, "displayModeBar": False},
                                key="vs_spy_intra_chart")

            # ── Historical comparison (daily) ─────────────────────────────────
            st.markdown('<div class="section-head">Historical Record vs S&P 500</div>',
                        unsafe_allow_html=True)

            if pt_history.empty or len(pt_history) < 2:
                st.info(
                    "Only one day of history so far — historical comparison will appear "
                    "after the scheduler runs the first EOD update tonight. "
                    "Check back after 4:45 PM ET."
                )
            else:
                start_date = str(pt_history["date"].min().date())
                with st.spinner("Fetching SPY history…"):
                    spy_closes = _spy_history(start_date)

                if spy_closes.empty:
                    st.error("Could not fetch SPY data.")
                else:
                    # Align history and SPY on the same dates
                    hist = pt_history.set_index("date")["portfolio_value"].copy()
                    hist.index = pd.to_datetime(hist.index)
                    hist = hist[~hist.index.duplicated(keep='last')]
                    spy_aligned = spy_closes.reindex(hist.index).ffill()

                    # Normalize both to $100k at portfolio inception
                    port_norm = hist / hist.iloc[0] * PT_INITIAL_CAPITAL
                    spy_norm  = spy_aligned / spy_aligned.iloc[0] * PT_INITIAL_CAPITAL

                    # Daily returns
                    port_ret  = hist.pct_change().dropna() * 100
                    spy_ret   = spy_aligned.pct_change().dropna() * 100

                    # Summary metrics
                    port_total = (hist.iloc[-1] / hist.iloc[0] - 1) * 100
                    spy_total  = (spy_aligned.iloc[-1] / spy_aligned.iloc[0] - 1) * 100
                    total_alpha = port_total - spy_total
                    win_days    = int((port_ret.values > spy_ret.reindex(port_ret.index).values).sum())
                    total_days  = len(port_ret)

                    # ── Beta & correlation (require aligned common dates) ─────
                    # Minimum 3 observations guard avoids divide-by-zero and
                    # misleading stats from just 1–2 data points.
                    # Beta = Cov(portfolio, SPY) / Var(SPY) — OLS slope.
                    # corr is computed but not currently displayed; retained for
                    # potential future use in the caption or metrics row.
                    common = port_ret.index.intersection(spy_ret.index)
                    if len(common) >= 3:
                        beta = float(np.cov(port_ret[common], spy_ret[common])[0, 1] /
                                     np.var(spy_ret[common])) if np.var(spy_ret[common]) > 0 else 1.0
                        corr = float(port_ret[common].corr(spy_ret[common]))
                    else:
                        # Fallback to neutral defaults when there is insufficient history
                        beta, corr = 1.0, 1.0

                    h1, h2, h3, h4, h5 = st.columns(5)
                    # Portfolio Return and S&P 500 Return: informational, no delta colouring
                    with h1: st.metric("Portfolio Return", f"{port_total:+.2f}%",
                                       help="Total return since portfolio inception.")
                    with h2: st.metric("S&P 500 Return",  f"{spy_total:+.2f}%",
                                       help="SPY total return over the same period.")
                    # Total Alpha: delta_color switches dynamically so "Underperforming"
                    # renders in red (inverse) and "Outperforming" renders in green (normal).
                    with h3: st.metric("Total Alpha",     f"{total_alpha:+.2f}%",
                                       delta="Outperforming" if total_alpha > 0 else "Underperforming",
                                       delta_color="normal" if total_alpha > 0 else "inverse",
                                       help="Portfolio return minus SPY return. Positive = beating the index.")
                    with h4: st.metric("Beta",            f"{beta:.2f}",
                                       help="Portfolio sensitivity to S&P moves. 1.0 = moves with the market. "
                                            "<1 = less volatile than S&P, >1 = more volatile.")
                    with h5: st.metric("Days Beating S&P", f"{win_days}/{total_days}",
                                       help="Number of days the portfolio's daily return exceeded SPY's daily return.")

                    # Historical chart
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(
                        x=port_norm.index, y=port_norm.values,
                        name="My Portfolio", line=dict(color="#4a9eff", width=2.2),
                        hovertemplate="<b>Portfolio</b><br>%{x|%b %d, %Y}  $%{y:,.0f}<extra></extra>",
                    ))
                    fig_hist.add_trace(go.Scatter(
                        x=spy_norm.index, y=spy_norm.values,
                        name="S&P 500 (SPY)", line=dict(color="#ffb86c", width=2, dash="dot"),
                        hovertemplate="<b>S&P 500</b><br>%{x|%b %d, %Y}  $%{y:,.0f}<extra></extra>",
                    ))
                    fig_hist.add_hline(y=PT_INITIAL_CAPITAL, line_color="#444", line_dash="dot",
                                       line_width=1, annotation_text="  $100k start",
                                       annotation_font_color="#555")
                    fig_hist.update_layout(**_layout(
                        height=320, uirevision="vs_spy_hist",
                        dragmode="pan",
                        title=dict(text="Portfolio vs S&P 500 — both normalized to $100k at inception",
                                   font=dict(size=12)),
                        yaxis=dict(tickprefix="$", tickformat=",.0f", autorange=True),
                        xaxis=dict(type="date", tickformat="%b %d", rangeslider=dict(visible=False)),
                        margin=dict(l=70),
                    ))
                    st.plotly_chart(fig_hist, theme=None, use_container_width=True,
                                   config={"scrollZoom": True, "displayModeBar": False})

                    # Daily comparison table
                    st.markdown('<div class="section-head">Daily Breakdown</div>',
                                unsafe_allow_html=True)
                    if len(common) > 0:
                        tbl_rows = []
                        for d in reversed(common):
                            pr = float(port_ret[d]) if d in port_ret.index else 0.0
                            sr = float(spy_ret[d]) if d in spy_ret.index else 0.0
                            tbl_rows.append({
                                "Date"         : d.strftime("%Y-%m-%d"),
                                "Portfolio %"  : round(pr, 2),
                                "S&P 500 %"    : round(sr, 2),
                                "Alpha"        : round(pr - sr, 2),
                                "Beating S&P"  : "✓" if pr > sr else "✗",
                            })
                        tbl = pd.DataFrame(tbl_rows)

                        def _color_alpha(v):
                            if isinstance(v, (int, float)):
                                if v > 0: return "color:#50fa7b"
                                if v < 0: return "color:#ff5555"
                            return ""
                        def _color_beat(v):
                            if v == "✓": return "color:#50fa7b;font-weight:600"
                            if v == "✗": return "color:#ff5555"
                            return ""

                        st.dataframe(
                            tbl.style
                            .map(_color_alpha, subset=["Portfolio %", "S&P 500 %", "Alpha"])
                            .map(_color_beat, subset=["Beating S&P"])
                            .format({"Portfolio %": "{:+.2f}%", "S&P 500 %": "{:+.2f}%",
                                     "Alpha": "{:+.2f}%"}),
                            use_container_width=True, hide_index=True,
                        )

                    st.caption(
                        "Beta and correlation are meaningful only with 20+ days of data. "
                        "Alpha = portfolio daily return − SPY daily return. "
                        "Both series normalized to $100k at portfolio inception date."
                    )

        _vs_sp_section()


if __name__ == "__main__":
    main()
