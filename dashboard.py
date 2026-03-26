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
  ^GSPC benchmark        → go.Scatter (purple dotted, normalised to close-to-close baseline)
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
from pipeline.data_pipeline import ASSET_CLASS
from scheduler import check_kill_switch, ORDERS_FILE
from ui.styles import CSS, _layout, PALETTE, LABELS
from ui.charts import (
    metrics, metrics_table,
    chart_equity, chart_drawdown, chart_monte_carlo, chart_mc_histogram,
    chart_walk_forward, chart_asset_sharpe, chart_macro_overlay,
    chart_monthly_heatmap, chart_ma_spread, chart_paper_portfolio,
    chart_beta_rolling, chart_active_return, chart_dead_weight,
)
from ui.data_loaders import (
    load_portfolio_curves, load_ticker_curves, load_walk_forward,
    load_walk_forward_atr, load_oos_selection, load_macro, load_fred_features,
    run_monte_carlo,
    load_correlation_diagnostic, load_regime_correlation, load_dead_weight,
)
from ui.styles import get_color, get_label

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
    fred          = load_fred_features()
    n_assets      = len(ticker_curves) if ticker_curves else 20

    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("## 📈 Strategy Performance Dashboard")
        st.markdown(
            f"<span style='color:#666;font-size:.82rem'>"
            f"Multi-asset systematic strategy · 2015–2025 · {n_assets} assets "
            f"(incl. survivorship anchors GE/INTC/VZ) · "
            f"Multi-signal: MA crossover + momentum breakout + dip-buy · "
            f"Two-sided bonds &amp; commodities · Cross-sectional momentum tilt"
            f"</span>",
            unsafe_allow_html=True,
        )

    if df_port.empty:
        st.error("Run `python run.py portfolio` first to generate portfolio data.")
        return

    # ── Top metrics ──────────────────────────────────────────────────────────
    # Select production method via two-stage gap-filtered OOS Sharpe.
    # Stage 1: filter to OOS Sharpe > 0.9 AND IS-OOS gap in [-0.20, +0.50]
    # Stage 2: among qualifying methods, pick highest OOS Sharpe
    # Fallback: if no method passes gap filter, use highest OOS Sharpe
    _prod_method = "equal_weight"
    if not oos_sel.empty and "oos_sharpe" in oos_sel.columns and "method" in oos_sel.columns:
        _filtered = pd.DataFrame()
        if "is_sharpe" in oos_sel.columns:
            _gaps = oos_sel["is_sharpe"] - oos_sel["oos_sharpe"]
            _mask = (oos_sel["oos_sharpe"] > 0.9) & (_gaps >= -0.20) & (_gaps <= 0.50)
            _filtered = oos_sel[_mask]
        if not _filtered.empty:
            _best_idx = _filtered["oos_sharpe"].idxmax()
        else:
            _best_idx = oos_sel["oos_sharpe"].idxmax()
        _best_m   = oos_sel.loc[_best_idx, "method"]
        # Map oos_selection method name → df_port column name (spaces → underscores)
        _best_col = _best_m.replace(" ", "_").replace("-", "_")
        if _best_col in df_port.columns:
            _prod_method = _best_col
    _prod_label = get_label(_prod_method)

    st.markdown(f'<div class="section-head">{_prod_label} vs buy &amp; hold</div>',
                unsafe_allow_html=True)

    eq_ret  = df_port[_prod_method].pct_change().dropna() if _prod_method in df_port.columns else pd.Series(dtype=float)
    bnh_ret = df_port["buy_hold"].pct_change().dropna()   if "buy_hold"   in df_port.columns else pd.Series(dtype=float)
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "  📊  Equity Curves",
        "  🔍  Alpha Decomposition",
        "  🎲  Monte Carlo",
        "  🔁  Walk-Forward",
        "  🌍  Macro Overlay",
        "  📡  Live Signals",
        "  📈  vs S&P 500",
    ])

    # ── Tab 1: Equity curves + drawdown + summary table ──────────────────────
    with tab1:
        # Compute top 5 methods by final equity value
        _non_bh  = [c for c in df_port.columns if c != "buy_hold"]
        _top5    = sorted(_non_bh, key=lambda c: df_port[c].iloc[-1] if not df_port[c].empty else 0,
                          reverse=True)[:5]
        _default = _top5 + (["buy_hold"] if "buy_hold" in df_port.columns else [])
        _options = list(df_port.columns)
        _option_labels = {c: get_label(c) for c in _options}
        _selected_cols = st.multiselect(
            "Methods to display",
            options=_options,
            default=_default,
            format_func=lambda c: _option_labels.get(c, c),
        )
        if not _selected_cols:
            _selected_cols = _default
        st.plotly_chart(chart_equity(df_port, columns=_selected_cols), theme=None, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})
        st.plotly_chart(chart_drawdown(df_port, columns=_selected_cols), theme=None, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

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

    # ── Tab 2: Alpha Decomposition ───────────────────────────────────────────
    with tab2:
        st.markdown("### Alpha Decomposition")
        _corr_diag   = load_correlation_diagnostic()
        _regime_corr = load_regime_correlation()
        _df_dw       = load_dead_weight()
        _oos_alpha   = load_oos_selection()

        if _corr_diag.empty and _regime_corr.empty and _df_dw.empty and _oos_alpha.empty:
            st.info("Run `python -m pipeline.correlation_diagnostic` to generate alpha decomposition data.")
        else:
            # ── Section A: 4-column metric row ──────────────────────────────
            _best_oos_act_sharpe = None
            _best_ols_beta       = None
            _alpha_total_ratio   = None
            _avg_dw_pct          = None

            if not _oos_alpha.empty:
                if "oos_act_sharpe" in _oos_alpha.columns:
                    _best_oos_act_sharpe = float(_oos_alpha["oos_act_sharpe"].max())
                if "oos_sharpe" in _oos_alpha.columns:
                    _best_oos_idx  = _oos_alpha["oos_sharpe"].idxmax()
                    _best_oos_meth = _oos_alpha.loc[_best_oos_idx, "method"] if "method" in _oos_alpha.columns else ""
                    if "is_sharpe" in _oos_alpha.columns and "oos_sharpe" in _oos_alpha.columns:
                        _is_v  = float(_oos_alpha.loc[_best_oos_idx, "is_sharpe"])
                        _oos_v = float(_oos_alpha.loc[_best_oos_idx, "oos_sharpe"])
                        if _is_v != 0:
                            _alpha_total_ratio = _oos_v / _is_v

            if not _corr_diag.empty:
                _beta_col = next((c for c in _corr_diag.columns if "beta" in c.lower()), None)
                if _beta_col:
                    _best_ols_beta = float(_corr_diag[_beta_col].mean())

            if not _df_dw.empty:
                _dw_col = "dead_weight_pct" if "dead_weight_pct" in _df_dw.columns else \
                          next((c for c in _df_dw.columns if "dead" in c.lower()), None)
                if _dw_col:
                    _avg_dw_pct = float(_df_dw[_dw_col].mean())

            _ma1, _ma2, _ma3, _ma4 = st.columns(4)
            with _ma1:
                st.metric("Best OOS Active Sharpe",
                          f"{_best_oos_act_sharpe:.3f}" if _best_oos_act_sharpe is not None else "—",
                          help="Highest OOS active Sharpe (alpha / active risk) across all methods.")
            with _ma2:
                st.metric("Avg OLS Beta",
                          f"{_best_ols_beta:.3f}" if _best_ols_beta is not None else "—",
                          help="Average OLS beta to SPY across methods in the correlation diagnostic.")
            with _ma3:
                st.metric("OOS / IS Sharpe Ratio",
                          f"{_alpha_total_ratio:.2f}" if _alpha_total_ratio is not None else "—",
                          help="OOS Sharpe divided by IS Sharpe for the best OOS method. "
                               "1.0 = perfect transfer. <0.5 = likely overfitting.")
            with _ma4:
                st.metric("Avg Dead Weight %",
                          f"{_avg_dw_pct*100:.1f}%" if _avg_dw_pct is not None else "—",
                          help="Average dead-weight % across universe tickers. "
                               "<50% = signal better than random on down days.")

            # ── Section B: Beta rolling + Active return charts ──────────────
            _bcol, _acol = st.columns(2)
            _top3_methods = None
            if not df_port.empty:
                _non_bh_cols = [c for c in df_port.columns if c != "buy_hold"]
                _top3_methods = sorted(_non_bh_cols,
                                       key=lambda c: df_port[c].iloc[-1] if not df_port[c].empty else 0,
                                       reverse=True)[:3]
            with _bcol:
                _fig_beta = chart_beta_rolling(df_port, methods=_top3_methods) if not df_port.empty else None
                if _fig_beta:
                    st.plotly_chart(_fig_beta, theme=None, use_container_width=True,
                                    config={"scrollZoom": True, "displayModeBar": True})
                else:
                    st.info("Insufficient data for rolling beta chart (need ≥60 days).")
            with _acol:
                _fig_act = chart_active_return(df_port, methods=_top3_methods) if not df_port.empty else None
                if _fig_act:
                    st.plotly_chart(_fig_act, theme=None, use_container_width=True,
                                    config={"scrollZoom": True, "displayModeBar": True})
                else:
                    st.info("Insufficient data for active return chart (need ≥60 days).")

            # ── Section C: IS vs OOS table from oos_selection ───────────────
            if not _oos_alpha.empty:
                st.markdown("")
                st.markdown('<div class="section-head">IS vs OOS method comparison</div>',
                            unsafe_allow_html=True)

                _disp_cols = {
                    "method"        : "Method",
                    "is_sharpe"     : "IS Sharpe",
                    "oos_sharpe"    : "OOS Sharpe",
                    "is_act_sharpe" : "IS Active Sharpe",
                    "oos_act_sharpe": "OOS Active Sharpe",
                }
                _disp_oos = _oos_alpha.rename(columns={k: v for k, v in _disp_cols.items()
                                                        if k in _oos_alpha.columns})
                _fmt_cols  = {v: "{:.3f}" for k, v in _disp_cols.items()
                              if k != "method" and v in _disp_oos.columns}

                if "OOS Sharpe" in _disp_oos.columns:
                    _best_oos_sharpe_method = _disp_oos.loc[_disp_oos["OOS Sharpe"].idxmax(), "Method"] \
                        if "Method" in _disp_oos.columns else ""
                    _best_act_sharpe_method = ""
                    if "OOS Active Sharpe" in _disp_oos.columns:
                        _best_act_sharpe_method = _disp_oos.loc[
                            _disp_oos["OOS Active Sharpe"].idxmax(), "Method"
                        ] if "Method" in _disp_oos.columns else ""
                    # Identify min IS-OOS gap for OOS > 0.8
                    _yellow_method = ""
                    if "IS Sharpe" in _disp_oos.columns and "OOS Sharpe" in _disp_oos.columns \
                            and "Method" in _disp_oos.columns:
                        _oos_gt08 = _disp_oos[_disp_oos["OOS Sharpe"] > 0.8].copy()
                        if not _oos_gt08.empty:
                            _oos_gt08 = _oos_gt08.copy()
                            _oos_gt08["_gap"] = (_oos_gt08["IS Sharpe"] - _oos_gt08["OOS Sharpe"]).abs()
                            _yellow_method = _oos_gt08.loc[_oos_gt08["_gap"].idxmin(), "Method"]

                    def _style_oos_table(row):
                        styles = [""] * len(row)
                        if "Method" in row.index:
                            m = row["Method"]
                            if m == _best_oos_sharpe_method:
                                styles = ["background-color:#0d2b0d"] * len(row)
                            elif m == _best_act_sharpe_method:
                                styles = ["background-color:#0a1a2e"] * len(row)
                            elif m == _yellow_method:
                                styles = ["background-color:#2b2200"] * len(row)
                        return styles

                    st.dataframe(
                        _disp_oos.style.apply(_style_oos_table, axis=1).format(_fmt_cols),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(
                        "Green = best OOS Sharpe · Blue = best OOS Active Sharpe · "
                        "Yellow = smallest IS-OOS gap (OOS > 0.8)"
                    )
                else:
                    st.dataframe(_disp_oos, use_container_width=True, hide_index=True)

            # ── Section D: Dead weight chart ─────────────────────────────────
            if not _df_dw.empty:
                st.markdown("")
                _fig_dw = chart_dead_weight(_df_dw)
                if _fig_dw:
                    st.plotly_chart(_fig_dw, theme=None, use_container_width=True,
                                    config={"scrollZoom": True, "displayModeBar": True})
                    st.caption(
                        "Dead weight = fraction of LONG signal days where the asset fell. "
                        "<50% = signal has directional edge. Red = above 45% (weak)."
                    )

            # ── Section E: Regime correlation table ──────────────────────────
            if not _regime_corr.empty:
                st.markdown("")
                st.markdown('<div class="section-head">Regime correlation table</div>',
                            unsafe_allow_html=True)
                st.dataframe(_regime_corr, use_container_width=True)
                st.caption(
                    "Pairwise return correlation across market regimes. "
                    "Low/negative correlation = better diversification in the regime."
                )

    # ── Tab 3: Monte Carlo ───────────────────────────────────────────────────
    with tab3:
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

    # ── Tab 4: Walk-forward + per-asset ──────────────────────────────────────
    with tab4:
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
            _wf_col_map = {
                "method"        : "Method",
                "is_sharpe"     : "IS Sharpe",
                "oos_sharpe"    : "OOS Sharpe",
                "is_act_sharpe" : "IS Active Sharpe",
                "oos_act_sharpe": "OOS Active Sharpe",
            }
            display_oos = oos_sel.rename(columns={k: v for k, v in _wf_col_map.items()
                                                   if k in oos_sel.columns})
            _wf_fmt = {v: "{:.3f}" for k, v in _wf_col_map.items()
                       if k != "method" and v in display_oos.columns}
            # Determine highlight methods
            _wf_best_oos = oos_sel.loc[oos_sel["oos_sharpe"].idxmax(), "method"] \
                if "oos_sharpe" in oos_sel.columns else ""
            _wf_best_act = ""
            if "oos_act_sharpe" in oos_sel.columns and "method" in oos_sel.columns:
                _wf_best_act = oos_sel.loc[oos_sel["oos_act_sharpe"].idxmax(), "method"]
            _wf_yellow = ""
            if "is_sharpe" in oos_sel.columns and "oos_sharpe" in oos_sel.columns \
                    and "method" in oos_sel.columns:
                _wf_gt08 = oos_sel[oos_sel["oos_sharpe"] > 0.8].copy()
                if not _wf_gt08.empty:
                    _wf_gt08["_gap"] = (_wf_gt08["is_sharpe"] - _wf_gt08["oos_sharpe"]).abs()
                    _wf_yellow = _wf_gt08.loc[_wf_gt08["_gap"].idxmin(), "method"]

            def _highlight_best(row):
                if "Method" not in row.index:
                    return [""] * len(row)
                m = row["Method"]
                if m == _wf_best_oos:
                    return ["background-color:#1a3a1a"] * len(row)
                if m == _wf_best_act:
                    return ["background-color:#0a1a2e"] * len(row)
                if m == _wf_yellow:
                    return ["background-color:#2b2200"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display_oos.style.apply(_highlight_best, axis=1).format(_wf_fmt),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Green = best OOS Sharpe · Blue = best OOS Active Sharpe · "
                "Yellow = smallest IS-OOS gap (OOS > 0.8)"
            )
            st.markdown(
                f"<span style='color:#50fa7b;font-size:.82rem'>"
                f"✓ Production method: <b>{_prod_label}</b> — selected by highest OOS Sharpe "
                f"among methods with IS-OOS gap in [−0.20, +0.50]</span>",
                unsafe_allow_html=True,
            )

    # ── Tab 5: Macro overlay ─────────────────────────────────────────────────
    with tab5:
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
            st.plotly_chart(chart_macro_overlay(df_port, macro, fred=fred),
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

            # FRED indicator scores (dynamic — only shown when fred_features.parquet exists)
            if "fred_macro_score" in macro.columns:
                st.markdown('<div class="section-head">FRED indicator summary</div>',
                            unsafe_allow_html=True)
                _fms = macro["fred_macro_score"].dropna()
                f1, f2, f3, f4 = st.columns(4)
                with f1: st.metric("FRED Score (latest)", f"{_fms.iloc[-1]:+.2f}" if not _fms.empty else "N/A",
                                     help="Composite of 8 FRED indicators (credit spreads, sentiment, "
                                          "claims, PMI, USD, inflation, VIX). Range −1 to +1.")
                with f2: st.metric("FRED Bullish days", f"{(_fms > 0).mean()*100:.0f}%" if not _fms.empty else "N/A",
                                     help="Days where FRED composite score > 0 (majority of "
                                          "indicators in bullish regime).")
                with f3:
                    if not fred.empty and "hy_oas" in fred.columns:
                        _hy = fred["hy_oas"].dropna()
                        st.metric("HY Spread (latest)", f"{_hy.iloc[-1]:.2f}%" if not _hy.empty else "N/A",
                                    help="ICE BofA High Yield OAS. Rising = credit stress / risk-off. "
                                         "Spikes above 5% historically precede equity drawdowns.")
                    else:
                        st.metric("HY Spread", "N/A", help="Run fred_features.py to populate.")
                with f4:
                    if not fred.empty and "ism_pmi" in fred.columns:
                        _pmi = fred["ism_pmi"].dropna()
                        st.metric("Mfg Production (latest)", f"{_pmi.iloc[-1]:.1f}" if not _pmi.empty else "N/A",
                                    help="Industrial Production: Manufacturing (IPMAN). "
                                         "Rising = expansion, falling = contraction. "
                                         "Leading indicator of economic activity.")
                    else:
                        st.metric("Mfg Production", "N/A", help="Run fred_features.py to populate.")

    # ── Tab 6: Paper Trading & Live Signals ──────────────────────────────────
    with tab6:

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

            # ── Fee totals (from trades.csv) ─────────────────────────────
            total_fees = 0.0
            buy_fees   = 0.0
            sell_fees  = 0.0
            if not pt_trades.empty and "commission" in pt_trades.columns:
                total_fees = pt_trades["commission"].sum()
                buy_fees   = pt_trades.loc[pt_trades["action"] == "BUY", "commission"].sum()
                sell_fees  = pt_trades.loc[pt_trades["action"] == "SELL", "commission"].sum()

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

                # prev_pv = previous trading day's close (entry baseline for intraday
                # daily-return calculation and the "Entry" reference line on the chart).
                # If EOD already ran today, state["portfolio_value"] is today's close —
                # we need the second-to-last history row instead.
                _today_et_str = now.strftime('%Y-%m-%d')
                if (pt_state.get("last_eod_date") == _today_et_str
                        and not pt_history.empty and len(pt_history) >= 2):
                    prev_pv = float(pt_history.sort_values("date").iloc[-2]["portfolio_value"])
                else:
                    prev_pv = pt_state.get("portfolio_value", PT_INITIAL_CAPITAL)

                # ── Open position market values (from fresh per-ticker prices) ─
                # When the market is closed live_prices will be empty — fall back
                # to last EOD close stored in state rather than entry prices,
                # otherwise every position shows 0% change overnight.
                tot_invested = sum(pos["cost_basis"] for pos in pt_state.get("positions", {}).values())
                _eod_pv = pt_state.get("portfolio_value", PT_INITIAL_CAPITAL)
                if live_prices:
                    tot_cur_val = sum(
                        pos["shares"] * (live_prices.get(t) or pos.get("last_close") or pos["entry_price"])
                        for t, pos in pt_state.get("positions", {}).items()
                    )
                else:
                    # Market closed — use last EOD close per position, or aggregate EOD value
                    _has_closes = any(p.get("last_close") for p in pt_state.get("positions", {}).values())
                    if _has_closes:
                        tot_cur_val = sum(
                            pos["shares"] * (pos.get("last_close") or pos["entry_price"])
                            for pos in pt_state.get("positions", {}).values()
                        )
                    else:
                        tot_cur_val = _eod_pv - cash
                tot_unreal   = tot_cur_val - tot_invested
                tot_chg_pct  = (tot_cur_val / tot_invested - 1) * 100 if tot_invested else 0.0

                # pv from live per-ticker prices (consistent with positions table;
                # avoids the intraday OHLC sum having stale zeros at 5-min boundaries)
                pv        = (tot_cur_val + cash) if live_prices else _eod_pv
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
                # Use yesterday's cash (not today's) to avoid mixing days when
                # trades executed today changed the cash balance.
                prev_cash = float(pt_history["cash"].iloc[-1]) if (
                    not pt_history.empty and "cash" in pt_history.columns
                ) else cash
                prev_positions_val   = prev_pv - prev_cash
                today_unrealized     = tot_cur_val - prev_positions_val
                today_unrealized_pct = (today_unrealized / prev_positions_val * 100
                                        if prev_positions_val else 0.0)

                # ── GROUP 1: Portfolio Overview ───────────────────────────────
                st.markdown('<div class="section-head">Portfolio Overview</div>', unsafe_allow_html=True)
                pm1, pm2, pm3, pm4, pm5, pm6 = st.columns(6)
                with pm1:
                    st.metric("Portfolio Value", f"${pv:,.0f}",
                              help="Live value: cash + all open positions marked to last 5-min bar.")
                with pm2:
                    _total_pnl = pv - PT_INITIAL_CAPITAL
                    st.metric("Total Return", f"{total_ret:+.2f}% (${_total_pnl:+,.0f})",
                              help="Overall % return vs $100k starting capital since inception.")
                with pm3:
                    st.metric("Realized P&L", f"${all_realized:+,.0f}",
                              help="Locked-in profit/loss from closed trades only.")
                with pm4:
                    st.metric("Cash", f"${cash:,.0f}",
                              help="Uninvested cash after all trades and commissions.")
                with pm5:
                    st.metric("Total Fees", f"${total_fees:,.2f}",
                              help=f"Cumulative commissions (0.05% per trade). "
                                   f"Entry: ${buy_fees:,.2f} · Exit: ${sell_fees:,.2f}.")
                with pm6:
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

                # ── GROUP 2b: Capital Flow (where did the money go?) ────────
                # Splits realised gains, losses, unrealized, and fees so the
                # user sees exactly where every dollar went.
                realized_gains  = 0.0
                realized_losses = 0.0
                if not pt_trades.empty and "pnl" in pt_trades.columns:
                    _sells_pnl = pd.to_numeric(
                        pt_trades.loc[pt_trades["action"] == "SELL", "pnl"],
                        errors="coerce",
                    ).fillna(0)
                    realized_gains  = float(_sells_pnl[_sells_pnl > 0].sum())
                    realized_losses = float(_sells_pnl[_sells_pnl < 0].sum())

                st.markdown('<div class="section-head">Capital Flow</div>',
                            unsafe_allow_html=True)
                wf1, wf2, wf3, wf4 = st.columns(4)
                with wf1:
                    st.metric("Starting Capital", f"${PT_INITIAL_CAPITAL:,.0f}")
                with wf2:
                    st.metric("Realized Gains", f"${realized_gains:+,.0f}",
                              delta_color="normal",
                              help="Sum of positive P&L from closed trades.")
                with wf3:
                    st.metric("Realized Losses", f"${realized_losses:+,.0f}",
                              delta_color="normal",
                              help="Sum of negative P&L from closed trades.")
                with wf4:
                    st.metric("Unrealized P&L", f"${tot_unreal:+,.0f}",
                              help="Open position value minus cost basis.")
                # Reconciliation: pnl already includes sell commissions,
                # so only subtract buy fees to avoid double-counting.
                _reconciled = PT_INITIAL_CAPITAL + realized_gains + realized_losses + tot_unreal - buy_fees
                _parts = [f"\\$100,000 start"]
                if realized_gains:
                    _parts.append(f"+ \\${realized_gains:,.0f} gains")
                if realized_losses:
                    _parts.append(f"− \\${abs(realized_losses):,.0f} losses")
                if tot_unreal >= 0:
                    _parts.append(f"+ \\${tot_unreal:,.0f} unrealized")
                else:
                    _parts.append(f"− \\${abs(tot_unreal):,.0f} unrealized")
                if buy_fees:
                    _parts.append(f"− \\${buy_fees:,.0f} entry fees")
                _parts.append(f"= \\${_reconciled:,.0f}")
                st.caption("  ".join(_parts))

                # ── Equity chart ──────────────────────────────────────────────
                # Timeframe selector — resample 5-min candles to user choice
                _tf_options = {"5m": "5T", "15m": "15T", "1H": "1h", "4H": "4h"}
                _tf = st.radio("Timeframe", list(_tf_options.keys()),
                               index=0, horizontal=True, label_visibility="collapsed")
                _chart_intra = intraday_df
                if not intraday_df.empty and _tf != "5m":
                    _rule = _tf_options[_tf]
                    _chart_intra = intraday_df.resample(_rule).agg({
                        "open": "first", "high": "max", "low": "min", "close": "last"
                    }).dropna()

                st.plotly_chart(
                    chart_paper_portfolio(pt_history, _chart_intra, pt_trades,
                                          now=now, entry_value=prev_pv,
                                          spy_curve=spy_curve),
                    theme=None, use_container_width=True,
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["autoScale2d"],
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
                        cur = (live_prices.get(ticker)
                               or pos.get("last_close")
                               or pos["entry_price"])
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
                    for _col in ["shares", "price", "value", "commission", "pnl"]:
                        if _col in recent.columns:
                            recent[_col] = pd.to_numeric(recent[_col], errors="coerce")
                    _tfmt = {"shares": "{:.2f}"}
                    for _col in ["price", "value", "commission", "pnl"]:
                        if _col in recent.columns:
                            _tfmt[_col] = "${:,.2f}"
                    st.dataframe(
                        recent.style
                        .map(_color_action, subset=["action"])
                        .format(_tfmt, na_rep="—"),
                        use_container_width=True, hide_index=True,
                    )

                # ── GROUP 5: Trade Analytics ────────────────────────────────
                if not pt_trades.empty and "pnl" in pt_trades.columns:
                    sells_only = pt_trades[
                        (pt_trades["action"] == "SELL") &
                        pt_trades["pnl"].notna() &
                        (pt_trades["pnl"].astype(str).str.strip() != "")
                    ].copy()
                    if not sells_only.empty:
                        sells_only["pnl_num"] = pd.to_numeric(sells_only["pnl"], errors="coerce").fillna(0)
                        wins  = sells_only[sells_only["pnl_num"] > 0]
                        losses = sells_only[sells_only["pnl_num"] < 0]
                        win_rate = len(wins) / len(sells_only) * 100 if len(sells_only) else 0
                        avg_win  = wins["pnl_num"].mean() if len(wins) else 0
                        avg_loss = losses["pnl_num"].mean() if len(losses) else 0
                        profit_factor = (wins["pnl_num"].sum() / abs(losses["pnl_num"].sum())
                                         if len(losses) and losses["pnl_num"].sum() != 0 else float('inf'))
                        largest_win  = wins["pnl_num"].max() if len(wins) else 0
                        largest_loss = losses["pnl_num"].min() if len(losses) else 0
                        avg_hold = None
                        if "date" in sells_only.columns and not pt_trades.empty:
                            try:
                                _buy_dates = pt_trades[pt_trades["action"] == "BUY"].groupby("ticker")["date"].first()
                                _sell_dates = sells_only.groupby("ticker")["date"].first()
                                _common = _buy_dates.index.intersection(_sell_dates.index)
                                if len(_common):
                                    _holds = (_sell_dates[_common] - _buy_dates[_common]).dt.days
                                    avg_hold = _holds.mean()
                            except Exception:
                                pass

                        st.markdown('<div class="section-head">Trade Analytics</div>',
                                    unsafe_allow_html=True)
                        ta1, ta2, ta3, ta4, ta5, ta6 = st.columns(6)
                        with ta1:
                            st.metric("Win Rate", f"{win_rate:.0f}%",
                                      help=f"{len(wins)} wins / {len(sells_only)} closed trades")
                        with ta2:
                            st.metric("Avg Win", f"${avg_win:+,.0f}",
                                      help="Average dollar P&L on winning trades")
                        with ta3:
                            st.metric("Avg Loss", f"${avg_loss:+,.0f}",
                                      help="Average dollar P&L on losing trades")
                        with ta4:
                            st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor < 100 else "∞",
                                      help="Gross profit / gross loss. >1.5 is good, >2.0 is exceptional.")
                        with ta5:
                            st.metric("Best Trade", f"${largest_win:+,.0f}")
                        with ta6:
                            st.metric("Worst Trade", f"${largest_loss:+,.0f}")

                # ── GROUP 6: Drawdown + Rolling Sharpe + Allocation ────────────
                if not pt_history.empty and len(pt_history) >= 3:
                    _pv_s = pt_history.set_index("date")["portfolio_value"]
                    _pv_s.index = pd.to_datetime(_pv_s.index)

                    _chart_cols = st.columns(2)

                    # Drawdown chart
                    with _chart_cols[0]:
                        _peak = _pv_s.cummax()
                        _dd_pct = ((_pv_s - _peak) / _peak * 100)
                        _fig_dd = go.Figure()
                        _fig_dd.add_trace(go.Scatter(
                            x=_dd_pct.index, y=_dd_pct.values,
                            fill="tozeroy", fillcolor="rgba(255,85,85,0.15)",
                            line=dict(color="#ff5555", width=1.2),
                            hovertemplate="<b>%{x|%b %d}</b><br>DD: %{y:.1f}%<extra></extra>",
                        ))
                        _fig_dd.add_hline(y=-12, line_color="#ffb86c", line_dash="dash",
                                          annotation_text="  -12% circuit breaker",
                                          annotation_font_color="#ffb86c")
                        _fig_dd.update_layout(
                            paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                            font=dict(color="#c0c0c0", size=11), height=220,
                            dragmode="pan", uirevision="paper_drawdown",
                            title=dict(text="Portfolio Drawdown", font=dict(size=12)),
                            yaxis=dict(title="DD %", gridcolor="#2a2a2a"),
                            xaxis=dict(gridcolor="#2a2a2a"),
                            margin=dict(l=50, r=20, t=35, b=30), showlegend=False,
                        )
                        st.plotly_chart(_fig_dd, theme=None, use_container_width=True,
                                        config={"scrollZoom": True, "displayModeBar": False})

                    # Rolling 21-day Sharpe
                    with _chart_cols[1]:
                        _daily_ret = _pv_s.pct_change().dropna()
                        if len(_daily_ret) >= 21:
                            _roll_mean = _daily_ret.rolling(21).mean()
                            _roll_std  = _daily_ret.rolling(21).std()
                            _roll_sharpe = (_roll_mean / _roll_std * np.sqrt(252)).dropna()
                            _fig_rs = go.Figure()
                            _fig_rs.add_trace(go.Scatter(
                                x=_roll_sharpe.index, y=_roll_sharpe.values,
                                line=dict(color="#4a9eff", width=1.2),
                                hovertemplate="<b>%{x|%b %d}</b><br>Sharpe: %{y:.2f}<extra></extra>",
                            ))
                            _fig_rs.add_hline(y=0, line_color="#555", line_dash="dot")
                            _fig_rs.add_hline(y=1, line_color="#50fa7b", line_dash="dot",
                                              line_width=0.8)
                            _fig_rs.update_layout(
                                paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                                font=dict(color="#c0c0c0", size=11), height=220,
                                dragmode="pan", uirevision="paper_rolling_sharpe",
                                title=dict(text="Rolling 21-Day Sharpe", font=dict(size=12)),
                                yaxis=dict(title="Sharpe", gridcolor="#2a2a2a"),
                                xaxis=dict(gridcolor="#2a2a2a"),
                                margin=dict(l=50, r=20, t=35, b=30), showlegend=False,
                            )
                            st.plotly_chart(_fig_rs, theme=None, use_container_width=True,
                                            config={"scrollZoom": True, "displayModeBar": False})

                # ── GROUP 7: Asset Class Allocation ────────────────────────────
                if pt_state.get("positions"):
                    st.markdown('<div class="section-head">Allocation by Asset Class</div>',
                                unsafe_allow_html=True)
                    _alloc_rows = []
                    for _t, _p in pt_state["positions"].items():
                        _cur_price = (live_prices.get(_t)
                                      or _p.get("last_close")
                                      or _p["entry_price"])
                        _mkt_val = _p["shares"] * _cur_price
                        _ac = ASSET_CLASS.get(_t, "other")
                        _alloc_rows.append({"asset_class": _ac, "ticker": _t, "value": _mkt_val})
                    _alloc_df = pd.DataFrame(_alloc_rows)
                    _ac_totals = _alloc_df.groupby("asset_class")["value"].sum()

                    _alloc_c1, _alloc_c2 = st.columns([1, 2])
                    with _alloc_c1:
                        _ac_colors = {
                            "equity_index": "#4a9eff", "sector_etf": "#50fa7b",
                            "stock": "#ffb86c", "bond": "#bd93f9",
                            "commodity": "#ff79c6", "other": "#888",
                        }
                        _fig_pie = go.Figure(go.Pie(
                            labels=_ac_totals.index.tolist(),
                            values=[round(v, 2) for v in _ac_totals.values.tolist()],
                            marker=dict(colors=[_ac_colors.get(a, "#888") for a in _ac_totals.index]),
                            textinfo="label+percent",
                            textfont=dict(size=11),
                            hole=0.4,
                            hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
                        ))
                        _fig_pie.update_layout(
                            paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                            font=dict(color="#c0c0c0", size=11), height=250,
                            margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                        )
                        st.plotly_chart(_fig_pie, theme=None, use_container_width=True)
                    with _alloc_c2:
                        _alloc_detail = _alloc_df.copy()
                        _alloc_detail["pct"] = _alloc_detail["value"] / _alloc_detail["value"].sum() * 100
                        _alloc_detail = _alloc_detail.sort_values("value", ascending=False)
                        st.dataframe(
                            _alloc_detail[["ticker", "asset_class", "value", "pct"]].style.format(
                                {"value": "${:,.0f}", "pct": "{:.1f}%"}
                            ),
                            use_container_width=True, hide_index=True, height=250,
                        )

                # ── Strategy info + risk state ─────────────────────────────────
                _strat_name = pt_state.get("strategy", "unknown")
                _rp_date = pt_state.get("last_rp_date", "—")
                st.caption(
                    f"Strategy: **{_strat_name}**  ·  "
                    f"Init: {pt_state.get('initialized_date','?')}  ·  "
                    f"Last EOD: {pt_state.get('last_eod_date','?')}  ·  "
                    f"RP weights: {_rp_date}  ·  "
                    "Filters: RSI<70  ·  3x ATR stop  ·  5-day hold  ·  15% kill switch"
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
                    if isinstance(v, str) and v == "LONG":       return "color:#50fa7b;font-weight:600"
                    if isinstance(v, str) and v == "FLAT":       return "color:#666"
                    if isinstance(v, str) and v == "FAST LONG":  return "color:#61afef;font-weight:600"
                    if isinstance(v, str) and v == "FAST FLAT":  return "color:#555"
                    if isinstance(v, str) and v == "✓":          return "color:#50fa7b;font-weight:600"
                    if isinstance(v, (int, float)):
                        if v > 0: return "color:#50fa7b"
                        if v < 0: return "color:#ff5555"
                    return ""

                _live_numeric_cols = ["Day Chg %", "MA Spread %", "20d Ret %", "60d Ret %", "Dist High %"]
                _live_color_cols   = ["Signal"] + _live_numeric_cols
                # Add extra columns if they exist
                for _ec in ["Breakout", "Squeeze", "Fast", "Mom Rank"]:
                    if _ec in live_df.columns:
                        _live_color_cols.append(_ec)
                _live_fmt = {
                    "Price":      "${:.2f}",
                    "Day Chg %":  "{:+.2f}%",
                    "MA Spread %":"{:+.2f}%",
                    "RSI":        "{:.1f}",
                    "20d Ret %":  "{:+.1f}%",
                    "60d Ret %":  "{:+.1f}%",
                    "Dist High %":"{:+.1f}%",
                }

                st.dataframe(
                    live_df.style
                    .apply(_style_live, axis=1)
                    .map(_color_cell, subset=[c for c in _live_color_cols if c in live_df.columns])
                    .format({k: v for k, v in _live_fmt.items() if k in live_df.columns}),
                    use_container_width=True, hide_index=True,
                )

        _live_section()

    # ── Tab 7: vs S&P 500 ────────────────────────────────────────────────────
    with tab7:

        @st.cache_data(ttl=300, show_spinner=False)
        def _spy_history(start_date: str):
            """Fetch daily ^GSPC closing prices from inception to today."""
            import yfinance as yf
            try:
                raw = yf.download("^GSPC", start=start_date, auto_adjust=True,
                                  progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.droplevel(1)
                raw.index = pd.to_datetime(raw.index).tz_localize(None)
                if "Close" in raw.columns and not raw.empty:
                    return raw["Close"].dropna()
            except Exception:
                pass
            try:
                raw = yf.Ticker("^GSPC").history(start=start_date, auto_adjust=True)
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
            st.markdown('<div class="section-head">Overall vs S&P 500</div>',
                        unsafe_allow_html=True)

            intraday_df, live_prices, spy_intraday, spy_pct_from_prev = get_intraday_curve()

            cash    = pt_state["cash"]
            # If EOD already ran today, state["portfolio_value"] is today's
            # close — use second-to-last history row as yesterday's baseline.
            _today_et_str_sp = now_et.strftime('%Y-%m-%d')
            if (pt_state.get("last_eod_date") == _today_et_str_sp
                    and not pt_history.empty and len(pt_history) >= 2):
                prev_pv = float(pt_history.sort_values("date").iloc[-2]["portfolio_value"])
            else:
                prev_pv = pt_state.get("portfolio_value", PT_INITIAL_CAPITAL)
            tot_invested = sum(pos["cost_basis"] for pos in pt_state.get("positions", {}).values())
            tot_cur_val = sum(
                pos["shares"] * (live_prices.get(t) or pos.get("last_close") or pos["entry_price"])
                for t, pos in pt_state.get("positions", {}).items()
            )
            pv = (tot_cur_val + cash) if live_prices else pt_state.get("portfolio_value", prev_pv)

            # Portfolio today % — measured from yesterday's EOD value (same baseline
            # as SPY's close-to-close metric) so chart and alpha metric agree.
            port_today_pct = (pv / prev_pv - 1) * 100 if prev_pv else 0.0

            # SPY % from yesterday's close.
            # spy_pct_from_prev is None when the daily fetch failed — fall back to
            # intraday-only calculation (misses the open gap but is always available).
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
                    "^GSPC (S&P 500 index) % change from yesterday's close — "
                    "matches Yahoo Finance and TradingView exactly. "
                    f"Last bar: {_last_bar_str} ET. "
                    "Data source: Yahoo Finance free tier (≈15 min delay)."
                )
                st.metric("S&P 500 Today", f"{spy_today_pct:+.2f}%", help=_spy_help)
            with c3:
                st.metric("Alpha (today)", f"{alpha_today:+.2f}%",
                          help="Portfolio return minus S&P return today. Positive = beating the index.")
            with c4:
                st.metric("Portfolio Value", f"${pv:,.0f}",
                          help="Live portfolio value (positions × latest 5-min bar + cash).")

            # Daily alpha bar chart — all trading days including today
            _alpha_chart_shown = False
            if not pt_history.empty and len(pt_history) >= 2:
                _bar_start = str(pt_history["date"].min().date())
                _bar_spy = _spy_history(_bar_start)
                if not _bar_spy.empty:
                    _bar_hist = pt_history.set_index("date")["portfolio_value"].copy()
                    _bar_hist.index = pd.to_datetime(_bar_hist.index)
                    _bar_hist = _bar_hist[~_bar_hist.index.duplicated(keep='last')]
                    # Always use live portfolio value for today (overwrite stale EOD)
                    _bar_today = pd.Timestamp(now_et.date())
                    if pv > 0:
                        _bar_hist[_bar_today] = pv
                    _bar_dates = _bar_spy.index.intersection(
                        pd.date_range(_bar_hist.index.min(), _bar_hist.index.max(), freq="B")
                    )
                    _bar_hf = _bar_hist.reindex(_bar_dates).ffill()
                    _bar_sa = _bar_spy.reindex(_bar_dates).ffill()
                    _bar_valid = _bar_hf.notna() & _bar_sa.notna()
                    _bar_hf = _bar_hf[_bar_valid]
                    _bar_sa = _bar_sa[_bar_valid]
                    if len(_bar_hf) >= 2:
                        _bar_pr = _bar_hf.pct_change().dropna() * 100
                        _bar_sr = _bar_sa.pct_change().dropna() * 100
                        _bar_common = _bar_pr.index.intersection(_bar_sr.index)
                        _bar_alpha = _bar_pr[_bar_common] - _bar_sr[_bar_common]
                        if len(_bar_alpha) > 0:
                            fig_alpha = go.Figure()
                            _bar_colors = ['#50fa7b' if a >= 0 else '#ff5555'
                                           for a in _bar_alpha.values]
                            fig_alpha.add_trace(go.Bar(
                                x=_bar_alpha.index, y=_bar_alpha.values,
                                marker_color=_bar_colors,
                                hovertemplate=(
                                    "<b>%{x|%b %d}</b><br>"
                                    "Alpha: %{y:+.2f}%<extra></extra>"
                                ),
                            ))
                            fig_alpha.add_hline(y=0, line_color="#555", line_width=1)
                            _avg_alpha = float(_bar_alpha.mean())
                            fig_alpha.add_hline(
                                y=_avg_alpha, line_color="#4a9eff", line_dash="dash",
                                line_width=1,
                                annotation_text=f"  avg {_avg_alpha:+.2f}%",
                                annotation_font_color="#4a9eff",
                            )
                            fig_alpha.update_layout(**_layout(
                                height=300, uirevision="vs_spy_daily_alpha",
                                dragmode="pan",
                                title=dict(
                                    text="Daily Alpha vs S&P 500 (Portfolio Return − Index Return)",
                                    font=dict(size=12),
                                ),
                                yaxis=dict(title="Alpha %", gridcolor="#2a2a2a",
                                           zeroline=True, zerolinecolor="#555"),
                                xaxis=dict(type="date", tickformat="%b %d",
                                           gridcolor="#2a2a2a"),
                                margin=dict(l=60, r=20, t=35, b=30),
                                showlegend=False, bargap=0.15,
                            ))
                            st.plotly_chart(
                                fig_alpha, theme=None, use_container_width=True,
                                config={"scrollZoom": True, "displayModeBar": True},
                                key="vs_spy_daily_alpha",
                            )
                            _alpha_chart_shown = True
            if not _alpha_chart_shown:
                st.info("Need at least 2 trading days to show the daily alpha chart.")

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
                with st.spinner("Fetching S&P 500 history…"):
                    spy_closes = _spy_history(start_date)

                if spy_closes.empty:
                    st.error("Could not fetch ^GSPC data.")
                else:
                    # Build aligned daily series using ALL SPY trading days
                    hist = pt_history.set_index("date")["portfolio_value"].copy()
                    hist.index = pd.to_datetime(hist.index)
                    hist = hist[~hist.index.duplicated(keep='last')]
                    # Always use live portfolio value for today (overwrite stale EOD)
                    _today_ts = pd.Timestamp(now_et.date())
                    if pv > 0:
                        hist[_today_ts] = pv

                    all_dates = spy_closes.index.intersection(
                        pd.date_range(hist.index.min(), hist.index.max(), freq="B")
                    )
                    hist_full = hist.reindex(all_dates).ffill()
                    spy_close_aligned = spy_closes.reindex(all_dates).ffill()

                    _valid = hist_full.notna() & spy_close_aligned.notna()
                    hist_full = hist_full[_valid]
                    spy_close_aligned = spy_close_aligned[_valid]

                    # Normalize both to $100k at inception
                    _port_scale = PT_INITIAL_CAPITAL / hist_full.iloc[0] if hist_full.iloc[0] else 1
                    _spy_scale  = PT_INITIAL_CAPITAL / spy_close_aligned.iloc[0] if spy_close_aligned.iloc[0] else 1
                    port_norm = hist_full * _port_scale
                    spy_norm  = spy_close_aligned * _spy_scale

                    # Daily returns
                    port_ret = hist_full.pct_change().dropna() * 100
                    spy_ret  = spy_close_aligned.pct_change().dropna() * 100
                    common   = port_ret.index.intersection(spy_ret.index)

                    # Summary
                    port_total  = (hist_full.iloc[-1] / hist_full.iloc[0] - 1) * 100
                    spy_total   = (spy_close_aligned.iloc[-1] / spy_close_aligned.iloc[0] - 1) * 100
                    total_alpha = port_total - spy_total
                    if len(common):
                        win_days   = int((port_ret.reindex(common).values > spy_ret.reindex(common).values).sum())
                        total_days = len(common)
                    else:
                        win_days, total_days = 0, 0

                    if len(common) >= 5:
                        beta = float(np.cov(port_ret[common], spy_ret[common])[0, 1] /
                                     np.var(spy_ret[common])) if np.var(spy_ret[common]) > 0 else 1.0
                    else:
                        beta = None

                    h1, h2, h3, h4, h5 = st.columns(5)
                    with h1: st.metric("Portfolio Return", f"{port_total:+.2f}%",
                                       help="Total return since portfolio inception.")
                    with h2: st.metric("S&P 500 Return",  f"{spy_total:+.2f}%",
                                       help="^GSPC (S&P 500 index) return over the same period.")
                    with h3: st.metric("Total Alpha",     f"{total_alpha:+.2f}%",
                                       help="Portfolio return minus ^GSPC return.")
                    with h4: st.metric("Beta", f"{beta:.2f}" if beta is not None else "N/A",
                                       help="Requires 5+ trading days of data."
                                            if beta is None else
                                            "Portfolio sensitivity to S&P moves. "
                                            "1.0 = moves with the market.")
                    with h5: st.metric("Days Beating S&P", f"{win_days}/{total_days}",
                                       help="Trading days where portfolio daily return exceeded ^GSPC.")

                    # Historical chart — line graph
                    fig_hist = go.Figure()

                    fig_hist.add_trace(go.Scatter(
                        x=port_norm.index, y=port_norm.values,
                        name="My Portfolio",
                        mode="lines+markers",
                        line=dict(color="#4a9eff", width=2.5),
                        marker=dict(size=7, color="#4a9eff"),
                        hovertemplate="<b>Portfolio</b><br>%{x|%b %d}  $%{y:,.0f}<extra></extra>",
                    ))
                    fig_hist.add_trace(go.Scatter(
                        x=spy_norm.index, y=spy_norm.values,
                        name="S&P 500 (^GSPC)",
                        mode="lines+markers",
                        line=dict(color="#ffb86c", width=2),
                        marker=dict(size=6, color="#ffb86c"),
                        hovertemplate="<b>S&P 500</b><br>%{x|%b %d}  $%{y:,.0f}<extra></extra>",
                    ))

                    fig_hist.add_hline(y=PT_INITIAL_CAPITAL, line_color="#444", line_dash="dot",
                                       line_width=1, annotation_text="  $100k start",
                                       annotation_font_color="#555")
                    fig_hist.update_layout(**_layout(
                        height=350, uirevision="vs_spy_hist",
                        dragmode="pan",
                        title=dict(text="Portfolio vs S&P 500 — both normalized to $100k at inception",
                                   font=dict(size=12)),
                        yaxis=dict(tickprefix="$", tickformat=",.0f", autorange=True),
                        xaxis=dict(type="date", tickformat="%b %d",
                                   rangeslider=dict(visible=False)),
                        margin=dict(l=70),
                    ))
                    st.plotly_chart(fig_hist, theme=None, use_container_width=True,
                                   config={"scrollZoom": True, "displayModeBar": True})

                    # Daily breakdown — every trading day
                    st.markdown('<div class="section-head">Daily Breakdown</div>',
                                unsafe_allow_html=True)
                    # Build table from ALL aligned dates (including inception)
                    tbl_rows = []
                    for d in reversed(list(hist_full.index)):
                        pr = float(port_ret[d]) if d in port_ret.index else 0.0
                        sr = float(spy_ret[d])  if d in spy_ret.index  else 0.0
                        pv_d = float(hist_full[d])
                        sv_d = float(spy_norm[d]) if d in spy_norm.index else 0.0
                        tbl_rows.append({
                            "Date"         : d.strftime("%Y-%m-%d"),
                            "Portfolio $"  : round(pv_d, 0),
                            "Portfolio %"  : round(pr, 2),
                            "S&P 500 %"    : round(sr, 2),
                            "Alpha"        : round(pr - sr, 2),
                            "Beating S&P"  : "✓" if pr > sr else ("—" if pr == sr == 0 else "✗"),
                        })
                    if tbl_rows:
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
                            .format({"Portfolio $": "${:,.0f}",
                                     "Portfolio %": "{:+.2f}%", "S&P 500 %": "{:+.2f}%",
                                     "Alpha": "{:+.2f}%"}),
                            use_container_width=True, hide_index=True,
                        )

                    st.caption(
                        "Beta requires 5+ trading days of data. "
                        "Alpha = portfolio daily return − ^GSPC daily return. "
                        "Both series normalized to $100k at portfolio inception date."
                    )

        _vs_sp_section()


if __name__ == "__main__":
    main()
