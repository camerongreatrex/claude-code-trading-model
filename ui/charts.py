# ui/charts.py
# All chart-builder functions and the metrics/metrics_table helpers.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from paper_trader import INITIAL_CAPITAL as PT_INITIAL_CAPITAL
from ui.styles import _layout, PALETTE, LABELS


def metrics(ret: pd.Series) -> dict:
    """
    Compute a standard set of annualised risk/return metrics from a daily
    returns series.

    Args:
        ret: Daily return series (fractional, e.g. 0.01 for +1%).
             NaN values are dropped before calculation.

    Returns:
        Dict with keys:
            ann_r    — Compound Annual Growth Rate (CAGR)
            vol      — Annualised volatility (std dev × √252)
            sharpe   — Sharpe ratio (ann_r / vol, risk-free = 0)
            max_dd   — Maximum drawdown (negative fraction, e.g. -0.20)
            calmar   — Calmar ratio (ann_r / |max_dd|)
            win_rate — Fraction of non-zero days with a positive return
            total    — Total return over the full period (fraction)
        Returns empty dict if ret is empty after dropping NaNs.
    """
    ret = ret.dropna()
    if ret.empty:
        return {}
    cum    = (1 + ret).cumprod()
    total  = cum.iloc[-1] - 1
    n_y    = len(ret) / 252
    ann_r  = (1 + total) ** (1 / n_y) - 1 if n_y > 0 else 0
    vol    = ret.std() * 252 ** .5
    sharpe = ann_r / vol if vol > 0 else 0
    peak   = cum.cummax()
    max_dd = ((cum - peak) / peak).min()
    calmar = ann_r / abs(max_dd) if max_dd != 0 else 0
    # Only count days with actual trades (non-zero return) to avoid inflating
    # the win rate with cash/flat days where the portfolio didn't move.
    active = ret[ret != 0]
    wr     = (active > 0).sum() / len(active) if len(active) > 0 else 0
    return dict(ann_r=ann_r, vol=vol, sharpe=sharpe, max_dd=max_dd,
                calmar=calmar, win_rate=wr, total=total)


# ── Summary table ─────────────────────────────────────────────────────────────
def metrics_table(df_port: pd.DataFrame) -> pd.DataFrame:
    """
    Build a human-readable summary statistics table for all sizing methods.

    Calls metrics() for each method column and formats values as display
    strings with consistent sign/decimal conventions.  The resulting
    DataFrame is rendered in the Overview tab with green/red cell colouring
    applied via Streamlit's .style.map().

    Args:
        df_port: Portfolio curves DataFrame (same as load_portfolio_curves()).

    Returns:
        DataFrame with columns [Method, Ann. Return, Volatility, Sharpe,
        Max DD, Calmar, Win Rate].  One row per sizing method present in df_port.
    """
    rows = []
    for col in ["equal_weight", "atr_sized", "atr_pca_macro", "eq_dd_control",
                "vol_target", "buy_hold"]:
        if col not in df_port.columns:
            continue
        ret = df_port[col].pct_change().dropna()
        m   = metrics(ret)
        rows.append({
            "Method"      : LABELS[col],
            "Ann. Return" : f"{m['ann_r']*100:+.1f}%",
            "Volatility"  : f"{m['vol']*100:.1f}%",
            "Sharpe"      : f"{m['sharpe']:.2f}",
            "Max DD"      : f"{m['max_dd']*100:.1f}%",
            "Calmar"      : f"{m['calmar']:.2f}",
            "Win Rate"    : f"{m['win_rate']*100:.0f}%",
        })
    return pd.DataFrame(rows)


# ── Chart builders ────────────────────────────────────────────────────────────
def chart_equity(df: pd.DataFrame, height: int = 420) -> go.Figure:
    """
    Build a multi-line equity curve chart for all portfolio sizing methods.

    Adds one go.Scatter trace per method present in df.  Buy & Hold is
    rendered as a dotted line to visually distinguish the benchmark from
    the strategy variants.

    Args:
        df:     DataFrame indexed by date with one column per sizing method
                (keys defined in PALETTE/LABELS).
        height: Chart height in pixels.

    Returns:
        go.Figure with dragmode="pan" and scrollZoom enabled via config at
        the call site.
    """
    fig = go.Figure()
    for col in ["equal_weight", "atr_sized", "atr_pca_macro", "eq_dd_control", "vol_target", "buy_hold"]:
        if col not in df.columns:
            continue
        dash = "dot" if col == "buy_hold" else "solid"
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=LABELS[col],
            line=dict(color=PALETTE[col], width=1.8, dash=dash),
            hovertemplate=(
                f"<b>{LABELS[col]}</b><br>"
                "$%{y:,.0f}<br>"
                "<i>Total portfolio value on this date (started at $100k).<br>"
                "A rising line means the strategy is making money.</i>"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(**_layout(
        height=height,
        dragmode="pan",
        title=dict(text="Portfolio Equity Curves — $100 k starting capital",
                   font=dict(size=13)),
        yaxis=dict(title="Value ($)", fixedrange=False),
        xaxis=dict(fixedrange=False),
    ))
    return fig


def chart_drawdown(df: pd.DataFrame, height: int = 420) -> go.Figure:
    """
    Build a drawdown chart comparing the equal-weight strategy to Buy & Hold.

    Computes percentage drawdown as (price − rolling_peak) / rolling_peak × 100
    and renders each as a filled area trace so shallow drawdowns are immediately
    visible against the dark background.

    Only the equal_weight and buy_hold columns are plotted (the others would
    create a cluttered chart; this pair provides the most useful comparison).

    Args:
        df:     DataFrame indexed by date with at least columns
                [equal_weight, buy_hold].
        height: Chart height in pixels.

    Returns:
        go.Figure with fill="tozeroy" traces and dragmode="pan".
    """
    fig = go.Figure()
    for col, alpha in [("equal_weight", 0.18), ("buy_hold", 0.10)]:
        if col not in df.columns:
            continue
        p  = df[col]
        dd = (p - p.cummax()) / p.cummax() * 100
        r, g, b = (int(PALETTE[col].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        fig.add_trace(go.Scatter(
            x=df.index, y=dd, name=LABELS[col],
            line=dict(color=PALETTE[col], width=1.4),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},{alpha})",
            hovertemplate=(
                f"<b>{LABELS[col]}</b><br>"
                "%{y:.1f}% from peak<br>"
                "<i>How far the portfolio has fallen from its all-time high.<br>"
                "-15% means it lost 15% from the top before recovering.<br>"
                "Smaller magnitude = shallower dip = better risk control.</i>"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(**_layout(
        height=height,
        dragmode="pan",
        title=dict(text="Drawdown (%)", font=dict(size=13)),
        yaxis=dict(title="DD (%)", fixedrange=False),
        xaxis=dict(fixedrange=False),
    ))
    return fig


def chart_monte_carlo(eq_curves: np.ndarray, actual: np.ndarray,
                      height: int = 420) -> go.Figure:
    """
    Build the Monte Carlo fan chart showing simulated equity path distribution.

    Traces added (in order):
      1. Faint individual sample paths — up to 80 paths merged into ONE
         go.Scatter with None separators so rendering stays fast during
         zoom/pan (one WebGL draw call instead of 80).
      2. 5–95th percentile band (go.Scatter fill="toself", blue tint).
      3. 25–75th percentile band (go.Scatter fill="toself", green tint).
      4. Median path (go.Scatter, white line).
      5. Actual historical equity curve (go.Scatter, blue line) overlaid
         so the user can see where history sits within the fan.

    Args:
        eq_curves: Array of shape (n_paths, n_days) with cumulative growth
                   factors (1.0 = break even), produced by run_monte_carlo().
        actual:    1-D array of the real cumulative growth factor series.
        height:    Chart height in pixels.

    Returns:
        go.Figure with dragmode="pan".
    """
    n   = eq_curves.shape[1]
    xs  = np.arange(n)
    p5, p25, p50, p75, p95 = (np.percentile(eq_curves, q, axis=0)
                               for q in (5, 25, 50, 75, 95))
    fig = go.Figure()

    # Faint individual paths — merged into ONE trace with None separators.
    # One trace instead of 80 = drastically faster zoom/pan rendering.
    sample_idx = np.random.default_rng(0).choice(len(eq_curves),
                                                   size=min(80, len(eq_curves)),
                                                   replace=False)
    xs_m: list = []
    ys_m: list = []
    for i in sample_idx:
        xs_m.extend(xs.tolist())
        xs_m.append(None)
        ys_m.extend(eq_curves[i].tolist())
        ys_m.append(None)
    fig.add_trace(go.Scatter(
        x=xs_m, y=ys_m,
        line=dict(color="rgba(180,180,180,0.05)", width=0.8),
        showlegend=False, hoverinfo="skip",
    ))

    # 5–95 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself", fillcolor="rgba(74,158,255,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5 – 95th pct", hoverinfo="skip",
    ))
    # 25–75 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself", fillcolor="rgba(80,250,123,0.11)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25 – 75th pct", hoverinfo="skip",
    ))
    # Median
    fig.add_trace(go.Scatter(
        x=xs, y=p50, name="Median path",
        line=dict(color="#e0e0e0", width=1.8),
        hovertemplate=(
            "<b>Day %{x}</b><br>"
            "Median simulated: %{y:.2f}× starting capital<br>"
            "<i>Half of all simulated paths are above this, half below.<br>"
            "1.5 = 50% gain. Think of this as the 'expected' outcome.</i>"
            "<extra></extra>"
        ),
    ))
    # Actual
    fig.add_trace(go.Scatter(
        x=xs[:len(actual)], y=actual, name="Historical",
        line=dict(color=PALETTE["equal_weight"], width=2.2),
        hovertemplate=(
            "<b>Day %{x}</b><br>"
            "Actual history: %{y:.2f}× starting capital<br>"
            "<i>What the real strategy returned, overlaid on the simulated range.<br>"
            "Staying near the median = no unusual luck or bad luck at play.</i>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(**_layout(
        height=height,
        dragmode="pan",
        title=dict(
            text=f"Monte Carlo Bootstrap  ·  {len(eq_curves):,} paths  ·  21-day block resampling",
            font=dict(size=13),
        ),
        xaxis=dict(title="Trading days", fixedrange=False),
        yaxis=dict(title="Growth of $1", fixedrange=False),
    ))
    return fig


def chart_mc_histogram(eq_curves: np.ndarray, height: int = 280) -> go.Figure:
    """
    Build a histogram of terminal (final-day) total returns from Monte Carlo paths.

    Adds vertical reference lines at the median and 5th-percentile so the
    user can immediately see the expected outcome and the stress-test floor.

    Args:
        eq_curves: Array of shape (n_paths, n_days) with cumulative growth
                   factors (same array produced by run_monte_carlo()).
        height:    Chart height in pixels.

    Returns:
        go.Figure with showlegend=False and dragmode="pan".
    """
    finals = (eq_curves[:, -1] - 1) * 100   # % total return
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=finals, nbinsx=60,
        marker_color=PALETTE["equal_weight"], opacity=0.75,
        hovertemplate=(
            "Total return: %{x:.1f}%<br>"
            "Simulated paths: %{y}<br>"
            "<i>%{y} out of 1,000 scenarios ended with this return.<br>"
            "A cluster to the right of 0% = strategy has a positive edge.</i>"
            "<extra></extra>"
        ),
        name="Path distribution",
    ))
    med = np.median(finals)
    fig.add_vline(x=med, line_color="#e0e0e0", line_dash="dot",
                  annotation_text=f"  Median {med:.1f}%",
                  annotation_font_color="#c0c0c0")
    fig.add_vline(x=np.percentile(finals, 5),
                  line_color=PALETTE["neg"], line_dash="dash",
                  annotation_text="  5th pct",
                  annotation_font_color=PALETTE["neg"])
    fig.update_layout(**_layout(
        height=height, showlegend=False,
        dragmode="pan",
        title=dict(text="Distribution of Terminal Returns", font=dict(size=13)),
        xaxis=dict(title="Total return (%)", fixedrange=False),
        yaxis=dict(title="Paths", fixedrange=False),
    ))
    return fig


def chart_walk_forward(wf: pd.DataFrame, height: int = 280) -> go.Figure:
    """
    Build a bar chart of OOS Sharpe ratios from walk-forward validation.

    Each bar represents one test window (1 year of genuinely unseen data
    after a 3-year training period).  Bars are coloured green/red based on
    sign so poor windows are immediately visible.

    Args:
        wf:     DataFrame with columns [period, sharpe].
        height: Chart height in pixels.

    Returns:
        go.Figure with a horizontal zero-line reference and dragmode="pan".
    """
    colors = [PALETTE["pos"] if s >= 0 else PALETTE["neg"] for s in wf["sharpe"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=wf["period"], y=wf["sharpe"],
        marker_color=colors,
        text=[f"{s:+.2f}" for s in wf["sharpe"]],
        textposition="outside",
        textfont=dict(size=11, color="#c0c0c0"),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "OOS Sharpe: %{y:.2f}<br>"
            "<i>Return-per-unit-of-risk on data the strategy had never seen.<br>"
            ">1.0 = strong · >0.5 = acceptable · <0 = lost money this window.<br>"
            "Consistent positives here = strategy isn't just overfitted.</i>"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_color="#555", line_dash="dot")
    fig.update_layout(**_layout(
        height=height, showlegend=False,
        dragmode="pan",
        title=dict(text="Walk-Forward OOS Sharpe  (3 yr train / 1 yr test)",
                   font=dict(size=13)),
        xaxis=dict(title=None, fixedrange=False),
        yaxis=dict(title="Sharpe ratio", fixedrange=False),
    ))
    return fig


def chart_asset_sharpe(ticker_curves: dict, height: int = 480) -> go.Figure:
    """
    Build a grouped horizontal bar chart comparing per-asset Sharpe ratios
    for the regime strategy vs Buy & Hold.

    For each ticker the annualised Sharpe is computed from its individual
    equity curve.  Tickers are sorted by strategy Sharpe so the best
    performers appear at the top.

    Args:
        ticker_curves: Dict mapping ticker str → DataFrame with columns
                       [regime, buy_hold] indexed by date.
        height:        Chart height in pixels.

    Returns:
        go.Figure with barmode="group" and dragmode="pan".
    """
    rows = []
    for ticker, df in ticker_curves.items():
        for col, label in [("regime", "Strategy"), ("buy_hold", "Buy & Hold")]:
            if col not in df.columns:
                continue
            r   = df[col].pct_change().dropna()
            ann = (1 + r).prod() ** (252 / len(r)) - 1
            vol = r.std() * 252 ** .5
            rows.append(dict(ticker=ticker, method=label,
                             sharpe=ann / vol if vol > 0 else 0))
    mdf    = pd.DataFrame(rows)
    strat  = mdf[mdf.method == "Strategy"].set_index("ticker")["sharpe"]
    bnh    = mdf[mdf.method == "Buy & Hold"].set_index("ticker")["sharpe"]
    order  = strat.sort_values().index.tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=order, x=[bnh.get(t, 0) for t in order], name="Buy & Hold",
        orientation="h", marker_color=PALETTE["buy_hold"], opacity=0.7,
        hovertemplate=(
            "<b>%{y}</b> — Buy & Hold<br>"
            "Sharpe: %{x:.2f}<br>"
            "<i>Return-per-unit-of-risk if you just held this stock forever.<br>"
            ">1.0 good · <0 = lost money on a risk-adjusted basis.</i>"
            "<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        y=order, x=[strat.get(t, 0) for t in order], name="Regime Strategy",
        orientation="h", marker_color=PALETTE["equal_weight"], opacity=0.9,
        hovertemplate=(
            "<b>%{y}</b> — MA Strategy<br>"
            "Sharpe: %{x:.2f}<br>"
            "<i>Return-per-unit-of-risk using the golden-cross signal.<br>"
            "Higher than Buy & Hold = the signal added real value here.</i>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(**_layout(
        height=height, barmode="group",
        dragmode="pan",
        title=dict(text="Sharpe Ratio  ·  Strategy vs Buy & Hold by Asset",
                   font=dict(size=13)),
        xaxis=dict(title="Sharpe ratio", fixedrange=False),
        yaxis=dict(title=None, fixedrange=False),
    ))
    return fig


def chart_macro_overlay(df_port: pd.DataFrame, macro: pd.DataFrame,
                         height: int = 560) -> go.Figure:
    """
    Build a three-panel subplot chart overlaying the portfolio equity curve
    with key macro regime indicators.

    Subplots (shared x-axis):
      Row 1 — Portfolio vs Buy & Hold equity curves (go.Scatter).
      Row 2 — VIX level with fill, and reference lines at 20 (caution)
               and 30 (fear).  VIX z-score > 2.5 triggers a hard signal
               gate in the strategy.
      Row 3 — 10Y–2Y Treasury yield spread.  Negative = inverted curve
               (recession warning, position sizing reduced).  Reference
               line at 0 marks inversion threshold.

    Args:
        df_port: Portfolio equity curve DataFrame (same as load_portfolio_curves()).
        macro:   Macro features DataFrame (same as load_macro()).
        height:  Total chart height in pixels across all three rows.

    Returns:
        go.Figure using make_subplots with shared x-axes and dragmode="pan".
    """
    macro = macro.reindex(df_port.index).ffill()
    fig   = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          vertical_spacing=0.04,
                          row_heights=[0.5, 0.25, 0.25],
                          subplot_titles=("Portfolio vs Buy & Hold",
                                          "VIX (fear gauge)", "10Y–2Y Yield Spread"))
    # Equity
    for col in ["equal_weight", "buy_hold"]:
        if col in df_port.columns:
            fig.add_trace(go.Scatter(
                x=df_port.index, y=df_port[col], name=LABELS[col],
                line=dict(color=PALETTE[col], width=1.6,
                          dash="dot" if col == "buy_hold" else "solid"),
                hovertemplate=(
                    f"<b>{LABELS[col]}</b><br>"
                    "$%{y:,.0f}<br>"
                    "<i>Portfolio value on this date (started at $100k).</i>"
                    "<extra></extra>"
                ),
            ), row=1, col=1)
    # VIX
    if "vix" in macro.columns:
        fig.add_trace(go.Scatter(
            x=macro.index, y=macro["vix"], name="VIX",
            line=dict(color=PALETTE["neg"], width=1),
            fill="tozeroy", fillcolor="rgba(255,85,85,0.08)", showlegend=False,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "VIX: %{y:.1f}<br>"
                "<i>Market fear gauge. <15 = calm, 15-20 = normal,<br>"
                "20-30 = anxious, >30 = fear, >40 = panic.<br>"
                "High VIX = strategy reduces position sizes.</i>"
                "<extra></extra>"
            ),
        ), row=2, col=1)
        for lvl, clr in [(20, "#555"), (30, PALETTE["neg"])]:
            fig.add_hline(y=lvl, line_color=clr, line_dash="dot",
                          line_width=1, row=2, col=1)
    # Yield curve
    if "yield_curve" in macro.columns:
        yc = macro["yield_curve"]
        fig.add_trace(go.Scatter(
            x=macro.index, y=yc, name="10Y–2Y",
            line=dict(color=PALETTE["atr_pca_macro"], width=1.2),
            showlegend=False,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "10Y-2Y Spread: %{y:+.2f}%<br>"
                "<i>Difference between 10-year and 2-year Treasury rates.<br>"
                "Negative (inverted) = recession warning, strategy shrinks positions.<br>"
                "Above 1% (steep) = economic expansion, normal/larger sizing.</i>"
                "<extra></extra>"
            ),
        ), row=3, col=1)
        fig.add_hline(y=0, line_color="#555", line_dash="dot",
                      line_width=1, row=3, col=1)

    fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                      font=dict(color="#c0c0c0", size=11),
                      height=height, showlegend=True, dragmode="pan",
                      legend=dict(bgcolor="#252525", bordercolor="#333", borderwidth=1),
                      margin=dict(l=56, r=20, t=44, b=36))
    for r in range(1, 4):
        fig.update_xaxes(gridcolor="#2a2a2a", fixedrange=False, row=r, col=1)
        fig.update_yaxes(gridcolor="#2a2a2a", fixedrange=False, row=r, col=1)
    return fig


# ── Monthly returns heatmap ───────────────────────────────────────────────────
def chart_monthly_heatmap(ret: pd.Series, title: str = "Equal Weight — Monthly Returns",
                          height: int = 340) -> go.Figure:
    """
    Build a calendar heatmap of monthly returns (year × month grid).

    The colour scale is centred at zero (zmid=0) and clipped at ±10% so
    that small monthly moves still show visible colour — extreme outliers
    don't wash out the scale.  Cell text shows the exact % return.

    Args:
        ret:    Daily returns series (fractional).  Resampled internally
                to month-end using compounded multiplication.
        title:  Chart title string.
        height: Chart height in pixels.

    Returns:
        go.Figure using go.Heatmap with year on y-axis (reversed so most
        recent year is at the top) and month on x-axis (top).
    """
    monthly = (1 + ret).resample("ME").prod() - 1
    df_m = monthly.to_frame("ret")
    df_m["year"]  = df_m.index.year
    df_m["month"] = df_m.index.month

    pivot = df_m.pivot(index="year", columns="month", values="ret")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_labels[c - 1] for c in pivot.columns]

    text = pivot.map(lambda v: f"{v*100:.1f}%" if not pd.isna(v) else "")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        text=text.values,
        texttemplate="%{text}",
        colorscale=[[0, "#ff5555"], [0.45, "#1c1c1c"], [0.55, "#1c1c1c"], [1, "#50fa7b"]],
        zmid=0, zmin=-0.10, zmax=0.10,
        showscale=True,
        colorbar=dict(tickformat=".0%", len=0.8),
        hovertemplate=(
            "<b>%{y} %{x}</b><br>"
            "Return: %{text}<br>"
            "<i>Strategy return for this single calendar month.<br>"
            "Green = gain, Red = loss. Darker = bigger move.<br>"
            "Consistent green rows = the strategy performs well year-round.</i>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(**_layout(
        height=height,
        dragmode="pan",
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title=None, side="top", fixedrange=False),
        yaxis=dict(title=None, autorange="reversed", fixedrange=False),
        margin=dict(l=56, r=80, t=56, b=16),
    ))
    return fig


# ── MA spread bar chart (live tab) ────────────────────────────────────────────
def chart_ma_spread(live_df: pd.DataFrame, height: int = 400) -> go.Figure:
    """
    Build a horizontal bar chart showing the MA spread for every ticker in
    the live universe.

    Each bar represents how far the fast MA is above or below the slow MA
    as a percentage.  Positive (green) = golden cross = LONG signal.
    Negative (red) = death cross = strategy holds cash for that ticker.
    Tickers are sorted ascending so the weakest signals appear at the top.

    Args:
        live_df: DataFrame returned by get_live_signals(), must contain
                 columns [Ticker, MA Spread %].
        height:  Chart height in pixels.

    Returns:
        go.Figure with a vertical zero-line, showlegend=False, dragmode="pan".
    """
    df = live_df.sort_values("MA Spread %")
    colors = [PALETTE["pos"] if v >= 0 else PALETTE["neg"] for v in df["MA Spread %"]]
    fig = go.Figure(go.Bar(
        y=df["Ticker"], x=df["MA Spread %"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in df["MA Spread %"]],
        textposition="outside",
        textfont=dict(size=10, color="#c0c0c0"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "MA Spread: %{x:+.2f}%<br>"
            "<i>How far the fast moving average is above/below the slow one.<br>"
            "Positive (green) = golden cross — fast MA crossed above slow MA = LONG signal.<br>"
            "Negative (red) = death cross — price trending down = strategy sits in cash.</i>"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_color="#555", line_width=1.5)
    fig.update_layout(**_layout(
        height=height, showlegend=False,
        dragmode="pan",
        title=dict(text="MA Spread — Fast vs Slow MA  (positive = golden cross / LONG)",
                   font=dict(size=13)),
        xaxis=dict(title="Spread (%)", fixedrange=False),
        yaxis=dict(title=None, fixedrange=False),
    ))
    return fig


# ── TradingView-style portfolio chart ────────────────────────────────────────
def chart_paper_portfolio(history_df: pd.DataFrame,
                          intraday_df: pd.DataFrame,
                          trades_df: pd.DataFrame,
                          height: int = 440,
                          now: datetime = None,
                          entry_value: float = None,
                          x_range: list = None,
                          spy_curve=None) -> go.Figure:
    """
    Build the TradingView-style paper portfolio equity curve figure.

    Combines three traces:
      1. Historical daily closes (go.Scatter, blue line) — from history.csv,
         excluding today (which is shown at higher resolution by the candles)
      2. Today's 5-minute candles (go.Candlestick, green/red) — from yfinance
      3. Post-close flat line (go.Scatter, blue dotted) — extends from the last
         candle to the current time so the chart never looks "cut off"
      4. SPY benchmark (go.Scatter, purple dotted) — normalised to the same
         starting value as the portfolio, showing relative performance
      5. Buy/sell markers (go.Scatter with triangle symbols) on historical dates
      6. Reference lines at initial capital ($100k) and entry value

    Args:
        history_df:    Daily history DataFrame from load_history().
        intraday_df:   Today's OHLC DataFrame from get_intraday_curve().
        trades_df:     Trade log DataFrame from load_trades().
        height:        Chart height in pixels.
        now:           Current ET-naive Timestamp (used for the "Now" line
                       and x-axis right edge computation).
        entry_value:   Previous day's portfolio value (used for the entry
                       reference line).
        x_range:       [left, right] x-axis range strings in ISO format.
                       Passed from session_state so it stays constant across
                       5-second refreshes, allowing uirevision to preserve zoom.
        spy_curve:     pd.Series of SPY benchmark values indexed by ET datetime.

    Returns:
        go.Figure ready for st.plotly_chart().

    Key Plotly settings:
        uirevision="paper_portfolio" — preserves user zoom across fragment refreshes
        dragmode="pan"               — drag to pan (not box-zoom, which is confusing)
        scrollZoom=True              — mouse wheel to zoom
    """
    fig = go.Figure()

    # Exclude today from daily history — intraday trace covers today
    if not history_df.empty:
        _today_str = pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d')
        history_df = history_df[history_df["date"].astype(str) < _today_str]

    # ── Historical equity curve ───────────────────────────────────────────────
    if not history_df.empty:
        fig.add_trace(go.Scatter(
            x=history_df["date"],
            y=history_df["portfolio_value"],
            name="Portfolio (daily)",
            line=dict(color="#4a9eff", width=2.2),
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "End-of-day value: <b>$%{y:,.0f}</b><br>"
                "<i>Recorded at 4:45 PM ET after the market close.<br>"
                "Each point = one trading day of holding the portfolio.</i>"
                "<extra></extra>"
            ),
        ))

    # ── Today's intraday candlesticks ─────────────────────────────────────────
    if not intraday_df.empty and {"open","high","low","close"}.issubset(intraday_df.columns):
        fig.add_trace(go.Candlestick(
            x=intraday_df.index,
            open=intraday_df["open"],
            high=intraday_df["high"],
            low=intraday_df["low"],
            close=intraday_df["close"],
            name="Today (live)",
            increasing=dict(line=dict(color="#50fa7b", width=1),
                            fillcolor="rgba(80,250,123,0.7)"),
            decreasing=dict(line=dict(color="#ff5555", width=1),
                            fillcolor="rgba(255,85,85,0.7)"),
        ))
        # After the last real candle, draw a flat dotted line to current time so
        # the chart never looks "cut off" — market closed = straight horizontal line.
        if now is not None and now > intraday_df.index[-1]:
            _lv = float(intraday_df["close"].iloc[-1])
            fig.add_trace(go.Scatter(
                x=[intraday_df.index[-1], now],
                y=[_lv, _lv],
                mode="lines",
                line=dict(color="#4a9eff", width=1.5, dash="dot"),
                showlegend=False,
                hovertemplate="<b>Portfolio</b> (market closed)<br>%{x|%H:%M} $%{y:,.0f}<extra></extra>",
            ))

    # ── S&P 500 benchmark (subtle dashed line, normalized to same start) ──────
    if spy_curve is not None and not spy_curve.empty:
        # Extend SPY benchmark to current time (flat after close)
        _spy_x = list(spy_curve.index)
        _spy_y = list(spy_curve.values)
        if now is not None and now > pd.Timestamp(_spy_x[-1]):
            _spy_x.append(now)
            _spy_y.append(_spy_y[-1])
        fig.add_trace(go.Scatter(
            x=_spy_x,
            y=_spy_y,
            name="S&P 500",
            line=dict(color="#7878aa", width=1.2, dash="dot"),
            opacity=0.6,
            hovertemplate=(
                "<b>S&P 500 Benchmark</b><br>"
                "%{x|%H:%M}<br>"
                "Equivalent value: <b>$%{y:,.0f}</b><br>"
                "<i>$100k in SPY — normalized to the same starting value. "
                "Drift shows relative over/under-performance vs the index.</i>"
                "<extra></extra>"
            ),
        ))

    # ── Trade markers ─────────────────────────────────────────────────────────
    if not trades_df.empty and not history_df.empty:
        # Map trade date → portfolio_value on that day for marker y-position
        hist_val = history_df.set_index("date")["portfolio_value"]

        buys  = trades_df[trades_df["action"] == "BUY"].copy()
        sells = trades_df[trades_df["action"] == "SELL"].copy()

        def _marker_y(dates: pd.Series) -> list:
            """
            Look up portfolio value for each trade date to position the marker
            at the correct y-coordinate on the equity curve.

            Falls back to the last known portfolio value for any trade date
            that is not found in history (e.g. weekend trades or pre-history).
            The isinstance(v, pd.Series) guard handles duplicate index entries
            that pandas returns as a Series rather than a scalar.
            """
            last = float(hist_val.iloc[-1])
            out  = []
            for d in dates:
                try:
                    v = hist_val[d]
                    out.append(float(v.iloc[0]) if isinstance(v, pd.Series) else float(v))
                except KeyError:
                    out.append(last)
            return out

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["date"], y=_marker_y(buys["date"]),
                mode="markers+text",
                name="Buy",
                marker=dict(symbol="triangle-up", size=12,
                            color=PALETTE["pos"], line=dict(color="#1c1c1c", width=1)),
                text=buys["ticker"].tolist(),
                textposition="top center",
                textfont=dict(size=9, color=PALETTE["pos"]),
                hovertemplate=(
                    "<b>BUY %{text}</b><br>"
                    "%{x|%b %d, %Y}<br>"
                    "Entry price: $%{y:,.2f}<br>"
                    "<i>Strategy opened a long position — MA50 crossed above MA200 (golden cross).</i>"
                    "<extra></extra>"
                ),
            ))

        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["date"], y=_marker_y(sells["date"]),
                mode="markers+text",
                name="Sell",
                marker=dict(symbol="triangle-down", size=12,
                            color=PALETTE["neg"], line=dict(color="#1c1c1c", width=1)),
                text=sells["ticker"].tolist(),
                textposition="bottom center",
                textfont=dict(size=9, color=PALETTE["neg"]),
                hovertemplate=(
                    "<b>SELL %{text}</b><br>"
                    "%{x|%b %d, %Y}<br>"
                    "Exit price: $%{y:,.2f}<br>"
                    "<i>Strategy closed the position — MA50 crossed below MA200 (death cross) or trailing stop hit.</i>"
                    "<extra></extra>"
                ),
            ))

    # Reference line at initial capital ($100k)
    fig.add_hline(
        y=PT_INITIAL_CAPITAL, line_color="#444", line_dash="dot", line_width=1,
        annotation_text=f"  Start ${PT_INITIAL_CAPITAL/1000:.0f}k",
        annotation_font_color="#555",
    )
    # Reference line at today's actual entry value (after commissions / market gaps)
    # Shows WHY the portfolio opened below $100k — commission drag + gap risk
    if entry_value and abs(entry_value - PT_INITIAL_CAPITAL) > 1:
        fig.add_hline(
            y=entry_value, line_color="#666", line_dash="dash", line_width=1,
            annotation_text=f"  Entry ${entry_value:,.0f}",
            annotation_font_color="#888",
        )

    # "Now" vertical line — use add_shape + add_annotation with ISO string
    # so the x coordinate is in the same system as the intraday data (naive ET strings)
    if now is not None:
        now_str = now.strftime('%Y-%m-%dT%H:%M:%S')
        fig.add_shape(
            type="line",
            x0=now_str, x1=now_str,
            y0=0, y1=1, yref="paper",
            line=dict(color="#ffb86c", dash="dot", width=1.2),
        )
        fig.add_annotation(
            x=now_str, y=0.99, yref="paper",
            text=f"  {now.strftime('%H:%M')}",
            showarrow=False,
            font=dict(color="#ffb86c", size=11),
            xanchor="left",
        )

    if x_range is None:
        _now = now or pd.Timestamp.now(tz='America/New_York').replace(tzinfo=None)
        _today = _now.strftime('%Y-%m-%d')
        _right = max(_now, pd.Timestamp(_today + " 16:05:00"))
        x_range = [f"{_today}T09:25:00", _right.strftime('%Y-%m-%dT%H:%M:00')]

    fig.update_layout(**_layout(
        height=height,
        # uirevision = constant string → Plotly.js preserves zoom/pan across data
        # updates (same as TradingView live feed — data refreshes, viewport stays)
        uirevision="paper_portfolio",
        title=dict(text="Paper Portfolio — Equity Curve  (live via Yahoo Finance)",
                   font=dict(size=13)),
        dragmode="pan",
        yaxis=dict(
            title=dict(text="Value ($)", standoff=20),
            tickprefix="$", tickformat=",.0f",
            autorange=True,
            fixedrange=False,
        ),
        margin=dict(l=80),
        xaxis=dict(
            title=None,
            type="date",
            tickformat="%b %d\n%H:%M",
            range=x_range,
            fixedrange=False,
            rangeslider=dict(visible=False),
            rangeselector=dict(
                bgcolor="#252525",
                activecolor="#3a3a3a",
                bordercolor="#444",
                font=dict(color="#c0c0c0", size=10),
                buttons=[
                    dict(count=2,  label="2H",  step="hour", stepmode="backward"),
                    dict(count=1,  label="Today", step="day", stepmode="todate"),
                    dict(count=7,  label="1W",  step="day",  stepmode="backward"),
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
            ),
        ),
    ))
    return fig
