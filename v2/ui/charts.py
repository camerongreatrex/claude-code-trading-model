# v2/ui/charts.py
# Chart builders and metrics helpers for the v2 macro regime rotation dashboard.
# Mirrors v1/ui/charts.py styling.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from v2.ui.styles import _layout, PALETTE, LABELS, get_color, get_label

PAPER_INITIAL_CAPITAL = 100_000


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(ret: pd.Series) -> dict:
    """Compute standard annualised risk/return metrics from daily returns."""
    ret = ret.dropna()
    if ret.empty:
        return {}
    cum    = (1 + ret).cumprod()
    total  = cum.iloc[-1] - 1
    n_y    = len(ret) / 252
    ann_r  = (1 + total) ** (1 / n_y) - 1 if n_y > 0 else 0
    vol    = ret.std() * 252 ** 0.5
    sharpe = ann_r / vol if vol > 0 else 0
    peak   = cum.cummax()
    max_dd = ((cum - peak) / peak).min()
    calmar = ann_r / abs(max_dd) if max_dd != 0 else 0
    active = ret[ret != 0]
    wr     = (active > 0).sum() / len(active) if len(active) > 0 else 0
    var_95 = float(-np.percentile(ret, 5))
    gross_win = ret[ret > 0].sum()
    gross_loss = -ret[ret < 0].sum()
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    return dict(ann_r=ann_r, vol=vol, sharpe=sharpe, max_dd=max_dd,
                calmar=calmar, win_rate=wr, total=total, var_95=var_95,
                profit_factor=profit_factor)


def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-curve summary table (Ann. Return, Vol, Sharpe, etc)."""
    rows = []
    for col in df.columns:
        ret = df[col].pct_change().dropna()
        m = metrics(ret)
        if not m:
            continue
        pf = m.get("profit_factor", float("inf"))
        rows.append({
            "Method"         : get_label(col),
            "Ann. Return"    : f"{m['ann_r']*100:+.1f}%",
            "Volatility"     : f"{m['vol']*100:.1f}%",
            "Sharpe"         : f"{m['sharpe']:.2f}",
            "Max DD"         : f"{m['max_dd']*100:.1f}%",
            "Calmar"         : f"{m['calmar']:.2f}",
            "Win Rate"       : f"{m['win_rate']*100:.0f}%",
            "Profit Factor"  : f"{pf:.2f}" if pf < 100 else "∞",
            "VaR 95%"        : f"{m['var_95']*100:.2f}%",
        })
    return pd.DataFrame(rows)


# ── Equity curves ────────────────────────────────────────────────────────────
def chart_equity(df: pd.DataFrame, columns: list = None, height: int = 420) -> go.Figure:
    """Multi-line equity curves normalized to start at 1.0."""
    if columns is None:
        columns = [c for c in df.columns if c in df]

    fig = go.Figure()
    for col in columns:
        if col not in df.columns:
            continue
        dash = "dot" if col == "spy" else "solid"
        color = get_color(col)
        label = get_label(col)
        # Normalize to start at $100k for consistent $ view
        series = df[col] * 100_000
        fig.add_trace(go.Scatter(
            x=df.index, y=series.values, name=label,
            line=dict(color=color, width=1.9, dash=dash),
            hovertemplate=f"<b>{label}</b><br>$%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(**_layout(
        height=height, dragmode="pan", uirevision="v2_equity",
        title=dict(text="Portfolio Growth — $100k start", font=dict(size=13)),
        yaxis=dict(title="Value ($)", fixedrange=False),
        xaxis=dict(fixedrange=False),
    ))
    return fig


def chart_equity_risk_adjusted(df: pd.DataFrame, columns: list = None,
                                height: int = 420, target_vol: float = 0.10) -> go.Figure:
    """Equity curves scaled so every strategy has `target_vol` annualized vol."""
    if columns is None:
        columns = list(df.columns)
    fig = go.Figure()
    for col in columns:
        if col not in df.columns:
            continue
        ret = df[col].pct_change().dropna()
        if ret.empty:
            continue
        realized = ret.std() * np.sqrt(252)
        if realized <= 0.001:
            continue
        scale = target_vol / realized
        scaled_ret = ret * scale
        eq = 100_000 * (1 + scaled_ret).cumprod()
        dash = "dot" if col == "spy" else "solid"
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name=f"{get_label(col)} ({realized*100:.0f}%→{int(target_vol*100)}%)",
            line=dict(color=get_color(col), width=1.9, dash=dash),
            hovertemplate=(
                f"<b>{get_label(col)}</b><br>"
                f"$%{{y:,.0f}} (scaled to {int(target_vol*100)}% vol)<br>"
                f"Realized: {realized*100:.1f}% · Scale: {scale:.2f}×<extra></extra>"
            ),
        ))
    fig.add_hline(y=100_000, line_color="#444", line_dash="dot", line_width=1)
    fig.update_layout(**_layout(
        height=height, dragmode="pan", uirevision="v2_equity_risk",
        title=dict(text=f"Risk-Adjusted Growth — all normalized to {int(target_vol*100)}% vol",
                   font=dict(size=13)),
        yaxis=dict(title="Value ($)", fixedrange=False),
        xaxis=dict(fixedrange=False),
    ))
    return fig


def chart_drawdown(df: pd.DataFrame, columns: list = None, height: int = 420) -> go.Figure:
    """Drawdown as filled area per curve."""
    if columns is None:
        columns = list(df.columns)
    alphas = [0.20, 0.14, 0.10, 0.08]
    fig = go.Figure()
    for idx, col in enumerate(columns):
        if col not in df.columns:
            continue
        p = df[col]
        dd = (p - p.cummax()) / p.cummax() * 100
        color = get_color(col)
        r, g, b = (int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        a = alphas[min(idx, len(alphas) - 1)]
        fig.add_trace(go.Scatter(
            x=df.index, y=dd, name=get_label(col),
            line=dict(color=color, width=1.4),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},{a})",
            hovertemplate=f"<b>{get_label(col)}</b><br>%{{y:.1f}}% from peak<extra></extra>",
        ))
    fig.update_layout(**_layout(
        height=height, dragmode="pan", uirevision="v2_drawdown",
        title=dict(text="Drawdown (%)", font=dict(size=13)),
        yaxis=dict(title="DD (%)", fixedrange=False),
        xaxis=dict(fixedrange=False),
    ))
    return fig


# ── Monte Carlo ──────────────────────────────────────────────────────────────
def chart_monte_carlo(eq_curves: np.ndarray, actual: np.ndarray,
                      height: int = 420) -> go.Figure:
    """Monte Carlo fan chart."""
    n = eq_curves.shape[1]
    xs = np.arange(n)
    p5, p25, p50, p75, p95 = (np.percentile(eq_curves, q, axis=0)
                              for q in (5, 25, 50, 75, 95))
    fig = go.Figure()

    sample_idx = np.random.default_rng(0).choice(
        len(eq_curves), size=min(80, len(eq_curves)), replace=False)
    xs_m, ys_m = [], []
    for i in sample_idx:
        xs_m.extend(xs.tolist()); xs_m.append(None)
        ys_m.extend(eq_curves[i].tolist()); ys_m.append(None)
    fig.add_trace(go.Scatter(
        x=xs_m, y=ys_m,
        line=dict(color="rgba(180,180,180,0.05)", width=0.8),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself", fillcolor="rgba(74,158,255,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="5–95th pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself", fillcolor="rgba(80,250,123,0.11)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75th pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=p50, name="Median",
        line=dict(color="#e0e0e0", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=xs[:len(actual)], y=actual, name="Historical",
        line=dict(color=PALETTE["strategy_net"], width=2.2),
    ))
    fig.update_layout(**_layout(
        height=height, dragmode="pan", uirevision="v2_mc",
        title=dict(text=f"Monte Carlo Bootstrap · {len(eq_curves):,} paths · 21-day blocks",
                   font=dict(size=13)),
        xaxis=dict(title="Trading days", fixedrange=False),
        yaxis=dict(title="Growth of $1", fixedrange=False),
    ))
    return fig


def chart_mc_histogram(eq_curves: np.ndarray, height: int = 280) -> go.Figure:
    finals = eq_curves[:, -1]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=(finals - 1) * 100, nbinsx=50,
        marker_color=PALETTE["strategy_net"], opacity=0.82, name="Terminal return",
    ))
    fig.add_vline(x=(np.median(finals) - 1) * 100, line_color="#e0e0e0",
                  line_dash="dash", annotation_text="Median",
                  annotation_font_color="#e0e0e0")
    fig.add_vline(x=(np.percentile(finals, 5) - 1) * 100, line_color=PALETTE["neg"],
                  line_dash="dot", annotation_text="5th pct",
                  annotation_font_color=PALETTE["neg"])
    fig.update_layout(**_layout(
        height=height, dragmode="pan", uirevision="v2_mc_hist",
        title=dict(text="Distribution of final-day total return", font=dict(size=13)),
        xaxis=dict(title="Total return (%)", fixedrange=False),
        yaxis=dict(title="Paths", fixedrange=False),
        showlegend=False,
    ))
    return fig


# ── Walk-forward ─────────────────────────────────────────────────────────────
def chart_walk_forward(wf: pd.DataFrame, height: int = 360) -> go.Figure:
    """Bar chart of OOS Sharpe by walk-forward window."""
    if wf.empty:
        return go.Figure()

    sharpe_col = "oos_sharpe" if "oos_sharpe" in wf.columns else "sharpe"
    label_col  = "window" if "window" in wf.columns else "period"

    s = wf[sharpe_col].astype(float)
    colors = [PALETTE["pos"] if v > 0.2 else PALETTE["neg"] for v in s]
    x_labels = [f"WF{int(w)}" if str(w).isdigit() else str(w) for w in wf[label_col]]

    fig = go.Figure(go.Bar(
        x=x_labels, y=s,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>OOS Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0.2, line_color="#555", line_dash="dot",
                  annotation_text="Gate (0.20)", annotation_font_color="#777")
    fig.update_layout(**_layout(
        height=height, dragmode="pan", uirevision="v2_wf",
        title=dict(text="Walk-Forward OOS Sharpe", font=dict(size=13)),
        yaxis=dict(title="Sharpe", fixedrange=False),
        xaxis=dict(fixedrange=False),
        showlegend=False,
    ))
    return fig


# ── Monthly heatmap ──────────────────────────────────────────────────────────
def chart_monthly_heatmap(ret: pd.Series, height: int = 420) -> go.Figure:
    ret = ret.dropna()
    if ret.empty:
        return go.Figure()
    monthly = (1 + ret).resample("M").prod() - 1
    table = monthly.groupby([monthly.index.year, monthly.index.month]).sum().unstack(fill_value=np.nan)
    table = table.sort_index()

    fig = go.Figure(go.Heatmap(
        z=table.values * 100,
        x=[pd.Timestamp(2000, m, 1).strftime("%b") for m in table.columns],
        y=table.index.astype(str),
        colorscale=[[0, "#ff5555"], [0.5, "#1c1c1c"], [1, "#50fa7b"]],
        zmid=0,
        colorbar=dict(title="%", tickformat=".0f"),
        text=table.values * 100,
        texttemplate="%{text:.1f}",
        textfont={"size": 9, "color": "#e0e0e0"},
        hovertemplate="<b>%{y} %{x}</b><br>%{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(**_layout(
        height=height, uirevision="v2_monthly",
        title=dict(text="Monthly returns (%)", font=dict(size=13)),
        yaxis=dict(autorange="reversed"),
    ))
    return fig


# ── Regime probabilities stacked ─────────────────────────────────────────────
def chart_regime_stack(probs: pd.DataFrame, height: int = 360) -> go.Figure:
    order = ["expansion", "recovery", "late_cycle", "slowdown", "stagflation", "recession"]
    present = [c for c in order if c in probs.columns]
    fig = go.Figure()
    for col in present:
        fig.add_trace(go.Scatter(
            x=probs.index, y=probs[col] * 100,
            name=get_label(col),
            line=dict(width=0, color=get_color(col)),
            stackgroup="one",
            hovertemplate=f"<b>{get_label(col)}</b><br>%{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**_layout(
        height=height, dragmode="pan", uirevision="v2_regime_stack",
        title=dict(text="Regime probabilities (stacked)", font=dict(size=13)),
        yaxis=dict(title="Probability (%)", fixedrange=False, range=[0, 100]),
        xaxis=dict(fixedrange=False),
    ))
    return fig


def chart_regime_label_strip(labels: pd.Series, height: int = 140) -> go.Figure:
    regime_map = {0: "Expansion", 1: "Slowdown", 2: "Recession",
                  3: "Recovery", 4: "Stagflation", 5: "Late Cycle"}
    color_map = {0: "#50fa7b", 1: "#ffb86c", 2: "#ff5555",
                 3: "#61afef", 4: "#bd93f9", 5: "#e5c07b"}
    fig = go.Figure()
    for rid, name in regime_map.items():
        mask = labels == rid
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=labels.index[mask], y=np.ones(mask.sum()),
            mode="markers", name=name,
            marker=dict(color=color_map[rid], size=4, symbol="square"),
            hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<extra></extra>",
        ))
    fig.update_layout(**_layout(
        height=height, uirevision="v2_regime_strip",
        title=dict(text="Dominant regime timeline", font=dict(size=13)),
        yaxis=dict(visible=False, range=[0.5, 1.5]),
        xaxis=dict(fixedrange=False),
        showlegend=True,
    ))
    return fig


# ── Current allocation bar chart ─────────────────────────────────────────────
def chart_current_allocation(weights: pd.Series, asset_class_map: dict,
                              height: int = 360) -> go.Figure:
    """Latest weights as horizontal bar chart, coloured by asset class."""
    w = weights[weights > 0.005].sort_values(ascending=True)
    colors = [PALETTE.get(asset_class_map.get(t, "other"), "#888") for t in w.index]
    fig = go.Figure(go.Bar(
        x=w.values * 100, y=w.index, orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(**_layout(
        height=height, uirevision="v2_alloc",
        title=dict(text="Current allocation", font=dict(size=13)),
        xaxis=dict(title="Weight (%)", fixedrange=False),
        yaxis=dict(fixedrange=False),
        showlegend=False,
    ))
    return fig


# ── Macro overlay ────────────────────────────────────────────────────────────
def chart_macro_overlay(eq_df: pd.DataFrame, market: pd.DataFrame,
                         height: int = 540) -> go.Figure:
    """Equity curve with VIX and HY OAS subplots stacked below."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.225, 0.225],
        vertical_spacing=0.04,
        subplot_titles=("Strategy vs SPY", "VIX", "HY OAS (credit spread)"),
    )

    if "strategy_net" in eq_df.columns:
        fig.add_trace(go.Scatter(
            x=eq_df.index, y=eq_df["strategy_net"].values,
            name="Strategy (Net)",
            line=dict(color=PALETTE["strategy_net"], width=1.9),
        ), row=1, col=1)
    if "spy" in eq_df.columns:
        fig.add_trace(go.Scatter(
            x=eq_df.index, y=eq_df["spy"].values,
            name="SPY", line=dict(color=PALETTE["spy"], width=1.6, dash="dot"),
        ), row=1, col=1)

    if "vix" in market.columns:
        fig.add_trace(go.Scatter(
            x=market.index, y=market["vix"], name="VIX",
            line=dict(color="#ff79c6", width=1.4), showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=20, line_color="#555", line_dash="dot", row=2, col=1)
        fig.add_hline(y=30, line_color="#ff5555", line_dash="dot", row=2, col=1)

    if "hy_oas" in market.columns:
        fig.add_trace(go.Scatter(
            x=market.index, y=market["hy_oas"], name="HY OAS",
            line=dict(color="#e5c07b", width=1.4), showlegend=False,
        ), row=3, col=1)

    fig.update_layout(**_layout(
        height=height, dragmode="pan", uirevision="v2_macro",
        title=dict(text="Macro overlay", font=dict(size=13)),
    ))
    for row in (1, 2, 3):
        fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a", row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a", row=row, col=1)
    return fig


# ── Regime attribution table ─────────────────────────────────────────────────
def chart_regime_bars(attribution: dict, height: int = 340) -> go.Figure:
    """Per-regime Sharpe bars."""
    if not attribution:
        return go.Figure()
    names = list(attribution.keys())
    sharpes = [attribution[n]["sharpe"] for n in names]
    colors = [PALETTE["pos"] if s > 0 else PALETTE["neg"] for s in sharpes]
    fig = go.Figure(go.Bar(
        x=names, y=sharpes, marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        height=height, uirevision="v2_regime_bars",
        title=dict(text="Per-regime net Sharpe", font=dict(size=13)),
        yaxis=dict(title="Sharpe"),
        showlegend=False,
    ))
    return fig


# ── Paper portfolio chart (V1-style, adapted for V2) ──────────────────────────
def chart_paper_portfolio(
    history_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    spy_curve: pd.Series | None = None,
    entry_value: float | None = None,
    height: int = 460,
) -> go.Figure:
    """
    V1-style paper portfolio equity curve for V2. Shows:
      - Daily NAV as candlesticks (open/close bracket each day)
      - SPY benchmark normalised to the same starting NAV (dotted purple)
      - BUY / SELL markers (triangles) on trade dates
      - Reference lines at $100k start and optional entry value
    """
    fig = go.Figure()

    # ── Daily NAV candlesticks ──────────────────────────────────────────────
    if not history_df.empty and "portfolio_value" in history_df.columns:
        hist = history_df.copy()
        hist["date"] = pd.to_datetime(hist["date"])
        # Drop market-holiday duplicates
        hist = hist[hist["portfolio_value"].ne(hist["portfolio_value"].shift(1))].reset_index(drop=True)
        if not hist.empty:
            closes = hist["portfolio_value"].values
            opens = np.concatenate([[closes[0]], closes[:-1]])
            highs = np.maximum(opens, closes)
            lows = np.minimum(opens, closes)
            fig.add_trace(go.Candlestick(
                x=hist["date"],
                open=opens, high=highs, low=lows, close=closes,
                name="NAV (daily)",
                increasing=dict(line=dict(color="#50fa7b", width=1.2),
                                fillcolor="rgba(80,250,123,0.7)"),
                decreasing=dict(line=dict(color="#ff5555", width=1.2),
                                fillcolor="rgba(255,85,85,0.7)"),
            ))

    # ── SPY benchmark overlay (normalised) ──────────────────────────────────
    if spy_curve is not None and not spy_curve.empty and not history_df.empty:
        start_nav = float(history_df["portfolio_value"].iloc[0])
        spy_aligned = spy_curve.reindex(pd.to_datetime(history_df["date"]), method="ffill")
        if len(spy_aligned.dropna()) > 0:
            spy_norm = spy_aligned / spy_aligned.dropna().iloc[0] * start_nav
            fig.add_trace(go.Scatter(
                x=history_df["date"], y=spy_norm.values,
                name="SPY (normalised)",
                line=dict(color=PALETTE["spy"], width=1.6, dash="dot"),
                hovertemplate="<b>SPY</b> %{x|%b %d}<br>$%{y:,.0f}<extra></extra>",
            ))

    # ── Trade markers ───────────────────────────────────────────────────────
    if not trades_df.empty and not history_df.empty:
        t = trades_df.copy()
        t["date"] = pd.to_datetime(t["date"])
        hist = history_df.copy()
        hist["date"] = pd.to_datetime(hist["date"])
        hist_val = hist.set_index("date")["portfolio_value"]
        last_pv = float(hist_val.iloc[-1])

        def _marker_rows(subset: pd.DataFrame) -> pd.DataFrame:
            g = (subset.groupby(subset["date"].dt.normalize())["ticker"]
                 .apply(lambda x: ", ".join(sorted(set(x))))
                 .reset_index())
            g.columns = ["date", "tickers"]
            g["y"] = g["date"].map(lambda d: float(hist_val.get(d, last_pv)))
            g["n"] = g["tickers"].str.split(",").str.len()
            return g

        buys = t[t["action"] == "BUY"]
        sells = t[t["action"] == "SELL"]
        if not buys.empty:
            bg = _marker_rows(buys)
            fig.add_trace(go.Scatter(
                x=bg["date"], y=bg["y"],
                mode="markers+text", name="Buy",
                marker=dict(symbol="triangle-up", size=13,
                            color=PALETTE["pos"],
                            line=dict(color="#1c1c1c", width=1)),
                text=bg["n"].apply(lambda n: f"+{n}"),
                textposition="top center",
                textfont=dict(size=9, color=PALETTE["pos"]),
                customdata=bg["tickers"].tolist(),
                hovertemplate="<b>BUY — %{customdata}</b><br>%{x|%b %d, %Y}<extra></extra>",
            ))
        if not sells.empty:
            sg = _marker_rows(sells)
            fig.add_trace(go.Scatter(
                x=sg["date"], y=sg["y"],
                mode="markers+text", name="Sell",
                marker=dict(symbol="triangle-down", size=13,
                            color=PALETTE["neg"],
                            line=dict(color="#1c1c1c", width=1)),
                text=sg["n"].apply(lambda n: f"−{n}"),
                textposition="bottom center",
                textfont=dict(size=9, color=PALETTE["neg"]),
                customdata=sg["tickers"].tolist(),
                hovertemplate="<b>SELL — %{customdata}</b><br>%{x|%b %d, %Y}<extra></extra>",
            ))

    fig.add_hline(
        y=PAPER_INITIAL_CAPITAL, line_color="#444", line_dash="dot", line_width=1,
        annotation_text=f"  Start ${PAPER_INITIAL_CAPITAL/1000:.0f}k",
        annotation_font_color="#555",
    )
    if entry_value and abs(entry_value - PAPER_INITIAL_CAPITAL) > 1:
        fig.add_hline(
            y=entry_value, line_color="#666", line_dash="dash", line_width=1,
            annotation_text=f"  Entry ${entry_value:,.0f}",
            annotation_font_color="#888",
        )

    fig.update_layout(**_layout(
        height=height, uirevision="v2_paper_portfolio",
        title=dict(text="Paper Portfolio — Equity Curve vs SPY", font=dict(size=13)),
        dragmode="pan",
        yaxis=dict(title=dict(text="Value ($)", standoff=20),
                   tickprefix="$", tickformat=",.0f",
                   autorange=True, fixedrange=False),
        xaxis=dict(type="date", tickformat="%b %d\n%Y",
                   rangeslider=dict(visible=False),
                   rangeselector=dict(
                       bgcolor="#252525", activecolor="#3a3a3a",
                       bordercolor="#444",
                       font=dict(color="#c0c0c0", size=10),
                       buttons=[
                           dict(count=7, label="1W", step="day", stepmode="backward"),
                           dict(count=1, label="1M", step="month", stepmode="backward"),
                           dict(count=3, label="3M", step="month", stepmode="backward"),
                           dict(step="all", label="All"),
                       ],
                   )),
    ))
    return fig


# ── Regime transition signals ────────────────────────────────────────────────
def chart_regime_transitions(labels: pd.Series, lookback_days: int = 252,
                              height: int = 320) -> go.Figure:
    """Regime label strip zoomed to the last N days with transition markers."""
    from v2.ui.styles import PALETTE

    regime_map = {0: "expansion", 1: "slowdown", 2: "recession",
                  3: "recovery", 4: "stagflation", 5: "late_cycle"}
    recent = labels.tail(lookback_days)
    if recent.empty:
        return go.Figure()

    fig = go.Figure()
    # Shaded background bands per regime
    changes = recent.ne(recent.shift()).cumsum()
    for seg_id, seg in recent.groupby(changes):
        rname = regime_map.get(int(seg.iloc[0]), "expansion")
        color = PALETTE.get(rname, "#888")
        fig.add_shape(
            type="rect",
            x0=seg.index[0], x1=seg.index[-1],
            y0=0, y1=1, yref="paper",
            fillcolor=color, opacity=0.25, line=dict(width=0),
            layer="below",
        )

    # Transition markers
    transitions = recent[recent.ne(recent.shift()).fillna(False)]
    if not transitions.empty:
        labels_text = [regime_map.get(int(v), "?").replace("_", " ").title()
                       for v in transitions.values]
        fig.add_trace(go.Scatter(
            x=transitions.index, y=[0.5] * len(transitions),
            mode="markers+text",
            marker=dict(size=10, color="#ffffff",
                        line=dict(color="#1c1c1c", width=1)),
            text=labels_text, textposition="top center",
            textfont=dict(size=10, color="#c0c0c0"),
            name="Regime change",
            hovertemplate="<b>%{text}</b><br>%{x|%Y-%m-%d}<extra></extra>",
        ))

    fig.update_layout(**_layout(
        height=height, uirevision="v2_regime_transitions",
        title=dict(text=f"Regime transitions — last {lookback_days} trading days",
                   font=dict(size=13)),
        yaxis=dict(visible=False, range=[0, 1]),
        xaxis=dict(showgrid=False),
        showlegend=False,
    ))
    return fig
