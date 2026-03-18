"""
Usage:
    streamlit run dashboard.py
    python -m streamlit run dashboard.py on windows
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from live_signals import get_live_signals
from paper_trader import (
    load_state, load_trades, load_history,
    get_intraday_curve, INITIAL_CAPITAL as PT_INITIAL_CAPITAL,
)
from scheduler import check_kill_switch, ORDERS_FILE

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Greatrex Quant Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Fine-grained CSS tweaks (dark grey tone-on-tone) ──────────────────────────
st.markdown("""
<style>
  /* Tighten top padding */
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #252525;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 14px 16px;
  }
  div[data-testid="stMetricValue"]  { font-size: 1.5rem; font-weight: 600; }
  div[data-testid="stMetricLabel"]  { color: #888; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em; }
  div[data-testid="stMetricDelta"]  { font-size: 0.78rem; }

  /* Section dividers */
  .section-head {
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #666;
    margin: 0.5rem 0 0.75rem 0;
  }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Plotly chart borders */
  .js-plotly-plot .plotly { border-radius: 10px; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    background: #252525;
    border-radius: 8px 8px 0 0;
    padding: 6px 18px;
    color: #888;
    font-size: 0.82rem;
  }
  .stTabs [aria-selected="true"] { color: #e0e0e0; background: #2d2d2d; }
</style>
""", unsafe_allow_html=True)

# ── Shared Plotly layout defaults ─────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor="#1c1c1c",
    plot_bgcolor ="#1c1c1c",
    font         =dict(color="#c0c0c0", family="Inter, system-ui, sans-serif", size=12),
    margin       =dict(l=56, r=20, t=44, b=36),
    hovermode    ="x unified",
    legend       =dict(bgcolor="#252525", bordercolor="#333", borderwidth=1,
                       font=dict(size=11)),
    xaxis        =dict(gridcolor="#2a2a2a", zeroline=False, showgrid=True),
    yaxis        =dict(gridcolor="#2a2a2a", zeroline=False, showgrid=True),
)

def _layout(**overrides) -> dict:
    """Merge _LAYOUT with per-chart overrides, deep-merging dict values."""
    base = dict(_LAYOUT)
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base

PALETTE = {
    "equal_weight"  : "#4a9eff",
    "atr_sized"     : "#50fa7b",
    "atr_pca_macro" : "#bd93f9",
    "eq_dd_control" : "#ffb86c",
    "vol_target"    : "#ff79c6",
    "buy_hold"      : "#6272a4",
    "pos"           : "#50fa7b",
    "neg"           : "#ff5555",
}
LABELS = {
    "equal_weight"  : "Equal Weight",
    "atr_sized"     : "ATR Sized",
    "atr_pca_macro" : "ATR + PCA + Macro",
    "eq_dd_control" : "Equal Wt + DD Control",
    "vol_target"    : "Vol Target",
    "buy_hold"      : "Buy & Hold",
}

# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_portfolio_curves() -> pd.DataFrame:
    path = Path("data/results/portfolio_comparison.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data
def load_ticker_curves() -> dict:
    curves = {}
    for f in Path("data/results").glob("*_curves.parquet"):
        name = f.stem.replace("_curves", "")
        if name == "portfolio":
            continue
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        curves[name] = df
    return curves

@st.cache_data
def load_walk_forward() -> pd.DataFrame:
    path = Path("data/results/walk_forward_regime.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_walk_forward_atr() -> pd.DataFrame:
    path = Path("data/results/walk_forward_atr_pca.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_oos_selection() -> pd.DataFrame:
    path = Path("data/results/oos_selection.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_macro() -> pd.DataFrame:
    path = Path("data/macro/macro_features.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def metrics(ret: pd.Series) -> dict:
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
    active = ret[ret != 0]
    wr     = (active > 0).sum() / len(active) if len(active) > 0 else 0
    return dict(ann_r=ann_r, vol=vol, sharpe=sharpe, max_dd=max_dd,
                calmar=calmar, win_rate=wr, total=total)


# ── Monte Carlo (block bootstrap) ────────────────────────────────────────────
@st.cache_data
def run_monte_carlo(returns_bytes: bytes, n_paths: int = 1000,
                    block_size: int = 21, seed: int = 42) -> np.ndarray:
    """
    Block bootstrap — resample 21-day blocks with replacement to preserve
    short-term autocorrelation structure. Returns array (n_paths × n_days)
    of equity curves normalised to start at 1.0.
    """
    import io
    ret = pd.read_parquet(io.BytesIO(returns_bytes)).values.ravel()
    n   = len(ret)
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n))
    for i in range(n_paths):
        sampled: list = []
        while len(sampled) < n:
            s = rng.integers(0, max(1, n - block_size))
            sampled.extend(ret[s: s + block_size].tolist())
        paths[i] = sampled[:n]
    return np.cumprod(1 + paths, axis=1)


# ── Chart builders ────────────────────────────────────────────────────────────
def chart_equity(df: pd.DataFrame, height: int = 420) -> go.Figure:
    fig = go.Figure()
    for col in ["equal_weight", "atr_sized", "atr_pca_macro", "eq_dd_control", "vol_target", "buy_hold"]:
        if col not in df.columns:
            continue
        dash = "dot" if col == "buy_hold" else "solid"
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=LABELS[col],
            line=dict(color=PALETTE[col], width=1.8, dash=dash),
            hovertemplate=f"<b>{LABELS[col]}</b>  $%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(**_layout(
        height=height,
        title=dict(text="Portfolio Equity Curves — $100 k starting capital",
                   font=dict(size=13)),
        yaxis=dict(title="Value ($)"),
    ))
    return fig


def chart_drawdown(df: pd.DataFrame, height: int = 240) -> go.Figure:
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
            hovertemplate=f"<b>{LABELS[col]}</b>  %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**_layout(
        height=height,
        title=dict(text="Drawdown (%)", font=dict(size=13)),
        yaxis=dict(title="DD (%)"),
    ))
    return fig


def chart_monte_carlo(eq_curves: np.ndarray, actual: np.ndarray,
                      height: int = 420) -> go.Figure:
    n   = eq_curves.shape[1]
    xs  = np.arange(n)
    p5, p25, p50, p75, p95 = (np.percentile(eq_curves, q, axis=0)
                               for q in (5, 25, 50, 75, 95))
    fig = go.Figure()

    # Faint individual paths (sample 80)
    sample_idx = np.random.default_rng(0).choice(len(eq_curves),
                                                   size=min(80, len(eq_curves)),
                                                   replace=False)
    for i in sample_idx:
        fig.add_trace(go.Scatter(
            x=xs, y=eq_curves[i],
            line=dict(color="rgba(180,180,180,0.05)", width=1),
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
        hovertemplate="Median: %{y:.3f}<extra></extra>",
    ))
    # Actual
    fig.add_trace(go.Scatter(
        x=xs[:len(actual)], y=actual, name="Historical",
        line=dict(color=PALETTE["equal_weight"], width=2.2),
        hovertemplate="Historical: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(**_layout(
        height=height,
        title=dict(
            text=f"Monte Carlo Bootstrap  ·  {len(eq_curves):,} paths  ·  21-day block resampling",
            font=dict(size=13),
        ),
        xaxis=dict(title="Trading days"),
        yaxis=dict(title="Growth of $1"),
    ))
    return fig


def chart_mc_histogram(eq_curves: np.ndarray, height: int = 280) -> go.Figure:
    finals = (eq_curves[:, -1] - 1) * 100   # % total return
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=finals, nbinsx=60,
        marker_color=PALETTE["equal_weight"], opacity=0.75,
        hovertemplate="Return: %{x:.1f}%  Count: %{y}<extra></extra>",
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
        title=dict(text="Distribution of Terminal Returns", font=dict(size=13)),
        xaxis=dict(title="Total return (%)"),
        yaxis=dict(title="Paths"),
    ))
    return fig


def chart_walk_forward(wf: pd.DataFrame, height: int = 280) -> go.Figure:
    colors = [PALETTE["pos"] if s >= 0 else PALETTE["neg"] for s in wf["sharpe"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=wf["period"], y=wf["sharpe"],
        marker_color=colors,
        text=[f"{s:+.2f}" for s in wf["sharpe"]],
        textposition="outside",
        textfont=dict(size=11, color="#c0c0c0"),
        hovertemplate="<b>%{x}</b><br>OOS Sharpe: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#555", line_dash="dot")
    fig.update_layout(**_layout(
        height=height, showlegend=False,
        title=dict(text="Walk-Forward OOS Sharpe  (3 yr train / 1 yr test)",
                   font=dict(size=13)),
        xaxis=dict(title=None),
        yaxis=dict(title="Sharpe ratio"),
    ))
    return fig


def chart_asset_sharpe(ticker_curves: dict, height: int = 480) -> go.Figure:
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
        hovertemplate="<b>%{y}</b> B&H Sharpe: %{x:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=order, x=[strat.get(t, 0) for t in order], name="Regime Strategy",
        orientation="h", marker_color=PALETTE["equal_weight"], opacity=0.9,
        hovertemplate="<b>%{y}</b> Strategy Sharpe: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        height=height, barmode="group",
        title=dict(text="Sharpe Ratio  ·  Strategy vs Buy & Hold by Asset",
                   font=dict(size=13)),
        xaxis=dict(title="Sharpe ratio"),
        yaxis=dict(title=None),
    ))
    return fig


def chart_macro_overlay(df_port: pd.DataFrame, macro: pd.DataFrame,
                         height: int = 560) -> go.Figure:
    macro = macro.reindex(df_port.index, method="ffill")
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
            ), row=1, col=1)
    # VIX
    if "vix" in macro.columns:
        fig.add_trace(go.Scatter(
            x=macro.index, y=macro["vix"], name="VIX",
            line=dict(color=PALETTE["neg"], width=1),
            fill="tozeroy", fillcolor="rgba(255,85,85,0.08)", showlegend=False,
        ), row=2, col=1)
        for lvl, clr in [(20, "#555"), (30, PALETTE["neg"])]:
            fig.add_hline(y=lvl, line_color=clr, line_dash="dot",
                          line_width=1, row=2, col=1)
    # Yield curve
    if "yield_curve" in macro.columns:
        yc = macro["yield_curve"]
        pos = yc.clip(lower=0); neg = yc.clip(upper=0)
        fig.add_trace(go.Scatter(
            x=macro.index, y=yc, name="10Y–2Y",
            line=dict(color=PALETTE["atr_pca_macro"], width=1.2),
            showlegend=False,
        ), row=3, col=1)
        fig.add_hline(y=0, line_color="#555", line_dash="dot",
                      line_width=1, row=3, col=1)

    fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                      font=dict(color="#c0c0c0", size=11),
                      height=height, showlegend=True,
                      legend=dict(bgcolor="#252525", bordercolor="#333", borderwidth=1),
                      margin=dict(l=56, r=20, t=44, b=36))
    for r in range(1, 4):
        fig.update_xaxes(gridcolor="#2a2a2a", row=r, col=1)
        fig.update_yaxes(gridcolor="#2a2a2a", row=r, col=1)
    return fig


# ── Monthly returns heatmap ───────────────────────────────────────────────────
def chart_monthly_heatmap(ret: pd.Series, title: str = "Equal Weight — Monthly Returns",
                          height: int = 340) -> go.Figure:
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
        hovertemplate="<b>%{y} %{x}</b>: %{text}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        height=height,
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title=None, side="top"),
        yaxis=dict(title=None, autorange="reversed"),
        margin=dict(l=56, r=80, t=56, b=16),
    ))
    return fig


# ── MA spread bar chart (live tab) ────────────────────────────────────────────
def chart_ma_spread(live_df: pd.DataFrame, height: int = 400) -> go.Figure:
    df = live_df.sort_values("MA Spread %")
    colors = [PALETTE["pos"] if v >= 0 else PALETTE["neg"] for v in df["MA Spread %"]]
    fig = go.Figure(go.Bar(
        y=df["Ticker"], x=df["MA Spread %"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in df["MA Spread %"]],
        textposition="outside",
        textfont=dict(size=10, color="#c0c0c0"),
        hovertemplate="<b>%{y}</b>  MA Spread: %{x:.2f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#555", line_width=1.5)
    fig.update_layout(**_layout(
        height=height, showlegend=False,
        title=dict(text="MA Spread — Fast vs Slow MA  (positive = golden cross / LONG)",
                   font=dict(size=13)),
        xaxis=dict(title="Spread (%)"),
        yaxis=dict(title=None),
    ))
    return fig


# ── TradingView-style portfolio chart ────────────────────────────────────────
def chart_paper_portfolio(history_df: pd.DataFrame,
                          intraday_df: pd.DataFrame,
                          trades_df: pd.DataFrame,
                          height: int = 440) -> go.Figure:
    """
    TradingView-style equity curve combining:
      - Historical daily closes (from paper_trader history.csv)
      - Today's 5-minute intraday values (live from yfinance)
      - Buy/sell trade markers
    """
    fig = go.Figure()

    # ── Historical equity curve ───────────────────────────────────────────────
    if not history_df.empty:
        fig.add_trace(go.Scatter(
            x=history_df["date"],
            y=history_df["portfolio_value"],
            name="Portfolio (daily)",
            line=dict(color="#4a9eff", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(74,158,255,0.07)",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>",
        ))

    # ── Today's intraday curve ────────────────────────────────────────────────
    if not intraday_df.empty:
        fig.add_trace(go.Scatter(
            x=intraday_df.index,
            y=intraday_df["portfolio_value"],
            name="Today (live)",
            line=dict(color="#50fa7b", width=1.8),
            hovertemplate="<b>%{x|%H:%M}</b><br>$%{y:,.0f}<extra></extra>",
        ))

    # ── Trade markers ─────────────────────────────────────────────────────────
    if not trades_df.empty and not history_df.empty:
        # Map trade date → portfolio_value on that day for marker y-position
        hist_val = history_df.set_index("date")["portfolio_value"]

        buys  = trades_df[trades_df["action"] == "BUY"].copy()
        sells = trades_df[trades_df["action"] == "SELL"].copy()

        def _marker_y(dates: pd.Series) -> list:
            return [float(hist_val.get(d, hist_val.iloc[-1])) for d in dates]

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
                hovertemplate="<b>BUY %{text}</b><br>%{x|%Y-%m-%d}<extra></extra>",
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
                hovertemplate="<b>SELL %{text}</b><br>%{x|%Y-%m-%d}<extra></extra>",
            ))

    # Reference line at initial capital
    fig.add_hline(
        y=PT_INITIAL_CAPITAL, line_color="#444", line_dash="dot", line_width=1,
        annotation_text=f"  Start ${PT_INITIAL_CAPITAL/1000:.0f}k",
        annotation_font_color="#555",
    )

    fig.update_layout(**_layout(
        height=height,
        title=dict(text="Paper Portfolio — Equity Curve  (live via Yahoo Finance)",
                   font=dict(size=13)),
        yaxis=dict(title="Value ($)", tickprefix="$", tickformat=",.0f"),
        xaxis=dict(
            title=None,
            rangeslider=dict(visible=False),
            rangeselector=dict(
                bgcolor="#252525",
                activecolor="#3a3a3a",
                bordercolor="#444",
                font=dict(color="#c0c0c0", size=10),
                buttons=[
                    dict(count=7,  label="1W",  step="day",   stepmode="backward"),
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(count=3,  label="3M",  step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
            ),
        ),
    ))
    return fig


# ── Summary table ─────────────────────────────────────────────────────────────
def metrics_table(df_port: pd.DataFrame) -> pd.DataFrame:
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


# ── Main layout ───────────────────────────────────────────────────────────────
def main():
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
        d = val - ref
        s = f"{d*100:+.1f}%" if pct else f"{d:+.2f}"
        return s

    with c1: st.metric("Ann. Return",  f"{m_eq['ann_r']*100:.1f}%",
                        delta=delta_str(m_eq['ann_r'], m_bnh['ann_r']),
                        help="Compound Annual Growth Rate (CAGR) — the annualised return "
                             "if capital was invested for the full period. Delta shows "
                             "outperformance vs buy & hold.")
    with c2: st.metric("Sharpe Ratio", f"{m_eq['sharpe']:.2f}",
                        delta=delta_str(m_eq['sharpe'], m_bnh['sharpe'], pct=False),
                        help="(Annualised return) / (Annualised volatility). A risk-adjusted "
                             "return score. >1.0 is good; >2.0 is exceptional. Higher = "
                             "more return earned per unit of risk taken.")
    with c3: st.metric("Volatility",   f"{m_eq['vol']*100:.1f}%",
                        delta=delta_str(m_eq['vol'], m_bnh['vol']), delta_color="inverse",
                        help="Annualised standard deviation of daily returns. Measures how "
                             "much returns vary day-to-day. Lower is better for a given "
                             "level of return. Delta (inverse): green = lower vol than B&H.")
    with c4: st.metric("Max Drawdown", f"{m_eq['max_dd']*100:.1f}%",
                        delta=delta_str(m_eq['max_dd'], m_bnh['max_dd']), delta_color="inverse",
                        help="Largest peak-to-trough decline in the equity curve. −20% means "
                             "the strategy fell 20% from its peak before recovering. Lower "
                             "magnitude = better. Delta (inverse): green = shallower DD than B&H.")
    with c5: st.metric("Calmar Ratio", f"{m_eq['calmar']:.2f}",
                        help="Annualised return divided by the absolute maximum drawdown. "
                             "Measures how much return you earned relative to the worst loss. "
                             ">1.0 is considered good; institutional target is often >0.5.")
    with c6: st.metric("Win Rate",     f"{m_eq['win_rate']*100:.0f}%",
                        help="Percentage of active trading days where the strategy had a "
                             "positive return. Even a 50% win rate can be very profitable "
                             "if winning days are larger than losing days (see Profit Factor).")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "  📊  Equity Curves",
        "  🎲  Monte Carlo",
        "  🔁  Walk-Forward",
        "  🌍  Macro Overlay",
        "  📡  Live Signals",
    ])

    # ── Tab 1: Equity curves + drawdown + summary table ──────────────────────
    with tab1:
        st.plotly_chart(chart_equity(df_port), width="stretch")
        st.plotly_chart(chart_drawdown(df_port), width="stretch")

        st.markdown('<div class="section-head">All portfolio methods — summary</div>',
                    unsafe_allow_html=True)
        tbl = metrics_table(df_port)
        st.dataframe(
            tbl.style.applymap(
                lambda v: "color:#50fa7b" if (isinstance(v, str) and v.startswith("+")) else
                          "color:#ff5555" if (isinstance(v, str) and "-" in v and "%" in v and v != "-0.0%") else "",
            ),
            width="stretch", hide_index=True,
        )

        # Monthly returns heatmap
        st.markdown("")
        st.markdown('<div class="section-head">Monthly returns — equal weight strategy</div>',
                    unsafe_allow_html=True)
        if not eq_ret.empty:
            st.plotly_chart(chart_monthly_heatmap(eq_ret), width="stretch")

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
                            width="stretch")
            st.plotly_chart(chart_mc_histogram(eq_curves),
                            width="stretch")

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
                st.plotly_chart(chart_walk_forward(wf), width="stretch")
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
                                width="stretch")

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
                width="stretch", hide_index=True,
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
                            width="stretch")

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
        if st.button("Refresh  (live data + portfolio)", type="secondary"):
            st.cache_data.clear()
            st.rerun()

        # ── Paper Portfolio ───────────────────────────────────────────────────
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
            pv        = pt_state.get("portfolio_value", PT_INITIAL_CAPITAL)
            cash      = pt_state["cash"]
            invested  = pv - cash
            total_ret = (pv / PT_INITIAL_CAPITAL - 1) * 100
            n_pos     = len(pt_state.get("positions", {}))

            # Daily return from history
            daily_ret = 0.0
            if len(pt_history) >= 2:
                daily_ret = float(pt_history["daily_return"].iloc[-1])

            st.markdown("### Paper Portfolio")
            pm1, pm2, pm3, pm4, pm5 = st.columns(5)
            with pm1: st.metric("Portfolio Value", f"${pv:,.0f}",
                                 help="Current estimated portfolio value (cash + open positions at last close)")
            with pm2: st.metric("Total Return",    f"{total_ret:+.2f}%",
                                 help="Total return since paper trading started")
            with pm3: st.metric("Today's Return",  f"{daily_ret:+.2f}%",
                                 help="Portfolio return on last trading day recorded")
            with pm4: st.metric("Cash",            f"${cash:,.0f}",
                                 help="Uninvested cash available for new positions")
            with pm5: st.metric("Open Positions",  str(n_pos),
                                 help="Number of assets currently held long")

            st.markdown("")

            # TradingView-style equity chart
            with st.spinner("Loading intraday data…"):
                intraday_df = get_intraday_curve()

            st.plotly_chart(
                chart_paper_portfolio(pt_history, intraday_df, pt_trades),
                width="stretch",
            )

            # ── Open positions table ──────────────────────────────────────────
            if pt_state.get("positions"):
                st.markdown('<div class="section-head">Open positions</div>',
                            unsafe_allow_html=True)

                @st.cache_data(ttl=120, show_spinner=False)
                def _pos_prices(tickers_key: str) -> dict:
                    import yfinance as yf
                    tickers = tickers_key.split(",")
                    prices  = {}
                    for t in tickers:
                        try:
                            tk = yf.Ticker(t)
                            prices[t] = tk.fast_info.get("last_price") or tk.fast_info.get("previousClose")
                        except Exception:
                            pass
                    return prices

                tickers_key  = ",".join(sorted(pt_state["positions"]))
                live_prices  = _pos_prices(tickers_key)

                pos_rows = []
                for ticker, pos in pt_state["positions"].items():
                    cur = live_prices.get(ticker, pos["entry_price"])
                    cur = cur if cur else pos["entry_price"]
                    cur_val  = pos["shares"] * cur
                    entry_val = pos["cost_basis"]
                    unreal_pnl = cur_val - entry_val
                    unreal_pct = (cur / pos["entry_price"] - 1) * 100
                    pos_rows.append({
                        "Ticker"     : ticker,
                        "Shares"     : round(pos["shares"], 3),
                        "Entry"      : pos["entry_price"],
                        "Current"    : round(cur, 2),
                        "Chg %"      : round(unreal_pct, 2),
                        "Unreal P&L" : round(unreal_pnl, 2),
                        "Entry Date" : pos["entry_date"],
                    })

                pos_df = pd.DataFrame(pos_rows)

                def _color_pnl(v):
                    if isinstance(v, (int, float)):
                        if v > 0: return "color:#50fa7b"
                        if v < 0: return "color:#ff5555"
                    return ""

                st.dataframe(
                    pos_df.style
                    .applymap(_color_pnl, subset=["Chg %", "Unreal P&L"])
                    .format({"Entry": "${:.2f}", "Current": "${:.2f}",
                             "Chg %": "{:+.2f}%", "Unreal P&L": "${:+,.2f}"}),
                    width="stretch", hide_index=True,
                )

            # ── Recent trades ─────────────────────────────────────────────────
            if not pt_trades.empty:
                st.markdown("")
                st.markdown('<div class="section-head">Recent trades</div>',
                            unsafe_allow_html=True)
                recent = pt_trades.sort_values("date", ascending=False).head(20)
                def _color_action(v):
                    if v == "BUY":  return "color:#50fa7b;font-weight:600"
                    if v == "SELL": return "color:#ff5555;font-weight:600"
                    return ""
                st.dataframe(
                    recent.style.applymap(_color_action, subset=["action"]),
                    width="stretch", hide_index=True,
                )

            # ── Kill switch status ────────────────────────────────────────
            kill = check_kill_switch()
            if kill:
                st.error(
                    "KILL SWITCH ACTIVE — rolling 20-day drawdown exceeded 15%. "
                    "Trading is paused.  Review the portfolio before re-enabling."
                )
            else:
                st.success("Kill switch: OK  (drawdown within limits)")

            # ── Tomorrow's order sheet ────────────────────────────────────
            if ORDERS_FILE.exists():
                import json as _json
                with open(ORDERS_FILE, encoding="utf-8") as _f:
                    sheet = _json.load(_f)

                st.markdown("")
                st.markdown(
                    f'<div class="section-head">Tomorrow\'s order sheet '
                    f'— generated {sheet.get("generated_at","?")}</div>',
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
                        orders_df.style.applymap(_color_order, subset=["action"]),
                        width="stretch", hide_index=True,
                    )
            else:
                st.markdown(
                    "<span style='color:#555;font-size:.78rem'>"
                    "No order sheet yet — run <code>python scheduler.py</code> "
                    "or <code>python paper_trader.py run</code> after 4:45 PM ET."
                    "</span>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<span style='color:#555;font-size:.75rem'>"
                f"Initialised: {pt_state.get('initialized_date','?')}  ·  "
                f"Last EOD update: {pt_state.get('last_eod_date','?')}  ·  "
                "Improvements vs backtest: RSI &lt; 70 entry filter  ·  "
                "3× ATR trailing stop  ·  5-day min hold  ·  "
                "data validation  ·  15% kill switch"
                "</span>",
                unsafe_allow_html=True,
            )

        # ── Live Signals ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Universe Signal State")
        st.markdown(
            "<span style='color:#888;font-size:.82rem'>"
            "MA crossover logic mirrors the backtest. Signal flips at daily close. "
            "Cached 5 min."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        @st.cache_data(ttl=300, show_spinner=False)
        def _cached_live():
            return get_live_signals()

        with st.spinner("Fetching live prices via yfinance…"):
            live_df, fetch_ts = _cached_live()

        if live_df.empty:
            st.error("Could not fetch live data. Check your internet connection.")
        else:
            n_long  = (live_df["Signal"] == "LONG").sum()
            n_flat  = (live_df["Signal"] == "FLAT").sum()
            exp_pct = n_long / len(live_df) * 100

            st.markdown(
                f"<span style='color:#666;font-size:.78rem'>Last fetched: {fetch_ts}</span>",
                unsafe_allow_html=True,
            )
            lv1, lv2, lv3, lv4 = st.columns(4)
            with lv1: st.metric("Long signals",   str(n_long))
            with lv2: st.metric("Flat (cash)",    str(n_flat))
            with lv3: st.metric("Gross exposure", f"{exp_pct:.0f}%")
            avg_rsi_long = live_df.loc[live_df["Signal"] == "LONG", "RSI"].mean()
            with lv4: st.metric("Avg RSI (longs)", f"{avg_rsi_long:.1f}" if n_long > 0 else "—")

            st.markdown("")
            st.plotly_chart(chart_ma_spread(live_df), width="stretch")

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
                .applymap(_color_cell, subset=["Signal", "Day Chg %", "MA Spread %",
                                               "20d Ret %", "60d Ret %", "Dist High %"])
                .format({"Price": "${:.2f}", "Day Chg %": "{:+.2f}%",
                         "MA Spread %": "{:+.2f}%", "RSI": "{:.1f}",
                         "20d Ret %": "{:+.1f}%", "60d Ret %": "{:+.1f}%",
                         "Dist High %": "{:+.1f}%"}),
                width="stretch", hide_index=True,
            )


if __name__ == "__main__":
    main()
