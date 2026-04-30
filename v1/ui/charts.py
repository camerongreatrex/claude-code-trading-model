# ui/charts.py
# All chart-builder functions and the metrics/metrics_table helpers.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from v1.scripts.paper_trader import INITIAL_CAPITAL as PT_INITIAL_CAPITAL
from v1.ui.styles import _layout, PALETTE, LABELS, get_color, get_label


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
            var_95   — Historical 1-day 95% VaR (positive fraction, e.g. 0.015 = 1.5% loss threshold)
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
    # Historical 1-day 95% VaR: loss exceeded only 5% of trading days.
    # Expressed as a positive fraction (e.g. 0.015 = 1.5% daily loss threshold).
    var_95 = float(-np.percentile(ret, 5))
    return dict(ann_r=ann_r, vol=vol, sharpe=sharpe, max_dd=max_dd,
                calmar=calmar, win_rate=wr, total=total, var_95=var_95)


# ── Summary table ─────────────────────────────────────────────────────────────
def metrics_table(df_port: pd.DataFrame, oos_sel: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build a human-readable summary statistics table for all sizing methods.

    Calls metrics() for each method column and formats values as display
    strings with consistent sign/decimal conventions.  The resulting
    DataFrame is rendered in the Overview tab with green/red cell colouring
    applied via Streamlit's .style.map().

    Args:
        df_port:  Portfolio curves DataFrame (same as load_portfolio_curves()).
        oos_sel:  Optional OOS selection DataFrame from oos_selection.parquet.
                  If provided, an "OOS Sharpe" column is added right after "Method".

    Returns:
        DataFrame with columns [Method, OOS Sharpe (if available), Ann. Return,
        Volatility, Sharpe, Max DD, Calmar, Win Rate, VaR 95%].
        One row per sizing method present in df_port.
    """
    from v1.ui.styles import TIER_SHOW, TIER_AVAILABLE
    _visible = TIER_SHOW | TIER_AVAILABLE

    rows = []
    for col in df_port.columns:
        if col not in _visible:
            continue
        ret = df_port[col].pct_change().dropna()
        m   = metrics(ret)
        if not m:
            continue
        # Look up OOS Sharpe for this method
        oos_sharpe_str = "—"
        if oos_sel is not None and not oos_sel.empty and "method" in oos_sel.columns:
            _oos_m = oos_sel[
                oos_sel["method"].str.replace(" ", "_").str.replace("-", "_") == col
            ]
            if _oos_m.empty:
                _oos_m = oos_sel[oos_sel["method"] == col.replace("_", " ")]
            if not _oos_m.empty and "oos_sharpe" in _oos_m.columns:
                oos_sharpe_str = f"{float(_oos_m['oos_sharpe'].iloc[0]):.2f}"
        rows.append({
            "Method"      : get_label(col),
            "OOS Sharpe"  : oos_sharpe_str,
            "Ann. Return" : f"{m['ann_r']*100:+.1f}%",
            "Volatility"  : f"{m['vol']*100:.1f}%",
            "Sharpe"      : f"{m['sharpe']:.2f}",
            "Max DD"      : f"{m['max_dd']*100:.1f}%",
            "Calmar"      : f"{m['calmar']:.2f}",
            "Win Rate"    : f"{m['win_rate']*100:.0f}%",
            "VaR 95%"     : f"{m['var_95']*100:.2f}%",
        })
    result = pd.DataFrame(rows)
    # Sort by OOS Sharpe descending (best strategies at top)
    if not result.empty and "OOS Sharpe" in result.columns:
        result["_sort"] = result["OOS Sharpe"].apply(
            lambda x: float(x) if x != "—" else -999
        )
        result = result.sort_values("_sort", ascending=False).drop(columns=["_sort"])
        result = result.reset_index(drop=True)
    return result


# ── Chart builders ────────────────────────────────────────────────────────────
def chart_equity(df: pd.DataFrame, height: int = 420,
                 columns: list = None) -> go.Figure:
    """
    Build a multi-line equity curve chart for portfolio sizing methods.

    Adds one go.Scatter trace per method present in df.  Buy & Hold is
    rendered as a dotted line to visually distinguish the benchmark from
    the strategy variants.

    Args:
        df:      DataFrame indexed by date with one column per sizing method.
        height:  Chart height in pixels.
        columns: Optional list of column names to plot. If None, plots the
                 top 5 columns by final value plus buy_hold.

    Returns:
        go.Figure with dragmode="pan" and scrollZoom enabled via config at
        the call site.
    """
    if columns is None:
        non_bh = [c for c in df.columns if c != "buy_hold"]
        top5   = sorted(non_bh, key=lambda c: df[c].iloc[-1] if not df[c].empty else 0,
                        reverse=True)[:5]
        columns = top5 + (["buy_hold"] if "buy_hold" in df.columns else [])
    fig = go.Figure()
    for col in columns:
        if col not in df.columns:
            continue
        dash  = "dot" if col == "buy_hold" else "solid"
        color = get_color(col)
        label = get_label(col)
        if col == "buy_hold":
            hovertemplate = (
                "<b>Buy & Hold benchmark</b><br>"
                "$%{y:,.0f}<extra></extra>"
            )
        else:
            hovertemplate = (
                f"<b>{label}</b><br>"
                "$%{y:,.0f}<extra></extra>"
            )
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=label,
            line=dict(color=color, width=1.8, dash=dash),
            hovertemplate=hovertemplate,
        ))
    # Add OOS region indicator: grey tint for training warm-up, faint green for OOS
    if len(df) > 756:  # 3 years of trading days → first OOS window starts after that
        oos_start = df.index[756]
        fig.add_vrect(
            x0=df.index[0], x1=oos_start,
            fillcolor="rgba(100,100,100,0.08)",
            line_width=0,
            annotation_text="Training",
            annotation_position="top left",
            annotation_font_color="#555",
            annotation_font_size=10,
        )
        fig.add_vrect(
            x0=oos_start, x1=df.index[-1],
            fillcolor="rgba(80,250,123,0.03)",
            line_width=0,
            annotation_text="Walk-Forward OOS Region",
            annotation_position="top left",
            annotation_font_color="#50fa7b",
            annotation_font_size=10,
        )
    fig.update_layout(**_layout(
        height=height,
        dragmode="pan",
        uirevision="backtest_equity",
        title=dict(
            text="Portfolio Growth — $100k start  ·  Zero leverage (gross ≤ 1.0×)",
            font=dict(size=13),
        ),
        yaxis=dict(title="Value ($)", fixedrange=False),
        xaxis=dict(fixedrange=False),
    ))
    return fig


def chart_equity_risk_adjusted(df: pd.DataFrame, height: int = 420,
                               columns: list = None) -> go.Figure:
    """
    Equity curves scaled so every strategy has 10% annualized volatility.

    This is the fair comparison: at equal risk, the strategy with higher
    Sharpe ratio produces higher return. Buy & Hold (Sharpe ~0.75) will
    appear BELOW strategies with Sharpe > 0.75.

    This is how institutional investors compare strategies — they normalize
    for risk first, then compare returns. A hedge fund running at 10% vol
    with Sharpe 1.5 earns 15% annualized; the S&P at 10% vol with Sharpe
    0.75 earns only 7.5%.
    """
    if columns is None:
        columns = list(df.columns)

    TARGET_VOL = 0.10
    fig = go.Figure()

    for col in columns:
        if col not in df.columns:
            continue
        ret = df[col].pct_change().dropna()
        if ret.empty:
            continue
        realized_vol = ret.std() * np.sqrt(252)
        if realized_vol <= 0.001:
            continue

        scale_factor = TARGET_VOL / realized_vol
        scaled_ret = ret * scale_factor
        scaled_equity = 100_000 * (1 + scaled_ret).cumprod()

        dash = "dot" if col == "buy_hold" else "solid"
        color = get_color(col)
        label = get_label(col)

        fig.add_trace(go.Scatter(
            x=scaled_equity.index, y=scaled_equity.values,
            name=f"{label} ({realized_vol*100:.0f}%→10%)",
            line=dict(color=color, width=1.8, dash=dash),
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"$%{{y:,.0f}} (scaled to 10% vol)<br>"
                f"<i>Original vol: {realized_vol*100:.1f}% · Scale: {scale_factor:.2f}×<br>"
                "At equal volatility, higher line = better risk-adjusted return.</i>"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=100_000, line_color="#444", line_dash="dot", line_width=1)

    fig.update_layout(**_layout(
        height=height,
        dragmode="pan",
        uirevision="equity_risk_adj",
        title=dict(
            text="Risk-Adjusted Growth — all strategies normalized to 10% volatility",
            font=dict(size=13),
        ),
        yaxis=dict(title="Value ($) at 10% vol", tickprefix="$", tickformat=",.0f",
                   fixedrange=False),
        xaxis=dict(fixedrange=False),
    ))
    return fig


def chart_drawdown(df: pd.DataFrame, height: int = 420,
                   columns: list = None) -> go.Figure:
    """
    Build a drawdown chart for selected portfolio sizing methods.

    Computes percentage drawdown as (price − rolling_peak) / rolling_peak × 100
    and renders each as a filled area trace so shallow drawdowns are immediately
    visible against the dark background.

    Args:
        df:      DataFrame indexed by date with one column per sizing method.
        height:  Chart height in pixels.
        columns: Optional list of column names to plot. If None, plots the
                 top 3 columns by final value plus buy_hold.

    Returns:
        go.Figure with fill="tozeroy" traces and dragmode="pan".
    """
    if columns is None:
        non_bh = [c for c in df.columns if c != "buy_hold"]
        top3   = sorted(non_bh, key=lambda c: df[c].iloc[-1] if not df[c].empty else 0,
                        reverse=True)[:3]
        columns = top3 + (["buy_hold"] if "buy_hold" in df.columns else [])
    alphas = [0.18, 0.14, 0.10, 0.08, 0.06]
    fig = go.Figure()
    for idx, col in enumerate(columns):
        if col not in df.columns:
            continue
        alpha = alphas[min(idx, len(alphas) - 1)]
        color = get_color(col)
        label = get_label(col)
        p  = df[col]
        dd = (p - p.cummax()) / p.cummax() * 100
        r, g, b = (int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        fig.add_trace(go.Scatter(
            x=df.index, y=dd, name=label,
            line=dict(color=color, width=1.4),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},{alpha})",
            hovertemplate=(
                f"<b>{label}</b><br>"
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
        uirevision="backtest_drawdown",
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
        uirevision="monte_carlo_fan",
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
        uirevision="mc_histogram",
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
        uirevision="walk_forward_sharpe",
        title=dict(text="Walk-Forward OOS Sharpe  (3 yr train / 1 yr test)",
                   font=dict(size=13)),
        xaxis=dict(title=None, fixedrange=False),
        yaxis=dict(title="Sharpe ratio", fixedrange=False),
        margin=dict(t=60),
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
        uirevision="asset_sharpe",
        title=dict(text="Sharpe Ratio  ·  Strategy vs Buy & Hold by Asset",
                   font=dict(size=13)),
        xaxis=dict(title="Sharpe ratio", fixedrange=False),
        yaxis=dict(title=None, fixedrange=False),
    ))
    return fig


def chart_macro_overlay(df_port: pd.DataFrame, macro: pd.DataFrame,
                         fred: pd.DataFrame = None,
                         height: int = 560,
                         primary_method: str = None) -> go.Figure:
    """
    Build a multi-panel subplot chart overlaying the portfolio equity curve
    with key macro regime indicators.

    Subplots (shared x-axis):
      Row 1 — Portfolio vs Buy & Hold equity curves (go.Scatter).
      Row 2 — VIX level with fill, and reference lines at 20 (caution)
               and 30 (fear).
      Row 3 — 10Y–2Y Treasury yield spread.
      Row 4 — HY credit spread (if FRED data available).
      Row 5 — FRED macro score (if available).

    Args:
        df_port: Portfolio equity curve DataFrame (same as load_portfolio_curves()).
        macro:   Macro features DataFrame (same as load_macro()).
        fred:    FRED features DataFrame (from load_fred_features()), optional.
        height:  Total chart height in pixels across all rows.

    Returns:
        go.Figure using make_subplots with shared x-axes and dragmode="pan".
    """
    macro = macro.reindex(df_port.index).ffill()

    # Determine how many rows based on available FRED data
    has_credit = fred is not None and not fred.empty and "hy_oas" in fred.columns
    has_fred_score = "fred_macro_score" in macro.columns
    n_rows = 3 + int(has_credit) + int(has_fred_score)

    row_heights = [0.35, 0.18, 0.18]
    subtitles = ["Portfolio vs Buy & Hold", "VIX (fear gauge)", "10Y–2Y Yield Spread"]
    if has_credit:
        row_heights.append(0.15)
        subtitles.append("HY Credit Spread (OAS)")
    if has_fred_score:
        row_heights.append(0.14)
        subtitles.append("FRED Macro Score")
    # Normalize row heights to sum to 1
    _total = sum(row_heights)
    row_heights = [h / _total for h in row_heights]

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.035,
                        row_heights=row_heights,
                        subplot_titles=tuple(subtitles))

    # Row 1: Equity — show the production method (defaults to whatever is in
    # df_port if `primary_method` is not provided / not present) plus B&H.
    _row1_methods = []
    if primary_method and primary_method in df_port.columns:
        _row1_methods.append(primary_method)
    elif "equal_weight" in df_port.columns:
        _row1_methods.append("equal_weight")
    if "buy_hold" in df_port.columns:
        _row1_methods.append("buy_hold")
    for col in _row1_methods:
        _label = LABELS.get(col, col.replace("_", " ").title())
        _color = PALETTE.get(col, "#888")
        fig.add_trace(go.Scatter(
            x=df_port.index, y=df_port[col], name=_label,
            line=dict(color=_color, width=1.6,
                      dash="dot" if col == "buy_hold" else "solid"),
            hovertemplate=(
                f"<b>{_label}</b><br>"
                "$%{y:,.0f}<br>"
                "<i>Portfolio value on this date (started at $100k).</i>"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # Row 2: VIX
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

    # Row 3: Yield curve
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

    # Row 4: HY credit spread (dynamic — only when FRED data available)
    _next_row = 4
    if has_credit:
        _fred_aligned = fred.reindex(df_port.index).ffill()
        fig.add_trace(go.Scatter(
            x=_fred_aligned.index, y=_fred_aligned["hy_oas"],
            name="HY OAS", showlegend=False,
            line=dict(color="#ffb86c", width=1.2),
            fill="tozeroy", fillcolor="rgba(255,184,108,0.06)",
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "HY OAS: %{y:.2f}%<br>"
                "<i>ICE BofA High Yield credit spread.<br>"
                "Rising = risk-off / credit stress.<br>"
                "Spikes above 5% historically precede drawdowns.</i>"
                "<extra></extra>"
            ),
        ), row=_next_row, col=1)
        fig.add_hline(y=5, line_color=PALETTE["neg"], line_dash="dot",
                      line_width=1, row=_next_row, col=1)
        _next_row += 1

    # Row 5: FRED macro score (dynamic)
    if has_fred_score:
        fig.add_trace(go.Scatter(
            x=macro.index, y=macro["fred_macro_score"],
            name="FRED Score", showlegend=False,
            line=dict(color="#8be9fd", width=1.2),
            fill="tozeroy", fillcolor="rgba(139,233,253,0.06)",
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "FRED Score: %{y:+.2f}<br>"
                "<i>Composite of 8 FRED indicators (credit, sentiment,<br>"
                "claims, PMI, USD, inflation, VIX). Range −1 to +1.<br>"
                "+1 = all bullish, −1 = all bearish.</i>"
                "<extra></extra>"
            ),
        ), row=_next_row, col=1)
        fig.add_hline(y=0, line_color="#555", line_dash="dot",
                      line_width=1, row=_next_row, col=1)

    _dyn_height = height if n_rows <= 3 else height + 120 * (n_rows - 3)
    fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                      font=dict(color="#c0c0c0", size=11),
                      height=_dyn_height, showlegend=True, dragmode="pan",
                      uirevision="macro_overlay",
                      legend=dict(bgcolor="#252525", bordercolor="#333", borderwidth=1),
                      margin=dict(l=56, r=20, t=44, b=36))
    for r in range(1, n_rows + 1):
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
        uirevision="monthly_heatmap",
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
        uirevision="ma_spread",
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
                          x_range: list = None) -> go.Figure:
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

    # ── Resample intraday data to the requested timeframe ───────────────────
    candle_df = pd.DataFrame()
    if not intraday_df.empty and {"open", "high", "low", "close"}.issubset(intraday_df.columns):
        candle_df = intraday_df.copy()

    # Dates covered by intraday candles — exclude these from daily history
    _intra_dates = set()
    if not candle_df.empty:
        _intra_dates = set(candle_df.index.strftime('%Y-%m-%d'))

    # ── Historical daily candlesticks (for dates NOT covered by intraday) ───
    if not history_df.empty:
        history_df = history_df.copy()
        # Drop market-holiday rows where portfolio value didn't change (e.g. Good Friday).
        # shift(1) is NaN for the first row so it is always kept.
        history_df = history_df[
            history_df["portfolio_value"].ne(history_df["portfolio_value"].shift(1))
        ].reset_index(drop=True)
        history_df["_ds"] = history_df["date"].dt.strftime('%Y-%m-%d')
        history_df = history_df[~history_df["_ds"].isin(_intra_dates)]
        if not history_df.empty:
            history_df["plot_ts"] = history_df["_ds"] + "T16:00:00"
            _closes = history_df["portfolio_value"].values
            _opens = np.concatenate([[_closes[0]], _closes[:-1]])
            _highs = np.maximum(_opens, _closes)
            _lows  = np.minimum(_opens, _closes)
            fig.add_trace(go.Candlestick(
                x=history_df["plot_ts"],
                open=_opens, high=_highs, low=_lows, close=_closes,
                name="Portfolio (daily)",
                increasing=dict(line=dict(color="#50fa7b", width=1.2),
                                fillcolor="rgba(80,250,123,0.7)"),
                decreasing=dict(line=dict(color="#ff5555", width=1.2),
                                fillcolor="rgba(255,85,85,0.7)"),
            ))

    # ── Intraday candlesticks (multi-day 5-min data) ────────────────────────
    if not candle_df.empty:
        fig.add_trace(go.Candlestick(
            x=candle_df.index,
            open=candle_df["open"],
            high=candle_df["high"],
            low=candle_df["low"],
            close=candle_df["close"],
            name="Portfolio (5m)",
            increasing=dict(line=dict(color="#50fa7b", width=1),
                            fillcolor="rgba(80,250,123,0.7)"),
            decreasing=dict(line=dict(color="#ff5555", width=1),
                            fillcolor="rgba(255,85,85,0.7)"),
        ))
        # Flat dotted line from last candle to now when market is closed
        if now is not None and now > candle_df.index[-1]:
            _lv = float(candle_df["close"].iloc[-1])
            fig.add_trace(go.Scatter(
                x=[candle_df.index[-1], now],
                y=[_lv, _lv],
                mode="lines",
                line=dict(color="#4a9eff", width=1.5, dash="dot"),
                showlegend=False,
                hovertemplate="<b>Portfolio</b> (market closed)<br>%{x|%H:%M} $%{y:,.0f}<extra></extra>",
            ))


    # ── Trade markers ─────────────────────────────────────────────────────────
    if not trades_df.empty and not history_df.empty:
        # Map trade date → portfolio_value on that day for marker y-position.
        # Index by normalized date string so Timestamp vs string mismatches don't cause KeyErrors.
        hist_val = (
            history_df.assign(_d=history_df["date"].dt.strftime('%Y-%m-%d'))
            .set_index("_d")["portfolio_value"]
        )
        _last_pv = float(hist_val.iloc[-1])

        def _marker_y(date_series: pd.Series) -> list:
            """One y value per grouped date string; falls back to last known PV."""
            out = []
            for d in date_series:
                key = pd.Timestamp(d).strftime('%Y-%m-%d') if not isinstance(d, str) else d
                try:
                    v = hist_val[key]
                    out.append(float(v.iloc[0]) if isinstance(v, pd.Series) else float(v))
                except KeyError:
                    out.append(_last_pv)
            return out

        def _group_trades(subset: pd.DataFrame) -> pd.DataFrame:
            """Collapse to one row per trading date; tickers joined as comma list."""
            subset = subset.copy()
            subset["_date_str"] = subset["date"].dt.strftime('%Y-%m-%d')
            grp = (
                subset.groupby("_date_str")["ticker"]
                .apply(lambda t: ", ".join(sorted(set(t))))
                .reset_index()
            )
            grp.columns = ["date_str", "tickers"]
            # x-position: plot at 4 PM ET same as history dots
            grp["plot_x"] = grp["date_str"] + "T16:00:00"
            grp["y"] = _marker_y(grp["date_str"])
            grp["n"] = grp["tickers"].apply(lambda t: len(t.split(",")))
            return grp

        buys  = trades_df[trades_df["action"] == "BUY"]
        sells = trades_df[trades_df["action"] == "SELL"]

        if not buys.empty:
            bg = _group_trades(buys)
            fig.add_trace(go.Scatter(
                x=bg["plot_x"], y=bg["y"],
                mode="markers+text",
                name="Buy",
                marker=dict(symbol="triangle-up", size=14,
                            color=PALETTE["pos"], line=dict(color="#1c1c1c", width=1)),
                text=bg["n"].apply(lambda n: f"+{n}"),
                textposition="top center",
                textfont=dict(size=9, color=PALETTE["pos"]),
                customdata=bg["tickers"].tolist(),
                hovertemplate=(
                    "<b>BUY — %{customdata}</b><br>"
                    "%{x|%b %d, %Y}<br>"
                    "<extra></extra>"
                ),
            ))

        if not sells.empty:
            sg = _group_trades(sells)
            fig.add_trace(go.Scatter(
                x=sg["plot_x"], y=sg["y"],
                mode="markers+text",
                name="Sell",
                marker=dict(symbol="triangle-down", size=14,
                            color=PALETTE["neg"], line=dict(color="#1c1c1c", width=1)),
                text=sg["n"].apply(lambda n: f"−{n}"),
                textposition="bottom center",
                textfont=dict(size=9, color=PALETTE["neg"]),
                customdata=sg["tickers"].tolist(),
                hovertemplate=(
                    "<b>SELL — %{customdata}</b><br>"
                    "%{x|%b %d, %Y}<br>"
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
            autorange=True,
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


def chart_beta_rolling(df_port: pd.DataFrame, spy_col: str = "buy_hold",
                       methods: list = None, height: int = 320) -> go.Figure:
    """Rolling 60-day OLS beta to SPY for selected methods."""
    spy_ret = df_port[spy_col].pct_change().dropna() if spy_col in df_port.columns else None
    if spy_ret is None or len(spy_ret) < 60:
        return None
    fig = go.Figure()
    if methods is None:
        candidates = [c for c in df_port.columns if c != spy_col and c != "buy_hold"]
        methods = sorted(candidates, key=lambda c: df_port[c].iloc[-1], reverse=True)[:3]
    for col in methods:
        if col not in df_port.columns:
            continue
        ret = df_port[col].pct_change().dropna()
        aligned = pd.concat([ret, spy_ret], axis=1).dropna()
        aligned.columns = ["strat", "spy"]
        if len(aligned) < 60:
            continue
        rolling_beta = (aligned["strat"].rolling(60).cov(aligned["spy"]) /
                        aligned["spy"].rolling(60).var())
        fig.add_trace(go.Scatter(
            x=rolling_beta.index, y=rolling_beta.values,
            name=get_label(col), line=dict(color=get_color(col), width=1.5),
        ))
    fig.add_hline(y=0, line_color="#555", line_dash="dot")
    fig.add_hline(y=1, line_color="#555", line_dash="dot",
                  annotation_text="  β=1 (index)", annotation_font_color="#666")
    fig.update_layout(**_layout(height=height, dragmode="pan",
        uirevision="beta_rolling",
        title=dict(text="Rolling 60-Day Beta to S&P 500", font=dict(size=13)),
        yaxis=dict(title="Beta", fixedrange=False),
        xaxis=dict(fixedrange=False)))
    return fig


def chart_active_return(df_port: pd.DataFrame, spy_col: str = "buy_hold",
                        methods: list = None, height: int = 320) -> go.Figure:
    """Cumulative active return (strategy return minus beta × SPY return)."""
    spy_ret = df_port[spy_col].pct_change().dropna() if spy_col in df_port.columns else None
    if spy_ret is None or len(spy_ret) < 60:
        return None
    fig = go.Figure()
    if methods is None:
        candidates = [c for c in df_port.columns if c != spy_col and c != "buy_hold"]
        methods = sorted(candidates, key=lambda c: df_port[c].iloc[-1], reverse=True)[:3]
    for col in methods:
        if col not in df_port.columns:
            continue
        ret = df_port[col].pct_change().dropna()
        aligned = pd.concat([ret, spy_ret], axis=1).dropna()
        aligned.columns = ["strat", "spy"]
        if len(aligned) < 60:
            continue
        beta = (aligned["strat"].cov(aligned["spy"]) / aligned["spy"].var())
        active_ret = aligned["strat"] - beta * aligned["spy"]
        cum_active = (1 + active_ret).cumprod()
        fig.add_trace(go.Scatter(
            x=cum_active.index, y=cum_active.values,
            name=get_label(col), line=dict(color=get_color(col), width=1.5),
        ))
    fig.add_hline(y=1.0, line_color="#555", line_dash="dot",
                  annotation_text="  zero active return", annotation_font_color="#666")
    fig.update_layout(**_layout(height=height, dragmode="pan",
        uirevision="active_return",
        title=dict(text="Cumulative Active Return (vs SPY Beta)", font=dict(size=13)),
        yaxis=dict(title="Cumulative Active Growth", fixedrange=False),
        xaxis=dict(fixedrange=False)))
    return fig


def chart_dead_weight(df_dw: pd.DataFrame, height: int = 380) -> go.Figure:
    """Horizontal bar chart of dead-weight % by ticker."""
    if df_dw is None or df_dw.empty:
        return None
    df = df_dw.copy()
    # Detect dead_weight column name
    dw_col = "dead_weight_pct" if "dead_weight_pct" in df.columns else \
             next((c for c in df.columns if "dead" in c.lower()), None)
    tick_col = "ticker" if "ticker" in df.columns else df.columns[0]
    if dw_col is None:
        return None
    df = df.sort_values(dw_col, ascending=True)
    colors = ["#e06c75" if v > 0.45 else "#e5c07b" if v > 0.40 else "#98c379"
              for v in df[dw_col]]
    fig = go.Figure(go.Bar(
        x=df[dw_col] * 100, y=df[tick_col],
        orientation="h", marker_color=colors,
        text=[f"{v*100:.1f}%" for v in df[dw_col]], textposition="outside",
    ))
    fig.add_vline(x=50, line_color="#e06c75", line_dash="dash",
                  annotation_text="  Random (50%)", annotation_font_color="#e06c75")
    fig.update_layout(**_layout(height=height, dragmode="pan",
        uirevision="dead_weight",
        title=dict(text="Dead Weight by Ticker (Long Signal on Down Days)", font=dict(size=13)),
        xaxis=dict(title="Dead Weight %", fixedrange=False),
        yaxis=dict(fixedrange=True),
        margin=dict(l=60, r=60)))
    return fig
