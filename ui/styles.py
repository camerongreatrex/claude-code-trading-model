# ui/styles.py
# CSS constant, Plotly layout helper, shared colour palette and display labels.

import plotly.graph_objects as go  # noqa: F401 — available for callers

# ── Fine-grained CSS tweaks (dark grey tone-on-tone) ──────────────────────────
CSS = """
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
  /* Prevent fragment grey-out on 5s auto-refresh */
  [data-stale="true"] { opacity: 1 !important; transition: none !important; }
  [data-stale="true"] * { opacity: 1 !important; transition: none !important; }
</style>
"""

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

# ── Chart colour palette and display labels ───────────────────────────────────
# PALETTE maps sizing-method keys to hex colours used consistently across all
# chart traces so each method always appears in the same colour.
# "pos"/"neg" are generic green/red used for bar charts, P&L cells, etc.
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
# LABELS maps the same keys to human-readable legend/table strings.
LABELS = {
    "equal_weight"  : "Equal Weight",
    "atr_sized"     : "ATR Sized",
    "atr_pca_macro" : "ATR + PCA + Macro",
    "eq_dd_control" : "Equal Wt + DD Control",
    "vol_target"    : "Vol Target",
    "buy_hold"      : "Buy & Hold",
}
