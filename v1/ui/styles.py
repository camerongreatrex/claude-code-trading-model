# ui/styles.py
# CSS, Plotly layout helper, palette and labels.

import plotly.graph_objects as go  # noqa: F401 — available for callers

from v1.config.params import V1_PRODUCTION_METHOD

# ── CSS (dark tone-on-tone) ──────────────────────────
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

# ── Plotly layout defaults ─────────────────────────────────────────────
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
    """Merge _LAYOUT with overrides; deep-merges dict values."""
    base = dict(_LAYOUT)
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base

# ── Palette and labels ───────────────────────────────────
# PALETTE: sizing-method key → hex colour (consistent across charts).
# "pos"/"neg" are generic green/red for bars and P&L cells.
PALETTE = {
    "equal_weight"          : "#4a9eff",
    "atr_sized"             : "#50fa7b",
    "atr_pca_macro"         : "#bd93f9",
    "eq_dd_control"         : "#ffb86c",
    "vol_target"            : "#ff79c6",
    "buy_hold"              : "#6272a4",
    "pos"                   : "#50fa7b",
    "neg"                   : "#ff5555",
    "multi_equal_weight"    : "#56b6c2",
    "multi_atr_pure"        : "#e06c75",
    "multi_atr_macro"       : "#d19a66",
    "multi_fast_atr"        : "#61afef",
    "fast_atr"              : "#98c379",
    "multi_mom_tilt"        : "#c678dd",
    "multi_fast_mom_tilt"   : "#e5c07b",
    "multi_fast_atr_vol"    : "#4ec9b0",
    "multi_fast_mom_vol"    : "#b5cea8",
    "pair_atr"              : "#be5046",
    "multi_pair_atr"        : "#7c3aed",
    "rp_macro"              : "#f472b6",
    "multi_fast_atr_earn"   : "#9d8ecb",
    "multi_fast_mom_earn"   : "#6b8ecb",
    "half_kelly"            : "#e2b86d",
    "atr_pca"               : "#56d6a0",
    "composite_vol_target"  : "#a78bfa",
    "risk_parity"           : "#f9a8d4",
    "rp_regime_aware"       : "#34d399",
    "rp_regime_vix"         : "#0891b2",
    "rp_regime_dw"          : "#0e7490",
    "rp_regime_vix_dw"      : "#155e75",
    "rp_blend"              : "#fbbf24",
    "signal_gated_mv_regime": "#818cf8",
    "ir_optimized"          : "#94a3b8",
    "ensemble_atr_pca_macro": "#fb923c",
    "hrp"                   : "#00b4d8",
    "multi_hrp"             : "#48cae4",
    "multi_hrp_mom"         : "#e63946",
    "regime_adaptive"       : "#f59e0b",
    "adaptive_blend"        : "#10b981",
    "multi_mom_portable"    : "#06b6d4",
    "multi_mom_port_low"    : "#0284c7",
    "multi_mom_carry"       : "#a3e635",   # lime
    "portable_carry"        : "#facc15",   # yellow
    "dbmf"                  : "#06b6d4",   # cyan-500  — managed futures
    "wtmf"                  : "#0e7490",   # cyan-700  — managed futures (conservative)
    "top11_vt14sm25_x1.5_cap1": "#22c55e", # green-500 — prior zero-lev candidate
    "top11_adx22_momt_ac55_cap1": "#16a34a", # green-600 — production (zero-leverage)
}
# LABELS: keys → legend/table strings.
LABELS = {
    "equal_weight"          : "Equal Weight",
    "atr_sized"             : "ATR Sized",
    "atr_pca_macro"         : "ATR + PCA + Macro",
    "eq_dd_control"         : "Equal Wt + DD Control",
    "vol_target"            : "Vol Target",
    "buy_hold"              : "Buy & Hold",
    "multi_equal_weight"    : "Multi Equal Weight",
    "multi_atr_pure"        : "Multi ATR",
    "multi_atr_macro"       : "Multi ATR + Macro",
    "multi_fast_atr"        : "Multi Fast ATR",
    "fast_atr"              : "Fast ATR",
    "multi_mom_tilt"        : "Momentum Tilt",
    "multi_fast_mom_tilt"   : "Fast Momentum Tilt",
    "multi_fast_atr_vol"    : "Multi Fast ATR + Vol",
    "multi_fast_mom_vol"    : "Fast Momentum + Vol",
    "pair_atr"              : "Pair ATR",
    "multi_pair_atr"        : "Multi Pair ATR",
    "rp_macro"              : "Risk Parity + Macro",
    "multi_fast_atr_earn"   : "Multi Fast ATR (Earn)",
    "multi_fast_mom_earn"   : "Multi Fast Mom (Earn)",
    "half_kelly"            : "Half-Kelly",
    "atr_pca"               : "ATR + PCA",
    "composite_vol_target"  : "Composite + Vol Target",
    "risk_parity"           : "Risk Parity",
    "rp_regime_aware"       : "RP Regime Aware",
    "rp_regime_vix"         : "RP Regime + VIX",
    "rp_regime_dw"          : "RP Regime + DW",
    "rp_regime_vix_dw"      : "RP Regime + VIX + DW",
    "rp_blend"              : "RP Blend",
    "signal_gated_mv_regime": "Signal-Gated MV",
    "ir_optimized"          : "IR Optimized",
    "ensemble_atr_pca_macro": "Ensemble ATR+PCA+Macro",
    "hrp"                   : "HRP",
    "multi_hrp"             : "Multi HRP",
    "multi_hrp_mom"         : "Multi HRP + Mom",
    "regime_adaptive"       : "Regime Adaptive",
    "adaptive_blend"        : "Adaptive Blend",
    "multi_mom_portable"    : "Portable Alpha",
    "multi_mom_port_low"    : "Portable Alpha (Low Beta)",
    "multi_mom_carry"       : "Momentum + Carry",
    "portable_carry"        : "Portable Carry",
    "top11_vt14sm25_x1.5_cap1": "Top-11 VT (Zero-Lev)",
    "top11_adx22_momt_ac55_cap1": "Top-11 ADX+Momt+AC (Zero-Lev) ★",
}

# ── Display tiers ───────────────────────────────────────────────────
# TIER_SHOW: shown by default (max 8 for readability).
# TIER_AVAILABLE: hidden, selectable via multiselect.
# Else: hidden from dashboard (still computed in portfolio.py).

TIER_SHOW = {
    # Production (pinned 2026-04-28): top11_adx22_momt_ac55_cap1 — zero-lev top-N
    # + ADX22 + 63d momentum tilt + 55% asset-class quota. OOS 21.00% / 2.324 Sh
    # / -6.33 DD across 2022-2025. Others are zero-lev legacy candidates.
    V1_PRODUCTION_METHOD,       # PRODUCTION ★ — OOS Sh 2.32
    "top11_vt14sm25_x1.5_cap1", # Prior zero-lev candidate — OOS Sh 2.22
    "multi_mom_tilt",           # OOS Sh 1.56 — momentum tilt
    "multi_equal_weight",       # OOS Sh 1.47 — equal weight
    "multi_atr_pure",           # OOS Sh 1.57 — ATR sized
    "adaptive_blend",           # OOS Sh 1.57 — blend
    "regime_adaptive",          # OOS Sh 1.58 — regime adaptive
    "buy_hold",                 # Benchmark
}

TIER_AVAILABLE = set()   # all non-top-5 hidden; add back via set for research

# Anything not in TIER_SHOW/TIER_AVAILABLE is hidden from the dashboard
# (half_kelly, ir_optimized, vol_target, atr_*, hrp, etc.).


def get_color(method_key: str) -> str:
    return PALETTE.get(method_key, "#888888")


def get_label(method_key: str) -> str:
    return LABELS.get(method_key, method_key.replace("_", " ").title())
