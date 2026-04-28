"""
config.py
---------
Global configuration toggle between v1 (trend-following) and v2 (macro regime rotation).
"""

# Active model version: "v1" or "v2"
MODEL_VERSION = "v2"

# Shared constants
CAPITAL = 100_000
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"

# ── V1 production method (single source of truth) ────────────────────────────
# Pinned 2026-04-28 (V2): top11_adx22_momt_ac55_cap1 — zero-leverage top-N
# concentration with three overlay refinements:
#   A. ADX≥22 filter — drop weak trends entirely before top-N selection.
#   B. 63d momentum tilt (0.7-1.3 percentile-rank scale) — within selected
#      longs, size strongest 3-month performers up to 1.3x and weakest down
#      to 0.7x.
#   C. 55% asset-class quota — no class > 55% of gross to force diversification.
# Walk-forward OOS (2022-2025, 3 windows): 21.00% AnnRet / 2.324 Sharpe /
# -6.33 worst DD — +0.79pp AnnRet, +0.10 Sharpe, +0.62pp DD reduction vs
# prior top11 V1 (20.21% / 2.221 / -6.95).  Beats prior 1.5x leverage prod
# (16.95% / 1.570 / -8.32 OOS) on every dimension at zero leverage.  Change
# this constant to swap production everywhere (dashboard tiers, paper trader,
# portfolio.py pin, etc.).
V1_PRODUCTION_METHOD = "top11_adx22_momt_ac55_cap1"

# ── V2 Configuration ──────────────────────────────────────────────────────────
# Macro regime rotation: cross-asset ETF allocation driven by hybrid regime
# classifier (realized macro + market-implied signals).
V2_CONFIG = {
    "target_vol": 0.12,              # 12% annualized vol target
    "vol_scale_cap": 1.5,            # max leverage scalar
    "vol_scale_floor": 0.3,          # min leverage scalar
    "max_dd_trigger": -0.12,         # drawdown circuit breaker threshold
    "dd_exposure_cut": 0.40,         # cut gross exposure by this fraction
    "dd_restore_threshold": -0.08,   # restore when DD recovers above this
    "stress_override_threshold": 2.0, # std devs for stress override
    "max_single_etf": 0.30,          # no single ETF > 30%
    "max_asset_class": 0.60,         # no asset class > 60%
    "rebalance_freq": "M",           # monthly rebalance
}
