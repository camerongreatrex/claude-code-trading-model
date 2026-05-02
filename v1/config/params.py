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
# Pinned 2026-05-01 (V4N-B): top11_adx22_momt_ac55_cap1 + 12% diversifier sleeve
# + profit-take overlay + conditional vol-carry overlay.  Zero-leverage.
#   A. ADX≥22 filter — drop weak trends before top-N selection.
#   B. 63d momentum tilt (0.7-1.3) — within longs, scale by 3m return rank.
#   C. 55% asset-class quota — no class > 55% of gross.
#   D. 12% diversifier sleeve — equal-weight TLT/GLD/DBMF/VGSH always-on.
#   E. Profit-take overlay — scale a position by 0.7 if its 10d cum/std
#      z-score >= 1.5 (take 30% off parabolic moves).
#   F. Conditional vol-carry — scale gross by 0.5 when vix_zscore >= 1.5
#      AND VIX rising over the last 5 days (avoid cutting after the storm).
# Walk-forward 1-yr OOS (3 windows, true OOS):
#   V4N-B: 2.56 mean Sh / 19.13% AnnRet / -3.83% DD / 1.98 worst Sh
#   vs V3: 2.33 mean Sh / 19.63% AnnRet / -4.60% DD / 1.62 worst Sh
#   = +0.23 Sh, -0.50pp Ann, -0.77pp DD, +0.36 worst-Sh.  Strict Pareto on
#   risk-adjusted return + worst-case stability.
# 6-month walk-forward stability: std 0.79 (V3 0.99) → much less variable.
# Beats the ML-based V4B (Sh 2.50, Ann 18.31%) on every dimension AND uses
# zero ML — purely deterministic overlays, no model state, no retraining.
# See feedback_position_sizing.md.
# Single source of truth — change here to swap everywhere (dashboard, paper
# trader, portfolio.py pin, etc.).
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
