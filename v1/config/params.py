"""
config.py — global params for v1 (trend-following).
"""

CAPITAL = 100_000
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"

# ── V1 production method (single source of truth) ────────────────────────────
# Pinned 2026-05-05 (V4N-D): V4N-B stack + Family-O asym vol boost +
# Family-S4 fear-regime top-RS concentration.  Zero-leverage.
#   A. ADX≥22 filter — drop weak trends before top-N selection.
#   B. 63d momentum tilt (0.7-1.3) — within longs, scale by 3m return rank.
#   C. 55% asset-class quota — no class > 55% of gross.
#   D. 12% diversifier sleeve — equal-weight TLT/GLD/DBMF/VGSH always-on.
#   E. Profit-take overlay — scale a position by 0.7 if its 10d cum/std
#      z-score >= 1.5 (take 30% off parabolic moves).
#   F. Conditional vol-carry — scale gross by 0.5 when vix_zscore >= 1.5
#      AND VIX rising over the last 5 days (avoid cutting after the storm).
#   G. Asymmetric vol boost (Family O) — scale gross by 1.15× when
#      vix_zscore <= -0.5 (calm), by 0.9× when >= +1.0 (fear).
#   H. Fear-regime top-RS concentration (Family S4) — when vix_zscore >= +1.0,
#      drop bottom longs, redistribute notional equally to top-3 by 63d RS.
# OOS walk-forward (1610 days, OOS warmup 756d):
#   V4N-D: Sh 2.52 / Ann 23.18% / DD -4.20% / Cal 5.52
#   V4N-B: Sh 2.55 / Ann 19.05% / DD -4.49% / Cal 4.24
#   = +4.13pp AnnRet, -0.29pp DD, +1.28 Calmar.  Wins 6 of 7 calendar windows
#   (only loss: Bear 2022 H1 -1.1pp vs O baseline).  Anti-overfit gate:
#   bull windows >= O-0.3pp AND ≥2 stress windows beat O+0.5pp — passes.
# Single source of truth — change here to swap everywhere (dashboard, paper
# trader, portfolio.py pin, etc.).
V1_PRODUCTION_METHOD = "top11_adx22_momt_ac55_cap1"
V1_PRODUCTION_LABEL  = "V4N-D"

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
