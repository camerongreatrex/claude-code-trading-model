"""
config.py — global params for v1 (trend-following).
"""

CAPITAL = 100_000
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"

# ── V1 production method (single source of truth) ────────────────────────────
# Pinned 2026-05-09 (V4N-F): V4N-E stack + bull-regime sleeve swap.
# Zero-leverage.
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
#   I. Acceleration kicker (Family A) — within active longs, boost names where
#      10d/42d cum log return ratio >= 1.05 by 1.35×, renormalize to preserve
#      gross.  Pure tilt, no leverage added.
#   J. Bull-regime sleeve swap (V4N-F) — when SPY > 50dMA AND vix_zscore <= 0,
#      shave up to 10% of capital from TLT and add to XLK to capture mega-cap
#      participation in broad bull markets.  Capped at 12% per-name; never
#      shorts TLT.
# OOS walk-forward (1610 days, OOS warmup 756d):
#   V4N-F: Sh 2.63 / Ann 24.78% / DD -4.05% / Cal 6.11
#   V4N-E: Sh 2.58 / Ann 23.98% / DD -4.23% / Cal 5.67
#   = +0.80pp AnnRet, +0.18pp DD (less negative), +0.44 Calmar.  Pareto win
#   on Sh/Ann/DD vs V4N-E.  Wins 7/7 calendar windows vs V4N-E baseline.
#   SPY-deficit narrows ~1pp/yr in bull years (2019/20/21) — top-N momo
#   structurally lags broad cap-weighted indices in low-vol bull regimes.
# Single source of truth — change here to swap everywhere (dashboard, paper
# trader, portfolio.py pin, etc.).
V1_PRODUCTION_METHOD = "top11_adx22_momt_ac55_cap1"
V1_PRODUCTION_LABEL  = "V4N-F"

# Live paper-trader execution (no daily top-up/trim).
# rank_rotate: enter top-N on signal; exit on signal=0 or rank drop after min hold.
# Walk-forward OOS (42-ticker panel): ~20% net ann, ~640 trades/yr, Sharpe ~2.0.
LIVE_REBALANCE_MODE  = "signal_only"
LIVE_REBALANCE_DAYS  = 21
LIVE_MIN_HOLD_DAYS   = 5    # min hold before rank_exit (matches signal pipeline)
# Top-N slots. Walk-forward execution study (2026-05): top11 beats top14 on
# net OOS return after fees on current 42-ticker panel; revisit after run.py
# builds features for Phase-9 names.
LIVE_TOP_N           = 11

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
