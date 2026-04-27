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
# Pinned 2026-04-27: atr_lev_1.5x with the pl_5_10 + ts_40 profit-lock /
# time-stop overlay (signal_generation.apply_profit_lock_timestop).  Pareto-
# dominates atr_pure on AnnRet / Sharpe / MaxDD / Calmar; IS-OOS gap ~-0.22.
# Change this single constant to swap the production method everywhere
# (dashboard tiers, paper trader, portfolio.py pin, etc.).
V1_PRODUCTION_METHOD = "atr_lev_1.5x"

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
