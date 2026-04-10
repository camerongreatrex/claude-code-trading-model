---
name: Trading Strategy Logic
description: Strategy audit results, live strategy, cleanup decisions, and key architectural changes (Apr 2026)
type: project
---

## Live Strategy (as of 2026-04-08)

**Active: `multi_mom_tilt`** (switched from `portable_carry`)

- OOS Sharpe 1.473, IS Sharpe 1.164, IS-OOS gap -0.309
- Ann return 9.5%, max drawdown -7.4%, composite score 0.706
- Sizing: ATR + 30% cross-sectional momentum tilt, no hedge overhead

**Why:** Strategy audit showed portable_carry composite score 0.526 despite highest OOS Sharpe (1.613), because IS-OOS gap of -0.673 indicates regime concentration (hedge only helps in 2020/2022 OOS windows). multi_mom_tilt wins on robustness, raw return, and simplicity.

## Strategy Audit (2026-04-08)

Top 5 retained strategies (from composite score formula):
1. multi_mom_tilt — OOS 1.399 / composite 0.775
2. multi_equal_weight — OOS 1.348 / composite 0.753
3. adaptive_blend — OOS 1.395 / composite 0.745
4. multi_atr_pure — OOS 1.348 / composite 0.738
5. regime_adaptive — OOS 1.384 / composite 0.708

Composite score = 0.35×norm(oos_sharpe) + 0.25×norm(ann_ret) + 0.20×norm(capture_ratio) + 0.20×(1-norm(abs_is_oos_gap))

Removed (commented out, not deleted): equal_weight, risk_parity, rp_regime_aware, rp_regime_dw, rp_blend, multi_mom_portable, multi_mom_port_low, multi_mom_carry, portable_carry

## Architecture Changes (2026-04-08)

### Runtime Reduction
- portfolio.py `all_methods` trimmed from 14 → 5 strategies
- Walk-forward runtime: 48.5s → 17.8s (63% reduction)

### INDEX_ETF_CAP
- Added `INDEX_ETF_CAP = 0.08` and `INDEX_ETF_TICKERS = {"SPY", "IWM", "EEM", "EFA", "VWO"}` in both pipeline/portfolio.py and paper_trader.py
- SPY/IWM/EEM capped at 8% of portfolio (was getting 13%+ via ATR sizing due to low ATR relative to price)
- Applies in: portfolio.py `atr_sizes()` (after per-ticker loop, before gross cap) and paper_trader.py `_compute_position_size()` (before final return)

### Tier Cleanup
- TIER_SHOW = {multi_mom_tilt, multi_equal_weight, adaptive_blend, multi_atr_pure, regime_adaptive, buy_hold}
- TIER_AVAILABLE = set() (empty — all non-top-5 hidden from dashboard)

### Key Files Changed
- pipeline/portfolio.py — all_methods, atr_sizes(), constants
- paper_trader.py — STRATEGIES dict, _OOS_NAME_MAP, _compute_position_size()
- ui/styles.py — TIER_SHOW, TIER_AVAILABLE
- strategy_audit.py — new audit script in repo root

**Why:** 
- IS-OOS gap > 0.40 negative = regime-concentrated OOS performance that will revert in live trading
- SPY/index ETF 8% cap redirects capital to higher-alpha sector ETFs and stocks
- Fewer strategies in walk-forward = faster daily pipeline runs
