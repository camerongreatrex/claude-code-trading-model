---
name: V2 Beta-Neutral Model
description: Complete restructure to beta-neutral cross-sectional equity system with S&P 500 universe, walk-forward validated Apr 2026
type: project
---

V2 model built as a parallel system alongside v1 (trend-following). v2 is a dollar-neutral long/short strategy on ~500 S&P 500 stocks with weekly rebalance.

**Architecture:** v1/ (archived original), v2/ (new beta-neutral), shared/ (common modules), config.py (MODEL_VERSION toggle)

**Walk-forward results (Apr 2026):**
- Window 1 (2025): OOS Sharpe +0.271 (PASS)
- Window 2 (2024): OOS Sharpe +1.297 (PASS)
- Window 3 (2023): OOS Sharpe -1.124 (FAIL — momentum reversal year post-2022 bear)
- Full-sample: Sharpe 0.28, beta -0.011, corr to SPY -0.027, max DD -13.1%

**Why:** v1 was a slow trend-following long-only system that underperformed SPY on up days. Beta neutrality unlocks S&P 500 universe since we're no longer fighting market beta.

**How to apply:** Window 3 failure is a known momentum crash exposure. The model is structurally sound but needs monitoring during momentum reversal periods. User may choose to proceed to paper trading or adjust signal weights.

**Key files:** v2/universe.py, v2/signal_generation.py, v2/portfolio.py, v2/rebalancer.py, v2/risk_model.py, v2/backtester.py, v2/paper_trader.py, v2/data_pipeline.py
