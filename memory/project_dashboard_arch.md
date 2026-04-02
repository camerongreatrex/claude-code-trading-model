---
name: Dashboard Architecture
description: Tab structure, fragment setup, key functions, data flow for dashboard.py and paper_trader.py
type: project
---

## Tab Structure (6 tabs as of Mar 26 2026)
```python
tab_port, tab_sig, tab_sp, tab_bt, tab_val, tab_risk = st.tabs([
    "  📈  Portfolio",
    "  📡  Signals",
    "  📊  vs S&P 500",
    "  🔬  Backtest",
    "  🔍  Validation",
    "  ⚠️  Risk",
])
```
- `tab_val` has sub-tabs: Walk-Forward, Alpha Decomposition
- `tab_risk` has sub-tabs: Monte Carlo, Macro Overlay
- Portfolio tab has NO S&P benchmark line (removed — S&P comparison lives only in `tab_sp`)

## Fragment Setup
- `_portfolio_fragment()` — `@st.fragment(run_every=5)` — Portfolio tab only
- `_signals_fragment()` — `@st.fragment(run_every=60)` — Signals tab only

## Key Functions (paper_trader.py)
- `_resolve_prev_pv(state, history)` — returns yesterday's EOD portfolio value; single source of truth used by both `get_intraday_curve()` and `compute_live_portfolio_metrics()`
- `get_intraday_curve()` — returns `(intraday_df, live_prices, spy_curve, spy_pct_from_prev)`
- `compute_live_portfolio_metrics(state, history, live_prices)` — single source of truth for all portfolio numbers shown in dashboard

## EOD Data Fix (Mar 26 2026)
- EOD was storing stale pipeline prices (morning fetch) as "close" — fixed by overlaying fresh yfinance daily closes for held positions after signal computation
- `state.json` and `history.csv` March 26 row corrected: portfolio_value $100,205 → $98,717.61

## Top Metrics Row (8 columns)
c1=Portfolio Value, c2=Daily Return, c3=Total Return, c4=Sharpe, c5=Max DD, c6=Win Rate, c7=Profit Factor, c8=Active Sharpe

## Variable Naming
- `_prod_method` — underscore key e.g. `"multi_mom_tilt"` (used for df column lookup)
- `_prod_oos_key` — raw oos_sel method name e.g. `"multi mom tilt"` (used for oos_sel row lookup)

## Known Correct Values (Mar 26 2026 EOD)
- ^GSPC: 6477.16 (-1.741% from Mar 25's 6591.90)
- Portfolio: $98,717.61 (down -0.39% from $99,107.23)
- Portfolio beats S&P by ~1.35pp on Mar 26
