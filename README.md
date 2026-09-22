# Algorithmic Trading System — V4N-F

A systematic, fully-automated paper trading system that picks a small basket of ETFs and stocks every day, sizes positions with risk-aware overlays, and runs hands-off until you tell it to stop. The current production strategy is **V4N-F** — a zero-leverage, top-11 trend-following stack with five sizing overlays and a bull-market sector swap.

**Out-of-sample walk-forward (2022–2025):**
Sharpe **2.63** · Annual return **24.78%** · Max drawdown **−4.05%** · Calmar **6.11**

---

## Contents

1. [What this thing actually does](#what-this-thing-actually-does)
2. [The strategy in plain English (V4N-F)](#the-strategy-in-plain-english-v4n-f)
3. [Quick start](#quick-start)
4. [Daily operation](#daily-operation)
5. [The 3-month live test (V4N-F)](#the-3-month-live-test-v4n-f)
6. [Reading the dashboard](#reading-the-dashboard)
7. [Files and folders](#files-and-folders)
8. [Pipeline commands](#pipeline-commands)
9. [Troubleshooting](#troubleshooting)

---

## What this thing actually does

Every weekday after the market closes, the system:

1. **Downloads** the day's prices for ~37 tickers (ETFs, sector funds, big stocks).
2. **Scores** each one using trend signals (moving averages, momentum, volatility).
3. **Picks the top 11** names that pass the trend filter.
4. **Sizes each position** so no single name or asset class dominates.
5. **Applies five overlays** that scale up or down depending on the market regime.
6. **Buys / sells** what's needed to match the target portfolio (paper money — no real broker).
7. **Saves the result** to disk and updates a live dashboard.

You don't have to do anything once it's running. A background scheduler handles it.

---

## The strategy in plain English (V4N-F)

V4N-F is the production sizer. The full key is `top11_adx22_momt_ac55_cap1`. Here's what each piece does, in order:

### Base: Top-11 trend-following

Out of ~37 tickers, only the 11 with the strongest trend signal get money. Everything else is ignored that day. The trend signal is a combination of moving-average crossover, Donchian breakout, and a dip-buy confirmation.

### Five overlays applied on top

| # | Name | What it does | Why it helps |
|---|---|---|---|
| **A** | **ADX ≥ 22 filter** | Drops names whose trend strength (ADX) is below 22 *before* picking the top 11. | A name can look like it's trending but really just be drifting sideways. ADX confirms the trend has real force. |
| **B** | **63-day momentum tilt** | Inside the 11 longs, names with the strongest 3-month return get up to 30% more dollars; weakest get up to 30% less. | Winners keep winning — concentrate dollars in the strongest movers without dropping the diversifying names entirely. |
| **C** | **55% asset-class quota** | No single asset class (e.g. tech sector, bonds, commodities) can exceed 55% of total dollars. | Stops the portfolio from becoming "100% tech" in a tech bull run. Forces some diversification. |
| **G** | **Asymmetric vol boost** | When VIX is *low* (calm market) → scale gross exposure by 1.15×. When VIX is *high* (fear) → scale by 0.9×. | Lean in when markets are calm, pull back slightly when they're not. Asymmetric on purpose — you keep more upside than you give back in vol. |
| **H** | **Fear-regime top-RS concentration** | When VIX is high (≥ +1 z-score), drop the weakest longs and redistribute their notional equally to the **top-3 by 63-day relative strength**. | In a fear regime, mediocre names get crushed. Concentrate into the few names actually holding up. |
| **I** | **Acceleration kicker** | Names whose 10-day return / 42-day return ratio ≥ 1.05 get boosted by 1.35×. Gross is renormalized so total dollars don't grow. | Catches names that are *accelerating* — pure tilt, no extra leverage. |
| **J** | **Bull-regime sleeve swap** *(this is the V4N-F bit)* | When SPY is above its 50-day average **and** VIX z-score ≤ 0, shave up to 10% of capital from TLT (long bonds) and rotate it into XLK (tech sector). Capped at 12% per name. | Top-N momentum strategies structurally lag the broad market in calm bull runs. Swapping some bond exposure for tech closes ~1pp/yr of the SPY deficit without raising drawdown. |

### What "zero leverage" means

Total gross exposure is capped at 100% of capital. The system never borrows money. If overlays would push gross above 1.0, they're scaled back down. This is what makes the drawdown so small (−4% max).

---

## Quick start

**1. Clone and install**

```bash
git clone https://github.com/camerongreatrex/Algorithmic-Trading-Model.git
cd Algorithmic-Trading-2
pip install -r requirements.txt
```

**2. Run the full pipeline** — downloads data, generates signals, runs backtests

```bash
python run.py
```

**3. Initialize the paper portfolio**

```bash
PYTHONPATH=. python v1/scripts/paper_trader.py init
```

**4. Start the scheduler** in one terminal

```bash
PYTHONPATH=. python v1/scripts/scheduler.py
```

**5. Open the dashboard** in a second terminal

```bash
PYTHONPATH=. streamlit run v1/ui/dashboard.py
```

Opens at `http://localhost:8501`.

---

## Daily operation

**Do nothing.** The scheduler handles everything automatically:

- Checks the clock every 30 seconds.
- At **4:45 PM ET** on weekdays, fetches live prices, recomputes signals, executes paper trades, saves state.
- Prints a heartbeat every 30 minutes.
- If your PC was off, replays missed weekdays automatically on startup.

### Catch-up on restart

When you reopen your laptop after time off:

1. Scheduler starts → compares `last_eod_date` against today (ET).
2. Any missed weekdays are replayed using yfinance historical data.
3. Today is **not** replayed until 4:45 PM (market still open).

### GitHub Actions backup

`.github/workflows/eod.yml` runs the same EOD update on GitHub's servers as a backup when your PC is off. Cron fires at 20:45 and 21:45 UTC (covers DST). A duplicate-day guard prevents double processing.

---

## The 3-month live test (V4N-F)

V4N-F runs as an isolated 3-month forward test (2026-05-11 → 2026-08-11) using a separate `PT_INSTANCE`. This keeps it from touching the main paper-trading state.

### How instance routing works

The paper trader reads `PT_INSTANCE` from the environment and routes all reads/writes to `data/v1/paper_trading_{instance}/`. So setting `PT_INSTANCE=v4nf_3mo` writes to `data/v1/paper_trading_v4nf_3mo/`.

### Run the V4N-F instance

```bash
PT_INSTANCE=v4nf_3mo PYTHONPATH=. python v1/scripts/scheduler.py
```

Or use the instance wrapper (same CLI as the main paper trader):

```bash
PYTHONPATH=. python v1/scripts/paper_trader_v4nf.py status
PYTHONPATH=. python v1/scripts/paper_trader_v4nf.py catchup --fill-gaps
```

The scheduler watches this instance separately. The dashboard's **V4N-F 3-Month Test** tab reads from the same folder.

### Reinitialize from scratch

```bash
PT_INSTANCE=v4nf_3mo PYTHONPATH=. python v1/scripts/_init_v4nf_at_open.py
```

This wipes the instance's `state.json` / `history.csv` / `trades.csv`, buys each position at today's **open** price, marks the portfolio to today's **close**, and saves a day-1 history row containing the open→close return.

---

## Reading the dashboard

The dashboard has several top-level tabs. The two most important:

### V4N-F 3-Month Test → Portfolio

- **Top status row:** Portfolio value, cash, invested, open positions, days elapsed.
- **Line chart:** Portfolio equity curve since inception, with a dashed line at $100,000 (starting capital). Anything above the dashed line is profit.

### V4N-F 3-Month Test → vs S&P 500

- **Three big numbers:** Your portfolio's total return, the S&P 500's total return over the same window, and the alpha (your return minus SPY's).
- **Chart:** Both equity curves normalized to $100k at the May-11 market open. The day-1 open→close gain is included in both series.
- **Daily breakdown:** Day-by-day table — your return, SPY's return, and daily alpha.

> Both series anchor at $100k at the May-11 **open**, not close. This is why the headline "Portfolio Return" includes the day-1 open→close gain.

### Other tabs

| Tab | What it shows |
|---|---|
| **Overview** | Long-run backtest comparison: V4N-F vs every other candidate strategy ever tested |
| **Live Signals** | Today's signal scores for every ticker — which ones are firing right now |
| **Positions** | Current paper-trader holdings, cost basis, unrealized P&L |
| **Alpha Decomposition** | OOS Active Sharpe, OLS beta vs SPY, dead-weight scoring |
| **Walk-Forward** | Out-of-sample results from rolling 3-year-train / 1-year-test windows |

---

## Files and folders

```
Algorithmic-Trading-2/
├── run.py                       # Pipeline orchestrator
├── README.md                    # This file
├── requirements.txt
├── data/
│   └── v1/
│       ├── paper_trading/        # Main paper trader state
│       └── paper_trading_v4nf_3mo/  # V4N-F 3-month test instance
│           ├── state.json        # Cash, positions, last EOD date
│           ├── history.csv       # Daily PV snapshots
│           └── trades.csv        # Every buy/sell ever executed
└── v1/
    ├── config/params.py          # V1_PRODUCTION_METHOD pin lives here
    ├── pipeline/                 # Data → features → signals → backtest
    ├── portfolio/portfolio.py    # Sizing logic (where the overlays live)
    ├── scripts/
    │   ├── paper_trader.py       # Main paper-trader engine
    │   ├── paper_trader_v4nf.py  # V4N-F instance entry point
    │   ├── scheduler.py          # 4:45 PM ET daily runner
    │   ├── live_signals.py       # Real-time signal compute
    │   └── _init_v4nf_at_open.py # One-shot V4N-F init at OPEN price
    └── ui/
        ├── dashboard.py          # Streamlit dashboard
        └── styles.py             # CSS, palette, strategy labels
```

### Key state files

| File | Contents |
|---|---|
| `state.json` | Current cash, open positions (shares, cost basis, last close), last EOD date |
| `trades.csv` | Append-only log — every BUY and SELL with price, value, commission, P&L |
| `history.csv` | Daily snapshot — date, portfolio value, cash, invested, n_positions, daily return |

---

## Pipeline commands

| Command | What it does |
|---|---|
| `python run.py` | Full rebuild: download → features → signals → backtest → portfolio |
| `python run.py signals` | Skip downloads, re-run from feature engineering |
| `python run.py backtest` | Re-run backtester + portfolio only |
| `PYTHONPATH=. python v1/scripts/paper_trader.py status` | Print current paper-trader state |
| `PYTHONPATH=. python v1/scripts/paper_trader.py run` or `eod` | Force an EOD update for today |
| `PYTHONPATH=. python v1/scripts/paper_trader.py catchup` | Replay missed days since last EOD |
| `PYTHONPATH=. python v1/scripts/paper_trader.py catchup --fill-gaps` | Fill holes in history, then tail catch-up |
| `PYTHONPATH=. python v1/scripts/paper_trader_v4nf.py catchup --fill-gaps` | Same for the 3-month V4N-F instance |

---

## Troubleshooting

**Dashboard shows "Paper trading not yet initialised."**
Run `PYTHONPATH=. python v1/scripts/paper_trader.py init` (or `_init_v4nf_at_open.py` for the V4N-F instance).

**Numbers between subtabs disagree.**
The Portfolio subtab and the vs-S&P subtab both anchor at the May-11 **open**. If one shows a different return, the dashboard is probably caching a stale fragment — hard-refresh the browser.

**Catch-up replays wrong prices.**
Catch-up uses yfinance adjusted closes. If yfinance has a bad day, re-run `paper_trader.py eod` or delete the bad row in `history.csv` and re-run.

**Too many rebalance trades / flat P&L since daily rebalance.**
Live execution defaults to `signal_only` + `top_n=11` in `v1/config/params.py`. Walk-forward execution study (3y/1y OOS, fee-aware):

```bash
PYTHONPATH=. python v1/scripts/walkforward_execution_study.py
```

Results saved to `data/v1/results/walkforward_execution_study.json`. On the current 42-ticker panel, **top11 + signal_only** ranked first: ~20% OOS net ann, ~640 trades/year, **0% rebalance churn** vs ~35% for daily. After adding Phase-9 tickers in `data_pipeline.py`, run `python run.py` and re-run the study.

**Strategy not updating after a config change.**
Changes to `V1_PRODUCTION_METHOD` take effect on the **next** EOD run. Open positions are not retroactively re-sized.

**GitHub Actions run failed.**
Check the **Actions** tab. Most common cause: a `git push` conflict (you pushed locally at the same time). Re-run manually from the Actions tab.
