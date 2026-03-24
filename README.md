# Algorithmic Trading System

A systematic multi-asset strategy trading 37 tickers across equities, bonds, commodities, and sector ETFs. Uses a multi-signal approach combining MA crossover (golden cross), momentum breakout (Donchian), and dip-buy filters, with two-sided signals for bonds and commodities, ATR position sizing, and cross-sectional momentum tilt. Walk-forward validated out-of-sample with best OOS Sharpe of 1.156 (Multi Mom Tilt). Fully automated: the scheduler runs the daily pipeline at market close, catches up missed days when your PC restarts, and a Streamlit dashboard shows live portfolio, signal state, and alpha decomposition analytics.

---

## Quick Start

**1. Clone and install dependencies**
```bash
git clone <repo-url> && cd Algorithmic-Trading-2
pip install -r requirements.txt
```

**2. Run the full pipeline once** (downloads data, engineers features, generates signals, runs backtests — takes ~5–10 min)
```bash
python run.py
```

**3. Initialize the paper portfolio** (enters all current LONG signals at today's prices)
```bash
python paper_trader.py init
```

**4. Start the scheduler** (leave this terminal open — it runs the daily EOD update automatically)
```bash
python scheduler.py
```

**5. Open the dashboard** (in a second terminal)
```bash
streamlit run dashboard.py
```
Opens at http://localhost:8501.

---

## Daily Operation

**Do nothing.** The scheduler handles everything automatically:

- It sleeps in a loop, checking the clock every 30 seconds.
- At **4:45 PM ET** on weekdays it fetches live prices, evaluates signals, executes paper trades, and saves the updated portfolio state.
- A heartbeat prints to the terminal every 30 minutes so you can verify it's alive.

### Two terminals

| Terminal | Command | Purpose |
|---|---|---|
| 1 | `python scheduler.py` | Runs the EOD pipeline on schedule. Leave it running. |
| 2 | `streamlit run dashboard.py` | Live dashboard with charts, positions, signals. |

### What happens when you close your laptop and reopen it

The catch-up system detects missed trading days automatically:

1. On scheduler startup, `catchup()` compares `last_eod_date` in `state.json` against today (ET).
2. Any weekdays strictly between the last processed date and today are replayed in order using historical yfinance data.
3. The current day is **not** replayed — it hasn't closed yet. The normal 4:45 PM run handles it.
4. The dashboard also calls `catchup()` on first load (cached for 1 hour) so charts are up-to-date even if the scheduler hasn't started yet.

**Example**: PC off Thursday through Sunday → Monday 9 AM startup → catch-up replays Thursday and Friday → scheduler waits for Monday's 4:45 PM close.

---

## GitHub Actions Backup

The workflow `.github/workflows/eod.yml` runs the EOD update on GitHub's servers as a backup when your PC is off.

### How it works

- Two cron schedules (20:45 and 21:45 UTC) ensure it fires at ~4:45 PM ET regardless of daylight saving time.
- It runs `catchup()` then `end_of_day_update()` from a clean checkout.
- Updated `state.json`, `history.csv`, and `trades.csv` are committed and pushed back to the repo.
- A duplicate-day guard prevents double processing if both cron times fire on the same day.

### How to enable

Just push to GitHub — the workflow runs on schedule automatically. No secrets or API keys needed (yfinance is free).

### How to verify it ran

Check the **Actions** tab in your GitHub repo. Each run shows logs with the tickers processed and portfolio value.

### Limitations

- `data/macro/macro_features.parquet` is gitignored, so the VIX gate and macro multiplier are inactive in the GH Actions run. The core MA-crossover signal still works correctly.
- If a `git push` fails due to a merge conflict (you pushed at the same time), the Actions run will fail visibly in the Actions tab. Re-run manually or let the next day's run pick it up.

---

## Reset the Paper Portfolio

**1.** Delete the state and history files:
```bash
rm data/paper_trading/state.json
rm data/paper_trading/history.csv
rm data/paper_trading/trades.csv
```

**2.** Re-initialize (enters all current LONG signals):
```bash
python paper_trader.py init
```

**3.** Start the scheduler:
```bash
python scheduler.py
```

The scheduler will process the next EOD at 4:45 PM ET. The dashboard picks up the new state automatically.

---

## Pipeline Shortcuts

| Command | What it does |
|---|---|
| `python run.py` | Full rebuild: download → features → research → macro → signals → backtest → portfolio (~5–10 min) |
| `python run.py signals` | Skip data download, re-run from feature engineering onward |
| `python run.py backtest` | Re-run backtester + portfolio only (seconds) |
| `python run.py portfolio` | Run portfolio stage only (seconds) |
| `python run.py macro` | Re-run from macro features onward |

---

## Architecture Overview

| Module | Purpose |
|---|---|
| `pipeline/data_pipeline.py` | Downloads and cleans OHLCV data from yfinance for 37 tickers |
| `pipeline/feature_engineering.py` | Computes technical features (ATR, RSI, MACD, ADX, Bollinger, OBV, etc.) |
| `pipeline/feature_research.py` | Calculates information coefficients (IC) for feature selection |
| `pipeline/macro_features.py` | Fetches VIX and yield curve data for macro regime filtering |
| `pipeline/signal_generation.py` | Generates MA-crossover signals with asset-class routing and post-processors |
| `pipeline/backtester.py` | Runs full historical backtests with transaction costs |
| `pipeline/portfolio.py` | ATR + PCA + macro position sizing, walk-forward OOS selection |
| `pipeline/risk_model.py` | PCA-based risk decomposition and correlation analysis |
| `pipeline/sensitivity.py` | Parameter sensitivity sweeps for MA crossover windows |
| `pipeline/regime_analysis.py` | Macro regime classification and conditional performance stats |
| `pipeline/correlation_diagnostic.py` | Signal correlation analysis, dead-weight scoring, regime correlation tables |
| `paper_trader.py` | Live paper trading execution engine (buy/sell/catch-up) |
| `scheduler.py` | Automated 4:45 PM ET daily runner with kill switch and order sheet |
| `live_signals.py` | Real-time signal computation from yfinance for the dashboard |
| `dashboard.py` | Streamlit dashboard: equity curves, Monte Carlo, live signals, paper trading |
| `run.py` | Pipeline orchestrator — runs each stage as a subprocess in sequence |

---

## Strategy Summary

- **Universe**: 37 tickers — broad equity (SPY, IWM, EEM, EFA, VWO, EWZ, EWJ, FXI, CCJ), bonds (TLT, HYG, TIP, BWX, EMB), commodities (GLD, DBC, UUP, DBA, FXE, FXY), sectors (XLE, XLU, XLF, VNQ, XLC, XLI, XLK, XLP, XLV), stocks (JPM, JNJ, XOM, AMZN, NEE, BRK-B, GS, COST, MSFT, NVDA, AAPL), survivorship anchors (GE, INTC, VZ)
- **Multi-signal**: MA crossover (golden cross MA50/200, MA100/300 for sectors) + Donchian momentum breakout (20-day high) + dip-buy filter
- **Two-sided**: bonds and commodities receive short (−1) signals as well as long (+1); paper trading execution keeps bonds/commodities flat on short signals
- **Cross-sectional momentum tilt**: position weights tilted towards highest 63-day momentum rank
- **Position sizing**: ATR-normalized (risk per trade / ATR × price), capped at max position %
- **Entry filter**: RSI-14 < 70 (skip overbought entries)
- **Exit**: 3× ATR trailing stop with tightening after profit target, or death cross
- **Min hold**: 5 trading days to prevent whipsaw
- **Macro overlay**: VIX z-score > 2.5 blocks all signals; inverted yield curve reduces sizing
- **Validation**: 3-year train / 1-year test rolling walk-forward

---

## Key Results (Out-of-Sample Walk-Forward)

| Method | OOS Sharpe | OOS Active Sharpe | IS Sharpe |
|---|---|---|---|
| **Multi Mom Tilt ★** | **1.156** | **0.789** | 1.187 |
| Multi Equal Weight | 1.147 | 0.798 | 1.165 |
| Multi Fast ATR | 1.137 | 0.800 | 1.167 |
| Equal Weight | 1.126 | 0.768 | 1.152 |
| Buy & Hold (benchmark) | ~0.65 | — | ~0.65 |

Best method: **Multi Mom Tilt** — highest OOS Sharpe (1.156) with minimal IS→OOS degradation (IS: 1.187). All results are out-of-sample (walk-forward validated, not curve-fit).
