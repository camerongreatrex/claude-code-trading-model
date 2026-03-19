# Algorithmic Trading System

## Daily Use — 3 commands, 2 terminals

**Terminal 1 — Scheduler** (leave it running all day)
```
python scheduler.py
```
Auto-fires the EOD pipeline at 4:45 PM ET. If your PC was off for days it catches up automatically on startup.

**Terminal 2 — Dashboard**
```
python -m streamlit run dashboard.py
```
Opens at http://localhost:8501

**When you want fresh backtest data** (~5–10 min, run occasionally)
```
python run.py
```

---

## Reset the Paper Portfolio

1. Replace `data/paper_trading/state.json` with:
```json
{
  "cash": 100000.0,
  "positions": {},
  "initial_capital": 100000.0,
  "initialized_date": "TODAY'S DATE",
  "last_eod_date": null,
  "portfolio_value": 100000.0
}
```

2. Replace `data/paper_trading/history.csv` with just the header:
```
date,portfolio_value,cash,invested,n_positions,daily_return
```

3. Run the two daily commands above. The scheduler will enter new positions at 4:45 PM ET.

---

## Pipeline Shortcuts

| Command | What it does |
|---|---|
| `python run.py` | Full rebuild (download → features → signals → backtest) |
| `python run.py signals` | Skip download, re-run from feature engineering |
| `python run.py backtest` | Re-run backtester + portfolio only |
| `python run.py macro` | Re-run from macro features onward |
