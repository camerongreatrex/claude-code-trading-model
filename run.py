"""
run.py
------
Root-level dispatcher — routes to v1 or v2 pipeline orchestrator.

Usage
─────
  python run.py              — full v1 run
  python run.py v2           — full v2 run
  python run.py v2 backtest  — v2 backtest only
  python run.py signals      — v1 shortcut: skip data download
  python run.py backtest     — v1 shortcut: backtester + portfolio
"""

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "v2":
        # Strip "v2" from argv so run_v2 sees its own args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from v2.scripts.run_v2 import main
        main()
    else:
        from v1.scripts.run_v1 import main
        main()
