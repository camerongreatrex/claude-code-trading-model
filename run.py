"""
run.py — root dispatcher into v1 pipeline.

  python run.py            full run
  python run.py signals    skip data download
  python run.py backtest   backtester + portfolio only
"""

if __name__ == "__main__":
    from v1.scripts.run_v1 import main
    main()
