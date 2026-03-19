"""
run.py
------
Pipeline orchestrator — runs each module as a subprocess in sequence.

Architecture
────────────
Each pipeline stage is a standalone Python module that reads from and
writes to the ``data/`` directory. Running stages as subprocesses (rather
than importing them) ensures each step starts in a clean interpreter state,
preventing silent state corruption between stages.

Data flow
─────────
  data_pipeline.py       reads  yfinance            writes  data/raw/
  feature_engineering.py reads  data/raw/            writes  data/features/
  macro_features.py      reads  yfinance + FRED       writes  data/macro/
  signal_generation.py   reads  data/features/ + macro  writes  data/signals/
  backtester.py          reads  data/signals/ + features writes  data/results/
  portfolio.py           reads  data/signals/ + features writes  data/results/

Shortcut flags (skip expensive upstream stages when data is still fresh)
────────────────────────────────────────────────────────────────────────
  python run.py              — full run (all 6 stages, ~5–10 min)
  python run.py signals      — skip data download; restart from feature_engineering
  python run.py portfolio    — run portfolio stage only (seconds)
  python run.py backtest     — run backtester + portfolio
  python run.py macro        — run from macro_features onward
"""

import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("data_pipeline",       "Downloading and cleaning market data"),
    ("feature_engineering", "Engineering features"),
    ("macro_features",      "Fetching macro data"),
    ("signal_generation",   "Generating signals"),
    ("backtester",          "Running backtests"),
    ("portfolio",           "Building portfolio"),
]

# shortcut entry points — skip early steps when data already exists
SHORTCUTS = {
    "signals"   : "feature_engineering",  # skip data download
    "portfolio" : "portfolio",             # run portfolio only
    "backtest"  : "backtester",            # run backtester + portfolio
    "macro"     : "macro_features",        # run from macro onward
}


def run_step(module: str, description: str) -> bool:
    """
    Execute one pipeline stage as a subprocess and print timing.

    Args:
        module:      Name of the Python file to run (without .py extension).
        description: Human-readable label printed to the console.

    Returns:
        True if the step exited with code 0 (success), False otherwise.
        On failure the pipeline halts immediately — later steps depend on
        earlier outputs, so continuing after an error would produce corrupt results.
    """
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  running {module}.py")
    print(f"{'='*60}")

    start  = time.time()
    result = subprocess.run([sys.executable, f"{module}.py"], capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED: {module}.py exited with code {result.returncode}")
        return False

    print(f"\n  Done in {elapsed:.1f}s")
    return True


def main():
    """
    Parse the optional shortcut argument, slice the STEPS list, and run
    each stage in order. Exits with code 1 on the first failure so CI
    systems can detect broken runs.
    """
    # determine start step from command line arg
    start_from = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in SHORTCUTS:
            start_from = SHORTCUTS[arg]
        else:
            print(f"Unknown shortcut '{arg}'. Options: {list(SHORTCUTS.keys())}")
            sys.exit(1)

    # find which index to start from
    step_names  = [s[0] for s in STEPS]
    start_index = step_names.index(start_from) if start_from else 0

    steps_to_run = STEPS[start_index:]
    total        = len(steps_to_run)

    print(f"\nRunning {total} step(s) starting from '{steps_to_run[0][0]}'")

    overall_start = time.time()

    for i, (module, description) in enumerate(steps_to_run, 1):
        print(f"\n[{i}/{total}]", end="")
        ok = run_step(module, description)
        if not ok:
            print(f"\nPipeline stopped at step {i}/{total}: {module}")
            sys.exit(1)

    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  All {total} steps completed in {total_time:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()