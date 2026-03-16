"""
Runs the full pipeline in order.
Usage:
    python run.py           — full run
    python run.py signals   — skip data download, rerun from feature engineering
    python run.py portfolio — skip everything, just rerun portfolio
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