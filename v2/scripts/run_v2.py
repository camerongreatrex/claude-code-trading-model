"""
run_v2.py
---------
Pipeline orchestrator for the v2 macro regime rotation system.
Runs each module as a subprocess in sequence.

Usage
─────
  python -m v2.scripts.run_v2              — full run (universe → regime → backtest)
  python -m v2.scripts.run_v2 backtest     — backtest only
  python -m v2.scripts.run_v2 regime       — from regime classification onward
"""

import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("v2.pipeline.data_pipeline",      "Building cross-asset ETF universe"),
    ("v2.regimes.macro_features",      "Fetching macro regime features (FRED)"),
    ("v2.regimes.market_features",     "Fetching market-implied stress signals"),
    ("v2.regimes.classification",      "Running hybrid regime classifier"),
    ("v2.regimes.validation",          "Validating regime classification (NBER)"),
    ("v2.pipeline.backtester",         "Running full backtest (portfolio + risk + walk-forward)"),
]

SHORTCUTS = {
    "backtest"  : "v2.pipeline.backtester",         # backtest only
    "regime"    : "v2.regimes.macro_features",      # from regime classification onward
}


def run_step(module: str, description: str) -> bool:
    """Execute one pipeline stage as a subprocess and print timing."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  running {module}")
    print(f"{'='*60}")

    start  = time.time()
    result = subprocess.run([sys.executable, "-m", module], capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED: {module} exited with code {result.returncode}")
        return False

    print(f"\n  Done in {elapsed:.1f}s")
    return True


def main():
    """Run the v2 macro regime rotation pipeline."""
    start_from = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in SHORTCUTS:
            start_from = SHORTCUTS[arg]
        else:
            print(f"Unknown shortcut '{arg}'. Options: {list(SHORTCUTS.keys())}")
            sys.exit(1)

    step_names = [s[0] for s in STEPS]
    start_index = step_names.index(start_from) if start_from else 0
    steps_to_run = STEPS[start_index:]
    total = len(steps_to_run)

    print(f"\n{'='*60}")
    print(f"  V2 Macro Regime Rotation Pipeline")
    print(f"  Running {total} step(s)")
    print(f"{'='*60}")

    overall_start = time.time()

    for i, (module, description) in enumerate(steps_to_run, 1):
        print(f"\n[{i}/{total}]", end="")
        ok = run_step(module, description)
        if not ok:
            print(f"\nV2 pipeline stopped at step {i}/{total}: {module}")
            sys.exit(1)

    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  V2 pipeline: {total} steps completed in {total_time:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
