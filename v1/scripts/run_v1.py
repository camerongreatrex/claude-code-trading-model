"""
run_v1.py
---------
Pipeline orchestrator for the v1 trend-following momentum system.
Runs each module as a subprocess in sequence.

Usage
─────
  python -m v1.scripts.run_v1              — full run (all stages)
  python -m v1.scripts.run_v1 signals      — skip data download
  python -m v1.scripts.run_v1 portfolio    — portfolio only
  python -m v1.scripts.run_v1 backtest     — backtester + portfolio
  python -m v1.scripts.run_v1 macro        — from macro onward
  python -m v1.scripts.run_v1 diagnostic   — correlation diagnostic
  python -m v1.scripts.run_v1 capture      — capture diagnostic
  python -m v1.scripts.run_v1 audit        — integrity audit
  python -m v1.scripts.run_v1 screen       — universe screen
"""

import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("v1.pipeline.data_pipeline",       "Downloading and cleaning market data"),
    ("v1.pipeline.feature_engineering", "Engineering features"),
    ("v1.pipeline.feature_research",    "Computing feature IC (information coefficients)"),
    ("v1.pipeline.macro_features",      "Fetching macro data"),
    ("v1.pipeline.signal_generation",   "Generating signals"),
    ("v1.pipeline.carry_signal",        "Computing carry signals"),
    ("v1.pipeline.backtester",          "Running backtests"),
    ("v1.portfolio.portfolio",           "Building portfolio"),
]

# shortcut entry points — skip early steps when data already exists
SHORTCUTS = {
    "signals"   : "v1.pipeline.feature_engineering",  # skip data download
    "portfolio" : "v1.portfolio.portfolio",             # run portfolio only
    "backtest"  : "v1.pipeline.backtester",            # run backtester + portfolio
    "macro"     : "v1.pipeline.macro_features",        # run from macro onward
}

# standalone diagnostics — run directly, not part of the main pipeline
DIAGNOSTICS = {
    "diagnostic": "v1.risk.correlation_diagnostic",    # regime / beta / dead-weight analysis
    "capture"   : "v1.risk.capture_diagnostic",        # upside/downside capture decomposition
    "audit"     : "v1.validation.integrity_audit",     # look-ahead / WF / sensitivity / cost checks
    "screen"    : "v1.pipeline.universe_screen",       # liquidity + correlation screen for new tickers
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
    """
    Parse the optional shortcut argument, slice the STEPS list, and run
    each stage in order. Exits with code 1 on the first failure so CI
    systems can detect broken runs.
    """
    # determine start step from command line arg
    start_from = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in DIAGNOSTICS:
            # run the diagnostic module directly and exit — not part of the pipeline
            module = DIAGNOSTICS[arg]
            print(f"\nRunning diagnostic: {module}\n")
            result = subprocess.run([sys.executable, "-m", module], capture_output=False)
            sys.exit(result.returncode)
        elif arg in SHORTCUTS:
            start_from = SHORTCUTS[arg]
        else:
            all_opts = list(SHORTCUTS.keys()) + list(DIAGNOSTICS.keys())
            print(f"Unknown shortcut '{arg}'. Options: {all_opts}")
            sys.exit(1)

    # find which index to start from
    step_names  = [s[0] for s in STEPS]
    start_index = step_names.index(start_from) if start_from else 0

    steps_to_run = STEPS[start_index:]
    total        = len(steps_to_run)

    print(f"\nV1 Momentum Pipeline")
    print(f"Running {total} step(s) starting from '{steps_to_run[0][0]}'")

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
