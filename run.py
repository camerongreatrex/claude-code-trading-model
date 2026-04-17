"""
run.py
------
Pipeline orchestrator — runs each module as a subprocess in sequence.

Supports both v1 (trend-following) and v2 (macro regime rotation) pipelines.

Usage
─────
  python run.py              — full v1 run (all stages)
  python run.py v2           — full v2 run (universe → regime → allocation → backtest)
  python run.py v2 backtest  — v2 backtest only
  python run.py v2 regime    — v2 from regime classification onward
  python run.py signals      — v1 shortcut: skip data download
  python run.py portfolio    — v1 shortcut: portfolio only
  python run.py backtest     — v1 shortcut: backtester + portfolio
  python run.py macro        — v1 shortcut: from macro onward
"""

import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("pipeline.data_pipeline",       "Downloading and cleaning market data"),
    ("pipeline.feature_engineering", "Engineering features"),
    ("pipeline.feature_research",    "Computing feature IC (information coefficients)"),
    ("pipeline.macro_features",      "Fetching macro data"),
    ("pipeline.signal_generation",   "Generating signals"),
    ("pipeline.carry_signal",        "Computing carry signals"),
    ("pipeline.backtester",          "Running backtests"),
    ("pipeline.portfolio",           "Building portfolio"),
]

# shortcut entry points — skip early steps when data already exists
SHORTCUTS = {
    "signals"   : "pipeline.feature_engineering",  # skip data download
    "portfolio" : "pipeline.portfolio",             # run portfolio only
    "backtest"  : "pipeline.backtester",            # run backtester + portfolio
    "macro"     : "pipeline.macro_features",        # run from macro onward
}

# standalone diagnostics — run directly, not part of the main pipeline
DIAGNOSTICS = {
    "diagnostic": "pipeline.correlation_diagnostic",  # regime / beta / dead-weight analysis
    "capture"   : "pipeline.capture_diagnostic",      # upside/downside capture decomposition
    "audit"     : "pipeline.integrity_audit",         # look-ahead / WF / sensitivity / cost checks
    "screen"    : "pipeline.universe_screen",         # liquidity + correlation screen for new tickers
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


# ── V2 pipeline steps (macro regime rotation) ────────────────────────────────
V2_STEPS = [
    ("v2.universe",                "Building cross-asset ETF universe"),
    ("v2.regimes.macro_features",  "Fetching macro regime features (FRED)"),
    ("v2.regimes.market_features", "Fetching market-implied stress signals"),
    ("v2.regimes.classifier",      "Running hybrid regime classifier"),
    ("v2.regimes.validation",      "Validating regime classification (NBER)"),
    ("v2.backtester",              "Running full backtest (portfolio + risk + walk-forward)"),
]

V2_SHORTCUTS = {
    "backtest"  : "v2.backtester",              # backtest only (portfolio already built)
    "regime"    : "v2.regimes.macro_features",   # from regime classification onward
}


def main():
    """
    Parse the optional shortcut argument, slice the STEPS list, and run
    each stage in order. Exits with code 1 on the first failure so CI
    systems can detect broken runs.
    """
    # Check for v2 pipeline
    if len(sys.argv) > 1 and sys.argv[1].lower() == "v2":
        main_v2()
        return

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
            all_opts = list(SHORTCUTS.keys()) + list(DIAGNOSTICS.keys()) + ["v2"]
            print(f"Unknown shortcut '{arg}'. Options: {all_opts}")
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


def main_v2():
    """Run the v2 macro regime rotation pipeline."""
    # Check for v2 shortcut
    start_from = None
    if len(sys.argv) > 2:
        arg = sys.argv[2].lower()
        if arg in V2_SHORTCUTS:
            start_from = V2_SHORTCUTS[arg]
        else:
            print(f"Unknown v2 shortcut '{arg}'. Options: {list(V2_SHORTCUTS.keys())}")
            sys.exit(1)

    step_names = [s[0] for s in V2_STEPS]
    start_index = step_names.index(start_from) if start_from else 0
    steps_to_run = V2_STEPS[start_index:]
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