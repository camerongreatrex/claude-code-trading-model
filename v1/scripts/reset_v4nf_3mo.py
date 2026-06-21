"""
Reset the V4N-F 3-month forward test ledger.

Wipes paper_trading_v4nf_3mo/ and writes a pending-init marker so the
scheduler auto-inits on the next trading day using the same rank_rotate
model as main paper trading (no top-up).

Run: PYTHONPATH=. python v1/scripts/reset_v4nf_3mo.py [--start YYYY-MM-DD]

Default start: next NYSE weekday from today.
Test window: 3 calendar months from start date.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

PT_DIR = Path("data/v1/paper_trading_v4nf_3mo")
ARTEFACTS = (
    "state.json", "history.csv", "trades.csv", "orders_tomorrow.json",
    "history.csv.pre_catchup.bak", "state.json.pre_catchup.bak",
    "trades.csv.pre_catchup.bak", "trades.csv.bad.bak",
)
PENDING_FILE = PT_DIR / "pending_init.json"


def _next_trading_day(after: date) -> date:
    d = after + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset v4nf_3mo 3-month test")
    parser.add_argument(
        "--start", default="",
        help="First trading day (YYYY-MM-DD). Default: next weekday.",
    )
    args = parser.parse_args()

    PT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARTEFACTS:
        fp = PT_DIR / name
        if fp.exists():
            fp.unlink()
            print(f"  removed {name}")

    today = date.today()
    start = date.fromisoformat(args.start) if args.start else _next_trading_day(today)
    end = start + timedelta(days=92)   # ~3 months

    meta = {
        "test_start": str(start),
        "test_end": str(end),
        "execution": "rank_rotate",
        "note": "Same sizer + rank_rotate as main; no daily top-up",
        "reset_on": str(today),
    }
    PENDING_FILE.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\n3-month test reset.")
    print(f"  Start : {start}  (auto-init on first EOD after this date)")
    print(f"  End   : {end}")
    print(f"  Model : rank_rotate (matches main paper trader)")
    print(f"\nOr init manually on {start}:")
    print("  PT_INSTANCE=v4nf_3mo PYTHONPATH=. python v1/scripts/paper_trader.py init")


if __name__ == "__main__":
    main()
