"""
Init V4N-F 3-month test using the same rank_rotate model as main paper trading.

Delegates to paper_trader.init_positions() — no frozen sizer, no daily top-up.

Run on first trading day of test window:
  PT_INSTANCE=v4nf_3mo PYTHONPATH=. python v1/scripts/_init_v4nf_at_open.py
  — or —
  PT_INSTANCE=v4nf_3mo PYTHONPATH=. python v1/scripts/paper_trader.py init
"""

from __future__ import annotations

import json
import os
from pathlib import Path

assert os.environ.get("PT_INSTANCE") == "v4nf_3mo", \
    "Must set PT_INSTANCE=v4nf_3mo before running this script."

from v1.scripts.paper_trader import PT_DIR, init_positions  # noqa: E402

PENDING = PT_DIR / "pending_init.json"


def main() -> None:
    init_positions()
    if PENDING.exists():
        meta = json.loads(PENDING.read_text())
        meta["initialized"] = True
        PENDING.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"\n3-month test window: {meta.get('test_start')} → {meta.get('test_end')}")


if __name__ == "__main__":
    main()
