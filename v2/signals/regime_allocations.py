"""
v2/signals/regime_allocations.py
--------------------------------
Sharpe-weighted target allocations for each of the 6 regimes.

Each regime has a predefined target portfolio based on which asset classes
historically perform best in that macro environment. Weights are informed
by academic literature (Ilmanen 2011, Ang 2014, QUANTT 2026) and calibrated
to the ETF universe in v2/universe.py.

These are "ideal" allocations — the classifier's continuous regime
probabilities blend across them in v2/portfolio.py.

Design Principles
─────────────────
  - Long-only (no shorting, bonds provide defensiveness)
  - Weights per regime sum to 1.0
  - No single ETF > 30% (enforced in portfolio.py)
  - No asset class > 60% (enforced in portfolio.py)
  - Tilt toward historically strong performers in each regime
"""

import pandas as pd
from v2.universe import get_asset_class_map

# ── Regime Target Allocations ─────────────────────────────────────────────────
# Each dict maps ticker → weight. Weights sum to 1.0.
# Omitted tickers get 0 weight.

EXPANSION = {
    # Risk-on: heavy equities, growth/tech tilt, minimal bonds
    "SPY": 0.22, "QQQ": 0.13, "IWM": 0.07, "IWF": 0.06,
    "EFA": 0.06, "EEM": 0.05,
    "XLK": 0.06, "XLF": 0.05,
    "VNQ": 0.05,
    "HYG": 0.04,
    "LQD": 0.03,
    "GLD": 0.03,
    "DBC": 0.03,
    "SHY": 0.01,
    "IEF": 0.05,
    "TLT": 0.06,
}

SLOWDOWN = {
    # Defensive rotation: reduce equity beta, add duration + quality
    "SPY": 0.12, "IWD": 0.05, "EFA": 0.03,
    "XLP": 0.05, "XLV": 0.05, "XLU": 0.05,
    "IEF": 0.12, "TLT": 0.10, "LQD": 0.08,
    "TIP": 0.05,
    "GLD": 0.08, "SLV": 0.02,
    "SHY": 0.10,
    "VNQ": 0.05,
    "UUP": 0.05,
}

RECESSION = {
    # Maximum defensiveness: treasuries, gold, cash proxy, quality equities
    "SHY": 0.20, "IEF": 0.15, "TLT": 0.15,
    "GLD": 0.12, "SLV": 0.03,
    "TIP": 0.05,
    "SPY": 0.05,
    "XLP": 0.05, "XLV": 0.05, "XLU": 0.05,
    "UUP": 0.05,
    "FXY": 0.05,
}

RECOVERY = {
    # Aggressive risk-on: small cap, value, EM, cyclicals, commodities
    "SPY": 0.15, "IWM": 0.10, "IWD": 0.08,
    "EFA": 0.06, "EEM": 0.06,
    "XLF": 0.06, "XLE": 0.06, "XLK": 0.05,
    "VNQ": 0.06, "VNQI": 0.03,
    "HYG": 0.05,
    "DBC": 0.05, "GLD": 0.02, "USO": 0.02,
    "LQD": 0.03,
    "IEF": 0.04,
    "TLT": 0.03,
    "SHY": 0.02,
    "QQQ": 0.03,
}

STAGFLATION = {
    # Real assets + inflation protection: commodities, TIPS, gold, energy
    "GLD": 0.15, "SLV": 0.05, "DBC": 0.10, "USO": 0.05,
    "TIP": 0.12,
    "XLE": 0.08, "XLP": 0.05, "XLU": 0.05,
    "SPY": 0.05, "IWD": 0.03,
    "SHY": 0.10,
    "IEF": 0.05,
    "FXY": 0.03,
    "UUP": 0.04,
    "EFA": 0.03,
    "VNQ": 0.02,
}

LATE_CYCLE = {
    # Quality + defensives, reduce beta, add gold/bonds
    "SPY": 0.14, "IWD": 0.06,
    "XLP": 0.06, "XLV": 0.06, "XLU": 0.05,
    "IEF": 0.10, "TLT": 0.06, "LQD": 0.05,
    "TIP": 0.04,
    "GLD": 0.10, "SLV": 0.02,
    "SHY": 0.10,
    "VNQ": 0.04,
    "UUP": 0.04,
    "EFA": 0.04,
    "FXY": 0.02,
    "EEM": 0.02,
}

# Map regime index → allocation dict
REGIME_ALLOCATIONS = {
    0: EXPANSION,
    1: SLOWDOWN,
    2: RECESSION,
    3: RECOVERY,
    4: STAGFLATION,
    5: LATE_CYCLE,
}

REGIME_NAMES = {
    0: "Expansion", 1: "Slowdown", 2: "Recession",
    3: "Recovery", 4: "Stagflation", 5: "Late Cycle",
}


def get_regime_allocation(regime_id: int) -> dict[str, float]:
    """Get target allocation for a single regime."""
    return REGIME_ALLOCATIONS[regime_id]


def get_all_tickers() -> list[str]:
    """Get union of all tickers across all regime allocations."""
    tickers = set()
    for alloc in REGIME_ALLOCATIONS.values():
        tickers.update(alloc.keys())
    return sorted(tickers)


def validate_allocations():
    """Verify all allocations sum to 1.0 and respect constraints."""
    issues = []
    ac_map = get_asset_class_map()

    for regime_id, alloc in REGIME_ALLOCATIONS.items():
        name = REGIME_NAMES[regime_id]
        total = sum(alloc.values())
        if abs(total - 1.0) > 0.001:
            issues.append(f"{name}: weights sum to {total:.3f}, not 1.0")

        # Check single-ETF cap
        for ticker, weight in alloc.items():
            if weight > 0.30:
                issues.append(f"{name}: {ticker} = {weight:.1%} > 30% cap")

        # Check asset-class cap
        ac_weights = {}
        for ticker, weight in alloc.items():
            ac = ac_map.get(ticker, "unknown")
            ac_weights[ac] = ac_weights.get(ac, 0) + weight
        for ac, weight in ac_weights.items():
            if weight > 0.60:
                issues.append(f"{name}: {ac} = {weight:.1%} > 60% cap")

    return issues


if __name__ == "__main__":
    print("Regime Target Allocations\n")

    all_tickers = get_all_tickers()
    ac_map = get_asset_class_map()

    for regime_id, name in REGIME_NAMES.items():
        alloc = REGIME_ALLOCATIONS[regime_id]
        total = sum(alloc.values())
        print(f"\n  {name.upper()} (total={total:.1%})")

        # Group by asset class
        by_ac = {}
        for ticker, weight in sorted(alloc.items(), key=lambda x: -x[1]):
            ac = ac_map.get(ticker, "other")
            by_ac.setdefault(ac, []).append((ticker, weight))

        for ac in sorted(by_ac.keys()):
            ac_total = sum(w for _, w in by_ac[ac])
            print(f"    {ac:15s} ({ac_total:5.1%}): ", end="")
            print(", ".join(f"{t} {w:.0%}" for t, w in by_ac[ac]))

    issues = validate_allocations()
    if issues:
        print(f"\n  VALIDATION ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  All allocations valid (sum=1.0, ETF<=30%, AC<=60%)")
