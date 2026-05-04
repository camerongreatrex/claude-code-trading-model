"""
Parameter sensitivity sweep for MA crossover strategy. Sweeps each param
(ma_fast, ma_slow, rsi_thresh, atr_mult, min_hold) holding others at default,
runs walk-forward (3yr/1yr equal-weight), reports OOS Sharpe per value.
Constant params patched on signal_generation module then restored.
Fragility: >0.2 OOS Sharpe drop on +/-1 step flags overfit.
Output: data/research/sensitivity.parquet.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import v1.pipeline.signal_generation as sg
from v1.pipeline.data_pipeline import TICKER_LIST, ASSET_CLASS

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURE_DIR  = Path("data/v1/features")
RESEARCH_DIR = Path("data/v1/research")
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

# ── Walk-forward config (must match portfolio.py) ──────────────────────────────
TRAIN_DAYS = 756   # 3 years
TEST_DAYS  = 252   # 1 year

# ── Default parameter values (must match production constants) ─────────────────
DEFAULTS = {
    "ma_fast"    : 50,
    "ma_slow"    : 200,
    "rsi_thresh" : 70,
    "atr_mult"   : 3.0,
    "min_hold"   : 5,
}

# ── Sweep ranges ───────────────────────────────────────────────────────────────
SWEEPS = {
    "ma_fast"    : [30, 40, 50, 60, 70],
    "ma_slow"    : [150, 175, 200, 225, 250],
    "rsi_thresh" : [60, 65, 70, 75, 80],
    "atr_mult"   : [2.0, 2.5, 3.0, 3.5, 4.0],
    "min_hold"   : [3, 5, 7, 10],
}

# Maps sensitivity param names -> signal_generation module attribute names
_CONST_MAP = {
    "rsi_thresh" : "RSI_ENTRY_THRESH",
    "atr_mult"   : "ATR_TRAILING_MULT",
    "min_hold"   : "MIN_HOLD_DAYS",
}

# OOS Sharpe drop from default that flags a parameter as fragile
FRAGILITY_THRESHOLD = 0.2


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data():
    """Load feature DFs, macro, and aligned (T x N) log-return DataFrame."""
    macro    = sg.load_macro()
    features = {}
    for ticker in TICKER_LIST:
        path = FEATURE_DIR / f"{ticker}.parquet"
        if path.exists():
            features[ticker] = pd.read_parquet(path)

    returns = pd.DataFrame(
        {t: df["log_return"] for t, df in features.items()}
    ).dropna()

    return features, macro, returns


# ── Signal generation with parameterised MA windows ───────────────────────────

def _gen_signals(
    features: dict,
    macro: pd.DataFrame,
    ma_fast: int,
    ma_slow: int,
) -> pd.DataFrame:
    """
    Generate signal_regime for all tickers with configurable MA windows.
    Non-bonds: long-only MA crossover + VIX gate + RSI/min-hold/ATR filters.
    Bonds: two-sided momentum (no MA windows).
    """
    all_sigs = {}
    for ticker, df in features.items():
        asset_class = ASSET_CLASS[ticker]
        gate        = sg.vix_gate(macro, df.index)

        if asset_class == "bond":
            # bond_regime always 1 -> simplify to momentum * gate.
            mom      = sg.momentum_rule(df)
            signal_r = mom * gate

        else:
            # MA crossover; post-processors read from (possibly patched) sg module.
            ma_f     = df["Close"].rolling(ma_fast).mean()
            ma_s     = df["Close"].rolling(ma_slow).mean()
            signal_r = (ma_f > ma_s).astype(int) * gate
            signal_r = sg.apply_rsi_entry_filter(signal_r, df["rsi_14"])
            signal_r = sg.apply_min_hold_filter(signal_r)
            signal_r = sg.apply_trailing_stop_signal(signal_r, df["Close"], df["atr_14"])
            signal_r = signal_r * gate

        all_sigs[ticker] = signal_r

    return pd.DataFrame(all_sigs).dropna()


# ── Walk-forward OOS evaluation ────────────────────────────────────────────────

def _walk_forward_oos(signals: pd.DataFrame, returns: pd.DataFrame) -> list:
    """
    Equal-weight walk-forward (3yr/1yr): per non-overlapping test window,
    equal-weight avg strat return -> annualised Sharpe. Returns list per period.
    Strat[t] = pos[t-1] * ret[t] (no look-ahead).
    """
    # Align returns to signals index (MA warm-up may shorten signals)
    common_idx = signals.index.intersection(returns.index)
    sig_aligned = signals.reindex(common_idx)
    ret_aligned = returns.reindex(common_idx)
    tickers     = [t for t in sig_aligned.columns if t in ret_aligned.columns]

    if not tickers or len(common_idx) < TRAIN_DAYS + TEST_DAYS:
        return []

    # Pre-compute strat[t] = ret[t] * sig[t-1]
    strat_all = ret_aligned[tickers].multiply(
        sig_aligned[tickers].shift(1).fillna(0),
    )

    oos_sharpes = []
    start = TRAIN_DAYS

    while start + TEST_DAYS <= len(common_idx):
        # Equal-weight average over active tickers
        period_ret = strat_all.iloc[start : start + TEST_DAYS][tickers].mean(axis=1)

        std = float(period_ret.std())
        sharpe = float(period_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        oos_sharpes.append(sharpe)
        start += TEST_DAYS

    return oos_sharpes


# ── Per-iteration sweep driver ─────────────────────────────────────────────────

def _sweep_one(
    features: dict,
    macro: pd.DataFrame,
    returns: pd.DataFrame,
    param_name: str,
    value,
) -> dict:
    """
    One sweep iteration. MA params -> direct args; constants -> patch sg attr
    and restore in finally. Returns parameter, value, oos_sharpe, oos_std, n_periods.
    """
    ma_fast = DEFAULTS["ma_fast"]
    ma_slow = DEFAULTS["ma_slow"]

    # Determine whether we need to patch a module constant
    patched_attr = None
    original_val = None

    if param_name == "ma_fast":
        ma_fast = int(value)
    elif param_name == "ma_slow":
        ma_slow = int(value)
    else:
        patched_attr = _CONST_MAP[param_name]
        original_val = getattr(sg, patched_attr)
        setattr(sg, patched_attr, type(original_val)(value))

    try:
        signals = _gen_signals(features, macro, ma_fast, ma_slow)
        oos     = _walk_forward_oos(signals, returns)
    finally:
        if patched_attr is not None:
            setattr(sg, patched_attr, original_val)

    mean_oos = float(np.mean(oos)) if oos else 0.0
    std_oos  = float(np.std(oos))  if oos else 0.0

    return {
        "parameter"  : param_name,
        "value"      : float(value),
        "oos_sharpe" : round(mean_oos, 4),
        "oos_std"    : round(std_oos,  4),
        "n_periods"  : len(oos),
    }


# ── Summary table ──────────────────────────────────────────────────────────────

def _print_summary(results: pd.DataFrame) -> None:
    """
    Print sensitivity table. Flags [FRAGILE] when value drops > FRAGILITY_THRESHOLD
    below default; also warns on any adjacent step drop > threshold.
    """
    W = 72
    print("\n" + "=" * W)
    print("  PARAMETER SENSITIVITY  --  OOS Sharpe (equal weight, 3yr/1yr WF)")
    print("=" * W)

    fragile_params = []

    for param in SWEEPS:
        subset  = results[results["parameter"] == param].sort_values("value").reset_index(drop=True)
        default = float(DEFAULTS[param])

        default_rows   = subset[subset["value"] == default]
        default_sharpe = float(default_rows["oos_sharpe"].iloc[0]) if len(default_rows) else None

        print(f"\n  {param}  (default = {default})")
        print(f"  {'Value':<10} {'OOS Sharpe':>12} {'Std':>8}  {'vs default':>12}")
        print("  " + "-" * 48)

        param_fragile = False
        values  = subset["value"].tolist()
        sharpes = subset["oos_sharpe"].tolist()

        for idx, row in subset.iterrows():
            v     = float(row["value"])
            s     = float(row["oos_sharpe"])
            delta = (s - default_sharpe) if default_sharpe is not None else 0.0
            tag   = " <-- default" if v == default else ""
            flag  = ""

            if default_sharpe is not None and v != default and (default_sharpe - s) > FRAGILITY_THRESHOLD:
                flag          = " [FRAGILE]"
                param_fragile = True

            print(f"  {v:<10g} {s:>12.3f} {row['oos_std']:>8.3f}  {delta:>+12.3f}{tag}{flag}")

        # Check adjacent-step drops anywhere in the sweep
        for j in range(len(values) - 1):
            drop = abs(sharpes[j] - sharpes[j + 1])
            if drop > FRAGILITY_THRESHOLD:
                v1, v2 = values[j], values[j + 1]
                print(f"  *** step drop  {v1} -> {v2}:  -{drop:.3f} > {FRAGILITY_THRESHOLD}")
                param_fragile = True

        if param_fragile:
            fragile_params.append(param)

    print("\n" + "=" * W)
    if fragile_params:
        print(f"  FRAGILE PARAMETERS: {', '.join(fragile_params)}")
        print(f"  A >{FRAGILITY_THRESHOLD} OOS Sharpe drop on a single step = potential overfit.")
    else:
        print(f"  All parameters robust -- no single-step OOS Sharpe drop > {FRAGILITY_THRESHOLD}.")
    print("=" * W + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    W = 72
    print("=" * W)
    print("  Parameter Sensitivity Sweep")
    print(f"  Walk-forward: {TRAIN_DAYS}-day train / {TEST_DAYS}-day test, equal weight")
    print(f"  Fragility threshold: >{FRAGILITY_THRESHOLD} OOS Sharpe drop on +/-1 step")
    print("=" * W)

    print("\nLoading features and macro data...")
    features, macro, returns = _load_data()
    print(f"  {len(features)} tickers  |  returns matrix: {returns.shape}")

    # Total sweep iterations (for progress display)
    total = sum(len(v) for v in SWEEPS.values())
    run   = 0
    rows  = []

    for param_name, values in SWEEPS.items():
        print(f"\n  Sweeping {param_name} over {values} ...")
        for value in values:
            run += 1
            result = _sweep_one(features, macro, returns, param_name, value)
            rows.append(result)

            default_sharpe = _sweep_one(
                features, macro, returns, param_name, DEFAULTS[param_name]
            )["oos_sharpe"] if value == DEFAULTS[param_name] else None

            marker = " <-- default" if float(value) == float(DEFAULTS[param_name]) else ""
            print(
                f"  [{run:2d}/{total}]  {param_name}={str(value):<8}  "
                f"OOS Sharpe={result['oos_sharpe']:+.3f}  "
                f"Std={result['oos_std']:.3f}  "
                f"n={result['n_periods']}"
                f"{marker}"
            )

    results = pd.DataFrame(rows)

    out_path = RESEARCH_DIR / "sensitivity.parquet"
    results.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"\nResults saved -> {out_path}")

    _print_summary(results)


if __name__ == "__main__":
    main()
