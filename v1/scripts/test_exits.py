"""
test_exits.py
-------------
Test alternative exit overlays on top of the existing signal_multi using the
locked-in atr_lev_1.5x sizing.

Each overlay can only fire EARLIER than current exits — it converts 1's to 0's
based on a different exit rule (Chandelier high, profit-lock, time-stop on
no-new-high, vol-regime tighten, ATR channel).  Once an overlay-triggered exit
fires, the position stays flat until upstream signal_multi cycles 0 -> 1 (a
fresh entry per the existing pipeline).

Run: python -m v1.scripts.test_exits
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes, defensive_tilt_overlay,
    portfolio_returns, summarise, MAX_POSITION_PCT, CAPITAL,
)


# ── Per-ticker overlay engine ────────────────────────────────────────────────
def _apply_overlay(signal: pd.Series, close: pd.Series, atr: pd.Series,
                   vix: pd.Series, ma20: pd.Series,
                   overlay_fn) -> pd.Series:
    """
    Walk one ticker bar-by-bar tracking entry price, highest-high since entry,
    and bars-since-high.  Hand state to overlay_fn(state) -> bool to decide
    whether to force an exit.  Position stays flat after overlay-exit until
    upstream signal cycles 0 -> 1.
    """
    out = signal.copy().astype(int)
    in_pos = False
    entry = 0.0
    high  = 0.0
    bars_since_high = 0
    overlay_exited = False

    s_arr = signal.values
    p_arr = close.values
    a_arr = atr.values
    v_arr = vix.values if vix is not None else np.full(len(signal), 20.0)
    m_arr = ma20.values if ma20 is not None else p_arr

    for i in range(len(out)):
        s = int(s_arr[i]) if not np.isnan(s_arr[i]) else 0
        p = float(p_arr[i])
        a = float(a_arr[i]) if not np.isnan(a_arr[i]) else 0.0
        v = float(v_arr[i]) if not np.isnan(v_arr[i]) else 20.0
        m = float(m_arr[i]) if not np.isnan(m_arr[i]) else p

        if not in_pos:
            if s > 0:
                in_pos = True
                entry  = p
                high   = p
                bars_since_high = 0
                overlay_exited  = False
            continue

        if s == 0:
            in_pos = False
            overlay_exited = False
            continue

        if overlay_exited:
            out.iloc[i] = 0
            continue

        if p > high:
            high = p
            bars_since_high = 0
        else:
            bars_since_high += 1

        state = dict(price=p, atr=a, vix=v, ma20=m, entry=entry,
                     high=high, bars_since_high=bars_since_high)
        if overlay_fn(state):
            out.iloc[i]    = 0
            overlay_exited = True

    return out


# ── Overlay rules ────────────────────────────────────────────────────────────
def chandelier(k: float = 3.0):
    """Exit if close < highest_high_since_entry - k * ATR."""
    def fn(s):
        if s["atr"] <= 0:
            return False
        return s["price"] < s["high"] - k * s["atr"]
    return fn


def profit_lock(target1: float = 0.05, lock1: float = 0.0,
                target2: float = 0.10, lock2: float = 0.05):
    """Once gain > target1, exit if price < entry*(1+lock1).
       Once gain > target2, exit if price < entry*(1+lock2)."""
    def fn(s):
        gain = (s["high"] - s["entry"]) / s["entry"] if s["entry"] > 0 else 0.0
        if gain >= target2:
            return s["price"] < s["entry"] * (1 + lock2)
        if gain >= target1:
            return s["price"] < s["entry"] * (1 + lock1)
        return False
    return fn


def time_stop_no_high(n_bars: int = 30):
    """Exit if no new high in n_bars trading days."""
    def fn(s):
        return s["bars_since_high"] >= n_bars
    return fn


def vol_regime_tighten(vix_thresh: float = 25.0, k: float = 1.0):
    """In high VIX (>thresh), tighten the chandelier to k*ATR."""
    def fn(s):
        if s["vix"] <= vix_thresh or s["atr"] <= 0:
            return False
        return s["price"] < s["high"] - k * s["atr"]
    return fn


def atr_channel(k: float = 2.0):
    """Exit if close < MA20 - k * ATR (channel break-down)."""
    def fn(s):
        if s["atr"] <= 0:
            return False
        return s["price"] < s["ma20"] - k * s["atr"]
    return fn


def combo_chandelier_timestop(k_chand: float = 3.0, n_bars: int = 60):
    """Chandelier OR time-stop — whichever fires first."""
    c = chandelier(k_chand)
    t = time_stop_no_high(n_bars)
    return lambda s: c(s) or t(s)


def combo_or(*fns):
    return lambda s: any(f(s) for f in fns)


# ── Build overlay-modified signals across all tickers ────────────────────────
def apply_overlay_all(signals: pd.DataFrame, features: dict,
                      macro: pd.DataFrame, overlay_fn) -> pd.DataFrame:
    """Apply overlay_fn per ticker; return new signal frame."""
    if "vix" in macro.columns:
        vix_full = macro["vix"].reindex(signals.index).ffill().fillna(20.0)
    else:
        vix_full = pd.Series(20.0, index=signals.index)

    out = signals.copy()
    for t in signals.columns:
        if t not in features:
            continue
        f = features[t]
        close = f["Close"].reindex(signals.index).ffill()
        atr   = f["atr_14"].reindex(signals.index).ffill()
        ma20  = close.rolling(20).mean()
        out[t] = _apply_overlay(signals[t], close, atr, vix_full, ma20, overlay_fn)
    return out


def lev_15_sizes(signals, features, macro):
    """atr_lev_1.5x sizing pipeline (mirrors portfolio.py)."""
    sz = atr_sizes(signals, features, CAPITAL)
    sz = (sz * 1.5).clip(-CAPITAL * MAX_POSITION_PCT,
                          CAPITAL * MAX_POSITION_PCT)
    return defensive_tilt_overlay(sz, signals, macro, CAPITAL)


def main() -> None:
    SIGNAL_DIR  = Path("data/v1/signals")
    FEATURE_DIR = Path("data/v1/features")
    MACRO_DIR   = Path("data/shared/macro")

    print("Loading signals + features...")
    signals_raw = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")

    features: dict = {}
    returns = pd.DataFrame()
    for t in signals_raw.columns:
        p = FEATURE_DIR / f"{t}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            features[t] = f
            returns[t]  = f["log_return"]

    returns = returns.dropna()
    signals = signals_raw.reindex(returns.index).fillna(0)

    macro_path = MACRO_DIR / "macro_features.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else pd.DataFrame()

    # ── Variant catalogue ───────────────────────────────────────────────────
    variants: dict = {
        "baseline (no overlay)":        None,
        # tight ts probe around the winner
        "pl_5_10 + ts_35":              combo_or(
            profit_lock(0.05, 0.0, 0.10, 0.05), time_stop_no_high(35)),
        "pl_5_10 + ts_38":              combo_or(
            profit_lock(0.05, 0.0, 0.10, 0.05), time_stop_no_high(38)),
        "pl_5_10 + ts_40":              combo_or(
            profit_lock(0.05, 0.0, 0.10, 0.05), time_stop_no_high(40)),
        "pl_5_10 + ts_42":              combo_or(
            profit_lock(0.05, 0.0, 0.10, 0.05), time_stop_no_high(42)),
        "pl_5_10 + ts_45":              combo_or(
            profit_lock(0.05, 0.0, 0.10, 0.05), time_stop_no_high(45)),
        # nearby pl combinations
        "pl_5_12 + ts_40":              combo_or(
            profit_lock(0.05, 0.0, 0.12, 0.06), time_stop_no_high(40)),
        "pl_4_10 + ts_40":              combo_or(
            profit_lock(0.04, 0.0, 0.10, 0.04), time_stop_no_high(40)),
        "pl_6_12 + ts_40":              combo_or(
            profit_lock(0.06, 0.0, 0.12, 0.06), time_stop_no_high(40)),
        "pl_5_10 + ts_40 + chand_3":    combo_or(
            profit_lock(0.05, 0.0, 0.10, 0.05),
            time_stop_no_high(40), chandelier(3.0)),
    }

    print("\n{:<28} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Variant", "AnnRet%", "Sharpe", "MaxDD%", "Calmar", "Δexits"
    ))
    print("-" * 80)

    base_signals = signals.copy()
    base_active_days = int((base_signals == 1).sum().sum())

    for name, fn in variants.items():
        if fn is None:
            sig_v = base_signals
        else:
            sig_v = apply_overlay_all(base_signals, features, macro, fn)
        active_days = int((sig_v == 1).sum().sum())
        delta_exits = base_active_days - active_days  # +ve = overlay cut exposure

        sz  = lev_15_sizes(sig_v, features, macro)
        ret = portfolio_returns(sz, returns)
        s   = summarise(ret, name)
        print(f"{s['label']:<28} {s['ann_return']:>10} {s['sharpe']:>10} "
              f"{s['max_drawdown']:>10} {s['calmar']:>10} {delta_exits:>+10d}")


if __name__ == "__main__":
    main()
