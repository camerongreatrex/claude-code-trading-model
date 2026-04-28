"""
test_push_returns.py
--------------------
Goal: push the production atr_lev_1.5x baseline (15.30 / 1.353 / -8.67) higher
WITHOUT increasing leverage. Tests are run on the saved signal_multi (which
already has the pl_5_10 + ts_40 overlay baked in by signal_generation.py).

Three experiment groups
1. SIZER variants at 1.5x leverage — replace atr_lev_1.5x with hybrids that
   may capture more upside while keeping the per-name $10k cap and 1.5x gross.
2. EXIT-OVERLAY variants ON TOP of the existing overlay — even tighter / longer
   time-stops, three-tier profit lock (5/10/20), ATR-channel re-entry filter.
3. ENTRY filters — quality screens that drop weak entries (low ADX, weak
   trend) to lift Sharpe without changing the sizer.

Run: python -m v1.scripts.test_push_returns
"""

from pathlib import Path
import numpy as np
import pandas as pd

from v1.portfolio.portfolio import (
    atr_sizes, kelly_sizes, defensive_tilt_overlay,
    portfolio_returns, summarise, MAX_POSITION_PCT, CAPITAL,
    compute_strategy_returns,
    INDEX_ETF_TICKERS, INDEX_ETF_CAP, RISK_PER_TRADE,
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _clip(sz: pd.DataFrame) -> pd.DataFrame:
    return sz.clip(-CAPITAL * MAX_POSITION_PCT, CAPITAL * MAX_POSITION_PCT)


def lev_15(sz_base: pd.DataFrame, signals: pd.DataFrame, macro: pd.DataFrame
           ) -> pd.DataFrame:
    """Apply 1.5x leverage + per-name clip + defensive overlay (mirrors
    portfolio.atr_lev_1.5x recipe but on any base sizer)."""
    sz = _clip(sz_base * 1.5)
    return defensive_tilt_overlay(sz, signals, macro, CAPITAL)


def lev_10(sz_base: pd.DataFrame, signals: pd.DataFrame, macro: pd.DataFrame
           ) -> pd.DataFrame:
    """Apply 1.0x leverage (no boost) + per-name clip + defensive overlay."""
    sz = _clip(sz_base)
    return defensive_tilt_overlay(sz, signals, macro, CAPITAL)


def kelly_sizes_scaled(signals, returns, capital, scale=0.5, lookback=252):
    sizes = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for ticker in signals.columns:
        if ticker not in returns.columns:
            continue
        strat_ret = compute_strategy_returns(signals[ticker], returns[ticker])
        rolling_wr = strat_ret.rolling(lookback).apply(
            lambda x: float((x[x != 0] > 0).mean()) if (x != 0).any() else 0.5
        )
        rolling_pf = strat_ret.rolling(lookback).apply(
            lambda x: float(x[x > 0].sum() / x[x < 0].abs().sum())
            if x[x < 0].abs().sum() > 0 else 1.0
        )
        kelly = (rolling_wr - (1 - rolling_wr) / rolling_pf.clip(0.01)).clip(0) * scale
        sizes[ticker] = (signals[ticker] * kelly * capital).clip(
            -capital * MAX_POSITION_PCT, capital * MAX_POSITION_PCT
        )
    return sizes.fillna(0)


def sharpe_weighted_atr(signals, features, returns, capital, lookback=126,
                         tilt_mag=0.30):
    """ATR sizing × (1 + tilt_mag * sharpe_clip)."""
    base = atr_sizes(signals, features, capital)
    out  = base.copy()
    for t in signals.columns:
        if t not in returns.columns:
            continue
        strat_ret = signals[t].shift(1) * returns[t]
        rolling_mean = strat_ret.rolling(lookback).mean() * 252
        rolling_std  = strat_ret.rolling(lookback).std() * np.sqrt(252)
        sharpe = (rolling_mean / rolling_std.replace(0, np.nan)).fillna(0)
        scale = (1.0 + tilt_mag * sharpe.clip(-2, 2)).clip(0.4, 1.6)
        out[t] = base[t] * scale
    return _clip(out)


def momentum_quality_atr(signals, features, returns, capital,
                         mom_window=63, tilt_mag=0.40):
    """ATR sizing × (1 + tilt_mag * cross-sectional rank of mom/vol)."""
    base = atr_sizes(signals, features, capital)
    out  = base.copy()
    for t in signals.columns:
        if t not in returns.columns:
            continue
        mom = returns[t].rolling(mom_window).sum()
        vol = returns[t].rolling(mom_window).std()
        ratio = (mom / vol.replace(0, np.nan)).fillna(0)
        rank = ratio.rolling(252, min_periods=63).apply(
            lambda x: (x.iloc[-1] > x).mean() if not x.empty else 0.5
        ).fillna(0.5)
        scale = 1.0 + tilt_mag * (rank - 0.5) * 2  # rank 0.5 → 1.0; rank 1 → 1+tilt; rank 0 → 1-tilt
        out[t] = base[t] * scale
    return _clip(out)


def adx_filtered_atr(signals, features, capital, adx_min=18.0):
    """ATR sizing but zero out positions where adx < adx_min (kills weak trends)."""
    base = atr_sizes(signals, features, capital)
    out  = base.copy()
    for t in signals.columns:
        if t not in features:
            continue
        adx = features[t].get("adx")
        if adx is None:
            continue
        adx = adx.reindex(signals.index).ffill().fillna(0.0)
        mask = (adx >= adx_min).astype(float)
        out[t] = base[t] * mask
    return _clip(out)


def trend_filter_atr(signals, features, capital, lookback=20, slope_min=0.0):
    """ATR sizing but zero where MA(20) slope is negative."""
    base = atr_sizes(signals, features, capital)
    out  = base.copy()
    for t in signals.columns:
        if t not in features:
            continue
        close = features[t]["Close"].reindex(signals.index).ffill()
        ma   = close.rolling(lookback).mean()
        slope = ma.diff(5) / ma.shift(5)
        mask = (slope > slope_min).astype(float)
        out[t] = base[t] * mask
    return _clip(out)


def vol_target_atr(signals, features, returns, capital,
                   target_vol=0.13, window=63, scale_max=2.5):
    """Symmetric vol target: scale UP on calm, DOWN on storms."""
    base = atr_sizes(signals, features, capital)
    weights = base.shift(1) / capital
    port_ret = (weights * returns.reindex(columns=base.columns)).sum(axis=1)
    realized = port_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
    realized = realized.replace(0, np.nan).fillna(target_vol)
    scaler = (target_vol / realized).clip(0.3, scale_max).shift(1).fillna(1.0)
    return _clip(base.multiply(scaler, axis=0))


# ── Exit overlays ───────────────────────────────────────────────────────────
def _apply_overlay(signal: pd.Series, close: pd.Series, atr: pd.Series,
                   overlay_fn) -> pd.Series:
    out = signal.copy().astype(int)
    in_pos, entry, high, bars_since_high = False, 0.0, 0.0, 0
    overlay_exited = False
    s_arr = signal.values
    p_arr = close.values
    a_arr = atr.values

    for i in range(len(out)):
        s = int(s_arr[i]) if not np.isnan(s_arr[i]) else 0
        p = float(p_arr[i])
        a = float(a_arr[i]) if not np.isnan(a_arr[i]) else 0.0

        if not in_pos:
            if s > 0:
                in_pos = True
                entry, high, bars_since_high, overlay_exited = p, p, 0, False
            continue
        if s == 0:
            in_pos = False
            overlay_exited = False
            continue
        if overlay_exited:
            out.iloc[i] = 0
            continue

        if p > high:
            high, bars_since_high = p, 0
        else:
            bars_since_high += 1

        state = dict(price=p, atr=a, entry=entry, high=high,
                     bars_since_high=bars_since_high)
        if overlay_fn(state):
            out.iloc[i] = 0
            overlay_exited = True
    return out


def three_tier_lock(t1=0.05, l1=0.0, t2=0.10, l2=0.05, t3=0.20, l3=0.12):
    def fn(s):
        gain = (s["high"] - s["entry"]) / s["entry"] if s["entry"] > 0 else 0.0
        if gain >= t3:
            return s["price"] < s["entry"] * (1 + l3)
        if gain >= t2:
            return s["price"] < s["entry"] * (1 + l2)
        if gain >= t1:
            return s["price"] < s["entry"] * (1 + l1)
        return False
    return fn


def time_stop_no_high(n_bars=40):
    def fn(s):
        return s["bars_since_high"] >= n_bars
    return fn


def combo_or(*fns):
    return lambda s: any(f(s) for f in fns)


def apply_overlay_all(signals, features, overlay_fn):
    out = signals.copy()
    for t in signals.columns:
        if t not in features:
            continue
        f = features[t]
        close = f["Close"].reindex(signals.index).ffill()
        atr   = f["atr_14"].reindex(signals.index).ffill()
        out[t] = _apply_overlay(signals[t], close, atr, overlay_fn)
    return out


# ── Main ────────────────────────────────────────────────────────────────────
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

    # ============== EXPERIMENT 1: Sizer variants at 1.5x leverage ==========
    print("\n" + "=" * 86)
    print("EXPERIMENT 1: alternative SIZERS at 1.5x leverage on production signals")
    print("=" * 86)

    sz_atr   = atr_sizes(signals, features, CAPITAL)
    sz_half  = kelly_sizes_scaled(signals, returns, CAPITAL, scale=0.5)
    sz_qtr   = kelly_sizes_scaled(signals, returns, CAPITAL, scale=0.25)
    sz_swt   = sharpe_weighted_atr(signals, features, returns, CAPITAL, tilt_mag=0.30)
    sz_swt2  = sharpe_weighted_atr(signals, features, returns, CAPITAL, tilt_mag=0.50)
    sz_mqa   = momentum_quality_atr(signals, features, returns, CAPITAL, tilt_mag=0.40)
    sz_mqa2  = momentum_quality_atr(signals, features, returns, CAPITAL, tilt_mag=0.60)
    sz_adx   = adx_filtered_atr(signals, features, CAPITAL, adx_min=18.0)
    sz_adx2  = adx_filtered_atr(signals, features, CAPITAL, adx_min=22.0)
    sz_trf   = trend_filter_atr(signals, features, CAPITAL)
    sz_vt13  = vol_target_atr(signals, features, returns, CAPITAL,
                              target_vol=0.13, scale_max=2.5)
    sz_vt12  = vol_target_atr(signals, features, returns, CAPITAL,
                              target_vol=0.12, scale_max=2.0)

    sizer_variants = {
        "atr_lev_1.5x (BASE)":        lev_15(sz_atr,   signals, macro),
        "atr_85/kelly_15 @1.5x":      lev_15(_clip(0.85*sz_atr + 0.15*sz_half), signals, macro),
        "atr_70/kelly_30 @1.5x":      lev_15(_clip(0.70*sz_atr + 0.30*sz_half), signals, macro),
        "atr_50/kelly_50 @1.5x":      lev_15(_clip(0.50*sz_atr + 0.50*sz_half), signals, macro),
        "atr_70/qtrkelly_30 @1.5x":   lev_15(_clip(0.70*sz_atr + 0.30*sz_qtr ), signals, macro),
        "sharpe_wt_atr_30 @1.5x":     lev_15(sz_swt,   signals, macro),
        "sharpe_wt_atr_50 @1.5x":     lev_15(sz_swt2,  signals, macro),
        "mom_quality_atr_40 @1.5x":   lev_15(sz_mqa,   signals, macro),
        "mom_quality_atr_60 @1.5x":   lev_15(sz_mqa2,  signals, macro),
        "adx_18_filter @1.5x":        lev_15(sz_adx,   signals, macro),
        "adx_22_filter @1.5x":        lev_15(sz_adx2,  signals, macro),
        "trend_slope_filter @1.5x":   lev_15(sz_trf,   signals, macro),
        "vol_target_13 @1.5x":        lev_15(sz_vt13,  signals, macro),
        "vol_target_12 @1.5x":        lev_15(sz_vt12,  signals, macro),
        # blends
        "0.7 atr+0.3 mqa @1.5x":      lev_15(_clip(0.7*sz_atr + 0.3*sz_mqa), signals, macro),
        "0.5 atr+0.5 swt @1.5x":      lev_15(_clip(0.5*sz_atr + 0.5*sz_swt), signals, macro),
    }
    print(f"\n{'Variant':<32} {'AnnRet%':>9} {'Sharpe':>8} {'MaxDD%':>8} "
          f"{'Calmar':>8} {'Gross':>8}")
    print("-" * 86)
    rows = []
    for name, sz in sizer_variants.items():
        ret = portfolio_returns(sz, returns)
        s   = summarise(ret, name)
        gross = float(sz.abs().sum(axis=1).mean()) / CAPITAL
        rows.append((name, s, gross))
        print(f"{name:<32} {s['ann_return']:>9} {s['sharpe']:>8} "
              f"{s['max_drawdown']:>8} {s['calmar']:>8} {gross:>8.2f}")

    print("\nTop 5 by Sharpe:")
    for name, s, _ in sorted(rows, key=lambda r: -float(r[1]["sharpe"]))[:5]:
        print(f"  {name:<32} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}")
    print("\nTop 5 by Calmar:")
    for name, s, _ in sorted(rows, key=lambda r: -float(r[1]["calmar"]))[:5]:
        print(f"  {name:<32} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}")

    # ============== EXPERIMENT 2: EXIT overlays on top of production ======
    print("\n" + "=" * 86)
    print("EXPERIMENT 2: additional EXIT overlays on top of production signals")
    print("=" * 86)
    exit_variants = {
        "production (no extra)":         None,
        "3tier_5_10_20_lock":            three_tier_lock(),
        "3tier_5_12_20_lock":            three_tier_lock(t2=0.12, l2=0.06, t3=0.20, l3=0.12),
        "3tier + ts_50":                 combo_or(three_tier_lock(), time_stop_no_high(50)),
        "3tier + ts_60":                 combo_or(three_tier_lock(), time_stop_no_high(60)),
        "3tier_5_10_25_lock":            three_tier_lock(t3=0.25, l3=0.15),
        "3tier_5_10_30_lock":            three_tier_lock(t3=0.30, l3=0.20),
    }
    for name, fn in exit_variants.items():
        sig_v = signals if fn is None else apply_overlay_all(signals, features, fn)
        sz = lev_15(atr_sizes(sig_v, features, CAPITAL), sig_v, macro)
        ret = portfolio_returns(sz, returns)
        s = summarise(ret, name)
        print(f"{name:<28} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}")

    # ============== EXPERIMENT 3: 1.0x leverage variants  =================
    print("\n" + "=" * 86)
    print("EXPERIMENT 3: best sizers tested at 1.0x leverage (no margin)")
    print("=" * 86)
    onex_variants = {
        "atr_pure @1.0x (BASE 1x)":   lev_10(sz_atr,   signals, macro),
        "0.7 atr+0.3 mqa @1.0x":      lev_10(_clip(0.7*sz_atr + 0.3*sz_mqa), signals, macro),
        "sharpe_wt_atr_30 @1.0x":     lev_10(sz_swt,   signals, macro),
        "sharpe_wt_atr_50 @1.0x":     lev_10(sz_swt2,  signals, macro),
        "mom_quality_atr_40 @1.0x":   lev_10(sz_mqa,   signals, macro),
        "vol_target_13 @1.0x":        lev_10(sz_vt13,  signals, macro),
        "vol_target_12 @1.0x":        lev_10(sz_vt12,  signals, macro),
    }
    for name, sz in onex_variants.items():
        ret = portfolio_returns(sz, returns)
        s = summarise(ret, name)
        gross = float(sz.abs().sum(axis=1).mean()) / CAPITAL
        print(f"{name:<28} ret {s['ann_return']:>6}  Sh {s['sharpe']:>6}  "
              f"DD {s['max_drawdown']:>6}  Cal {s['calmar']:>6}  gross {gross:>4.2f}")


if __name__ == "__main__":
    main()
