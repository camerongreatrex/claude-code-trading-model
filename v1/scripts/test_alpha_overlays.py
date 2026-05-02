"""
Sweep five candidate alpha overlays on top of V3 production
(top11_adx22_momt_ac55_cap1 + 12% diversifier sleeve):

  1. Trend-quality filter   — drop/halve names whose 63d return came from <=3 days
  2. Vol-carry (VIX scaling) — scale gross down when VIX-zscore is high, up when calm
  3. Earnings event filter   — flatten any V3 name within +-N trading days of earnings
  4. Mean-reversion sleeve   — small contrarian sleeve buying gap-down names
  5. FF5+Mom residual signal — long-only sleeve of names with strongest 63d alpha
                                (residual return after FF5+Mom regression)

For each overlay we report OOS Sh/Ann/MaxDD/Calmar vs V3 baseline.  Zero leverage
(gross <= 1.0 x capital) is enforced after every overlay.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import v1.portfolio.portfolio as _pp
from v1.portfolio.portfolio import (
    top_n_adx_momt_ac_sizes,
    diversifier_sleeve_overlay,
    portfolio_returns,
    defensive_tilt_overlay,
    _get_macro,
    SIGNAL_DIR, FEATURE_DIR, CAPITAL,
)
from v1.pipeline.data_pipeline import TICKERS, TICKER_LIST
from v1.pipeline.signal_generation import ATR_PARTIAL_REMAIN
from v1.pipeline.backtester import sharpe_ratio, max_drawdown

OOS_WARMUP = 756
EARN_PATH = ROOT / "data" / "v1" / "raw" / "earnings_dates.json"
FF_PATH   = ROOT / "data" / "shared" / "macro" / "famafrench" / "ff5_mom_daily.parquet"


def metrics(r):
    if len(r) < 30:
        return {}
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    mdd = max_drawdown((1 + r).cumprod())
    return dict(
        sharpe=sharpe_ratio(r), ann=ann, mdd=mdd,
        calmar=ann / abs(mdd) if mdd < 0 else float("nan"),
        vol=r.std() * np.sqrt(252),
    )


def load_inputs():
    multi = pd.read_parquet(SIGNAL_DIR / "multi_signals.parquet")
    feats: dict = {}
    rets = pd.DataFrame()
    for t in TICKER_LIST:
        fp = FEATURE_DIR / f"{t}.parquet"
        if not fp.exists():
            continue
        f = pd.read_parquet(fp)
        feats[t] = f
        rets[t] = f["log_return"]
    rets = rets.dropna()
    sig = multi.reindex(rets.index).fillna(0.0)
    hs_path = SIGNAL_DIR / "half_size.parquet"
    if hs_path.exists():
        hs = pd.read_parquet(hs_path).reindex(rets.index).fillna(False)
        hs = hs.reindex(columns=sig.columns, fill_value=False)
        sig = sig * np.where(hs, ATR_PARTIAL_REMAIN, 1.0)
    return sig, feats, rets, _get_macro()


def make_v3_sizer():
    base_kw = dict(
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )

    def sizer(sig, feats, rets, macro):
        s = top_n_adx_momt_ac_sizes(sig, feats, rets, CAPITAL, **base_kw)
        s = defensive_tilt_overlay(s, sig, macro, CAPITAL)
        s = diversifier_sleeve_overlay(
            s, CAPITAL,
            sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
            sleeve_pct=0.12,
        )
        return s

    return sizer


def zero_lev_clip(sizes: pd.DataFrame, capital: float = CAPITAL) -> pd.DataFrame:
    gross = sizes.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return sizes.multiply(cap_scale, axis=0)


# ============================================================================
# 1. TREND-QUALITY FILTER
# ============================================================================
def trend_quality_filter(sizes: pd.DataFrame, rets: pd.DataFrame,
                          lookback: int = 63, top_k_days: int = 3,
                          quality_floor: float = 0.5,
                          punish_factor: float = 0.5) -> pd.DataFrame:
    """
    For each long position, compute what fraction of the 63d cum log-return
    came from the top-K abs-return days.  If > quality_floor, halve position
    (signal that the trend is fragile / driven by 1-2 jumps).
    """
    out = sizes.copy()
    cands = [c for c in sizes.columns if c in rets.columns]
    for t in cands:
        r = rets[t]
        roll_sum = r.rolling(lookback).sum()
        # Rolling sum of top-K abs values via apply (acceptable: small universe)
        top_sum = r.abs().rolling(lookback).apply(
            lambda x: np.sort(np.abs(x))[-top_k_days:].sum(), raw=True
        )
        ratio = (top_sum / roll_sum.abs().replace(0, np.nan)).clip(0, 5)
        bad = (ratio > quality_floor).reindex(out.index).fillna(False).values
        col = out[t].values.astype(float)
        col = np.where(bad & (col > 0), col * punish_factor, col)
        out[t] = col
    return zero_lev_clip(out)


# ============================================================================
# 2. VOL-CARRY OVERLAY (VIX scaling)
# ============================================================================
def vol_carry_overlay(sizes: pd.DataFrame, macro: pd.DataFrame,
                       calm_z: float = -0.5, fear_z: float = 1.5,
                       calm_mult: float = 1.0, fear_mult: float = 0.6) -> pd.DataFrame:
    """
    Scale daily gross by VIX zscore.  Calm regime (z <= calm_z) -> calm_mult.
    Fear regime (z >= fear_z) -> fear_mult.  Linear interp between.
    Zero leverage cap means calm_mult > 1 won't add leverage, so we use 1.0
    for calm and trim for fear (asymmetric: cut risk in stress).
    """
    if "vix_zscore" not in macro.columns:
        return sizes
    z = macro["vix_zscore"].reindex(sizes.index).ffill().bfill()
    # Linear ramp: at calm_z -> calm_mult, at fear_z -> fear_mult
    span = max(fear_z - calm_z, 1e-6)
    raw = calm_mult + (fear_mult - calm_mult) * ((z - calm_z) / span).clip(0, 1)
    mult = raw.clip(lower=fear_mult, upper=calm_mult)
    return sizes.multiply(mult, axis=0)


# ============================================================================
# 3. EARNINGS EVENT FILTER
# ============================================================================
def earnings_event_filter(sizes: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """
    Zero out positions in any name within +-`window` trading days of an
    earnings announcement (idiosyncratic gap risk).
    """
    if not EARN_PATH.exists():
        return sizes
    earn = json.load(open(EARN_PATH))
    out = sizes.copy()
    idx = out.index
    for t, dates in earn.items():
        if t not in out.columns:
            continue
        mask = pd.Series(False, index=idx)
        for d in dates:
            try:
                d = pd.Timestamp(d)
            except Exception:
                continue
            # Find nearest trading day, mask +-window
            pos = idx.searchsorted(d)
            lo = max(0, pos - window)
            hi = min(len(idx), pos + window + 1)
            mask.iloc[lo:hi] = True
        out.loc[mask, t] = 0.0
    return out


# ============================================================================
# 4. MEAN-REVERSION SLEEVE (gap-down buy)
# ============================================================================
def mean_rev_sleeve(rets: pd.DataFrame, capital: float, candidates: list,
                     z_thresh: float = -2.0, hold_days: int = 2,
                     sleeve_pct: float = 0.05, lookback: int = 60) -> pd.DataFrame:
    """
    Each day, rank current 1d returns vs name's recent vol.  Names with
    z <= -z_thresh get a long position for `hold_days` (mean-revert).
    Equal-weight within picks; total dollar exposure capped at sleeve_pct.
    """
    cands = [t for t in candidates if t in rets.columns]
    R = rets[cands]
    vol = R.rolling(lookback).std()
    z = R / vol.replace(0, np.nan)
    triggers = (z <= z_thresh).fillna(False)
    sizes = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    # Forward-fill triggers for hold_days
    held = triggers.copy()
    for shift in range(1, hold_days):
        held = held | triggers.shift(shift).fillna(False)
    # Daily count of active picks; size each pick equally up to sleeve_pct
    n_active = held.sum(axis=1).replace(0, np.nan)
    per_name = (sleeve_pct * capital) / n_active
    for t in cands:
        sizes[t] = held[t].astype(float) * per_name.fillna(0)
    return sizes.fillna(0.0)


# ============================================================================
# 5. FAMA-FRENCH RESIDUAL ALPHA SIGNAL
# ============================================================================
def ff_residual_sleeve(rets: pd.DataFrame, capital: float, candidates: list,
                        regress_window: int = 252, mom_window: int = 63,
                        k: int = 5, sleeve_pct: float = 0.10,
                        rebal_freq: str = "ME") -> pd.DataFrame:
    """
    For each name in `candidates`, run rolling 252d regression of daily return
    on FF5 + Mom factors.  Compute residual (idiosyncratic) returns.  Each
    rebalance, rank names by trailing `mom_window` cumulative residual; long
    top-K equal-weight.  This is "alpha momentum" — return after stripping
    market/size/value/quality/investment/momentum factor exposure.
    """
    if not FF_PATH.exists():
        return pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    ff = pd.read_parquet(FF_PATH)
    # FF data is in % daily returns; convert to decimal
    ff = ff / 100.0
    # Align to portfolio daily index
    ff = ff.reindex(rets.index).ffill()
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    F = ff[factor_cols].fillna(0.0).values  # T x 6
    cands = [t for t in candidates if t in rets.columns]
    residuals = pd.DataFrame(index=rets.index, columns=cands, dtype=float)

    rf = ff["RF"].fillna(0.0).values

    for t in cands:
        y_full = rets[t].values - rf  # excess return
        res = np.full_like(y_full, np.nan)
        for end in range(regress_window, len(y_full)):
            start = end - regress_window
            X = F[start:end]
            y = y_full[start:end]
            mask = ~np.isnan(y) & np.isfinite(y)
            if mask.sum() < regress_window // 2:
                continue
            X_m, y_m = X[mask], y[mask]
            try:
                # OLS: beta = (X'X)^-1 X'y
                XtX = X_m.T @ X_m
                Xty = X_m.T @ y_m
                beta = np.linalg.solve(XtX + 1e-8 * np.eye(6), Xty)
            except Exception:
                continue
            # Today's residual = today's excess return - factor exposure
            res[end] = y_full[end] - F[end] @ beta
        residuals[t] = res

    # Rank by trailing mom_window cumulative residual at each rebal
    cum_res = residuals.rolling(mom_window, min_periods=int(mom_window * 0.6)).sum()
    sizes = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    eom = rets.index.to_series().groupby(pd.Grouper(freq=rebal_freq)).max()
    for d in eom:
        if d not in cum_res.index:
            continue
        row = cum_res.loc[d].dropna()
        if len(row) < k:
            continue
        longs = row.sort_values().index[-k:].tolist()
        each = (sleeve_pct * capital) / k
        for t in longs:
            sizes.loc[d, t] = each
    return sizes.replace(0.0, np.nan).ffill().fillna(0.0)


# ============================================================================
# DRIVER
# ============================================================================
def add_sleeve_to_v3(sizes_v3: pd.DataFrame, sleeve: pd.DataFrame,
                      sleeve_pct: float, capital: float = CAPITAL) -> pd.DataFrame:
    """V3 reduced by sleeve_pct; sleeve added; final zero-lev clip."""
    v3_scaled = sizes_v3 * (1.0 - sleeve_pct)
    combined = v3_scaled.add(sleeve, fill_value=0.0)
    return zero_lev_clip(combined, capital)


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in {time.time()-t0:.1f}s")

    stocks = [t for t, c in TICKERS.items() if c == "stock" and t in rets.columns]
    sector_etfs = [t for t, c in TICKERS.items() if c == "sector_etf" and t in rets.columns]
    universe = stocks + sector_etfs

    # V3 baseline
    sizer_v3 = make_v3_sizer()
    sizes_v3 = sizer_v3(sig, feats, rets, macro)
    pr_v3 = portfolio_returns(sizes_v3, rets).dropna()
    m_v3 = metrics(pr_v3.iloc[OOS_WARMUP:])

    print("\n" + "=" * 92)
    print("V3 BASELINE + each candidate overlay (OOS slice after 756 day warmup)")
    print("=" * 92)
    print(f"{'Variant':<60} {'Sh':>6} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    print(f"{'V3 baseline':<60} {m_v3['sharpe']:>6.2f} "
          f"{m_v3['ann']*100:>6.2f}% {m_v3['mdd']*100:>6.2f}% "
          f"{m_v3['calmar']:>5.2f} {m_v3['vol']*100:>5.1f}%")
    print("-" * 92)

    # ---- 1. Trend-quality filter ----
    print("\n[1] TREND-QUALITY FILTER (drop/halve names with concentrated trend)")
    for floor, punish in [(0.40, 0.5), (0.50, 0.5), (0.60, 0.5),
                           (0.50, 0.0), (0.40, 0.0)]:
        s = trend_quality_filter(sizes_v3, rets,
                                  quality_floor=floor, punish_factor=punish)
        pr = portfolio_returns(s, rets).dropna()
        m = metrics(pr.iloc[OOS_WARMUP:])
        label = f"V3 + tq floor={floor:.2f} punish={punish:.1f}"
        print(f"{label:<60} {m['sharpe']:>6.2f} "
              f"{m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # ---- 2. Vol-carry overlay ----
    print("\n[2] VOL-CARRY OVERLAY (scale gross down in high VIX-z)")
    for cz, fz, cm, fm in [(-0.5, 1.5, 1.0, 0.7),
                            (-0.5, 1.5, 1.0, 0.5),
                            (0.0, 1.0, 1.0, 0.6),
                            (0.0, 2.0, 1.0, 0.5),
                            (-1.0, 2.0, 1.0, 0.4)]:
        s = vol_carry_overlay(sizes_v3, macro, cz, fz, cm, fm)
        pr = portfolio_returns(s, rets).dropna()
        m = metrics(pr.iloc[OOS_WARMUP:])
        label = f"V3 + vc calm={cz:+.1f}->1.0 fear={fz:+.1f}->{fm:.1f}"
        print(f"{label:<60} {m['sharpe']:>6.2f} "
              f"{m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # ---- 3. Earnings event filter ----
    print("\n[3] EARNINGS EVENT FILTER (flatten positions ±N days of earnings)")
    for w in [1, 2, 3, 5]:
        s = earnings_event_filter(sizes_v3, window=w)
        pr = portfolio_returns(s, rets).dropna()
        m = metrics(pr.iloc[OOS_WARMUP:])
        label = f"V3 + earn ±{w}d"
        print(f"{label:<60} {m['sharpe']:>6.2f} "
              f"{m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # ---- 4. Mean-reversion sleeve ----
    print("\n[4] MEAN-REV SLEEVE (gap-down buy, 2-day hold)")
    for sp, zt, hd in [(0.05, -2.0, 2), (0.05, -2.5, 2),
                        (0.10, -2.0, 2), (0.10, -2.5, 3),
                        (0.05, -1.5, 2)]:
        sleeve = mean_rev_sleeve(rets, CAPITAL, universe,
                                  z_thresh=zt, hold_days=hd, sleeve_pct=sp)
        s = add_sleeve_to_v3(sizes_v3, sleeve, sp)
        pr = portfolio_returns(s, rets).dropna()
        m = metrics(pr.iloc[OOS_WARMUP:])
        label = f"V3 + mr {int(sp*100)}% z={zt:.1f} hold={hd}d"
        print(f"{label:<60} {m['sharpe']:>6.2f} "
              f"{m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    # ---- 5. FF-residual sleeve ----
    print("\n[5] FF5+Mom RESIDUAL ALPHA SLEEVE (long top-K by 63d residual)")
    for sp, k, mw in [(0.05, 5, 63), (0.10, 5, 63), (0.15, 5, 63),
                       (0.10, 5, 126), (0.10, 7, 63)]:
        sleeve = ff_residual_sleeve(rets, CAPITAL, universe,
                                     k=k, sleeve_pct=sp, mom_window=mw)
        s = add_sleeve_to_v3(sizes_v3, sleeve, sp)
        pr = portfolio_returns(s, rets).dropna()
        m = metrics(pr.iloc[OOS_WARMUP:])
        label = f"V3 + ff {int(sp*100)}% K={k} mw={mw}"
        print(f"{label:<60} {m['sharpe']:>6.2f} "
              f"{m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
