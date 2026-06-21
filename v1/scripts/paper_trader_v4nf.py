"""
paper_trader_v4nf.py — FROZEN V4N-F overlay snapshot (pinned 2026-05-11).

DO NOT EDIT.  This is a snapshot of the V4N-F production sizer + every overlay
it depends on, frozen so the live 3-month test (data/v1/paper_trading_v4nf_3mo)
keeps trading the exact V4N-F logic regardless of what changes happen to
paper_trader.py during ongoing research.

Routing: paper_trader._compute_topn_v2_targets() delegates to
compute_v4nf_targets() in this module when PT_INSTANCE=v4nf_3mo.

Risk constants (RISK_PER_TRADE, MAX_POSITION_PCT, INDEX_ETF_CAP, INDEX_ETF_TICKERS)
are inlined here so future edits to paper_trader.py / portfolio.py cannot leak in.

Test window: 2026-05-11 → 2026-08-11.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Static lookups (safe to import — these rarely change) ────────────────────
from v1.pipeline.data_pipeline import ASSET_CLASS

# ── FROZEN constants (snapshotted from paper_trader.py / portfolio.py 2026-05-11) ──
_RISK_PER_TRADE    = 0.01
_MAX_POSITION_PCT  = 0.12
_INDEX_ETF_CAP     = 0.08
_INDEX_ETF_TICKERS = {"SPY", "IWM", "EEM", "EFA", "VWO"}

# ── Module-level VIX cache (independent from paper_trader._VIX_STATE_CACHE) ──
_V4NF_VIX_CACHE: dict | None = None


def _v4nf_get_vix_state(roc_days: int = 5) -> dict:
    """Frozen VIX z-score + ROC fetch.  Returns {z, roc, ok}."""
    global _V4NF_VIX_CACHE
    if _V4NF_VIX_CACHE is not None:
        return _V4NF_VIX_CACHE
    state = {"z": float("nan"), "roc": 0.0, "ok": False}
    vix = None
    try:
        h = yf.Ticker("^VIX").history(period="120d", auto_adjust=True)
        if h is not None and not h.empty and "Close" in h.columns:
            vix = h["Close"].dropna()
    except Exception as _e:
        print(f"  [v4nf vix] Ticker.history failed ({_e}) — trying download")
    if vix is None or vix.empty:
        try:
            d = yf.download("^VIX", period="120d", interval="1d",
                            auto_adjust=True, progress=False)
            if d is not None and not d.empty and "Close" in d.columns:
                col = d["Close"]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                vix = col.dropna()
        except Exception as _e:
            print(f"  [v4nf vix] download failed ({_e})")
    if vix is None or len(vix) < 60 + roc_days + 1:
        print("  [v4nf vix] insufficient history — assuming calm")
        _V4NF_VIX_CACHE = state
        return state
    roll = vix.rolling(60)
    z_series = (vix - roll.mean()) / roll.std()
    state = {
        "z":   float(z_series.iloc[-1]),
        "roc": float(vix.iloc[-1] - vix.iloc[-1 - roc_days]),
        "ok":  True,
    }
    _V4NF_VIX_CACHE = state
    return state


def _v4nf_is_bull_regime(spy_ma: int = 50, vix_z_thresh: float = 0.0) -> bool:
    """Frozen bull-regime probe: SPY > spy_ma-day MA AND VIX z <= threshold."""
    try:
        spy_path = Path("data/v1/features/SPY.parquet")
        if not spy_path.exists():
            return False
        spy_close = pd.read_parquet(spy_path)["Close"].dropna()
        if len(spy_close) < spy_ma + 1:
            return False
        spy_avg = float(spy_close.rolling(spy_ma).mean().iloc[-1])
        spy_now = float(spy_close.iloc[-1])
        if not (spy_now > spy_avg):
            return False
        st = _v4nf_get_vix_state()
        if not st["ok"]:
            return False
        return float(st["z"]) <= vix_z_thresh
    except Exception as _e:
        print(f"  [v4nf bull_regime] probe failed ({_e}) — assuming non-bull")
        return False


def _v4nf_apply_profit_take(positions: dict, signals: dict | None,
                              lookback: int = 10, sigma_thresh: float = 1.5,
                              scale: float = 0.7) -> dict:
    """Frozen profit-take: scale positions where 10d z >= 1.5 by 0.7."""
    if not positions or not signals:
        return positions
    out = dict(positions)
    for t in list(out.keys()):
        if out[t] <= 0:
            continue
        s = signals.get(t)
        if not s:
            continue
        rets = s.get("log_returns") or []
        if len(rets) < lookback:
            continue
        r = np.asarray(rets[-lookback:], dtype=float)
        cum = float(r.sum())
        std = float(r.std() * np.sqrt(lookback))
        if std <= 0:
            continue
        z = cum / std
        if z >= sigma_thresh:
            out[t] = out[t] * scale
    return out


def _v4nf_get_cond_vol_carry_mult(fear_z: float = 1.5, roc_days: int = 5,
                                    fear_mult: float = 0.5) -> float:
    """Frozen cond vol-carry: 0.5x when VIX z >= 1.5 AND rising over 5d."""
    state = _v4nf_get_vix_state(roc_days=roc_days)
    if not state["ok"]:
        return 1.0
    if state["z"] >= fear_z and state["roc"] > 0:
        return fear_mult
    return 1.0


def _v4nf_apply_asym_vol_boost(positions: dict, calm_boost: float = 1.15,
                                 calm_z: float = -0.5, fear_cut: float = 0.9,
                                 fear_z: float = 1.0) -> dict:
    """Frozen asym vol boost: calm 1.15x (z<=-0.5), fear 0.9x (z>=+1.0)."""
    if not positions:
        return positions
    state = _v4nf_get_vix_state()
    if not state["ok"]:
        return positions
    z = state["z"]
    if z <= calm_z:
        scale = calm_boost
    elif z >= fear_z:
        scale = fear_cut
    else:
        return positions
    return {t: v * scale for t, v in positions.items()}


def _v4nf_apply_fear_topRS(positions: dict, signals: dict, top_k: int = 3,
                             fear_z: float = 1.0) -> dict:
    """Frozen fear top-RS concentration: in fear (z>=+1.0), keep top_k by ret_63d."""
    if not positions:
        return positions
    state = _v4nf_get_vix_state()
    if not state["ok"] or state["z"] < fear_z:
        return positions
    longs = {t: v for t, v in positions.items() if v > 0}
    if len(longs) < top_k:
        return positions
    rs = {}
    for t in longs:
        s = signals.get(t)
        if not s:
            continue
        r = s.get("ret_63d")
        if r is None or (isinstance(r, float) and np.isnan(r)):
            continue
        rs[t] = float(r)
    if len(rs) < top_k:
        return positions
    keep = set(sorted(rs, key=lambda t: -rs[t])[:top_k])
    drop = [t for t in longs if t not in keep]
    dropped_notional = sum(longs[t] for t in drop)
    each = dropped_notional / top_k
    out = dict(positions)
    for t in keep:
        out[t] = out[t] + each
    for t in drop:
        out.pop(t, None)
    return out


def _v4nf_apply_accel_kicker(positions: dict, signals: dict,
                               accel_thresh: float = 1.05,
                               accel_boost: float = 1.35,
                               short_w: int = 10, long_w: int = 42) -> dict:
    """Frozen accel kicker: boost names where 10d/42d cum ratio >= 1.05 by 1.35x."""
    if not positions:
        return positions
    longs = {t: v for t, v in positions.items() if v > 0}
    if not longs:
        return positions
    ratios: dict[str, float] = {}
    for t in longs:
        s = signals.get(t)
        if not s:
            continue
        rets = s.get("log_returns") or []
        if len(rets) < long_w + 1:
            continue
        prior = rets[:-1]
        s_sum = float(np.sum(prior[-short_w:]))
        l_sum = float(np.sum(prior[-long_w:]))
        ratios[t] = (1.0 + s_sum) / (1.0 + l_sum) if (1.0 + l_sum) != 0 else 1.0
    if not ratios:
        return positions
    accel = [t for t, r in ratios.items() if r >= accel_thresh]
    if not accel:
        return positions
    out = dict(positions)
    gross_before = sum(longs.values())
    for t in accel:
        out[t] = out[t] * accel_boost
    new_long_sum = sum(out[t] for t in longs)
    scale = gross_before / new_long_sum if new_long_sum > 0 else 1.0
    for t in longs:
        out[t] = out[t] * scale
    return out


def _v4nf_apply_bull_sleeve_swap(positions: dict, pv: float,
                                   src: str = "TLT", dst: str = "XLK",
                                   swap_pct: float = 0.10,
                                   spy_ma: int = 50,
                                   vix_z_thresh: float = 0.0) -> dict:
    """Frozen V4N-F bull sleeve swap: shave src, add to dst when bull."""
    if not positions or swap_pct <= 0:
        return positions
    if not _v4nf_is_bull_regime(spy_ma=spy_ma, vix_z_thresh=vix_z_thresh):
        return positions
    out = dict(positions)
    src_now = float(out.get(src, 0.0))
    if src_now <= 0:
        return out
    shave = min(swap_pct * pv, src_now)
    if shave <= 0:
        return out
    out[src] = src_now - shave
    if out[src] <= 1.0:
        out.pop(src, None)
    dst_cap = pv * _MAX_POSITION_PCT
    dst_now = float(out.get(dst, 0.0))
    out[dst] = min(dst_now + shave, dst_cap)
    return out


def compute_v4nf_targets(
    pv: float, signals: dict,
    *,
    load_history_fn,
    top_n: int = 11, target_vol: float = 0.14, scale_max: float = 2.5,
    vt_window: int = 63, lev_x: float = 1.5, max_gross: float = 1.0,
    adx_threshold: float = 22.0, mom_lo: float = 0.7, mom_hi: float = 1.3,
    ac_quota: float = 0.55, gross_floor: float = 0.95,
) -> dict:
    """
    FROZEN V4N-F target sizer (pinned 2026-05-11).

    Mirrors paper_trader._compute_topn_v2_targets at the moment V4N-F was
    promoted to production.  Wired via PT_INSTANCE=v4nf_3mo routing in
    paper_trader._compute_topn_v2_targets so the 3-month live test cannot
    drift if overlay code in paper_trader.py is later edited.

    `load_history_fn` is injected so this module stays decoupled from
    paper_trader's PT_DIR routing — caller passes paper_trader.load_history.
    """
    longs = {t: s for t, s in signals.items() if s.get("signal") == 1}
    if not longs:
        return {}

    # Step A: ADX filter
    if adx_threshold > 0:
        longs = {t: s for t, s in longs.items()
                 if float(s.get("adx", s.get("adx_14", 0)) or 0) >= adx_threshold}
        if not longs:
            return {}

    # Step 1: ATR base sizes
    base = {}
    for t, s in longs.items():
        atr   = float(s.get("atr", 0) or 0)
        close = float(s.get("close", 1) or 1)
        if atr <= 0:
            base[t] = pv / max(len(longs), 1)
        else:
            base[t] = (pv * _RISK_PER_TRADE / atr) * close

    # Step 2: vol-target scalar
    scaler = 1.0
    try:
        hist = load_history_fn()
        if not hist.empty and "portfolio_value" in hist.columns and len(hist) >= 20:
            pv_series = pd.Series(hist["portfolio_value"].astype(float).values,
                                  index=pd.to_datetime(hist["date"]))
            port_ret  = pv_series.pct_change().dropna().tail(vt_window)
            if len(port_ret) >= 10:
                realised = float(port_ret.std() * np.sqrt(252))
                if realised > 0:
                    scaler = float(np.clip(target_vol / realised, 0.3, scale_max))
    except Exception:
        pass

    # Step 3: top-N by adx × |signal|
    conviction = {}
    for t, s in longs.items():
        adx_val = float(s.get("adx", s.get("adx_14", 0)) or 0)
        sig_abs = abs(float(s.get("signal", 0)))
        conviction[t] = adx_val * sig_abs
    keep = set(t for t, _ in sorted(conviction.items(), key=lambda x: -x[1])[:top_n])

    # Step B: momentum tilt
    in_set = [t for t in longs if t in keep]
    mom_vals = [(t, float(longs[t].get("ret_63d", 0) or 0)) for t in in_set]
    if mom_vals:
        sorted_mom = sorted(mom_vals, key=lambda x: x[1])
        n = len(sorted_mom)
        tilt = {t: mom_lo + (mom_hi - mom_lo) * (i / max(n - 1, 1))
                for i, (t, _) in enumerate(sorted_mom)}
    else:
        tilt = {}

    # Steps 4-5: leverage scalar
    raw = {}
    for t in longs:
        if t in keep:
            raw[t] = base[t] * scaler * lev_x * tilt.get(t, 1.0)
        else:
            raw[t] = 0.0

    # Step C: AC quota
    if ac_quota and ac_quota > 0:
        gross_t = sum(abs(v) for v in raw.values())
        if gross_t > 0:
            by_cls = {}
            for t, v in raw.items():
                ac = ASSET_CLASS.get(t, "other")
                by_cls.setdefault(ac, []).append((t, v))
            for cls, items in by_cls.items():
                cls_g = sum(abs(v) for _, v in items)
                if cls_g > ac_quota * gross_t and cls_g > 0:
                    cls_scale = (ac_quota * gross_t) / cls_g
                    for t, _ in items:
                        raw[t] *= cls_scale

    # Final gross targeting
    gross = sum(abs(v) for v in raw.values())
    if gross > 0:
        upper = (max_gross * pv) / gross
        lift  = (gross_floor * pv) / gross if gross_floor and gross_floor > 0 else upper
        gscale = min(lift, upper)
        if gscale != 1.0:
            raw = {t: v * gscale for t, v in raw.items()}

    # Per-name + index ETF caps
    out = {}
    for t, v in raw.items():
        if v <= 0:
            continue
        capped = min(v, pv * _MAX_POSITION_PCT)
        if t in _INDEX_ETF_TICKERS:
            capped = min(capped, pv * _INDEX_ETF_CAP)
        out[t] = capped

    # Diversifier sleeve (12% TLT/GLD/DBMF/VGSH)
    sleeve_tickers = ("TLT", "GLD", "DBMF", "VGSH")
    sleeve_pct     = 0.12
    available_sleeve = [t for t in sleeve_tickers if t in signals]
    if available_sleeve and sleeve_pct > 0:
        out = {t: v * (1.0 - sleeve_pct) for t, v in out.items()}
        each = (sleeve_pct * pv) / len(available_sleeve)
        for t in available_sleeve:
            out[t] = min(out.get(t, 0.0) + each, pv * _MAX_POSITION_PCT)

    # V4N-B
    out = _v4nf_apply_profit_take(out, signals=signals,
                                    lookback=10, sigma_thresh=1.5, scale=0.7)
    vc_mult = _v4nf_get_cond_vol_carry_mult(fear_z=1.5, roc_days=5, fear_mult=0.5)
    if vc_mult < 1.0:
        out = {t: v * vc_mult for t, v in out.items()}

    # V4N-D
    out = _v4nf_apply_asym_vol_boost(
        out, calm_boost=1.15, calm_z=-0.5, fear_cut=0.9, fear_z=1.0,
    )
    out = _v4nf_apply_fear_topRS(out, signals=signals, top_k=3, fear_z=1.0)

    # V4N-E
    out = _v4nf_apply_accel_kicker(
        out, signals=signals,
        accel_thresh=1.05, accel_boost=1.35, short_w=10, long_w=42,
    )

    # V4N-F
    out = _v4nf_apply_bull_sleeve_swap(
        out, pv=pv, src="TLT", dst="XLK", swap_pct=0.10,
        spy_ma=50, vix_z_thresh=0.0,
    )

    # Final gross cap
    gross2 = sum(out.values())
    if gross2 > max_gross * pv and gross2 > 0:
        s = (max_gross * pv) / gross2
        out = {t: v * s for t, v in out.items()}
    return out


if __name__ == "__main__":
    os.environ.setdefault("PT_INSTANCE", "v4nf_3mo")
    from v1.scripts import paper_trader as _pt  # noqa: E402

    with _pt.use_pt_instance("v4nf_3mo"):
        _pt.main_cli()
