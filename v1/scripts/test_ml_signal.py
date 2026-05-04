"""
ML viability probe: can HistGBM on per-ticker+macro features beat linear V3 momentum?
Target: 5d fwd log-return >0; rolling 504d train, retrain quarterly, 1d lagged feats.
Two evals: (A) standalone top-K sleeve; (B) layered V3 + ML sleeve, zero-lev clipped.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

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
from sklearn.ensemble import HistGradientBoostingClassifier

OOS_WARMUP = 756

PER_TICKER_FEATS = [
    "mom_5", "mom_20", "mom_60", "rsi_14", "adx", "plus_di", "minus_di",
    "macd_hist", "zscore_20", "zscore_60", "bb_pct_b", "volume_zscore",
]
MACRO_FEATS = ["vix_zscore", "yield_curve", "vix_term_ratio", "curve_momentum"]


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


def build_panel(feats: dict, macro: pd.DataFrame, rets: pd.DataFrame,
                 universe: list, fwd: int = 5) -> pd.DataFrame:
    """Long panel rows=(date,ticker); target = fwd-day cum log-return >0; feats lagged 1d."""
    macro_aligned = macro.reindex(rets.index).ffill()
    macro_view = macro_aligned[[c for c in MACRO_FEATS if c in macro_aligned.columns]]
    parts = []
    for t in universe:
        if t not in feats:
            continue
        df = feats[t].copy()
        cols = [c for c in PER_TICKER_FEATS if c in df.columns]
        if not cols:
            continue
        x = df[cols].copy()
        if "atr_14" in df.columns and "Close" in df.columns:
            x["atr_pct"] = (df["atr_14"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
        x = x.shift(1)
        x = x.join(macro_view, how="left")
        fwd_ret = rets[t].rolling(fwd).sum().shift(-fwd)
        x["target"] = (fwd_ret > 0).astype(int)
        x["fwd_ret"] = fwd_ret
        x["ticker"] = t
        x = x.dropna(subset=["target"])
        parts.append(x)
    panel = pd.concat(parts, axis=0)
    panel.index.name = "date"
    return panel


def walkforward_predict(panel: pd.DataFrame, train_days: int = 504,
                         retrain_freq: int = 63, min_train: int = 252) -> pd.Series:
    """Walk-forward predict P(target=1); train on last train_days, retrain every retrain_freq."""
    feature_cols = [c for c in panel.columns
                     if c not in ("target", "fwd_ret", "ticker")]
    panel = panel.copy()
    panel = panel.sort_index()
    dates = panel.index.unique().sort_values()
    preds = pd.Series(np.nan, index=panel.index, dtype=float)

    last_train_idx = -10**9
    model = None
    for i, d in enumerate(dates):
        if i < min_train:
            continue
        if i - last_train_idx >= retrain_freq or model is None:
            train_start_pos = max(0, i - train_days)
            train_dates = dates[train_start_pos: i]
            train = panel.loc[panel.index.isin(train_dates)]
            X = train[feature_cols].values
            y = train["target"].values
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            if mask.sum() < 200 or len(np.unique(y[mask])) < 2:
                continue
            model = HistGradientBoostingClassifier(
                max_depth=4, max_iter=120, learning_rate=0.05,
                min_samples_leaf=30, l2_regularization=1.0,
                random_state=42,
            )
            model.fit(X[mask], y[mask])
            last_train_idx = i
        today = panel.loc[panel.index == d]
        Xt = today[feature_cols].values
        mask = ~np.isnan(Xt).any(axis=1)
        if mask.sum() == 0:
            continue
        p = np.full(len(today), np.nan)
        p[mask] = model.predict_proba(Xt[mask])[:, 1]
        preds.loc[panel.index == d] = p
    return preds


def ml_sleeve_from_preds(panel: pd.DataFrame, preds: pd.Series, rets: pd.DataFrame,
                          k: int = 5, sleeve_pct: float = 0.10,
                          rebal_freq: str = "W") -> pd.DataFrame:
    """Each rebalance: long top-K by predicted prob, equal-weight; held until next rebal."""
    panel = panel.copy()
    panel["pred"] = preds.values
    sizes = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    each = (sleeve_pct * CAPITAL) / k
    rebal_dates = rets.index.to_series().groupby(pd.Grouper(freq=rebal_freq)).max()
    for d in rebal_dates:
        if d not in panel.index:
            continue
        day = panel.loc[panel.index == d, ["ticker", "pred"]].dropna()
        if len(day) < k:
            continue
        top = day.sort_values("pred", ascending=False).head(k)["ticker"].tolist()
        for t in top:
            if t in sizes.columns:
                sizes.loc[d, t] = each
    return sizes.replace(0.0, np.nan).ffill().fillna(0.0)


def add_sleeve_to_v3(sizes_v3: pd.DataFrame, sleeve: pd.DataFrame,
                      sleeve_pct: float, capital: float = CAPITAL) -> pd.DataFrame:
    v3_scaled = sizes_v3 * (1.0 - sleeve_pct)
    combined = v3_scaled.add(sleeve, fill_value=0.0)
    gross = combined.abs().sum(axis=1).replace(0, np.nan)
    cap_scale = (capital / gross).clip(upper=1.0).fillna(1.0)
    return combined.multiply(cap_scale, axis=0)


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in {time.time()-t0:.1f}s")

    stocks = [t for t, c in TICKERS.items() if c == "stock" and t in rets.columns]
    sectors = [t for t, c in TICKERS.items() if c == "sector_etf" and t in rets.columns]
    universe = stocks + sectors
    print(f"Universe: {len(universe)} tickers ({len(stocks)} stocks + "
          f"{len(sectors)} sectors)")

    # V3 baseline
    sizer_v3 = make_v3_sizer()
    sizes_v3 = sizer_v3(sig, feats, rets, macro)
    pr_v3 = portfolio_returns(sizes_v3, rets).dropna()
    m_v3 = metrics(pr_v3.iloc[OOS_WARMUP:])
    print(f"\nV3 baseline OOS: Sh {m_v3['sharpe']:.2f} Ann {m_v3['ann']*100:.2f}% "
          f"DD {m_v3['mdd']*100:.2f}% Cal {m_v3['calmar']:.2f}")

    # Build panel + walk-forward predict
    print("\nBuilding feature panel (5d forward target)...")
    t1 = time.time()
    panel = build_panel(feats, macro, rets, universe, fwd=5)
    print(f"  Panel: {len(panel):,} rows, "
          f"{len([c for c in panel.columns if c not in ('target','fwd_ret','ticker')])} features in {time.time()-t1:.1f}s")

    print("Walk-forward training (504d window, retrain quarterly)...")
    t1 = time.time()
    preds = walkforward_predict(panel, train_days=504, retrain_freq=63)
    print(f"  Predictions generated in {time.time()-t1:.1f}s "
          f"({preds.notna().sum():,} valid)")

    # Sanity: cross-sectional quintile spread
    panel_eval = panel.copy()
    panel_eval["pred"] = preds.values
    panel_eval = panel_eval.dropna(subset=["pred", "fwd_ret"])
    panel_eval["decile"] = (panel_eval.groupby(level=0)["pred"]
                              .transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")))
    decile_ret = panel_eval.groupby("decile")["fwd_ret"].mean() * 252 / 5
    print("\nQuintile fwd-return (annualized, 5d horizon):")
    for q, r in decile_ret.items():
        print(f"  Q{int(q)+1}: {r*100:+6.2f}%")
    spread = decile_ret.iloc[-1] - decile_ret.iloc[0]
    print(f"  Top-Bottom spread: {spread*100:+.2f}% — "
          f"{'POSITIVE' if spread > 0 else 'NEGATIVE'} signal")

    # ---- Standalone ML sleeve ----  (header)
    print("\n" + "=" * 92)
    print(f"{'Variant':<48} {'Sh':>6} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    print("=" * 92)
    print(f"{'V3 baseline':<48} {m_v3['sharpe']:>6.2f} "
          f"{m_v3['ann']*100:>6.2f}% {m_v3['mdd']*100:>6.2f}% "
          f"{m_v3['calmar']:>5.2f} {m_v3['vol']*100:>5.1f}%")

    print("\n[Standalone] ML sleeve only (sleeve_pct=1.0 = full capital):")
    for k, freq in [(3, "W"), (5, "W"), (5, "ME"), (7, "W"), (10, "W")]:
        sleeve = ml_sleeve_from_preds(panel, preds, rets,
                                       k=k, sleeve_pct=1.0, rebal_freq=freq)
        pr = portfolio_returns(sleeve, rets).dropna()
        m = metrics(pr.iloc[OOS_WARMUP:]) if len(pr) > OOS_WARMUP else metrics(pr)
        if not m:
            continue
        label = f"ML K={k} rebal={freq}"
        print(f"{label:<48} {m['sharpe']:>6.2f} "
              f"{m.get('ann', 0)*100:>6.2f}% {m.get('mdd', 0)*100:>6.2f}% "
              f"{m.get('calmar', float('nan')):>5.2f} {m.get('vol', 0)*100:>5.1f}%")

    # ---- Layered ----
    print("\n[Layered] V3 + ML sleeve at varying sizes:")
    for k, sp in [(5, 0.05), (5, 0.10), (5, 0.15), (7, 0.10), (3, 0.10)]:
        sleeve = ml_sleeve_from_preds(panel, preds, rets,
                                       k=k, sleeve_pct=sp, rebal_freq="W")
        s = add_sleeve_to_v3(sizes_v3, sleeve, sp)
        pr = portfolio_returns(s, rets).dropna()
        m = metrics(pr.iloc[OOS_WARMUP:])
        label = f"V3 + ML {int(sp*100)}% K={k}"
        print(f"{label:<48} {m['sharpe']:>6.2f} "
              f"{m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
              f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
