"""
ML as a second-opinion FILTER on V3 picks (not parallel alpha): classify which
V3 longs earn positive 5d return; downweight/skip low-confidence picks at test.
Conditioned on V3's filter (high-mom, ADX>22) so model learns residual variation.
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

from v1.portfolio.portfolio import (
    top_n_adx_momt_ac_sizes,
    diversifier_sleeve_overlay,
    portfolio_returns,
    defensive_tilt_overlay,
    _get_macro,
    SIGNAL_DIR, FEATURE_DIR, CAPITAL,
)
from v1.pipeline.data_pipeline import TICKER_LIST
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


def make_v3_sizes(sig, feats, rets, macro):
    base_kw = dict(
        top_n=11, target_vol=0.14, scale_max=2.5, vt_window=63,
        lev_x=1.5, max_gross=1.0,
        adx_threshold=22.0, mom_window=63, mom_lo=0.7, mom_hi=1.3,
        ac_quota=0.55, gross_floor=0.95,
    )
    s = top_n_adx_momt_ac_sizes(sig, feats, rets, CAPITAL, **base_kw)
    s = defensive_tilt_overlay(s, sig, macro, CAPITAL)
    s = diversifier_sleeve_overlay(
        s, CAPITAL,
        sleeve_tickers=("TLT", "GLD", "DBMF", "VGSH"),
        sleeve_pct=0.12,
    )
    return s


def build_position_panel(sizes_v3: pd.DataFrame, feats: dict,
                          macro: pd.DataFrame, rets: pd.DataFrame,
                          fwd: int = 5) -> pd.DataFrame:
    """Panel of (date, ticker) rows only where V3 holds long; target = fwd cum log-ret >0."""
    macro_aligned = macro.reindex(rets.index).ffill()
    macro_view = macro_aligned[[c for c in MACRO_FEATS if c in macro_aligned.columns]]
    parts = []
    for t in sizes_v3.columns:
        if t not in feats or t not in rets.columns:
            continue
        df = feats[t]
        cols = [c for c in PER_TICKER_FEATS if c in df.columns]
        if not cols:
            continue
        x = df[cols].copy()
        if "atr_14" in df.columns and "Close" in df.columns:
            x["atr_pct"] = (df["atr_14"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
        x = x.shift(1)
        x = x.join(macro_view, how="left")
        held = sizes_v3[t].reindex(x.index) > 0
        x = x[held]
        if len(x) == 0:
            continue
        fwd_ret = rets[t].rolling(fwd).sum().shift(-fwd)
        x["target"] = (fwd_ret.reindex(x.index) > 0).astype(int)
        x["fwd_ret"] = fwd_ret.reindex(x.index)
        x["ticker"] = t
        x = x.dropna(subset=["target"])
        parts.append(x)
    panel = pd.concat(parts, axis=0).sort_index()
    panel.index.name = "date"
    return panel


def walkforward_predict(panel: pd.DataFrame, train_days: int = 504,
                         retrain_freq: int = 63, min_train: int = 252) -> pd.Series:
    feature_cols = [c for c in panel.columns
                     if c not in ("target", "fwd_ret", "ticker")]
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
            if mask.sum() < 100 or len(np.unique(y[mask])) < 2:
                continue
            model = HistGradientBoostingClassifier(
                max_depth=3, max_iter=80, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=1.0,
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


def apply_filter(sizes_v3: pd.DataFrame, panel: pd.DataFrame, preds: pd.Series,
                  threshold: float = 0.5, mode: str = "skip",
                  scale_low: float = 0.5) -> pd.DataFrame:
    """Where pred<threshold: mode='skip' zero, mode='scale' x scale_low. No re-leverage."""
    out = sizes_v3.copy()
    panel = panel.copy()
    panel["pred"] = preds.values
    flagged = panel[panel["pred"].notna() & (panel["pred"] < threshold)]
    for (d, row) in flagged.iterrows():
        t = row["ticker"]
        if t not in out.columns:
            continue
        if mode == "skip":
            out.loc[d, t] = 0.0
        elif mode == "scale":
            out.loc[d, t] = out.loc[d, t] * scale_low
    return out


def main():
    t0 = time.time()
    sig, feats, rets, macro = load_inputs()
    print(f"Loaded {len(rets)} days, {len(rets.columns)} tickers in {time.time()-t0:.1f}s")

    sizes_v3 = make_v3_sizes(sig, feats, rets, macro)
    pr_v3 = portfolio_returns(sizes_v3, rets).dropna()
    m_v3 = metrics(pr_v3.iloc[OOS_WARMUP:])
    print(f"V3 baseline OOS: Sh {m_v3['sharpe']:.2f} Ann {m_v3['ann']*100:.2f}% "
          f"DD {m_v3['mdd']*100:.2f}% Cal {m_v3['calmar']:.2f}")

    print("\nBuilding position panel (V3 longs only)...")
    panel = build_position_panel(sizes_v3, feats, macro, rets, fwd=5)
    print(f"  {len(panel):,} (date, ticker) position rows; "
          f"{panel['target'].mean()*100:.1f}% positive (base rate)")

    print("Walk-forward predict P(V3 long earns positive 5d)...")
    preds = walkforward_predict(panel, train_days=504, retrain_freq=63)
    print(f"  Valid predictions: {preds.notna().sum():,}")

    # Sanity: are predictions actually informative?
    pe = panel.copy()
    pe["pred"] = preds.values
    pe = pe.dropna(subset=["pred", "fwd_ret"])
    pe["q"] = pe.groupby(level=0)["pred"].transform(
        lambda x: pd.qcut(x, 3, labels=False, duplicates="drop") if len(x) >= 3 else np.nan)
    print("\nTercile fwd-return (annualized, 5d):")
    for q, r in pe.groupby("q")["fwd_ret"].mean().items():
        print(f"  T{int(q)+1}: {r*252/5*100:+6.2f}%")

    # ----  filter sweep ----
    print("\n" + "=" * 88)
    print(f"{'Variant':<48} {'Sh':>6} {'Ann':>7} {'DD':>7} {'Cal':>5} {'Vol':>6}")
    print("=" * 88)
    print(f"{'V3 baseline':<48} {m_v3['sharpe']:>6.2f} "
          f"{m_v3['ann']*100:>6.2f}% {m_v3['mdd']*100:>6.2f}% "
          f"{m_v3['calmar']:>5.2f} {m_v3['vol']*100:>5.1f}%")

    for thr in [0.40, 0.45, 0.50, 0.55]:
        for mode in ["skip", "scale"]:
            kw = dict(threshold=thr, mode=mode)
            if mode == "scale":
                kw["scale_low"] = 0.5
            s = apply_filter(sizes_v3, panel, preds, **kw)
            pr = portfolio_returns(s, rets).dropna()
            m = metrics(pr.iloc[OOS_WARMUP:])
            label = f"V3 + ml-filter thr={thr:.2f} {mode}"
            print(f"{label:<48} {m['sharpe']:>6.2f} "
                  f"{m['ann']*100:>6.2f}% {m['mdd']*100:>6.2f}% "
                  f"{m['calmar']:>5.2f} {m['vol']*100:>5.1f}%")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
