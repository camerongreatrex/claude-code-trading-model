"""
garch_vol.py — GARCH(1,1) conditional volatility forecasts per V1 ticker.
Recursion: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}. Walk-forward refit on expanding
window every REFIT_DAYS (no look-ahead). Output: data/v1/risk/garch_conditional_vol.parquet
consumed by portfolio.py garch_kelly_sizes().
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import DataScaleWarning, ConvergenceWarning

from v1.pipeline.data_pipeline import TICKER_LIST

FEATURE_DIR = Path("data/v1/features")
OUTPUT_DIR  = Path("data/v1/risk")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Walk-forward params ──────────────────────────────────────────────────────
INITIAL_WINDOW = 504    # 2 years for first fit
REFIT_DAYS     = 63     # quarterly expanding-window refit
ANNUALISATION  = np.sqrt(252)
MIN_RETURNS    = 252    # min 1 year to fit GARCH


def _fit_garch_params(returns: np.ndarray) -> tuple[float, float, float, float] | None:
    """Fit GARCH(1,1) on returns array; returns (ω, α, β, σ²_last) or None.
    Scales returns by 100 for fitting (arch convention) and unscales ω back."""
    if len(returns) < MIN_RETURNS:
        return None

    scaled = returns * 100.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataScaleWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            model = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
            res   = model.fit(disp="off", show_warning=False)
    except Exception:
        return None

    params = res.params
    omega_s = float(params.get("omega", np.nan))
    alpha   = float(params.get("alpha[1]", np.nan))
    beta    = float(params.get("beta[1]",  np.nan))

    if any(np.isnan(x) for x in (omega_s, alpha, beta)):
        return None
    # Stationarity guard: alpha + beta < 1 required.
    if alpha + beta >= 0.999:
        return None

    # Unscale variance by 100² = 10_000.
    omega       = omega_s / 10_000.0
    cond_vol_s  = np.asarray(res.conditional_volatility)
    last_var_s  = float(cond_vol_s[-1] ** 2)
    last_var    = last_var_s / 10_000.0
    return omega, alpha, beta, last_var


def _roll_conditional_var(
    returns: np.ndarray, omega: float, alpha: float, beta: float, var0: float
) -> np.ndarray:
    """Roll GARCH conditional variance forward via σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}.
    Each output entry is forecast for that day from prior shock/variance only."""
    out      = np.empty(len(returns), dtype=float)
    var_prev = max(var0, 1e-12)
    eps_prev = returns[0] if len(returns) > 0 else 0.0
    for i in range(len(returns)):
        var_t       = omega + alpha * (eps_prev ** 2) + beta * var_prev
        out[i]      = max(var_t, 1e-12)
        eps_prev    = returns[i]
        var_prev    = var_t
    return out


def garch_vol_series(returns: pd.Series) -> pd.Series:
    """Walk-forward GARCH(1,1) annualised cond vol for one ticker (no look-ahead).
    First INITIAL_WINDOW bars are NaN; refit on expanding window every REFIT_DAYS;
    keep prior params if refit fails."""
    r = returns.dropna().values
    n = len(r)
    if n <= INITIAL_WINDOW:
        return pd.Series(np.nan, index=returns.index, dtype=float)

    out      = np.full(n, np.nan, dtype=float)
    fit_idx  = INITIAL_WINDOW
    params   = _fit_garch_params(r[:fit_idx])
    if params is None:
        # Fall back to long-run sample variance if initial fit fails.
        long_run_var = float(np.var(r[:fit_idx]))
        params       = (long_run_var * 0.05, 0.1, 0.85, long_run_var)

    omega, alpha, beta, var_prev = params
    next_refit = fit_idx + REFIT_DAYS

    while fit_idx < n:
        end_block      = min(next_refit, n)
        block_returns  = r[fit_idx:end_block]
        block_var      = _roll_conditional_var(block_returns, omega, alpha, beta, var_prev)
        out[fit_idx:end_block] = block_var
        var_prev = float(block_var[-1])
        fit_idx  = end_block

        if fit_idx < n:
            # Refit on expanding window.
            new_params = _fit_garch_params(r[:fit_idx])
            if new_params is not None:
                omega, alpha, beta, var_after_fit = new_params
                # Seed next block with most recent in-sample variance.
                var_prev = var_after_fit
            next_refit = fit_idx + REFIT_DAYS

    cond_vol = np.sqrt(out) * ANNUALISATION
    full_index = returns.index
    aligned = pd.Series(np.nan, index=full_index, dtype=float)
    aligned.loc[returns.dropna().index] = cond_vol
    return aligned


def build_garch_matrix() -> pd.DataFrame:
    """Build wide GARCH cond vol matrix (T × N annualised) for TICKER_LIST.
    Tickers without feature data are silently omitted."""
    series = {}
    for ticker in TICKER_LIST:
        path = FEATURE_DIR / f"{ticker}.parquet"
        if not path.exists():
            print(f"  {ticker}: feature parquet missing — skipped")
            continue
        feat = pd.read_parquet(path)
        if "log_return" not in feat.columns:
            print(f"  {ticker}: no log_return column — skipped")
            continue
        vol = garch_vol_series(feat["log_return"])
        n_valid = int(vol.notna().sum())
        if n_valid == 0:
            print(f"  {ticker}: GARCH fit failed for all windows — skipped")
            continue
        series[ticker] = vol
        last = vol.dropna().iloc[-1]
        print(f"  {ticker:<6}: {n_valid} forecast days  |  latest cond vol {last*100:.1f}% annualised")

    matrix = pd.DataFrame(series).sort_index()
    return matrix


def main():
    print("Building GARCH(1,1) conditional volatility matrix...\n")
    matrix = build_garch_matrix()

    out_path = OUTPUT_DIR / "garch_conditional_vol.parquet"
    matrix.to_parquet(out_path)
    print(f"\nGARCH conditional vol matrix saved → {out_path}  shape={matrix.shape}")

    # Summary stats
    latest = matrix.dropna(how="all").iloc[-1].dropna()
    if len(latest) > 0:
        print(f"\nLatest conditional vol (annualised, %):")
        for t, v in latest.sort_values().items():
            print(f"  {t:<6} {v*100:>6.1f}%")


if __name__ == "__main__":
    main()
