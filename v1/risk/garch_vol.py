"""
garch_vol.py
------------
GARCH(1,1) conditional volatility forecasts for every V1 ticker.

Why GARCH instead of rolling std?
─────────────────────────────────
Rolling realised vol uses a flat window — it treats every day in the window
equally, so the response to a vol spike lags by ~half the window length.
GARCH(1,1) captures volatility clustering: the conditional variance equation

    σ²_t  =  ω  +  α · ε²_{t-1}  +  β · σ²_{t-1}

gives this period's vol a direct dependency on yesterday's shock and yesterday's
variance.  In practice this means vol estimates *rise during the spike*, not
~30 days after — letting the position sizer cut exposure before drawdowns
rather than after.

Walk-forward design (no look-ahead)
───────────────────────────────────
For each ticker:
  1. Fit GARCH(1,1) on the initial in-sample window (default 504 days = 2 years).
  2. Use the fitted (ω, α, β) to roll the conditional variance forward day-by-
     day via the GARCH recursion (ε² and σ² from the prior bar only).
  3. Refit on an EXPANDING window every REFIT_DAYS (default 63 = quarterly).
     Each refit uses returns up to and including day t, so the new parameters
     apply strictly from day t+1 onward — no look-ahead.

Output
──────
data/v1/risk/garch_conditional_vol.parquet
    Wide DataFrame (T × N) of annualised conditional vol per ticker.
    Indexed by date, columns are tickers.

Consumed by
───────────
portfolio.py  — garch_kelly_sizes() uses this matrix for vol-conditional sizing.
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

# ── GARCH walk-forward parameters ────────────────────────────────────────────
INITIAL_WINDOW = 504    # 2 years of daily returns for the first fit
REFIT_DAYS     = 63     # refit on expanding window every quarter
ANNUALISATION  = np.sqrt(252)
MIN_RETURNS    = 252    # require at least 1 year of returns to fit GARCH


def _fit_garch_params(returns: np.ndarray) -> tuple[float, float, float, float] | None:
    """
    Fit GARCH(1,1) on a returns array and return (ω, α, β, σ²_last).

    Returns are scaled by 100 for fitting (arch package convention — avoids
    the "DataScaleWarning" and improves numerical conditioning), then the
    fitted ω is unscaled at the end so the variance is in raw return units.

    Args:
        returns: 1D numpy array of daily log returns (raw, not scaled).

    Returns:
        Tuple of (omega, alpha, beta, last_conditional_variance_raw)
        or None if the fit failed to converge.
    """
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
    # Stationarity guard: alpha + beta < 1 is required for the GARCH process to
    # be stationary.  When it isn't, fall back to long-run variance.
    if alpha + beta >= 0.999:
        return None

    # Unscale: variance scales by 100², so divide by 10_000 to map back.
    omega       = omega_s / 10_000.0
    cond_vol_s  = np.asarray(res.conditional_volatility)
    last_var_s  = float(cond_vol_s[-1] ** 2)
    last_var    = last_var_s / 10_000.0
    return omega, alpha, beta, last_var


def _roll_conditional_var(
    returns: np.ndarray, omega: float, alpha: float, beta: float, var0: float
) -> np.ndarray:
    """
    Roll the GARCH conditional variance forward day-by-day from a starting
    variance, using the supplied (ω, α, β).

    Recursion:
      σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}

    Args:
        returns: Daily log returns BETWEEN refit dates (1D array).
        omega:   GARCH constant.
        alpha:   ARCH coefficient on lagged squared shock.
        beta:    GARCH coefficient on lagged variance.
        var0:    Starting conditional variance (from the most recent fit).

    Returns:
        Conditional variance series of the same length as `returns`.  Each
        entry is the GARCH forecast for the variance OF that day, formed
        from the prior day's shock and variance only.
    """
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
    """
    Walk-forward GARCH(1,1) conditional vol forecast for one ticker.

    Strict no-look-ahead:
      - First INITIAL_WINDOW bars produce no forecast (NaN).
      - From bar INITIAL_WINDOW onward, fit GARCH on returns[0:INITIAL_WINDOW]
        and compute the next-day vol forecast.
      - Every REFIT_DAYS, refit on the EXPANDING window of returns observed so
        far.  Between refits, roll the conditional variance forward via the
        GARCH recursion using the most-recent fitted parameters.
      - If a refit fails to converge, keep the prior parameters.

    Args:
        returns: Daily log return Series indexed by date.

    Returns:
        Annualised conditional vol Series aligned to `returns.index`.
        NaN for the first INITIAL_WINDOW bars.
    """
    r = returns.dropna().values
    n = len(r)
    if n <= INITIAL_WINDOW:
        return pd.Series(np.nan, index=returns.index, dtype=float)

    out      = np.full(n, np.nan, dtype=float)
    fit_idx  = INITIAL_WINDOW
    params   = _fit_garch_params(r[:fit_idx])
    if params is None:
        # Fall back to long-run sample std if even the initial fit fails.
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
            # Refit on EXPANDING window of all returns observed so far.
            new_params = _fit_garch_params(r[:fit_idx])
            if new_params is not None:
                omega, alpha, beta, var_after_fit = new_params
                # Use the most recent in-sample conditional variance as the
                # starting point for the next block — keeps the recursion
                # consistent across the refit boundary.
                var_prev = var_after_fit
            next_refit = fit_idx + REFIT_DAYS

    cond_vol = np.sqrt(out) * ANNUALISATION
    full_index = returns.index
    aligned = pd.Series(np.nan, index=full_index, dtype=float)
    aligned.loc[returns.dropna().index] = cond_vol
    return aligned


def build_garch_matrix() -> pd.DataFrame:
    """
    Build the wide GARCH conditional vol matrix for every ticker in TICKER_LIST.

    Returns:
        Wide DataFrame (T × N) of annualised conditional vol.  Tickers without
        feature data are silently omitted.
    """
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
