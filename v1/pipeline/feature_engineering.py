"""
feature_engineering.py — Transforms raw OHLCV into features for signal_generation.

Pure math: no I/O during transformation, no asset-class logic. Features are
computed identically per ticker; signal_generation.py decides which to act on.

Features:
  Technical: log_return, dollar_volume, true_range, atr_14, mom_5/20/60,
    rsi_14 (Wilder EWM), macd_line/signal/hist (12/26/9), adx/plus_di/minus_di,
    zscore_20/60, bollinger (20, 2σ): bb_middle/upper/lower/pct_b/bandwidth,
    volume_zscore, obv, obv_zscore.
  Vol regime: vol_regime (21d>252d), vol_ratio (5d/63d), garch_vol (RM EWMA λ=0.94).
  Cross-sectional (needs closes_matrix): xsec_mom_20/63 (pct rank), xsec_vol_rank,
    relative_strength (60d return − universe avg). All strictly backward-looking.

I/O:
  Reads  data/raw/{TICKER}.parquet, data/raw/closes_matrix.parquet (optional)
  Writes data/features/{TICKER}.parquet, data/features/returns_matrix.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path("data/v1/raw")
FEATURE_DIR = Path("data/v1/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

from .data_pipeline import TICKER_LIST, ASSET_CLASS


# ── Per-ticker technical features ──────────────────────────────────────────────

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Foundational features required by other indicators (ADX, RSI, MACD).
    Must be called first.

    TR = max(H−L, |H−prev_C|, |L−prev_C|) — accounts for overnight gaps.
    ATR_14 = 14d rolling mean of TR (Wilder).
    Returns df + log_return, dollar_volume, true_range, atr_14.
    """
    df = df.copy()

    df["log_return"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["dollar_volume"] = df["Close"] * df["Volume"]

    prev = df["Close"].shift(1)
    df["true_range"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)

    df["atr_14"] = df["true_range"].rolling(14, min_periods=1).mean()
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price momentum over 5, 20, 60 days (~1w, 1m, 3m).
    signal_generation.py requires all three to agree (3-way confirmation).
    Returns df + mom_5, mom_20, mom_60.
    """
    df = df.copy()
    df["mom_5"]  = df["Close"].pct_change(5)
    df["mom_20"] = df["Close"].pct_change(20)
    df["mom_60"] = df["Close"].pct_change(60)
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Wilder's RSI using EWM (com=window-1, α=1/window). Default window=14 (Wilder 1978).
    >70 overbought, <30 oversold, 50 neutral.
    Returns df + rsi_{window}.
    """
    df    = df.copy()
    delta = df["Close"].diff()
    gains  = delta.clip(lower=0)
    losses = delta.clip(upper=0).abs()

    avg_gain = gains.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = losses.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))  # >70 overbought, <30 oversold
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD (default 12/26/9). hist>0 bullish, hist<0 bearish, zero-cross =
    momentum reversal (used by signal_generation.momentum_rule()).
    Returns df + macd_line, macd_signal, macd_hist.
    """
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["macd_line"]   = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]  # crossing zero = momentum turning
    return df


def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ADX trend-strength (not direction). Default window=14.
    <20 choppy (unreliable), 20-25 developing, >25 confirmed, >40 strong.
    +DI>-DI uptrend, -DI>+DI downtrend.
    Used in signal_generation.py as regime filter for MA crossovers
    (without it, choppy markets cause whipsaws).
    Requires true_range from add_base_features(). Returns df + adx, plus_di, minus_di.
    """
    df = df.copy()
    high, low = df["High"], df["Low"]
    prev_high, prev_low = high.shift(1), low.shift(1)

    plus_dm  = (high - prev_high).clip(lower=0)
    plus_dm[plus_dm < (prev_low - low).clip(lower=0)] = 0

    minus_dm = (prev_low - low).clip(lower=0)
    minus_dm[minus_dm < (high - prev_high).clip(lower=0)] = 0

    atr_s    = df["true_range"].ewm(span=window, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=window,  adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(span=window, adjust=False).mean() / atr_s

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    df["adx"]      = dx.ewm(span=window, adjust=False).mean()
    df["plus_di"]  = plus_di   # +DI > -DI = uptrend
    df["minus_di"] = minus_di  # -DI > +DI = downtrend
    return df


def add_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling price z-score (std-devs from mean) over 20 and 60 days.
    Used by signal_generation.mean_reversion_rule(): -2 = long candidate,
    +2 = overextended.
    Returns df + zscore_20, zscore_60.
    """
    df = df.copy()
    for window in [20, 60]:
        roll = df["Close"].rolling(window)
        df[f"zscore_{window}"] = (df["Close"] - roll.mean()) / roll.std()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands (default 20, 2σ).
    pct_b: 0 lower (long), 0.5 mid, 1 upper (short), >1 breakout.
    bandwidth rising = vol expanding; falling = squeeze.
    Returns df + bb_middle, bb_upper, bb_lower, bb_pct_b, bb_bandwidth.
    """
    df     = df.copy()
    middle = df["Close"].rolling(window).mean()
    std    = df["Close"].rolling(window).std()
    upper  = middle + num_std * std
    lower  = middle - num_std * std

    df["bb_middle"]    = middle
    df["bb_upper"]     = upper
    df["bb_lower"]     = lower
    df["bb_pct_b"]     = (df["Close"] - lower) / (upper - lower)
    df["bb_bandwidth"] = (upper - lower) / middle
    return df


def add_volume_signals(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Volume-based features for flow confirmation.

    Args:
        df:     DataFrame with columns [Close, Volume].
        window: Rolling window for z-score computation (default 20).

    Returns:
        Copy of df with columns: volume_zscore, obv, obv_zscore.

    Notes:
        volume_zscore: How unusual is today's volume vs the recent average?
            High volume on an up day = institutional buying (bullish).
            High volume on a down day = institutional selling (bearish).

        OBV (On-Balance Volume): Adds volume on up days, subtracts on down days.
            Rising OBV while price is flat = accumulation (smart money buying).
            Falling OBV while price is flat = distribution (smart money selling).
            OBV divergence from price is a leading indicator of reversals.

        obv_zscore: Normalised OBV used as a flow-strength signal in
            compute_scores() in signal_generation.py.
    """
    df       = df.copy()
    vol_roll = df["Volume"].rolling(window)

    df["volume_zscore"] = (df["Volume"] - vol_roll.mean()) / vol_roll.std()
    df["obv"]           = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()

    obv_roll = df["obv"].rolling(window)
    df["obv_zscore"] = (df["obv"] - obv_roll.mean()) / obv_roll.std()
    return df


def add_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility regime features that capture volatility clustering and term structure.

    A golden cross (or any MA signal) occurring in a low-vol regime is far more
    reliable than one occurring during a vol spike.  These features let downstream
    models condition on the current volatility environment.

    Args:
        df: DataFrame with column [log_return] (added by add_base_features()).

    Returns:
        Copy of df with columns: vol_regime, vol_ratio, garch_vol.

    Features
    ────────
    vol_regime (binary)
        1  = short-term realized vol (21-day) is ABOVE long-term (252-day) →
             vol is expanding / regime is stressed.
        0  = vol is contracting or normal.
        Rationale: comparing a 1-month window to a 1-year window is the
        simplest non-parametric way to detect a vol-regime shift without
        overfitting to a specific threshold.

    vol_ratio (continuous, >0)
        5-day realized vol divided by 63-day realized vol.
        >1 = short-term stress elevated relative to the 3-month baseline.
        <1 = vol term structure is normal / in contango.
        This is the realized-vol analogue of the VIX9D/VIX term ratio in
        macro_features.py, but computed from actual price moves rather than
        implied vol.

    garch_vol (continuous, annualised %)
        RiskMetrics EWMA 1-day forward variance estimate:
            σ²_t = λ·σ²_{t-1} + (1−λ)·r²_{t-1},  λ = 0.94
        Reported as annualised daily vol: sqrt(σ²_t) × sqrt(252).
        λ = 0.94 is J.P. Morgan's original 1994 RiskMetrics constant for
        daily data — chosen to give a half-life of ~11 trading days, matching
        the empirically observed speed of vol mean-reversion in equities.

        Unlike simple rolling vol, EWMA vol reacts immediately to a vol spike
        and decays smoothly afterward — essential for capturing vol clustering
        (the "GARCH effect": large moves cluster in time).

    Implementation note on vol_regime / vol_ratio:
        All realized vols are annualised (×√252) before comparison so the
        ratio is dimensionless and comparable across different base-vol regimes.
        min_periods equals the window length so the first observation is only
        produced once the full window of data is available.
    """
    df = df.copy()
    r  = df["log_return"]

    # Realised vol at multiple horizons (annualised)
    rv5   = r.rolling(5,   min_periods=5).std()   * np.sqrt(252)
    rv21  = r.rolling(21,  min_periods=21).std()  * np.sqrt(252)
    rv63  = r.rolling(63,  min_periods=63).std()  * np.sqrt(252)
    rv252 = r.rolling(252, min_periods=252).std() * np.sqrt(252)

    # vol_regime: 1 when short-term vol is above long-term vol (expanding)
    df["vol_regime"] = (rv21 > rv252).astype(int)

    # vol_ratio: 5d / 63d realised vol term structure
    # >1 = short-term stress elevated vs 3-month baseline
    df["vol_ratio"] = rv5 / rv63

    # garch_vol: RiskMetrics EWMA variance, lambda = 0.94
    # σ²_t = λ·σ²_{t-1} + (1−λ)·r²_{t-1}
    # Equivalent to EWM of r² with alpha = 1 − λ = 0.06
    lam      = 0.94
    ewma_var = (r ** 2).ewm(alpha=1.0 - lam, adjust=False).mean()
    df["garch_vol"] = np.sqrt(ewma_var) * np.sqrt(252)  # annualised 1-day forward vol

    return df


# ── Cross-sectional features ───────────────────────────────────────────────────

def add_cross_sectional(
    df: pd.DataFrame,
    ticker: str,
    closes_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cross-sectional features capturing how this ticker ranks within
    the full universe on each date.

    Three features are added:

    xsec_mom_20 — percentile rank (0–1) of this ticker's 20-day return
        compared to every other ticker in the universe on the same date.
        0 = worst recent performer, 1 = best recent performer.
        Rank is computed cross-sectionally across tickers at each date T
        using only data available up to T (pct_change(20) looks 20 days
        back from T — no look-ahead).

    xsec_vol_rank — percentile rank (0–1) of this ticker's trailing
        20-day realised volatility (annualised, computed as the rolling
        std of daily log returns × √252) vs all other tickers at date T.
        0 = quietest ticker, 1 = most volatile.
        Uses rolling(20).std() — only uses data up to T ✓.

    relative_strength — this ticker's 60-day percentage return minus the
        equal-weighted average 60-day return across the full universe at
        date T.  Positive = outperformed the universe over the past 60
        days; negative = underperformed.
        Uses pct_change(60) on both sides — no look-ahead ✓.

    Args:
        df:             Single-ticker feature DataFrame (output of per-ticker
                        add_*() functions).  Index must be DatetimeIndex.
        ticker:         This ticker's symbol — used to select its column from
                        closes_matrix.
        closes_matrix:  DataFrame of aligned close prices for the full
                        universe (dates × tickers).  Sourced from
                        data/raw/closes_matrix.parquet written by
                        data_pipeline.main().

    Returns:
        Copy of df with four additional columns: xsec_mom_20,
        xsec_vol_rank, relative_strength, xsec_mom_63.

    Note:
        If ticker is not a column of closes_matrix, all three features
        are set to NaN and a warning is printed rather than raising.
        NaN rows are removed by the dropna() call in engineer().
    """
    df = df.copy()

    if ticker not in closes_matrix.columns:
        print(f"  Warning: {ticker} not found in closes_matrix — xsec features will be NaN")
        df["xsec_mom_20"]      = np.nan
        df["xsec_vol_rank"]    = np.nan
        df["relative_strength"] = np.nan
        df["xsec_mom_63"]      = np.nan
        return df

    # Align closes_matrix to the dates in this ticker's DataFrame.
    # forward-fill to handle sparse rows (e.g. cross-listed tickers with
    # different holiday calendars) without introducing look-ahead.
    closes = closes_matrix.reindex(df.index, method="ffill")

    # ── xsec_mom_20: 20-day return percentile rank across universe ─────────────
    mom20      = closes.pct_change(20)                # each ticker's 20d return at each T
    xsec_mom20 = mom20.rank(axis=1, pct=True, na_option="keep")[ticker]
    df["xsec_mom_20"] = xsec_mom20

    # ── xsec_vol_rank: 20-day realised vol percentile rank ─────────────────────
    log_ret      = np.log(closes / closes.shift(1))
    vol20        = log_ret.rolling(20).std() * np.sqrt(252)    # annualised
    xsec_vol_rk  = vol20.rank(axis=1, pct=True, na_option="keep")[ticker]
    df["xsec_vol_rank"] = xsec_vol_rk

    # ── relative_strength: 60-day return vs equal-weighted universe ────────────
    mom60           = closes.pct_change(60)
    universe_avg60  = mom60.mean(axis=1)               # equal-weight avg at each date
    df["relative_strength"] = mom60[ticker] - universe_avg60

    # ── xsec_mom_63: 63-day (3-month) return percentile rank ───────────────────
    # Jegadeesh-Titman (1993) momentum sweet spot: 63 trading days captures
    # the strongest and most-replicated momentum window.  Shorter windows
    # (< 20d) capture mean-reversion; longer (> 252d) capture reversal.
    # rank(pct=True) → [0, 1] percentile rank. 0 = worst, 1 = best.
    # Uses only trailing data (pct_change looks back) — no look-ahead. ✓
    mom63      = closes.pct_change(63)
    xsec_mom63 = mom63.rank(axis=1, pct=True, na_option="keep")[ticker]
    df["xsec_mom_63"] = xsec_mom63

    return df


def add_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Donchian channel breakout and squeeze detection features.

    donchian_high_20  — rolling 20-day high (resistance level)
    donchian_low_20   — rolling 20-day low  (support level)
    breakout_20       — 1 when close >= 20-day high (new high breakout)
    donchian_width_pct — channel width as fraction of close price
    squeeze           — 1 when width is in the bottom 20th percentile of its
                        own 252-day history (volatility compression before explosive move)

    All lookbacks are strictly backward — no look-ahead.

    References:
      Donchian channel: Richard Dennis / Turtle Traders (1983 — published).
      Squeeze percentile: Bollinger, "Bollinger on Bollinger Bands" (2001).
    """
    df["donchian_high_20"]   = df["High"].rolling(20).max()
    df["donchian_low_20"]    = df["Low"].rolling(20).min()
    df["breakout_20"]        = (df["Close"] >= df["donchian_high_20"]).astype(int)
    df["donchian_width_pct"] = (df["donchian_high_20"] - df["donchian_low_20"]) / df["Close"]
    width_rank = df["donchian_width_pct"].rolling(252, min_periods=63).rank(pct=True)
    df["squeeze"] = (width_rank < 0.20).astype(int)
    return df


# ── Pipeline entry point ───────────────────────────────────────────────────────

def engineer(
    df: pd.DataFrame,
    ticker: str = None,
    closes_matrix: pd.DataFrame = None,
    cross_asset_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Apply all feature transformations in the correct dependency order.

    Args:
        df:             Raw OHLCV DataFrame from data_pipeline.py.
        ticker:         Ticker symbol for cross-sectional feature computation.
                        If None, cross-sectional features are skipped.
        closes_matrix:  Full-universe aligned close prices DataFrame.
                        If None, cross-sectional features are skipped.
                        Supply both ticker and closes_matrix to enable
                        xsec_mom_20, xsec_vol_rank, and relative_strength.
        cross_asset_df: Cross-asset lead-lag features from
                        cross_asset_signals.py.  If provided, columns are
                        merged via left join + ffill so every ticker gets
                        the same market-level signals aligned to its dates.

    Returns:
        Feature-enriched DataFrame with all columns from each add_*
        function.  Rows with any NaN are dropped — this removes the
        warm-up period at the start of history where rolling windows
        do not yet have enough data.

    Order matters:
        add_base_features() MUST run first — it produces true_range and
        log_return which are required by add_adx(), add_rsi(), and
        add_volatility_regime().

    Backward compatibility:
        Calling engineer(df) with no extra arguments returns exactly the
        same output as before cross-sectional features were added.  The
        paper_trader and scheduler call this form and are unaffected.
    """
    df = add_base_features(df)
    df = add_momentum(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_adx(df)
    df = add_zscore(df)
    df = add_bollinger_bands(df)
    df = add_breakout_features(df)
    df = add_volume_signals(df)
    df = add_volatility_regime(df)

    if ticker is not None and closes_matrix is not None:
        df = add_cross_sectional(df, ticker, closes_matrix)

    if cross_asset_df is not None and not cross_asset_df.empty:
        # Left join: keep only dates already in df, forward-fill to handle
        # any gaps (e.g. cross-asset data may have slightly different trading
        # calendars).  These are market-level signals — identical for every ticker.
        ca = cross_asset_df.reindex(df.index).ffill()
        for col in ca.columns:
            df[col] = ca[col]

    df = df.dropna()
    return df


# ── Script entry point ─────────────────────────────────────────────────────────

def main():
    print("Engineering features...\n")

    # Load closes matrix for cross-sectional features.
    # Written by data_pipeline.main() — must exist before this step.
    closes_path = DATA_DIR / "closes_matrix.parquet"
    if closes_path.exists():
        closes_matrix = pd.read_parquet(closes_path)
        print(f"Closes matrix loaded: {closes_matrix.shape} (dates x tickers)\n")
    else:
        closes_matrix = None
        print("Warning: closes_matrix.parquet not found — cross-sectional features "
              "will be skipped. Run python run.py (full pipeline) to include them.\n")

    # Load cross-asset lead-lag features (from cross_asset_signals.py).
    ca_path = Path("data/v1/signals/cross_asset_features.parquet")
    if ca_path.exists():
        cross_asset_df = pd.read_parquet(ca_path)
        cross_asset_df.index = pd.to_datetime(cross_asset_df.index)
        print(f"Cross-asset features loaded: {cross_asset_df.shape} "
              f"({list(cross_asset_df.columns)})\n")
    else:
        cross_asset_df = None
        print("Warning: cross_asset_features.parquet not found — cross-asset features "
              "will be skipped. Run python -m v1.pipeline.cross_asset_signals first.\n")

    all_data = {}

    for ticker in TICKER_LIST:
        raw_path = DATA_DIR / f"{ticker}.parquet"
        if not raw_path.exists():
            print(f"  {ticker}: raw parquet missing — skipped (run data_pipeline.py first)")
            continue
        raw = pd.read_parquet(raw_path)
        df  = engineer(raw, ticker=ticker, closes_matrix=closes_matrix,
                       cross_asset_df=cross_asset_df)
        out = FEATURE_DIR / f"{ticker}.parquet"
        df.to_parquet(out, engine="pyarrow", compression="snappy")

        xsec_note = " + xsec" if closes_matrix is not None else ""
        ca_note   = " + ca"   if cross_asset_df is not None else ""
        print(f"  {ticker}: {len(df)} rows, {len(df.columns)} cols{xsec_note}{ca_note}  ->  {out}")
        all_data[ticker] = df

    returns = pd.DataFrame({t: d["log_return"] for t, d in all_data.items()}).dropna()
    returns.to_parquet(FEATURE_DIR / "returns_matrix.parquet")

    print(f"\nReturns matrix: {returns.shape}")
    print("\nAnnualised volatility:")
    for t in returns.columns:
        ann_vol = returns[t].std() * (252 ** 0.5) * 100
        print(f"  {t} ({ASSET_CLASS[t]}): {ann_vol:.1f}%")


if __name__ == "__main__":
    main()
