"""
pipeline/feature_engineer.py
-----------------------------
Generates financial features from clean OHLCV data for risk model input.

Features produced:
  - log_return            : Daily log returns
  - rolling_volatility_20 : 20-day rolling std of log returns
  - momentum_14           : 14-day price momentum (close / close_14d_ago - 1)
  - rsi_14                : 14-period Relative Strength Index
  - volume_zscore         : Z-score normalized volume
  - beta_rolling          : Rolling 60-day beta vs benchmark (Nifty 50)

Output: data/processed/features_<symbol>.csv

Standalone:
    python pipeline/feature_engineer.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    BENCHMARK_SYMBOL,
    BETA_WINDOW,
    MOMENTUM_WINDOW,
    PROC_DATA_DIR,
    ROLLING_VOL_WINDOW,
    RSI_WINDOW,
    SYMBOLS,
    START_DATE,
    END_DATE,
)
from utils.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Individual feature functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_log_return(df: pd.DataFrame) -> pd.Series:
    """Daily log returns: ln(close_t / close_{t-1})"""
    return np.log(df["close"] / df["close"].shift(1))


def compute_rolling_volatility(log_ret: pd.Series, window: int = ROLLING_VOL_WINDOW) -> pd.Series:
    """Rolling standard deviation of log returns × √252 (annualised)."""
    return log_ret.rolling(window=window, min_periods=max(5, window // 2)).std() * np.sqrt(252)


def compute_momentum(df: pd.DataFrame, window: int = MOMENTUM_WINDOW) -> pd.Series:
    """Momentum: (close_t / close_{t-window}) - 1"""
    return df["close"] / df["close"].shift(window) - 1


def compute_rsi(df: pd.DataFrame, window: int = RSI_WINDOW) -> pd.Series:
    """
    14-period Relative Strength Index using Wilder's smoothing (EWM).
    RSI = 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss
    """
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling Z-score of volume."""
    rolling_mean = df["volume"].rolling(window=window, min_periods=5).mean()
    rolling_std  = df["volume"].rolling(window=window, min_periods=5).std()
    return (df["volume"] - rolling_mean) / rolling_std.replace(0, np.nan)


def compute_rolling_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = BETA_WINDOW,
) -> pd.Series:
    """
    Rolling beta of the instrument vs. benchmark over `window` trading days.
    beta = Cov(r_i, r_m) / Var(r_m)
    """
    if benchmark_returns is None or benchmark_returns.empty:
        return pd.Series(np.nan, index=returns.index)

    aligned = pd.DataFrame({"stock": returns, "bench": benchmark_returns}).dropna()

    betas = []
    for i in range(len(returns)):
        if i < window:
            betas.append(np.nan)
            continue
        window_data = aligned.iloc[max(0, i - window):i]
        if len(window_data) < window // 2:
            betas.append(np.nan)
            continue
        cov = window_data["stock"].cov(window_data["bench"])
        var = window_data["bench"].var()
        betas.append(cov / var if var != 0 else np.nan)

    return pd.Series(betas, index=returns.index)


# ──────────────────────────────────────────────────────────────────────────────
# Main feature engineering function
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, symbol: str, benchmark_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate all features from clean OHLCV data.

    Parameters
    ----------
    df           : pd.DataFrame — preprocessed OHLCV data (sorted by date)
    symbol       : str
    benchmark_df : pd.DataFrame — benchmark OHLCV (e.g. Nifty 50), optional

    Returns
    -------
    pd.DataFrame with original OHLCV + all engineered features
    """
    log.info(f"[FeatureEngineer] Generating features for {symbol} ({len(df)} rows)")
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Core return series
    df["log_return"] = compute_log_return(df)

    # Rolling volatility
    df[f"rolling_volatility_{ROLLING_VOL_WINDOW}"] = compute_rolling_volatility(
        df["log_return"], ROLLING_VOL_WINDOW
    )

    # Momentum
    df[f"momentum_{MOMENTUM_WINDOW}"] = compute_momentum(df, MOMENTUM_WINDOW)

    # RSI
    df[f"rsi_{RSI_WINDOW}"] = compute_rsi(df, RSI_WINDOW)

    # Volume Z-score
    if "volume" in df.columns:
        df["volume_zscore"] = compute_volume_zscore(df)
    else:
        df["volume_zscore"] = np.nan

    # Rolling beta
    bench_returns = None
    if benchmark_df is not None and not benchmark_df.empty:
        bench_ret = np.log(benchmark_df["close"] / benchmark_df["close"].shift(1))
        bench_ret.index = benchmark_df["date"]
        bench_ret = bench_ret.reindex(df["date"].values)
        bench_returns = bench_ret.values

    df["beta_rolling"] = compute_rolling_beta(
        df["log_return"],
        pd.Series(bench_returns, index=df.index) if bench_returns is not None else pd.Series(dtype=float),
        window=BETA_WINDOW,
    )

    feature_cols = [
        "log_return",
        f"rolling_volatility_{ROLLING_VOL_WINDOW}",
        f"momentum_{MOMENTUM_WINDOW}",
        f"rsi_{RSI_WINDOW}",
        "volume_zscore",
        "beta_rolling",
    ]
    non_null = df[feature_cols].notna().any().sum()
    log.info(f"[FeatureEngineer] {symbol}: {non_null}/{len(feature_cols)} feature columns populated")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Load / save helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_preprocessed(symbol: str) -> pd.DataFrame:
    safe_sym = symbol.replace(".", "_")
    path = os.path.join(PROC_DATA_DIR, f"preprocessed_{safe_sym}.csv")
    if not os.path.exists(path):
        log.warning(f"[FeatureEngineer] Preprocessed file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    log.debug(f"[FeatureEngineer] Loaded {path} ({len(df)} rows)")
    return df


def load_benchmark() -> pd.DataFrame:
    """Load benchmark (Nifty 50) preprocessed data if available."""
    safe_bench = BENCHMARK_SYMBOL.replace("^", "").replace(".", "_")
    for pattern in [f"preprocessed_{safe_bench}.csv", f"yahoo_{safe_bench}_*.csv"]:
        import glob
        matches = glob.glob(os.path.join(PROC_DATA_DIR, pattern))
        if matches:
            try:
                df = pd.read_csv(matches[0], parse_dates=["date"])
                log.info(f"[FeatureEngineer] Benchmark loaded: {matches[0]}")
                return df
            except Exception:
                pass
    log.warning("[FeatureEngineer] Benchmark data not found; beta_rolling will be NaN")
    return pd.DataFrame()


def save_features(df: pd.DataFrame, symbol: str) -> str:
    safe_sym = symbol.replace(".", "_")
    filename = f"features_{safe_sym}.csv"
    filepath = os.path.join(PROC_DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    log.info(f"[FeatureEngineer] Saved → {filepath}  ({len(df)} rows)")
    return filepath


def engineer_all(symbols: list = None) -> dict:
    """Run feature engineering for all symbols."""
    if symbols is None:
        symbols = SYMBOLS

    benchmark_df = load_benchmark()
    results = {}

    for symbol in symbols:
        df = load_preprocessed(symbol)
        if df.empty:
            log.warning(f"[FeatureEngineer] Skipping {symbol} — no preprocessed data")
            continue

        features = engineer_features(df, symbol, benchmark_df)
        results[symbol] = features
        save_features(features, symbol)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== FeatureEngineer: starting ===")
    results = engineer_all()
    log.info("=== FeatureEngineer: complete ===")
    for sym, df in results.items():
        print(f"  ✅ {sym}: {len(df)} rows, {df.shape[1]} columns")
