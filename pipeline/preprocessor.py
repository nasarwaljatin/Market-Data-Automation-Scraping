"""
pipeline/preprocessor.py
------------------------
Cleans and preprocesses resolved OHLCV data:

  1. Missing value handling — forward-fill (up to MAX_FFILL_DAYS); flag large gaps
  2. Outlier detection      — Z-score + IQR dual method with configurable thresholds
  3. Corporate action adj.  — split & dividend adjustments applied to historical prices

Standalone:
    python pipeline/preprocessor.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    IQR_OUTLIER_MULTIPLIER,
    MAX_FFILL_DAYS,
    PROC_DATA_DIR,
    SYMBOLS,
    ZSCORE_OUTLIER_THRESHOLD,
)
from utils.logger import get_logger

log = get_logger(__name__)

PRICE_COLS = ["open", "high", "low", "close"]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Missing value handling
# ──────────────────────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Forward-fill gaps up to MAX_FFILL_DAYS. Flag and log larger gaps.

    Parameters
    ----------
    df     : pd.DataFrame — must have 'date' and OHLCV columns
    symbol : str

    Returns
    -------
    pd.DataFrame with missing values handled and 'missing_flag' column added
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["missing_flag"] = False

    for col in PRICE_COLS + ["volume"]:
        if col not in df.columns:
            continue

        null_mask = df[col].isna()
        n_missing = null_mask.sum()

        if n_missing == 0:
            continue

        # Identify run lengths of consecutive NaNs
        runs = (null_mask != null_mask.shift()).cumsum()
        run_lengths = null_mask.groupby(runs).transform("sum")

        large_gaps = null_mask & (run_lengths > MAX_FFILL_DAYS)
        if large_gaps.any():
            log.warning(
                f"[Preprocessor] {symbol}/{col}: {large_gaps.sum()} values in gaps "
                f"> {MAX_FFILL_DAYS} days — will remain NaN after ffill"
            )
            df.loc[large_gaps, "missing_flag"] = True

        # Forward-fill within the limit
        df[col] = df[col].fillna(method="ffill", limit=MAX_FFILL_DAYS)
        remaining = df[col].isna().sum()
        log.debug(
            f"[Preprocessor] {symbol}/{col}: {n_missing} missing → "
            f"{remaining} remaining after ffill(limit={MAX_FFILL_DAYS})"
        )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Outlier detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_outliers(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Flag outliers in OHLCV columns using both Z-score and IQR methods.
    A value is flagged if it triggers either method.

    Adds an 'outlier_flag' boolean column.

    Parameters
    ----------
    df     : pd.DataFrame
    symbol : str

    Returns
    -------
    pd.DataFrame with 'outlier_flag' column
    """
    df = df.copy()
    df["outlier_flag"] = False

    for col in PRICE_COLS:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) < 10:
            continue

        # ── Z-score method ──
        z_scores  = (df[col] - series.mean()) / series.std()
        z_outlier = z_scores.abs() > ZSCORE_OUTLIER_THRESHOLD

        # ── IQR method ──
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr     = q3 - q1
        lower   = q1 - IQR_OUTLIER_MULTIPLIER * iqr
        upper   = q3 + IQR_OUTLIER_MULTIPLIER * iqr
        iqr_outlier = (df[col] < lower) | (df[col] > upper)

        combined = z_outlier | iqr_outlier
        df.loc[combined.fillna(False), "outlier_flag"] = True

        n_out = combined.sum()
        if n_out:
            log.warning(
                f"[Preprocessor] {symbol}/{col}: {n_out} outlier(s) flagged "
                f"(Z>{ZSCORE_OUTLIER_THRESHOLD} or IQR×{IQR_OUTLIER_MULTIPLIER})"
            )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Corporate action adjustments
# ──────────────────────────────────────────────────────────────────────────────

def apply_corporate_actions(
    df: pd.DataFrame,
    dividends: pd.DataFrame = None,
    splits: pd.DataFrame = None,
    symbol: str = "",
) -> pd.DataFrame:
    """
    Adjust historical prices for stock splits and cash dividends.

    Adjustment approach (backward adjustment — standard finance convention):
      - Split   : multiply all prices BEFORE the split date by (1 / split_ratio)
      - Dividend: multiply all prices BEFORE the ex-date by
                  (1 - dividend / close_on_ex_date)

    Parameters
    ----------
    df        : pd.DataFrame — OHLCV DataFrame sorted by date
    dividends : pd.DataFrame — columns: [date, dividend]  (optional)
    splits    : pd.DataFrame — columns: [date, ratio]     (optional, ratio > 1 = split)
    symbol    : str          — for logging

    Returns
    -------
    pd.DataFrame with adjusted OHLCV prices
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Split adjustments ─────────────────────────────────────────
    if splits is not None and not splits.empty:
        splits = splits.copy()
        splits["date"] = pd.to_datetime(splits["date"])
        for _, row in splits.iterrows():
            split_date = row["date"]
            ratio      = float(row.get("ratio", 1.0))
            if ratio <= 0 or ratio == 1.0:
                continue
            mask = df["date"] < split_date
            for col in PRICE_COLS:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] / ratio
            if "volume" in df.columns:
                df.loc[mask, "volume"] = df.loc[mask, "volume"] * ratio
            log.info(f"[Preprocessor] {symbol}: applied split ratio={ratio:.4f} on {split_date.date()}")

    # ── Dividend adjustments ──────────────────────────────────────
    if dividends is not None and not dividends.empty:
        dividends = dividends.copy()
        dividends["date"] = pd.to_datetime(dividends["date"])
        for _, row in dividends.iterrows():
            ex_date  = row["date"]
            dividend = float(row.get("dividend", 0.0))
            if dividend <= 0:
                continue

            ex_close_row = df[df["date"] == ex_date]
            if ex_close_row.empty:
                # Use closest prior close
                prior = df[df["date"] < ex_date]
                if prior.empty:
                    continue
                ex_close = prior.iloc[-1]["close"]
            else:
                ex_close = ex_close_row.iloc[0]["close"]

            if ex_close <= 0:
                continue

            adj_factor = 1 - dividend / ex_close
            mask = df["date"] < ex_date
            for col in PRICE_COLS:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] * adj_factor
            log.info(
                f"[Preprocessor] {symbol}: dividend={dividend:.4f} on {ex_date.date()}, "
                f"adj_factor={adj_factor:.6f}"
            )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Full preprocessing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    symbol: str,
    dividends: pd.DataFrame = None,
    splits: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Run all preprocessing steps in order:
      missing values → outlier detection → corporate action adjustment

    Parameters
    ----------
    df        : pd.DataFrame
    symbol    : str
    dividends : pd.DataFrame (optional)
    splits    : pd.DataFrame (optional)

    Returns
    -------
    pd.DataFrame — clean, adjusted OHLCV data
    """
    log.info(f"[Preprocessor] Starting for {symbol} ({len(df)} rows)")
    df = handle_missing_values(df, symbol)
    df = detect_outliers(df, symbol)
    df = apply_corporate_actions(df, dividends, splits, symbol)
    log.info(f"[Preprocessor] Complete for {symbol} ({len(df)} rows)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Load / save helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_resolved(symbol: str) -> pd.DataFrame:
    safe_sym = symbol.replace(".", "_")
    path = os.path.join(PROC_DATA_DIR, f"resolved_{safe_sym}.csv")
    if not os.path.exists(path):
        log.warning(f"[Preprocessor] Resolved file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    log.debug(f"[Preprocessor] Loaded {path} ({len(df)} rows)")
    return df


def save_preprocessed(df: pd.DataFrame, symbol: str) -> str:
    safe_sym = symbol.replace(".", "_")
    filename = f"preprocessed_{safe_sym}.csv"
    filepath = os.path.join(PROC_DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    log.info(f"[Preprocessor] Saved → {filepath}  ({len(df)} rows)")
    return filepath


def preprocess_all(symbols: list = None) -> dict:
    """Run preprocessing for all symbols."""
    if symbols is None:
        symbols = SYMBOLS

    results = {}
    for symbol in symbols:
        df = load_resolved(symbol)
        if df.empty:
            log.warning(f"[Preprocessor] Skipping {symbol} — no resolved data")
            continue

        # Load dividends/splits from Yahoo raw data if available
        dividends = _load_corporate_actions(symbol, "dividends")
        splits    = _load_corporate_actions(symbol, "splits")

        cleaned = preprocess(df, symbol, dividends=dividends, splits=splits)
        results[symbol] = cleaned
        save_preprocessed(cleaned, symbol)

    return results


def _load_corporate_actions(symbol: str, action_type: str) -> pd.DataFrame:
    """Load dividend or split data extracted during Yahoo fetch (if available)."""
    safe_sym = symbol.replace(".", "_")
    path = os.path.join(PROC_DATA_DIR, f"yahoo_{safe_sym}_{action_type}.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path, parse_dates=["date"])
        except Exception:
            pass
    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== Preprocessor: starting ===")
    results = preprocess_all()
    log.info("=== Preprocessor: complete ===")
    for sym, df in results.items():
        outliers = df["outlier_flag"].sum() if "outlier_flag" in df.columns else 0
        missing  = df["missing_flag"].sum()  if "missing_flag"  in df.columns else 0
        print(f"  ✅ {sym}: {len(df)} rows | outliers={outliers} | missing_flags={missing}")
