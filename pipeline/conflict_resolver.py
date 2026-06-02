"""
pipeline/conflict_resolver.py
------------------------------
Detects and resolves discrepancies when the same instrument has differing
OHLCV values across NSE, Yahoo Finance, and Google Finance.

Resolution strategy:
  - Weighted priority: NSE (3) > Yahoo (2) > Google (1)
  - If sources agree within CONFLICT_THRESHOLD → take priority-source value
  - If they disagree beyond threshold → flag conflict, apply weighted average
    of available sources, log the conflict

Standalone:
    python pipeline/conflict_resolver.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    CONFLICT_THRESHOLD,
    PROC_DATA_DIR,
    SOURCE_PRIORITY,
    SYMBOLS,
)
from utils.logger import get_logger

log = get_logger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# ──────────────────────────────────────────────────────────────────────────────
# Core resolution logic
# ──────────────────────────────────────────────────────────────────────────────

def merge_sources(source_frames: dict) -> pd.DataFrame:
    """
    Merge DataFrames from multiple sources on the 'date' column.

    Parameters
    ----------
    source_frames : dict — { source_name: pd.DataFrame }
        Each DataFrame must have the unified OHLCV schema.

    Returns
    -------
    pd.DataFrame with all-source columns suffixed by _<source>
    """
    if not source_frames:
        return pd.DataFrame()

    merged = None
    for source, df in source_frames.items():
        if df.empty:
            continue
        sub = df[["date"] + [c for c in OHLCV_COLS if c in df.columns]].copy()
        sub = sub.rename(columns={c: f"{c}_{source}" for c in OHLCV_COLS if c in sub.columns})
        if merged is None:
            merged = sub
        else:
            merged = pd.merge(merged, sub, on="date", how="outer")

    if merged is None:
        return pd.DataFrame()

    return merged.sort_values("date").reset_index(drop=True)


def resolve_conflicts(merged: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Apply priority-based conflict resolution across merged source data.

    For each OHLCV field and each date row:
      1. Collect values from all sources that have data for that row.
      2. Compute the max relative deviation among available values.
      3. If deviation ≤ CONFLICT_THRESHOLD → use highest-priority source value.
      4. If deviation > CONFLICT_THRESHOLD → flag as conflict, use
         priority-weighted average and log.

    Parameters
    ----------
    merged : pd.DataFrame  — output of merge_sources()
    symbol : str           — instrument symbol for logging

    Returns
    -------
    pd.DataFrame with resolved OHLCV columns + 'conflict_flag' boolean column
    """
    result = pd.DataFrame({"date": merged["date"]})
    result["symbol"] = symbol
    result["conflict_flag"] = False

    sources_available = sorted(
        SOURCE_PRIORITY.keys(),
        key=lambda s: SOURCE_PRIORITY[s],
        reverse=True,           # highest priority first
    )

    conflict_count = 0

    for col in OHLCV_COLS:
        col_data = {}
        for src in sources_available:
            col_name = f"{col}_{src}"
            if col_name in merged.columns:
                col_data[src] = merged[col_name]

        if not col_data:
            result[col] = np.nan
            continue

        resolved = []
        flags = []

        for i in range(len(merged)):
            row_vals = {
                src: vals.iloc[i]
                for src, vals in col_data.items()
                if not pd.isna(vals.iloc[i])
            }

            if not row_vals:
                resolved.append(np.nan)
                flags.append(False)
                continue

            if len(row_vals) == 1:
                # Only one source — no conflict possible
                resolved.append(next(iter(row_vals.values())))
                flags.append(False)
                continue

            values = list(row_vals.values())
            ref    = max(abs(v) for v in values) or 1.0
            max_dev = (max(values) - min(values)) / ref

            if max_dev <= CONFLICT_THRESHOLD:
                # Sources agree → use highest-priority source
                best = next(
                    (row_vals[s] for s in sources_available if s in row_vals),
                    np.nan,
                )
                resolved.append(best)
                flags.append(False)
            else:
                # Conflict: weighted average
                total_weight = sum(SOURCE_PRIORITY[s] for s in row_vals)
                weighted_val = sum(
                    row_vals[s] * SOURCE_PRIORITY[s] for s in row_vals
                ) / total_weight
                resolved.append(weighted_val)
                flags.append(True)
                conflict_count += 1
                log.debug(
                    f"[Conflict] {symbol} | {col} | row {i} | "
                    f"vals={row_vals} | deviation={max_dev:.4f} → "
                    f"weighted={weighted_val:.4f}"
                )

        result[col] = resolved
        result["conflict_flag"] = result["conflict_flag"] | pd.Series(flags)

    if conflict_count:
        log.warning(
            f"[ConflictResolver] {symbol}: {conflict_count} conflict cells detected "
            f"(threshold={CONFLICT_THRESHOLD:.2%})"
        )
    else:
        log.info(f"[ConflictResolver] {symbol}: no conflicts detected")

    return result.reset_index(drop=True)


def resolve_symbol(source_frames: dict, symbol: str) -> pd.DataFrame:
    """
    Full resolution for one symbol: merge → resolve conflicts.

    Parameters
    ----------
    source_frames : dict — { source_name: pd.DataFrame }
    symbol        : str

    Returns
    -------
    pd.DataFrame — resolved OHLCV with conflict_flag column
    """
    merged = merge_sources(source_frames)
    if merged.empty:
        log.warning(f"[ConflictResolver] No data to resolve for {symbol}")
        return pd.DataFrame()
    return resolve_conflicts(merged, symbol)


# ──────────────────────────────────────────────────────────────────────────────
# Load standardized data from disk
# ──────────────────────────────────────────────────────────────────────────────

def load_standardized(symbol: str) -> dict:
    """
    Load standardized CSV files for a symbol from data/processed/.

    Returns
    -------
    dict — { source: pd.DataFrame }
    """
    safe_sym = symbol.replace(".", "_")
    frames = {}
    for source in SOURCE_PRIORITY.keys():
        path = os.path.join(PROC_DATA_DIR, f"standardized_{safe_sym}_{source}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=["date"])
                frames[source] = df
                log.debug(f"[ConflictResolver] Loaded {path} ({len(df)} rows)")
            except Exception as exc:
                log.error(f"[ConflictResolver] Failed to load {path}: {exc}")
    return frames


def save_resolved(df: pd.DataFrame, symbol: str) -> str:
    """Save resolved DataFrame to data/processed/resolved_<symbol>.csv"""
    safe_sym = symbol.replace(".", "_")
    filename = f"resolved_{safe_sym}.csv"
    filepath = os.path.join(PROC_DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    log.info(f"[ConflictResolver] Saved → {filepath}  ({len(df)} rows)")
    return filepath


def resolve_all(symbols: list = None) -> dict:
    """
    Run full conflict resolution for all symbols.

    Returns
    -------
    dict — { symbol: pd.DataFrame (resolved) }
    """
    if symbols is None:
        symbols = SYMBOLS

    resolved = {}
    for symbol in symbols:
        frames = load_standardized(symbol)
        df = resolve_symbol(frames, symbol)
        if not df.empty:
            resolved[symbol] = df
            save_resolved(df, symbol)

    return resolved


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== ConflictResolver: starting ===")
    resolved = resolve_all()
    log.info("=== ConflictResolver: complete ===")
    for sym, df in resolved.items():
        conflicts = df["conflict_flag"].sum() if "conflict_flag" in df.columns else 0
        print(f"  ✅ {sym}: {len(df)} rows, {conflicts} conflict cells resolved")
