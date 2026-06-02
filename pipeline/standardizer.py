"""
pipeline/standardizer.py
-------------------------
Reads raw CSV files from data/raw/, standardizes schemas, and normalizes
all timestamps to IST (UTC+5:30).

Unified output schema:
    date | open | high | low | close | volume | source | symbol

Can be run standalone:
    python pipeline/standardizer.py
"""

import os
import sys
import glob

import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import RAW_DATA_DIR, PROC_DATA_DIR, TIMEZONE, SYMBOLS, NSE_SYMBOLS
from utils.logger import get_logger

log = get_logger(__name__)

IST = pytz.timezone(TIMEZONE)

# Unified column names in the desired order
UNIFIED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "source", "symbol"]

# ──────────────────────────────────────────────────────────────────────────────
# Schema alignment helpers
# ──────────────────────────────────────────────────────────────────────────────

_COLUMN_ALIASES = {
    # Generic
    "date": "date", "datetime": "date", "timestamp": "date", "time": "date",
    "open": "open",   "open_price": "open",
    "high": "high",   "high_price": "high",
    "low":  "low",    "low_price":  "low",
    "close": "close", "close_price": "close", "last": "close", "ltp": "close",
    "volume": "volume", "vol": "volume", "qty": "volume",
    "source": "source",
}


def align_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw columns to the unified OHLCV schema.

    Parameters
    ----------
    df : pd.DataFrame — raw data from any source

    Returns
    -------
    pd.DataFrame with standardized column names
    """
    rename = {}
    for col in df.columns:
        mapped = _COLUMN_ALIASES.get(col.lower().replace(" ", "_"))
        if mapped:
            rename[col] = mapped

    df = df.rename(columns=rename)

    # Cast numerics
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalize_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'date' column to IST (UTC+5:30) timezone-aware Timestamps,
    then strip timezone info for storage (naive IST).

    Parameters
    ----------
    df : pd.DataFrame — must contain a 'date' column

    Returns
    -------
    pd.DataFrame with 'date' column in IST (tz-naive, stored as IST wall-clock time)
    """
    if "date" not in df.columns:
        log.warning("normalize_timezone: no 'date' column found, skipping")
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)

    # Localize naive timestamps to IST directly
    def _to_ist(ts):
        if pd.isna(ts):
            return ts
        if ts.tzinfo is None:
            return IST.localize(ts)
        return ts.astimezone(IST)

    df["date"] = df["date"].apply(_to_ist)
    # Strip tz for storage (keeps wall-clock IST time)
    df["date"] = df["date"].apply(lambda ts: ts.replace(tzinfo=None) if not pd.isna(ts) else ts)
    return df


def standardize_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Full standardization pipeline for a single raw DataFrame.

    1. Align column schema
    2. Normalize timezone → IST
    3. Add symbol column
    4. Drop rows missing essential fields
    5. Sort by date
    6. Return unified DataFrame

    Parameters
    ----------
    df     : pd.DataFrame — raw source data
    symbol : str          — instrument symbol (e.g. 'RELIANCE.NS')

    Returns
    -------
    pd.DataFrame with UNIFIED_COLUMNS schema
    """
    df = align_schema(df)
    df = normalize_timezone(df)

    df["symbol"] = symbol

    # Drop rows missing date or close price
    df = df.dropna(subset=["date", "close"])

    # Ensure all unified columns exist
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[UNIFIED_COLUMNS].sort_values("date").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# File-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_files(source: str, symbol: str) -> pd.DataFrame:
    """
    Load all raw CSV files matching a source and symbol from data/raw/.

    Parameters
    ----------
    source : str — 'nse', 'yahoo', or 'google'
    symbol : str — symbol string (dots replaced with underscores in filenames)

    Returns
    -------
    pd.DataFrame (concatenated, unsorted)
    """
    safe_symbol = symbol.replace(".", "_")
    nse_symbol  = symbol.replace(".NS", "")

    patterns = [
        os.path.join(RAW_DATA_DIR, f"{source}_{safe_symbol}_*.csv"),
        os.path.join(RAW_DATA_DIR, f"{source}_{nse_symbol}_*.csv"),
    ]

    frames = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                df = pd.read_csv(path)
                df["_file"] = os.path.basename(path)
                frames.append(df)
                log.debug(f"Loaded raw file: {path}  ({len(df)} rows)")
            except Exception as exc:
                log.error(f"Failed to read {path}: {exc}")

    if not frames:
        log.warning(f"No raw files found for source={source}, symbol={symbol}")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).drop(columns=["_file"])


def standardize_all(symbols: list = None) -> dict:
    """
    Standardize all raw data files for the given symbols.

    Parameters
    ----------
    symbols : list — list of symbols (default: from config.SYMBOLS)

    Returns
    -------
    dict — { symbol: { source: pd.DataFrame } }
    """
    if symbols is None:
        symbols = SYMBOLS

    result = {}
    for symbol in symbols:
        result[symbol] = {}
        nse_sym = symbol.replace(".NS", "")

        for source, sym_key in [("nse", nse_sym), ("yahoo", symbol), ("google", nse_sym)]:
            raw = load_raw_files(source, sym_key)
            if raw.empty:
                log.warning(f"[Standardizer] No raw data: source={source}, symbol={symbol}")
                continue

            std = standardize_dataframe(raw, symbol)
            result[symbol][source] = std
            log.info(f"[Standardizer] {symbol} / {source}: {len(std)} rows standardized")

    return result


def save_standardized(std_data: dict) -> dict:
    """
    Save standardized DataFrames to data/processed/standardized_<symbol>_<source>.csv

    Returns
    -------
    dict — { symbol: { source: filepath } }
    """
    paths = {}
    for symbol, sources in std_data.items():
        paths[symbol] = {}
        safe_sym = symbol.replace(".", "_")
        for source, df in sources.items():
            filename = f"standardized_{safe_sym}_{source}.csv"
            filepath = os.path.join(PROC_DATA_DIR, filename)
            df.to_csv(filepath, index=False)
            paths[symbol][source] = filepath
            log.info(f"[Standardizer] Saved → {filepath}")

    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== Standardizer: starting ===")
    std_data = standardize_all()
    paths = save_standardized(std_data)
    log.info("=== Standardizer: complete ===")

    for symbol, srcs in paths.items():
        for source, path in srcs.items():
            print(f"  ✅ {symbol}/{source} → {path}")
