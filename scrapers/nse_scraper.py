"""
scrapers/nse_scraper.py
-----------------------
Fetches historical OHLCV data from NSE using:
  1. nsepython library (primary)
  2. NSE JSON API via requests (fallback)

CLI Usage:
    python scrapers/nse_scraper.py --symbol RELIANCE --start 2023-01-01 --end 2024-01-01

Output:
    data/raw/nse_<SYMBOL>_<start>_<end>.csv
"""

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests

# Allow running as standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    NSE_HEADERS,
    NSE_HISTORY_URL,
    NSE_SYMBOLS,
    RAW_DATA_DIR,
    START_DATE,
    END_DATE,
)
from utils.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Primary: nsepython
# ──────────────────────────────────────────────────────────────────────────────

def fetch_via_nsepython(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical equity data using the nsepython library.

    Parameters
    ----------
    symbol : str  — NSE symbol without suffix, e.g. 'RELIANCE'
    start  : str  — Start date 'YYYY-MM-DD'
    end    : str  — End date   'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame with columns: date, open, high, low, close, volume, source
    """
    try:
        from nsepython import equity_history
        log.info(f"[NSE] Fetching {symbol} via nsepython ({start} → {end})")

        # nsepython date format: DD-MM-YYYY
        s = datetime.strptime(start, "%Y-%m-%d").strftime("%d-%m-%Y")
        e = datetime.strptime(end,   "%Y-%m-%d").strftime("%d-%m-%Y")

        raw = equity_history(symbol, "EQ", s, e)

        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            log.warning(f"[NSE] nsepython returned empty data for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        df = _normalize_nse_columns(df)
        df["source"] = "nse"
        log.info(f"[NSE] nsepython → {len(df)} rows for {symbol}")
        return df

    except Exception as exc:
        log.warning(f"[NSE] nsepython failed for {symbol}: {exc}")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Fallback: NSE JSON API (chunked to 1-year windows)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_via_nse_api(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical data directly from the NSE API endpoint.
    NSE limits queries to ~1 year; this function chunks automatically.

    Returns
    -------
    pd.DataFrame with unified OHLCV columns + source='nse'
    """
    log.info(f"[NSE] Fetching {symbol} via NSE API ({start} → {end})")
    session = _create_nse_session()
    all_frames = []

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    # Chunk into 365-day windows
    chunk_start = start_dt
    while chunk_start < end_dt:
        from datetime import timedelta
        chunk_end = min(chunk_start + timedelta(days=364), end_dt)

        params = {
            "symbol":    symbol,
            "series":    "EQ",
            "from":      chunk_start.strftime("%d-%m-%Y"),
            "to":        chunk_end.strftime("%d-%m-%Y"),
            "csv":       "true",
        }
        try:
            resp = session.get(NSE_HISTORY_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            records = data.get("data", [])
            if records:
                df = pd.DataFrame(records)
                df = _normalize_nse_columns(df)
                df["source"] = "nse"
                all_frames.append(df)
                log.debug(f"[NSE] API chunk {chunk_start.date()}–{chunk_end.date()}: {len(df)} rows")
        except Exception as exc:
            log.error(f"[NSE] API chunk failed ({chunk_start.date()}–{chunk_end.date()}): {exc}")

        chunk_start = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.5)   # polite delay

    if not all_frames:
        log.warning(f"[NSE] API returned no data for {symbol}")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["date"])
    log.info(f"[NSE] API → {len(result)} rows for {symbol}")
    return result


def _create_nse_session() -> requests.Session:
    """Create a requests session with NSE cookies (required for API access)."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Hit the main page to acquire cookies
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
    except Exception as exc:
        log.warning(f"[NSE] Cookie init failed: {exc}")
    return session


def _normalize_nse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw NSE columns to the unified OHLCV schema."""
    col_map = {
        # nsepython / API variants
        "CH_TIMESTAMP":       "date",
        "CH_OPENING_PRICE":   "open",
        "CH_TRADE_HIGH_PRICE":"high",
        "CH_TRADE_LOW_PRICE": "low",
        "CH_CLOSING_PRICE":   "close",
        "CH_TOT_TRADED_QTY":  "volume",
        # lowercase variants
        "date":   "date",
        "open":   "open",
        "high":   "high",
        "low":    "low",
        "close":  "close",
        "volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────

def fetch_nse_data(symbol: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Fetch NSE data for a symbol. Tries nsepython first, falls back to NSE API.

    Parameters
    ----------
    symbol : str  — NSE symbol (e.g. 'RELIANCE')
    start  : str  — 'YYYY-MM-DD'
    end    : str  — 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame
    """
    df = fetch_via_nsepython(symbol, start, end)
    if df.empty:
        df = fetch_via_nse_api(symbol, start, end)
    return df


def save_nse_data(symbol: str, start: str, end: str) -> str:
    """
    Fetch and save NSE data to data/raw/.

    Returns
    -------
    str — path to saved CSV, or empty string on failure
    """
    df = fetch_nse_data(symbol, start, end)
    if df.empty:
        log.error(f"[NSE] No data to save for {symbol}")
        return ""

    filename = f"nse_{symbol}_{start}_{end}.csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    log.info(f"[NSE] Saved → {filepath}  ({len(df)} rows)")
    return filepath


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE historical data fetcher")
    parser.add_argument("--symbol", default=NSE_SYMBOLS[0], help="NSE symbol (e.g. RELIANCE)")
    parser.add_argument("--start",  default=START_DATE,      help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=END_DATE,         help="End date   YYYY-MM-DD")
    args = parser.parse_args()

    path = save_nse_data(args.symbol, args.start, args.end)
    if path:
        print(f"✅ NSE data saved: {path}")
    else:
        print("❌ Failed to fetch NSE data.")
        sys.exit(1)
