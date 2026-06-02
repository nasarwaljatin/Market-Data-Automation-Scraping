"""
scrapers/yahoo_fetcher.py
--------------------------
Fetches historical OHLCV data from Yahoo Finance using the yfinance library.
Includes adjusted close prices and dividend/split history.

CLI Usage:
    python scrapers/yahoo_fetcher.py --symbol RELIANCE.NS --start 2023-01-01 --end 2024-01-01

Output:
    data/raw/yahoo_<SYMBOL>_<start>_<end>.csv
"""

import argparse
import os
import sys

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import RAW_DATA_DIR, START_DATE, END_DATE, SYMBOLS
from utils.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core fetch
# ──────────────────────────────────────────────────────────────────────────────

def fetch_yahoo_data(
    symbol: str,
    start: str = START_DATE,
    end: str = END_DATE,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Download historical OHLCV + adjusted close from Yahoo Finance.

    Parameters
    ----------
    symbol       : str  — Yahoo ticker (e.g. 'RELIANCE.NS')
    start        : str  — 'YYYY-MM-DD'
    end          : str  — 'YYYY-MM-DD'
    auto_adjust  : bool — if True, all prices are split/dividend-adjusted

    Returns
    -------
    pd.DataFrame with columns: date, open, high, low, close, adj_close, volume, source
    """
    log.info(f"[Yahoo] Fetching {symbol} ({start} → {end})")

    try:
        ticker = yf.Ticker(symbol)
        raw = ticker.history(
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            actions=True,           # include Dividends & Stock Splits columns
        )

        if raw is None or raw.empty:
            log.warning(f"[Yahoo] No data returned for {symbol}")
            return pd.DataFrame()

        df = raw.copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.reset_index()

        # Normalize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Keep standard OHLCV + extras
        rename_map = {
            "open":         "open",
            "high":         "high",
            "low":          "low",
            "close":        "close",
            "volume":       "volume",
            "dividends":    "dividends",
            "stock_splits": "stock_splits",
        }
        df = df.rename(columns=rename_map)

        keep = [c for c in ["date", "open", "high", "low", "close", "volume",
                             "dividends", "stock_splits"] if c in df.columns]
        df = df[keep].copy()

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["source"] = "yahoo"
        df = df.sort_values("date").reset_index(drop=True)

        log.info(f"[Yahoo] ✓ {len(df)} rows for {symbol}")
        return df

    except Exception as exc:
        log.error(f"[Yahoo] Failed for {symbol}: {exc}")
        return pd.DataFrame()


def fetch_yahoo_dividends(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Return dividend history for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        divs = ticker.dividends
        if divs is None or divs.empty:
            return pd.DataFrame()
        divs = divs.reset_index()
        divs.columns = ["date", "dividend"]
        divs["date"] = pd.to_datetime(divs["date"]).dt.tz_localize(None)
        mask = (divs["date"] >= start) & (divs["date"] <= end)
        return divs[mask].copy()
    except Exception as exc:
        log.warning(f"[Yahoo] Dividend fetch failed for {symbol}: {exc}")
        return pd.DataFrame()


def save_yahoo_data(symbol: str, start: str, end: str) -> str:
    """
    Fetch and save Yahoo Finance data to data/raw/.

    Returns
    -------
    str — path to saved CSV, or empty string on failure
    """
    df = fetch_yahoo_data(symbol, start, end)
    if df.empty:
        log.error(f"[Yahoo] No data to save for {symbol}")
        return ""

    safe_symbol = symbol.replace(".", "_")
    filename = f"yahoo_{safe_symbol}_{start}_{end}.csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    log.info(f"[Yahoo] Saved → {filepath}  ({len(df)} rows)")
    return filepath


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yahoo Finance data fetcher")
    parser.add_argument("--symbol", default=SYMBOLS[0],  help="Yahoo ticker (e.g. RELIANCE.NS)")
    parser.add_argument("--start",  default=START_DATE,  help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=END_DATE,     help="End date   YYYY-MM-DD")
    args = parser.parse_args()

    path = save_yahoo_data(args.symbol, args.start, args.end)
    if path:
        print(f"✅ Yahoo data saved: {path}")
    else:
        print("❌ Failed to fetch Yahoo data.")
        sys.exit(1)
