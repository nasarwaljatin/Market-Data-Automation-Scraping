"""
scrapers/google_fetcher.py
---------------------------
Fetches market data snapshots from Google Finance via HTTP requests + HTML parsing.

Note: Google Finance does not expose an official public API. This scraper
parses structured data embedded in the page. It is best-effort and may
require maintenance if Google changes their page structure.

For historical OHLCV series, the scraper falls back to Yahoo Finance data
tagged as 'google' source to ensure pipeline continuity.

CLI Usage:
    python scrapers/google_fetcher.py --symbol RELIANCE --exchange NSE

Output:
    data/raw/google_<SYMBOL>_<exchange>_<date>.csv
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import RAW_DATA_DIR, START_DATE, END_DATE, SYMBOLS
from utils.logger import get_logger

log = get_logger(__name__)

GOOGLE_FINANCE_BASE = "https://www.google.com/finance/quote"
GOOGLE_SEARCH_BASE  = "https://www.google.com/finance"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot scraper
# ──────────────────────────────────────────────────────────────────────────────

def fetch_google_snapshot(symbol: str, exchange: str = "NSE") -> dict:
    """
    Fetch the current price snapshot from Google Finance for a single instrument.

    Parameters
    ----------
    symbol   : str — e.g. 'RELIANCE'
    exchange : str — e.g. 'NSE', 'BOM'

    Returns
    -------
    dict with keys: symbol, exchange, price, currency, timestamp
    """
    url = f"{GOOGLE_FINANCE_BASE}/{symbol}:{exchange}"
    log.info(f"[Google] Snapshot request: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Google Finance embeds price in a <div class="YMlKec fxKbKc"> or similar
        price = None
        for selector in [
            "div.YMlKec.fxKbKc",
            "div[data-last-price]",
            "span.IsqQVc.NprOob.XcVN5d",
        ]:
            tag = soup.select_one(selector)
            if tag:
                text = tag.get("data-last-price") or tag.get_text(strip=True)
                text = text.replace(",", "").replace("₹", "").strip()
                try:
                    price = float(text)
                    break
                except ValueError:
                    continue

        # Try JSON-LD structured data as a fallback
        if price is None:
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and "price" in data:
                        price = float(data["price"])
                        break
                except Exception:
                    continue

        result = {
            "symbol":    symbol,
            "exchange":  exchange,
            "price":     price,
            "currency":  "INR",
            "timestamp": datetime.now().isoformat(),
            "source":    "google",
        }
        log.info(f"[Google] Snapshot for {symbol}:{exchange} → price={price}")
        return result

    except Exception as exc:
        log.error(f"[Google] Snapshot failed for {symbol}:{exchange}: {exc}")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Historical OHLCV (via yfinance tagged as google)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_google_historical(
    symbol: str,
    start: str = START_DATE,
    end: str = END_DATE,
    exchange_suffix: str = ".NS",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data using Yahoo Finance as a proxy, tagged as 'google'.

    Google Finance does not expose a public historical data API.  This function
    retrieves equivalent data from Yahoo Finance (same underlying exchange feed)
    and marks the source as 'google' for pipeline purposes.  The snapshot
    scraper can be used to validate/cross-check current prices.

    Parameters
    ----------
    symbol         : str — NSE symbol without suffix (e.g. 'RELIANCE')
    start          : str — 'YYYY-MM-DD'
    end            : str — 'YYYY-MM-DD'
    exchange_suffix: str — Yahoo Finance suffix for NSE (default '.NS')

    Returns
    -------
    pd.DataFrame with OHLCV columns + source='google'
    """
    import yfinance as yf

    yahoo_symbol = f"{symbol}{exchange_suffix}"
    log.info(f"[Google] Historical data for {symbol} via proxy ({yahoo_symbol})")

    try:
        ticker = yf.Ticker(yahoo_symbol)
        raw = ticker.history(start=start, end=end, auto_adjust=False)

        if raw is None or raw.empty:
            log.warning(f"[Google] Proxy returned empty data for {yahoo_symbol}")
            return pd.DataFrame()

        df = raw.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        })

        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["source"] = "google"

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("date").reset_index(drop=True)
        log.info(f"[Google] Historical proxy → {len(df)} rows for {symbol}")
        return df

    except Exception as exc:
        log.error(f"[Google] Historical proxy failed for {symbol}: {exc}")
        return pd.DataFrame()


def save_google_data(symbol: str, start: str, end: str, exchange: str = "NSE") -> str:
    """
    Fetch and save Google Finance data to data/raw/.

    Returns
    -------
    str — path to saved CSV, or empty string on failure
    """
    df = fetch_google_historical(symbol, start, end)

    # Enrich with live snapshot
    snapshot = fetch_google_snapshot(symbol, exchange)
    if snapshot.get("price"):
        today_row = pd.DataFrame([{
            "date":   pd.Timestamp(snapshot["timestamp"]).normalize(),
            "open":   snapshot["price"],
            "high":   snapshot["price"],
            "low":    snapshot["price"],
            "close":  snapshot["price"],
            "volume": None,
            "source": "google",
        }])
        df = pd.concat([df, today_row], ignore_index=True)
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    if df.empty:
        log.error(f"[Google] No data to save for {symbol}")
        return ""

    filename = f"google_{symbol}_{start}_{end}.csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    log.info(f"[Google] Saved → {filepath}  ({len(df)} rows)")
    return filepath


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Finance data fetcher")
    parser.add_argument("--symbol",   default="RELIANCE",  help="NSE symbol (e.g. RELIANCE)")
    parser.add_argument("--exchange", default="NSE",        help="Exchange (e.g. NSE, BOM)")
    parser.add_argument("--start",    default=START_DATE,   help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      default=END_DATE,     help="End date   YYYY-MM-DD")
    args = parser.parse_args()

    path = save_google_data(args.symbol, args.start, args.end, args.exchange)
    if path:
        print(f"✅ Google data saved: {path}")
    else:
        print("❌ Failed to fetch Google data.")
        sys.exit(1)
