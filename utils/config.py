"""
utils/config.py
---------------
Central configuration for the Market Data Automation pipeline.
Edit SYMBOLS, date ranges, and source settings here.
"""

from dotenv import load_dotenv
import os

load_dotenv()  # Load any .env file if present

# ─────────────────────────────────────────────
# Target instruments
# ─────────────────────────────────────────────
# Yahoo Finance / NSE tickers (suffix .NS for NSE-listed equities)
SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]

# NSE-style symbols (no suffix) used by nsepython and NSE scrapers
NSE_SYMBOLS = [s.replace(".NS", "") for s in SYMBOLS]

# Benchmark index for rolling beta calculation
BENCHMARK_SYMBOL = "^NSEI"          # Nifty 50
BENCHMARK_NSE    = "NIFTY 50"

# ─────────────────────────────────────────────
# Date range
# ─────────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

# ─────────────────────────────────────────────
# Timezone
# ─────────────────────────────────────────────
TIMEZONE = "Asia/Kolkata"   # IST — UTC+5:30

# ─────────────────────────────────────────────
# Source priority (for conflict resolution)
# Higher value = higher trust
# ─────────────────────────────────────────────
SOURCE_PRIORITY = {
    "nse":    3,
    "yahoo":  2,
    "google": 1,
}

# ─────────────────────────────────────────────
# Conflict resolution thresholds
# ─────────────────────────────────────────────
# Relative difference (fraction) above which two sources are considered in conflict
CONFLICT_THRESHOLD = 0.01   # 1 %

# ─────────────────────────────────────────────
# Preprocessing thresholds
# ─────────────────────────────────────────────
ZSCORE_OUTLIER_THRESHOLD = 3.5   # std-devs
IQR_OUTLIER_MULTIPLIER   = 1.5   # IQR fence
MAX_FFILL_DAYS           = 3     # max consecutive forward-fill days

# ─────────────────────────────────────────────
# Feature engineering windows
# ─────────────────────────────────────────────
ROLLING_VOL_WINDOW  = 20    # days
MOMENTUM_WINDOW     = 14    # days
RSI_WINDOW          = 14    # periods
BETA_WINDOW         = 60    # trading days

# ─────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROC_DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
LOG_FILE       = os.path.join(LOG_DIR, "pipeline.log")

# Ensure directories exist at import time
for _dir in (RAW_DATA_DIR, PROC_DATA_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────────
# NSE scraper endpoints
# ─────────────────────────────────────────────
NSE_BASE_URL          = "https://www.nseindia.com"
NSE_HISTORY_URL       = "https://www.nseindia.com/api/historical/cm/equity"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}
