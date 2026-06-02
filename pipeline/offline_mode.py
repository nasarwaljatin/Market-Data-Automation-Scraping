"""
pipeline/offline_mode.py
-------------------------
Offline / demo mode for the Market Data Automation pipeline.

When internet is unavailable, this module generates realistic synthetic
OHLCV data using a Geometric Brownian Motion (GBM) model so every
pipeline stage (standardizer → conflict resolver → preprocessor →
feature engineer → risk model) can still be exercised end-to-end.

Usage (automatic — called by run_pipeline.py when --offline flag is set):
    python pipeline/run_pipeline.py --offline

Or as a standalone generator:
    python pipeline/offline_mode.py
    python pipeline/offline_mode.py --symbols RELIANCE.NS TCS.NS --start 2022-01-01 --end 2024-01-01
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    RAW_DATA_DIR,
    SYMBOLS,
    NSE_SYMBOLS,
    START_DATE,
    END_DATE,
    TIMEZONE,
)
from utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Realistic starting prices and volatility for common NSE symbols
# ─────────────────────────────────────────────────────────────────────────────
SYMBOL_PARAMS = {
    "RELIANCE":    {"S0": 2400.0,  "mu": 0.15, "sigma": 0.22},
    "TCS":         {"S0": 3500.0,  "mu": 0.12, "sigma": 0.18},
    "INFY":        {"S0": 1500.0,  "mu": 0.13, "sigma": 0.20},
    "HDFCBANK":    {"S0": 1600.0,  "mu": 0.11, "sigma": 0.19},
    "ICICIBANK":   {"S0": 900.0,   "mu": 0.14, "sigma": 0.21},
    "DEFAULT":     {"S0": 1000.0,  "mu": 0.12, "sigma": 0.20},
}

NSE_HOLIDAYS = set()   # Can be populated with actual NSE holiday dates


def _trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Return business days between start and end (Mon–Fri), excluding known holidays."""
    all_bdays = pd.bdate_range(start=start, end=end, freq="B")
    return all_bdays[~all_bdays.isin(NSE_HOLIDAYS)]


def _gbm_prices(
    S0: float,
    mu: float,
    sigma: float,
    n: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate n daily close prices using Geometric Brownian Motion.

    dS = S * (mu*dt + sigma*dW)

    Parameters
    ----------
    S0    : float — initial price
    mu    : float — annualised drift
    sigma : float — annualised volatility
    n     : int   — number of trading days
    seed  : int   — random seed for reproducibility

    Returns
    -------
    np.ndarray of shape (n,) — simulated close prices
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0 / 252.0
    daily_returns = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n))
    prices = S0 * np.cumprod(daily_returns)
    return prices


def generate_synthetic_ohlcv(
    symbol: str,
    start: str = START_DATE,
    end: str = END_DATE,
    source: str = "nse",
    seed_offset: int = 0,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic OHLCV DataFrame for a symbol.

    Parameters
    ----------
    symbol      : str — NSE symbol (no .NS suffix)
    start       : str — 'YYYY-MM-DD'
    end         : str — 'YYYY-MM-DD'
    source      : str — label for the 'source' column
    seed_offset : int — shifts the random seed so each source differs slightly

    Returns
    -------
    pd.DataFrame with columns: date, open, high, low, close, volume, source
    """
    nse_sym = symbol.replace(".NS", "")
    params  = SYMBOL_PARAMS.get(nse_sym, SYMBOL_PARAMS["DEFAULT"])

    trading_days = _trading_days(start, end)
    n = len(trading_days)

    if n == 0:
        log.warning(f"[Offline] No trading days for {symbol} in {start}→{end}")
        return pd.DataFrame()

    # ── Simulate close prices ─────────────────────────────────────
    seed = abs(hash(nse_sym)) % (2 ** 31) + seed_offset
    close = _gbm_prices(params["S0"], params["mu"], params["sigma"], n, seed)

    # ── Derive OHLV from close ────────────────────────────────────
    rng    = np.random.default_rng(seed + 1)
    spread = params["sigma"] / np.sqrt(252) * close   # daily 1σ in price

    # Realistic intraday spread
    high   = close + rng.uniform(0.0, 1.0, n) * spread * 1.5
    low    = close - rng.uniform(0.0, 1.0, n) * spread * 1.5
    open_  = close * np.exp(rng.normal(0, params["sigma"] / np.sqrt(252 * 4), n))

    # Clip to valid OHLCV constraints
    high   = np.maximum(high,  np.maximum(close, open_))
    low    = np.minimum(low,   np.minimum(close, open_))

    # Simulate volume: log-normal with trend
    base_vol = 5_000_000 if params["S0"] < 1000 else 2_000_000
    volume   = rng.lognormal(mean=np.log(base_vol), sigma=0.5, size=n).astype(int)

    df = pd.DataFrame({
        "date":   trading_days,
        "open":   open_.round(2),
        "high":   high.round(2),
        "low":    low.round(2),
        "close":  close.round(2),
        "volume": volume,
        "source": source,
    })

    log.info(f"[Offline] Generated {len(df)} synthetic rows for {symbol}/{source}")
    return df


def generate_all_sources(
    symbol: str,
    start: str = START_DATE,
    end: str = END_DATE,
) -> dict:
    """
    Generate synthetic data for all three sources with slight price variations
    to simulate real cross-source discrepancies.

    Returns
    -------
    dict — { 'nse': df, 'yahoo': df, 'google': df }
    """
    nse_sym = symbol.replace(".NS", "")
    return {
        "nse":    generate_synthetic_ohlcv(nse_sym, start, end, source="nse",    seed_offset=0),
        "yahoo":  generate_synthetic_ohlcv(nse_sym, start, end, source="yahoo",  seed_offset=100),
        "google": generate_synthetic_ohlcv(nse_sym, start, end, source="google", seed_offset=200),
    }


def save_offline_data(
    symbols: list = None,
    start: str = START_DATE,
    end: str = END_DATE,
) -> dict:
    """
    Generate and save offline synthetic data for all symbols to data/raw/.

    Returns
    -------
    dict — { symbol: { source: filepath } }
    """
    if symbols is None:
        symbols = SYMBOLS

    saved = {}
    for symbol in symbols:
        nse_sym  = symbol.replace(".NS", "")
        saved[symbol] = {}
        sources  = generate_all_sources(symbol, start, end)

        for source, df in sources.items():
            if df.empty:
                continue

            if source == "nse":
                filename = f"nse_{nse_sym}_{start}_{end}.csv"
            elif source == "yahoo":
                safe = symbol.replace(".", "_")
                filename = f"yahoo_{safe}_{start}_{end}.csv"
            else:
                filename = f"google_{nse_sym}_{start}_{end}.csv"

            filepath = os.path.join(RAW_DATA_DIR, filename)
            df.to_csv(filepath, index=False)
            saved[symbol][source] = filepath
            log.info(f"[Offline] Saved → {filepath}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic offline market data")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols (Yahoo format)")
    parser.add_argument("--start",   default=START_DATE,  help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=END_DATE,    help="End date   YYYY-MM-DD")
    args = parser.parse_args()

    log.info("=== Offline Data Generator: starting ===")
    saved = save_offline_data(args.symbols, args.start, args.end)
    log.info("=== Offline Data Generator: complete ===")

    for sym, sources in saved.items():
        for src, path in sources.items():
            print(f"  ✅ {sym}/{src} → {path}")
