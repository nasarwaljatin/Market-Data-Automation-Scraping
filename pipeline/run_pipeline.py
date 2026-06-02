"""
pipeline/run_pipeline.py
------------------------
Full pipeline orchestrator.

Runs all stages in sequence:
  1. Fetch   — NSE, Yahoo Finance, Google Finance  (OR offline synthetic data)
  2. Standardize — schema alignment + IST timezone normalization
  3. Resolve conflicts — priority-weighted cross-source resolution
  4. Preprocess — missing values, outliers, corporate actions
  5. Feature engineering — log returns, volatility, RSI, momentum, beta
  6. Save outputs to data/processed/

Usage:
    python pipeline/run_pipeline.py                         # online mode
    python pipeline/run_pipeline.py --offline               # offline/demo mode
    python pipeline/run_pipeline.py --symbols RELIANCE.NS TCS.NS --start 2023-01-01 --end 2024-01-01
    python pipeline/run_pipeline.py --skip-fetch            # use existing raw files
"""

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import SYMBOLS, NSE_SYMBOLS, START_DATE, END_DATE
from utils.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1a: Online data ingestion
# ──────────────────────────────────────────────────────────────────────────────

def stage_fetch(symbols: list, nse_symbols: list, start: str, end: str) -> None:
    """Fetch raw data from all three sources for all symbols (requires internet)."""
    from scrapers.nse_scraper    import save_nse_data
    from scrapers.yahoo_fetcher  import save_yahoo_data
    from scrapers.google_fetcher import save_google_data

    log.info("── Stage 1: Data Ingestion (ONLINE) ─────────────────────────")

    for nse_sym, yahoo_sym in zip(nse_symbols, symbols):
        log.info(f"  Fetching NSE:    {nse_sym}")
        try:
            save_nse_data(nse_sym, start, end)
        except Exception as exc:
            log.error(f"  NSE fetch failed for {nse_sym}: {exc}")

        log.info(f"  Fetching Yahoo:  {yahoo_sym}")
        try:
            save_yahoo_data(yahoo_sym, start, end)
        except Exception as exc:
            log.error(f"  Yahoo fetch failed for {yahoo_sym}: {exc}")

        log.info(f"  Fetching Google: {nse_sym}")
        try:
            save_google_data(nse_sym, start, end)
        except Exception as exc:
            log.error(f"  Google fetch failed for {nse_sym}: {exc}")

        time.sleep(1)   # polite delay between symbols


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1b: Offline synthetic data generation
# ──────────────────────────────────────────────────────────────────────────────

def stage_fetch_offline(symbols: list, start: str, end: str) -> None:
    """Generate synthetic GBM market data (no internet required)."""
    from pipeline.offline_mode import save_offline_data

    log.info("── Stage 1: Data Generation (OFFLINE / DEMO MODE) ───────────")
    log.info("   Using Geometric Brownian Motion to simulate realistic OHLCV data.")
    save_offline_data(symbols, start, end)
    log.info("   Offline data ready in data/raw/")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Standardization
# ──────────────────────────────────────────────────────────────────────────────

def stage_standardize(symbols: list) -> dict:
    """Align schemas and normalize timezones for all sources."""
    from pipeline.standardizer import standardize_all, save_standardized

    log.info("── Stage 2: Standardization ─────────────────────────────────")
    std_data = standardize_all(symbols)
    save_standardized(std_data)
    return std_data


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Conflict resolution
# ──────────────────────────────────────────────────────────────────────────────

def stage_resolve(symbols: list) -> dict:
    """Detect and resolve cross-source OHLCV discrepancies."""
    from pipeline.conflict_resolver import resolve_all

    log.info("── Stage 3: Conflict Resolution ─────────────────────────────")
    return resolve_all(symbols)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def stage_preprocess(symbols: list) -> dict:
    """Handle missing values, detect outliers, apply corporate actions."""
    from pipeline.preprocessor import preprocess_all

    log.info("── Stage 4: Preprocessing ───────────────────────────────────")
    return preprocess_all(symbols)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5: Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def stage_features(symbols: list) -> dict:
    """Generate all financial features for risk model input."""
    from pipeline.feature_engineer import engineer_all

    log.info("── Stage 5: Feature Engineering ─────────────────────────────")
    return engineer_all(symbols)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    symbols: list = None,
    start: str = START_DATE,
    end: str = END_DATE,
    skip_fetch: bool = False,
    offline: bool = False,
) -> dict:
    """
    Execute the full market data pipeline.

    Parameters
    ----------
    symbols    : list  — Yahoo-style symbols (e.g. ['RELIANCE.NS', 'TCS.NS'])
    start      : str   — 'YYYY-MM-DD'
    end        : str   — 'YYYY-MM-DD'
    skip_fetch : bool  — if True, skip data ingestion (use existing raw files)
    offline    : bool  — if True, use synthetic GBM data (no internet needed)

    Returns
    -------
    dict — { symbol: pd.DataFrame } of final feature-engineered data
    """
    if symbols is None:
        symbols = SYMBOLS

    nse_symbols = [s.replace(".NS", "") for s in symbols]

    mode_label = "OFFLINE (synthetic GBM)" if offline else "ONLINE (live sources)"
    log.info("=" * 65)
    log.info("  Market Data Automation Pipeline")
    log.info(f"  Mode      : {mode_label}")
    log.info(f"  Symbols   : {', '.join(symbols)}")
    log.info(f"  Date range: {start} → {end}")
    log.info(f"  Skip fetch: {skip_fetch}")
    log.info("=" * 65)

    t0 = time.time()

    # ── Stage 1 ───────────────────────────────────────────────────
    if skip_fetch:
        log.info("── Stage 1: Skipped (--skip-fetch) ──────────────────────────")
    elif offline:
        stage_fetch_offline(symbols, start, end)
    else:
        stage_fetch(symbols, nse_symbols, start, end)

    # ── Stage 2 ───────────────────────────────────────────────────
    stage_standardize(symbols)

    # ── Stage 3 ───────────────────────────────────────────────────
    stage_resolve(symbols)

    # ── Stage 4 ───────────────────────────────────────────────────
    stage_preprocess(symbols)

    # ── Stage 5 ───────────────────────────────────────────────────
    final = stage_features(symbols)

    elapsed = time.time() - t0
    log.info("=" * 65)
    log.info(f"  Pipeline complete in {elapsed:.1f}s")
    log.info(f"  Output directory: data/processed/")
    log.info("=" * 65)

    return final


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Market Data Automation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/run_pipeline.py
  python pipeline/run_pipeline.py --symbols RELIANCE.NS TCS.NS --start 2023-01-01 --end 2024-01-01
  python pipeline/run_pipeline.py --skip-fetch
        """,
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to process (Yahoo Finance format, e.g. RELIANCE.NS TCS.NS)"
    )
    parser.add_argument("--start",      default=START_DATE, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",        default=END_DATE,   help="End date   YYYY-MM-DD")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data ingestion stage")
    parser.add_argument("--offline",    action="store_true",
                        help="Run in offline mode using synthetic GBM data (no internet needed)")
    args = parser.parse_args()

    results = run_pipeline(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        skip_fetch=args.skip_fetch,
        offline=args.offline,
    )

    print("\n── Pipeline Summary ──────────────────────────────────────────")
    for sym, df in results.items():
        if df is not None and not df.empty:
            print(f"  ✅ {sym}: {len(df)} rows, {df.shape[1]} columns → data/processed/features_{sym.replace('.', '_')}.csv")
        else:
            print(f"  ❌ {sym}: no output generated")
    print()
