"""
run_offline_demo.py
--------------------
Quick-start demo: runs the entire pipeline in OFFLINE mode
using synthetic GBM data — no internet connection needed.

Usage:
    python run_offline_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.run_pipeline import run_pipeline
from models.risk_model import run_risk_model, print_report

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Market Data Automation — OFFLINE DEMO")
    print("  (No internet connection required)")
    print("=" * 60 + "\n")

    # Run pipeline in offline mode with a shorter date range for speed
    results = run_pipeline(
        symbols=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        start="2022-01-01",
        end="2024-01-01",
        offline=True,
    )

    print("\n── Risk Model Output ─────────────────────────────────────────")
    reports = run_risk_model(["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    for r in reports:
        print_report(r)

    print("\n✅ Offline demo complete! Check data/processed/ for outputs.\n")
