#!/usr/bin/env python3
"""
Check which tickers are available on GCS jwss_data_store.

Reads tickers from universe files, checks GCS availability using list_available_months.
Outputs available and missing tickers.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volume_price_trade.data.gcs_loader import list_available_months

def check_availability(universe_file: str):
    """Check GCS availability for tickers in file."""
    with open(universe_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    available = []
    missing = []

    for ticker in tickers:
        months = list_available_months(ticker)
        if months:
            available.append(ticker)
            print(f"AVAILABLE: {ticker} - {len(months)} months")
        else:
            missing.append(ticker)
            print(f"MISSING: {ticker}")

    return available, missing

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check GCS availability for tickers.")
    parser.add_argument("--universe-file", required=True, help="File with tickers")
    args = parser.parse_args()

    available, missing = check_availability(args.universe_file)
    print(f"\nSummary for {args.universe_file}:")
    print(f"Available: {len(available)} tickers")
    print(f"Missing: {len(missing)} tickers")

    # Write missing to file
    missing_file = args.universe_file.replace('.txt', '_missing.txt')
    with open(missing_file, 'w') as f:
        for ticker in missing:
            f.write(ticker + '\n')
    print(f"Missing tickers written to {missing_file}")