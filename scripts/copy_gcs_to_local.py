#!/usr/bin/env python3
"""
Copy available GCS data to local data/ directory.

Reads tickers from file, loads from GCS, saves to data/{ticker}.parquet.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volume_price_trade.data.gcs_loader import load_minute_bars
from volume_price_trade.utils.io import save_parquet

def copy_gcs_to_local(universe_file: str, start_date: str, end_date: str, output_dir: str):
    """Copy GCS data to local for available tickers."""
    with open(universe_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        try:
            df = load_minute_bars(ticker=ticker, start=start_date, end=end_date)
            if df.empty:
                print(f"No data for {ticker}")
                continue

            output_file = output_path / f"{ticker}.parquet"
            save_parquet(df, str(output_file))
            print(f"Copied {ticker}: {len(df)} rows")

        except Exception as e:
            print(f"Failed to copy {ticker}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Copy GCS data to local.")
    parser.add_argument("--universe-file", required=True, help="File with available tickers")
    parser.add_argument("--start-date", default="2020-10-01", help="Start date")
    parser.add_argument("--end-date", default="2025-08-31", help="End date")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()

    copy_gcs_to_local(args.universe_file, args.start_date, args.end_date, args.output_dir)