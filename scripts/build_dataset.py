#!/usr/bin/env python
"""CLI to load and inspect minute bar data from GCS."""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volume_price_trade.data.gcs_loader import load_minute_bars, list_available_months
from volume_price_trade.utils.validation import validate_bars


import logging

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Load and inspect minute bar data")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Load config
    root = Path(__file__).resolve().parents[1]
    with open(root / "config/base.yaml") as f:
        cfg = yaml.safe_load(f)

    print(f"Loading data for {args.ticker} from {args.start} to {args.end}")

    # List available months
    months = list_available_months(args.ticker)
    print(f"Available months for {args.ticker}: {months}")

    # Load data
    df = load_minute_bars(args.ticker, args.start, args.end)

    if df.empty:
        print("No data found for the specified date range")
        return

    # Validate data
    try:
        validate_bars(df, cfg)
        print("Data validation passed")
    except AssertionError as e:
        print(f"Data validation failed: {e}")
        return

    # Print basic stats
    print("\nDataset statistics:")
    print(f"Total bars: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Trading days: {df.index.date.nunique()}")

    # Print column info
    print(f"\nColumns: {list(df.columns)}")

    # Print basic price stats
    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        print("\nPrice statistics:")
        print(f"Average daily range: {(df['high'] - df['low']).mean():.2f}")
        print(f"Average close price: {df['close'].mean():.2f}")

    # Print volume stats
    if "volume" in df.columns:
        print("\nVolume statistics:")
        print(f"Average daily volume: {df.groupby(df.index.date)['volume'].sum().mean():.0f}")
        print(f"Average minute volume: {df['volume'].mean():.0f}")


if __name__ == "__main__":
    main()