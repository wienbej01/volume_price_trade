#!/usr/bin/env python3
"""
Download minute bars from Polygon API for a list of tickers.

Reads tickers from a file, downloads 5 years of data (2020-01-01 to 2025-09-20),
saves to data/polygon/{ticker}.parquet with UTC tz-aware timestamps.
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volume_price_trade.data.polygon_loader import load_minute_bars_polygon
from volume_price_trade.utils.io import save_parquet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_polygon_data(universe_file: str, start_date: str, end_date: str, output_dir: str):
    """Download data for tickers in universe file."""
    logger.info(f"Starting Polygon download for universe: {universe_file}")

    # Read tickers
    with open(universe_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    logger.info(f"Found {len(tickers)} tickers to download")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        logger.info(f"Downloading {ticker}...")
        try:
            df = load_minute_bars_polygon(
                ticker=ticker,
                start=start_date,
                end=end_date,
                adjusted=True,
                throttle_sleep=1.2  # Respect rate limits
            )

            if df.empty:
                logger.warning(f"No data for {ticker}")
                continue

            output_file = output_path / f"{ticker}.parquet"
            save_parquet(df, str(output_file))
            logger.info(f"Saved {len(df)} rows for {ticker} to {output_file}")

        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            continue

    logger.info("Polygon download complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Polygon minute data for tickers in a file.")
    parser.add_argument("--universe-file", type=str, required=True, help="File with one ticker per line")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2025-09-20", help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="data/polygon", help="Output directory")

    args = parser.parse_args()
    download_polygon_data(args.universe_file, args.start_date, args.end_date, args.output_dir)