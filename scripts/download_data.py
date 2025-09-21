
import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volume_price_trade.data.gcs_loader import load_minute_bars

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_data(tickers: list, start_date: str, end_date: str, output_dir: str):
    """Downloads data for a list of tickers and saves it to Parquet files."""
    logger.info(f"--- Starting Data Download for {len(tickers)} tickers ---")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        logger.info(f"Downloading data for {ticker}...")
        try:
            df = load_minute_bars(ticker=ticker, start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data found for {ticker}.")
            else:
                output_file = output_path / f"{ticker}.parquet"
                df.to_parquet(output_file)
                logger.info(f"Saved {len(df)} rows of data for {ticker} to {output_file}")

        except Exception as e:
            logger.error(f"An error occurred while downloading data for {ticker}: {e}", exc_info=True)

    logger.info(f"--- Finished Data Download ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data for a list of tickers.")
    parser.add_argument("--tickers", type=str, nargs="+", required=True, help="List of ticker symbols to download.")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default="2023-01-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory to save the data.")
    args = parser.parse_args()

    download_data(tickers=args.tickers, start_date=args.start_date, end_date=args.end_date, output_dir=args.output_dir)
