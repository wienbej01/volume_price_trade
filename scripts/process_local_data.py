
import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volume_price_trade.features.feature_union import build_feature_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_local_data(tickers: list, start_date: str, end_date: str, input_dir: str, output_dir: str):
    """Processes the downloaded data and creates the feature matrix."""
    logger.info(f"--- Starting Data Processing for {len(tickers)} tickers ---")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        logger.info(f"Processing data for {ticker}...")
        try:
            input_file = input_path / f"{ticker}.parquet"
            if not input_file.exists():
                logger.warning(f"No data found for {ticker} at {input_file}")
                continue

            df = pd.read_parquet(input_file)

            # Dummy config for now
            config = {
                'features': {
                    'atr_window': 20,
                    'rvol_windows': [5, 20],
                    'volume_profile': {
                        'bin_size': 0.05,
                        'value_area': 0.70,
                        'rolling_sessions': 20
                    }
                }
            }

            feature_matrix = build_feature_matrix(df, config)

            output_file = output_path / f"{ticker}_features.parquet"
            feature_matrix.to_parquet(output_file)
            logger.info(f"Saved {len(feature_matrix)} rows of feature data for {ticker} to {output_file}")

        except Exception as e:
            logger.error(f"An error occurred while processing data for {ticker}: {e}", exc_info=True)

    logger.info(f"--- Finished Data Processing ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the downloaded data and create the feature matrix.")
    parser.add_argument("--tickers", type=str, nargs="+", required=True, help="List of ticker symbols to process.")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default="2023-03-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--input_dir", type=str, default="data", help="Input directory where the data is saved.")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output directory to save the processed data.")
    args = parser.parse_args()

    process_local_data(tickers=args.tickers, start_date=args.start_date, end_date=args.end_date, input_dir=args.input_dir, output_dir=args.output_dir)
