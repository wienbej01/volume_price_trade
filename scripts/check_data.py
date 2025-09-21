
import argparse
import pyarrow.parquet as pq
from pathlib import Path

def check_data(tickers: list):
    """Checks the metadata of the Parquet files for a list of tickers."""
    print("--- Checking Data Files ---")

    for ticker in tickers:
        print(f"Checking data for {ticker}...")
        try:
            file_path = Path(f"data/{ticker}.parquet")
            if not file_path.exists():
                print(f"  File not found: {file_path}")
                continue

            metadata = pq.read_metadata(file_path)
            print(f"  Metadata for {file_path}:")
            print(metadata)

        except Exception as e:
            print(f"  An error occurred while checking data for {ticker}: {e}")

    print("--- Finished Checking Data Files ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the metadata of Parquet files.")
    parser.add_argument("--tickers", type=str, nargs="+", required=True, help="List of ticker symbols to check.")
    args = parser.parse_args()

    check_data(tickers=args.tickers)
