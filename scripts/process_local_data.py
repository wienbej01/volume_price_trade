
import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import yaml
import time

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volume_price_trade.features.feature_union import build_feature_matrix
from volume_price_trade.utils.preprocess import preprocess_bars

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def _load_config() -> dict:
    cfg_path = Path("config/base.yaml")
    if not cfg_path.exists():
        logger.warning("config/base.yaml not found; using minimal defaults")
        return {}
    with cfg_path.open("r") as f:
        return yaml.safe_load(f) or {}

def process_local_data(tickers: list, start_date: str, end_date: str, input_dir: str, output_dir: str):
    """Processes the downloaded data and creates the feature matrix."""
    logger.info(f"--- Starting Feature Precompute for {len(tickers)} tickers ---")
    logger.info(f"Window: {start_date} .. {end_date}")

    cfg = _load_config()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse window bounds (UTC)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    start_utc = start_dt.tz_localize("UTC") if start_dt.tz is None else start_dt.tz_convert("UTC")
    end_utc = end_dt.tz_localize("UTC") if end_dt.tz is None else end_dt.tz_convert("UTC")

    for i, ticker in enumerate(tickers, 1):
        t0 = time.perf_counter()
        logger.info(f"[{i}/{len(tickers)}] Processing {ticker} ...")
        try:
            input_file = input_path / f"{ticker}.parquet"
            if not input_file.exists():
                logger.warning(f"No data found for {ticker} at {input_file}")
                continue

            # Load raw bars and normalize/tz/dedupe
            df = pd.read_parquet(input_file)
            df = preprocess_bars(df, ticker=ticker)

            # Filter by requested window
            df = df[(df.index >= start_utc) & (df.index <= end_utc)]
            if df.empty:
                logger.warning(f"[{i}/{len(tickers)}] No bars in range for {ticker} {start_date}..{end_date}")
                continue

            logger.info(f"[{i}/{len(tickers)}] Building feature matrix for {ticker} on {len(df)} rows")
            t_feat = time.perf_counter()
            # Use full project config; pass ticker for cache metadata
            feature_matrix = build_feature_matrix(df, cfg, ticker=ticker)
            logger.info(f"[{i}/{len(tickers)}] Features built in {time.perf_counter()-t_feat:.2f}s, shape={feature_matrix.shape}")

            output_file = output_path / f"{ticker}_features.parquet"
            feature_matrix.to_parquet(output_file)
            logger.info(f"[{i}/{len(tickers)}] Saved {len(feature_matrix)} rows to {output_file} in {time.perf_counter()-t0:.2f}s")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}", exc_info=True)
            continue

    logger.info(f"--- Finished Feature Precompute ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute feature matrices and save to processed_data.")
    parser.add_argument("--tickers", type=str, nargs="+", required=True, help="List of ticker symbols to process.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--input_dir", type=str, default="data", help="Input directory where the data is saved.")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output directory to save the processed data.")
    args = parser.parse_args()

    process_local_data(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
