"""Build X/y with purging and group keys (date/ticker)."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import time
from ..features.feature_union import build_feature_matrix
from ..labels.targets import triple_barrier_labels
from ..data.calendar import next_session_close
from ..data.gcs_loader import load_minute_bars
from pathlib import Path



# Configure logger
logger = logging.getLogger(__name__)


def make_dataset(
    tickers: List[str],
    start: str,
    end: str,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build dataset with features, labels, and metadata.
    
    Process:
    1. Load bars per ticker → features (feature_union) → labels (triple-barrier)
    2. Add meta: ticker, session_date
    3. Purge overlapping events; apply embargo days from config
    4. Return X, y, meta
    
    Args:
        tickers: List of ticker symbols to process
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)
        config: Configuration dictionary
        
    Returns:
        Tuple of (X, y, meta) where:
        - X: DataFrame with features
        - y: Series with labels
        - meta: DataFrame with metadata (ticker, session_date)
    """
    # Convert date strings to datetime
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    # Initialize empty DataFrames to store results
    all_X = pd.DataFrame()
    all_y = pd.Series(dtype='object')
    all_meta = pd.DataFrame()
    
    # Get configuration parameters
    horizons_min = config.get('horizons_minutes', [60])[0]  # Use first horizon as default
    atr_mult_sl = config.get('risk', {}).get('atr_stop_mult', 1.5)
    r_mult_tp = config.get('risk', {}).get('take_profit_R', 2.0)
    eod_flat_minutes_before_close = config.get('sessions', {}).get('eod_flat_minutes_before_close', 1)
    embargo_days = config.get('cv', {}).get('walk_forward', {}).get('embargo_days', 5)
    # Data source toggles
    use_processed = config.get('data', {}).get('use_processed', True)
    event_stride = int(config.get('data', {}).get('event_stride_minutes', 1) or 1)
    
    # Process each ticker
    for i, ticker in enumerate(tickers, 1):
        
        try:
            t0_ticker = time.perf_counter()
            logger.info(f"[{i}/{len(tickers)}] Start {ticker}")
            labeled_df: pd.DataFrame

            # 0. Fast-path: processed_data cache
            processed_df = pd.DataFrame()
            if use_processed:
                processed_df = _load_processed_features(ticker, start_date, end_date)
                if not processed_df.empty:
                    logger.info(f"Using processed features for {ticker}: shape {processed_df.shape}")

            if not processed_df.empty:
                # If labels are included, use directly
                if 'y_class' in processed_df.columns:
                    labeled_df = processed_df.copy()
                else:
                    # Build labels directly from processed features' OHLCV to avoid external bar loads
                    processed_start = processed_df.index.min()
                    processed_end = processed_df.index.max()
                    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing = [c for c in ohlcv_cols if c not in processed_df.columns]
                    if missing:
                        logger.warning(f"Processed features missing OHLCV columns {missing} for {ticker}; skipping")
                        continue
                    logger.info(f"Generating labels for {ticker} from processed OHLCV (date range: {processed_start} to {processed_end})")
                    labels_only = triple_barrier_labels(
                        processed_df[ohlcv_cols],
                        horizons_min=horizons_min,
                        atr_mult_sl=atr_mult_sl,
                        r_mult_tp=r_mult_tp,
                        eod_flat=True
                    )[["y_class", "horizon_minutes", "event_end_time"]]
                    labeled_df = processed_df.join(labels_only, how='inner')
            else:
                # 1. Load bars for the ticker
                bars_df = _load_bars_for_ticker(ticker, start_date, end_date, config)
                logger.info(f"Loaded bars for {ticker}: shape {bars_df.shape}")
                
                if bars_df.empty:
                    logger.warning(f"No data found for ticker {ticker} between {start} and {end}")
                    continue
                    
                # 2. Build features
                logger.info(f"[{i}/{len(tickers)}] Building features for {ticker}")
                t_feat = time.perf_counter()
                features_df = build_feature_matrix(bars_df, config, ticker=ticker)
                logger.info(f"[{i}/{len(tickers)}] Built features for {ticker} in {time.perf_counter()-t_feat:.2f}s shape={features_df.shape}")
                
                # 3. Generate triple-barrier labels
                logger.info(f"[{i}/{len(tickers)}] Generating labels for {ticker}")
                t_lab = time.perf_counter()
                labeled_df = triple_barrier_labels(
                    features_df,
                    horizons_min=horizons_min,
                    atr_mult_sl=atr_mult_sl,
                    r_mult_tp=r_mult_tp,
                    eod_flat=True
                )
                logger.info(f"[{i}/{len(tickers)}] Labels generated for {ticker} in {time.perf_counter()-t_lab:.2f}s shape={labeled_df.shape}")

            # 4. Optional event stride downsampling to reduce event starts (speeds purging)
            if 'event_stride' not in locals():
                event_stride = int(config.get('data', {}).get('event_stride_minutes', 1) or 1)
            if event_stride > 1:
                orig_len = len(labeled_df)
                labeled_df = labeled_df.iloc[::event_stride]
                logger.info(f"Applied event_stride_minutes={event_stride} for {ticker}: {orig_len} -> {len(labeled_df)} rows")

            # 5. Add metadata
            labeled_df['ticker'] = ticker
            labeled_df['session_date'] = pd.to_datetime(labeled_df.index).date
            
            # 5. Purge overlapping events and apply embargo
            logger.info(f"[{i}/{len(tickers)}] Purging overlapping events for {ticker}")
            t_purge = time.perf_counter()
            purged_df = _purge_overlapping_events(
                labeled_df,
                embargo_days=embargo_days,
                horizon_minutes=horizons_min
            )
            logger.info(f"[{i}/{len(tickers)}] Purge complete for {ticker} in {time.perf_counter()-t_purge:.2f}s -> {len(purged_df)} rows")
            
            # 6. Extract X, y, and meta
            # X contains all feature columns (exclude OHLCV and label columns)
            feature_columns = [col for col in purged_df.columns
                             if col not in ['open', 'high', 'low', 'close', 'volume',
                                          'y_class', 'horizon_minutes', 'event_end_time',
                                          'ticker', 'session_date', 'session', 'date_et', 'n', 'vw']]
            
            ticker_X = purged_df[feature_columns]
            ticker_y = purged_df['y_class']
            ticker_meta = purged_df[['ticker', 'session_date']].copy()
            ticker_meta['timestamp'] = purged_df.index
            
            # Append to overall results
            all_X = pd.concat([all_X, ticker_X], axis=0)
            all_y = pd.concat([all_y, ticker_y], axis=0)
            all_meta = pd.concat([all_meta, ticker_meta], axis=0)
            
            logger.info(f"[{i}/{len(tickers)}] Completed {ticker}: {len(ticker_X)} samples in {time.perf_counter()-t0_ticker:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Final validation
    if all_X.empty:
        logger.error("No data processed successfully")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()
    
    # Ensure indices are aligned
    all_X = all_X.sort_index()
    all_y = all_y.sort_index()
    all_meta = all_meta.sort_index()
    
    logger.info(f"Dataset construction complete: {len(all_X)} samples, {len(all_X.columns)} features")
    logger.info(f"Final shapes - X: {all_X.shape}, y: {all_y.shape}, meta: {all_meta.shape}")

    return all_X, all_y, all_meta


def _load_processed_features(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load precomputed feature matrix from processed_data if available.

    Args:
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with processed features indexed by timestamp, or empty DataFrame
    """
    try:
        file_path = Path(f"processed_data/{ticker}_features.parquet")
        if not file_path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(file_path)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            else:
                df.index = pd.to_datetime(df.index)

        # TZ normalize to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        # Filter by date range
        start_dt = start_date.tz_localize('UTC')
        end_dt = end_date.tz_localize('UTC')
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        # Drop duplicate indices if any
        if df.index.has_duplicates:
            logger.warning(f"Duplicate timestamps in processed features for {ticker}; dropping duplicates.")
            df = df[~df.index.duplicated(keep='first')]

        return df
    except Exception as e:
        logger.error(f"Error loading processed features for {ticker}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def _load_bars_for_ticker(ticker: str, start_date: datetime, end_date: datetime, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load bars for a specific ticker and date range with source fallback:
    data/*.parquet -> GCS -> Polygon.
    """
    # Normalize bounds
    start_utc = start_date.tz_localize('UTC') if start_date.tz is None else start_date.tz_convert('UTC')
    end_utc = end_date.tz_localize('UTC') if end_date.tz is None else end_date.tz_convert('UTC')

    # 1) Local parquet: data/{ticker}.parquet
    try:
        file_path = Path(f"data/{ticker}.parquet")
        if file_path.exists():
            bars_df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(bars_df)} rows from {file_path}. Head:\n{bars_df.head()}")
            # Preprocess: tz, sort, dedupe
            from ..utils.preprocess import preprocess_bars
            bars_df = preprocess_bars(bars_df, ticker=ticker)

            # Filter by date range
            bars_df = bars_df[(bars_df.index >= start_utc) & (bars_df.index <= end_utc)]

            if not bars_df.empty:
                return bars_df
        else:
            logger.info(f"Local parquet not found for {ticker} at {file_path}")
    except Exception as e:
        logger.warning(f"Local parquet load failed for {ticker}: {e}")

    # 2) GCS fallback via loader if enabled
    try:
        use_gcsfs = config.get('data', {}).get('gcs', {}).get('use_gcsfs', True)
        use_gcs = config.get('data', {}).get('use_gcs', True)
        if use_gcs and use_gcsfs:
            logger.info(f"Attempting GCS load for {ticker}")
            bars_df = load_minute_bars(
                ticker=ticker,
                start=start_utc.strftime('%Y-%m-%d'),
                end=end_utc.strftime('%Y-%m-%d')
            )
            if not bars_df.empty:
                # Preprocess: tz, sort, dedupe
                from ..utils.preprocess import preprocess_bars
                bars_df = preprocess_bars(bars_df, ticker=ticker)

                # Filter again to be safe
                bars_df = bars_df[(bars_df.index >= start_utc) & (bars_df.index <= end_utc)]
                return bars_df
            logger.warning(f"GCS returned no data for {ticker} {start_utc}..{end_utc}")
    except Exception as e:
        logger.warning(f"GCS load failed for {ticker}: {e}")

    # 3) Polygon fallback if enabled
    try:
        use_polygon = config.get('data', {}).get('use_polygon', False)
        if use_polygon:
            logger.info(f"Attempting Polygon load for {ticker}")
            from ..data.polygon_loader import load_minute_bars_polygon
            bars_df = load_minute_bars_polygon(
                ticker=ticker,
                start=start_utc.strftime('%Y-%m-%d'),
                end=end_utc.strftime('%Y-%m-%d')
            )
            if not bars_df.empty:
                # Preprocess: tz, sort, dedupe
                from ..utils.preprocess import preprocess_bars
                bars_df = preprocess_bars(bars_df, ticker=ticker)

                # Filter again to be safe
                bars_df = bars_df[(bars_df.index >= start_utc) & (bars_df.index <= end_utc)]
                return bars_df
            logger.warning(f"Polygon returned no data for {ticker} {start_utc}..{end_utc}")
    except Exception as e:
        logger.warning(f"Polygon load failed for {ticker}: {e}")

    logger.warning(f"No bars available for {ticker} after local, GCS, and Polygon fallbacks.")
    return pd.DataFrame()


def _purge_overlapping_events(
    df: pd.DataFrame,
    embargo_days: int,
    horizon_minutes: int
) -> pd.DataFrame:
    """
    Greedy, vectorized-style purge of overlapping events with embargo.
    Keeps the earliest event, then skips subsequent starts until the previous
    event's (end + embargo) time is passed.

    Significantly faster than iterrows-based approach on large datasets.
    """
    if df.empty:
        return df.copy()

    # Sort once
    df_sorted = df.sort_index()

    # Effective event end time: fill NaT with (start + horizon)
    effective_end = df_sorted['event_end_time'].copy()
    fallback_end = df_sorted.index + pd.to_timedelta(horizon_minutes, unit='m')
    effective_end = effective_end.where(~effective_end.isna(), fallback_end)

    embargo_delta = pd.to_timedelta(embargo_days, unit='D')
    embargo_end = effective_end + embargo_delta

    # Convert to int64 nanoseconds arrays for fast comparisons
    idx = df_sorted.index
    if idx.tz is not None:
        idx_ns = idx.tz_convert('UTC').tz_localize(None).asi8
        embargo_end_ns = embargo_end.dt.tz_convert('UTC').dt.tz_localize(None).astype('int64').to_numpy()
    else:
        idx_ns = idx.asi8
        embargo_end_ns = embargo_end.astype('int64').to_numpy()

    keep = np.zeros(len(df_sorted), dtype=bool)
    last_ns = np.int64(-2**63)  # effectively -inf

    # Greedy selection in pure Python loop over arrays (fast)
    for i in range(len(idx_ns)):
        if idx_ns[i] >= last_ns:
            keep[i] = True
            last_ns = embargo_end_ns[i]

    purged_df = df_sorted.iloc[keep]
    return purged_df
