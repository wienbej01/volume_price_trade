"""Build X/y with purging and group keys (date/ticker)."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from ..features.feature_union import build_feature_matrix
from ..labels.targets import triple_barrier_labels
from ..data.calendar import next_session_close

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
    
    # Process each ticker
    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        
        try:
            # 1. Load bars for the ticker
            # Note: In a real implementation, this would load from a database or file
            # For now, we'll assume the data is passed in or loaded elsewhere
            # This is a placeholder - actual data loading would depend on the data source
            bars_df = _load_bars_for_ticker(ticker, start_date, end_date)
            
            if bars_df.empty:
                logger.warning(f"No data found for ticker {ticker} between {start} and {end}")
                continue
                
            # 2. Build features
            logger.info(f"Building features for {ticker}")
            features_df = build_feature_matrix(bars_df, config)
            
            # 3. Generate triple-barrier labels
            logger.info(f"Generating labels for {ticker}")
            labeled_df = triple_barrier_labels(
                features_df,
                horizons_min=horizons_min,
                atr_mult_sl=atr_mult_sl,
                r_mult_tp=r_mult_tp,
                eod_flat=True
            )
            
            # 4. Add metadata
            labeled_df['ticker'] = ticker
            labeled_df['session_date'] = labeled_df.index.date
            
            # 5. Purge overlapping events and apply embargo
            logger.info(f"Purging overlapping events for {ticker}")
            purged_df = _purge_overlapping_events(
                labeled_df,
                embargo_days=embargo_days,
                horizon_minutes=horizons_min
            )
            
            # 6. Extract X, y, and meta
            # X contains all feature columns (exclude OHLCV and label columns)
            feature_columns = [col for col in purged_df.columns
                             if col not in ['open', 'high', 'low', 'close', 'volume',
                                          'y_class', 'horizon_minutes', 'event_end_time',
                                          'ticker', 'session_date']]
            
            ticker_X = purged_df[feature_columns]
            ticker_y = purged_df['y_class']
            ticker_meta = purged_df[['ticker', 'session_date']]
            
            # Append to overall results
            all_X = pd.concat([all_X, ticker_X], axis=0)
            all_y = pd.concat([all_y, ticker_y], axis=0)
            all_meta = pd.concat([all_meta, ticker_meta], axis=0)
            
            logger.info(f"Completed processing for {ticker}: {len(ticker_X)} samples")
            
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
    
    return all_X, all_y, all_meta


def _load_bars_for_ticker(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load bars for a specific ticker and date range.
    
    Note: This is a placeholder function. In a real implementation,
    this would load from a database, CSV files, or API.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with OHLCV data
    """
    # This is a placeholder - actual implementation would depend on data source
    # For now, return an empty DataFrame
    logger.warning(f"_load_bars_for_ticker is a placeholder. No data loaded for {ticker}")
    return pd.DataFrame()


def _purge_overlapping_events(
    df: pd.DataFrame,
    embargo_days: int,
    horizon_minutes: int
) -> pd.DataFrame:
    """
    Purge overlapping events and apply embargo.
    
    Args:
        df: DataFrame with labels and event times
        embargo_days: Number of days to embargo after each event
        horizon_minutes: Horizon in minutes for each event
        
    Returns:
        DataFrame with purged events
    """
    if df.empty:
        return df.copy()
    
    # Sort by index (timestamp)
    df_sorted = df.sort_index()
    
    # Initialize list to keep track of non-overlapping events
    keep_indices = []
    
    # Track the last event end time + embargo
    last_embargo_end = None
    
    # Iterate through each event
    for idx, row in df_sorted.iterrows():
        event_end_time = row['event_end_time']
        
        # Skip if event_end_time is NaT
        if pd.isna(event_end_time):
            keep_indices.append(idx)
            continue
            
        # Calculate embargo period
        embargo_end = event_end_time + pd.Timedelta(days=embargo_days)
        
        # If this is the first event or after the last event's embargo period
        if last_embargo_end is None or idx >= last_embargo_end:
            keep_indices.append(idx)
            last_embargo_end = embargo_end
    
    # Return DataFrame with only non-overlapping events
    purged_df = df_sorted.loc[keep_indices]
    
    return purged_df
