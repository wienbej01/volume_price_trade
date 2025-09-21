"""Purged walk-forward splits + ticker holdout."""

import pandas as pd
import numpy as np
from typing import Tuple, Generator, Dict, Any
from datetime import datetime, timedelta
import logging

# Configure logger
logger = logging.getLogger(__name__)


def purged_walk_forward_splits(meta_df: pd.DataFrame, config: Dict[str, Any]) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate purged walk-forward cross-validation splits with time-blocked validation.
    
    This function creates time-ordered folds for cross-validation that:
    - Respects temporal ordering to avoid look-ahead bias
    - Includes embargo periods between training and validation
    - Handles holdout tickers separately
    
    Args:
        meta_df: DataFrame with metadata including at least 'ticker' and 'session_date' columns
        config: Configuration dictionary with cv.walk_forward and tickers sections
        
    Yields:
        Tuple of (train_idx, val_idx) for each fold, where idx are integer indices
        that can be used to index into the original DataFrame
    """
    # Extract configuration parameters
    train_months = config.get('cv', {}).get('walk_forward', {}).get('train_months', 24)
    val_months = config.get('cv', {}).get('walk_forward', {}).get('val_months', 1)
    embargo_days = config.get('cv', {}).get('walk_forward', {}).get('embargo_days', 5)
    oos_tickers = config.get('tickers', {}).get('oos', [])
    
    # Validate input
    if meta_df.empty:
        logger.warning("Empty meta_df provided")
        return
        
    if 'session_date' not in meta_df.columns:
        raise ValueError("meta_df must contain 'session_date' column")
        
    if 'ticker' not in meta_df.columns:
        raise ValueError("meta_df must contain 'ticker' column")
    
    # Convert session_date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(meta_df['session_date']):
        meta_df['session_date'] = pd.to_datetime(meta_df['session_date'])
    
    # Create a mask for holdout (OOS) tickers
    holdout_mask = ~meta_df['ticker'].isin(oos_tickers)
    
    # Filter out holdout tickers for CV splits
    cv_df = meta_df[holdout_mask].copy()
    
    if cv_df.empty:
        logger.warning("No data available after filtering holdout tickers")
        return
    
    # Sort by date to ensure proper time ordering
    cv_df = cv_df.sort_values('session_date')
    
    # Get unique session dates excluding holdout tickers
    non_holdout_dates = meta_df.loc[holdout_mask, 'session_date']
    # Normalize to midnight to ensure clean day boundaries
    unique_dates = pd.Series(pd.to_datetime(non_holdout_dates).dt.normalize().unique()).sort_values()

    available_days = len(unique_dates)
    approx_train_days = int(train_months * 21)  # ~21 trading days per month
    approx_val_days = int(val_months * 21)
    required_days = approx_train_days + approx_val_days + embargo_days

    # Fallback to day-count based splits when data is scarce
    use_day_mode = available_days < required_days

    if use_day_mode:
        logger.warning(f"Insufficient data for month-based CV; falling back to day-based splits "
                       f"(available_days={available_days}, required_days~{required_days})")

        # Heuristics for tiny datasets
        val_days = max(1, min(approx_val_days if approx_val_days > 0 else 1, max(1, available_days // 5)))
        train_days = max(1, min(approx_train_days if approx_train_days > 0 else 1,
                                max(1, available_days - val_days - max(embargo_days, 0))))

        # Ensure feasibility
        if train_days + val_days + embargo_days > available_days:
            spare = available_days - embargo_days - train_days
            val_days = max(1, spare)

        start_idx = 0
        while True:
            train_end_idx = start_idx + train_days
            val_start_idx = train_end_idx + embargo_days
            val_end_idx = val_start_idx + val_days

            if val_end_idx > available_days:
                break

            train_start = pd.to_datetime(unique_dates.iloc[start_idx])
            # exclusive end bounds
            train_end_excl = (pd.to_datetime(unique_dates.iloc[train_end_idx])
                              if train_end_idx < available_days
                              else pd.to_datetime(unique_dates.iloc[-1]) + pd.DateOffset(days=1))
            val_start = pd.to_datetime(unique_dates.iloc[val_start_idx])
            val_end_excl = (pd.to_datetime(unique_dates.iloc[val_end_idx])
                            if val_end_idx < available_days
                            else pd.to_datetime(unique_dates.iloc[-1]) + pd.DateOffset(days=1))

            overall_mask = holdout_mask
            train_mask = overall_mask & (meta_df['session_date'] >= train_start) & (meta_df['session_date'] < train_end_excl)
            val_mask = overall_mask & (meta_df['session_date'] >= val_start) & (meta_df['session_date'] < val_end_excl)

            train_indices = np.where(train_mask.to_numpy())[0]
            val_indices = np.where(val_mask.to_numpy())[0]

            if len(train_indices) == 0 or len(val_indices) == 0:
                start_idx += 1
                continue

            logger.info(
                f"Generated fold (fallback-day): train {train_start.date()}..{(train_end_excl - pd.Timedelta(days=1)).date()}, "
                f"val {val_start.date()}..{(val_end_excl - pd.Timedelta(days=1)).date()}"
            )
            yield train_indices, val_indices

            start_idx = val_end_idx
    else:
        # Month-based splits (default path)
        start_date = pd.to_datetime(unique_dates.min())
        end_date = pd.to_datetime(unique_dates.max())

        current_train_start = start_date

        # Generate walk-forward splits
        while True:
            # Calculate train and validation periods
            train_end = current_train_start + pd.DateOffset(months=train_months)
            val_start = train_end + pd.DateOffset(days=embargo_days)
            val_end = val_start + pd.DateOffset(months=val_months)

            # Check if we've exceeded the data range
            if val_end > end_date:
                break

            overall_mask = holdout_mask
            train_mask = overall_mask & (meta_df['session_date'] >= current_train_start) & (meta_df['session_date'] < train_end)
            val_mask = overall_mask & (meta_df['session_date'] >= val_start) & (meta_df['session_date'] < val_end)

            train_indices = np.where(train_mask.to_numpy())[0]
            val_indices = np.where(val_mask.to_numpy())[0]

            # Skip if either train or validation set is empty
            if len(train_indices) == 0 or len(val_indices) == 0:
                logger.warning("Skipping fold with empty train or validation set")
                current_train_start = train_end
                continue

            logger.info(
                f"Generated fold: train {current_train_start.date()} to {train_end.date()}, "
                f"val {val_start.date()} to {val_end.date()}"
            )

            # Yield the indices for this fold
            yield train_indices, val_indices

            # Move to next fold
            current_train_start = val_end


def create_holdout_mask(meta_df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """
    Create a boolean mask for holdout (OOS) tickers.
    
    Args:
        meta_df: DataFrame with metadata including 'ticker' column
        config: Configuration dictionary with tickers section
        
    Returns:
        Boolean Series where True indicates the sample is NOT a holdout (i.e., can be used for CV)
    """
    oos_tickers = config.get('tickers', {}).get('oos', [])
    
    if 'ticker' not in meta_df.columns:
        raise ValueError("meta_df must contain 'ticker' column")
    
    # Create mask: True for non-holdout tickers, False for holdout tickers
    holdout_mask = ~meta_df['ticker'].isin(oos_tickers)
    
    logger.info(f"Created holdout mask: {holdout_mask.sum()} non-holdout samples, "
               f"{len(holdout_mask) - holdout_mask.sum()} holdout samples")
    
    return holdout_mask
