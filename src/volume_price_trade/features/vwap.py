"""VWAP (Volume Weighted Average Price) features for M2 (Features v1)."""

import pandas as pd
import numpy as np
from ..data.calendar import is_rth
from .ta_basic import atr


def compute_vwap_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute VWAP-based features.
    
    Args:
        df: DataFrame with OHLCV data
        cfg: Configuration dictionary
        
    Returns:
        DataFrame with VWAP features
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Calculate typical price: (high + low + close) / 3
    result_df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate price * volume for VWAP calculation
    result_df['price_volume'] = result_df['typical_price'] * df['volume']
    
    # Identify session boundaries using is_rth function
    # Create a column to track when we're in a new session
    result_df['is_new_session'] = False
    
    # For each row, check if it's the first row or if the previous row was not in RTH
    # and the current row is in RTH (indicating a new session)
    for i in range(1, len(result_df)):
        prev_idx = result_df.index[i-1]
        curr_idx = result_df.index[i]
        
        prev_in_rth = is_rth(prev_idx)
        curr_in_rth = is_rth(curr_idx)
        
        # New session starts when transitioning from non-RTH to RTH
        if not prev_in_rth and curr_in_rth:
            result_df.loc[curr_idx, 'is_new_session'] = True
    
    # Handle the first row
    if len(result_df) > 0:
        first_idx = result_df.index[0]
        result_df.loc[first_idx, 'is_new_session'] = is_rth(first_idx)
    
    # Calculate cumulative values for session VWAP
    result_df['cumulative_price_volume'] = np.nan
    result_df['cumulative_volume'] = np.nan
    
    # Initialize variables for tracking session
    cum_price_vol = 0
    cum_vol = 0
    
    for i, idx in enumerate(result_df.index):
        if result_df.loc[idx, 'is_new_session']:
            # Reset for new session
            cum_price_vol = result_df.loc[idx, 'price_volume']
            cum_vol = result_df.loc[idx, 'volume']
        else:
            # Add to current session
            cum_price_vol += result_df.loc[idx, 'price_volume']
            cum_vol += result_df.loc[idx, 'volume']
        
        result_df.loc[idx, 'cumulative_price_volume'] = cum_price_vol
        result_df.loc[idx, 'cumulative_volume'] = cum_vol
    
    # Calculate session VWAP
    result_df['vwap_session'] = result_df['cumulative_price_volume'] / result_df['cumulative_volume']
    
    # Calculate rolling VWAP (20 bars)
    window = 20
    result_df['rolling_price_volume'] = result_df['price_volume'].rolling(window=window).sum()
    result_df['rolling_volume'] = df['volume'].rolling(window=window).sum()
    result_df['vwap_rolling_20'] = result_df['rolling_price_volume'] / result_df['rolling_volume']
    
    # Calculate ATR for normalization (using window=20 as specified)
    atr_window = cfg.get('atr_window', 20)
    atr_values = atr(df, window=atr_window)
    result_df['atr'] = atr_values
    
    # Calculate distance from close to session VWAP in ATR units
    result_df['dist_close_to_vwap_session_atr'] = (df['close'] - result_df['vwap_session']) / result_df['atr']
    
    # Binary flag if price is above VWAP
    result_df['above_vwap_session'] = (df['close'] > result_df['vwap_session']).astype(int)
    
    # VWAP crossover events
    # First, shift the previous close and vwap values to compare with current values
    result_df['prev_close'] = df['close'].shift(1)
    result_df['prev_vwap_session'] = result_df['vwap_session'].shift(1)
    
    # Initialize crossover columns
    result_df['vwap_cross_up'] = 0
    result_df['vwap_cross_down'] = 0
    
    # Detect crossovers
    for i, idx in enumerate(result_df.index):
        if i == 0:  # Skip first row as we don't have previous values
            continue
            
        prev_close = result_df.loc[idx, 'prev_close']
        prev_vwap = result_df.loc[idx, 'prev_vwap_session']
        curr_close = result_df.loc[idx, 'close']
        curr_vwap = result_df.loc[idx, 'vwap_session']
        
        # Cross up: previous close <= previous VWAP and current close > current VWAP
        if prev_close <= prev_vwap and curr_close > curr_vwap:
            result_df.loc[idx, 'vwap_cross_up'] = 1
        
        # Cross down: previous close >= previous VWAP and current close < current VWAP
        if prev_close >= prev_vwap and curr_close < curr_vwap:
            result_df.loc[idx, 'vwap_cross_down'] = 1
    
    # Clean up temporary columns
    columns_to_drop = [
        'typical_price', 'price_volume', 'is_new_session',
        'cumulative_price_volume', 'cumulative_volume',
        'rolling_price_volume', 'rolling_volume', 'atr',
        'prev_close', 'prev_vwap_session'
    ]
    
    # Only drop columns that exist
    result_df = result_df.drop(columns=[col for col in columns_to_drop if col in result_df.columns])
    
    return result_df