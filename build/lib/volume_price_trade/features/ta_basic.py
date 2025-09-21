"""ATR, RVOL, returns, ranges, and rolling statistics for M2 (Features v1)."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate True Range.
    
    True Range is the maximum of:
    - High - Low
    - Absolute value of High - Previous Close
    - Absolute value of Low - Previous Close
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        Series with True Range values
    """
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return true_range


def atr(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with OHLC data
        window: Window size for ATR calculation
        
    Returns:
        Series with ATR values
    """
    tr = true_range(df)
    atr_values = tr.rolling(window=window).mean()
    
    return atr_values


def rvol(df: pd.DataFrame, windows: list = [5, 20]) -> pd.DataFrame:
    """
    Calculate Relative Volume (RVOL).
    
    Args:
        df: DataFrame with volume data
        windows: List of window sizes for RVOL calculation
        
    Returns:
        DataFrame with RVOL values for each window
    """
    result = pd.DataFrame(index=df.index)
    
    for window in windows:
        rolling_mean_vol = df['volume'].rolling(window=window).mean()
        result[f'rvol_{window}'] = df['volume'] / rolling_mean_vol
    
    return result


def returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price returns.
    
    Args:
        df: DataFrame with close price data
        
    Returns:
        DataFrame with log returns and percentage returns
    """
    result = pd.DataFrame(index=df.index)
    
    # Calculate log returns: log(close/close_prev)
    result['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate percentage returns: (close-close_prev)/close_prev
    result['pct_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    return result


def rolling_stats(df: pd.DataFrame, windows: list = [5, 10, 20], 
                  columns: list = ['close', 'volume']) -> pd.DataFrame:
    """
    Calculate rolling statistics for specified columns.
    
    Args:
        df: DataFrame with OHLCV data
        windows: List of window sizes for rolling calculations
        columns: List of columns to calculate statistics for
        
    Returns:
        DataFrame with rolling statistics
    """
    result = pd.DataFrame(index=df.index)
    
    for window in windows:
        for col in columns:
            if col in df.columns:
                result[f'{col}_mean_{window}'] = df[col].rolling(window=window).mean()
                result[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
                result[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
                result[f'{col}_max_{window}'] = df[col].rolling(window=window).max()
    
    return result


def compute_ta_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute technical analysis features.
    
    Args:
        df: DataFrame with OHLCV data
        cfg: Configuration dictionary with ATR window and RVOL windows
        
    Returns:
        DataFrame with TA features
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Get configuration parameters
    atr_window = cfg.get('atr_window', 20)
    rvol_windows = cfg.get('rvol_windows', [5, 20])
    
    # 1. Calculate True Range
    tr_values = true_range(df)
    result_df['true_range'] = tr_values
    
    # 2. Calculate ATR
    atr_values = atr(df, window=atr_window)
    result_df[f'atr_{atr_window}'] = atr_values
    
    # 3. Calculate RVOL
    rvol_df = rvol(df, windows=rvol_windows)
    for col in rvol_df.columns:
        result_df[col] = rvol_df[col]
    
    # 4. Calculate Returns
    returns_df = returns(df)
    for col in returns_df.columns:
        result_df[col] = returns_df[col]
    
    # 5. Calculate Rolling Stats
    # Use windows from RVOL configuration plus ATR window for consistency
    rolling_windows = list(set(rvol_windows + [atr_window]))
    rolling_windows.sort()  # Sort for consistent column ordering
    rolling_df = rolling_stats(df, windows=rolling_windows)
    for col in rolling_df.columns:
        result_df[col] = rolling_df[col]
    
    return result_df


# Legacy function for backward compatibility
def add_ta(df):
    """
    Legacy function for backward compatibility.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with TA features using default configuration
    """
    default_cfg = {
        'atr_window': 20,
        'rvol_windows': [5, 20]
    }
    return compute_ta_features(df, default_cfg)
