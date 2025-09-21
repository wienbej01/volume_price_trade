"""Top-5 VPA events: climax, VDU, churn, effort-vs-result, breakout-conf."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .ta_basic import atr, rvol


def detect_climax_up(df: pd.DataFrame, idx: int, atr_threshold: float, rvol_threshold: float) -> int:
    """
    Detect climax up pattern.
    
    Args:
        df: DataFrame with OHLCV data
        idx: Current index to check
        atr_threshold: ATR threshold for body size
        rvol_threshold: RVOL threshold for volume
        
    Returns:
        Binary flag (1 if pattern detected, 0 otherwise)
    """
    if idx < 1:
        return 0
    
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    
    # Check if current close is higher than previous close
    if current['close'] <= previous['close']:
        return 0
    
    # Calculate body size as absolute difference between open and close
    body_size = abs(current['close'] - current['open'])
    
    # Get ATR value
    atr_col = [col for col in df.columns if col.startswith('atr_')][0]
    atr_value = df.iloc[idx][atr_col]
    
    # Check if body size > k*ATR
    if body_size <= atr_threshold * atr_value:
        return 0
    
    # Get RVOL value (use the first available RVOL column)
    rvol_col = [col for col in df.columns if col.startswith('rvol_')][0]
    rvol_value = df.iloc[idx][rvol_col]
    
    # Check if RVOL > threshold
    if rvol_value <= rvol_threshold:
        return 0
    
    return 1


def detect_climax_down(df: pd.DataFrame, idx: int, atr_threshold: float, rvol_threshold: float) -> int:
    """
    Detect climax down pattern.
    
    Args:
        df: DataFrame with OHLCV data
        idx: Current index to check
        atr_threshold: ATR threshold for body size
        rvol_threshold: RVOL threshold for volume
        
    Returns:
        Binary flag (1 if pattern detected, 0 otherwise)
    """
    if idx < 1:
        return 0
    
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    
    # Check if current close is lower than previous close
    if current['close'] >= previous['close']:
        return 0
    
    # Calculate body size as absolute difference between open and close
    body_size = abs(current['close'] - current['open'])
    
    # Get ATR value
    atr_col = [col for col in df.columns if col.startswith('atr_')][0]
    atr_value = df.iloc[idx][atr_col]
    
    # Check if body size > k*ATR
    if body_size <= atr_threshold * atr_value:
        return 0
    
    # Get RVOL value (use the first available RVOL column)
    rvol_col = [col for col in df.columns if col.startswith('rvol_')][0]
    rvol_value = df.iloc[idx][rvol_col]
    
    # Check if RVOL > threshold
    if rvol_value <= rvol_threshold:
        return 0
    
    return 1


def detect_vdu(df: pd.DataFrame, idx: int, rvol_threshold: float) -> int:
    """
    Detect Volume Down Up (VDU) pattern.
    
    Args:
        df: DataFrame with OHLCV data
        idx: Current index to check
        rvol_threshold: RVOL threshold for volume
        
    Returns:
        Binary flag (1 if pattern detected, 0 otherwise)
    """
    if idx < 1:
        return 0
    
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    
    # Check if current bar is up (close > open)
    if current['close'] <= current['open']:
        return 0
    
    # Check if previous bar is down (close < open)
    if previous['close'] >= previous['open']:
        return 0
    
    # Get RVOL value (use the first available RVOL column)
    rvol_col = [col for col in df.columns if col.startswith('rvol_')][0]
    
    # Check if both bars have high RVOL
    if df.iloc[idx][rvol_col] <= rvol_threshold or df.iloc[idx - 1][rvol_col] <= rvol_threshold:
        return 0
    
    return 1


def detect_churn(df: pd.DataFrame, idx: int, atr_threshold: float, rvol_threshold: float) -> int:
    """
    Detect churn pattern.
    
    Args:
        df: DataFrame with OHLCV data
        idx: Current index to check
        atr_threshold: ATR threshold for body size
        rvol_threshold: RVOL threshold for volume
        
    Returns:
        Binary flag (1 if pattern detected, 0 otherwise)
    """
    if idx < 1:
        return 0
    
    current = df.iloc[idx]
    
    # Calculate body size as absolute difference between open and close
    body_size = abs(current['close'] - current['open'])
    
    # Get ATR value
    atr_col = [col for col in df.columns if col.startswith('atr_')][0]
    atr_value = df.iloc[idx][atr_col]
    
    # Check if body size is small relative to ATR
    if body_size > atr_threshold * atr_value:
        return 0
    
    # Get RVOL value (use the first available RVOL column)
    rvol_col = [col for col in df.columns if col.startswith('rvol_')][0]
    rvol_value = df.iloc[idx][rvol_col]
    
    # Check if RVOL > threshold
    if rvol_value <= rvol_threshold:
        return 0
    
    return 1


def detect_effort_no_result(df: pd.DataFrame, idx: int, atr_threshold: float, rvol_threshold: float) -> int:
    """
    Detect effort no result pattern.
    
    Args:
        df: DataFrame with OHLCV data
        idx: Current index to check
        atr_threshold: ATR threshold for body size
        rvol_threshold: RVOL threshold for volume
        
    Returns:
        Binary flag (1 if pattern detected, 0 otherwise)
    """
    if idx < 1:
        return 0
    
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    
    # Calculate body size as absolute difference between open and close
    body_size = abs(current['close'] - current['open'])
    
    # Get ATR value
    atr_col = [col for col in df.columns if col.startswith('atr_')][0]
    atr_value = df.iloc[idx][atr_col]
    
    # Check if body size is significant relative to ATR
    if body_size <= atr_threshold * atr_value:
        return 0
    
    # Get RVOL value (use the first available RVOL column)
    rvol_col = [col for col in df.columns if col.startswith('rvol_')][0]
    rvol_value = df.iloc[idx][rvol_col]
    
    # Check if RVOL > threshold
    if rvol_value <= rvol_threshold:
        return 0
    
    # Determine direction of effort (up or down)
    effort_up = current['close'] > current['open']
    
    # Check if price closes opposite to effort direction
    if effort_up and current['close'] <= previous['close']:
        return 1
    elif not effort_up and current['close'] >= previous['close']:
        return 1
    
    return 0


def detect_breakout(df: pd.DataFrame, idx: int, rvol_threshold: float, lookback: int = 20) -> int:
    """
    Detect breakout confirmation pattern.
    
    Args:
        df: DataFrame with OHLCV data
        idx: Current index to check
        rvol_threshold: RVOL threshold for volume
        lookback: Number of periods to look back for range calculation
        
    Returns:
        Binary flag (1 if pattern detected, 0 otherwise)
    """
    if idx < lookback:
        return 0
    
    current = df.iloc[idx]
    
    # Calculate range over lookback period
    lookback_data = df.iloc[idx - lookback:idx]
    range_high = lookback_data['high'].max()
    range_low = lookback_data['low'].min()
    
    # Check if current bar breaks the range
    breaks_high = current['high'] > range_high
    breaks_low = current['low'] < range_low
    
    if not (breaks_high or breaks_low):
        return 0
    
    # Get RVOL value (use the first available RVOL column)
    rvol_col = [col for col in df.columns if col.startswith('rvol_')][0]
    rvol_value = df.iloc[idx][rvol_col]
    
    # Check if RVOL > threshold
    if rvol_value <= rvol_threshold:
        return 0
    
    return 1


def compute_vpa_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute Volume Price Analysis (VPA) features.
    
    Args:
        df: DataFrame with OHLCV data
        cfg: Configuration dictionary with VPA parameters
        
    Returns:
        DataFrame with VPA binary flags
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Get configuration parameters
    rvol_climax = cfg.get('rvol_climax', 2.5)
    vdu_threshold = cfg.get('vdu_threshold', 0.4)
    atr_threshold = cfg.get('atr_threshold', 0.5)  # Default multiplier for ATR
    breakout_lookback = cfg.get('breakout_lookback', 20)
    
    # Initialize result columns
    result_df['vpa_climax_up'] = 0
    result_df['vpa_climax_down'] = 0
    result_df['vpa_vdu'] = 0
    result_df['vpa_churn'] = 0
    result_df['vpa_effort_no_result'] = 0
    result_df['vpa_breakout_conf'] = 0
    
    # Iterate through the DataFrame to detect patterns
    for i, idx in enumerate(df.index):
        result_df.at[idx, 'vpa_climax_up'] = detect_climax_up(
            df, i, atr_threshold, rvol_climax
        )
        result_df.at[idx, 'vpa_climax_down'] = detect_climax_down(
            df, i, atr_threshold, rvol_climax
        )
        result_df.at[idx, 'vpa_vdu'] = detect_vdu(
            df, i, rvol_climax
        )
        result_df.at[idx, 'vpa_churn'] = detect_churn(
            df, i, atr_threshold, rvol_climax
        )
        result_df.at[idx, 'vpa_effort_no_result'] = detect_effort_no_result(
            df, i, atr_threshold, rvol_climax
        )
        result_df.at[idx, 'vpa_breakout_conf'] = detect_breakout(
            df, i, rvol_climax, breakout_lookback
        )
    
    return result_df
