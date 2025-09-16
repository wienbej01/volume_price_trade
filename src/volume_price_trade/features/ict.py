"""Top-5 ICT proxies: FVG, liquidity sweep, displacement, EQ distance, killzones."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .ta_basic import atr
from datetime import time


def detect_fvg_up(df: pd.DataFrame, idx: int, min_size_atr: float = 0.25) -> bool:
    """
    Detect FVG (Fair Value Gap) up pattern at the given index.
    
    FVG Up: 3-bar pattern where Bar1 high < Bar2 low < Bar3 high (gap up)
    
    Args:
        df: DataFrame with OHLC data
        idx: Current index to check
        min_size_atr: Minimum gap size as a multiple of ATR
        
    Returns:
        True if FVG up pattern is detected, False otherwise
    """
    if idx < 2:
        return False
    
    # Get the three bars
    bar1 = df.iloc[idx-2]
    bar2 = df.iloc[idx-1]
    bar3 = df.iloc[idx]
    
    # Check if we have enough data for ATR calculation
    if idx < 20:  # Default ATR window is 20
        # For testing purposes, use a smaller window if we don't have enough data
        atr_window = min(idx, 5)  # Use at least 5 bars or all available if less
        if atr_window < 2:
            return False
    else:
        atr_window = 20
    
    # Calculate ATR at the current index with dynamic window
    atr_value = atr(df.iloc[:idx+1], window=atr_window).iloc[-1]
    
    # Check FVG up condition: Bar1 high < Bar2 low < Bar3 high
    gap_exists = bar1['high'] < bar2['low'] < bar3['high']
    
    # Check if gap size is significant enough
    gap_size = bar2['low'] - bar1['high']
    min_gap_size = min_size_atr * atr_value
    
    return gap_exists and gap_size >= min_gap_size


def detect_fvg_down(df: pd.DataFrame, idx: int, min_size_atr: float = 0.25) -> bool:
    """
    Detect FVG (Fair Value Gap) down pattern at the given index.
    
    FVG Down: 3-bar pattern where Bar1 low > Bar2 high > Bar3 low (gap down)
    
    Args:
        df: DataFrame with OHLC data
        idx: Current index to check
        min_size_atr: Minimum gap size as a multiple of ATR
        
    Returns:
        True if FVG down pattern is detected, False otherwise
    """
    if idx < 2:
        return False
    
    # Get the three bars
    bar1 = df.iloc[idx-2]
    bar2 = df.iloc[idx-1]
    bar3 = df.iloc[idx]
    
    # Check if we have enough data for ATR calculation
    if idx < 20:  # Default ATR window is 20
        # For testing purposes, use a smaller window if we don't have enough data
        atr_window = min(idx, 5)  # Use at least 5 bars or all available if less
        if atr_window < 2:
            return False
    else:
        atr_window = 20
    
    # Calculate ATR at the current index with dynamic window
    atr_value = atr(df.iloc[:idx+1], window=atr_window).iloc[-1]
    
    # Check FVG down condition: Bar1 low > Bar2 high > Bar3 low
    gap_exists = bar1['low'] > bar2['high'] > bar3['low']
    
    # Check if gap size is significant enough
    gap_size = bar1['low'] - bar2['high']
    min_gap_size = min_size_atr * atr_value
    
    return gap_exists and gap_size >= min_gap_size


def detect_liquidity_sweep_up(df: pd.DataFrame, idx: int, lookback: int = 10) -> bool:
    """
    Detect liquidity sweep up pattern at the given index.
    
    Liquidity Sweep Up: Price breaks prior swing high (capturing buy-side liquidity)
    but then reverses direction. This can be either:
    1. Price momentarily surpasses the highs but fails to close above them, then reverses
    2. Price closes above the highs but quickly reverses with strong selling momentum
    
    Args:
        df: DataFrame with OHLC data
        idx: Current index to check
        lookback: Number of bars to look back for swing high
        
    Returns:
        True if liquidity sweep up pattern is detected, False otherwise
    """
    if idx < lookback + 1:
        return False
    
    # Current and previous bars
    current_bar = df.iloc[idx]
    prev_bar = df.iloc[idx-1]
    
    # Find the highest high in the lookback period (excluding current bar)
    lookback_data = df.iloc[idx-lookback:idx]
    swing_high = lookback_data['high'].max()
    
    # Check if price broke the swing high
    broke_high = current_bar['high'] > swing_high
    
    # Check for reversal pattern - either:
    # 1. Failed to close above the swing high (bearish rejection)
    # 2. Closed above but then reversed strongly (next bar shows strong selling)
    failed_to_close_above = current_bar['close'] < swing_high
    
    # Check for significant wick (at least 30% of total range) indicating rejection
    wick_size = current_bar['high'] - max(current_bar['open'], current_bar['close'])
    total_range = current_bar['high'] - current_bar['low']
    significant_wick = total_range > 0 and (wick_size / total_range) >= 0.3
    
    return broke_high and (failed_to_close_above or significant_wick)


def detect_liquidity_sweep_down(df: pd.DataFrame, idx: int, lookback: int = 10) -> bool:
    """
    Detect liquidity sweep down pattern at the given index.
    
    Liquidity Sweep Down: Price breaks prior swing low (capturing sell-side liquidity)
    but then reverses direction. This can be either:
    1. Price momentarily surpasses the lows but fails to close below them, then reverses
    2. Price closes below the lows but quickly reverses with strong buying momentum
    
    Args:
        df: DataFrame with OHLC data
        idx: Current index to check
        lookback: Number of bars to look back for swing low
        
    Returns:
        True if liquidity sweep down pattern is detected, False otherwise
    """
    if idx < lookback + 1:
        return False
    
    # Current and previous bars
    current_bar = df.iloc[idx]
    prev_bar = df.iloc[idx-1]
    
    # Find the lowest low in the lookback period (excluding current bar)
    lookback_data = df.iloc[idx-lookback:idx]
    swing_low = lookback_data['low'].min()
    
    # Check if price broke the swing low
    broke_low = current_bar['low'] < swing_low
    
    # Check for reversal pattern - either:
    # 1. Failed to close below the swing low (bullish rejection)
    # 2. Closed below but then reversed strongly (next bar shows strong buying)
    failed_to_close_below = current_bar['close'] > swing_low
    
    # Check for significant wick (at least 30% of total range) indicating rejection
    wick_size = min(current_bar['open'], current_bar['close']) - current_bar['low']
    total_range = current_bar['high'] - current_bar['low']
    significant_wick = total_range > 0 and (wick_size / total_range) >= 0.3
    
    return broke_low and (failed_to_close_below or significant_wick)


def detect_displacement_up(df: pd.DataFrame, idx: int, atr_threshold: float = 1.2) -> bool:
    """
    Detect upward displacement pattern at the given index.
    
    Displacement Up: Large body movement > atr_threshold * ATR
    
    Args:
        df: DataFrame with OHLC data
        idx: Current index to check
        atr_threshold: Threshold for displacement as a multiple of ATR
        
    Returns:
        True if upward displacement pattern is detected, False otherwise
    """
    if idx < 1:
        return False
    
    # Check if we have enough data for ATR calculation
    if idx < 20:  # Default ATR window is 20
        # For testing purposes, use a smaller window if we don't have enough data
        atr_window = min(idx, 5)  # Use at least 5 bars or all available if less
        if atr_window < 2:
            return False
    else:
        atr_window = 20
    
    # Current and previous bars
    current_bar = df.iloc[idx]
    prev_bar = df.iloc[idx-1]
    
    # Calculate ATR at the current index with dynamic window
    atr_value = atr(df.iloc[:idx+1], window=atr_window).iloc[-1]
    
    # Calculate body size
    body_size = current_bar['close'] - current_bar['open']
    
    # Check if body size is positive and exceeds threshold
    threshold = atr_threshold * atr_value
    
    return body_size > threshold


def detect_displacement_down(df: pd.DataFrame, idx: int, atr_threshold: float = 1.2) -> bool:
    """
    Detect downward displacement pattern at the given index.
    
    Displacement Down: Large body movement > atr_threshold * ATR (downward)
    
    Args:
        df: DataFrame with OHLC data
        idx: Current index to check
        atr_threshold: Threshold for displacement as a multiple of ATR
        
    Returns:
        True if downward displacement pattern is detected, False otherwise
    """
    if idx < 1:
        return False
    
    # Check if we have enough data for ATR calculation
    if idx < 20:  # Default ATR window is 20
        # For testing purposes, use a smaller window if we don't have enough data
        atr_window = min(idx, 5)  # Use at least 5 bars or all available if less
        if atr_window < 2:
            return False
    else:
        atr_window = 20
    
    # Current and previous bars
    current_bar = df.iloc[idx]
    prev_bar = df.iloc[idx-1]
    
    # Calculate ATR at the current index with dynamic window
    atr_value = atr(df.iloc[:idx+1], window=atr_window).iloc[-1]
    
    # Calculate body size (negative for downward movement)
    body_size = current_bar['open'] - current_bar['close']
    
    # Check if body size is positive and exceeds threshold
    threshold = atr_threshold * atr_value
    
    return body_size > threshold


def calculate_equilibrium_distance(df: pd.DataFrame, idx: int, lookback: int = 20) -> float:
    """
    Calculate distance to equilibrium (50% level of recent swing) at the given index.
    
    Args:
        df: DataFrame with OHLC data
        idx: Current index to check
        lookback: Number of bars to look back for swing high/low
        
    Returns:
        Distance to equilibrium as a percentage of the swing range
    """
    if idx < lookback:
        return 0.0
    
    # Get data for the lookback period
    lookback_data = df.iloc[idx-lookback:idx+1]
    
    # Find swing high and low
    swing_high = lookback_data['high'].max()
    swing_low = lookback_data['low'].min()
    
    # Calculate equilibrium (50% level)
    equilibrium = (swing_high + swing_low) / 2
    
    # Current close price
    current_close = df.iloc[idx]['close']
    
    # Calculate distance to equilibrium as a percentage of the swing range
    swing_range = swing_high - swing_low
    if swing_range > 0:
        distance_pct = ((current_close - equilibrium) / swing_range) * 100
    else:
        distance_pct = 0.0
    
    return distance_pct


def is_in_killzone(ts: pd.Timestamp, killzone_ranges: Dict[str, list]) -> Dict[str, bool]:
    """
    Check if timestamp is in any of the specified killzones.
    
    Args:
        ts: Timestamp to check
        killzone_ranges: Dictionary with killzone names and time ranges
        
    Returns:
        Dictionary with killzone names as keys and boolean values indicating
        if the timestamp is in each killzone
    """
    result = {}
    
    # Extract time from timestamp
    ts_time = ts.time()
    
    for zone_name, time_range in killzone_ranges.items():
        # Parse time strings
        start_time = time.fromisoformat(time_range[0])
        end_time = time.fromisoformat(time_range[1])
        
        # Check if timestamp time is within the killzone range
        result[f'killzone_{zone_name}'] = start_time <= ts_time <= end_time
    
    return result


def compute_ict_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute ICT (Inner Circle Trader) features.
    
    Args:
        df: DataFrame with OHLCV data
        cfg: Configuration dictionary with ICT parameters
        
    Returns:
        DataFrame with ICT features
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Get configuration parameters
    fvg_min_size_atr = cfg.get('fvg_min_size_atr', 0.25)
    displacement_body_atr = cfg.get('displacement_body_atr', 1.2)
    killzone_ranges = cfg.get('time_of_day', {}).get('killzones', {
        'ny_open': ["09:30", "11:30"],
        'lunch': ["12:00", "13:30"],
        'pm_drive': ["13:30", "16:00"]
    })
    
    # Initialize feature columns
    result_df['ict_fvg_up'] = False
    result_df['ict_fvg_down'] = False
    result_df['ict_liquidity_sweep_up'] = False
    result_df['ict_liquidity_sweep_down'] = False
    result_df['ict_displacement_up'] = False
    result_df['ict_displacement_down'] = False
    result_df['ict_dist_to_eq'] = 0.0
    
    # Initialize killzone columns
    for zone_name in killzone_ranges.keys():
        result_df[f'ict_killzone_{zone_name}'] = False
    
    # Compute features for each row
    for pos_idx in range(len(df)):
        # Get the actual index label
        idx_label = df.index[pos_idx]
        
        # FVG detection
        result_df.at[idx_label, 'ict_fvg_up'] = detect_fvg_up(df, pos_idx, fvg_min_size_atr)
        result_df.at[idx_label, 'ict_fvg_down'] = detect_fvg_down(df, pos_idx, fvg_min_size_atr)
        
        # Liquidity sweep detection
        result_df.at[idx_label, 'ict_liquidity_sweep_up'] = detect_liquidity_sweep_up(df, pos_idx)
        result_df.at[idx_label, 'ict_liquidity_sweep_down'] = detect_liquidity_sweep_down(df, pos_idx)
        
        # Displacement detection
        result_df.at[idx_label, 'ict_displacement_up'] = detect_displacement_up(df, pos_idx, displacement_body_atr)
        result_df.at[idx_label, 'ict_displacement_down'] = detect_displacement_down(df, pos_idx, displacement_body_atr)
        
        # Equilibrium distance
        result_df.at[idx_label, 'ict_dist_to_eq'] = calculate_equilibrium_distance(df, pos_idx)
        
        # Killzone detection
        if 'timestamp' in df.columns:
            killzone_flags = is_in_killzone(df.iloc[pos_idx]['timestamp'], killzone_ranges)
            for zone_name, is_in_zone in killzone_flags.items():
                result_df.at[idx_label, f'ict_{zone_name}'] = is_in_zone
        else:
            # Use index as timestamp if no timestamp column
            killzone_flags = is_in_killzone(idx_label, killzone_ranges)
            for zone_name, is_in_zone in killzone_flags.items():
                result_df.at[idx_label, f'ict_{zone_name}'] = is_in_zone
    
    return result_df


# Legacy function for backward compatibility
def compute_ict_events(df):
    """
    Legacy function for backward compatibility.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with ICT features using default configuration
    """
    default_cfg = {
        'fvg_min_size_atr': 0.25,
        'displacement_body_atr': 1.2,
        'time_of_day': {
            'killzones': {
                'ny_open': ["09:30", "11:30"],
                'lunch': ["12:00", "13:30"],
                'pm_drive': ["13:30", "16:00"]
            }
        }
    }
    return compute_ict_features(df, default_cfg)
