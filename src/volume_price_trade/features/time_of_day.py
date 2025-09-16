"""Minute-of-day cyclical encodings + killzone flags."""

import pandas as pd
import numpy as np
import math
from typing import Dict, Any, Tuple
from datetime import time
from ..data.calendar import is_rth


def minute_of_day(ts: pd.Timestamp) -> int:
    """
    Extract minute of day from timestamp (0-1439).
    
    Args:
        ts: Timestamp to extract minute of day from
        
    Returns:
        Minute of day (0-1439)
    """
    return ts.hour * 60 + ts.minute


def cyclical_encoding(minute: int) -> Tuple[float, float]:
    """
    Convert minute to sin/cos encoding.
    
    Args:
        minute: Minute of day (0-1439)
        
    Returns:
        Tuple of (sin_minute, cos_minute) values
    """
    # Normalize to [0, 2π] range
    normalized = 2 * math.pi * minute / 1440
    return math.sin(normalized), math.cos(normalized)


def is_in_session(ts: pd.Timestamp, session_ranges: Dict[str, list]) -> Dict[str, bool]:
    """
    Check if timestamp is in session.
    
    Args:
        ts: Timestamp to check
        session_ranges: Dictionary with session names and time ranges
        
    Returns:
        Dictionary with session names as keys and boolean values indicating if in session
    """
    result = {}
    current_time = ts.time()
    
    for session_name, time_range in session_ranges.items():
        start_time_str, end_time_str = time_range
        start_hour, start_minute = map(int, start_time_str.split(":"))
        end_hour, end_minute = map(int, end_time_str.split(":"))
        
        start_time = time(start_hour, start_minute)
        end_time = time(end_hour, end_minute)
        
        # Check if current time is within session (end time is exclusive)
        result[session_name] = start_time <= current_time < end_time
        
    return result


def time_to_market_close(ts: pd.Timestamp, rth_end: str = "16:00") -> int:
    """
    Calculate minutes until market close (negative after close).
    
    Args:
        ts: Timestamp to calculate from
        rth_end: Market close time in "HH:MM" format
        
    Returns:
        Minutes until market close (negative after close)
    """
    end_hour, end_minute = map(int, rth_end.split(":"))
    end_time = time(end_hour, end_minute)
    current_time = ts.time()
    
    # Calculate minutes until close
    minutes_until_close = (end_hour - ts.hour) * 60 + (end_minute - ts.minute)
    
    # If current time is after close, return negative value
    if current_time >= end_time:
        return minutes_until_close
    
    return minutes_until_close


def time_since_market_open(ts: pd.Timestamp, rth_start: str = "09:30") -> int:
    """
    Calculate minutes since market open (negative before open).
    
    Args:
        ts: Timestamp to calculate from
        rth_start: Market open time in "HH:MM" format
        
    Returns:
        Minutes since market open (negative before open)
    """
    start_hour, start_minute = map(int, rth_start.split(":"))
    start_time = time(start_hour, start_minute)
    current_time = ts.time()
    
    # Calculate minutes since open
    minutes_since_open = (ts.hour - start_hour) * 60 + (ts.minute - start_minute)
    
    # If current time is before open, return negative value
    if current_time < start_time:
        return minutes_since_open
    
    return minutes_since_open


def compute_time_of_day_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute Time of Day (ToD) features.
    
    Args:
        df: DataFrame with timestamp data
        cfg: Configuration dictionary with time parameters
        
    Returns:
        DataFrame with ToD features
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Extract configuration values
    session_ranges = cfg.get("time_of_day", {}).get("killzones", {
        "ny_open": ["09:30", "11:30"],
        "lunch": ["12:00", "13:30"],
        "pm_drive": ["13:30", "16:00"]
    })
    
    rth_start = cfg.get("sessions", {}).get("rth_start", "09:30")
    rth_end = cfg.get("sessions", {}).get("rth_end", "16:00")
    
    # Ensure timestamps are in the correct timezone
    timestamps = pd.to_datetime(result_df.index)
    
    # Initialize feature columns
    result_df["minute_of_day"] = 0
    result_df["sin_minute"] = 0.0
    result_df["cos_minute"] = 0.0
    result_df["hour_of_day"] = 0
    result_df["minute_of_hour"] = 0
    result_df["is_rth"] = False
    result_df["time_to_close"] = 0
    result_df["time_since_open"] = 0
    
    # Add session flags
    for session_name in session_ranges.keys():
        result_df[f"is_{session_name}"] = False
    
    # Compute features for each timestamp
    for i, ts in enumerate(timestamps):
        # 1. Minute of day
        mod = minute_of_day(ts)
        result_df.iloc[i, result_df.columns.get_loc("minute_of_day")] = mod
        
        # 2. Cyclical encoding
        sin_min, cos_min = cyclical_encoding(mod)
        result_df.iloc[i, result_df.columns.get_loc("sin_minute")] = sin_min
        result_df.iloc[i, result_df.columns.get_loc("cos_minute")] = cos_min
        
        # 3. Hour and minute components
        result_df.iloc[i, result_df.columns.get_loc("hour_of_day")] = ts.hour
        result_df.iloc[i, result_df.columns.get_loc("minute_of_hour")] = ts.minute
        
        # 4. RTH flag
        result_df.iloc[i, result_df.columns.get_loc("is_rth")] = is_rth(ts)
        
        # 5. Time to close and since open
        result_df.iloc[i, result_df.columns.get_loc("time_to_close")] = time_to_market_close(ts, rth_end)
        result_df.iloc[i, result_df.columns.get_loc("time_since_open")] = time_since_market_open(ts, rth_start)
        
        # 6. Session flags
        session_flags = is_in_session(ts, session_ranges)
        for session_name, in_session in session_flags.items():
            result_df.iloc[i, result_df.columns.get_loc(f"is_{session_name}")] = in_session
    
    return result_df
