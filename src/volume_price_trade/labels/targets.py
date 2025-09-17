"""Triple-barrier style labels constrained to 5–240m and EOD flat."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
from ..data.calendar import next_session_close
from ..features.ta_basic import atr


def triple_barrier_labels(
    df: pd.DataFrame,
    horizons_min: int,
    atr_mult_sl: float,
    r_mult_tp: float,
    eod_flat: bool
) -> pd.DataFrame:
    """
    Generate triple-barrier labels for each bar in the DataFrame.
    
    For each bar, compute forward path up to min(horizon_minutes, EOD):
    - Stop = ATR * atr_mult_sl
    - Target = Stop * take_profit_R
    - Determine which (TP/SL/timeout) occurs first; set y_class in {up, down, hold}
    - Also output horizon_minutes actually used and event_end_time
    
    Args:
        df: DataFrame with OHLCV data and datetime index
        horizons_min: Maximum horizon in minutes (5-240)
        atr_mult_sl: Multiplier for ATR to calculate stop loss
        r_mult_tp: R-multiple for take profit (Target = Stop * r_mult_tp)
        eod_flat: Whether to apply EOD flat position (close all positions at EOD)
        
    Returns:
        DataFrame with original data plus label columns:
        - y_class: Label in {up, down, hold}
        - horizon_minutes: Actual horizon used (may be less than horizons_min due to EOD)
        - event_end_time: Timestamp when the event (TP/SL/timeout) occurred
    """
    # Make a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Initialize output columns
    result_df['y_class'] = 'hold'
    result_df['horizon_minutes'] = 0
    result_df['event_end_time'] = pd.NaT
    
    # Calculate ATR for stop loss calculation
    atr_window = 20  # Default ATR window, can be made configurable
    atr_series = atr(df, atr_window)
    
    # For each row in the DataFrame
    for i, (idx, row) in enumerate(df.iterrows()):
        current_time = idx
        current_close = row['close']
        current_atr = atr_series.iloc[i]
        
        # Skip if ATR is NaN
        if np.isnan(current_atr):
            continue
            
        # Calculate stop loss and take profit levels
        stop_loss_points = current_atr * atr_mult_sl
        take_profit_points = stop_loss_points * r_mult_tp
        
        # Define barriers
        upper_barrier = current_close + take_profit_points  # Long TP
        lower_barrier = current_close - stop_loss_points     # Long SL
        
        # Calculate maximum horizon (EOD or specified horizon)
        max_horizon = pd.Timedelta(minutes=horizons_min)
        
        if eod_flat:
            # Get EOD time
            eod_time = next_session_close(current_time)
            eod_horizon = eod_time - current_time
            
            # Use the minimum of specified horizon and EOD horizon
            actual_horizon = min(max_horizon, eod_horizon)
        else:
            actual_horizon = max_horizon
            
        # Get the end time for this horizon
        end_time = current_time + actual_horizon
        
        # Get future data within the horizon
        future_data = df.loc[current_time:end_time]
        
        # Skip if no future data (e.g., at the end of the dataset)
        if len(future_data) <= 1:
            continue
            
        # Remove the current row from future_data
        future_data = future_data.iloc[1:]
        
        # Initialize variables to track which barrier was hit first
        hit_upper = False
        hit_lower = False
        event_time = end_time  # Default to timeout (end of horizon)
        
        # Check each future bar to see which barrier is hit first
        for future_idx, future_row in future_data.iterrows():
            future_high = future_row['high']
            future_low = future_row['low']
            
            # Check if upper barrier is hit
            if not hit_upper and future_high >= upper_barrier:
                hit_upper = True
                event_time = future_idx
                break  # Exit loop as soon as a barrier is hit
                
            # Check if lower barrier is hit
            if not hit_lower and future_low <= lower_barrier:
                hit_lower = True
                event_time = future_idx
                break  # Exit loop as soon as a barrier is hit
        
        # Determine the label based on which barrier was hit
        if hit_upper:
            label = 'up'  # Long position would have hit take profit
        elif hit_lower:
            label = 'down'  # Long position would have hit stop loss
        else:
            label = 'hold'  # Neither barrier was hit (timeout)
            
        # Update the result DataFrame
        result_df.at[idx, 'y_class'] = label  # type: ignore
        result_df.at[idx, 'horizon_minutes'] = int(actual_horizon.total_seconds() / 60)  # type: ignore
        result_df.at[idx, 'event_end_time'] = event_time  # type: ignore
    
    return result_df
