"""Triple-barrier style labels constrained to 5–240m and EOD flat."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union, cast
from ..data.calendar import next_session_close
from ..features.ta_basic import atr


def triple_barrier_labels(
    df: pd.DataFrame,
    horizons_min: int,
    atr_mult_sl: float,
    r_mult_tp: float,
    eod_flat: bool,
    mode: str = "fast"
) -> pd.DataFrame:
    """
    Generate triple-barrier labels for each bar in the DataFrame.

    Args:
        df: DataFrame with OHLCV data and datetime index
        horizons_min: Maximum horizon in minutes (5-240)
        atr_mult_sl: Multiplier for ATR to calculate stop loss
        r_mult_tp: R-multiple for take profit (Target = Stop * r_mult_tp)
        eod_flat: Whether to apply EOD flat position (close all positions at EOD)
        mode: "fast" (vectorized) or "precise" (loop-based exact)

    Returns:
        DataFrame with original data plus label columns:
        - y_class: Label in {up, down, hold}
        - horizon_minutes: Actual horizon used (may be less than horizons_min due to EOD)
        - event_end_time: Timestamp when the event (TP/SL/timeout) occurred
    """
    if mode == "fast":
        return _triple_barrier_labels_fast(df, horizons_min, atr_mult_sl, r_mult_tp)
    elif mode == "precise":
        return _triple_barrier_labels_precise(df, horizons_min, atr_mult_sl, r_mult_tp, eod_flat)
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'fast' or 'precise'.")


def _triple_barrier_labels_fast(
    df: pd.DataFrame,
    horizons_min: int,
    atr_mult_sl: float,
    r_mult_tp: float
) -> pd.DataFrame:
    """
    Performance-optimized implementation:
    - Uses vectorized rolling max/min on future windows to avoid O(N*H) Python loops
    - Approximates event_end_time as current_time + horizons_min minutes
      (sufficient for purging/embargo and dramatically faster)
    - Keeps ATR-based barrier sizing identical

    Args:
        df: DataFrame with OHLCV data and datetime index
        horizons_min: Maximum horizon in minutes (5-240)
        atr_mult_sl: Multiplier for ATR to calculate stop loss
        r_mult_tp: R-multiple for take profit (Target = Stop * r_mult_tp)

    Returns:
        DataFrame with original data plus:
        - y_class: {'up','down','hold'}
        - horizon_minutes: int (set to horizons_min where ATR is available, else 0)
        - event_end_time: timestamp approximated as index + horizons_min minutes
    """
    # Defensive copy and datetime index
    result_df = df.copy()
    result_df.index = pd.to_datetime(result_df.index)

    # Initialize output columns
    result_df['y_class'] = 'hold'
    result_df['horizon_minutes'] = 0

    # Initialize event_end_time with tz-awareness matching index
    if result_df.index.tz is not None:
        result_df['event_end_time'] = pd.Series(
            [pd.NaT] * len(result_df),
            index=result_df.index,
            dtype=f'datetime64[ns, {result_df.index.tz}]'
        )
    else:
        result_df['event_end_time'] = pd.NaT

    # Compute ATR for barrier sizing
    atr_window = 20
    atr_series = atr(df, atr_window)

    # Barriers
    stop_loss_points = atr_series * atr_mult_sl
    take_profit_points = stop_loss_points * r_mult_tp
    upper_barrier = result_df['close'] + take_profit_points
    lower_barrier = result_df['close'] - stop_loss_points

    # Window in bars (assumes 1-minute bars, consistent with dataset)
    window = max(1, int(horizons_min))

    # Forward-looking extrema excluding the current bar
    # Reverse -> rolling -> reverse -> shift(-1) to exclude current
    future_high_max = result_df['high'][::-1].rolling(window=window, min_periods=1).max()[::-1].shift(-1)
    future_low_min = result_df['low'][::-1].rolling(window=window, min_periods=1).min()[::-1].shift(-1)

    # Determine hits; mask rows where ATR is NaN to avoid spurious labels
    valid_atr = ~atr_series.isna()
    hit_up = (future_high_max >= upper_barrier) & valid_atr
    hit_down = (future_low_min <= lower_barrier) & valid_atr

    # Resolve labels (if both hits occur within window, prefer 'up' deterministically)
    y = np.where(hit_up, 'up', np.where(hit_down, 'down', 'hold'))
    result_df['y_class'] = y

    # Horizon minutes and event_end_time approximation
    result_df.loc[valid_atr, 'horizon_minutes'] = window
    horizon_delta = pd.to_timedelta(window, unit='m')
    # Set end time only where ATR is valid; leave as NaT otherwise
    if result_df.index.tz is not None:
        result_df.loc[valid_atr, 'event_end_time'] = result_df.index[valid_atr] + horizon_delta  # type: ignore
    else:
        result_df.loc[valid_atr, 'event_end_time'] = (result_df.index[valid_atr] + horizon_delta).to_numpy()  # type: ignore

    return result_df


def _triple_barrier_labels_precise(
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
    """
    # Make a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    # Ensure index is DatetimeIndex for mypy
    result_df.index = pd.to_datetime(result_df.index)
    
    # Initialize output columns
    result_df['y_class'] = 'hold'
    result_df['horizon_minutes'] = 0
    # Initialize event_end_time column with proper timezone-aware dtype
    if result_df.index.tz is not None:
        result_df['event_end_time'] = pd.Series([pd.NaT] * len(result_df), index=result_df.index, dtype=f'datetime64[ns, {result_df.index.tz}]')
    else:
        result_df['event_end_time'] = pd.NaT
    
    # Calculate ATR for stop loss calculation
    atr_window = 20  # Default ATR window, can be made configurable
    atr_series = atr(df, atr_window)
    
    # For each row in the DataFrame
    for i, (idx, row) in enumerate(df.iterrows()):
        # Ensure idx is Timestamp
        current_time = pd.Timestamp(idx)  # type: ignore
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
                event_time = pd.Timestamp(future_idx)  # type: ignore
                break  # Exit loop as soon as a barrier is hit
                
            # Check if lower barrier is hit
            if not hit_lower and future_low <= lower_barrier:
                hit_lower = True
                event_time = pd.Timestamp(future_idx)  # type: ignore
                break  # Exit loop as soon as a barrier is hit
        
        # Determine the label based on which barrier was hit
        if hit_upper:
            label = 'up'  # Long position would have hit take profit
        elif hit_lower:
            label = 'down'  # Long position would have hit stop loss
        else:
            label = 'hold'  # Neither barrier was hit (timeout)
            
        # Update the result DataFrame
        idx_ts = pd.Timestamp(idx)  # type: ignore
        result_df.loc[idx_ts, 'y_class'] = label
        result_df.loc[idx_ts, 'horizon_minutes'] = int(actual_horizon.total_seconds() / 60)
        # event_time is already timezone-aware and matches column dtype
        result_df.loc[idx_ts, 'event_end_time'] = event_time
    
    return result_df
    # Defensive copy and datetime index
    result_df = df.copy()
    result_df.index = pd.to_datetime(result_df.index)

    # Initialize output columns
    result_df['y_class'] = 'hold'
    result_df['horizon_minutes'] = 0

    # Initialize event_end_time with tz-awareness matching index
    if result_df.index.tz is not None:
        result_df['event_end_time'] = pd.Series(
            [pd.NaT] * len(result_df),
            index=result_df.index,
            dtype=f'datetime64[ns, {result_df.index.tz}]'
        )
    else:
        result_df['event_end_time'] = pd.NaT

    # Compute ATR for barrier sizing
    atr_window = 20
    atr_series = atr(df, atr_window)

    # Barriers
    stop_loss_points = atr_series * atr_mult_sl
    take_profit_points = stop_loss_points * r_mult_tp
    upper_barrier = result_df['close'] + take_profit_points
    lower_barrier = result_df['close'] - stop_loss_points

    # Window in bars (assumes 1-minute bars, consistent with dataset)
    window = max(1, int(horizons_min))

    # Forward-looking extrema excluding the current bar
    # Reverse -> rolling -> reverse -> shift(-1) to exclude current
    future_high_max = result_df['high'][::-1].rolling(window=window, min_periods=1).max()[::-1].shift(-1)
    future_low_min = result_df['low'][::-1].rolling(window=window, min_periods=1).min()[::-1].shift(-1)

    # Determine hits; mask rows where ATR is NaN to avoid spurious labels
    valid_atr = ~atr_series.isna()
    hit_up = (future_high_max >= upper_barrier) & valid_atr
    hit_down = (future_low_min <= lower_barrier) & valid_atr

    # Resolve labels (if both hits occur within window, prefer 'up' deterministically)
    y = np.where(hit_up, 'up', np.where(hit_down, 'down', 'hold'))
    result_df['y_class'] = y

    # Horizon minutes and event_end_time approximation
    result_df.loc[valid_atr, 'horizon_minutes'] = window
    horizon_delta = pd.to_timedelta(window, unit='m')
    # Set end time only where ATR is valid; leave as NaT otherwise
    if result_df.index.tz is not None:
        result_df.loc[valid_atr, 'event_end_time'] = result_df.index[valid_atr] + horizon_delta  # type: ignore
    else:
        result_df.loc[valid_atr, 'event_end_time'] = (result_df.index[valid_atr] + horizon_delta).to_numpy()  # type: ignore

    return result_df
