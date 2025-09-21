"""Volume Profile features (POC, VAH/VAL, HVN/LVN, distances)."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
from .ta_basic import atr
from ..data.calendar import is_rth
from numba import jit
import time

# Configure logger
logger = logging.getLogger(__name__)

@jit(nopython=True)
def _calculate_volume_profile_numba(
    lows: np.ndarray,
    highs: np.ndarray,
    volumes: np.ndarray,
    low_min: float,
    high_max: float,
    bin_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-optimized volume profile calculation."""
    price_bins = np.arange(low_min, high_max + bin_size, bin_size)
    volume_per_bin = np.zeros_like(price_bins, dtype=np.float64)

    for i in range(len(lows)):
        start_bin_idx = int((lows[i] - low_min) / bin_size)
        end_bin_idx = int((highs[i] - low_min) / bin_size)

        if end_bin_idx > start_bin_idx:
            volume_per_bin[start_bin_idx:end_bin_idx] += volumes[i] / (end_bin_idx - start_bin_idx)

    return price_bins, volume_per_bin

def calculate_volume_profile(bars: pd.DataFrame, bin_size: float) -> pd.DataFrame:
    """
    Calculate volume profile for a set of bars using a vectorized approach.
    
    Args:
        bars: DataFrame with OHLCV data
        bin_size: Size of each price bin
        
    Returns:
        DataFrame with price bins and volume counts
    """
    if bars.empty:
        return pd.DataFrame(columns=['price', 'volume'])

    low_min = bars['low'].min()
    high_max = bars['high'].max()

    if pd.isna(low_min) or pd.isna(high_max):
        return pd.DataFrame(columns=['price', 'volume'])

    price_bins, volume_per_bin = _calculate_volume_profile_numba(
        bars['low'].values,
        bars['high'].values,
        bars['volume'].values,
        low_min,
        high_max,
        bin_size,
    )

    # Create the profile DataFrame
    profile = pd.DataFrame({
        'price': price_bins,
        'volume': volume_per_bin
    })

    return profile


def calculate_value_area(profile: pd.DataFrame, value_area_pct: float) -> Tuple[float, float]:
    """
    Calculate value area boundaries.
    
    Args:
        profile: DataFrame with price and volume data
        value_area_pct: Percentage of volume to include in value area (e.g., 0.70 for 70%)
        
    Returns:
        Tuple of (value_area_low, value_area_high)
    """
    if profile.empty or profile['volume'].sum() == 0:
        return (np.nan, np.nan)
    
    # Sort by price
    profile = profile.sort_values('price').copy()
    
    # Find POC
    poc_price = profile.loc[profile['volume'].idxmax()]['price']
    
    # Calculate cumulative volume
    profile['cum_volume'] = profile['volume'].cumsum()
    total_volume = profile['volume'].sum()
    
    # Find value area
    target_volume = total_volume * value_area_pct
    
    # Find the row where cumulative volume exceeds half of the target volume
    half_target_volume = target_volume / 2
    
    # Find the index of the POC
    poc_index = profile[profile['price'] == poc_price].index[0]
    
    # Find the indices of the value area
    try:
        upper_bound_index = profile[profile['cum_volume'] >= profile.loc[poc_index, 'cum_volume'] + half_target_volume].index[0]
    except IndexError:
        upper_bound_index = profile.index[-1]
        
    try:
        lower_bound_index = profile[profile['cum_volume'] <= profile.loc[poc_index, 'cum_volume'] - half_target_volume].index[-1]
    except IndexError:
        lower_bound_index = profile.index[0]
        
    value_area_low = profile.loc[lower_bound_index, 'price']
    value_area_high = profile.loc[upper_bound_index, 'price']
    
    return (value_area_low, value_area_high)

def find_poc(profile: pd.DataFrame) -> float:
    """
    Find point of control (price level with highest volume).

    Args:
        profile: DataFrame with price and volume data

    Returns:
        Price level with highest volume
    """
    if profile.empty or profile['volume'].sum() == 0:
        return np.nan

    # Find price with maximum volume (return scalar float)
    poc_idx = profile['volume'].idxmax()
    poc_price = float(profile.loc[poc_idx, 'price'])  # type: ignore
    return poc_price

def find_hvn_lvn(profile: pd.DataFrame, threshold_pct: float = 0.2) -> Tuple[List[float], List[float]]:
    """
    Find high and low volume nodes.
    
    Args:
        profile: DataFrame with price and volume data
        threshold_pct: Threshold for identifying HVN/LVN as percentage of POC volume
        
    Returns:
        Tuple of (hvn_prices, lvn_prices) - lists of price levels
    """
    start_time = time.time()
    if profile.empty or profile['volume'].sum() == 0:
        return ([], [])
    
    # Find POC volume
    poc_volume = profile['volume'].max()
    
    # Calculate threshold volumes
    hvn_threshold = poc_volume * (1 - threshold_pct)
    lvn_threshold = poc_volume * threshold_pct
    
    # Find HVN (high volume nodes) - prices with volume close to POC
    hvn_prices = profile.loc[profile['volume'] >= hvn_threshold, 'price'].tolist()
    
    # Find LVN (low volume nodes) - prices with low volume
    lvn_prices = profile.loc[profile['volume'] <= lvn_threshold, 'price'].tolist()
    
    end_time = time.time()
    logger.info(f"        find_hvn_lvn took {end_time - start_time:.4f} seconds")
    return (hvn_prices, lvn_prices)


def find_poc(profile: pd.DataFrame) -> float:
    """
    Find point of control (price level with highest volume).

    Args:
        profile: DataFrame with price and volume data

    Returns:
        Price level with highest volume
    """
    if profile.empty or profile['volume'].sum() == 0:
        return np.nan

    # Find price with maximum volume (return scalar float)
    poc_idx = profile['volume'].idxmax()
    poc_price = float(profile.loc[poc_idx, 'price'])  # type: ignore
    return poc_price


def find_hvn_lvn(profile: pd.DataFrame, threshold_pct: float = 0.2) -> Tuple[List[float], List[float]]:
    """
    Find high and low volume nodes.
    
    Args:
        profile: DataFrame with price and volume data
        threshold_pct: Threshold for identifying HVN/LVN as percentage of POC volume
        
    Returns:
        Tuple of (hvn_prices, lvn_prices) - lists of price levels
    """
    if profile.empty or profile['volume'].sum() == 0:
        return ([], [])
    
    # Find POC volume
    poc_volume = profile['volume'].max()
    
    # Calculate threshold volumes
    hvn_threshold = poc_volume * (1 - threshold_pct)
    lvn_threshold = poc_volume * threshold_pct
    
    # Find HVN (high volume nodes) - prices with volume close to POC
    hvn_prices = profile.loc[profile['volume'] >= hvn_threshold, 'price'].tolist()
    
    # Find LVN (low volume nodes) - prices with low volume
    lvn_prices = profile.loc[profile['volume'] <= lvn_threshold, 'price'].tolist()
    
    return (hvn_prices, lvn_prices)


import time

# Configure logger
logger = logging.getLogger(__name__)


def compute_volume_profile_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Enhanced volume profile features computation with NaN reduction.
    
    Key improvements:
    1. Handle insufficient rolling sessions gracefully
    2. Use smaller ATR window for volume profile to reduce warm-up period
    3. Implement forward-fill for POC values within sessions
    4. Add fallback calculations for edge cases
    
    Args:
        df: DataFrame with OHLCV data
        cfg: Configuration dictionary with volume profile parameters
        
    Returns:
        DataFrame with volume profile features
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Get configuration parameters with safer defaults
    bin_size = cfg.get('bin_size', 0.05)
    value_area_pct = cfg.get('value_area', 0.70)
    rolling_sessions = cfg.get('rolling_sessions', 20)
    hvn_lvn_threshold = cfg.get('hvn_lvn_threshold', 0.2)
    hvn_lvn_atr_distance = cfg.get('hvn_lvn_atr_distance', 0.5)
    
    # Use smaller ATR window for volume profile to reduce NaN values
    # Default to 10 instead of 20 to cut warm-up period in half
    atr_window = min(cfg.get('atr_window', cfg.get('features', {}).get('atr_window', 20)), 10)
    
    # Initialize feature columns
    result_df['vp_poc'] = np.nan
    result_df['vp_vah'] = np.nan
    result_df['vp_val'] = np.nan
    result_df['vp_dist_to_poc_atr'] = np.nan
    result_df['vp_inside_value'] = 0
    result_df['vp_hvn_near'] = 0
    result_df['vp_lvn_near'] = 0
    result_df['vp_poc_shift_dir'] = 0
    
    # Filter for RTH only
    try:
        rth_mask = [is_rth(idx) for idx in df.index]
        rth_df = df[rth_mask].copy()
    except Exception as e:
        logger.warning(f"Error filtering RTH data: {e}")
        rth_df = df.copy()

    if rth_df.empty:
        logger.warning("No RTH data found, returning empty volume profile features")
        return result_df

    # Calculate ATR for distance normalization
    atr_values = atr(rth_df, window=atr_window)

    # Group by date to process each session separately
    rth_df['date'] = pd.to_datetime(rth_df.index).date
    
    # Sort by date
    rth_df = rth_df.sort_index()
    
    # Get unique dates
    unique_dates = sorted(rth_df['date'].unique())
    
    # Adjust rolling sessions if insufficient data
    available_sessions = len(unique_dates)
    if available_sessions < rolling_sessions:
        logger.warning(f"Insufficient dates for rolling sessions: need {rolling_sessions}, have {available_sessions}")
        # Use available sessions but ensure at least 1
        adjusted_rolling_sessions = max(1, available_sessions - 1)
        logger.info(f"Adjusted rolling sessions to: {adjusted_rolling_sessions}")
    else:
        adjusted_rolling_sessions = rolling_sessions

    def process_session(session_data):
        date = session_data['date'].iloc[0]
        i = unique_dates.index(date)

        # Get rolling window of previous sessions
        start_idx = max(0, i - adjusted_rolling_sessions + 1)
        previous_dates = unique_dates[start_idx:i]
        
        # Combine previous sessions and current session for rolling calculation
        if previous_dates:
            previous_sessions = rth_df[rth_df['date'].isin(previous_dates)]
            rolling_bars = pd.concat([previous_sessions, session_data])
        else:
            rolling_bars = session_data

        # Calculate volume profile for rolling window
        profile = calculate_volume_profile_safe(rolling_bars, bin_size)

        if profile.empty:
            return None

        # Find POC
        poc = find_poc_safe(profile)

        # Calculate value area
        val, vah = calculate_value_area_safe(profile, value_area_pct)

        # Find HVN/LVN
        hvn_prices, lvn_prices = find_hvn_lvn_safe(profile, hvn_lvn_threshold)

        # Update features for each bar in current session
        session_features = session_data.copy()
        session_features['vp_poc'] = poc
        session_features['vp_vah'] = vah
        session_features['vp_val'] = val

        # Calculate distance to POC in ATR units
        current_atr = atr_values.loc[session_features.index]
        dist_to_poc = np.abs(session_features['close'] - poc)
        dist_atr = np.minimum(dist_to_poc / current_atr, 10.0)
        session_features['vp_dist_to_poc_atr'] = dist_atr

        # Check if price is inside value area
        session_features['vp_inside_value'] = np.where((session_features['close'] >= val) & (session_features['close'] <= vah), 1, 0)

        # Check if HVN/LVN are nearby
        atr_distance = hvn_lvn_atr_distance * current_atr
        session_features['vp_hvn_near'] = 0
        session_features['vp_lvn_near'] = 0
        for price in hvn_prices:
            session_features['vp_hvn_near'] = np.where(np.abs(session_features['close'] - price) <= atr_distance, 1, session_features['vp_hvn_near'])
        for price in lvn_prices:
            session_features['vp_lvn_near'] = np.where(np.abs(session_features['close'] - price) <= atr_distance, 1, session_features['vp_lvn_near'])

        # Calculate POC shift direction
        if i > 0:
            prev_date = unique_dates[i-1]
            prev_session = rth_df[rth_df['date'] == prev_date]
            if not prev_session.empty:
                prev_profile = calculate_volume_profile_safe(prev_session, bin_size)
                if not prev_profile.empty:
                    prev_poc = find_poc_safe(prev_profile)
                    if not np.isnan(poc) and not np.isnan(prev_poc):
                        if poc > prev_poc:
                            session_features['vp_poc_shift_dir'] = 1
                        elif poc < prev_poc:
                            session_features['vp_poc_shift_dir'] = -1
                        else:
                            session_features['vp_poc_shift_dir'] = 0
        return session_features

    # Process each session
    processed_sessions = rth_df.groupby('date').apply(process_session)

    # Update the result_df with the computed features
    result_df.update(processed_sessions)

    # Apply forward-fill within each date to reduce NaN values
    for date in unique_dates:
        date_mask = [idx.date() == date for idx in result_df.index]
        date_indices = result_df[date_mask].index
        
        if len(date_indices) > 1:
            # Forward-fill POC, VAH, VAL within the same session
            result_df.loc[date_indices, 'vp_poc'] = result_df.loc[date_indices, 'vp_poc'].ffill()
            result_df.loc[date_indices, 'vp_vah'] = result_df.loc[date_indices, 'vp_vah'].ffill()
            result_df.loc[date_indices, 'vp_val'] = result_df.loc[date_indices, 'vp_val'].ffill()
    
    return result_df



# Legacy function for backward compatibility
def rolling_volume_profile(df, sessions: int, bin_size: float, value_area: float):
    """
    Legacy function for backward compatibility.
    
    Args:
        df: DataFrame with OHLCV data
        sessions: Number of rolling sessions
        bin_size: Size of each price bin
        value_area: Percentage of volume to include in value area
        
    Returns:
        DataFrame with volume profile features
    """
    cfg = {
        'bin_size': bin_size,
        'value_area': value_area,
        'rolling_sessions': sessions,
        'atr_window': 20,
        'hvn_lvn_threshold': 0.2,
        'hvn_lvn_atr_distance': 0.5
    }
    return compute_volume_profile_features(df, cfg)


def calculate_volume_profile_safe(bars: pd.DataFrame, bin_size: float) -> pd.DataFrame:
    """Safe volume profile calculation with error handling."""
    try:
        if bars.empty:
            return pd.DataFrame(columns=['price', 'volume'])
        
        # Determine price range
        low_min = bars['low'].min()
        high_max = bars['high'].max()
        
        # Check for valid price range
        if np.isnan(low_min) or np.isnan(high_max) or low_min >= high_max:
            return pd.DataFrame(columns=['price', 'volume'])
        
        # Create price bins
        price_bins = np.arange(low_min, high_max + bin_size, bin_size)
        
        # Initialize volume profile
        profile = pd.DataFrame({
            'price': price_bins[:-1],  # Use left edge of bins
            'volume': 0.0
        })
        
        # Distribute volume across price range for each bar
        for _, bar in bars.iterrows():
            bar_low = bar['low']
            bar_high = bar['high']
            bar_volume = bar['volume']
            
            # Skip if volume is zero or negative
            if bar_volume <= 0:
                continue
            
            # Find bins that overlap with this bar's range
            mask = (profile['price'] >= bar_low) & (profile['price'] < bar_high)
            
            if mask.any():
                # Distribute volume proportionally based on overlap
                overlapping_bins = profile.loc[mask]
                
                # Calculate overlap for each bin
                bin_highs = overlapping_bins['price'] + bin_size
                bin_lows = overlapping_bins['price']
                
                overlap_lows = np.maximum(bin_lows, bar_low)
                overlap_highs = np.minimum(bin_highs, bar_high)
                
                overlaps = overlap_highs - overlap_lows
                total_overlap = overlaps.sum()
                
                if total_overlap > 0:
                    # Distribute volume based on overlap proportion
                    volume_distribution = (overlaps / total_overlap) * bar_volume
                    profile.loc[mask, 'volume'] += volume_distribution
        
        return profile
    except Exception as e:
        logger.warning(f"Error in volume profile calculation: {e}")
        return pd.DataFrame(columns=['price', 'volume'])


def find_poc_safe(profile: pd.DataFrame) -> float:
    """Safe POC calculation with error handling."""
    try:
        if profile.empty or profile['volume'].sum() == 0:
            return np.nan

        # Find price with maximum volume
        poc_idx = profile['volume'].idxmax()
        poc_row = profile.loc[poc_idx]
        return float(poc_row['price'])
    except Exception as e:
        logger.warning(f"Error finding POC: {e}")
        return np.nan


def calculate_value_area_safe(profile: pd.DataFrame, value_area_pct: float) -> Tuple[float, float]:
    """Safe value area calculation with error handling."""
    try:
        if profile.empty or profile['volume'].sum() == 0:
            return (np.nan, np.nan)
        
        # Sort by volume descending
        sorted_profile = profile.sort_values('volume', ascending=False).copy()
        
        # Calculate cumulative volume
        sorted_profile['cum_volume'] = sorted_profile['volume'].cumsum()
        total_volume = sorted_profile['volume'].sum()
        sorted_profile['cum_volume_pct'] = sorted_profile['cum_volume'] / total_volume
        
        # Find POC (price with highest volume)
        poc_price = sorted_profile.iloc[0]['price']
        
        # Find value area boundaries
        target_volume_pct = value_area_pct
        
        # Start from POC and expand outward until we reach target volume percentage
        included_prices = [poc_price]
        included_volume = sorted_profile.iloc[0]['volume']
        
        # Initialize pointers for expansion
        above_ptr = 0
        below_ptr = 0
        
        # Get prices above and below POC
        prices_above = sorted_profile[sorted_profile['price'] > poc_price].sort_values('price')
        prices_below = sorted_profile[sorted_profile['price'] < poc_price].sort_values('price', ascending=False)
        
        # Expand outward from POC until we reach target volume
        while (included_volume / total_volume) < target_volume_pct:
            # Check if we have more prices to include
            can_add_above = above_ptr < len(prices_above)
            can_add_below = below_ptr < len(prices_below)
            
            if not can_add_above and not can_add_below:
                break
            
            # Determine which side to add based on volume
            above_volume = prices_above.iloc[above_ptr]['volume'] if can_add_above else 0
            below_volume = prices_below.iloc[below_ptr]['volume'] if can_add_below else 0
            
            if (can_add_above and above_volume >= below_volume) or not can_add_below:
                # Add price above
                included_volume += above_volume
                included_prices.append(prices_above.iloc[above_ptr]['price'])
                above_ptr += 1
            else:
                # Add price below
                included_volume += below_volume
                included_prices.append(prices_below.iloc[below_ptr]['price'])
                below_ptr += 1
        
        # Calculate value area boundaries
        value_area_low = min(included_prices)
        value_area_high = max(included_prices)
        
        return (value_area_low, value_area_high)
    except Exception as e:
        logger.warning(f"Error calculating value area: {e}")
        return (np.nan, np.nan)


def find_hvn_lvn_safe(profile: pd.DataFrame, threshold_pct: float = 0.2) -> Tuple[List[float], List[float]]:
    """Safe HVN/LVN calculation with error handling."""
    try:
        if profile.empty or profile['volume'].sum() == 0:
            return ([], [])
        
        # Find POC volume
        poc_volume = profile['volume'].max()
        
        # Calculate threshold volumes
        hvn_threshold = poc_volume * (1 - threshold_pct)
        lvn_threshold = poc_volume * threshold_pct
        
        # Find HVN (high volume nodes) - prices with volume close to POC
        hvn_prices = profile.loc[profile['volume'] >= hvn_threshold, 'price'].tolist()
        
        # Find LVN (low volume nodes) - prices with low volume
        lvn_prices = profile.loc[profile['volume'] <= lvn_threshold, 'price'].tolist()
        
        return (hvn_prices, lvn_prices)
    except Exception as e:
        logger.warning(f"Error finding HVN/LVN: {e}")
        return ([], [])
