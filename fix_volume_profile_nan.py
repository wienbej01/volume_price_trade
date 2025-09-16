#!/usr/bin/env python3
"""
Fix for volume profile NaN issues.
This script addresses the high NaN values in volume profile features,
particularly vp_dist_to_poc_atr which has 70.50% NaN values.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from volume_price_trade.features.volume_profile import compute_volume_profile_features
from volume_price_trade.features.ta_basic import atr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def enhanced_compute_volume_profile_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Enhanced volume profile features computation with NaN reduction.
    
    Key improvements:
    1. Handle insufficient rolling sessions gracefully
    2. Use smaller ATR window for volume profile to reduce warm-up period
    3. Implement forward-fill for POC values within sessions
    4. Add fallback calculations for edge cases
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
    
    # Calculate ATR for distance normalization
    atr_values = atr(df, window=atr_window)
    
    # Filter for RTH only
    try:
        from volume_price_trade.data.calendar import is_rth
        rth_mask = [is_rth(idx) for idx in df.index]
        rth_df = df[rth_mask].copy()
    except Exception as e:
        logger.warning(f"Error filtering RTH data: {e}")
        rth_df = df.copy()
    
    if rth_df.empty:
        logger.warning("No RTH data found, returning empty volume profile features")
        return result_df
    
    # Group by date to process each session separately
    rth_df['date'] = rth_df.index.date
    
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
    
    # Process each date
    for i, date in enumerate(unique_dates):
        # Get current session data
        current_session = rth_df[rth_df['date'] == date]
        
        # Get rolling window of previous sessions
        start_idx = max(0, i - adjusted_rolling_sessions + 1)
        previous_dates = unique_dates[start_idx:i]
        
        # Combine previous sessions and current session for rolling calculation
        if previous_dates:
            previous_sessions = rth_df[rth_df['date'].isin(previous_dates)]
            rolling_bars = pd.concat([previous_sessions, current_session])
        else:
            rolling_bars = current_session
        
        # Skip if insufficient data
        if rolling_bars.empty:
            continue
        
        # Calculate volume profile for rolling window
        try:
            profile = calculate_volume_profile_safe(rolling_bars, bin_size)
        except Exception as e:
            logger.warning(f"Error calculating volume profile for date {date}: {e}")
            continue
        
        if profile.empty:
            continue
        
        # Find POC
        poc = find_poc_safe(profile)
        
        # Calculate value area
        val, vah = calculate_value_area_safe(profile, value_area_pct)
        
        # Find HVN/LVN
        hvn_prices, lvn_prices = find_hvn_lvn_safe(profile, hvn_lvn_threshold)
        
        # Update features for each bar in current session
        for idx, row in current_session.iterrows():
            # Get ATR value for this bar
            current_atr = atr_values.loc[idx] if idx in atr_values.index else np.nan
            
            # Set POC, VAH, VAL
            result_df.loc[idx, 'vp_poc'] = poc
            result_df.loc[idx, 'vp_vah'] = vah
            result_df.loc[idx, 'vp_val'] = val
            
            # Calculate distance to POC in ATR units
            if not np.isnan(poc) and not np.isnan(current_atr) and current_atr > 0:
                close_price = row['close']
                dist_to_poc = abs(close_price - poc)
                # Cap the distance to avoid extreme values
                dist_atr = min(dist_to_poc / current_atr, 10.0)
                result_df.loc[idx, 'vp_dist_to_poc_atr'] = dist_atr
            elif not np.isnan(poc):
                # Fallback: use a default ATR value if POC is available but ATR is not
                fallback_atr = atr_values.dropna().mean() if not atr_values.dropna().empty else 1.0
                close_price = row['close']
                dist_to_poc = abs(close_price - poc)
                dist_atr = min(dist_to_poc / fallback_atr, 10.0)
                result_df.loc[idx, 'vp_dist_to_poc_atr'] = dist_atr
            
            # Check if price is inside value area
            if not np.isnan(val) and not np.isnan(vah):
                close_price = row['close']
                result_df.loc[idx, 'vp_inside_value'] = 1 if val <= close_price <= vah else 0
            
            # Check if HVN/LVN are nearby
            if not np.isnan(current_atr) and current_atr > 0:
                close_price = row['close']
                atr_distance = hvn_lvn_atr_distance * current_atr
                
                # Check HVN nearby
                hvn_near = any(abs(price - close_price) <= atr_distance for price in hvn_prices)
                result_df.loc[idx, 'vp_hvn_near'] = 1 if hvn_near else 0
                
                # Check LVN nearby
                lvn_near = any(abs(price - close_price) <= atr_distance for price in lvn_prices)
                result_df.loc[idx, 'vp_lvn_near'] = 1 if lvn_near else 0
            elif hvn_prices or lvn_prices:
                # Fallback: use fixed distance if ATR is not available
                close_price = row['close']
                fixed_distance = 0.5  # Default fixed distance
                hvn_near = any(abs(price - close_price) <= fixed_distance for price in hvn_prices)
                lvn_near = any(abs(price - close_price) <= fixed_distance for price in lvn_prices)
                result_df.loc[idx, 'vp_hvn_near'] = 1 if hvn_near else 0
                result_df.loc[idx, 'vp_lvn_near'] = 1 if lvn_near else 0
        
        # Calculate POC shift direction (compared to previous session)
        if i > 0:
            prev_date = unique_dates[i-1]
            prev_session = rth_df[rth_df['date'] == prev_date]
            
            if not prev_session.empty:
                # Calculate volume profile for previous session only
                try:
                    prev_profile = calculate_volume_profile_safe(prev_session, bin_size)
                    
                    if not prev_profile.empty:
                        prev_poc = find_poc_safe(prev_profile)
                        
                        if not np.isnan(poc) and not np.isnan(prev_poc):
                            # Set POC shift direction for all bars in current session
                            if poc > prev_poc:
                                result_df.loc[current_session.index, 'vp_poc_shift_dir'] = 1
                            elif poc < prev_poc:
                                result_df.loc[current_session.index, 'vp_poc_shift_dir'] = -1
                            else:
                                result_df.loc[current_session.index, 'vp_poc_shift_dir'] = 0
                except Exception as e:
                    logger.warning(f"Error calculating previous session POC for date {date}: {e}")
    
    # Apply forward-fill within each date to reduce NaN values
    for date in unique_dates:
        date_mask = result_df.index.date == date
        date_indices = result_df[date_mask].index
        
        if len(date_indices) > 1:
            # Forward-fill POC, VAH, VAL within the same session
            result_df.loc[date_indices, 'vp_poc'] = result_df.loc[date_indices, 'vp_poc'].ffill()
            result_df.loc[date_indices, 'vp_vah'] = result_df.loc[date_indices, 'vp_vah'].ffill()
            result_df.loc[date_indices, 'vp_val'] = result_df.loc[date_indices, 'vp_val'].ffill()
    
    return result_df

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
        poc_row = profile.loc[profile['volume'].idxmax()]
        return poc_row['price']
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

def test_volume_profile_fix():
    """Test the volume profile NaN fix."""
    logger.info("Testing volume profile NaN fix")
    
    # Create sample data
    np.random.seed(42)
    start_date = pd.Timestamp('2023-01-02 09:30:00-05:00')
    dates = pd.date_range(start=start_date, periods=200, freq='5min')
    
    base_price = 100.0
    returns = np.random.normal(0, 0.001, 200)
    prices = [base_price]
    
    for i in range(1, 200):
        momentum = 0.1 * returns[i-1] if i > 0 else 0
        price_change = returns[i] + momentum
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    data = {
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 200))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 200))),
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, 200).astype(int)
    }
    
    for i in range(200):
        data['high'][i] = max(data['open'][i], data['high'][i], data['close'][i])
        data['low'][i] = min(data['open'][i], data['low'][i], data['close'][i])
    
    df = pd.DataFrame(data, index=dates)
    
    # Test configuration
    config = {
        'bin_size': 0.05,
        'value_area': 0.70,
        'rolling_sessions': 20,
        'atr_window': 20,
        'hvn_lvn_threshold': 0.2,
        'hvn_lvn_atr_distance': 0.5
    }
    
    # Test original implementation
    logger.info("Testing original implementation...")
    original_vp = compute_volume_profile_features(df, config)
    
    # Test enhanced implementation
    logger.info("Testing enhanced implementation...")
    enhanced_vp = enhanced_compute_volume_profile_features(df, config)
    
    # Compare results
    logger.info("\nCOMPARISON RESULTS:")
    logger.info("="*50)
    
    # Overall NaN comparison
    original_nan = original_vp.isna().sum().sum()
    enhanced_nan = enhanced_vp.isna().sum().sum()
    original_cells = original_vp.shape[0] * original_vp.shape[1]
    enhanced_cells = enhanced_vp.shape[0] * enhanced_vp.shape[1]
    
    original_pct = (original_nan / original_cells) * 100
    enhanced_pct = (enhanced_nan / enhanced_cells) * 100
    
    logger.info(f"Original implementation: {original_nan}/{original_cells} NaN values ({original_pct:.2f}%)")
    logger.info(f"Enhanced implementation: {enhanced_nan}/{enhanced_cells} NaN values ({enhanced_pct:.2f}%)")
    logger.info(f"Improvement: {original_pct - enhanced_pct:.2f} percentage points")
    
    # Feature-specific comparison
    vp_features = ['vp_poc', 'vp_vah', 'vp_val', 'vp_dist_to_poc_atr']
    
    logger.info("\nFeature-specific comparison:")
    for feature in vp_features:
        if feature in original_vp.columns and feature in enhanced_vp.columns:
            orig_nan = original_vp[feature].isna().sum()
            enh_nan = enhanced_vp[feature].isna().sum()
            orig_pct = (orig_nan / len(original_vp)) * 100
            enh_pct = (enh_nan / len(enhanced_vp)) * 100
            logger.info(f"  {feature}: {orig_pct:.2f}% -> {enh_pct:.2f}% ({orig_pct - enh_pct:.2f} improvement)")
    
    return original_vp, enhanced_vp

if __name__ == "__main__":
    original_vp, enhanced_vp = test_volume_profile_fix()