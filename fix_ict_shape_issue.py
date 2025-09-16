#!/usr/bin/env python3
"""
Fix for the ICT shape doubling issue.
The problem is in the compute_ict_features function where it uses .at[idx, ...] 
with positional indices instead of label-based indexing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from volume_price_trade.features.ict import compute_ict_features

def create_sample_data(n_bars=20):
    """Create sample OHLCV data."""
    start_date = pd.Timestamp('2023-01-02 09:30:00-05:00')
    dates = pd.date_range(start=start_date, periods=n_bars, freq='5min')
    
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.001, n_bars)
    prices = [base_price]
    
    for i in range(1, n_bars):
        momentum = 0.1 * returns[i-1] if i > 0 else 0
        price_change = returns[i] + momentum
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    data = {
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, n_bars).astype(int)
    }
    
    for i in range(n_bars):
        data['high'][i] = max(data['open'][i], data['high'][i], data['close'][i])
        data['low'][i] = min(data['open'][i], data['low'][i], data['close'][i])
    
    return pd.DataFrame(data, index=dates)

def compute_ict_features_fixed(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Fixed version of compute_ict_features that properly handles indexing.
    """
    # Import the detection functions from the original module
    from volume_price_trade.features.ict import (
        detect_fvg_up, detect_fvg_down, detect_liquidity_sweep_up, detect_liquidity_sweep_down,
        detect_displacement_up, detect_displacement_down, calculate_equilibrium_distance, is_in_killzone
    )
    
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
    
    # Compute features for each row - FIXED VERSION
    # Use iloc for positional access and then set values by index label
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

def test_fix():
    """Test the fix for the ICT shape doubling issue."""
    print("Testing ICT shape doubling fix...")
    
    # Create test data
    df = create_sample_data(20)
    print(f"Input data shape: {df.shape}")
    
    # Test original version
    print("\nTesting original version...")
    cfg = {
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
    
    try:
        original_result = compute_ict_features(df, cfg)
        print(f"Original result shape: {original_result.shape}")
        if len(original_result) != len(df):
            print(f"❌ BUG CONFIRMED: Original output length ({len(original_result)}) != input length ({len(df)})")
        else:
            print("✅ Original version works correctly")
    except Exception as e:
        print(f"❌ Original version failed: {e}")
    
    # Test fixed version
    print("\nTesting fixed version...")
    try:
        fixed_result = compute_ict_features_fixed(df, cfg)
        print(f"Fixed result shape: {fixed_result.shape}")
        if len(fixed_result) == len(df):
            print("✅ Fixed version maintains correct shape")
            
            # Check if indices match
            if fixed_result.index.equals(df.index):
                print("✅ Fixed version maintains correct index")
            else:
                print("❌ Fixed version has incorrect index")
                
        else:
            print(f"❌ Fixed version still has shape issues: {len(fixed_result)} != {len(df)}")
    except Exception as e:
        print(f"❌ Fixed version failed: {e}")

if __name__ == "__main__":
    test_fix()