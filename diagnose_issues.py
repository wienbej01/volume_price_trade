#!/usr/bin/env python3
"""
Diagnostic script to investigate the critical issues found in integration tests:
1. Future leakage detection (vp_poc NaN values at the end)
2. ICT module shape doubling
3. High NaN count
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from volume_price_trade.features.feature_union import build_feature_matrix
from volume_price_trade.features.volume_profile import compute_volume_profile_features
from volume_price_trade.features.ict import compute_ict_features
from volume_price_trade.features.ta_basic import compute_ta_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML files."""
    config_path = Path("config/base.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    features_path = Path("config/features.yaml")
    with open(features_path, "r") as f:
        features_config = yaml.safe_load(f)
    
    config.update(features_config)
    return config

def create_sample_data(n_bars=50):
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

def investigate_volume_profile_issue():
    """Investigate the volume profile NaN issue."""
    print("\n" + "="*60)
    print("INVESTIGATING VOLUME PROFILE NaN ISSUE")
    print("="*60)
    
    config = load_config()
    df = create_sample_data(50)
    
    print(f"Input data shape: {df.shape}")
    print(f"Input data index range: {df.index.min()} to {df.index.max()}")
    
    # Compute volume profile features
    vp_features = compute_volume_profile_features(df, config.get('volume_profile', {}))
    
    print(f"Volume profile features shape: {vp_features.shape}")
    
    # Check vp_poc column specifically
    if 'vp_poc' in vp_features.columns:
        vp_poc_series = vp_features['vp_poc']
        nan_count = vp_poc_series.isna().sum()
        total_count = len(vp_poc_series)
        
        print(f"vp_poc column:")
        print(f"  - Total values: {total_count}")
        print(f"  - NaN values: {nan_count}")
        print(f"  - Non-NaN values: {total_count - nan_count}")
        
        # Check where NaN values are located
        if nan_count > 0:
            nan_indices = vp_poc_series[vp_poc_series.isna()].index
            print(f"  - NaN indices: {len(nan_indices)} locations")
            print(f"  - First NaN: {nan_indices.min() if len(nan_indices) > 0 else 'N/A'}")
            print(f"  - Last NaN: {nan_indices.max() if len(nan_indices) > 0 else 'N/A'}")
            
            # Check if NaNs are at the end
            last_valid_idx = vp_poc_series.last_valid_index()
            last_df_idx = vp_poc_series.index[-1]
            print(f"  - Last valid index: {last_valid_idx}")
            print(f"  - Last dataframe index: {last_df_idx}")
            print(f"  - NaNs at end: {last_valid_idx != last_df_idx}")
            
            # Show sample values
            print(f"  - Sample values:")
            for i in range(min(5, len(vp_poc_series))):
                idx = vp_poc_series.index[i]
                val = vp_poc_series.iloc[i]
                print(f"    {idx}: {val}")
            
            if len(vp_poc_series) > 5:
                print("    ...")
                for i in range(max(5, len(vp_poc_series)-5), len(vp_poc_series)):
                    idx = vp_poc_series.index[i]
                    val = vp_poc_series.iloc[i]
                    print(f"    {idx}: {val}")
    
    # Check other volume profile columns
    vp_cols = [col for col in vp_features.columns if col.startswith('vp_')]
    print(f"\nAll volume profile columns:")
    for col in vp_cols:
        nan_count = vp_features[col].isna().sum()
        total_count = len(vp_features[col])
        print(f"  - {col}: {nan_count}/{total_count} NaN values ({nan_count/total_count*100:.1f}%)")

def investigate_ict_shape_issue():
    """Investigate the ICT shape doubling issue."""
    print("\n" + "="*60)
    print("INVESTIGATING ICT SHAPE DOUBLING ISSUE")
    print("="*60)
    
    config = load_config()
    df = create_sample_data(50)
    
    print(f"Input data shape: {df.shape}")
    print(f"Input data index range: {df.index.min()} to {df.index.max()}")
    
    # Compute TA features first (needed by ICT)
    ta_features = compute_ta_features(df, config.get('ta', {}))
    print(f"TA features shape: {ta_features.shape}")
    
    # Create working dataframe with TA features
    working_df = df.copy()
    ta_cols = [col for col in ta_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    for col in ta_cols:
        working_df[col] = ta_features[col]
    
    print(f"Working dataframe shape: {working_df.shape}")
    
    # Compute ICT features
    ict_features = compute_ict_features(working_df, config.get('ict', {}))
    
    print(f"ICT features shape: {ict_features.shape}")
    
    if len(ict_features) != len(df):
        print(f"WARNING: ICT output length ({len(ict_features)}) != input length ({len(df)})")
        
        # Check the index of both dataframes
        print(f"Input index range: {df.index.min()} to {df.index.max()}")
        
        # Check if ICT index is valid before trying to get min/max
        try:
            print(f"ICT index range: {ict_features.index.min()} to {ict_features.index.max()}")
        except Exception as e:
            print(f"Error getting ICT index range: {e}")
            print(f"ICT index type: {type(ict_features.index)}")
            print(f"ICT index sample: {ict_features.index[:5] if len(ict_features.index) > 0 else 'Empty'}")
        
        # Check if ICT features are duplicated
        if len(ict_features) == 2 * len(df):
            print("ICT features appear to be duplicated!")
            
            # Show first few rows
            print("\nFirst 5 rows of ICT features:")
            print(ict_features.head())
            
            print("\nRows 50-55 of ICT features (should be beyond input):")
            print(ict_features.iloc[50:55])
            
            # Check if the second half is a duplicate
            first_half = ict_features.iloc[:len(df)]
            second_half = ict_features.iloc[len(df):]
            
            if first_half.index.equals(second_half.index):
                print("Second half has identical indices to first half!")
            else:
                print("Second half has different indices from first half")
                print(f"First half index range: {first_half.index.min()} to {first_half.index.max()}")
                print(f"Second half index range: {second_half.index.min()} to {second_half.index.max()}")

def investigate_nan_counts():
    """Investigate overall NaN counts in the feature matrix."""
    print("\n" + "="*60)
    print("INVESTIGATING OVERALL NaN COUNTS")
    print("="*60)
    
    config = load_config()
    df = create_sample_data(100)
    
    print(f"Input data shape: {df.shape}")
    
    # Build complete feature matrix
    feature_matrix = build_feature_matrix(df, config)
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    
    # Count NaN values by column
    nan_counts = feature_matrix.isna().sum()
    total_nans = nan_counts.sum()
    total_cells = feature_matrix.shape[0] * feature_matrix.shape[1]
    
    print(f"Total NaN values: {total_nans}")
    print(f"Total cells: {total_cells}")
    print(f"NaN percentage: {total_nans/total_cells*100:.2f}%")
    
    # Show columns with most NaN values
    print("\nTop 10 columns with most NaN values:")
    top_nan_cols = nan_counts.sort_values(ascending=False).head(10)
    for col, count in top_nan_cols.items():
        percentage = count / len(feature_matrix) * 100
        print(f"  - {col}: {count} ({percentage:.1f}%)")
    
    # Show columns with no NaN values
    no_nan_cols = nan_counts[nan_counts == 0]
    print(f"\nColumns with no NaN values: {len(no_nan_cols)}")
    if len(no_nan_cols) > 0:
        print(f"  {', '.join(no_nan_cols.index.tolist())}")

def check_configuration_integration():
    """Check why configuration integration test failed."""
    print("\n" + "="*60)
    print("INVESTIGATING CONFIGURATION INTEGRATION ISSUE")
    print("="*60)
    
    config = load_config()
    df = create_sample_data(50)
    
    # Test with default configuration
    print("Testing with default configuration...")
    feature_matrix_default = build_feature_matrix(df, config)
    print(f"Default config shape: {feature_matrix_default.shape}")
    
    # Test with modified configuration
    print("\nTesting with modified configuration...")
    modified_config = config.copy()
    modified_config['atr_window'] = 10  # Change ATR window
    modified_config['rvol_windows'] = [10, 30]  # Change RVOL windows
    
    feature_matrix_modified = build_feature_matrix(df, modified_config)
    print(f"Modified config shape: {feature_matrix_modified.shape}")
    
    # Check if shapes are different
    if feature_matrix_default.shape != feature_matrix_modified.shape:
        print("✓ Configuration changes had effect on output shape")
    else:
        print("✗ Configuration changes did NOT affect output shape")
        
        # Check if actual values are different
        # Compare ATR columns
        atr_col_default = [col for col in feature_matrix_default.columns if col.startswith('atr_')][0]
        atr_col_modified = [col for col in feature_matrix_modified.columns if col.startswith('atr_')][0]
        
        if atr_col_default != atr_col_modified:
            print(f"ATR column names differ: {atr_col_default} vs {atr_col_modified}")
        else:
            # Compare values
            default_values = feature_matrix_default[atr_col_default].dropna()
            modified_values = feature_matrix_modified[atr_col_modified].dropna()
            
            if len(default_values) > 0 and len(modified_values) > 0:
                correlation = default_values.corr(modified_values)
                print(f"Correlation between default and modified ATR values: {correlation:.4f}")
                
                if correlation < 0.99:
                    print("✓ Configuration changes had effect on ATR values")
                else:
                    print("✗ Configuration changes did NOT significantly affect ATR values")

def main():
    """Run all diagnostic investigations."""
    print("DIAGNOSTIC INVESTIGATION OF CRITICAL ISSUES")
    print("="*80)
    
    investigate_volume_profile_issue()
    investigate_ict_shape_issue()
    investigate_nan_counts()
    check_configuration_integration()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC INVESTIGATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()