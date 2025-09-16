#!/usr/bin/env python3
"""
Test script to verify volume profile NaN reduction fixes.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from volume_price_trade.features.feature_union import build_feature_matrix
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    try:
        with open('config/base.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # Return default config if file not found
        return {
            'features': {
                'atr_window': 20,
                'rvol_windows': [5, 20],
                'volume_profile': {
                    'bin_size': 0.05,
                    'value_area': 0.70,
                    'rolling_sessions': 20,
                    'hvn_lvn_threshold': 0.2,
                    'hvn_lvn_atr_distance': 0.5
                }
            }
        }

def create_test_data(n_bars=500):
    """Create test OHLCV data."""
    np.random.seed(42)
    start_date = pd.Timestamp('2023-01-02 09:30:00-05:00')
    dates = pd.date_range(start=start_date, periods=n_bars, freq='5min')
    
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
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_volume_profile_improvements():
    """Test the volume profile improvements."""
    logger.info("Testing volume profile NaN reduction improvements")
    
    # Create test data with more bars to get better statistics
    df = create_test_data(500)
    config = load_config()
    
    logger.info(f"Test data shape: {df.shape}")
    
    # Build feature matrix with enhanced volume profile
    feature_matrix = build_feature_matrix(df, config)
    
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    
    # Analyze NaN values
    total_cells = feature_matrix.shape[0] * feature_matrix.shape[1]
    total_nans = feature_matrix.isna().sum().sum()
    overall_nan_pct = (total_nans / total_cells) * 100
    
    logger.info(f"Overall NaN percentage: {overall_nan_pct:.2f}% ({total_nans}/{total_cells})")
    
    # Analyze volume profile features specifically
    vp_features = ['vp_poc', 'vp_vah', 'vp_val', 'vp_dist_to_poc_atr', 'vp_inside_value', 'vp_hvn_near', 'vp_lvn_near', 'vp_poc_shift_dir']
    
    logger.info("\nVolume Profile Feature Analysis:")
    logger.info("=" * 50)
    
    for feature in vp_features:
        if feature in feature_matrix.columns:
            nan_count = feature_matrix[feature].isna().sum()
            feature_pct = (nan_count / len(feature_matrix)) * 100
            logger.info(f"  {feature}: {nan_count}/{len(feature_matrix)} NaN values ({feature_pct:.2f}%)")
    
    # Compare with expected improvements
    logger.info("\nExpected vs Actual Improvements:")
    logger.info("=" * 50)
    
    expected_improvements = {
        'vp_poc': 61.00,
        'vp_vah': 61.00,
        'vp_val': 61.00,
        'vp_dist_to_poc_atr': 70.50
    }
    
    for feature, expected_pct in expected_improvements.items():
        if feature in feature_matrix.columns:
            actual_pct = (feature_matrix[feature].isna().sum() / len(feature_matrix)) * 100
            improvement = expected_pct - actual_pct
            logger.info(f"  {feature}: {expected_pct:.2f}% -> {actual_pct:.2f}% ({improvement:.2f} improvement)")
    
    # Check if we achieved the target of < 3% overall NaN
    target_achieved = overall_nan_pct < 3.0
    logger.info(f"\nTarget < 3% overall NaN: {'✓ ACHIEVED' if target_achieved else '✗ NOT ACHIEVED'}")
    
    return feature_matrix, overall_nan_pct

if __name__ == "__main__":
    feature_matrix, overall_nan_pct = test_volume_profile_improvements()
    
    # Save results
    with open('volume_profile_test_results.txt', 'w') as f:
        f.write("Volume Profile Fix Test Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Overall NaN percentage: {overall_nan_pct:.2f}%\n")
        f.write(f"Feature matrix shape: {feature_matrix.shape}\n")
        
        vp_features = ['vp_poc', 'vp_vah', 'vp_val', 'vp_dist_to_poc_atr', 'vp_inside_value', 'vp_hvn_near', 'vp_lvn_near', 'vp_poc_shift_dir']
        
        f.write("\nVolume Profile Feature NaN Counts:\n")
        for feature in vp_features:
            if feature in feature_matrix.columns:
                nan_count = feature_matrix[feature].isna().sum()
                feature_pct = (nan_count / len(feature_matrix)) * 100
                f.write(f"  {feature}: {nan_count}/{len(feature_matrix)} ({feature_pct:.2f}%)\n")
    
    logger.info(f"\nResults saved to volume_profile_test_results.txt")