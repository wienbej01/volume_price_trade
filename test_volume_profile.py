"""Test script for volume profile features."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Import the volume profile function
from src.volume_price_trade.features.volume_profile import compute_volume_profile_features

def create_sample_data():
    """Create sample OHLCV data for testing."""
    # Create date range for 5 trading days
    ET = pytz.timezone("America/New_York")
    start_date = ET.localize(datetime(2023, 1, 2, 9, 30))  # Start of trading day
    
    # Create 1-minute bars for 5 days (390 minutes per day)
    dates = [start_date + timedelta(minutes=i) for i in range(5 * 390)]
    
    # Create sample OHLCV data
    np.random.seed(42)  # For reproducible results
    
    # Base price around $150
    base_price = 150.0
    
    # Create price series with some randomness
    price_changes = np.random.normal(0, 0.05, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        prices.append(prices[-1] + change)
    
    # Create OHLCV data
    data = {
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.1)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.1)) for p in prices],
        'close': prices,
        'volume': [int(np.random.uniform(100, 1000)) for _ in prices]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    return df

def test_volume_profile():
    """Test the volume profile implementation."""
    print("Creating sample data...")
    df = create_sample_data()
    
    print(f"Sample data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Configuration for volume profile
    cfg = {
        'bin_size': 0.05,
        'value_area': 0.70,
        'rolling_sessions': 3,
        'atr_window': 20,
        'hvn_lvn_threshold': 0.2,
        'hvn_lvn_atr_distance': 0.5
    }
    
    print("\nComputing volume profile features...")
    result_df = compute_volume_profile_features(df, cfg)
    
    # Check if features were added
    expected_features = [
        'vp_poc', 'vp_vah', 'vp_val', 'vp_dist_to_poc_atr', 
        'vp_inside_value', 'vp_hvn_near', 'vp_lvn_near', 'vp_poc_shift_dir'
    ]
    
    print("\nChecking if features were added...")
    for feature in expected_features:
        if feature in result_df.columns:
            non_null_count = result_df[feature].notna().sum()
            print(f"✓ {feature}: {non_null_count} non-null values")
        else:
            print(f"✗ {feature}: NOT FOUND")
    
    # Display some sample values
    print("\nSample feature values (last 5 rows):")
    sample_cols = ['close'] + expected_features
    print(result_df[sample_cols].tail())
    
    # Check for any errors
    print("\nChecking for potential issues...")
    for feature in expected_features:
        if feature in result_df.columns:
            # Check for infinite values
            inf_count = np.isinf(result_df[feature]).sum()
            if inf_count > 0:
                print(f"⚠ {feature}: {inf_count} infinite values")
            
            # Check for extreme values
            if feature in ['vp_dist_to_poc_atr']:
                extreme_high = (result_df[feature] > 10).sum()
                if extreme_high > 0:
                    print(f"⚠ {feature}: {extreme_high} values > 10")
    
    print("\nVolume profile features test completed!")
    return result_df

if __name__ == "__main__":
    test_volume_profile()