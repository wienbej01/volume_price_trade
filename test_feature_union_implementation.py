#!/usr/bin/env python3
"""
Test script for feature_union.py implementation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from volume_price_trade.features.feature_union import build_feature_matrix

def create_sample_data(n_bars=100):
    """Create sample OHLCV data for testing."""
    import pytz
    
    # Generate timestamps with timezone
    et = pytz.timezone("America/New_York")
    start_time = et.localize(datetime(2023, 1, 2, 9, 30))  # Market open on a weekday
    timestamps = [start_time + timedelta(minutes=i*5) for i in range(n_bars)]
    
    # Generate price data with random walk
    np.random.seed(42)  # For reproducible results
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n_bars):
        change = np.random.normal(0, 0.5)  # Random price change
        new_price = prices[-1] + change
        prices.append(max(new_price, 50.0))  # Prevent negative prices
    
    # Create OHLCV data
    data = {
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.2)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.2)) for p in prices],
        'close': prices,
        'volume': [int(np.random.normal(10000, 2000)) for _ in range(n_bars)]
    }
    
    # Ensure high >= open/close >= low
    for i in range(n_bars):
        high = max(data['open'][i], data['close'][i], data['high'][i])
        low = min(data['open'][i], data['close'][i], data['low'][i])
        data['high'][i] = high
        data['low'][i] = low
    
    # Create DataFrame
    df = pd.DataFrame(data, index=timestamps)
    
    return df

def test_feature_union():
    """Test the feature union implementation."""
    print("Creating sample data...")
    df = create_sample_data(100)
    
    print(f"Sample data shape: {df.shape}")
    print(f"Sample data columns: {list(df.columns)}")
    print(f"Sample data index type: {type(df.index)}")
    
    # Create configuration
    cfg = {
        'ta': {
            'atr_window': 20,
            'rvol_windows': [5, 20]
        },
        'volume_profile': {
            'bin_size': 0.05,
            'value_area': 0.70,
            'rolling_sessions': 20,
            'atr_window': 20
        },
        'vpa': {
            'rvol_climax': 2.5,
            'vdu_threshold': 0.4,
            'atr_threshold': 0.5,
            'breakout_lookback': 20
        },
        'ict': {
            'fvg_min_size_atr': 0.25,
            'displacement_body_atr': 1.2,
            'time_of_day': {
                'killzones': {
                    'ny_open': ["09:30", "11:30"],
                    'lunch': ["12:00", "13:30"],
                    'pm_drive': ["13:30", "16:00"]
                }
            }
        },
        'time_of_day': {
            'killzones': {
                'ny_open': ["09:30", "11:30"],
                'lunch': ["12:00", "13:30"],
                'pm_drive': ["13:30", "16:00"]
            },
            'sessions': {
                'rth_start': "09:30",
                'rth_end': "16:00"
            }
        },
        'vwap': {
            'atr_window': 20
        }
    }
    
    print("\nBuilding feature matrix...")
    try:
        feature_matrix = build_feature_matrix(df, cfg)
        
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Feature matrix columns: {list(feature_matrix.columns)}")
        
        # Check for NaN values
        nan_counts = feature_matrix.isna().sum()
        print("\nNaN value counts:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  {col}: {count}")
        
        # Check data types
        print("\nData types:")
        for col in feature_matrix.columns:
            print(f"  {col}: {feature_matrix[col].dtype}")
        
        # Save sample of feature matrix
        feature_matrix.head(20).to_csv('feature_matrix_sample.csv')
        print("\nSample of feature matrix saved to 'feature_matrix_sample.csv'")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_union()
    sys.exit(0 if success else 1)