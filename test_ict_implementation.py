"""Test script for ICT features implementation."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from volume_price_trade.features.ict import compute_ict_features

def create_test_data(n_bars=50):
    """Create test OHLCV data with some known patterns."""
    # Create a date range
    start_date = datetime(2023, 1, 1, 9, 30)
    dates = [start_date + timedelta(minutes=i*5) for i in range(n_bars)]
    
    # Create base price data with some randomness
    np.random.seed(42)  # For reproducible results
    base_price = 100
    prices = [base_price]
    
    for i in range(1, n_bars):
        # Random walk with some drift
        change = np.random.normal(0, 0.5)
        prices.append(prices[-1] + change)
    
    # Create OHLCV data
    data = {
        'timestamp': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.3)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.3)) for p in prices],
        'close': [p + np.random.normal(0, 0.2) for p in prices],
        'volume': [int(np.random.normal(10000, 2000)) for _ in range(n_bars)]
    }
    
    # Create some specific patterns for testing
    
    # FVG Up pattern at index 10-12
    # Bar1 (index 10): high should be < Bar2 (index 11) low
    # Bar2 (index 11): low should be < Bar3 (index 12) high
    data['open'][10] = 100.5
    data['high'][10] = 101.0
    data['low'][10] = 100.3
    data['close'][10] = 100.8
    
    data['open'][11] = 101.6  # Set open > Bar1 high to ensure gap
    data['high'][11] = 102.0
    data['low'][11] = 101.5  # Gap up
    data['close'][11] = 101.8
    
    data['open'][12] = 101.6
    data['high'][12] = 102.5  # Should be > Bar2 low
    data['low'][12] = 101.8  # Should be > Bar1 high
    data['close'][12] = 102.2
    
    # FVG Down pattern at index 20-22
    # Bar1 (index 20): low should be > Bar2 (index 21) high
    # Bar2 (index 21): high should be > Bar3 (index 22) low
    data['open'][20] = 102.5
    data['high'][20] = 102.8
    data['low'][20] = 102.6  # Set low > Bar2 high to ensure gap
    data['close'][20] = 102.7
    
    data['open'][21] = 102.4  # Set open < Bar1 low to ensure gap
    data['high'][21] = 102.5  # Gap down
    data['low'][21] = 102.0
    data['close'][21] = 102.2
    
    data['open'][22] = 102.1
    data['high'][22] = 102.3  # Should be < Bar1 low
    data['low'][22] = 101.8  # Should be < Bar2 high
    data['close'][22] = 102.0
    
    # Liquidity sweep up at index 30
    # First create a swing high around index 25-28
    for i in range(25, 29):
        data['open'][i] = 103.0 + (i-25)*0.05
        data['high'][i] = 104.0 + (i-25)*0.1
        data['low'][i] = 102.8 + (i-25)*0.05
        data['close'][i] = 103.5 + (i-25)*0.08
    
    # Create a liquidity sweep up at index 30 with long up wick
    data['open'][30] = 103.8
    data['high'][30] = 104.5  # Long up wick breaks previous high
    data['low'][30] = 103.7
    data['close'][30] = 103.9  # Closes back inside (below swing high)
    
    # Liquidity sweep down at index 35
    # First create a swing low around index 32-34
    # Index 33 will be our swing low (local minimum)
    data['open'][32] = 101.9
    data['high'][32] = 102.1
    data['low'][32] = 101.7
    data['close'][32] = 101.8
    
    data['open'][33] = 101.8  # Swing low bar
    data['high'][33] = 101.9
    data['low'][33] = 101.5  # This is the swing low (local minimum)
    data['close'][33] = 101.6
    
    data['open'][34] = 101.7
    data['high'][34] = 101.8
    data['low'][34] = 101.6
    data['close'][34] = 101.7
    
    # Create a liquidity sweep down at index 35 with long down wick
    data['open'][35] = 101.7
    data['high'][35] = 101.8
    data['low'][35] = 101.4  # Long down wick breaks previous swing low (101.5)
    data['close'][35] = 101.6  # Closes back inside (above swing low)
    
    # Displacement up at index 40
    data['open'][40] = 103.0
    data['high'][40] = 105.2
    data['low'][40] = 102.8
    data['close'][40] = 105.0  # Large body up
    
    # Displacement down at index 45
    data['open'][45] = 105.0
    data['high'][45] = 105.2
    data['low'][45] = 102.8
    data['close'][45] = 103.0  # Large body down
    
    # Ensure high >= open, close and low <= open, close
    for i in range(n_bars):
        # Skip the pattern indices to preserve our carefully crafted patterns
        if 10 <= i <= 12 or 20 <= i <= 22 or 25 <= i <= 30 or 32 <= i <= 35 or i == 40 or i == 45:
            continue
        data['high'][i] = max(data['open'][i], data['close'][i], data['high'][i])
        data['low'][i] = min(data['open'][i], data['close'][i], data['low'][i])
    
    return pd.DataFrame(data)

def test_ict_features():
    """Test the ICT features implementation."""
    print("Creating test data...")
    df = create_test_data()
    
    print("Test data created:")
    print(df.head())
    print("\n")
    
    print("Testing ICT features...")
    
    # Create configuration
    cfg = {
        'fvg_min_size_atr': 0.1,  # Lower threshold for testing
        'displacement_body_atr': 0.5,  # Lower threshold for testing
        'time_of_day': {
            'killzones': {
                'ny_open': ["09:30", "11:30"],
                'lunch': ["12:00", "13:30"],
                'pm_drive': ["13:30", "16:00"]
            }
        }
    }
    
    # Compute ICT features
    result_df = compute_ict_features(df, cfg)
    
    print("ICT features computed successfully!")
    print("\nResult columns:")
    print(result_df.columns.tolist())
    
    print("\nSample of ICT features:")
    ict_cols = [col for col in result_df.columns if col.startswith('ict_')]
    print(result_df[ict_cols].head(10))
    
    # Check for expected patterns
    print("\nChecking for expected patterns:")
    
    # Debug: Print the data around the expected patterns
    print("\nDebug data around FVG Up pattern (indices 10-12):")
    print(df.loc[10:12, ['open', 'high', 'low', 'close']])
    print(f"FVG Up condition check:")
    print(f"Bar1 high ({df.loc[10, 'high']}) < Bar2 low ({df.loc[11, 'low']}): {df.loc[10, 'high'] < df.loc[11, 'low']}")
    print(f"Bar2 low ({df.loc[11, 'low']}) < Bar3 high ({df.loc[12, 'high']}): {df.loc[11, 'low'] < df.loc[12, 'high']}")
    
    print("\nDebug data around FVG Down pattern (indices 20-22):")
    print(df.loc[20:22, ['open', 'high', 'low', 'close']])
    print(f"FVG Down condition check:")
    print(f"Bar1 low ({df.loc[20, 'low']}) > Bar2 high ({df.loc[21, 'high']}): {df.loc[20, 'low'] > df.loc[21, 'high']}")
    print(f"Bar2 high ({df.loc[21, 'high']}) > Bar3 low ({df.loc[22, 'low']}): {df.loc[21, 'high'] > df.loc[22, 'low']}")
    
    print("\nDebug data around Liquidity Sweep Up pattern (indices 28-30):")
    print(df.loc[28:30, ['open', 'high', 'low', 'close']])
    
    print("\nDebug data around Liquidity Sweep Down pattern (indices 33-35):")
    print(df.loc[33:35, ['open', 'high', 'low', 'close']])
    
    # Find the lowest low in the lookback period (excluding current bar)
    # The function uses default lookback of 10 bars, so let's check indices 25-34
    lookback_data = df.loc[25:34]  # Look at indices 25-34 (excluding 35)
    swing_low = lookback_data['low'].min()
    swing_low_idx = lookback_data['low'].idxmin()
    print(f"Liquidity Sweep Down condition check:")
    print(f"Full lookback period (indices 25-34) low values:")
    print(lookback_data['low'])
    print(f"Swing low in lookback period: {swing_low} at index {swing_low_idx}")
    print(f"Current bar low ({df.loc[35, 'low']}) < swing_low ({swing_low}): {df.loc[35, 'low'] < swing_low}")
    print(f"Current bar close ({df.loc[35, 'close']}) > swing_low ({swing_low}): {df.loc[35, 'close'] > swing_low}")
    
    # Calculate wick size
    current_bar = df.loc[35]
    wick_size = min(current_bar['open'], current_bar['close']) - current_bar['low']
    total_range = current_bar['high'] - current_bar['low']
    wick_significant = total_range > 0 and (wick_size / total_range) >= 0.3
    print(f"Wick size: {wick_size}, Total range: {total_range}, Wick significant: {wick_significant}")
    
    # Calculate ATR values for debugging
    print("\nATR values at pattern indices:")
    from volume_price_trade.features.ta_basic import atr
    atr_values = atr(df)
    print(f"ATR at index 12: {atr_values.iloc[12]}")
    print(f"ATR at index 22: {atr_values.iloc[22]}")
    print(f"ATR at index 30: {atr_values.iloc[30]}")
    print(f"ATR at index 35: {atr_values.iloc[35]}")
    
    # FVG Up at index 12
    if result_df.loc[12, 'ict_fvg_up']:
        print("✓ FVG Up pattern detected at index 12")
    else:
        print("✗ FVG Up pattern NOT detected at index 12")
    
    # FVG Down at index 22
    if result_df.loc[22, 'ict_fvg_down']:
        print("✓ FVG Down pattern detected at index 22")
    else:
        print("✗ FVG Down pattern NOT detected at index 22")
    
    # Liquidity sweep up at index 30
    if result_df.loc[30, 'ict_liquidity_sweep_up']:
        print("✓ Liquidity sweep up pattern detected at index 30")
    else:
        print("✗ Liquidity sweep up pattern NOT detected at index 30")
    
    # Liquidity sweep down at index 35
    if result_df.loc[35, 'ict_liquidity_sweep_down']:
        print("✓ Liquidity sweep down pattern detected at index 35")
    else:
        print("✗ Liquidity sweep down pattern NOT detected at index 35")
    
    # Displacement up at index 40
    if result_df.loc[40, 'ict_displacement_up']:
        print("✓ Displacement up pattern detected at index 40")
    else:
        print("✗ Displacement up pattern NOT detected at index 40")
    
    # Displacement down at index 45
    if result_df.loc[45, 'ict_displacement_down']:
        print("✓ Displacement down pattern detected at index 45")
    else:
        print("✗ Displacement down pattern NOT detected at index 45")
    
    # Killzone flags
    print("\nChecking killzone flags:")
    for i in range(min(5, len(result_df))):
        ts = result_df.loc[i, 'timestamp']
        time_str = ts.strftime("%H:%M")
        killzone_ny_open = result_df.loc[i, 'ict_killzone_ny_open']
        killzone_lunch = result_df.loc[i, 'ict_killzone_lunch']
        killzone_pm_drive = result_df.loc[i, 'ict_killzone_pm_drive']
        print(f"Index {i} ({time_str}): NY Open={killzone_ny_open}, Lunch={killzone_lunch}, PM Drive={killzone_pm_drive}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_ict_features()