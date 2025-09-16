#!/usr/bin/env python3
"""
Test script to verify that configuration integration fixes are working correctly.

This script tests the specific configuration parameters that were reported as failing:
- atr_window (should affect ATR calculations)
- rvol_windows (should affect RVOL calculations) 
- bin_size (should affect volume profile calculations)
- fvg_min_size_atr (should affect ICT calculations)
- rvol_climax (should affect VPA calculations)
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from volume_price_trade.features.feature_union import build_feature_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data(n_bars: int = 200) -> pd.DataFrame:
    """Create sample OHLCV data for testing with more volatility to generate patterns."""
    logger.info(f"Creating sample OHLCV data with {n_bars} bars")
    
    # Create datetime index
    start_date = pd.Timestamp('2023-01-02 09:30:00-05:00')
    dates = pd.date_range(start=start_date, periods=n_bars, freq='5min')
    
    # Generate more volatile price data to create FVG patterns and volume profile differences
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n_bars):
        # Create more volatility and some gaps
        if i % 20 == 0:  # Create gaps every 20 bars
            gap_size = np.random.choice([-0.5, 0.5])  # 50 cent gaps
            new_price = prices[-1] * (1 + gap_size/100)
        else:
            price_change = np.random.normal(0, 0.003)  # Higher volatility
            new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data with more realistic patterns
    data = {
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.004, n_bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.004, n_bars))),
        'close': prices,
        'volume': np.random.lognormal(11, 0.8, n_bars).astype(int)  # Higher volume variation
    }
    
    # Ensure high >= low and high/low include open/close
    for i in range(n_bars):
        data['high'][i] = max(data['open'][i], data['high'][i], data['close'][i])
        data['low'][i] = min(data['open'][i], data['low'][i], data['close'][i])
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_configuration_changes():
    """Test that configuration changes actually affect feature output."""
    logger.info("Testing configuration changes affect feature output")
    
    # Load base configuration
    config_path = Path("config/base.yaml")
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    # Create test data
    df = create_sample_data(100)
    
    # Test 1: Default configuration
    logger.info("Test 1: Running with default configuration")
    default_features = build_feature_matrix(df, base_config)
    
    # Test 2: Modified configuration
    logger.info("Test 2: Running with modified configuration")
    modified_config = yaml.safe_load(yaml.dump(base_config))  # Deep copy via string conversion
    
    # Modify key parameters
    modified_config['features']['atr_window'] = 10  # Changed from 20
    modified_config['features']['rvol_windows'] = [10, 30]  # Changed from [5, 20]
    modified_config['features']['volume_profile']['bin_size'] = 0.1  # Changed from 0.05
    modified_config['features']['ict']['fvg_min_size_atr'] = 0.5  # Changed from 0.25
    modified_config['features']['vpa']['rvol_climax'] = 3.0  # Changed from 2.5
    
    modified_features = build_feature_matrix(df, modified_config)
    
    # Compare results
    results = {
        'success': True,
        'tests': {},
        'issues': []
    }
    
    # Test ATR window change
    atr_columns_default = [col for col in default_features.columns if col.startswith('atr_')]
    atr_columns_modified = [col for col in modified_features.columns if col.startswith('atr_')]
    
    logger.info(f"Default ATR columns: {atr_columns_default}")
    logger.info(f"Modified ATR columns: {atr_columns_modified}")
    
    atr_changed = atr_columns_default != atr_columns_modified
    results['tests']['atr_window'] = {
        'expected_change': 'atr_20 -> atr_10',
        'actual_change': f"{atr_columns_default} -> {atr_columns_modified}",
        'success': atr_changed
    }
    
    # Test RVOL windows change
    rvol_columns_default = [col for col in default_features.columns if col.startswith('rvol_')]
    rvol_columns_modified = [col for col in modified_features.columns if col.startswith('rvol_')]
    
    logger.info(f"Default RVOL columns: {rvol_columns_default}")
    logger.info(f"Modified RVOL columns: {rvol_columns_modified}")
    
    rvol_changed = rvol_columns_default != rvol_columns_modified
    results['tests']['rvol_windows'] = {
        'expected_change': 'rvol_5, rvol_20 -> rvol_10, rvol_30',
        'actual_change': f"{rvol_columns_default} -> {rvol_columns_modified}",
        'success': rvol_changed
    }
    
    # Test Volume Profile bin_size change (should affect VP features)
    vp_features_default = [col for col in default_features.columns if col.startswith('vp_')]
    vp_features_modified = [col for col in modified_features.columns if col.startswith('vp_')]
    
    # For bin_size, check if any non-NaN VP POC values changed
    if 'vp_poc' in default_features.columns and 'vp_poc' in modified_features.columns:
        vp_poc_default_nonan = default_features['vp_poc'].dropna()
        vp_poc_modified_nonan = modified_features['vp_poc'].dropna()
        
        if len(vp_poc_default_nonan) > 0 and len(vp_poc_modified_nonan) > 0:
            # Compare first non-NaN values
            vp_poc_default = vp_poc_default_nonan.iloc[0]
            vp_poc_modified = vp_poc_modified_nonan.iloc[0]
            vp_changed = vp_poc_default != vp_poc_modified
        else:
            vp_poc_default = 0
            vp_poc_modified = 0
            vp_changed = False
    else:
        vp_poc_default = 0
        vp_poc_modified = 0
        vp_changed = False
    
    logger.info(f"Default VP POC (first non-NaN): {vp_poc_default}")
    logger.info(f"Modified VP POC (first non-NaN): {vp_poc_modified}")
    
    results['tests']['bin_size'] = {
        'expected_change': 'VP POC values should change with different bin_size',
        'actual_change': f"VP POC: {vp_poc_default} -> {vp_poc_modified}",
        'success': vp_changed
    }
    
    # Test ICT fvg_min_size_atr change
    ict_fvg_up_default = default_features['ict_fvg_up'].sum() if 'ict_fvg_up' in default_features.columns else 0
    ict_fvg_up_modified = modified_features['ict_fvg_up'].sum() if 'ict_fvg_up' in modified_features.columns else 0
    
    logger.info(f"Default ICT FVG up count: {ict_fvg_up_default}")
    logger.info(f"Modified ICT FVG up count: {ict_fvg_up_modified}")
    
    ict_changed = ict_fvg_up_default != ict_fvg_up_modified
    results['tests']['fvg_min_size_atr'] = {
        'expected_change': 'FVG detection should change with different threshold',
        'actual_change': f"FVG up count: {ict_fvg_up_default} -> {ict_fvg_up_modified}",
        'success': ict_changed
    }
    
    # Test VPA rvol_climax change
    vpa_climax_up_default = default_features['vpa_climax_up'].sum() if 'vpa_climax_up' in default_features.columns else 0
    vpa_climax_up_modified = modified_features['vpa_climax_up'].sum() if 'vpa_climax_up' in modified_features.columns else 0
    
    logger.info(f"Default VPA climax up count: {vpa_climax_up_default}")
    logger.info(f"Modified VPA climax up count: {vpa_climax_up_modified}")
    
    vpa_changed = vpa_climax_up_default != vpa_climax_up_modified
    results['tests']['rvol_climax'] = {
        'expected_change': 'Climax detection should change with different RVOL threshold',
        'actual_change': f"Climax up count: {vpa_climax_up_default} -> {vpa_climax_up_modified}",
        'success': vpa_changed
    }
    
    # Overall assessment
    passed_tests = sum(1 for test in results['tests'].values() if test['success'])
    total_tests = len(results['tests'])
    
    results['summary'] = {
        'passed_tests': passed_tests,
        'total_tests': total_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
        'overall_success': passed_tests == total_tests
    }
    
    return results

def main():
    """Main function to test configuration fixes."""
    print("Testing configuration integration fixes...")
    print("=" * 60)
    
    results = test_configuration_changes()
    
    # Print results
    print("\nCONFIGURATION INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    for test_name, test_result in results['tests'].items():
        status = "✅ PASS" if test_result['success'] else "❌ FAIL"
        print(f"{status} {test_name}")
        print(f"   Expected: {test_result['expected_change']}")
        print(f"   Actual:   {test_result['actual_change']}")
        print()
    
    # Print summary
    summary = results['summary']
    print("SUMMARY")
    print("-" * 30)
    print(f"Tests passed: {summary['passed_tests']}/{summary['total_tests']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    
    if summary['overall_success']:
        print("🎉 ALL TESTS PASSED! Configuration integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Configuration integration needs further investigation.")
    
    print("=" * 60)
    
    return 0 if summary['overall_success'] else 1

if __name__ == "__main__":
    exit(main())