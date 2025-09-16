#!/usr/bin/env python3
"""
Debug script to check configuration flow through feature_union.
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

# Monkey patch the feature modules to log their received configuration
from volume_price_trade.features import volume_profile, ict

# Store original functions
original_compute_volume_profile_features = volume_profile.compute_volume_profile_features
original_compute_ict_features = ict.compute_ict_features

# Patched functions that log configuration
def patched_compute_volume_profile_features(df, cfg):
    logging.getLogger("volume_profile").info(f"Volume Profile received config: {cfg}")
    return original_compute_volume_profile_features(df, cfg)

def patched_compute_ict_features(df, cfg):
    logging.getLogger("ict").info(f"ICT received config: {cfg}")
    return original_compute_ict_features(df, cfg)

# Apply patches
volume_profile.compute_volume_profile_features = patched_compute_volume_profile_features
ict.compute_ict_features = patched_compute_ict_features

from volume_price_trade.features.feature_union import build_feature_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data(n_bars: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    start_date = pd.Timestamp('2023-01-02 09:30:00-05:00')
    dates = pd.date_range(start=start_date, periods=n_bars, freq='5min')
    
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n_bars):
        if i % 15 == 0:
            gap_size = np.random.choice([-0.8, 0.8])
            new_price = prices[-1] * (1 + gap_size/100)
        else:
            price_change = np.random.normal(0, 0.004)
            new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    data = {
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_bars))),
        'close': prices,
        'volume': np.random.lognormal(12, 1.0, n_bars).astype(int)
    }
    
    for i in range(n_bars):
        data['high'][i] = max(data['open'][i], data['high'][i], data['close'][i])
        data['low'][i] = min(data['open'][i], data['low'][i], data['close'][i])
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_feature_union_config_routing():
    """Test configuration routing through feature_union."""
    logger.info("=" * 60)
    logger.info("TESTING FEATURE UNION CONFIGURATION ROUTING")
    logger.info("=" * 60)
    
    # Load config
    config_path = Path("config/base.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Original config features.volume_profile: {config['features']['volume_profile']}")
    logger.info(f"Original config features.ict: {config['features']['ict']}")
    
    # Create test data
    df = create_sample_data(100)
    
    # Test 1: Default configuration
    logger.info("\n--- Test 1: Default configuration ---")
    default_features = build_feature_matrix(df, config)
    
    # Test 2: Modified configuration
    logger.info("\n--- Test 2: Modified configuration ---")
    modified_config = yaml.safe_load(yaml.dump(config))  # Deep copy
    
    # Modify parameters
    modified_config['features']['volume_profile']['bin_size'] = 0.2
    modified_config['features']['ict']['fvg_min_size_atr'] = 0.1
    
    logger.info(f"Modified config features.volume_profile: {modified_config['features']['volume_profile']}")
    logger.info(f"Modified config features.ict: {modified_config['features']['ict']}")
    
    modified_features = build_feature_matrix(df, modified_config)
    
    # Compare results
    logger.info("\n--- Results Comparison ---")
    
    # Volume Profile comparison
    if 'vp_poc' in default_features.columns and 'vp_poc' in modified_features.columns:
        vp_poc_default = default_features['vp_poc'].dropna()
        vp_poc_modified = modified_features['vp_poc'].dropna()
        
        if len(vp_poc_default) > 0 and len(vp_poc_modified) > 0:
            logger.info(f"Default VP POC (first): {vp_poc_default.iloc[0]}")
            logger.info(f"Modified VP POC (first): {vp_poc_modified.iloc[0]}")
            logger.info(f"VP POC changed: {vp_poc_default.iloc[0] != vp_poc_modified.iloc[0]}")
    
    # ICT comparison
    if 'ict_fvg_up' in default_features.columns and 'ict_fvg_up' in modified_features.columns:
        fvg_up_default = default_features['ict_fvg_up'].sum()
        fvg_up_modified = modified_features['ict_fvg_up'].sum()
        
        logger.info(f"Default FVG up count: {fvg_up_default}")
        logger.info(f"Modified FVG up count: {fvg_up_modified}")
        logger.info(f"FVG changed: {fvg_up_default != fvg_up_modified}")

def main():
    """Main function."""
    test_feature_union_config_routing()
    
    logger.info("\n" + "=" * 60)
    logger.info("DEBUGGING COMPLETE")
    logger.info("=" * 60)
    
    # Restore original functions
    volume_profile.compute_volume_profile_features = original_compute_volume_profile_features
    ict.compute_ict_features = original_compute_ict_features

if __name__ == "__main__":
    main()