#!/usr/bin/env python3
"""
Debug script to trace configuration flow to individual modules.
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
from volume_price_trade.features.volume_profile import compute_volume_profile_features
from volume_price_trade.features.ict import compute_ict_features

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
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
        if i % 15 == 0:  # Create gaps every 15 bars for FVG patterns
            gap_size = np.random.choice([-0.8, 0.8])  # Larger gaps
            new_price = prices[-1] * (1 + gap_size/100)
        else:
            price_change = np.random.normal(0, 0.004)  # Higher volatility
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

def debug_volume_profile_config():
    """Debug volume profile configuration flow."""
    logger.info("=" * 50)
    logger.info("DEBUGGING VOLUME PROFILE CONFIGURATION")
    logger.info("=" * 50)
    
    # Load config
    config_path = Path("config/base.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Original config features.volume_profile: {config['features']['volume_profile']}")
    logger.info(f"Original config features.atr_window: {config['features'].get('atr_window', 'NOT_FOUND')}")
    
    # Create test data
    df = create_sample_data(100)
    
    # Test 1: Direct call with default config
    logger.info("\n--- Test 1: Direct call with default config ---")
    vp_config_default = config['features']['volume_profile'].copy()
    logger.info(f"VP config passed to module: {vp_config_default}")
    
    vp_features_default = compute_volume_profile_features(df, vp_config_default)
    vp_poc_default = vp_features_default['vp_poc'].dropna()
    logger.info(f"VP POC values (first 5): {vp_poc_default.head().tolist()}")
    
    # Test 2: Direct call with modified config
    logger.info("\n--- Test 2: Direct call with modified config ---")
    vp_config_modified = vp_config_default.copy()
    vp_config_modified['bin_size'] = 0.2  # Much larger change
    logger.info(f"Modified VP config passed to module: {vp_config_modified}")
    
    vp_features_modified = compute_volume_profile_features(df, vp_config_modified)
    vp_poc_modified = vp_features_modified['vp_poc'].dropna()
    logger.info(f"Modified VP POC values (first 5): {vp_poc_modified.head().tolist()}")
    
    # Compare
    if len(vp_poc_default) > 0 and len(vp_poc_modified) > 0:
        logger.info(f"VP POC comparison: {vp_poc_default.iloc[0]} vs {vp_poc_modified.iloc[0]}")
        logger.info(f"VP POC changed: {vp_poc_default.iloc[0] != vp_poc_modified.iloc[0]}")

def debug_ict_config():
    """Debug ICT configuration flow."""
    logger.info("=" * 50)
    logger.info("DEBUGGING ICT CONFIGURATION")
    logger.info("=" * 50)
    
    # Load config
    config_path = Path("config/base.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Original config features.ict: {config['features']['ict']}")
    
    # Create test data with TA features (required for ICT)
    df = create_sample_data(100)
    
    # Create TA features first
    from volume_price_trade.features.ta_basic import compute_ta_features
    ta_config = {
        'atr_window': config['features']['atr_window'],
        'rvol_windows': config['features']['rvol_windows']
    }
    ta_features = compute_ta_features(df, ta_config)
    
    working_df = df.copy()
    ta_cols = [col for col in ta_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    for col in ta_cols:
        working_df[col] = ta_features[col]
    
    # Test 1: Direct call with default config
    logger.info("\n--- Test 1: Direct call with default config ---")
    ict_config_default = config['features']['ict'].copy()
    logger.info(f"ICT config passed to module: {ict_config_default}")
    
    ict_features_default = compute_ict_features(working_df, ict_config_default)
    fvg_up_default = ict_features_default['ict_fvg_up'].sum()
    logger.info(f"FVG up count: {fvg_up_default}")
    
    # Test 2: Direct call with modified config
    logger.info("\n--- Test 2: Direct call with modified config ---")
    ict_config_modified = ict_config_default.copy()
    ict_config_modified['fvg_min_size_atr'] = 0.1  # Much smaller threshold
    logger.info(f"Modified ICT config passed to module: {ict_config_modified}")
    
    ict_features_modified = compute_ict_features(working_df, ict_config_modified)
    fvg_up_modified = ict_features_modified['ict_fvg_up'].sum()
    logger.info(f"Modified FVG up count: {fvg_up_modified}")
    
    # Compare
    logger.info(f"FVG comparison: {fvg_up_default} vs {fvg_up_modified}")
    logger.info(f"FVG changed: {fvg_up_default != fvg_up_modified}")

def main():
    """Main function to debug configuration flow."""
    debug_volume_profile_config()
    debug_ict_config()
    
    logger.info("\n" + "=" * 50)
    logger.info("DEBUGGING COMPLETE")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()