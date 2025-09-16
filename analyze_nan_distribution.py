#!/usr/bin/env python3
"""
Comprehensive analysis of NaN distribution across all features.
This script will help identify the root causes of NaN values in the feature pipeline.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import sys
import os
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from volume_price_trade.features.feature_union import build_feature_matrix
from volume_price_trade.features.ta_basic import compute_ta_features
from volume_price_trade.features.volume_profile import compute_volume_profile_features
from volume_price_trade.features.vpa import compute_vpa_features
from volume_price_trade.features.ict import compute_ict_features
from volume_price_trade.features.time_of_day import compute_time_of_day_features
from volume_price_trade.features.vwap import compute_vwap_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML files."""
    config_path = Path("config/base.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    features_path = Path("config/features.yaml")
    if features_path.exists():
        with open(features_path, "r") as f:
            features_config = yaml.safe_load(f)
        config.update(features_config)
    
    return config

def create_sample_data(n_bars=200):
    """Create sample OHLCV data for testing."""
    logger.info(f"Creating sample OHLCV data with {n_bars} bars")
    
    # Create datetime index (trading days only)
    start_date = pd.Timestamp('2023-01-02 09:30:00-05:00')  # Start on a trading day
    dates = pd.date_range(start=start_date, periods=n_bars, freq='5min')
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducible results
    base_price = 100.0
    
    # Generate price movements with some autocorrelation
    returns = np.random.normal(0, 0.001, n_bars)
    prices = [base_price]
    
    for i in range(1, n_bars):
        # Add some momentum
        momentum = 0.1 * returns[i-1] if i > 0 else 0
        price_change = returns[i] + momentum
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    data = {
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_bars))),
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, n_bars).astype(int)
    }
    
    # Ensure high >= low and high/low include open/close
    for i in range(n_bars):
        data['high'][i] = max(data['open'][i], data['high'][i], data['close'][i])
        data['low'][i] = min(data['open'][i], data['low'][i], data['close'][i])
    
    df = pd.DataFrame(data, index=dates)
    logger.info(f"Created sample data with shape: {df.shape}")
    return df

def analyze_nan_distribution(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze NaN distribution across all features."""
    logger.info("Analyzing NaN distribution across all features")
    
    # Build complete feature matrix
    feature_matrix = build_feature_matrix(df, config)
    
    # Overall NaN statistics
    total_cells = feature_matrix.shape[0] * feature_matrix.shape[1]
    total_nans = feature_matrix.isna().sum().sum()
    overall_nan_percentage = (total_nans / total_cells) * 100
    
    logger.info(f"Overall NaN percentage: {overall_nan_percentage:.2f}% ({total_nans}/{total_cells})")
    
    # NaN analysis by feature category
    feature_categories = {
        'ta_features': [col for col in feature_matrix.columns if col.startswith(('true_range', 'atr_', 'rvol_', 'log_return', 'pct_return', 'close_mean', 'close_std', 'close_min', 'close_max', 'volume_mean', 'volume_std', 'volume_min', 'volume_max'))],
        'volume_profile': [col for col in feature_matrix.columns if col.startswith('vp_')],
        'vpa_features': [col for col in feature_matrix.columns if col.startswith('vpa_')],
        'ict_features': [col for col in feature_matrix.columns if col.startswith('ict_')],
        'time_of_day': [col for col in feature_matrix.columns if col in ['minute_of_day', 'sin_minute', 'cos_minute', 'hour_of_day', 'minute_of_hour', 'is_rth', 'time_to_close', 'time_since_open', 'is_ny_open', 'is_lunch', 'is_pm_drive']],
        'vwap_features': [col for col in feature_matrix.columns if col.startswith(('vwap_', 'dist_close_to_vwap_', 'above_vwap_', 'vwap_cross_'))]
    }
    
    category_analysis = {}
    
    for category, columns in feature_categories.items():
        if columns:
            category_df = feature_matrix[columns]
            category_cells = category_df.shape[0] * category_df.shape[1]
            category_nans = category_df.isna().sum().sum()
            category_nan_percentage = (category_nans / category_cells) * 100
            
            category_analysis[category] = {
                'nan_count': category_nans,
                'nan_percentage': category_nan_percentage,
                'feature_count': len(columns),
                'columns': columns
            }
            
            logger.info(f"{category}: {category_nan_percentage:.2f}% NaN ({category_nans}/{category_cells})")
    
    # Individual feature analysis
    individual_analysis = {}
    for col in feature_matrix.columns:
        if col not in ['open', 'high', 'low', 'close', 'volume']:  # Skip original OHLCV
            nan_count = feature_matrix[col].isna().sum()
            nan_percentage = (nan_count / len(feature_matrix)) * 100
            individual_analysis[col] = {
                'nan_count': nan_count,
                'nan_percentage': nan_percentage
            }
    
    # Sort by NaN percentage (descending)
    sorted_individual = sorted(individual_analysis.items(), key=lambda x: x[1]['nan_percentage'], reverse=True)
    
    logger.info("\nTop 10 features with highest NaN percentage:")
    for col, analysis in sorted_individual[:10]:
        logger.info(f"  {col}: {analysis['nan_percentage']:.2f}% ({analysis['nan_count']} values)")
    
    return {
        'overall_nan_percentage': overall_nan_percentage,
        'total_nan_count': total_nans,
        'total_cells': total_cells,
        'category_analysis': category_analysis,
        'individual_analysis': individual_analysis,
        'feature_matrix': feature_matrix
    }

def investigate_vp_dist_to_poc_atr_issue(df: pd.DataFrame, config: Dict[str, Any]):
    """Investigate the specific vp_dist_to_poc_atr NaN issue."""
    logger.info("\n" + "="*60)
    logger.info("INVESTIGATING vp_dist_to_poc_atr NaN ISSUE")
    logger.info("="*60)
    
    # Compute volume profile features
    vp_features = compute_volume_profile_features(df, config.get('volume_profile', {}))
    
    # Check vp_dist_to_poc_atr column specifically
    if 'vp_dist_to_poc_atr' in vp_features.columns:
        vp_dist_series = vp_features['vp_dist_to_poc_atr']
        nan_count = vp_dist_series.isna().sum()
        total_count = len(vp_dist_series)
        nan_percentage = (nan_count / total_count) * 100
        
        logger.info(f"vp_dist_to_poc_atr analysis:")
        logger.info(f"  - Total values: {total_count}")
        logger.info(f"  - NaN values: {nan_count} ({nan_percentage:.2f}%)")
        logger.info(f"  - Non-NaN values: {total_count - nan_count}")
        
        # Check where NaN values are located
        if nan_count > 0:
            nan_indices = vp_dist_series[vp_dist_series.isna()].index
            logger.info(f"  - NaN locations: {len(nan_indices)} positions")
            
            # Check if NaNs are at the beginning (warm-up) or distributed
            first_valid_idx = vp_dist_series.first_valid_index()
            last_valid_idx = vp_dist_series.last_valid_index()
            
            logger.info(f"  - First valid index: {first_valid_idx}")
            logger.info(f"  - Last valid index: {last_valid_idx}")
            logger.info(f"  - First dataframe index: {vp_dist_series.index[0]}")
            logger.info(f"  - Last dataframe index: {vp_dist_series.index[-1]}")
            
            # Analyze the cause of NaN values
            logger.info("\nAnalyzing causes of NaN values:")
            
            # Check ATR availability
            atr_window = config.get('volume_profile', {}).get('atr_window', 20)
            logger.info(f"  - ATR window used: {atr_window}")
            
            # Check if ATR values are available
            from volume_price_trade.features.ta_basic import atr
            atr_values = atr(df, window=atr_window)
            atr_nan_count = atr_values.isna().sum()
            logger.info(f"  - ATR NaN count: {atr_nan_count}")
            
            # Check POC availability
            if 'vp_poc' in vp_features.columns:
                poc_nan_count = vp_features['vp_poc'].isna().sum()
                logger.info(f"  - POC NaN count: {poc_nan_count}")
            
            # Check if NaN values correspond to missing ATR or POC
            sample_nan_indices = nan_indices[:min(5, len(nan_indices))]
            logger.info(f"\nSample NaN indices and their causes:")
            for idx in sample_nan_indices:
                atr_val = atr_values.loc[idx] if idx in atr_values.index else 'N/A'
                poc_val = vp_features.loc[idx, 'vp_poc'] if 'vp_poc' in vp_features.columns else 'N/A'
                logger.info(f"  - {idx}: ATR={atr_val}, POC={poc_val}")

def investigate_atr_calculation(df: pd.DataFrame, config: Dict[str, Any]):
    """Investigate ATR calculation issues."""
    logger.info("\n" + "="*60)
    logger.info("INVESTIGATING ATR CALCULATION")
    logger.info("="*60)
    
    from volume_price_trade.features.ta_basic import atr, true_range
    
    # Test different ATR windows
    atr_windows = [5, 10, 20, 30]
    
    for window in atr_windows:
        atr_values = atr(df, window=window)
        nan_count = atr_values.isna().sum()
        total_count = len(atr_values)
        nan_percentage = (nan_count / total_count) * 100
        
        logger.info(f"ATR window {window}: {nan_count}/{total_count} NaN values ({nan_percentage:.2f}%)")
        
        if nan_count > 0:
            first_valid = atr_values.first_valid_index()
            logger.info(f"  - First valid value at: {first_valid}")
            logger.info(f"  - Warm-up period: {atr_values.index.get_loc(first_valid)} bars")

def investigate_volume_profile_data_requirements(df: pd.DataFrame, config: Dict[str, Any]):
    """Investigate volume profile data requirements and edge cases."""
    logger.info("\n" + "="*60)
    logger.info("INVESTIGATING VOLUME PROFILE DATA REQUIREMENTS")
    logger.info("="*60)
    
    # Check RTH filtering
    from volume_price_trade.data.calendar import is_rth
    
    rth_mask = [is_rth(idx) for idx in df.index]
    rth_count = sum(rth_mask)
    total_count = len(df)
    
    logger.info(f"RTH data analysis:")
    logger.info(f"  - Total bars: {total_count}")
    logger.info(f"  - RTH bars: {rth_count} ({rth_count/total_count*100:.2f}%)")
    logger.info(f"  - Non-RTH bars: {total_count - rth_count} ({(total_count-rth_count)/total_count*100:.2f}%)")
    
    # Check rolling sessions requirement
    rolling_sessions = config.get('volume_profile', {}).get('rolling_sessions', 20)
    logger.info(f"Rolling sessions required: {rolling_sessions}")
    
    # Group by date to see session distribution
    df_dates = df.copy()
    df_dates['date'] = df_dates.index.date
    unique_dates = df_dates['date'].unique()
    
    logger.info(f"Unique dates in dataset: {len(unique_dates)}")
    
    if len(unique_dates) < rolling_sessions:
        logger.warning(f"Insufficient dates for rolling sessions: need {rolling_sessions}, have {len(unique_dates)}")
    
    # Check each date's RTH data
    for date in unique_dates[:5]:  # Check first 5 dates
        date_data = df_dates[df_dates['date'] == date]
        date_rth_count = sum([is_rth(idx) for idx in date_data.index])
        logger.info(f"  - Date {date}: {date_rth_count} RTH bars")

def main():
    """Main function to run all NaN analysis."""
    logger.info("COMPREHENSIVE NaN ANALYSIS")
    logger.info("="*80)
    
    # Load configuration
    config = load_config()
    
    # Create test data
    df = create_sample_data(200)
    
    # Run comprehensive NaN analysis
    analysis_results = analyze_nan_distribution(df, config)
    
    # Investigate specific issues
    investigate_vp_dist_to_poc_atr_issue(df, config)
    investigate_atr_calculation(df, config)
    investigate_volume_profile_data_requirements(df, config)
    
    # Save detailed analysis
    with open('nan_analysis_results.txt', 'w') as f:
        f.write("COMPREHENSIVE NaN ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Overall NaN percentage: {analysis_results['overall_nan_percentage']:.2f}%\n")
        f.write(f"Total NaN count: {analysis_results['total_nan_count']}\n")
        f.write(f"Total cells: {analysis_results['total_cells']}\n\n")
        
        f.write("CATEGORY ANALYSIS:\n")
        f.write("-"*40 + "\n")
        for category, data in analysis_results['category_analysis'].items():
            f.write(f"{category}: {data['nan_percentage']:.2f}% NaN ({data['nan_count']}/{data['feature_count']} features)\n")
        
        f.write("\nTOP 10 FEATURES WITH HIGHEST NaN PERCENTAGE:\n")
        f.write("-"*50 + "\n")
        sorted_individual = sorted(analysis_results['individual_analysis'].items(), key=lambda x: x[1]['nan_percentage'], reverse=True)
        for col, analysis in sorted_individual[:10]:
            f.write(f"{col}: {analysis['nan_percentage']:.2f}% ({analysis['nan_count']} values)\n")
    
    logger.info(f"\nAnalysis complete. Results saved to nan_analysis_results.txt")
    
    return analysis_results

if __name__ == "__main__":
    results = main()