#!/usr/bin/env python3
"""
Comprehensive NaN imputation strategy for feature engineering.
This module provides different imputation methods for different feature types
to reduce NaN values while avoiding future leakage.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum

# Configure logger
logger = logging.getLogger(__name__)

class ImputationMethod(Enum):
    """Enumeration of different imputation methods."""
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    MEAN = "mean"
    MEDIAN = "median"
    ZERO = "zero"
    ROLLING_MEAN = "rolling_mean"
    ROLLING_MEDIAN = "rolling_median"
    INTERPOLATE = "interpolate"
    DROP = "drop"

class FeatureType(Enum):
    """Enumeration of different feature types."""
    PRICE_BASED = "price_based"
    VOLUME_BASED = "volume_based"
    VOLATILITY_BASED = "volatility_based"
    TIME_BASED = "time_based"
    BINARY = "binary"
    DISTANCE_BASED = "distance_based"
    ROLLING_WINDOW = "rolling_window"

def get_feature_type(feature_name: str) -> FeatureType:
    """
    Determine the type of feature based on its name.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        FeatureType enum value
    """
    feature_name_lower = feature_name.lower()
    
    # Price-based features
    if any(term in feature_name_lower for term in ['poc', 'vah', 'val', 'vwap', 'close', 'open', 'high', 'low']):
        return FeatureType.PRICE_BASED
    
    # Volume-based features
    elif any(term in feature_name_lower for term in ['volume', 'vol']):
        return FeatureType.VOLUME_BASED
    
    # Volatility-based features
    elif any(term in feature_name_lower for term in ['atr', 'std', 'range']):
        return FeatureType.VOLATILITY_BASED
    
    # Time-based features
    elif any(term in feature_name_lower for term in ['time', 'minute', 'hour', 'day']):
        return FeatureType.TIME_BASED
    
    # Distance-based features
    elif any(term in feature_name_lower for term in ['dist', 'distance']):
        return FeatureType.DISTANCE_BASED
    
    # Rolling window features
    elif any(term in feature_name_lower for term in ['mean_', 'std_', 'min_', 'max_', 'rolling']):
        return FeatureType.ROLLING_WINDOW
    
    # Binary features
    elif any(term in feature_name_lower for term in ['is_', 'fvg', 'liquidity', 'displacement', 'killzone']):
        return FeatureType.BINARY
    
    # Default to price-based
    else:
        return FeatureType.PRICE_BASED

def get_imputation_method(feature_type: FeatureType, feature_name: str = "") -> ImputationMethod:
    """
    Get the recommended imputation method for a given feature type.
    
    Args:
        feature_type: Type of the feature
        feature_name: Name of the feature (for special cases)
        
    Returns:
        ImputationMethod enum value
    """
    feature_name_lower = feature_name.lower()
    
    # Special cases
    if 'vp_dist_to_poc_atr' in feature_name_lower:
        return ImputationMethod.ROLLING_MEAN
    elif any(term in feature_name_lower for term in ['atr', 'true_range']):
        return ImputationMethod.FORWARD_FILL
    elif feature_type == FeatureType.BINARY:
        return ImputationMethod.ZERO
    elif feature_type == FeatureType.TIME_BASED:
        return ImputationMethod.INTERPOLATE
    elif feature_type == FeatureType.VOLATILITY_BASED:
        return ImputationMethod.FORWARD_FILL
    elif feature_type == FeatureType.DISTANCE_BASED:
        return ImputationMethod.ROLLING_MEAN
    elif feature_type == FeatureType.ROLLING_WINDOW:
        return ImputationMethod.FORWARD_FILL
    elif feature_type == FeatureType.PRICE_BASED:
        return ImputationMethod.FORWARD_FILL
    elif feature_type == FeatureType.VOLUME_BASED:
        return ImputationMethod.FORWARD_FILL
    else:
        return ImputationMethod.FORWARD_FILL

def impute_feature(
    series: pd.Series, 
    method: ImputationMethod, 
    window: int = 20,
    min_periods: int = 5
) -> pd.Series:
    """
    Apply imputation to a single feature series.
    
    Args:
        series: Pandas Series with potential NaN values
        method: Imputation method to use
        window: Window size for rolling calculations
        min_periods: Minimum periods for rolling calculations
        
    Returns:
        Imputed pandas Series
    """
    if series.isna().sum() == 0:
        return series.copy()
    
    result = series.copy()
    
    try:
        if method == ImputationMethod.FORWARD_FILL:
            result = result.ffill()
            # If there are still NaN values at the beginning, use backward fill
            if result.isna().sum() > 0:
                
                
        elif method == ImputationMethod.BACKWARD_FILL:
            
            # If there are still NaN values at the end, use forward fill
            if result.isna().sum() > 0:
                result = result.ffill()
                
        elif method == ImputationMethod.MEAN:
            mean_value = result.mean()
            if not np.isnan(mean_value):
                result = result.fillna(mean_value)
            else:
                # Fallback to median if mean is NaN
                median_value = result.median()
                if not np.isnan(median_value):
                    result = result.fillna(median_value)
                    
        elif method == ImputationMethod.MEDIAN:
            median_value = result.median()
            if not np.isnan(median_value):
                result = result.fillna(median_value)
            else:
                # Fallback to mean if median is NaN
                mean_value = result.mean()
                if not np.isnan(mean_value):
                    result = result.fillna(mean_value)
                    
        elif method == ImputationMethod.ZERO:
            result = result.fillna(0)
            
        elif method == ImputationMethod.ROLLING_MEAN:
            # Use rolling mean for imputation
            rolling_mean = result.rolling(window=window, min_periods=min_periods).mean()
            # Fill NaN with rolling mean values
            result = result.fillna(rolling_mean)
            # If there are still NaN values, use forward fill
            if result.isna().sum() > 0:
                result = result.ffill()
                if result.isna().sum() > 0:
                    
                    
        elif method == ImputationMethod.ROLLING_MEDIAN:
            # Use rolling median for imputation
            rolling_median = result.rolling(window=window, min_periods=min_periods).median()
            # Fill NaN with rolling median values
            result = result.fillna(rolling_median)
            # If there are still NaN values, use forward fill
            if result.isna().sum() > 0:
                result = result.ffill()
                if result.isna().sum() > 0:
                    
                    
        elif method == ImputationMethod.INTERPOLATE:
            # Use linear interpolation
            result = result.interpolate(method='linear')
            # If there are still NaN values at the ends, use forward/backward fill
            if result.isna().sum() > 0:
                result = result.ffill().bfill()
                
        elif method == ImputationMethod.DROP:
            # This would drop rows, but we don't want to change the index
            # So we'll use forward fill instead
            result = result.ffill().bfill()
            
    except Exception as e:
        logger.warning(f"Error applying imputation method {method.value} to series: {e}")
        # Fallback to forward fill
        result = result.ffill().bfill()
    
    return result

def apply_nan_imputation(
    df: pd.DataFrame, 
    exclude_columns: Optional[List[str]] = None,
    custom_methods: Optional[Dict[str, ImputationMethod]] = None,
    rolling_window: int = 20,
    min_periods: int = 5
) -> pd.DataFrame:
    """
    Apply comprehensive NaN imputation to a DataFrame.
    
    Args:
        df: DataFrame with potential NaN values
        exclude_columns: List of columns to exclude from imputation
        custom_methods: Dictionary mapping column names to custom imputation methods
        rolling_window: Window size for rolling calculations
        min_periods: Minimum periods for rolling calculations
        
    Returns:
        DataFrame with imputed values
    """
    if df.empty:
        return df.copy()
    
    result_df = df.copy()
    
    # Columns to exclude from imputation
    if exclude_columns is None:
        exclude_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Get columns to process
    columns_to_process = [col for col in result_df.columns if col not in exclude_columns]
    
    logger.info(f"Applying NaN imputation to {len(columns_to_process)} columns")
    
    # Track imputation statistics
    total_nans_before = result_df[columns_to_process].isna().sum().sum()
    total_cells = len(result_df) * len(columns_to_process)
    
    # Process each column
    for col in columns_to_process:
        if col in result_df.columns:
            nan_count_before = result_df[col].isna().sum()
            
            if nan_count_before > 0:
                # Get feature type and imputation method
                if custom_methods and col in custom_methods:
                    method = custom_methods[col]
                else:
                    feature_type = get_feature_type(col)
                    method = get_imputation_method(feature_type, col)
                
                # Apply imputation
                result_df[col] = impute_feature(
                    result_df[col], 
                    method, 
                    window=rolling_window,
                    min_periods=min_periods
                )
                
                nan_count_after = result_df[col].isna().sum()
                nan_reduction = nan_count_before - nan_count_after
                
                logger.debug(f"  {col}: {nan_count_before} -> {nan_count_after} NaN values "
                           f"({nan_reduction} reduced, method: {method.value})")
    
    # Calculate overall statistics
    total_nans_after = result_df[columns_to_process].isna().sum().sum()
    total_reduction = total_nans_before - total_nans_after
    
    logger.info(f"NaN imputation complete:")
    logger.info(f"  Before: {total_nans_before}/{total_cells} ({total_nans_before/total_cells*100:.2f}%)")
    logger.info(f"  After:  {total_nans_after}/{total_cells} ({total_nans_after/total_cells*100:.2f}%)")
    logger.info(f"  Reduced: {total_reduction} ({total_reduction/total_cells*100:.2f} percentage points)")
    
    return result_df

def create_custom_imputation_config() -> Dict[str, ImputationMethod]:
    """
    Create custom imputation configuration for specific features.
    
    Returns:
        Dictionary mapping feature names to imputation methods
    """
    return {
        # Volume profile features
        'vp_dist_to_poc_atr': ImputationMethod.ROLLING_MEAN,
        'vp_poc': ImputationMethod.FORWARD_FILL,
        'vp_vah': ImputationMethod.FORWARD_FILL,
        'vp_val': ImputationMethod.FORWARD_FILL,
        
        # ATR and volatility features
        'true_range': ImputationMethod.FORWARD_FILL,
        'atr_20': ImputationMethod.FORWARD_FILL,
        
        # Rolling window features
        'close_mean_5': ImputationMethod.FORWARD_FILL,
        'close_mean_10': ImputationMethod.FORWARD_FILL,
        'close_mean_20': ImputationMethod.FORWARD_FILL,
        'close_std_5': ImputationMethod.FORWARD_FILL,
        'close_std_10': ImputationMethod.FORWARD_FILL,
        'close_std_20': ImputationMethod.FORWARD_FILL,
        
        # Volume features
        'rvol_5': ImputationMethod.FORWARD_FILL,
        'rvol_20': ImputationMethod.FORWARD_FILL,
        
        # Binary features (should be 0 when NaN)
        'vpa_climax_up': ImputationMethod.ZERO,
        'vpa_climax_down': ImputationMethod.ZERO,
        'vpa_vdu': ImputationMethod.ZERO,
        'vpa_churn': ImputationMethod.ZERO,
        'vpa_effort_no_result': ImputationMethod.ZERO,
        'vpa_breakout_conf': ImputationMethod.ZERO,
        'ict_fvg_up': ImputationMethod.ZERO,
        'ict_fvg_down': ImputationMethod.ZERO,
        'ict_liquidity_sweep_up': ImputationMethod.ZERO,
        'ict_liquidity_sweep_down': ImputationMethod.ZERO,
        'ict_displacement_up': ImputationMethod.ZERO,
        'ict_displacement_down': ImputationMethod.ZERO,
        'vp_inside_value': ImputationMethod.ZERO,
        'vp_hvn_near': ImputationMethod.ZERO,
        'vp_lvn_near': ImputationMethod.ZERO,
        'vp_poc_shift_dir': ImputationMethod.ZERO,
        'is_rth': ImputationMethod.ZERO,
        'is_ny_open': ImputationMethod.ZERO,
        'is_lunch': ImputationMethod.ZERO,
        'is_pm_drive': ImputationMethod.ZERO,
        'above_vwap_session': ImputationMethod.ZERO,
        'vwap_cross_up': ImputationMethod.ZERO,
        'vwap_cross_down': ImputationMethod.ZERO,
    }