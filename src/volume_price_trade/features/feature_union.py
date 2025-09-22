"""Merge feature sets; schema/version validation."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
from .ta_basic import compute_ta_features
from .volume_profile import compute_volume_profile_features
from .vpa import compute_vpa_features
from .ict import compute_ict_features
from .time_of_day import compute_time_of_day_features
from .vwap import compute_vwap_features
from ..utils.nan_imputation import apply_nan_imputation, create_custom_imputation_config

# Configure logger
logger = logging.getLogger(__name__)


def validate_input_data(df: pd.DataFrame) -> bool:
    """
    Validate input DataFrame structure.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check if DataFrame is empty
    if df.empty:
        logger.error("Input DataFrame is empty")
        return False
    
    # Check for required OHLCV columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check if index is datetime or can be converted to datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            logger.info("Converted index to datetime")
        except Exception as e:
            logger.error(f"Failed to convert index to datetime: {e}")
            return False
    
    # Check if index is monotonic increasing
    if not df.index.is_monotonic_increasing:
        logger.warning("Index is not monotonic increasing, sorting DataFrame")
        df = df.sort_index()
    
    return True


def align_features(feature_dfs: List[pd.DataFrame], original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align multiple feature DataFrames with the original DataFrame.
    
    Args:
        feature_dfs: List of feature DataFrames to align
        original_df: Original DataFrame with timestamp index
        
    Returns:
        Aligned DataFrame with all features
    """
    if not feature_dfs:
        logger.warning("No feature DataFrames to align")
        return original_df.copy()
    
    # Start with a copy of the original DataFrame
    aligned_df = original_df.copy()
    
    # Merge each feature DataFrame using left join to preserve original timestamps
    for i, feature_df in enumerate(feature_dfs):
        if feature_df is None or feature_df.empty:
            logger.warning(f"Feature DataFrame {i} is empty, skipping")
            continue
            
        try:
            # Get feature columns (exclude original OHLCV columns)
            feature_columns = [col for col in feature_df.columns
                             if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            if not feature_columns:
                logger.warning(f"No feature columns found in DataFrame {i}")
                continue
                
            # Select only feature columns to avoid duplicating OHLCV data
            features_to_merge = feature_df[feature_columns]
            
            # Merge with aligned DataFrame using left join on index
            aligned_df = aligned_df.merge(
                features_to_merge,
                left_index=True,
                right_index=True,
                how='left',
                suffixes=('', f'_dup_{i}')
            )
            
            # Handle any duplicate columns that might have been created
            dup_columns = [col for col in aligned_df.columns if col.endswith(f'_dup_{i}')]
            for dup_col in dup_columns:
                orig_col = dup_col.replace(f'_dup_{i}', '')
                # If original column exists, use the non-duplicate version
                if orig_col in aligned_df.columns:
                    aligned_df.drop(dup_col, axis=1, inplace=True)
            
        except Exception as e:
            logger.error(f"Error aligning feature DataFrame {i}: {e}")
            continue
    
    return aligned_df


def get_feature_column_order() -> List[str]:
    """
    Return stable column order for features.
    
    Returns:
        List of column names in desired order
    """
    # Define column order by feature module
    column_order = []
    
    # Original OHLCV columns (if present)
    column_order.extend(['open', 'high', 'low', 'close', 'volume'])
    
    # TA features
    ta_columns = [
        'true_range', 'atr_20', 'rvol_5', 'rvol_20',
        'log_return', 'pct_return',
        'close_mean_5', 'close_std_5', 'close_min_5', 'close_max_5',
        'close_mean_10', 'close_std_10', 'close_min_10', 'close_max_10',
        'close_mean_20', 'close_std_20', 'close_min_20', 'close_max_20',
        'volume_mean_5', 'volume_std_5', 'volume_min_5', 'volume_max_5',
        'volume_mean_10', 'volume_std_10', 'volume_min_10', 'volume_max_10',
        'volume_mean_20', 'volume_std_20', 'volume_min_20', 'volume_max_20'
    ]
    column_order.extend(ta_columns)
    
    # Volume Profile features
    vp_columns = [
        'vp_poc', 'vp_vah', 'vp_val', 'vp_dist_to_poc_atr',
        'vp_inside_value', 'vp_hvn_near', 'vp_lvn_near', 'vp_poc_shift_dir'
    ]
    column_order.extend(vp_columns)
    
    # VPA features
    vpa_columns = [
        'vpa_climax_up', 'vpa_climax_down', 'vpa_vdu',
        'vpa_churn', 'vpa_effort_no_result', 'vpa_breakout_conf'
    ]
    column_order.extend(vpa_columns)
    
    # ICT features
    ict_columns = [
        'ict_fvg_up', 'ict_fvg_down', 'ict_liquidity_sweep_up',
        'ict_liquidity_sweep_down', 'ict_displacement_up', 'ict_displacement_down',
        'ict_dist_to_eq', 'ict_killzone_ny_open', 'ict_killzone_lunch',
        'ict_killzone_pm_drive'
    ]
    column_order.extend(ict_columns)
    
    # Time of Day features
    tod_columns = [
        'minute_of_day', 'sin_minute', 'cos_minute',
        'hour_of_day', 'minute_of_hour', 'is_rth',
        'time_to_close', 'time_since_open', 'is_ny_open',
        'is_lunch', 'is_pm_drive'
    ]
    column_order.extend(tod_columns)
    
    # VWAP features
    vwap_columns = [
        'vwap_session', 'vwap_rolling_20', 'dist_close_to_vwap_session_atr',
        'above_vwap_session', 'vwap_cross_up', 'vwap_cross_down'
    ]
    column_order.extend(vwap_columns)
    
    return column_order


def check_future_leakage(feature_df: pd.DataFrame) -> bool:
    """
    Validate that no future data is used in calculations.
    
    Args:
        feature_df: DataFrame with features to check
        
    Returns:
        True if no future leakage detected, False otherwise
    """
    # This is a basic check - more sophisticated checks may be needed
    # depending on specific feature calculations
    
    # Check for NaN values at the beginning (expected for rolling calculations)
    # but not at the end (which might indicate future leakage)
    for col in feature_df.columns:
        # Skip original OHLCV columns
        if col in ['open', 'high', 'low', 'close', 'volume']:
            continue
            
        col_data = feature_df[col]
        
        # Check if there are NaN values at the end of the series
        # (which might indicate future leakage)
        if col_data.isna().any():
            last_valid_idx = col_data.last_valid_index()
            if last_valid_idx != feature_df.index[-1]:
                logger.warning(f"Column {col} has NaN values at the end, possible future leakage")
                return False
    
    return True


def build_feature_matrix(df: pd.DataFrame, cfg: Dict[str, Any], ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Build complete feature matrix from all feature modules.
    
    Args:
        df: DataFrame with OHLCV data and timestamp index
        cfg: Configuration dictionary
        ticker: Optional ticker for caching
        
    Returns:
        DataFrame with all features aligned by timestamp
    """
    # Try to load from cache first
    if ticker:
        from .cache import load_cached_features
        cached_df, version_hash = load_cached_features(ticker, cfg)
        if cached_df is not None:
            logger.info(f"Using cached features for {ticker}")
            return cached_df

    # Validate input data
    if not validate_input_data(df):
        logger.error("Input data validation failed")
        return df.copy()
    
    logger.info("Building feature matrix")
    
    # List to store feature DataFrames from each module
    feature_dfs = []
    
    # 1. Compute TA features first (needed by other modules)
    logger.info("Computing TA features")
    # Extract TA configuration from features section
    features_config = cfg.get('features', {})
    enable_cfg = features_config.get('enable', {})
    en_ta = bool(enable_cfg.get('ta', True))
    en_vp = bool(enable_cfg.get('volume_profile', True))
    en_vpa = bool(enable_cfg.get('vpa', True))
    en_ict = bool(enable_cfg.get('ict', True))
    en_tod = bool(enable_cfg.get('time_of_day', True))
    en_vwap = bool(enable_cfg.get('vwap', True))

    ta_config = {
        'atr_window': features_config.get('atr_window', 20),
        'rvol_windows': features_config.get('rvol_windows', [5, 20])
    }

    # Create a working DataFrame with TA features for dependent modules
    working_df = df.copy()

    if en_ta:
        t0 = time.perf_counter()
        ta_features = compute_ta_features(df, ta_config)
        logger.info(f"TA features shape: {ta_features.shape} (took {time.perf_counter()-t0:.2f}s)")
        feature_dfs.append(ta_features)

        ta_feature_cols = [col for col in ta_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        for col in ta_feature_cols:
            working_df[col] = ta_features[col]
    else:
        logger.info("TA features disabled via config")
    
    # 2. Compute Volume Profile features
    if en_vp:
        logger.info("Computing Volume Profile features")
        t0 = time.perf_counter()
        vp_features = compute_volume_profile_features(df, cfg.get('features', {}).get('volume_profile', {}))
        logger.info(f"Volume Profile features shape: {vp_features.shape} (took {time.perf_counter()-t0:.2f}s)")
        feature_dfs.append(vp_features)
    else:
        logger.info("Volume Profile features disabled via config")
    
    # 3. Compute VPA features (depends on TA features)
    if en_vpa:
        logger.info("Computing VPA features")
        t0 = time.perf_counter()
        vpa_features = compute_vpa_features(working_df, cfg.get('vpa', {}))
        logger.info(f"VPA features shape: {vpa_features.shape} (took {time.perf_counter()-t0:.2f}s)")
        feature_dfs.append(vpa_features)
    else:
        logger.info("VPA features disabled via config")
    
    # 4. Compute ICT features (depends on TA features)
    if en_ict:
        logger.info("Computing ICT features")
        t0 = time.perf_counter()
        ict_features = compute_ict_features(working_df, cfg.get('features', {}).get('ict', {}))
        logger.info(f"ICT features shape: {ict_features.shape} (took {time.perf_counter()-t0:.2f}s)")
        feature_dfs.append(ict_features)
    else:
        logger.info("ICT features disabled via config")
    
    # 5. Compute Time of Day features
    if en_tod:
        logger.info("Computing Time of Day features")
        t0 = time.perf_counter()
        tod_features = compute_time_of_day_features(df, cfg.get('time_of_day', {}))
        logger.info(f"Time of Day features shape: {tod_features.shape} (took {time.perf_counter()-t0:.2f}s)")
        feature_dfs.append(tod_features)
    else:
        logger.info("Time of Day features disabled via config")
    
    # 6. Compute VWAP features
    if en_vwap:
        logger.info("Computing VWAP features")
        t0 = time.perf_counter()
        vwap_features = compute_vwap_features(df, cfg.get('vwap', {}))
        logger.info(f"VWAP features shape: {vwap_features.shape} (took {time.perf_counter()-t0:.2f}s)")
        feature_dfs.append(vwap_features)
    else:
        logger.info("VWAP features disabled via config")
    
    # Align all features
    logger.info("Aligning features")
    feature_matrix = align_features(feature_dfs, df)
    
    # Check for future leakage
    if not check_future_leakage(feature_matrix):
        logger.warning("Potential future leakage detected in feature matrix")
    
    # Reorder columns according to defined order
    logger.info("Reordering columns")
    desired_order = get_feature_column_order()
    
    # Filter to only include columns that exist in the DataFrame
    existing_columns = [col for col in desired_order if col in feature_matrix.columns]
    
    # Add any columns not in the desired order at the end
    additional_columns = [col for col in feature_matrix.columns
                         if col not in existing_columns and col not in ['open', 'high', 'low', 'close', 'volume']]
    
    # Create final column order
    final_column_order = existing_columns + additional_columns
    
    # Reorder columns
    feature_matrix = feature_matrix[final_column_order]
    
    logger.info(f"Feature matrix built with {len(feature_matrix.columns)} features")
    
    # Apply NaN imputation to reduce overall NaN percentage
    logger.info("Applying NaN imputation")
    
    # Get custom imputation configuration
    custom_imputation_config = create_custom_imputation_config()
    
    # Apply imputation
    feature_matrix = apply_nan_imputation(
        feature_matrix,
        exclude_columns=['open', 'high', 'low', 'close', 'volume'],
        custom_methods=custom_imputation_config,
        rolling_window=20,
        min_periods=5
    )
    
    logger.info("Feature matrix construction complete")

    # Save to cache
    if ticker:
        from .cache import save_cached_features, _hash_config
        version_hash = _hash_config(cfg)
        save_cached_features(ticker, feature_matrix, cfg, version_hash)

    return feature_matrix
    # Validate input data
    if not validate_input_data(df):
        logger.error("Input data validation failed")
        return df.copy()
    
    logger.info("Building feature matrix")
    
    # List to store feature DataFrames from each module
    feature_dfs = []
    
    # 1. Compute TA features first (needed by other modules)
    logger.info("Computing TA features")
    # Extract TA configuration from features section
    features_config = cfg.get('features', {})
    ta_config = {
        'atr_window': features_config.get('atr_window', 20),
        'rvol_windows': features_config.get('rvol_windows', [5, 20])
    }
    ta_features = compute_ta_features(df, ta_config)
    logger.info(f"TA features shape: {ta_features.shape}")
    feature_dfs.append(ta_features)
    
    # Create a working DataFrame with TA features for dependent modules
    working_df = df.copy()
    ta_feature_cols = [col for col in ta_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    for col in ta_feature_cols:
        working_df[col] = ta_features[col]
    
    # 2. Compute Volume Profile features
    logger.info("Computing Volume Profile features")
    vp_features = compute_volume_profile_features(df, cfg.get('features', {}).get('volume_profile', {}))
    logger.info(f"Volume Profile features shape: {vp_features.shape}")
    feature_dfs.append(vp_features)
    
    # 3. Compute VPA features (depends on TA features)
    logger.info("Computing VPA features")
    vpa_features = compute_vpa_features(working_df, cfg.get('vpa', {}))
    logger.info(f"VPA features shape: {vpa_features.shape}")
    feature_dfs.append(vpa_features)
    
    # 4. Compute ICT features (depends on TA features)
    logger.info("Computing ICT features")
    ict_features = compute_ict_features(working_df, cfg.get('features', {}).get('ict', {}))
    logger.info(f"ICT features shape: {ict_features.shape}")
    feature_dfs.append(ict_features)
    
    # 5. Compute Time of Day features
    logger.info("Computing Time of Day features")
    tod_features = compute_time_of_day_features(df, cfg.get('time_of_day', {}))
    logger.info(f"Time of Day features shape: {tod_features.shape}")
    feature_dfs.append(tod_features)
    
    # 6. Compute VWAP features
    logger.info("Computing VWAP features")
    vwap_features = compute_vwap_features(df, cfg.get('vwap', {}))
    logger.info(f"VWAP features shape: {vwap_features.shape}")
    feature_dfs.append(vwap_features)
    
    # Align all features
    logger.info("Aligning features")
    feature_matrix = align_features(feature_dfs, df)
    
    # Check for future leakage
    if not check_future_leakage(feature_matrix):
        logger.warning("Potential future leakage detected in feature matrix")
    
    # Reorder columns according to defined order
    logger.info("Reordering columns")
    desired_order = get_feature_column_order()
    
    # Filter to only include columns that exist in the DataFrame
    existing_columns = [col for col in desired_order if col in feature_matrix.columns]
    
    # Add any columns not in the desired order at the end
    additional_columns = [col for col in feature_matrix.columns
                         if col not in existing_columns and col not in ['open', 'high', 'low', 'close', 'volume']]
    
    # Create final column order
    final_column_order = existing_columns + additional_columns
    
    # Reorder columns
    feature_matrix = feature_matrix[final_column_order]
    
    logger.info(f"Feature matrix built with {len(feature_matrix.columns)} features")
    
    # Apply NaN imputation to reduce overall NaN percentage
    logger.info("Applying NaN imputation")
    
    # Get custom imputation configuration
    custom_imputation_config = create_custom_imputation_config()
    
    # Apply imputation
    feature_matrix = apply_nan_imputation(
        feature_matrix,
        exclude_columns=['open', 'high', 'low', 'close', 'volume'],
        custom_methods=custom_imputation_config,
        rolling_window=20,
        min_periods=5
    )
    
    logger.info("Feature matrix construction complete")
    
    return feature_matrix
