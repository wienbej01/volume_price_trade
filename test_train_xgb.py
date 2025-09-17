"""Test script for train_xgb function."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.volume_price_trade.ml.models import train_xgb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_train_xgb():
    """Test the train_xgb function with synthetic data."""
    logger.info("Starting test for train_xgb function")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Convert to pandas DataFrame and Series
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Create metadata DataFrame
    dates = pd.date_range(start='2020-01-01', periods=len(X_df), freq='D')
    tickers = ['AAPL'] * len(X_df)
    meta_df = pd.DataFrame({
        'ticker': tickers,
        'session_date': dates
    }, index=X_df.index)
    
    # Create simple train/validation splits (simulating walk-forward CV)
    n_splits = 3
    split_size = len(X_df) // (n_splits + 1)
    splits = []
    
    for i in range(n_splits):
        train_start = i * split_size
        train_end = (i + 1) * split_size
        val_start = train_end
        val_end = (i + 2) * split_size
        
        if val_end > len(X_df):
            val_end = len(X_df)
        
        train_idx = np.arange(train_start, train_end)
        val_idx = np.arange(val_start, val_end)
        
        splits.append((train_idx, val_idx))
    
    # Test with XGBoost
    logger.info("Testing with XGBoost")
    xgb_config = {
        'model': {
            'type': 'xgboost',
            'params': {
                'n_estimators': 100,
                'max_depth': 3,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
        }
    }
    
    try:
        xgb_model, xgb_metrics = train_xgb(X_df, y_series, meta_df, splits, xgb_config)
        logger.info(f"XGBoost training successful. Metrics: {xgb_metrics}")
        print(f"XGBoost average accuracy: {np.mean(xgb_metrics['accuracy']):.4f}")
        print(f"XGBoost average F1: {np.mean(xgb_metrics['f1']):.4f}")
        print(f"XGBoost average AUC: {np.mean(xgb_metrics['auc']):.4f}")
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Test with LightGBM
    logger.info("Testing with LightGBM")
    lgb_config = {
        'model': {
            'type': 'lightgbm',
            'params': {
                'n_estimators': 100,
                'max_depth': 3,
                'random_state': 42,
                'verbose': -1
            }
        }
    }
    
    try:
        lgb_model, lgb_metrics = train_xgb(X_df, y_series, meta_df, splits, lgb_config)
        logger.info(f"LightGBM training successful. Metrics: {lgb_metrics}")
        print(f"LightGBM average accuracy: {np.mean(lgb_metrics['accuracy']):.4f}")
        print(f"LightGBM average F1: {np.mean(lgb_metrics['f1']):.4f}")
        print(f"LightGBM average AUC: {np.mean(lgb_metrics['auc']):.4f}")
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Test with multiclass classification
    logger.info("Testing with multiclass classification")
    X_multi, y_multi = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    X_multi_df = pd.DataFrame(X_multi, columns=[f'feature_{i}' for i in range(X_multi.shape[1])])
    y_multi_series = pd.Series(y_multi, name='target')
    
    try:
        multi_model, multi_metrics = train_xgb(X_multi_df, y_multi_series, meta_df, splits, xgb_config)
        logger.info(f"Multiclass training successful. Metrics: {multi_metrics}")
        print(f"Multiclass average accuracy: {np.mean(multi_metrics['accuracy']):.4f}")
        print(f"Multiclass average F1: {np.mean(multi_metrics['f1']):.4f}")
        # Note: AUC is not calculated for multiclass in our implementation
    except Exception as e:
        logger.error(f"Multiclass training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    logger.info("All tests passed successfully!")
    return True

if __name__ == "__main__":
    test_train_xgb()