"""XGBoost/LightGBM baselines as pipelines."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Union, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
import logging
import xgboost as xgb
import lightgbm as lgb

# Configure logger
logger = logging.getLogger(__name__)


def train_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    config: Dict[str, Any]
) -> Tuple[Union[xgb.XGBModel, lgb.LGBMModel], Dict[str, List[float]]]:
    """
    Train XGBoost or LightGBM classifier using cross-validation splits.
    
    Args:
        X: DataFrame with features
        y: Series with labels
        meta: DataFrame with metadata (ticker, session_date, etc.)
        splits: List of (train_indices, val_indices) tuples for cross-validation
        config: Configuration dictionary with model settings
        
    Returns:
        Tuple of (trained_model, metrics) where:
        - trained_model: The trained XGBoost or LightGBM model
        - metrics: Dictionary with metrics per fold (e.g., {'accuracy': [0.85, 0.87], ...})
    """
    # Extract model configuration
    model_type = config.get('model', {}).get('type', 'xgboost').lower()
    model_params = config.get('model', {}).get('params', {})
    
    # Validate inputs
    if X.empty or y.empty:
        raise ValueError("X and y cannot be empty")
    
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    if not splits:
        raise ValueError("At least one split must be provided")
    
    # Initialize metrics dictionary
    metrics: Dict[str, List[float]] = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'log_loss': []
    }
    
    # Add AUC for binary classification
    unique_classes = np.unique(y)
    is_binary = len(unique_classes) == 2
    if is_binary:
        metrics['auc'] = []
    
    # Encode labels if they are strings
    label_encoder = None
    y_encoded = y.copy()
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        label_encoder = LabelEncoder()
        y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index)
        logger.info(f"Encoded labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Initialize the model based on type
    model: Any
    if model_type == 'xgboost':
        # Ensure base_score is valid for binary classification
        if is_binary:
            model_params = model_params.copy()
            # Set base_score to a valid value for binary classification
            if 'base_score' not in model_params:
                model_params['base_score'] = 0.5  # Use 0.5 as default for balanced classes

        model = xgb.XGBClassifier(**model_params)
        logger.info(f"Initialized XGBoost model with params: {model_params}")
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(**model_params)
        logger.info(f"Initialized LightGBM model with params: {model_params}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'xgboost', 'lightgbm'")
    
    # Train model on each fold and collect metrics
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Training fold {fold_idx + 1}/{len(splits)}")
        
        try:
            # Split data for this fold
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = np.asarray(model.predict_proba(X_val))

            # Calculate metrics
            fold_metrics = _calculate_metrics(np.asarray(y_val), y_pred, y_pred_proba, is_binary)
            
            # Append fold metrics to overall metrics
            for metric_name, metric_value in fold_metrics.items():
                metrics[metric_name].append(metric_value)
            
            logger.info(f"Fold {fold_idx + 1} metrics: {fold_metrics}")
            
        except Exception as e:
            logger.error(f"Error in fold {fold_idx + 1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    # Train final model on all data
    logger.info("Training final model on all data")
    model.fit(X, y_encoded)
    
    # Log average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    logger.info(f"Average metrics across all folds: {avg_metrics}")
    
    return model, metrics


def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    is_binary: bool
) -> Dict[str, float]:
    """
    Calculate classification metrics for a single fold.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        is_binary: Whether this is a binary classification problem
        
    Returns:
        Dictionary with metric values
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')

    # Log loss with error handling for single-class cases
    try:
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
    except ValueError as e:
        if "only one label" in str(e):
            # Handle case where validation set has only one class
            # Use a reasonable default value
            metrics['log_loss'] = 0.693  # log(2), which is the entropy of a fair coin
            logger.warning(f"Log loss calculation failed due to single class in validation set: {e}")
        else:
            raise
    
    # AUC for binary classification
    if is_binary:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    return metrics
