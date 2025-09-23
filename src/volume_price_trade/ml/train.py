"""Train and persist artifacts."""

import os
import yaml
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import logging
import json
import joblib
from pathlib import Path

from .dataset import make_dataset
from .cv import purged_walk_forward_splits
from .models import train_xgb
from .train_report import generate_train_report
from ..utils.io import save_joblib, save_json, save_yaml
from ..utils.logging import get_logger, setup_run_logger

# Configure logger
logger = get_logger(__name__)


def train_model(config_path: str, sample_days: Optional[int] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Train a model using the provided configuration and save artifacts.
    
    Args:
        config_path: Path to the configuration YAML file
        sample_days: Optional number of days to sample for faster testing
        
    Returns:
        Dictionary with run information including run_id, metrics, and artifact paths
    """
    # Generate unique run ID
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    
    # Set up run-specific logger
    logger = setup_run_logger(run_id)
    logger.info(f"Starting training run {run_id}")

    # Log config and sample days
    logger.info(f"Using config: {config_path}")
    if sample_days:
        logger.info(f"Using sample days: {sample_days}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create artifacts directory
    artifacts_dir = Path(f"artifacts/models/{run_id}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get training tickers from config
        train_tickers = config.get('tickers', {}).get('train', [])
        if not train_tickers:
            raise ValueError("No training tickers found in config")
        
        # Determine date range
        if start_date is not None and end_date is not None:
            # Use provided date range
            start_date_str = start_date
            end_date_str = end_date
            logger.info(f"Using provided date range: {start_date_str} to {end_date_str}")
        else:
            # End date is last day of previous month to ensure data is available
            end_date_dt = datetime.now().replace(day=1) - pd.DateOffset(days=1)
            end_date_str = end_date_dt.strftime('%Y-%m-%d')

            # If sample_days is provided, adjust the start date
            if sample_days is not None and sample_days > 0:
                start_date_dt = end_date_dt - pd.DateOffset(days=sample_days)
                start_date_str = start_date_dt.strftime('%Y-%m-%d')
                logger.info(f"Using sampled data: {sample_days} days from {start_date_str} to {end_date_str}")
            else:
                # Default to 2 years of data
                start_date_dt = end_date_dt - pd.DateOffset(years=2)
                start_date_str = start_date_dt.strftime('%Y-%m-%d')
                logger.info(f"Using default 2-year date range: {start_date_str} to {end_date_str}")
        
        # Load dataset
        logger.info("Loading dataset...")
        X, y, meta = make_dataset(
            tickers=train_tickers,
            start=start_date_str,
            end=end_date_str,
            config=config
        )
        
        if X.empty or y.empty:
            raise ValueError("Failed to load dataset - empty features or labels")
        
        logger.info(f"Dataset loaded: {len(X)} samples, {len(X.columns)} features")
        
        # Generate CV splits
        logger.info("Generating cross-validation splits...")
        cv_splits = list(purged_walk_forward_splits(meta, config))
        
        if not cv_splits:
            raise ValueError("No CV splits generated - check data and config")
        
        logger.info(f"Generated {len(cv_splits)} CV splits")
        
        # Train model
        logger.info("Training model...")
        model, metrics = train_xgb(X, y, meta, cv_splits, config)
        
        # Save artifacts
        logger.info("Saving artifacts...")
        
        # Save model
        model_path = str(artifacts_dir / "model.joblib")
        save_joblib(model, model_path)
        
        # Save metrics
        metrics_path = str(artifacts_dir / "metrics.json")
        save_json(metrics, metrics_path)
        
        # Save configuration
        config_path_artifact = str(artifacts_dir / "config.yaml")
        save_yaml(config, config_path_artifact)
        
        # Save feature names
        feature_names_path = str(artifacts_dir / "feature_names.json")
        save_json({"feature_names": list(X.columns)}, feature_names_path)
        
        # Save run metadata
        run_metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config_path": config_path,
            "sample_days": sample_days,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "n_samples": len(X),
            "n_features": len(X.columns),
            "n_folds": len(cv_splits),
            "metrics": {k: np.mean(v) for k, v in metrics.items()},
            "artifacts": {
                "model": model_path,
                "metrics": metrics_path,
                "config": config_path_artifact,
                "feature_names": feature_names_path
            }
        }

        metadata_path = str(artifacts_dir / "run_metadata.json")
        save_json(run_metadata, metadata_path)

        # Generate training report
        logger.info("Generating training report...")
        train_report_path = str(artifacts_dir / "train_report.html")
        generate_train_report(run_metadata, metrics, list(X.columns), train_report_path)
        run_metadata["artifacts"]["train_report"] = train_report_path
        # Update metadata with report path
        save_json(run_metadata, metadata_path)
        
        # Log final metrics
        logger.info("Final metrics:")
        for metric, value in run_metadata["metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")

        # Update project state
        logger.info("Updating project state...")
        _update_project_state(run_id, run_metadata)
        
        logger.info(f"Training run {run_id} completed successfully")
        
        return run_metadata
        
    except Exception as e:
        logger.error(f"Error in training run {run_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def _update_project_state(run_id: str, run_metadata: Dict[str, Any]) -> None:
    """
    Update the project state file with a new run entry.
    
    Args:
        run_id: Unique identifier for the run
        run_metadata: Dictionary with run information
    """
    project_state_path = "codex/state/project_state.yaml"
    
    # Load existing project state
    try:
        with open(project_state_path, 'r') as f:
            project_state = yaml.safe_load(f)
    except FileNotFoundError:
        # Create minimal project state if it doesn't exist
        project_state = {
            "project": "volume_price_trade",
            "version": "0.1.0",
            "milestones": [],
            "runs": [],
            "artifacts": []
        }
    
    # Add new run entry
    run_entry = {
        "id": run_id,
        "step": "M4-training",
        "timestamp": run_metadata["timestamp"],
        "notes": f"Model training run with {run_metadata['n_samples']} samples, "
                f"{run_metadata['n_features']} features, {run_metadata['n_folds']} CV folds. "
                f"Average accuracy: {run_metadata['metrics']['accuracy']:.4f}"
    }
    
    project_state["runs"].append(run_entry)
    
    # Add artifact entry
    artifact_entry = {
        "id": f"A{len(project_state['artifacts']):02d}",
        "type": "model_training",
        "description": f"Trained model with run ID {run_id}",
        "timestamp": run_metadata["timestamp"],
        "related_run": run_id
    }
    
    project_state["artifacts"].append(artifact_entry)
    
    # Update milestone status if needed
    for milestone in project_state["milestones"]:
        if milestone["id"] == "M4" and milestone["status"] == "pending":
            milestone["status"] = "in_progress"
            break
    
    # Save updated project state
    save_yaml(project_state, project_state_path)
    
    logger.info(f"Updated project state with run {run_id}")
