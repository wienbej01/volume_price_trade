"""Artifact IO helpers (parquet/feather)."""

import pandas as pd
import joblib
import json
import yaml
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to parquet file.
    
    Args:
        df: DataFrame to save
        path: File path to save to
    """
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info(f"Saved DataFrame to {path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {path}: {e}")
        raise


def save_joblib(obj: Any, path: str) -> None:
    """
    Save object using joblib.
    
    Args:
        obj: Object to save
        path: File path to save to
    """
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)
        logger.info(f"Saved object to {path}")
    except Exception as e:
        logger.error(f"Error saving object to {path}: {e}")
        raise


def save_json(obj: Dict, path: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        obj: Dictionary to save
        path: File path to save to
    """
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)
        logger.info(f"Saved JSON to {path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {path}: {e}")
        raise


def save_yaml(obj: Dict, path: str) -> None:
    """
    Save dictionary to YAML file.
    
    Args:
        obj: Dictionary to save
        path: File path to save to
    """
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(obj, f, default_flow_style=False)
        logger.info(f"Saved YAML to {path}")
    except Exception as e:
        logger.error(f"Error saving YAML to {path}: {e}")
        raise
