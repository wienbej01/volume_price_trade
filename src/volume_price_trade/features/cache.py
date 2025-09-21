"""Feature cache utilities for processed_data with versioning."""

import pandas as pd
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = Path("processed_data")


def _hash_config(cfg: Dict[str, Any]) -> str:
    """Generate a hash of the config dict for versioning."""
    cfg_str = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(cfg_str.encode()).hexdigest()[:16]


def load_cached_features(ticker: str, cfg: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load cached features if they exist and match the config.

    Args:
        ticker: Stock ticker
        cfg: Configuration dict used to generate features

    Returns:
        Tuple of (DataFrame or None, version_hash or None)
    """
    version_hash = _hash_config(cfg)
    cache_path = CACHE_DIR / f"{ticker}_features_{version_hash}.parquet"
    meta_path = CACHE_DIR / f"{ticker}_features_{version_hash}.meta.json"

    if not cache_path.exists() or not meta_path.exists():
        logger.info(f"No cache found for {ticker} with hash {version_hash}")
        return None, version_hash

    try:
        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)
        cached_hash = meta.get("config_hash")
        if cached_hash != version_hash:
            logger.info(f"Config hash mismatch for {ticker}; cache invalid.")
            return None, version_hash

        # Load DataFrame
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded cached features for {ticker} from {cache_path}")
        return df, version_hash
    except Exception as e:
        logger.warning(f"Failed to load cache for {ticker}: {e}")
        return None, version_hash


def save_cached_features(ticker: str, df: pd.DataFrame, cfg: Dict[str, Any], version_hash: str) -> None:
    """
    Save features to cache with metadata.

    Args:
        ticker: Stock ticker
        df: Feature DataFrame
        cfg: Configuration dict used to generate features
        version_hash: Precomputed config hash
    """
    cache_path = CACHE_DIR / f"{ticker}_features_{version_hash}.parquet"
    meta_path = CACHE_DIR / f"{ticker}_features_{version_hash}.meta.json"

    try:
        # Save DataFrame
        df.to_parquet(cache_path)
        logger.info(f"Saved features for {ticker} to {cache_path}")

        # Save metadata
        meta = {
            "config_hash": version_hash,
            "ticker": ticker,
            "shape": df.shape,
            "columns": list(df.columns),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved metadata for {ticker} to {meta_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache for {ticker}: {e}")