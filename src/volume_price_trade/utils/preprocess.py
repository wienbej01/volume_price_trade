"""Global preprocessing utilities: dedupe, tz normalize, monotonic sort."""

import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def preprocess_bars(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Apply global preprocessing to raw bars:
    1. Ensure DatetimeIndex (from 'timestamp' col if needed)
    2. TZ normalize to UTC
    3. Sort index ascending (monotonic)
    4. Remove duplicate timestamps (keep first)

    Args:
        df: Raw DataFrame with potential 'timestamp' column or DatetimeIndex
        ticker: Optional ticker for logging context

    Returns:
        Cleaned DataFrame with UTC DatetimeIndex, sorted, deduped
    """
    context = f" for {ticker}" if ticker else ""
    if df.empty:
        logger.info(f"Empty DataFrame{context}; returning as-is.")
        return df

    # 1. Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        else:
            logger.warning(f"No timestamp column or DatetimeIndex{context}; returning as-is.")
            return df

    # 2. TZ normalize to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
        logger.info(f"Localized index to UTC{context}.")
    else:
        df.index = df.index.tz_convert('UTC')
        logger.info(f"Converted index to UTC{context}.")

    # 3. Sort index ascending (monotonic)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        logger.info(f"Sorted index ascending{context}.")

    # 4. Remove duplicate timestamps (keep first)
    if df.index.has_duplicates:
        dupes = df.index.duplicated(keep='first').sum()
        df = df[~df.index.duplicated(keep='first')]
        logger.info(f"Removed {dupes} duplicate timestamps{context}.")

    return df