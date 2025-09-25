"""Global preprocessing utilities: dedupe, tz normalize, monotonic sort."""

import pandas as pd
import logging
from typing import Optional
from volume_price_trade.utils.ohlc import coerce_timestamps

logger = logging.getLogger(__name__)


from volume_price_trade.utils.ohlc import coerce_timestamps


def preprocess_bars(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Apply global preprocessing to raw bars using the centralized `coerce_timestamps` function.
    This ensures timezone normalization (to UTC), monotonic sorting, and deduplication.

    Args:
        df: Raw DataFrame with potential 'timestamp' column or DatetimeIndex.
        ticker: Optional ticker for logging context. If provided, 'ticker' column
                is expected for per-ticker operations.

    Returns:
        Cleaned DataFrame with a UTC DatetimeIndex, sorted, and deduped.
    """
    context = f" for {ticker}" if ticker else ""
    if df.empty:
        logger.info(f"Empty DataFrame{context}; returning as-is.")
        return df

    # Ensure DatetimeIndex before coercion
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index(pd.to_datetime(df['timestamp'])).drop(columns=['timestamp'])
        else:
            logger.warning(f"No timestamp column or DatetimeIndex{context}; returning as-is.")
            return df

    logger.info(f"Applying timestamp coercion (UTC, sort, dedupe){context}.")
    # Use per_ticker logic only if a ticker context is provided and column exists
    use_per_ticker = ticker is not None and "ticker" in df.columns
    return coerce_timestamps(df, per_ticker=use_per_ticker)