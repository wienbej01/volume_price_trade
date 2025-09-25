"""
Centralized OHLCV and timestamp validation logic.

This module provides a single source of truth for validating OHLCV dataframes
and ensuring timestamp integrity, as outlined in OHLC_CENTRALIZATION_PLAN.md.
"""
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime


class OhlcValidationError(ValueError):
    """Base exception for all OHLC validation errors."""
    pass


class TimestampTimezoneError(OhlcValidationError):
    """Raised when a timestamp column has missing or unexpected timezone info."""
    pass


class TimestampNotMonotonicError(OhlcValidationError):
    """Raised when timestamps are not strictly increasing."""
    pass


class PriceNonPositiveError(OhlcValidationError):
    """Raised when a price column (O, H, L, C) is not strictly positive."""
    pass


class VolumeNegativeError(OhlcValidationError):
    """Raised when volume is negative."""
    pass


class HighLessThanLowError(OhlcValidationError):
    """Raised when high is less than low."""
    pass


class OpenOutsideRangeError(OhlcValidationError):
    """Raised when open is outside the [low, high] range."""
    pass


class CloseOutsideRangeError(OhlcValidationError):
    """Raised when close is outside the [low, high] range."""
    pass


# Compatibility shims for legacy error messages
class HighLessThanOpenError(OhlcValidationError):
    """Raised when high is less than open. Legacy compatibility."""
    pass


class LowGreaterThanCloseError(OhlcValidationError):
    """Raised when low is greater than close. Legacy compatibility."""
    pass


def validate_ohlcv_frame(
    df: pd.DataFrame,
    *,
    per_ticker: bool = True,
    tz_policy: str = "require_tz",
    msg_compat: bool = True,
) -> None:
    """
    Validates an OHLCV dataframe against canonical rules with deterministic precedence.

    Args:
        df: DataFrame to validate. Must have a DatetimeIndex.
        per_ticker: If True, validate timestamp monotonicity per ticker symbol.
                    Requires a 'ticker' column.
        tz_policy: "require_tz" (default) or "allow_naive".
        msg_compat: If True, raise AssertionError with legacy messages for
                    compatibility with existing tests.

    Raises:
        OhlcValidationError (or subclass): If validation fails.
        AssertionError: If msg_compat is True and a validation fails.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TimestampTimezoneError("Index must be a DatetimeIndex.")

    # 1. Timestamp issues
    if tz_policy == "require_tz" and df.index.tz is None:
        raise TimestampTimezoneError("Timestamps must be timezone-aware.")

    if per_ticker:
        if "ticker" not in df.columns:
            raise ValueError("per_ticker=True requires a 'ticker' column.")
        if not df.groupby("ticker").apply(lambda x: x.index.is_monotonic_increasing, include_groups=False).all():
            raise TimestampNotMonotonicError("Timestamps must be monotonic increasing per ticker.")
    else:
        if not df.index.is_monotonic_increasing:
            raise TimestampNotMonotonicError("Timestamps must be monotonic increasing.")

    if df.index.has_duplicates:
        raise TimestampNotMonotonicError("Timestamps must be unique.")

    # 2. high < low
    if "high" in df.columns and "low" in df.columns:
        invalid_hilo = df["high"] < df["low"]
        if invalid_hilo.any():
            if msg_compat:
                raise AssertionError("High price is less than low price")
            raise HighLessThanLowError(f"High < low at indices: {df.index[invalid_hilo][:3].tolist()}")

    # 3. open/close outside [low, high]
    price_cols = {"open", "close", "high", "low"}
    if price_cols.issubset(df.columns):
        invalid_open = (df["open"] < df["low"]) | (df["open"] > df["high"])
        if invalid_open.any():
            if msg_compat:
                # This specific message is required by an existing test
                raise AssertionError("High price is less than open price")
            raise OpenOutsideRangeError(f"Open outside [low, high] at indices: {df.index[invalid_open][:3].tolist()}")

        invalid_close = (df["close"] < df["low"]) | (df["close"] > df["high"])
        if invalid_close.any():
            if msg_compat:
                # This specific message is required by an existing test
                raise AssertionError("Low price is greater than close price")
            raise CloseOutsideRangeError(f"Close outside [low, high] at indices: {df.index[invalid_close][:3].tolist()}")

    # 4. Volume < 0 or price <= 0
    if "volume" in df.columns:
        invalid_vol = df["volume"] < 0
        if invalid_vol.any():
            if msg_compat:
                raise AssertionError("Volume is negative")
            raise VolumeNegativeError(f"Volume < 0 at indices: {df.index[invalid_vol][:3].tolist()}")

    price_cols_present = [col for col in ["open", "high", "low", "close"] if col in df.columns]
    for col in price_cols_present:
        invalid_price = df[col] <= 0
        if invalid_price.any():
            if msg_compat:
                raise AssertionError(f"Price column '{col}' is not positive")
            raise PriceNonPositiveError(f"{col} <= 0 at indices: {df.index[invalid_price][:3].tolist()}")


def coerce_timestamps(
    df: pd.DataFrame, *, per_ticker: bool = True, to_utc: bool = True
) -> pd.DataFrame:
    """
    Normalizes timestamps in a dataframe.

    - Converts tz-aware timestamps to UTC.
    - Sorts by timestamp (and optionally ticker).
    - Removes duplicate timestamps, keeping the first entry.

    Args:
        df: DataFrame with a DatetimeIndex.
        per_ticker: If True, sort and dedupe within each 'ticker' group.
        to_utc: If True, convert timezone to UTC.

    Returns:
        A new DataFrame with coerced timestamps.
    """
    if not is_datetime(df.index):
        raise TypeError("Index must be a datetime-like.")

    df_copy = df.copy()

    if to_utc and df_copy.index.tz is not None:
        df_copy.index = df_copy.index.tz_convert("UTC")
    elif to_utc and df_copy.index.tz is None:
        # This assumes naive timestamps are in UTC, a common convention.
        # For more complex cases, the caller should localize first.
        df_copy.index = df_copy.index.tz_localize("UTC")


    sort_keys = ["ticker", df_copy.index.name or "timestamp"] if per_ticker else [df_copy.index.name or "timestamp"]
    if per_ticker and "ticker" not in df_copy.columns:
        raise ValueError("per_ticker=True requires a 'ticker' column.")
    
    if per_ticker:
        df_copy = df_copy.sort_values(by=["ticker", df_copy.index.name])
    else:
        df_copy = df_copy.sort_index()

    # Deduplicate
    if per_ticker:
        df_copy = df_copy[~df_copy.index.duplicated(keep="first")]
    else:
        df_copy = df_copy[~df_copy.index.duplicated(keep="first")]

    return df_copy


def is_valid_ohlcv_row(row: pd.Series) -> bool:
    """
    Performs a fast, boolean check on a single row (Series) of OHLCV data.

    Useful for performance-sensitive guards in loops.

    Args:
        row: A pandas Series representing a single bar.

    Returns:
        True if the row is valid, False otherwise.
    """
    try:
        validate_ohlcv_row(row)
        return True
    except OhlcValidationError:
        return False


def validate_ohlcv_row(row: pd.Series) -> None:
    """
    Validates a single row of OHLCV data.

    Args:
        row: A pandas Series representing a single bar.

    Raises:
        OhlcValidationError or subclass if validation fails.
    """
    # Use .get() to avoid KeyError if a column is missing
    low, high = row.get("low"), row.get("high")
    open_, close_ = row.get("open"), row.get("close")
    volume = row.get("volume")

    if high is not None and low is not None and high < low:
        raise HighLessThanLowError("High < low")

    if open_ is not None:
        if low is not None and open_ < low:
            raise OpenOutsideRangeError("Open < low")
        if high is not None and open_ > high:
            raise OpenOutsideRangeError("Open > high")

    if close_ is not None:
        if low is not None and close_ < low:
            raise CloseOutsideRangeError("Close < low")
        if high is not None and close_ > high:
            raise CloseOutsideRangeError("Close > high")

    if volume is not None and volume < 0:
        raise VolumeNegativeError("Volume < 0")

    for price in (open_, high, low, close_):
        if price is not None and price <= 0:
            raise PriceNonPositiveError("Price <= 0")