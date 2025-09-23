"""Schema checks and guards."""

import pandas as pd
from typing import Any
from volume_price_trade.utils.ohlc import validate_ohlcv_frame

def validate_bars(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Validate minute bar DataFrame against schema.

    Args:
        df: DataFrame with minute bar data
        cfg: Configuration dictionary with schema definition

    Raises:
        AssertionError: If validation fails
    """
    # Check DataFrame is not empty
    assert not df.empty, "DataFrame is empty"

    # Get expected columns from config
    expected_cols = list(cfg["data"]["schema"].values()) + ["ticker"]

    # Check all required columns exist
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Check ticker column is non-empty string
    if "ticker" in df.columns:
        assert df["ticker"].notna().all(), "Ticker column contains NaN values"
        assert (df["ticker"].astype(str).str.len() > 0).all(), \
            "Ticker column contains empty strings"

    # Delegate to centralized OHLCV validator with legacy message compatibility
    # This handles all timestamp, price, and volume validation with deterministic precedence
    validate_ohlcv_frame(df, per_ticker=True, tz_policy="allow_naive", msg_compat=True)