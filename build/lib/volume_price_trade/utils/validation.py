"""Schema checks and guards."""

import pandas as pd
from typing import Any

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

    # Check timestamp column exists and is monotonic
    assert "timestamp" in df.columns, "Missing timestamp column"
    assert df["timestamp"].is_monotonic_increasing, "Timestamp is not monotonic"

    # Check numeric columns are non-negative where appropriate
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            assert df[col].notna().all(), f"Column {col} contains NaN values"
            if col == "volume":
                assert (df[col] >= 0).all(), f"Column {col} contains negative values"
            else:
                assert (df[col] > 0).all(), f"Column {col} contains non-positive values"

    # Check price relationships in the order expected by tests
    if all(col in df.columns for col in ["high", "open"]):
        assert (df["high"] >= df["open"]).all(), "High price is less than open price"

    if all(col in df.columns for col in ["high", "close"]):
        assert (df["high"] >= df["close"]).all(), "High price is less than close price"

    if all(col in df.columns for col in ["low", "open"]):
        assert (df["low"] <= df["open"]).all(), "Low price is greater than open price"

    if all(col in df.columns for col in ["low", "close"]):
        assert (df["low"] <= df["close"]).all(), "Low price is greater than close price"

    if all(col in df.columns for col in ["high", "low"]):
        assert (df["high"] >= df["low"]).all(), "High price is less than low price"

    # Check ticker column is non-empty string
    if "ticker" in df.columns:
        assert df["ticker"].notna().all(), "Ticker column contains NaN values"
        assert (df["ticker"].astype(str).str.len() > 0).all(), \
            "Ticker column contains empty strings"