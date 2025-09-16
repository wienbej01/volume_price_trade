"""Unit tests for validation functions."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from volume_price_trade.utils.validation import validate_bars


class TestValidateBars:
    """Test validate_bars function with synthetic data."""
    
    @pytest.fixture
    def valid_config(self):
        """Return a valid configuration dictionary."""
        return {
            "data": {
                "schema": {
                    "timestamp": "timestamp",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "ticker": "ticker"
                }
            }
        }
    
    @pytest.fixture
    def valid_df(self):
        """Return a valid DataFrame with synthetic bar data."""
        dates = pd.date_range("2023-06-13 09:30:00", periods=5, freq="1min")
        return pd.DataFrame({
            "timestamp": dates,
            "open": [100.0, 100.5, 101.0, 100.8, 101.2],
            "high": [100.5, 101.0, 101.2, 101.1, 101.5],
            "low": [99.8, 100.2, 100.8, 100.5, 100.9],
            "close": [100.3, 100.8, 100.9, 100.9, 101.3],
            "volume": [1000, 1200, 800, 1500, 900],
            "ticker": ["AAPL"] * 5
        })
    
    def test_valid_data_passes(self, valid_df, valid_config):
        """Test that valid data passes validation."""
        validate_bars(valid_df, valid_config)  # Should not raise
    
    def test_missing_column_fails(self, valid_config):
        """Test that missing required column fails validation."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-06-13 09:30:00", periods=5, freq="1min"),
            "open": [100.0, 100.5, 101.0, 100.8, 101.2],
            # Missing other required columns
        })
        
        with pytest.raises(AssertionError, match="Missing required column"):
            validate_bars(df, valid_config)
    
    def test_non_monotonic_timestamp_fails(self, valid_df, valid_config):
        """Test that non-monotonic timestamp fails validation."""
        # Swap two rows to make timestamp non-monotonic
        valid_df.loc[1, "timestamp"] = valid_df.loc[2, "timestamp"]
        valid_df.loc[2, "timestamp"] = valid_df.loc[1, "timestamp"]
        
        with pytest.raises(AssertionError, match="Timestamp is not monotonic"):
            validate_bars(valid_df, valid_config)
    
    def test_negative_volume_fails(self, valid_df, valid_config):
        """Test that negative volume fails validation."""
        valid_df.loc[0, "volume"] = -100
        
        with pytest.raises(AssertionError, match="contains negative values"):
            validate_bars(valid_df, valid_config)
    
    def test_zero_price_fails(self, valid_df, valid_config):
        """Test that zero price fails validation."""
        valid_df.loc[0, "open"] = 0.0
        
        with pytest.raises(AssertionError, match="contains non-positive values"):
            validate_bars(valid_df, valid_config)
    
    def test_high_less_than_low_fails(self, valid_df, valid_config):
        """Test that high < low fails validation."""
        valid_df.loc[0, "high"] = 99.0
        valid_df.loc[0, "low"] = 100.0
        
        with pytest.raises(AssertionError, match="High price is less than low price"):
            validate_bars(valid_df, valid_config)
    
    def test_high_less_than_open_fails(self, valid_df, valid_config):
        """Test that high < open fails validation."""
        valid_df.loc[0, "high"] = 99.0
        valid_df.loc[0, "open"] = 100.0
        
        with pytest.raises(AssertionError, match="High price is less than open price"):
            validate_bars(valid_df, valid_config)
    
    def test_low_greater_than_close_fails(self, valid_df, valid_config):
        """Test that low > close fails validation."""
        valid_df.loc[0, "low"] = 101.0
        valid_df.loc[0, "close"] = 100.0
        
        with pytest.raises(AssertionError, match="Low price is greater than close price"):
            validate_bars(valid_df, valid_config)
    
    def test_nan_values_fails(self, valid_df, valid_config):
        """Test that NaN values fail validation."""
        valid_df.loc[0, "open"] = np.nan
        
        with pytest.raises(AssertionError, match="contains NaN values"):
            validate_bars(valid_df, valid_config)
    
    def test_synthetic_bars_monotonic_timestamp(self, valid_config):
        """Test validation of monotonic timestamp with synthetic bars."""
        # Create synthetic bar data with monotonic timestamps
        dates = pd.date_range("2023-06-13 09:30:00", periods=10, freq="1min")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0 + i * 0.1 for i in range(10)],
            "high": [100.5 + i * 0.1 for i in range(10)],
            "low": [99.8 + i * 0.1 for i in range(10)],
            "close": [100.3 + i * 0.1 for i in range(10)],
            "volume": [1000 + i * 100 for i in range(10)],
            "ticker": ["AAPL"] * 10
        })
        
        # Should pass validation
        validate_bars(df, valid_config)
        
        # Make timestamps non-monotonic
        df.loc[5, "timestamp"] = df.loc[4, "timestamp"]
        
        # Should fail validation
        with pytest.raises(AssertionError, match="Timestamp is not monotonic"):
            validate_bars(df, valid_config)
    
    def test_synthetic_bars_required_columns(self, valid_config):
        """Test that all required columns exist in synthetic bars."""
        # Create synthetic bar data with all required columns
        dates = pd.date_range("2023-06-13 09:30:00", periods=5, freq="1min")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0, 100.5, 101.0, 100.8, 101.2],
            "high": [100.5, 101.0, 101.2, 101.1, 101.5],
            "low": [99.8, 100.2, 100.8, 100.5, 100.9],
            "close": [100.3, 100.8, 100.9, 100.9, 101.3],
            "volume": [1000, 1200, 800, 1500, 900],
            "ticker": ["AAPL"] * 5
        })
        
        # Should pass validation
        validate_bars(df, valid_config)
        
        # Remove each required column one by one and test
        required_columns = ["timestamp", "open", "high", "low", "close", "volume", "ticker"]
        
        for col in required_columns:
            df_missing = df.drop(columns=[col])
            with pytest.raises(AssertionError, match=f"Missing required column: {col}"):
                validate_bars(df_missing, valid_config)
    
    def test_synthetic_bars_non_negative_volume(self, valid_config):
        """Test validation of non-negative volume in synthetic bars."""
        # Create synthetic bar data with non-negative volume
        dates = pd.date_range("2023-06-13 09:30:00", periods=5, freq="1min")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0, 100.5, 101.0, 100.8, 101.2],
            "high": [100.5, 101.0, 101.2, 101.1, 101.5],
            "low": [99.8, 100.2, 100.8, 100.5, 100.9],
            "close": [100.3, 100.8, 100.9, 100.9, 101.3],
            "volume": [1000, 1200, 800, 1500, 900],  # All non-negative
            "ticker": ["AAPL"] * 5
        })
        
        # Should pass validation
        validate_bars(df, valid_config)
        
        # Set one volume to negative
        df.loc[2, "volume"] = -100
        
        # Should fail validation
        with pytest.raises(AssertionError, match="contains negative values"):
            validate_bars(df, valid_config)
        
        # Set one volume to zero (should pass)
        df.loc[2, "volume"] = 0
        validate_bars(df, valid_config)  # Should not raise
    
    def test_synthetic_bars_price_relationships(self, valid_config):
        """Test validation of price relationships (high >= low) in synthetic bars."""
        # Create synthetic bar data with valid price relationships
        dates = pd.date_range("2023-06-13 09:30:00", periods=5, freq="1min")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0, 100.5, 101.0, 100.8, 101.2],
            "high": [100.5, 101.0, 101.2, 101.1, 101.5],  # high >= low
            "low": [99.8, 100.2, 100.8, 100.5, 100.9],   # low <= high
            "close": [100.3, 100.8, 100.9, 100.9, 101.3],
            "volume": [1000, 1200, 800, 1500, 900],
            "ticker": ["AAPL"] * 5
        })
        
        # Should pass validation
        validate_bars(df, valid_config)
        
        # Make high < low
        df.loc[2, "high"] = 100.5
        df.loc[2, "low"] = 101.0
        
        # Should fail validation
        with pytest.raises(AssertionError, match="High price is less than low price"):
            validate_bars(df, valid_config)
    
    def test_synthetic_bars_ticker_column(self, valid_config):
        """Test validation of ticker column being non-empty string in synthetic bars."""
        # Create synthetic bar data with valid ticker
        dates = pd.date_range("2023-06-13 09:30:00", periods=5, freq="1min")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0, 100.5, 101.0, 100.8, 101.2],
            "high": [100.5, 101.0, 101.2, 101.1, 101.5],
            "low": [99.8, 100.2, 100.8, 100.5, 100.9],
            "close": [100.3, 100.8, 100.9, 100.9, 101.3],
            "volume": [1000, 1200, 800, 1500, 900],
            "ticker": ["AAPL"] * 5  # All non-empty strings
        })
        
        # Should pass validation
        validate_bars(df, valid_config)
        
        # Set one ticker to empty string
        df.loc[2, "ticker"] = ""
        
        # Should fail validation
        with pytest.raises(AssertionError, match="contains empty strings"):
            validate_bars(df, valid_config)
        
        # Set one ticker to NaN
        df.loc[2, "ticker"] = np.nan
        
        # Should fail validation
        with pytest.raises(AssertionError, match="contains NaN values"):
            validate_bars(df, valid_config)
        
        # Set ticker to valid non-empty string
        df.loc[2, "ticker"] = "MSFT"
        validate_bars(df, valid_config)  # Should not raise
    
    def test_empty_dataframe_fails(self, valid_config):
        """Test that empty DataFrame fails validation."""
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "ticker"])
        
        with pytest.raises(AssertionError, match="DataFrame is empty"):
            validate_bars(df, valid_config)
    
    def test_empty_ticker_fails(self, valid_df, valid_config):
        """Test that empty ticker fails validation."""
        valid_df.loc[0, "ticker"] = ""
        
        with pytest.raises(AssertionError, match="contains empty strings"):
            validate_bars(valid_df, valid_config)
    
    def test_nan_ticker_fails(self, valid_df, valid_config):
        """Test that NaN ticker fails validation."""
        valid_df.loc[0, "ticker"] = np.nan
        
        with pytest.raises(AssertionError, match="contains NaN values"):
            validate_bars(valid_df, valid_config)