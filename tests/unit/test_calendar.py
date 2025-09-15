"""Unit tests for calendar functions."""

import pytest
import pandas as pd
import pytz
from datetime import time

from volume_price_trade.data.calendar import is_rth, next_session_close, ET, RTH_START, RTH_END


class TestIsRTH:
    """Test is_rth function with edge cases."""
    
    def test_weekday_rth(self):
        """Test weekday during regular trading hours."""
        # Tuesday 10:30 ET
        ts = pd.Timestamp("2023-06-13 10:30:00").tz_localize(ET)
        assert is_rth(ts) is True
    
    def test_weekday_before_rth(self):
        """Test weekday before regular trading hours."""
        # Tuesday 09:00 ET
        ts = pd.Timestamp("2023-06-13 09:00:00").tz_localize(ET)
        assert is_rth(ts) is False
    
    def test_weekday_after_rth(self):
        """Test weekday after regular trading hours."""
        # Tuesday 17:00 ET
        ts = pd.Timestamp("2023-06-13 17:00:00").tz_localize(ET)
        assert is_rth(ts) is False
    
    def test_saturday(self):
        """Test Saturday during what would be RTH on weekday."""
        # Saturday 10:30 ET
        ts = pd.Timestamp("2023-06-17 10:30:00").tz_localize(ET)
        assert is_rth(ts) is False
    
    def test_sunday(self):
        """Test Sunday during what would be RTH on weekday."""
        # Sunday 10:30 ET
        ts = pd.Timestamp("2023-06-18 10:30:00").tz_localize(ET)
        assert is_rth(ts) is False
    
    def test_rth_start_edge(self):
        """Test exactly at RTH start."""
        # Tuesday 09:30:00 ET
        ts = pd.Timestamp("2023-06-13 09:30:00").tz_localize(ET)
        assert is_rth(ts) is True
    
    def test_rth_end_edge(self):
        """Test exactly at RTH end."""
        # Tuesday 16:00:00 ET
        ts = pd.Timestamp("2023-06-13 16:00:00").tz_localize(ET)
        assert is_rth(ts) is False
    
    def test_utc_timestamp(self):
        """Test UTC timestamp conversion."""
        # Tuesday 14:30 UTC = 10:30 ET
        ts = pd.Timestamp("2023-06-13 14:30:00").tz_localize("UTC")
        assert is_rth(ts) is True
    
    def test_string_timestamp(self):
        """Test string timestamp input."""
        ts = "2023-06-13 10:30:00"
        assert is_rth(ts) is True


class TestNextSessionClose:
    """Test next_session_close function."""
    
    def test_before_close_same_day(self):
        """Test timestamp before today's close."""
        # Tuesday 10:30 ET
        ts = pd.Timestamp("2023-06-13 10:30:00").tz_localize(ET)
        result = next_session_close(ts)
        
        # Should be same day at 16:00 ET
        expected = pd.Timestamp("2023-06-13 16:00:00").tz_localize(ET)
        assert result == expected
    
    def test_after_close_same_day(self):
        """Test timestamp after today's close."""
        # Tuesday 17:00 ET
        ts = pd.Timestamp("2023-06-13 17:00:00").tz_localize(ET)
        result = next_session_close(ts)
        
        # Should be next day at 16:00 ET
        expected = pd.Timestamp("2023-06-14 16:00:00").tz_localize(ET)
        assert result == expected
    
    def test_friday_after_close(self):
        """Test Friday after close - should skip weekend."""
        # Friday 17:00 ET
        ts = pd.Timestamp("2023-06-16 17:00:00").tz_localize(ET)
        result = next_session_close(ts)
        
        # Should be Monday at 16:00 ET
        expected = pd.Timestamp("2023-06-19 16:00:00").tz_localize(ET)
        assert result == expected
    
    def test_saturday(self):
        """Test Saturday - should skip to Monday."""
        # Saturday 10:30 ET
        ts = pd.Timestamp("2023-06-17 10:30:00").tz_localize(ET)
        result = next_session_close(ts)
        
        # Should be Monday at 16:00 ET
        expected = pd.Timestamp("2023-06-19 16:00:00").tz_localize(ET)
        assert result == expected
    
    def test_sunday(self):
        """Test Sunday - should skip to Monday."""
        # Sunday 10:30 ET
        ts = pd.Timestamp("2023-06-18 10:30:00").tz_localize(ET)
        result = next_session_close(ts)
        
        # Should be Monday at 16:00 ET
        expected = pd.Timestamp("2023-06-19 16:00:00").tz_localize(ET)
        assert result == expected

    def test_utc_timestamp(self):
        """Test UTC timestamp conversion."""
        # Tuesday 14:30 UTC = 10:30 ET
        ts = pd.Timestamp("2023-06-13 14:30:00").tz_localize("UTC")
        result = next_session_close(ts)

        # Should be same day at 16:00 ET = 20:00 UTC
        expected = pd.Timestamp("2023-06-13 20:00:00").tz_localize("UTC")
        assert result == expected

    def test_string_timestamp(self):
        """Test string timestamp input."""
        ts = "2023-06-13 10:30:00"
        result = next_session_close(ts)

        # Should be same day at 16:00 ET
        expected = pd.Timestamp("2023-06-13 16:00:00").tz_localize(ET)
        assert result == expected