"""Unit tests for calendar functions."""

import pytest
import pandas as pd
import pytz
from datetime import time

from volume_price_trade.data.calendar import is_rth, next_session_close, ET, RTH_START, RTH_END, get_early_close_dates
from datetime import date

class TestGetEarlyCloseDates:
    """Test get_early_close_dates function."""

    def test_2023(self):
        """Test early close dates for 2023."""
        dates = get_early_close_dates((2023,))
        assert date(2023, 7, 3) in dates
        assert date(2023, 11, 24) in dates
        assert date(2023, 12, 24) not in dates # Sunday

    def test_2024(self):
        """Test early close dates for 2024 (leap year)."""
        dates = get_early_close_dates((2024,))
        assert date(2024, 7, 3) in dates
        assert date(2024, 11, 29) in dates
        assert date(2024, 12, 24) in dates

    def test_2025(self):
        """Test early close dates for 2025."""
        dates = get_early_close_dates((2025,))
        assert date(2025, 7, 3) in dates
        assert date(2025, 11, 28) in dates
        assert date(2025, 12, 24) in dates

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

    def test_early_close_day_during_rth(self):
        """Test RTH on an early close day."""
        # July 3, 2023, 10:30 ET
        ts = pd.Timestamp("2023-07-03 10:30:00").tz_localize(ET)
        assert is_rth(ts) is True

    def test_early_close_day_after_close(self):
        """Test after close on an early close day."""
        # July 3, 2023, 14:00 ET
        ts = pd.Timestamp("2023-07-03 14:00:00").tz_localize(ET)
        assert is_rth(ts) is False

    def test_early_close_edge(self):
        """Test exactly at early close time."""
        # July 3, 2023, 13:00 ET
        ts = pd.Timestamp("2023-07-03 13:00:00").tz_localize(ET)
        assert is_rth(ts) is False

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

    def test_early_close_day_before_close(self):
        """Test before close on an early close day."""
        # July 3, 2023, 10:30 ET
        ts = pd.Timestamp("2023-07-03 10:30:00").tz_localize(ET)
        result = next_session_close(ts)
        expected = pd.Timestamp("2023-07-03 13:00:00").tz_localize(ET)
        assert result == expected

    def test_early_close_day_after_close(self):
        """Test after close on an early close day."""
        # July 3, 2023, 14:00 ET
        ts = pd.Timestamp("2023-07-03 14:00:00").tz_localize(ET)
        result = next_session_close(ts)
        # Next day is July 4th, a holiday, so it should be July 5th
        expected = pd.Timestamp("2023-07-05 16:00:00").tz_localize(ET)
        assert result == expected

    def test_day_before_early_close(self):
        """Test the day before an early close day."""
        # July 2, 2023 (Sunday)
        ts = pd.Timestamp("2023-07-02 10:00:00").tz_localize(ET)
        result = next_session_close(ts)
        # Next session is July 3, an early close day
        expected = pd.Timestamp("2023-07-03 13:00:00").tz_localize(ET)
        assert result == expected