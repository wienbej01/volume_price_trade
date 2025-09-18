"""Tests for dataset builder with purging and embargo functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from src.volume_price_trade.ml.dataset import make_dataset, _purge_overlapping_events


class TestDatasetBuilder:
    """Test cases for make_dataset function."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create test configuration
        self.config = {
            'horizons_minutes': [60],
            'risk': {
                'atr_stop_mult': 1.5,
                'take_profit_R': 2.0
            },
            'sessions': {
                'eod_flat_minutes_before_close': 1
            },
            'cv': {
                'walk_forward': {
                    'embargo_days': 5
                }
            },
            'features': {
                'atr_window': 20,
                'rvol_windows': [5, 20]
            }
        }
        
        # Test parameters
        self.tickers = ['AAPL', 'MSFT']
        self.start = '2023-01-02'
        self.end = '2023-01-06'

    def test_make_dataset_basic(self, monkeypatch):
        """Test basic functionality of make_dataset."""
        # Mock _load_bars_for_ticker to return empty DataFrame
        # This simulates the case where no data is available for the tickers
        def mock_load_bars(*args, **kwargs):
            return pd.DataFrame()

        monkeypatch.setattr('src.volume_price_trade.ml.dataset._load_bars_for_ticker', mock_load_bars)

        # Call make_dataset
        X, y, meta = make_dataset(
            self.tickers,
            self.start,
            self.end,
            self.config
        )

        # Since no data is loaded, we expect empty results
        assert X.empty
        assert y.empty
        assert meta.empty

    def test_make_dataset_empty_tickers(self):
        """Test make_dataset with empty tickers list."""
        X, y, meta = make_dataset(
            [],
            self.start,
            self.end,
            self.config
        )
        
        # Should return empty DataFrames
        assert X.empty
        assert y.empty
        assert meta.empty

    def test_make_dataset_invalid_date_range(self):
        """Test make_dataset with invalid date range."""
        X, y, meta = make_dataset(
            self.tickers,
            '2023-01-06',  # End before start
            '2023-01-02',
            self.config
        )
        
        # Should return empty DataFrames
        assert X.empty
        assert y.empty
        assert meta.empty


class TestPurgeOverlappingEvents:
    """Test cases for _purge_overlapping_events function."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create a test DataFrame with overlapping events
        dates = pd.date_range('2023-01-02 09:30:00', periods=100, freq='1min', tz='America/New_York')
        
        # Create labels with some overlapping events
        np.random.seed(42)
        labels = np.random.choice(['up', 'down', 'hold'], size=100)
        
        # Create event end times with some overlaps
        event_end_times = []
        for i, date in enumerate(dates):
            # Random horizon between 5 and 60 minutes
            horizon = np.random.randint(5, 61)
            event_end_times.append(date + pd.Timedelta(minutes=horizon))
        
        self.test_df = pd.DataFrame({
            'y_class': labels,
            'event_end_time': event_end_times,
            'ticker': 'AAPL',
            'session_date': dates.date
        }, index=dates)
        
        # Test parameters
        self.embargo_days = 5
        self.horizon_minutes = 60

    def test_purge_overlapping_events_basic(self):
        """Test basic functionality of _purge_overlapping_events."""
        # Apply purging
        result = _purge_overlapping_events(
            self.test_df,
            self.embargo_days,
            self.horizon_minutes
        )
        
        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that result has the same columns as input
        assert list(result.columns) == list(self.test_df.columns)
        
        # Check that result has fewer or equal rows than input
        assert len(result) <= len(self.test_df)

    def test_purge_overlapping_events_no_overlaps(self):
        """Test purging with no overlapping events."""
        # Create a DataFrame with no overlapping events
        dates = pd.date_range('2023-01-02 09:30:00', periods=5, freq='1D', tz='America/New_York')
        
        # Event end times are far apart (no overlap)
        event_end_times = [
            dates[0] + pd.Timedelta(hours=1),
            dates[1] + pd.Timedelta(hours=1),
            dates[2] + pd.Timedelta(hours=1),
            dates[3] + pd.Timedelta(hours=1),
            dates[4] + pd.Timedelta(hours=1)
        ]
        
        df = pd.DataFrame({
            'y_class': ['up', 'down', 'up', 'down', 'hold'],
            'event_end_time': event_end_times,
            'ticker': 'AAPL',
            'session_date': dates.date
        }, index=dates)
        
        # Apply purging with short embargo (0 days)
        result = _purge_overlapping_events(
            df,
            embargo_days=0,  # No embargo
            horizon_minutes=60
        )
        
        # Should return all rows since no overlaps and no embargo
        assert len(result) == len(df)

    def test_purge_overlapping_events_with_overlaps(self):
        """Test purging with overlapping events."""
        # Create a DataFrame with overlapping events
        dates = pd.date_range('2023-01-02 09:30:00', periods=5, freq='30min', tz='America/New_York')
        
        # Event end times overlap (short embargo)
        event_end_times = [
            dates[0] + pd.Timedelta(hours=1),
            dates[1] + pd.Timedelta(hours=1),
            dates[2] + pd.Timedelta(hours=1),
            dates[3] + pd.Timedelta(hours=1),
            dates[4] + pd.Timedelta(hours=1)
        ]
        
        df = pd.DataFrame({
            'y_class': ['up', 'down', 'up', 'down', 'hold'],
            'event_end_time': event_end_times,
            'ticker': 'AAPL',
            'session_date': dates.date
        }, index=dates)
        
        # Apply purging with short embargo
        result = _purge_overlapping_events(
            df,
            embargo_days=0,  # No embargo
            horizon_minutes=60
        )
        
        # Should return fewer rows due to overlaps
        assert len(result) < len(df)

    def test_purge_overlapping_events_embargo_enforcement(self):
        """Test that embargo is correctly enforced."""
        # Create a DataFrame with events that need embargo
        dates = pd.date_range('2023-01-02 09:30:00', periods=3, freq='1D', tz='America/New_York')
        
        # Event end times are close together (need embargo)
        event_end_times = [
            dates[0] + pd.Timedelta(hours=1),
            dates[1] + pd.Timedelta(hours=1),
            dates[2] + pd.Timedelta(hours=1)
        ]
        
        df = pd.DataFrame({
            'y_class': ['up', 'down', 'hold'],
            'event_end_time': event_end_times,
            'ticker': 'AAPL',
            'session_date': dates.date
        }, index=dates)
        
        # Apply purging with embargo
        result = _purge_overlapping_events(
            df,
            embargo_days=2,  # 2-day embargo
            horizon_minutes=60
        )
        
        # Should return fewer rows due to embargo
        assert len(result) < len(df)
        
        # Check that remaining events respect the embargo
        for i in range(1, len(result)):
            prev_event_end = result.iloc[i-1]['event_end_time']
            curr_event_start = result.index[i]
            
            # Current event should be after embargo period of previous event
            embargo_end = prev_event_end + pd.Timedelta(days=2)
            assert curr_event_start >= embargo_end

    def test_purge_overlapping_events_empty_dataframe(self):
        """Test purging with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['y_class', 'event_end_time', 'ticker', 'session_date'])
        
        # Apply purging
        result = _purge_overlapping_events(
            empty_df,
            self.embargo_days,
            self.horizon_minutes
        )
        
        # Should return empty DataFrame with same columns
        assert result.empty
        assert list(result.columns) == list(empty_df.columns)

    def test_purge_overlapping_events_nat_event_end_time(self):
        """Test purging with NaT event_end_time values."""
        # Create a DataFrame with some NaT event_end_time values
        dates = pd.date_range('2023-01-02 09:30:00', periods=5, freq='1D', tz='America/New_York')
        
        # Some event end times are NaT
        event_end_times = [
            dates[0] + pd.Timedelta(hours=1),
            pd.NaT,  # NaT value
            dates[2] + pd.Timedelta(hours=1),
            pd.NaT,  # NaT value
            dates[4] + pd.Timedelta(hours=1)
        ]
        
        df = pd.DataFrame({
            'y_class': ['up', 'down', 'up', 'down', 'hold'],
            'event_end_time': event_end_times,
            'ticker': 'AAPL',
            'session_date': dates.date
        }, index=dates)
        
        # Apply purging
        result = _purge_overlapping_events(
            df,
            embargo_days=0,
            horizon_minutes=60
        )
        
        # Should handle NaT values gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(df)

    def test_purge_overlapping_events_sorting(self):
        """Test that output DataFrame is sorted by index."""
        # Create a DataFrame with unsorted index
        dates = pd.date_range('2023-01-02 09:30:00', periods=5, freq='1D', tz='America/New_York')
        
        # Shuffle the dates
        shuffled_dates = dates[[2, 0, 4, 1, 3]]
        
        event_end_times = [
            shuffled_dates[0] + pd.Timedelta(hours=1),
            shuffled_dates[1] + pd.Timedelta(hours=1),
            shuffled_dates[2] + pd.Timedelta(hours=1),
            shuffled_dates[3] + pd.Timedelta(hours=1),
            shuffled_dates[4] + pd.Timedelta(hours=1)
        ]
        
        df = pd.DataFrame({
            'y_class': ['up', 'down', 'up', 'down', 'hold'],
            'event_end_time': event_end_times,
            'ticker': 'AAPL',
            'session_date': shuffled_dates.date
        }, index=shuffled_dates)
        
        # Apply purging
        result = _purge_overlapping_events(
            df,
            embargo_days=0,
            horizon_minutes=60
        )
        
        # Check that result is sorted by index
        assert result.index.is_monotonic_increasing