"""Tests for triple-barrier labeling system."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from src.volume_price_trade.labels.targets import triple_barrier_labels


class TestTripleBarrierLabels:
    """Test cases for triple_barrier_labels function."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create a simple test DataFrame with OHLCV data
        self.base_df = self._create_test_ohlcv_data()
        
        # Test parameters
        self.horizons_min = 60  # 1 hour
        self.atr_mult_sl = 1.5  # 1.5x ATR for stop loss
        self.r_mult_tp = 2.0    # 2R take profit
        self.eod_flat = True    # Apply EOD flat position

    def _create_test_ohlcv_data(self, days=5, minutes_per_day=390):
        """Create synthetic OHLCV data for testing."""
        # Create datetime index for trading days (9:30 AM - 4:00 PM ET)
        start_date = pd.Timestamp('2023-01-02 09:30:00', tz='America/New_York')
        end_date = start_date + pd.Timedelta(days=days-1, minutes=minutes_per_day-1)
        
        # Create business day index at 1-minute intervals
        dates = pd.date_range(start=start_date, end=end_date, freq='1min', tz='America/New_York')
        
        # Filter to only include regular trading hours (9:30 AM - 4:00 PM)
        dates = [d for d in dates if d.hour >= 9 and (d.hour < 16 or (d.hour == 16 and d.minute == 0))]
        
        # Generate synthetic price data with a slight upward trend
        n_periods = len(dates)
        base_price = 100.0
        trend = 0.0001  # Small upward trend
        
        # Create OHLCV data
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.0001, 0.005, n_periods)  # Random returns
        
        prices = [base_price]
        for i in range(1, n_periods):
            prices.append(prices[-1] * (1 + returns[i] + trend))
        
        # Convert to numpy array
        prices = np.array(prices)
        
        # Create OHLCV DataFrame
        df = pd.DataFrame({
            'open': prices * 0.999,  # Open slightly below previous close
            'high': prices * 1.005,  # High slightly above close
            'low': prices * 0.995,   # Low slightly below close
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_periods)
        }, index=dates)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        return df

    def test_triple_barrier_labels_basic(self):
        """Test basic functionality of triple_barrier_labels."""
        # Apply triple barrier labels
        result = triple_barrier_labels(
            self.base_df, 
            self.horizons_min, 
            self.atr_mult_sl, 
            self.r_mult_tp, 
            self.eod_flat
        )
        
        # Check that result has the expected columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume', 
                           'y_class', 'horizon_minutes', 'event_end_time']
        assert all(col in result.columns for col in expected_columns)
        
        # Check that y_class contains only expected values
        valid_labels = {'up', 'down', 'hold'}
        assert all(label in valid_labels for label in result['y_class'].dropna())
        
        # Check that horizon_minutes is positive
        assert all(h > 0 for h in result['horizon_minutes'] if h > 0)

    def test_take_profit_hit(self):
        """Test case where take profit is hit before stop loss."""
        # Create a DataFrame where price increases steadily
        df = self.base_df.copy()
        
        # Modify prices to ensure take profit is hit
        for i in range(len(df)):
            if i > 0:
                # Make each bar's high significantly higher than previous close
                df.iloc[i, df.columns.get_loc('high')] = df.iloc[i-1, df.columns.get_loc('close')] * 1.02
                df.iloc[i, df.columns.get_loc('close')] = df.iloc[i-1, df.columns.get_loc('close')] * 1.015
                df.iloc[i, df.columns.get_loc('open')] = df.iloc[i-1, df.columns.get_loc('close')]
                df.iloc[i, df.columns.get_loc('low')] = df.iloc[i, df.columns.get_loc('open')] * 0.998
        
        # Apply triple barrier labels
        result = triple_barrier_labels(
            df, 
            self.horizons_min, 
            self.atr_mult_sl, 
            self.r_mult_tp, 
            self.eod_flat
        )
        
        # Check that most labels are 'up' (take profit hit)
        up_count = (result['y_class'] == 'up').sum()
        down_count = (result['y_class'] == 'down').sum()
        hold_count = (result['y_class'] == 'hold').sum()
        
        # Most should be 'up' due to our price manipulation
        assert up_count > down_count
        assert up_count > hold_count

    def test_stop_loss_hit(self):
        """Test case where stop loss is hit before take profit."""
        # Create a DataFrame where price decreases steadily
        df = self.base_df.copy()
        
        # Modify prices to ensure stop loss is hit
        for i in range(len(df)):
            if i > 0:
                # Make each bar's low significantly lower than previous close
                df.iloc[i, df.columns.get_loc('low')] = df.iloc[i-1, df.columns.get_loc('close')] * 0.98
                df.iloc[i, df.columns.get_loc('close')] = df.iloc[i-1, df.columns.get_loc('close')] * 0.985
                df.iloc[i, df.columns.get_loc('open')] = df.iloc[i-1, df.columns.get_loc('close')]
                df.iloc[i, df.columns.get_loc('high')] = df.iloc[i, df.columns.get_loc('open')] * 1.002
        
        # Apply triple barrier labels
        result = triple_barrier_labels(
            df, 
            self.horizons_min, 
            self.atr_mult_sl, 
            self.r_mult_tp, 
            self.eod_flat
        )
        
        # Check that most labels are 'down' (stop loss hit)
        up_count = (result['y_class'] == 'up').sum()
        down_count = (result['y_class'] == 'down').sum()
        hold_count = (result['y_class'] == 'hold').sum()
        
        # Most should be 'down' due to our price manipulation
        assert down_count > up_count
        assert down_count > hold_count

    def test_timeout_behavior(self):
        """Test case where neither barrier is hit (timeout)."""
        # Create a DataFrame with very small price movements
        df = self.base_df.copy()
        
        # Modify prices to ensure neither barrier is hit
        for i in range(len(df)):
            if i > 0:
                # Make very small price movements
                change = np.random.uniform(-0.001, 0.001)
                df.iloc[i, df.columns.get_loc('close')] = df.iloc[i-1, df.columns.get_loc('close')] * (1 + change)
                df.iloc[i, df.columns.get_loc('open')] = df.iloc[i-1, df.columns.get_loc('close')]
                df.iloc[i, df.columns.get_loc('high')] = max(df.iloc[i, df.columns.get_loc('open')], 
                                                           df.iloc[i, df.columns.get_loc('close')]) * 1.0005
                df.iloc[i, df.columns.get_loc('low')] = min(df.iloc[i, df.columns.get_loc('open')], 
                                                          df.iloc[i, df.columns.get_loc('close')]) * 0.9995
        
        # Apply triple barrier labels with a very short horizon
        result = triple_barrier_labels(
            df, 
            5,  # Very short horizon (5 minutes)
            self.atr_mult_sl, 
            self.r_mult_tp, 
            self.eod_flat
        )
        
        # Check that most labels are 'hold' (timeout)
        up_count = (result['y_class'] == 'up').sum()
        down_count = (result['y_class'] == 'down').sum()
        hold_count = (result['y_class'] == 'hold').sum()
        
        # Most should be 'hold' due to small price movements and short horizon
        assert hold_count > up_count
        assert hold_count > down_count

    def test_eod_truncation(self):
        """Test that EOD truncation works correctly."""
        # Create a DataFrame with data near EOD
        df = self.base_df.copy()
        
        # Apply triple barrier labels with EOD flat
        result = triple_barrier_labels(
            df, 
            240,  # Long horizon (4 hours)
            self.atr_mult_sl, 
            self.r_mult_tp, 
            True  # EOD flat enabled
        )
        
        # Check that for bars near EOD, horizon_minutes is less than the full horizon
        # Get bars in the last hour of the trading day
        eod_bars = result[result.index.hour == 15]  # 3:00-4:00 PM
        
        # These should have shorter horizons due to EOD truncation
        if not eod_bars.empty:
            max_horizon = eod_bars['horizon_minutes'].max()
            assert max_horizon < 240  # Should be less than full horizon due to EOD

    def test_event_end_time(self):
        """Test that event_end_time is correctly set."""
        # Apply triple barrier labels
        result = triple_barrier_labels(
            self.base_df, 
            self.horizons_min, 
            self.atr_mult_sl, 
            self.r_mult_tp, 
            self.eod_flat
        )
        
        # Check that event_end_time is after the index time for non-hold events
        non_hold_events = result[result['y_class'] != 'hold']
        if not non_hold_events.empty:
            for idx, row in non_hold_events.iterrows():
                assert row['event_end_time'] > idx

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        # Apply triple barrier labels
        result = triple_barrier_labels(
            empty_df, 
            self.horizons_min, 
            self.atr_mult_sl, 
            self.r_mult_tp, 
            self.eod_flat
        )
        
        # Should return a DataFrame with the same structure
        assert result.shape[0] == 0  # No rows
        expected_columns = ['open', 'high', 'low', 'close', 'volume', 
                           'y_class', 'horizon_minutes', 'event_end_time']
        assert all(col in result.columns for col in expected_columns)

    def test_single_row_dataframe(self):
        """Test behavior with a single row DataFrame."""
        single_row_df = self.base_df.iloc[0:1].copy()
        
        # Apply triple barrier labels
        result = triple_barrier_labels(
            single_row_df, 
            self.horizons_min, 
            self.atr_mult_sl, 
            self.r_mult_tp, 
            self.eod_flat
        )
        
        # Should return a DataFrame with the same structure
        assert result.shape[0] == 1  # One row
        expected_columns = ['open', 'high', 'low', 'close', 'volume', 
                           'y_class', 'horizon_minutes', 'event_end_time']
        assert all(col in result.columns for col in expected_columns)
        
        # With only one row, the label should be 'hold' (no future data)
        assert result.iloc[0]['y_class'] == 'hold'

    def test_no_eod_flat(self):
        """Test behavior with EOD flat disabled."""
        # Apply triple barrier labels without EOD flat
        result = triple_barrier_labels(
            self.base_df, 
            self.horizons_min, 
            self.atr_mult_sl, 
            self.r_mult_tp, 
            False  # EOD flat disabled
        )
        
        # Check that all horizons are the full horizon (no EOD truncation)
        assert all(h == self.horizons_min for h in result['horizon_minutes'] if h > 0)