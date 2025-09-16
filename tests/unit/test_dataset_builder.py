
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
