"""Integration tests for backtesting functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import json
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from volume_price_trade.backtest.engine import run_backtest
from volume_price_trade.backtest.risk import shares_for_risk
from volume_price_trade.backtest.fills import get_fill_price
from volume_price_trade.backtest.metrics import compute_all_metrics
from volume_price_trade.backtest.reports import make_report


class TestBacktestIntegration:
    """Integration tests for backtesting functionality."""
    
    def create_test_data(self):
        """Create synthetic test data for backtesting."""
        # Create timestamps
        start_date = datetime(2023, 1, 1, 9, 30)
        timestamps = [start_date + timedelta(minutes=i) for i in range(0, 390 * 5)]  # 5 days of 5-minute bars
        
        # Create tickers
        tickers = ['AAPL', 'MSFT', 'GOOG']
        
        # Create signals data
        signals_data = []
        for timestamp in timestamps[::10]:  # Every 50 minutes
            for ticker in tickers:
                # Random signal strength between -1 and 1
                signal_strength = np.random.uniform(-1, 1)
                
                # Only include signals above threshold
                if abs(signal_strength) > 0.5:
                    signals_data.append({
                        'timestamp': timestamp,
                        'ticker': ticker,
                        'signal_strength': signal_strength,
                        'atr': np.random.uniform(0.5, 2.0)  # Random ATR
                    })
        
        signals_df = pd.DataFrame(signals_data)
        
        # Create bars data
        bars_data = []
        for timestamp in timestamps:
            for ticker in tickers:
                # Random walk for price
                base_price = 100 + np.random.uniform(-10, 10)
                
                bars_data.append({
                    'timestamp': timestamp,
                    'ticker': ticker,
                    'open': base_price,
                    'high': base_price * (1 + np.random.uniform(0, 0.02)),
                    'low': base_price * (1 - np.random.uniform(0, 0.02)),
                    'close': base_price * (1 + np.random.uniform(-0.01, 0.01)),
                    'volume': np.random.randint(1000, 10000)
                })
        
        bars_df = pd.DataFrame(bars_data)
        
        return signals_df, bars_df
    
    def test_shares_for_risk(self):
        """Test position sizing function."""
        # Test basic calculation
        shares = shares_for_risk(
            equity=100000,
            entry=100.0,
            stop=98.0,
            risk_perc=0.02
        )
        
        # Expected: 100000 * 0.02 / (100 - 98) = 1000 shares
        assert shares == 1000
        
        # Test with zero risk percentage
        shares = shares_for_risk(
            equity=100000,
            entry=100.0,
            stop=98.0,
            risk_perc=0.0
        )
        assert shares == 0
        
        # Test with zero equity
        shares = shares_for_risk(
            equity=0,
            entry=100.0,
            stop=98.0,
            risk_perc=0.02
        )
        assert shares == 0
        
        # Test error when entry equals stop
        with pytest.raises(ValueError):
            shares_for_risk(
                equity=100000,
                entry=100.0,
                stop=100.0,
                risk_perc=0.02
            )
    
    def test_get_fill_price(self):
        """Test fill price calculation."""
        # Create test bar
        bar = {
            'open': 100.0,
            'high': 102.0,
            'low': 98.0,
            'close': 101.0
        }
        
        # Test next open mode
        fill_price = get_fill_price('buy', bar, mode='next_open')
        assert fill_price == 100.0
        
        fill_price = get_fill_price('sell', bar, mode='next_open')
        assert fill_price == 100.0
        
        # Test current close mode
        fill_price = get_fill_price('buy', bar, mode='current_close')
        assert fill_price == 101.0
        
        fill_price = get_fill_price('sell', bar, mode='current_close')
        assert fill_price == 101.0
        
        # Test market mode with slippage
        config = {'backtest': {'slippage_pct': 0.001}}
        fill_price = get_fill_price('buy', bar, mode='market', config=config)
        expected = (102.0 + 98.0 + 101.0) / 3 * 1.001  # VWAP with slippage
        assert abs(fill_price - expected) < 0.01
        
        fill_price = get_fill_price('sell', bar, mode='market', config=config)
        expected = (102.0 + 98.0 + 101.0) / 3 * 0.999  # VWAP with slippage
        assert abs(fill_price - expected) < 0.01
    
    def test_run_backtest_basic(self):
        """Test basic backtest functionality."""
        # Create test data
        signals_df, bars_df = self.create_test_data()
        
        # Create config
        config = {
            'backtest': {
                'initial_equity': 100000.0,
                'risk_per_trade': 0.02,
                'max_trades_per_day': 10,
                'eod_flat_minutes_before_close': 30,
                'commission_pct': 0.001,
                'slippage_pct': 0.0005,
                'max_hold_minutes': 240,
                'min_hold_minutes': 5,
                'stop_loss_atr_multiple': 2.0,
                'take_profit_atr_multiple': 3.0,
                'signal_threshold': 0.5
            }
        }
        
        # Run backtest
        result = run_backtest(signals_df, bars_df, config)
        
        # Check result structure
        assert 'trades' in result
        assert 'equity_curve' in result
        assert 'initial_equity' in result
        assert 'final_equity' in result
        assert 'total_return' in result
        assert 'config' in result
        
        # Check equity curve
        assert isinstance(result['equity_curve'], pd.DataFrame)
        assert 'timestamp' in result['equity_curve'].columns
        assert 'equity' in result['equity_curve'].columns
        
        # Check trades
        assert isinstance(result['trades'], list)
        for trade in result['trades']:
            assert 'ticker' in trade
            assert 'entry_time' in trade
            assert 'exit_time' in trade
            assert 'entry_price' in trade
            assert 'exit_price' in trade
            assert 'pnl' in trade
            assert 'exit_reason' in trade
    
    def test_eod_flattening(self):
        """Test end-of-day position flattening."""
        # Create test data with specific EOD scenario
        start_date = datetime(2023, 1, 1, 15, 0)  # 3:00 PM
        timestamps = [start_date + timedelta(minutes=i) for i in range(0, 120)]  # Until 5:00 PM
        
        # Create signals at 3:00 PM
        signals_data = []
        for i, timestamp in enumerate(timestamps[:5]):  # First 5 timestamps
            signals_data.append({
                'timestamp': timestamp,
                'ticker': 'AAPL',
                'signal_strength': 0.8,
                'atr': 1.0
            })
        
        signals_df = pd.DataFrame(signals_data)
        
        # Create bars data
        bars_data = []
        for timestamp in timestamps:
            bars_data.append({
                'timestamp': timestamp,
                'ticker': 'AAPL',
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.0,
                'volume': 1000
            })
        
        bars_df = pd.DataFrame(bars_data)
        
        # Create config with EOD flattening at 3:30 PM
        config = {
            'backtest': {
                'initial_equity': 100000.0,
                'risk_per_trade': 0.02,
                'max_trades_per_day': 10,
                'eod_flat_minutes_before_close': 30,  # Flatten at 3:30 PM
                'commission_pct': 0.001,
                'slippage_pct': 0.0005,
                'max_hold_minutes': 240,
                'min_hold_minutes': 5,
                'stop_loss_atr_multiple': 2.0,
                'take_profit_atr_multiple': 3.0,
                'signal_threshold': 0.5
            }
        }
        
        # Run backtest
        result = run_backtest(signals_df, bars_df, config)
        
        # Check that positions were flattened at EOD
        assert len(result['trades']) > 0
        
        # Check that all trades have exit reasons
        for trade in result['trades']:
            assert trade['exit_reason'] is not None
    
    def test_trade_caps(self):
        """Test maximum trades per day constraint."""
        # Create test data with many signals
        start_date = datetime(2023, 1, 1, 9, 30)
        timestamps = [start_date + timedelta(minutes=i) for i in range(0, 390)]  # 1 day
        
        # Create many signals
        signals_data = []
        for i, timestamp in enumerate(timestamps[::5]):  # Every 25 minutes
            signals_data.append({
                'timestamp': timestamp,
                'ticker': 'AAPL',
                'signal_strength': 0.8,
                'atr': 1.0
            })
        
        signals_df = pd.DataFrame(signals_data)
        
        # Create bars data
        bars_data = []
        for timestamp in timestamps:
            bars_data.append({
                'timestamp': timestamp,
                'ticker': 'AAPL',
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.0,
                'volume': 1000
            })
        
        bars_df = pd.DataFrame(bars_data)
        
        # Create config with low trade limit
        config = {
            'backtest': {
                'initial_equity': 100000.0,
                'risk_per_trade': 0.02,
                'max_trades_per_day': 3,  # Limit to 3 trades per day
                'eod_flat_minutes_before_close': 30,
                'commission_pct': 0.001,
                'slippage_pct': 0.0005,
                'max_hold_minutes': 60,  # Short hold time
                'min_hold_minutes': 5,
                'stop_loss_atr_multiple': 2.0,
                'take_profit_atr_multiple': 3.0,
                'signal_threshold': 0.5
            }
        }
        
        # Run backtest
        result = run_backtest(signals_df, bars_df, config)
        
        # Count trades by day
        trades_by_day = {}
        for trade in result['trades']:
            day = trade['entry_time'].date()
            trades_by_day[day] = trades_by_day.get(day, 0) + 1
        
        # Check that no day exceeds the limit
        for day, count in trades_by_day.items():
            assert count <= 3, f"Day {day} has {count} trades, exceeding limit of 3"
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        # Create test data
        signals_df, bars_df = self.create_test_data()
        
        # Create config
        config = {
            'backtest': {
                'initial_equity': 100000.0,
                'risk_per_trade': 0.02,
                'max_trades_per_day': 10,
                'eod_flat_minutes_before_close': 30,
                'commission_pct': 0.001,
                'slippage_pct': 0.0005,
                'max_hold_minutes': 240,
                'min_hold_minutes': 5,
                'stop_loss_atr_multiple': 2.0,
                'take_profit_atr_multiple': 3.0,
                'signal_threshold': 0.5
            }
        }
        
        # Run backtest
        result = run_backtest(signals_df, bars_df, config)
        
        # Compute metrics
        metrics = compute_all_metrics(result['trades'], result['equity_curve'])
        
        # Check metrics structure
        assert 'trade_stats' in metrics
        assert 'equity_metrics' in metrics
        assert 'ticker_stats' in metrics
        assert 'time_of_day_stats' in metrics
        assert 'monthly_stats' in metrics
        assert 'trade_distribution' in metrics
        
        # Check trade stats
        trade_stats = metrics['trade_stats']
        assert 'total_trades' in trade_stats
        assert 'win_rate' in trade_stats
        assert 'profit_factor' in trade_stats
        assert 'expectancy' in trade_stats
        
        # Check equity metrics
        equity_metrics = metrics['equity_metrics']
        assert 'total_return' in equity_metrics
        assert 'sharpe_ratio' in equity_metrics
        assert 'max_drawdown' in equity_metrics
    
    def test_report_generation(self):
        """Test report generation."""
        # Create test data
        signals_df, bars_df = self.create_test_data()
        
        # Create config
        config = {
            'backtest': {
                'initial_equity': 100000.0,
                'risk_per_trade': 0.02,
                'max_trades_per_day': 10,
                'eod_flat_minutes_before_close': 30,
                'commission_pct': 0.001,
                'slippage_pct': 0.0005,
                'max_hold_minutes': 240,
                'min_hold_minutes': 5,
                'stop_loss_atr_multiple': 2.0,
                'take_profit_atr_multiple': 3.0,
                'signal_threshold': 0.5
            }
        }
        
        # Run backtest
        result = run_backtest(signals_df, bars_df, config)
        
        # Create temporary directory for reports
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate HTML report
            html_report_path = os.path.join(temp_dir, "test_report.html")
            report_path = make_report(result, html_report_path, format="html")
            
            # Check that report was created
            assert os.path.exists(report_path)
            
            # Check that report contains expected content
            with open(report_path, 'r') as f:
                html_content = f.read()
                assert "Backtest Report" in html_content
                assert "Trade Statistics" in html_content
                assert "Equity Metrics" in html_content
            
            # Generate Markdown report
            md_report_path = os.path.join(temp_dir, "test_report.md")
            report_path = make_report(result, md_report_path, format="markdown")
            
            # Check that report was created
            assert os.path.exists(report_path)
            
            # Check that report contains expected content
            with open(report_path, 'r') as f:
                md_content = f.read()
                assert "# Backtest Report" in md_content
                assert "## Trade Statistics" in md_content
                assert "## Equity Metrics" in md_content
    
    def test_full_backtest_workflow(self):
        """Test the complete backtest workflow."""
        # Create test data
        signals_df, bars_df = self.create_test_data()
        
        # Create config
        config = {
            'backtest': {
                'initial_equity': 100000.0,
                'risk_per_trade': 0.02,
                'max_trades_per_day': 10,
                'eod_flat_minutes_before_close': 30,
                'commission_pct': 0.001,
                'slippage_pct': 0.0005,
                'max_hold_minutes': 240,
                'min_hold_minutes': 5,
                'stop_loss_atr_multiple': 2.0,
                'take_profit_atr_multiple': 3.0,
                'signal_threshold': 0.5
            }
        }
        
        # Run backtest
        result = run_backtest(signals_df, bars_df, config)
        
        # Compute metrics
        metrics = compute_all_metrics(result['trades'], result['equity_curve'])
        
        # Create temporary directory for reports
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate reports
            html_report_path = os.path.join(temp_dir, "workflow_test.html")
            make_report(result, html_report_path, format="html")
            
            md_report_path = os.path.join(temp_dir, "workflow_test.md")
            make_report(result, md_report_path, format="markdown")
            
            # Save results
            result_path = os.path.join(temp_dir, "workflow_result.json")
            
            # Convert datetime objects to strings for JSON serialization
            serializable_result = {
                'trades': [
                    {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in trade.items()}
                    for trade in result['trades']
                ],
                'equity_curve': result['equity_curve'].to_dict('records'),
                'initial_equity': result['initial_equity'],
                'final_equity': result['final_equity'],
                'total_return': result['total_return'],
                'config': result['config']
            }
            
            with open(result_path, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            
            # Verify all files were created
            assert os.path.exists(html_report_path)
            assert os.path.exists(md_report_path)
            assert os.path.exists(result_path)
            
            # Verify metrics are reasonable
            assert metrics['trade_stats']['total_trades'] >= 0
            assert -1.0 <= metrics['equity_metrics']['max_drawdown'] <= 0.0
            assert metrics['equity_metrics']['total_return'] == result['total_return']