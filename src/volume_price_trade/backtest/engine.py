"""Core simulator: 1 position per ticker, hold 5–240m, force flat EOD."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from .risk import shares_for_risk, calculate_position_size
from .fills import get_fill_price, calculate_commission

# Configure logger
logger = logging.getLogger(__name__)


class Position:
    """Represents a trading position."""

    def __init__(self, ticker: str, entry_time: datetime, entry_price: float,
                 shares: int, direction: str, stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None, max_hold_minutes: Optional[int] = None):
        """
        Initialize a position.

        Args:
            ticker: Ticker symbol
            entry_time: Entry timestamp
            entry_price: Entry price
            shares: Number of shares
            direction: 'long' or 'short'
            stop_loss: Stop loss price
            take_profit: Take profit price
            max_hold_minutes: Maximum hold time in minutes
        """
        self.ticker = ticker
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.shares = shares
        self.direction = direction
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_minutes = max_hold_minutes
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.exit_reason: Optional[str] = None
        self.commission = 0.0
        self.slippage = 0.0
        
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.exit_time is None
    
    def close(self, exit_time: datetime, exit_price: float, reason: str):
        """Close the position."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = reason
    
    def pnl(self) -> float:
        """Calculate profit/loss."""
        if self.exit_price is None:
            return 0.0

        assert self.exit_price is not None  # For mypy
        if self.direction == 'long':
            return (self.exit_price - self.entry_price) * self.shares - self.commission
        else:  # short
            return (self.entry_price - self.exit_price) * self.shares - self.commission
    
    def return_pct(self) -> float:
        """Calculate return as percentage."""
        if self.exit_price is None:
            return 0.0

        assert self.exit_price is not None  # For mypy
        if self.direction == 'long':
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        else:  # short
            return (self.entry_price - self.exit_price) / self.entry_price * 100
    
    def hold_minutes(self) -> Optional[int]:
        """Calculate hold time in minutes."""
        if self.exit_time is None:
            return None
        assert self.exit_time is not None  # For mypy
        return int((self.exit_time - self.entry_time).total_seconds() / 60)


def run_backtest(
    signals_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a backtest on signals with bars data.
    
    Args:
        signals_df: DataFrame with signals (timestamp, ticker, signal_strength, etc.)
        bars_df: DataFrame with OHLCV data (timestamp, ticker, open, high, low, close, volume)
        config: Configuration dictionary with backtest settings
        
    Returns:
        Dictionary with backtest results including trades, equity curve, and metrics
    """
    # Extract configuration
    from ..data.calendar import next_session_close
    initial_equity = config.get('backtest', {}).get('initial_equity', 100000.0)
    risk_per_trade = config.get('backtest', {}).get('risk_per_trade', 0.02)
    max_trades_per_day = config.get('backtest', {}).get('max_trades_per_day', 10)
    eod_flat_minutes_before_close = config.get('backtest', {}).get('eod_flat_minutes_before_close', 30)
    commission_pct = config.get('backtest', {}).get('commission_pct', 0.001)
    slippage_pct = config.get('backtest', {}).get('slippage_pct', 0.0005)
    max_hold_minutes = config.get('backtest', {}).get('max_hold_minutes', 240)
    min_hold_minutes = config.get('backtest', {}).get('min_hold_minutes', 5)
    stop_loss_atr_multiple = config.get('backtest', {}).get('stop_loss_atr_multiple', 2.0)
    take_profit_atr_multiple = config.get('backtest', {}).get('take_profit_atr_multiple', 3.0)
    signal_threshold = config.get('backtest', {}).get('signal_threshold', 0.5)
    
    # Initialize state
    equity = initial_equity
    positions: Dict[str, Position] = {}  # ticker -> Position
    trades = []
    equity_curve = []
    daily_trade_count: Dict[datetime, int] = defaultdict(int)
    
    # Ensure data is sorted by timestamp
    signals_df = signals_df.sort_values('timestamp')
    bars_df = bars_df.sort_values('timestamp')

    # Get unique timestamps from both signals and bars to ensure we process all market timestamps
    # This ensures EOD flat logic runs even on days with no signals
    signal_timestamps = set(signals_df['timestamp'].unique())
    bar_timestamps = set(bars_df['timestamp'].unique())
    unique_timestamps = sorted(signal_timestamps.union(bar_timestamps))
    
    # Process each timestamp
    for timestamp in unique_timestamps:
        timestamp_signals = signals_df[signals_df['timestamp'] == timestamp]
        
        # Get bars for this timestamp
        timestamp_bars = bars_df[bars_df['timestamp'] == timestamp]
        
        if timestamp_bars.empty:
            continue
            
        # Check if we need to flatten positions at EOD
        # Use calendar helper for accurate session close time
        session_close = next_session_close(timestamp)
        flatten_time = session_close - timedelta(minutes=eod_flat_minutes_before_close)
        if timestamp >= flatten_time and timestamp < session_close:
            # Close all open positions
            for ticker, position in list(positions.items()):
                if position.is_open():
                    # Get the bar for this ticker
                    ticker_bar = timestamp_bars[timestamp_bars['ticker'] == ticker]
                    if not ticker_bar.empty:
                        bar = ticker_bar.iloc[0]
                        exit_price = get_fill_price(
                            'sell' if position.direction == 'long' else 'buy',
                            bar,
                            mode='current_close',
                            slippage_pct=slippage_pct,
                            config=config
                        )
                        
                        # Calculate commission
                        commission = calculate_commission(
                            position.shares,
                            exit_price,
                            commission_pct=commission_pct,
                            config=config
                        )
                        
                        # Close position
                        position.close(timestamp, exit_price, 'eod_flat')
                        position.commission = commission
                        
                        # Update equity
                        equity += position.pnl()
                        
                        # Record trade
                        trades.append(_position_to_trade(position))
                        
                        # Remove from positions
                        del positions[ticker]
                        
                        logger.debug(f"Closed position {ticker} at EOD: PnL={position.pnl():.2f}")
        
        # Process signals
        for _, signal in timestamp_signals.iterrows():
            ticker = signal['ticker']
            signal_strength = signal.get('signal_strength', 1.0)
            
            # Skip if signal is below threshold
            if abs(signal_strength) < signal_threshold:
                continue
                
            # Skip if we already have a position for this ticker
            if ticker in positions and positions[ticker].is_open():
                continue
                
            # Check daily trade limit
            trade_date = timestamp.date()
            if daily_trade_count[trade_date] >= max_trades_per_day:
                continue
                
            # Get the bar for this ticker
            ticker_bar = timestamp_bars[timestamp_bars['ticker'] == ticker]
            if ticker_bar.empty:
                continue
                
            bar = ticker_bar.iloc[0]
            
            # Determine direction based on signal
            direction = 'long' if signal_strength > 0 else 'short'
            
            # Calculate stop loss and take profit (robust to NaN ATR/entry)
            atr_val = signal.get('atr', np.nan)
            try:
                import pandas as pd  # ensure available
            except Exception:
                pass
            if (isinstance(atr_val, float) and (np.isnan(atr_val) or not np.isfinite(atr_val))) or atr_val is None or atr_val <= 0:
                # Fallback to current bar range
                fallback = (bar['high'] - bar['low'])
                atr_val = float(fallback) if np.isfinite(fallback) and fallback > 0 else 1e-6

            if direction == 'long':
                stop_loss = bar['close'] - (stop_loss_atr_multiple * atr_val)
                take_profit = bar['close'] + (take_profit_atr_multiple * atr_val)
            else:  # short
                stop_loss = bar['close'] + (stop_loss_atr_multiple * atr_val)
                take_profit = bar['close'] - (take_profit_atr_multiple * atr_val)

            # Entry price fallback
            entry_price = bar.get('open', np.nan)
            if (isinstance(entry_price, float) and (np.isnan(entry_price) or entry_price <= 0)) or entry_price is None:
                entry_price = bar.get('close', np.nan)
            if not (isinstance(entry_price, float) and np.isfinite(entry_price) and entry_price > 0):
                # Cannot size position without a valid entry; skip this signal
                continue

            # Calculate position size
            position_info = calculate_position_size(
                equity=equity,
                entry=float(entry_price),
                stop=float(stop_loss) if np.isfinite(stop_loss) else entry_price,
                risk_perc=risk_per_trade,
                commission_pct=commission_pct
            )
            
            shares = int(position_info['shares'])

            # Skip if position size is zero
            if shares == 0:
                continue

            # Create position
            position = Position(
                ticker=ticker,
                entry_time=timestamp,
                entry_price=bar['open'],
                shares=shares,
                direction=direction,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_hold_minutes=max_hold_minutes
            )
            
            # Add commission
            position.commission = position_info['commission']
            
            # Add to positions
            positions[ticker] = position
            
            # Update daily trade count
            daily_trade_count[trade_date] += 1
            
            # Update equity (reduce by commission)
            equity -= position.commission
            
            logger.debug(f"Opened position {ticker}: {direction} {shares} shares @ {bar['open']:.2f}")
        
        # Check for exits on open positions
        for ticker, position in list(positions.items()):
            if not position.is_open():
                continue
                
            # Get the bar for this ticker
            ticker_bar = timestamp_bars[timestamp_bars['ticker'] == ticker]
            if ticker_bar.empty:
                continue
                
            bar = ticker_bar.iloc[0]
            
            # Check if position should be closed
            close_reason: Optional[str] = None
            close_price: Optional[float] = None
            
            # Check stop loss
            if position.stop_loss is not None:
                if (position.direction == 'long' and bar['low'] <= position.stop_loss) or \
                    (position.direction == 'short' and bar['high'] >= position.stop_loss):
                    close_reason = 'stop_loss'
                    close_price = position.stop_loss

            # Check take profit
            if close_reason is None and position.take_profit is not None:
                if (position.direction == 'long' and bar['high'] >= position.take_profit) or \
                    (position.direction == 'short' and bar['low'] <= position.take_profit):
                    close_reason = 'take_profit'
                    close_price = position.take_profit

            # Check max hold time
            if close_reason is None and position.max_hold_minutes is not None:
                hold_minutes = (timestamp - position.entry_time).total_seconds() / 60
                if hold_minutes >= position.max_hold_minutes:
                    close_reason = 'timeout'
                    close_price = get_fill_price(
                        'sell' if position.direction == 'long' else 'buy',
                        bar,
                        mode='current_close',
                        slippage_pct=slippage_pct,
                        config=config
                    )

            # Check minimum hold time
            if close_reason is not None:
                hold_minutes = (timestamp - position.entry_time).total_seconds() / 60
                if hold_minutes < min_hold_minutes:
                    # Don't exit if minimum hold time not reached
                    close_reason = None
                    close_price = None

            # Close position if needed
            if close_reason is not None:
                if close_price is None:
                    # Use market fill price if not set
                    close_price = get_fill_price(
                        'sell' if position.direction == 'long' else 'buy',
                        bar,
                        mode='market',
                        slippage_pct=slippage_pct,
                        config=config
                    )

                # Close position
                position.close(timestamp, close_price, close_reason)

                # Update equity
                equity += position.pnl()

                # Record trade
                trades.append(_position_to_trade(position))

                # Remove from positions
                del positions[ticker]

                logger.debug(f"Closed position {ticker} ({close_reason}): PnL={position.pnl():.2f}")
        
        # Record equity curve
        equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'open_positions': len([p for p in positions.values() if p.is_open()])
        })
    
    # Close any remaining positions at the end
    for ticker, position in positions.items():
        if position.is_open():
            # Get the last bar for this ticker
            last_bar = bars_df[bars_df['ticker'] == ticker].iloc[-1]
            
            exit_price = get_fill_price(
                'sell' if position.direction == 'long' else 'buy',
                last_bar,
                mode='current_close',
                slippage_pct=slippage_pct,
                config=config
            )
            
            # Close position
            position.close(last_bar['timestamp'], exit_price, 'end_of_backtest')
            
            # Update equity
            equity += position.pnl()
            
            # Record trade
            trades.append(_position_to_trade(position))
            
            logger.debug(f"Closed position {ticker} at end of backtest: PnL={position.pnl():.2f}")
    
    # Create result dictionary
    result = {
        'trades': trades,
        'equity_curve': pd.DataFrame(equity_curve),
        'initial_equity': initial_equity,
        'final_equity': equity,
        'total_return': (equity - initial_equity) / initial_equity,
        'config': config
    }
    
    logger.info(f"Backtest completed: {len(trades)} trades, "
               f"Total return: {result['total_return']:.2%}")
    
    return result


# Remove the old _is_eod_flatten_time function as it's no longer used


def _position_to_trade(position: Position) -> Dict[str, Any]:
    """
    Convert a Position object to a trade dictionary.
    
    Args:
        position: Position object
        
    Returns:
        Dictionary with trade information
    """
    return {
        'ticker': position.ticker,
        'entry_time': position.entry_time,
        'entry_price': position.entry_price,
        'exit_time': position.exit_time,
        'exit_price': position.exit_price,
        'direction': position.direction,
        'shares': position.shares,
        'pnl': position.pnl(),
        'return_pct': position.return_pct(),
        'hold_minutes': position.hold_minutes(),
        'exit_reason': position.exit_reason,
        'commission': position.commission
    }
