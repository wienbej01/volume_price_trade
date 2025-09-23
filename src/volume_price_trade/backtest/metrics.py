"""Trade/equity stats, drawdowns, heatmaps."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime, timedelta

# Configure logger
logger = logging.getLogger(__name__)


def compute_trade_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a list of trades.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary with trade statistics
    """
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_r_multiple': 0.0,
            'avg_hold_minutes': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'total_pnl': 0.0,
            'total_return': 0.0
        }
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(trades)
    
    # Basic counts
    total_trades = len(df)
    winning_trades = len(df[df['pnl'] > 0])
    losing_trades = len(df[df['pnl'] < 0])
    
    # Win rate
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Profit factor
    gross_profit = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expectancy
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0.0
    avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # R-multiple (assuming risk is 1R per trade)
    # This is a simplified calculation - in practice you'd track actual risk per trade
    risk_per_trade = df['entry_price'] * df['shares'] * 0.02  # Assuming 2% risk
    df['r_multiple'] = df['pnl'] / risk_per_trade
    avg_r_multiple = df['r_multiple'].mean()
    
    # Hold time
    avg_hold_minutes = df['hold_minutes'].mean()
    
    # Extremes
    max_win = df['pnl'].max()
    max_loss = df['pnl'].min()
    
    # Totals
    total_pnl = df['pnl'].sum()
    
    # Calculate return percentage (assuming initial equity)
    # This is a simplified calculation
    initial_equity = 100000.0  # Default assumption
    total_return = total_pnl / initial_equity
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_r_multiple': avg_r_multiple,
        'avg_hold_minutes': avg_hold_minutes,
        'max_win': max_win,
        'max_loss': max_loss,
        'total_pnl': total_pnl,
        'total_return': total_return
    }


def compute_equity_metrics(equity_curve: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute equity curve metrics including drawdowns.
    
    Args:
        equity_curve: DataFrame with timestamp and equity columns
        
    Returns:
        Dictionary with equity metrics
    """
    if equity_curve.empty:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'calmar_ratio': 0.0
        }
    
    # Make a copy to avoid modifying the original
    df = equity_curve.copy()
    
    # Calculate returns
    df['return'] = df['equity'].pct_change()
    
    # Total return
    initial_equity = df['equity'].iloc[0]
    final_equity = df['equity'].iloc[-1]
    total_return = (final_equity - initial_equity) / initial_equity
    
    # Annualized return
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    # Volatility (annualized)
    volatility = df['return'].std() * np.sqrt(252) if len(df) > 1 else 0.0
    
    # Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
    
    # Drawdown analysis
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
    df['drawdown_pct'] = df['drawdown'] * 100
    
    max_drawdown = df['drawdown'].min()
    
    # Max drawdown duration
    is_drawdown = df['drawdown'] < 0
    drawdown_periods = (is_drawdown != is_drawdown.shift()).cumsum()
    drawdown_duration = drawdown_periods[is_drawdown].value_counts().max() if is_drawdown.any() else 0
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': drawdown_duration,
        'calmar_ratio': calmar_ratio
    }


def compute_ticker_stats(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute statistics per ticker.

    Args:
        trades: List of trade dictionaries

    Returns:
        Dictionary with ticker as key and stats as value
    """
    if not trades:
        return {}

    # Convert to DataFrame
    df = pd.DataFrame(trades)

    # Group by ticker
    ticker_stats: Dict[str, Dict[str, Any]] = {}

    for ticker, group in df.groupby('ticker'):
        ticker_trades = group.to_dict('records')
        # Ensure ticker is string and ticker_trades has string keys
        ticker_str = str(ticker)
        ticker_trades_typed = [{str(k): v for k, v in t.items()} for t in ticker_trades]
        ticker_stats[ticker_str] = compute_trade_stats(ticker_trades_typed)

    return ticker_stats


def compute_time_of_day_stats(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute statistics by time of day.

    Args:
        trades: List of trade dictionaries

    Returns:
        Dictionary with time period as key and stats as value
    """
    if not trades:
        return {}

    # Convert to DataFrame
    df = pd.DataFrame(trades)

    # Convert entry times to ET and compute minutes since midnight ET
    times_et = pd.to_datetime(df['entry_time'], utc=True).dt.tz_convert('America/New_York')
    df['time_minutes'] = times_et.dt.hour * 60 + times_et.dt.minute

    # Define time periods based on market hours (9:30 AM - 4:00 PM ET)
    # Convert to minutes since midnight for easier comparison
    time_periods = {
        'morning': (570, 690),    # 9:30 AM - 11:30 AM ET
        'midday': (690, 840),     # 11:30 AM - 2:00 PM ET
        'afternoon': (840, 960)   # 2:00 PM - 4:00 PM ET
    }

    time_stats: Dict[str, Dict[str, Any]] = {}

    for period, (start_minutes, end_minutes) in time_periods.items():
        period_trades = df[(df['time_minutes'] >= start_minutes) & (df['time_minutes'] < end_minutes)]
        period_trades_typed = [{str(k): v for k, v in t.items()} for t in period_trades.to_dict('records')]
        time_stats[period] = compute_trade_stats(period_trades_typed)

    return time_stats


def compute_monthly_stats(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute statistics by month.

    Args:
        trades: List of trade dictionaries

    Returns:
        Dictionary with month as key and stats as value
    """
    if not trades:
        return {}

    # Convert to DataFrame
    df = pd.DataFrame(trades)

    # Extract year-month from entry time
    df['year_month'] = pd.to_datetime(df['entry_time']).dt.to_period('M')

    # Group by month
    monthly_stats: Dict[str, Dict[str, Any]] = {}

    for month, group in df.groupby('year_month'):
        month_trades = group.to_dict('records')
        month_trades_typed = [{str(k): v for k, v in t.items()} for t in month_trades]
        monthly_stats[str(month)] = compute_trade_stats(month_trades_typed)

    return monthly_stats


def compute_trade_distribution(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute trade distribution statistics.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary with distribution statistics
    """
    if not trades:
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    
    # PnL distribution
    pnl_stats = {
        'mean': df['pnl'].mean(),
        'median': df['pnl'].median(),
        'std': df['pnl'].std(),
        'min': df['pnl'].min(),
        'max': df['pnl'].max(),
        'q25': df['pnl'].quantile(0.25),
        'q75': df['pnl'].quantile(0.75),
        'skew': df['pnl'].skew(),
        'kurtosis': df['pnl'].kurtosis()
    }
    
    # Return distribution
    return_stats = {
        'mean': df['return_pct'].mean(),
        'median': df['return_pct'].median(),
        'std': df['return_pct'].std(),
        'min': df['return_pct'].min(),
        'max': df['return_pct'].max(),
        'q25': df['return_pct'].quantile(0.25),
        'q75': df['return_pct'].quantile(0.75)
    }
    
    # Hold time distribution
    hold_time_stats = {
        'mean': df['hold_minutes'].mean(),
        'median': df['hold_minutes'].median(),
        'std': df['hold_minutes'].std(),
        'min': df['hold_minutes'].min(),
        'max': df['hold_minutes'].max()
    }
    
    return {
        'pnl_distribution': pnl_stats,
        'return_distribution': return_stats,
        'hold_time_distribution': hold_time_stats
    }


def compute_all_metrics(
    trades: List[Dict[str, Any]],
    equity_curve: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute all metrics for a backtest.
    
    Args:
        trades: List of trade dictionaries
        equity_curve: DataFrame with equity curve data
        
    Returns:
        Dictionary with all metrics
    """
    # Basic trade stats
    trade_stats = compute_trade_stats(trades)
    
    # Equity metrics
    equity_metrics = compute_equity_metrics(equity_curve)
    
    # Ticker stats
    ticker_stats = compute_ticker_stats(trades)
    
    # Time of day stats
    time_stats = compute_time_of_day_stats(trades)
    
    # Monthly stats
    monthly_stats = compute_monthly_stats(trades)
    
    # Trade distribution
    trade_distribution = compute_trade_distribution(trades)
    
    # Combine all metrics
    all_metrics = {
        'trade_stats': trade_stats,
        'equity_metrics': equity_metrics,
        'ticker_stats': ticker_stats,
        'time_of_day_stats': time_stats,
        'monthly_stats': monthly_stats,
        'trade_distribution': trade_distribution
    }
    
    return all_metrics
