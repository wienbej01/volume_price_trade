"""2% VaR position sizing given stop distance."""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any
import logging

# Configure logger
logger = logging.getLogger(__name__)


def shares_for_risk(equity: float, entry: float, stop: float, risk_perc: float) -> int:
    """
    Calculate the number of shares to trade based on risk percentage.
    
    This function implements position sizing based on a fixed percentage
    of equity risk per trade, using the stop loss distance to determine
    the appropriate position size.
    
    Args:
        equity: Total account equity
        entry: Entry price per share
        stop: Stop loss price per share
        risk_perc: Risk percentage as a decimal (e.g., 0.02 for 2%)
        
    Returns:
        Integer number of shares to trade
        
    Raises:
        ValueError: If entry equals stop (division by zero)
    """
    # Validate inputs
    if equity <= 0:
        logger.warning("Equity must be positive, returning 0 shares")
        return 0
        
    if entry <= 0 or stop <= 0:
        logger.warning("Entry and stop prices must be positive, returning 0 shares")
        return 0
        
    if risk_perc <= 0:
        logger.warning("Risk percentage must be positive, returning 0 shares")
        return 0
    
    # Calculate risk amount in currency
    risk_amount = equity * risk_perc
    
    # Calculate stop distance
    stop_distance = abs(entry - stop)
    
    # Avoid division by zero
    if stop_distance == 0:
        raise ValueError("Entry price cannot equal stop price")
    
    # Calculate number of shares
    shares = risk_amount / stop_distance
    
    # Return as integer (floor to ensure we don't exceed risk)
    return int(np.floor(shares))


def calculate_position_size(
    equity: float,
    entry: float,
    stop: float,
    risk_perc: float,
    max_position_pct: float = 1.0,
    commission_pct: float = 0.0
) -> Dict[str, Union[int, float]]:
    """
    Calculate position size with additional constraints.
    
    Args:
        equity: Total account equity
        entry: Entry price per share
        stop: Stop loss price per share
        risk_perc: Risk percentage as a decimal
        max_position_pct: Maximum position size as percentage of equity
        commission_pct: Commission percentage as a decimal
        
    Returns:
        Dictionary with position details:
        - shares: Number of shares
        - position_value: Total value of position
        - risk_amount: Risk amount in currency
        - commission: Commission amount
    """
    # Calculate base shares from risk
    shares = shares_for_risk(equity, entry, stop, risk_perc)
    
    # Calculate position value
    position_value = shares * entry
    
    # Apply maximum position size constraint
    max_position_value = equity * max_position_pct
    if position_value > max_position_value:
        # Scale down shares to fit within max position size
        shares = int(np.floor(max_position_value / entry))
        position_value = shares * entry
    
    # Calculate commission
    commission = position_value * commission_pct
    
    # Calculate actual risk amount
    stop_distance = abs(entry - stop)
    risk_amount = shares * stop_distance
    
    return {
        'shares': shares,
        'position_value': position_value,
        'risk_amount': risk_amount,
        'commission': commission
    }
