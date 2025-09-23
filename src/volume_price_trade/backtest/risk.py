"""2% VaR position sizing given stop distance."""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any
import logging

# Configure logger
logger = logging.getLogger(__name__)


def shares_for_risk(equity: float, entry: float, stop: float, risk_perc: float) -> int:
    """
    Calculate the number of shares to trade based on risk percentage with robust guards.
    """
    # Validate base inputs
    if equity <= 0 or risk_perc <= 0:
        logger.warning("Non-positive equity or risk_perc; returning 0 shares")
        return 0

    # Guard against NaN/inf and non-positive prices
    vals = [entry, stop]
    if any((x is None) for x in vals):
        logger.warning("Entry/stop is None; returning 0 shares")
        return 0
    if any((isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))) for x in vals):
        logger.warning("Entry/stop NaN or non-finite; returning 0 shares")
        return 0
    if entry <= 0 or stop <= 0:
        logger.warning("Entry/stop must be positive; returning 0 shares")
        return 0

    # Risk amount and stop distance
    risk_amount = float(equity) * float(risk_perc)
    stop_distance = abs(float(entry) - float(stop))

    if not np.isfinite(stop_distance) or stop_distance <= 0:
        logger.warning("Invalid or zero stop distance; returning 0 shares")
        return 0

    shares = risk_amount / stop_distance
    if not np.isfinite(shares) or shares <= 0:
        return 0

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
