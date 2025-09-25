"""Entry/exit price modeling, slippage."""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional
import logging

from ..utils.ohlc import is_valid_ohlcv_row

# Configure logger
logger = logging.getLogger(__name__)


def get_fill_price(
    side: str, 
    bar: Union[pd.Series, Dict[str, float]], 
    mode: str = "next_open",
    slippage_pct: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate fill price for a trade with optional slippage.
    
    Args:
        side: Trade side ('buy' or 'sell')
        bar: Price bar data with OHLC values
        mode: Fill mode ('next_open', 'current_close', 'market', 'limit')
        slippage_pct: Slippage percentage as decimal (overrides config)
        config: Configuration dictionary with slippage settings
        
    Returns:
        Fill price as a float
        
    Raises:
        ValueError: If side is not 'buy' or 'sell', or if mode is invalid
    """
    # In debug mode, perform a quick validation on the bar data.
    # This has no overhead in production (when python -O is used).
    if __debug__:
        assert is_valid_ohlcv_row(pd.Series(bar) if isinstance(bar, dict) else bar), \
            f"Invalid OHLCV row provided to get_fill_price: {bar}"

    # Validate side
    side = side.lower()
    if side not in ['buy', 'sell']:
        raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
    
    # Get slippage from config or parameter
    if slippage_pct is None:
        slippage_pct = config.get('backtest', {}).get('slippage_pct', 0.0005) if config else 0.0005
    
    # Convert bar to Series if it's a dict
    if isinstance(bar, dict):
        bar = pd.Series(bar)
    
    # Ensure bar has required fields
    required_fields = ['open', 'high', 'low', 'close']
    for field in required_fields:
        if field not in bar:
            raise ValueError(f"Bar data missing required field: {field}")
    
    # Calculate base price based on mode
    if mode == "next_open":
        # For next open, we use the open price
        base_price = bar['open']
    elif mode == "current_close":
        # For current close, we use the close price
        base_price = bar['close']
    elif mode == "market":
        # For market orders, use VWAP approximation (H+L+C)/3
        base_price = (bar['high'] + bar['low'] + bar['close']) / 3
    elif mode == "limit":
        # For limit orders, we need to know the limit price
        # This would be passed separately in a real implementation
        raise NotImplementedError("Limit orders require limit price parameter")
    else:
        raise ValueError(f"Invalid fill mode: {mode}")
    
    # Apply slippage
    if slippage_pct > 0:
        if side == 'buy':
            # Buys get filled at higher price (unfavorable)
            fill_price = base_price * (1 + slippage_pct)
        else:
            # Sells get filled at lower price (unfavorable)
            fill_price = base_price * (1 - slippage_pct)
    else:
        fill_price = base_price
    
    logger.debug(f"Fill price calculation: side={side}, mode={mode}, base_price={base_price:.4f}, "
                f"slippage_pct={slippage_pct:.4f}, fill_price={fill_price:.4f}")
    
    return fill_price


def get_fill_price_with_spread(
    side: str,
    bar: Union[pd.Series, Dict[str, float]],
    mode: str = "next_open",
    slippage_pct: Optional[float] = None,
    spread_pct: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate fill price with bid-ask spread and slippage.
    
    Args:
        side: Trade side ('buy' or 'sell')
        bar: Price bar data with OHLC values
        mode: Fill mode ('next_open', 'current_close', 'market', 'limit')
        slippage_pct: Slippage percentage as decimal
        spread_pct: Bid-ask spread percentage as decimal
        config: Configuration dictionary with slippage and spread settings
        
    Returns:
        Fill price as a float
    """
    # Get base fill price without spread
    base_fill_price = get_fill_price(side, bar, mode, slippage_pct, config)
    
    # Get spread from config or parameter
    if spread_pct is None:
        spread_pct = config.get('backtest', {}).get('spread_pct', 0.001) if config else 0.001
    
    # Apply spread
    if spread_pct > 0:
        if side == 'buy':
            # Buys pay ask price (higher)
            fill_price = base_fill_price * (1 + spread_pct / 2)
        else:
            # Sells receive bid price (lower)
            fill_price = base_fill_price * (1 - spread_pct / 2)
    else:
        fill_price = base_fill_price
    
    logger.debug(f"Fill price with spread: side={side}, base_fill_price={base_fill_price:.4f}, "
                f"spread_pct={spread_pct:.4f}, fill_price={fill_price:.4f}")
    
    return fill_price


def calculate_commission(
    shares: int,
    price: float,
    commission_pct: Optional[float] = None,
    min_commission: float = 0.0,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate commission for a trade.
    
    Args:
        shares: Number of shares
        price: Price per share
        commission_pct: Commission percentage as decimal
        min_commission: Minimum commission amount
        config: Configuration dictionary with commission settings
        
    Returns:
        Commission amount as a float
    """
    # Get commission from config or parameter
    if commission_pct is None:
        commission_pct = config.get('backtest', {}).get('commission_pct', 0.001) if config else 0.001
    
    # Calculate commission
    trade_value = shares * price
    commission = trade_value * commission_pct
    
    # Apply minimum commission
    commission = max(commission, min_commission)
    
    return commission
