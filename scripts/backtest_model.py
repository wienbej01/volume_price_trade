#!/usr/bin/env python
"""Backtest script for volume price trade models."""

import argparse
import sys
import yaml
import uuid
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volume_price_trade.ml.predict import load_model, predict
from volume_price_trade.backtest.engine import run_backtest
from volume_price_trade.backtest.reports import make_report
from volume_price_trade.ml.dataset import make_dataset
from volume_price_trade.utils.io import save_yaml
from volume_price_trade.utils.logging import get_logger, setup_run_logger

# Configure logger
logger = get_logger(__name__)


def generate_signals(
    model_path: str,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    threshold: float = 0.5,
    config: dict = None
) -> pd.DataFrame:
    """
    Generate trading signals from model predictions.
    
    Args:
        model_path: Path to the trained model
        X: Feature DataFrame
        meta: Metadata DataFrame with timestamps and tickers
        threshold: Signal threshold for taking trades
        config: Configuration dictionary
        
    Returns:
        DataFrame with signals (timestamp, ticker, signal_strength)
    """
    logger.info("Generating signals from model predictions...")
    
    # Load model
    model = load_model(model_path)
    
    # Make predictions
    predictions = predict(model, X)
    
    # Create signals DataFrame
    signals_df = meta[['timestamp', 'ticker']].copy()
    
    # For binary classification, use positive class probability
    if len(predictions.shape) == 2 and predictions.shape[1] == 2:
        signals_df['signal_strength'] = predictions[:, 1] - 0.5  # Center around 0
    else:
        # For regression or multi-class, use the raw predictions
        signals_df['signal_strength'] = predictions - 0.5  # Center around 0
    
    # Add ATR if available (for stop loss/take profit calculations)
    if 'atr' in X.columns:
        signals_df['atr'] = X['atr'].values
    else:
        # Calculate a simple ATR approximation if not available
        high_cols = [col for col in X.columns if 'high' in col.lower()]
        low_cols = [col for col in X.columns if 'low' in col.lower()]
        
        if high_cols and low_cols:
            signals_df['atr'] = X[high_cols[0]] - X[low_cols[0]]
        else:
            # Use a default ATR value
            signals_df['atr'] = 1.0
    
    # Filter signals by threshold
    signals_df = signals_df[abs(signals_df['signal_strength']) >= threshold].copy()
    
    logger.info(f"Generated {len(signals_df)} signals with threshold {threshold}")
    
    return signals_df


def prepare_bars_data(
    tickers: list,
    start_date: str,
    end_date: str,
    config: dict = None
) -> pd.DataFrame:
    """
    Prepare bars data for backtesting.
    
    Args:
        tickers: List of tickers
        start_date: Start date string
        end_date: End date string
        config: Configuration dictionary
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info("Preparing bars data for backtesting...")
    
    # This is a simplified implementation
    # In a real system, you would load actual bar data
    
    # For now, we'll create synthetic bars data based on the features
    # This is just for demonstration purposes
    
    all_bars = []
    
    for ticker in tickers:
        # Load dataset for this ticker
        try:
            X, y, meta = make_dataset(
                tickers=[ticker],
                start=start_date,
                end=end_date,
                config=config
            )
            
            if X.empty:
                logger.warning(f"No data found for ticker {ticker}")
                continue
            
            # Extract OHLC data from features if available
            ohlc_cols = {}
            for col in X.columns:
                if 'open' in col.lower():
                    ohlc_cols['open'] = col
                elif 'high' in col.lower():
                    ohlc_cols['high'] = col
                elif 'low' in col.lower():
                    ohlc_cols['low'] = col
                elif 'close' in col.lower():
                    ohlc_cols['close'] = col
                elif 'volume' in col.lower():
                    ohlc_cols['volume'] = col
            
            # Create bars DataFrame
            bars_df = meta[['timestamp', 'ticker']].copy()
            
            # Add OHLC data
            for ohlc_key, col_name in ohlc_cols.items():
                bars_df[ohlc_key] = X[col_name].values
            
            # Fill missing OHLC data with defaults if needed
            if 'open' not in bars_df.columns:
                bars_df['open'] = 100.0  # Default price
            if 'high' not in bars_df.columns:
                bars_df['high'] = bars_df['open'] * 1.01  # 1% higher
            if 'low' not in bars_df.columns:
                bars_df['low'] = bars_df['open'] * 0.99  # 1% lower
            if 'close' not in bars_df.columns:
                bars_df['close'] = bars_df['open']  # Same as open
            if 'volume' not in bars_df.columns:
                bars_df['volume'] = 10000  # Default volume
            
            all_bars.append(bars_df)
            
        except Exception as e:
            logger.error(f"Error preparing bars data for ticker {ticker}: {e}")
            continue
    
    if not all_bars:
        raise ValueError("No bars data could be prepared")
    
    # Combine all bars
    combined_bars = pd.concat(all_bars, ignore_index=True)
    
    logger.info(f"Prepared {len(combined_bars)} bars for backtesting")
    
    return combined_bars


def main():
    """Main entry point for the backtest script."""
    parser = argparse.ArgumentParser(description="Backtest a volume price trade model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Path to configuration YAML file (default: config/base.yaml)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=None,
        help="List of tickers to backtest (default: use config)"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--signal_threshold",
        type=float,
        default=0.5,
        help="Signal threshold for taking trades (default: 0.5)"
    )
    parser.add_argument(
        "--report_format",
        type=str,
        choices=["html", "markdown"],
        default="html",
        help="Report format (default: html)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for reports (default: artifacts/reports/<run_id>)"
    )
    
    args = parser.parse_args()
    
    # Generate unique run ID
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    
    # Set up run-specific logger
    logger = setup_run_logger(run_id)
    logger.info(f"Starting backtest run {run_id}")
    
    # Validate model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get tickers from args or config
    if args.tickers:
        tickers = args.tickers
    else:
        tickers = config.get('tickers', {}).get('train', [])
    
    if not tickers:
        print("Error: No tickers specified")
        sys.exit(1)
    
    # Determine date range
    if args.start_date:
        start_date = args.start_date
    else:
        # Default to 6 months ago
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    if args.end_date:
        end_date = args.end_date
    else:
        # Default to today
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"artifacts/reports/{run_id}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset for features and metadata
        logger.info("Loading dataset for backtesting...")
        X, y, meta = make_dataset(
            tickers=tickers,
            start=start_date,
            end=end_date,
            config=config
        )
        
        if X.empty:
            raise ValueError("Failed to load dataset - empty features")
        
        logger.info(f"Dataset loaded: {len(X)} samples, {len(X.columns)} features")
        
        # Generate signals
        signals_df = generate_signals(
            model_path=str(model_path),
            X=X,
            meta=meta,
            threshold=args.signal_threshold,
            config=config
        )
        
        if signals_df.empty:
            raise ValueError("No signals generated - check model and threshold")
        
        # Prepare bars data
        bars_df = prepare_bars_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            config=config
        )
        
        # Run backtest
        logger.info("Running backtest...")
        bt_result = run_backtest(
            signals_df=signals_df,
            bars_df=bars_df,
            config=config
        )
        
        # Generate report
        logger.info("Generating report...")
        report_path = str(output_dir / f"backtest_report.{args.report_format}")
        make_report(
            bt_result=bt_result,
            path=report_path,
            format=args.report_format,
            include_plots=True
        )
        
        # Save backtest result
        result_path = str(output_dir / "backtest_result.json")
        bt_result['trades'] = [
            {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in trade.items()}
            for trade in bt_result['trades']
        ]
        bt_result['equity_curve'] = bt_result['equity_curve'].to_dict('records')
        
        with open(result_path, 'w') as f:
            json.dump(bt_result, f, indent=2, default=str)
        
        # Print summary
        print(f"\nBacktest completed successfully!")
        print(f"Run ID: {run_id}")
        print(f"Total Trades: {len(bt_result['trades'])}")
        print(f"Total Return: {bt_result['total_return']:.2%}")
        print(f"Sharpe Ratio: {bt_result['config']['backtest'].get('sharpe_ratio', 'N/A')}")
        print(f"Max Drawdown: {bt_result['config']['backtest'].get('max_drawdown', 'N/A'):.2%}")
        print(f"Report saved to: {report_path}")
        print(f"Results saved to: {result_path}")
        
    except Exception as e:
        logger.error(f"Error in backtest run {run_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
