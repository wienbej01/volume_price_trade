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
import joblib
from typing import Dict, List, Optional
# moved import of is_rth below after sys.path insertion

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volume_price_trade.data.calendar import is_rth
from volume_price_trade.backtest.engine import run_backtest
from volume_price_trade.backtest.reports import make_report
from volume_price_trade.backtest.metrics import compute_all_metrics
from volume_price_trade.ml.dataset import make_dataset
from volume_price_trade.utils.io import save_yaml
from volume_price_trade.utils.logging import get_logger, setup_run_logger

# Configure logger
logger = get_logger(__name__)


def generate_signals(
    model_path: str,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    bars_df: pd.DataFrame,
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
    model = joblib.load(model_path)
    
    # Make predictions
    predictions = model.predict_proba(X)
    
    # Create signals DataFrame
    signals_df = meta[['timestamp', 'ticker']].copy()
    
    # For binary classification, use positive class probability
    if len(predictions.shape) == 2 and predictions.shape[1] == 2:
        signals_df['signal_strength'] = predictions[:, 1] - 0.5  # Center around 0
    elif len(predictions.shape) == 2 and predictions.shape[1] > 2:
        # For multi-class, use the maximum probability minus 1/num_classes
        signals_df['signal_strength'] = np.max(predictions, axis=1) - (1.0 / predictions.shape[1])
    else:
        # For regression or single probability, use the raw predictions
        signals_df['signal_strength'] = predictions.flatten() - 0.5  # Center around 0
    
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

    # Build per-ticker RTH timestamp arrays from bars for alignment
    def _safe_is_rth(ts):
        try:
            return is_rth(ts)
        except Exception:
            return False

    rth_timestamps_by_ticker: Dict[str, np.ndarray] = {}
    for tkr in signals_df['ticker'].unique():
        t_bars = bars_df[bars_df['ticker'] == tkr]
        # Ensure sorted and tz-aware timestamps
        t_bars = t_bars.sort_values('timestamp')
        rth_mask = t_bars['timestamp'].apply(_safe_is_rth)
        rth_ts = t_bars.loc[rth_mask, 'timestamp'].to_numpy()
        rth_timestamps_by_ticker[tkr] = rth_ts

    # Align each signal timestamp to the next available RTH bar for that ticker
    pre_align = len(signals_df)
    aligned_ts: List[pd.Timestamp] = []
    dropped = 0
    for _, row in signals_df.iterrows():
        ts = row['timestamp']
        tkr = row['ticker']
        if _safe_is_rth(ts):
            aligned_ts.append(ts)
            continue
        arr = rth_timestamps_by_ticker.get(tkr)
        if arr is None or len(arr) == 0:
            aligned_ts.append(pd.NaT)
            dropped += 1
            continue
        # Find the next RTH timestamp at or after the signal time
        idx = np.searchsorted(arr, ts, side='left')
        if idx >= len(arr):
            aligned_ts.append(pd.NaT)
            dropped += 1
        else:
            aligned_ts.append(arr[idx])

    signals_df['timestamp'] = aligned_ts
    # Drop signals that couldn't be aligned to RTH
    pre_drop = len(signals_df)
    signals_df = signals_df.dropna(subset=['timestamp'])
    post_drop = len(signals_df)
    logger.info(f"RTH alignment: input={pre_align}, dropped_no_alignment={pre_drop - post_drop}, kept={post_drop}")

    # Filter signals by threshold
    signals_df = signals_df[abs(signals_df['signal_strength']) >= threshold].copy()
    signals_df = signals_df.drop_duplicates(subset=['timestamp', 'ticker'])
    
    logger.info(f"Generated {len(signals_df)} signals after RTH alignment with threshold {threshold}")
    
    return signals_df


def prepare_bars_data(
    tickers: list,
    start_date: str,
    end_date: str,
    config: dict = None
) -> pd.DataFrame:
    """
    Prepare bars data for backtesting.

    This function will:
    - Try local data/{ticker}.parquet
    - Fallback to GCS parquet via gcsfs if enabled
    - Fallback to Polygon if enabled
    - Always apply global preprocessing (UTC tz, sort, dedupe)
    - Return a normalized DataFrame with columns: timestamp, open, high, low, close, volume, ticker
    """
    logger.info("Preparing bars data for backtesting...")

    all_bars: list[pd.DataFrame] = []

    # Normalize bounds to UTC
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    start_utc = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt.tz_convert('UTC')
    end_utc = end_dt.tz_localize('UTC') if end_dt.tz is None else end_dt.tz_convert('UTC')

    # Lazy imports to avoid circulars and ensure src path is used
    from volume_price_trade.utils.preprocess import preprocess_bars
    from volume_price_trade.data.gcs_loader import load_minute_bars as gcs_load_minute_bars
    try:
        from volume_price_trade.data.polygon_loader import load_minute_bars_polygon
    except Exception:
        load_minute_bars_polygon = None  # type: ignore

    for ticker in tickers:
        try:
            df = pd.DataFrame()

            # 1) Local parquet
            file_path = Path(f"data/{ticker}.parquet")
            if file_path.exists():
                raw = pd.read_parquet(file_path)
                df = preprocess_bars(raw, ticker=ticker)
                df = df[(df.index >= start_utc) & (df.index <= end_utc)]
            else:
                logger.info(f"Local parquet not found for {ticker} at {file_path}")

            # 2) GCS fallback
            use_gcs = bool((config or {}).get('data', {}).get('use_gcs', True))
            use_gcsfs = bool((config or {}).get('data', {}).get('gcs', {}).get('use_gcsfs', True))
            if (df is None or df.empty) and use_gcs and use_gcsfs:
                try:
                    df = gcs_load_minute_bars(
                        ticker=ticker,
                        start=start_utc.strftime('%Y-%m-%d'),
                        end=end_utc.strftime('%Y-%m-%d')
                    )
                    # gcs_loader already preprocesses and sets index
                    df = df[(df.index >= start_utc) & (df.index <= end_utc)]
                except Exception as e:
                    logger.warning(f"GCS load failed for {ticker}: {e}")
                    df = pd.DataFrame()

            # 3) Polygon fallback
            use_polygon = bool((config or {}).get('data', {}).get('use_polygon', False))
            if (df is None or df.empty) and use_polygon and load_minute_bars_polygon is not None:
                try:
                    raw = load_minute_bars_polygon(
                        ticker=ticker,
                        start=start_utc.strftime('%Y-%m-%d'),
                        end=end_utc.strftime('%Y-%m-%d')
                    )
                    if raw is not None and not raw.empty:
                        df = preprocess_bars(raw, ticker=ticker)
                        df = df[(df.index >= start_utc) & (df.index <= end_utc)]
                except Exception as e:
                    logger.warning(f"Polygon load failed for {ticker}: {e}")

            if df is None or df.empty:
                logger.warning(f"No bars found for {ticker} in range {start_date}..{end_date}")
                continue

            # Normalize to expected columns
            tmp = df[['open', 'high', 'low', 'close', 'volume']].copy()
            tmp['ticker'] = ticker
            tmp = tmp.reset_index().rename(columns={'index': 'timestamp'})
            all_bars.append(tmp[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ticker']])

        except Exception as e:
            logger.error(f"Error preparing bars data for ticker {ticker}: {e}", exc_info=True)
            continue

    if not all_bars:
        raise ValueError("No bars data could be prepared")

    # Combine and normalize timestamps to UTC tz-aware
    combined_bars = pd.concat(all_bars, ignore_index=True)
    combined_bars['timestamp'] = pd.to_datetime(combined_bars['timestamp'], utc=True)
    combined_bars = combined_bars.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

    logger.info(f"Prepared {len(combined_bars)} bars for backtesting across {len(tickers)} tickers")

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
        default=0.1,
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
        
        # Prepare bars data (used for RTH alignment)
        bars_df = prepare_bars_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            config=config
        )

        # Generate signals (snap to next RTH bar)
        signals_df = generate_signals(
            model_path=str(model_path),
            X=X,
            meta=meta,
            bars_df=bars_df,
            threshold=args.signal_threshold,
            config=config
        )
        
        if signals_df.empty:
            print("No signals were generated after RTH alignment; the signals_df is empty.")
            raise ValueError("No signals generated - check model, threshold, and RTH alignment")
        
        signals_df = signals_df.reset_index(drop=True)
        
        # Create backtest config (map project config into backtest engine schema)
        risk_cfg = config.get('risk', {}) or {}
        sess_cfg = config.get('sessions', {}) or {}

        backtest_cfg = {
            # Intraday constraints
            'min_hold_minutes': 5,
            'max_hold_minutes': 240,
            'eod_flat_minutes_before_close': int(sess_cfg.get('eod_flat_minutes_before_close', 1)),

            # Risk and sizing
            'initial_equity': float(risk_cfg.get('start_equity', 100000.0)),
            'risk_per_trade': float(risk_cfg.get('risk_perc', 0.02)),
            'max_trades_per_day': int(risk_cfg.get('max_trades_per_day', 5)),

            # Execution frictions (fallback defaults if not present)
            'commission_pct': float(risk_cfg.get('commission_pct', 0.001)),
            'slippage_pct': float(risk_cfg.get('slippage_pct', 0.0005)),

            # Signal gating
            'signal_threshold': float(args.signal_threshold),
        }

        backtest_config = {'backtest': backtest_cfg}

        # Run backtest
        logger.info("Running backtest...")
        bt_result = run_backtest(
            signals_df=signals_df,
            bars_df=bars_df,
            config=backtest_config
        )

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = compute_all_metrics(
            trades=bt_result['trades'],
            equity_curve=bt_result['equity_curve']
        )
        bt_result['metrics'] = metrics

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
        print(f"Sharpe Ratio: {bt_result['metrics']['equity_metrics']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {bt_result['metrics']['equity_metrics']['max_drawdown']:.2%}")
        print(f"Win Rate: {bt_result['metrics']['trade_stats']['win_rate']:.2%}")
        print(f"Report saved to: {report_path}")
        print(f"Results saved to: {result_path}")
        
    except Exception as e:
        logger.error(f"Error in backtest run {run_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
