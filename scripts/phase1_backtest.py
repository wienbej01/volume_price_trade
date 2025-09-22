#!/usr/bin/env python3
"""
Phase 1 Backtest: Test trained model on unseen data.

Uses OOS tickers from config, 6 months backtest period.
No look-forward bias: backtest on future data not seen in training.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volume_price_trade.backtest.engine import run_backtest
from volume_price_trade.ml.dataset import make_dataset
import pandas as pd
import numpy as np
import joblib

def main(model_run_id=None):
    if not model_run_id:
        # Find latest run
        artifacts_dir = Path("artifacts/models")
        if not artifacts_dir.exists():
            print("No artifacts found. Run training first.")
            return

        runs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
        if not runs:
            print("No model runs found.")
            return

        latest_run = max(runs, key=lambda x: x.stat().st_mtime)
        model_run_id = latest_run.name
        print(f"Using latest run: {model_run_id}")

    model_dir = f"artifacts/models/{model_run_id}"
    config_path = f"{model_dir}/config.yaml"

    # Load config to get OOS tickers
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    oos_tickers = config.get('tickers', {}).get('oos', [])
    if not oos_tickers:
        print("No OOS tickers in config. Using default.")
        oos_tickers = ["REGN", "LLY", "TMO"]  # Sample

    print(f"Phase 1 Backtest: {len(oos_tickers)} OOS tickers")
    print(f"Model: {model_run_id}")
    print(f"Tickers: {', '.join(oos_tickers[:5])}..." if len(oos_tickers) > 5 else ', '.join(oos_tickers))

    # Backtest period: 6 months after training end
    # Training uses ~2 years to end of last month
    end_date = datetime.now().replace(day=1) - timedelta(days=1)
    start_date = end_date - timedelta(days=180)  # 6 months back

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Backtest period: {start_str} to {end_str}")

    # 1) Load dataset (features/meta) for OOS tickers and window
    from volume_price_trade.ml.dataset import make_dataset
    X, y, meta = make_dataset(
        tickers=oos_tickers,
        start=start_str,
        end=end_str,
        config=config
    )
    if X.empty:
        print("Error: No features built for backtest window/tickers")
        return

    # 2) Load model
    model_path = Path(model_dir) / "model.joblib"
    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        return
    model = joblib.load(str(model_path))

    # 3) Generate signals (binary/multiclass robust)
    preds = model.predict_proba(X)
    signals_df = meta[['timestamp', 'ticker']].copy()
    if preds.ndim == 2 and preds.shape[1] == 2:
        signals_df['signal_strength'] = preds[:, 1] - 0.5
    elif preds.ndim == 2 and preds.shape[1] > 2:
        signals_df['signal_strength'] = np.max(preds, axis=1) - (1.0 / preds.shape[1])
    else:
        preds = np.asarray(preds).reshape(-1)
        signals_df['signal_strength'] = preds - 0.5

    # Provide ATR to engine if available (use TA feature name)
    if 'atr_20' in X.columns:
        signals_df['atr'] = X['atr_20'].values

    # Optional thresholding for sparsity
    threshold = float(config.get('backtest', {}).get('signal_threshold', 0.1))
    signals_df = signals_df[signals_df['signal_strength'].abs() >= threshold].copy()

    # 4) Prepare bars data (OHLCV) for same window
    def _prepare_bars_data(tickers, start_date, end_date):
        all_bars = []
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        for t in tickers:
            fp = Path(f"data/{t}.parquet")
            if not fp.exists():
                continue
            dfb = pd.read_parquet(fp)
            # Ensure DatetimeIndex
            if not isinstance(dfb.index, pd.DatetimeIndex):
                if 'timestamp' in dfb.columns:
                    dfb['timestamp'] = pd.to_datetime(dfb['timestamp'])
                    dfb = dfb.set_index('timestamp')
                else:
                    dfb.index = pd.to_datetime(dfb.index)
            if dfb.index.tz is None:
                dfb.index = dfb.index.tz_localize('UTC')
            else:
                dfb.index = dfb.index.tz_convert('UTC')
            dfb = dfb[(dfb.index >= start_dt) & (dfb.index <= end_dt)].copy()
            if dfb.empty:
                continue
            dfb['ticker'] = t
            dfb = dfb.reset_index().rename(columns={'index': 'timestamp'})
            all_bars.append(dfb)
        if not all_bars:
            return pd.DataFrame()
        return pd.concat(all_bars, axis=0)

    bars_df = _prepare_bars_data(oos_tickers, start_str, end_str)
    if bars_df.empty:
        print("Error: No bars data available for backtest tickers/window")
        return

    # 5) Map risk/session config to backtest config
    risk = config.get('risk', {})
    sessions = config.get('sessions', {})
    backtest_config = {
        'backtest': {
            'initial_equity': float(risk.get('start_equity', 10000)),
            'risk_per_trade': float(risk.get('risk_perc', 0.02)),
            'max_trades_per_day': int(risk.get('max_trades_per_day', 5)),
            'eod_flat_minutes_before_close': int(sessions.get('eod_flat_minutes_before_close', 1)),
            'stop_loss_atr_multiple': float(risk.get('atr_stop_mult', 1.5)),
            'take_profit_atr_multiple': float(risk.get('take_profit_R', 2.0)),
            'signal_threshold': threshold
        }
    }

    # 6) Run backtest
    results = run_backtest(
        signals_df=signals_df,
        bars_df=bars_df,
        config=backtest_config
    )

    # 7) Print summary
    print("\nBacktest completed!")
    print(f"Total Trades: {len(results.get('trades', []))}")
    print(f"Total Return: {results.get('total_return', 0):.2%}")
    eq = results.get('equity_curve')
    if isinstance(eq, pd.DataFrame):
        print(f"Equity Curve Points: {len(eq)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Backtest")
    parser.add_argument("--model-run-id", help="Specific run ID to backtest")
    args = parser.parse_args()

    main(args.model_run_id)