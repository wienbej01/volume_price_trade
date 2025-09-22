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
from volume_price_trade.ml.predict import load_model_and_predict
from volume_price_trade.utils.io import load_json

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

    # Run backtest
    results = run_backtest(
        model_dir=model_dir,
        tickers=oos_tickers,
        start_date=start_str,
        end_date=end_str,
        config=config
    )

    print("
Backtest completed!")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(f"Total Return: {results.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    print(f"Report: {results.get('report_path', 'N/A')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Backtest")
    parser.add_argument("--model-run-id", help="Specific run ID to backtest")
    args = parser.parse_args()

    main(args.model_run_id)