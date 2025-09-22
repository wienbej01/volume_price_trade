#!/usr/bin/env python3
"""
Phase 1 Training: Baseline model on 5 tickers, 2 years data.

No look-forward bias: uses fixed historical tickers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volume_price_trade.ml.train import train_model

def main():
    config_path = "config/base.yaml"  # Has 5 tickers, 2 years default

    print("Phase 1 Training: 5 tickers, 2 years")
    print("Tickers: AAPL, MSFT, AMZN, GOOGL, TSLA")
    print("Period: ~2 years (default)")

    result = train_model(config_path=config_path, sample_days=None)

    print("\nTraining completed!")
    print(f"Run ID: {result['run_id']}")
    print(f"Samples: {result['n_samples']}")
    print(f"Features: {result['n_features']}")
    print(f"Average Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Artifacts: artifacts/models/{result['run_id']}/")

    return result

if __name__ == "__main__":
    main()