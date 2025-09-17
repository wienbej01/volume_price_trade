#!/usr/bin/env python
"""Training script for volume price trade models."""

import argparse
import sys
from pathlib import Path

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volume_price_trade.ml.train import train_model


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train a volume price trade model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Path to configuration YAML file (default: config/base.yaml)"
    )
    parser.add_argument(
        "--sample_days",
        type=int,
        default=None,
        help="Number of days to sample for faster testing (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    try:
        # Train the model
        result = train_model(
            config_path=str(config_path),
            sample_days=args.sample_days
        )
        
        # Print summary
        print(f"\nTraining completed successfully!")
        print(f"Run ID: {result['run_id']}")
        print(f"Samples: {result['n_samples']}")
        print(f"Features: {result['n_features']}")
        print(f"CV Folds: {result['n_folds']}")
        print(f"Average Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"Artifacts saved to: artifacts/models/{result['run_id']}/")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
