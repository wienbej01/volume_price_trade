#!/usr/bin/env python
"""Test script to verify the training CLI works correctly."""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from volume_price_trade.ml.train import train_model


def create_mock_data():
    """Create mock market data for testing."""
    # Generate timestamp range
    start_date = datetime.now() - timedelta(days=10)
    end_date = datetime.now() - timedelta(days=1)
    
    # Create minute timestamps
    timestamps = pd.date_range(
        start=start_date,
        end=end_date,
        freq='1min',
        tz='America/New_York'
    )
    
    # Filter to regular trading hours (9:30 AM - 4:00 PM)
    timestamps = [ts for ts in timestamps 
                  if ts.time() >= pd.Timestamp('09:30').time() and 
                     ts.time() <= pd.Timestamp('16:00').time()]
    
    # Create mock OHLCV data
    n_samples = len(timestamps)
    base_price = 100.0
    
    data = {
        'timestamp': timestamps,
        'open': base_price + np.random.normal(0, 0.1, n_samples),
        'high': base_price + np.random.normal(0.2, 0.1, n_samples),
        'low': base_price + np.random.normal(-0.2, 0.1, n_samples),
        'close': base_price + np.random.normal(0, 0.1, n_samples),
        'volume': np.random.randint(1000, 10000, n_samples),
        'ticker': 'TEST'
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    
    return df


def test_training_cli():
    """Test the training CLI with mock data."""
    print("Testing training CLI...")
    
    # Create temporary directory for artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock config
        config = {
            'project': 'test',
            'data': {
                'gcs': {
                    'bucket': 'test',
                    'root': 'test'
                },
                'schema': {
                    'timestamp': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'ticker': 'ticker'
                }
            },
            'sessions': {
                'tz': 'America/New_York',
                'rth_start': '09:30',
                'rth_end': '16:00',
                'eod_flat_minutes_before_close': 1
            },
            'risk': {
                'start_equity': 10000,
                'risk_perc': 0.02,
                'max_trades_per_day': 5,
                'atr_stop_mult': 1.5,
                'take_profit_R': 2.0,
                'slippage_cents': 1,
                'commission_per_share': 0.005
            },
            'features': {
                'atr_window': 20,
                'rvol_windows': [5, 20],
                'volume_profile': {
                    'bin_size': 0.05,
                    'value_area': 0.70,
                    'rolling_sessions': 20
                },
                'ict': {
                    'fvg_min_size_atr': 0.25,
                    'displacement_body_atr': 1.2
                },
                'vpa': {
                    'rvol_climax': 2.5,
                    'vdu_threshold': 0.4,
                    'atr_threshold': 0.5,
                    'breakout_lookback': 20
                }
            },
            'time_of_day': {
                'killzones': {
                    'ny_open': ['09:30', '11:30'],
                    'lunch': ['12:00', '13:30'],
                    'pm_drive': ['13:30', '16:00']
                }
            },
            'horizons_minutes': [60],
            'cv': {
                'walk_forward': {
                    'train_months': 1,
                    'val_months': 1,
                    'embargo_days': 1
                },
                'ticker_holdout': 0
            },
            'tickers': {
                'train': ['TEST'],
                'oos': []
            },
            'model': {
                'type': 'xgboost',
                'params': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                }
            }
        }
        
        # Save config to temporary file
        config_path = os.path.join(temp_dir, 'test_config.yaml')
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Mock the data loading function
        from volume_price_trade.ml import dataset
        
        original_load_bars = dataset._load_bars_for_ticker
        
        def mock_load_bars(ticker, start_date, end_date):
            return create_mock_data()
        
        dataset._load_bars_for_ticker = mock_load_bars
        
        try:
            # Test training with sample_days
            result = train_model(
                config_path=config_path,
                sample_days=5
            )
            
            print("✓ Training completed successfully!")
            print(f"Run ID: {result['run_id']}")
            print(f"Samples: {result['n_samples']}")
            print(f"Features: {result['n_features']}")
            print(f"CV Folds: {result['n_folds']}")
            print(f"Average Accuracy: {result['metrics']['accuracy']:.4f}")
            
            # Check if artifacts were created
            artifacts_dir = Path(f"artifacts/models/{result['run_id']}")
            if artifacts_dir.exists():
                print("✓ Artifacts directory created")
                
                expected_files = ['model.joblib', 'metrics.json', 'config.yaml', 
                                'feature_names.json', 'run_metadata.json']
                
                for file in expected_files:
                    if (artifacts_dir / file).exists():
                        print(f"✓ {file} created")
                    else:
                        print(f"✗ {file} missing")
            else:
                print("✗ Artifacts directory not created")
            
            # Check if project state was updated
            project_state_path = Path("codex/state/project_state.yaml")
            if project_state_path.exists():
                print("✓ Project state file exists")
                
                with open(project_state_path, 'r') as f:
                    project_state = yaml.safe_load(f)
                
                # Check if run was added
                run_ids = [run['id'] for run in project_state.get('runs', [])]
                if result['run_id'] in run_ids:
                    print("✓ Run added to project state")
                else:
                    print("✗ Run not found in project state")
            else:
                print("✗ Project state file not found")
            
            return True
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Restore original function
            dataset._load_bars_for_ticker = original_load_bars


if __name__ == "__main__":
    success = test_training_cli()
    if success:
        print("\n✓ All tests passed! Training CLI is working correctly.")
        sys.exit(0)
    else:
        print("\n✗ Tests failed!")
        sys.exit(1)