# Scaling Tracker: Volume Price Trade Model

Tracking training and backtesting performance across scaling phases.

## Scaling Plan

### Phase 1: Baseline (5 tickers, 2 years)
- **Tickers:** AAPL, MSFT, AMZN, GOOGL, TSLA
- **Period:** ~2 years historical
- **Goal:** Establish baseline, validate pipeline
- **Hardware:** 8 cores, 32GB RAM

### Phase 2: Time Scaling (5 tickers, 3-4 years)
- **Tickers:** Same 5
- **Period:** 3-4 years
- **Goal:** Test temporal robustness

### Phase 3: Ticker Scaling (20-50 tickers, 3 years)
- **Tickers:** Diverse set from universe
- **Period:** 3 years
- **Goal:** Add microstructure diversity

### Phase 4: Full Scale (100-150 tickers, 4 years)
- **Tickers:** Full train universe
- **Period:** 4 years
- **Goal:** Maximum sample efficiency

### Phase 5: Optimization
- **Ensemble:** Multiple models if needed
- **Backtest Universe:** 50-100 tickers ($10-20 range)

## Performance Tracking

### Training Metrics (CV Average)

| Phase | Run ID | Samples | Features | Accuracy | AUC | Log Loss | Notes |
|-------|--------|---------|----------|----------|-----|----------|-------|
| 1     |        |         |          |          |     |          |       |
| 2     |        |         |          |          |     |          |       |
| 3     |        |         |          |          |     |          |       |
| 4     |        |         |          |          |     |          |       |
| 5     |        |         |          |          |     |          |       |

### Backtesting Metrics

| Phase | Run ID | Trades | Total Return | Sharpe | Max DD | Win Rate | Notes |
|-------|--------|--------|--------------|--------|--------|----------|-------|
| 1     |        |        |              |        |        |          |       |
| 2     |        |        |              |        |        |          |       |
| 3     |        |        |              |        |        |          |       |
| 4     |        |        |              |        |        |          |       |
| 5     |        |        |              |        |        |          |       |

## Commands

### Training
```bash
# Phase 1
python scripts/phase1_train.py

# General
python scripts/train_model.py --config config/base.yaml --sample_days 365
```

### Backtesting
```bash
# Phase 1
python scripts/phase1_backtest.py

# General
python scripts/backtest_model.py --model-dir artifacts/models/run-XXXX --start-date 2024-01-01 --end-date 2024-06-30
```

## Data Sources

- **Local:** data/*.parquet (178 files)
- **GCS:** jwss_data_store/stocks/ (fallback)
- **Polygon:** API (missing data)

## Hardware Monitoring

- **Memory:** Monitor during training (>28GB = issue)
- **Time:** Training should complete in <4 hours
- **CPU:** 8 cores utilized

## Decision Rules

- **Advance Phase:** If OOS accuracy >52%, Sharpe >1.0
- **Stop Scaling:** If memory >32GB or overfitting detected
- **Optimize:** If metrics degrade, reduce features/estimators

## Logs

- Training: logs/run-{run_id}.log
- Artifacts: artifacts/models/{run_id}/
- Reports: artifacts/reports/{run_id}/

Update this file after each run with metrics from console output and artifact files.