# Volume Price Trade - Project Plan

## Overview
This project implements a minimal, auditable intraday ML backtesting stack for SP500 1m data on GCS, featuring Volume Profile, VPA, ICT features, purged walk-forward CV, and a risk-aware backtester.

## Architecture
The system is organized into the following modules:
- **Data**: GCS Parquet loader, calendar helpers
- **Features**: Volume Profile, VPA, ICT, Time-of-Day, TA basics
- **Labels**: Triple-barrier within 5-240m; EOD truncation
- **ML**: XGBoost/LightGBM baselines; purged walk-forward + ticker holdout
- **Backtest**: Long/short/hold; 2% VaR; 5-240m holding; force flat EOD
- **Reports**: Trade/equity analytics; calibration; OOS summaries

## Milestones

### M1: Data & Sessions
**Objectives**:
- Implement GCS Parquet data loader with proper timezone handling
- Create calendar helpers for trading sessions and early close days
- Ensure EOD flattening (NYSE) is properly handled

**Acceptance Criteria**:
- GCS loader can fetch 1m SP500 data
- Calendar helper identifies trading days and early closes
- Data is properly timezone-aware (America/New_York)

### M2: Features v1 (VP/VPA/ICT/ToD)
**Objectives**:
- Implement Volume Profile features (top-5)
- Implement Volume Price Analysis features (top-5)
- Implement ICT (Inner Circle Trader) features (top-5)
- Implement Time-of-Day features

**Acceptance Criteria**:
- All features generate properly typed pandas DataFrames
- Features are deterministic given same input
- Features pass unit tests with edge cases

### M3: Labels & Dataset
**Objectives**:
- Implement triple-barrier labeling with configurable horizons (5-240m)
- Ensure EOD truncation of labels
- Create dataset builder that combines features and labels

**Acceptance Criteria**:
- Labels are properly generated for all horizons
- Dataset builder creates feature-label pairs with proper alignment
- Labels respect EOD flattening rule

### M4: ML Baselines & CV
**Objectives**:
- Implement XGBoost and LightGBM baseline models
- Create purged walk-forward cross-validation
- Implement ticker holdout validation

**Acceptance Criteria**:
- Models train on feature-label datasets
- CV properly purges overlapping data
- Holdout validation preserves temporal integrity

### M5: Backtester & Risk
**Objectives**:
- Implement backtester with long/short/hold capabilities
- Integrate risk management (2% VaR per trade, max 5 trades/day)
- Ensure position sizing and EOD flattening

**Acceptance Criteria**:
- Backtester executes trades with proper fills
- Risk limits are enforced (VaR, max trades)
- Positions are flattened at EOD

### M6: Reports
**Objectives**:
- Generate trade analytics and equity curves
- Create model calibration reports
- Summarize out-of-sample performance

**Acceptance Criteria**:
- Reports show key performance metrics
- Equity curves are properly plotted
- OOS summary is comprehensive

### M7: Scale-out & Final OOS
**Objectives**:
- Optimize for production deployment
- Final validation on holdout set
- Documentation and examples

**Acceptance Criteria**:
- System runs efficiently on full dataset
- Final OOS performance meets expectations
- Documentation is complete

## Configuration
All configuration is managed through `config/base.yaml` as the single source of truth for:
- Tickers (train and OOS)
- Risk parameters
- Trading sessions
- Feature parameters
- CV parameters

## Quality Assurance
- All code must pass `bash scripts/run_checks.sh` (imports, ruff, mypy, pytest)
- Code is typed where feasible
- Prioritize clarity and determinism
- No secrets in code or logs
- Only fetch external data from GCS paths defined in config