# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure for volume_price_trade
- Configuration schema in config/base.yaml
- Project plan with 7 milestones (M1-M7)
- Persona and policies documentation
- Basic module structure for data, features, labels, ML, backtest, and reports
- Development tooling (ruff, mypy, pytest)
- Smoke import test for core dependencies
- Complete M1: Data & Sessions implementation with GCS Parquet loader and calendar helpers
- Complete M2: Features implementation with 68 features across 6 modules (TA, VP, VPA, ICT, ToD, VWAP)
- Complete M3: Labels & Dataset with triple-barrier labeling and EOD truncation
- Complete M4: ML Baselines & CV with XGBoost/LightGBM models and purged walk-forward validation
- Complete M5: Backtester & Risk with position management, risk controls, and fill simulation
- Partial M6: Reports implementation with HTML/Markdown reporting and metrics calculation
- Report bundling script for organizing outputs (scripts/make_report.py)
- Comprehensive trade analytics including equity curves, drawdowns, and performance metrics
- Time-based analysis (time of day, monthly statistics)
- Ticker-specific performance breakdowns

### Changed
- Updated project structure to include comprehensive backtesting module
- Enhanced feature set with ICT, Volume Profile, and VPA implementations
- Improved risk management with position sizing based on VaR

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Fixed ICT shape doubling bug
- Fixed volume profile NaN handling issues

### Security
- Nothing yet