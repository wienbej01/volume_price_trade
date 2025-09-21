# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand and assist with the `volume_price_trade` project.

## Project Overview

`volume_price_trade` is a Python-based machine learning backtesting framework for intraday trading strategies. The project focuses on using Volume Profile, Volume Price Analysis (VPA), and Inner Circle Trader (ICT) concepts on 1-minute S&P 500 stock data.

The project is structured as a proper Python package with a `src` layout. It uses a configuration-driven approach, with `config/base.yaml` as the single source of truth for all parameters.

### Key Technologies

*   **Programming Language:** Python 3.10
*   **Core Libraries:** pandas, scikit-learn, XGBoost, LightGBM
*   **Data:** Parquet files stored in Google Cloud Storage (GCS)
*   **Testing & Linting:** pytest, mypy, ruff

### Architecture

The project is organized into the following modules:

*   `data`: Handles data loading from GCS and time-zone conversions.
*   `features`: Implements various feature engineering techniques, including Volume Profile, VPA, ICT, and time-of-day features.
*   `labels`: Creates triple-barrier labels for training.
*   `ml`: Contains the machine learning pipeline, including model training, cross-validation, and prediction.
*   `backtest`: Implements the backtesting engine, including risk management, fill simulation, and trade execution.
*   `reports`: Generates reports with performance metrics and visualizations.

## Building and Running

### Setup

1.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv && source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -U pip
    pip install -e ".[dev]"
    ```

### Running Checks

To ensure code quality, run the following script:

```bash
bash scripts/run_checks.sh
```
This will run import checks, linting with ruff, static type checking with mypy, and unit tests with pytest.

### Running a Backtest

To run a backtest with a trained model, use the following command:

```bash
python scripts/backtest_model.py --model artifacts/models/run-bbdead19/model.joblib
```

## Development Conventions

*   **Configuration:** All project parameters are managed in `config/base.yaml`.
*   **Coding Style:** The project uses `ruff` for linting and `mypy` for static type checking. Please adhere to the configurations in `pyproject.toml`.
*   **Testing:** All new code should be accompanied by unit tests in the `tests` directory.
*   **Commits:** Follow conventional commit standards.
*   **No Secrets:** Do not commit any secrets or credentials to the repository.
