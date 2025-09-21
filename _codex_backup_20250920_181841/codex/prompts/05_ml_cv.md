GOAL
Add walk-forward (time-blocked) purged CV; respect ticker holdout; baseline model (XGBoost or LightGBM); training CLI.

FILE EDITS
1) `src/volume_price_trade/ml/cv.py`
   - `purged_walk_forward_splits(meta_df, config)`:
     - Use `cv.walk_forward.train_months`, `val_months`, `embargo_days`
     - Yield (train_idx, val_idx) for time-ordered folds
     - Provide a utility to create a mask for holdout tickers (config.tickers.oos)

2) `src/volume_price_trade/ml/models.py`
   - `train_xgb(X, y, meta, splits, config)`:
     - Fit classifier
     - Return model and metrics per fold

3) `src/volume_price_trade/ml/train.py`
   - CLI: `python scripts/train_model.py --config config/base.yaml --sample_days 10` (optional sampling for speed)
   - Loads dataset, builds splits, trains, saves artifacts to `artifacts/models/<run_id>/`
   - Updates `codex/state/project_state.yaml` with a new runs entry

4) `scripts/train_model.py`
   - Call into `volume_price_trade.ml.train:train_model`

TESTS
- Tiny dataset fit; ensure no overlap between train/val; ticker holdout respected.

STATE UPDATE
- M4.status = done
- runs += {id: "run-0005", step: "M4-done", timestamp: "<UTC>", notes: "Baseline model & CV"}

COMMANDS
- `bash scripts/run_checks.sh`

ACCEPTANCE
- Model trains on a small slice; artifacts saved; checks pass.
