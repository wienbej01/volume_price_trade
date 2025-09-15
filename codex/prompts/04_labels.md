GOAL
Implement triple-barrier labels within 5–240m with EOD truncation, and dataset builder with purged/embargoed splits.

FILE EDITS
1) `src/volume_price_trade/labels/targets.py`
   - `triple_barrier_labels(df, horizons_min, atr_mult_sl, r_mult_tp, eod_flat)`
   - For each bar, compute forward path up to min(240m, EOD):
     - Stop = ATR * atr_mult_sl; Target = Stop * take_profit_R
     - Determine which (TP/SL/timeout) occurs first; set `y_class` in {up, down, hold}
     - Also output `horizon_minutes` actually used and `event_end_time`

2) `src/volume_price_trade/ml/dataset.py`
   - `make_dataset(tickers, start, end, config)`:
     - Load bars per ticker → features (feature_union) → labels (triple-barrier)
     - Add meta: `ticker`, `session_date`
     - Purge overlapping events; apply embargo days from config
     - Return X, y, meta

TESTS
- Synthetic series tests:
  - SL hits before TP, vice versa, and timeout behavior
  - EOD truncation verified
  - Purging removes overlaps; embargo enforced

STATE UPDATE
- M3.status = done
- runs += {id: "run-0004", step: "M3-done", timestamp: "<UTC>", notes: "Labels & dataset built"}

COMMANDS
- `bash scripts/run_checks.sh`

ACCEPTANCE
- Deterministic labels; tests cover SL/TP/timeout; checks pass.
