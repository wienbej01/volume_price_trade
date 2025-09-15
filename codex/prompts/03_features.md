GOAL
Implement first-pass features and a feature union ensuring no lookahead.

FILE EDITS
1) `src/volume_price_trade/features/ta_basic.py`
   - Implement ATR(20), RVOL(5,20), returns, true range, rolling stats.

2) `src/volume_price_trade/features/volume_profile.py`
   - Rolling VP over `rolling_sessions` RTH days (config.features.volume_profile).
   - For each bar: `poc`, `vah`, `val`, `dist_to_poc_atr`, `inside_value`, `hvn_near`, `lvn_near`, `poc_shift_dir`.
   - Use a simplified binning (e.g., $0.05) and spread volume across high–low range per 1m bar.

3) `src/volume_price_trade/features/vpa.py`
   - Binary flags: `vpa_climax_up/down`, `vpa_vdu`, `vpa_churn`, `vpa_effort_no_result`, `vpa_breakout_conf`.
   - Normalize thresholds by ATR & RVOL from config.

4) `src/volume_price_trade/features/ict.py`
   - `ict_fvg_up/down` (3-bar FVG), `ict_liquidity_sweep_up/down` (wick breaks prior swing then close back inside),
     `ict_displacement_up/down` (body > k*ATR), `dist_to_eq` (swing 50%).
   - Killzone flags derived from config.time_of_day.

5) `src/volume_price_trade/features/time_of_day.py`
   - minute-of-day sin/cos encodings; flags for open/lunch/PM drive.

6) `src/volume_price_trade/features/feature_union.py`
   - `build_feature_matrix(df, cfg)`:
     - call TA, VP, VPA, ICT, ToD
     - align on timestamp, left-join without future leakage (only use info <= t)
     - return feature DataFrame with stable column order

TESTS
- Unit tests with toy data for each detector (deterministic).
- Ensure shapes/indices align.

STATE UPDATE
- M2.status = done
- runs += {id: "run-0003", step: "M2-done", timestamp: "<UTC>", notes: "Features v1"}

COMMANDS
- `bash scripts/run_checks.sh`

ACCEPTANCE
- All tests pass; ruff/mypy clean; feature frames align without NaN explosions beyond expected warm-ups.
