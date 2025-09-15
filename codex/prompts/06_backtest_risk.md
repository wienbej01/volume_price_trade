GOAL
Turn classifier outputs into trades with risk, slippage, and constraints. Produce basic metrics.

FILE EDITS
1) `src/volume_price_trade/backtest/risk.py`
   - `shares_for_risk(equity, entry, stop, risk_perc)` → integer shares

2) `src/volume_price_trade/backtest/fills.py`
   - `get_fill_price(side, bar, mode="next_open")` with optional slippage from config

3) `src/volume_price_trade/backtest/engine.py`
   - `run_backtest(signals_df, bars_df, config)`:
     - One position per ticker
     - Entry: next-bar open after signal
     - Exit: SL/TP or horizon timeout (5–240m); force flat at EOD (`eod_flat_minutes_before_close`)
     - Max trades per day = config.risk.max_trades_per_day
     - Track equity curve; commission and slippage applied

4) `src/volume_price_trade/backtest/metrics.py`
   - Trade stats: win%, PF, expectancy, avg R, avg hold, drawdowns
   - Per-ticker and time-of-day breakdowns

5) `src/volume_price_trade/backtest/reports.py`
   - Save Markdown/HTML with equity curve and tables under `artifacts/reports/<run_id>/`

6) `scripts/backtest_model.py`
   - Glue: load model, generate signals (simple threshold), run engine, save report path

TESTS
- Integration test on synthetic bars and signals; validate EOD flattening and trade caps.

STATE UPDATE
- M5.status = done
- runs += {id: "run-0006", step: "M5-done", timestamp: "<UTC>", notes: "Backtester & metrics"}

COMMANDS
- `bash scripts/run_checks.sh`

ACCEPTANCE
- Backtest produces sensible stats and a saved report dir; checks pass.
