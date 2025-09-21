GOAL
Implement GCS Parquet loader, NYSE calendar helpers, and basic schema validation. Wire a small smoke path in the CLI.

IMPLEMENTATION NOTES
- Use gcsfs + pandas to read Parquet.
- Conform to config schema names from `config/base.yaml`.
- Ensure timestamps are tz-aware UTC in DataFrame; add `ticker` column.
- Calendar: `is_rth(ts)` for NY RTH 09:30–16:00 (US/Eastern). `next_session_close(ts)` returns the same-day close time (ignore early-closes for now; leave TODO).

FILE EDITS
1) `src/volume_price_trade/data/gcs_loader.py`
   - Implement:
     ```python
     from __future__ import annotations
     from typing import Iterable
     import pandas as pd
     import pyarrow.parquet as pq
     import gcsfs, pandas as pd
     from datetime import datetime
     import pytz

     def list_available_months(ticker: str) -> list[str]: ...
     def load_minute_bars(ticker: str, start: str, end: str) -> pd.DataFrame: ...
     ```
   - Logic: construct GCS paths `gs://{bucket}/{root}/{TICKER}/{YYYY}/{TICKER_YYYY-MM}.parquet` for months overlapping [start, end]; read and concat; enforce columns, UTC tz; add `ticker`.

2) `src/volume_price_trade/data/calendar.py`
   - Implement:
     ```python
     import pandas as pd
     import pytz
     from datetime import time

     ET = pytz.timezone("America/New_York")
     RTH_START = time(9, 30)
     RTH_END = time(16, 0)

     def is_rth(ts: pd.Timestamp) -> bool: ...
     def next_session_close(ts: pd.Timestamp) -> pd.Timestamp: ...
     ```

3) `src/volume_price_trade/utils/validation.py`
   - Implement:
     ```python
     import pandas as pd

     def validate_bars(df: pd.DataFrame, cfg: dict) -> None:
         # assert columns exist; monotonic timestamp; nonnegative volume; high>=low
         ...
     ```

4) `scripts/build_dataset.py`
   - Add a `--ticker`, `--start`, `--end` CLI that:
     - loads bars via loader
     - prints basic counts and date range
     - exits (no ML yet)

TESTS
- Add unit tests for `is_rth` edge times.
- Add validation test on synthetic bars (monotonic ts, col names).

STATE UPDATE
- Set M1.status = done
- runs +=
  - id: "run-0002"
    step: "M1-done"
    timestamp: "<UTC>"
    notes: "Data loader/calendar/validation implemented"

COMMANDS
- `bash scripts/run_checks.sh`

ACCEPTANCE
- Checks pass; CLI can load and print dataset stats (if data is present in GCS).
