# OHLC Price Logic Centralization Plan

## Overview
Centralize all OHLCV and timestamp validation logic into a single source-of-truth module to eliminate inconsistencies, reduce duplication, and ensure deterministic behavior across the trading system.

## Current State Analysis
### Where Price Logic Exists Today
- **Validation and preprocessing**
  - `validate_bars()`: Current OHLCV assertions with message precedence issues
  - `preprocess_bars()`: Timestamp normalization, monotonic sort, dedupe
- **Backtest and execution**
  - `get_fill_price()`, `get_fill_price_with_spread()`: Fill price derivations from OHLC modes
  - `run_backtest()`: Entry/exit logic relies on bar[open/high/low/close], ATR fallback to (high-low)
- **Features and signals**
  - `compute_vwap_features()`: Typical price, rolling VWAP; assumes sane OHLC
  - `ta_basic.py`: ATR and returns; depends on high/low range validity
  - `volume_profile.py`: Uses low/high/volume extensively
  - `vpa.py`, `ict.py`: Pattern logic based on open/close and high/low
- **Labels**
  - `triple_barrier_labels()`: First-hit detection with high/low barriers
- **Tests**
  - `test_validation.py`: Explicit expectations for messages and monotonic timestamps
  - `test_triple_barrier_labels.py`: Enforces high ≥ max(open, close) and low ≤ min(open, close) setup

## Canonical OHLCV and Timestamp Rules (Axioms)
### Per Row Axioms
- **Finite and positive values**
  - open, high, low, close: finite and > 0
  - volume: finite and ≥ 0
- **Range consistency**
  - high ≥ low
  - low ≤ open ≤ high
  - low ≤ close ≤ high

### Timestamp Axioms
- tz-aware and normalized to UTC internally
- strictly increasing and unique
  - Default enforcement: per ticker series
  - For unit tests expecting global order: configurable global enforcement

**No auto-correction by default (fail-fast). Optional coercion helper available at ingestion.**

## Error Taxonomy and Deterministic Precedence
### Exception Hierarchy
- `OhlcValidationError` (base)
- `PriceNonPositiveError`, `VolumeNegativeError`
- `HighLessThanLowError`
- `OpenOutsideRangeError`, `CloseOutsideRangeError`
- `HighLessThanOpenError`, `LowGreaterThanCloseError` (compat shims)
- `TimestampNotMonotonicError`, `TimestampTimezoneError`

### Validation Precedence
1. Timestamp issues (presence, tz, monotonic, unique)
2. high < low
3. open/close outside [low, high] (use specific OpenOutsideRangeError/CloseOutsideRangeError)
4. Volume < 0 or price ≤ 0

**Compatibility mode (`msg_compat=True`) maps errors to current test strings in `validate_bars()` to avoid test churn.**

## Single-Source-of-Truth Module Design
### New Module: `ohlc.py`
**API:**
- `validate_ohlcv_frame(df, *, per_ticker=True, tz_policy="require_tz", msg_compat=True)` → None | raises
- `validate_ohlcv_row(row)` → None | raises
- `is_valid_ohlcv_row(row)` → bool (fast guard)
- `coerce_timestamps(df, *, per_ticker=True, to_utc=True)` → DataFrame (normalize tz, sort, dedupe)

**Implementation:**
- Vectorized masks per axiom; report first violation by precedence
- Exceptions include count and first N offending indices for diagnostics
- `msg_compat=True` maps to legacy messages used in `test_validation.py`

## Integration Plan
### 1. Refactor `validate_bars()`
- Replace ad-hoc assertions with a call to `validate_ohlcv_frame(..., msg_compat=True)`

### 2. Preprocessing
- Optionally call `coerce_timestamps` in `preprocess_bars()` (or at ingestion boundary) to ensure UTC tz, sorted, deduped

### 3. Backtest and fills
- At start of `run_backtest()`, add optional one-time frame validation (config flag), assuming upstream validation in pipelines
- In `get_fill_price()`, add DEBUG-guarded `is_valid_ohlcv_row(row)` assertions (no runtime overhead in prod)

### 4. Features and labels
- Rely on prevalidated inputs; add lightweight optional entry assertions in `compute_vwap_features()` and `triple_barrier_labels()` during debug

## Test Strategy
### New Unit Suite: `tests/unit/test_ohlc_rules.py`
- Each axiom: positive/negative cases
- Timestamp policies: tz presence, UTC coercion, monotonic per_ticker vs global
- Precedence order verification
- `msg_compat` mapping to legacy messages

### Existing Tests
- Keep existing tests; `validate_bars()` delegates to the new module with `msg_compat=True` to preserve expected messages
- Full pytest run to confirm no regressions

## Rollout Steps
1. Implement `ohlc.py` with exceptions and API
2. Add `test_ohlc_rules.py`
3. Refactor `validate_bars()` to delegate; run unit tests for validation
4. Integrate optional checks in `preprocess_bars()`, `run_backtest()`, `get_fill_price()`
5. Replace any ad-hoc price checks in features/labels with centralized assertions (only at module boundaries)
6. Update README and inline module docstrings

## Acceptance Criteria
- All existing tests pass without modifying their assertions
- New ohlc validator tests pass
- Single authoritative place for OHLC and timestamp validation rules, with deterministic precedence and clear error taxonomy
- Optional debug-time guards in fills/engine; low overhead in production

## Risks and Mitigations
- **Test coupling to legacy messages**: mitigated via `msg_compat` mapping
- **Performance overhead**: validation concentrated at ingestion/one-time checks; DEBUG-only guards elsewhere
- **Timezone corner cases**: consistent UTC normalization and explicit policies enforced by `coerce_timestamps()`

## Estimation
- Implementation and unit tests: ~4-6 hours
- Refactor and integration + full test run: ~2 hours
- Documentation and cleanup: ~1 hour