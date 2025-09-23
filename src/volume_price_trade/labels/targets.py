"""Triple-barrier labels with 5-minute event sampling and weighted decisioning.

This implementation:
- Samples event start times at calendar 5-minute boundaries by default.
- Sizes barriers using ATR (stop = ATR * atr_mult_sl; take-profit = stop * r_mult_tp).
- Truncates horizons at End-Of-Day when eod_flat is True.
- Evaluates outcomes using only future bars (bar_close + 1 bar), avoiding lookahead.
- Returns only the sampled event rows (downsampled).

Outputs per event:
- y_class in {'up','down','hold'}
- horizon_minutes (actual, after EOD truncation if any)
- event_end_time (timestamp of first barrier hit or timeout)

Assumptions and integrity:
- No lookahead: label for a start time t uses data from (t+1 ... horizon_end].
- Tie-handling when both barriers are touched within the same bar is deterministic via `tie_break`.
- Weights control how ties are resolved (and can be used downstream as sample weights),
  but to avoid leakage into X given the existing dataset pipeline, weights are NOT returned
  as a column; instead they are stored in result.attrs['label_weights'].
"""

import pandas as pd
import numpy as np
from typing import Optional

from ..features.ta_basic import atr
from .trade_horizons import sample_event_index, compute_horizon_end


def triple_barrier_labels(
    df: pd.DataFrame,
    horizons_min: int,
    atr_mult_sl: float,
    r_mult_tp: float,
    eod_flat: bool,
    w_up: float = 1.0,
    w_down: float = 1.0,
    event_freq: str = "5min",
    mode: str = "precise",
    tie_break: str = "tp",
) -> pd.DataFrame:
    """
    Weighted triple-barrier labels with 5-minute event sampling.

    Args:
        df: DataFrame with minute OHLCV and datetime index. Must contain ['open','high','low','close','volume'].
        horizons_min: Maximum horizon in minutes (inclusive range 5..240).
        atr_mult_sl: Stop-loss size in ATRs (stop = ATR * atr_mult_sl).
        r_mult_tp: Take-profit R-multiple relative to stop (tp = stop * r_mult_tp).
        eod_flat: Truncate horizon at next session close when True.
        w_up: Weight associated with a take-profit hit (used for tie-breaking and downstream).
        w_down: Weight associated with a stop-loss hit (used for tie-breaking and downstream).
        event_freq: Sampling interval for event starts; default "5min".
        mode: "precise" (default) uses first-hit detection with EOD truncation; "fast" uses rolling extrema.
        tie_break: When both barriers are touched within the same bar, choose "tp" or "sl" deterministically.

    Notes and guarantees:
    - Signals are evaluated at bar close + 1 bar (current bar excluded).
    - Purging/embargo should use event_end_time to keep temporal integrity.
    - To prevent label leakage into X, this function does not add any label-like numeric columns.
    """
    if horizons_min < 5 or horizons_min > 240:
        raise ValueError("horizons_min must be within [5, 240] minutes")

    # Defensive index normalization
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Select event start timestamps aligned to the requested frequency
    event_idx = sample_event_index(df.index, freq=event_freq)

    if mode == "precise":
        result = _triple_barrier_labels_precise_weighted(
            df=df,
            event_idx=event_idx,
            horizons_min=horizons_min,
            atr_mult_sl=atr_mult_sl,
            r_mult_tp=r_mult_tp,
            eod_flat=eod_flat,
            tie_break=tie_break,
        )
    elif mode == "fast":
        result = _triple_barrier_labels_fast_weighted(
            df=df,
            event_idx=event_idx,
            horizons_min=horizons_min,
            atr_mult_sl=atr_mult_sl,
            r_mult_tp=r_mult_tp,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'precise' or 'fast'.")

    # Store weights in metadata to enable downstream use without leaking into X
    result.attrs["label_weights"] = {"up": float(w_up), "down": float(w_down), "hold": 0.0}
    result.attrs["event_freq"] = event_freq
    result.attrs["horizons_min"] = horizons_min
    return result


def _triple_barrier_labels_precise_weighted(
    df: pd.DataFrame,
    event_idx: pd.DatetimeIndex,
    horizons_min: int,
    atr_mult_sl: float,
    r_mult_tp: float,
    eod_flat: bool,
    tie_break: str,
) -> pd.DataFrame:
    """
    Precise first-hit detection with EOD truncation.
    """
    out_idx = []
    y_class = []
    horizon_minutes = []
    event_end_time = []

    # Barrier sizing uses ATR on full series
    atr_window = 20
    atr_series = atr(df, atr_window)

    # Iterate over events; exclude starts without ATR (warmup)
    for ts in event_idx:
        ts = pd.Timestamp(ts)
        current_atr = atr_series.get(ts, np.nan)
        if np.isnan(current_atr):
            # For single-row DataFrames or early events without ATR, assign 'hold'
            out_idx.append(ts)
            y_class.append("hold")
            horizon_minutes.append(horizons_min)  # Use full horizon since no ATR available
            event_end_time.append(ts + pd.Timedelta(minutes=horizons_min))
            continue

        start_close = float(df.at[ts, "close"])
        stop_pts = float(current_atr) * float(atr_mult_sl)
        tp_pts = stop_pts * float(r_mult_tp)
        upper = start_close + tp_pts
        lower = start_close - stop_pts

        # Compute scan horizon (may truncate at EOD)
        end_time = compute_horizon_end(ts, horizons_min, eod_flat=eod_flat)

        # Timezone harmonization: align end_time to the same tz as df.index (expected ET)
        # to avoid "Both dates must have the same UTC offset" during slicing.
        if df.index.tz is not None:
            _et_tz = df.index.tz  # typically America/New_York
            end_time = pd.Timestamp(end_time)
            if end_time.tzinfo is None:
                end_time = end_time.tz_localize(_et_tz)
            else:
                end_time = end_time.tz_convert(_et_tz)

        # Future path strictly after the event bar (bar_close + 1 bar)
        future = df.loc[ts:end_time]
        if len(future) <= 1:
            # No future bars to evaluate; skip
            continue
        future = future.iloc[1:]

        label = "hold"
        ev_time = end_time

        # First-hit search, tie-broken deterministically within a bar
        for f_ts, row in future.iterrows():
            hi = row["high"]
            lo = row["low"]
            up_hit = hi >= upper
            dn_hit = lo <= lower

            if up_hit and dn_hit:
                label = "down" if tie_break.lower() == "sl" else "up"
                ev_time = pd.Timestamp(f_ts)
                break
            elif up_hit:
                label = "up"
                ev_time = pd.Timestamp(f_ts)
                break
            elif dn_hit:
                label = "down"
                ev_time = pd.Timestamp(f_ts)
                break
            # else continue

        out_idx.append(ts)
        y_class.append(label)
        horizon_minutes.append(int((end_time - ts).total_seconds() // 60))
        event_end_time.append(ev_time)

    # Assemble output DataFrame on the kept event rows
    result = df.loc[out_idx].copy() if out_idx else df.iloc[0:0].copy()
    result["y_class"] = y_class
    result["horizon_minutes"] = horizon_minutes
    # Preserve timezone dtype where applicable
    if result.index.tz is not None:
        result["event_end_time"] = pd.Series(event_end_time, index=result.index, dtype=f"datetime64[ns, {result.index.tz}]")
    else:
        result["event_end_time"] = pd.Series(event_end_time, index=result.index, dtype="datetime64[ns]")
    return result


def _triple_barrier_labels_fast_weighted(
    df: pd.DataFrame,
    event_idx: pd.DatetimeIndex,
    horizons_min: int,
    atr_mult_sl: float,
    r_mult_tp: float,
) -> pd.DataFrame:
    """
    Fast, vectorized approximation:
    - Uses future rolling extrema (excludes current bar via shift(-1)).
    - Approximates event_end_time as start + horizons_min (no EOD truncation).
    - Cannot determine true first-hit ordering when both barriers are touched within the window;
      deterministically prefers 'up' in that rare tie case.

    Use precise mode when exact ordering and EOD truncation are required.
    """
    result_df = df.copy()
    result_df.index = pd.to_datetime(result_df.index)

    atr_window = 20
    atr_series = atr(df, atr_window)

    # Barrier series
    stop_loss_points = atr_series * atr_mult_sl
    take_profit_points = stop_loss_points * r_mult_tp
    upper_barrier = result_df["close"] + take_profit_points
    lower_barrier = result_df["close"] - stop_loss_points

    # Window length in one-minute bars
    window = max(1, int(horizons_min))

    # Future extrema excluding current bar
    future_high_max = result_df["high"][::-1].rolling(window=window, min_periods=1).max()[::-1].shift(-1)
    future_low_min = result_df["low"][::-1].rolling(window=window, min_periods=1).min()[::-1].shift(-1)

    valid_atr = ~atr_series.isna()
    hit_up = (future_high_max >= upper_barrier) & valid_atr
    hit_down = (future_low_min <= lower_barrier) & valid_atr

    # Prepare output only for sampled events
    evt_mask = pd.Series(False, index=result_df.index)
    evt_mask.loc[event_idx] = True

    y_cls = np.where(hit_up, "up", np.where(hit_down, "down", "hold"))

    horizon_delta = pd.to_timedelta(window, unit="m")

    out = result_df.loc[evt_mask].copy()
    out["y_class"] = y_cls[evt_mask]
    out["horizon_minutes"] = np.where(valid_atr[evt_mask], window, 0)
    # event_end_time approximation
    if out.index.tz is not None:
        out["event_end_time"] = out.index + horizon_delta  # type: ignore
    else:
        out["event_end_time"] = (out.index + horizon_delta).to_numpy()  # type: ignore
    return out
