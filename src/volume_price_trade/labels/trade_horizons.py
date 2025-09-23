"""Encodes allowed holding windows, 5-minute event sampling, and EOD truncation helpers."""

from typing import Optional
import pandas as pd
from ..data.calendar import next_session_close


def sample_event_index(index: pd.DatetimeIndex, freq: str = "5min") -> pd.DatetimeIndex:
    """
    Select event start timestamps aligned to a calendar frequency boundary.

    Rules:
    - If freq in {"1min", "1T", "", None}: return the original index (no downsampling).
    - Otherwise, select timestamps that are exactly equal to their floor(freq),
      i.e., index == index.floor(freq). This is leakage-free and session-agnostic.

    Args:
        index: Minute-level DatetimeIndex (tz-aware or naive).
        freq: Pandas offset alias like "5min" ("5T").

    Returns:
        Subset of index containing only timestamps aligned to the given freq.
    """
    idx = pd.DatetimeIndex(index)
    if freq in (None, "", "1min", "1T"):
        return idx
    floored = idx.floor(freq)
    mask = idx == floored
    return idx[mask]


def compute_horizon_end(start: pd.Timestamp, horizon_minutes: int, eod_flat: bool = True) -> pd.Timestamp:
    """
    Compute the end timestamp for an event horizon, optionally truncated to EOD.

    Args:
        start: Event start time (bar close at t, label evaluated from t+1 bar).
        horizon_minutes: Max horizon length in minutes.
        eod_flat: When True, limit to next regular-trading-hours session close.

    Returns:
        The timestamp of the end of the horizon or the session close, whichever comes first.
    """
    horizon = pd.Timedelta(minutes=int(horizon_minutes))
    if eod_flat:
        eod = next_session_close(start)
        return min(start + horizon, eod)
    return start + horizon


def enumerate_horizons(ts, max_minutes=240): ...
