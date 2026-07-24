"""Map sparse observed bars onto frozen exchange-session sample rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from tools.session_calendar import SessionCalendar, expected_bins


@dataclass(frozen=True, slots=True)
class SampleRows:
    as_of: int
    entry: int
    target: int
    as_of_ordinal: int


@dataclass(frozen=True, slots=True)
class SessionSamples:
    rows: tuple[SampleRows, ...]
    opportunities: int


def session_samples(
    timestamps: Sequence[str],
    minutes: int,
    calendar: SessionCalendar,
    start: date,
    end: date,
    history_bars: int,
    horizon_bars: int,
    alignment_horizon_bars: int,
) -> SessionSamples:
    """Return rows whose history, entry, and fixed-horizon target exist."""
    values = (history_bars, horizon_bars, alignment_horizon_bars)
    if any(type(value) is not int or value < 1 for value in values) or \
       horizon_bars > alignment_horizon_bars:
        raise ValueError("invalid sample history or horizon")
    grid = tuple(expected_bins(calendar, start, end, minutes))
    ordinal_by_time = {
        item.timestamp: ordinal for ordinal, item in enumerate(grid)
    }
    observed, previous = [-1] * len(grid), -1
    for row, timestamp in enumerate(timestamps):
        ordinal = ordinal_by_time.get(timestamp, -1) \
            if isinstance(timestamp, str) else -1
        if ordinal <= previous:
            raise ValueError(
                "observed timestamps must be ordered expected session bins",
            )
        observed[ordinal] = row
        previous = ordinal

    first_target = history_bars + alignment_horizon_bars - 1
    rows, streak = [], 0
    for as_of_ordinal, row in enumerate(observed):
        streak = streak + 1 if row >= 0 else 0
        target_ordinal = as_of_ordinal + horizon_bars
        if first_target <= target_ordinal < len(grid) and \
           streak >= history_bars and \
           observed[as_of_ordinal + 1] >= 0 and \
           observed[target_ordinal] >= 0:
            rows.append(SampleRows(
                observed[as_of_ordinal], observed[as_of_ordinal + 1],
                observed[target_ordinal], as_of_ordinal,
            ))
    return SessionSamples(
        tuple(rows), max(0, len(grid) - first_target),
    )
