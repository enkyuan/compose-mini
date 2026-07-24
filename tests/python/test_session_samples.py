#!/usr/bin/env python3
"""Verify frozen exchange grids and continuity-safe sample rows."""

from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.session_calendar import (
    DEFAULT_CALENDAR, SessionCalendar, expected_bins,
)
from tools.session_samples import SampleRows, SessionSamples, session_samples

START = date(2024, 11, 1)
END = date(2024, 11, 5)


def raises(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def grid() -> tuple[str, ...]:
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    return tuple(item.timestamp for item in expected_bins(
        calendar, START, END, 30,
    ))


def samples(
    timestamps: tuple[str, ...], history: int = 17,
    horizon: int = 13, alignment: int = 13,
) -> SessionSamples:
    return session_samples(
        timestamps, 30, SessionCalendar.read(DEFAULT_CALENDAR), START, END,
        history, horizon, alignment,
    )


def test_expected_bins() -> None:
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    bins = tuple(expected_bins(calendar, START, date(2024, 11, 4), 30))
    assert len(bins) == 26
    assert (bins[0].session, bins[0].minute, bins[0].timestamp) == (
        START, 570, "2024-11-01T13:30:00Z",
    )
    assert bins[13].timestamp == "2024-11-04T14:30:00Z"
    early = tuple(expected_bins(
        calendar, date(2024, 11, 29), date(2024, 11, 29), 30,
    ))
    assert len(early) == 7 and early[-1].timestamp == "2024-11-29T17:30:00Z"
    remainder = tuple(expected_bins(
        calendar, date(2024, 11, 29), date(2024, 11, 29), 59,
    ))
    assert remainder and remainder[-1].minute + 59 <= 780
    raises(expected_bins, calendar, date(2024, 7, 21), START, 30)
    raises(expected_bins, calendar, START, END, True)


def test_complete_samples() -> None:
    result = samples(grid())
    assert result.opportunities == 10 and len(result.rows) == 10
    assert result.rows[0] == SampleRows(16, 17, 29, 16)
    assert result.rows[-1] == SampleRows(25, 26, 38, 25)
    stationary = samples(grid(), 18)
    assert stationary.opportunities == 9
    assert stationary.rows[0] == SampleRows(17, 18, 30, 17)
    assert stationary.rows[0].as_of - 18 + 1 == 0
    shorter = samples(grid(), horizon=1)
    assert shorter.opportunities == 10
    assert shorter.rows[0] == SampleRows(28, 29, 29, 28)


def test_missing_rows() -> None:
    timestamps = grid()

    def without(index: int) -> SessionSamples:
        return samples(timestamps[:index] + timestamps[index + 1:])

    missing_history = without(5)
    assert missing_history.opportunities == 10
    assert all(row.as_of_ordinal != 16 for row in missing_history.rows)
    assert next(
        row for row in missing_history.rows if row.as_of_ordinal == 22
    ) == SampleRows(21, 22, 34, 22)
    assert all(row.as_of_ordinal != 16 for row in without(17).rows)
    assert all(row.as_of_ordinal != 16 for row in without(29).rows)
    holding_gap = without(20)
    assert next(
        row for row in holding_gap.rows if row.as_of_ordinal == 16
    ) == SampleRows(16, 17, 28, 16)


def test_early_close_boundary() -> None:
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    start, end = date(2024, 11, 27), date(2024, 12, 2)
    timestamps = tuple(
        item.timestamp for item in expected_bins(calendar, start, end, 30)
    )
    result = session_samples(
        timestamps, 30, calendar, start, end, 17, 13, 13,
    )
    assert len(timestamps) == 33
    assert result.opportunities == len(result.rows) == 4
    assert result.rows[0] == SampleRows(16, 17, 29, 16)


def test_invalid_observations() -> None:
    timestamps = grid()
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    raises(samples, (timestamps[0], timestamps[0], *timestamps[1:]))
    raises(samples, tuple(reversed(timestamps)))
    raises(samples, (timestamps[0].replace("Z", "+00:00"), *timestamps[1:]))
    raises(samples, ("2024-10-31T13:30:00Z", *timestamps))
    for history, horizon, alignment in (
        (0, 13, 13), (17, 14, 13), (17, True, 13),
    ):
        raises(
            session_samples, timestamps, 30, calendar, START, END,
            history, horizon, alignment,
        )
    raises(
        session_samples, [[]], 30, calendar, START, END, 17, 13, 13,
    )


def main() -> None:
    test_expected_bins()
    test_complete_samples()
    test_missing_rows()
    test_early_close_boundary()
    test_invalid_observations()
    print("session grid and sample tests passed")


if __name__ == "__main__":
    main()
