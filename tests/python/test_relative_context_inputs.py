#!/usr/bin/env python3
"""Verify stock rows pair only with the exact same causal SPY cells."""

from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.context_diagnostic_inputs import timestamp_rows
from tools.relative_context_inputs import align_spy_rows
from tools.session_calendar import (
    DEFAULT_CALENDAR, SessionCalendar, expected_bins,
)
from tools.session_samples import SampleRows, SessionSamples, session_samples
from tools.universe_contract import PackedRows
from tools.universe_scaling_contract import timestamp_grid_sha256


TIMESTAMPS = tuple(f"2026-01-02T{hour:02d}:00:00Z" for hour in range(8))
STOCK_ROWS = (
    SampleRows(1, 2, 4, 101),
    SampleRows(2, 3, 5, 102),
)
STOCK = PackedRows(STOCK_ROWS, (1, 1))
SPY = SessionSamples(STOCK_ROWS, 8)


def rejects(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("invalid relative-context input was accepted")


def verify_complete_grid() -> None:
    aligned = align_spy_rows(TIMESTAMPS, STOCK, TIMESTAMPS, SPY)
    assert aligned == STOCK
    offset = 0
    for count in STOCK.counts:
        stop = offset + count
        stock_grid = timestamp_rows(
            TIMESTAMPS, STOCK.rows[offset:stop],
        )
        spy_grid = timestamp_rows(
            TIMESTAMPS, aligned.rows[offset:stop],
        )
        assert stock_grid == spy_grid
        assert timestamp_grid_sha256(stock_grid) == \
            timestamp_grid_sha256(spy_grid)
        offset = stop

    training_only = PackedRows((STOCK_ROWS[0],), (1, 0))
    aligned = align_spy_rows(
        TIMESTAMPS, training_only, TIMESTAMPS,
        SessionSamples((SPY.rows[0],), 1),
    )
    assert aligned == training_only


def verify_sparse_stock_grid() -> None:
    stock_timestamps = TIMESTAMPS[1:]
    stock = PackedRows((
        SampleRows(0, 1, 3, 101),
        SampleRows(2, 3, 5, 103),
    ), (1, 1))
    spy = SessionSamples((
        SampleRows(1, 2, 4, 101),
        SampleRows(2, 3, 5, 102),
        SampleRows(3, 4, 6, 103),
    ), 8)
    aligned = align_spy_rows(
        stock_timestamps, stock, TIMESTAMPS, spy,
    )

    assert aligned.counts == stock.counts
    assert aligned.rows != stock.rows
    assert tuple(row.as_of_ordinal for row in aligned.rows) == (101, 103)
    assert timestamp_rows(stock_timestamps, stock.rows) == \
        timestamp_rows(TIMESTAMPS, aligned.rows)


def verify_alignment_guards() -> None:
    missing = SessionSamples((SPY.rows[0],), 8)
    duplicate = SessionSamples((
        SPY.rows[0],
        SampleRows(2, 3, 5, SPY.rows[0].as_of_ordinal),
    ), 8)
    shifted = SessionSamples((
        SampleRows(1, 2, 5, 101),
        SPY.rows[1],
    ), 8)
    extra = TIMESTAMPS[:2] + ("2026-01-02T01:30:00Z",) + TIMESTAMPS[2:]
    nonadjacent = SessionSamples((
        SampleRows(1, 3, 5, 101),
        SampleRows(3, 4, 6, 102),
    ), 9)

    for value in (missing, duplicate, shifted):
        rejects(align_spy_rows, TIMESTAMPS, STOCK, TIMESTAMPS, value)
    rejects(align_spy_rows, TIMESTAMPS, STOCK, extra, nonadjacent)

    invalid_timestamps = (
        object(),
        "abcdefgh",
        list(TIMESTAMPS),
        (),
        (*TIMESTAMPS[:-1], ""),
        (TIMESTAMPS[1], TIMESTAMPS[0], *TIMESTAMPS[2:]),
    )
    for value in invalid_timestamps:
        rejects(align_spy_rows, value, STOCK, TIMESTAMPS, SPY)
        rejects(align_spy_rows, TIMESTAMPS, STOCK, value, SPY)

    for counts in ((), (0, 2), (1, -1), (1, True), (1, 0, 1), (1, 0)):
        rejects(
            align_spy_rows, TIMESTAMPS,
            PackedRows(STOCK_ROWS, counts), TIMESTAMPS, SPY,
        )
    rejects(
        align_spy_rows, TIMESTAMPS,
        PackedRows(list(STOCK_ROWS), (1, 1)), TIMESTAMPS, SPY,
    )
    rejects(
        align_spy_rows, TIMESTAMPS,
        PackedRows(STOCK_ROWS, [1, 1]), TIMESTAMPS, SPY,
    )
    rejects(
        align_spy_rows, TIMESTAMPS,
        PackedRows((STOCK_ROWS[0], object()), (1, 1)),
        TIMESTAMPS, SPY,
    )
    rejects(
        align_spy_rows, TIMESTAMPS, object(), TIMESTAMPS, SPY,
    )
    rejects(
        align_spy_rows, TIMESTAMPS, STOCK, TIMESTAMPS,
        SessionSamples(list(SPY.rows), 8),
    )
    rejects(
        align_spy_rows, TIMESTAMPS, STOCK, TIMESTAMPS, object(),
    )
    rejects(
        align_spy_rows, TIMESTAMPS, STOCK, TIMESTAMPS,
        SessionSamples((*SPY.rows, object()), 8),
    )
    for opportunities in (-1, True, 1):
        rejects(
            align_spy_rows, TIMESTAMPS, STOCK, TIMESTAMPS,
            SessionSamples(SPY.rows, opportunities),
        )
    for ordinal in (-1, True):
        rejects(
            align_spy_rows, TIMESTAMPS, STOCK, TIMESTAMPS,
            SessionSamples((
                SampleRows(1, 2, 4, ordinal),
                SampleRows(2, 3, 5, 102),
            ), 8),
        )


def verify_calendar_boundary() -> None:
    day = date(2024, 11, 29)
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    valid = tuple(
        item.timestamp for item in expected_bins(calendar, day, day, 30)
    )
    assert valid[-1] == "2024-11-29T17:30:00Z"
    rejects(
        session_samples, (*valid, "2024-11-29T18:00:00Z"),
        30, calendar, day, day, 1, 1, 1,
    )


def main() -> None:
    verify_complete_grid()
    verify_sparse_stock_grid()
    verify_alignment_guards()
    verify_calendar_boundary()
    print("relative-context input tests passed")


if __name__ == "__main__":
    main()
