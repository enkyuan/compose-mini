#!/usr/bin/env python3
"""Verify context candidates share source-numbered timestamp cells."""

from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.context_diagnostic_inputs import (
    context_all_phase_rows, context_grid_sha256, context_phase_rows,
    timestamp_rows,
)
from tools.session_calendar import (
    DEFAULT_CALENDAR, SessionCalendar, expected_bins,
)
from tools.session_samples import SampleRows, session_samples
from tools.universe_scaling_contract import timestamp_grid_sha256

START = date(2024, 11, 1)
END = date(2026, 7, 21)
MINUTES = 30
HORIZON = ALIGNMENT = 13


def raises(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def grid() -> tuple[str, ...]:
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    return tuple(
        item.timestamp
        for item in expected_bins(calendar, START, END, MINUTES)
    )


def test_source_numbering_is_preserved() -> None:
    timestamps = grid()
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    packed = context_phase_rows(
        timestamps, MINUTES, calendar, START, END, "fold-1",
        HORIZON, ALIGNMENT,
    )
    assert packed.counts == (3_792, 538)

    source = session_samples(
        timestamps, MINUTES, calendar, START, END, 17,
        HORIZON, ALIGNMENT, opportunity_stop=4_393,
    )
    source_by_target = {
        timestamps[row.target]: row for row in source.rows
    }
    assert all(
        source_by_target[timestamps[row.target]] == row
        for row in packed.rows
    )
    assert packed.rows[0].as_of_ordinal == 67
    assert dict(context_all_phase_rows(
        timestamps, MINUTES, calendar, START, END, HORIZON, ALIGNMENT,
    ))["fold-1"] == packed


def test_missing_history_only_removes_affected_cells() -> None:
    timestamps = grid()
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    complete = context_phase_rows(
        timestamps, MINUTES, calendar, START, END, "fold-1",
        HORIZON, ALIGNMENT,
    )
    early = context_phase_rows(
        timestamps[:30] + timestamps[31:], MINUTES, calendar,
        START, END, "fold-1", HORIZON, ALIGNMENT,
    )
    boundary = context_phase_rows(
        timestamps[:3860] + timestamps[3861:], MINUTES, calendar,
        START, END, "fold-1", HORIZON, ALIGNMENT,
    )

    def ordinals(
        packed: object,
    ) -> tuple[set[int], set[int]]:
        split = packed.counts[0]
        return (
            {row.as_of_ordinal for row in packed.rows[:split]},
            {row.as_of_ordinal for row in packed.rows[split:]},
        )

    train, evaluation = ordinals(complete)
    early_train, early_evaluation = ordinals(early)
    boundary_train, boundary_evaluation = ordinals(boundary)
    assert train - early_train == set(range(67, 98))
    assert evaluation == early_evaluation
    assert train - boundary_train == {3_847}
    assert evaluation - boundary_evaluation == set(range(3_871, 3_928))


def test_shifted_source_grid_fails() -> None:
    timestamps = grid()
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    raises(
        context_phase_rows, timestamps, MINUTES, calendar,
        date(2024, 11, 4), END, "fold-1", HORIZON, ALIGNMENT,
    )


def test_timestamp_and_aggregate_hashes_bind_order() -> None:
    timestamps = grid()
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    packed = context_phase_rows(
        timestamps, MINUTES, calendar, START, END, "calibration",
        HORIZON, ALIGNMENT,
    )
    training = timestamp_rows(timestamps, packed.rows[:packed.counts[0]])
    evaluation = timestamp_rows(timestamps, packed.rows[packed.counts[0]:])
    assert timestamp_grid_sha256(training) != \
        timestamp_grid_sha256(evaluation)

    grids = {"A": training, "B": training}
    names = ("A", "B")
    digest = context_grid_sha256("training", names, grids)
    assert digest == context_grid_sha256("training", names, grids)
    assert digest != context_grid_sha256(
        "training", tuple(reversed(names)),
        {"B": training, "A": training},
    )
    assert digest != context_grid_sha256(
        "training", names, {"A": training, "B": evaluation},
    )
    raises(context_grid_sha256, "truth", names, grids)
    raises(context_grid_sha256, "training", ("A",), {})
    raises(context_grid_sha256, "training", ("A",), {"A": ()})
    raises(
        context_grid_sha256, "training", ("A",),
        {"A": ("not-a-timestamp-row",)},
    )
    raises(
        context_grid_sha256, "training", names,
        {"A": training, "C": training},
    )
    raises(timestamp_rows, timestamps, (*packed.rows, object()))
    for row in (
        SampleRows(-1, 0, 1, 0),
        SampleRows(2, 1, 3, 2),
        SampleRows(0, 1, 0, 0),
        SampleRows(True, 1, 2, 0),
    ):
        raises(timestamp_rows, timestamps, (row,))
    valid = packed.rows[:2]
    raises(timestamp_rows, timestamps, tuple(reversed(valid)))


def main() -> None:
    test_source_numbering_is_preserved()
    test_missing_history_only_removes_affected_cells()
    test_shifted_source_grid_fails()
    test_timestamp_and_aggregate_hashes_bind_order()
    print("context diagnostic input tests passed")


if __name__ == "__main__":
    main()
