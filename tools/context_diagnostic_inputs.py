"""Derive the common timestamp cells for temporal-context diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import hashlib
import json

from tools.chronology import IndexRange
from tools.context_diagnostic_contract import (
    HISTORY_LENGTHS, MAX_HISTORY, PHASE_RANGES, SOURCE_OPPORTUNITIES,
    TARGET_PHASES,
)
from tools.session_calendar import SessionCalendar
from tools.session_samples import SampleRows, session_samples
from tools.universe_contract import PackedRows, pack_rows
from tools.universe_scaling_contract import timestamp_grid_sha256


def _context_phase_rows(
    timestamps: Sequence[str], minutes: int, calendar: SessionCalendar,
    start: date, end: date, phases: Sequence[str],
    horizon: int, alignment: int,
) -> tuple[tuple[str, PackedRows], ...]:
    names = tuple(phases)
    if not names or any(name not in PHASE_RANGES for name in names) or \
       len(set(names)) != len(names):
        raise ValueError("context phases are invalid")
    stop = max(PHASE_RANGES[name][-1][1] for name in names)
    source = session_samples(
        timestamps, minutes, calendar, start, end, HISTORY_LENGTHS[0],
        horizon, alignment, opportunity_stop=stop,
    )
    if source.opportunities != SOURCE_OPPORTUNITIES:
        raise ValueError("context source opportunity grid changed")
    eligible = {
        row.as_of_ordinal for row in session_samples(
            timestamps, minutes, calendar, start, end, MAX_HISTORY,
            horizon, alignment,
        ).rows
    }
    rows = tuple(
        row for row in source.rows if row.as_of_ordinal in eligible
    )
    return tuple(
        (
            name,
            pack_rows(
                rows,
                tuple(
                    IndexRange(*item) for item in PHASE_RANGES[name]
                ),
                HISTORY_LENGTHS[0], horizon, alignment,
            ),
        )
        for name in names
    )


def context_phase_rows(
    timestamps: Sequence[str], minutes: int, calendar: SessionCalendar,
    start: date, end: date, phase: str, horizon: int, alignment: int,
) -> PackedRows:
    """Filter one source phase to cells eligible for every history."""
    return _context_phase_rows(
        timestamps, minutes, calendar, start, end, (phase,),
        horizon, alignment,
    )[0][1]


def context_all_phase_rows(
    timestamps: Sequence[str], minutes: int, calendar: SessionCalendar,
    start: date, end: date, horizon: int, alignment: int,
) -> tuple[tuple[str, PackedRows], ...]:
    """Filter every frozen source phase with one eligibility scan."""
    return _context_phase_rows(
        timestamps, minutes, calendar, start, end, TARGET_PHASES,
        horizon, alignment,
    )


def timestamp_rows(
    timestamps: Sequence[str], rows: Sequence[SampleRows],
) -> tuple[tuple[str, str, str], ...]:
    """Resolve indexed sample rows to their causal timestamp triples."""
    values = tuple(timestamps)
    samples = tuple(rows)
    if any(
        type(row) is not SampleRows or any(
            type(value) is not int or value < 0 for value in (
                row.as_of, row.entry, row.target, row.as_of_ordinal,
            )
        ) or row.entry != row.as_of + 1 or row.target <= row.as_of or
        max(row.as_of, row.entry, row.target) >= len(values)
        for row in samples
    ) or any(
        left.as_of_ordinal >= right.as_of_ordinal
        for left, right in zip(samples, samples[1:])
    ):
        raise ValueError("context sample rows are invalid")
    result = tuple(
        (values[row.as_of], values[row.entry], values[row.target])
        for row in samples
    )
    timestamp_grid_sha256(result)
    return result


def context_grid_sha256(
    role: str, expected: Sequence[str],
    grids: Mapping[str, Sequence[Sequence[str]]],
) -> str:
    """Hash one ordered multi-series training or evaluation grid."""
    names = tuple(expected)
    if role not in ("training", "evaluation") or \
       not names or any(not isinstance(name, str) or not name for name in names) \
       or len(set(names)) != len(names) or \
       not isinstance(grids, Mapping) or tuple(grids) != names:
        raise ValueError("context grid inputs are invalid")
    records = []
    for series, rows in grids.items():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) \
           or not rows:
            raise ValueError("context grid rows are invalid")
        records.append({
            "count": len(rows),
            "grid_sha256": timestamp_grid_sha256(rows),
            "series": series,
        })
    return hashlib.sha256(json.dumps({
        "records": records,
        "role": role,
        "schema": 1,
    }, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
