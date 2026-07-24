"""Build common target-time blocks for one frozen sparse-bar universe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tools.chronology import (
    IndexRange, holdout_split, purged_split, split_ranges,
    walk_forward_splits,
)
from tools.session_samples import SampleRows

COHORT_SIZES = (11, 22, 33, 55)
TRANSFER_TRAIN_SIZES = (11, 22, 33, 44)
UNSEEN_START = 44


@dataclass(frozen=True, slots=True)
class CalendarBlocks:
    folds: tuple[tuple[IndexRange, ...], ...]
    holdout: tuple[IndexRange, ...]


@dataclass(frozen=True, slots=True)
class PackedRows:
    rows: tuple[SampleRows, ...]
    counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class UniverseRoles:
    cohorts: tuple[tuple[int, tuple[str, ...]], ...]
    transfer_training: tuple[tuple[int, tuple[str, ...]], ...]
    unseen: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Coverage:
    requested: tuple[str, ...]
    evaluable: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing


def common_calendar(
    opportunities: int, folds: int, fraction: float, embargo: int,
) -> CalendarBlocks:
    """Return half-open opportunity ranges shared by every stock."""
    validation = tuple(
        split_ranges(
            purged_split(split, embargo, preserve_last=False), embargo,
        )
        for split in walk_forward_splits(
            opportunities, folds, fraction, reserved_blocks=2,
        )
    )
    holdout = split_ranges(
        purged_split(holdout_split(opportunities, fraction), embargo),
        embargo,
    )
    return CalendarBlocks(validation, holdout)


def pack_rows(
    rows: Sequence[SampleRows], blocks: Sequence[IndexRange],
    history_bars: int, horizon_bars: int, alignment_horizon_bars: int,
) -> PackedRows:
    """Pack retained rows by common target-opportunity interval."""
    values = tuple(rows)
    dimensions = (history_bars, horizon_bars, alignment_horizon_bars)
    if not blocks or \
       any(not isinstance(block, IndexRange) for block in blocks) or \
       any(
           left.stop > right.start
           for left, right in zip(blocks, blocks[1:], strict=False)
       ) or \
       any(type(value) is not int or value < 1 for value in dimensions) or \
       horizon_bars > alignment_horizon_bars or \
       any(
           not isinstance(row, SampleRows) or
           any(type(value) is not int or value < 0 for value in (
               row.as_of, row.entry, row.target, row.as_of_ordinal,
           ))
           for row in values
       ) or \
       any(
           left.as_of_ordinal >= right.as_of_ordinal
           for left, right in zip(values, values[1:], strict=False)
       ):
        raise ValueError("sample rows or target blocks are invalid")
    first_target = history_bars + alignment_horizon_bars - 1
    opportunities = tuple(
        row.as_of_ordinal + horizon_bars - first_target for row in values
    )
    if any(opportunity < 0 for opportunity in opportunities):
        raise ValueError("sample precedes the first target opportunity")
    grouped = tuple(
        tuple(
            row for row, opportunity in zip(
                values, opportunities, strict=True,
            )
            if block.start <= opportunity < block.stop
        )
        for block in blocks
    )
    return PackedRows(
        tuple(row for group in grouped for row in group),
        tuple(map(len, grouped)),
    )


def universe_roles(series: Sequence[str]) -> UniverseRoles:
    names = tuple(series)
    if len(names) != COHORT_SIZES[-1] or \
       any(not isinstance(name, str) or not name for name in names) or \
       len(set(names)) != len(names):
        raise ValueError("universe requires 55 ordered unique series")

    def prefixes(sizes: Sequence[int]) -> tuple[tuple[int, tuple[str, ...]], ...]:
        return tuple((size, names[:size]) for size in sizes)

    return UniverseRoles(
        prefixes(COHORT_SIZES),
        prefixes(TRANSFER_TRAIN_SIZES),
        names[UNSEEN_START:COHORT_SIZES[-1]],
    )


def coverage(
    series: Sequence[str], sample_counts: Mapping[str, int],
) -> Coverage:
    names = tuple(series)
    if not names or \
       any(not isinstance(name, str) or not name for name in names) or \
       len(set(names)) != len(names) or set(sample_counts) != set(names) or \
       any(type(count) is not int or count < 0 for count in sample_counts.values()):
        raise ValueError("coverage inputs are invalid")
    evaluable = tuple(name for name in names if sample_counts[name])
    missing = tuple(name for name in names if not sample_counts[name])
    return Coverage(names, evaluable, missing)
