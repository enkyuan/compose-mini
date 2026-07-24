"""Define deterministic chronological split counts and index ranges."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndexRange:
    start: int
    stop: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.stop) is not int or \
           not 0 <= self.start < self.stop:
            raise ValueError("index range must be nonempty and ordered")


def walk_forward_splits(
    samples: int, folds: int, fraction: float, reserved_blocks: int = 1,
) -> tuple[tuple[int, int], ...]:
    if type(samples) is not int or type(folds) is not int or \
       type(reserved_blocks) is not int or samples < 1 or folds < 1 or \
       reserved_blocks < 1 or not isinstance(fraction, float) or \
       not 0.0 < fraction < 1.0:
        raise ValueError("walk-forward split arguments are invalid")
    block = int(samples * fraction)
    initial = samples - (folds + reserved_blocks) * block
    if min(block, initial) <= 0:
        raise ValueError("series is too short for the requested folds")
    return tuple((initial + fold * block, block) for fold in range(folds))


def holdout_split(samples: int, fraction: float) -> tuple[int, int, int]:
    if type(samples) is not int or samples < 1 or \
       not isinstance(fraction, float) or not 0.0 < fraction < 1.0:
        raise ValueError("holdout split arguments are invalid")
    block = int(samples * fraction)
    if block <= 0 or samples - 2 * block <= 0:
        raise ValueError("series is too short for validation and test holdouts")
    return samples - 2 * block, block, block


def purged_split(
    split: tuple[int, ...], gap: int, preserve_last: bool = True,
) -> tuple[int, ...]:
    """Remove boundary labels whose horizons overlap the following split."""
    if not split or type(gap) is not int or gap < 0 or \
       type(preserve_last) is not bool or \
       any(type(count) is not int or count < 1 for count in split):
        raise ValueError("split counts or embargo are invalid")
    purge_count = len(split) - int(preserve_last)
    result = tuple(
        count - gap if index < purge_count else count
        for index, count in enumerate(split)
    )
    if min(result) <= 0:
        raise ValueError("series blocks must exceed the horizon embargo")
    return result


def split_ranges(
    split: Sequence[int], gap: int,
) -> tuple[IndexRange, ...]:
    if not split or type(gap) is not int or gap < 0 or \
       any(type(count) is not int or count < 1 for count in split):
        raise ValueError("split counts or gap are invalid")
    ranges, start = [], 0
    for count in split:
        ranges.append(IndexRange(start, start + count))
        start += count + gap
    return tuple(ranges)
