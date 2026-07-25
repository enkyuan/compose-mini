"""Measure stock-selection signal after removing each group's market mean."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from math import fsum, isfinite, sqrt
from statistics import fmean

from tools.universe_scaling import (
    BOOTSTRAP_BLOCK_DAYS, EffectiveCount, circular_block_interval,
    effective_count,
)

CROSS_SECTION_SEED = 20_260_725


def _day(value: str) -> str:
    try:
        parsed = str(date.fromisoformat(value))
    except (TypeError, ValueError):
        raise ValueError("cross-sectional day must be an ISO date") from None
    if parsed != value:
        raise ValueError("cross-sectional day must be an ISO date")
    return parsed


@dataclass(frozen=True, slots=True)
class CrossSectionCell:
    day: str
    group: str
    series: str
    actual: float
    predicted: float

    def __post_init__(self) -> None:
        _day(self.day)
        values = (self.actual, self.predicted)
        if any(
            not isinstance(value, str) or not value
            for value in (self.group, self.series)
        ) or any(
            type(value) not in (int, float) or not isfinite(value)
            for value in values
        ):
            raise ValueError("cross-sectional cell is invalid")


@dataclass(frozen=True, slots=True)
class BlockInterval:
    block_days: int
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class CrossSectionResult:
    r2: float | None
    paired_mean: float
    intervals: tuple[BlockInterval, ...]
    raw_breadth: int
    group_count: int
    date_count: int
    eligible_spearman_groups: int
    excluded_spearman_groups: int
    mean_spearman: float | None
    effective_breadth: EffectiveCount
    meets_statistical_gate: bool


def _names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(values)
    if len(names) < 2 or any(
        not isinstance(name, str) or not name for name in names
    ) or len(set(names)) != len(names):
        raise ValueError("series names must be unique")
    return names


def _groups(values: Sequence[tuple[str, str]]) -> dict[str, str]:
    groups = tuple(values)
    if not groups or any(
        not isinstance(item, tuple) or len(item) != 2
        for item in groups
    ):
        raise ValueError("expected groups must be day/group pairs")
    result = {
        group: _day(day)
        for day, group in groups
        if isinstance(group, str) and group
    }
    if len(result) != len(groups):
        raise ValueError("expected group names must be nonempty and unique")
    return result


def _ranks(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result, start = [0.0] * len(values), 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and \
              values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end + 1) / 2.0
        for index in ordered[start:end]:
            result[index] = rank
        start = end
    return tuple(result)


def _center(values: Sequence[float]) -> tuple[float, ...]:
    try:
        mean = fmean(values)
        centered = tuple(value - mean for value in values)
    except OverflowError:
        raise ValueError("cross-sectional values overflowed") from None
    if not all(map(isfinite, centered)):
        raise ValueError("cross-sectional values must remain finite")
    return centered


def _spearman(
    left: Sequence[float], right: Sequence[float],
) -> float | None:
    ranks = _ranks(left), _ranks(right)
    centered = tuple(
        tuple(value - fmean(values) for value in values)
        for values in ranks
    )
    scales = tuple(
        sqrt(fsum(value * value for value in values))
        for values in centered
    )
    if 0.0 in scales:
        return None
    value = fsum(
        first * second
        for first, second in zip(*centered, strict=True)
    ) / (scales[0] * scales[1])
    if not isfinite(value):
        raise ValueError("cross-sectional correlation is non-finite")
    return value


def cross_section_diagnostics(
    cells: Sequence[CrossSectionCell],
    series: Sequence[str],
    *,
    expected_groups: Sequence[tuple[str, str]],
    block_days: Sequence[int] = BOOTSTRAP_BLOCK_DAYS,
    replicates: int = 10_000,
    seed: int = CROSS_SECTION_SEED,
) -> CrossSectionResult:
    """Compute diagnostics for a caller-authenticated group grid."""
    names, groups = _names(series), _groups(expected_groups)
    rows = tuple(cells)
    blocks = tuple(block_days)
    if (
        not rows
        or any(not isinstance(row, CrossSectionCell) for row in rows)
        or not blocks
        or any(type(width) is not int or width < 1 for width in blocks)
        or any(
            left >= right
            for left, right in zip(blocks, blocks[1:], strict=False)
        )
    ):
        raise ValueError("cross-sectional inputs are invalid")

    grouped: dict[str, tuple[str, dict[str, CrossSectionCell]]] = {}
    for row in rows:
        day, members = grouped.setdefault(row.group, (row.day, {}))
        if day != row.day or row.series in members:
            raise ValueError("cross-sectional group is duplicated")
        members[row.series] = row
    if {
        group: day for group, (day, _) in grouped.items()
    } != groups or any(
        set(members) != set(names) for _, members in grouped.values()
    ):
        raise ValueError("cross-sectional groups must be complete")

    losses = {name: {} for name in names}
    residual, total, correlations = 0.0, 0.0, []
    for _, (day, members) in sorted(
        grouped.items(), key=lambda item: (item[1][0], item[0]),
    ):
        actual = tuple(members[name].actual for name in names)
        predicted = tuple(members[name].predicted for name in names)
        centered = tuple(map(_center, (actual, predicted)))
        correlation = _spearman(*centered)
        if correlation is not None:
            correlations.append(correlation)
        for name, observed, forecast in zip(
            names, *centered, strict=True,
        ):
            error = observed - forecast
            residual += error * error
            total += observed * observed
            losses[name].setdefault(day, []).append(
                abs(error) - abs(observed)
            )
    if not all(map(isfinite, (residual, total))):
        raise ValueError("cross-sectional loss is non-finite")

    paired = {
        name: {
            day: tuple(values) for day, values in by_day.items()
        }
        for name, by_day in losses.items()
    }
    daily = {
        name: {
            day: fmean(values) for day, values in by_day.items()
        }
        for name, by_day in paired.items()
    }
    intervals = tuple(
        BlockInterval(
            width,
            *circular_block_interval(
                paired, width, replicates=replicates, seed=seed,
            ),
        )
        for width in blocks
    )
    r2 = 1.0 - residual / total if total > 0.0 else None
    if r2 is not None and not isfinite(r2):
        raise ValueError("cross-sectional R-squared is non-finite")
    mean = fmean(
        value for by_day in paired.values()
        for values in by_day.values() for value in values
    )
    eligible, groups = len(correlations), len(grouped)
    meets_gate = r2 is not None and r2 > 0.0 and \
        max(interval.upper for interval in intervals) < 0.0
    return CrossSectionResult(
        r2, mean, intervals, len(names), groups,
        len(set(day for day, _ in grouped.values())),
        eligible, groups - eligible,
        fmean(correlations) if correlations else None,
        effective_count(daily), meets_gate,
    )
