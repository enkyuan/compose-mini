"""Compute deterministic stock-macro universe-scaling statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from math import exp, fsum, isfinite
from random import Random
from statistics import fmean

from tools.universe_contract import COHORT_SIZES, universe_roles

BOOTSTRAP_BLOCK_DAYS = (5, 10, 20)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_724


@dataclass(frozen=True, slots=True)
class ForecastPoint:
    target_time: str
    actual_return: float
    predicted_return: float
    reference_price: float
    outcome_price: float

    def __post_init__(self) -> None:
        values = (
            self.actual_return, self.predicted_return,
            self.reference_price, self.outcome_price,
        )
        if not isinstance(self.target_time, str) or not self.target_time or \
           any(type(value) not in (int, float) or not isfinite(value)
               for value in values) or min(values[2:]) <= 0:
            raise ValueError("forecast point is invalid")
        _day(self.target_time)


@dataclass(frozen=True, slots=True)
class EffectiveCount:
    value: float | None
    included: tuple[str, ...]
    excluded: tuple[str, ...]
    reason: str | None


def _names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(not isinstance(name, str) or not name for name in names) or \
       len(set(names)) != len(names):
        raise ValueError("series names must be nonempty and unique")
    return names


def _day(timestamp: str) -> str:
    try:
        value = str(date.fromisoformat(timestamp[:10]))
    except (TypeError, ValueError):
        raise ValueError("target time must begin with an ISO date") from None
    if not timestamp.startswith(value):
        raise ValueError("target time must begin with an ISO date")
    return value


def validate_nested_cohorts(
    master: Sequence[str],
    cohorts: Mapping[int, Sequence[str]],
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Return exact ordered prefixes after validating the frozen cohort ladder."""
    names = _names(master)
    if tuple(cohorts) != COHORT_SIZES or len(names) != COHORT_SIZES[-1]:
        raise ValueError("cohorts must be the exact 11/22/33/55 ladder")
    ordered = tuple((size, tuple(cohorts[size])) for size in COHORT_SIZES)
    if any(members != names[:size] for size, members in ordered):
        raise ValueError("cohorts must be exact master prefixes")
    return ordered


def cohort_views(
    master: Sequence[str],
    size: int,
    core_size: int = COHORT_SIZES[0],
) -> dict[str, tuple[str, ...]]:
    """Partition one prefix into invariant core, added, and all-stock views."""
    names = _names(master)
    if type(size) is not int or type(core_size) is not int or \
       not 1 <= core_size <= size <= len(names):
        raise ValueError("cohort view sizes are invalid")
    members = names[:size]
    return {
        "core": members[:core_size],
        "added": members[core_size:],
        "all": members,
    }


def unseen_view(master: Sequence[str]) -> tuple[str, ...]:
    return universe_roles(_names(master)).unseen


def _series(
    values: Mapping[str, Sequence[ForecastPoint]],
) -> tuple[tuple[str, tuple[ForecastPoint, ...]], ...]:
    if not isinstance(values, Mapping) or not values or any(
        not isinstance(name, str) or not name for name in values
    ):
        raise ValueError("forecast series must be a mapping")
    names = _names(sorted(values))
    result = []
    for name in names:
        points = tuple(values[name])
        if not points or \
           any(not isinstance(point, ForecastPoint) for point in points) or \
           any(left.target_time >= right.target_time
               for left, right in zip(points, points[1:], strict=False)):
            raise ValueError("forecast points must be nonempty and ordered")
        result.append((name, points))
    return tuple(result)


def stock_macro_metrics(
    values: Mapping[str, Sequence[ForecastPoint]],
) -> dict[str, float]:
    """Average each metric within stock, then equally across stocks."""
    def close_error(point: ForecastPoint) -> float:
        try:
            predicted = point.reference_price * exp(point.predicted_return)
        except OverflowError:
            raise ValueError("reconstructed close must be finite") from None
        if not isfinite(predicted):
            raise ValueError("reconstructed close must be finite")
        return abs(point.outcome_price - predicted)

    metrics = []
    for _, points in _series(values):
        errors = tuple(
            point.actual_return - point.predicted_return for point in points
        )
        squares = tuple(error * error for error in errors)
        if not all(map(isfinite, (*errors, *squares))):
            raise ValueError("return errors must remain finite")
        metrics.append((
            fmean(abs(error) for error in errors),
            fmean(squares),
            fmean(
                (point.predicted_return > 0) - (point.predicted_return < 0) ==
                (point.actual_return > 0) - (point.actual_return < 0)
                for point in points
            ),
            fmean(map(close_error, points)),
        ))
    result = dict(zip(
        ("return_mae", "return_mse", "direction_accuracy", "close_mae"),
        (fmean(column) for column in zip(*metrics, strict=True)),
        strict=True,
    ))
    if not all(map(isfinite, result.values())):
        raise ValueError("stock-macro metrics must remain finite")
    return result


def _paired_daily(
    candidate: Mapping[str, Sequence[ForecastPoint]],
    reference: Mapping[str, Sequence[ForecastPoint]],
) -> dict[str, dict[str, tuple[float, ...]]]:
    candidates, references = _series(candidate), _series(reference)
    if tuple(name for name, _ in candidates) != tuple(
        name for name, _ in references
    ):
        raise ValueError("paired forecasts must use identical stocks")
    daily: dict[str, dict[str, list[float]]] = {}
    for (name, left), (_, right) in zip(candidates, references, strict=True):
        if len(left) != len(right):
            raise ValueError("paired forecasts must use identical target grids")
        daily[name] = {}
        for candidate_point, reference_point in zip(left, right, strict=True):
            if (
                candidate_point.target_time,
                candidate_point.actual_return,
                candidate_point.reference_price,
                candidate_point.outcome_price,
            ) != (
                reference_point.target_time,
                reference_point.actual_return,
                reference_point.reference_price,
                reference_point.outcome_price,
            ):
                raise ValueError("paired forecasts must use identical targets")
            gain = (
                abs(reference_point.actual_return -
                    reference_point.predicted_return) -
                abs(candidate_point.actual_return -
                    candidate_point.predicted_return)
            )
            daily[name].setdefault(_day(candidate_point.target_time), []).append(gain)
    return {
        name: {day: tuple(items) for day, items in by_day.items()}
        for name, by_day in daily.items()
    }


def _union_dates(
    values: Mapping[str, Mapping[str, Sequence[float]]],
) -> tuple[str, ...]:
    if not isinstance(values, Mapping) or not values or any(
        not isinstance(name, str) or not name or
        not isinstance(by_date, Mapping) or not by_date or any(
            not isinstance(day, str) or not day or
            isinstance(items, (str, bytes)) or
            not isinstance(items, Sequence) or not items or any(
                type(value) not in (int, float) or not isfinite(value)
                for value in items
            )
            for day, items in by_date.items()
        )
        for name, by_date in values.items()
    ):
        raise ValueError("daily values must be nonempty mappings")
    return tuple(sorted(set().union(*(
        set(by_date) for by_date in values.values()
    ))))


def _common_dates(
    values: Mapping[str, Mapping[str, Sequence[float]]],
) -> tuple[str, ...]:
    _union_dates(values)
    dates = tuple(sorted(set.intersection(*map(
        set, values.values(),
    ))))
    if not dates:
        raise ValueError("daily values need finite common-date observations")
    return dates


def _macro(
    values: Mapping[str, Mapping[str, Sequence[float]]],
    dates: Sequence[str],
) -> float:
    return fmean(
        fmean(
            value for day in dates if day in values[name]
            for value in values[name][day]
        )
        for name in sorted(values)
    )


def circular_block_means(
    values: Mapping[str, Mapping[str, Sequence[float]]],
    block_days: int,
    *,
    session_dates: Sequence[str] | None = None,
    sample_days: int | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, ...]:
    """Resample one stock-macro mean over shared circular date blocks."""
    observed = _union_dates(values)
    if session_dates is None:
        dates = _common_dates(values)
    elif isinstance(session_dates, (str, bytes)) or \
         not isinstance(session_dates, Sequence):
        raise ValueError("bootstrap session grid is invalid")
    else:
        dates = tuple(session_dates)
        if dates != observed or any(
            tuple(by_date) != tuple(day for day in dates if day in by_date)
            for by_date in values.values()
        ):
            raise ValueError("bootstrap session grid is invalid")
    count = len(dates) if sample_days is None else sample_days
    if type(block_days) is not int or type(count) is not int or \
       type(replicates) is not int or type(seed) is not int or \
       not 1 <= block_days <= min(len(dates), count) or replicates < 2:
        raise ValueError("bootstrap parameters are invalid")
    daily = tuple(
        tuple(
            (
                (sum(values[name][day]), len(values[name][day]))
                if day in values[name] else (0.0, 0)
            )
            for day in dates
        )
        for name in sorted(values)
    )
    full_blocks, remainder = divmod(count, block_days)

    def blocks(stock: Sequence[tuple[float, int]], width: int) -> tuple[
        tuple[float, int], ...
    ]:
        return tuple((
            sum(
                stock[(start + offset) % len(dates)][0]
                for offset in range(width)
            ),
            sum(
                stock[(start + offset) % len(dates)][1]
                for offset in range(width)
            ),
        ) for start in range(len(dates)))

    prepared = tuple((
        blocks(stock, block_days),
        blocks(stock, remainder) if remainder else (),
    ) for stock in daily)
    if any(
        any(count == 0 for _, count in full) or
        any(count == 0 for _, count in partial)
        for full, partial in prepared
    ):
        raise ValueError("bootstrap session coverage is invalid")
    generator, samples = Random(seed), []
    for _ in range(replicates):
        starts = tuple(
            generator.randrange(len(dates)) for _ in range(full_blocks)
        )
        tail = generator.randrange(len(dates)) if remainder else None
        samples.append(fmean(
            (
                sum(full[start][0] for start in starts) +
                (partial[tail][0] if tail is not None else 0.0)
            ) / (
                sum(full[start][1] for start in starts) +
                (partial[tail][1] if tail is not None else 0)
            )
            for full, partial in prepared
        ))
    return tuple(sorted(samples))


def circular_block_interval(
    values: Mapping[str, Mapping[str, Sequence[float]]],
    block_days: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap one paired stock-macro mean over its observed date count."""
    samples = circular_block_means(
        values, block_days, replicates=replicates, seed=seed,
    )
    return (
        samples[int(0.025 * (replicates - 1))],
        samples[int(0.975 * (replicates - 1))],
    )


def effective_count(
    values: Mapping[str, Mapping[str, float]],
) -> EffectiveCount:
    """Estimate cross-stock breadth from aligned daily loss differentials."""
    if not isinstance(values, Mapping) or not values or any(
        not isinstance(name, str) or not name or
        not isinstance(column, Mapping) or not column
        for name, column in values.items()
    ):
        raise ValueError("effective-count columns are invalid")
    names = tuple(sorted(values))
    dates = tuple(sorted(next(iter(values.values()), {})))
    if any(tuple(sorted(values[name])) != dates for name in names):
        raise ValueError("effective-count columns must be exactly aligned")
    if len(dates) < 2:
        return EffectiveCount(None, names, (), "fewer-than-two-aligned-dates")
    columns = {
        name: tuple(float(values[name][day]) for day in dates) for name in names
    }
    if any(not all(map(isfinite, column)) for column in columns.values()):
        raise ValueError("effective-count values must be finite")
    excluded = tuple(
        name for name, column in columns.items()
        if all(value == column[0] for value in column[1:])
    )
    included = tuple(name for name in names if name not in excluded)
    if len(included) < 2:
        return EffectiveCount(
            None, included, excluded, "fewer-than-two-nonconstant-stocks",
        )
    means = {name: fmean(columns[name]) for name in included}
    centered = {
        name: tuple(value - means[name] for value in columns[name])
        for name in included
    }
    trace = fsum(
        value * value for column in centered.values() for value in column
    )
    denominator = fsum(
        fsum(centered[name][index] for name in included) ** 2
        for index in range(len(dates))
    )
    if denominator <= 0 or not all(map(isfinite, (trace, denominator))):
        return EffectiveCount(
            None, included, excluded, "nonpositive-or-nonfinite-denominator",
        )
    result = len(included) * trace / denominator
    return (
        EffectiveCount(result, included, excluded, None)
        if isfinite(result) else
        EffectiveCount(
            None, included, excluded, "nonpositive-or-nonfinite-denominator",
        )
    )


def paired_comparison(
    candidate: Mapping[str, Sequence[ForecastPoint]],
    reference: Mapping[str, Sequence[ForecastPoint]],
    *,
    block_days: Sequence[int] = BOOTSTRAP_BLOCK_DAYS,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Compare aligned forecasts; positive gain means the candidate is better."""
    daily = _paired_daily(candidate, reference)
    dates = _common_dates(daily)
    per_stock = {
        name: fmean(value for day in dates for value in daily[name][day])
        for name in sorted(daily)
    }
    daily_means = {
        name: {day: fmean(daily[name][day]) for day in dates}
        for name in sorted(daily)
    }
    blocks = tuple(block_days)
    if not blocks or any(type(block) is not int for block in blocks) or \
       any(left >= right for left, right in zip(
           blocks, blocks[1:], strict=False,
       )):
        raise ValueError("bootstrap block sizes are invalid")
    return {
        "common_dates": dates,
        "mean_gain": _macro(daily, dates),
        "per_stock_mean_gain": per_stock,
        "wins": sum(value > 0 for value in per_stock.values()),
        "ties": sum(value == 0 for value in per_stock.values()),
        "losses": sum(value < 0 for value in per_stock.values()),
        "intervals": {
            str(block): circular_block_interval(
                daily, block, replicates, seed,
            )
            for block in blocks
        },
        "effective_count": asdict(effective_count(daily_means)),
    }
