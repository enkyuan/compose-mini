#!/usr/bin/env python3
"""Verify pure stock-macro universe-scaling statistics."""

from dataclasses import replace
import json
import math
from pathlib import Path
from random import Random
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.universe_scaling import (
    ForecastPoint, cohort_views, circular_block_interval,
    circular_block_means, effective_count, paired_comparison,
    stock_macro_metrics, unseen_view,
    validate_nested_cohorts,
)
from tools.universe_contract import universe_roles


def raises(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def point(day: int, actual: float, predicted: float) -> ForecastPoint:
    return ForecastPoint(
        f"2026-01-{day:02d}T20:00:00Z", actual, predicted, 100.0, 101.0,
    )


def test_cohorts() -> None:
    master = tuple(f"S{index:02}" for index in range(55))
    cohorts = {size: master[:size] for size in (11, 22, 33, 55)}
    assert validate_nested_cohorts(master, cohorts) == tuple(cohorts.items())
    assert cohort_views(master, 22) == {
        "core": master[:11], "added": master[11:22], "all": master[:22],
    }
    assert unseen_view(master) == master[44:55]
    roles = universe_roles(master)
    assert tuple(size for size, _ in roles.transfer_training) == (11, 22, 33, 44)
    assert set(roles.transfer_training[-1][1]).isdisjoint(roles.unseen)
    for invalid in (
        {11: master[:11], 22: master[:22], 55: master},
        {11: master[:11], 33: master[:33], 22: master[:22], 55: master},
        cohorts | {22: (*master[:21], master[0])},
        cohorts | {33: master[1:34]},
    ):
        raises(validate_nested_cohorts, master, invalid)
    raises(cohort_views, master, 10)
    raises(unseen_view, master[:-1])


def test_macro_metrics() -> None:
    values = {
        "A": (point(1, 1.0, 0.0),),
        "B": tuple(point(day, 4.0, 1.0) for day in (1, 2, 3)),
    }
    metrics = stock_macro_metrics(values)
    assert metrics["return_mae"] == 2.0
    assert metrics["return_mse"] == 5.0
    assert metrics["direction_accuracy"] == 0.5
    assert metrics["return_mae"] != (1.0 + 3.0 * 3) / 4
    raises(stock_macro_metrics, {})
    raises(
        stock_macro_metrics,
        {"A": (point(2, 0.0, 0.0), point(1, 0.0, 0.0))},
    )
    raises(ForecastPoint, "bad", 0.0, 0.0, 1.0, 1.0)
    raises(stock_macro_metrics, {1: (point(1, 0.0, 0.0),)})
    raises(stock_macro_metrics, {"A": (point(1, 1e308, -1e308),)})


def test_paired_comparison() -> None:
    actuals = {
        "A": tuple(point(day, float(day % 3 - 1), 0.0) for day in range(1, 21)),
        "B": tuple(point(day, float((day + 1) % 3 - 1), 0.0)
                   for day in range(1, 21)),
    }
    reference = {
        name: tuple(replace(item, predicted_return=item.actual_return + 1.0)
                    for item in points)
        for name, points in actuals.items()
    }
    candidate = {
        name: tuple(replace(item, predicted_return=item.actual_return)
                    for item in points)
        for name, points in actuals.items()
    }
    result = paired_comparison(
        candidate, reference, block_days=(5, 10, 20), replicates=200,
    )
    assert result["mean_gain"] == 1.0
    assert result["wins"] == 2 and not result["ties"] and not result["losses"]
    assert all(interval == (1.0, 1.0)
               for interval in result["intervals"].values())
    assert result == paired_comparison(
        candidate, reference, block_days=(5, 10, 20), replicates=200,
    )
    assert result == paired_comparison(
        dict(reversed(tuple(candidate.items()))),
        dict(reversed(tuple(reference.items()))),
        block_days=(5, 10, 20), replicates=200,
    )
    json.dumps(result, allow_nan=False, sort_keys=True)
    assert result["effective_count"]["value"] is None

    mismatched = dict(candidate)
    mismatched["A"] = mismatched["A"][:-1]
    raises(paired_comparison, mismatched, reference, block_days=(5,), replicates=5)
    changed = dict(candidate)
    changed["A"] = (
        replace(changed["A"][0], actual_return=99.0), *changed["A"][1:],
    )
    raises(paired_comparison, changed, reference, block_days=(5,), replicates=5)
    raises(
        paired_comparison, candidate, reference,
        block_days=(10, 5), replicates=5,
    )


def reference_block_means(
    blocks: dict[str, dict[str, tuple[float, ...]]],
    width: int,
    sample_days: int,
    replicates: int,
    seed: int,
    session_dates: tuple[str, ...] | None = None,
) -> tuple[float, ...]:
    """Reproduce the original circular-block loop as a test oracle."""
    dates = session_dates or tuple(next(iter(blocks.values())))
    generator, samples = Random(seed), []
    for _ in range(replicates):
        selected = []
        while len(selected) < sample_days:
            start = generator.randrange(len(dates))
            selected.extend(
                (start + offset) % len(dates) for offset in range(width)
            )
        selected = selected[:sample_days]
        samples.append(fmean(
            sum(
                sum(blocks[name].get(dates[index], ())) for index in selected
            ) / sum(
                len(blocks[name].get(dates[index], ()))
                for index in selected
            )
            for name in sorted(blocks)
        ))
    return tuple(sorted(samples))


def test_bootstrap_and_effective_count() -> None:
    blocks = {
        "A": {
            f"2026-01-{day:02d}": (float(day), float(day + 1))
            for day in range(1, 22)
        },
        "B": {
            f"2026-01-{day:02d}": (float(day * day % 13),)
            for day in range(1, 22)
        },
    }
    dates = tuple(blocks["A"])
    for width in (5, 10, 20):
        observed = circular_block_interval(blocks, width, 200, 7)
        samples = reference_block_means(
            blocks, width, len(dates), 200, 7,
        )
        assert observed == (
            samples[int(0.025 * 199)], samples[int(0.975 * 199)],
        )
        assert observed[0] < observed[1]
        assert observed == circular_block_interval(blocks, width, 200, 7)
    extended = circular_block_means(
        blocks, 5, sample_days=60, replicates=200, seed=7,
    )
    assert extended == reference_block_means(blocks, 5, 60, 200, 7)
    masked = {
        **blocks,
        "B": {
            day: values for index, (day, values) in enumerate(
                blocks["B"].items(),
            ) if not 5 <= index < 11
        },
    }
    assert circular_block_means(
        masked, 10, session_dates=dates, sample_days=60,
        replicates=200, seed=7,
    ) == reference_block_means(masked, 10, 60, 200, 7, dates)
    raises(
        circular_block_means, masked, 10, session_dates=dates[:-1],
        sample_days=60, replicates=200, seed=7,
    )
    raises(
        circular_block_means, masked, 5, session_dates=dates,
        sample_days=60, replicates=200, seed=7,
    )
    complete = {
        name: dict(tuple(values.items())[:20])
        for name, values in blocks.items()
    }
    for width in (5, 10, 20):
        samples = reference_block_means(complete, width, 20, 200, 7)
        assert circular_block_interval(complete, width, 200, 7) == (
            samples[int(0.025 * 199)], samples[int(0.975 * 199)],
        )
    raises(circular_block_interval, blocks, 22, 10, 7)
    raises(circular_block_interval, {}, 1, 10, 7)
    raises(
        circular_block_means, blocks, 5,
        sample_days=4, replicates=10, seed=7,
    )

    hand = effective_count({
        "A": {"d1": 0.0, "d2": 1.0, "d3": 2.0},
        "B": {"d1": 0.0, "d2": 2.0, "d3": 4.0},
        "C": {"d1": 1.0, "d2": 1.0, "d3": 1.0},
    })
    assert math.isclose(hand.value or 0.0, 10 / 9)
    assert hand.included == ("A", "B") and hand.excluded == ("C",)
    assert hand.reason is None

    short = effective_count({"A": {"d1": 1.0}, "B": {"d1": 2.0}})
    assert short.value is None and short.reason == "fewer-than-two-aligned-dates"
    constant = effective_count({
        "A": {"d1": 1.0, "d2": 1.0},
        "B": {"d1": 2.0, "d2": 2.0},
    })
    assert constant.value is None
    inverse = effective_count({
        "A": {"d1": -1.0, "d2": 1.0},
        "B": {"d1": 1.0, "d2": -1.0},
    })
    assert inverse.value is None
    raises(effective_count, {
        "A": {"d1": 0.0, "d2": 1.0},
        "B": {"d1": 0.0, "d3": 1.0},
    })
    raises(effective_count, {
        "A": {"d1": 0.0, "d2": math.inf},
        "B": {"d1": 1.0, "d2": 2.0},
    })
    reversed_blocks = dict(reversed(tuple(blocks.items())))
    assert circular_block_interval(reversed_blocks, 5, 200, 7) == \
        circular_block_interval(blocks, 5, 200, 7)


def main() -> None:
    test_cohorts()
    test_macro_metrics()
    test_paired_comparison()
    test_bootstrap_and_effective_count()
    print("universe scaling tests passed")


if __name__ == "__main__":
    main()
