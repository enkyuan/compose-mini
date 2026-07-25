#!/usr/bin/env python3
"""Verify pure cross-sectional stock-selection diagnostics."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from math import isclose
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.universe_cross_section import (
    CrossSectionCell, cross_section_diagnostics,
)

SERIES = ("A", "B", "C")
GROUPS = (("2026-01-02", "g1"), ("2026-01-05", "g2"))


def raises(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def perfect_cells() -> tuple[CrossSectionCell, ...]:
    return tuple(
        CrossSectionCell(day, group, series, actual, actual)
        for day, group, actuals in (
            ("2026-01-02", "g1", (-1.0, 0.0, 1.0)),
            ("2026-01-05", "g2", (-2.0, 0.0, 2.0)),
        )
        for series, actual in zip(SERIES, actuals, strict=True)
    )


def test_literal_diagnostics() -> None:
    cells = perfect_cells()
    result = cross_section_diagnostics(
        cells, SERIES, expected_groups=GROUPS,
        block_days=(1, 2), replicates=100, seed=7,
    )
    assert result.r2 == 1.0
    assert result.paired_mean == -1.0
    assert result.raw_breadth == 3
    assert result.group_count == result.date_count == 2
    assert result.eligible_spearman_groups == 2
    assert result.excluded_spearman_groups == 0
    assert isclose(result.mean_spearman or 0.0, 1.0)
    assert tuple(item.block_days for item in result.intervals) == (1, 2)
    assert max(item.upper for item in result.intervals) < 0.0
    assert result.effective_breadth.value == 1.0
    assert result.effective_breadth.included == ("A", "C")
    assert result.effective_breadth.excluded == ("B",)
    assert result.meets_statistical_gate

    shifted = tuple(
        replace(
            cell,
            actual=cell.actual + 100.0,
            predicted=cell.predicted - 50.0,
        )
        for cell in reversed(cells)
    )
    assert cross_section_diagnostics(
        shifted, SERIES, expected_groups=GROUPS,
        block_days=(1, 2), replicates=100, seed=7,
    ) == result

    try:
        result.meets_statistical_gate = False  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("cross-sectional result is mutable")


def test_common_signal_has_no_stock_selection() -> None:
    cells = tuple(
        replace(cell, predicted=10.0) for cell in perfect_cells()
    )
    result = cross_section_diagnostics(
        cells, SERIES, expected_groups=GROUPS,
        block_days=(1, 2), replicates=20, seed=7,
    )
    assert result.r2 == result.paired_mean == 0.0
    assert result.mean_spearman is None
    assert result.eligible_spearman_groups == 0
    assert result.excluded_spearman_groups == 2
    assert not result.meets_statistical_gate


def test_rank_and_level_metrics_are_distinct() -> None:
    cells = tuple(
        replace(cell, predicted=2.0 * cell.actual)
        for cell in perfect_cells()
    )
    result = cross_section_diagnostics(
        cells, SERIES, expected_groups=GROUPS,
        block_days=(1, 2), replicates=20, seed=7,
    )
    assert result.r2 == result.paired_mean == 0.0
    assert isclose(result.mean_spearman or 0.0, 1.0)
    assert not result.meets_statistical_gate

    reversed_ranks = tuple(
        replace(cell, predicted=-cell.actual) for cell in perfect_cells()
    )
    assert isclose(
        cross_section_diagnostics(
            reversed_ranks, SERIES, expected_groups=GROUPS,
            block_days=(1, 2), replicates=20, seed=7,
        ).mean_spearman or 0.0,
        -1.0,
    )

    constant = tuple(replace(cell, actual=5.0) for cell in cells)
    assert cross_section_diagnostics(
        constant, SERIES, expected_groups=GROUPS,
        block_days=(1, 2), replicates=20, seed=7,
    ).r2 is None


def test_same_day_groups_remain_distinct() -> None:
    cells = (
        *perfect_cells(),
        *(
            CrossSectionCell("2026-01-02", "g3", series, actual, 0.0)
            for series, actual in zip(SERIES, (-3.0, 0.0, 3.0), strict=True)
        ),
    )
    captured = []

    def interval(
        values: object, width: int, **_: object,
    ) -> tuple[float, float]:
        captured.append((width, values))
        return -1.0, -0.5

    with patch(
        "tools.universe_cross_section.circular_block_interval",
        side_effect=interval,
    ):
        result = cross_section_diagnostics(
            cells, SERIES, expected_groups=(*GROUPS, ("2026-01-02", "g3")),
            block_days=(1, 2), replicates=20, seed=7,
        )
    assert result.group_count == 3
    assert result.date_count == 2
    assert result.meets_statistical_gate
    expected = {
        "A": {"2026-01-02": (-1.0, 0.0), "2026-01-05": (-2.0,)},
        "B": {"2026-01-02": (0.0, 0.0), "2026-01-05": (0.0,)},
        "C": {"2026-01-02": (-1.0, 0.0), "2026-01-05": (-2.0,)},
    }
    assert captured == [(1, expected), (2, expected)]


def test_rejections() -> None:
    cells = perfect_cells()
    for invalid in (
        (),
        cells[:-1],
        (*cells, cells[0]),
        (replace(cells[0], day="2026-01-03"), *cells[1:]),
    ):
        raises(
            cross_section_diagnostics, invalid, SERIES,
            expected_groups=GROUPS,
            block_days=(1, 2), replicates=20, seed=7,
        )
    raises(
        cross_section_diagnostics, cells[:3], SERIES,
        expected_groups=GROUPS,
        block_days=(1,), replicates=20, seed=7,
    )
    for names in ((), ("A",), ("A", "A"), ("A", "B", 1)):
        raises(
            cross_section_diagnostics, cells, names,
            expected_groups=GROUPS,
            block_days=(1, 2), replicates=20, seed=7,
        )
    for groups in (
        (), (("2026-01-02", "g1"), ("2026-01-02", "g1")),
        (("invalid", "g1"),), (("2026-01-02", ""),),
    ):
        raises(
            cross_section_diagnostics, cells, SERIES,
            expected_groups=groups,
            block_days=(1, 2), replicates=20, seed=7,
        )
    for blocks in ((), (0,), (1, 1), (2, 1), (3,), (True,)):
        raises(
            cross_section_diagnostics, cells, SERIES,
            expected_groups=GROUPS,
            block_days=blocks, replicates=20, seed=7,
        )
    for replicates, seed in ((1, 7), (True, 7), (20, True)):
        raises(
            cross_section_diagnostics, cells, SERIES,
            expected_groups=GROUPS,
            block_days=(1, 2), replicates=replicates, seed=seed,
        )
    large = tuple(replace(cell, actual=1e308) for cell in cells)
    raises(
        cross_section_diagnostics, large, SERIES,
        expected_groups=GROUPS,
        block_days=(1, 2), replicates=20, seed=7,
    )
    tiny_truth = tuple(
        replace(cell, actual=cell.actual * 1e-160,
                predicted=cell.actual * 1e153)
        for cell in cells
    )
    raises(
        cross_section_diagnostics, tiny_truth, SERIES,
        expected_groups=GROUPS,
        block_days=(1, 2), replicates=20, seed=7,
    )
    days = tuple(
        str(date(2026, 1, 1) + timedelta(days=offset))
        for offset in range(19)
    )
    short = tuple(
        CrossSectionCell(day, f"g{offset}", series, actual, actual)
        for offset, day in enumerate(days)
        for series, actual in zip(SERIES, (-1.0, 0.0, 1.0), strict=True)
    )
    raises(
        cross_section_diagnostics, short, SERIES,
        expected_groups=tuple(
            (day, f"g{offset}") for offset, day in enumerate(days)
        ),
        block_days=(20,), replicates=20, seed=7,
    )
    for value in (
        ("", "g", "A", 0.0, 0.0),
        ("2026-01-02", "", "A", 0.0, 0.0),
        ("2026-01-02", "g", "", 0.0, 0.0),
        ("2026-01-02", "g", "A", True, 0.0),
        ("2026-01-02", "g", "A", 0.0, float("nan")),
    ):
        raises(CrossSectionCell, *value)


def main() -> None:
    test_literal_diagnostics()
    test_common_signal_has_no_stock_selection()
    test_rank_and_level_metrics_are_distinct()
    test_same_day_groups_remain_distinct()
    test_rejections()


if __name__ == "__main__":
    main()
