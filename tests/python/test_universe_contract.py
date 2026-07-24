#!/usr/bin/env python3
"""Verify common-calendar universe roles, blocks, and coverage."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.chronology import IndexRange
from tools.session_samples import SampleRows
from tools.universe_contract import (
    common_calendar, coverage, fixed_update_budget, pack_rows, universe_roles,
)


def raises(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_fixed_blocks() -> None:
    plan = common_calendar(5_505, 2, 0.1, 12)
    assert plan.folds == (
        (IndexRange(0, 3_293), IndexRange(3_305, 3_843)),
        (IndexRange(0, 3_843), IndexRange(3_855, 4_393)),
    )
    assert plan.holdout == (
        IndexRange(0, 4_393),
        IndexRange(4_405, 4_943),
        IndexRange(4_955, 5_505),
    )
    for arguments in ((True, 2, 0.1, 12), (20, 2, 1.0, 2), (20, 2, 0.2, -1)):
        raises(common_calendar, *arguments)


def test_fixed_update_budget() -> None:
    budgets = tuple(
        fixed_update_budget(samples, 128, 100)
        for samples in (34_992, 41_042, 47_092)
    )
    assert tuple(item.updates_per_checkpoint for item in budgets) == \
        (274, 321, 368)
    assert tuple(item.total_updates for item in budgets) == \
        (27_400, 32_100, 36_800)
    for arguments in ((0, 128, 100), (10, True, 100), (10, 128, 0)):
        raises(fixed_update_budget, *arguments)


def test_sparse_rows() -> None:
    blocks = common_calendar(20, 2, 0.2, 2).holdout
    opportunities = (0, 1, 2, 4, 5, 12, 13, 16, 17, 18, 19)
    rows = tuple(
        SampleRows(index, index + 1, index + 2, opportunity + 3)
        for index, opportunity in enumerate(opportunities)
    )
    packed = pack_rows(rows, blocks, 4, 2, 2)
    assert packed.counts == (5, 2, 4)
    assert packed.rows == rows
    assert pack_rows(rows, blocks[:2], 4, 2, 2).rows == rows[:7]
    raises(pack_rows, tuple(reversed(rows)), blocks, 4, 2, 2)
    raises(pack_rows, rows, (object(),), 4, 2, 2)
    raises(pack_rows, rows, tuple(reversed(blocks)), 4, 2, 2)
    raises(
        pack_rows, rows, (IndexRange(0, 10), IndexRange(9, 14)), 4, 2, 2,
    )
    raises(pack_rows, rows, blocks, 4, 3, 2)
    raises(
        pack_rows, (SampleRows(0, 1, 2, 0),), blocks, 4, 2, 2,
    )


def test_roles_and_coverage() -> None:
    names = tuple(f"S{index:02}" for index in range(55))
    roles = universe_roles(names)
    assert tuple(size for size, _ in roles.cohorts) == (11, 22, 33, 55)
    assert roles.cohorts[0][1] == names[:11]
    assert roles.transfer_training[-1] == (44, names[:44])
    assert roles.unseen == names[44:55]
    observed = coverage(roles.unseen, {
        name: int(name != "S49") for name in roles.unseen
    })
    assert not observed.complete
    assert observed.evaluable == tuple(
        name for name in roles.unseen if name != "S49"
    )
    assert observed.missing == ("S49",)
    raises(universe_roles, names[:-1])
    raises(universe_roles, (*names[:-1], names[0]))
    raises(coverage, roles.unseen, {name: 1 for name in names})
    raises(coverage, (), {})


def main() -> None:
    test_fixed_blocks()
    test_fixed_update_budget()
    test_sparse_rows()
    test_roles_and_coverage()
    print("universe contract tests passed")


if __name__ == "__main__":
    main()
