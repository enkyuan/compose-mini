#!/usr/bin/env python3
"""Verify deterministic shared-portfolio execution on synthetic opportunities."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtest import Costs, execute_long
from tools.portfolio_backtest import PortfolioOpportunity, run_phase


class HostileText(str):
    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other

    def __ne__(self, other: object) -> bool:
        return False


class NumericSubclass(float):
    pass


def raises(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def opportunity(
    series: str = "AAA", rank: int = 1,
    as_of: str = "2026-01-02T14:30:00Z",
    entry: str = "2026-01-02T15:00:00Z",
    target: str = "2026-01-02T16:00:00Z",
    reference: float = 10.0, outcome: float = 11.0,
    mean: float | None = 0.1, deviation: float | None = 0.0,
) -> PortfolioOpportunity:
    return PortfolioOpportunity(
        "fold-1", series, rank, as_of, entry, target, reference, outcome,
        mean, deviation,
    )


def run(
    values: object, *, action: str = "long_above",
    costs: Costs = Costs(0.0, 0.0, 0.0), cash: float = 100.0,
    phase: str = "fold-1", safety: float = 0.0, disagreement: float = 0.0,
) -> dict[str, object]:
    return run_phase(
        values, cash, costs, phase=phase, action=action,
        safety_bps=safety, disagreement_lambda=disagreement,
    )  # type: ignore[arg-type]


def test_compounds_and_labels_realized_evidence() -> None:
    first = opportunity()
    second = opportunity(
        as_of="2026-01-02T16:00:01Z",
        entry="2026-01-02T16:30:00Z",
        target="2026-01-02T17:00:00Z",
    )
    report = run((first, second))
    assert report == run((second, first))
    assert report["terminal_equity"] == 121.0
    assert math.isclose(
        report["terminal_log_growth"], math.log(1.21), rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert report["gross_turnover"] == 4.41
    assert report["completed_trades"] == 2
    assert report["realized_exit_max_drawdown"] == 0.0
    assert report["evidence_role"] == "hypothetical-development-execution"
    assert report["risk_scope"] == "realized-exit-events-only"
    assert report["costs"] == {
        "spread_bps": 0.0, "slippage_bps": 0.0, "fee_bps": 0.0,
    }
    assert report["policy"] == {
        "safety_bps": 0.0, "disagreement_lambda": 0.0,
        "minimum_score": 0.0,
    }
    assert [item["equity"] for item in report["realized_equity"]] == [
        100.0, 110.0, 121.0,
    ]
    assert report["trades"][0]["actual_log_return"] == math.log(11.0 / 10.0)
    assert report["rejections"] == {
        "below_threshold": 0, "not_selected": 0, "position_active": 0,
    }
    text = json.dumps(report, sort_keys=True)
    for forbidden in ("projected", "daily", "weekly", "monthly", "bar_close"):
        assert forbidden not in text


def test_reuses_exact_cost_contract() -> None:
    costs = Costs(2.0, 3.0, 5.0)
    expected = execute_long(100.0, 50.0, 55.0, costs)
    trade = run(
        (opportunity(reference=50.0, outcome=55.0, mean=1.0),), costs=costs,
    )["trades"][0]
    for field in (
        "shares", "entry_execution_price", "exit_execution_price",
        "entry_notional", "exit_notional", "cash_after",
    ):
        assert trade[field] == getattr(expected, field)


def test_threshold_rank_order_and_overlap() -> None:
    costs, safety = Costs(1.0, 1.0, 0.0), 3.0
    threshold = costs.break_even_log_return + safety / 10_000.0
    abstained = run(
        (opportunity(mean=threshold),), costs=costs, safety=safety,
    )
    assert abstained["terminal_equity"] == 100.0
    assert abstained["rejections"]["below_threshold"] == 1

    lower = opportunity("AAA", 1)
    higher = opportunity("BBB", 2)
    selected = run((higher, lower))
    assert selected == run((lower, higher))
    assert selected["trades"][0]["series"] == "AAA"
    assert selected["rejections"]["not_selected"] == 1
    penalized = run((
        opportunity("AAA", 1, mean=0.3, deviation=0.2),
        opportunity("BBB", 2, mean=0.15, deviation=0.0),
    ), disagreement=1.0)
    assert penalized["trades"][0]["series"] == "BBB"

    at_exit = (
        opportunity(),
        opportunity(
            "BBB", 2, "2026-01-02T15:30:00Z",
            "2026-01-02T16:00:00Z", "2026-01-02T17:00:00Z",
        ),
        opportunity(
            "CCC", 3, "2026-01-02T16:00:00Z",
            "2026-01-02T16:30:00Z", "2026-01-02T17:30:00Z",
        ),
    )
    overlap = run(at_exit)
    assert overlap["completed_trades"] == 2
    assert overlap["rejections"]["position_active"] == 1


def test_actions_and_immutability() -> None:
    values = (
        opportunity("BBB", 2, mean=None, deviation=None),
        opportunity("AAA", 1, mean=None, deviation=None),
    )
    always = run(values, action="always_up")
    assert always["trades"][0]["series"] == "AAA"
    assert always["rejections"]["not_selected"] == 1
    cash = run((), action="cash")
    assert cash["terminal_equity"] == 100.0
    assert cash["completed_trades"] == 0
    raises(run, values, action="cash")
    raises(run, (opportunity(),), action="always_up")
    raises(run, values, action="always_up", safety=1.0)
    try:
        values[0].series = "changed"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("portfolio opportunity is mutable")


def test_extreme_finite_log_ratios() -> None:
    smallest, largest = sys.float_info.min, sys.float_info.max
    report = run((
        opportunity(
            reference=largest, outcome=smallest, mean=1.0,
        ),
    ), cash=largest)
    expected = math.log(smallest) - math.log(largest)
    trade = report["trades"][0]
    assert trade["actual_log_return"] == expected
    assert trade["net_log_growth"] == expected
    assert report["terminal_log_growth"] == expected
    assert report["terminal_equity"] == smallest
    assert report["gross_turnover"] == 1.0


def test_rejects_numeric_subclasses() -> None:
    base = opportunity()
    for value in (
        replace(base, reference_price=NumericSubclass(10.0)),
        replace(base, outcome_price=NumericSubclass(11.0)),
        replace(base, prediction_mean=NumericSubclass(0.1)),
        replace(base, prediction_pstdev=NumericSubclass(0.0)),
    ):
        raises(run, (value,))
    for arguments in (
        {"cash": NumericSubclass(100.0)},
        {"safety": NumericSubclass(0.0)},
        {"disagreement": NumericSubclass(0.0)},
    ):
        raises(run, (base,), **arguments)


def test_rejects_malformed_inputs() -> None:
    base = opportunity()
    invalid = (
        replace(base, phase="test"),
        replace(base, phase=HostileText("calibration")),
        replace(base, series=""),
        replace(base, series=HostileText("AAA")),
        replace(base, manifest_rank=0),
        replace(base, manifest_rank=True),  # type: ignore[arg-type]
        replace(base, as_of="2026-01-02T14:30:00+00:00"),
        replace(base, entry_time=base.as_of),
        replace(base, target_time="2026-01-02T14:00:00Z"),
        replace(base, reference_price=0.0),
        replace(base, reference_price=math.inf),
        replace(base, outcome_price=math.nan),
        replace(base, prediction_mean=None),
        replace(base, prediction_mean=math.inf),
        replace(base, prediction_pstdev=-1.0),
    )
    for value in invalid:
        raises(run, (value,))
    for values in (
        (base, base),
        (base, replace(base, target_time="2026-01-02T17:00:00Z")),
        (
            base,
            opportunity(
                entry="2026-01-02T17:00:00Z",
                target="2026-01-02T18:00:00Z", rank=2,
            ),
        ),
        (base, opportunity("BBB", 1)),
    ):
        raises(run, values)
    for values, kwargs in (
        ((base,), {"action": "short"}),
        ((base,), {"cash": 0.0}),
        ((base,), {"cash": math.nan}),
        ((base,), {"safety": -1.0}),
        ((base,), {"disagreement": math.inf}),
        ((base,), {"costs": object()}),
        ((base,), {"phase": "test"}),
        (
            (
                opportunity(
                    reference=sys.float_info.min, outcome=sys.float_info.max,
                ),
            ),
            {"cash": sys.float_info.min},
        ),
        (object(), {}),
    ):
        raises(run, values, **kwargs)


def main() -> None:
    test_compounds_and_labels_realized_evidence()
    test_reuses_exact_cost_contract()
    test_threshold_rank_order_and_overlap()
    test_actions_and_immutability()
    test_extreme_finite_log_ratios()
    test_rejects_numeric_subclasses()
    test_rejects_malformed_inputs()


if __name__ == "__main__":
    main()
