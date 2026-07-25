#!/usr/bin/env python3
"""Verify the predeclared shared-portfolio policy selection."""

from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.select_portfolio_policy import (
    PortfolioPolicy, PortfolioTrial, registered_portfolio_policies,
    select_portfolio_trial,
)


def raises(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def trials(
    growth: float = 0.0, turnover: float = 1.0,
) -> tuple[PortfolioTrial, ...]:
    return tuple(
        PortfolioTrial(
            policy,
            0.0 if policy.action == "cash" else growth,
            0.0 if policy.action == "cash" else turnover,
        )
        for policy in registered_portfolio_policies()
    )


def tied_forecasts(*policies: PortfolioPolicy) -> tuple[PortfolioTrial, ...]:
    return tuple(
        replace(
            trial,
            terminal_log_growth=1.0
            if trial.policy in policies else -1.0,
        ) if trial.policy.action == "long_above" else trial
        for trial in trials()
    )


def bounded_repeat(value: PortfolioTrial) -> Iterator[PortfolioTrial]:
    yield from (value for _ in range(14))
    raise AssertionError("portfolio selector consumed a fifteenth trial")


def test_registered_grid() -> None:
    policies = registered_portfolio_policies()
    assert policies == tuple(
        PortfolioPolicy("long_above", safety, disagreement)
        for safety in (0.0, 3.0, 6.0, 10.0)
        for disagreement in (0.0, 0.5, 1.0)
    ) + (PortfolioPolicy("cash", 0.0, 0.0),)
    assert len(policies) == len(set(policies)) == 13
    try:
        policies[0].safety_bps = 1.0  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("portfolio policy is mutable")


def test_selection_precedence() -> None:
    values = trials()
    best_growth = replace(values[0], terminal_log_growth=0.1)
    assert select_portfolio_trial((best_growth, *values[1:])) is best_growth

    turnover_values = trials(1.0)
    lower_turnover = replace(turnover_values[1], gross_turnover=0.5)
    assert select_portfolio_trial(
        (turnover_values[0], lower_turnover, *turnover_values[2:])
    ) is lower_turnover

    assert select_portfolio_trial(trials(0.0, 0.0)).policy.action == "cash"

    higher_safety = PortfolioPolicy("long_above", 10.0, 1.0)
    lower_safety = PortfolioPolicy("long_above", 6.0, 0.0)
    safety_tie = tied_forecasts(higher_safety, lower_safety)
    assert select_portfolio_trial(safety_tie).policy == higher_safety
    assert select_portfolio_trial(reversed(safety_tie)).policy == higher_safety

    same_safety = tied_forecasts(
        higher_safety, PortfolioPolicy("long_above", 10.0, 0.0),
    )
    selected = select_portfolio_trial(same_safety)
    assert selected.policy == PortfolioPolicy("long_above", 10.0, 0.0)
    assert select_portfolio_trial(
        reversed(same_safety)
    ).policy == selected.policy
    try:
        selected.gross_turnover = 0.0  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("portfolio trial is mutable")


def test_selection_fail_closed() -> None:
    values = trials()
    duplicate = (values[0], *values[:-1])
    unregistered = replace(
        values[0], policy=PortfolioPolicy("long_above", 1.0, 0.0),
    )
    malformed = replace(
        values[0], policy=PortfolioPolicy("cash", 0, 0),  # type: ignore[arg-type]
    )
    cash = values[-1]
    for invalid in (
        values[:-1], values + (values[0],), duplicate,
        (unregistered, *values[1:]),
        (malformed, *values[1:]),
        (replace(values[0], terminal_log_growth=math.nan), *values[1:]),
        (replace(values[0], gross_turnover=-1.0), *values[1:]),
        (replace(values[0], gross_turnover=math.inf), *values[1:]),
        (*values[:-1], replace(cash, terminal_log_growth=0.01)),
        (*values[:-1], replace(cash, gross_turnover=0.01)),
        (
            replace(
                values[0],
                terminal_log_growth=type("Number", (float,), {})(0.0),
            ),
            *values[1:],
        ),
        (object(), *values[1:]),
        bounded_repeat(values[0]),
    ):
        raises(select_portfolio_trial, invalid)
    assert select_portfolio_trial(
        value for value in values
    ).policy.action == "cash"


def main() -> None:
    test_registered_grid()
    test_selection_precedence()
    test_selection_fail_closed()


if __name__ == "__main__":
    main()
