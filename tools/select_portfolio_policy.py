"""Register and select the shared development-portfolio policy."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

from tools.backtest import (
    POLICY_SAFETY_GRID, SEEDED_DISAGREEMENT_GRID, _finite, select_trial,
)


@dataclass(frozen=True, slots=True)
class PortfolioPolicy:
    action: str
    safety_bps: float
    disagreement_lambda: float


@dataclass(frozen=True, slots=True)
class PortfolioTrial:
    policy: PortfolioPolicy
    terminal_log_growth: float
    gross_turnover: float


_POLICIES = tuple(
    PortfolioPolicy("long_above", safety, disagreement)
    for safety in POLICY_SAFETY_GRID
    for disagreement in SEEDED_DISAGREEMENT_GRID
) + (PortfolioPolicy("cash", 0.0, 0.0),)


def registered_portfolio_policies() -> tuple[PortfolioPolicy, ...]:
    """Return the immutable policy grid registered before result access."""
    return _POLICIES


def _validate_trial(value: object) -> PortfolioTrial:
    if type(value) is not PortfolioTrial or \
       type(value.policy) is not PortfolioPolicy or \
       type(value.policy.action) is not str or \
       type(value.policy.safety_bps) is not float or \
       type(value.policy.disagreement_lambda) is not float or \
       value.policy not in _POLICIES or \
       type(value.terminal_log_growth) not in (int, float) or \
       type(value.gross_turnover) not in (int, float):
        raise ValueError("portfolio trial policy is invalid")
    growth = _finite(
        value.terminal_log_growth, "portfolio terminal log growth",
    )
    turnover = _finite(value.gross_turnover, "portfolio gross turnover")
    if turnover < 0.0:
        raise ValueError("portfolio gross turnover must be nonnegative")
    if value.policy.action == "cash" and (growth != 0.0 or turnover != 0.0):
        raise ValueError("cash trial must have zero growth and turnover")
    return value


def select_portfolio_trial(
    values: Iterable[PortfolioTrial],
) -> PortfolioTrial:
    """Select the exact grid's growth-first, cost-aware winning trial."""
    try:
        trials = tuple(
            _validate_trial(value)
            for value in islice(iter(values), len(_POLICIES) + 1)
        )
    except TypeError as error:
        raise ValueError("portfolio trials are invalid") from error
    policies = tuple(trial.policy for trial in trials)
    if len(trials) != len(_POLICIES) or \
       len(set(policies)) != len(policies) or set(policies) != set(_POLICIES):
        raise ValueError("portfolio trial grid is incomplete")
    selected = select_trial(tuple({
        "objective": trial.terminal_log_growth,
        "mean_gross_turnover": trial.gross_turnover,
        "safety_bps":
            None if trial.policy.action == "cash"
            else trial.policy.safety_bps,
        "disagreement_lambda":
            None if trial.policy.action == "cash"
            else trial.policy.disagreement_lambda,
        "trial": trial,
    } for trial in trials))["trial"]
    if type(selected) is not PortfolioTrial:
        raise ValueError("portfolio trial selection is invalid")
    return selected
