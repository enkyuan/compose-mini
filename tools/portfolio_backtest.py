"""Execute one shared development long-or-cash portfolio phase."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import math

from tools.backtest import (
    POLICY_COST_FIELDS, Costs, LongExecution, _finite, _name,
    _validated_costs, execute_long,
)

PHASES = frozenset(("fold-1", "calibration"))
ACTIONS = frozenset(("long_above", "always_up", "cash"))


@dataclass(frozen=True, slots=True)
class PortfolioOpportunity:
    phase: str
    series: str
    manifest_rank: int
    as_of: str
    entry_time: str
    target_time: str
    reference_price: float
    outcome_price: float
    prediction_mean: float | None
    prediction_pstdev: float | None


@dataclass(frozen=True, slots=True)
class _Position:
    opportunity: PortfolioOpportunity
    score: float | None
    cash_before: float
    execution: LongExecution


def _timestamp(value: object, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(f"{label} must be a canonical UTC timestamp") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    return value


def _log_ratio(numerator: float, denominator: float, label: str) -> float:
    try:
        ratio = numerator / denominator
        value = math.log(numerator) - math.log(denominator) \
            if ratio == 0.0 or not math.isfinite(ratio) else math.log(ratio)
        return _finite(value, label)
    except (OverflowError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} must be finite") from error


def _opportunity(
    value: object, phase: str, action: str,
) -> PortfolioOpportunity:
    if type(value) is not PortfolioOpportunity:
        raise ValueError("portfolio opportunity is invalid")
    (
        item_phase, raw_series, rank, raw_as_of, raw_entry, raw_target,
        raw_reference, raw_outcome, raw_mean, raw_deviation,
    ) = (
        value.phase, value.series, value.manifest_rank,
        value.as_of, value.entry_time, value.target_time,
        value.reference_price, value.outcome_price,
        value.prediction_mean, value.prediction_pstdev,
    )
    if type(item_phase) is not str or type(raw_series) is not str or \
       type(rank) is not int or rank < 1:
        raise ValueError("portfolio opportunity is invalid")
    series = _name(raw_series, "series")
    as_of = _timestamp(raw_as_of, "as_of")
    entry = _timestamp(raw_entry, "entry_time")
    target = _timestamp(raw_target, "target_time")
    reference = _finite(raw_reference, "reference_price")
    outcome = _finite(raw_outcome, "outcome_price")
    if item_phase != phase or not as_of < entry <= target or \
       min(reference, outcome) <= 0.0:
        raise ValueError("portfolio opportunity is invalid")
    mean = deviation = None
    if action == "long_above":
        mean = _finite(raw_mean, "prediction_mean")
        deviation = _finite(raw_deviation, "prediction_pstdev")
        if deviation < 0.0:
            raise ValueError("prediction_pstdev must be nonnegative")
    elif raw_mean is not None or raw_deviation is not None:
        raise ValueError("nonforecast actions require absent predictions")
    return PortfolioOpportunity(
        phase, series, rank,
        as_of, entry, target, reference, outcome, mean, deviation,
    )


def _validate_opportunities(
    values: Sequence[PortfolioOpportunity], phase: str, action: str,
) -> tuple[PortfolioOpportunity, ...]:
    normalized = tuple(_opportunity(value, phase, action) for value in values)
    series_ranks: dict[str, int] = {}
    rank_series: dict[int, str] = {}
    series_entries: set[tuple[str, str]] = set()
    for item in normalized:
        series_entry = item.series, item.entry_time
        if series_ranks.setdefault(item.series, item.manifest_rank) != \
                item.manifest_rank or \
           rank_series.setdefault(item.manifest_rank, item.series) != \
                item.series or series_entry in series_entries:
            raise ValueError("portfolio opportunities are inconsistent")
        series_entries.add(series_entry)
    return tuple(sorted(
        normalized,
        key=lambda item: (
            item.entry_time, item.manifest_rank, item.series,
            item.as_of, item.target_time,
        ),
    ))


def run_phase(
    opportunities: Sequence[PortfolioOpportunity], initial_cash: float,
    costs: Costs, *, phase: str, action: str, safety_bps: float = 0.0,
    disagreement_lambda: float = 0.0,
) -> dict[str, object]:
    """Run one validated phase without loading ledgers, prices, or labels."""
    if type(phase) is not str or phase not in PHASES or \
       type(action) is not str or action not in ACTIONS:
        raise ValueError("portfolio phase configuration is invalid")
    frozen_costs = _validated_costs(costs)
    cost_values = tuple(
        getattr(frozen_costs, field) for field in POLICY_COST_FIELDS
    )
    break_even = frozen_costs.break_even_log_return
    try:
        raw = tuple(opportunities)
    except TypeError as error:
        raise ValueError("portfolio opportunities must be iterable") from error
    if action == "cash" and raw or action != "cash" and not raw:
        raise ValueError("portfolio action has invalid opportunity coverage")
    start_cash = cash = _finite(initial_cash, "initial_cash")
    safety = _finite(safety_bps, "safety_bps")
    disagreement = _finite(disagreement_lambda, "disagreement_lambda")
    if cash <= 0.0 or min(safety, disagreement) < 0.0 or \
       action != "long_above" and (safety != 0.0 or disagreement != 0.0):
        raise ValueError("portfolio policy parameters are invalid")
    items = () if action == "cash" else \
        _validate_opportunities(raw, phase, action)
    execution_costs = Costs(*cost_values)
    threshold = _finite(
        break_even + safety / 10_000.0,
        "portfolio threshold",
    )
    grouped: dict[str, list[PortfolioOpportunity]] = defaultdict(list)
    for item in items:
        grouped[item.entry_time].append(item)

    rejections = {
        "below_threshold": 0, "not_selected": 0, "position_active": 0,
    }
    trades: list[dict[str, object]] = []
    realized = [{"event": "initial", "time": None, "equity": cash}]
    active: _Position | None = None
    turnover = 0.0

    def close(position: _Position) -> float:
        nonlocal turnover
        item, execution = position.opportunity, position.execution
        turnover = _finite(
            turnover + execution.entry_notional + execution.exit_notional,
            "portfolio turnover",
        )
        trades.append({
            "series": item.series, "manifest_rank": item.manifest_rank,
            "as_of": item.as_of, "entry_time": item.entry_time,
            "target_time": item.target_time, "score": position.score,
            "actual_log_return": _log_ratio(
                item.outcome_price, item.reference_price, "actual_log_return",
            ),
            "reference_price": item.reference_price,
            "outcome_price": item.outcome_price,
            "shares": execution.shares,
            "entry_execution_price": execution.entry_execution_price,
            "exit_execution_price": execution.exit_execution_price,
            "entry_notional": execution.entry_notional,
            "exit_notional": execution.exit_notional,
            "cash_before": position.cash_before,
            "cash_after": execution.cash_after,
            "net_log_growth": _log_ratio(
                execution.cash_after, position.cash_before, "net_log_growth",
            ),
        })
        realized.append({
            "event": "exit", "time": item.target_time,
            "equity": execution.cash_after,
        })
        return execution.cash_after

    for entry_time in sorted(grouped):
        group = grouped[entry_time]
        if active is not None and entry_time > active.opportunity.target_time:
            cash, active = close(active), None
        if active is not None:
            rejections["position_active"] += len(group)
            continue
        if action == "always_up":
            passing = tuple((None, item) for item in group)
        else:
            scored = tuple((
                _finite(
                    item.prediction_mean -
                    disagreement * item.prediction_pstdev,
                    "portfolio score",
                ),
                item,
            ) for item in group)
            passing = tuple(pair for pair in scored if pair[0] > threshold)
            rejections["below_threshold"] += len(scored) - len(passing)
        if not passing:
            continue
        score, selected = min(
            passing,
            key=lambda pair: (
                0.0 if pair[0] is None else -pair[0],
                pair[1].manifest_rank,
            ),
        )
        rejections["not_selected"] += len(passing) - 1
        active = _Position(
            selected, score, cash,
            execute_long(
                cash, selected.reference_price, selected.outcome_price,
                execution_costs,
            ),
        )
    if active is not None:
        cash = close(active)

    equities = tuple(item["equity"] for item in realized)
    peak, drawdown = start_cash, 0.0
    for equity in equities:
        peak = max(peak, equity)
        drawdown = max(drawdown, 1.0 - equity / peak)
    return {
        "schema": 1, "phase": phase, "action": action,
        "evidence_role": "hypothetical-development-execution",
        "risk_scope": "realized-exit-events-only", "cash_yield": 0.0,
        "costs": dict(zip(POLICY_COST_FIELDS, cost_values, strict=True)),
        "policy": {
            "safety_bps": safety, "disagreement_lambda": disagreement,
            "minimum_score": threshold if action == "long_above" else None,
        },
        "initial_equity": start_cash, "terminal_equity": cash,
        "terminal_log_growth": _log_ratio(
            cash, start_cash, "terminal_log_growth",
        ),
        "realized_exit_max_drawdown": drawdown,
        "gross_turnover": _finite(
            turnover / start_cash, "portfolio gross turnover",
        ),
        "completed_trades": len(trades),
        "coverage": {
            "opportunities": len(items), "entry_groups": len(grouped),
        },
        "rejections": rejections, "realized_equity": realized,
        "trades": trades,
    }
