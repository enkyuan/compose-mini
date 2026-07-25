"""Evaluate and select the frozen temporal-context diagnostic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log
from statistics import fmean

from tools.files import file_sha256, freeze_inputs, verify_frozen
from tools.context_diagnostic_contract import (
    CONTROL_MODELS, HISTORY_LENGTHS, MODELS, PRIMARY_MODEL, SEEDS,
    TARGET_PHASES, ContextPhase, ContextPredictionEvidence, ContextReceipt,
    context_phase_sha256, expected_context_predictions,
)
from tools.panel_contract import (
    FileBinding, _directory_identity, _exact_json, _regular_inputs,
    _verify_identities, read_canonical_json,
)
from tools.run_context_diagnostic import (
    RunClaim, context_access_value, phase_artifacts,
    publish_context_outcome, validate_context_ledgers,
)
from tools.universe_scaling import (
    BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_REPLICATES, ForecastPoint,
    paired_comparison, stock_macro_metrics,
)
from tools.universe_scaling_contract import timestamp_grid_sha256

BOOTSTRAP_SEED = 20_260_725


@dataclass(frozen=True, slots=True)
class ContextTruthRow:
    """Bind one prediction cell to its development-only market outcome."""

    as_of: str
    entry_time: str
    target_time: str
    reference_price: float
    outcome_price: float

    def __post_init__(self) -> None:
        times = (self.as_of, self.entry_time, self.target_time)
        prices = (self.reference_price, self.outcome_price)
        if any(not isinstance(value, str) or not value for value in times) or \
           not self.as_of < self.entry_time <= self.target_time or any(
               type(value) not in (int, float) or not isfinite(value)
               for value in prices
           ) or min(prices) <= 0:
            raise ValueError("context truth row is invalid")
        ForecastPoint(
            self.target_time, self.actual_return, 0.0,
            self.reference_price, self.outcome_price,
        )

    @property
    def actual_return(self) -> float:
        """Derive the sole return truth from its executable prices."""
        return log(self.outcome_price / self.reference_price)


def _truth(
    phase: ContextPhase,
    values: Mapping[str, Sequence[ContextTruthRow]],
) -> dict[str, tuple[ContextTruthRow, ...]]:
    names = tuple(series for series, _, _ in phase.evaluation_rows)
    if not isinstance(values, Mapping) or tuple(values) != names:
        raise ValueError("context truth must match the evaluation order")
    result = {}
    for series, count, expected_grid in phase.evaluation_rows:
        source = values[series]
        if not isinstance(source, Sequence) or isinstance(source, str):
            raise ValueError(f"{series} context truth rows are invalid")
        rows = tuple(source)
        if len(rows) != count or any(
            not isinstance(row, ContextTruthRow) for row in rows
        ):
            raise ValueError(f"{series} context truth grid changed")
        grid = tuple(
            (row.as_of, row.entry_time, row.target_time) for row in rows
        )
        if timestamp_grid_sha256(grid) != expected_grid:
            raise ValueError(f"{series} context truth grid changed")
        result[series] = rows
    return result


def _ensembles(
    master: Sequence[str], phase: ContextPhase,
    evidence: Sequence[ContextPredictionEvidence],
) -> dict[tuple[str, int, str], tuple[float, ...]]:
    expected = expected_context_predictions(master, phase)
    records = tuple(evidence)
    if len(records) != len(expected) or any(
        not isinstance(record, ContextPredictionEvidence) or
        record.prediction != prediction
        for record, prediction in zip(records, expected, strict=True)
    ):
        raise ValueError("context prediction evidence changed")
    grouped: dict[
        tuple[str, int, str], list[tuple[int | None, tuple[float, ...]]]
    ] = {}
    for record in records:
        fit = record.prediction.fit
        grouped.setdefault(
            (fit.model, fit.history, record.prediction.series), [],
        ).append((fit.seed, record.values))

    result = {}
    evaluation = tuple(series for series, _, _ in phase.evaluation_rows)
    for model in MODELS:
        seeds = (None,) if model == CONTROL_MODELS[0] else SEEDS
        for history in HISTORY_LENGTHS:
            for series in evaluation:
                values = grouped.get((model, history, series), [])
                if tuple(seed for seed, _ in values) != seeds:
                    raise ValueError("context prediction seed closure changed")
                vectors = tuple(predictions for _, predictions in values)
                result[model, history, series] = (
                    vectors[0] if seeds == (None,) else
                    tuple(fmean(column) for column in zip(
                        *vectors, strict=True,
                    ))
                )
    if len(result) != len(grouped):
        raise ValueError("context prediction family changed")
    return result


def _points(
    truth: Mapping[str, Sequence[ContextTruthRow]],
    predictions: Mapping[tuple[str, int, str], Sequence[float]],
    model: str, history: int,
) -> dict[str, tuple[ForecastPoint, ...]]:
    return {
        series: tuple(
            ForecastPoint(
                row.target_time, row.actual_return, predicted,
                row.reference_price, row.outcome_price,
            )
            for row, predicted in zip(
                rows, predictions[model, history, series], strict=True,
            )
        )
        for series, rows in truth.items()
    }


def _common_dates(
    truth: Mapping[str, Sequence[ContextTruthRow]],
) -> tuple[str, ...]:
    dates = tuple(sorted(set.intersection(*(
        {row.target_time[:10] for row in rows}
        for rows in truth.values()
    ))))
    if not dates:
        raise ValueError("context truth has no common target dates")
    return dates


def evaluate_context_phase(
    master: Sequence[str], phase: ContextPhase,
    evidence: Sequence[ContextPredictionEvidence],
    truth: Mapping[str, Sequence[ContextTruthRow]],
) -> dict[str, object]:
    """Compute the predeclared development metrics for one received phase."""
    rows = _truth(phase, truth)
    predictions = _ensembles(master, phase, evidence)
    forecasts = {
        (model, history): _points(
            rows, predictions, model, history,
        )
        for model in MODELS for history in HISTORY_LENGTHS
    }
    metrics = {
        model: {
            str(history): stock_macro_metrics(forecasts[model, history])
            for history in HISTORY_LENGTHS
        }
        for model in MODELS
    }
    expected_dates = _common_dates(rows)
    primary = {}
    for history in HISTORY_LENGTHS[1:]:
        comparison = paired_comparison(
            forecasts[PRIMARY_MODEL, history],
            forecasts[PRIMARY_MODEL, HISTORY_LENGTHS[0]],
            block_days=BOOTSTRAP_BLOCK_DAYS,
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        if tuple(comparison["common_dates"]) != expected_dates:
            raise ValueError("context bootstrap date grid changed")
        primary[str(history)] = comparison
    return {
        "descriptive_metrics": metrics,
        "evidence_role": "development-diagnostic-not-forward-clean",
        "phase": phase.phase,
        "phase_sha256": context_phase_sha256(phase),
        "primary": primary,
        "schema": 1,
    }


def _interval(value: object) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str) or \
       len(value) != 2 or any(
           type(item) not in (int, float) for item in value
       ):
        raise ValueError("context bootstrap interval is invalid")
    try:
        bounds = tuple(float(item) for item in value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("context bootstrap interval is invalid") from error
    if any(not isfinite(item) for item in bounds) or bounds[0] > bounds[1]:
        raise ValueError("context bootstrap interval is invalid")
    return bounds


def _primary_intervals(
    value: object, phase: str, phase_sha256: str,
) -> dict[str, dict[str, tuple[float, float]]]:
    if not isinstance(value, Mapping) or set(value) != {
        "descriptive_metrics", "evidence_role", "phase",
        "phase_sha256", "primary", "schema",
    } or type(value.get("schema")) is not int or \
       value.get("schema") != 1 or value.get("phase") != phase or \
       value.get("phase_sha256") != phase_sha256 or \
       value.get("evidence_role") != \
            "development-diagnostic-not-forward-clean":
        raise ValueError("context evaluation phase changed")
    try:
        primary = value["primary"]
        histories = tuple(map(str, HISTORY_LENGTHS[1:]))
        if not isinstance(primary, Mapping) or tuple(primary) != histories:
            raise ValueError("context primary family changed")
        blocks = tuple(map(str, BOOTSTRAP_BLOCK_DAYS))
        result = {}
        for history in histories:
            intervals = primary[history]["intervals"]
            if not isinstance(intervals, Mapping) or \
               set(intervals) != set(blocks):
                raise ValueError("context bootstrap family changed")
            result[history] = {
                block: _interval(intervals[block]) for block in blocks
            }
        return result
    except (KeyError, TypeError) as error:
        raise ValueError("context primary intervals are missing") from error


def _select_context_history(
    phases: Sequence[ContextPhase],
    evaluations: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Select 34, then 68, only when every frozen lower bound is positive."""
    bound = tuple(phases)
    if len(bound) != len(TARGET_PHASES) or any(
        not isinstance(phase, ContextPhase) for phase in bound
    ) or tuple(phase.phase for phase in bound) != TARGET_PHASES or \
       not isinstance(evaluations, Mapping) or \
       tuple(evaluations) != TARGET_PHASES:
        raise ValueError("context evaluations must follow phase order")
    master = (
        *(series for series, _ in bound[0].training_rows),
        *(series for series, _, _ in bound[0].evaluation_rows),
    )
    for phase in bound:
        expected_context_predictions(master, phase)
    phase_hashes = {
        phase.phase: context_phase_sha256(phase) for phase in bound
    }
    intervals = {
        phase: _primary_intervals(
            evaluations[phase], phase, phase_hashes[phase],
        )
        for phase in TARGET_PHASES
    }
    qualifies = {
        str(history): min(
            intervals[phase][str(history)][str(block)][0]
            for phase in TARGET_PHASES for block in BOOTSTRAP_BLOCK_DAYS
        ) > 0
        for history in HISTORY_LENGTHS[1:]
    }
    selected = next((
        history for history in HISTORY_LENGTHS[1:]
        if qualifies[str(history)]
    ), HISTORY_LENGTHS[0])
    return {
        "qualifies": qualifies,
        "selected_history": selected,
    }


def _binding_value(binding: FileBinding) -> dict[str, str]:
    return {"path": binding.path, "sha256": binding.sha256}


def finalize_context_history(
    claim: RunClaim, master: Sequence[str],
    phases: Sequence[ContextPhase],
) -> dict[str, object]:
    """Publish one terminal decision from the live phase evidence closure."""
    bound = tuple(phases)
    if not isinstance(claim, RunClaim) or \
       len(bound) != len(TARGET_PHASES) or any(
           not isinstance(phase, ContextPhase) for phase in bound
       ) or tuple(phase.phase for phase in bound) != TARGET_PHASES or \
       tuple(claim.completed) != TARGET_PHASES:
        raise ValueError("context phase evidence is incomplete")
    provenance = {
        (
            evidence.source_failure_sha256,
            evidence.config_sha256,
            evidence.source_tree_sha256,
        )
        for evidence in claim.completed.values()
    }
    if len(provenance) != 1:
        raise ValueError("context phase provenance changed")
    source_failure, config, source_tree = next(iter(provenance))
    sources = []
    bindings = []
    identities = []
    for phase in bound:
        try:
            evidence = claim.completed[phase.phase]
        except (AttributeError, KeyError) as error:
            raise ValueError("context phase evidence is incomplete") \
                from error
        artifacts = phase_artifacts(claim.root, claim.attempt, phase)
        paths = tuple(artifacts)
        if evidence.phase != phase.phase or len(evidence.bindings) != \
                len(paths) or len(evidence.identities) != len(paths):
            raise ValueError("context phase evidence changed")
        sources.extend(paths)
        bindings.extend(evidence.bindings)
        identities.extend(evidence.identities)

    paths = (claim.attempt, *sources)
    expected_bindings = (claim.attempt_binding, *bindings)
    expected_identities = (claim.attempt_identity, *identities)
    observed = _regular_inputs(paths)
    if tuple(identity for _, identity in observed) != expected_identities:
        raise ValueError("context phase evidence identity changed")

    with freeze_inputs(paths) as frozen:
        by_path = dict(zip(paths, frozen, strict=True))
        for path, binding in zip(paths, expected_bindings, strict=True):
            try:
                logical = path.relative_to(claim.root).as_posix()
            except ValueError as error:
                raise ValueError("context phase evidence escaped the root") \
                    from error
            if binding.path != logical or \
               binding.sha256 != by_path[path].sha256:
                raise ValueError("context phase evidence binding changed")

        evaluations = {}
        inputs = []
        for phase in bound:
            evidence = claim.completed[phase.phase]
            artifacts = phase_artifacts(
                claim.root, claim.attempt, phase,
            )
            fit, prediction, receipt, access, evaluation = tuple(artifacts)
            fit_binding, prediction_binding, receipt_binding, \
                access_binding, evaluation_binding = evidence.bindings
            parsed_receipt = ContextReceipt.parse(
                read_canonical_json(by_path[receipt].snapshot),
            )
            parsed_receipt.validate(
                phase, claim.attempt_binding, fit_binding,
                prediction_binding, evidence.source_tree_sha256,
                claim.directory_identity,
            )
            validate_context_ledgers(
                master, phase, by_path[fit].snapshot,
                by_path[prediction].snapshot,
                evidence.source_failure_sha256,
                evidence.config_sha256,
            )
            if not _exact_json(
                read_canonical_json(by_path[access].snapshot),
                context_access_value(
                    claim.attempt_binding, receipt_binding, phase,
                ),
            ):
                raise ValueError("context truth access changed")
            evaluations[phase.phase] = read_canonical_json(
                by_path[evaluation].snapshot,
            )
            inputs.append({
                "access": _binding_value(access_binding),
                "evaluation": _binding_value(evaluation_binding),
                "fits": _binding_value(fit_binding),
                "phase": phase.phase,
                "predictions": _binding_value(prediction_binding),
                "receipt": _binding_value(receipt_binding),
            })

        decision = _select_context_history(bound, evaluations)
        output = {
            "decision": decision,
            "evidence_role": "development-diagnostic-not-forward-clean",
            "inputs": {
                "attempt": _binding_value(claim.attempt_binding),
                "phases": inputs,
            },
            "integrity": {
                "config_sha256": config,
                "source_failure_sha256": source_failure,
                "source_tree_sha256": source_tree,
            },
            "schema": 1,
        }

        if _directory_identity(claim.path) != claim.directory_identity:
            raise ValueError("context run directory changed")
        _verify_identities(observed)
        verify_frozen(frozen)

    def verify() -> None:
        _verify_identities(observed)
        if any(
            file_sha256(path) != binding.sha256
            for path, binding in zip(
                paths, expected_bindings, strict=True,
            )
        ) or _directory_identity(claim.path) != claim.directory_identity:
            raise ValueError("context phase evidence changed")
        _verify_identities(observed)

    publish_context_outcome(claim, output, verify)
    return output
