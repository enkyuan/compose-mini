"""Score authenticated SPY-residual predictions without trading claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from datetime import date
from math import fsum, isfinite, sqrt
from statistics import fmean

from tools.context_diagnostic_contract import (
    ContextPhase, context_phase_sha256, context_phase_value,
)
from tools.panel_contract import (
    FileBinding, _exact_json, _integer, _object, _sha256, _string,
)
from tools.relative_context_contract import (
    EVIDENCE_ROLE, MODELS, PAIRED_COMPARISONS, PHASE_BUDGETS,
    RESIDUAL_CONFIG, SEEDS, SPY_RESIDUAL_TARGET, ResidualPhaseInput,
    ResidualPredictionEvidence, ResidualTruthRow,
    expected_residual_predictions, expected_residual_protocol,
    residual_fit_provenance_id, residual_phase_sha256,
)
from tools.universe_cross_section import (
    CROSS_SECTION_SEED, _center, _spearman,
)
from tools.universe_scaling import (
    BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_REPLICATES, circular_block_interval,
)
from tools.universe_scaling_contract import timestamp_grid_sha256

BOOTSTRAP_SEED = CROSS_SECTION_SEED
NEURAL_MODELS = MODELS[1:]
TERMINAL_SERIES = (
    "KRYS", "TGT", "STM", "SSNC", "NWL", "AAON",
    "GEV", "SWKS", "BMRN", "ACI", "HUN",
)
# The bound source attempt fixes each pooled/common/date evaluation shape.
TERMINAL_PHASE_SHAPES = {
    "fold-1": {
        "dates": 38, "observations": 5_848, "stocks": len(TERMINAL_SERIES),
        "timestamps": 468,
    },
    "calibration": {
        "dates": 36, "observations": 5_827, "stocks": len(TERMINAL_SERIES),
        "timestamps": 447,
    },
}


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _square_sum(values: Sequence[float], label: str) -> float:
    try:
        result = fsum(value * value for value in values)
    except OverflowError:
        raise ValueError(f"{label} is non-finite") from None
    if not isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def _day(timestamp: str) -> str:
    try:
        value = str(date.fromisoformat(timestamp[:10]))
    except (TypeError, ValueError):
        raise ValueError("residual target must begin with an ISO date") \
            from None
    if not timestamp.startswith(value):
        raise ValueError("residual target must begin with an ISO date")
    return value


def _binding_value(binding: FileBinding) -> dict[str, str]:
    return {"path": binding.path, "sha256": binding.sha256}


def _truth(
    source: ContextPhase, phase: ResidualPhaseInput,
    values: Mapping[str, Sequence[ResidualTruthRow]],
) -> tuple[
    dict[str, tuple[ResidualTruthRow, ...]],
    tuple[tuple[str, str, str], ...],
]:
    if ResidualPhaseInput.parse(asdict(phase), source) != phase:
        raise ValueError("residual phase changed")
    names = tuple(series for series, _, _ in source.evaluation_rows)
    if not isinstance(values, Mapping) or tuple(values) != names:
        raise ValueError("residual truth must match the evaluation order")
    result, grids = {}, {}
    for series, count, expected_grid in source.evaluation_rows:
        rows = tuple(values[series]) if isinstance(
            values[series], Sequence,
        ) and not isinstance(values[series], (str, bytes)) else ()
        grid = tuple((row.as_of, row.entry, row.target) for row in rows
                     if isinstance(row, ResidualTruthRow))
        if len(rows) != count or len(grid) != count or \
           timestamp_grid_sha256(grid) != expected_grid:
            raise ValueError(f"{series} residual truth grid changed")
        targets = tuple(target for _, _, target in grid)
        if len(set(grid)) != len(grid) or len(set(targets)) != len(targets) or \
           any(left >= right for left, right in zip(targets, targets[1:])):
            raise ValueError(f"{series} residual target grid changed")
        tuple(map(_day, targets))
        result[series] = rows
        grids[series] = grid
    if not grids:
        raise ValueError("residual truth is empty")
    shared = set.intersection(*(set(grid) for grid in grids.values()))
    common = tuple(row for row in next(iter(grids.values())) if row in shared)
    if not common or any(
        tuple(row for row in grid if row in shared) != common
        for grid in grids.values()
    ):
        raise ValueError("residual common grid changed")
    return result, common


def _population_std(values: Sequence[float]) -> float:
    try:
        mean = fmean(values)
        result = sqrt(fmean((value - mean) ** 2 for value in values))
    except OverflowError:
        raise ValueError("seed dispersion is non-finite") from None
    if not isfinite(result):
        raise ValueError("seed dispersion is non-finite")
    return result


def _predictions(
    master: Sequence[str], source: ContextPhase, phase: ResidualPhaseInput,
    evidence: Sequence[ResidualPredictionEvidence],
) -> tuple[
    dict[str, dict[str, tuple[float, ...]]],
    dict[str, dict[str, tuple[float, ...]]],
]:
    expected, records = expected_residual_predictions(master, source), \
        tuple(evidence)
    if len(records) != len(expected):
        raise ValueError("residual prediction evidence changed")
    grouped: dict[
        tuple[str, str], list[tuple[int | None, tuple[float, ...]]]
    ] = {}
    states = {}
    for record, prediction in zip(records, expected, strict=True):
        if not isinstance(record, ResidualPredictionEvidence) or \
           record.prediction != prediction or \
           type(record.values) is not tuple or \
           len(record.values) != prediction.prediction_count:
            raise ValueError("residual prediction evidence changed")
        values = tuple(
            _finite(value, "residual prediction") for value in record.values
        )
        identity = (
            _sha256(record.fit_provenance_id, "residual fit provenance"),
            _sha256(record.state_fingerprint, "residual state fingerprint"),
        )
        if identity[0] != residual_fit_provenance_id(
            prediction.fit, source, phase, master,
        ):
            raise ValueError("residual fit provenance changed")
        previous = states.setdefault(prediction.fit, identity)
        if previous != identity:
            raise ValueError("residual fitted state changed across stocks")
        grouped.setdefault(
            (prediction.fit.model, prediction.series), [],
        ).append((prediction.fit.seed, values))

    names = tuple(series for series, _, _ in source.evaluation_rows)
    ensembles, dispersion = {}, {}
    for model in MODELS:
        seeds = (None,) if model == MODELS[0] else SEEDS
        ensembles[model] = {}
        model_dispersion = {}
        for series in names:
            members = grouped.get((model, series), ())
            if tuple(seed for seed, _ in members) != seeds:
                raise ValueError("residual prediction seed closure changed")
            vectors = tuple(values for _, values in members)
            ensembles[model][series] = (
                vectors[0] if len(vectors) == 1 else
                tuple(fmean(column) for column in zip(
                    *vectors, strict=True,
                ))
            )
            if len(vectors) > 1:
                model_dispersion[series] = tuple(
                    _population_std(column) for column in zip(
                        *vectors, strict=True,
                    )
                )
        if model_dispersion:
            dispersion[model] = model_dispersion
    if set(grouped) != {
        (model, series) for model in MODELS for series in names
    } or set(dispersion) != set(NEURAL_MODELS):
        raise ValueError("residual prediction family changed")
    return ensembles, dispersion


def _pooled_r2(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, float]:
    actual = tuple(
        row.value for rows in truth.values() for row in rows
    )
    denominator = _square_sum(actual, "raw residual denominator")
    if denominator == 0.0:
        raise ValueError("raw residual denominator is zero")
    result = {}
    for model in MODELS:
        predicted = tuple(
            value for series in truth for value in predictions[model][series]
        )
        error = tuple(
            observed - forecast for observed, forecast in zip(
                actual, predicted, strict=True,
            )
        )
        result[model] = 1.0 - _square_sum(
            error, f"{model} raw residual error",
        ) / denominator
    if not all(map(isfinite, result.values())):
        raise ValueError("raw residual R-squared is non-finite")
    return result


def _paired_metrics(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    candidate: str, reference: str,
) -> dict[str, object]:
    forecast = {
        **predictions,
        "zero": {series: (0.0,) * len(rows)
                 for series, rows in truth.items()},
    }
    daily = {}
    for series, rows in truth.items():
        by_day: dict[str, list[float]] = {}
        for row, left, right in zip(
            rows, forecast[candidate][series], forecast[reference][series],
            strict=True,
        ):
            by_day.setdefault(_day(row.target), []).append(
                _finite(
                    abs(row.value - right) - abs(row.value - left),
                    "paired residual gain",
                )
            )
        daily[series] = {
            day: tuple(values) for day, values in by_day.items()
        }
    dates = tuple(sorted(set.intersection(*(
        set(values) for values in daily.values()
    ))))
    if not dates:
        raise ValueError("residual gains have no common target dates")
    per_stock = {
        series: fmean(
            value for day in dates for value in daily[series][day]
        )
        for series in truth
    }
    intervals = {
        str(width): circular_block_interval(
            daily, width, replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        for width in BOOTSTRAP_BLOCK_DAYS
    }
    return {
        "candidate": candidate,
        "date_count": len(dates),
        "intervals": intervals,
        "losses": sum(value < 0.0 for value in per_stock.values()),
        "mean_gain": fmean(per_stock.values()),
        "per_stock_mean_gain": per_stock,
        "reference": reference,
        "ties": sum(value == 0.0 for value in per_stock.values()),
        "wins": sum(value > 0.0 for value in per_stock.values()),
    }


def _secondary(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    common: Sequence[tuple[str, str, str]],
) -> dict[str, object]:
    names, groups = tuple(truth), tuple(common)
    indices = {
        series: {
            (row.as_of, row.entry, row.target): index
            for index, row in enumerate(truth[series])
        }
        for series in names
    }
    actual = tuple(
        tuple(truth[series][indices[series][group]].value for series in names)
        for group in groups
    )
    centered_actual = tuple(map(_center, actual))
    denominator = _square_sum(
        tuple(value for column in centered_actual for value in column),
        "centered residual denominator",
    )
    if denominator == 0.0:
        raise ValueError("centered residual denominator is zero")

    r2, rank = {}, {}
    for model in MODELS:
        forecast = tuple(
            tuple(
                predictions[model][series][indices[series][group]]
                for series in names
            )
            for group in groups
        )
        centered_forecast = tuple(map(_center, forecast))
        errors = tuple(
            observed - predicted
            for left, right in zip(
                centered_actual, centered_forecast, strict=True,
            )
            for observed, predicted in zip(left, right, strict=True)
        )
        r2[model] = 1.0 - _square_sum(
            errors, f"{model} centered residual error",
        ) / denominator
        correlations = tuple(
            max(-1.0, min(1.0, value))
            for left, right in zip(actual, forecast, strict=True)
            if (value := _spearman(left, right)) is not None
        )
        if not correlations:
            raise ValueError(f"{model} has no valid RankIC timestamps")
        rank[model] = {
            "excluded_timestamp_count": len(groups) - len(correlations),
            "mean": fmean(correlations),
            "valid_timestamp_count": len(correlations),
        }
    if not all(map(isfinite, (*r2.values(), *(
        value["mean"] for value in rank.values()
    )))):
        raise ValueError("residual cross-sectional metrics are non-finite")
    return {
        "centered_cross_sectional_r2": r2,
        "spearman_rank_ic": rank,
    }


def _seed_dispersion(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    common: Sequence[tuple[str, str, str]],
    values: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, float]:
    groups = set(common)
    result = {}
    for model in NEURAL_MODELS:
        per_stock = []
        for series, rows in truth.items():
            selected = tuple(
                value for row, value in zip(
                    rows, values[model][series], strict=True,
                )
                if (row.as_of, row.entry, row.target) in groups
            )
            if len(selected) != len(groups):
                raise ValueError("residual dispersion grid changed")
            per_stock.append(fmean(selected))
        result[model] = fmean(per_stock)
    if not all(isfinite(value) and value >= 0.0 for value in result.values()):
        raise ValueError("residual seed dispersion is invalid")
    return result


def evaluate_residual_phase(
    master: Sequence[str], source: ContextPhase, phase: ResidualPhaseInput,
    evidence: Sequence[ResidualPredictionEvidence],
    truth: Mapping[str, Sequence[ResidualTruthRow]],
) -> dict[str, object]:
    """Compute the frozen development metrics for one residual phase."""
    context_phase_value(source, master)
    rows, common = _truth(source, phase, truth)
    predictions, dispersion = _predictions(
        master, source, phase, evidence,
    )
    protocol = expected_residual_protocol()
    return {
        "evidence_role": EVIDENCE_ROLE,
        "integrity": {
            "config_sha256": RESIDUAL_CONFIG.sha256,
            "evaluation_grid_sha256": source.evaluation_grid_sha256,
            "residual_phase_sha256": residual_phase_sha256(phase),
            "source_phase_sha256": context_phase_sha256(source),
        },
        "locks": protocol["locks"],
        "observation_count": sum(
            count for _, count, _ in source.evaluation_rows
        ),
        "phase": source.phase,
        "primary": {
            "pooled_raw_residual_r2_vs_zero":
                _pooled_r2(rows, predictions),
            "paired_absolute_error": [
                _paired_metrics(rows, predictions, candidate, reference)
                for candidate, reference in PAIRED_COMPARISONS
            ],
        },
        "schema": 1,
        "secondary": _secondary(rows, predictions, common),
        "seed_dispersion": _seed_dispersion(
            rows, common, dispersion,
        ),
        "stock_count": len(rows),
        "target_kind": SPY_RESIDUAL_TARGET,
        "timestamp_count": len(common),
    }


def _counts(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} is invalid")
    return value


def _ordered_numbers(
    value: object, keys: Sequence[str], label: str, *,
    minimum: float | None = None, maximum: float | None = None,
) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} changed")
    result = {
        key: _finite(value[key], f"{label}.{key}") for key in keys
    }
    if minimum is not None and min(result.values()) < minimum:
        raise ValueError(f"{label} changed")
    if maximum is not None and max(result.values()) > maximum:
        raise ValueError(f"{label} changed")
    return result


def _validate_phase_evaluation(
    value: object, phase: ResidualPhaseInput,
) -> dict[str, object]:
    fields = {
        "schema", "phase", "evidence_role", "target_kind",
        "observation_count", "stock_count", "timestamp_count", "primary",
        "secondary", "seed_dispersion", "locks", "integrity",
    }
    item = _object(value, fields, "residual phase evaluation")
    expected_integrity = {
        "config_sha256": RESIDUAL_CONFIG.sha256,
        "evaluation_grid_sha256": phase.aligned_evaluation_grid_sha256,
        "residual_phase_sha256": residual_phase_sha256(phase),
        "source_phase_sha256": phase.source_phase_sha256,
    }
    shape = TERMINAL_PHASE_SHAPES[phase.phase]
    count = _integer(item["timestamp_count"], "residual timestamp count")
    stock_count = _integer(item["stock_count"], "residual stock count")
    if _integer(item["schema"], "residual evaluation schema") != 1 or \
       _string(item["phase"], "residual evaluation phase") != phase.phase or \
       item["evidence_role"] != EVIDENCE_ROLE or \
       item["target_kind"] != SPY_RESIDUAL_TARGET or \
       not _exact_json(
           item["locks"], expected_residual_protocol()["locks"],
       ) or \
       item["integrity"] != expected_integrity or \
       _integer(item["observation_count"], "residual observation count") != \
            shape["observations"] or stock_count != shape["stocks"] or \
       count != shape["timestamps"]:
        raise ValueError("residual phase evaluation changed")

    primary = _object(
        item["primary"], {
            "pooled_raw_residual_r2_vs_zero", "paired_absolute_error",
        }, "residual primary metrics",
    )
    _ordered_numbers(
        primary["pooled_raw_residual_r2_vs_zero"], MODELS, "residual R2",
        maximum=1.0,
    )
    paired = primary["paired_absolute_error"]
    if not isinstance(paired, list) or len(paired) != len(PAIRED_COMPARISONS):
        raise ValueError("residual paired comparisons changed")
    names: tuple[str, ...] | None = None
    common_date_count: int | None = None
    for raw, (candidate, reference) in zip(
        paired, PAIRED_COMPARISONS, strict=True,
    ):
        comparison = _object(raw, {
            "candidate", "date_count", "intervals", "losses", "mean_gain",
            "per_stock_mean_gain", "reference", "ties", "wins",
        }, "residual paired comparison")
        per_stock_value = comparison["per_stock_mean_gain"]
        if not isinstance(per_stock_value, dict) or \
           len(per_stock_value) != stock_count or any(
               not isinstance(name, str) or not name
               for name in per_stock_value
           ) or set(per_stock_value) != set(TERMINAL_SERIES):
            raise ValueError("residual per-stock gains changed")
        current_names = tuple(per_stock_value)
        if names is None:
            names = current_names
        elif set(current_names) != set(names):
            raise ValueError("residual per-stock gains changed")
        per_stock = _ordered_numbers(
            per_stock_value, current_names,
            "residual per-stock gains",
        )
        outcomes = tuple(
            _counts(comparison[key], f"residual {key}")
            for key in ("wins", "ties", "losses")
        )
        date_count = _integer(
            comparison["date_count"], "residual date count",
        )
        intervals = comparison["intervals"]
        expected_outcomes = (
            sum(value > 0.0 for value in per_stock.values()),
            sum(value == 0.0 for value in per_stock.values()),
            sum(value < 0.0 for value in per_stock.values()),
        )
        if common_date_count is None:
            common_date_count = date_count
        if comparison["candidate"] != candidate or \
           comparison["reference"] != reference or \
           date_count != shape["dates"] or date_count != common_date_count or \
           outcomes != expected_outcomes or \
           _finite(comparison["mean_gain"], "residual mean gain") != \
                fmean(per_stock.values()) or \
           not isinstance(intervals, dict) or \
           set(intervals) != set(map(str, BOOTSTRAP_BLOCK_DAYS)):
            raise ValueError("residual paired comparison changed")
        for width in BOOTSTRAP_BLOCK_DAYS:
            bounds = intervals[str(width)]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError("residual bootstrap interval changed")
            lower, upper = (
                _finite(bound, "residual bootstrap bound")
                for bound in bounds
            )
            if lower > upper:
                raise ValueError("residual bootstrap interval changed")

    secondary = _object(item["secondary"], {
        "centered_cross_sectional_r2", "spearman_rank_ic",
    }, "residual secondary metrics")
    _ordered_numbers(
        secondary["centered_cross_sectional_r2"], MODELS,
        "residual centered R2", maximum=1.0,
    )
    rank = secondary["spearman_rank_ic"]
    if not isinstance(rank, dict) or set(rank) != set(MODELS):
        raise ValueError("residual RankIC family changed")
    for model in MODELS:
        metric = _object(rank[model], {
            "excluded_timestamp_count", "mean", "valid_timestamp_count",
        }, f"{model} RankIC")
        valid = _integer(
            metric["valid_timestamp_count"], f"{model} valid RankIC count",
        )
        excluded = _counts(
            metric["excluded_timestamp_count"],
            f"{model} excluded RankIC count",
        )
        if valid + excluded != count:
            raise ValueError("residual RankIC counts changed")
        mean = _finite(metric["mean"], f"{model} mean RankIC")
        if not -1.0 <= mean <= 1.0:
            raise ValueError("residual RankIC changed")
    _ordered_numbers(
        item["seed_dispersion"], NEURAL_MODELS,
        "residual seed dispersion", minimum=0.0,
    )
    return deepcopy(item)


def _phase_inputs(
    values: Sequence[Mapping[str, object]], phases: Sequence[ResidualPhaseInput],
) -> tuple[dict[str, object], ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or \
       len(values) != len(phases):
        raise ValueError("residual phase inputs changed")
    result, paths = [], set()
    for raw, source in zip(values, phases, strict=True):
        item = _object(raw, {
            "access", "evaluation", "fits", "phase", "predictions", "receipt",
        }, "residual phase inputs")
        if _string(item["phase"], "residual phase input") != source.phase:
            raise ValueError("residual phase input order changed")
        parsed = {
            name: FileBinding.parse(
                item[name], f"{source.phase} {name}",
            )
            for name in ("access", "evaluation", "fits", "predictions",
                         "receipt")
        }
        current = tuple(binding.path for binding in parsed.values())
        if len(set(current)) != len(current) or paths.intersection(current):
            raise ValueError("residual phase input paths must be distinct")
        paths.update(current)
        result.append({
            "access": _binding_value(parsed["access"]),
            "evaluation": _binding_value(parsed["evaluation"]),
            "fits": _binding_value(parsed["fits"]),
            "phase": source.phase,
            "predictions": _binding_value(parsed["predictions"]),
            "receipt": _binding_value(parsed["receipt"]),
        })
    return tuple(result)


def finalize_residual_run(
    attempt: FileBinding, phases: Sequence[ResidualPhaseInput],
    evaluations: Mapping[str, Mapping[str, object]],
    phase_inputs: Sequence[Mapping[str, object]], source_tree_sha256: str,
) -> dict[str, object]:
    """Bind two received phase evaluations into one non-trading outcome."""
    bound = tuple(phases)
    if not isinstance(attempt, FileBinding) or \
       FileBinding.parse(
           _binding_value(attempt), "residual attempt",
       ) != attempt or len(bound) != len(PHASE_BUDGETS) or any(
           not isinstance(phase, ResidualPhaseInput) for phase in bound
       ):
        raise ValueError("residual terminal inputs changed")
    phase_names = tuple(name for name, _ in PHASE_BUDGETS)
    if tuple(phase.phase for phase in bound) != phase_names or \
       len({phase.scaler_inputs_sha256 for phase in bound}) != len(bound) or \
       any(
           _sha256(value, f"{phase.phase} {label}") != value
           for phase in bound
           for label, value in (
               ("source phase", phase.source_phase_sha256),
               ("training grid", phase.aligned_training_grid_sha256),
               ("evaluation grid", phase.aligned_evaluation_grid_sha256),
               ("scaler inputs", phase.scaler_inputs_sha256),
           )
       ) or \
       not isinstance(evaluations, Mapping) or \
       tuple(evaluations) != phase_names:
        raise ValueError("residual evaluations must follow phase order")
    inputs = _phase_inputs(phase_inputs, bound)
    if attempt.path in {
        binding["path"] for item in inputs
        for name, binding in item.items()
        if name != "phase" and isinstance(binding, dict)
    }:
        raise ValueError("residual input paths must be distinct")
    source_tree = _sha256(source_tree_sha256, "residual source tree")
    return {
        "decision": {
            "output_role": "residual-only-not-executable-return",
        },
        "evidence_role": EVIDENCE_ROLE,
        "inputs": {
            "attempt": _binding_value(attempt),
            "phases": list(inputs),
        },
        "integrity": {
            "config_sha256": RESIDUAL_CONFIG.sha256,
            "source_tree_sha256": source_tree,
        },
        "locks": expected_residual_protocol()["locks"],
        "phases": [
            _validate_phase_evaluation(
                evaluations[phase.phase], phase,
            )
            for phase in bound
        ],
        "schema": 1,
    }
