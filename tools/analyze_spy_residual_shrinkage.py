#!/usr/bin/env python3
"""Test whether one global scale rescues SPY-residual forecasts."""

from __future__ import annotations

import sys

_BOOTSTRAP_FLAGS = ("-I", "-S", "-B")
_BOOTSTRAP_CACHE_PREFIX: str | None = None
_PACKAGE_NAME = "tools.analyze_spy_residual_shrinkage"


def _require_isolated_execution(*, bootstrapped: bool = False) -> None:
    flags = sys.flags
    if not flags.isolated or not getattr(flags, "safe_path", False) or \
       not flags.no_user_site or not flags.no_site or \
       not flags.dont_write_bytecode or not flags.ignore_environment or \
       not sys.dont_write_bytecode or bootstrapped and (
           _BOOTSTRAP_CACHE_PREFIX is None or
           sys.pycache_prefix != _BOOTSTRAP_CACHE_PREFIX
       ):
        raise ValueError(
            "residual shrinkage requires isolated bytecode-free Python",
        )


def _require_exact_launch(*, pristine: bool = False) -> None:
    if pristine and ("ctypes" in sys.modules or "_ctypes" in sys.modules):
        raise ValueError("residual shrinkage launch inspection is loaded")

    from ctypes import POINTER, byref, c_int, c_wchar_p, pythonapi
    import os

    argc = c_int()
    argv = POINTER(c_wchar_p)()
    get_argv = pythonapi.Py_GetArgcArgv
    get_argv.argtypes = (POINTER(c_int), POINTER(POINTER(c_wchar_p)))
    get_argv.restype = None
    get_argv(byref(argc), byref(argv))
    observed = tuple(argv[index] for index in range(argc.value))
    canonical = lambda values: (os.path.realpath(values[0]), *values[1:])
    expected = (
        os.path.realpath(sys.executable), *_BOOTSTRAP_FLAGS, *sys.argv,
    )
    if not observed or canonical(observed) != expected or \
       canonical(tuple(sys.orig_argv)) != expected or \
       os.path.realpath(sys.argv[0]) != os.path.realpath(__file__):
        raise ValueError("residual shrinkage requires its exact script launch")


def _register_package_alias() -> None:
    module = sys.modules[__name__]
    if _PACKAGE_NAME in sys.modules:
        raise ValueError("residual shrinkage package alias is unsafe")
    sys.modules[_PACKAGE_NAME] = module


def _require_package_alias() -> None:
    if sys.modules.get(_PACKAGE_NAME) is not sys.modules[__name__]:
        raise ValueError("residual shrinkage package alias changed")


def _bootstrap_main() -> None:
    """Authenticate the package namespace before repository imports."""
    global _BOOTSTRAP_CACHE_PREFIX

    from importlib.machinery import PathFinder
    import os
    import stat
    import tempfile

    while True:
        prefix = os.path.join(
            tempfile.gettempdir(),
            f"compose-mini-residual-shrinkage-{os.urandom(32).hex()}",
        )
        if not os.path.lexists(prefix):
            break
    sys.pycache_prefix = prefix
    sys.dont_write_bytecode = True

    tools = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(tools)
    initializer = os.path.join(tools, "__init__.py")
    if not stat.S_ISDIR(os.lstat(tools).st_mode) or \
       not stat.S_ISREG(os.lstat(initializer).st_mode):
        raise ValueError("tools namespace is not a real package")
    for entry in os.scandir(tools):
        mode = entry.stat(follow_symlinks=False).st_mode
        valid = (
            stat.S_ISDIR(mode) if entry.name == "__pycache__" else
            entry.name.endswith(".py") and stat.S_ISREG(mode)
        )
        if not valid:
            raise ValueError("tools namespace contains an unsafe entry")
    if any(
        name == "tools" or name.startswith("tools.") for name in sys.modules
    ):
        raise ValueError("tools namespace is already loaded")
    spec = PathFinder.find_spec("tools", (*sys.path, root))
    locations = tuple(
        os.path.realpath(path)
        for path in (spec.submodule_search_locations or ())
    ) if spec is not None else ()
    if spec is None or os.path.realpath(spec.origin or "") != \
            os.path.realpath(initializer) or \
            locations != (os.path.realpath(tools),):
        raise ValueError("tools namespace resolver is unsafe")
    sys.path.append(root)
    import tools as package
    if os.path.realpath(package.__file__ or "") != \
            os.path.realpath(initializer) or tuple(
                map(os.path.realpath, package.__path__)
            ) != locations:
        raise ValueError("tools namespace import is unsafe")
    _register_package_alias()
    _BOOTSTRAP_CACHE_PREFIX = prefix


if __name__ == "__main__":
    _require_isolated_execution()
    _require_exact_launch(pristine=True)
    _bootstrap_main()

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite, log
from pathlib import Path
from statistics import fmean
import argparse
import json
import os
import stat

from tools.arm_context_diagnostic import _validate_commit
from tools.arm_spy_residual import (
    ResidualLease, _directory_members, authenticate_residual_attempt,
)
from tools.context_diagnostic_contract import ContextAttempt, ContextPhase
from tools.context_diagnostic_inputs import context_bar_prefix
from tools.data_v1 import FEATURE_COUNT
from tools.files import (
    FrozenInput, freeze_inputs, verify_frozen, write_json_exclusive,
)
from tools.finalize_spy_residual import (
    _binding_value, _day, _finite,
    _paired_metrics as _paired_mae_metrics,
    _pooled_r2 as _model_pooled_r2, _predictions, _secondary, _square_sum,
    _truth, finalize_residual_run,
)
from tools.panel_contract import (
    FileBinding, SourceTree, _absent, _directory_identity, _exact_json,
    _open_directory, _tree_digest, mkdir_nofollow, read_canonical_json,
)
from tools.relative_context_contract import (
    MODELS, ResidualAttempt, ResidualPhaseInput, ResidualReceipt,
    ResidualTruthRow, expected_residual_protocol,
)
from tools.run_context_diagnostic import (
    _binding as _file_binding, _frozen as _frozen_input, phase_artifacts,
)
from tools.run_spy_residual import (
    _attempt_path, _master, _outcome_path, _single_link_inputs,
    _verify_single_link_inputs, read_residual_attempt,
    residual_access_value, validate_residual_ledgers,
)
from tools.spy_residual_controller import _PhaseRows, _collect_inputs
from tools.universe_cross_section import CROSS_SECTION_SEED
from tools.universe_scaling import (
    BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_REPLICATES, circular_block_interval,
)

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SOURCE_PATHS = ("tools/analyze_spy_residual_shrinkage.py",)
SOURCE_ATTEMPT = FileBinding(
    "experiments/h13-spy-residual-20260725-01-attempt.json",
    "0fb90623c90b418dfff93d35dde1bb49024c25d3b2b27b799ce752b8deed9ea3",
)
SOURCE_OUTCOME = FileBinding(
    "experiments/h13-spy-residual-20260725-01-outcome.json",
    "132c17cd7dde7abcdf625581d6b399d7e9b0011f4a9c8bc6d4d0f4065d3a0488",
)
SOURCE_IMPLEMENTATION = "0bc33956ddbff9f706d1341b77f01e71a0b07496"
CANDIDATE = "shrunk_transformer"
MODEL = "panel_transformer"
REFERENCES = ("zero", MODEL, "global_ridge", "global_mlp")
EVIDENCE_ROLE = "development-post-hoc-not-forward-clean"


@dataclass(frozen=True, slots=True)
class ShrinkageFit:
    """Record the sufficient statistics for one zero-anchored fit."""

    scale: float
    unclipped_scale: float | None
    numerator: float
    denominator: float
    observation_count: int


@dataclass(frozen=True, slots=True)
class AuthenticatedPhase:
    """Retain one terminal phase without exposing its truth."""

    source: ContextPhase
    phase: ResidualPhaseInput
    predictions: Mapping[str, Mapping[str, tuple[float, ...]]]
    evaluation: Mapping[str, object]


def _completed_run(
    attempt_path: Path,
    logical_path: Path,
    frozen: Mapping[Path, FrozenInput],
    run_identity: tuple[int, int],
    context: ContextAttempt,
) -> tuple[
    ResidualAttempt, tuple[AuthenticatedPhase, ...], dict[str, object],
]:
    """Authenticate the completed run without regenerating its truth."""
    attempt = ResidualAttempt.read(
        _frozen_input(frozen, attempt_path).snapshot,
        logical_path, ROOT, context,
    )
    attempt_binding = _file_binding(
        ROOT, _frozen_input(frozen, attempt_path),
    )
    if attempt_binding != SOURCE_ATTEMPT or \
       attempt.implementation_commit != SOURCE_IMPLEMENTATION:
        raise ValueError("shrinkage source attempt changed")
    evaluations, phase_inputs, phases = {}, [], []
    for source, phase in zip(
        context.phases, attempt.phases, strict=True,
    ):
        artifacts = phase_artifacts(ROOT, attempt_path, source)
        fits, predictions, receipt, access, evaluation = tuple(artifacts)
        bindings = tuple(
            _file_binding(ROOT, _frozen_input(frozen, path))
            for path in artifacts
        )
        fit_binding, prediction_binding, receipt_binding, \
            access_binding, evaluation_binding = bindings
        parsed_receipt = ResidualReceipt.parse(read_canonical_json(
            _frozen_input(frozen, receipt).snapshot,
        ))
        parsed_receipt.validate(
            source, phase, attempt_binding, fit_binding,
            prediction_binding, attempt.source_tree.sha256, run_identity,
        )
        evidence = validate_residual_ledgers(
            _master(source), source, phase,
            _frozen_input(frozen, fits).snapshot,
            _frozen_input(frozen, predictions).snapshot,
        )
        if not _exact_json(
            read_canonical_json(_frozen_input(frozen, access).snapshot),
            residual_access_value(
                attempt_binding, receipt_binding, source,
            ),
        ):
            raise ValueError("residual truth access changed")
        evaluation_value = read_canonical_json(
            _frozen_input(frozen, evaluation).snapshot,
        )
        ensembles, _ = _predictions(
            _master(source), source, phase, evidence,
        )
        phase_input = {
            "access": _binding_value(access_binding),
            "evaluation": _binding_value(evaluation_binding),
            "fits": _binding_value(fit_binding),
            "phase": source.phase,
            "predictions": _binding_value(prediction_binding),
            "receipt": _binding_value(receipt_binding),
        }
        evaluations[source.phase] = evaluation_value
        phase_inputs.append(phase_input)
        phases.append(AuthenticatedPhase(
            source, phase, ensembles, evaluation_value,
        ))

    outcome_path = _outcome_path(attempt_path)
    expected = finalize_residual_run(
        attempt_binding, attempt.phases, evaluations, phase_inputs,
        attempt.source_tree.sha256,
    )
    outcome_binding = _file_binding(
        ROOT, _frozen_input(frozen, outcome_path),
    )
    if outcome_binding != SOURCE_OUTCOME or not _exact_json(
        read_canonical_json(_frozen_input(frozen, outcome_path).snapshot),
        expected,
    ):
        raise ValueError("residual terminal outcome changed")
    source = _file_binding(
        ROOT, _frozen_input(frozen, ROOT / ANALYSIS_SOURCE_PATHS[0]),
    )
    return attempt, tuple(phases), {
        "analysis_source": _binding_value(source),
        "attempt": _binding_value(attempt_binding),
        "outcome": _binding_value(outcome_binding),
        "phases": phase_inputs,
    }


def _phase_truth(
    state: _PhaseRows,
    lease: ResidualLease,
) -> tuple[
    Mapping[str, tuple[ResidualTruthRow, ...]],
    tuple[tuple[str, str, str], ...],
]:
    """Regenerate one authenticated phase's residual truth in memory."""
    if not isinstance(state, _PhaseRows) or \
       not isinstance(lease, ResidualLease):
        raise TypeError("residual truth inputs are invalid")
    lease()
    benchmark = dict(lease.benchmark)
    try:
        spy_csv = benchmark["spy_csv"]
    except KeyError as error:
        raise ValueError("residual SPY snapshot is missing") from error
    verify_frozen((spy_csv,))
    csv = dict(lease.context.snapshots.csv)
    stock = {
        series: (timestamps, packed)
        for series, timestamps, packed in state.stock
    }
    aligned = dict(state.spy)
    truth = {}
    for series, _, _ in state.source.evaluation_rows:
        timestamps, packed = stock[series]
        boundary = packed.counts[0]
        samples = packed.rows[boundary:]
        spy_samples = aligned[series].rows[boundary:]
        if not samples or len(samples) != len(spy_samples):
            raise ValueError(f"{series} residual truth grid changed")
        stock_bars = context_bar_prefix(
            csv[series].snapshot, timestamps,
            timestamps[samples[-1].target],
        )
        spy_bars = context_bar_prefix(
            spy_csv.snapshot, state.spy_timestamps,
            state.spy_timestamps[spy_samples[-1].target],
        )
        truth[series] = tuple(
            ResidualTruthRow(
                timestamps[row.as_of],
                timestamps[row.entry],
                timestamps[row.target],
                log(
                    stock_bars[row.target * FEATURE_COUNT + 3] /
                    stock_bars[row.entry * FEATURE_COUNT]
                ) - log(
                    spy_bars[spy_row.target * FEATURE_COUNT + 3] /
                    spy_bars[spy_row.entry * FEATURE_COUNT]
                ),
            )
            for row, spy_row in zip(
                samples, spy_samples, strict=True,
            )
        )
    lease()
    verify_frozen((spy_csv,))
    return _truth(state.source, state.binding, truth)


def _pairs(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Sequence[float]],
    excluded: Collection[str] = (),
) -> tuple[tuple[float, float], ...]:
    """Return validated truth/forecast pairs in their bound order."""
    if not isinstance(truth, Mapping) or not isinstance(predictions, Mapping):
        raise TypeError("shrinkage inputs must be mappings")
    names = tuple(truth)
    if not names or names != tuple(predictions) or \
       any(type(name) is not str or not name for name in names):
        raise ValueError("shrinkage series order changed")
    if isinstance(excluded, (str, bytes)) or not isinstance(
        excluded, Collection,
    ):
        raise TypeError("excluded series must be a collection")
    omitted = tuple(excluded)
    if len(omitted) != len(set(omitted)) or \
       any(type(name) is not str for name in omitted) or \
       not set(omitted).issubset(names):
        raise ValueError("excluded series are invalid")

    result = []
    for name in names:
        rows, values = truth[name], predictions[name]
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or \
           isinstance(values, (str, bytes)) or \
           not isinstance(values, Sequence) or \
           not rows or len(rows) != len(values):
            raise ValueError(f"{name} shrinkage rows changed")
        validated = []
        for row, value in zip(rows, values, strict=True):
            if not isinstance(row, ResidualTruthRow):
                raise TypeError(f"{name} residual truth row is invalid")
            validated.append((
                _finite(row.value, f"{name} residual truth"),
                _finite(value, f"{name} residual prediction"),
            ))
        if name not in omitted:
            result.extend(validated)
    if not result:
        raise ValueError("shrinkage observations are empty")
    return tuple(result)


def fit_zero_anchored_scale(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Sequence[float]],
    excluded: Collection[str] = (),
) -> ShrinkageFit:
    """Fit the clipped least-squares scale against the zero forecast."""
    pairs = _pairs(truth, predictions, excluded)
    forecast = tuple(predicted for _, predicted in pairs)
    denominator = _square_sum(forecast, "shrinkage denominator")
    try:
        numerator = fsum(actual * predicted for actual, predicted in pairs)
    except OverflowError:
        raise ValueError("shrinkage numerator is non-finite") from None
    if not isfinite(numerator):
        raise ValueError("shrinkage numerator is non-finite")
    unclipped = None if denominator == 0.0 else numerator / denominator
    if unclipped is not None and not isfinite(unclipped):
        raise ValueError("unclipped shrinkage scale is non-finite")
    return ShrinkageFit(
        0.0 if unclipped is None else min(1.0, max(0.0, unclipped)),
        unclipped, numerator, denominator, len(pairs),
    )


def zero_anchored_scale(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Sequence[float]],
    excluded: Collection[str] = (),
) -> float:
    """Return the conservative MSE-minimizing scale in ``[0, 1]``."""
    return fit_zero_anchored_scale(truth, predictions, excluded).scale


def scale_predictions(
    predictions: Mapping[str, Sequence[float]], scale: float,
) -> dict[str, tuple[float, ...]]:
    """Apply one validated scale without changing the prediction shape."""
    factor = _finite(scale, "shrinkage scale")
    if not 0.0 <= factor <= 1.0 or not isinstance(predictions, Mapping) or \
       not predictions:
        raise ValueError("shrinkage scale or predictions are invalid")
    result = {}
    for name, values in predictions.items():
        if type(name) is not str or not name or \
           isinstance(values, (str, bytes)) or \
           not isinstance(values, Sequence) or not values:
            raise ValueError("shrinkage prediction shape changed")
        result[name] = tuple(
            factor * _finite(value, f"{name} residual prediction")
            for value in values
        )
    return result


def pooled_r2(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Sequence[float]],
) -> float:
    """Score residual forecasts against the zero-residual baseline."""
    pairs = _pairs(truth, predictions)
    denominator = _square_sum(
        tuple(actual for actual, _ in pairs), "raw residual denominator",
    )
    if denominator == 0.0:
        raise ValueError("raw residual denominator is zero")
    errors = tuple(actual - predicted for actual, predicted in pairs)
    result = 1.0 - _square_sum(
        errors, "shrunk residual error",
    ) / denominator
    if not isfinite(result):
        raise ValueError("shrunk residual R-squared is non-finite")
    return result


def fit_diagnostic(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Fit once and expose only sufficient and stability statistics."""
    fit = fit_zero_anchored_scale(truth, predictions)
    without = {
        name: fit_zero_anchored_scale(
            truth, predictions, (name,),
        ).scale
        for name in truth
    }
    values = tuple(without.values())
    return {
        "denominator": fit.denominator,
        "numerator": fit.numerator,
        "observation_count": fit.observation_count,
        "pooled_raw_residual_r2_vs_zero": pooled_r2(
            truth, scale_predictions(predictions, fit.scale),
        ),
        "scale": fit.scale,
        "scale_leave_one_stock_out": without,
        "scale_leave_one_stock_out_delta": {
            name: value - fit.scale for name, value in without.items()
        },
        "scale_leave_one_stock_out_range": max(values) - min(values),
        "unclipped_scale": fit.unclipped_scale,
    }


def _paired_mse_metrics(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    candidate: str,
    reference: str,
) -> dict[str, object]:
    """Compare squared errors over the existing common-date panel."""
    forecast = {
        **predictions,
        "zero": {
            series: (0.0,) * len(rows) for series, rows in truth.items()
        },
    }
    if candidate not in forecast or reference not in forecast:
        raise ValueError("paired residual model is missing")
    daily = {}
    for series, rows in truth.items():
        by_day: dict[str, list[float]] = {}
        for row, left, right in zip(
            rows, forecast[candidate][series], forecast[reference][series],
            strict=True,
        ):
            by_day.setdefault(_day(row.target), []).append(_finite(
                (row.value - right) ** 2 - (row.value - left) ** 2,
                "paired residual MSE gain",
            ))
        daily[series] = {
            day: tuple(values) for day, values in by_day.items()
        }
    dates = tuple(sorted(set.intersection(*(
        set(values) for values in daily.values()
    ))))
    if not dates:
        raise ValueError("residual MSE gains have no common target dates")
    per_stock = {
        series: fmean(
            value for day in dates for value in daily[series][day]
        )
        for series in truth
    }
    return {
        "candidate": candidate,
        "date_count": len(dates),
        "intervals": {
            str(width): circular_block_interval(
                daily, width, BOOTSTRAP_REPLICATES, CROSS_SECTION_SEED,
            )
            for width in BOOTSTRAP_BLOCK_DAYS
        },
        "losses": sum(value < 0.0 for value in per_stock.values()),
        "mean_gain": fmean(per_stock.values()),
        "per_stock_mean_gain": per_stock,
        "reference": reference,
        "ties": sum(value == 0.0 for value in per_stock.values()),
        "wins": sum(value > 0.0 for value in per_stock.values()),
    }


def _leave_one_out_r2(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """Drop one stock without refitting or changing any timestamp grid."""
    pairs = {
        name: _pairs({name: truth[name]}, {name: predictions[name]})
        for name in truth
    }
    truth_energy = {
        name: _square_sum(
            tuple(actual for actual, _ in values),
            f"{name} raw residual denominator",
        )
        for name, values in pairs.items()
    }
    error_energy = {
        name: _square_sum(
            tuple(actual - predicted for actual, predicted in values),
            f"{name} shrunk residual error",
        )
        for name, values in pairs.items()
    }
    result = {}
    for omitted in truth:
        denominator = fsum(
            value for name, value in truth_energy.items()
            if name != omitted
        )
        if denominator == 0.0:
            raise ValueError("leave-one-out residual denominator is zero")
        result[omitted] = 1.0 - fsum(
            value for name, value in error_energy.items()
            if name != omitted
        ) / denominator
    if not all(map(isfinite, result.values())):
        raise ValueError("leave-one-out residual R-squared is non-finite")
    return result


def evaluate_frozen_scale(
    scale: float,
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    common: Sequence[tuple[str, str, str]],
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    transformer_seed_dispersion: float,
    transformer_rank_ic: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate one fold-1 scale on a later development phase."""
    factor = _finite(scale, "frozen shrinkage scale")
    if not 0.0 <= factor <= 1.0 or tuple(predictions) != MODELS:
        raise ValueError("frozen shrinkage inputs changed")
    scaled = scale_predictions(predictions[MODEL], factor)
    family = {**predictions, CANDIDATE: scaled}
    scored = {**predictions, MODEL: scaled}
    raw_r2 = _model_pooled_r2(truth, scored)[MODEL]
    if factor == 0.0:
        centered_r2, rank = 0.0, None
    else:
        secondary = _secondary(truth, scored, common)
        centered_r2 = secondary["centered_cross_sectional_r2"][MODEL]
        computed_rank = secondary["spearman_rank_ic"][MODEL]
        if not _exact_json(computed_rank, transformer_rank_ic):
            raise ValueError("positive shrinkage changed Transformer RankIC")
        rank = dict(transformer_rank_ic)
    mse = {
        reference: _paired_mse_metrics(
            truth, family, CANDIDATE, reference,
        )
        for reference in REFERENCES
    }
    mae = {
        reference: _paired_mae_metrics(
            truth, family, CANDIDATE, reference,
        )
        for reference in REFERENCES
    }
    leave_one_out = _leave_one_out_r2(truth, scaled)
    primary = mse["zero"]
    decision = factor > 0.0 and raw_r2 > 0.0 and \
        primary["intervals"]["20"][0] > 0.0 and \
        all(value > 0.0 for value in leave_one_out.values()) and \
        primary["wins"] >= 6
    dispersion = _finite(
        transformer_seed_dispersion, "Transformer seed dispersion",
    )
    if dispersion < 0.0:
        raise ValueError("Transformer seed dispersion is negative")
    return {
        "centered_cross_sectional_r2": centered_r2,
        "later_residual_holdout_preregistration_warranted": decision,
        "paired_absolute_error": mae,
        "paired_squared_error": mse,
        "pooled_raw_residual_r2_vs_zero": raw_r2,
        "pooled_raw_residual_r2_without_stock": leave_one_out,
        "scaled_seed_dispersion": factor * dispersion,
        "spearman_rank_ic": rank,
    }


def _json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("shrinkage report is not finite JSON") from error
    if not isinstance(result, dict):
        raise ValueError("shrinkage report must be an object")
    return result


def _fit_report(
    inputs: Mapping[str, object],
    diagnostic: Mapping[str, object],
    implementation_commit: str,
) -> dict[str, object]:
    """Bind the tuning-only scale before calibration truth is opened."""
    return _json_mapping({
        "candidate": {
            "fit_phase": "fold-1",
            "model": MODEL,
            "parameter_stability_only": True,
            "seed_aggregation": "arithmetic-mean-before-scaling",
        },
        "decision": {
            "output_role": "residual-magnitude-diagnostic-only",
        },
        "evidence_role": EVIDENCE_ROLE,
        "fit": dict(diagnostic),
        "inputs": dict(inputs),
        "integrity": {
            "implementation_commit": implementation_commit,
        },
        "locks": dict(expected_residual_protocol()["locks"]),
        "schema": 1,
    })


def _final_report(
    inputs: Mapping[str, object],
    fit_binding: FileBinding,
    fit_value: Mapping[str, object],
    evaluation: Mapping[str, object],
    implementation_commit: str,
) -> dict[str, object]:
    """Bind the adaptive calibration result without executable claims."""
    warranted = evaluation[
        "later_residual_holdout_preregistration_warranted"
    ]
    if type(warranted) is not bool:
        raise ValueError("shrinkage decision is invalid")
    return _json_mapping({
        "decision": {
            "later_residual_holdout_preregistration_warranted": warranted,
            "output_role":
                "adaptive-residual-only-not-executable-return",
        },
        "evidence_role": EVIDENCE_ROLE,
        "evaluation": dict(evaluation),
        "fit": {
            "artifact": _binding_value(fit_binding),
            "phase": "fold-1",
            "scale": fit_value["fit"]["scale"],
        },
        "inputs": dict(inputs),
        "integrity": {
            "implementation_commit": implementation_commit,
        },
        "locks": dict(expected_residual_protocol()["locks"]),
        "schema": 1,
    })


def _validate_analysis_commit(
    commit: str, source: FileBinding,
) -> None:
    if type(commit) is not str or len(commit) != 40 or any(
        byte not in "0123456789abcdef" for byte in commit
    ) or source.path != ANALYSIS_SOURCE_PATHS[0]:
        raise ValueError("shrinkage implementation commit is invalid")
    files = (source,)
    _validate_commit(
        commit, SourceTree(
            str(ROOT.resolve(strict=True)), files, _tree_digest(files),
        ),
    )


def _create_output_directory(path: Path) -> tuple[int, tuple[int, int]]:
    _absent(path, "shrinkage output directory")
    identity = mkdir_nofollow(path)
    descriptor, opened = _open_directory(path)
    if opened != identity or os.listdir(descriptor):
        os.close(descriptor)
        raise ValueError("shrinkage output directory changed")
    os.fsync(descriptor)
    return descriptor, identity


def _validate_published(
    path: Path, expected: Mapping[str, object],
) -> tuple[tuple[Path, tuple[int, int]], ...]:
    identities = _single_link_inputs((path,), "shrinkage output")
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or \
       stat.S_IMODE(metadata.st_mode) != 0o600 or \
       not _exact_json(read_canonical_json(path), expected):
        raise ValueError("shrinkage output changed")
    return identities


def _publish(
    path: Path,
    value: Mapping[str, object],
    directory: int,
    verify: object,
) -> None:
    if not callable(verify):
        raise TypeError("shrinkage verifier is invalid")
    verify()
    write_json_exclusive(
        path, value, directory, before_link=verify,
    )


def _analyze_phases(
    states: Sequence[_PhaseRows],
    phases: Sequence[AuthenticatedPhase],
    lease: ResidualLease,
    inputs: Mapping[str, object],
    implementation_commit: str,
    fit_path: Path,
    result_path: Path,
    directory: int,
    output_identity: tuple[int, int],
    verify: Callable[[], None],
) -> Mapping[str, object]:
    """Seal fold-1 tuning evidence before opening calibration truth."""
    verify()
    fold_truth, _ = _phase_truth(states[0], lease)
    fit_value = _fit_report(
        inputs,
        fit_diagnostic(fold_truth, phases[0].predictions[MODEL]),
        implementation_commit,
    )
    del fold_truth
    _publish(fit_path, fit_value, directory, verify)
    fit_identities = _validate_published(fit_path, fit_value)
    output = fit_path.parent
    if _directory_members(output, (fit_path.name,)) != output_identity:
        raise ValueError("shrinkage fit topology changed")

    with freeze_inputs((fit_path,)) as fit_frozen:
        published_fit = read_canonical_json(fit_frozen[0].snapshot)
        if not _exact_json(published_fit, fit_value):
            raise ValueError("shrinkage fit changed")
        fit_binding = _file_binding(ROOT, fit_frozen[0])

        def verify_fit() -> None:
            verify()
            _verify_single_link_inputs(
                fit_identities, "shrinkage fit",
            )
            verify_frozen(fit_frozen)

        verify_fit()
        scale = _finite(
            published_fit["fit"]["scale"], "frozen shrinkage scale",
        )
        calibration_truth, common = _phase_truth(states[1], lease)
        calibration = phases[1]
        evaluation = evaluate_frozen_scale(
            scale, calibration_truth, common, calibration.predictions,
            calibration.evaluation["seed_dispersion"][MODEL],
            calibration.evaluation["secondary"][
                "spearman_rank_ic"
            ][MODEL],
        )
        del calibration_truth
        result = _final_report(
            inputs, fit_binding, published_fit, evaluation,
            implementation_commit,
        )
        _publish(result_path, result, directory, verify_fit)
        result_identities = _validate_published(result_path, result)
        with freeze_inputs((result_path,)) as result_frozen:
            if not _exact_json(
                read_canonical_json(result_frozen[0].snapshot), result,
            ):
                raise ValueError("shrinkage result changed")
            verify_fit()
            _verify_single_link_inputs(
                result_identities, "shrinkage result",
            )
            verify_frozen(result_frozen)
            if _directory_members(
                output, (fit_path.name, result_path.name),
            ) != output_identity:
                raise ValueError("shrinkage result topology changed")
        return result


def analyze_residual_shrinkage(
    attempt_path: Path, implementation_commit: str,
) -> Mapping[str, object]:
    """Fit on fold-1, seal the fit, then inspect calibration once."""
    _require_isolated_execution(bootstrapped=True)
    _require_exact_launch()
    _require_package_alias()
    live, context = read_residual_attempt(attempt_path)
    live.primary_python.validate_live("shrinkage primary Python")
    if Path(sys.executable).resolve(strict=True) != Path(
        live.primary_python.path,
    ).resolve(strict=True):
        raise ValueError("shrinkage requires the bound primary Python")
    absolute, logical = _attempt_path(attempt_path)
    if logical.as_posix() != SOURCE_ATTEMPT.path:
        raise ValueError("shrinkage requires the frozen source attempt")
    run = ROOT / live.run_dir
    outcome = _outcome_path(absolute)
    artifacts = tuple(
        path
        for source in context.phases
        for path in phase_artifacts(ROOT, absolute, source)
    )
    source = ROOT / ANALYSIS_SOURCE_PATHS[0]
    paths = (absolute, outcome, *artifacts, source)
    identities = _single_link_inputs(paths, "shrinkage inputs")
    run_names = tuple(path.name for path in artifacts)
    run_identity = _directory_members(run, run_names)

    with freeze_inputs(paths) as snapshots:
        frozen = {item.source: item for item in snapshots}
        attempt, phases, inputs = _completed_run(
            absolute, logical, frozen, run_identity, context,
        )
        if attempt != live:
            raise ValueError("residual attempt changed")
        source_binding = FileBinding.parse(
            inputs["analysis_source"], "shrinkage analysis source",
        )
        _validate_analysis_commit(implementation_commit, source_binding)

        def verify_inputs() -> None:
            _verify_single_link_inputs(identities, "shrinkage inputs")
            verify_frozen(snapshots)
            if _directory_members(run, run_names) != run_identity:
                raise ValueError("residual run directory changed")

        verify_inputs()
        with authenticate_residual_attempt(attempt) as lease:
            try:
                spy = dict(lease.benchmark)["spy_csv"]
            except KeyError as error:
                raise ValueError("residual SPY snapshot is missing") from error
            states = _collect_inputs(context, lease.context, spy)
            if tuple(
                (state.source, state.binding) for state in states
            ) != tuple(
                (phase.source, phase.phase) for phase in phases
            ):
                raise ValueError("shrinkage phase binding changed")
            output = run.with_name(f"{run.name}-shrinkage")
            fit_path, result_path = (
                output / "shrinkage-fit.json",
                output / "shrinkage.json",
            )
            _absent(fit_path, "shrinkage fit")
            _absent(result_path, "shrinkage result")
            directory, output_identity = _create_output_directory(output)
            try:
                def verify() -> None:
                    lease()
                    verify_inputs()
                    if _directory_identity(output) != output_identity:
                        raise ValueError(
                            "shrinkage output directory changed",
                        )

                return _analyze_phases(
                    states, phases, lease, inputs,
                    implementation_commit, fit_path, result_path,
                    directory, output_identity, verify,
                )
            finally:
                os.close(directory)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--implementation-commit", required=True)
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_args()
    try:
        report = analyze_residual_shrinkage(
            arguments.attempt, arguments.implementation_commit,
        )
    except (
        IndexError, KeyError, OSError, OverflowError, TypeError,
        UnicodeError, ValueError,
    ) as error:
        print(f"integrity error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps({
        "later_residual_holdout_preregistration_warranted":
            report["decision"][
                "later_residual_holdout_preregistration_warranted"
            ],
        "scale": report["fit"]["scale"],
        "status": "analyzed",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
