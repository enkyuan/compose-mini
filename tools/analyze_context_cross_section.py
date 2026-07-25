#!/usr/bin/env python3
"""Publish post-hoc stock-selection diagnostics for one completed context run."""

from __future__ import annotations

import sys

_BOOTSTRAP_FLAGS = ("-I", "-S", "-B")
_BOOTSTRAP_CACHE_PREFIX: str | None = None
_PACKAGE_NAME = "tools.analyze_context_cross_section"


def _require_isolated_execution(*, bootstrapped: bool = False) -> None:
    flags = sys.flags
    if not flags.isolated or not getattr(flags, "safe_path", False) or \
       not flags.no_user_site or not flags.no_site or \
       not flags.dont_write_bytecode or \
       not flags.ignore_environment or not sys.dont_write_bytecode or \
       bootstrapped and (
           _BOOTSTRAP_CACHE_PREFIX is None or
           sys.pycache_prefix != _BOOTSTRAP_CACHE_PREFIX
       ):
        raise ValueError(
            "context analysis requires isolated bytecode-free Python",
        )


def _require_exact_launch(*, pristine: bool = False) -> None:
    if pristine and ("ctypes" in sys.modules or "_ctypes" in sys.modules):
        raise ValueError("context analysis launch inspection is already loaded")

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
        raise ValueError("context analysis requires its exact script launch")


def _register_package_alias() -> None:
    module = sys.modules[__name__]
    if _PACKAGE_NAME in sys.modules:
        raise ValueError("context analysis package alias is unsafe")
    sys.modules[_PACKAGE_NAME] = module


def _require_package_alias() -> None:
    if sys.modules.get(_PACKAGE_NAME) is not sys.modules[__name__]:
        raise ValueError("context analysis package alias changed")


def _bootstrap_main() -> None:
    """Authenticate the package namespace before importing repository code."""
    global _BOOTSTRAP_CACHE_PREFIX

    from importlib.machinery import PathFinder
    import os
    import stat
    import tempfile

    while True:
        prefix = os.path.join(
            tempfile.gettempdir(),
            f"compose-mini-context-analysis-{os.urandom(32).hex()}",
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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Protocol
import argparse
import json
import os

ROOT = Path(__file__).resolve().parents[1]

from tools.arm_context_diagnostic import (
    ContextLease, authenticate_context_attempt,
)
from tools.context_cross_section import evaluate_context_cross_section
from tools.context_diagnostic_contract import (
    HISTORY_LENGTHS, PRIMARY_MODEL, TARGET_PHASES, ContextAttempt,
    ContextPhase, ContextPredictionEvidence, ContextReceipt,
    context_phase_sha256, validate_context_sweep,
)
from tools.context_diagnostic_inputs import context_bar_prefix, timestamp_rows
from tools.data_v1 import (
    CLOSE_RETURN_TARGET, EXECUTABLE_RETURN_TARGET, FEATURE_COUNT,
)
from tools.fetch_universe import UniverseManifest
from tools.files import (
    FrozenInput, freeze_inputs, verify_frozen, write_json_exclusive,
)
from tools.finalize_context_diagnostic import (
    ContextTruthRow, _select_context_history,
)
from tools.panel_contract import (
    FileBinding, _absent, _directory_identity, _exact_json,
    _open_directory, _regular_inputs, _verify_identities,
    read_canonical_json,
)
from tools.run_context_diagnostic import (
    _attempt_path, context_access_value, phase_artifacts,
    read_context_attempt, validate_context_ledgers,
)
from tools.run_universe_scaling import _expose_torch_package
from tools.session_calendar import SessionCalendar
from tools.universe_contract import PackedRows
from tools.universe_scaling import BOOTSTRAP_BLOCK_DAYS

ANALYSIS_SOURCE_PATHS = (
    "tools/analyze_context_cross_section.py",
    "tools/context_cross_section.py",
    "tools/universe_cross_section.py",
)
EVIDENCE_ROLE = "development-post-hoc-not-forward-clean"
SELECTED_HISTORY = HISTORY_LENGTHS[0]


class ContextSweep(Protocol):
    target_kind: str
    target_horizon_bars: int
    alignment_horizon_bars: int


PhaseRows = Callable[
    [
        ContextAttempt, ContextPhase, ContextLease, ContextSweep,
        UniverseManifest, SessionCalendar,
    ],
    tuple[Mapping[str, tuple[str, ...]], Mapping[str, PackedRows]],
]


def _binding(path: Path, frozen: FrozenInput) -> FileBinding:
    if frozen.source != path:
        raise ValueError("context analysis input changed")
    try:
        logical = path.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("context analysis input escaped the repository") \
            from error
    return FileBinding(logical, frozen.sha256)


def _value(binding: FileBinding) -> dict[str, str]:
    return asdict(binding)


def _completed_run(
    attempt_path: Path, logical_path: Path,
    frozen: Mapping[Path, FrozenInput], run_identity: tuple[int, int],
) -> tuple[
    ContextAttempt,
    Mapping[str, tuple[ContextPredictionEvidence, ...]],
    Mapping[str, object],
]:
    attempt = ContextAttempt.read(
        frozen[attempt_path].snapshot, logical_path, ROOT,
    )
    attempt_binding = _binding(attempt_path, frozen[attempt_path])
    evaluations, evidence, phase_inputs = {}, {}, []
    for phase in attempt.phases:
        artifacts = phase_artifacts(ROOT, attempt_path, phase)
        fit, prediction, receipt, access, evaluation = tuple(artifacts)
        bindings = tuple(
            _binding(path, frozen[path]) for path in artifacts
        )
        fit_binding, prediction_binding, receipt_binding, \
            access_binding, evaluation_binding = bindings
        parsed_receipt = ContextReceipt.parse(
            read_canonical_json(frozen[receipt].snapshot),
        )
        parsed_receipt.validate(
            phase, attempt_binding, fit_binding, prediction_binding,
            attempt.source_tree.sha256, run_identity,
        )
        evidence[phase.phase] = validate_context_ledgers(
            attempt.master, phase, frozen[fit].snapshot,
            frozen[prediction].snapshot,
            attempt.source_binding("failure").sha256,
            attempt.config.sha256,
        )
        if not _exact_json(
            read_canonical_json(frozen[access].snapshot),
            context_access_value(
                attempt_binding, receipt_binding, phase,
            ),
        ):
            raise ValueError("context truth access changed")
        evaluations[phase.phase] = read_canonical_json(
            frozen[evaluation].snapshot,
        )
        phase_inputs.append({
            "access": _value(access_binding),
            "evaluation": _value(evaluation_binding),
            "fits": _value(fit_binding),
            "phase": phase.phase,
            "predictions": _value(prediction_binding),
            "receipt": _value(receipt_binding),
        })

    decision = _select_context_history(attempt.phases, evaluations)
    if decision["selected_history"] != SELECTED_HISTORY:
        raise ValueError("context analysis requires selected history 17")
    outcome_path = ROOT / attempt.run_dir / "outcome.json"
    expected = {
        "decision": decision,
        "evidence_role": "development-diagnostic-not-forward-clean",
        "inputs": {
            "attempt": _value(attempt_binding),
            "phases": phase_inputs,
        },
        "integrity": {
            "config_sha256": attempt.config.sha256,
            "source_failure_sha256":
                attempt.source_binding("failure").sha256,
            "source_tree_sha256": attempt.source_tree.sha256,
        },
        "schema": 1,
    }
    if not _exact_json(
        read_canonical_json(frozen[outcome_path].snapshot), expected,
    ):
        raise ValueError("context terminal outcome changed")

    sources = tuple(
        _binding(ROOT / path, frozen[ROOT / path])
        for path in ANALYSIS_SOURCE_PATHS
    )
    return attempt, evidence, {
        "analysis_sources": list(map(_value, sources)),
        "attempt": _value(attempt_binding),
        "outcome": _value(_binding(
            outcome_path, frozen[outcome_path],
        )),
    }


def _common_groups(
    phase: ContextPhase,
    timestamps: Mapping[str, Sequence[str]],
    packed: Mapping[str, PackedRows],
) -> tuple[tuple[str, str, str], ...]:
    """Derive the complete evaluation grid without reading market values."""
    names = tuple(series for series, _, _ in phase.evaluation_rows)
    grids = {
        series: timestamp_rows(
            timestamps[series],
            packed[series].rows[packed[series].counts[0]:],
        )
        for series in names
    }
    membership = {name: set(grid) for name, grid in grids.items()}
    common = tuple(
        row for row in grids[names[0]]
        if all(row in membership[name] for name in names[1:])
    )
    required = set(common)
    if not common or any(
        tuple(row for row in grids[name] if row in required) != common
        for name in names
    ):
        raise ValueError("context common group grid changed")
    return common


def _phase_truth(
    attempt: ContextAttempt, phase: ContextPhase, lease: ContextLease,
    sweep: ContextSweep, manifest: UniverseManifest,
    calendar: SessionCalendar, phase_rows: PhaseRows,
) -> tuple[
    Mapping[str, tuple[ContextTruthRow, ...]],
    tuple[tuple[str, str, str], ...],
]:
    timestamps, packed = phase_rows(
        attempt, phase, lease, sweep, manifest, calendar,
    )
    groups = _common_groups(phase, timestamps, packed)
    lease()
    csv, truth = dict(lease.snapshots.csv), {}
    for series, _, _ in phase.evaluation_rows:
        samples = packed[series]
        rows = samples.rows[samples.counts[0]:]
        bars = context_bar_prefix(
            csv[series].snapshot, timestamps[series],
            timestamps[series][rows[-1].target],
        )
        truth[series] = tuple(
            ContextTruthRow(
                timestamps[series][row.as_of],
                timestamps[series][row.entry],
                timestamps[series][row.target],
                float(bars[
                    row.as_of * FEATURE_COUNT + 3
                    if sweep.target_kind == CLOSE_RETURN_TARGET
                    else row.entry * FEATURE_COUNT
                ]),
                float(bars[row.target * FEATURE_COUNT + 3]),
            )
            for row in rows
        )
    lease()
    return truth, groups


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        byte not in "0123456789abcdef" for byte in value
    ):
        raise ValueError(f"{label} hash changed")
    return value


def _float(
    value: object, label: str, *,
    minimum: float | None = None, maximum: float | None = None,
) -> float:
    if type(value) is not float or not isfinite(value) or \
       minimum is not None and value < minimum or \
       maximum is not None and value > maximum:
        raise ValueError(f"{label} changed")
    return value


def _optional_float(
    value: object, label: str, *,
    minimum: float | None = None, maximum: float | None = None,
) -> float | None:
    return (
        None if value is None else
        _float(value, label, minimum=minimum, maximum=maximum)
    )


def _names(value: object, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ) or len(set(value)) != len(value):
        raise ValueError(f"{label} changed")
    return list(value)


def _effective_breadth(
    value: object, series: Sequence[str],
) -> dict[str, object]:
    fields = {"value", "included", "excluded", "reason"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("context effective breadth changed")
    included = _names(value["included"], "context included breadth")
    excluded = _names(value["excluded"], "context excluded breadth")
    reason = value["reason"]
    estimate = _optional_float(
        value["value"], "context effective breadth", minimum=0.0,
    )
    names = tuple(series)
    if included != sorted(included) or excluded != sorted(excluded) or \
       set(included) & set(excluded) or \
       set(included) | set(excluded) != set(names) or \
       reason not in (
           None, "fewer-than-two-aligned-dates",
           "fewer-than-two-nonconstant-stocks",
           "nonpositive-or-nonfinite-denominator",
       ) or (estimate is None) == (reason is None):
        raise ValueError("context effective breadth changed")
    return {
        "excluded": excluded,
        "included": included,
        "reason": reason,
        "value": estimate,
    }


def _diagnostic(
    value: object, series: Sequence[str],
) -> dict[str, object]:
    fields = {
        "r2", "paired_mean", "intervals", "raw_breadth",
        "group_count", "date_count", "eligible_spearman_groups",
        "excluded_spearman_groups", "mean_spearman",
        "effective_breadth", "meets_statistical_gate",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("context diagnostic fields changed")
    integer_fields = (
        "raw_breadth", "group_count", "date_count",
        "eligible_spearman_groups", "excluded_spearman_groups",
    )
    if any(type(value[name]) is not int for name in integer_fields) or \
       value["raw_breadth"] != len(series) or \
       value["group_count"] < 1 or \
       value["date_count"] < max(BOOTSTRAP_BLOCK_DAYS) or \
       value["date_count"] > value["group_count"] or \
       min(
           value["eligible_spearman_groups"],
           value["excluded_spearman_groups"],
       ) < 0 or \
       value["eligible_spearman_groups"] + \
            value["excluded_spearman_groups"] != value["group_count"] or \
       type(value["meets_statistical_gate"]) is not bool:
        raise ValueError("context diagnostic counts changed")
    intervals = value["intervals"]
    blocks = tuple(map(str, BOOTSTRAP_BLOCK_DAYS))
    if not isinstance(intervals, dict) or tuple(intervals) != blocks:
        raise ValueError("context diagnostic intervals changed")
    safe_intervals = {}
    for block in blocks:
        bounds = intervals[block]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("context diagnostic interval changed")
        lower, upper = (
            _float(bound, "context diagnostic interval")
            for bound in bounds
        )
        if lower > upper:
            raise ValueError("context diagnostic interval changed")
        safe_intervals[block] = (lower, upper)
    r2 = _optional_float(
        value["r2"], "context R-squared", maximum=1.0,
    )
    mean_spearman = _optional_float(
        value["mean_spearman"], "context mean Spearman",
        minimum=-1.0, maximum=1.0,
    )
    if (mean_spearman is None) != (
        value["eligible_spearman_groups"] == 0
    ):
        raise ValueError("context Spearman availability changed")
    gate = r2 is not None and r2 > 0.0 and \
        max(bounds[1] for bounds in safe_intervals.values()) < 0.0
    if value["meets_statistical_gate"] != gate:
        raise ValueError("context statistical gate changed")
    return {
        "date_count": value["date_count"],
        "effective_breadth": _effective_breadth(
            value["effective_breadth"], series,
        ),
        "eligible_spearman_groups":
            value["eligible_spearman_groups"],
        "excluded_spearman_groups":
            value["excluded_spearman_groups"],
        "group_count": value["group_count"],
        "intervals": safe_intervals,
        "mean_spearman": mean_spearman,
        "meets_statistical_gate": value["meets_statistical_gate"],
        "paired_mean": _float(
            value["paired_mean"], "context paired mean",
        ),
        "r2": r2,
        "raw_breadth": value["raw_breadth"],
    }


def _report(
    inputs: Mapping[str, object],
    phases: Sequence[Mapping[str, object]],
    contracts: Sequence[ContextPhase],
) -> dict[str, object]:
    if not isinstance(inputs, dict) or set(inputs) != {
        "analysis_sources", "attempt", "outcome",
    }:
        raise ValueError("context cross-sectional inputs changed")
    attempt = _value(FileBinding.parse(
        inputs["attempt"], "context analysis attempt",
    ))
    outcome = _value(FileBinding.parse(
        inputs["outcome"], "context analysis outcome",
    ))
    raw_sources = inputs["analysis_sources"]
    if not isinstance(raw_sources, list) or \
       len(raw_sources) != len(ANALYSIS_SOURCE_PATHS):
        raise ValueError("context analysis sources changed")
    sources = [
        _value(FileBinding.parse(
            value, f"context analysis source {index}",
        ))
        for index, value in enumerate(raw_sources)
    ]
    if tuple(value["path"] for value in sources) != ANALYSIS_SOURCE_PATHS:
        raise ValueError("context analysis source order changed")

    fields = {
        "diagnostic", "evidence_role", "group_grid_sha256", "history",
        "model", "phase", "phase_sha256", "schema",
    }
    values = tuple(phases)
    bound = tuple(contracts)
    if len(values) != len(TARGET_PHASES) or \
       len(bound) != len(TARGET_PHASES) or any(
           not isinstance(phase, ContextPhase) for phase in bound
       ) or tuple(phase.phase for phase in bound) != TARGET_PHASES:
        raise ValueError("context cross-sectional results changed")
    safe = []
    for value, contract in zip(values, bound, strict=True):
        phase = contract.phase
        if not isinstance(value, dict) or set(value) != fields or \
           type(value["schema"]) is not int or value["schema"] != 1 or \
           value["evidence_role"] != EVIDENCE_ROLE or \
           value["model"] != PRIMARY_MODEL or \
           type(value["history"]) is not int or \
           value["history"] != SELECTED_HISTORY or \
           value["phase"] != phase or \
           value["phase_sha256"] != context_phase_sha256(contract):
            raise ValueError("context cross-sectional results changed")
        names = tuple(
            series for series, _, _ in contract.evaluation_rows
        )
        safe.append({
            "diagnostic": _diagnostic(value["diagnostic"], names),
            "evidence_role": EVIDENCE_ROLE,
            "group_grid_sha256": _hash(
                value["group_grid_sha256"], "context group grid",
            ),
            "history": SELECTED_HISTORY,
            "model": PRIMARY_MODEL,
            "phase": phase,
            "phase_sha256": _hash(
                value["phase_sha256"], "context phase",
            ),
            "schema": 1,
        })
    return {
        "evidence_role": EVIDENCE_ROLE,
        "inputs": {
            "analysis_sources": sources,
            "attempt": attempt,
            "outcome": outcome,
        },
        "locks": {
            "backtest_run": False,
            "forward_clean": False,
            "trading_authorized": False,
            "universe_expansion_authorized": False,
        },
        "phases": safe,
        "schema": 1,
    }


def _publish(
    output: Path, report: Mapping[str, object], directory: int,
    verify: Callable[[], None],
) -> None:
    verify()
    write_json_exclusive(
        output, report, directory, before_link=verify,
    )


def analyze_context_cross_section(
    attempt_path: Path,
) -> Mapping[str, object]:
    """Validate one completed run and exclusively publish its diagnostics."""
    _require_isolated_execution(bootstrapped=True)
    _require_exact_launch()
    _require_package_alias()
    live = read_context_attempt(attempt_path)
    if Path(sys.executable).resolve(strict=True) != Path(
        live.torch_probe.python.path,
    ).resolve(strict=True):
        raise ValueError("analysis requires the attempt's bound Torch Python")
    absolute, logical = _attempt_path(attempt_path)
    run = ROOT / live.run_dir
    output = run / "cross-section.json"
    _absent(output, "context cross-sectional artifact")
    paths = (
        absolute, run / "outcome.json",
        *(
            path for phase in live.phases
            for path in phase_artifacts(ROOT, absolute, phase)
        ),
        *(ROOT / path for path in ANALYSIS_SOURCE_PATHS),
    )
    identities = _regular_inputs(paths)
    directory, run_identity = _open_directory(run)
    try:
        with freeze_inputs(paths) as snapshots:
            frozen = {
                item.source: item for item in snapshots
            }
            attempt, evidence, inputs = _completed_run(
                absolute, logical, frozen, run_identity,
            )
            if attempt != live:
                raise ValueError("context attempt changed")
            _verify_identities(identities)
            verify_frozen(snapshots)
            with authenticate_context_attempt(attempt) as lease:
                _expose_torch_package(Path(
                    attempt.torch_probe.package_tree.root,
                ))
                from tools.context_diagnostic_controller import _phase_rows
                from tools.experiment import Sweep

                validate_context_sweep(read_canonical_json(
                    lease.snapshots.config.snapshot,
                ))
                sweep = Sweep.read(lease.snapshots.config.snapshot)
                if sweep.target_kind not in (
                    CLOSE_RETURN_TARGET, EXECUTABLE_RETURN_TARGET,
                ):
                    raise ValueError("context target kind changed")
                manifest = UniverseManifest.read(
                    lease.snapshots.manifest.snapshot,
                )
                calendar = SessionCalendar.read(
                    lease.snapshots.calendar.snapshot,
                )
                results = []
                for phase in attempt.phases:
                    truth, groups = _phase_truth(
                        attempt, phase, lease, sweep, manifest,
                        calendar, _phase_rows,
                    )
                    results.append(evaluate_context_cross_section(
                        attempt.master, phase, evidence[phase.phase],
                        truth, SELECTED_HISTORY, groups,
                    ))
                    lease()
                report = _report(inputs, results, attempt.phases)

                def verify() -> None:
                    lease()
                    _verify_identities(identities)
                    verify_frozen(snapshots)
                    if _directory_identity(run) != run_identity:
                        raise ValueError("context run directory changed")

                _publish(output, report, directory, verify)
        return report
    finally:
        os.close(directory)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_args()
    try:
        report = analyze_context_cross_section(arguments.attempt)
    except (
        IndexError, KeyError, OSError, OverflowError, TypeError,
        UnicodeError, ValueError,
    ) as error:
        print(f"integrity error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    outcome = ROOT / report["inputs"]["outcome"]["path"]
    print(json.dumps({
        "output": str(outcome.with_name("cross-section.json")),
        "phases": [value["phase"] for value in report["phases"]],
        "status": "analyzed",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
