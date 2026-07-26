"""Authenticate completed calibration evidence without execution authority."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Protocol
import os
import stat

from tools.arm_context_diagnostic import (
    _derive_phases, _source_attempt, _source_outputs, _validate_commit,
    _validate_summary,
)
from tools.analyze_context_cross_section import (
    _completed_run as _completed_context_run,
)
from tools.context_cross_section import CONTEXT_ANALYSIS_SOURCE_PATHS
from tools.context_diagnostic_contract import (
    CONTEXT_CONFIG, SOURCE_EVIDENCE, ContextAttempt, ContextPhase,
    validate_context_sweep,
)
from tools.files import FrozenInput, freeze_inputs, verify_frozen
from tools.finalize_universe_scaling import (
    _fetch_bindings, _master_from_snapshot,
)
from tools.panel_contract import (
    FileBinding, SourceTree, _directory_identity, _exact_json,
    _open_directory, _regular_inputs, _verify_identities,
    read_canonical_json,
)
from tools.relative_context_contract import (
    RESIDUAL_BENCHMARK, RESIDUAL_CONFIG, RESIDUAL_SOURCE,
    ResidualAttempt, ResidualPhaseInput, ResidualTruthRow,
    expected_source_context_outcome,
    validate_residual_protocol, validate_source_context_outcome,
    validate_spy_fetch_report,
)
from tools.session_calendar import SessionCalendar
from tools.residual_phase_evidence import (
    phase_market_regimes, phase_truth,
)
from tools.spy_residual_controller import _collect_snapshot_inputs

Verify = Callable[[], None]
ROOT = Path(__file__).resolve().parents[1]
CONTEXT_CROSS_SECTION = FileBinding(
    "reports/h13-context-diagnostic-20260725-03/cross-section.json",
    "00b112f9755041a26ba1a46444346069da9c0b5c3b4496d2dbdcb9fe31405aea",
)


class TerminalPhase(Protocol):
    """Describe the terminal values supplied by the private analyzer seam."""

    source: ContextPhase
    phase: ResidualPhaseInput
    predictions: Mapping[str, Mapping[str, tuple[float, ...]]]
    evaluation: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ExecutionProvenance:
    """Bind one historical or current implementation to exact source bytes."""

    implementation_commit: str
    source_tree: SourceTree

    def __post_init__(self) -> None:
        if len(self.implementation_commit) != 40 or any(
            byte not in "0123456789abcdef"
            for byte in self.implementation_commit
        ) or not isinstance(self.source_tree, SourceTree):
            raise ValueError("execution provenance is invalid")


@dataclass(frozen=True, slots=True)
class CalibrationSeries:
    """Hold one ordered calibration series needed by sensitivity math."""

    name: str
    truth: tuple[ResidualTruthRow, ...]
    transformer_mean: tuple[float, ...]
    regimes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or \
           type(self.truth) is not tuple or not self.truth or \
           any(type(row) is not ResidualTruthRow for row in self.truth) or \
           type(self.transformer_mean) is not tuple or \
           type(self.regimes) is not tuple or \
           len(self.truth) != len(self.transformer_mean) or \
           len(self.truth) != len(self.regimes) or \
           any(
               type(value) is not float or not isfinite(value)
               for value in self.transformer_mean
           ) or \
           any(label not in ("negative", "nonnegative")
               for label in self.regimes):
            raise ValueError("calibration series is invalid")


@dataclass(frozen=True, slots=True)
class CompletedCalibrationEvidence:
    """Expose immutable calibration values and no model-running capability."""

    source: ContextPhase
    binding: ResidualPhaseInput
    series: tuple[CalibrationSeries, ...]
    scaling_execution: ExecutionProvenance
    scaling_finalizer: ExecutionProvenance
    context_execution: ExecutionProvenance
    residual_execution: ExecutionProvenance
    current_interpretation: ExecutionProvenance
    _verify: Verify = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        expected = tuple(
            name for name, _, _ in self.source.evaluation_rows
        ) if isinstance(self.source, ContextPhase) else ()
        counts = dict(
            (name, count)
            for name, count, _ in self.source.evaluation_rows
        ) if isinstance(self.source, ContextPhase) else {}
        if not isinstance(self.source, ContextPhase) or \
           self.source.phase != "calibration" or \
           not isinstance(self.binding, ResidualPhaseInput) or \
           self.binding.phase != self.source.phase or \
           type(self.series) is not tuple or \
           tuple(item.name for item in self.series) != expected or \
           any(type(item) is not CalibrationSeries for item in self.series) or \
           any(len(item.truth) != counts[item.name] for item in self.series) or \
           any(type(item) is not ExecutionProvenance for item in (
               self.scaling_execution, self.scaling_finalizer,
               self.context_execution, self.residual_execution,
               self.current_interpretation,
           )) or \
           not callable(self._verify):
            raise ValueError("completed calibration evidence is invalid")

    def verify(self) -> None:
        """Recheck every frozen input and directory identity."""
        self._verify()


def _path(binding: FileBinding) -> Path:
    path = Path(binding.path)
    return path if path.is_absolute() else ROOT / path


def _binding(path: Path, frozen: FrozenInput) -> FileBinding:
    if frozen.source != path:
        raise ValueError("calibration evidence input changed")
    try:
        logical = path.relative_to(ROOT).as_posix()
    except ValueError:
        logical = str(path)
    return FileBinding(logical, frozen.sha256)


def _require_binding(
    path: Path,
    frozen: FrozenInput,
    expected: FileBinding,
    label: str,
) -> None:
    if frozen.source != path or frozen.sha256 != expected.sha256 or \
       path != _path(expected):
        raise ValueError(f"{label} binding changed")


def _single_link_inputs(
    paths: Sequence[Path], label: str,
) -> tuple[tuple[Path, tuple[int, int]], ...]:
    identities = _regular_inputs(paths)
    if any(
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        for path in paths
        for metadata in (path.stat(follow_symlinks=False),)
    ):
        raise ValueError(f"{label} must be single-link regular files")
    return identities


def _verify_single_link_inputs(
    identities: Sequence[tuple[Path, tuple[int, int]]], label: str,
) -> None:
    _verify_identities(identities)
    if _single_link_inputs(
        tuple(path for path, _ in identities), label,
    ) != tuple(identities):
        raise ValueError(f"{label} identity changed")


def _directory_members(
    path: Path, names: Sequence[str],
) -> tuple[int, int]:
    descriptor, identity = _open_directory(path)
    try:
        entries = tuple(sorted(os.listdir(descriptor)))
        if entries != tuple(sorted(names)):
            raise ValueError("calibration evidence directory changed")
        for name in entries:
            metadata = os.stat(
                name, dir_fd=descriptor, follow_symlinks=False,
            )
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("calibration evidence entry is unsafe")
    finally:
        os.close(descriptor)
    return identity


def _context_artifacts(
    context: ContextAttempt,
) -> tuple[tuple[Path, FileBinding], ...]:
    expected = expected_source_context_outcome()
    raw = expected["inputs"]["phases"]
    if not isinstance(raw, list):
        raise ValueError("context terminal phases changed")
    result = []
    for index, phase in enumerate(raw):
        if not isinstance(phase, Mapping) or \
           phase.get("phase") != context.phases[index].phase:
            raise ValueError("context terminal phase changed")
        for name in ("fits", "predictions", "receipt", "access", "evaluation"):
            result.append((
                _path(FileBinding.parse(
                    phase[name], f"context phase[{index}].{name}",
                )),
                FileBinding.parse(
                    phase[name], f"context phase[{index}].{name}",
                ),
            ))
    return tuple(result)


def _manifest_range(
    snapshot: Path,
) -> tuple[date, date, int]:
    value = read_canonical_json(snapshot)
    if not isinstance(value, Mapping):
        raise ValueError("calibration manifest changed")
    start, end, interval = value.get("start"), value.get("end"), \
        value.get("interval_minutes")
    if type(start) is not str or type(end) is not str or \
       type(interval) is not int:
        raise ValueError("calibration manifest range changed")
    try:
        parsed = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError as error:
        raise ValueError("calibration manifest range changed") from error
    if tuple(map(str, parsed)) != (start, end):
        raise ValueError("calibration manifest range changed")
    return *parsed, interval


@contextmanager
def _bind_completed_calibration(
    attempt: ResidualAttempt,
    context: ContextAttempt,
    phases: Sequence[TerminalPhase],
    current_interpretation: ExecutionProvenance,
    verify_terminal: Verify,
) -> Iterator[CompletedCalibrationEvidence]:
    """Hold completed calibration values without granting execution access."""
    if not isinstance(attempt, ResidualAttempt) or \
       not isinstance(context, ContextAttempt) or \
       isinstance(phases, (str, bytes)) or not isinstance(phases, Sequence) or \
       not isinstance(current_interpretation, ExecutionProvenance) or \
       not callable(verify_terminal):
        raise ValueError("completed calibration request is invalid")
    terminal = tuple(phases)
    if tuple(phase.source for phase in terminal) != context.phases or \
       tuple(phase.phase for phase in terminal) != attempt.phases:
        raise ValueError("completed calibration phases changed")

    context_artifacts = _context_artifacts(context)
    roots = (
        _path(RESIDUAL_SOURCE["context_attempt"]),
        _path(RESIDUAL_SOURCE["context_outcome"]),
        *(path for path, _ in context_artifacts),
        _path(CONTEXT_CROSS_SECTION),
        *(ROOT / path for path in CONTEXT_ANALYSIS_SOURCE_PATHS),
        *(_path(binding) for binding in SOURCE_EVIDENCE.values()),
        _path(CONTEXT_CONFIG),
        _path(RESIDUAL_CONFIG),
        *(_path(binding) for binding in RESIDUAL_BENCHMARK.values()),
    )
    root_paths = tuple(dict.fromkeys(roots))
    root_identities = _single_link_inputs(
        root_paths, "calibration evidence roots",
    )
    with freeze_inputs(root_paths) as root_snapshots:
        root_frozen = dict(zip(root_paths, root_snapshots, strict=True))
        context_path = _path(RESIDUAL_SOURCE["context_attempt"])
        published_context = ContextAttempt.read(
            root_frozen[context_path].snapshot,
            Path(RESIDUAL_SOURCE["context_attempt"].path), ROOT,
        )
        if published_context != context or \
           _binding(context_path, root_frozen[context_path]) != \
                RESIDUAL_SOURCE["context_attempt"]:
            raise ValueError("source context attempt changed")
        context_outcome = _path(RESIDUAL_SOURCE["context_outcome"])
        if _binding(
            context_outcome, root_frozen[context_outcome],
        ) != RESIDUAL_SOURCE["context_outcome"] or not _exact_json(
            read_canonical_json(root_frozen[context_outcome].snapshot),
            validate_source_context_outcome(
                read_canonical_json(root_frozen[context_outcome].snapshot),
            ),
        ):
            raise ValueError("source context outcome changed")
        if any(
            _binding(path, root_frozen[path]) != binding
            for path, binding in context_artifacts
        ) or _binding(
            _path(CONTEXT_CROSS_SECTION),
            root_frozen[_path(CONTEXT_CROSS_SECTION)],
        ) != CONTEXT_CROSS_SECTION:
            raise ValueError("source context artifact changed")
        authenticated_context, _, _ = _completed_context_run(
            context_path, Path(RESIDUAL_SOURCE["context_attempt"].path),
            root_frozen, _directory_identity(context_outcome.parent),
        )
        if authenticated_context != context:
            raise ValueError("source context completion changed")

        scaling = _source_attempt(
            root_frozen[_path(SOURCE_EVIDENCE["attempt"])],
        )
        outputs = _source_outputs(
            root_frozen[_path(SOURCE_EVIDENCE["failure"])], scaling,
        )
        if _binding(
            _path(SOURCE_EVIDENCE["fits"]),
            root_frozen[_path(SOURCE_EVIDENCE["fits"])],
        ) != SOURCE_EVIDENCE["fits"]:
            raise ValueError("source scaling fits changed")
        context_config = root_frozen[_path(CONTEXT_CONFIG)]
        residual_config = root_frozen[_path(RESIDUAL_CONFIG)]
        if _binding(_path(CONTEXT_CONFIG), context_config) != \
                CONTEXT_CONFIG or _binding(
                    _path(RESIDUAL_CONFIG), residual_config,
                ) != RESIDUAL_CONFIG:
            raise ValueError("calibration configuration changed")
        config = validate_context_sweep(
            read_canonical_json(context_config.snapshot),
        )
        validate_residual_protocol(
            read_canonical_json(residual_config.snapshot),
        )
        spy_report = root_frozen[
            _path(RESIDUAL_BENCHMARK["fetch_report"])
        ]
        spy_csv = root_frozen[_path(RESIDUAL_BENCHMARK["spy_csv"])]
        if _binding(
            _path(RESIDUAL_BENCHMARK["fetch_report"]), spy_report,
        ) != RESIDUAL_BENCHMARK["fetch_report"] or \
           _binding(
               _path(RESIDUAL_BENCHMARK["spy_csv"]), spy_csv,
           ) != RESIDUAL_BENCHMARK["spy_csv"]:
            raise ValueError("calibration SPY bundle changed")
        validate_spy_fetch_report(
            read_canonical_json(spy_report.snapshot), ROOT,
        )

        fetch_path = _path(scaling.fetch_report)
        fetch_identities = _single_link_inputs(
            (fetch_path,), "calibration fetch report",
        )
        with freeze_inputs((fetch_path,)) as fetch_snapshots:
            if _binding(fetch_path, fetch_snapshots[0]) != \
                    scaling.fetch_report:
                raise ValueError("calibration fetch report changed")
            names, csv_bindings = _fetch_bindings(
                fetch_snapshots[0].snapshot,
            )
            data_paths = tuple(dict.fromkeys((
                *(_path(item.file) for item in scaling.manifests),
                _path(scaling.session_calendar), _path(scaling.config),
                *(_path(binding) for binding in outputs.values()),
                *(_path(binding) for binding in csv_bindings),
            )))
            data_identities = _single_link_inputs(
                data_paths, "calibration source data",
            )
            with freeze_inputs(data_paths) as data_snapshots:
                data = dict(zip(data_paths, data_snapshots, strict=True))
                for index, item in enumerate(scaling.manifests):
                    path = _path(item.file)
                    _require_binding(
                        path, data[path], item.file,
                        f"calibration manifest[{index}]",
                    )
                for label, binding in (
                    ("calibration calendar", scaling.session_calendar),
                    ("calibration scaling config", scaling.config),
                ):
                    path = _path(binding)
                    _require_binding(path, data[path], binding, label)
                for name, binding in outputs.items():
                    path = _path(binding)
                    _require_binding(
                        path, data[path], binding,
                        f"calibration scaling {name}",
                    )
                for index, binding in enumerate(csv_bindings):
                    path = _path(binding)
                    _require_binding(
                        path, data[path], binding,
                        f"calibration CSV[{index}]",
                    )
                _validate_summary(
                    data[_path(outputs["summary"])], outputs,
                )
                manifest = data[_path(scaling.manifests[-1].file)]
                if _master_from_snapshot(manifest.snapshot) != names or \
                   names != context.master:
                    raise ValueError("calibration universe changed")
                if _derive_phases(
                    scaling, names, csv_bindings, data, context_config,
                ) != context.phases:
                    raise ValueError("context phase derivation changed")
                start, end, interval = _manifest_range(manifest.snapshot)
                calendar = SessionCalendar.read(
                    data[_path(scaling.session_calendar)].snapshot,
                )
                stock_csv = tuple(
                    (name, data[_path(binding)])
                    for name, binding in zip(
                        names, csv_bindings, strict=True,
                    )
                )
                context_run = context_outcome.parent
                context_names = (
                    *(path.name for path, _ in context_artifacts),
                    context_outcome.name, CONTEXT_CROSS_SECTION.path.rsplit(
                        "/", 1,
                    )[-1],
                )
                context_identity = _directory_members(
                    context_run, context_names,
                )
                scaling_run = ROOT / scaling.run_dir
                scaling_names = tuple(
                    _path(outputs[name]).name
                    for name in ("fits", "predictions", "summary")
                )
                scaling_identity = _directory_members(
                    scaling_run, scaling_names,
                )
                spy_dir = _path(RESIDUAL_BENCHMARK["spy_csv"]).parent
                spy_names = tuple(
                    _path(binding).name
                    for binding in RESIDUAL_BENCHMARK.values()
                )
                spy_identity = _directory_members(spy_dir, spy_names)

                def verify() -> None:
                    verify_terminal()
                    _verify_single_link_inputs(
                        root_identities, "calibration evidence roots",
                    )
                    _verify_single_link_inputs(
                        fetch_identities, "calibration fetch report",
                    )
                    _verify_single_link_inputs(
                        data_identities, "calibration source data",
                    )
                    verify_frozen(root_snapshots)
                    verify_frozen(fetch_snapshots)
                    verify_frozen(data_snapshots)
                    if _directory_identity(context_run) != \
                            context_identity or _directory_members(
                                context_run, context_names,
                            ) != context_identity or \
                       _directory_identity(scaling_run) != \
                            scaling_identity or _directory_members(
                                scaling_run, scaling_names,
                            ) != scaling_identity or \
                       _directory_identity(spy_dir) != spy_identity or \
                       _directory_members(spy_dir, spy_names) != spy_identity:
                        raise ValueError(
                            "calibration evidence topology changed",
                        )

                _validate_commit(
                    scaling.implementation_commit, scaling.source_tree,
                )
                _validate_commit(
                    scaling.implementation_commit, scaling.finalizer_tree,
                )
                _validate_commit(
                    context.implementation_commit, context.source_tree,
                )
                _validate_commit(
                    attempt.implementation_commit, attempt.source_tree,
                )
                _validate_commit(
                    current_interpretation.implementation_commit,
                    current_interpretation.source_tree,
                )
                states = _collect_snapshot_inputs(
                    context, config, start, end, interval, calendar,
                    stock_csv, spy_csv, verify,
                )
                if tuple(state.binding for state in states) != \
                        attempt.phases or tuple(
                            (state.source, state.binding) for state in states
                        ) != tuple(
                            (phase.source, phase.phase) for phase in terminal
                        ):
                    raise ValueError(
                        "calibration phase derivation changed",
                    )
                matches = tuple(
                    (state, phase)
                    for state, phase in zip(states, terminal, strict=True)
                    if state.source.phase == "calibration"
                )
                if len(matches) != 1:
                    raise ValueError("calibration phase selection changed")
                state, phase = matches[0]
                regimes = phase_market_regimes(state, spy_csv, verify)
                truth, _ = phase_truth(
                    state, dict(stock_csv), spy_csv, verify,
                )
                try:
                    predictions = phase.predictions["panel_transformer"]
                except KeyError as error:
                    raise ValueError(
                        "calibration transformer predictions are missing",
                    ) from error
                if tuple(truth) != tuple(predictions) or \
                   tuple(truth) != tuple(regimes):
                    raise ValueError("calibration series order changed")
                evidence = CompletedCalibrationEvidence(
                    state.source, state.binding,
                    tuple(
                        CalibrationSeries(
                            name, truth[name],
                            tuple(predictions[name]), tuple(regimes[name]),
                        )
                        for name in truth
                    ),
                    ExecutionProvenance(
                        scaling.implementation_commit, scaling.source_tree,
                    ),
                    ExecutionProvenance(
                        scaling.implementation_commit,
                        scaling.finalizer_tree,
                    ),
                    ExecutionProvenance(
                        context.implementation_commit, context.source_tree,
                    ),
                    ExecutionProvenance(
                        attempt.implementation_commit, attempt.source_tree,
                    ),
                    current_interpretation, verify,
                )
                verify()
                try:
                    yield evidence
                finally:
                    verify()
