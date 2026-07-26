#!/usr/bin/env python3
"""Execute one receipt-gated stock-minus-SPY calibration attempt."""

from __future__ import annotations

import sys

_BOOTSTRAP_FLAGS = ("-I", "-S", "-B")
_BOOTSTRAP_CACHE_PREFIX: str | None = None
_PACKAGE_NAME = "tools.run_spy_residual"


def _require_isolated_execution(*, bootstrapped: bool = False) -> None:
    flags = sys.flags
    if not flags.isolated or not getattr(flags, "safe_path", False) or \
       not flags.no_user_site or not flags.no_site or \
       not flags.dont_write_bytecode or \
       not flags.ignore_environment or not sys.dont_write_bytecode:
        raise ValueError(
            "residual runner requires isolated bytecode-free Python",
        )
    if bootstrapped and (
        _BOOTSTRAP_CACHE_PREFIX is None or
        sys.pycache_prefix != _BOOTSTRAP_CACHE_PREFIX
    ):
        raise ValueError("residual runner requires authenticated bootstrap")


def _require_exact_launch(*, pristine: bool = False) -> None:
    if pristine and ("ctypes" in sys.modules or "_ctypes" in sys.modules):
        raise ValueError("residual runner launch inspection is already loaded")

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
        raise ValueError("residual runner requires the exact bound launch")


def _register_package_alias() -> None:
    module = sys.modules[__name__]
    if _PACKAGE_NAME in sys.modules:
        raise ValueError("residual runner package alias is unsafe")
    sys.modules[_PACKAGE_NAME] = module


def _require_package_alias() -> None:
    if sys.modules.get(_PACKAGE_NAME) is not sys.modules[__name__]:
        raise ValueError("residual runner package alias changed")


def _bootstrap_main() -> None:
    """Authenticate the import namespace before exposing repository code."""
    global _BOOTSTRAP_CACHE_PREFIX

    from importlib.machinery import PathFinder
    import os
    import stat
    import tempfile

    while True:
        prefix = os.path.join(
            tempfile.gettempdir(),
            f"compose-mini-residual-runner-{os.urandom(32).hex()}",
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

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
import argparse
import json
import os
import signal
import stat

from tools.context_diagnostic_contract import (
    ContextAttempt, ContextFit, ContextPhase, ContextPrediction,
)
from tools.files import ExclusiveTemp, FrozenInput, freeze_inputs, verify_frozen
from tools.panel_contract import (
    FileBinding, _absent, _directory_identity, _exact_json, _open_directory,
    _regular_inputs, _verify_identities, mkdir_nofollow,
    read_canonical_json, read_canonical_json_lines, selected_source_tree,
    source_tree,
)
from tools.relative_context_contract import (
    PHASE_BUDGETS, RESIDUAL_SOURCE, RESIDUAL_SOURCE_PATHS,
    ResidualAttempt, ResidualFitEvidence, ResidualPhaseInput,
    ResidualPredictionEvidence, ResidualReceipt, ResidualTruthRow,
    expected_residual_fits, expected_residual_predictions,
    residual_fit_record, residual_phase_sha256, residual_prediction_record,
    validate_residual_fit_records, validate_residual_prediction_records,
)
from tools.run_context_diagnostic import (
    Interrupted, PhaseArtifacts, _TerminalOutcome, _binding, _logical,
    _run_path, _verify_terminal_outcome, _write_json, _write_ledger,
    phase_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)

ResidualFitOne = Callable[
    [ContextFit], tuple[str, float, object],
]
ResidualPredictOne = Callable[
    [ContextPrediction, object], tuple[float, ...],
]
ResidualTruthReader = Callable[
    [], Mapping[str, tuple[ResidualTruthRow, ...]],
]
Verify = Callable[[], None]


@dataclass(frozen=True, slots=True)
class ResidualPhaseEvidence:
    """Retain one completed phase until its terminal outcome is durable."""

    phase: str
    bindings: tuple[FileBinding, ...]
    identities: tuple[tuple[int, int], ...]
    source_phase_sha256: str
    residual_phase_sha256: str
    source_tree_sha256: str


@dataclass(frozen=True, slots=True)
class ResidualRunClaim:
    """Authorize one process to use one immutable residual run directory."""

    root: Path
    path: Path
    directory_identity: tuple[int, int]
    attempt_path: Path
    attempt_binding: FileBinding
    attempt_identity: tuple[int, int]
    _completed: dict[str, ResidualPhaseEvidence] = field(
        default_factory=dict, compare=False, repr=False,
    )

    @property
    def completed(self) -> Mapping[str, ResidualPhaseEvidence]:
        """Expose completed evidence without granting lifecycle mutation."""
        return MappingProxyType(self._completed)


@dataclass(slots=True)
class _ClaimState:
    claim: ResidualRunClaim
    started: set[str]
    outcome_parent_identity: tuple[int, int]


_CLAIMS: dict[int, _ClaimState] = {}
_ACTIVE_ATTEMPT: Path | None = None
_TERMINAL_OUTCOME: _TerminalOutcome | None = None
_SIGNAL_NUMBER: int | None = None


def _state(claim: ResidualRunClaim) -> _ClaimState:
    try:
        value = _CLAIMS[id(claim)]
    except (KeyError, TypeError) as error:
        raise ValueError("residual run claim is not registered") from error
    if value.claim is not claim:
        raise ValueError("residual run claim is not registered")
    return value


def _outcome_path(attempt_path: Path) -> Path:
    suffix = "-attempt.json"
    if attempt_path.parent.name != "experiments" or \
       not attempt_path.name.endswith(suffix):
        raise ValueError("residual attempt path is invalid")
    return attempt_path.with_name(
        f"{attempt_path.name[:-len(suffix)]}-outcome.json",
    )


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


def _master(source: ContextPhase) -> tuple[str, ...]:
    master = (
        *(series for series, _ in source.training_rows),
        *(series for series, _, _ in source.evaluation_rows),
    )
    if not master or len(set(master)) != len(master):
        raise ValueError("residual source universe changed")
    return master


def _artifacts(
    claim: ResidualRunClaim, source: ContextPhase,
) -> PhaseArtifacts:
    artifacts = phase_artifacts(claim.root, claim.attempt_path, source)
    paths = (claim.attempt_path, *artifacts)
    tuple(_logical(claim.root, path, "residual path") for path in paths)
    if len(set(paths)) != len(paths) or \
       {path.parent for path in artifacts} != {claim.path}:
        raise ValueError("residual output topology is invalid")
    return artifacts


def claim_residual_run(
    root: Path, attempt_path: Path,
) -> ResidualRunClaim:
    """Claim the canonical run directory before any residual truth access."""
    run = _run_path(root, attempt_path)
    outcome = _outcome_path(attempt_path)
    _absent(outcome, "residual outcome")
    identities = _single_link_inputs(
        (attempt_path,), "residual attempt",
    )
    run_parent, run_parent_identity = _open_directory(run.parent)
    outcome_parent, outcome_parent_identity = _open_directory(outcome.parent)
    try:
        with freeze_inputs((attempt_path,)) as frozen:
            binding = _binding(root, frozen[0])
            created_identity = mkdir_nofollow(run)
            directory, directory_identity = _open_directory(run)
            try:
                if directory_identity != created_identity:
                    raise ValueError("residual run claim changed")
                os.fsync(directory)
            finally:
                os.close(directory)
            os.fsync(run_parent)
            if _directory_identity(run.parent) != run_parent_identity or \
               _directory_identity(run) != directory_identity or \
               _directory_identity(outcome.parent) != \
                    outcome_parent_identity:
                raise ValueError("residual run claim changed")
            _verify_identities(identities)
            verify_frozen(frozen)
    finally:
        os.close(outcome_parent)
        os.close(run_parent)
    claim = ResidualRunClaim(
        root, run, directory_identity, attempt_path, binding,
        identities[0][1],
    )
    _CLAIMS[id(claim)] = _ClaimState(
        claim, set(), outcome_parent_identity,
    )
    return claim


def _verify_claim(claim: ResidualRunClaim) -> None:
    _state(claim)
    if claim.root != claim.root.resolve(strict=True) or \
       claim.path != _run_path(claim.root, claim.attempt_path) or \
       _directory_identity(claim.path) != claim.directory_identity or \
       _directory_identity(_outcome_path(claim.attempt_path).parent) != \
            _state(claim).outcome_parent_identity:
        raise ValueError("residual run claim changed")
    _verify_single_link_inputs((
        (claim.attempt_path, claim.attempt_identity),
    ), "residual attempt")


def _require_phase_prefix(
    claim: ResidualRunClaim, source: ContextPhase, *, started: bool,
) -> None:
    names = tuple(name for name, _ in PHASE_BUDGETS)
    try:
        index = names.index(source.phase)
    except ValueError as error:
        raise ValueError("residual phase order changed") from error
    expected_completed = names[:index]
    expected_started = set(names[:index + int(started)])
    state = _state(claim)
    if tuple(claim._completed) != expected_completed or \
       state.started != expected_started or any(
           type(value) is not ResidualPhaseEvidence or value.phase != name
           for name, value in claim._completed.items()
       ):
        raise ValueError("residual phase order changed")


def _start(
    claim: ResidualRunClaim, source: ContextPhase,
) -> PhaseArtifacts:
    _verify_claim(claim)
    _require_phase_prefix(claim, source, started=False)
    _state(claim).started.add(source.phase)
    artifacts = _artifacts(claim, source)
    for path in artifacts:
        _absent(path, "residual phase output")
    return artifacts


def _frozen(
    values: Mapping[Path, FrozenInput], path: Path,
) -> FrozenInput:
    try:
        return values[path]
    except KeyError as error:
        raise ValueError(f"unfrozen residual input: {path}") from error


def validate_residual_ledgers(
    master: Sequence[str], source: ContextPhase, phase: ResidualPhaseInput,
    fits: Path, predictions: Path,
) -> tuple[ResidualPredictionEvidence, ...]:
    """Authenticate the exact ordered residual fit and prediction closure."""
    fit_records = read_canonical_json_lines(fits)
    validate_residual_fit_records(
        fit_records, master, source, phase,
    )
    return validate_residual_prediction_records(
        read_canonical_json_lines(predictions),
        master, source, phase, fit_records,
    )


def residual_access_value(
    attempt: FileBinding, receipt: FileBinding, source: ContextPhase,
) -> dict[str, object]:
    """Bind one durable truth access to its authenticated receipt."""
    return {
        "attempt": asdict(attempt),
        "phase": source.phase,
        "receipt": asdict(receipt),
        "schema": 1,
    }


def _evaluate_residual_phase(
    master: Sequence[str], source: ContextPhase, phase: ResidualPhaseInput,
    predictions: Sequence[ResidualPredictionEvidence],
    truth: Mapping[str, Sequence[ResidualTruthRow]],
) -> dict[str, object]:
    from tools.finalize_spy_residual import evaluate_residual_phase

    return evaluate_residual_phase(
        master, source, phase, predictions, truth,
    )


def _truth_and_evaluation(
    claim: ResidualRunClaim, attempt: ResidualAttempt,
    source: ContextPhase, phase: ResidualPhaseInput,
    read_truth: ResidualTruthReader, verify_inputs: Verify,
) -> tuple[Mapping[str, object], ResidualPhaseEvidence]:
    _require_phase_prefix(claim, source, started=True)
    artifacts = _artifacts(claim, source)
    _absent(artifacts.access, "residual truth access")
    _absent(artifacts.evaluation, "residual phase evaluation")
    paths = (
        claim.attempt_path, artifacts.fits,
        artifacts.predictions, artifacts.receipt,
    )
    identities = _single_link_inputs(paths, "residual receipt inputs")
    with freeze_inputs(paths) as frozen:
        by_path = {item.source: item for item in frozen}
        attempt_binding = _binding(
            claim.root, _frozen(by_path, claim.attempt_path),
        )
        fit_binding = _binding(
            claim.root, _frozen(by_path, artifacts.fits),
        )
        prediction_binding = _binding(
            claim.root, _frozen(by_path, artifacts.predictions),
        )
        receipt_binding = _binding(
            claim.root, _frozen(by_path, artifacts.receipt),
        )
        receipt = ResidualReceipt.parse(read_canonical_json(
            _frozen(by_path, artifacts.receipt).snapshot,
        ))
        receipt.validate(
            source, phase, attempt_binding, fit_binding,
            prediction_binding, attempt.source_tree.sha256,
            claim.directory_identity,
        )
        predictions = validate_residual_ledgers(
            _master(source), source, phase,
            _frozen(by_path, artifacts.fits).snapshot,
            _frozen(by_path, artifacts.predictions).snapshot,
        )
        access_value = residual_access_value(
            attempt_binding, receipt_binding, source,
        )

        def verify() -> None:
            verify_inputs()
            _verify_claim(claim)
            _verify_single_link_inputs(
                identities, "residual receipt inputs",
            )
            verify_frozen(frozen)

        verify()
        _write_json(
            artifacts.access, access_value, verify,
            claim.directory_identity,
        )

        def restore_access() -> None:
            try:
                _absent(artifacts.access, "residual truth access")
            except ValueError:
                return
            _write_json(
                artifacts.access, access_value, _verify_claim_only,
                claim.directory_identity,
            )

        def _verify_claim_only() -> None:
            _verify_claim(claim)

        try:
            access_identities = _single_link_inputs(
                (artifacts.access,), "residual truth access",
            )
            with freeze_inputs((artifacts.access,)) as access_frozen:
                if not _exact_json(
                    read_canonical_json(access_frozen[0].snapshot),
                    access_value,
                ):
                    raise ValueError("residual truth access changed")

                def verify_access() -> None:
                    verify()
                    _verify_single_link_inputs(
                        access_identities, "residual truth access",
                    )
                    verify_frozen(access_frozen)

                verify_access()
                truth = read_truth()
                if not isinstance(truth, Mapping):
                    raise ValueError("residual truth must be an object")
                verify_access()
                evaluation = _evaluate_residual_phase(
                    _master(source), source, phase, predictions, truth,
                )
                if not isinstance(evaluation, Mapping):
                    raise ValueError(
                        "residual phase evaluation must be an object",
                    )
                try:
                    expected = json.loads(json.dumps(
                        evaluation, allow_nan=False,
                    ))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "residual phase evaluation is not JSON",
                    ) from error
                verify_access()
                _write_json(
                    artifacts.evaluation, evaluation, verify_access,
                    claim.directory_identity,
                )
                evaluation_identities = _single_link_inputs(
                    (artifacts.evaluation,), "residual evaluation",
                )
                with freeze_inputs((
                    artifacts.evaluation,
                )) as evaluation_frozen:
                    published = read_canonical_json(
                        evaluation_frozen[0].snapshot,
                    )
                    if not _exact_json(published, expected):
                        raise ValueError(
                            "residual phase evaluation changed",
                        )

                    def verify_evaluation() -> None:
                        verify_access()
                        _verify_single_link_inputs(
                            evaluation_identities,
                            "residual evaluation",
                        )
                        verify_frozen(evaluation_frozen)

                    verify_evaluation()
                    identity_by_path = dict((
                        *identities, *access_identities,
                        *evaluation_identities,
                    ))
                    evidence = ResidualPhaseEvidence(
                        source.phase,
                        (
                            fit_binding, prediction_binding, receipt_binding,
                            _binding(claim.root, access_frozen[0]),
                            _binding(claim.root, evaluation_frozen[0]),
                        ),
                        tuple(
                            identity_by_path[path] for path in artifacts
                        ),
                        phase.source_phase_sha256,
                        residual_phase_sha256(phase),
                        attempt.source_tree.sha256,
                    )
                    evaluation = published
        except BaseException:
            restore_access()
            raise
    return evaluation, evidence


def execute_residual_phase(
    claim: ResidualRunClaim, attempt: ResidualAttempt,
    source: ContextPhase, phase: ResidualPhaseInput,
    fit_one: ResidualFitOne, predict_one: ResidualPredictOne,
    read_truth: ResidualTruthReader, verify: Verify,
) -> Mapping[str, object]:
    """Fit and predict before durably authorizing one residual truth read."""
    if not isinstance(attempt, ResidualAttempt) or \
       not isinstance(source, ContextPhase) or \
       not isinstance(phase, ResidualPhaseInput) or \
       ResidualPhaseInput.parse(asdict(phase), source) != phase or \
       not all(callable(item) for item in (
           fit_one, predict_one, read_truth, verify,
       )):
        raise ValueError("residual phase inputs are invalid")
    try:
        index = tuple(name for name, _ in PHASE_BUDGETS).index(source.phase)
    except ValueError as error:
        raise ValueError("residual phase order changed") from error
    if index >= len(attempt.phases) or attempt.phases[index] != phase:
        raise ValueError("residual phase binding changed")
    artifacts = _start(claim, source)
    master = _master(source)
    attempt_identities = (
        (claim.attempt_path, claim.attempt_identity),
    )
    with freeze_inputs((claim.attempt_path,)) as attempt_frozen:
        if _binding(claim.root, attempt_frozen[0]) != claim.attempt_binding:
            raise ValueError("residual attempt changed after its claim")

        def verify_attempt() -> None:
            verify()
            _verify_claim(claim)
            _verify_identities(attempt_identities)
            verify_frozen(attempt_frozen)

        verify_attempt()
        fitted, fit_records = {}, []
        for fit in expected_residual_fits(master, source):
            result = fit_one(fit)
            if type(result) is not tuple or len(result) != 3:
                raise ValueError("residual fit callback changed")
            fingerprint, loss, model = result
            fit_records.append(residual_fit_record(
                fit, source, phase, master, fingerprint, loss,
            ))
            fitted[fit] = model
        fit_evidence = validate_residual_fit_records(
            fit_records, master, source, phase,
        )
        verify_attempt()

        by_fit = {item.fit: item for item in fit_evidence}
        prediction_records = []
        for prediction in expected_residual_predictions(master, source):
            values = predict_one(
                prediction, fitted[prediction.fit],
            )
            if type(values) is not tuple:
                raise ValueError("residual prediction callback changed")
            prediction_records.append(residual_prediction_record(
                prediction, by_fit[prediction.fit],
                values,
            ))
        validate_residual_prediction_records(
            prediction_records, master, source, phase, fit_records,
        )
        verify_attempt()

        _write_ledger(
            artifacts.fits, fit_records, verify_attempt,
            claim.directory_identity,
        )
        _write_ledger(
            artifacts.predictions, prediction_records, verify_attempt,
            claim.directory_identity,
        )
        ledger_paths = (artifacts.fits, artifacts.predictions)
        ledger_identities = _single_link_inputs(
            ledger_paths, "residual ledgers",
        )
        with freeze_inputs(ledger_paths) as frozen:
            by_path = {item.source: item for item in frozen}
            fit_binding = _binding(
                claim.root, _frozen(by_path, artifacts.fits),
            )
            prediction_binding = _binding(
                claim.root, _frozen(by_path, artifacts.predictions),
            )
            validate_residual_ledgers(
                master, source, phase,
                _frozen(by_path, artifacts.fits).snapshot,
                _frozen(by_path, artifacts.predictions).snapshot,
            )
            receipt = ResidualReceipt(
                source.phase, claim.attempt_binding,
                fit_binding, prediction_binding,
                phase.source_phase_sha256, residual_phase_sha256(phase),
                source.evaluation_grid_sha256,
                attempt.source_tree.sha256, claim.directory_identity,
                len(fit_records), len(prediction_records),
            )

            def verify_ledgers() -> None:
                verify_attempt()
                _verify_single_link_inputs(
                    ledger_identities, "residual ledgers",
                )
                verify_frozen(frozen)

            _write_json(
                artifacts.receipt, receipt.value(), verify_ledgers,
                claim.directory_identity,
            )
            receipt_identities = _single_link_inputs(
                (artifacts.receipt,), "residual receipt",
            )
            with freeze_inputs((artifacts.receipt,)) as receipt_frozen:
                published = ResidualReceipt.parse(read_canonical_json(
                    receipt_frozen[0].snapshot,
                ))
                published.validate(
                    source, phase, claim.attempt_binding,
                    fit_binding, prediction_binding,
                    attempt.source_tree.sha256, claim.directory_identity,
                )
                verify_ledgers()
                _verify_single_link_inputs(
                    receipt_identities, "residual receipt",
                )
                verify_frozen(receipt_frozen)

    evaluation, evidence = _truth_and_evaluation(
        claim, attempt, source, phase, read_truth, verify,
    )
    claim._completed[source.phase] = evidence
    return evaluation


@contextmanager
def _completed_phase_inputs(
    claim: ResidualRunClaim, attempt: ResidualAttempt,
    sources: Sequence[ContextPhase],
) -> Iterator[
    tuple[
        FileBinding, dict[str, Mapping[str, object]],
        tuple[dict[str, object], ...], Verify,
    ]
]:
    names = tuple(name for name, _ in PHASE_BUDGETS)
    if tuple(claim._completed) != names or \
       tuple(source.phase for source in sources) != names:
        raise ValueError("residual phases are incomplete")
    artifacts = tuple(
        (source, _artifacts(claim, source))
        for source in sources
    )
    paths = (
        claim.attempt_path,
        *(path for _, values in artifacts for path in values),
    )
    identities = _single_link_inputs(
        paths, "residual terminal inputs",
    )
    with freeze_inputs(paths) as frozen:
        by_path = {item.source: item for item in frozen}
        attempt_binding = _binding(
            claim.root, _frozen(by_path, claim.attempt_path),
        )
        if attempt_binding != claim.attempt_binding:
            raise ValueError("residual terminal attempt changed")
        evaluations, inputs = {}, []
        identity_by_path = dict(identities)
        for source, values in artifacts:
            phase = attempt.phases[names.index(source.phase)]
            fits, predictions, receipt, access, evaluation = tuple(values)
            bindings = tuple(
                _binding(claim.root, _frozen(by_path, path))
                for path in values
            )
            parsed_receipt = ResidualReceipt.parse(
                read_canonical_json(_frozen(by_path, receipt).snapshot),
            )
            parsed_receipt.validate(
                source, phase, attempt_binding, bindings[0], bindings[1],
                attempt.source_tree.sha256, claim.directory_identity,
            )
            validate_residual_ledgers(
                _master(source), source, phase,
                _frozen(by_path, fits).snapshot,
                _frozen(by_path, predictions).snapshot,
            )
            if not _exact_json(
                read_canonical_json(_frozen(by_path, access).snapshot),
                residual_access_value(attempt_binding, bindings[2], source),
            ):
                raise ValueError("residual truth access changed")
            evaluations[source.phase] = read_canonical_json(
                _frozen(by_path, evaluation).snapshot,
            )
            evidence = claim._completed[source.phase]
            expected_evidence = ResidualPhaseEvidence(
                source.phase, bindings,
                tuple(identity_by_path[path] for path in values),
                phase.source_phase_sha256, residual_phase_sha256(phase),
                attempt.source_tree.sha256,
            )
            if evidence != expected_evidence:
                raise ValueError("residual phase completion changed")
            inputs.append({
                **{
                    name: asdict(binding)
                    for name, binding in zip(
                        ("fits", "predictions", "receipt",
                         "access", "evaluation"),
                        bindings, strict=True,
                    )
                },
                "phase": source.phase,
            })

        def verify() -> None:
            _verify_claim(claim)
            _verify_single_link_inputs(
                identities, "residual terminal inputs",
            )
            verify_frozen(frozen)

        verify()
        yield attempt_binding, evaluations, tuple(inputs), verify
        verify()


def publish_residual_outcome(
    claim: ResidualRunClaim, value: Mapping[str, object], verify: Verify,
) -> Path:
    """Publish one inode-bound terminal outcome outside the run directory."""
    global _TERMINAL_OUTCOME

    _verify_claim(claim)
    if not isinstance(value, Mapping) or not callable(verify):
        raise ValueError("residual outcome inputs are invalid")
    outcome = _outcome_path(claim.attempt_path)
    _absent(outcome, "residual outcome")
    active = _ACTIVE_ATTEMPT == Path(os.path.abspath(claim.attempt_path))

    def committed(
        binding: ExclusiveTemp, digest: str, size: int,
    ) -> None:
        global _TERMINAL_OUTCOME

        _TERMINAL_OUTCOME = _TerminalOutcome(
            outcome, binding.identity,
            _state(claim).outcome_parent_identity,
            binding.mode, size, digest,
        )

    _write_json(
        outcome, value, verify,
        _state(claim).outcome_parent_identity,
        accept_committed_error=True,
        on_committed=committed if active else None,
    )
    if active and _TERMINAL_OUTCOME is None:
        raise OSError("terminal residual outcome was not authenticated")
    return outcome


def _finalize_residual_attempt(
    claim: ResidualRunClaim, attempt: ResidualAttempt,
    sources: Sequence[ContextPhase], verify_inputs: Verify,
) -> Mapping[str, object]:
    from tools.finalize_spy_residual import finalize_residual_run

    with _completed_phase_inputs(
        claim, attempt, sources,
    ) as (attempt_binding, evaluations, inputs, verify):
        value = finalize_residual_run(
            attempt_binding, attempt.phases, evaluations, inputs,
            attempt.source_tree.sha256,
        )

        def verify_terminal() -> None:
            verify_inputs()
            verify()

        publish_residual_outcome(claim, value, verify_terminal)
    return value


def _attempt_path(path: Path) -> tuple[Path, Path]:
    absolute = path if path.is_absolute() else ROOT / path
    try:
        lexical = Path(os.path.abspath(absolute))
        resolved = absolute.resolve(strict=True)
        if lexical != resolved:
            raise ValueError("residual attempt must not contain symlinks")
        _single_link_inputs((resolved,), "residual attempt")
        logical = resolved.relative_to(ROOT.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError(
            "residual attempt must be inside the repository",
        ) from error
    return absolute, logical


def _read_source_context() -> ContextAttempt:
    binding = RESIDUAL_SOURCE["context_attempt"]
    path = ROOT / binding.path
    identities = _single_link_inputs(
        (path,), "source context attempt",
    )
    with freeze_inputs((path,)) as frozen:
        if frozen[0].sha256 != binding.sha256:
            raise ValueError("source context attempt changed")
        context = ContextAttempt.read(
            frozen[0].snapshot, Path(binding.path), ROOT,
        )
        _verify_identities(identities)
        verify_frozen(frozen)
        return context


def read_residual_attempt(path: Path) -> tuple[ResidualAttempt, ContextAttempt]:
    """Read one canonical attempt and its fixed source context."""
    _require_package_alias()
    absolute, logical = _attempt_path(path)
    context = _read_source_context()
    identities = _single_link_inputs(
        (absolute,), "residual attempt",
    )
    with freeze_inputs((absolute,)) as frozen:
        attempt = ResidualAttempt.read(
            frozen[0].snapshot, logical, ROOT, context,
        )
        _verify_identities(identities)
        verify_frozen(frozen)
    return attempt, context


def _validate_environment(attempt: ResidualAttempt) -> None:
    actual = dict(os.environ)
    expected = dict(attempt.environment)
    if any(
        actual.get(name) != value for name, value in expected.items()
    ) or set(actual) - set(expected) - {
        "LC_CTYPE", "__CF_USER_TEXT_ENCODING",
    }:
        raise ValueError("residual runner environment changed")


def _validate_controller(attempt: ResidualAttempt) -> None:
    _require_isolated_execution(bootstrapped=True)
    _require_exact_launch()
    if tuple(sys.argv) != attempt.runner_argv or \
       Path(sys.executable).resolve(strict=True) != Path(
           attempt.torch_probe.python.path,
       ).resolve(strict=True):
        raise ValueError("residual runner command changed")
    _validate_environment(attempt)
    attempt.primary_python.validate_live("residual primary Python")
    attempt.torch_probe.python.validate_live("residual Torch Python")
    if source_tree(
        Path(attempt.torch_probe.package_tree.root),
    ) != attempt.torch_probe.package_tree or \
       selected_source_tree(ROOT, RESIDUAL_SOURCE_PATHS) != \
            attempt.source_tree:
        raise ValueError("residual runner source or Torch package changed")


def _prepare_phase(
    context: ContextAttempt, source: ContextPhase,
    phase: ResidualPhaseInput, lease: object,
) -> tuple[object, ResidualTruthReader]:
    from tools.arm_spy_residual import ResidualLease
    from tools.spy_residual_controller import prepare_residual_phase

    if not isinstance(lease, ResidualLease):
        raise ValueError("residual lease changed")
    benchmark = dict(lease.benchmark)
    try:
        spy = benchmark["spy_csv"]
    except KeyError as error:
        raise ValueError("residual SPY snapshot is missing") from error
    return prepare_residual_phase(
        context, source, phase, lease.context, spy,
    )


def _failure_value(
    claim: ResidualRunClaim, stage: str,
) -> dict[str, object]:
    return {
        "attempt": asdict(claim.attempt_binding),
        "schema": 1,
        "stage": stage,
        "status": "integrity-failure",
    }


def execute_residual_attempt(path: Path) -> Mapping[str, object]:
    """Run both residual phases and publish exactly one terminal outcome."""
    attempt, context = read_residual_attempt(path)
    _validate_controller(attempt)
    from tools.arm_spy_residual import authenticate_residual_attempt
    from tools.run_universe_scaling import _expose_torch_package

    claim: ResidualRunClaim | None = None
    stage = "authenticate"
    try:
        with authenticate_residual_attempt(attempt) as lease:
            _expose_torch_package(Path(
                attempt.torch_probe.package_tree.root,
            ))
            _require_package_alias()
            from tools.spy_residual_runtime import ResidualRuntime
            import torch

            claim = claim_residual_run(
                ROOT, ROOT / attempt.attempt_path,
            )
            for source, phase in zip(
                context.phases, attempt.phases, strict=True,
            ):
                stage = source.phase
                prepared, read_truth = _prepare_phase(
                    context, source, phase, lease,
                )
                runtime = ResidualRuntime(
                    prepared, torch.device("cpu"),
                    attempt.source_tree.sha256,
                )
                execute_residual_phase(
                    claim, attempt, source, phase,
                    runtime.fit_one, runtime.predict_one,
                    read_truth, lease,
                )
            stage = "finalize"
            if claim is None:
                raise ValueError("residual run was not claimed")
            return _finalize_residual_attempt(
                claim, attempt, context.phases, lease,
            )
    except BaseException:
        if claim is not None:
            try:
                publish_residual_outcome(
                    claim, _failure_value(claim, stage),
                    lambda: _verify_claim(claim),
                )
            except (OSError, ValueError):
                pass
        raise


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    return parser.parse_args(argv)


def _run(path: Path) -> int:
    """Execute once, mapping the first signal to its shell exit status."""
    global _ACTIVE_ATTEMPT, _SIGNAL_NUMBER, _TERMINAL_OUTCOME

    _ACTIVE_ATTEMPT = Path(os.path.abspath(path))
    _TERMINAL_OUTCOME = None
    _SIGNAL_NUMBER = None
    previous = []
    failure: BaseException | None = None
    completed = False

    def interrupt(number: int, _frame: object) -> None:
        global _SIGNAL_NUMBER

        if _TERMINAL_OUTCOME is None and _SIGNAL_NUMBER is None:
            _SIGNAL_NUMBER = number
            raise Interrupted(number)

    def remember(error: BaseException) -> None:
        nonlocal failure
        if failure is None:
            failure = error

    try:
        for number in SIGNALS:
            previous.append((number, signal.getsignal(number)))
            signal.signal(number, interrupt)
        execute_residual_attempt(path)
        completed = True
    except BaseException as error:
        remember(error)
    if completed and _TERMINAL_OUTCOME is None:
        remember(ValueError(
            "residual run returned without a terminal outcome",
        ))
    elif _TERMINAL_OUTCOME is not None:
        try:
            _verify_terminal_outcome(_TERMINAL_OUTCOME)
        except BaseException as error:
            remember(error)
    for number, handler in previous:
        try:
            signal.signal(number, handler)
        except Interrupted as error:
            try:
                signal.signal(number, handler)
            except BaseException as retry_error:
                remember(retry_error)
            else:
                if not completed:
                    remember(error)
        except BaseException as error:
            remember(error)
    _ACTIVE_ATTEMPT = _TERMINAL_OUTCOME = None
    _SIGNAL_NUMBER = None
    if isinstance(failure, Interrupted):
        return 128 + failure.number
    if failure is not None:
        raise failure
    return 0


def main() -> None:
    try:
        code = _run(parse_args().attempt)
    except (
        KeyError, OSError, OverflowError, TypeError, UnicodeError, ValueError,
    ) as error:
        print(f"residual runner error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
