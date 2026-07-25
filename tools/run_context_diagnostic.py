#!/usr/bin/env python3
"""Enforce one-shot, receipt-gated context diagnostic phases."""

from __future__ import annotations

import sys

_BOOTSTRAP_FLAGS = ("-I", "-S", "-B")
_BOOTSTRAP_CACHE_PREFIX: str | None = None
_PACKAGE_NAME = "tools.run_context_diagnostic"


def _require_isolated_execution(*, bootstrapped: bool = False) -> None:
    flags = sys.flags
    if not flags.isolated or not getattr(flags, "safe_path", False) or \
       not flags.no_user_site or not flags.no_site or \
       not flags.dont_write_bytecode or \
       not flags.ignore_environment or not sys.dont_write_bytecode:
        raise ValueError(
            "context runner requires isolated bytecode-free Python",
        )
    if bootstrapped and (
        _BOOTSTRAP_CACHE_PREFIX is None or
        sys.pycache_prefix != _BOOTSTRAP_CACHE_PREFIX
    ):
        raise ValueError("context runner requires authenticated bootstrap")


def _require_exact_launch(*, pristine: bool = False) -> None:
    if pristine and ("ctypes" in sys.modules or "_ctypes" in sys.modules):
        raise ValueError("context runner launch inspection is already loaded")

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
        raise ValueError("context runner requires the exact bound launch")


def _register_package_alias() -> None:
    module = sys.modules[__name__]
    if _PACKAGE_NAME in sys.modules:
        raise ValueError("context runner package alias is unsafe")
    sys.modules[_PACKAGE_NAME] = module


def _require_package_alias() -> None:
    if sys.modules.get(_PACKAGE_NAME) is not sys.modules[__name__]:
        raise ValueError("context runner package alias changed")


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
            f"compose-mini-context-runner-{os.urandom(32).hex()}",
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
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TextIO
import argparse
import hashlib
import json
import os
import signal
import stat

from tools.arm_context_diagnostic import ContextLease, authenticate_context_attempt
from tools.context_diagnostic_contract import (
    CONTEXT_SOURCE_PATHS, TARGET_PHASES, ContextAttempt, ContextFit, ContextPhase,
    ContextPrediction, ContextPredictionEvidence,
    ContextReceipt,
    context_family_sha256, context_fit_record, context_prediction_record,
    context_phase_sha256, context_provenance_id,
    expected_context_fits, expected_context_predictions,
    validate_context_fit_records, validate_context_prediction_records,
)
from tools.files import (
    ExclusiveTemp, FrozenInput, _owns_entry, exclusive_text, freeze_inputs,
    verify_frozen,
)
from tools.panel_contract import (
    FileBinding, _absent, _directory_identity, _open_directory,
    _exact_json, _regular_identity, _regular_inputs, _sha256,
    _verify_identities, mkdir_nofollow, read_canonical_json,
    read_canonical_json_lines, selected_source_tree, source_tree,
)

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)

FitOne = Callable[[ContextFit], tuple[str, float, object]]
PredictOne = Callable[[ContextPrediction, object], Sequence[float]]
Verify = Callable[[], None]
ReadTruth = Callable[
    [Sequence[ContextPredictionEvidence]], Mapping[str, object],
]


class Interrupted(Exception):
    """Carry the first process signal to the CLI exit boundary."""

    def __init__(self, number: int) -> None:
        self.number = number


class PublicationCommitted(OSError):
    """Report an error after an owned output reached its public name."""


@dataclass(frozen=True, slots=True)
class _TerminalOutcome:
    path: Path
    identity: tuple[int, int]
    directory_identity: tuple[int, int]
    mode: int
    size: int
    sha256: str


_ACTIVE_ATTEMPT: Path | None = None
_TERMINAL_OUTCOME: _TerminalOutcome | None = None
_SIGNAL_NUMBER: int | None = None


@dataclass(frozen=True, slots=True)
class PhaseArtifacts:
    """Name the ordered evidence, label access, and phase evaluation."""

    fits: Path
    predictions: Path
    receipt: Path
    access: Path
    evaluation: Path

    def __iter__(self) -> Iterator[Path]:
        return iter((
            self.fits, self.predictions, self.receipt,
            self.access, self.evaluation,
        ))


@dataclass(frozen=True, slots=True)
class PhaseEvidence:
    """Retain the exact phase files until the terminal decision is durable."""

    phase: str
    bindings: tuple[FileBinding, ...]
    identities: tuple[tuple[int, int], ...]
    source_failure_sha256: str
    config_sha256: str
    source_tree_sha256: str


@dataclass(frozen=True, slots=True)
class RunClaim:
    """Authorize one process to use one durable canonical run directory."""

    root: Path
    path: Path
    attempt: Path
    attempt_binding: FileBinding
    attempt_identity: tuple[int, int]
    directory_identity: tuple[int, int]
    _started: set[str] = field(
        default_factory=set, compare=False, repr=False,
    )
    _completed: dict[str, PhaseEvidence] = field(
        default_factory=dict, compare=False, repr=False,
    )

    @property
    def started(self) -> frozenset[str]:
        """Expose phase starts without granting lifecycle mutation."""
        return frozenset(self._started)

    @property
    def completed(self) -> Mapping[str, PhaseEvidence]:
        """Expose completed evidence through a read-only live view."""
        return MappingProxyType(self._completed)

    @property
    def armed(self) -> bool:
        """Report the immutable module-owned execution mode."""
        return _claim_mode(self)


_CLAIMS: dict[int, tuple[RunClaim, bool]] = {}


def _claim_mode(claim: RunClaim) -> bool:
    try:
        registered, armed = _CLAIMS[id(claim)]
    except (KeyError, TypeError) as error:
        raise ValueError("context run claim is not registered") from error
    if registered is not claim:
        raise ValueError("context run claim is not registered")
    return armed


def _registered(claim: object) -> bool:
    if not isinstance(claim, RunClaim):
        return False
    try:
        _claim_mode(claim)
    except ValueError:
        return False
    return True


def _logical(root: Path, path: Path, label: str) -> str:
    if not isinstance(path, Path) or path != Path(os.path.abspath(path)):
        raise ValueError(f"{label} must be an absolute normalized path")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} is outside the repository") from error
    if relative == Path(".") or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ValueError(f"{label} is not normalized")
    return relative.as_posix()


def _run_path(root: Path, attempt: Path) -> Path:
    if not isinstance(root, Path) or root != root.resolve(strict=True):
        raise ValueError("context repository is invalid")
    _logical(root, attempt, "context attempt")
    suffix = "-attempt.json"
    if attempt.parent != root / "experiments" or \
       not attempt.name.endswith(suffix):
        raise ValueError("context attempt path is invalid")
    run_id = attempt.name[:-len(suffix)]
    if run_id in ("", ".", "..") or Path(run_id).name != run_id:
        raise ValueError("context run id is invalid")
    return root / "reports" / run_id


def phase_artifacts(
    root: Path, attempt: Path, phase: ContextPhase,
) -> PhaseArtifacts:
    """Derive every phase output from the attempt's unique run id."""
    if not isinstance(phase, ContextPhase):
        raise ValueError("context phase is invalid")
    run = _run_path(root, attempt)
    prefix = run / phase.phase
    return PhaseArtifacts(
        prefix.with_name(f"{prefix.name}-fits.jsonl"),
        prefix.with_name(f"{prefix.name}-predictions.jsonl"),
        prefix.with_name(f"{prefix.name}-receipt.json"),
        prefix.with_name(f"{prefix.name}-truth-access.json"),
        prefix.with_name(f"{prefix.name}-evaluation.json"),
    )


def claim_run(root: Path, attempt: Path) -> RunClaim:
    """Claim the one canonical run directory for an immutable attempt."""
    run = _run_path(root, attempt)
    identities = _regular_inputs((attempt,))
    parent, parent_identity = _open_directory(run.parent)
    try:
        with freeze_inputs((attempt,)) as frozen:
            binding = _binding(root, frozen[0])
            try:
                ContextAttempt.read(
                    frozen[0].snapshot, attempt.relative_to(root), root,
                )
            except ValueError:
                armed = False
            else:
                armed = True
            created_identity = mkdir_nofollow(run)
            directory, directory_identity = _open_directory(run)
            try:
                if directory_identity != created_identity:
                    raise ValueError("context run claim changed")
                os.fsync(directory)
            finally:
                os.close(directory)
            os.fsync(parent)
            if _directory_identity(run.parent) != parent_identity or \
               _directory_identity(run) != directory_identity:
                raise ValueError("context run claim changed")
            _verify_identities(identities)
            verify_frozen(frozen)
    finally:
        os.close(parent)
    claim = RunClaim(
        root, run, attempt, binding, identities[0][1],
        directory_identity,
    )
    _CLAIMS[id(claim)] = (claim, armed)
    return claim


def _artifacts(
    root: Path, attempt: Path, phase: ContextPhase,
) -> PhaseArtifacts:
    artifacts = phase_artifacts(root, attempt, phase)
    values = (attempt, *artifacts)
    tuple(_logical(root, path, "context path") for path in values)
    if len(set(values)) != len(values) or \
       {path.parent for path in artifacts} != {_run_path(root, attempt)}:
        raise ValueError("context output topology is invalid")
    return artifacts


def _require_phase_prefix(
    claim: RunClaim, phase: ContextPhase, *, started: bool,
) -> None:
    try:
        index = TARGET_PHASES.index(phase.phase)
    except ValueError as error:
        raise ValueError("context phase order changed") from error
    completed = TARGET_PHASES[:index]
    begun = TARGET_PHASES[:index + int(started)]
    if tuple(claim._completed) != completed or \
       claim._started != set(begun) or any(
           type(evidence) is not PhaseEvidence or evidence.phase != name
           for name, evidence in claim._completed.items()
       ):
        raise ValueError("context phase order changed")


def _start(claim: RunClaim, phase: ContextPhase) -> PhaseArtifacts:
    if not _registered(claim) or \
       not isinstance(phase, ContextPhase) or \
       claim.path != _run_path(claim.root, claim.attempt) or \
       _directory_identity(claim.path) != claim.directory_identity:
        raise ValueError("context run claim is invalid")
    _verify_identities(((claim.attempt, claim.attempt_identity),))
    _require_phase_prefix(claim, phase, started=False)
    claim._started.add(phase.phase)
    artifacts = _artifacts(claim.root, claim.attempt, phase)
    for path in artifacts:
        _absent(path, "context phase output")
    return artifacts


def _publish_text(
    path: Path, write: Callable[[TextIO], None], verify: Verify,
    directory_identity: tuple[int, int], *,
    accept_committed_error: bool = False,
    on_committed: Callable[[ExclusiveTemp], None] | None = None,
) -> ExclusiveTemp:
    """Publish into the claimed directory and clean only an owned temp."""
    directory, identity = _open_directory(path.parent)
    if identity != directory_identity:
        os.close(directory)
        raise ValueError("context output directory changed")
    temporary: ExclusiveTemp | None = None
    completed = False
    marked = False

    def capture(value: ExclusiveTemp) -> None:
        nonlocal temporary
        temporary = value

    def before_link(value: ExclusiveTemp) -> None:
        if value != temporary or \
           _directory_identity(path.parent) != directory_identity:
            raise OSError("context output temporary changed")
        verify()

    def committed() -> bool:
        for retry in range(2):
            try:
                return temporary is not None and \
                    _regular_identity(path) == temporary.identity and \
                    _directory_identity(path.parent) == directory_identity
            except Interrupted:
                if retry:
                    raise
            except (OSError, ValueError):
                return False
        return False

    def mark_committed() -> None:
        nonlocal marked
        if marked or on_committed is None:
            return
        for retry in range(2):
            try:
                if temporary is None:
                    raise OSError(
                        "context output temporary was not created",
                    )
                on_committed(temporary)
                marked = True
                return
            except Interrupted:
                if retry:
                    raise

    failure: BaseException | None = None
    try:
        exclusive_text(
            path, write, directory,
            before_link_with_temp=before_link,
            on_temp_created=capture,
        )
        os.fsync(directory)
        if not committed():
            raise OSError("context output changed after publication")
        mark_committed()
        completed = True
    except BaseException as error:
        failure = error
    try:
        # Never remove a public output or a foreign replacement.
        if not completed and temporary is not None and \
           _owns_entry(directory, temporary, (1,)):
            os.unlink(temporary.name, dir_fd=directory)
            os.fsync(directory)
    except BaseException as error:
        failure = error if failure is None else failure
    try:
        os.close(directory)
    except BaseException as error:
        failure = error if failure is None else failure
    if failure is None:
        if temporary is None:
            raise OSError("context output temporary was not created")
        return temporary
    if committed():
        mark_committed()
        if accept_committed_error:
            if temporary is None:
                raise OSError("context output temporary was not created")
            return temporary
        raise PublicationCommitted(
            "context output committed before publication failed",
        ) from failure
    raise failure


def _write_ledger(
    path: Path, records: Sequence[Mapping[str, object]], verify: Verify,
    directory_identity: tuple[int, int],
) -> None:
    def write(file: TextIO) -> None:
        for record in records:
            json.dump(record, file, allow_nan=False, sort_keys=True)
            file.write("\n")

    _publish_text(path, write, verify, directory_identity)


def _write_json(
    path: Path, value: Mapping[str, object], verify: Verify,
    directory_identity: tuple[int, int], *,
    accept_committed_error: bool = False,
    on_committed: Callable[[ExclusiveTemp, str, int], None] | None = None,
) -> tuple[ExclusiveTemp, str, int]:
    text = json.dumps(
        value, allow_nan=False, indent=2, sort_keys=True,
    ) + "\n"
    payload = text.encode()
    digest, size = hashlib.sha256(payload).hexdigest(), len(payload)

    def write(file: TextIO) -> None:
        file.write(text)

    def commit(binding: ExclusiveTemp) -> None:
        if on_committed is not None:
            on_committed(binding, digest, size)

    published = _publish_text(
        path, write, verify, directory_identity,
        accept_committed_error=accept_committed_error,
        on_committed=commit if on_committed is not None else None,
    )
    return published, digest, size


def _verify_terminal_outcome(value: _TerminalOutcome) -> None:
    directory, directory_identity = _open_directory(value.path.parent)
    descriptor = None
    try:
        if directory_identity != value.directory_identity:
            raise ValueError("terminal context directory changed")
        descriptor = os.open(
            value.path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK,
            dir_fd=directory,
        )

        def state(metadata: os.stat_result) -> tuple[object, ...]:
            return (
                (metadata.st_dev, metadata.st_ino),
                stat.S_IFMT(metadata.st_mode),
                metadata.st_nlink,
                stat.S_IMODE(metadata.st_mode),
                metadata.st_size,
            )

        expected = (
            value.identity, stat.S_IFREG, 1, value.mode, value.size,
        )
        if state(os.fstat(descriptor)) != expected:
            raise ValueError("terminal context outcome changed")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 16):
            digest.update(chunk)
        public = os.stat(
            value.path.name, dir_fd=directory, follow_symlinks=False,
        )
        if state(os.fstat(descriptor)) != expected or \
           state(public) != expected or digest.hexdigest() != value.sha256 or \
           _directory_identity(value.path.parent) != \
                value.directory_identity:
            raise ValueError("terminal context outcome changed")
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            os.close(directory)


def _frozen(
    values: Mapping[Path, FrozenInput], path: Path,
) -> FrozenInput:
    try:
        return values[path]
    except KeyError as error:
        raise ValueError(f"unfrozen context input: {path}") from error


def _binding(root: Path, frozen: FrozenInput) -> FileBinding:
    return FileBinding(
        _logical(root, frozen.source, "context binding"), frozen.sha256,
    )


def validate_context_ledgers(
    master: Sequence[str], phase: ContextPhase,
    fit_path: Path, prediction_path: Path,
    source_failure_sha256: str, config_sha256: str,
) -> tuple[ContextPredictionEvidence, ...]:
    fits = read_canonical_json_lines(fit_path)
    validate_context_fit_records(
        fits, master, phase,
        source_failure_sha256, config_sha256,
    )
    return validate_context_prediction_records(
        read_canonical_json_lines(prediction_path), master, phase, fits,
        source_failure_sha256, config_sha256,
    )


def context_access_value(
    attempt: FileBinding, receipt: FileBinding, phase: ContextPhase,
) -> dict[str, object]:
    return {
        "attempt": {"path": attempt.path, "sha256": attempt.sha256},
        "phase": phase.phase,
        "receipt": {"path": receipt.path, "sha256": receipt.sha256},
        "schema": 1,
    }


def _authorized_truth_evidence(
    claim: RunClaim, master: Sequence[str], phase: ContextPhase,
    source_failure_sha256: str, config_sha256: str,
    source_tree_sha256: str, reader: ReadTruth,
) -> tuple[Mapping[str, object], PhaseEvidence]:
    """Consume one truth access and durably publish its phase evaluation."""
    if not _registered(claim) or \
       not isinstance(phase, ContextPhase) or \
       claim.path != _run_path(claim.root, claim.attempt) or \
       _directory_identity(claim.path) != claim.directory_identity or \
       phase.phase in claim._completed:
        raise ValueError("context run claim is invalid")
    _require_phase_prefix(claim, phase, started=True)
    root, attempt = claim.root, claim.attempt
    artifacts = _artifacts(root, attempt, phase)
    _absent(artifacts.access, "context truth access")
    _absent(artifacts.evaluation, "context phase evaluation")
    hashes = tuple(map(
        _sha256,
        (source_failure_sha256, config_sha256, source_tree_sha256),
        ("source failure", "context config", "context source tree"),
    ))
    paths = (attempt, artifacts.fits, artifacts.predictions, artifacts.receipt)
    identities = _regular_inputs(paths)
    with freeze_inputs(paths) as frozen:
        by_path = {item.source: item for item in frozen}
        attempt_binding = _binding(root, _frozen(by_path, attempt))
        fit_binding = _binding(root, _frozen(by_path, artifacts.fits))
        prediction_binding = _binding(
            root, _frozen(by_path, artifacts.predictions),
        )
        receipt = ContextReceipt.parse(read_canonical_json(
            _frozen(by_path, artifacts.receipt).snapshot,
        ))
        receipt_binding = _binding(
            root, _frozen(by_path, artifacts.receipt),
        )
        access_value = context_access_value(
            attempt_binding, receipt_binding, phase,
        )
        run_identity = _directory_identity(artifacts.receipt.parent)
        receipt.validate(
            phase, attempt_binding, fit_binding, prediction_binding,
            hashes[2], run_identity,
        )
        predictions = validate_context_ledgers(
            master, phase,
            _frozen(by_path, artifacts.fits).snapshot,
            _frozen(by_path, artifacts.predictions).snapshot,
            hashes[0], hashes[1],
        )

        def verify() -> None:
            if _directory_identity(artifacts.receipt.parent) != run_identity:
                raise ValueError("context run directory changed")
            _verify_identities(identities)
            verify_frozen(frozen)

        verify()
        _write_json(
            artifacts.access, access_value, verify, run_identity,
        )

        def verify_run() -> None:
            if _directory_identity(artifacts.access.parent) != run_identity:
                raise ValueError("context run directory changed")

        def restore_access() -> None:
            # A callback cannot make consumed truth retryable by unlinking it.
            try:
                _absent(artifacts.access, "context truth access")
            except ValueError:
                return
            _write_json(
                artifacts.access, access_value, verify_run, run_identity,
            )

        try:
            access_identities = _regular_inputs((artifacts.access,))
            with freeze_inputs((artifacts.access,)) as access_frozen:
                if not _exact_json(
                    read_canonical_json(access_frozen[0].snapshot),
                    access_value,
                ):
                    raise ValueError("context truth access changed")

                def verify_access() -> None:
                    verify()
                    _verify_identities(access_identities)
                    verify_frozen(access_frozen)

                verify_access()
                evaluation = reader(predictions)
                if not isinstance(evaluation, Mapping):
                    raise ValueError(
                        "context phase evaluation must be an object",
                    )
                try:
                    expected_evaluation = json.loads(json.dumps(
                        evaluation, allow_nan=False,
                    ))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "context phase evaluation is not JSON",
                    ) from error
                verify_access()
                _write_json(
                    artifacts.evaluation, evaluation,
                    verify_access, run_identity,
                )
                evaluation_identities = _regular_inputs((
                    artifacts.evaluation,
                ))
                with freeze_inputs((
                    artifacts.evaluation,
                )) as evaluation_frozen:
                    published = read_canonical_json(
                        evaluation_frozen[0].snapshot,
                    )
                    if not _exact_json(published, expected_evaluation):
                        raise ValueError(
                            "context phase evaluation changed",
                        )

                    def verify_evaluation() -> None:
                        verify_access()
                        _verify_identities(evaluation_identities)
                        verify_frozen(evaluation_frozen)

                    verify_evaluation()
                    identity_by_path = dict((
                        *identities, *access_identities,
                        *evaluation_identities,
                    ))
                    bindings = (
                        fit_binding, prediction_binding, receipt_binding,
                        _binding(root, access_frozen[0]),
                        _binding(root, evaluation_frozen[0]),
                    )
                    completion = PhaseEvidence(
                        phase.phase, bindings,
                        tuple(
                            identity_by_path[path] for path in artifacts
                        ),
                        *hashes,
                    )
                    evaluation = published
        except BaseException:
            restore_access()
            raise
    return evaluation, completion


def read_authorized_truth(
    claim: RunClaim, master: Sequence[str], phase: ContextPhase,
    source_failure_sha256: str, config_sha256: str,
    source_tree_sha256: str, reader: ReadTruth,
) -> Mapping[str, object]:
    """Publish authorized truth and retain its live completion evidence."""
    if not _registered(claim) or _claim_mode(claim):
        raise ValueError("armed context truth requires authentication")
    evaluation, evidence = _authorized_truth_evidence(
        claim, master, phase, source_failure_sha256, config_sha256,
        source_tree_sha256, reader,
    )
    claim._completed[phase.phase] = evidence
    return evaluation


def publish_context_outcome(
    claim: RunClaim, value: Mapping[str, object], verify: Verify,
) -> Path:
    """Atomically close one live run while its phase evidence is frozen."""
    global _TERMINAL_OUTCOME

    if not _registered(claim) or \
       not isinstance(value, Mapping) or \
       claim.path != _run_path(claim.root, claim.attempt) or \
       _directory_identity(claim.path) != claim.directory_identity:
        raise ValueError("context outcome claim is invalid")
    outcome = claim.path / "outcome.json"
    _absent(outcome, "context outcome")
    active = _ACTIVE_ATTEMPT == Path(os.path.abspath(claim.attempt))

    def committed(
        binding: ExclusiveTemp, digest: str, size: int,
    ) -> None:
        global _TERMINAL_OUTCOME

        _TERMINAL_OUTCOME = _TerminalOutcome(
            outcome, binding.identity, claim.directory_identity,
            binding.mode, size, digest,
        )

    _write_json(
        outcome, value, verify, claim.directory_identity,
        accept_committed_error=True,
        on_committed=committed if active else None,
    )
    if active and _TERMINAL_OUTCOME is None:
        raise OSError("terminal context outcome was not authenticated")
    return outcome


def _execute_started_phase(
    artifacts: PhaseArtifacts,
    claim: RunClaim, master: Sequence[str], phase: ContextPhase,
    source_failure_sha256: str, config_sha256: str,
    source_tree_sha256: str, fit_one: FitOne,
    predict_one: PredictOne, truth: ReadTruth, verify_inputs: Verify,
) -> tuple[Mapping[str, object], PhaseEvidence]:
    root, attempt = claim.root, claim.attempt
    source = _sha256(source_failure_sha256, "source failure")
    config = _sha256(config_sha256, "context config")
    source_tree = _sha256(source_tree_sha256, "context source tree")
    attempt_identities = ((attempt, claim.attempt_identity),)
    with freeze_inputs((attempt,)) as attempt_frozen:
        attempt_binding = _binding(root, attempt_frozen[0])
        if attempt_binding != claim.attempt_binding:
            raise ValueError("context attempt changed after the run claim")

        def verify_attempt() -> None:
            verify_inputs()
            if _directory_identity(claim.path) != claim.directory_identity:
                raise ValueError("context run directory changed")
            _verify_identities(attempt_identities)
            verify_frozen(attempt_frozen)

        verify_attempt()
        fitted, fit_records = {}, []
        for fit in expected_context_fits(master, phase):
            state, loss, model = fit_one(fit)
            fit_records.append(context_fit_record(
                fit, context_provenance_id(fit, phase, source, config),
                state, loss,
            ))
            fitted[fit] = model
        fit_evidence = validate_context_fit_records(
            fit_records, master, phase, source, config,
        )
        verify_attempt()

        prediction_records = []
        by_fit = {item.fit: item for item in fit_evidence}
        for prediction in expected_context_predictions(master, phase):
            prediction_records.append(context_prediction_record(
                prediction, by_fit[prediction.fit],
                predict_one(prediction, fitted[prediction.fit]),
            ))
        validate_context_prediction_records(
            prediction_records, master, phase, fit_records, source, config,
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
        ledger_identities = _regular_inputs(ledger_paths)
        with freeze_inputs(ledger_paths) as frozen:
            by_path = {item.source: item for item in frozen}
            fit_binding = _binding(root, _frozen(by_path, artifacts.fits))
            prediction_binding = _binding(
                root, _frozen(by_path, artifacts.predictions),
            )
            validate_context_ledgers(
                master, phase,
                _frozen(by_path, artifacts.fits).snapshot,
                _frozen(by_path, artifacts.predictions).snapshot,
                source, config,
            )
            receipt = ContextReceipt(
                phase.phase, attempt_binding, fit_binding,
                prediction_binding, phase.evaluation_grid_sha256,
                context_family_sha256(), context_phase_sha256(phase),
                source_tree, claim.directory_identity,
                len(fit_evidence), len(prediction_records),
            )

            def verify() -> None:
                verify_attempt()
                _verify_identities(ledger_identities)
                verify_frozen(frozen)

            _write_json(
                artifacts.receipt, receipt.value(), verify,
                claim.directory_identity,
            )
            verify()
    return _authorized_truth_evidence(
        claim, master, phase, source, config, source_tree, truth,
    )


def execute_phase(
    claim: RunClaim, master: Sequence[str], phase: ContextPhase,
    source_failure_sha256: str, config_sha256: str,
    source_tree_sha256: str, fit_one: FitOne,
    predict_one: PredictOne, truth: ReadTruth,
) -> Mapping[str, object]:
    """Fit, predict, publish, and only then authorize one phase's truth."""
    if not _registered(claim) or _claim_mode(claim):
        raise ValueError("armed context execution requires authentication")
    evaluation, evidence = _execute_started_phase(
        _start(claim, phase), claim, master, phase,
        source_failure_sha256, config_sha256, source_tree_sha256,
        fit_one, predict_one, truth, lambda: None,
    )
    claim._completed[phase.phase] = evidence
    return evaluation


def _validate_completed_prefix(
    claim: RunClaim, attempt: ContextAttempt,
) -> None:
    """Require every skipped phase to have its complete live evidence."""
    names = tuple(claim._completed)
    if names != TARGET_PHASES[:len(names)] or \
       len(names) > len(attempt.phases):
        raise ValueError("armed context phase prefix changed")
    provenance = (
        attempt.source_binding("failure").sha256,
        attempt.config.sha256,
        attempt.source_tree.sha256,
    )
    for phase in attempt.phases[:len(names)]:
        evidence = claim._completed[phase.phase]
        artifacts = _artifacts(claim.root, claim.attempt, phase)
        paths = tuple(artifacts)
        if type(evidence) is not PhaseEvidence or \
           evidence.phase != phase.phase or \
           (
               evidence.source_failure_sha256,
               evidence.config_sha256,
               evidence.source_tree_sha256,
           ) != provenance or \
           len(evidence.bindings) != len(paths) or \
           len(evidence.identities) != len(paths):
            raise ValueError("armed context phase evidence changed")
        identities = _regular_inputs(paths)
        if tuple(item for _, item in identities) != evidence.identities:
            raise ValueError("armed context phase identity changed")
        with freeze_inputs(paths) as frozen:
            by_path = dict(zip(paths, frozen, strict=True))
            bindings = tuple(
                _binding(claim.root, by_path[path]) for path in paths
            )
            if bindings != evidence.bindings:
                raise ValueError("armed context phase binding changed")
            fits, predictions, receipt, access, evaluation = paths
            parsed = ContextReceipt.parse(read_canonical_json(
                by_path[receipt].snapshot,
            ))
            parsed.validate(
                phase, claim.attempt_binding, bindings[0], bindings[1],
                provenance[2], claim.directory_identity,
            )
            validate_context_ledgers(
                attempt.master, phase, by_path[fits].snapshot,
                by_path[predictions].snapshot,
                provenance[0], provenance[1],
            )
            if not _exact_json(
                read_canonical_json(by_path[access].snapshot),
                context_access_value(
                    claim.attempt_binding, bindings[2], phase,
                ),
            ) or not isinstance(
                read_canonical_json(by_path[evaluation].snapshot),
                Mapping,
            ):
                raise ValueError("armed context phase result changed")
            _verify_identities(identities)
            verify_frozen(frozen)


def _prepare_context_phase(
    attempt: ContextAttempt, phase: ContextPhase, lease: ContextLease,
) -> tuple[FitOne, PredictOne, ReadTruth]:
    from tools.context_diagnostic_controller import prepare_context_phase

    return prepare_context_phase(attempt, phase, lease)


def _claimed_attempt(claim: RunClaim) -> ContextAttempt:
    """Re-read the live attempt through its immutable run claim."""
    if not _registered(claim) or not _claim_mode(claim) or \
       claim.root != claim.root.resolve(strict=True):
        raise ValueError("armed context execution is invalid")
    attempt_identity = ((claim.attempt, claim.attempt_identity),)
    with freeze_inputs((claim.attempt,)) as frozen:
        binding = _binding(claim.root, frozen[0])
        if binding != claim.attempt_binding:
            raise ValueError("armed context attempt changed")
        logical = claim.attempt.relative_to(claim.root)
        attempt = ContextAttempt.read(
            frozen[0].snapshot, logical, claim.root,
        )
        _verify_identities(attempt_identity)
        verify_frozen(frozen)
    return attempt


def _execute_authenticated_phase(
    claim: RunClaim, attempt: ContextAttempt, lease: ContextLease,
) -> tuple[Mapping[str, object], PhaseEvidence, ContextPhase]:
    """Execute the next phase while its source-derived lease is live."""
    if not _registered(claim) or not _claim_mode(claim) or \
       claim.root != claim.root.resolve(strict=True) or \
       Path(sys.executable).resolve(strict=True) != Path(
           attempt.torch_probe.python.path,
       ).resolve(strict=True):
        raise ValueError("context phase requires its bound attempt and Python")
    _validate_completed_prefix(claim, attempt)
    index = len(claim._completed)
    if index >= len(attempt.phases):
        raise ValueError("armed context phases are complete")
    phase = attempt.phases[index]
    artifacts = _start(claim, phase)
    callbacks = _prepare_context_phase(attempt, phase, lease)
    if not isinstance(callbacks, tuple) or len(callbacks) != 3 or \
       any(not callable(callback) for callback in callbacks):
        raise ValueError("armed context preparation is invalid")
    evaluation, evidence = _execute_started_phase(
        artifacts, claim, attempt.master, phase,
        attempt.source_binding("failure").sha256,
        attempt.config.sha256, attempt.source_tree.sha256,
        *callbacks, lease,
    )
    return evaluation, evidence, phase


def execute_armed_phase(claim: RunClaim) -> Mapping[str, object]:
    """Execute the next armed phase without caller-supplied provenance."""
    attempt = _claimed_attempt(claim)
    with authenticate_context_attempt(attempt) as lease:
        evaluation, evidence, phase = _execute_authenticated_phase(
            claim, attempt, lease,
        )
    claim._completed[phase.phase] = evidence
    return evaluation


def _attempt_path(path: Path) -> tuple[Path, Path]:
    absolute = path if path.is_absolute() else ROOT / path
    try:
        lexical = Path(os.path.abspath(absolute))
        resolved = absolute.resolve(strict=True)
        if lexical != resolved:
            raise ValueError("context attempt must not contain symlinks")
        _regular_identity(resolved)
        logical = resolved.relative_to(ROOT.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError("context attempt must be inside the repository") \
            from error
    return absolute, logical


def read_context_attempt(path: Path) -> ContextAttempt:
    """Read one canonical attempt without claiming its run directory."""
    _require_package_alias()
    absolute, logical = _attempt_path(path)
    return ContextAttempt.read(absolute, logical, ROOT)


def _validate_environment(attempt: ContextAttempt) -> None:
    actual = dict(os.environ)
    expected = dict(attempt.environment)
    if any(
        actual.get(name) != value for name, value in expected.items()
    ) or set(actual) - set(expected) - {
        "LC_CTYPE", "__CF_USER_TEXT_ENCODING",
    }:
        raise ValueError("context runner environment changed")


def _validate_controller(attempt: ContextAttempt) -> None:
    _require_isolated_execution(bootstrapped=True)
    _require_exact_launch()
    if tuple(sys.argv) != attempt.runner_argv or \
       Path(sys.executable).resolve(strict=True) != Path(
           attempt.torch_probe.python.path,
       ).resolve(strict=True):
        raise ValueError("context runner command changed")
    _validate_environment(attempt)
    attempt.torch_probe.python.validate_live("context Torch Python")
    package = Path(attempt.torch_probe.package_tree.root)
    if source_tree(package) != attempt.torch_probe.package_tree or \
       selected_source_tree(ROOT, CONTEXT_SOURCE_PATHS) != \
            attempt.source_tree:
        raise ValueError("context runner source or Torch package changed")


def _failure_value(
    claim: RunClaim, stage: str,
) -> dict[str, object]:
    return {
        "attempt": {
            "path": claim.attempt_binding.path,
            "sha256": claim.attempt_binding.sha256,
        },
        "stage": stage,
        "status": "integrity-failure",
        "schema": 1,
    }


def execute_context_attempt(path: Path) -> Mapping[str, object]:
    """Run both phases sequentially and finalize exactly one terminal result."""
    attempt = read_context_attempt(path)
    _validate_controller(attempt)
    from tools.run_universe_scaling import _expose_torch_package

    claim = None
    stage = "authenticate"

    def verify_claim() -> None:
        if claim is None:
            raise ValueError("context run was not claimed")
        if _directory_identity(claim.path) != claim.directory_identity:
            raise ValueError("context run directory changed")
        _verify_identities(((claim.attempt, claim.attempt_identity),))

    try:
        with authenticate_context_attempt(attempt) as lease:
            _expose_torch_package(Path(
                attempt.torch_probe.package_tree.root,
            ))
            _require_package_alias()
            from tools.finalize_context_diagnostic import finalize_context_history

            claim = claim_run(ROOT, ROOT / attempt.attempt_path)
            for stage in TARGET_PHASES:
                if _claimed_attempt(claim) != attempt:
                    raise ValueError("context attempt changed after its claim")
                evaluation, evidence, phase = _execute_authenticated_phase(
                    claim, attempt, lease,
                )
                claim._completed[phase.phase] = evidence
        stage = "finalize"
        return finalize_context_history(
            claim, attempt.master, attempt.phases,
        )
    except BaseException:
        if claim is not None:
            try:
                publish_context_outcome(
                    claim, _failure_value(claim, stage), verify_claim,
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
    """Execute once, mapping the first process signal to its shell status."""
    global _ACTIVE_ATTEMPT, _SIGNAL_NUMBER
    global _TERMINAL_OUTCOME

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
        execute_context_attempt(path)
        completed = True
    except BaseException as error:
        remember(error)
    if completed and _TERMINAL_OUTCOME is None:
        remember(ValueError("context run returned without a terminal outcome"))
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
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"context runner error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
