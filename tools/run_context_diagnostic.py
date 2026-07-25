#!/usr/bin/env python3
"""Enforce one-shot, receipt-gated context diagnostic phases."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO
import json
import os

from tools.context_diagnostic_contract import (
    TARGET_PHASES, ContextFit, ContextPhase, ContextPrediction,
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
    read_canonical_json_lines,
)

FitOne = Callable[[ContextFit], tuple[str, float, object]]
PredictOne = Callable[[ContextPrediction, object], Sequence[float]]
ReadTruth = Callable[[], Mapping[str, object]]
Verify = Callable[[], None]


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
    started: set[str] = field(
        default_factory=set, compare=False, repr=False,
    )
    completed: dict[str, PhaseEvidence] = field(
        default_factory=dict, compare=False, repr=False,
    )


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
            mkdir_nofollow(run)
            directory, directory_identity = _open_directory(run)
            try:
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
    return RunClaim(
        root, run, attempt, binding, identities[0][1],
        directory_identity,
    )


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
    if tuple(claim.completed) != completed or \
       claim.started != set(begun):
        raise ValueError("context phase order changed")


def _start(claim: RunClaim, phase: ContextPhase) -> PhaseArtifacts:
    if not isinstance(claim, RunClaim) or \
       not isinstance(phase, ContextPhase) or \
       claim.path != _run_path(claim.root, claim.attempt) or \
       _directory_identity(claim.path) != claim.directory_identity:
        raise ValueError("context run claim is invalid")
    _verify_identities(((claim.attempt, claim.attempt_identity),))
    _require_phase_prefix(claim, phase, started=False)
    claim.started.add(phase.phase)
    artifacts = _artifacts(claim.root, claim.attempt, phase)
    for path in artifacts:
        _absent(path, "context phase output")
    return artifacts


def _publish_text(
    path: Path, write: Callable[[TextIO], None], verify: Verify,
    directory_identity: tuple[int, int],
) -> None:
    """Publish into the claimed directory and clean only an owned temp."""
    directory, identity = _open_directory(path.parent)
    if identity != directory_identity:
        os.close(directory)
        raise ValueError("context output directory changed")
    temporary: ExclusiveTemp | None = None
    completed = False

    def capture(value: ExclusiveTemp) -> None:
        nonlocal temporary
        temporary = value

    def before_link(value: ExclusiveTemp) -> None:
        if value != temporary or \
           _directory_identity(path.parent) != directory_identity:
            raise OSError("context output temporary changed")
        verify()

    try:
        exclusive_text(
            path, write, directory,
            before_link_with_temp=before_link,
            on_temp_created=capture,
        )
        os.fsync(directory)
        if temporary is None or \
           _regular_identity(path) != temporary.identity or \
           _directory_identity(path.parent) != directory_identity:
            raise OSError("context output changed after publication")
        completed = True
    finally:
        try:
            # Never remove a public output or a foreign replacement.
            if not completed and temporary is not None and \
               _owns_entry(directory, temporary, (1,)):
                os.unlink(temporary.name, dir_fd=directory)
                os.fsync(directory)
        finally:
            os.close(directory)


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
    directory_identity: tuple[int, int],
) -> None:
    def write(file: TextIO) -> None:
        json.dump(value, file, allow_nan=False, indent=2, sort_keys=True)
        file.write("\n")

    _publish_text(path, write, verify, directory_identity)


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
) -> None:
    fits = read_canonical_json_lines(fit_path)
    validate_context_fit_records(
        fits, master, phase,
        source_failure_sha256, config_sha256,
    )
    validate_context_prediction_records(
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


def read_authorized_truth(
    claim: RunClaim, master: Sequence[str], phase: ContextPhase,
    source_failure_sha256: str, config_sha256: str,
    source_tree_sha256: str, reader: ReadTruth,
) -> Mapping[str, object]:
    """Consume one truth access and durably publish its phase evaluation."""
    if not isinstance(claim, RunClaim) or \
       not isinstance(phase, ContextPhase) or \
       claim.path != _run_path(claim.root, claim.attempt) or \
       _directory_identity(claim.path) != claim.directory_identity or \
       phase.phase in claim.completed:
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
        validate_context_ledgers(
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
                evaluation = reader()
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
                    # Keep post-truth evidence live until outcome publication.
                    claim.completed[phase.phase] = PhaseEvidence(
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
    return evaluation


def publish_context_outcome(
    claim: RunClaim, value: Mapping[str, object], verify: Verify,
) -> Path:
    """Atomically close one live run while its phase evidence is frozen."""
    if not isinstance(claim, RunClaim) or not isinstance(value, Mapping) or \
       claim.path != _run_path(claim.root, claim.attempt) or \
       _directory_identity(claim.path) != claim.directory_identity:
        raise ValueError("context outcome claim is invalid")
    outcome = claim.path / "outcome.json"
    _absent(outcome, "context outcome")
    _write_json(outcome, value, verify, claim.directory_identity)
    verify()
    return outcome


def execute_phase(
    claim: RunClaim, master: Sequence[str], phase: ContextPhase,
    source_failure_sha256: str, config_sha256: str,
    source_tree_sha256: str, fit_one: FitOne,
    predict_one: PredictOne, truth: ReadTruth,
) -> Mapping[str, object]:
    """Fit, predict, publish, and only then authorize one phase's truth."""
    artifacts = _start(claim, phase)
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
    return read_authorized_truth(
        claim, master, phase, source, config, source_tree, truth,
    )
