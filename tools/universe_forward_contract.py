"""Load a passing scaling result and bind its forward selections."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from tools.files import freeze_inputs, verify_frozen
from tools.finalize_universe_scaling import (
    FitClosure, LOCKS, _timestamp, validate_fit_ledger,
)
from tools.panel_contract import (
    RUN_ID, FileBinding, _object, _regular_inputs, _sha256, _string,
    _verify_identities, read_canonical_json, read_canonical_json_lines,
)
from tools.universe_scaling_contract import (
    EXPECTED_BUDGETS, SEEDS, FitJob, ScalingAttempt, fit_provenance_id,
    question_uses,
)

OUTCOME_FIELDS = {
    "schema", "attempt", "started", "ended", "stage", "exit", "status",
    "outputs", "integrity",
}
SUMMARY_FIELDS = {
    "schema", "status", "evidence_role", "ensemble", "fold_role",
    "fixed_epoch_role", "gate_source", "model_binding_role",
    "prediction_evidence", "locks", "results", "paired_calibration",
    "gates", "inputs",
}
GATES = {
    "unseen_mae_improvement",
    "positive_paired_intervals",
    "majority_unseen_improved",
    "core_degradation",
    "pooled_and_local_controls",
    "direction_majority",
    "close_mae",
    "unseen_33_to_44_marginal",
}
PRIOR_PHASE = MappingProxyType({
    "fold-1": "fold-0", "calibration": "fold-1",
})
FORWARD_QUESTION = "cohort-scaling"
FORWARD_MODE = "fixed-update"
FORWARD_COHORT = 55
FORWARD_MODEL = "panel_transformer"


@dataclass(frozen=True, slots=True)
class CheckpointSelection:
    target_phase: str
    source_phase: str
    seed: int
    checkpoint: int
    provenance_id: str
    model_fingerprint: str


@dataclass(frozen=True, slots=True)
class PassingScalingOutcome:
    run_id: str
    outcome: FileBinding
    attempt: FileBinding
    fits: FileBinding
    predictions: FileBinding
    summary: FileBinding
    manifest: ScalingAttempt
    selections: tuple[CheckpointSelection, ...]


def _external(value: object, label: str) -> FileBinding:
    if not isinstance(value, FileBinding):
        raise ValueError(f"{label} is invalid")
    return FileBinding.parse(
        {"path": value.path, "sha256": value.sha256}, label,
    )


def _present(value: object, label: str) -> FileBinding:
    item = _object(value, {"path", "state", "sha256"}, label)
    if item["state"] != "present":
        raise ValueError(f"{label} is not present")
    return FileBinding.parse(
        {name: item[name] for name in ("path", "sha256")},
        label, relative=False,
    )


def _rooted(binding: FileBinding, root: Path, label: str) -> Path:
    path = Path(binding.path)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} is outside the repository") from error
    if path != root / relative or relative == Path(".") or \
       any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"{label} is not normalized")
    return path


def _summary(
    value: object,
    bindings: tuple[FileBinding, FileBinding, FileBinding],
) -> None:
    item = _object(value, SUMMARY_FIELDS, "scaling summary")
    locks = _object(item["locks"], set(LOCKS), "summary locks")
    if type(item["schema"]) is not int or item["schema"] != 1 or \
       item["status"] != "pass" or \
       item["evidence_role"] != \
            "development-diagnostic-not-forward-clean" or \
       item["gate_source"] != "fixed-update-calibration-only" or \
       type(locks["reserved_test_materialized_samples"]) is not int or \
       locks["reserved_test_materialized_samples"] != 0 or any(
           locks[name] is not False
           for name in ("policy_selected", "backtest_run",
                        "trading_authorized")
       ):
        raise ValueError("scaling summary is not a passing diagnostic")
    gates = _object(item["gates"], {*GATES, "all_pass"}, "summary gates")
    if gates["all_pass"] is not True or any(
        not isinstance(gates[name], Mapping) or
        gates[name].get("pass") is not True
        for name in GATES
    ):
        raise ValueError("scaling summary gates did not all pass")
    inputs = _object(
        item["inputs"], {"attempt", "fits", "predictions"}, "summary inputs",
    )
    observed = tuple(
        FileBinding.parse(inputs[name], f"summary inputs.{name}",
                          relative=False)
        for name in ("attempt", "fits", "predictions")
    )
    if observed != bindings:
        raise ValueError("scaling summary inputs changed")


def _outcome(
    value: object, expected: FileBinding, root: Path,
) -> tuple[
    str, FileBinding, FileBinding, FileBinding, FileBinding, str, FileBinding,
]:
    item = _object(value, OUTCOME_FIELDS, "scaling outcome")
    if type(item["schema"]) is not int or item["schema"] != 1 or \
       item["status"] != "pass" or item["stage"] != "analysis" or \
       type(item["exit"]) is not int or item["exit"] != 0:
        raise ValueError("scaling outcome is not a terminal pass")
    started = _timestamp(item["started"], "outcome start")
    ended = _timestamp(item["ended"], "outcome end")
    if ended < started:
        raise ValueError("outcome timestamps are reversed")

    attempt_value = _object(
        item["attempt"], {"path", "sha256", "run_id"}, "outcome attempt",
    )
    run_id = _string(attempt_value["run_id"], "outcome run_id")
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("outcome run_id is invalid")
    attempt = FileBinding.parse(
        {name: attempt_value[name] for name in ("path", "sha256")},
        "outcome attempt", relative=False,
    )
    outputs = _object(
        item["outputs"], {"fits", "predictions", "summary", "outcome"},
        "outcome outputs",
    )
    fits = _present(outputs["fits"], "outcome outputs.fits")
    predictions = _present(
        outputs["predictions"], "outcome outputs.predictions",
    )
    summary = _present(outputs["summary"], "outcome outputs.summary")
    self_record = _object(
        outputs["outcome"], {"path", "state", "sha256"},
        "outcome outputs.outcome",
    )
    if self_record != {
        "path": str(root / expected.path),
        "state": "absent", "sha256": None,
    }:
        raise ValueError("terminal outcome binding changed")
    integrity = _object(
        item["integrity"], {"trusted_finalizer_tree", "primary_python"},
        "outcome integrity",
    )
    finalizer = _sha256(
        integrity["trusted_finalizer_tree"], "finalizer tree",
    )
    primary = FileBinding.parse(
        integrity["primary_python"], "primary Python", relative=False,
    )
    return (
        run_id, attempt, fits, predictions, summary, finalizer, primary,
    )


def _manifest(
    value: ScalingAttempt,
    run_id: str,
    bindings: tuple[
        FileBinding, FileBinding, FileBinding, FileBinding,
    ],
    outcome: Path,
    finalizer: str,
    primary: FileBinding,
    root: Path,
) -> None:
    attempt, fits, predictions, summary = bindings
    expected = {
        name: root / value.outputs[name]
        for name in ("fits", "predictions", "summary", "outcome")
    }
    if value.run_id != run_id or \
       root / value.attempt_path != Path(attempt.path) or \
       expected != {
           "fits": Path(fits.path),
           "predictions": Path(predictions.path),
           "summary": Path(summary.path),
           "outcome": outcome,
       } or value.finalizer_tree.sha256 != finalizer or \
       FileBinding(
           value.primary_python.path, value.primary_python.sha256,
       ) != primary:
        raise ValueError("scaling outcome disagrees with its armed attempt")


def read_passing_scaling_outcome(
    outcome: FileBinding, *, root: Path,
) -> PassingScalingOutcome:
    """Load one exact terminal PASS and its validated physical-fit closure."""
    expected = _external(outcome, "scaling outcome")
    if not isinstance(root, Path):
        raise ValueError("repository root must be resolved")
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("repository root is unavailable") from error
    if root != resolved_root:
        raise ValueError("repository root must be resolved")
    outcome_path = root / expected.path
    outcome_identity = _regular_inputs((outcome_path,))
    with freeze_inputs((outcome_path,)) as (outcome_input,):
        if outcome_input.sha256 != expected.sha256:
            raise ValueError("scaling outcome hash changed")
        values = _outcome(
            read_canonical_json(outcome_input.snapshot), expected, root,
        )
        run_id, attempt, fits, predictions, summary, finalizer, primary = \
            values
        consumed = tuple(
            _rooted(binding, root, label)
            for binding, label in (
                (attempt, "attempt"),
                (summary, "summary"),
                (fits, "fit ledger"),
            )
        )
        identities = _regular_inputs(consumed)
        with freeze_inputs(consumed) as inputs:
            attempt_input, summary_input, fits_input = inputs
            for binding, frozen, label in (
                (attempt, attempt_input, "attempt"),
                (summary, summary_input, "summary"),
                (fits, fits_input, "fit ledger"),
            ):
                binding.validate(frozen, label)
            logical_attempt = Path(attempt.path).relative_to(root)
            manifest = ScalingAttempt.read(
                attempt_input.snapshot, logical_attempt, root,
            )
            _manifest(
                manifest, run_id,
                (attempt, fits, predictions, summary),
                outcome_path, finalizer, primary, root,
            )
            _summary(
                read_canonical_json(summary_input.snapshot),
                (attempt, fits, predictions),
            )
            closure = validate_fit_ledger(
                read_canonical_json_lines(fits_input.snapshot),
                manifest.coverage.master, manifest.coverage,
            )
            selections = tuple(
                _selection(closure, target, seed)
                for target in PRIOR_PHASE for seed in SEEDS
            )
            _verify_identities(identities)
            verify_frozen(inputs)
        _verify_identities(outcome_identity)
        verify_frozen((outcome_input,))
    return PassingScalingOutcome(
        run_id, expected, attempt, fits, predictions, summary,
        manifest, selections,
    )


def _selection(
    closure: FitClosure, target_phase: str, seed: int,
) -> CheckpointSelection:
    if len(closure.jobs) != len(closure.records):
        raise ValueError("forward fit closure changed")
    source_phase = PRIOR_PHASE[target_phase]
    expected = FitJob(
        "pooled", FORWARD_MODE, FORWARD_COHORT, source_phase,
        FORWARD_MODEL, seed, closure.master[:FORWARD_COHORT],
    )
    if (FORWARD_QUESTION, FORWARD_COHORT) not in question_uses(
        expected, closure.master,
    ):
        raise ValueError("forward checkpoint question is invalid")
    matches = tuple(
        record for job, record in zip(
            closure.jobs, closure.records, strict=True,
        )
        if job == expected
    )
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise ValueError("forward checkpoint is not unique")
    record = matches[0]
    provenance = fit_provenance_id(expected)
    checkpoint = record.get("selected_checkpoint")
    fingerprint = record.get("model_fingerprint")
    limit = dict(EXPECTED_BUDGETS)[source_phase].checkpoints
    if record.get("provenance_id") != provenance or \
       type(checkpoint) is not int or not 1 <= checkpoint <= limit:
        raise ValueError("forward checkpoint selection is invalid")
    fingerprint = _sha256(fingerprint, "forward model fingerprint")
    return CheckpointSelection(
        target_phase, source_phase, seed, checkpoint, provenance, fingerprint,
    )


def resolve_prior_checkpoint(
    source: PassingScalingOutcome, target_phase: str, seed: int,
) -> CheckpointSelection:
    """Return one immutable selection authenticated by the PASS reader."""
    if not isinstance(source, PassingScalingOutcome) or \
       not isinstance(target_phase, str) or target_phase not in PRIOR_PHASE or \
       type(seed) is not int or seed not in SEEDS:
        raise ValueError("forward checkpoint query is invalid")
    matches = tuple(
        item for item in source.selections
        if item.target_phase == target_phase and item.seed == seed
    )
    if len(matches) != 1:
        raise ValueError("forward checkpoint is not unique")
    return matches[0]
