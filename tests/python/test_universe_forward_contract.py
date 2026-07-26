#!/usr/bin/env python3
"""Verify terminal scaling evidence and PASS-only forward selection."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tools.universe_forward_contract as contract
from tools.files import file_sha256, write_json
from tools.finalize_universe_scaling import FitClosure
from tools.panel_contract import FileBinding
from tools.universe_forward_contract import (
    CheckpointSelection, ForwardFitSpec, ForwardPredictionSpec,
    forward_fit_specs, forward_model_fingerprint, forward_prediction_specs,
    read_passing_scaling_outcome, read_scaling_failure,
    resolve_prior_checkpoint,
)
from tools.universe_scaling_contract import (
    EXPECTED_BUDGETS, PHASES, SEEDS, FitJob, PhaseCoverage,
    ScalingCoverage, SeriesCoverage, fit_provenance_id,
    timestamp_grid_sha256,
)

MASTER = tuple(f"S{index:02}" for index in range(55))
MISSING_RANKS = {
    "fold-0": frozenset((15, 29)),
    "fold-1": frozenset((15, 29, 40)),
    "calibration": frozenset((15, 29, 33, 40)),
}
EMPTY_GRID_SHA256 = timestamp_grid_sha256(())
GATES = (
    "unseen_mae_improvement",
    "positive_paired_intervals",
    "majority_unseen_improved",
    "core_degradation",
    "pooled_and_local_controls",
    "direction_majority",
    "close_mae",
    "unseen_33_to_44_marginal",
)


def raises(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def digest(index: int) -> str:
    return f"{index:064x}"


def coverage() -> ScalingCoverage:
    return ScalingCoverage(tuple(
        PhaseCoverage(phase, tuple(
            SeriesCoverage(
                series,
                1_000 + phase_index,
                0 if rank in MISSING_RANKS[phase]
                else 100 * phase_index + rank,
                EMPTY_GRID_SHA256 if rank in MISSING_RANKS[phase]
                else digest(1_000 + 100 * phase_index + rank),
            )
            for rank, series in enumerate(MASTER, 1)
        ))
        for phase_index, phase in enumerate(PHASES, 1)
    ))


def binding(path: Path, sha256: str) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256}


def write_lines(path: Path, values: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, allow_nan=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def fit(
    phase: str, seed: int, checkpoint: object = 1,
    *, mode: str = "fixed-update",
) -> tuple[FitJob, dict[str, object]]:
    job = FitJob(
        "pooled", mode, 55, phase, "panel_transformer", seed, MASTER,
    )
    provenance = fit_provenance_id(job)
    return job, {
        "provenance_id": provenance,
        "model_fingerprint": digest(seed),
        "selected_checkpoint": checkpoint,
    }


def fit_closure(
    values: tuple[tuple[FitJob, dict[str, object]], ...],
) -> FitClosure:
    jobs, records = zip(*values, strict=True)
    return FitClosure(
        records, MASTER, {}, {}, {}, jobs,
        {fit_provenance_id(job): job for job in jobs},
        {
            fit_provenance_id(job): record["model_fingerprint"]
            for job, record in values
        },
    )


def fixture(
    root: Path,
    *,
    status: str = "pass",
    change_summary: object | None = None,
    change_outcome: object | None = None,
) -> tuple[FileBinding, object, FitClosure]:
    failed = status == "gate-failure"
    attempt_path = root / "experiments/run-attempt.json"
    outcome_path = root / "experiments/run-outcome.json"
    fits_path = root / "reports/run/fits.jsonl"
    predictions_path = root / "reports/run/predictions.jsonl"
    summary_path = root / "reports/run/summary.json"
    write_json(attempt_path, {"fixture": "attempt"})
    write_lines(fits_path, ({"fixture": "fit"},))

    attempt = binding(attempt_path, file_sha256(attempt_path))
    fits = binding(fits_path, file_sha256(fits_path))
    predictions = binding(predictions_path, digest(3))
    summary = {
        "schema": 1,
        "status": status,
        "evidence_role": "development-diagnostic-not-forward-clean",
        "ensemble": "arithmetic-mean-of-five-neural-returns",
        "fold_role": "checkpoint-selection-audit-development-diagnostic",
        "fixed_epoch_role":
            "descriptive-cohort-sized-draw-data-plus-compute-curve",
        "gate_source": "fixed-update-calibration-only",
        "model_binding_role":
            "cross-ledger-consistency-not-independent-execution-proof",
        "prediction_evidence": {},
        "locks": {
            "reserved_test_materialized_samples": 0,
            "policy_selected": False,
            "backtest_run": False,
            "trading_authorized": False,
        },
        "results": [],
        "paired_calibration": {},
        "gates": {
            **{
                name: {"pass": not failed or index > 0}
                for index, name in enumerate(GATES)
            },
            "all_pass": not failed,
        },
        "inputs": {
            "attempt": attempt, "fits": fits, "predictions": predictions,
        },
    }
    if change_summary is not None:
        change_summary(summary)  # type: ignore[operator]
    write_json(summary_path, summary)

    outputs = {
        name: value | {"state": "present"}
        for name, value in (
            ("fits", fits),
            ("predictions", predictions),
            ("summary", binding(summary_path, file_sha256(summary_path))),
        )
    }
    outputs["outcome"] = {
        "path": str(outcome_path), "state": "absent", "sha256": None,
    }
    outcome = {
        "schema": 1,
        "attempt": attempt | {"run_id": "run"},
        "started": "2026-07-24T00:00:00Z",
        "ended": "2026-07-24T01:00:00Z",
        "stage": "analysis",
        "exit": 3 if failed else 0,
        "status": status,
        "outputs": outputs,
        "integrity": {
            "trusted_finalizer_tree": digest(6),
            "primary_python": {
                "path": "/python", "sha256": digest(7),
            },
        },
    }
    if change_outcome is not None:
        change_outcome(outcome)  # type: ignore[operator]
    write_json(outcome_path, outcome)

    closure = fit_closure(tuple(
        fit(phase, seed, 1 + index)
        for phase in ("fold-0", "fold-1")
        for index, seed in enumerate(SEEDS)
    ))
    manifest = SimpleNamespace(
        run_id="run",
        attempt_path="experiments/run-attempt.json",
        outputs={
            "fits": "reports/run/fits.jsonl",
            "predictions": "reports/run/predictions.jsonl",
            "summary": "reports/run/summary.json",
            "outcome": "experiments/run-outcome.json",
        },
        finalizer_tree=SimpleNamespace(sha256=digest(6)),
        primary_python=SimpleNamespace(path="/python", sha256=digest(7)),
        coverage=coverage(),
    )
    return FileBinding(
        "experiments/run-outcome.json", file_sha256(outcome_path),
    ), manifest, closure


def read_fixture(
    expected: FileBinding, manifest: object,
    closure: FitClosure, root: Path, *args: object,
    reader: Callable[..., object] = read_passing_scaling_outcome,
) -> object:
    def read_attempt(
        snapshot: Path, logical: Path, repository: Path,
    ) -> object:
        assert json.loads(snapshot.read_text(encoding="utf-8")) == {
            "fixture": "attempt",
        }
        assert logical == Path("experiments/run-attempt.json")
        assert repository == root
        return manifest

    def validate_fits(
        records: object, master: object, coverage: object,
    ) -> FitClosure:
        assert records == ({"fixture": "fit"},)
        assert master == MASTER
        assert coverage is manifest.coverage
        return closure

    with patch.object(
        contract.ScalingAttempt, "read", side_effect=read_attempt,
    ), patch.object(
        contract, "validate_fit_ledger", side_effect=validate_fits,
    ):
        return reader(expected, *args, root=root)


def test_pass_reader() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        expected, manifest, closure = fixture(root)
        value = read_fixture(expected, manifest, closure, root)
        assert value.run_id == "run"
        assert value.outcome == expected
        assert value.fits.path == str(root / "reports/run/fits.jsonl")
        assert value.manifest is manifest
        assert len(value.selections) == 2 * len(SEEDS)
        assert not hasattr(value, "closure")

        raises(
            read_fixture, replace(expected, sha256=digest(99)),
            manifest, closure, root,
        )
        outcome_path = root / expected.path
        outcome_path.write_text(
            outcome_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        changed = replace(expected, sha256=file_sha256(outcome_path))
        raises(read_fixture, changed, manifest, closure, root)


def test_pass_rejections() -> None:
    cases = (
        (None, lambda item: item.__setitem__("status", "gate-failure")),
        (None, lambda item: item.__setitem__("exit", True)),
        (None, lambda item: item.__setitem__(
            "started", "2026-07-24T02:00:00Z",
        )),
        (lambda item: item["gates"].__setitem__("all_pass", 1), None),
        (lambda item: item["gates"][GATES[0]].__setitem__("pass", False),
         None),
        (lambda item: item["locks"].__setitem__(
            "trading_authorized", True,
        ), None),
        (lambda item: item["inputs"]["fits"].__setitem__(
            "sha256", digest(99),
        ), None),
    )
    for change_summary, change_outcome in cases:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            values = fixture(
                root, change_summary=change_summary,
                change_outcome=change_outcome,
            )
            raises(read_fixture, *values, root)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        expected, manifest, closure = fixture(root)
        manifest.finalizer_tree.sha256 = digest(99)
        raises(read_fixture, expected, manifest, closure, root)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        expected, manifest, closure = fixture(
            root, change_outcome=lambda item:
            item["outputs"]["fits"].__setitem__(
                "path", "/outside/fits.jsonl",
            ),
        )
        raises(read_fixture, expected, manifest, closure, root)


def test_failure_reader() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        expected, manifest, closure = fixture(root, status="gate-failure")
        value = read_fixture(
            expected, manifest, closure, root,
            reader=read_scaling_failure,
        )
        assert value.run_id == "run"
        assert value.outcome == expected
        assert value.summary.path == str(root / "reports/run/summary.json")
        assert value.failed_gates == (GATES[0],)
        assert not hasattr(value, "manifest")

        try:
            value.failed_gates = ()  # type: ignore[misc]
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("scaling failure is mutable")

        raises(
            read_fixture, replace(expected, sha256=digest(99)),
            manifest, closure, root, reader=read_scaling_failure,
        )


def test_failure_rejections() -> None:
    cases = (
        (None, lambda item: item.__setitem__("status", "pass")),
        (None, lambda item: item.__setitem__("exit", 0)),
        (lambda item: item.__setitem__("status", "pass"), None),
        (lambda item: item["gates"].__setitem__("all_pass", True), None),
        (lambda item: item["gates"][GATES[0]].__setitem__("pass", True),
         None),
        (lambda item: item["gates"][GATES[0]].__setitem__("pass", 0), None),
        (lambda item: item["locks"].__setitem__(
            "trading_authorized", True,
        ), None),
        (None, lambda item: item["outputs"]["fits"].__setitem__(
            "sha256", digest(99),
        )),
    )
    for change_summary, change_outcome in cases:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            values = fixture(
                root, status="gate-failure",
                change_summary=change_summary,
                change_outcome=change_outcome,
            )
            raises(
                read_fixture, *values, root, reader=read_scaling_failure,
            )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        expected, manifest, closure = fixture(
            root, status="gate-failure",
            change_outcome=lambda item:
            item["outputs"]["summary"].__setitem__("path", "/outside.json"),
        )
        raises(
            read_fixture, expected, manifest, closure, root,
            reader=read_scaling_failure,
        )


def test_checkpoint_selection() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        expected, manifest, closure = fixture(root)
        source = read_fixture(expected, manifest, closure, root)
    for target, prior in (
        ("fold-1", "fold-0"), ("calibration", "fold-1"),
    ):
        for index, seed in enumerate(SEEDS):
            assert resolve_prior_checkpoint(source, target, seed) == \
                CheckpointSelection(
                    target, prior, seed, 1 + index,
                    fit_provenance_id(fit(prior, seed)[0]), digest(seed),
                )

    for target in ("fold-0", "unknown", 1):
        raises(resolve_prior_checkpoint, source, target, SEEDS[0])
    for seed in (True, 0, 7.0, "7"):
        raises(resolve_prior_checkpoint, source, "fold-1", seed)

    job, record = fit("fold-0", SEEDS[0])
    for checkpoint in (None, True, 0, 101, 1.0, "1"):
        changed = fit_closure(
            ((job, record | {"selected_checkpoint": checkpoint}),),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            expected, manifest, _ = fixture(root)
            raises(read_fixture, expected, manifest, changed, root)
    duplicate = fit_closure(((job, record), (job, dict(record))))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        expected, manifest, _ = fixture(root)
        raises(read_fixture, expected, manifest, duplicate, root)
    try:
        source.selections[0].checkpoint = 99  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("checkpoint selection is mutable")
    try:
        replace(source, closure=closure)
    except TypeError:
        pass
    else:
        raise AssertionError("mutable fit closure remains exposed")
    raises(resolve_prior_checkpoint, object(), "fold-1", SEEDS[0])


def test_forward_fit_specs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        expected, manifest, closure = fixture(root)
        source = read_fixture(expected, manifest, closure, root)
        specs = tuple(
            item for phase in ("fold-1", "calibration")
            for item in read_fixture(
                expected, manifest, closure, root, phase,
                reader=forward_fit_specs,
            )
        )
        raises(
            read_fixture, replace(expected, sha256=digest(90)),
            manifest, closure, root, "fold-1",
            reader=forward_fit_specs,
        )
    budgets = dict(EXPECTED_BUDGETS)
    assert tuple(
        (item.selection.target_phase, item.selection.seed)
        for item in specs
    ) == tuple(
        (phase, seed)
        for phase in ("fold-1", "calibration") for seed in SEEDS
    )
    assert all(
        item.optimizer_updates ==
        item.selection.checkpoint *
        budgets[item.selection.target_phase].updates_per_checkpoint
        for item in specs
    )
    assert len({item.provenance_id for item in specs}) == len(specs)
    assert all(
        item.provenance_id != item.selection.provenance_id for item in specs
    )

    assert contract._forward_provenance_id(
        digest(8), MASTER, specs[0].selection, specs[0].optimizer_updates,
    ) == \
        "87df33de81a8ad27776a428f717bea1390b840df63c89e6a7302db832188e60b"
    first = specs[0]
    selections = source.selections
    mutations = (
        (digest(90), MASTER, first.selection, first.optimizer_updates),
        (
            source.outcome.sha256, MASTER,
            replace(first.selection, checkpoint=2),
            2 * budgets["fold-1"].updates_per_checkpoint,
        ),
        (
            source.outcome.sha256, MASTER,
            replace(first.selection, provenance_id=digest(91)),
            first.optimizer_updates,
        ),
        (
            source.outcome.sha256, MASTER,
            replace(first.selection, model_fingerprint=digest(92)),
            first.optimizer_updates,
        ),
        (
            source.outcome.sha256, MASTER[1:] + MASTER[:1],
            first.selection, first.optimizer_updates,
        ),
    )
    assert all(
        contract._forward_provenance_id(*value) !=
        first.provenance_id
        for value in mutations
    )

    invalid_selection = replace(
        source, selections=(
            replace(selections[0], provenance_id=digest(91)),
            *selections[1:],
        ),
    )
    raises(contract._forward_fit_specs, invalid_selection, "fold-1")
    for changed in (selections[:-1], (selections[0], *selections)):
        raises(
            contract._forward_fit_specs,
            replace(source, selections=changed), "fold-1",
        )
    raises(
        contract._forward_fit_specs,
        replace(
            source,
            selections=(
                SimpleNamespace(**asdict(selections[0])), *selections[1:],
            ),
        ),
        "fold-1",
    )
    for manifest in (
        SimpleNamespace(),
        SimpleNamespace(coverage=SimpleNamespace()),
        SimpleNamespace(coverage=SimpleNamespace(
            master=(*MASTER[:-1], MASTER[0]),
        )),
    ):
        raises(
            contract._forward_fit_specs,
            replace(source, manifest=manifest), "fold-1",
        )

    try:
        first.optimizer_updates = 1  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("forward fit specification is mutable")
    assert isinstance(first, ForwardFitSpec)
    for target in ("fold-0", "unknown", 1):
        raises(forward_fit_specs, source.outcome, target, root=root)
    raises(forward_fit_specs, object(), "fold-1", root=root)


def test_forward_model_fingerprint() -> None:
    provenance, state = digest(1), digest(2)
    expected = "848a2e3da573430c0b0e5444dee403e2cd0499f499a97b9e0cf6776d0da62495"
    assert forward_model_fingerprint(provenance, state) == expected
    assert forward_model_fingerprint(state, provenance) != expected
    assert forward_model_fingerprint(digest(3), state) != expected
    assert forward_model_fingerprint(provenance, digest(3)) != expected
    for invalid in (None, True, 1, "A" * 64, "0" * 63, "g" * 64):
        raises(forward_model_fingerprint, invalid, state)
        raises(forward_model_fingerprint, provenance, invalid)


def test_forward_prediction_specs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        expected, manifest, closure = fixture(root)
        source = read_fixture(expected, manifest, closure, root)
        predictions_path = root / "reports/run/predictions.jsonl"
        assert not predictions_path.exists()
        assert not (root / "data").exists()
        fits = {
            phase: read_fixture(
                expected, manifest, closure, root, phase,
                reader=forward_fit_specs,
            )
            for phase in ("fold-1", "calibration")
        }
        specs = tuple(
            item for phase in fits
            for item in read_fixture(
                expected, manifest, closure, root, phase,
                reader=forward_prediction_specs,
            )
        )
        raises(
            read_fixture, replace(expected, sha256=digest(90)),
            manifest, closure, root, "fold-1",
            reader=forward_prediction_specs,
        )
        assert not predictions_path.exists()
        assert not (root / "data").exists()

    phases = {
        phase.phase: phase for phase in manifest.coverage.phases
    }
    expected_specs = tuple(
        (
            fit, record.series, rank, record.validation_rows,
            record.timestamp_sha256,
        )
        for phase, phase_fits in fits.items()
        for fit in phase_fits
        for rank, record in enumerate(phases[phase].series, 1)
        if record.validation_rows
    )
    assert tuple(
        (
            item.fit, item.series, item.manifest_rank,
            item.prediction_count, item.timestamp_sha256,
        )
        for item in specs
    ) == expected_specs
    assert len(specs) == len(SEEDS) * (52 + 51) == 515
    assert tuple(
        sum(
            item.fit.selection.target_phase == phase for item in specs
        )
        for phase in ("fold-1", "calibration")
    ) == (260, 255)
    assert sum(item.prediction_count for item in specs) == len(SEEDS) * sum(
        record.validation_rows
        for phase in ("fold-1", "calibration")
        for record in phases[phase].series
    )
    assert all(
        item.series == MASTER[item.manifest_rank - 1] and
        item.fit.selection.target_phase in ("fold-1", "calibration")
        for item in specs
    )

    first = specs[0]
    try:
        first.prediction_count = 1  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("forward prediction specification is mutable")
    assert isinstance(first, ForwardPredictionSpec)

    def with_coverage(value: ScalingCoverage) -> object:
        changed = SimpleNamespace(**{
            **vars(source.manifest), "coverage": value,
        })
        return replace(source, manifest=changed)

    original = manifest.coverage
    fold1 = original.phases[1]
    calibration = original.phases[2]
    calibration_index = next(
        index for index, item in enumerate(calibration.series)
        if item.validation_rows
    )
    calibration_records = list(calibration.series)
    calibration_records[calibration_index] = replace(
        calibration_records[calibration_index],
        validation_rows=
            calibration_records[calibration_index].validation_rows + 1,
        timestamp_sha256=digest(9_000),
    )
    changed_calibration = replace(
        original, phases=(
            *original.phases[:2],
            replace(calibration, series=tuple(calibration_records)),
        ),
    )
    assert contract._forward_prediction_specs(
        with_coverage(changed_calibration), "fold-1",
    ) == contract._forward_prediction_specs(source, "fold-1")
    assert contract._forward_prediction_specs(
        with_coverage(changed_calibration), "calibration",
    ) != contract._forward_prediction_specs(source, "calibration")

    reordered = replace(
        fold1, series=(
            fold1.series[1], fold1.series[0], *fold1.series[2:],
        ),
    )
    duplicate = replace(
        fold1, series=(
            fold1.series[0], fold1.series[0], *fold1.series[2:],
        ),
    )
    empty_mismatch = replace(
        fold1, series=(
            replace(fold1.series[0], timestamp_sha256=EMPTY_GRID_SHA256),
            *fold1.series[1:],
        ),
    )
    zero_mismatch = replace(
        fold1, series=(
            *fold1.series[:14],
            replace(fold1.series[14], timestamp_sha256=digest(9_001)),
            *fold1.series[15:],
        ),
    )
    bool_count = replace(
        fold1, series=(
            replace(fold1.series[0], validation_rows=True),
            *fold1.series[1:],
        ),
    )
    negative_count = replace(
        fold1, series=(
            replace(fold1.series[0], validation_rows=-1),
            *fold1.series[1:],
        ),
    )
    malformed_digest = replace(
        fold1, series=(
            replace(fold1.series[0], timestamp_sha256="invalid"),
            *fold1.series[1:],
        ),
    )
    calibration_bool = replace(
        calibration, series=(
            replace(calibration.series[0], validation_rows=True),
            *calibration.series[1:],
        ),
    )
    for target, changed in (
        ("fold-1", replace(
            original, phases=(
                original.phases[0], fold1, calibration_bool,
            ),
        )),
        ("calibration", replace(
            original, phases=(
                original.phases[0], bool_count, calibration,
            ),
        )),
    ):
        raises(
            contract._forward_prediction_specs,
            with_coverage(changed), target,
        )
    for changed in (
        replace(original, phases=(
            original.phases[1], original.phases[0], original.phases[2],
        )),
        replace(original, phases=(
            original.phases[0], reordered, original.phases[2],
        )),
        replace(original, phases=(
            original.phases[0], duplicate, original.phases[2],
        )),
        replace(original, phases=(
            original.phases[0], empty_mismatch, original.phases[2],
        )),
        replace(original, phases=(
            original.phases[0], zero_mismatch, original.phases[2],
        )),
        replace(original, phases=(
            original.phases[0], bool_count, original.phases[2],
        )),
        replace(original, phases=(
            original.phases[0], negative_count, original.phases[2],
        )),
        replace(original, phases=(
            original.phases[0], malformed_digest, original.phases[2],
        )),
    ):
        raises(
            contract._forward_prediction_specs,
            with_coverage(changed), "fold-1",
        )

    for target in ("fold-0", "unknown", 1):
        raises(forward_prediction_specs, source.outcome, target, root=root)
    raises(forward_prediction_specs, object(), "fold-1", root=root)


def main() -> None:
    test_pass_reader()
    test_pass_rejections()
    test_failure_reader()
    test_failure_rejections()
    test_checkpoint_selection()
    test_forward_fit_specs()
    test_forward_model_fingerprint()
    test_forward_prediction_specs()


if __name__ == "__main__":
    main()
