#!/usr/bin/env python3
"""Verify the one-shot context phase and receipt boundary."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.context_diagnostic_contract import (
    BATCH_SIZE, CONTROL_COHORT, EXPECTED_FITS_PER_PHASE,
    EXPECTED_PREDICTIONS_PER_PHASE, HISTORY_LENGTHS, PHASE_RANGES, SEEDS,
    ContextPhase, context_phase_sha256,
)
from tools.finalize_context_diagnostic import finalize_context_history
from tools.files import write_json
import tools.run_context_diagnostic as runner
from tools.run_context_diagnostic import (
    RunClaim, claim_run, execute_phase, phase_artifacts,
    read_authorized_truth,
)
from tools.universe_scaling import BOOTSTRAP_BLOCK_DAYS
from tools.universe_scaling_contract import FitJob, fit_provenance_id

MASTER = tuple(f"S{index:02d}" for index in range(55))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def raises(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except (OSError, ValueError):
        return
    raise AssertionError("expected a lifecycle failure")


def phase_value(name: str = "fold-1") -> dict[str, object]:
    training = [
        {"count": 100 + index, "series": series}
        for index, series in enumerate(MASTER[:44])
    ]
    source_phase = {"fold-1": "fold-0", "calibration": "fold-1"}[name]
    return {
        "evaluation_grid_sha256": digest(f"{name}-evaluation"),
        "evaluation_rows": [
            {
                "count": 2,
                "grid_sha256": digest(f"{name}-{series}"),
                "series": series,
            }
            for series in MASTER[44:]
        ],
        "phase": name,
        "prior_selections": [
            {
                "model": model,
                "seed": seed,
                "selected_checkpoint": index + 1,
                "source_model_fingerprint": digest(
                    f"{name}-{model}-{seed}-model",
                ),
                "source_provenance_id": fit_provenance_id(FitJob(
                    "pooled", "fixed-update", 44, source_phase,
                    model, seed, MASTER[:44],
                )),
            }
            for model in ("global_mlp", "panel_transformer")
            for index, seed in enumerate(SEEDS)
        ],
        "source_ranges": list(map(list, PHASE_RANGES[name])),
        "scaler_inputs_sha256": digest(f"{name}-scaler-inputs"),
        "training_grid_sha256": digest(f"{name}-training"),
        "training_rows": training,
        "updates_per_checkpoint": (
            sum(
                row["count"] for row in training[:CONTROL_COHORT]
            ) + BATCH_SIZE - 1
        ) // BATCH_SIZE,
    }


def evaluation_for(
    phase: ContextPhase, lower: float = 0.1,
) -> dict[str, object]:
    return {
        "descriptive_metrics": {},
        "evidence_role": "development-diagnostic-not-forward-clean",
        "phase": phase.phase,
        "phase_sha256": context_phase_sha256(phase),
        "primary": {
            str(history): {
                "intervals": {
                    str(block): (lower, lower + 0.1)
                    for block in BOOTSTRAP_BLOCK_DAYS
                },
            }
            for history in HISTORY_LENGTHS[1:]
        },
        "schema": 1,
    }


def setup_attempt(parent: str) -> tuple[Path, Path]:
    root = (Path(parent) / "repository").resolve()
    (root / "experiments").mkdir(parents=True)
    (root / "reports").mkdir()
    attempt = root / "experiments" / "context-run-attempt.json"
    write_json(attempt, {"schema": 1})
    return root, attempt


def setup_run(parent: str) -> tuple[RunClaim, ContextPhase]:
    root, attempt = setup_attempt(parent)
    return claim_run(root, attempt), ContextPhase.parse(
        phase_value(), MASTER,
    )


def execute_fixture(
    claim: RunClaim, phase: ContextPhase, *,
    fail_fit: int | None = None,
    fail_prediction: int | None = None,
    mutate_attempt: str | None = None,
    mutate_run: bool = False,
    mutate_access: str | None = None,
    mutate_truth_input: str | None = None,
    truth_failure: bool = False,
    evaluation: Mapping[str, object] | None = None,
) -> tuple[list[str], Mapping[str, object]]:
    events: list[str] = []
    fits = 0
    predictions = 0

    def fit_one(fit: object) -> tuple[str, float, object]:
        nonlocal fits
        fits += 1
        events.append(f"fit:{fits}")
        if mutate_attempt == "replace" and fits == 1:
            write_json(claim.attempt, {"schema": 2})
        if mutate_attempt == "in-place" and fits == 1:
            claim.attempt.write_text('{"schema":2}\n', encoding="utf-8")
        if mutate_run and fits == 1:
            claim.path.rename(claim.path.with_name("original-run"))
            claim.path.mkdir()
        if fits == fail_fit:
            raise RuntimeError("synthetic fit failure")
        return digest(f"state-{fits}"), fits / 100.0, fit

    def predict_one(prediction: object, model: object) -> list[float]:
        nonlocal predictions
        assert fits == EXPECTED_FITS_PER_PHASE
        assert model == prediction.fit
        predictions += 1
        events.append(f"prediction:{predictions}")
        if predictions == fail_prediction:
            raise RuntimeError("synthetic prediction failure")
        return [predictions / 100.0] * prediction.prediction_count

    def truth() -> Mapping[str, object]:
        artifacts = phase_artifacts(
            claim.root, claim.attempt, phase,
        )
        access = artifacts.access
        assert access.exists()
        events.append("truth")
        if mutate_access in ("delete", "replace"):
            access.unlink()
            if mutate_access == "replace":
                write_json(access, {"schema": 2})
        if mutate_access == "in-place":
            access.write_text('{"schema":2}\n', encoding="utf-8")
        if mutate_truth_input == "run":
            claim.path.rename(claim.path.with_name("truth-original-run"))
            claim.path.mkdir()
        elif mutate_truth_input:
            path = (
                claim.attempt if mutate_truth_input == "attempt"
                else getattr(artifacts, mutate_truth_input)
            )
            path.write_text('{"schema":2}\n', encoding="utf-8")
        if truth_failure:
            raise RuntimeError("synthetic truth failure")
        return (
            {"authorized": True} if evaluation is None else evaluation
        )

    result = execute_phase(
        claim, MASTER, phase, digest("source-failure"),
        digest("config"), digest("source-tree"),
        fit_one, predict_one, truth,
    )
    return events, result


def test_claim_is_parent_synced() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-claim-sync-", dir=ROOT,
    ) as parent:
        root, attempt = setup_attempt(parent)
        original = runner.os.fsync
        identities: list[tuple[int, int]] = []

        def fsync(descriptor: int) -> None:
            value = os.fstat(descriptor)
            identities.append((value.st_dev, value.st_ino))
            original(descriptor)

        runner.os.fsync = fsync
        try:
            claim = claim_run(root, attempt)
        finally:
            runner.os.fsync = original
        parent_value = (root / "reports").stat()
        assert identities == [
            claim.directory_identity,
            (parent_value.st_dev, parent_value.st_ino),
        ]


def test_claim_sync_failure_burns_run() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-claim-failure-", dir=ROOT,
    ) as parent:
        root, attempt = setup_attempt(parent)
        original = runner.os.fsync
        calls = 0

        def fsync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic parent fsync failure")
            original(descriptor)

        runner.os.fsync = fsync
        try:
            raises(claim_run, root, attempt)
        finally:
            runner.os.fsync = original
        assert (root / "reports" / "context-run").is_dir()
        raises(claim_run, root, attempt)


def test_claim_and_phase_barrier() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-driver-", dir=ROOT,
    ) as parent:
        claim, phase = setup_run(parent)
        raises(claim_run, claim.root, claim.attempt)
        artifacts = phase_artifacts(claim.root, claim.attempt, phase)
        assert {path.parent for path in artifacts} == {claim.path}

        events, result = execute_fixture(claim, phase)
        assert result == {"authorized": True}
        assert events[:EXPECTED_FITS_PER_PHASE] == [
            f"fit:{index}"
            for index in range(1, EXPECTED_FITS_PER_PHASE + 1)
        ]
        assert events[EXPECTED_FITS_PER_PHASE:-1] == [
            f"prediction:{index}"
            for index in range(1, EXPECTED_PREDICTIONS_PER_PHASE + 1)
        ]
        assert events[-1] == "truth"
        assert all(path.exists() for path in artifacts)
        assert artifacts.evaluation.read_text(encoding="utf-8") == \
            '{\n  "authorized": true\n}\n'

        work = 0

        def fit_again(_fit: object) -> tuple[str, float, object]:
            nonlocal work
            work += 1
            raise AssertionError("completed phase was recomputed")

        raises(
            execute_phase,
            claim, MASTER, phase, digest("source-failure"),
            digest("config"), digest("source-tree"), fit_again,
            lambda _prediction, _model: (), lambda: None,
        )
        assert work == 0

        calls = 0

        def truth() -> object:
            nonlocal calls
            calls += 1
            return None

        raises(
            read_authorized_truth,
            claim, MASTER, phase,
            digest("source-failure"), digest("config"),
            digest("source-tree"), truth,
        )
        assert calls == 0


def test_failed_phase_is_not_retryable() -> None:
    for failure in ("fit", "prediction"):
        with tempfile.TemporaryDirectory(
            prefix=f"context-{failure}-", dir=ROOT,
        ) as parent:
            claim, phase = setup_run(parent)
            try:
                execute_fixture(
                    claim, phase,
                    fail_fit=(
                        EXPECTED_FITS_PER_PHASE
                        if failure == "fit" else None
                    ),
                    fail_prediction=(
                        EXPECTED_PREDICTIONS_PER_PHASE
                        if failure == "prediction" else None
                    ),
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("synthetic phase failure was ignored")
            assert not any(
                path.exists()
                for path in phase_artifacts(
                    claim.root, claim.attempt, phase,
                )
            )

            work = 0

            def fit_again(_fit: object) -> tuple[str, float, object]:
                nonlocal work
                work += 1
                raise AssertionError("failed phase was recomputed")

            raises(
                execute_phase,
                claim, MASTER, phase, digest("source-failure"),
                digest("config"), digest("source-tree"), fit_again,
                lambda _prediction, _model: (), lambda: None,
            )
            assert work == 0


def test_calibration_requires_completed_fold() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-calibration-first-", dir=ROOT,
    ) as parent:
        claim, _fold = setup_run(parent)
        calibration = ContextPhase.parse(
            phase_value("calibration"), MASTER,
        )
        raises(execute_fixture, claim, calibration)
        assert not claim.started and not claim.completed
        assert not any(path.exists() for path in phase_artifacts(
            claim.root, claim.attempt, calibration,
        ))

    with tempfile.TemporaryDirectory(
        prefix="context-calibration-failed-fold-", dir=ROOT,
    ) as parent:
        claim, fold = setup_run(parent)
        try:
            execute_fixture(claim, fold, fail_fit=1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("synthetic fold failure was ignored")
        calibration = ContextPhase.parse(
            phase_value("calibration"), MASTER,
        )
        raises(execute_fixture, claim, calibration)
        assert claim.started == {"fold-1"} and not claim.completed
        assert not any(path.exists() for path in phase_artifacts(
            claim.root, claim.attempt, calibration,
        ))


def test_attempt_mutation_blocks_publication() -> None:
    for mode in ("in-place", "replace"):
        with tempfile.TemporaryDirectory(
            prefix=f"context-attempt-{mode}-", dir=ROOT,
        ) as parent:
            claim, phase = setup_run(parent)
            raises(
                execute_fixture, claim, phase, mutate_attempt=mode,
            )
            assert not any(
                path.exists()
                for path in phase_artifacts(
                    claim.root, claim.attempt, phase,
                )
            )


def test_run_substitution_blocks_publication() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-run-swap-", dir=ROOT,
    ) as parent:
        claim, phase = setup_run(parent)
        raises(execute_fixture, claim, phase, mutate_run=True)
        assert not any(
            path.exists()
            for path in phase_artifacts(claim.root, claim.attempt, phase)
        )


def test_truth_failure_is_terminal() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-truth-", dir=ROOT,
    ) as parent:
        claim, phase = setup_run(parent)
        try:
            execute_fixture(claim, phase, truth_failure=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("synthetic truth failure was ignored")
        artifacts = phase_artifacts(claim.root, claim.attempt, phase)
        assert all(path.exists() for path in tuple(artifacts)[:-1])
        assert not artifacts.evaluation.exists()

        calls = 0

        def truth() -> object:
            nonlocal calls
            calls += 1
            return None

        raises(
            read_authorized_truth,
            claim, MASTER, phase,
            digest("source-failure"), digest("config"),
            digest("source-tree"), truth,
        )
        assert calls == 0


def test_truth_access_mutation_is_terminal() -> None:
    for mode in ("delete", "in-place", "replace"):
        with tempfile.TemporaryDirectory(
            prefix=f"context-access-{mode}-", dir=ROOT,
        ) as parent:
            claim, phase = setup_run(parent)
            raises(
                execute_fixture, claim, phase, mutate_access=mode,
            )
            artifacts = phase_artifacts(
                claim.root, claim.attempt, phase,
            )
            assert artifacts.access.exists()
            assert not artifacts.evaluation.exists()

            calls = 0

            def truth() -> object:
                nonlocal calls
                calls += 1
                return {"authorized": True}

            raises(
                read_authorized_truth,
                claim, MASTER, phase,
                digest("source-failure"), digest("config"),
                digest("source-tree"), truth,
            )
            assert calls == 0


def test_truth_boundary_revalidates_every_input() -> None:
    for name in ("attempt", "fits", "predictions", "receipt", "run"):
        with tempfile.TemporaryDirectory(
            prefix=f"context-truth-{name}-", dir=ROOT,
        ) as parent:
            claim, phase = setup_run(parent)
            raises(
                execute_fixture, claim, phase,
                mutate_truth_input=name,
            )
            run = (
                claim.path.with_name("truth-original-run")
                if name == "run" else claim.path
            )
            artifacts = phase_artifacts(
                claim.root, claim.attempt, phase,
            )
            assert (run / artifacts.access.name).exists()
            assert not (run / artifacts.evaluation.name).exists()
            assert not claim.completed


def completed_run(
    parent: str,
) -> tuple[RunClaim, tuple[ContextPhase, ContextPhase]]:
    claim, fold = setup_run(parent)
    calibration = ContextPhase.parse(
        phase_value("calibration"), MASTER,
    )
    phases = (fold, calibration)
    for phase in phases:
        execute_fixture(
            claim, phase, evaluation=evaluation_for(phase),
        )
    return claim, phases


def test_terminal_outcome_binds_phase_evaluations() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-outcome-", dir=ROOT,
    ) as parent:
        claim, phases = completed_run(parent)
        assert tuple(claim.completed) == ("fold-1", "calibration")
        output = finalize_context_history(claim, MASTER, phases)
        assert output["decision"] == {
            "qualifies": {"34": True, "68": True},
            "selected_history": 34,
        }
        outcome = claim.path / "outcome.json"
        assert json.loads(outcome.read_text(encoding="utf-8")) == output
        raises(finalize_context_history, claim, MASTER, phases)

    for mode in ("in-place", "replace"):
        with tempfile.TemporaryDirectory(
            prefix=f"context-evaluation-{mode}-", dir=ROOT,
        ) as parent:
            claim, phases = completed_run(parent)
            evaluation = phase_artifacts(
                claim.root, claim.attempt, phases[0],
            ).evaluation
            forged = evaluation_for(phases[0], lower=-0.2)
            if mode == "in-place":
                evaluation.write_text(
                    json.dumps(forged, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            else:
                evaluation.unlink()
                write_json(evaluation, forged)
            raises(finalize_context_history, claim, MASTER, phases)
            assert not (claim.path / "outcome.json").exists()


def test_terminal_outcome_rejects_mixed_provenance() -> None:
    fields = (
        "source_failure_sha256", "config_sha256", "source_tree_sha256",
    )
    for field in fields:
        with tempfile.TemporaryDirectory(
            prefix=f"context-provenance-{field}-", dir=ROOT,
        ) as parent:
            claim, phases = completed_run(parent)
            evidence = claim.completed["calibration"]
            claim.completed["calibration"] = replace(
                evidence, **{field: digest(f"changed-{field}")},
            )
            raises(finalize_context_history, claim, MASTER, phases)
            assert not (claim.path / "outcome.json").exists()


def test_terminal_outcome_publication_is_atomic() -> None:
    for failure, present in ((1, False), (2, True)):
        with tempfile.TemporaryDirectory(
            prefix=f"context-outcome-fsync-{failure}-", dir=ROOT,
        ) as parent:
            claim, phases = completed_run(parent)
            original = runner.os.fsync
            calls = 0

            def fsync(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                if calls == failure:
                    raise OSError("synthetic outcome fsync failure")
                original(descriptor)

            runner.os.fsync = fsync
            try:
                raises(finalize_context_history, claim, MASTER, phases)
            finally:
                runner.os.fsync = original
            assert (claim.path / "outcome.json").exists() is present
            assert not tuple(claim.path.glob(".*.tmp"))


def inject_fsync_failure(
    claim: RunClaim, phase: ContextPhase, failure: int,
) -> None:
    original = runner.os.fsync
    calls = 0

    def fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == failure:
            raise OSError("synthetic fsync failure")
        original(descriptor)

    runner.os.fsync = fsync
    try:
        raises(execute_fixture, claim, phase)
    finally:
        runner.os.fsync = original


def test_publication_failures_preserve_evidence_and_clean_temps() -> None:
    names = ("fits", "predictions", "receipt", "access", "evaluation")
    expected = (
        (), ("fits",), ("fits",), ("fits", "predictions"),
        ("fits", "predictions"), ("fits", "predictions", "receipt"),
        ("fits", "predictions", "receipt"),
        ("fits", "predictions", "receipt", "access"),
        ("fits", "predictions", "receipt", "access"), names,
    )
    for failure, present in enumerate(expected, 1):
        with tempfile.TemporaryDirectory(
            prefix=f"context-publish-{failure}-", dir=ROOT,
        ) as parent:
            claim, phase = setup_run(parent)
            inject_fsync_failure(claim, phase, failure)
            artifacts = phase_artifacts(claim.root, claim.attempt, phase)
            actual = tuple(
                name for name, path in zip(names, artifacts, strict=True)
                if path.exists()
            )
            assert actual == present
            assert not tuple(claim.path.glob(".*.tmp"))


def test_pre_access_failure_allows_one_validated_recovery() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-recovery-", dir=ROOT,
    ) as parent:
        claim, phase = setup_run(parent)
        inject_fsync_failure(claim, phase, 7)
        artifacts = phase_artifacts(claim.root, claim.attempt, phase)
        assert artifacts.receipt.exists() and \
            not artifacts.access.exists() and \
            not artifacts.evaluation.exists()

        original = claim.path.with_name("original-run")
        replacement = claim.path.with_name("replacement-run")
        claim.path.rename(original)
        claim.path.mkdir()
        for path in artifacts:
            if path not in (artifacts.access, artifacts.evaluation):
                path.hardlink_to(original / path.name)

        calls = 0

        def truth() -> object:
            nonlocal calls
            calls += 1
            return {"authorized": True}

        raises(
            read_authorized_truth,
            claim, MASTER, phase,
            digest("source-failure"), digest("config"),
            digest("source-tree"), truth,
        )
        assert calls == 0 and not artifacts.access.exists()
        claim.path.rename(replacement)
        original.rename(claim.path)

        assert read_authorized_truth(
            claim, MASTER, phase,
            digest("source-failure"), digest("config"),
            digest("source-tree"), truth,
        ) == {"authorized": True}
        assert calls == 1 and artifacts.access.exists() and \
            artifacts.evaluation.exists()


def main() -> None:
    test_claim_is_parent_synced()
    test_claim_sync_failure_burns_run()
    test_claim_and_phase_barrier()
    test_failed_phase_is_not_retryable()
    test_calibration_requires_completed_fold()
    test_attempt_mutation_blocks_publication()
    test_run_substitution_blocks_publication()
    test_truth_failure_is_terminal()
    test_truth_access_mutation_is_terminal()
    test_truth_boundary_revalidates_every_input()
    test_terminal_outcome_binds_phase_evaluations()
    test_terminal_outcome_rejects_mixed_provenance()
    test_terminal_outcome_publication_is_atomic()
    test_publication_failures_preserve_evidence_and_clean_temps()
    test_pre_access_failure_allows_one_validated_recovery()
    print("context diagnostic driver tests passed")


if __name__ == "__main__":
    main()
