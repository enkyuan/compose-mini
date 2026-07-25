#!/usr/bin/env python3
"""Verify the one-shot context phase and receipt boundary."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.context_diagnostic_contract import (
    BATCH_SIZE, CONTROL_COHORT, EXPECTED_FITS_PER_PHASE,
    EXPECTED_PREDICTIONS_PER_PHASE, HISTORY_LENGTHS, PHASE_RANGES, SEEDS,
    CONTEXT_CONFIG, CONTEXT_SOURCE_PATHS, PYTHON_FLAGS, SOURCE_EVIDENCE,
    ContextAttempt, ContextPhase, ContextReceipt, context_phase_sha256,
)
from tools.finalize_context_diagnostic import finalize_context_history
from tools.files import write_json
import tools.run_context_diagnostic as runner
from tools.run_context_diagnostic import (
    PhaseEvidence, RunClaim, claim_run, execute_armed_phase, execute_phase,
    phase_artifacts, read_authorized_truth,
)
from tools.panel_contract import FileBinding, SourceTree, _tree_digest
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


def setup_armed_attempt(parent: str) -> tuple[Path, Path]:
    root, attempt = setup_attempt(parent)
    files = tuple(
        FileBinding(path, digest(path)) for path in CONTEXT_SOURCE_PATHS
    )
    package_files = (FileBinding("torch.py", digest("torch.py")),)
    source_tree = SourceTree(
        str(root), files, _tree_digest(files),
    )
    package_tree = SourceTree(
        str(root / "torch"), package_files, _tree_digest(package_files),
    )
    python = str(Path(sys.executable).resolve())
    write_json(attempt, {
        "attempt_path": "experiments/context-run-attempt.json",
        "config": asdict(CONTEXT_CONFIG),
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": "reports/context-run/.pycache",
        },
        "implementation_commit": "1" * 40,
        "phases": [
            phase_value(name) for name in ("fold-1", "calibration")
        ],
        "primary_python": {
            "path": python,
            "sha256": digest("primary"),
            "version": "synthetic",
        },
        "run_dir": "reports/context-run",
        "run_id": "context-run",
        "schema": 1,
        "source": {
            name: asdict(binding)
            for name, binding in SOURCE_EVIDENCE.items()
        },
        "source_tree": asdict(source_tree),
        "status": "armed",
        "torch_argv": [python, *PYTHON_FLAGS],
        "torch_probe": {
            "config": "cpu",
            "cuda_version": None,
            "git_version": None,
            "package_tree": asdict(package_tree),
            "python": {
                "path": python,
                "sha256": digest("torch-python"),
                "version": "synthetic",
            },
            "version": "synthetic",
        },
    })
    return root, attempt


def setup_armed_run(parent: str) -> RunClaim:
    return claim_run(*setup_armed_attempt(parent))


def armed_callbacks(
    events: list[str], phase: ContextPhase,
) -> tuple[object, object, object]:
    fits = 0

    def fit_one(fit: object) -> tuple[str, float, object]:
        nonlocal fits
        fits += 1
        events.append(f"{phase.phase}:fit")
        return digest(f"{phase.phase}-state-{fits}"), 0.1, fit

    def predict_one(prediction: object, model: object) -> list[float]:
        assert fits == EXPECTED_FITS_PER_PHASE
        assert model == prediction.fit
        return [0.0] * prediction.prediction_count

    def truth(
        predictions: Sequence[object],
    ) -> Mapping[str, object]:
        assert len(predictions) == EXPECTED_PREDICTIONS_PER_PHASE
        events.append(f"{phase.phase}:truth")
        return {"phase": phase.phase}

    return fit_one, predict_one, truth


def test_armed_handoff_owns_phase_and_provenance() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-armed-", dir=ROOT,
    ) as parent:
        claim = setup_armed_run(parent)
        events: list[str] = []

        def prepare(
            attempt: object, phase: ContextPhase, lease: object,
        ) -> tuple[object, object, object]:
            assert claim.started == {
                *claim.completed, phase.phase,
            }
            assert attempt.master == MASTER
            assert callable(lease)
            events.append(f"prepare:{phase.phase}")
            return armed_callbacks(events, phase)

        try:
            execute_armed_phase(claim, prepare)  # type: ignore[call-arg]
        except TypeError:
            pass
        else:
            raise AssertionError("armed preparation remained injectable")
        with patch.object(
            runner, "authenticate_context_attempt",
        ), patch.object(runner, "_prepare_context_phase", new=prepare):
            fold = execute_armed_phase(claim)
            calibration = execute_armed_phase(claim)
        assert fold == {"phase": "fold-1"}
        assert calibration == {"phase": "calibration"}
        assert events[0] == "prepare:fold-1"
        assert events[-1] == "calibration:truth"
        assert tuple(claim.completed) == ("fold-1", "calibration")
        fold_phase = ContextPhase.parse(phase_value(), MASTER)
        receipt = ContextReceipt.parse(json.loads(
            phase_artifacts(
                claim.root, claim.attempt, fold_phase,
            ).receipt.read_text(encoding="utf-8"),
        ))
        assert receipt.source_tree_sha256 == \
            claim.completed["fold-1"].source_tree_sha256
        calls = len(events)
        with patch.object(
            runner, "authenticate_context_attempt",
        ), patch.object(runner, "_prepare_context_phase", new=prepare):
            raises(execute_armed_phase, claim)
        assert len(events) == calls


def test_failed_armed_preparation_is_terminal() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-armed-failure-", dir=ROOT,
    ) as parent:
        claim = setup_armed_run(parent)
        calls = 0

        def invalid(
            _attempt: object, _phase: ContextPhase, _lease: object,
        ) -> tuple[object, ...]:
            nonlocal calls
            calls += 1
            return ()

        with patch.object(
            runner, "authenticate_context_attempt",
        ), patch.object(runner, "_prepare_context_phase", new=invalid):
            raises(execute_armed_phase, claim)
        assert claim.started == {"fold-1"} and calls == 1
        with patch.object(
            runner, "authenticate_context_attempt",
        ), patch.object(runner, "_prepare_context_phase", new=invalid):
            raises(execute_armed_phase, claim)
        assert calls == 1


def test_armed_handoff_rejects_forged_state_and_attempt() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-armed-forgery-", dir=ROOT,
    ) as parent:
        claim = setup_armed_run(parent)
        prepared = 0

        def prepare(
            _attempt: object, _phase: ContextPhase, _lease: object,
        ) -> tuple[object, ...]:
            nonlocal prepared
            prepared += 1
            return ()

        with patch.object(
            runner, "authenticate_context_attempt",
            side_effect=ValueError("synthetic attempt"),
        ), patch.object(runner, "_prepare_context_phase", new=prepare):
            raises(execute_armed_phase, claim)
        assert not claim.started and prepared == 0

        attempt = ContextAttempt.read(
            claim.attempt, claim.attempt.relative_to(claim.root),
            claim.root,
        )
        claim._completed["fold-1"] = PhaseEvidence(
            "fold-1", (), (),
            attempt.source_binding("failure").sha256,
            attempt.config.sha256, attempt.source_tree.sha256,
        )
        claim._started.add("fold-1")
        with patch.object(
            runner, "authenticate_context_attempt",
        ), patch.object(runner, "_prepare_context_phase", new=prepare):
            raises(execute_armed_phase, claim)
        assert prepared == 0


def test_failed_authentication_exit_cannot_complete_a_phase() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-armed-lease-", dir=ROOT,
    ) as parent:
        claim = setup_armed_run(parent)
        prepared = 0

        def prepare(
            _attempt: object, phase: ContextPhase, _lease: object,
        ) -> tuple[object, object, object]:
            nonlocal prepared
            prepared += 1
            return armed_callbacks([], phase)

        @contextmanager
        def failing_lease(_attempt: object) -> Iterator[object]:
            yield lambda: None
            raise ValueError("context lease changed")

        with patch.object(
            runner, "authenticate_context_attempt", new=failing_lease,
        ), patch.object(runner, "_prepare_context_phase", new=prepare):
            raises(execute_armed_phase, claim)
        assert claim.started == {"fold-1"} and not claim.completed
        with patch.object(
            runner, "authenticate_context_attempt",
        ), patch.object(runner, "_prepare_context_phase", new=prepare):
            raises(execute_armed_phase, claim)
        assert prepared == 1


def test_loose_executor_cannot_complete_an_armed_attempt() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-armed-loose-", dir=ROOT,
    ) as parent:
        claim = setup_armed_run(parent)
        attempt = ContextAttempt.read(
            claim.attempt, claim.attempt.relative_to(claim.root),
            claim.root,
        )
        phase = attempt.phases[0]
        fit, predict, truth = armed_callbacks([], phase)
        for candidate in (claim, replace(claim)):
            raises(
                execute_phase, candidate, attempt.master, phase,
                attempt.source_binding("failure").sha256,
                attempt.config.sha256, attempt.source_tree.sha256,
                fit, predict, truth,
            )
        assert not claim.started and not claim.completed


def test_controller_authenticates_before_claim_and_finalizes_after_exit() -> None:
    root_entries = tuple(map(os.path.realpath, sys.path)).count(str(ROOT))
    import tools.finalize_context_diagnostic as finalizer
    import tools.run_universe_scaling as scaling_runner

    assert tuple(map(os.path.realpath, sys.path)).count(str(ROOT)) == \
        root_entries
    with tempfile.TemporaryDirectory(
        prefix="context-controller-", dir=ROOT,
    ) as parent:
        root, attempt_path = setup_armed_attempt(parent)
        events: list[str] = []
        original_claim = runner.claim_run

        @contextmanager
        def authenticate(_attempt: object) -> Iterator[object]:
            events.append("authenticate")
            yield lambda: None
            events.append("lease-exit")

        def claim(run_root: Path, path: Path) -> RunClaim:
            assert events == ["authenticate"]
            events.append("claim")
            return original_claim(run_root, path)

        def prepare(
            _attempt: object, phase: ContextPhase, _lease: object,
        ) -> tuple[object, object, object]:
            events.append(f"prepare:{phase.phase}")
            return armed_callbacks([], phase)

        def finalize(
            claim: RunClaim, _master: object, _phases: object,
        ) -> dict[str, object]:
            assert tuple(claim.completed) == ("fold-1", "calibration")
            events.append("finalize")
            return {"status": "complete"}

        with patch.object(
            runner, "ROOT", root,
        ), patch.object(
            runner, "_validate_controller",
        ), patch.object(
            runner, "authenticate_context_attempt", new=authenticate,
        ), patch.object(
            runner, "claim_run", new=claim,
        ), patch.object(
            runner, "_prepare_context_phase", new=prepare,
        ), patch.object(
            scaling_runner, "_expose_torch_package",
        ), patch.object(
            finalizer, "finalize_context_history", new=finalize,
        ):
            result = runner.execute_context_attempt(
                attempt_path.relative_to(root),
            )
        assert result == {"status": "complete"}
        assert tuple(
            event for event in events if event.startswith("prepare:")
        ) == ("prepare:fold-1", "prepare:calibration")
        assert events.index("claim") > events.index("authenticate")
        assert events.index("finalize") > events.index("lease-exit")


def test_armer_preserves_authenticated_repository_path() -> None:
    script = (
        f"import os,sys; root={str(ROOT)!r}; sys.path.append(root); "
        "import tools.arm_context_diagnostic; "
        "paths=tuple(map(os.path.realpath,sys.path)); "
        "assert paths.count(root)==1 and paths[-1]==root"
    )
    result = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_runner_package_alias_shares_finalizer_state() -> None:
    script = (
        "import importlib.util,sys; "
        f"root={str(ROOT)!r}; "
        f"path={str(ROOT / 'tools/run_context_diagnostic.py')!r}; "
        "sys.path.append(root); "
        "spec=importlib.util.spec_from_file_location("
        "'context_runner_script',path); "
        "runner=importlib.util.module_from_spec(spec); "
        "sys.modules[spec.name]=runner; "
        "spec.loader.exec_module(runner); "
        "runner._register_package_alias(); "
        "import tools.finalize_context_diagnostic as finalizer; "
        "assert finalizer.RunClaim is runner.RunClaim; "
        "assert finalizer.publish_context_outcome.__globals__['_CLAIMS'] "
        "is runner._CLAIMS"
    )
    result = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_controller_failure_publishes_one_terminal_outcome() -> None:
    import tools.run_universe_scaling as scaling_runner

    with tempfile.TemporaryDirectory(
        prefix="context-controller-failure-", dir=ROOT,
    ) as parent:
        root, attempt_path = setup_armed_attempt(parent)
        claimed: list[RunClaim] = []
        original_claim = runner.claim_run

        @contextmanager
        def authenticate(_attempt: object) -> Iterator[object]:
            yield lambda: None

        def claim(run_root: Path, path: Path) -> RunClaim:
            value = original_claim(run_root, path)
            claimed.append(value)
            return value

        def fail(*_args: object) -> object:
            raise ValueError("synthetic preparation failure")

        with patch.object(
            runner, "ROOT", root,
        ), patch.object(
            runner, "_validate_controller",
        ), patch.object(
            runner, "authenticate_context_attempt", new=authenticate,
        ), patch.object(
            runner, "claim_run", new=claim,
        ), patch.object(
            runner, "_prepare_context_phase", new=fail,
        ), patch.object(
            scaling_runner, "_expose_torch_package",
        ):
            raises(
                runner.execute_context_attempt,
                attempt_path.relative_to(root),
            )
        assert len(claimed) == 1
        outcome = json.loads(
            (claimed[0].path / "outcome.json").read_text(encoding="utf-8"),
        )
        assert outcome["stage"] == "fold-1"
        assert outcome["status"] == "integrity-failure"


def test_isolated_controller_bootstrap_rejects_an_unbound_attempt() -> None:
    result = subprocess.run(
        (
            sys.executable, "-I", "-S", "-B",
            "tools/run_context_diagnostic.py",
            "experiments/does-not-exist-attempt.json",
        ),
        cwd=ROOT, env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": "reports/context-smoke/.pycache",
        },
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        "context runner error: "
        "context attempt must be inside the repository\n"
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

    def truth(
        evidence: Sequence[object],
    ) -> Mapping[str, object]:
        assert len(evidence) == EXPECTED_PREDICTIONS_PER_PHASE
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


def test_claim_rejects_a_substituted_directory() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-claim-swap-", dir=ROOT,
    ) as parent:
        root, attempt = setup_attempt(parent)
        create = runner.mkdir_nofollow

        def swap(path: Path) -> tuple[int, int]:
            identity = create(path)
            path.rename(path.with_name("original-run"))
            path.mkdir()
            return identity

        with patch.object(runner, "mkdir_nofollow", side_effect=swap):
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
            lambda _prediction, _model: (), lambda _evidence: None,
        )
        assert work == 0

        calls = 0

        def truth(_evidence: object) -> object:
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
                lambda _prediction, _model: (), lambda _evidence: None,
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

        def truth(_evidence: object) -> object:
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

            def truth(_evidence: object) -> object:
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
            claim._completed["calibration"] = replace(
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
                if present:
                    finalize_context_history(claim, MASTER, phases)
                else:
                    raises(finalize_context_history, claim, MASTER, phases)
            finally:
                runner.os.fsync = original
            assert (claim.path / "outcome.json").exists() is present
            assert not tuple(claim.path.glob(".*.tmp"))


def test_terminal_outcome_rejects_a_forged_commit_signal() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-outcome-forgery-", dir=ROOT,
    ) as parent:
        claim, _ = setup_run(parent)

        def forged() -> None:
            raise runner.PublicationCommitted("forged")

        raises(
            runner.publish_context_outcome,
            claim, {"schema": 1}, forged,
        )
        assert not (claim.path / "outcome.json").exists()


def test_terminal_outcome_uses_the_public_link_as_signal_boundary() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-outcome-signal-", dir=ROOT,
    ) as parent:
        claim, _ = setup_run(parent)

        def interrupt() -> None:
            raise runner.Interrupted(signal.SIGTERM)

        try:
            runner.publish_context_outcome(claim, {"schema": 1}, interrupt)
        except runner.Interrupted as error:
            assert error.number == signal.SIGTERM
        else:
            raise AssertionError("pre-link signal was accepted")
        assert not (claim.path / "outcome.json").exists()

        directory, interrupted, original_open = None, False, runner._open_directory
        original_close = runner.os.close

        def open_directory(path: Path) -> tuple[int, tuple[int, int]]:
            nonlocal directory
            directory, identity = original_open(path)
            return directory, identity

        def close(descriptor: int) -> None:
            nonlocal interrupted
            original_close(descriptor)
            if descriptor == directory and not interrupted:
                interrupted = True
                raise runner.Interrupted(signal.SIGTERM)

        with patch.object(
            runner, "_open_directory", new=open_directory,
        ), patch.object(runner.os, "close", new=close):
            runner.publish_context_outcome(claim, {"schema": 1}, lambda: None)
        assert (claim.path / "outcome.json").exists()


def test_terminal_outcome_rechecks_after_a_confirmation_signal() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-outcome-confirmation-", dir=ROOT,
    ) as parent:
        claim, _ = setup_run(parent)
        original_fsync = runner.os.fsync
        original_identity = runner._regular_identity
        fsync_calls = identity_calls = 0

        def fsync(descriptor: int) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("synthetic directory fsync failure")
            original_fsync(descriptor)

        def regular_identity(path: Path) -> tuple[int, int]:
            nonlocal identity_calls
            identity_calls += 1
            if identity_calls == 1:
                raise runner.Interrupted(signal.SIGTERM)
            return original_identity(path)

        with patch.object(
            runner.os, "fsync", new=fsync,
        ), patch.object(
            runner, "_regular_identity", new=regular_identity,
        ):
            runner.publish_context_outcome(claim, {"schema": 1}, lambda: None)
        assert (claim.path / "outcome.json").exists()


def test_terminal_outcome_waits_for_snapshot_cleanup() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-outcome-cleanup-", dir=ROOT,
    ) as parent:
        claim, phases = completed_run(parent)
        original = tempfile.TemporaryDirectory.cleanup

        def cleanup(directory: tempfile.TemporaryDirectory[str]) -> None:
            original(directory)
            raise runner.Interrupted(signal.SIGTERM)

        with patch.object(
            tempfile.TemporaryDirectory, "cleanup", new=cleanup,
        ):
            try:
                finalize_context_history(claim, MASTER, phases)
            except runner.Interrupted as error:
                assert error.number == signal.SIGTERM
            else:
                raise AssertionError("snapshot cleanup signal was accepted")
        assert not (claim.path / "outcome.json").exists()


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

        def truth(_evidence: object) -> object:
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


def test_first_signal_sets_the_shell_status_and_restores_handlers() -> None:
    installed, restored = {}, []

    def install(number: int, handler: object) -> object:
        if number not in installed:
            installed[number] = handler
            return signal.SIG_DFL
        restored.append((number, handler))
        return installed[number]

    def interrupt(_path: Path) -> None:
        try:
            installed[signal.SIGTERM](signal.SIGTERM, None)
        except runner.Interrupted:
            installed[signal.SIGINT](signal.SIGINT, None)
            raise

    with patch.object(
        runner.signal, "signal", side_effect=install,
    ), patch.object(
        runner.signal, "getsignal", return_value=signal.SIG_DFL,
    ), patch.object(runner, "execute_context_attempt", new=interrupt):
        assert runner._run(Path("attempt.json")) == 128 + signal.SIGTERM
    assert tuple(number for number, _ in restored) == runner.SIGNALS
    assert all(handler == signal.SIG_DFL for _, handler in restored)


def test_signal_during_handler_installation_restores_partial_state() -> None:
    installed, restored = {}, []

    def install(number: int, handler: object) -> None:
        if handler == signal.SIG_DFL:
            restored.append(number)
            return
        installed[number] = handler
        if number == signal.SIGINT:
            installed[signal.SIGHUP](signal.SIGHUP, None)

    with patch.object(
        runner.signal, "signal", side_effect=install,
    ), patch.object(
        runner.signal, "getsignal", return_value=signal.SIG_DFL,
    ), patch.object(
        runner, "execute_context_attempt",
    ) as execute:
        assert runner._run(Path("attempt.json")) == 128 + signal.SIGHUP
        execute.assert_not_called()
    assert tuple(restored) == (signal.SIGHUP, signal.SIGINT)


def test_post_execution_signal_cannot_break_handler_restoration() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-restoration-signal-", dir=ROOT,
    ) as parent:
        claim, _ = setup_run(parent)
        installed, restored = {}, []
        interrupted = False

        def install(number: int, handler: object) -> None:
            nonlocal interrupted
            if handler != signal.SIG_DFL:
                installed[number] = handler
                return
            if not interrupted:
                interrupted = True
                installed[signal.SIGTERM](signal.SIGTERM, None)
            restored.append(number)

        def execute(_path: Path) -> None:
            runner.publish_context_outcome(
                claim, {"schema": 1}, lambda: None,
            )

        with patch.object(
            runner.signal, "signal", side_effect=install,
        ), patch.object(
            runner.signal, "getsignal", return_value=signal.SIG_DFL,
        ), patch.object(
            runner, "execute_context_attempt", new=execute,
        ):
            assert runner._run(claim.attempt) == 0
        assert tuple(restored) == runner.SIGNALS


def test_success_requires_an_authenticated_terminal_outcome() -> None:
    with patch.object(
        runner, "execute_context_attempt", return_value={},
    ):
        raises(runner._run, Path("attempt.json"))


def test_post_publication_signal_uses_the_terminal_outcome() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-post-publication-signal-", dir=ROOT,
    ) as parent:
        claim, _ = setup_run(parent)
        installed = {}

        def install(number: int, handler: object) -> None:
            installed[number] = handler

        def execute(_path: Path) -> None:
            runner.publish_context_outcome(
                claim, {"schema": 1}, lambda: None,
            )
            installed[signal.SIGTERM](signal.SIGTERM, None)

        with patch.object(
            runner.signal, "signal", side_effect=install,
        ), patch.object(
            runner.signal, "getsignal", return_value=signal.SIG_DFL,
        ), patch.object(
            runner, "execute_context_attempt", new=execute,
        ):
            assert runner._run(claim.attempt) == 0
        assert (claim.path / "outcome.json").exists()


def test_signal_during_terminal_verification_is_ignored() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-verification-signal-", dir=ROOT,
    ) as parent:
        claim, _ = setup_run(parent)
        original = runner._verify_terminal_outcome

        def verify(value: object) -> None:
            os.kill(os.getpid(), signal.SIGTERM)
            original(value)  # type: ignore[arg-type]

        def execute(_path: Path) -> None:
            runner.publish_context_outcome(
                claim, {"schema": 1}, lambda: None,
            )

        with patch.object(
            runner, "_verify_terminal_outcome", new=verify,
        ), patch.object(
            runner, "execute_context_attempt", new=execute,
        ):
            assert runner._run(claim.attempt) == 0
        assert (claim.path / "outcome.json").exists()


def test_pre_link_signal_aborts_terminal_publication() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-pre-link-signal-", dir=ROOT,
    ) as parent:
        claim, _ = setup_run(parent)

        def execute(_path: Path) -> None:
            runner.publish_context_outcome(
                claim, {"schema": 1},
                lambda: os.kill(os.getpid(), signal.SIGTERM),
            )

        with patch.object(
            runner, "execute_context_attempt", new=execute,
        ):
            assert runner._run(claim.attempt) == 128 + signal.SIGTERM
        assert not (claim.path / "outcome.json").exists()


def test_post_link_signal_authenticates_the_owned_outcome() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-post-link-signal-", dir=ROOT,
    ) as parent:
        claim, _ = setup_run(parent)
        original = runner.exclusive_text

        def exclusive(*args: object, **kwargs: object) -> None:
            original(*args, **kwargs)  # type: ignore[arg-type]
            os.kill(os.getpid(), signal.SIGTERM)

        def execute(_path: Path) -> None:
            runner.publish_context_outcome(
                claim, {"schema": 1}, lambda: None,
            )

        with patch.object(
            runner, "exclusive_text", new=exclusive,
        ), patch.object(
            runner, "execute_context_attempt", new=execute,
        ):
            assert runner._run(claim.attempt) == 0
        assert (claim.path / "outcome.json").exists()


def test_terminal_marker_rejects_a_post_link_replacement() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-terminal-replacement-", dir=ROOT,
    ) as parent:
        claim, _ = setup_run(parent)
        original = runner._write_json

        def write_json(
            path: Path, *args: object, **kwargs: object,
        ) -> object:
            published = original(
                path, *args, **kwargs,  # type: ignore[arg-type]
            )
            path.unlink()
            path.write_text('{"foreign": true}\n', encoding="utf-8")
            return published

        def execute(_path: Path) -> None:
            runner.publish_context_outcome(
                claim, {"schema": 1}, lambda: None,
            )

        with patch.object(
            runner, "_write_json", new=write_json,
        ), patch.object(
            runner, "execute_context_attempt", new=execute,
        ):
            raises(runner._run, claim.attempt)


def test_terminal_marker_binds_content_mode_and_link_count() -> None:
    def reject(name: str, mutate: object) -> None:
        with tempfile.TemporaryDirectory(
            prefix=f"context-terminal-{name}-", dir=ROOT,
        ) as parent:
            claim, _ = setup_run(parent)

            def execute(_path: Path) -> None:
                outcome = runner.publish_context_outcome(
                    claim, {"schema": 1}, lambda: None,
                )
                mutate(outcome)  # type: ignore[operator]

            with patch.object(
                runner, "execute_context_attempt", new=execute,
            ):
                raises(runner._run, claim.attempt)

    reject(
        "content",
        lambda path: path.write_text(
            '{"changed": true}\n', encoding="utf-8",
        ),
    )
    reject("mode", lambda path: path.chmod(0o400))
    reject(
        "links",
        lambda path: path.with_name("outcome-link.json").hardlink_to(path),
    )

    def replace_with_fifo(path: Path) -> None:
        path.unlink()
        os.mkfifo(path)

    reject("fifo", replace_with_fifo)


def main() -> None:
    test_armed_handoff_owns_phase_and_provenance()
    test_failed_armed_preparation_is_terminal()
    test_armed_handoff_rejects_forged_state_and_attempt()
    test_failed_authentication_exit_cannot_complete_a_phase()
    test_loose_executor_cannot_complete_an_armed_attempt()
    test_controller_authenticates_before_claim_and_finalizes_after_exit()
    test_armer_preserves_authenticated_repository_path()
    test_runner_package_alias_shares_finalizer_state()
    test_controller_failure_publishes_one_terminal_outcome()
    test_isolated_controller_bootstrap_rejects_an_unbound_attempt()
    test_claim_is_parent_synced()
    test_claim_sync_failure_burns_run()
    test_claim_rejects_a_substituted_directory()
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
    test_terminal_outcome_rejects_a_forged_commit_signal()
    test_terminal_outcome_uses_the_public_link_as_signal_boundary()
    test_terminal_outcome_rechecks_after_a_confirmation_signal()
    test_terminal_outcome_waits_for_snapshot_cleanup()
    test_publication_failures_preserve_evidence_and_clean_temps()
    test_pre_access_failure_allows_one_validated_recovery()
    test_first_signal_sets_the_shell_status_and_restores_handlers()
    test_signal_during_handler_installation_restores_partial_state()
    test_post_execution_signal_cannot_break_handler_restoration()
    test_success_requires_an_authenticated_terminal_outcome()
    test_post_publication_signal_uses_the_terminal_outcome()
    test_signal_during_terminal_verification_is_ignored()
    test_pre_link_signal_aborts_terminal_publication()
    test_post_link_signal_authenticates_the_owned_outcome()
    test_terminal_marker_rejects_a_post_link_replacement()
    test_terminal_marker_binds_content_mode_and_link_count()
    print("context diagnostic driver tests passed")


if __name__ == "__main__":
    main()
