#!/usr/bin/env python3
"""Verify the receipt-gated one-shot SPY-residual runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import hashlib
import json
import os
import operator
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from test_relative_context_contract import (
    MASTER, residual_attempt_value, source_context,
)
from tools.context_diagnostic_contract import ContextAttempt, ContextPhase
from tools.files import write_json
from tools.relative_context_contract import (
    EXPECTED_RESIDUAL_FITS_PER_PHASE,
    EXPECTED_RESIDUAL_PREDICTIONS_PER_PHASE,
    ResidualAttempt, ResidualPhaseInput,
)
import tools.run_spy_residual as runner


def digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


def rejects(
    function: Callable[..., object], *args: object, **kwargs: object,
) -> BaseException:
    try:
        function(*args, **kwargs)
    except (
        FrozenInstanceError, OSError, TypeError, ValueError,
        runner.Interrupted,
    ) as error:
        return error
    raise AssertionError("invalid residual lifecycle operation succeeded")


def setup_attempt(
    parent: str, run_id: str = "spy-residual-test",
) -> tuple[Path, Path, ResidualAttempt, ContextAttempt]:
    root = (Path(parent) / "repository").resolve()
    (root / "experiments").mkdir(parents=True)
    (root / "reports").mkdir()
    context = source_context(root)
    attempt_path = root / "experiments" / f"{run_id}-attempt.json"
    write_json(
        attempt_path, residual_attempt_value(root, context, run_id),
    )
    attempt = ResidualAttempt.read(
        attempt_path, attempt_path.relative_to(root), root, context,
    )
    return root, attempt_path, attempt, context


def setup_run(
    parent: str, run_id: str = "spy-residual-test",
) -> tuple[
    runner.ResidualRunClaim, ResidualAttempt,
    tuple[ContextPhase, ...], tuple[ResidualPhaseInput, ...],
]:
    root, attempt_path, attempt, context = setup_attempt(parent, run_id)
    return (
        runner.claim_residual_run(root, attempt_path),
        attempt, context.phases, attempt.phases,
    )


def callbacks(
    source: ContextPhase, events: list[str] | None = None,
) -> tuple[runner.ResidualFitOne, runner.ResidualPredictOne]:
    fits, predictions = iter(
        runner.expected_residual_fits(MASTER, source),
    ), iter(runner.expected_residual_predictions(MASTER, source))

    def fit_one(fit: object) -> tuple[str, float, object]:
        assert fit == next(fits)
        if events is not None:
            events.append("fit")
        return digest(fit), 0.1, fit

    def predict_one(
        prediction: object, model: object,
    ) -> tuple[float, ...]:
        assert prediction == next(predictions)
        assert model == prediction.fit
        if events is not None:
            events.append("predict")
        return (0.0,) * prediction.prediction_count

    return fit_one, predict_one


def execute(
    claim: runner.ResidualRunClaim, attempt: ResidualAttempt,
    source: ContextPhase, phase: ResidualPhaseInput,
    read_truth: runner.ResidualTruthReader = lambda: {},
    events: list[str] | None = None,
) -> Mapping[str, object]:
    fit_one, predict_one = callbacks(source, events)
    with patch.object(
        runner, "_evaluate_residual_phase",
        side_effect=lambda *_: (
            events.append("evaluate") if events is not None else None
        ) or {"phase": source.phase, "schema": 1},
    ):
        return runner.execute_residual_phase(
            claim, attempt, source, phase, fit_one, predict_one,
            read_truth, lambda: None,
        )


def test_claim_is_immutable_registered_and_unforgeable() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-claim-", dir=ROOT,
    ) as parent:
        claim, attempt, sources, phases = setup_run(parent)
        assert claim.completed == {}
        rejects(setattr, claim, "path", claim.path.parent)
        rejects(operator.setitem, claim.completed, "fold-1", object())

        forged = replace(claim)
        fit_one, predict_one = callbacks(sources[0])
        rejects(
            runner.execute_residual_phase,
            forged, attempt, sources[0], phases[0],
            fit_one, predict_one, lambda: {}, lambda: None,
        )
        assert not claim.completed


def test_phase_publishes_exact_label_free_order() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-order-", dir=ROOT,
    ) as parent:
        claim, attempt, sources, phases = setup_run(parent)
        source, phase = sources[0], phases[0]
        artifacts = runner.phase_artifacts(
            claim.root, claim.attempt_path, source,
        )
        events: list[str] = []
        write_ledger, write_json_file = (
            runner._write_ledger, runner._write_json,
        )

        def ledger(path: Path, *args: object, **kwargs: object) -> object:
            result = write_ledger(path, *args, **kwargs)
            events.append(path.name)
            return result

        def json_file(
            path: Path, *args: object, **kwargs: object,
        ) -> object:
            result = write_json_file(path, *args, **kwargs)
            events.append(path.name)
            return result

        def truth() -> Mapping[str, object]:
            assert artifacts.receipt.exists()
            assert artifacts.access.exists()
            assert not artifacts.evaluation.exists()
            events.append("truth")
            return {}

        with patch.object(
            runner, "_write_ledger", side_effect=ledger,
        ), patch.object(runner, "_write_json", side_effect=json_file):
            result = execute(
                claim, attempt, source, phase, truth, events,
            )

        fits = EXPECTED_RESIDUAL_FITS_PER_PHASE
        predictions = EXPECTED_RESIDUAL_PREDICTIONS_PER_PHASE
        assert events[:fits] == ["fit"] * fits
        assert events[fits:fits + predictions] == \
            ["predict"] * predictions
        assert events[fits + predictions:] == [
            artifacts.fits.name,
            artifacts.predictions.name,
            artifacts.receipt.name,
            artifacts.access.name,
            "truth",
            "evaluate",
            artifacts.evaluation.name,
        ]
        assert result == {"phase": source.phase, "schema": 1}
        assert tuple(claim.completed) == (source.phase,)
        runner.validate_residual_ledgers(
            MASTER, source, phase,
            artifacts.fits, artifacts.predictions,
        )


def test_interruption_never_completes_a_durable_boundary() -> None:
    for boundary in (
        "fits", "predictions", "receipt", "access", "evaluation",
    ):
        with tempfile.TemporaryDirectory(
            prefix=f"residual-interrupt-{boundary}-", dir=ROOT,
        ) as parent:
            claim, attempt, sources, phases = setup_run(parent)
            source, phase = sources[0], phases[0]
            target = getattr(
                runner.phase_artifacts(
                    claim.root, claim.attempt_path, source,
                ),
                boundary,
            )
            write_ledger, write_json_file = (
                runner._write_ledger, runner._write_json,
            )

            def interrupt(
                writer: Callable[..., object],
                path: Path, *args: object, **kwargs: object,
            ) -> object:
                result = writer(path, *args, **kwargs)
                if path == target:
                    raise runner.Interrupted(2)
                return result

            with patch.object(
                runner, "_write_ledger",
                side_effect=lambda path, *args, **kwargs: interrupt(
                    write_ledger, path, *args, **kwargs,
                ),
            ), patch.object(
                runner, "_write_json",
                side_effect=lambda path, *args, **kwargs: interrupt(
                    write_json_file, path, *args, **kwargs,
                ),
            ):
                error = rejects(
                    execute, claim, attempt, source, phase,
                )
            assert isinstance(error, runner.Interrupted)
            assert target.exists()
            assert not claim.completed


def test_truth_access_mutation_is_terminal_or_restored() -> None:
    for mode in ("delete", "replace", "hardlink"):
        with tempfile.TemporaryDirectory(
            prefix=f"residual-access-{mode}-", dir=ROOT,
        ) as parent:
            claim, attempt, sources, phases = setup_run(parent)
            source, phase = sources[0], phases[0]
            artifacts = runner.phase_artifacts(
                claim.root, claim.attempt_path, source,
            )
            alias = artifacts.access.with_name("access-alias.json")

            def truth() -> Mapping[str, object]:
                if mode == "delete":
                    artifacts.access.unlink()
                elif mode == "replace":
                    artifacts.access.unlink()
                    write_json(artifacts.access, {"forged": True})
                else:
                    os.link(artifacts.access, alias)
                return {}

            rejects(
                execute, claim, attempt, source, phase, truth,
            )
            assert artifacts.access.exists()
            assert not artifacts.evaluation.exists()
            assert not claim.completed
            if mode == "delete":
                assert "receipt" in json.loads(
                    artifacts.access.read_text(encoding="utf-8"),
                )
            elif mode == "replace":
                assert json.loads(
                    artifacts.access.read_text(encoding="utf-8"),
                ) == {"forged": True}
            else:
                assert alias.exists()


def test_terminal_reauthenticates_both_phases() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-terminal-", dir=ROOT,
    ) as parent:
        claim, attempt, sources, phases = setup_run(parent)
        for source, phase in zip(sources, phases, strict=True):
            execute(claim, attempt, source, phase)
        assert tuple(claim.completed) == ("fold-1", "calibration")

        evaluation = runner.phase_artifacts(
            claim.root, claim.attempt_path, sources[0],
        ).evaluation
        original = evaluation.read_bytes()
        evaluation.write_text("{}\n", encoding="utf-8")
        with patch(
            "tools.finalize_spy_residual.finalize_residual_run",
        ) as finalize:
            rejects(
                runner._finalize_residual_attempt,
                claim, attempt, sources, lambda: None,
            )
            finalize.assert_not_called()

        evaluation.write_bytes(original)
        terminal = {"schema": 1, "status": "completed"}
        with patch(
            "tools.finalize_spy_residual.finalize_residual_run",
            return_value=terminal,
        ) as finalize:
            assert runner._finalize_residual_attempt(
                claim, attempt, sources, lambda: None,
            ) == terminal
            assert finalize.call_count == 1
        outcome = claim.attempt_path.with_name(
            "spy-residual-test-outcome.json",
        )
        assert json.loads(outcome.read_text(encoding="utf-8")) == terminal


def test_failure_schema_never_authorizes_metrics_or_trading() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-failure-", dir=ROOT,
    ) as parent:
        claim, _, _, _ = setup_run(parent)
        assert runner._failure_value(claim, "fold-1") == {
            "attempt": {
                "path": "experiments/spy-residual-test-attempt.json",
                "sha256": claim.attempt_binding.sha256,
            },
            "schema": 1,
            "stage": "fold-1",
            "status": "integrity-failure",
        }


def test_controller_authenticates_before_claim_and_preparation() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-controller-", dir=ROOT,
    ) as parent:
        _, _, attempt, context = setup_attempt(parent)
        events: list[str] = []
        claim, lease, prepared = object(), object(), object()

        @contextmanager
        def authenticate(_attempt: object) -> object:
            events.append("authenticate:enter")
            yield lease
            events.append("authenticate:exit")

        def expose(_path: Path) -> None:
            assert events == ["authenticate:enter"]
            events.append("expose")

        def claim_run(_root: Path, _attempt: Path) -> object:
            assert events == ["authenticate:enter", "expose"]
            events.append("claim")
            return claim

        def prepare(
            _context: object, source: ContextPhase,
            _phase: object, actual_lease: object,
        ) -> tuple[object, object]:
            assert actual_lease is lease and "claim" in events
            events.append(f"prepare:{source.phase}")
            return prepared, lambda: {}

        class Runtime:
            def __init__(
                self, actual: object, device: object, tree: str,
            ) -> None:
                assert (actual, device, tree) == (
                    prepared, "cpu", attempt.source_tree.sha256,
                )
                events.append("runtime")
                self.fit_one = self.predict_one = lambda *_: None

        def execute_phase(
            actual_claim: object, _attempt: object,
            source: ContextPhase, _phase: object,
            *_callbacks: object,
        ) -> None:
            assert actual_claim is claim
            events.append(f"execute:{source.phase}")

        torch = ModuleType("torch")
        torch.device = lambda value: value  # type: ignore[attr-defined]
        runtime = ModuleType("tools.spy_residual_runtime")
        runtime.ResidualRuntime = Runtime  # type: ignore[attr-defined]
        with patch.object(
            runner, "read_residual_attempt",
            return_value=(attempt, context),
        ), patch.object(
            runner, "_validate_controller",
        ), patch.object(
            runner, "_require_package_alias",
        ), patch.object(
            runner, "claim_residual_run", side_effect=claim_run,
        ), patch.object(
            runner, "_prepare_phase", side_effect=prepare,
        ), patch.object(
            runner, "execute_residual_phase", side_effect=execute_phase,
        ), patch.object(
            runner, "_finalize_residual_attempt",
            side_effect=lambda *_: events.append("finalize") or {
                "status": "completed",
            },
        ), patch(
            "tools.arm_spy_residual.authenticate_residual_attempt",
            side_effect=authenticate,
        ), patch(
            "tools.run_universe_scaling._expose_torch_package",
            side_effect=expose,
        ), patch.dict(sys.modules, {
            "tools.spy_residual_runtime": runtime, "torch": torch,
        }):
            assert runner.execute_residual_attempt(Path("attempt.json")) == {
                "status": "completed",
            }

        assert events == [
            "authenticate:enter", "expose", "claim",
            "prepare:fold-1", "runtime", "execute:fold-1",
            "prepare:calibration", "runtime", "execute:calibration",
            "finalize", "authenticate:exit",
        ]


def main() -> None:
    test_claim_is_immutable_registered_and_unforgeable()
    test_phase_publishes_exact_label_free_order()
    test_interruption_never_completes_a_durable_boundary()
    test_truth_access_mutation_is_terminal_or_restored()
    test_terminal_reauthenticates_both_phases()
    test_failure_schema_never_authorizes_metrics_or_trading()
    test_controller_authenticates_before_claim_and_preparation()
    print("SPY residual runner tests passed")


if __name__ == "__main__":
    main()
