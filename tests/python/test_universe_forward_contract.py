#!/usr/bin/env python3
"""Verify the PASS-only universe-forward selection contract."""

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
    CheckpointSelection, ForwardFitSpec, forward_fit_specs,
    read_passing_scaling_outcome, resolve_prior_checkpoint,
)
from tools.universe_scaling_contract import (
    EXPECTED_BUDGETS, SEEDS, FitJob, fit_provenance_id,
)

MASTER = tuple(f"S{index:02}" for index in range(55))
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
    change_summary: object | None = None,
    change_outcome: object | None = None,
) -> tuple[FileBinding, object, FitClosure]:
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
        "status": "pass",
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
            **{name: {"pass": True} for name in GATES},
            "all_pass": True,
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
        "exit": 0,
        "status": "pass",
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
        coverage=SimpleNamespace(master=MASTER),
    )
    return FileBinding(
        "experiments/run-outcome.json", file_sha256(outcome_path),
    ), manifest, closure


def read_fixture(
    expected: FileBinding, manifest: object,
    closure: FitClosure, root: Path, target_phase: str | None = None,
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
        return (
            read_passing_scaling_outcome(expected, root=root)
            if target_phase is None else
            forward_fit_specs(expected, target_phase, root=root)
        )


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
            )
        )
        raises(
            read_fixture, replace(expected, sha256=digest(90)),
            manifest, closure, root, "fold-1",
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


def main() -> None:
    test_pass_reader()
    test_pass_rejections()
    test_checkpoint_selection()
    test_forward_fit_specs()


if __name__ == "__main__":
    main()
