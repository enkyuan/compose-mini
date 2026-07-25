#!/usr/bin/env python3
"""Verify context arming is exact, atomic, and one-shot."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch
import hashlib
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import arm_context_diagnostic as armer
from tools import context_diagnostic_contract as contract
from tools.context_diagnostic_contract import (
    BATCH_SIZE, PHASE_RANGES, SEEDS, ContextPhase,
)
from tools.files import FrozenInput, file_sha256, freeze_inputs, verify_frozen
from tools.panel_contract import (
    ExecutableBinding, FileBinding, SourceTree, TorchIdentity,
    _tree_digest, read_canonical_json, selected_source_tree,
)
from tools.universe_scaling_contract import FitJob, fit_provenance_id

MASTER = tuple(f"S{index:02d}" for index in range(55))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def binding(root: Path, path: Path) -> FileBinding:
    return FileBinding(path.relative_to(root).as_posix(), file_sha256(path))


def raises(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except (OSError, ValueError):
        return
    raise AssertionError("expected context arming failure")


def authenticate(attempt: object) -> None:
    with armer.authenticate_context_attempt(
        attempt,  # type: ignore[arg-type]
    ) as lease:
        assert type(lease) is armer.ContextLease
        assert callable(lease)
        lease()


def test_private_snapshot_mutation_fails_verification() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-snapshot-", dir=ROOT,
    ) as directory:
        source = write(Path(directory) / "source", "original")
        with freeze_inputs((source,)) as frozen:
            snapshot = frozen[0].snapshot
            snapshot.chmod(0o600)
            snapshot.write_text("changed", encoding="utf-8")
            raises(verify_frozen, frozen)


def phase_value(name: str) -> dict[str, object]:
    source_phase = {"fold-1": "fold-0", "calibration": "fold-1"}[name]
    training = [
        {"count": 100 + index, "series": series}
        for index, series in enumerate(MASTER[:44])
    ]
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
        "scaler_inputs_sha256": digest(f"{name}-scalers"),
        "source_ranges": list(map(list, PHASE_RANGES[name])),
        "training_grid_sha256": digest(f"{name}-training"),
        "training_rows": training,
        "updates_per_checkpoint": (
            sum(item["count"] for item in training[:11]) +
            BATCH_SIZE - 1
        ) // BATCH_SIZE,
    }


class ArmFixture:
    run_id = "context-arm-test"
    output = Path(f"experiments/{run_id}-attempt.json")

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        for name in ("experiments", "reports", "inputs", "source"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.source_paths = (
            "source/arm.py", "source/contract.py", "source/runner.py",
        )
        for name in self.source_paths:
            write(self.root / name, f"{name}\n")

        self.source_files = {
            name: write(self.root / f"inputs/{name}.json", f"{name}\n")
            for name in ("attempt", "failure", "fits")
        }
        self.source = MappingProxyType({
            name: binding(self.root, path)
            for name, path in self.source_files.items()
        })
        self.context_config_path = write(
            self.root / "inputs/context.json", "{}\n",
        )
        self.context_config = binding(
            self.root, self.context_config_path,
        )
        self.fetch_path = write(self.root / "inputs/fetch.json", "{}\n")
        self.manifest_path = write(
            self.root / "inputs/manifest.json", "{}\n",
        )
        self.calendar_path = write(
            self.root / "inputs/calendar.json", "{}\n",
        )
        self.scaling_config_path = write(
            self.root / "inputs/scaling.json", "{}\n",
        )
        self.csv = tuple(
            write(self.root / f"inputs/{name}.csv", f"{name}\n")
            for name in MASTER
        )
        self.outputs = {
            "fits": self.source["fits"],
            "predictions": binding(
                self.root,
                write(self.root / "inputs/predictions.jsonl", "{}\n"),
            ),
            "summary": binding(
                self.root,
                write(self.root / "inputs/summary.json", "{}\n"),
            ),
        }
        self.scaling = SimpleNamespace(
            attempt_path=self.source["attempt"].path,
            run_id="source-run",
            fetch_report=binding(self.root, self.fetch_path),
            manifests=(
                SimpleNamespace(file=binding(
                    self.root, self.manifest_path,
                )),
            ),
            session_calendar=binding(self.root, self.calendar_path),
            config=binding(self.root, self.scaling_config_path),
            outputs={
                name: value.path for name, value in self.outputs.items()
            },
            coverage=object(),
        )
        self.csv_bindings = tuple(
            FileBinding(str(path), file_sha256(path)) for path in self.csv
        )
        self.phases = tuple(
            ContextPhase.parse(phase_value(name), MASTER)
            for name in ("fold-1", "calibration")
        )
        self.primary = write(self.root / "inputs/primary", "primary\n")
        self.torch = write(self.root / "inputs/torch", "torch\n")
        package = self.root / "package"
        write(package / "module.py", "torch = True\n")
        self.primary_binding = ExecutableBinding(
            str(self.primary), file_sha256(self.primary), "primary",
        )
        self.torch_identity = TorchIdentity(
            ExecutableBinding(
                str(self.torch), file_sha256(self.torch), "torch",
            ),
            "synthetic", None, None, "cpu",
            selected_source_tree(package, ("module.py",)),
        )
        self.scaling.primary_python = self.primary_binding
        self.scaling.torch_probe = self.torch_identity

    @property
    def output_path(self) -> Path:
        return self.root / self.output

    def arm(self) -> object:
        return armer.arm(
            self.output, "1" * 40, self.run_id,
            self.primary, self.torch,
        )

    def derive_phases(
        self, _attempt: object, _master: object, _csv: object, _data: object,
        config: object,
    ) -> tuple[ContextPhase, ...]:
        assert isinstance(config, FrozenInput)
        assert config.source == self.context_config_path
        return self.phases

    @contextmanager
    def patched(self) -> Iterator[None]:
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(
                armer,
                ROOT=self.root,
                CONTEXT_SOURCE_PATHS=self.source_paths,
                SOURCE_EVIDENCE=self.source,
                CONTEXT_CONFIG=self.context_config,
                _source_attempt=lambda _snapshot: self.scaling,
                _source_outputs=lambda _failure, _attempt: self.outputs,
                _fetch_bindings=lambda _path: (
                    MASTER, self.csv_bindings,
                ),
                _master_from_snapshot=lambda _path: MASTER,
                _validate_summary=lambda _snapshot, _outputs: None,
                _derive_phases=self.derive_phases,
                _require_isolated_execution=lambda: None,
                _validate_commit=lambda _commit, _tree: None,
                source_tree=lambda _path: self.torch_identity.package_tree,
            ))
            stack.enter_context(patch.object(
                ExecutableBinding, "validate_live", return_value=None,
            ))
            stack.enter_context(patch.multiple(
                contract,
                CONTEXT_SOURCE_PATHS=self.source_paths,
                SOURCE_EVIDENCE=self.source,
                CONTEXT_CONFIG=self.context_config,
            ))
            yield


def test_exact_one_shot_attempt() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-arm-", dir=ROOT,
    ) as directory:
        fixture = ArmFixture(Path(directory))
        with fixture.patched():
            attempt = fixture.arm()
            authenticate(attempt)
            forged = replace(
                attempt,
                phases=(
                    replace(
                        attempt.phases[0],
                        training_grid_sha256=digest("forged-grid"),
                    ),
                    attempt.phases[1],
                ),
            )
            raises(authenticate, forged)
            assert attempt.master == MASTER
            assert attempt.runner_argv == (
                contract.CONTEXT_RUNNER, attempt.attempt_path,
            )
            assert "command" not in read_canonical_json(fixture.output_path)
            assert fixture.output_path.is_file()
            assert not (fixture.root / "reports" / fixture.run_id).exists()
            raises(fixture.arm)


def test_existing_destination_blocks_all_work() -> None:
    for destination in ("attempt", "run"):
        with tempfile.TemporaryDirectory(
            prefix=f"context-arm-{destination}-", dir=ROOT,
        ) as directory:
            fixture = ArmFixture(Path(directory))
            target = (
                fixture.output_path if destination == "attempt" else
                fixture.root / "reports" / fixture.run_id
            )
            target.mkdir() if destination == "run" else write(target, "{}\n")
            with fixture.patched():
                raises(fixture.arm)
            assert target.exists()


def test_input_mutation_prevents_publication() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-arm-mutation-", dir=ROOT,
    ) as directory:
        fixture = ArmFixture(Path(directory))

        def mutate(*args: object) -> object:
            fixture.csv[0].write_text("changed\n", encoding="utf-8")
            return fixture.phases

        with fixture.patched(), patch.object(
            armer, "_derive_phases", new=mutate,
        ):
            raises(fixture.arm)
        assert not fixture.output_path.exists()


def test_source_tree_mutation_prevents_publication() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-arm-source-", dir=ROOT,
    ) as directory:
        fixture = ArmFixture(Path(directory))

        def changed(root: Path, paths: object) -> object:
            tree = selected_source_tree(root, paths)
            return SimpleNamespace(
                root=tree.root, files=tree.files,
                sha256=digest("changed-tree"),
            )

        with fixture.patched(), patch.object(
            armer, "selected_source_tree", new=changed,
        ):
            raises(fixture.arm)
        assert not fixture.output_path.exists()


def test_publication_fsync_boundary_is_one_shot() -> None:
    for failure, present in ((1, False), (2, False), (3, True)):
        with tempfile.TemporaryDirectory(
            prefix=f"context-arm-fsync-{failure}-", dir=ROOT,
        ) as directory:
            fixture = ArmFixture(Path(directory))
            original = armer.os.fsync
            calls = 0

            def fsync(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                if calls == failure:
                    raise OSError("synthetic fsync failure")
                original(descriptor)

            with fixture.patched(), patch.object(
                armer.os, "fsync", new=fsync,
            ):
                raises(fixture.arm)
            assert fixture.output_path.exists() is present


def test_run_claim_race_burns_instead_of_reporting_success() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-arm-claim-race-", dir=ROOT,
    ) as directory:
        fixture = ArmFixture(Path(directory))
        publish = armer.write_json_exclusive

        def race(*args: object, **kwargs: object) -> None:
            publish(*args, **kwargs)
            (fixture.root / "reports" / fixture.run_id).mkdir()

        with fixture.patched(), patch.object(
            armer, "write_json_exclusive", new=race,
        ):
            raises(fixture.arm)
        assert fixture.output_path.is_file()


def test_commit_must_bind_the_source_tree() -> None:
    content = b"bound source\n"
    files = (
        FileBinding("tools/source.py", hashlib.sha256(content).hexdigest()),
    )
    tree = SourceTree("/repository", files, _tree_digest(files))

    def run(
        command: tuple[str, ...], **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            stdout="commit\n" if "cat-file" in command else content,
        )

    with patch.object(armer.subprocess, "run", side_effect=run):
        armer._validate_commit("1" * 40, tree)

    def changed(
        command: tuple[str, ...], **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            stdout="commit\n" if "cat-file" in command else b"changed\n",
        )

    with patch.object(armer.subprocess, "run", side_effect=changed):
        raises(armer._validate_commit, "1" * 40, tree)


def test_path_identity_is_fixed() -> None:
    with tempfile.TemporaryDirectory(
        prefix="context-arm-path-", dir=ROOT,
    ) as directory:
        fixture = ArmFixture(Path(directory))
        with fixture.patched():
            for output, run_id in (
                (Path("/tmp/context.json"), fixture.run_id),
                (Path("../context.json"), fixture.run_id),
                (fixture.output, "other"),
            ):
                raises(
                    armer.arm, output, "1" * 40, run_id,
                    fixture.primary, fixture.torch,
                )
            assert not fixture.output_path.exists()


def main() -> None:
    test_private_snapshot_mutation_fails_verification()
    test_exact_one_shot_attempt()
    test_existing_destination_blocks_all_work()
    test_input_mutation_prevents_publication()
    test_source_tree_mutation_prevents_publication()
    test_publication_fsync_boundary_is_one_shot()
    test_run_claim_race_burns_instead_of_reporting_success()
    test_commit_must_bind_the_source_tree()
    test_path_identity_is_fixed()
    print("context diagnostic armer tests passed")


if __name__ == "__main__":
    main()
