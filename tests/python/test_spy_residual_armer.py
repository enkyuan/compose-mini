#!/usr/bin/env python3
"""Verify residual arming is source-bound, atomic, and one-shot."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import os
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import arm_spy_residual as armer
from tools.files import ExclusiveTemp, FrozenInput
from tools.panel_contract import (
    ExecutableBinding, FileBinding, SourceTree, TorchIdentity,
)
from tools.relative_context_contract import (
    RESIDUAL_BENCHMARK, RESIDUAL_CALENDAR, RESIDUAL_CONFIG,
    RESIDUAL_SOURCE, ResidualAttempt, ResidualPhaseInput,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def raises(
    function: object, *args: object, **kwargs: object,
) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except (OSError, ValueError):
        return
    raise AssertionError("expected residual arming failure")


class Lease:
    def __init__(self, calendar: FrozenInput) -> None:
        self.snapshots = SimpleNamespace(calendar=calendar)
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


class ArmFixture:
    run_id = "h13-spy-residual-test"
    output = Path(f"experiments/{run_id}-attempt.json")

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        for name in ("experiments", "reports", "inputs"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        executable = write(self.root / "inputs/python", "python\n")
        torch = write(self.root / "inputs/torch", "torch\n")
        self.primary = ExecutableBinding(
            str(executable), digest("python"), "synthetic",
        )
        package = SourceTree(
            str(self.root / "torch"),
            (FileBinding("torch.py", digest("torch.py")),),
            digest("torch-tree"),
        )
        self.torch = TorchIdentity(
            ExecutableBinding(
                str(torch), digest("torch"), "synthetic",
            ),
            "synthetic", None, None, "cpu", package,
        )
        self.tree = SourceTree(
            str(self.root),
            (FileBinding("tools/source.py", digest("source")),),
            digest("source-tree"),
        )
        self.phases = tuple(
            ResidualPhaseInput(
                phase, digest(f"{phase}-source"),
                digest(f"{phase}-training"),
                digest(f"{phase}-evaluation"),
                digest(f"{phase}-scalers"),
            )
            for phase in ("fold-1", "calibration")
        )
        self.calendar = FrozenInput(
            self.root / RESIDUAL_CALENDAR.path,
            self.root / "inputs/calendar.snapshot",
            RESIDUAL_CALENDAR.sha256,
        )
        self.context_lease = Lease(self.calendar)
        self.benchmark = tuple(
            (
                name,
                FrozenInput(
                    self.root / binding.path,
                    self.root / f"inputs/{name}.snapshot",
                    binding.sha256,
                ),
            )
            for name, binding in RESIDUAL_BENCHMARK.items()
        )
        self.verify_calls = 0
        self.context = object()
        self.bound = armer._BoundResidual(
            self.context, self.phases, self.tree, self.primary,
            (str(executable), "-I", "-S", "-B"), self.torch,
            self.context_lease, self.benchmark, self.verify,
        )
        self.attempt = ResidualAttempt(
            self.output.as_posix(), self.run_id,
            f"reports/{self.run_id}", "1" * 40,
            tuple(RESIDUAL_SOURCE.items()), RESIDUAL_CONFIG,
            tuple(RESIDUAL_BENCHMARK.items()), self.phases, self.tree,
            self.primary, self.bound.torch_argv, self.torch,
            MappingProxyType({
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX":
                    f"reports/{self.run_id}/.pycache",
            }),
        )

    @property
    def output_path(self) -> Path:
        return self.root / self.output

    def verify(self) -> None:
        self.context_lease()
        self.verify_calls += 1

    @contextmanager
    def closure(self) -> Iterator[armer._BoundResidual]:
        yield self.bound

    @contextmanager
    def patched(self) -> Iterator[None]:
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(
                armer,
                ROOT=self.root,
                _bound_residual=self.closure,
                _parse_constructed=lambda *_args: self.attempt,
                _require_isolated_execution=lambda: None,
                _validate_commit=lambda *_args: None,
            ))
            stack.enter_context(patch.object(
                armer.ResidualAttempt, "read",
                return_value=self.attempt,
            ))
            yield

    def arm(self) -> ResidualAttempt:
        return armer.arm(self.output, "1" * 40, self.run_id)


def test_import_boundary_is_training_free() -> None:
    forbidden = (
        "torch", "tools.backtest", "tools.evaluate", "tools.train",
        "tools.run_spy_residual",
    )
    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in sys.modules for prefix in forbidden
    )


def test_attempt_value_inherits_runtime_and_localizes_cache() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-attempt-value-", dir=ROOT,
    ) as directory:
        fixture = ArmFixture(Path(directory))
        value = armer._attempt_value(
            fixture.output, "1" * 40, fixture.run_id, fixture.bound,
        )
        assert value["attempt_path"] == fixture.output.as_posix()
        assert value["run_dir"] == f"reports/{fixture.run_id}"
        assert value["primary_python"] == asdict(fixture.primary)
        assert value["torch_argv"] == list(fixture.bound.torch_argv)
        assert value["torch_probe"] == asdict(fixture.torch)
        assert value["environment"] == {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX":
                f"reports/{fixture.run_id}/.pycache",
        }


def test_exact_attempt_and_ordered_lease() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-arm-", dir=ROOT,
    ) as directory:
        fixture = ArmFixture(Path(directory))
        with fixture.patched():
            attempt = fixture.arm()
            with armer.authenticate_residual_attempt(attempt) as lease:
                assert type(lease) is armer.ResidualLease
                assert lease.context is fixture.context_lease
                assert lease.benchmark == fixture.benchmark
                assert tuple(
                    name for name, _ in lease.benchmark
                ) == tuple(RESIDUAL_BENCHMARK)
                calls = fixture.verify_calls
                context_calls = fixture.context_lease.calls
                lease()
                assert fixture.verify_calls == calls + 1
                assert fixture.context_lease.calls == context_calls + 1
            raises(
                lambda: armer.authenticate_residual_attempt(
                    replace(attempt, implementation_commit="2" * 40),
                ).__enter__(),
            )
        value = json.loads(fixture.output_path.read_text(encoding="utf-8"))
        assert value["attempt_path"] == fixture.output.as_posix()
        assert value["run_id"] == fixture.run_id
        assert value["run_dir"] == f"reports/{fixture.run_id}"
        assert not (fixture.root / "reports" / fixture.run_id).exists()


def test_existing_destination_blocks_before_discovery() -> None:
    for destination in ("attempt", "run", "outcome"):
        with tempfile.TemporaryDirectory(
            prefix=f"residual-arm-{destination}-", dir=ROOT,
        ) as directory:
            fixture = ArmFixture(Path(directory))
            target = {
                "attempt": fixture.output_path,
                "run": fixture.root / "reports" / fixture.run_id,
                "outcome": fixture.root / "experiments" /
                    f"{fixture.run_id}-outcome.json",
            }[destination]
            target.mkdir() if destination == "run" else write(target, "{}\n")
            with fixture.patched(), patch.object(
                armer, "_bound_residual",
                side_effect=AssertionError("closure discovery ran"),
            ):
                raises(fixture.arm)
            assert target.exists()


def test_attempt_identity_is_exact() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-arm-identity-", dir=ROOT,
    ) as directory:
        fixture = ArmFixture(Path(directory))
        with fixture.patched():
            for output, commit, run_id in (
                (fixture.output_path, "1" * 40, fixture.run_id),
                (Path("../attempt.json"), "1" * 40, fixture.run_id),
                (fixture.output, "g" * 40, fixture.run_id),
                (fixture.output, "1" * 40, "h13-spy-residual-other"),
            ):
                raises(armer.arm, output, commit, run_id)
            assert fixture.arm() == fixture.attempt


def test_publication_callbacks_precede_public_inode_verification() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-arm-order-", dir=ROOT,
    ) as directory:
        fixture = ArmFixture(Path(directory))
        publish = armer.write_json_exclusive
        inspect = armer._published_bytes
        events: list[str] = []

        def ordered(
            path: Path, value: object, descriptor: int,
            *, before_link_with_temp: object,
            on_temp_created: object,
        ) -> None:
            def capture(binding: ExclusiveTemp) -> None:
                events.append("temporary")
                on_temp_created(binding)  # type: ignore[operator]

            def verify(binding: ExclusiveTemp) -> None:
                events.append("pre-link")
                before_link_with_temp(binding)  # type: ignore[operator]

            publish(
                path, value, descriptor,
                before_link_with_temp=verify,
                on_temp_created=capture,
            )
            events.append("linked")

        def inspected(*args: object) -> None:
            events.append("public-inode")
            inspect(*args)  # type: ignore[arg-type]

        with fixture.patched(), patch.object(
            armer, "write_json_exclusive", new=ordered,
        ), patch.object(armer, "_published_bytes", new=inspected):
            fixture.arm()
        assert events == [
            "temporary", "pre-link", "linked", "public-inode",
        ]


def test_same_bytes_on_a_new_public_inode_burns_attempt() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-arm-public-inode-", dir=ROOT,
    ) as directory:
        fixture = ArmFixture(Path(directory))
        publish = armer.write_json_exclusive

        def substitute(
            path: Path, value: object, descriptor: int, **kwargs: object,
        ) -> None:
            publish(path, value, descriptor, **kwargs)
            payload = path.read_bytes()
            os.rename(
                path.name, ".held-public-inode",
                src_dir_fd=descriptor, dst_dir_fd=descriptor,
            )
            replacement = os.open(
                path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600, dir_fd=descriptor,
            )
            try:
                os.write(replacement, payload)
                os.fsync(replacement)
            finally:
                os.close(replacement)

        with fixture.patched(), patch.object(
            armer, "write_json_exclusive", new=substitute,
        ):
            raises(fixture.arm)
        assert fixture.output_path.is_file()


def test_cleanup_removes_only_the_owned_temporary() -> None:
    for foreign in (False, True):
        with tempfile.TemporaryDirectory(
            prefix=f"residual-arm-cleanup-{foreign}-", dir=ROOT,
        ) as directory:
            fixture = ArmFixture(Path(directory))
            name = f".{fixture.output.name}.controlled.tmp"

            def fail(
                _path: Path, _value: object, descriptor: int,
                *, on_temp_created: object, **_kwargs: object,
            ) -> None:
                file = os.open(
                    name, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600, dir_fd=descriptor,
                )
                metadata = os.fstat(file)
                os.close(file)
                on_temp_created(ExclusiveTemp(  # type: ignore[operator]
                    name, (metadata.st_dev, metadata.st_ino),
                    stat.S_IMODE(metadata.st_mode),
                ))
                if foreign:
                    os.rename(
                        name, f"{name}.held",
                        src_dir_fd=descriptor, dst_dir_fd=descriptor,
                    )
                    replacement = os.open(
                        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600, dir_fd=descriptor,
                    )
                    os.write(replacement, b"foreign")
                    os.close(replacement)
                raise OSError("synthetic publication failure")

            with fixture.patched(), patch.object(
                armer, "write_json_exclusive", new=fail,
            ):
                raises(fixture.arm)
            temporary = fixture.output_path.parent / name
            assert temporary.exists() is foreign
            if foreign:
                assert temporary.read_bytes() == b"foreign"


def test_bundle_topology_and_inode_revalidation() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-bundle-", dir=ROOT,
    ) as directory:
        base = Path(directory).resolve()

        def bundle(name: str) -> tuple[Path, Path, Path]:
            path = base / name
            path.mkdir()
            return (
                path,
                write(path / "fetch.json", "{}\n"),
                write(path / "spy.csv", "same bytes\n"),
            )

        exact, fetch, spy = bundle("exact")
        assert armer._directory_members(
            exact, ("fetch.json", "spy.csv"),
        ) == armer._directory_members(exact, ("fetch.json", "spy.csv"))

        extra, _, _ = bundle("extra")
        write(extra / "unexpected", "x\n")
        raises(
            armer._directory_members, extra, ("fetch.json", "spy.csv"),
        )

        linked, _, linked_spy = bundle("hardlink")
        os.link(linked_spy, base / "external-hardlink")
        raises(
            armer._directory_members, linked, ("fetch.json", "spy.csv"),
        )

        symbolic, _, symbolic_spy = bundle("symlink")
        held = base / "held-spy"
        symbolic_spy.rename(held)
        symbolic_spy.symlink_to(held)
        raises(
            armer._directory_members, symbolic, ("fetch.json", "spy.csv"),
        )
        alias = base / "bundle-alias"
        alias.symlink_to(exact, target_is_directory=True)
        raises(
            armer._directory_members, alias, ("fetch.json", "spy.csv"),
        )

        identities = armer._single_link_inputs(
            (fetch, spy), "SPY bundle",
        )
        old_spy = base / "old-spy"
        replacement = write(base / "replacement-spy", "same bytes\n")
        spy.rename(old_spy)
        replacement.rename(spy)
        armer._directory_members(exact, ("fetch.json", "spy.csv"))
        raises(armer._verify_identities, identities)


class Poison:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError(
            f"decoded source evidence was accessed through {name}",
        )


def test_source_predictions_are_discarded_before_derivation() -> None:
    with tempfile.TemporaryDirectory(
        prefix="residual-source-discard-", dir=ROOT,
    ) as directory:
        root = Path(directory).resolve()
        context = SimpleNamespace(
            primary_python=object(),
            torch_argv=("python", "-I", "-S", "-B"),
            torch_probe=object(),
        )
        calendar = FrozenInput(
            root / RESIDUAL_CALENDAR.path, root / "calendar.snapshot",
            RESIDUAL_CALENDAR.sha256,
        )
        lease = Lease(calendar)
        phases = ("fold-1", "calibration")
        tree = SourceTree(str(root), (), digest("tree"))
        source_run = root / "source-run"
        source_paths = tuple(
            root / binding.path for binding in RESIDUAL_SOURCE.values()
        )
        source_names = tuple(path.name for path in source_paths)
        derived = 0

        @contextmanager
        def frozen(
            paths: object,
        ) -> Iterator[tuple[FrozenInput, ...]]:
            yield tuple(
                FrozenInput(
                    path, root / f"snapshot-{index}",
                    digest(str(path)),
                )
                for index, path in enumerate(paths)  # type: ignore[arg-type]
            )

        @contextmanager
        def authenticated(_context: object) -> Iterator[Lease]:
            yield lease

        def derive(
            observed_context: object, observed_lease: object,
            spy: FrozenInput,
        ) -> tuple[str, ...]:
            nonlocal derived
            derived += 1
            assert observed_context is context
            assert observed_lease is lease
            assert spy.source == root / RESIDUAL_BENCHMARK["spy_csv"].path
            return phases

        with ExitStack() as stack:
            stack.enter_context(patch.multiple(
                armer,
                ROOT=root,
                RESIDUAL_SOURCE_PATHS=(),
                _binding_matches=lambda *_args: True,
                _completed_run=lambda *_args: (context, Poison(), {}),
                _directory_members=lambda *_args: (1, 2),
                _parse_source_context=lambda _snapshot: context,
                _single_link_inputs=lambda *_args: (),
                _snapshot_tree=lambda _frozen: tree,
                _source_paths=lambda _context: (
                    source_run, source_paths, source_names,
                ),
                _verify_identities=lambda _identities: None,
                authenticate_context_attempt=authenticated,
                derive_residual_phases=derive,
                freeze_inputs=frozen,
                read_canonical_json=lambda _path: {},
                selected_source_tree=lambda *_args: tree,
                validate_residual_protocol=lambda _value: {},
                validate_source_context_outcome=lambda _value: {},
                validate_spy_fetch_report=lambda *_args: {},
                verify_frozen=lambda _snapshots: None,
            ))
            with armer._bound_residual() as bound:
                assert bound.context_attempt is context
                assert bound.phases == phases
                assert tuple(
                    name for name, _ in bound.benchmark
                ) == tuple(RESIDUAL_BENCHMARK)
                bound.verify()
        assert derived == 1


def main() -> None:
    test_import_boundary_is_training_free()
    test_attempt_value_inherits_runtime_and_localizes_cache()
    test_exact_attempt_and_ordered_lease()
    test_existing_destination_blocks_before_discovery()
    test_attempt_identity_is_exact()
    test_publication_callbacks_precede_public_inode_verification()
    test_same_bytes_on_a_new_public_inode_burns_attempt()
    test_cleanup_removes_only_the_owned_temporary()
    test_bundle_topology_and_inode_revalidation()
    test_source_predictions_are_discarded_before_derivation()
    print("SPY residual armer tests passed")


if __name__ == "__main__":
    main()
