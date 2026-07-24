#!/usr/bin/env python3
"""Verify universe-scaling arming rejects filesystem races and aliases."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import arm_universe_scaling as armer
from tools.files import file_sha256
from tools.panel_contract import (
    FileBinding, TorchIdentity, executable_binding, selected_source_tree,
)
from tools.universe_scaling_contract import PHASES, TreeBinding
from tools.universe_scaling_inputs import (
    PhaseCoverage, ScalingCoverage, ScalingSeries,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


class ArmFixture:
    """Provide the smallest real filesystem needed by ``arm()``."""

    run_id = "scaling-arm-test"
    output = Path(f"experiments/{run_id}-attempt.json")

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.fetch_path = Path("inputs/fetch.json")
        self.calendar_path = Path("inputs/calendar.json")
        self.config_path = Path("inputs/config.json")
        self.selection_root = Path("selection")
        self.source_paths = ("src/runner.py", "src/shared.py")
        self.finalizer_paths = ("src/finalizer.py", "src/shared.py")
        for parent in ("experiments", "reports", "inputs", "selection"):
            (self.root / parent).mkdir(parents=True, exist_ok=True)

        self.names = tuple(
            "ENLC" if index == 49 else f"S{index:02d}"
            for index in range(55)
        )
        self.csvs = tuple(
            write(
                self.root / "data" / f"{name.lower()}-30m.csv",
                f"{name}\n",
            )
            for name in self.names
        )
        self.series = tuple(
            ScalingSeries(
                name, FileBinding(str(path), file_sha256(path)), 1,
            )
            for name, path in zip(self.names, self.csvs, strict=True)
        )
        write_json(self.root / self.fetch_path, {
            "manifest": {"path": "selection/manifests/liquid-common-55.json"},
            "session_calendar": {"path": self.calendar_path.as_posix()},
        })
        write_json(self.root / self.calendar_path, {})
        write_json(self.root / self.config_path, {})

        manifest_dir = self.root / self.selection_root / "manifests"
        manifest_dir.mkdir()
        self.manifest_paths = {
            size: write(
                manifest_dir / f"liquid-common-{size}.json",
                f"{size}\n",
            )
            for size in (11, 22, 33, 55)
        }
        self.manifest_sha256 = {
            size: file_sha256(path)
            for size, path in self.manifest_paths.items()
        }
        self.manifests = tuple(
            SimpleNamespace(series=tuple(
                SimpleNamespace(ticker=name)
                for name in self.names[:size]
            ))
            for size in self.manifest_paths
        )
        self.selection = TreeBinding(
            self.selection_root.as_posix(), len(self.manifest_paths),
            hashlib.sha256(b"selection").hexdigest(),
        )
        self.selection_members = tuple(self.manifest_paths.values())

        for path in {*self.source_paths, *self.finalizer_paths}:
            write(self.root / path, f"{path}\n")
        self.primary = write(self.root / "runtime/primary", "primary\n")
        self.torch = write(self.root / "runtime/torch", "torch\n")
        package = self.root / "torch-package"
        write(package / "module.py", "synthetic = True\n")
        self.torch_probe = TorchIdentity(
            executable_binding(self.torch, "synthetic torch"),
            "synthetic", None, None, "cpu",
            selected_source_tree(package, ("module.py",)),
        )

    @property
    def output_path(self) -> Path:
        return self.root / self.output

    def assert_unpublished(self) -> None:
        assert not any(path.exists() for path in (
            self.output_path,
            self.root / "reports" / self.run_id,
            self.root / "experiments" / f"{self.run_id}-outcome.json",
        ))

    def arm(self) -> object:
        return armer.arm(
            self.output, "1" * 40, self.run_id,
            self.primary, self.torch,
        )

    @contextmanager
    def patched(
        self, *, validate_data: bool = False,
    ) -> Iterator[SimpleNamespace]:
        sentinel = object()
        reads: list[tuple[Path, Path, Path]] = []

        def read_attempt(
            path: Path, logical_path: Path, repository_root: Path,
        ) -> object:
            assert path.is_file()
            assert logical_path == self.output
            assert repository_root == self.root
            reads.append((path, logical_path, repository_root))
            return sentinel

        manifest_values = iter(self.manifests)
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(
                armer,
                ROOT=self.root,
                FETCH_PATH=self.fetch_path,
                FETCH_SHA256=file_sha256(self.root / self.fetch_path),
                CALENDAR_PATH=self.calendar_path,
                CALENDAR_SHA256=file_sha256(self.root / self.calendar_path),
                CONFIG_PATH=self.config_path,
                CONFIG_SHA256=file_sha256(self.root / self.config_path),
                SELECTION_ROOT=self.selection_root,
                MANIFEST_SHA256=self.manifest_sha256,
                SOURCE_PATHS=self.source_paths,
                FINALIZER_SOURCE_PATHS=self.finalizer_paths,
                ScalingAttempt=SimpleNamespace(read=read_attempt),
            ))
            selection_binding = stack.enter_context(patch.object(
                armer, "selection_binding", return_value=self.selection,
            ))
            stack.enter_context(patch.object(
                armer, "selection_paths",
                return_value=self.selection_members,
            ))
            stack.enter_context(patch.object(
                armer, "fetch_series", return_value=self.series,
            ))
            stack.enter_context(patch.object(
                armer, "_version", return_value="synthetic primary",
            ))
            observe_torch = stack.enter_context(patch.object(
                armer, "observe_torch", return_value=self.torch_probe,
            ))
            read_json = stack.enter_context(patch.object(
                armer, "read_json", wraps=armer.read_json,
            ))
            stack.enter_context(patch.object(
                armer.UniverseManifest, "read",
                side_effect=lambda _path: next(manifest_values),
            ))
            stack.enter_context(patch.object(
                armer.SessionCalendar, "read", return_value=object(),
            ))
            stack.enter_context(patch.object(
                armer, "validate_config", return_value={},
            ))
            stack.enter_context(patch.object(
                armer, "load_frozen_bars", return_value=object(),
            ))
            stack.enter_context(patch.object(
                armer, "validate_fetch", return_value=None,
            ))
            coverage = stack.enter_context(patch.object(
                armer, "common_coverage",
                return_value=ScalingCoverage(tuple(
                    PhaseCoverage(phase, (), (), ())
                    for phase in PHASES
                ), ("ENLC",)),
            ))
            if not validate_data:
                stack.enter_context(patch.object(
                    armer, "_validate_data", return_value=None,
                ))
            yield SimpleNamespace(
                coverage=coverage, observe_torch=observe_torch,
                read_attempts=reads, read_json=read_json,
                selection_binding=selection_binding,
            )


def rejects(call: Callable[[], object], fixture: ArmFixture) -> Exception:
    try:
        call()
    except (OSError, TypeError, ValueError) as error:
        fixture.assert_unpublished()
        return error
    raise AssertionError("invalid universe-scaling attempt was armed")


def intercepted(
    action: Callable[[Callable[[], None]], None],
) -> Callable[..., None]:
    original = armer.write_json_exclusive

    def publish(
        path: Path, value: Mapping[str, object],
        directory_fd: int | None = None,
        before_link: Callable[[], None] | None = None,
    ) -> None:
        assert directory_fd is not None and callable(before_link)
        original(
            path, value, directory_fd,
            lambda: action(before_link),
        )

    return publish


def verify_existing_output_precedes_discovery(root: Path) -> None:
    fixture = ArmFixture(root)
    fixture.output_path.write_text("occupied\n", encoding="utf-8")
    with fixture.patched() as probes:
        try:
            fixture.arm()
        except ValueError as error:
            assert str(error) == "attempt output must be absent"
        else:
            raise AssertionError("existing attempt output was overwritten")
        probes.selection_binding.assert_not_called()
        probes.read_json.assert_not_called()
    assert fixture.output_path.read_text(encoding="utf-8") == "occupied\n"


def verify_publication_mutations() -> None:
    for label, target in (
        ("direct", lambda fixture: fixture.csvs[0]),
        ("source", lambda fixture: fixture.root / fixture.source_paths[0]),
        ("runtime", lambda fixture: fixture.primary),
    ):
        with tempfile.TemporaryDirectory(
            prefix=f"compose-mini-scaling-arm-{label}-",
        ) as directory:
            fixture = ArmFixture(Path(directory))

            def mutate(
                verify: Callable[[], None],
                path: Path = target(fixture),
            ) -> None:
                path.write_text("changed\n", encoding="utf-8")
                verify()

            with fixture.patched(), patch.object(
                armer, "write_json_exclusive", intercepted(mutate),
            ):
                error = rejects(fixture.arm, fixture)
            assert str(error) == "an input changed during the command"


def verify_output_parent_replacement(root: Path) -> None:
    fixture = ArmFixture(root)
    parent = fixture.output_path.parent
    moved = parent.with_name("moved-experiments")

    def replace_parent(verify: Callable[[], None]) -> None:
        parent.rename(moved)
        parent.mkdir()
        try:
            verify()
        finally:
            parent.rmdir()
            moved.rename(parent)

    with fixture.patched(), patch.object(
        armer, "write_json_exclusive", intercepted(replace_parent),
    ):
        error = rejects(fixture.arm, fixture)
    assert str(error) == "attempt output parent changed"


def verify_hardlink_alias(root: Path) -> None:
    fixture = ArmFixture(root)
    config = fixture.root / fixture.config_path
    config.unlink()
    os.link(fixture.root / fixture.calendar_path, config)
    with fixture.patched():
        error = rejects(fixture.arm, fixture)
    assert str(error) == "panel inputs must not alias each other"


def verify_nonpromotable_coverage(root: Path) -> None:
    fixture = ArmFixture(root)
    with fixture.patched(validate_data=True) as probes, patch.object(
        armer, "write_json_exclusive", wraps=armer.write_json_exclusive,
    ) as publish:
        error = rejects(fixture.arm, fixture)
        expected = "unseen calibration coverage is incomplete: ENLC"
        assert str(error) == expected, str(error)
        probes.coverage.assert_called_once()
        probes.observe_torch.assert_not_called()
        publish.assert_not_called()
        assert not probes.read_attempts


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-arm-existing-",
    ) as directory:
        verify_existing_output_precedes_discovery(Path(directory))
    verify_publication_mutations()
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-arm-parent-",
    ) as directory:
        verify_output_parent_replacement(Path(directory))
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-arm-hardlink-",
    ) as directory:
        verify_hardlink_alias(Path(directory))
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-arm-coverage-",
    ) as directory:
        verify_nonpromotable_coverage(Path(directory))
    print("universe scaling arming tests passed")


if __name__ == "__main__":
    main()
