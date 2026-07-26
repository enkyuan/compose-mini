#!/usr/bin/env python3
"""Arm one immutable development-only stock-minus-SPY calibration."""

from __future__ import annotations

import os
import sys

_BOOTSTRAP_FLAGS = ("-I", "-S", "-B")


def _require_isolated_execution() -> None:
    flags = sys.flags
    if not flags.isolated or not getattr(flags, "safe_path", False) or \
       not flags.no_user_site or not flags.no_site or \
       not flags.dont_write_bytecode or \
       not flags.ignore_environment or not sys.dont_write_bytecode:
        raise ValueError(
            "residual armer requires isolated bytecode-free Python",
        )


def _require_exact_launch(*, pristine: bool = False) -> None:
    if pristine and ("ctypes" in sys.modules or "_ctypes" in sys.modules):
        raise ValueError("residual armer launch inspection is already loaded")

    from ctypes import POINTER, byref, c_int, c_wchar_p, pythonapi

    argc = c_int()
    argv = POINTER(c_wchar_p)()
    get_argv = pythonapi.Py_GetArgcArgv
    get_argv.argtypes = (POINTER(c_int), POINTER(POINTER(c_wchar_p)))
    get_argv.restype = None
    get_argv(byref(argc), byref(argv))
    observed = tuple(argv[index] for index in range(argc.value))
    canonical = lambda values: (
        os.path.realpath(values[0]), *values[1:]
    )
    expected = (
        os.path.realpath(sys.executable), *_BOOTSTRAP_FLAGS, *sys.argv,
    )
    if not observed or canonical(observed) != expected or \
       canonical(tuple(sys.orig_argv)) != expected or \
       os.path.realpath(sys.argv[0]) != os.path.realpath(__file__):
        raise ValueError("residual armer requires its exact Python launch")


def _bootstrap_main() -> None:
    """Expose only this checkout's real tools package after launch checks."""
    from importlib.machinery import PathFinder
    import stat

    tools = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(tools)
    initializer = os.path.join(tools, "__init__.py")
    if not stat.S_ISDIR(os.lstat(tools).st_mode) or \
       not stat.S_ISREG(os.lstat(initializer).st_mode):
        raise ValueError("tools namespace is not a real package")
    for entry in os.scandir(tools):
        mode = entry.stat(follow_symlinks=False).st_mode
        valid = (
            stat.S_ISDIR(mode) if entry.name == "__pycache__" else
            entry.name.endswith(".py") and stat.S_ISREG(mode)
        )
        if not valid:
            raise ValueError("tools namespace contains an unsafe entry")
    if any(
        name == "tools" or name.startswith("tools.") for name in sys.modules
    ):
        raise ValueError("tools namespace is already loaded")
    spec = PathFinder.find_spec("tools", (*sys.path, root))
    locations = tuple(
        os.path.realpath(path)
        for path in (spec.submodule_search_locations or ())
    ) if spec is not None else ()
    if spec is None or os.path.realpath(spec.origin or "") != \
       os.path.realpath(initializer) or \
       locations != (os.path.realpath(tools),):
        raise ValueError("tools namespace resolver is unsafe")
    sys.path.append(root)
    import tools as package
    if os.path.realpath(package.__file__ or "") != \
       os.path.realpath(initializer) or tuple(
           map(os.path.realpath, package.__path__)
       ) != locations:
        raise ValueError("tools namespace import is unsafe")


if __name__ == "__main__":
    _require_isolated_execution()
    _require_exact_launch(pristine=True)
    _bootstrap_main()

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
import argparse
import hashlib
import json
import re
import stat
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in tuple(map(os.path.realpath, sys.path)):
    sys.path.insert(0, str(ROOT))

from tools.analyze_context_cross_section import _completed_run
from tools.arm_context_diagnostic import (
    ContextLease, _validate_commit, authenticate_context_attempt,
)
from tools.context_diagnostic_contract import (
    ContextAttempt, PYTHON_FLAGS,
)
from tools.files import (
    ExclusiveTemp, FrozenInput, _owns_entry, freeze_inputs, verify_frozen,
    write_json, write_json_exclusive,
)
from tools.panel_contract import (
    ExecutableBinding, FileBinding, SourceTree, TorchIdentity, _absent,
    _directory_identity, _open_directory, _regular_inputs,
    _tree_digest, _verify_identities, read_canonical_json,
    selected_source_tree,
)
from tools.relative_context_contract import (
    RESIDUAL_BENCHMARK, RESIDUAL_CALENDAR, RESIDUAL_CONFIG, RESIDUAL_SOURCE,
    RESIDUAL_SOURCE_PATHS, ResidualAttempt, ResidualPhaseInput,
    validate_residual_protocol,
    validate_source_context_outcome, validate_spy_fetch_report,
)
from tools.run_context_diagnostic import phase_artifacts
from tools.spy_residual_controller import derive_residual_phases

if PYTHON_FLAGS != _BOOTSTRAP_FLAGS:
    raise ValueError("residual Python isolation flags changed")

COMMIT = re.compile(r"[0-9a-f]{40}")
RUN_ID = re.compile(r"h13-spy-residual-[a-z0-9][a-z0-9-]*")
Verify = Callable[[], None]


@dataclass(frozen=True, slots=True)
class ResidualLease:
    """Keep the authenticated source and benchmark closures live."""

    context: ContextLease
    benchmark: tuple[tuple[str, FrozenInput], ...]
    _verifier: Verify = field(repr=False, compare=False)

    def __call__(self) -> None:
        self._verifier()


@dataclass(frozen=True, slots=True)
class _BoundResidual:
    context_attempt: ContextAttempt
    phases: tuple[ResidualPhaseInput, ...]
    tree: SourceTree
    primary: ExecutableBinding
    torch_argv: tuple[str, ...]
    torch: TorchIdentity
    context_lease: ContextLease
    benchmark: tuple[tuple[str, FrozenInput], ...]
    verify: Verify = field(repr=False, compare=False)


def _relative(path: Path, label: str) -> Path:
    if path.is_absolute() or not path.parts or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise ValueError(f"{label} must be a normalized relative path")
    return path


def _fresh(paths: Sequence[tuple[Path, str]]) -> None:
    resolved = tuple(
        (path.resolve(strict=False), label) for path, label in paths
    )
    for index, (path, _) in enumerate(resolved):
        if any(path == other for other, _ in resolved[:index]):
            raise ValueError("residual output topology is invalid")
    for path, label in paths:
        _absent(path, label)


def _frozen(
    values: Mapping[Path, FrozenInput], path: Path,
) -> FrozenInput:
    try:
        return values[path]
    except KeyError as error:
        raise ValueError(f"unfrozen residual input: {path}") from error


def _single_link_inputs(
    paths: Sequence[Path], label: str,
) -> tuple[tuple[Path, tuple[int, int]], ...]:
    identities = _regular_inputs(paths)
    if any(
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        for path in paths
        for metadata in (path.stat(follow_symlinks=False),)
    ):
        raise ValueError(f"{label} must be single-link regular files")
    return identities


def _directory_members(
    path: Path, names: Sequence[str],
) -> tuple[int, int]:
    try:
        if path != Path(os.path.abspath(path)) or \
           path != path.resolve(strict=True):
            raise ValueError("residual directory must not use an alias")
        descriptor, identity = _open_directory(path)
        try:
            entries = tuple(sorted(os.listdir(descriptor)))
            if entries != tuple(sorted(names)):
                raise ValueError("residual directory entries changed")
            for name in entries:
                metadata = os.stat(
                    name, dir_fd=descriptor, follow_symlinks=False,
                )
                if not stat.S_ISREG(metadata.st_mode) or \
                   metadata.st_nlink != 1:
                    raise ValueError(
                        "residual directory entry is unsafe",
                    )
        finally:
            os.close(descriptor)
        return identity
    except OSError as error:
        raise ValueError("residual directory is unavailable") from error


def _snapshot_tree(
    frozen: Mapping[Path, FrozenInput],
) -> SourceTree:
    files = tuple(
        FileBinding(path, _frozen(frozen, ROOT / path).sha256)
        for path in RESIDUAL_SOURCE_PATHS
    )
    return SourceTree(str(ROOT.resolve()), files, _tree_digest(files))


def _source_paths(
    context: ContextAttempt,
) -> tuple[Path, tuple[Path, ...], tuple[str, ...]]:
    attempt = ROOT / RESIDUAL_SOURCE["context_attempt"].path
    outcome = ROOT / RESIDUAL_SOURCE["context_outcome"].path
    artifacts = tuple(
        path
        for phase in context.phases
        for path in phase_artifacts(ROOT, attempt, phase)
    )
    run = outcome.parent
    names = (
        *(path.name for path in artifacts),
        outcome.name,
        "cross-section.json",
    )
    return run, (attempt, outcome, *artifacts, run / "cross-section.json"), \
        tuple(names)


def _parse_source_context(snapshot: FrozenInput) -> ContextAttempt:
    binding = RESIDUAL_SOURCE["context_attempt"]
    if snapshot.sha256 != binding.sha256:
        raise ValueError("source context attempt changed")
    return ContextAttempt.read(
        snapshot.snapshot, Path(binding.path), ROOT,
    )


def _parse_constructed(
    value: Mapping[str, object], logical_path: Path,
    context: ContextAttempt,
) -> ResidualAttempt:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-residual-attempt-",
    ) as directory:
        snapshot = Path(directory) / "attempt.json"
        write_json(snapshot, value)
        return ResidualAttempt.read(
            snapshot, logical_path, ROOT, context,
        )


def _attempt_value(
    output: Path, implementation_commit: str, run_id: str,
    bound: _BoundResidual,
) -> dict[str, object]:
    run_dir = Path("reports") / run_id
    return {
        "attempt_path": output.as_posix(),
        "benchmark": {
            name: asdict(binding)
            for name, binding in RESIDUAL_BENCHMARK.items()
        },
        "config": asdict(RESIDUAL_CONFIG),
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": f"{run_dir.as_posix()}/.pycache",
        },
        "implementation_commit": implementation_commit,
        "phases": [asdict(phase) for phase in bound.phases],
        "primary_python": asdict(bound.primary),
        "run_dir": run_dir.as_posix(),
        "run_id": run_id,
        "schema": 1,
        "source": {
            name: asdict(binding)
            for name, binding in RESIDUAL_SOURCE.items()
        },
        "source_tree": asdict(bound.tree),
        "status": "armed",
        "torch_argv": list(bound.torch_argv),
        "torch_probe": asdict(bound.torch),
    }


def _binding_matches(
    frozen: Mapping[Path, FrozenInput], binding: FileBinding,
) -> bool:
    return _frozen(frozen, ROOT / binding.path).sha256 == binding.sha256


@contextmanager
def _bound_residual() -> Iterator[_BoundResidual]:
    """Freeze and continuously verify the executable residual closure."""
    source_attempt_path = ROOT / RESIDUAL_SOURCE["context_attempt"].path
    attempt_identities = _single_link_inputs(
        (source_attempt_path,), "source context attempt",
    )
    with freeze_inputs((source_attempt_path,)) as initial:
        context = _parse_source_context(initial[0])
        _verify_identities(attempt_identities)
        verify_frozen(initial)

    source_run, source_paths, source_names = _source_paths(context)
    source_run_identity = _directory_members(source_run, source_names)
    config_path = ROOT / RESIDUAL_CONFIG.path
    benchmark_paths = tuple(
        ROOT / binding.path for binding in RESIDUAL_BENCHMARK.values()
    )
    source_code = tuple(ROOT / path for path in RESIDUAL_SOURCE_PATHS)
    paths = tuple(dict.fromkeys((
        *source_paths, config_path, *benchmark_paths, *source_code,
    )))
    identities = _single_link_inputs(paths, "residual closure")
    bundle = benchmark_paths[0].parent
    bundle_identity = _directory_members(
        bundle, tuple(path.name for path in benchmark_paths),
    )

    with freeze_inputs(paths) as snapshots:
        frozen = dict(zip(paths, snapshots, strict=True))
        context = _parse_source_context(
            _frozen(frozen, source_attempt_path),
        )
        if not _binding_matches(
            frozen, RESIDUAL_SOURCE["context_outcome"],
        ) or not _binding_matches(frozen, RESIDUAL_CONFIG) or any(
            not _binding_matches(frozen, binding)
            for binding in RESIDUAL_BENCHMARK.values()
        ):
            raise ValueError("residual fixed input changed")
        outcome = _frozen(
            frozen, ROOT / RESIDUAL_SOURCE["context_outcome"].path,
        )
        validate_source_context_outcome(
            read_canonical_json(outcome.snapshot),
        )
        validate_residual_protocol(read_canonical_json(
            _frozen(frozen, config_path).snapshot,
        ))
        validate_spy_fetch_report(
            read_canonical_json(_frozen(
                frozen, benchmark_paths[0],
            ).snapshot),
            ROOT,
        )

        authenticated, decoded, _ = _completed_run(
            source_attempt_path, Path(
                RESIDUAL_SOURCE["context_attempt"].path,
            ),
            frozen, source_run_identity,
        )
        del decoded
        if authenticated != context:
            raise ValueError("source context authentication changed")

        tree = _snapshot_tree(frozen)
        benchmark = tuple(
            (name, _frozen(frozen, ROOT / binding.path))
            for name, binding in RESIDUAL_BENCHMARK.items()
        )
        with authenticate_context_attempt(context) as context_lease:
            calendar = context_lease.snapshots.calendar
            if calendar.source != ROOT / RESIDUAL_CALENDAR.path or \
               calendar.sha256 != RESIDUAL_CALENDAR.sha256:
                raise ValueError("residual calendar differs from context")
            phases = derive_residual_phases(
                context, context_lease, dict(benchmark)["spy_csv"],
            )

            def verify() -> None:
                context_lease()
                if _directory_members(
                    source_run, source_names,
                ) != source_run_identity or _directory_members(
                    bundle, tuple(path.name for path in benchmark_paths),
                ) != bundle_identity:
                    raise ValueError("residual input directory changed")
                _verify_identities(identities)
                _single_link_inputs(paths, "residual closure")
                verify_frozen(snapshots)
                if selected_source_tree(
                    ROOT, RESIDUAL_SOURCE_PATHS,
                ) != tree:
                    raise ValueError("residual source tree changed")

            verify()
            bound = _BoundResidual(
                context, phases, tree, context.primary_python,
                context.torch_argv, context.torch_probe,
                context_lease, benchmark, verify,
            )
            yield bound
            verify()


@contextmanager
def authenticate_residual_attempt(
    attempt: ResidualAttempt,
) -> Iterator[ResidualLease]:
    """Hold one source-derived residual attempt lease through execution."""
    _require_isolated_execution()
    if not isinstance(attempt, ResidualAttempt):
        raise ValueError("residual attempt is invalid")
    logical = _relative(
        Path(attempt.attempt_path), "residual attempt path",
    )
    if logical != Path(
        f"experiments/{attempt.run_id}-attempt.json",
    ):
        raise ValueError("residual attempt identity changed")
    path = ROOT / logical
    identities = _single_link_inputs((path,), "residual attempt")
    with freeze_inputs((path,)) as frozen:
        with _bound_residual() as bound:
            published = ResidualAttempt.read(
                frozen[0].snapshot, Path(attempt.attempt_path),
                ROOT, bound.context_attempt,
            )
            _validate_commit(attempt.implementation_commit, bound.tree)
            expected = _parse_constructed(
                _attempt_value(
                    Path(attempt.attempt_path),
                    attempt.implementation_commit, attempt.run_id, bound,
                ),
                Path(attempt.attempt_path), bound.context_attempt,
            )
            if attempt != published or published != expected:
                raise ValueError(
                    "residual attempt is not source-derived",
                )

            def verify() -> None:
                bound.verify()
                _verify_identities(identities)
                _single_link_inputs((path,), "residual attempt")
                verify_frozen(frozen)

            verify()
            yield ResidualLease(
                bound.context_lease, bound.benchmark, verify,
            )
            verify()


def _published_bytes(
    path: Path, directory: int, binding: ExclusiveTemp, payload: bytes,
) -> None:
    descriptor = os.open(
        path.name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK,
        dir_fd=directory,
    )
    try:
        expected = (
            binding.identity, stat.S_IFREG, 1, binding.mode, len(payload),
        )

        def state(metadata: os.stat_result) -> tuple[object, ...]:
            return (
                (metadata.st_dev, metadata.st_ino),
                stat.S_IFMT(metadata.st_mode), metadata.st_nlink,
                stat.S_IMODE(metadata.st_mode), metadata.st_size,
            )

        metadata = os.fstat(descriptor)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 16):
            digest.update(chunk)
        public = os.stat(
            path.name, dir_fd=directory, follow_symlinks=False,
        )
        if state(metadata) != expected or state(public) != expected or \
           digest.digest() != hashlib.sha256(payload).digest() or \
           state(os.fstat(descriptor)) != expected:
            raise ValueError("published residual attempt changed")
    finally:
        os.close(descriptor)


def arm(
    output: Path, implementation_commit: str, run_id: str,
) -> ResidualAttempt:
    """Atomically publish one exact development residual attempt."""
    _require_isolated_execution()
    output = _relative(output, "residual attempt output")
    if not COMMIT.fullmatch(implementation_commit) or \
       not RUN_ID.fullmatch(run_id) or \
       output != Path(f"experiments/{run_id}-attempt.json"):
        raise ValueError("residual attempt identity is invalid")
    output_path = ROOT / output
    run_path = ROOT / "reports" / run_id
    outcome_path = ROOT / "experiments" / f"{run_id}-outcome.json"
    absent = (
        (output_path, "residual attempt output"),
        (run_path, "residual run directory"),
        (outcome_path, "residual outcome"),
    )
    _fresh(absent)

    with _bound_residual() as bound:
        _validate_commit(implementation_commit, bound.tree)
        value = _attempt_value(
            output, implementation_commit, run_id, bound,
        )
        constructed = _parse_constructed(
            value, output, bound.context_attempt,
        )
        text = json.dumps(
            value, allow_nan=False, indent=2, sort_keys=True,
        ) + "\n"
        payload = text.encode()
        parent, parent_identity = _open_directory(output_path.parent)
        temporary: ExclusiveTemp | None = None

        def capture(binding: ExclusiveTemp) -> None:
            nonlocal temporary
            temporary = binding

        def verify(binding: ExclusiveTemp) -> None:
            if binding != temporary or \
               _directory_identity(output_path.parent) != parent_identity:
                raise OSError("residual attempt temporary changed")
            bound.verify()
            _fresh(absent)

        failure: BaseException | None = None
        try:
            write_json_exclusive(
                output_path, value, parent,
                before_link_with_temp=verify,
                on_temp_created=capture,
            )
            os.fsync(parent)
            if temporary is None:
                raise OSError(
                    "residual attempt temporary was not created",
                )
            _published_bytes(
                output_path, parent, temporary, payload,
            )
            if _directory_identity(
                output_path.parent,
            ) != parent_identity:
                raise ValueError("residual attempt parent changed")
            bound.verify()
            _absent(run_path, "residual run directory")
            _absent(outcome_path, "residual outcome")
            if _single_link_inputs(
                (output_path,), "published residual attempt",
            )[0][1] != temporary.identity:
                raise ValueError("published residual attempt changed")
            published = ResidualAttempt.read(
                output_path, output, ROOT, bound.context_attempt,
            )
            if published != constructed or _single_link_inputs(
                (output_path,), "published residual attempt",
            )[0][1] != temporary.identity:
                raise ValueError("published residual attempt changed")
            return published
        except BaseException as error:
            failure = error
        finally:
            try:
                if temporary is not None and _owns_entry(
                    parent, temporary, (1,),
                ):
                    os.unlink(temporary.name, dir_fd=parent)
                    os.fsync(parent)
            except BaseException as error:
                if failure is None:
                    failure = error
            finally:
                try:
                    os.close(parent)
                except BaseException as error:
                    if failure is None:
                        failure = error
        if failure is not None:
            raise failure
        raise OSError("residual attempt publication failed")


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        attempt = arm(
            args.output, args.implementation_commit, args.run_id,
        )
    except (
        KeyError, OSError, OverflowError, TypeError, UnicodeError, ValueError,
    ) as error:
        print(f"integrity error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps({
        "attempt": attempt.attempt_path,
        "run_id": attempt.run_id,
        "status": "armed",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
