"""Validate immutable inputs for one calibration-only panel experiment."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib
import json
import math
import os
import re
import stat
import subprocess

from tools.files import FrozenInput, file_sha256, freeze_inputs, verify_frozen

SOURCE_PATHS = (
    "tools/experiment.py",
    "tools/panel_contract.py",
    "tools/train.py",
    "tools/artifact_v1.py",
    "tools/data_v1.py",
    "tools/backtest.py",
    "tools/files.py",
    "tools/float32.py",
    "tools/analyze_panel.py",
    "tools/analyze_universe.py",
    "tools/fetch_universe.py",
    "tools/fetch_massive.py",
    "tools/finalize_panel_attempt.py",
)
FINALIZER_SOURCE_PATHS = (
    "tools/finalize_panel_attempt.py",
    "tools/panel_contract.py",
    "tools/files.py",
)
COMMANDS = ("validate_attempt", "preflight", "experiment", "analyze")
OUTPUTS = (
    "experiment_report", "calibration_ledger", "analysis_report", "outcome",
)
NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
HEX = re.compile(r"[0-9a-f]{64}")


def _object(value: object, fields: set[str], label: str
            ) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is invalid")
    return value


def _integer(value: object, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} is invalid")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError(f"{label} is invalid")
    return value


def _relative(value: object, label: str) -> str:
    path = Path(_string(value, label))
    if path.is_absolute() or not path.parts or \
       any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative path")
    return path.as_posix()


def _finite_tree(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("manifest numbers must be finite")
    if isinstance(value, Mapping):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate manifest field: {key}")
        result[key] = value
    return result


def read_canonical_json(path: Path) -> Mapping[str, object]:
    """Decode one finite, duplicate-free canonical JSON object."""
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw, object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid numeric constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read canonical manifest: {error}") from error
    _finite_tree(value)
    if not isinstance(value, dict) or raw != (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode():
        raise ValueError("manifest must be a canonical JSON object")
    return value


def _regular_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"input is unavailable: {path}") from error
    if stat.S_IFMT(metadata.st_mode) != stat.S_IFREG:
        raise ValueError(f"input must be a nonsymlink regular file: {path}")
    return metadata.st_dev, metadata.st_ino


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"directory is unavailable: {path}") from error
    if stat.S_IFMT(metadata.st_mode) != stat.S_IFDIR:
        raise ValueError(f"directory must be nonsymlink: {path}")
    return metadata.st_dev, metadata.st_ino


def _open_directory(path: Path) -> tuple[int, tuple[int, int]]:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ValueError(f"directory is unavailable: {path}") from error
    metadata = os.fstat(descriptor)
    if stat.S_IFMT(metadata.st_mode) != stat.S_IFDIR:
        os.close(descriptor)
        raise ValueError(f"directory must be nonsymlink: {path}")
    return descriptor, (metadata.st_dev, metadata.st_ino)


def _regular_inputs(
    paths: Sequence[Path],
) -> tuple[tuple[Path, tuple[int, int]], ...]:
    identities = tuple(_regular_identity(path) for path in paths)
    if len(identities) != len(set(identities)):
        raise ValueError("panel inputs must not alias each other")
    return tuple(zip(paths, identities, strict=True))


def _verify_identities(
    identities: Sequence[tuple[Path, tuple[int, int]]],
) -> None:
    if any(_regular_identity(path) != identity
           for path, identity in identities):
        raise ValueError("panel input identity changed during the command")


def _absent(path: Path, label: str) -> None:
    if os.path.lexists(path) or os.path.lexists(Path(os.path.abspath(path))) or \
       os.path.lexists(path.resolve(strict=False)):
        raise ValueError(f"{label} must be absent")


def _tree_digest(files: Sequence[FileBinding]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda file: file.path):
        digest.update(
            item.path.encode("utf-8") + b"\0" +
            item.sha256.encode("ascii") + b"\n"
        )
    return digest.hexdigest()


@dataclass(frozen=True)
class FileBinding:
    path: str
    sha256: str

    @classmethod
    def parse(cls, value: object, label: str, *, relative: bool = True
              ) -> FileBinding:
        item = _object(value, {"path", "sha256"}, label)
        path = (_relative(item["path"], f"{label}.path") if relative else
                _string(item["path"], f"{label}.path"))
        return cls(path, _sha256(item["sha256"], f"{label}.sha256"))

    def validate(self, frozen: FrozenInput, label: str) -> None:
        if str(frozen.source) != self.path or frozen.sha256 != self.sha256:
            raise ValueError(f"{label} does not match its frozen binding")


@dataclass(frozen=True)
class ExecutableBinding(FileBinding):
    version: str

    @classmethod
    def parse(cls, value: object, label: str) -> ExecutableBinding:
        item = _object(value, {"path", "sha256", "version"}, label)
        path = _string(item["path"], f"{label}.path")
        if not Path(path).is_absolute():
            raise ValueError(f"{label}.path must be absolute")
        return cls(
            path, _sha256(item["sha256"], f"{label}.sha256"),
            _string(item["version"], f"{label}.version"),
        )

    def validate_live(self, label: str) -> None:
        path = Path(self.path)
        _regular_identity(path)
        if file_sha256(path) != self.sha256 or \
           _command_version(path) != self.version:
            raise ValueError(f"{label} runtime identity changed")


@dataclass(frozen=True)
class SourceTree:
    root: str
    files: tuple[FileBinding, ...]
    sha256: str

    @classmethod
    def parse(cls, value: object, label: str,
              expected: Sequence[str] | None = None) -> SourceTree:
        item = _object(value, {"root", "files", "sha256"}, label)
        root = _string(item["root"], f"{label}.root")
        if not Path(root).is_absolute():
            raise ValueError(f"{label}.root must be absolute")
        raw = item["files"]
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{label}.files must be nonempty")
        files = tuple(
            FileBinding.parse(entry, f"{label}.files[{index}]")
            for index, entry in enumerate(raw)
        )
        paths = tuple(file.path for file in files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)) or \
           expected is not None and set(paths) != set(expected):
            raise ValueError(f"{label}.files are invalid")
        digest = _sha256(item["sha256"], f"{label}.sha256")
        if digest != _tree_digest(files):
            raise ValueError(f"{label} tree digest is invalid")
        return cls(root, files, digest)

    def validate(self, frozen: Mapping[str, FrozenInput], label: str) -> None:
        if set(frozen) != {item.path for item in self.files}:
            raise ValueError(f"{label} source set changed")
        for item in self.files:
            if frozen[item.path].sha256 != item.sha256:
                raise ValueError(f"{label} source changed")


@dataclass(frozen=True)
class TorchIdentity:
    python: ExecutableBinding
    version: str
    git_version: str | None
    cuda_version: str | None
    config: str
    package_tree: SourceTree

    @classmethod
    def parse(cls, value: object) -> TorchIdentity:
        item = _object(
            value,
            {
                "python", "version", "git_version", "cuda_version", "config",
                "package_tree",
            },
            "runtime.torch_probe",
        )
        optional = (item["git_version"], item["cuda_version"])
        if any(value is not None and not isinstance(value, str)
               for value in optional):
            raise ValueError("Torch optional runtime fields are invalid")
        return cls(
            ExecutableBinding.parse(item["python"], "runtime.torch_python"),
            _string(item["version"], "runtime.torch_version"),
            item["git_version"], item["cuda_version"],
            _string(item["config"], "runtime.torch_config"),
            SourceTree.parse(item["package_tree"], "runtime.package_tree"),
        )


@dataclass(frozen=True)
class PanelSeries:
    name: str
    csv: FileBinding
    rows: int
    first_timestamp: str
    last_timestamp: str
    timestamp_sha256: str

    @classmethod
    def parse(cls, value: object, index: int) -> PanelSeries:
        label = f"series[{index}]"
        item = _object(
            value,
            {
                "name", "csv", "rows", "first_timestamp",
                "last_timestamp", "timestamp_sha256",
            },
            label,
        )
        name = _string(item["name"], f"{label}.name")
        if not NAME.fullmatch(name):
            raise ValueError(f"{label}.name is invalid")
        return cls(
            name,
            FileBinding.parse(item["csv"], f"{label}.csv"),
            _integer(item["rows"], f"{label}.rows"),
            _string(item["first_timestamp"], f"{label}.first_timestamp"),
            _string(item["last_timestamp"], f"{label}.last_timestamp"),
            _sha256(
                item["timestamp_sha256"], f"{label}.timestamp_sha256",
            ),
        )


@dataclass(frozen=True)
class PanelInputs:
    series: tuple[PanelSeries, ...]
    baseline_report: FileBinding
    baseline_ledger: FileBinding

    @classmethod
    def read(cls, path: Path) -> PanelInputs:
        value = _object(
            read_canonical_json(path),
            {"schema", "series", "baseline_report", "baseline_ledger"},
            "input manifest",
        )
        if value["schema"] != 1 or not isinstance(value["series"], list) or \
           not value["series"]:
            raise ValueError("input manifest schema or series is invalid")
        series = tuple(
            PanelSeries.parse(item, index)
            for index, item in enumerate(value["series"])
        )
        if len({item.name for item in series}) != len(series):
            raise ValueError("input manifest series names must be unique")
        return cls(
            series,
            FileBinding.parse(value["baseline_report"], "baseline_report"),
            FileBinding.parse(value["baseline_ledger"], "baseline_ledger"),
        )

    def validate_direct(
        self, series: Sequence[tuple[str, FrozenInput]],
        baseline_report: FrozenInput, baseline_ledger: FrozenInput,
    ) -> None:
        if tuple(name for name, _ in series) != \
                tuple(item.name for item in self.series):
            raise ValueError("series order does not match the input manifest")
        for declared, (_, frozen) in zip(self.series, series, strict=True):
            declared.csv.validate(frozen, f"{declared.name} CSV")
        self.baseline_report.validate(baseline_report, "baseline report")
        self.baseline_ledger.validate(baseline_ledger, "baseline ledger")

    def validate_timestamps(
        self, series: Sequence[tuple[str, FrozenInput, Sequence[str]]],
    ) -> None:
        if tuple(name for name, _, _ in series) != \
                tuple(item.name for item in self.series):
            raise ValueError("series order does not match the input manifest")
        for declared, (_, frozen, timestamps) in zip(
            self.series, series, strict=True,
        ):
            declared.csv.validate(frozen, f"{declared.name} CSV")
            if not timestamps:
                raise ValueError(f"{declared.name} timestamp grid is empty")
            digest = hashlib.sha256(
                "".join(f"{timestamp}\n" for timestamp in timestamps).encode(
                    "ascii",
                )
            ).hexdigest()
            if len(timestamps) != declared.rows or \
               timestamps[0] != declared.first_timestamp or \
               timestamps[-1] != declared.last_timestamp or \
               digest != declared.timestamp_sha256:
                raise ValueError(
                    f"{declared.name} timestamps do not match the input manifest"
                )


def _argv(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or \
       any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must be a nonempty argument array")
    return tuple(value)


@dataclass(frozen=True)
class PanelAttempt:
    run_id: str
    run_dir: str
    implementation_commit: str
    input_manifest: FileBinding
    config: FileBinding
    baseline_report: FileBinding
    baseline_ledger: FileBinding
    source_tree: SourceTree
    finalizer_tree: SourceTree
    primary_python: ExecutableBinding
    uv: ExecutableBinding
    torch_argv: tuple[str, ...]
    torch_probe: TorchIdentity
    environment: Mapping[str, str]
    commands: Mapping[str, tuple[str, ...]]
    expected_equivalent_runs: int
    expected_panel_fits: int
    outputs: Mapping[str, str]

    @classmethod
    def read(cls, path: Path) -> PanelAttempt:
        value = _object(
            read_canonical_json(path),
            {
                "schema", "run_id", "status", "run_dir",
                "implementation_commit", "input_manifest", "config",
                "baseline_report", "baseline_ledger", "source_tree",
                "finalizer_tree", "primary_python", "uv", "torch_argv",
                "torch_probe", "environment", "commands",
                "expected_equivalent_runs", "expected_panel_fits", "outputs",
            },
            "attempt manifest",
        )
        if value["schema"] != 1 or value["status"] != "armed":
            raise ValueError("attempt must be schema 1 and armed")
        run_id = _string(value["run_id"], "run_id")
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("run_id is invalid")
        commit = _string(value["implementation_commit"],
                         "implementation_commit")
        if len(commit) not in (40, 64) or \
           any(byte not in "0123456789abcdef" for byte in commit):
            raise ValueError("implementation_commit is invalid")
        environment = _object(
            value["environment"],
            {"PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX"},
            "environment",
        )
        if environment["PYTHONDONTWRITEBYTECODE"] != "1":
            raise ValueError("bytecode must be disabled")
        commands = _object(
            value["commands"], {*COMMANDS, "finalizer_prefix"}, "commands",
        )
        parsed_commands = {
            name: _argv(commands[name], f"commands.{name}")
            for name in (*COMMANDS, "finalizer_prefix")
        }
        outputs = _object(value["outputs"], set(OUTPUTS), "outputs")
        parsed_outputs = {
            name: _relative(outputs[name], f"outputs.{name}")
            for name in OUTPUTS
        }
        run_dir = _relative(value["run_dir"], "run_dir")
        if environment["PYTHONPYCACHEPREFIX"] != \
                f"{run_dir}/.pycache" or \
           parsed_outputs["experiment_report"] != \
                f"{run_dir}/experiment.json" or \
           parsed_outputs["calibration_ledger"] != \
                f"{run_dir}/calibration.jsonl" or \
           parsed_outputs["analysis_report"] != \
                f"{run_dir}/analysis.json" or \
           len({
               Path(path).resolve(strict=False)
               for path in parsed_outputs.values()
           }) != len(parsed_outputs):
            raise ValueError("attempt output or cache paths are invalid")
        source = SourceTree.parse(
            value["source_tree"], "source_tree", SOURCE_PATHS,
        )
        finalizer = SourceTree.parse(
            value["finalizer_tree"], "finalizer_tree",
            FINALIZER_SOURCE_PATHS,
        )
        if source.root != finalizer.root:
            raise ValueError("source roots do not match")
        uv = ExecutableBinding.parse(value["uv"], "uv")
        torch_argv = _argv(value["torch_argv"], "torch_argv")
        if torch_argv != (
            uv.path, "run", "--offline", "--with", "torch", "python",
        ):
            raise ValueError("torch_argv is invalid")
        return cls(
            run_id, run_dir, commit,
            FileBinding.parse(value["input_manifest"], "input_manifest"),
            FileBinding.parse(value["config"], "config"),
            FileBinding.parse(value["baseline_report"], "baseline_report"),
            FileBinding.parse(value["baseline_ledger"], "baseline_ledger"),
            source, finalizer,
            ExecutableBinding.parse(value["primary_python"], "primary_python"),
            uv, torch_argv, TorchIdentity.parse(value["torch_probe"]),
            MappingProxyType(dict(environment)),
            MappingProxyType(parsed_commands),
            _integer(
                value["expected_equivalent_runs"],
                "expected_equivalent_runs",
            ),
            _integer(value["expected_panel_fits"], "expected_panel_fits"),
            MappingProxyType(parsed_outputs),
        )

    def source_paths(self) -> tuple[Path, ...]:
        paths = dict.fromkeys(
            item.path for tree in (self.source_tree, self.finalizer_tree)
            for item in tree.files
        )
        return tuple(Path(self.source_tree.root) / path for path in paths)

    def validate_stage(
        self, stage: str, argv: Sequence[str], environment: Mapping[str, str],
        torch_probe: TorchIdentity,
    ) -> None:
        if stage not in COMMANDS or tuple(argv) != self.commands[stage]:
            raise ValueError("calling stage arguments do not match the attempt")
        if dict(self.environment) != {
            name: environment.get(name) for name in self.environment
        }:
            raise ValueError("panel environment does not match the attempt")
        self.primary_python.validate_live("primary Python")
        self.uv.validate_live("uv")
        if torch_probe != self.torch_probe:
            raise ValueError("Torch runtime identity changed")

    def validate_paths(self, stage: str) -> None:
        run_dir = Path(self.run_dir)
        cache = Path(self.environment["PYTHONPYCACHEPREFIX"])
        if stage in ("validate_attempt", "preflight"):
            _absent(run_dir, "run directory")
        elif not run_dir.is_dir() or run_dir.is_symlink():
            raise ValueError("run directory must be a nonsymlink directory")
        _absent(cache, "Python cache prefix")
        _absent(Path(self.outputs["outcome"]), "attempt outcome")
        if stage == "experiment":
            for name in (
                "experiment_report", "calibration_ledger", "analysis_report",
            ):
                _absent(Path(self.outputs[name]), f"{name} output")
        elif stage == "analyze":
            for name in ("experiment_report", "calibration_ledger"):
                _regular_identity(Path(self.outputs[name]))
            _absent(Path(self.outputs["analysis_report"]), "analysis output")


@dataclass(frozen=True)
class PanelExecution:
    attempt_input: FrozenInput
    input_manifest_input: FrozenInput
    config_input: FrozenInput
    baseline_report_input: FrozenInput
    baseline_ledger_input: FrozenInput
    source_inputs: Mapping[str, FrozenInput]
    attempt: PanelAttempt
    inputs: PanelInputs
    series: tuple[tuple[str, FrozenInput], ...]
    torch_probe: TorchIdentity
    observe_torch: Callable[[], TorchIdentity]
    argv: tuple[str, ...]
    frozen: tuple[FrozenInput, ...]
    identities: tuple[tuple[Path, tuple[int, int]], ...]
    run_directory_fd: int
    run_directory_identity: tuple[int, int]

    def validate(self) -> None:
        self.verify()
        self.attempt.input_manifest.validate(
            self.input_manifest_input, "input manifest",
        )
        self.attempt.config.validate(self.config_input, "sweep config")
        self.attempt.baseline_report.validate(
            self.baseline_report_input, "baseline report",
        )
        self.attempt.baseline_ledger.validate(
            self.baseline_ledger_input, "baseline ledger",
        )
        self.attempt.source_tree.validate(
            {
                item.path: self.source_inputs[item.path]
                for item in self.attempt.source_tree.files
            },
            "implementation",
        )
        self.attempt.finalizer_tree.validate(
            {
                item.path: self.source_inputs[item.path]
                for item in self.attempt.finalizer_tree.files
            },
            "finalizer",
        )
        self.attempt.validate_stage(
            "experiment", self.argv, os.environ, self.torch_probe,
        )
        self.attempt.validate_paths("experiment")
        self.inputs.validate_direct(
            self.series, self.baseline_report_input,
            self.baseline_ledger_input,
        )

    def verify(self) -> None:
        verify_frozen(self.frozen)
        _verify_identities(self.identities)
        metadata = os.fstat(self.run_directory_fd)
        if (metadata.st_dev, metadata.st_ino) != \
                self.run_directory_identity or \
           _directory_identity(Path(self.attempt.run_dir)) != \
                self.run_directory_identity:
            raise ValueError("panel run directory changed")
        if self.observe_torch() != self.attempt.torch_probe:
            raise ValueError("Torch runtime identity changed")

    def validate_outputs(
        self, report: Path, ledger: Path | None,
    ) -> None:
        actual = (str(report), None if ledger is None else str(ledger))
        expected = (
            self.attempt.outputs["experiment_report"],
            self.attempt.outputs["calibration_ledger"],
        )
        if actual != expected:
            raise ValueError("panel outputs do not match the attempt")

    def prepare_output(self, name: str, path: Path) -> None:
        if str(path) != self.attempt.outputs[name]:
            raise ValueError("panel output does not match the attempt")
        self.verify()
        try:
            os.stat(
                path.name, dir_fd=self.run_directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        raise ValueError(f"{name} output must be absent")

    def provenance(self) -> dict[str, object]:
        return {
            "attempt_manifest": {
                "path": str(self.attempt_input.source),
                "sha256": self.attempt_input.sha256,
                "run_id": self.attempt.run_id,
            },
            "input_manifest": {
                "path": str(self.input_manifest_input.source),
                "sha256": self.input_manifest_input.sha256,
            },
        }


def _command_version(path: Path) -> str:
    try:
        result = subprocess.run(
            (str(path), "--version"), check=True, capture_output=True,
            text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot identify runtime executable: {path}") from error
    return (result.stdout or result.stderr).strip()


def executable_binding(path: Path, version: str) -> ExecutableBinding:
    resolved = path.resolve(strict=True)
    _regular_identity(resolved)
    return ExecutableBinding(str(resolved), file_sha256(resolved), version)


def source_tree(root: Path) -> SourceTree:
    """Hash every nonsymlink regular file below one resolved package root."""
    resolved = root.resolve(strict=True)
    files = []
    for path in resolved.rglob("*"):
        metadata = path.stat(follow_symlinks=False)
        if stat.S_IFMT(metadata.st_mode) == stat.S_IFREG:
            files.append(FileBinding(
                path.relative_to(resolved).as_posix(), file_sha256(path),
            ))
    files.sort(key=lambda item: item.path)
    if not files:
        raise ValueError("runtime package contains no regular files")
    return SourceTree(str(resolved), tuple(files), _tree_digest(files))


@contextmanager
def freeze_panel_execution(
    attempt_path: Path, input_manifest_path: Path, config_path: Path,
    baseline_report_path: Path, baseline_ledger_path: Path,
    series: Sequence[tuple[str, Path]], root: Path,
    observe_torch: Callable[[], TorchIdentity], argv: Sequence[str],
) -> Iterator[PanelExecution]:
    """Freeze the discovered panel closure and yield one validated execution."""
    direct = (
        attempt_path, input_manifest_path, config_path, baseline_report_path,
        baseline_ledger_path, *(path for _, path in series),
    )
    discovery_identity = _regular_inputs(direct)
    with freeze_inputs((attempt_path,)) as discovery:
        discovered = PanelAttempt.read(discovery[0].snapshot)
        if Path(discovered.source_tree.root) != root.resolve(strict=True):
            raise ValueError("attempt source root does not match the repository")
        source_paths = discovered.source_paths()
        unique_sources = tuple(dict.fromkeys(source_paths))
        executable_paths = tuple(dict.fromkeys(map(Path, (
            discovered.primary_python.path, discovered.uv.path,
            discovered.torch_probe.python.path,
        ))))
        paths = tuple(dict.fromkeys(
            (*direct, *unique_sources, *executable_paths),
        ))
        identities = _regular_inputs(paths)
        _verify_identities(discovery_identity)
        with freeze_inputs(paths) as frozen:
            verify_frozen(discovery)
            _verify_identities(identities)
            by_source = dict(zip(paths, frozen, strict=True))
            attempt = PanelAttempt.read(by_source[attempt_path].snapshot)
            if attempt != discovered:
                raise ValueError("attempt changed during source discovery")
            for binding, label in (
                (attempt.primary_python, "primary Python"),
                (attempt.uv, "uv"),
                (attempt.torch_probe.python, "Torch Python"),
            ):
                binding.validate(by_source[Path(binding.path)], label)
            source_inputs = MappingProxyType({
                item.path: by_source[Path(tree.root) / item.path]
                for tree in (attempt.source_tree, attempt.finalizer_tree)
                for item in tree.files
            })
            directory_fd, directory_identity = _open_directory(
                Path(attempt.run_dir),
            )
            try:
                execution = PanelExecution(
                    by_source[attempt_path], by_source[input_manifest_path],
                    by_source[config_path], by_source[baseline_report_path],
                    by_source[baseline_ledger_path], source_inputs, attempt,
                    PanelInputs.read(by_source[input_manifest_path].snapshot),
                    tuple(
                        (name, by_source[path]) for name, path in series
                    ),
                    observe_torch(), observe_torch, tuple(argv), frozen,
                    identities, directory_fd, directory_identity,
                )
                execution.validate()
                yield execution
                execution.verify()
            finally:
                os.close(directory_fd)
