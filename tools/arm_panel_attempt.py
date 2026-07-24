#!/usr/bin/env python3
"""Arm one immutable conditioned-panel calibration attempt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import os
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.files import (
    FrozenInput, freeze_inputs, verify_frozen, write_json,
    write_json_exclusive,
)
from tools.panel_contract import (
    COMPARISON_PROFILE, FINALIZER_SOURCE_PATHS, RUN_ID, SERIES, SOURCE_PATHS,
    FileBinding, PanelAttempt, PanelInputs, PanelProfile, SourceTree,
    _absent, _directory_identity, _open_directory, _tree_digest,
    expected_panel_commands, observe_torch, panel_profile,
    read_canonical_json, regular_file_identities, selected_source_tree,
)

COMMIT = re.compile(r"[0-9a-f]{40}")
HISTORICAL_TEMPLATE = Path(
    "experiments/executable-h13-panel-attempt.json"
)
HISTORICAL_TEMPLATE_SHA256 = \
    "f564ac8ff99e222d37de9500ffc2e5447055ad4b34ed8c90d7a7a733ab4d48c9"
HISTORICAL_INPUTS = Path("experiments/executable-h13-panel-inputs.json")
HISTORICAL_INPUTS_SHA256 = \
    "088e8f4c5b71e574620e86b8e56dcbe5dcf7f7a53ce52b2c6c616ed1d575f588"


def _relative(path: Path, label: str) -> Path:
    if path.is_absolute() or not path.parts or \
       any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative path")
    return path


def _fresh(paths: Sequence[tuple[Path, str]]) -> None:
    resolved = tuple(
        (path.resolve(strict=False), label) for path, label in paths
    )
    for index, (path, label) in enumerate(resolved):
        for other, other_label in resolved[:index]:
            overlaps = path == other or path in other.parents or \
                other in path.parents
            cache_nesting = (
                label == "Python cache prefix" and
                other_label == "run directory" and
                path == other / ".pycache"
            ) or (
                label == "run directory" and
                other_label == "Python cache prefix" and
                other == path / ".pycache"
            )
            if overlaps and not cache_nesting:
                raise ValueError("attempt output path topology is invalid")
    for path, label in paths:
        _absent(path, label)


def _frozen(
    values: Mapping[Path, FrozenInput], path: Path,
) -> FrozenInput:
    try:
        return values[path]
    except KeyError as error:
        raise ValueError(f"unfrozen panel input: {path}") from error


def _snapshot_tree(
    paths: Sequence[str], frozen: Mapping[Path, FrozenInput],
) -> SourceTree:
    files = tuple(
        FileBinding(path, _frozen(frozen, ROOT / path).sha256)
        for path in sorted(paths)
    )
    return SourceTree(str(ROOT.resolve()), files, _tree_digest(files))


def _validate_constructed(
    attempt: PanelAttempt, profile: PanelProfile,
    commands: Mapping[str, Sequence[str]],
) -> None:
    expected = {
        name: tuple(command) for name, command in commands.items()
    }
    if profile != COMPARISON_PROFILE or \
       attempt.expected_equivalent_runs != profile.expected_runs or \
       attempt.expected_panel_fits != profile.expected_panel_fits or \
       dict(attempt.commands) != expected:
        raise ValueError("attempt profile, counts, or commands disagree")


def _parse_constructed(
    value: Mapping[str, object], profile: PanelProfile,
    commands: Mapping[str, Sequence[str]],
) -> PanelAttempt:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-attempt-",
    ) as directory:
        path = Path(directory) / "attempt.json"
        write_json(path, value)
        attempt = PanelAttempt.read(path)
    _validate_constructed(attempt, profile, commands)
    return attempt


def arm(
    output: Path, runtime_template: Path, input_manifest: Path,
    config: Path, implementation_commit: str, run_id: str,
    run_dir: Path, outcome: Path,
) -> PanelAttempt:
    output = _relative(output, "output")
    runtime_template = _relative(runtime_template, "runtime template")
    input_manifest = _relative(input_manifest, "input manifest")
    config = _relative(config, "config")
    run_dir = _relative(run_dir, "run directory")
    outcome = _relative(outcome, "outcome")
    if runtime_template != HISTORICAL_TEMPLATE or \
       input_manifest != HISTORICAL_INPUTS:
        raise ValueError("historical panel artifact path is invalid")
    if not COMMIT.fullmatch(implementation_commit):
        raise ValueError("implementation commit must be 40 lowercase hex bytes")
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("run_id is invalid")

    cache = Path(f"{run_dir}/.pycache")
    absent = (
        (output, "output"),
        (run_dir, "run directory"),
        (cache, "Python cache prefix"),
        (outcome, "attempt outcome"),
    )
    _fresh(absent)
    parent_identities = tuple(
        (parent, _directory_identity(parent))
        for parent in dict.fromkeys((run_dir.parent, outcome.parent))
    )
    output_fd, output_parent_identity = _open_directory(output.parent)
    try:
        discovery_paths = (runtime_template, input_manifest)
        regular_file_identities(discovery_paths)
        with freeze_inputs(discovery_paths) as discovery:
            discovery_hashes = tuple(item.sha256 for item in discovery)
            if discovery_hashes != (
                HISTORICAL_TEMPLATE_SHA256, HISTORICAL_INPUTS_SHA256,
            ):
                raise ValueError("historical panel artifact hash is invalid")
            template_discovery = PanelAttempt.read(discovery[0].snapshot)
            inputs_discovery = PanelInputs.read(discovery[1].snapshot)

        if Path(template_discovery.source_tree.root) != ROOT.resolve():
            raise ValueError("runtime template source root is not this repository")
        baseline_paths = (
            Path(inputs_discovery.baseline_report.path),
            Path(inputs_discovery.baseline_ledger.path),
        )
        csv_paths = tuple(
            Path(item.csv.path) for item in inputs_discovery.series
        )
        source_paths = tuple(
            ROOT / path
            for path in dict.fromkeys(
                (*SOURCE_PATHS, *FINALIZER_SOURCE_PATHS)
            )
        )
        runtime_paths = tuple(map(Path, dict.fromkeys((
            template_discovery.primary_python.path,
            template_discovery.uv.path,
            template_discovery.torch_probe.python.path,
        ))))
        paths = tuple(dict.fromkeys((
            runtime_template, input_manifest, config, *baseline_paths,
            *csv_paths, *source_paths, *runtime_paths,
        )))
        identities = regular_file_identities(paths)

        with freeze_inputs(paths) as frozen:
            by_path = {item.source: item for item in frozen}
            template = PanelAttempt.read(
                _frozen(by_path, runtime_template).snapshot
            )
            inputs = PanelInputs.read(
                _frozen(by_path, input_manifest).snapshot
            )
            if template != template_discovery or \
               inputs != inputs_discovery or \
               (
                   _frozen(by_path, runtime_template).sha256,
                   _frozen(by_path, input_manifest).sha256,
               ) != discovery_hashes:
                raise ValueError("runtime template or input manifest changed")

            profile = panel_profile(read_canonical_json(
                _frozen(by_path, config).snapshot
            ))
            if profile != COMPARISON_PROFILE:
                raise ValueError("config is not the conditioned-panel profile")
            if tuple(item.name for item in inputs.series) != SERIES:
                raise ValueError("input manifest series order is invalid")

            manifest_binding = FileBinding(
                input_manifest.as_posix(),
                _frozen(by_path, input_manifest).sha256,
            )
            if manifest_binding != template.input_manifest:
                raise ValueError(
                    "input manifest does not match the runtime template"
                )
            if inputs.baseline_report != template.baseline_report or \
               inputs.baseline_ledger != template.baseline_ledger:
                raise ValueError(
                    "baseline bindings do not match the runtime template"
                )
            baseline_report = _frozen(by_path, baseline_paths[0])
            baseline_ledger = _frozen(by_path, baseline_paths[1])
            series = tuple(
                (item.name, _frozen(by_path, Path(item.csv.path)))
                for item in inputs.series
            )
            inputs.validate_direct(
                series, baseline_report, baseline_ledger,
            )

            template.primary_python.validate_live("primary Python")
            template.uv.validate_live("uv")
            torch_probe = observe_torch(template.torch_argv, ROOT)
            if torch_probe != template.torch_probe:
                raise ValueError("Torch runtime identity changed")

            source_tree = _snapshot_tree(SOURCE_PATHS, by_path)
            finalizer_tree = _snapshot_tree(
                FINALIZER_SOURCE_PATHS, by_path,
            )
            verify_frozen(frozen)
            if regular_file_identities(paths) != identities:
                raise ValueError("panel input identity changed")

            outputs = {
                "experiment_report": f"{run_dir}/experiment.json",
                "calibration_ledger": f"{run_dir}/calibration.jsonl",
                "analysis_report": f"{run_dir}/analysis.json",
                "outcome": outcome.as_posix(),
            }
            commands = expected_panel_commands(
                output, input_manifest.as_posix(), config.as_posix(),
                inputs.baseline_report.path, inputs.baseline_ledger.path,
                outputs, inputs, profile,
            )
            environment = dict(template.environment)
            environment["PYTHONPYCACHEPREFIX"] = cache.as_posix()
            value = {
                "baseline_ledger": asdict(template.baseline_ledger),
                "baseline_report": asdict(template.baseline_report),
                "commands": {
                    name: list(command)
                    for name, command in commands.items()
                },
                "config": {
                    "path": config.as_posix(),
                    "sha256": _frozen(by_path, config).sha256,
                },
                "environment": environment,
                "expected_equivalent_runs": profile.expected_runs,
                "expected_panel_fits": profile.expected_panel_fits,
                "finalizer_tree": asdict(finalizer_tree),
                "implementation_commit": implementation_commit,
                "input_manifest": asdict(manifest_binding),
                "outputs": outputs,
                "primary_python": asdict(template.primary_python),
                "run_dir": run_dir.as_posix(),
                "run_id": run_id,
                "schema": 1,
                "source_tree": asdict(source_tree),
                "status": "armed",
                "torch_argv": list(template.torch_argv),
                "torch_probe": asdict(template.torch_probe),
                "uv": asdict(template.uv),
            }
            constructed = _parse_constructed(value, profile, commands)
            if constructed.primary_python != template.primary_python or \
               constructed.uv != template.uv or \
               constructed.torch_argv != template.torch_argv or \
               constructed.torch_probe != template.torch_probe:
                raise ValueError("constructed runtime differs from the template")

            def verify() -> None:
                verify_frozen(frozen)
                if regular_file_identities(paths) != identities:
                    raise ValueError("panel input identity changed")
                if selected_source_tree(ROOT, SOURCE_PATHS) != source_tree or \
                   selected_source_tree(
                       ROOT, FINALIZER_SOURCE_PATHS,
                   ) != finalizer_tree:
                    raise ValueError("selected source tree changed")
                template.primary_python.validate_live("primary Python")
                template.uv.validate_live("uv")
                if observe_torch(template.torch_argv, ROOT) != \
                        template.torch_probe:
                    raise ValueError("Torch runtime identity changed")
                _fresh(absent)
                if any(
                    _directory_identity(parent) != identity
                    for parent, identity in parent_identities
                ):
                    raise ValueError("run or outcome parent changed")
                metadata = os.fstat(output_fd)
                identity = metadata.st_dev, metadata.st_ino
                if identity != output_parent_identity or \
                   _directory_identity(output.parent) != \
                        output_parent_identity:
                    raise ValueError("output parent changed")

            write_json_exclusive(output, value, output_fd, verify)
            os.fsync(output_fd)
            published = PanelAttempt.read(output)
            _validate_constructed(published, profile, commands)
            if published != constructed:
                raise ValueError("published attempt differs from construction")
            return published
    finally:
        os.close(output_fd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runtime-template", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--outcome", required=True, type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        attempt = arm(
            args.output, args.runtime_template, args.input_manifest,
            args.config, args.implementation_commit, args.run_id,
            args.run_dir, args.outcome,
        )
    except (
        KeyError, OSError, OverflowError, TypeError, UnicodeError, ValueError,
    ) as error:
        print(f"integrity error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps({
        "attempt": str(args.output), "run_id": attempt.run_id,
        "status": "armed",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
