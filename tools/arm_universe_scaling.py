#!/usr/bin/env python3
"""Arm one immutable development-only universe-scaling attempt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_universe import (
    ObservedCsv, read_json, validate_config, validate_fetch,
)
from tools.data_v1 import read_timestamps
from tools.fetch_universe import FETCH_SCHEMA, UniverseManifest
from tools.files import (
    FrozenInput, freeze_inputs, verify_frozen, write_json,
    write_json_exclusive,
)
from tools.panel_contract import (
    FileBinding, SourceTree, _absent, _directory_identity, _open_directory,
    _tree_digest, executable_binding, observe_torch,
    regular_file_identities, selected_source_tree,
)
from tools.session_calendar import SessionCalendar
from tools.universe_contract import universe_roles
from tools.universe_scaling_contract import (
    CALENDAR_PATH, CALENDAR_SHA256, CONFIG_PATH, CONFIG_SHA256,
    EXPECTED_BUDGETS, EXPECTED_FIT_COUNT, EXPECTED_MISSING, FETCH_PATH,
    FETCH_SHA256, FINALIZER_SOURCE_PATHS, MANIFEST_BINDINGS, RUN_ID,
    SOURCE_PATHS, ScalingAttempt, expected_fit_jobs, expected_protocol,
    expected_scaling_commands,
)
from tools.universe_scaling_inputs import (
    ScalingCoverage, common_coverage, fetch_series, selection_binding,
    selection_paths,
)

COMMIT = re.compile(r"[0-9a-f]{40}")


def _relative(path: Path, label: str) -> Path:
    if path.is_absolute() or not path.parts or \
       any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative path")
    return path


def _fresh(paths: Sequence[tuple[Path, str]]) -> None:
    resolved = tuple(
        (path.resolve(strict=False), label) for path, label in paths
    )
    for index, (path, _) in enumerate(resolved):
        for other, _ in resolved[:index]:
            nested_cache = (
                path.name == ".pycache" and path.parent == other
            ) or (
                other.name == ".pycache" and other.parent == path
            )
            if not nested_cache and (
                path == other or path in other.parents or
                other in path.parents
            ):
                raise ValueError("attempt output path topology is invalid")
    for path, label in paths:
        _absent(path, label)


def _frozen(
    values: Mapping[Path, FrozenInput], path: Path,
) -> FrozenInput:
    try:
        return values[path]
    except KeyError as error:
        raise ValueError(f"unfrozen scaling input: {path}") from error


def _view(value: FrozenInput, source: str) -> FrozenInput:
    return FrozenInput(
        Path(source), value.snapshot, value.sha256,
        value.snapshot_identity,
    )


def _require_expected_coverage(
    coverage: ScalingCoverage, names: Sequence[str],
) -> None:
    coverage.require_promotable()
    if tuple(
        (item.phase, item.missing) for item in coverage.phases
    ) != EXPECTED_MISSING:
        raise ValueError("scaling coverage does not match the frozen benchmark")
    jobs = expected_fit_jobs(
        names, {item.phase: item.evaluable for item in coverage.phases},
    )
    if len(jobs) != EXPECTED_FIT_COUNT or len(set(jobs)) != len(jobs):
        raise ValueError("scaling fit schedule does not match the benchmark")


def _snapshot_tree(
    paths: Sequence[str], frozen: Mapping[Path, FrozenInput],
) -> SourceTree:
    files = tuple(
        FileBinding(path, _frozen(frozen, ROOT / path).sha256)
        for path in sorted(paths)
    )
    return SourceTree(str(ROOT.resolve()), files, _tree_digest(files))


def _version(path: Path) -> str:
    try:
        result = subprocess.run(
            (str(path), "--version"), check=True, capture_output=True,
            text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot identify runtime: {path}") from error
    return (result.stdout or result.stderr).strip()


def _parse_constructed(
    value: Mapping[str, object], logical_path: Path,
) -> ScalingAttempt:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-attempt-",
    ) as directory:
        snapshot = Path(directory) / "attempt.json"
        write_json(snapshot, value)
        return ScalingAttempt.read(snapshot, logical_path, ROOT)


def _validate_data(
    frozen: Mapping[Path, FrozenInput],
    discovery: Mapping[str, object],
) -> ScalingCoverage:
    report_input = _frozen(frozen, ROOT / FETCH_PATH)
    report = read_json(report_input.snapshot, canonical=True)
    if report != discovery or report_input.sha256 != FETCH_SHA256:
        raise ValueError("frozen fetch report changed")
    if type(report.get("fetch_schema")) is not int or \
       report["fetch_schema"] != FETCH_SCHEMA:
        raise ValueError("scaling fetch report must use schema 4")
    series = fetch_series(report)
    manifests = []
    for size, binding in MANIFEST_BINDINGS.items():
        path = ROOT / binding.path
        value = _frozen(frozen, path)
        if value.sha256 != binding.sha256:
            raise ValueError(f"cohort-{size} manifest changed")
        manifests.append(UniverseManifest.read(value.snapshot))
    names = tuple(item.ticker for item in manifests[-1].series)
    roles = universe_roles(names)
    if any(
        tuple(item.ticker for item in manifest.series) != members
        for manifest, (_, members) in zip(
            manifests, roles.cohorts, strict=True,
        )
    ):
        raise ValueError("cohort manifests are not exact nested prefixes")

    calendar_input = _frozen(frozen, ROOT / CALENDAR_PATH)
    config_input = _frozen(frozen, ROOT / CONFIG_PATH)
    if calendar_input.sha256 != CALENDAR_SHA256 or \
       config_input.sha256 != CONFIG_SHA256:
        raise ValueError("calendar or config binding changed")
    calendar = SessionCalendar.read(calendar_input.snapshot)
    validate_config(read_json(config_input.snapshot))
    bar_inputs = tuple(
        _frozen(frozen, Path(item.csv.path)) for item in series
    )
    observed = {
        item.name: ObservedCsv(
            item.csv.path, value.sha256, read_timestamps(value.snapshot),
        )
        for item, value in zip(series, bar_inputs, strict=True)
    }
    validate_fetch(
        report, manifests[-1],
        _view(
            _frozen(
                frozen, ROOT / MANIFEST_BINDINGS[55].path,
            ),
            MANIFEST_BINDINGS[55].path,
        ),
        observed,
        _view(calendar_input, CALENDAR_PATH.as_posix()),
    )
    coverage = common_coverage(
        manifests[-1], calendar,
        {name: item.timestamps for name, item in observed.items()},
    )
    _require_expected_coverage(coverage, names)
    return coverage


def arm(
    output: Path, implementation_commit: str, run_id: str,
    primary_python: Path, torch_python: Path,
) -> ScalingAttempt:
    output = _relative(output, "attempt output")
    if not COMMIT.fullmatch(implementation_commit) or \
       not RUN_ID.fullmatch(run_id) or \
       output != Path(f"experiments/{run_id}-attempt.json"):
        raise ValueError("attempt identity is invalid")
    run_dir = Path(f"reports/{run_id}")
    cache = run_dir / ".pycache"
    outcome = Path(f"experiments/{run_id}-outcome.json")
    output_path, run_path, cache_path, outcome_path = (
        ROOT / output, ROOT / run_dir, ROOT / cache, ROOT / outcome,
    )
    absent = (
        (output_path, "attempt output"), (run_path, "run directory"),
        (cache_path, "Python cache prefix"),
        (outcome_path, "attempt outcome"),
    )
    _fresh(absent)
    selection = selection_binding()
    members = selection_paths()
    fetch = ROOT / FETCH_PATH
    with freeze_inputs((fetch,)) as frozen_fetch:
        if frozen_fetch[0].sha256 != FETCH_SHA256:
            raise ValueError("fetch report does not match the benchmark")
        discovery = read_json(frozen_fetch[0].snapshot, canonical=True)
        series = fetch_series(discovery)

    direct = tuple(dict.fromkeys((
        *members, ROOT / FETCH_PATH, ROOT / CALENDAR_PATH,
        ROOT / CONFIG_PATH,
        *(ROOT / binding.path for binding in MANIFEST_BINDINGS.values()),
        *(Path(item.csv.path) for item in series),
    )))
    direct_identities = regular_file_identities(direct)
    with freeze_inputs(direct) as frozen_direct:
        direct_by_path = {item.source: item for item in frozen_direct}
        coverage = _validate_data(direct_by_path, discovery)
        verify_frozen(frozen_direct)
        if regular_file_identities(direct) != direct_identities or \
           selection_binding() != selection:
            raise ValueError("frozen data identity changed")

        source_paths = tuple(
            ROOT / path for path in dict.fromkeys((
                *SOURCE_PATHS, *FINALIZER_SOURCE_PATHS,
            ))
        )
        torch_launcher = Path(os.path.abspath(torch_python))
        runtime_paths = (
            primary_python.resolve(strict=True),
            torch_launcher.resolve(strict=True),
        )
        support = tuple(dict.fromkeys((*source_paths, *runtime_paths)))
        support_identities = regular_file_identities(support)
        with freeze_inputs(support) as frozen_support:
            support_by_path = {item.source: item for item in frozen_support}
            primary = executable_binding(runtime_paths[0], _version(
                runtime_paths[0],
            ))
            torch = observe_torch((str(torch_launcher),), ROOT)
            if primary.sha256 != _frozen(
                support_by_path, runtime_paths[0],
            ).sha256 or Path(torch.python.path) != runtime_paths[1] or \
                    torch.python.sha256 != _frozen(
                support_by_path, runtime_paths[1],
            ).sha256:
                raise ValueError("runtime changed while arming")
            source = _snapshot_tree(SOURCE_PATHS, support_by_path)
            finalizer = _snapshot_tree(
                FINALIZER_SOURCE_PATHS, support_by_path,
            )
            outputs = {
                "fits": f"{run_dir}/fits.jsonl",
                "predictions": f"{run_dir}/predictions.jsonl",
                "summary": f"{run_dir}/summary.json",
                "outcome": outcome.as_posix(),
            }
            commands = expected_scaling_commands(output, outputs)
            value = {
                "attempt_path": output.as_posix(),
                "budgets": [
                    {"phase": phase, **asdict(budget)}
                    for phase, budget in EXPECTED_BUDGETS
                ],
                "commands": {
                    name: list(command)
                    for name, command in commands.items()
                },
                "config": {
                    "path": CONFIG_PATH.as_posix(),
                    "sha256": CONFIG_SHA256,
                },
                "coverage": [
                    {
                        "phase": phase.phase,
                        "series": [
                            asdict(item) for item in phase.series
                        ],
                    }
                    for phase in coverage.phases
                ],
                "environment": {
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": cache.as_posix(),
                },
                "fetch_report": {
                    "path": FETCH_PATH.as_posix(),
                    "sha256": FETCH_SHA256,
                },
                "finalizer_tree": asdict(finalizer),
                "implementation_commit": implementation_commit,
                "manifests": [
                    {
                        "size": size, "path": binding.path,
                        "sha256": binding.sha256,
                    }
                    for size, binding in MANIFEST_BINDINGS.items()
                ],
                "outputs": outputs,
                "primary_python": asdict(primary),
                "protocol": expected_protocol(),
                "run_dir": run_dir.as_posix(),
                "run_id": run_id,
                "schema": 1,
                "selection_tree": asdict(selection),
                "session_calendar": {
                    "path": CALENDAR_PATH.as_posix(),
                    "sha256": CALENDAR_SHA256,
                },
                "source_tree": asdict(source),
                "status": "armed",
                "torch_argv": [torch.python.path],
                "torch_probe": asdict(torch),
            }
            constructed = _parse_constructed(value, output)
            parents = tuple(
                (path, _directory_identity(path))
                for path in dict.fromkeys((
                    output_path.parent, run_path.parent,
                ))
            )
            output_fd, output_parent = _open_directory(output_path.parent)
            try:
                def verify() -> None:
                    verify_frozen((*frozen_direct, *frozen_support))
                    if regular_file_identities(direct) != \
                            direct_identities or \
                       regular_file_identities(support) != \
                            support_identities or \
                       selection_binding() != selection or \
                       selected_source_tree(ROOT, SOURCE_PATHS) != source or \
                       selected_source_tree(
                           ROOT, FINALIZER_SOURCE_PATHS,
                       ) != finalizer or \
                       executable_binding(
                           runtime_paths[0], _version(runtime_paths[0]),
                       ) != primary or \
                       torch_launcher.resolve(strict=True) != \
                           runtime_paths[1] or \
                       observe_torch((str(torch_launcher),), ROOT) != torch:
                        raise ValueError("attempt input changed while arming")
                    _fresh(absent)
                    if any(
                        _directory_identity(path) != identity
                        for path, identity in parents
                    ) or _directory_identity(
                        output_path.parent,
                    ) != output_parent:
                        raise ValueError("attempt output parent changed")

                write_json_exclusive(output_path, value, output_fd, verify)
                os.fsync(output_fd)
            finally:
                os.close(output_fd)
            published = ScalingAttempt.read(output_path, output, ROOT)
            if published != constructed:
                raise ValueError("published attempt differs from construction")
            return published


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--primary-python", required=True, type=Path)
    parser.add_argument("--torch-python", required=True, type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        attempt = arm(
            args.output, args.implementation_commit, args.run_id,
            args.primary_python, args.torch_python,
        )
    except (
        KeyError, OSError, OverflowError, TypeError, UnicodeError, ValueError,
    ) as error:
        print(f"integrity error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps({
        "attempt": args.output.as_posix(),
        "run_id": attempt.run_id,
        "status": "armed",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
