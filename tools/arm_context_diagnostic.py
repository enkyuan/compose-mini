#!/usr/bin/env python3
"""Arm one immutable development-only temporal-context diagnostic."""

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
            "context armer requires isolated bytecode-free Python",
        )


def _require_exact_launch() -> None:
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
        raise ValueError("context armer requires its exact Python launch")


if __name__ == "__main__":
    _require_isolated_execution()
    _require_exact_launch()

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.context_diagnostic_contract import (
    BATCH_SIZE, CONTEXT_CONFIG, CONTEXT_SOURCE_PATHS, EVALUATION_RANKS,
    HISTORY_LENGTHS, NEURAL_MODELS, PHASE_RANGES, PRIOR_PHASE, PYTHON_FLAGS,
    SOURCE_EVIDENCE, TARGET_PHASES, TRAINING_COHORT, ContextAttempt,
    ContextPhase, ContextScalerInput, context_phase_value,
    context_scaler_inputs_sha256,
    validate_context_sweep,
)
from tools.context_diagnostic_inputs import (
    context_all_phase_rows, context_csv_prefix_sha256,
    context_cutoff_timestamp,
    context_grid_sha256, timestamp_rows,
)
from tools.data_v1 import read_timestamps_until
from tools.fetch_universe import UniverseManifest
from tools.files import (
    FrozenInput, freeze_inputs, verify_frozen, write_json,
    write_json_exclusive,
)
from tools.finalize_universe_scaling import (
    _fetch_bindings, _master_from_snapshot, validate_fit_ledger,
)
from tools.panel_contract import (
    ExecutableBinding, FileBinding, SourceTree, TorchIdentity, _absent,
    _directory_identity, _open_directory, _regular_inputs, _tree_digest,
    _verify_identities, read_canonical_json,
    read_canonical_json_lines, selected_source_tree, source_tree,
)
from tools.session_calendar import SessionCalendar
from tools.universe_contract import PackedRows, fixed_update_budget
from tools.universe_scaling_contract import (
    FitJob, ScalingAttempt, fit_provenance_id, timestamp_grid_sha256,
)

if PYTHON_FLAGS != _BOOTSTRAP_FLAGS:
    raise ValueError("context Python isolation flags changed")

COMMIT = re.compile(r"[0-9a-f]{40}")
SOURCE_PREDICTIONS_SHA256 = \
    "1ed1671fc435903b7e36620c3835d619059acdf754e20905ee2a1d276fea515c"
SOURCE_SUMMARY_SHA256 = \
    "22607ed864f0074bbc2e1fa7447d48ffce23003beb43a0d70a7e0060e6f57b2f"
GIT = "/usr/bin/git"
GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}
Verify = Callable[[], None]


@dataclass(frozen=True, slots=True)
class ContextSnapshots:
    """Expose only frozen snapshots needed by the authenticated controller."""

    config: FrozenInput
    manifest: FrozenInput
    calendar: FrozenInput
    csv: tuple[tuple[str, FrozenInput], ...]


@dataclass(frozen=True, slots=True)
class ContextLease:
    """Keep source, data, and runtime identities live through one phase."""

    snapshots: ContextSnapshots
    verify: Verify = field(repr=False, compare=False)

    def __call__(self) -> None:
        self.verify()


@dataclass(frozen=True, slots=True)
class _BoundContext:
    """Hold one fully derived source and runtime closure while it is frozen."""

    master: tuple[str, ...]
    phases: tuple[ContextPhase, ...]
    primary: ExecutableBinding
    torch_argv: tuple[str, ...]
    torch: TorchIdentity
    tree: SourceTree
    snapshots: ContextSnapshots
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
        if any(
            path == other or path in other.parents or other in path.parents
            for other, _ in resolved[:index]
        ):
            raise ValueError("context output topology is invalid")
    for path, label in paths:
        _absent(path, label)


def _frozen(
    values: Mapping[Path, FrozenInput], path: Path,
) -> FrozenInput:
    try:
        return values[path]
    except KeyError as error:
        raise ValueError(f"unfrozen context input: {path}") from error


def _snapshot_tree(
    frozen: Mapping[Path, FrozenInput],
) -> SourceTree:
    files = tuple(
        FileBinding(path, _frozen(frozen, ROOT / path).sha256)
        for path in CONTEXT_SOURCE_PATHS
    )
    return SourceTree(str(ROOT.resolve()), files, _tree_digest(files))


def _parse_constructed(
    value: Mapping[str, object], logical_path: Path,
) -> ContextAttempt:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-context-attempt-",
    ) as directory:
        snapshot = Path(directory) / "attempt.json"
        write_json(snapshot, value)
        return ContextAttempt.read(snapshot, logical_path, ROOT)


def _source_attempt(snapshot: FrozenInput) -> ScalingAttempt:
    binding = SOURCE_EVIDENCE["attempt"]
    if snapshot.sha256 != binding.sha256:
        raise ValueError("source attempt changed")
    return ScalingAttempt.read(
        snapshot.snapshot, Path(binding.path), ROOT,
    )


def _source_outputs(
    failure: FrozenInput, attempt: ScalingAttempt,
) -> dict[str, FileBinding]:
    if failure.sha256 != SOURCE_EVIDENCE["failure"].sha256:
        raise ValueError("source failure changed")
    value = read_canonical_json(failure.snapshot)
    if not isinstance(value, Mapping) or set(value) != {
        "attempt", "ended", "exit", "integrity", "outputs", "schema",
        "stage", "started", "status",
    } or type(value["schema"]) is not int or value["schema"] != 1 or \
       value["stage"] != "analysis" or \
       value["status"] != "gate-failure" or \
       type(value["exit"]) is not int or value["exit"] != 3:
        raise ValueError("source failure is not the declared terminal gate")
    source = value["attempt"]
    outputs = value["outputs"]
    if not isinstance(source, Mapping) or source != {
        "path": str(ROOT / attempt.attempt_path),
        "run_id": attempt.run_id,
        "sha256": SOURCE_EVIDENCE["attempt"].sha256,
    } or not isinstance(outputs, Mapping) or set(outputs) != {
        "fits", "outcome", "predictions", "summary",
    }:
        raise ValueError("source failure does not bind its attempt")

    expected = {
        "fits": SOURCE_EVIDENCE["fits"].sha256,
        "predictions": SOURCE_PREDICTIONS_SHA256,
        "summary": SOURCE_SUMMARY_SHA256,
    }
    result = {}
    for name, sha256 in expected.items():
        item = outputs[name]
        if not isinstance(item, Mapping) or item.get("state") != "present" or \
           item.get("sha256") != sha256 or \
           item.get("path") != str(ROOT / attempt.outputs[name]):
            raise ValueError(f"source {name} output changed")
        result[name] = FileBinding(attempt.outputs[name], sha256)
    outcome = outputs["outcome"]
    if not isinstance(outcome, Mapping) or outcome != {
        "path": str(ROOT / attempt.outputs["outcome"]),
        "sha256": None,
        "state": "absent",
    }:
        raise ValueError("source failure outcome state changed")
    return result


def _validate_summary(
    snapshot: FrozenInput, outputs: Mapping[str, FileBinding],
) -> None:
    if snapshot.sha256 != SOURCE_SUMMARY_SHA256:
        raise ValueError("source summary changed")
    value = read_canonical_json(snapshot.snapshot)
    if not isinstance(value, Mapping):
        raise ValueError("source summary is not an object")
    inputs = value.get("inputs")
    if type(value.get("schema")) is not int or \
       value.get("schema") != 1 or \
       value.get("status") != "gate-failure" or \
       value.get("evidence_role") != \
            "development-diagnostic-not-forward-clean" or \
       value.get("locks") != {
           "backtest_run": False,
           "policy_selected": False,
           "reserved_test_materialized_samples": 0,
           "trading_authorized": False,
       } or not isinstance(inputs, Mapping) or inputs != {
           "attempt": {
               "path": str(ROOT / SOURCE_EVIDENCE["attempt"].path),
               "sha256": SOURCE_EVIDENCE["attempt"].sha256,
           },
           "fits": {
               "path": str(ROOT / outputs["fits"].path),
               "sha256": outputs["fits"].sha256,
           },
           "predictions": {
               "path": str(ROOT / outputs["predictions"].path),
               "sha256": outputs["predictions"].sha256,
           },
       }:
        raise ValueError("source summary provenance changed")


def _prior_selections(
    phase: str, master: tuple[str, ...],
    records: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    members = master[:TRAINING_COHORT]
    values = []
    for model in NEURAL_MODELS:
        for seed in (7, 19, 31, 43, 61):
            job = FitJob(
                "pooled", "fixed-update", TRAINING_COHORT,
                PRIOR_PHASE[phase], model, seed, members,
            )
            provenance = fit_provenance_id(job)
            try:
                record = records[provenance]
            except KeyError as error:
                raise ValueError("source checkpoint selection is missing") \
                    from error
            values.append({
                "model": model,
                "seed": seed,
                "selected_checkpoint": record["selected_checkpoint"],
                "source_model_fingerprint": record["model_fingerprint"],
                "source_provenance_id": provenance,
            })
    return values


def _phase(
    name: str, master: tuple[str, ...],
    packed: Mapping[tuple[str, str], PackedRows],
    grids: Mapping[
        tuple[str, str],
        tuple[
            tuple[tuple[str, str, str], ...],
            tuple[tuple[str, str, str], ...],
        ],
    ],
    prefix_sha256: Mapping[tuple[str, str], str],
    fit_records: Mapping[str, Mapping[str, object]],
) -> ContextPhase:
    training = master[:TRAINING_COHORT]
    evaluation = tuple(master[rank - 1] for rank in EVALUATION_RANKS)
    training_rows = [
        {"count": packed[name, series].counts[0], "series": series}
        for series in training
    ]
    evaluation_rows = [
        {
            "count": packed[name, series].counts[1],
            "grid_sha256": timestamp_grid_sha256(
                grids[name, series][1],
            ),
            "series": series,
        }
        for series in evaluation
    ]
    scaler_inputs = tuple(
        ContextScalerInput(
            series, prefix_sha256[name, series],
            packed[name, series].counts[0],
            timestamp_grid_sha256(grids[name, series][0]),
        )
        for series in master
    )
    value = {
        "evaluation_grid_sha256": context_grid_sha256(
            "evaluation", evaluation, {
                series: grids[name, series][1] for series in evaluation
            },
        ),
        "evaluation_rows": evaluation_rows,
        "phase": name,
        "prior_selections": _prior_selections(
            name, master, fit_records,
        ),
        "scaler_inputs_sha256": context_scaler_inputs_sha256(
            master, scaler_inputs,
        ),
        "source_ranges": list(map(list, PHASE_RANGES[name])),
        "training_grid_sha256": context_grid_sha256(
            "training", training, {
                series: grids[name, series][0] for series in training
            },
        ),
        "training_rows": training_rows,
        "updates_per_checkpoint": fixed_update_budget(
            sum(item["count"] for item in training_rows[:11]),
            BATCH_SIZE, 1,
        ).updates_per_checkpoint,
    }
    return ContextPhase.parse(value, master)


def _derive_phases(
    attempt: ScalingAttempt, master: tuple[str, ...],
    csv: Sequence[FileBinding], data: Mapping[Path, FrozenInput],
    config: FrozenInput,
) -> tuple[ContextPhase, ...]:
    closure = validate_fit_ledger(
        read_canonical_json_lines(
            _frozen(data, ROOT / SOURCE_EVIDENCE["fits"].path).snapshot,
        ),
        master, attempt.coverage,
    )
    records = {
        record["provenance_id"]: record for record in closure.records
    }
    manifest = UniverseManifest.read(
        _frozen(data, ROOT / attempt.manifests[-1].file.path).snapshot,
    )
    calendar = SessionCalendar.read(
        _frozen(data, ROOT / attempt.session_calendar.path).snapshot,
    )
    sweep = validate_context_sweep(read_canonical_json(
        config.snapshot,
    ))
    cutoff = context_cutoff_timestamp(
        calendar, manifest.start, manifest.end, manifest.interval_minutes,
        sweep["target_horizon_bars"], sweep["alignment_horizon_bars"],
    )
    packed = {}
    grids = {}
    prefix_sha256 = {}
    for series, binding in zip(master, csv, strict=True):
        frozen = _frozen(data, Path(binding.path))
        timestamps = read_timestamps_until(frozen.snapshot, cutoff)
        rows_by_phase = context_all_phase_rows(
            timestamps, manifest.interval_minutes, calendar,
            manifest.start, manifest.end,
            sweep["target_horizon_bars"],
            sweep["alignment_horizon_bars"],
        )
        for name, rows in rows_by_phase:
            boundary = rows.counts[0]
            training = rows.rows[:boundary]
            if not training:
                raise ValueError("context training prefix is empty")
            prefix_sha256[name, series] = context_csv_prefix_sha256(
                frozen.snapshot, timestamps[training[-1].target],
            )
            packed[name, series] = rows
            grids[name, series] = (
                timestamp_rows(timestamps, rows.rows[:boundary]),
                timestamp_rows(timestamps, rows.rows[boundary:]),
            )
    return tuple(
        _phase(
            name, master, packed, grids, prefix_sha256, records,
        )
        for name in TARGET_PHASES
    )


def _validate_commit(commit: str, tree: SourceTree) -> None:
    """Require the declared commit to contain the exact bound source bytes."""
    try:
        kind = subprocess.run(
            (
                GIT, "--no-replace-objects", "-C", str(ROOT),
                "cat-file", "-t", commit,
            ),
            check=True, capture_output=True, text=True, timeout=30,
            env=GIT_ENVIRONMENT,
        ).stdout.strip()
        if kind != "commit":
            raise ValueError("implementation object is not a commit")
        for binding in tree.files:
            content = subprocess.run(
                (
                    GIT, "--no-replace-objects", "-C", str(ROOT), "show",
                    f"{commit}:{binding.path}",
                ),
                check=True, capture_output=True, timeout=30,
                env=GIT_ENVIRONMENT,
            ).stdout
            if hashlib.sha256(content).hexdigest() != binding.sha256:
                raise ValueError(
                    "implementation commit differs from the source tree",
                )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("implementation commit is unavailable") from error


@contextmanager
def _bound_context(
    primary_python: Path, torch_python: Path,
) -> Iterator[_BoundContext]:
    """Freeze, derive, and continuously verify the exact context inputs."""
    source_paths = tuple(
        ROOT / binding.path for binding in SOURCE_EVIDENCE.values()
    ) + (ROOT / CONTEXT_CONFIG.path,)
    source_identities = _regular_inputs(source_paths)
    with freeze_inputs(source_paths) as frozen_source:
        source = dict(zip(source_paths, frozen_source, strict=True))
        scaling = _source_attempt(
            _frozen(source, ROOT / SOURCE_EVIDENCE["attempt"].path),
        )
        outputs = _source_outputs(
            _frozen(source, ROOT / SOURCE_EVIDENCE["failure"].path),
            scaling,
        )
        if _frozen(
            source, ROOT / SOURCE_EVIDENCE["fits"].path,
        ).sha256 != outputs["fits"].sha256 or \
           _frozen(
               source, ROOT / CONTEXT_CONFIG.path,
           ).sha256 != CONTEXT_CONFIG.sha256:
            raise ValueError("context source binding changed")

        fetch_path = ROOT / scaling.fetch_report.path
        fetch_identities = _regular_inputs((fetch_path,))
        with freeze_inputs((fetch_path,)) as frozen_fetch:
            if frozen_fetch[0].sha256 != scaling.fetch_report.sha256:
                raise ValueError("source fetch report changed")
            names, csv = _fetch_bindings(frozen_fetch[0].snapshot)
            data_paths = tuple(dict.fromkeys((
                fetch_path,
                *(ROOT / item.file.path for item in scaling.manifests),
                ROOT / scaling.session_calendar.path,
                ROOT / scaling.config.path,
                *(Path(item.path) for item in csv),
                *(ROOT / item.path for item in outputs.values()),
            )))
            data_identities = _regular_inputs(data_paths)
            with freeze_inputs(data_paths) as frozen_data:
                data = dict(zip(data_paths, frozen_data, strict=True))
                bindings = (
                    *(item.file for item in scaling.manifests),
                    scaling.fetch_report, scaling.session_calendar,
                    scaling.config, *csv, *outputs.values(),
                )
                if any(
                    _frozen(data, (
                        ROOT / binding.path
                        if not Path(binding.path).is_absolute() else
                        Path(binding.path)
                    )).sha256 != binding.sha256
                    for binding in bindings
                ):
                    raise ValueError("source data binding changed")
                master = _master_from_snapshot(
                    _frozen(
                        data,
                        ROOT / scaling.manifests[-1].file.path,
                    ).snapshot,
                )
                if master != names:
                    raise ValueError("source universe order changed")
                _validate_summary(
                    _frozen(data, ROOT / outputs["summary"].path),
                    outputs,
                )
                phases = _derive_phases(
                    scaling, master, csv, data,
                    _frozen(source, ROOT / CONTEXT_CONFIG.path),
                )

                launcher = Path(os.path.abspath(torch_python))
                runtimes = (
                    primary_python.resolve(strict=True),
                    launcher.resolve(strict=True),
                )
                support_paths = tuple(
                    ROOT / path for path in CONTEXT_SOURCE_PATHS
                ) + runtimes
                support_identities = _regular_inputs(support_paths)
                with freeze_inputs(support_paths) as frozen_support:
                    support = dict(zip(
                        support_paths, frozen_support, strict=True,
                    ))
                    primary = scaling.primary_python
                    torch = scaling.torch_probe
                    torch_argv = (torch.python.path, *PYTHON_FLAGS)
                    if Path(primary.path) != runtimes[0] or \
                       primary.sha256 != _frozen(
                        support, runtimes[0],
                    ).sha256 or \
                       Path(torch.python.path) != runtimes[1] or \
                       torch.python.sha256 != _frozen(
                           support, runtimes[1],
                       ).sha256:
                        raise ValueError("context runtime changed")
                    primary.validate_live("context primary Python")
                    torch.python.validate_live("context Torch Python")
                    if source_tree(
                        Path(torch.package_tree.root),
                    ) != torch.package_tree:
                        raise ValueError("context Torch package changed")
                    tree = _snapshot_tree(support)

                    def stable() -> None:
                        _verify_identities(source_identities)
                        _verify_identities(fetch_identities)
                        _verify_identities(data_identities)
                        _verify_identities(support_identities)
                        verify_frozen(frozen_source)
                        verify_frozen(frozen_fetch)
                        verify_frozen(frozen_data)
                        verify_frozen(frozen_support)
                        if selected_source_tree(
                            ROOT, CONTEXT_SOURCE_PATHS,
                        ) != tree:
                            raise ValueError(
                                "context input changed while bound",
                            )

                    def verify() -> None:
                        stable()
                        primary.validate_live("context primary Python")
                        torch.python.validate_live(
                            "context Torch Python",
                        )
                        if source_tree(
                            Path(torch.package_tree.root),
                        ) != torch.package_tree:
                            raise ValueError(
                                "context Torch package changed",
                            )

                    verify()
                    snapshots = ContextSnapshots(
                        _frozen(source, ROOT / CONTEXT_CONFIG.path),
                        _frozen(
                            data,
                            ROOT / scaling.manifests[-1].file.path,
                        ),
                        _frozen(
                            data, ROOT / scaling.session_calendar.path,
                        ),
                        tuple(
                            (name, _frozen(data, Path(binding.path)))
                            for name, binding in zip(
                                names, csv, strict=True,
                            )
                        ),
                    )
                    yield _BoundContext(
                        master, phases, primary, torch_argv,
                        torch, tree, snapshots, verify,
                    )
                    verify()


@contextmanager
def authenticate_context_attempt(
    attempt: ContextAttempt,
) -> Iterator[ContextLease]:
    """Hold one source-derived attempt lease through execution."""
    _require_isolated_execution()
    if not isinstance(attempt, ContextAttempt):
        raise ValueError("context attempt is invalid")
    with _bound_context(
        Path(attempt.primary_python.path),
        Path(attempt.torch_probe.python.path),
    ) as bound:
        _validate_commit(attempt.implementation_commit, bound.tree)
        if attempt.master != bound.master or \
           attempt.phases != bound.phases or \
           attempt.primary_python != bound.primary or \
           attempt.torch_argv != bound.torch_argv or \
           attempt.torch_probe != bound.torch or \
           attempt.source_tree != bound.tree:
            raise ValueError("context attempt is not source-derived")
        bound.verify()
        yield ContextLease(bound.snapshots, bound.verify)


def arm(
    output: Path, implementation_commit: str, run_id: str,
    primary_python: Path, torch_python: Path,
) -> ContextAttempt:
    """Atomically publish one exact development context attempt."""
    _require_isolated_execution()
    output = _relative(output, "context attempt output")
    if not COMMIT.fullmatch(implementation_commit) or \
       not re.fullmatch(r"[a-z0-9][a-z0-9-]*", run_id) or \
       output != Path(f"experiments/{run_id}-attempt.json"):
        raise ValueError("context attempt identity is invalid")
    run_dir = Path(f"reports/{run_id}")
    output_path, run_path = ROOT / output, ROOT / run_dir
    absent = (
        (output_path, "context attempt output"),
        (run_path, "context run directory"),
    )
    _fresh(absent)

    with _bound_context(primary_python, torch_python) as bound:
        _validate_commit(implementation_commit, bound.tree)
        value = {
            "attempt_path": output.as_posix(),
            "config": asdict(CONTEXT_CONFIG),
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": f"{run_dir}/.pycache",
            },
            "implementation_commit": implementation_commit,
            "phases": [
                context_phase_value(phase, bound.master)
                for phase in bound.phases
            ],
            "primary_python": asdict(bound.primary),
            "run_dir": run_dir.as_posix(),
            "run_id": run_id,
            "schema": 1,
            "source": {
                name: asdict(binding)
                for name, binding in SOURCE_EVIDENCE.items()
            },
            "source_tree": asdict(bound.tree),
            "status": "armed",
            "torch_argv": list(bound.torch_argv),
            "torch_probe": asdict(bound.torch),
        }
        constructed = _parse_constructed(value, output)
        parents = tuple(
            (path, _directory_identity(path))
            for path in (output_path.parent, run_path.parent)
        )
        output_fd, output_parent = _open_directory(output_path.parent)
        try:
            def verify() -> None:
                bound.verify()
                if any(
                    _directory_identity(path) != identity
                    for path, identity in parents
                ) or _directory_identity(
                    output_path.parent,
                ) != output_parent:
                    raise ValueError(
                        "context output parent changed while arming",
                    )
                _fresh(absent)

            write_json_exclusive(
                output_path, value, output_fd, verify,
            )
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        _absent(run_path, "context run directory")
        published = ContextAttempt.read(output_path, output, ROOT)
        if published != constructed:
            raise ValueError("published context attempt changed")
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
        "attempt": attempt.attempt_path,
        "run_id": attempt.run_id,
        "status": "armed",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
