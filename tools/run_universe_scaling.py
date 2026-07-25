#!/usr/bin/env python3
"""Run one frozen universe-scaling benchmark and finalize every terminal path."""

from __future__ import annotations

import sys

_BOOTSTRAP_PRIMARY_FLAGS = ("-I", "-S", "-B")
_BOOTSTRAP_TORCH_FLAGS = ("-I", "-S", "-B")
_BOOTSTRAP_CACHE_PREFIX: str | None = None


def _launch_flags() -> tuple[str, ...]:
    return (
        _BOOTSTRAP_TORCH_FLAGS
        if len(sys.argv) > 1 and sys.argv[1] == "calibrate" else
        _BOOTSTRAP_PRIMARY_FLAGS
    )


def _require_isolated_execution(*, bootstrapped: bool = False) -> None:
    flags = sys.flags
    if not flags.isolated or not getattr(flags, "safe_path", False) or \
            not flags.no_user_site or not flags.no_site or \
            not flags.dont_write_bytecode or \
            not flags.ignore_environment or not sys.dont_write_bytecode:
        raise ValueError(
            "runner requires isolated bytecode-free Python execution"
        )
    if bootstrapped and (
        _BOOTSTRAP_CACHE_PREFIX is None or
        sys.pycache_prefix != _BOOTSTRAP_CACHE_PREFIX
    ):
        raise ValueError("runner requires authenticated script bootstrap")


def _require_exact_launch(*, pristine: bool = False) -> None:
    if pristine and ("ctypes" in sys.modules or "_ctypes" in sys.modules):
        raise ValueError("runner launch inspection is already loaded")

    from ctypes import POINTER, byref, c_int, c_wchar_p, pythonapi
    import os

    argc = c_int()
    argv = POINTER(c_wchar_p)()
    get_argv = pythonapi.Py_GetArgcArgv
    get_argv.argtypes = (POINTER(c_int), POINTER(POINTER(c_wchar_p)))
    get_argv.restype = None
    get_argv(byref(argc), byref(argv))
    observed = tuple(argv[index] for index in range(argc.value))
    canonical = lambda values: (os.path.realpath(values[0]), *values[1:])
    expected = (
        os.path.realpath(sys.executable), *_launch_flags(), *sys.argv,
    )
    if not observed or canonical(observed) != expected or \
            canonical(tuple(sys.orig_argv)) != expected or \
            os.path.realpath(sys.argv[0]) != os.path.realpath(__file__):
        raise ValueError("runner requires the exact bound Python launch")


def _bootstrap_main() -> None:
    """Authenticate the import namespace before exposing repository code."""
    global _BOOTSTRAP_CACHE_PREFIX

    from importlib.machinery import PathFinder
    import os
    import stat
    import tempfile

    while True:
        prefix = os.path.join(
            tempfile.gettempdir(),
            f"compose-mini-scaling-runner-{os.urandom(32).hex()}",
        )
        if not os.path.lexists(prefix):
            break
    sys.pycache_prefix = prefix
    sys.dont_write_bytecode = True

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
    _BOOTSTRAP_CACHE_PREFIX = prefix


if __name__ == "__main__":
    _require_isolated_execution()
    _require_exact_launch(pristine=True)
    _bootstrap_main()

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
import argparse
import hashlib
import json
import os
import signal
import stat
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if _BOOTSTRAP_CACHE_PREFIX is None and str(ROOT) not in tuple(
    map(os.path.realpath, sys.path),
):
    sys.path.insert(0, str(ROOT))

from tools.files import (
    freeze_inputs, rename_may_have_committed, rename_noreplace,
    verify_frozen,
)
from tools.panel_contract import (
    _absent, _directory_identity, _open_directory, _regular_identity,
    _regular_inputs, _verify_identities, iter_canonical_json_lines,
    mkdir_nofollow, selected_source_tree, source_tree,
)
from tools.universe_scaling_contract import (
    EXPECTED_BUDGETS, EXPECTED_FIT_COUNT, EXPECTED_PREDICTION_RECORDS,
    EXPECTED_PREDICTION_VALUES, FINALIZER_PYTHON_FLAGS,
    FIXED_EPOCH_BUDGET, PHASES, RUNNER_PRIMARY_PYTHON_FLAGS,
    RUNNER_TORCH_PYTHON_FLAGS, FitJob, ScalingAttempt, ScalingCoverage,
    expected_fit_jobs, fit_provenance_id, question_uses,
    required_prediction_series,
)

SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
UNKNOWN_TIME = "1970-01-01T00:00:00Z"


class Interrupted(Exception):
    pass


@dataclass(frozen=True, slots=True)
class PreflightCounts:
    phases: tuple[int, ...]
    fits: int
    prediction_records: int
    prediction_values: int


EXPECTED_PREFLIGHT_COUNTS = PreflightCounts(
    (53, 52, 51), EXPECTED_FIT_COUNT,
    EXPECTED_PREDICTION_RECORDS, EXPECTED_PREDICTION_VALUES,
)


def phase_jobs(jobs: Sequence[FitJob], phase: str) -> tuple[FitJob, ...]:
    """Return one phase without changing the canonical fit order."""
    if phase not in PHASES:
        raise ValueError("scaling phase is invalid")
    return tuple(job for job in jobs if job.phase == phase)


def preflight_counts(coverage: ScalingCoverage) -> PreflightCounts:
    """Derive physical work solely from the armed coverage and schedule."""
    if not isinstance(coverage, ScalingCoverage):
        raise ValueError("scaling coverage is invalid")
    evaluable = {
        phase.phase: phase.evaluable for phase in coverage.phases
    }
    jobs = expected_fit_jobs(coverage.master, evaluable)
    records = values = 0
    rows = {
        (phase.phase, item.series): item.validation_rows
        for phase in coverage.phases for item in phase.series
    }
    for job in jobs:
        destinations = required_prediction_series(
            job, coverage.master, evaluable,
        )
        records += len(destinations)
        values += sum(rows[(job.phase, name)] for name in destinations)
    return PreflightCounts(
        tuple(len(phase.evaluable) for phase in coverage.phases),
        len(jobs), records, values,
    )


def _write_record(file: object, value: Mapping[str, object]) -> None:
    file.write(json.dumps(
        value, allow_nan=False, sort_keys=True,
    ) + "\n")


def merge_spools(
    schedule: Sequence[str], spools: Mapping[str, Path], output: Path,
) -> None:
    """Publish phase spools in one exact canonical record order."""
    phases = tuple(schedule)
    if not phases or set(phases) != set(spools) or \
            any(phase not in PHASES for phase in spools):
        raise ValueError("phase spool schedule is invalid")
    directory_fd, _ = _open_directory(output.parent)
    readers = {
        phase: iter(iter_canonical_json_lines(path))
        for phase, path in spools.items()
    }
    descriptor = None
    try:
        descriptor = os.open(
            output.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL |
            getattr(os, "O_NOFOLLOW", 0),
            0o600, dir_fd=directory_fd,
        )
        with os.fdopen(
            descriptor, "w", encoding="utf-8", closefd=False,
        ) as file:
            for phase in phases:
                try:
                    record = next(readers[phase])
                except StopIteration:
                    raise ValueError(
                        f"{phase} spool ended before its schedule"
                    ) from None
                _write_record(file, record)
            if any(next(reader, None) is not None for reader in readers.values()):
                raise ValueError("phase spool contains extra records")
            file.flush()
            os.fsync(file.fileno())
        metadata = os.fstat(descriptor)
        if _regular_identity(output) != (
            metadata.st_dev, metadata.st_ino,
        ):
            raise ValueError("published ledger identity changed")
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _entry_state(
    directory_fd: int, name: str,
) -> tuple[tuple[int, int], int, int] | None:
    try:
        metadata = os.stat(
            name, dir_fd=directory_fd, follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    return (
        (metadata.st_dev, metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
        metadata.st_nlink,
    )


def _publish_complete(
    source_fd: int, source: str, target_fd: int, target: str,
) -> None:
    """Rename one complete private inode without replacing a run output."""
    state = _entry_state(source_fd, source)
    if state is None or state[1:] != (stat.S_IFREG, 1) or \
            _entry_state(target_fd, target) is not None:
        raise ValueError("ledger publication endpoints are invalid")
    failure = None
    try:
        rename_noreplace(source_fd, source, target_fd, target)
    except OSError as error:
        failure = error
    committed = (
        rename_may_have_committed(failure) and
        _entry_state(source_fd, source) is None and
        _entry_state(target_fd, target) == state
    )
    if not committed:
        if failure is not None:
            raise failure
        raise OSError("exclusive ledger publication changed")
    os.fsync(target_fd)


def _attempt_path(path: Path) -> tuple[Path, Path]:
    absolute = path if path.is_absolute() else ROOT / path
    try:
        lexical = Path(os.path.abspath(absolute))
        resolved = absolute.resolve(strict=True)
        if lexical != resolved:
            raise ValueError("attempt path must not contain symlinks")
        _regular_identity(resolved)
        logical = resolved.relative_to(ROOT.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError("attempt must be inside the repository") from error
    return absolute, logical


def read_attempt(path: Path) -> ScalingAttempt:
    absolute, logical = _attempt_path(path)
    return ScalingAttempt.read(absolute, logical, ROOT)


def _output_path(attempt: ScalingAttempt, name: str) -> Path:
    return ROOT / attempt.outputs[name]


def _validate_source(attempt: ScalingAttempt) -> None:
    if selected_source_tree(
        ROOT, tuple(item.path for item in attempt.source_tree.files),
    ) != attempt.source_tree or selected_source_tree(
        ROOT, tuple(item.path for item in attempt.finalizer_tree.files),
    ) != attempt.finalizer_tree:
        raise ValueError("scaling source closure changed")


def _validate_environment(attempt: ScalingAttempt) -> None:
    actual = dict(os.environ)
    expected = dict(attempt.environment)
    if any(
        actual.get(name) != value for name, value in expected.items()
    ) or set(actual) - set(expected) - {
        "LC_CTYPE", "__CF_USER_TEXT_ENCODING",
    }:
        raise ValueError("runner environment does not match the attempt")


def _expose_torch_package(package: Path) -> None:
    """Insert one attested Torch parent ahead of repository shadows."""
    from importlib.machinery import PathFinder

    if any(
        name == "torch" or name.startswith("torch.") for name in sys.modules
    ):
        raise ValueError("Torch namespace was loaded before validation")
    package = package.resolve(strict=True)
    initializer = package / "__init__.py"
    if not stat.S_ISREG(
        initializer.stat(follow_symlinks=False).st_mode,
    ):
        raise ValueError("Torch package initializer is unsafe")
    parent = str(package.parent)
    root = str(ROOT.resolve(strict=True))
    resolved = tuple(map(os.path.realpath, sys.path))
    if resolved.count(root) != 1 or resolved[-1] != root or \
            parent in resolved:
        raise ValueError("runner import path changed before Torch validation")
    index = len(sys.path) - 1
    search = (*sys.path[:index], parent)
    spec = PathFinder.find_spec("torch", search)
    locations = tuple(
        map(os.path.realpath, spec.submodule_search_locations or ())
    ) if spec is not None else ()
    if spec is None or os.path.realpath(spec.origin or "") != \
            str(initializer) or locations != (str(package),):
        raise ValueError("Torch package resolver is unsafe")
    sys.path[index] = parent


def _validate_runtime(attempt: ScalingAttempt, stage: str) -> None:
    expected = (
        Path(attempt.torch_probe.python.path)
        if stage == "calibrate" else
        Path(attempt.primary_python.path)
    )
    if Path(sys.executable).resolve(strict=True) != \
            expected.resolve(strict=True):
        raise ValueError(f"{stage} is not running under its bound Python")
    if stage == "calibrate":
        attempt.torch_probe.python.validate_live("Torch Python")
        package = Path(
            attempt.torch_probe.package_tree.root,
        ).resolve(strict=True)
        if source_tree(package) != attempt.torch_probe.package_tree:
            raise ValueError("Torch package closure changed")
    else:
        attempt.primary_python.validate_live("primary Python")


def _validate_stage_paths(attempt: ScalingAttempt, stage: str) -> None:
    run_dir = ROOT / attempt.run_dir
    cache = ROOT / attempt.environment["PYTHONPYCACHEPREFIX"]
    outputs = {
        name: _output_path(attempt, name)
        for name in ("fits", "predictions", "summary", "outcome")
    }
    _absent(cache, "Python cache prefix")
    _absent(outputs["outcome"], "attempt outcome")
    if stage in ("validate", "preflight"):
        _absent(run_dir, "run directory")
        for name in ("fits", "predictions", "summary"):
            _absent(outputs[name], f"{name} output")
        return
    _directory_identity(run_dir)
    if stage == "calibrate":
        if tuple(run_dir.iterdir()):
            raise ValueError("calibration run directory must be empty")
        return
    if stage != "analyze":
        raise ValueError("scaling stage is invalid")
    _regular_identity(outputs["fits"])
    _regular_identity(outputs["predictions"])
    _absent(outputs["summary"], "summary output")
    if {path.name for path in run_dir.iterdir()} != {
        outputs["fits"].name, outputs["predictions"].name,
    }:
        raise ValueError("analysis run directory members are invalid")


def _validate_stage(attempt: ScalingAttempt, stage: str) -> None:
    _require_isolated_execution(bootstrapped=True)
    _require_exact_launch()
    if stage not in attempt.commands or \
            tuple(sys.argv) != attempt.commands[stage]:
        raise ValueError("stage arguments do not match the armed attempt")
    _validate_environment(attempt)
    _validate_runtime(attempt, stage)
    _validate_source(attempt)
    if stage == "calibrate":
        _expose_torch_package(Path(
            attempt.torch_probe.package_tree.root,
        ))
    _validate_stage_paths(attempt, stage)


@contextmanager
def _validated_inputs(
    attempt: ScalingAttempt,
) -> Iterator[tuple[tuple[str, ...], Mapping[Path, object], object]]:
    from tools.finalize_universe_scaling import (
        _success_inputs, _validate_live_success, _validate_success_inputs,
    )

    success = _success_inputs(attempt)
    identities = _regular_inputs(success.paths)
    with freeze_inputs(success.paths) as frozen:
        by_path = dict(zip(success.paths, frozen, strict=True))
        master = _validate_success_inputs(attempt, success, by_path)
        _validate_live_success(attempt, success)
        yield master, by_path, success
        _verify_identities(identities)
        verify_frozen(frozen)
        _validate_live_success(attempt, success)


def _derive_coverage(
    attempt: ScalingAttempt, by_path: Mapping[Path, object],
) -> ScalingCoverage:
    from tools.analyze_universe import read_json
    from tools.arm_universe_scaling import _validate_data

    report = by_path[ROOT / attempt.fetch_report.path]
    coverage = _validate_data(
        by_path, read_json(report.snapshot, canonical=True),
    )
    if coverage != attempt.coverage:
        raise ValueError("live scaling coverage changed")
    return coverage


def preflight(attempt: ScalingAttempt) -> PreflightCounts:
    """Prove the full development closure without creating output or Torch."""
    with _validated_inputs(attempt) as (master, by_path, _):
        coverage = _derive_coverage(attempt, by_path)
        if master != coverage.master:
            raise ValueError("manifest and coverage series disagree")
        counts = preflight_counts(coverage)
        if counts != EXPECTED_PREFLIGHT_COUNTS:
            raise ValueError("scaling physical work closure changed")
        return counts


def _fingerprint_tensor(
    digest: object, name: str, tensor: object,
) -> None:
    values = tensor.detach().reshape(-1)
    metadata = json.dumps({
        "dtype": str(tensor.dtype),
        "name": name,
        "shape": list(tensor.shape),
    }, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    digest.update(metadata + b"\0")
    for start in range(0, values.numel(), 16_384):
        chunk = values[start:start + 16_384].to(
            device="cpu",
        ).float().tolist()
        digest.update(struct.pack(f"<{len(chunk)}f", *chunk))


def model_fingerprint(
    model: object, job: FitJob, candidate: object,
    members: Sequence[tuple[str, object]], runtime_sha256: str,
) -> str:
    """Bind one fitted model to its candidate, members, scalers, and state."""
    if len(runtime_sha256) != 64 or any(
        byte not in "0123456789abcdef" for byte in runtime_sha256
    ):
        raise ValueError("model runtime identity is invalid")
    digest = hashlib.sha256(json.dumps({
        "candidate": asdict(candidate),
        "fit_provenance_id": fit_provenance_id(job),
        "members": [name for name, _ in members],
        "model": job.model,
        "runtime_sha256": runtime_sha256,
    }, allow_nan=False, separators=(",", ":"), sort_keys=True).encode())
    for name, data in members:
        for field in (
            "feature_mean", "feature_scale", "target_mean", "target_scale",
        ):
            _fingerprint_tensor(
                digest, f"member:{name}:{field}", getattr(data, field),
            )
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in sorted(values):
            _fingerprint_tensor(digest, f"{kind}:{name}", tensor)
    return digest.hexdigest()


def _runtime_sha256(attempt: ScalingAttempt) -> str:
    encoded = json.dumps(
        asdict(attempt.torch_probe), allow_nan=False,
        separators=(",", ":"), sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fit_record(
    job: FitJob, fingerprint: str, fit: object | None,
    coverage: Mapping[tuple[str, str], tuple[int, int]],
    master: tuple[str, ...],
) -> dict[str, object]:
    fixed = job.kind == "pooled" and job.mode == "fixed-update"
    epochs = job.kind == "local" or job.mode == "fixed-epoch"
    rows = [
        {
            "series": name,
            "train_rows": coverage[(job.phase, name)][0],
            "validation_rows": coverage[(job.phase, name)][1],
        }
        for name in job.members
    ]
    trained = fit.epochs_trained if epochs else 0
    rows_per_epoch = (
        sum(item["train_rows"] for item in rows) +
        FIXED_EPOCH_BUDGET.batch_size - 1
    ) // FIXED_EPOCH_BUDGET.batch_size if epochs else 0
    updates = (
        dict(EXPECTED_BUDGETS)[job.phase].total_updates if fixed else
        trained * rows_per_epoch
    )
    return {
        "budget": (
            asdict(dict(EXPECTED_BUDGETS)[job.phase]) if fixed else
            asdict(FIXED_EPOCH_BUDGET) if epochs else None
        ),
        "cohort": job.cohort,
        "coverage": rows,
        "epochs_trained": trained,
        "kind": job.kind,
        "members": list(job.members),
        "mode": job.mode,
        "model": job.model,
        "model_fingerprint": fingerprint,
        "optimizer_updates": updates,
        "phase": job.phase,
        "provenance_id": fit_provenance_id(job),
        "question_uses": [
            {"question": question, "cohort": cohort}
            for question, cohort in question_uses(job, master)
        ],
        "schema": 1,
        "seed": job.seed,
        "selected_checkpoint": fit.best_checkpoint if fixed else None,
        "selected_epoch": fit.best_epoch if epochs else None,
    }


def _fit(
    job: FitJob, data: Mapping[str, object], candidate: object,
    sweep: object, device: object, experiment: ModuleType,
) -> tuple[object, object | None]:
    members = tuple(data[name] for name in job.members)
    if job.kind == "ridge":
        return experiment.stock_macro_linear_model(
            members, candidate.ridge,
        ).to(device), None
    if job.kind == "local":
        model, fit, _ = experiment._fit_neural(
            "transformer", candidate, members[0], sweep, job.seed, device,
        )
        return model, fit

    validation = tuple(
        index for index, member in enumerate(members)
        if len(member.validation)
    )
    runtime_model = "mlp" if job.model == "global_mlp" else job.model
    if job.mode == "fixed-update":
        budget = dict(EXPECTED_BUDGETS)[job.phase]
        model, fit = experiment._fit_shared_updates(
            runtime_model, candidate, members, sweep, job.seed,
            budget.updates_per_checkpoint, device,
            validation_indices=validation,
        )
        if fit.updates_trained != budget.total_updates:
            raise ValueError("fixed-update fit did not exhaust its budget")
        return model, fit
    model, fit, _ = experiment._fit_shared_epochs(
        runtime_model, candidate, members, sweep, job.seed, device,
        validation_indices=validation,
    )
    return model, fit


def _predictions(
    job: FitJob, model: object, data: Mapping[str, object],
    candidate: object, sweep: object, device: object, experiment: ModuleType,
    master: tuple[str, ...], evaluable: Mapping[str, Sequence[str]],
    grids: Mapping[tuple[str, str], str],
    fingerprint: str, runtime_sha256: str,
) -> Iterator[dict[str, object]]:
    from tools.float32 import encode_f32le_base64
    from tools.train import data_loaders, evaluate

    provenance = fit_provenance_id(job)
    for name in required_prediction_series(job, master, evaluable):
        target = data[name]
        evaluation = (
            experiment._conditioned(
                target, job.members.index(name),
            ) if job.model == "conditioned_panel_transformer" else target
        )
        loader = data_loaders(
            evaluation, sweep.batch_size, job.seed or 0,
        )[1]
        values: list[float] = []
        evaluate(
            model, loader, target.target_mean, target.target_scale,
            device, values,
        )
        if len(values) != len(target.validation):
            raise ValueError("prediction count changed during evaluation")
        yield {
            "grid_sha256": grids[(job.phase, name)],
            "model_fingerprint": fingerprint,
            "phase": job.phase,
            "predictions": encode_f32le_base64(values),
            "provenance_id": provenance,
            "schema": 2,
            "series": name,
        }
    if model_fingerprint(
        model, job, candidate, tuple(
            (name, data[name]) for name in job.members
        ), runtime_sha256,
    ) != fingerprint:
        raise ValueError("model changed while predictions were emitted")


def _phase_data(
    attempt: ScalingAttempt, phase: str, master: tuple[str, ...],
    by_path: Mapping[Path, object],
) -> tuple[object, object, Mapping[str, object]]:
    import torch

    from tools.data_v1 import read_bars
    from tools.experiment import Sweep, _prepare_packed
    from tools.fetch_universe import UniverseManifest
    from tools.session_calendar import SessionCalendar
    from tools.session_samples import session_samples
    from tools.universe_contract import common_calendar, pack_rows

    sweep = Sweep.read(
        by_path[ROOT / attempt.config.path].snapshot,
    )
    if len(sweep.candidates) != 1:
        raise ValueError("scaling sweep must contain one candidate")
    candidate = sweep.candidates[0]
    manifest = UniverseManifest.read(
        by_path[ROOT / attempt.manifests[-1].file.path].snapshot,
    )
    calendar = SessionCalendar.read(
        by_path[ROOT / attempt.session_calendar.path].snapshot,
    )
    protocol = attempt.protocol
    calendar_contract = protocol["calendar"]
    history = protocol["history_bars"]
    horizon = protocol["target_horizon_bars"]
    alignment = protocol["alignment_horizon_bars"]
    blocks = common_calendar(
        calendar_contract["opportunities"], protocol["folds"],
        protocol["fold_fraction"], alignment - 1,
    )
    observed = (
        tuple(tuple((item.start, item.stop) for item in fold)
              for fold in blocks.folds),
        tuple((item.start, item.stop) for item in blocks.holdout[:2]),
        (blocks.holdout[-1].start, blocks.holdout[-1].stop),
    )
    expected_blocks = (
        tuple(tuple(tuple(item) for item in fold)
              for fold in calendar_contract["folds"]),
        tuple(tuple(item) for item in calendar_contract["calibration"]),
        tuple(calendar_contract["reserved_test"]),
    )
    if observed != expected_blocks:
        raise ValueError("development calendar contract changed")
    ranges = dict(zip(
        PHASES, (*blocks.folds, blocks.holdout[:2]), strict=True,
    ))[phase]
    stop = expected_blocks[1][-1][1]

    from tools.finalize_universe_scaling import _fetch_bindings

    names, csv = _fetch_bindings(
        by_path[ROOT / attempt.fetch_report.path].snapshot,
    )
    if names != master:
        raise ValueError("fetch and manifest order changed")
    prepared = {}
    expected = {
        item.series: (item.train_rows, item.validation_rows)
        for item in next(
            item for item in attempt.coverage.phases
            if item.phase == phase
        ).series
    }
    for name, binding in zip(names, csv, strict=True):
        timestamps, rows = read_bars(
            by_path[Path(binding.path)].snapshot,
        )
        samples = session_samples(
            timestamps, manifest.interval_minutes, calendar,
            manifest.start, manifest.end, history, horizon, alignment,
            opportunity_stop=stop,
        )
        packed = pack_rows(
            samples.rows, ranges, history, horizon, alignment,
        )
        if packed.counts != expected[name]:
            raise ValueError(f"{name} {phase} prepared coverage changed")
        member = _prepare_packed(
            rows, candidate, packed, history, sweep,
        )
        if len(member.train) != packed.counts[0] or \
                len(member.validation) != packed.counts[1] or \
                len(member.test):
            raise ValueError("prepared scaling split changed")
        prepared[name] = member
    torch.use_deterministic_algorithms(True)
    return candidate, sweep, prepared


def calibrate(attempt: ScalingAttempt) -> None:
    """Fit every physical job once and publish compact canonical ledgers."""
    import torch
    from tools import experiment

    run_dir = ROOT / attempt.run_dir
    run_fd, run_identity = _open_directory(run_dir)
    try:
        with _validated_inputs(attempt) as (master, by_path, _):
            coverage = _derive_coverage(attempt, by_path)
            counts = preflight_counts(coverage)
            if counts != EXPECTED_PREFLIGHT_COUNTS:
                raise ValueError("scaling physical work closure changed")
            evaluable = {
                phase.phase: phase.evaluable
                for phase in coverage.phases
            }
            rows = {
                (phase.phase, item.series): (
                    item.train_rows, item.validation_rows,
                )
                for phase in coverage.phases for item in phase.series
            }
            grids = {
                (phase.phase, item.series): item.timestamp_sha256
                for phase in coverage.phases for item in phase.series
            }
            jobs = expected_fit_jobs(master, evaluable)
            device = torch.device("cpu")
            runtime_sha256 = _runtime_sha256(attempt)
            with tempfile.TemporaryDirectory(
                prefix=f".{attempt.run_id}-spool-",
                dir=run_dir.parent,
            ) as spool_name:
                spool = Path(spool_name).resolve(strict=True)
                fit_spools = {
                    phase: spool / f"{phase}-fits.jsonl"
                    for phase in PHASES
                }
                prediction_spools = {
                    phase: spool / f"{phase}-predictions.jsonl"
                    for phase in PHASES
                }
                for phase in PHASES:
                    candidate, sweep, data = _phase_data(
                        attempt, phase, master, by_path,
                    )
                    with fit_spools[phase].open(
                        "x", encoding="utf-8",
                    ) as fit_file, prediction_spools[phase].open(
                        "x", encoding="utf-8",
                    ) as prediction_file:
                        for job in phase_jobs(jobs, phase):
                            model, fit = _fit(
                                job, data, candidate, sweep, device,
                                experiment,
                            )
                            fingerprint = model_fingerprint(
                                model, job, candidate, tuple(
                                    (name, data[name])
                                    for name in job.members
                                ), runtime_sha256,
                            )
                            _write_record(
                                fit_file,
                                _fit_record(
                                    job, fingerprint, fit, rows, master,
                                ),
                            )
                            for record in _predictions(
                                job, model, data, candidate, sweep,
                                device, experiment, master, evaluable, grids,
                                fingerprint, runtime_sha256,
                            ):
                                if record["model_fingerprint"] != fingerprint:
                                    raise ValueError(
                                        "fit and prediction fingerprints differ"
                                    )
                                _write_record(prediction_file, record)
                    del data
                if _directory_identity(run_dir) != run_identity or \
                        tuple(run_dir.iterdir()):
                    raise ValueError("run directory changed before publication")
                merged_fits = spool / "fits.jsonl"
                merged_predictions = spool / "predictions.jsonl"
                merge_spools(
                    tuple(job.phase for job in jobs), fit_spools,
                    merged_fits,
                )
                prediction_schedule = tuple(
                    job.phase
                    for job in jobs
                    for _ in required_prediction_series(
                        job, master, evaluable,
                    )
                )
                if len(prediction_schedule) != \
                        EXPECTED_PREFLIGHT_COUNTS.prediction_records:
                    raise ValueError("published prediction count changed")
                merge_spools(
                    prediction_schedule, prediction_spools,
                    merged_predictions,
                )
                spool_fd, spool_identity = _open_directory(spool)
                try:
                    if spool_identity[0] != run_identity[0]:
                        raise ValueError(
                            "private spool is not on the run filesystem"
                        )
                    _publish_complete(
                        spool_fd, merged_fits.name, run_fd,
                        _output_path(attempt, "fits").name,
                    )
                    _publish_complete(
                        spool_fd, merged_predictions.name, run_fd,
                        _output_path(attempt, "predictions").name,
                    )
                finally:
                    os.close(spool_fd)
                if _directory_identity(run_dir) != run_identity or {
                    path.name for path in run_dir.iterdir()
                } != {
                    _output_path(attempt, "fits").name,
                    _output_path(attempt, "predictions").name,
                }:
                    raise ValueError(
                        "run directory changed during publication"
                    )
                _regular_identity(_output_path(attempt, "fits"))
                _regular_identity(_output_path(attempt, "predictions"))
    finally:
        os.close(run_fd)


def analyze(attempt: ScalingAttempt) -> int:
    """Revalidate the ledgers and return only the frozen development gate code."""
    from tools.finalize_universe_scaling import (
        _fetch_bindings, _prediction_line_bytes, analyze_ledgers,
        derive_market_truth,
    )
    from tools.panel_contract import (
        iter_canonical_json_lines, read_canonical_json_lines,
    )

    with _validated_inputs(attempt) as (master, by_path, _):
        coverage = _derive_coverage(attempt, by_path)
        names, csv = _fetch_bindings(
            by_path[ROOT / attempt.fetch_report.path].snapshot,
        )
        if names != master:
            raise ValueError("analysis fetch order changed")
        truth = derive_market_truth(
            by_path[ROOT / attempt.manifests[-1].file.path].snapshot,
            by_path[ROOT / attempt.session_calendar.path].snapshot,
            {
                name: by_path[Path(binding.path)].snapshot
                for name, binding in zip(names, csv, strict=True)
            },
            coverage, attempt.protocol,
        )
        outputs = tuple(
            _output_path(attempt, name)
            for name in ("fits", "predictions")
        )
        identities = _regular_inputs(outputs)
        with freeze_inputs(outputs) as frozen:
            result = analyze_ledgers(
                read_canonical_json_lines(frozen[0].snapshot),
                iter_canonical_json_lines(
                    frozen[1].snapshot,
                    max_line_bytes=_prediction_line_bytes(truth),
                ),
                master, coverage, truth,
            )
            _verify_identities(identities)
            verify_frozen(frozen)
    return {"pass": 0, "gate-failure": 3}[result["status"]]


def _interrupt(signum: int, _frame: object) -> None:
    signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    raise Interrupted(signum)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    reaped = False
    try:
        process.wait(timeout=2)
        reaped = True
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        if not reaped:
            process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if not reaped:
        process.wait()


def _run(command: Sequence[object], environment: Mapping[str, str]) -> int:
    process = subprocess.Popen(
        [str(item) for item in command], cwd=ROOT,
        env=dict(environment), start_new_session=True,
    )
    try:
        code = process.wait()
    except BaseException:
        _terminate(process)
        raise
    return 128 - code if code < 0 else min(code, 255)


def _finalizer(
    attempt: ScalingAttempt, started: str, ended: str,
    stage: str, code: int, status: str,
) -> tuple[object, ...]:
    flags = tuple(attempt.protocol["finalizer_python_flags"])
    if flags != FINALIZER_PYTHON_FLAGS:
        raise ValueError("finalizer Python flags changed")
    return (
        attempt.primary_python.path, *flags,
        *attempt.commands["finalizer_prefix"],
        "--started", started, "--ended", ended,
        "--stage", stage, "--exit", str(code), "--status", status,
    )


def _runner_flags(
    attempt: ScalingAttempt, stage: str,
) -> tuple[str, ...]:
    torch_stage = stage == "calibrate"
    key = (
        "runner_torch_python_flags" if torch_stage else
        "runner_primary_python_flags"
    )
    expected = (
        RUNNER_TORCH_PYTHON_FLAGS if torch_stage else
        RUNNER_PRIMARY_PYTHON_FLAGS
    )
    flags = tuple(attempt.protocol[key])
    if flags != expected:
        raise ValueError(f"{stage} Python flags changed")
    return flags


def _validate_orchestrator(attempt: ScalingAttempt) -> None:
    _require_isolated_execution(bootstrapped=True)
    _require_exact_launch()
    if tuple(sys.argv) != (
        attempt.commands["validate"][0], attempt.attempt_path,
    ):
        raise ValueError("orchestrator arguments do not match the attempt")
    if Path(sys.executable).resolve(strict=True) != Path(
        attempt.primary_python.path,
    ).resolve(strict=True):
        raise ValueError("orchestrator requires the bound primary Python")
    attempt.primary_python.validate_live("primary Python")
    _validate_environment(attempt)
    _validate_source(attempt)
    _validate_stage_paths(attempt, "validate")


def _drain_pending(
    interruptions: list[int], previous_mask: set[signal.Signals],
) -> None:
    eligible = set(SIGNALS) - set(previous_mask)
    pending = set(signal.sigpending()).intersection(eligible)
    while pending:
        interruptions.append(int(signal.sigwait(pending)))
        pending = set(signal.sigpending()).intersection(eligible)


def execute(attempt_path: Path) -> int:
    """Run all bound stages and invoke the finalizer exactly once."""
    attempt = read_attempt(attempt_path)
    _validate_orchestrator(attempt)
    environment = dict(attempt.environment)
    stage, code, status = "preflight", 1, "preflight-failure"
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    started = UNKNOWN_TIME
    handlers: dict[int, object] = {}
    interruptions: list[int] = []
    finalizing = False

    def interrupt(signum: int, frame: object) -> None:
        interruptions.append(signum)
        if not finalizing:
            _interrupt(signum, frame)

    finalizer_exit = 2
    try:
        try:
            for item in SIGNALS:
                handlers[item] = signal.getsignal(item)
            started = _utc_now()
            for item in SIGNALS:
                signal.signal(item, interrupt)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            for name in ("validate", "preflight"):
                code = _run(
                    (
                        attempt.primary_python.path,
                        *_runner_flags(attempt, name),
                        *attempt.commands[name],
                    ),
                    environment,
                )
                if code:
                    break
            if code == 0:
                stage, status = "setup", "setup-failure"
                try:
                    mkdir_nofollow(ROOT / attempt.run_dir)
                except (OSError, ValueError) as error:
                    print(f"setup error: {error}", file=sys.stderr)
                    code = 1
            if stage == "setup" and code == 0:
                stage, status = "experiment", "experiment-failure"
                code = _run(
                    (
                        *attempt.torch_argv,
                        *_runner_flags(attempt, "calibrate"),
                        *attempt.commands["calibrate"],
                    ),
                    environment,
                )
            if stage == "experiment" and code == 0:
                stage, status = "analysis", "analysis-integrity-failure"
                code = _run(
                    (
                        attempt.primary_python.path,
                        *_runner_flags(attempt, "analyze"),
                        *attempt.commands["analyze"],
                    ),
                    environment,
                )
                status = {
                    0: "pass", 3: "gate-failure",
                }.get(code, "analysis-integrity-failure")
            signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        except Interrupted as error:
            code = 128 + int(error.args[0])
            if stage == "analysis":
                status = "analysis-integrity-failure"
        except Exception as error:
            code = 2 if stage == "analysis" else 1
            if stage == "analysis":
                status = "analysis-integrity-failure"
            print(f"{stage} error: {error}", file=sys.stderr)
    except Interrupted:
        pass
    finally:
        while True:
            try:
                signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
                break
            except Interrupted:
                continue
        if interruptions:
            code = 128 + interruptions[-1]
            if stage == "analysis":
                status = "analysis-integrity-failure"
        try:
            try:
                ended = _utc_now()
            except Exception as error:
                print(f"timestamp error: {error}", file=sys.stderr)
                ended = started
            finalizing = True
            try:
                signal.pthread_sigmask(
                    signal.SIG_SETMASK, previous_mask,
                )
            except Exception as error:
                print(f"signal unmask error: {error}", file=sys.stderr)
            try:
                finalizer_exit = _run(
                    _finalizer(
                        attempt, started, ended, stage, code, status,
                    ),
                    environment,
                )
            except Exception as error:
                print(f"finalizer error: {error}", file=sys.stderr)
        except Exception as error:
            print(f"finalizer error: {error}", file=sys.stderr)
        finally:
            while True:
                try:
                    signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
                    break
                except Interrupted:
                    continue
            _drain_pending(interruptions, previous_mask)
            for item, handler in handlers.items():
                signal.signal(item, handler)
            finalizing = False
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    return (
        128 + interruptions[-1] if interruptions else
        finalizer_exit or code
    )


def _stage(stage: str, attempt_path: Path) -> int:
    attempt = read_attempt(attempt_path)
    _validate_stage(attempt, stage)
    if stage == "preflight":
        preflight(attempt)
    elif stage == "calibrate":
        calibrate(attempt)
    elif stage == "analyze":
        return analyze(attempt)
    return 0


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command_or_attempt")
    parser.add_argument("attempt", nargs="?")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        code = (
            _stage(args.command_or_attempt, Path(args.attempt))
            if args.attempt is not None else
            execute(Path(args.command_or_attempt))
        )
    except (
        KeyError, OSError, OverflowError, TypeError, UnicodeError, ValueError,
    ) as error:
        print(f"integrity error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    raise SystemExit(code)


if __name__ == "__main__":
    main()
