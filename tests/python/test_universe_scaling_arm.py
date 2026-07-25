#!/usr/bin/env python3
"""Verify universe-scaling arming rejects filesystem races and aliases."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import sys
import tempfile
from unittest.mock import call, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import arm_universe_scaling as armer
from tools import universe_scaling_inputs as inputs
from tools.files import file_sha256
from tools.panel_contract import (
    FileBinding, TorchIdentity, executable_binding, selected_source_tree,
)
from tools.universe_scaling_contract import (
    EXPECTED_BUDGETS, EXPECTED_MISSING, EXPECTED_PREDICTION_RECORDS,
    MODES, PHASES, SEEDS, FitJob, PhaseCoverage, ScalingCoverage,
    SeriesCoverage, TreeBinding, expected_fit_jobs, expected_protocol,
    question_uses, required_prediction_series, timestamp_grid_sha256,
)
from tools.universe_scaling_inputs import ScalingSeries


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def phase_train_rows(phase: str, index: int) -> int:
    budget = dict(EXPECTED_BUDGETS)[phase].control_samples
    quotient, remainder = divmod(budget, 11)
    return quotient + (index < remainder) if index < 11 else 1


def synthetic_master() -> tuple[str, ...]:
    names = [f"S{index:02d}" for index in range(55)]
    for index, name in zip(
        (14, 28, 32, 39), ("ALTR", "ZI", "FYBR", "INFA"), strict=True,
    ):
        names[index] = name
    return tuple(names)


def synthetic_coverage(names: tuple[str, ...]) -> ScalingCoverage:
    missing = {
        "fold-0": {14, 28},
        "fold-1": {14, 28, 39},
        "calibration": {14, 28, 32, 39},
    }
    phases = tuple(
        PhaseCoverage(phase, tuple(
            SeriesCoverage(
                name, phase_train_rows(phase, index),
                int(index not in missing[phase]),
                timestamp_grid_sha256((
                    (f"{phase}:{name}:as-of",
                     f"{phase}:{name}:entry",
                     f"{phase}:{name}:target"),
                )) if index not in missing[phase] else
                timestamp_grid_sha256(()),
            )
            for index, name in enumerate(names)
        ))
        for phase in PHASES
    )
    return ScalingCoverage(phases)


def coverage_value(coverage: ScalingCoverage) -> list[dict[str, object]]:
    return [
        {
            "phase": phase.phase,
            "series": [
                {
                    "series": item.series,
                    "train_rows": item.train_rows,
                    "validation_rows": item.validation_rows,
                    "timestamp_sha256": item.timestamp_sha256,
                }
                for item in phase.series
            ],
        }
        for phase in coverage.phases
    ]


def with_validation(
    coverage: ScalingCoverage, phase: str, index: int, rows: int,
) -> ScalingCoverage:
    phases = list(coverage.phases)
    phase_index = PHASES.index(phase)
    records = list(phases[phase_index].series)
    records[index] = replace(
        records[index], validation_rows=rows,
        timestamp_sha256=timestamp_grid_sha256((
            (f"{phase}:{records[index].series}:as-of",
             f"{phase}:{records[index].series}:entry",
             f"{phase}:{records[index].series}:target"),
        )) if rows else timestamp_grid_sha256(()),
    )
    phases[phase_index] = PhaseCoverage(phase, tuple(records))
    return ScalingCoverage(tuple(phases))


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

        self.names = tuple(f"S{index:02d}" for index in range(55))
        self.coverage = synthetic_coverage(self.names)
        self.timestamps = (("2026-07-24T14:30:00Z",),) * len(self.names)
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
        write_json(self.root / self.calendar_path, {})
        write_json(self.root / self.config_path, {})

        manifest_dir = self.root / self.selection_root / "manifests"
        manifest_dir.mkdir()
        self.manifest_paths = {
            size: write(
                manifest_dir / f"liquid-common-{size}.json",
                f"{size}\n",
            )
            for size in (11, 22, 33)
        }
        self.manifest_paths[55] = write(
            self.root / "inputs/overlay-55.json", "55\n",
        )
        self.manifest_bindings = {
            size: FileBinding(
                path.relative_to(self.root).as_posix(), file_sha256(path),
            )
            for size, path in self.manifest_paths.items()
        }
        write_json(self.root / self.fetch_path, {
            "fetch_schema": 4,
            "manifest": {
                "path": self.manifest_bindings[55].path,
                "sha256": self.manifest_bindings[55].sha256,
            },
            "session_calendar": {
                "path": self.calendar_path.as_posix(),
                "sha256": file_sha256(self.root / self.calendar_path),
            },
        })
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
        self.selection_members = tuple(
            self.manifest_paths[size] for size in (11, 22, 33)
        )

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
                MANIFEST_BINDINGS=self.manifest_bindings,
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
            read_timestamps = stack.enter_context(patch.object(
                armer, "read_timestamps", side_effect=self.timestamps,
            ))
            validate_fetch = stack.enter_context(patch.object(
                armer, "validate_fetch", return_value=None,
            ))
            coverage = stack.enter_context(patch.object(
                armer, "common_coverage",
                return_value=with_validation(
                    self.coverage, "calibration", 49, 0,
                ),
            ))
            if not validate_data:
                stack.enter_context(patch.object(
                    armer, "_validate_data", return_value=self.coverage,
                ))
            yield SimpleNamespace(
                coverage=coverage, observe_torch=observe_torch,
                read_attempts=reads, read_json=read_json,
                read_timestamps=read_timestamps,
                selection_binding=selection_binding,
                validate_fetch=validate_fetch,
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


def verify_serializes_coverage(root: Path) -> None:
    fixture = ArmFixture(root)
    with fixture.patched():
        fixture.arm()
    value = json.loads(fixture.output_path.read_text(encoding="utf-8"))
    assert value["coverage"] == coverage_value(fixture.coverage)


def verify_torch_venv_launcher(root: Path) -> None:
    fixture = ArmFixture(root)
    launcher = fixture.torch.with_name("torch-venv")
    launcher.symlink_to(fixture.torch)
    with fixture.patched() as probes:
        armer.arm(
            fixture.output, "1" * 40, fixture.run_id,
            fixture.primary, launcher,
        )
    assert probes.observe_torch.call_args_list == [
        call((str(launcher),), fixture.root),
        call((str(launcher),), fixture.root),
    ]
    value = json.loads(fixture.output_path.read_text(encoding="utf-8"))
    assert value["torch_argv"] == [str(fixture.torch)]


def verify_shared_torch_runtime(root: Path) -> None:
    fixture = ArmFixture(root)
    launcher = fixture.primary.with_name("torch-venv")
    launcher.symlink_to(fixture.primary)
    fixture.torch_probe = replace(
        fixture.torch_probe,
        python=executable_binding(fixture.primary, "synthetic torch"),
    )
    with fixture.patched() as probes:
        armer.arm(
            fixture.output, "1" * 40, fixture.run_id,
            fixture.primary, launcher,
        )
    probes.observe_torch.assert_has_calls([
        call((str(launcher),), fixture.root),
        call((str(launcher),), fixture.root),
    ])
    value = json.loads(fixture.output_path.read_text(encoding="utf-8"))
    assert value["torch_argv"] == [str(fixture.primary)]


def verify_nonpromotable_coverage(root: Path) -> None:
    fixture = ArmFixture(root)
    with fixture.patched(validate_data=True) as probes, patch.object(
        armer, "write_json_exclusive", wraps=armer.write_json_exclusive,
    ) as publish:
        error = rejects(fixture.arm, fixture)
        expected = (
            "unseen calibration coverage is incomplete: "
            f"{fixture.names[49]}"
        )
        assert str(error) == expected, str(error)
        fetch_args = probes.validate_fetch.call_args.args
        assert str(fetch_args[2].source) == \
            fixture.manifest_bindings[55].path
        assert fetch_args[2].sha256 == fixture.manifest_bindings[55].sha256
        assert str(fetch_args[4].source) == fixture.calendar_path.as_posix()
        observations = fetch_args[3]
        assert tuple(observations) == fixture.names
        assert all(
            isinstance(item, armer.ObservedCsv) and
            item.path == series.csv.path and
            item.sha256 == series.csv.sha256 and
            item.timestamps == timestamps and
            not hasattr(item, "opens") and not hasattr(item, "closes")
            for item, series, timestamps in zip(
                observations.values(), fixture.series, fixture.timestamps,
                strict=True,
            )
        )
        assert probes.coverage.call_args.args[2] == {
            name: timestamps
            for name, timestamps in zip(
                fixture.names, fixture.timestamps, strict=True,
            )
        }
        assert probes.read_timestamps.call_count == len(fixture.series)
        probes.coverage.assert_called_once()
        probes.observe_torch.assert_not_called()
        publish.assert_not_called()
        assert not probes.read_attempts


def verify_exact_coverage_and_fit_closure() -> None:
    names = synthetic_master()
    coverage = synthetic_coverage(names)
    assert tuple(
        (phase.phase, phase.missing) for phase in coverage.phases
    ) == EXPECTED_MISSING
    armer._require_expected_coverage(coverage, names)

    substituted = coverage
    for phase in PHASES:
        substituted = with_validation(substituted, phase, 14, 1)
        substituted = with_validation(substituted, phase, 15, 0)
    mutations = (
        substituted,
        ScalingCoverage(tuple(reversed(coverage.phases))),
    )
    for invalid in mutations:
        try:
            armer._require_expected_coverage(invalid, names)
        except ValueError:
            continue
        raise AssertionError("nonfrozen coverage was accepted")

    with patch.object(armer, "expected_fit_jobs", return_value=(object(),)):
        try:
            armer._require_expected_coverage(coverage, names)
        except ValueError as error:
            assert str(error) == \
                "scaling fit schedule does not match the benchmark"
        else:
            raise AssertionError("nonfrozen fit closure was accepted")


def verify_prediction_schedule() -> None:
    names = tuple(f"S{54 - index:02d}" for index in range(55))
    coverage = synthetic_coverage(names)
    evaluable = {
        phase.phase: phase.evaluable for phase in coverage.phases
    }
    jobs = expected_fit_jobs(names, evaluable)
    records = {
        kind: sum(
            len(required_prediction_series(job, names, evaluable))
            for job in jobs if job.kind == kind
        )
        for kind in ("pooled", "ridge", "local")
    }
    assert records == {
        "pooled": 12_960,
        "ridge": 476,
        "local": 780,
    }
    assert sum(records.values()) == EXPECTED_PREDICTION_RECORDS == 14_216

    conditioned = FitJob(
        "pooled", MODES[0], 11, PHASES[0],
        "conditioned_panel_transformer", SEEDS[0], names[:11],
    )
    pooled = replace(conditioned, model="panel_transformer")
    assert required_prediction_series(
        conditioned, names, evaluable,
    ) == names[:11]
    assert required_prediction_series(
        pooled, names, evaluable,
    ) == (*names[:11], *names[44:])
    local_missing = FitJob(
        "local", None, None, PHASES[0], "local_transformer",
        SEEDS[0], (names[14],),
    )
    assert required_prediction_series(
        local_missing, names, evaluable,
    ) == ()
    assert question_uses(local_missing, names)
    ridge = FitJob(
        "ridge", None, 11, PHASES[0], "global_ridge", None, names[:11],
    )
    zero = FitJob("zero", None, None, PHASES[0], "zero", None, ())
    invalid_axes = (
        replace(pooled, mode=None),
        replace(pooled, mode="invalid"),
        replace(pooled, cohort=11.0),
        replace(pooled, model="global_ridge"),
        replace(pooled, seed=None),
        replace(pooled, members=names[:10]),
        replace(conditioned, cohort=44, members=names[:44]),
        replace(ridge, mode=MODES[0]),
        replace(ridge, model="panel_transformer"),
        replace(ridge, seed=SEEDS[0]),
        replace(local_missing, mode=MODES[0]),
        replace(local_missing, cohort=11),
        replace(local_missing, model="panel_transformer"),
        replace(local_missing, seed=None),
        replace(local_missing, members=names[14:16]),
        replace(pooled, phase="invalid"),
        replace(pooled, kind="invalid"),
        zero,
    )
    for job in invalid_axes:
        for call in (
            lambda job=job: question_uses(job, names),
            lambda job=job: required_prediction_series(
                job, names, evaluable,
            ),
        ):
            try:
                call()
            except ValueError:
                continue
            raise AssertionError("invalid fit axes were accepted")

    invalid_evaluable = dict(evaluable)
    invalid_evaluable[PHASES[0]] = tuple(
        reversed(invalid_evaluable[PHASES[0]])
    )
    invalid = (
        (pooled, names[:-1], evaluable),
        (pooled, names, {PHASES[0]: evaluable[PHASES[0]]}),
        (pooled, names, invalid_evaluable),
    )
    for job, master, phase_evaluable in invalid:
        try:
            required_prediction_series(job, master, phase_evaluable)
        except ValueError:
            continue
        raise AssertionError("invalid prediction schedule was accepted")

    protocol = expected_protocol()
    assert protocol["device"] == "cpu"
    assert protocol["prediction_schema"] == 2


class PackedGrid:
    def __init__(self, train: int, row: object | None) -> None:
        self.train = train
        self.row = row

    def __getitem__(self, key: slice) -> tuple[object, ...]:
        assert key == slice(self.train, None, None)
        return () if self.row is None else (self.row,)


def derive_synthetic_coverage(
    timestamps: Mapping[str, tuple[str, ...]],
) -> ScalingCoverage:
    names = tuple(timestamps)
    manifest = SimpleNamespace(
        series=tuple(SimpleNamespace(ticker=name) for name in names),
        interval_minutes=30, start=object(), end=object(),
    )
    missing = {
        "fold-0": {14, 28},
        "fold-1": {14, 28, 39},
        "calibration": {14, 28, 32, 39},
    }
    calls = 0

    def pack(*_args: object) -> SimpleNamespace:
        nonlocal calls
        index, phase_index = divmod(calls, len(PHASES))
        phase = PHASES[phase_index]
        calls += 1
        train = phase_train_rows(phase, index)
        validation = int(index not in missing[phase])
        row = SimpleNamespace(as_of=0, entry=1, target=2) \
            if validation else None
        return SimpleNamespace(
            counts=(train, validation),
            rows=PackedGrid(train, row),
        )

    blocks = SimpleNamespace(
        folds=(("train-0", "validation-0"),
               ("train-1", "validation-1")),
        holdout=("train-calibration", "validation-calibration"),
    )
    samples = SimpleNamespace(rows=(), opportunities=5_505)
    with patch.object(inputs, "common_calendar", return_value=blocks), \
         patch.object(inputs, "session_samples", return_value=samples), \
         patch.object(inputs, "pack_rows", side_effect=pack):
        return inputs.common_coverage(
            manifest, SimpleNamespace(), timestamps,
        )


def verify_timestamp_grid_provenance() -> None:
    names = tuple(f"S{index:02d}" for index in range(55))
    timestamps = {
        name: (
            f"{name}:as-of", f"{name}:entry", f"{name}:target",
        )
        for name in names
    }
    coverage = derive_synthetic_coverage(timestamps)
    assert coverage.promotable
    expected = timestamp_grid_sha256((
        timestamps[names[0]],
    ))
    assert coverage.phases[0].series[0].timestamp_sha256 == expected
    assert coverage.phases[0].series[14].timestamp_sha256 == \
        timestamp_grid_sha256(())

    mutated = dict(timestamps)
    mutated[names[0]] = (
        timestamps[names[0]][0],
        timestamps[names[0]][1],
        "mutated-target",
    )
    rebound = derive_synthetic_coverage(mutated)
    assert rebound.phases[0].series[0].timestamp_sha256 != expected
    assert tuple(
        (phase.counts, phase.missing) for phase in rebound.phases
    ) == tuple(
        (phase.counts, phase.missing) for phase in coverage.phases
    )


def verify_coverage_parser() -> None:
    names = synthetic_master()
    value = coverage_value(synthetic_coverage(names))
    parsed = ScalingCoverage.parse(value)
    assert parsed.master == names
    assert tuple(
        (phase.phase, phase.missing) for phase in parsed.phases
    ) == tuple(
        (phase, tuple(names[index] for index in indices))
        for phase, indices in (
            ("fold-0", (14, 28)),
            ("fold-1", (14, 28, 39)),
            ("calibration", (14, 28, 32, 39)),
        )
    )

    mutations = []
    for mutate in (
        lambda item: item.pop(),
        lambda item: item.append(deepcopy(item[-1])),
        lambda item: item.reverse(),
    ):
        invalid = deepcopy(value)
        mutate(invalid)
        mutations.append(invalid)
    invalid = deepcopy(value)
    invalid[0]["series"][0]["series"] = names[1]
    mutations.append(invalid)
    invalid = deepcopy(value)
    invalid[0]["series"][0], invalid[0]["series"][1] = \
        invalid[0]["series"][1], invalid[0]["series"][0]
    mutations.append(invalid)
    for field, replacement in (
        ("train_rows", 0),
        ("validation_rows", -1),
        ("timestamp_sha256", "invalid"),
    ):
        invalid = deepcopy(value)
        invalid[0]["series"][0][field] = replacement
        mutations.append(invalid)
    invalid = deepcopy(value)
    invalid[0]["series"][0]["train_rows"] += 1
    mutations.append(invalid)
    invalid = deepcopy(value)
    invalid[0]["series"][0]["timestamp_sha256"] = \
        timestamp_grid_sha256(())
    mutations.append(invalid)
    invalid = deepcopy(value)
    invalid[0]["series"][0]["extra"] = True
    mutations.append(invalid)
    for invalid in mutations:
        try:
            ScalingCoverage.parse(invalid)
        except ValueError:
            continue
        raise AssertionError("invalid scaling coverage was accepted")


def verify_schema_precedes_data_and_runtime() -> None:
    for schema in (3, 4.0, None):
        with tempfile.TemporaryDirectory(
            prefix="compose-mini-scaling-arm-schema-",
        ) as directory:
            fixture = ArmFixture(Path(directory))
            report = json.loads(
                (fixture.root / fixture.fetch_path).read_text(
                    encoding="utf-8",
                )
            )
            if schema is None:
                del report["fetch_schema"]
            else:
                report["fetch_schema"] = schema
            write_json(fixture.root / fixture.fetch_path, report)
            with fixture.patched(validate_data=True) as probes:
                error = rejects(fixture.arm, fixture)
            assert str(error) == "scaling fetch report must use schema 4"
            probes.read_timestamps.assert_not_called()
            probes.validate_fetch.assert_not_called()
            probes.coverage.assert_not_called()
            probes.observe_torch.assert_not_called()


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
        prefix="compose-mini-scaling-arm-serialize-",
    ) as directory:
        verify_serializes_coverage(Path(directory))
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-arm-torch-venv-",
    ) as directory:
        verify_torch_venv_launcher(Path(directory))
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-arm-shared-runtime-",
    ) as directory:
        verify_shared_torch_runtime(Path(directory))
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-arm-coverage-",
    ) as directory:
        verify_nonpromotable_coverage(Path(directory))
    verify_exact_coverage_and_fit_closure()
    verify_prediction_schedule()
    verify_timestamp_grid_provenance()
    verify_coverage_parser()
    verify_schema_precedes_data_and_runtime()
    print("universe scaling arming tests passed")


if __name__ == "__main__":
    main()
