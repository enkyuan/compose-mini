#!/usr/bin/env python3
"""Verify the immutable development-only universe-scaling contract."""

from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
from math import inf, log, nextafter
from pathlib import Path
from statistics import fmean
import base64
import hashlib
import json
import os
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.universe_scaling_contract import (
    CALENDAR_SHA256, CONFIG_SHA256, CSV_ROOT, EXPECTED_BUDGETS,
    EXPECTED_FIT_COUNT, EXPECTED_MISSING, EXPECTED_PREDICTION_RECORDS,
    EXPECTED_PREDICTION_VALUES, FETCH_PATH, FETCH_SHA256,
    FINALIZER_PYTHON_FLAGS,
    FINALIZER_SOURCE_PATHS, FIXED_EPOCH_BUDGET, MANIFEST_BINDINGS, PHASES,
    MODES, POOLED_MODELS, RUNNER_PRIMARY_PYTHON_FLAGS,
    RUNNER_TORCH_PYTHON_FLAGS, SEEDS, SELECTION_FILES, SELECTION_SHA256,
    SOURCE_PATHS, TRAINING_COHORTS, FitJob, PhaseCoverage, ScalingAttempt,
    ScalingCoverage, SeriesCoverage,
    expected_fit_jobs, expected_protocol, expected_scaling_commands,
    question_uses, required_prediction_series, timestamp_grid_sha256,
)
from tools.universe_scaling_inputs import (
    common_coverage, fetch_series, selection_binding, selection_paths,
)
from tools.float32 import decode_f32le_base64, encode_f32le_base64
import tools.finalize_universe_scaling as finalizer
from tools.finalize_universe_scaling import (
    MarketTruth, PredictionTruth, _gate_results, _transition,
    build_development_summary, fit_provenance_id,
    validate_fit_ledger, validate_prediction_ledger,
)
from tools.universe_scaling import ForecastPoint, paired_comparison
from tools.universe_contract import PackedRows
from tools.session_samples import SampleRows, SessionSamples
import tools.files as file_tools
import tools.run_universe_scaling as runner


def sha256(index: int) -> str:
    return f"{index:064x}"


def tree(root: str, paths: tuple[str, ...], offset: int) -> dict[str, object]:
    files = [
        {"path": path, "sha256": sha256(offset + index)}
        for index, path in enumerate(sorted(paths))
    ]
    digest = hashlib.sha256()
    for item in files:
        digest.update(
            item["path"].encode() + b"\0" +
            item["sha256"].encode() + b"\n"
        )
    return {"root": root, "files": files, "sha256": digest.hexdigest()}


def binding(path: str, digest: str) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def executable(path: str, index: int) -> dict[str, str]:
    return binding(path, sha256(index)) | {"version": f"runtime {index}"}


def synthetic_master() -> tuple[str, ...]:
    names = [f"S{index:02d}" for index in range(55)]
    for index, name in (
        (14, "ALTR"), (28, "ZI"), (32, "FYBR"), (39, "INFA"),
    ):
        names[index] = name
    return tuple(names)


def synthetic_coverage(names: tuple[str, ...]) -> ScalingCoverage:
    missing = {
        "fold-0": {"ALTR", "ZI"},
        "fold-1": {"ALTR", "ZI", "INFA"},
        "calibration": {"ALTR", "ZI", "FYBR", "INFA"},
    }
    timestamps = (
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        "2026-01-03T00:00:00Z",
    )
    phases = []
    for phase, budget in EXPECTED_BUDGETS:
        base, remainder = divmod(budget.control_samples, 11)
        series = []
        for index, name in enumerate(names):
            validation = int(name not in missing[phase])
            train = (
                base + int(index < remainder) if index < 11 else
                100 + index
            )
            series.append(SeriesCoverage(
                name, train, validation,
                timestamp_grid_sha256((timestamps,))
                if validation else timestamp_grid_sha256(()),
            ))
        phases.append(PhaseCoverage(phase, tuple(series)))
    return ScalingCoverage(tuple(phases))


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


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def attempt_value(repository_root: str) -> dict[str, object]:
    attempt_path = "experiments/universe-scaling-run-attempt.json"
    run_dir = "reports/universe-scaling-run"
    outputs = {
        "fits": f"{run_dir}/fits.jsonl",
        "predictions": f"{run_dir}/predictions.jsonl",
        "summary": f"{run_dir}/summary.json",
        "outcome": "experiments/universe-scaling-run-outcome.json",
    }
    torch_python = executable("/runtime/torch-python", 9_001)
    coverage = synthetic_coverage(synthetic_master())
    return {
        "attempt_path": attempt_path,
        "budgets": [
            {
                "phase": phase,
                "control_samples": budget.control_samples,
                "batch_size": budget.batch_size,
                "checkpoints": budget.checkpoints,
                "updates_per_checkpoint": budget.updates_per_checkpoint,
                "total_updates": budget.total_updates,
            }
            for phase, budget in EXPECTED_BUDGETS
        ],
        "commands": {
            name: list(command)
            for name, command in expected_scaling_commands(
                Path(attempt_path), outputs,
            ).items()
        },
        "config": binding(
            "experiments/executable-h13-universe.example.json", CONFIG_SHA256,
        ),
        "coverage": coverage_value(coverage),
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": f"{run_dir}/.pycache",
        },
        "fetch_report": binding(
            FETCH_PATH.as_posix(), FETCH_SHA256,
        ),
        "finalizer_tree": tree(
            repository_root, FINALIZER_SOURCE_PATHS, 7_000,
        ),
        "implementation_commit": "1" * 40,
        "manifests": [
            {
                "size": size,
                **binding(file.path, file.sha256),
            }
            for size, file in MANIFEST_BINDINGS.items()
        ],
        "outputs": outputs,
        "primary_python": executable("/runtime/primary-python", 9_000),
        "protocol": expected_protocol(),
        "run_dir": run_dir,
        "run_id": "universe-scaling-run",
        "schema": 1,
        "selection_tree": {
            "root": "reports/universe-selection-20260724-06",
            "files": SELECTION_FILES,
            "sha256": SELECTION_SHA256,
        },
        "session_calendar": binding(
            "universes/us-equities-core-2024-07-22_2026-07-21.json",
            CALENDAR_SHA256,
        ),
        "source_tree": tree(repository_root, SOURCE_PATHS, 6_000),
        "status": "armed",
        "torch_argv": [torch_python["path"]],
        "torch_probe": {
            "python": torch_python,
            "version": "2.13.0",
            "git_version": None,
            "cuda_version": None,
            "config": "cpu",
            "package_tree": tree("/runtime/torch", ("torch.py",), 9_100),
        },
    }


def retarget(value: dict[str, object], run_id: str) -> None:
    attempt_path = f"experiments/{run_id}-attempt.json"
    run_dir = f"reports/{run_id}"
    outputs = {
        "fits": f"{run_dir}/fits.jsonl",
        "predictions": f"{run_dir}/predictions.jsonl",
        "summary": f"{run_dir}/summary.json",
        "outcome": f"experiments/{run_id}-outcome.json",
    }
    value.update(
        attempt_path=attempt_path, run_dir=run_dir, run_id=run_id,
        outputs=outputs,
    )
    value["environment"]["PYTHONPYCACHEPREFIX"] = f"{run_dir}/.pycache"
    value["commands"] = {
        name: list(command)
        for name, command in expected_scaling_commands(
            Path(attempt_path), outputs,
        ).items()
    }


def reject(
    directory: Path, repository: Path, value: dict[str, object],
    label: str = "mutation",
) -> None:
    path = directory / "attempt.json"
    write_json(path, value)
    try:
        ScalingAttempt.read(
            path, Path(value["attempt_path"]), repository,
        )
    except ValueError:
        return
    raise AssertionError(f"invalid scaling attempt was accepted: {label}")


def verify_attempt() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-contract-",
    ) as directory_name:
        directory = Path(directory_name)
        repository = directory / "repository"
        repository.mkdir()
        repository = repository.resolve()
        value = attempt_value(str(repository))
        path = directory / "attempt.json"
        write_json(path, value)
        attempt = ScalingAttempt.read(
            path, Path(value["attempt_path"]), repository,
        )
        assert attempt.training_cohorts == (11, 22, 33, 55)
        assert attempt.transfer_cohorts == (11, 22, 33, 44)
        assert attempt.unseen_ranks == tuple(range(45, 56))
        assert dict(attempt.budgets) == dict(EXPECTED_BUDGETS)
        assert tuple(attempt.outputs) == (
            "fits", "predictions", "summary", "outcome",
        )
        forbidden = ("--test", "policy", "backtest", "replay", "authorization")
        assert not any(
            token in argument
            for command in attempt.commands.values()
            for argument in command for token in forbidden
        )
        try:
            attempt.protocol["models"] = ()
        except TypeError:
            pass
        else:
            raise AssertionError("parsed protocol is mutable")
        try:
            attempt.protocol["calendar"]["folds"] = ()
        except TypeError:
            pass
        else:
            raise AssertionError("nested parsed protocol is mutable")
        protocol = expected_protocol()
        assert (
            protocol["batch_size"], protocol["epochs"], protocol["patience"],
        ) == (
            FIXED_EPOCH_BUDGET.batch_size, FIXED_EPOCH_BUDGET.epochs,
            FIXED_EPOCH_BUDGET.patience,
        )
        assert protocol["finalizer_python_flags"] == \
            list(FINALIZER_PYTHON_FLAGS) == ["-I", "-S", "-B"]
        assert protocol["runner_primary_python_flags"] == \
            list(RUNNER_PRIMARY_PYTHON_FLAGS) == ["-I", "-S", "-B"]
        assert protocol["runner_torch_python_flags"] == \
            list(RUNNER_TORCH_PYTHON_FLAGS) == ["-I", "-S", "-B"]
        assert finalizer._BOOTSTRAP_PYTHON_FLAGS == FINALIZER_PYTHON_FLAGS
        assert runner._BOOTSTRAP_PRIMARY_FLAGS == \
            RUNNER_PRIMARY_PYTHON_FLAGS
        assert runner._BOOTSTRAP_TORCH_FLAGS == RUNNER_TORCH_PYTHON_FLAGS
        protocol["models"].clear()
        assert expected_protocol()["models"]
        assert "tools/__init__.py" in SOURCE_PATHS
        assert "tools/__init__.py" in FINALIZER_SOURCE_PATHS
        assert {
            "tools/analyze_universe.py",
            "tools/arm_universe_scaling.py",
        } <= set(SOURCE_PATHS)
        assert {
            "tools/chronology.py", "tools/data_v1.py",
            "tools/session_calendar.py", "tools/session_samples.py",
            "tools/universe_contract.py",
        } <= set(FINALIZER_SOURCE_PATHS)

        mutations = []
        for label, mutate in (
            ("status", lambda item: item.update(status="complete")),
            ("environment", lambda item: item["environment"].update(
                MASSIVE_API_KEY="secret",
            )),
            ("cohort order", lambda item: item[
                "protocol"
            ]["training_cohorts"].reverse()),
            ("protocol", lambda item: item["protocol"].update(folds=3)),
            ("budget", lambda item: item[
                "budgets"
            ][0].update(total_updates=27_401)),
            ("missing identities", lambda item: (
                item["coverage"][0]["series"][14].update(
                    validation_rows=1,
                    timestamp_sha256=item["coverage"][0][
                        "series"
                    ][15]["timestamp_sha256"],
                ),
                item["coverage"][0]["series"][15].update(
                    validation_rows=0,
                    timestamp_sha256=timestamp_grid_sha256(()),
                ),
            )),
            ("manifest order", lambda item: item["manifests"].reverse()),
            ("manifest hash", lambda item: item[
                "manifests"
            ][0].update(sha256="0" * 64)),
            ("implementation commit", lambda item: item.update(
                implementation_commit="not-a-commit",
            )),
            ("selection root", lambda item: item[
                "selection_tree"
            ].update(root="reports/other-selection")),
            ("selection count", lambda item: item[
                "selection_tree"
            ].update(files=76)),
            ("selection digest", lambda item: item[
                "selection_tree"
            ].update(sha256="0" * 64)),
            ("fetch", lambda item: item[
                "fetch_report"
            ].update(sha256="0" * 64)),
            ("calendar", lambda item: item[
                "session_calendar"
            ].update(sha256="0" * 64)),
            ("config", lambda item: item[
                "config"
            ].update(sha256="0" * 64)),
            ("source root", lambda item: item[
                "source_tree"
            ].update(root="/other")),
            ("source digest", lambda item: item[
                "source_tree"
            ]["files"][0].update(
                sha256="0" * 64,
            )),
            ("finalizer root", lambda item: item[
                "finalizer_tree"
            ].update(root="/other")),
            ("coordinated roots", lambda item: (
                item["source_tree"].update(root="/other"),
                item["finalizer_tree"].update(root="/other"),
            )),
            ("primary path", lambda item: item[
                "primary_python"
            ].update(path="relative")),
            ("torch Python", lambda item: item[
                "torch_probe"
            ]["python"].update(path="/other")),
            ("torch argv", lambda item: item[
                "torch_argv"
            ].append("--test")),
            ("command", lambda item: item[
                "commands"
            ]["calibrate"].append("--test")),
            ("attempt path", lambda item: item.update(attempt_path="--help")),
            ("output alias", lambda item: item["outputs"].update(
                outcome=item["attempt_path"],
            )),
            ("extra output", lambda item: item[
                "outputs"
            ].update(test="test.jsonl")),
        ):
            invalid = deepcopy(value)
            mutate(invalid)
            mutations.append((label, invalid))
        for label, invalid in mutations:
            reject(directory, repository, invalid, label)
        collision = deepcopy(value)
        retarget(collision, "universe-selection-20260724-06")
        reject(
            directory, repository, collision,
            "selection output collision",
        )

        write_json(path, value)
        try:
            ScalingAttempt.read(
                path, Path("experiments/other-attempt.json"), repository,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("logical attempt path mismatch was accepted")
        path.write_text(json.dumps(value), encoding="utf-8")
        try:
            ScalingAttempt.read(
                path, Path(value["attempt_path"]), repository,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("noncanonical attempt JSON was accepted")
        path.write_text('{"schema":1,"schema":1}', encoding="utf-8")
        try:
            ScalingAttempt.read(
                path, Path(value["attempt_path"]), repository,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate attempt field was accepted")


def verify_isolated_startup() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-isolation-",
    ) as directory_name:
        directory = Path(directory_name)

        def copy_repository(name: str) -> Path:
            repository = directory / name
            shutil.copytree(
                ROOT / "tools", repository / "tools",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            return repository

        environment = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(directory),
            "PYTHONPYCACHEPREFIX": str(directory / "forged-cache"),
            "TMPDIR": str(directory / "private-tmp"),
        }

        def run(repository: Path, *flags: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                (
                    sys.executable, *flags,
                    str(repository / "tools/finalize_universe_scaling.py"),
                    "--help",
                ),
                cwd=repository,
                env=environment,
                capture_output=True, text=True, check=False,
            )

        repository = copy_repository("repository")
        (directory / "private-tmp").mkdir()
        markers = (
            repository / "tools-statistics-ran",
            repository / "root-statistics-ran",
            repository / "forged-pyc-ran",
        )
        for module, marker in (
            (repository / "tools" / "statistics.py", markers[0]),
            (repository / "statistics.py", markers[1]),
        ):
            module.write_text(
                f"open({str(marker)!r}, 'w').write('ran')\n"
                "def fmean(values):\n"
                "    return 0.0\n",
                encoding="ascii",
            )
        source = repository / "tools/files.py"
        malicious = directory / "malicious-files.py"
        payload = (
            f"open({str(markers[2])!r}, 'w').write('ran')\n"
            "raise RuntimeError('forged cache executed')\n"
        ).encode("ascii")
        payload += b"#" * (source.stat().st_size - len(payload))
        malicious.write_bytes(payload)
        os.utime(malicious, (int(source.stat().st_mtime),) * 2)
        cache = repository / "tools/__pycache__" / (
            f"files.{sys.implementation.cache_tag}.pyc"
        )
        cache.parent.mkdir()
        import py_compile
        py_compile.compile(
            str(malicious), cfile=str(cache), dfile=str(source), doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
        assert int.from_bytes(cache.read_bytes()[8:12], "little") == \
            int(source.stat().st_mtime)
        assert int.from_bytes(cache.read_bytes()[12:16], "little") == \
            source.stat().st_size
        before = tuple(sorted(
            path.relative_to(repository)
            for path in repository.rglob("*")
        ))

        unisolated = run(repository)
        assert unisolated.returncode != 0
        assert "isolated" in unisolated.stderr.lower()
        assert not any(path.exists() for path in markers)

        isolated = run(repository, *FINALIZER_PYTHON_FLAGS)
        assert isolated.returncode == 0, isolated.stderr
        assert "usage:" in isolated.stdout
        assert not any(path.exists() for path in markers)
        optimized = run(repository, *FINALIZER_PYTHON_FLAGS, "-O")
        assert optimized.returncode != 0
        assert "launch" in optimized.stderr.lower()

        runner_script = repository / "tools/run_universe_scaling.py"

        def run_runner(
            *flags: str, arguments: tuple[str, ...] = ("--help",),
        ) -> subprocess.CompletedProcess:
            return subprocess.run(
                (sys.executable, *flags, str(runner_script), *arguments),
                cwd=repository, env=environment,
                capture_output=True, text=True, check=False,
            )

        unisolated_runner = run_runner()
        assert unisolated_runner.returncode != 0
        assert "isolated" in unisolated_runner.stderr.lower()
        isolated_runner = run_runner(*RUNNER_PRIMARY_PYTHON_FLAGS)
        assert isolated_runner.returncode == 0, isolated_runner.stderr
        assert "usage:" in isolated_runner.stdout
        optimized_runner = run_runner(
            *RUNNER_PRIMARY_PYTHON_FLAGS, "-O",
        )
        assert optimized_runner.returncode != 0
        assert "launch" in optimized_runner.stderr.lower()
        torch_runner = run_runner(
            *RUNNER_TORCH_PYTHON_FLAGS,
            arguments=("calibrate", "missing-attempt.json"),
        )
        assert torch_runner.returncode == 2
        assert "launch" not in torch_runner.stderr.lower()

        runner_runpy = (
            "import runpy,sys;"
            f"sys.argv=[{str(runner_script)!r},'--help'];"
            f"runpy.run_path({str(runner_script)!r},run_name='__main__')"
        )
        rejected_runner = subprocess.run(
            (
                sys.executable, *RUNNER_PRIMARY_PYTHON_FLAGS,
                "-c", runner_runpy,
            ),
            cwd=repository, env=environment,
            capture_output=True, text=True, check=False,
        )
        assert rejected_runner.returncode != 0
        assert "launch" in rejected_runner.stderr.lower()
        preloaded_runner = subprocess.run(
            (
                sys.executable, *RUNNER_PRIMARY_PYTHON_FLAGS, "-c",
                "import ctypes,runpy,sys;"
                f"sys.argv=[{str(runner_script)!r},'--help'];"
                f"runpy.run_path({str(runner_script)!r},"
                "run_name='__main__')",
            ),
            cwd=repository, env=environment,
            capture_output=True, text=True, check=False,
        )
        assert preloaded_runner.returncode != 0
        assert "inspection" in preloaded_runner.stderr.lower()

        after = tuple(sorted(
            path.relative_to(repository)
            for path in repository.rglob("*")
        ))
        assert before == after, (
            set(after) - set(before), set(before) - set(after),
        )
        assert not tuple((directory / "private-tmp").iterdir())

        import_repository = copy_repository("isolated-import")
        import_code = (
            f"import sys;sys.path.append({str(import_repository)!r});"
            "import tools.finalize_universe_scaling as finalizer;"
            "assert callable(finalizer.analyze_ledgers)"
        )
        imported = subprocess.run(
            (sys.executable, *FINALIZER_PYTHON_FLAGS, "-c", import_code),
            cwd=import_repository, capture_output=True, text=True, check=False,
        )
        assert imported.returncode == 0, imported.stderr
        skipped = subprocess.run(
            (
                sys.executable, *FINALIZER_PYTHON_FLAGS, "-c",
                import_code +
                ";from pathlib import Path;"
                "finalizer.finalize("
                "Path('missing-attempt'),Path('missing-outcome'),"
                "'2026-07-24T00:00:00Z','2026-07-24T00:00:01Z',"
                "'setup',2,'setup-failure')",
            ),
            cwd=import_repository, capture_output=True, text=True, check=False,
        )
        assert skipped.returncode != 0
        assert "bootstrap" in skipped.stderr.lower()

        forged_prefix = str(directory / "forged-finalizer-cache")
        script = import_repository / "tools/finalize_universe_scaling.py"
        forged_argv = (
            str(script), "missing-attempt", "missing-outcome",
            "--started", "2026-07-24T00:00:00Z",
            "--ended", "2026-07-24T00:00:01Z",
            "--stage", "setup", "--exit", "2",
            "--status", "setup-failure",
        )
        forged = subprocess.run(
            (
                sys.executable, *FINALIZER_PYTHON_FLAGS, "-c",
                import_code +
                f";finalizer._BOOTSTRAP_CACHE_PREFIX={forged_prefix!r}"
                f";sys.pycache_prefix={forged_prefix!r}"
                f";sys.argv={list(forged_argv)!r}"
                ";sys.orig_argv=[sys.executable,'-I','-S','-B',*sys.argv]"
                ";from pathlib import Path"
                ";finalizer.finalize("
                "Path('missing-attempt'),Path('missing-outcome'),"
                "'2026-07-24T00:00:00Z','2026-07-24T00:00:01Z',"
                "'setup',2,'setup-failure')",
            ),
            cwd=import_repository, capture_output=True, text=True,
            check=False,
        )
        assert forged.returncode != 0
        assert "launch" in forged.stderr.lower(), forged.stderr

        foreign = directory / "foreign/tools"
        foreign.mkdir(parents=True)
        marker = directory / "foreign-tools-ran"
        (foreign / "__init__.py").write_text(
            f"open({str(marker)!r}, 'w').write('ran')\n",
            encoding="ascii",
        )
        script = repository / "tools/finalize_universe_scaling.py"
        runpy_code = (
            "import runpy,sys;"
            f"sys.argv=[{str(script)!r},'--help'];"
            f"runpy.run_path({str(script)!r},run_name='__main__')"
        )
        rejected = subprocess.run(
            (
                sys.executable, *FINALIZER_PYTHON_FLAGS,
                "-c", runpy_code,
            ),
            cwd=repository, env=environment,
            capture_output=True, text=True, check=False,
        )
        assert rejected.returncode != 0
        assert "launch" in rejected.stderr.lower()

        preloaded_ctypes = subprocess.run(
            (
                sys.executable, *FINALIZER_PYTHON_FLAGS, "-c",
                "import ctypes,runpy,sys;"
                f"sys.argv=[{str(script)!r},'--help'];"
                f"runpy.run_path({str(script)!r},run_name='__main__')",
            ),
            cwd=repository, env=environment,
            capture_output=True, text=True, check=False,
        )
        assert preloaded_ctypes.returncode != 0
        assert "inspection" in preloaded_ctypes.stderr.lower()

        code = (
            f"import sys;sys.path.append({str(import_repository)!r});"
            "import tools.finalize_universe_scaling as finalizer;"
            "[sys.modules.pop(name) for name in tuple(sys.modules) "
            "if name=='tools' or name.startswith('tools.')];"
            f"sys.path.insert(0,{str(foreign.parent)!r});"
            "finalizer._bootstrap_main()"
        )
        rejected = subprocess.run(
            (sys.executable, *FINALIZER_PYTHON_FLAGS, "-c", code),
            cwd=import_repository, env=environment,
            capture_output=True, text=True, check=False,
        )
        assert rejected.returncode != 0
        assert "namespace" in rejected.stderr.lower(), rejected.stderr
        assert not marker.exists()

        marker = directory / "spoofed-original-argv-ran"
        code = (
            "import runpy,sys,types;"
            "module=types.ModuleType('tools.files');"
            "noop=lambda *args,**kwargs:None;"
            "module.FrozenInput=noop;"
            "module.freeze_inputs=noop;"
            "module.verify_frozen=noop;"
            "module.write_json_exclusive=noop;"
            f"module.__getattr__=lambda name:"
            f"(open({str(marker)!r},'w').write(name),noop)[1];"
            "sys.modules['tools.files']=module;"
            f"sys.argv=[{str(script)!r},'--help'];"
            "sys.orig_argv=[sys.executable,'-I','-S','-B',*sys.argv];"
            f"runpy.run_path({str(script)!r},run_name='__main__')"
        )
        rejected = subprocess.run(
            (sys.executable, *FINALIZER_PYTHON_FLAGS, "-c", code),
            cwd=repository, env=environment,
            capture_output=True, text=True, check=False,
        )
        assert rejected.returncode != 0
        assert "launch" in rejected.stderr.lower()
        assert not marker.exists()

        for injected in ("tools", "tools.files"):
            marker = directory / f"preloaded-{injected.replace('.', '-')}-ran"
            code = (
                f"import sys;sys.path.append({str(import_repository)!r});"
                "import tools.finalize_universe_scaling as finalizer;"
                "[sys.modules.pop(name) for name in tuple(sys.modules) "
                "if name=='tools' or name.startswith('tools.')];"
                "import types;module=types.ModuleType('injected');"
                "noop=lambda *args,**kwargs:None;"
                "module.FrozenInput=noop;"
                "module.freeze_inputs=noop;"
                "module.verify_frozen=noop;"
                "module.write_json_exclusive=noop;"
                f"module.__getattr__=lambda name:"
                f"(open({str(marker)!r},'w').write(name),noop)[1];"
                f"sys.modules[{injected!r}]=module;"
                "finalizer._bootstrap_main()"
            )
            rejected = subprocess.run(
                (sys.executable, *FINALIZER_PYTHON_FLAGS, "-c", code),
                cwd=import_repository, env=environment,
                capture_output=True, text=True, check=False,
            )
            assert rejected.returncode != 0
            assert "namespace" in rejected.stderr.lower()
            assert not marker.exists()

        package_repository = copy_repository("package-shadow")
        marker = package_repository / "files-package-ran"
        package = package_repository / "tools/files"
        package.mkdir()
        (package / "__init__.py").write_text(
            f"open({str(marker)!r}, 'w').write('ran')\n",
            encoding="ascii",
        )
        rejected = run(package_repository, *FINALIZER_PYTHON_FLAGS)
        assert rejected.returncode != 0
        assert "namespace" in rejected.stderr.lower()
        assert not marker.exists()
        package_runner = package_repository / "tools/run_universe_scaling.py"
        rejected = subprocess.run(
            (
                sys.executable, *RUNNER_PRIMARY_PYTHON_FLAGS,
                str(package_runner), "--help",
            ),
            cwd=package_repository, env=environment,
            capture_output=True, text=True, check=False,
        )
        assert rejected.returncode != 0
        assert "namespace" in rejected.stderr.lower()
        assert not marker.exists()

        for name, install in (
            ("extension-shadow", lambda tools: (
                tools / "files.so"
            ).write_bytes(b"not an extension")),
            ("symlink-shadow", lambda tools: (
                tools / "shadow.py"
            ).symlink_to(tools / "files.py")),
        ):
            unsafe = copy_repository(name)
            install(unsafe / "tools")
            rejected = run(unsafe, *FINALIZER_PYTHON_FLAGS)
            assert rejected.returncode != 0
            assert "namespace" in rejected.stderr.lower()


def verify_input_derivation() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-series-",
    ) as directory_name:
        root = Path(directory_name).resolve()
        value = {"series": [
            {
                "ticker": f"S{index:02d}",
                "csv": {
                    "path": str(
                        root / CSV_ROOT / f"s{index:02d}-30m.csv"
                    ),
                    "rows": 5_000 + index,
                    "sha256": sha256(2_000 + index),
                },
            }
            for index in range(55)
        ]}
        series = fetch_series(value, root)
        assert tuple(item.name for item in series) == tuple(
            f"S{index:02d}" for index in range(55)
        )
        report = root / "fetch.json"
        write_json(report, value)
        with patch.object(finalizer, "ROOT", root):
            names, bindings = finalizer._fetch_bindings(report)
        assert names == tuple(item.name for item in series)
        assert bindings == tuple(item.csv for item in series)
        (root / "selection").mkdir()
        attempt = SimpleNamespace(
            config=SimpleNamespace(path=str(root / "config.json")),
            fetch_report=SimpleNamespace(
                path=str(report),
                sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            ),
            manifests=(),
            selection_tree=SimpleNamespace(root="selection"),
            session_calendar=SimpleNamespace(path=str(root / "calendar.json")),
            source_tree=SimpleNamespace(files=()),
            torch_probe=SimpleNamespace(
                package_tree=SimpleNamespace(
                    root=str(root / "torch-package"), files=(),
                ),
                python=SimpleNamespace(path=str(root / "torch-python")),
            ),
        )
        with patch.object(finalizer, "ROOT", root):
            success = finalizer._success_inputs(attempt)
        assert success.csv_names == names
        assert success.csv == bindings
        assert set(map(Path, (item.path for item in bindings))) <= \
            set(success.paths)
        invalid = deepcopy(value)
        invalid["series"][0]["csv"]["path"] = str(root / "outside.csv")
        try:
            fetch_series(invalid, root)
        except ValueError:
            pass
        else:
            raise AssertionError("fetch-derived CSV escaped its frozen root")

    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-selection-",
    ) as directory_name:
        root = Path(directory_name)
        real_parent = root / "real"
        selection = real_parent / "selection"
        selection.mkdir(parents=True)
        (selection / "manifest.json").write_text("data", encoding="ascii")
        root_link = root / "selection-link"
        root_link.symlink_to(selection, target_is_directory=True)
        parent_link = root / "parent-link"
        parent_link.symlink_to(real_parent, target_is_directory=True)
        for function in (selection_paths, selection_binding):
            for alias in (root_link, parent_link / "selection"):
                try:
                    function(alias)
                except ValueError:
                    continue
                raise AssertionError("symlinked selection root was accepted")

    names = [f"S{index:02d}" for index in range(55)]
    manifest = SimpleNamespace(
        series=tuple(SimpleNamespace(ticker=name) for name in names),
        interval_minutes=30, start="start", end="end",
    )
    timestamps = {name: (name,) for name in names}

    def samples(timestamps: tuple[str, ...], *_: object) -> object:
        return SimpleNamespace(
            opportunities=5_505, rows=timestamps,
        )

    def derive(missing: dict[str, tuple[int, ...]]) -> ScalingCoverage:
        calls = 0

        def packed(rows: tuple[str, ...], *_: object) -> PackedRows:
            nonlocal calls
            name, phase = rows[0], PHASES[calls % len(PHASES)]
            index = names.index(name)
            budget = dict(EXPECTED_BUDGETS)[phase]
            base, remainder = divmod(budget.control_samples, 11)
            train = base + (index < remainder) if index < 11 else 1
            calls += 1
            return PackedRows(
                (), (train, int(index not in missing[phase])),
            )

        with patch(
            "tools.universe_scaling_inputs.session_samples",
            side_effect=samples,
        ), patch(
            "tools.universe_scaling_inputs.pack_rows",
            side_effect=packed,
        ):
            result = common_coverage(
                manifest, SimpleNamespace(), timestamps,
            )
        assert calls == 55 * len(PHASES)
        return result

    missing = {
        "fold-0": (11, 20),
        "fold-1": (11, 20, 34),
        "calibration": (11, 20, 34, 40),
    }
    coverage = derive(missing)
    assert tuple(len(item.evaluable) for item in coverage.phases) == (
        53, 52, 51,
    )
    assert tuple(
        (item.phase, item.missing) for item in coverage.phases
    ) == tuple(
        (phase, tuple(names[index] for index in missing[phase]))
        for phase in PHASES
    )
    assert tuple(
        sum(train for _, train, _ in item.counts[:11])
        for item in coverage.phases
    ) == tuple(
        budget.control_samples for _, budget in EXPECTED_BUDGETS
    )
    assert coverage.promotable
    coverage.require_promotable()

    unseen_missing = dict(missing)
    unseen_missing["calibration"] += (49,)
    coverage = derive(unseen_missing)
    assert coverage.unseen_missing == (names[49],)
    try:
        coverage.require_promotable()
    except ValueError as error:
        assert str(error) == (
            f"unseen calibration coverage is incomplete: {names[49]}"
        )
    else:
        raise AssertionError("incomplete unseen coverage was accepted")

    core_missing = dict(missing)
    core_missing["fold-0"] = (0, *core_missing["fold-0"])
    try:
        derive(core_missing)
    except ValueError as error:
        assert str(error) == "core validation coverage is incomplete"
    else:
        raise AssertionError("missing core validation coverage was accepted")

    reordered = dict(reversed(tuple(timestamps.items())))
    try:
        common_coverage(manifest, SimpleNamespace(), reordered)
    except ValueError:
        pass
    else:
        raise AssertionError("reordered coverage timestamps were accepted")

    try:
        ScalingCoverage(()).require_promotable()
    except ValueError:
        pass
    else:
        raise AssertionError("phase-free coverage was accepted")


def verify_fit_schedule() -> None:
    names = tuple(f"S{index:02d}" for index in range(55))
    coverage = {
        phase: tuple(
            name for index, name in enumerate(names)
            if index not in range(11, 13 + phase_index)
        )
        for phase_index, phase in enumerate(PHASES)
    }
    jobs = expected_fit_jobs(names, coverage)
    modes = ("fixed-update", "fixed-epoch")
    cohorts = (11, 22, 33, 44, 55)
    phases = ("fold-0", "fold-1", "calibration")
    seeds = (7, 19, 31, 43, 61)
    training = (11, 22, 33, 55)
    expected = tuple(
        FitJob("pooled", mode, cohort, phase, model, seed, names[:cohort])
        for mode in modes
        for cohort in cohorts
        for phase in phases
        for model in (
            "global_mlp", "panel_transformer",
            *(("conditioned_panel_transformer",)
              if cohort in training else ()),
        )
        for seed in seeds
    ) + tuple(
        FitJob(
            "ridge", None, cohort, phase, "global_ridge", None,
            names[:cohort],
        )
        for cohort in cohorts for phase in phases
    ) + tuple(
        FitJob(
            "local", None, None, phase, "local_transformer", seed, (name,),
        )
        for phase in phases for name in coverage[phase] for seed in seeds
    )
    assert jobs == expected
    pooled = tuple(job for job in jobs if job.kind == "pooled")
    ridge = tuple(job for job in jobs if job.kind == "ridge")
    local = tuple(job for job in jobs if job.kind == "local")
    assert len(pooled) == 420
    assert len(ridge) == 15
    assert len(local) == 5 * sum(map(len, coverage.values()))
    assert len(jobs) == len(set(jobs)) == 1_215

    shared = next(
        job for job in pooled
        if (job.mode, job.cohort, job.phase, job.model, job.seed) ==
        ("fixed-update", 11, "fold-0", "panel_transformer", 7)
    )
    assert question_uses(shared, names) == (
        ("cohort-scaling", 11), ("unseen-transfer", 11),
    )
    conditioned = next(
        job for job in pooled
        if job.model == "conditioned_panel_transformer"
    )
    assert question_uses(conditioned, names) == (
        ("cohort-scaling", conditioned.cohort),
    )
    transfer = next(
        job for job in pooled
        if job.cohort == 44 and job.model == "panel_transformer"
    )
    assert transfer.members == names[:44]
    assert not set(names[44:]) & set(transfer.members)
    assert all(job.mode is None and job.seed is None for job in ridge)
    assert all(
        job.mode is None and job.cohort is None and len(job.members) == 1
        for job in local
    )
    assert {
        (job.phase, job.members[0], job.seed) for job in local
    } == {
        (phase, name, seed)
        for phase, members in coverage.items()
        for name in members
        for seed in (7, 19, 31, 43, 61)
    }
    local_uses = {
        11: (("cohort-scaling", 11), ("cohort-scaling", 22),
             ("cohort-scaling", 33), ("cohort-scaling", 55)),
        12: (("cohort-scaling", 22), ("cohort-scaling", 33),
             ("cohort-scaling", 55)),
        22: (("cohort-scaling", 22), ("cohort-scaling", 33),
             ("cohort-scaling", 55)),
        23: (("cohort-scaling", 33), ("cohort-scaling", 55)),
        33: (("cohort-scaling", 33), ("cohort-scaling", 55)),
        34: (("cohort-scaling", 55),),
        44: (("cohort-scaling", 55),),
        45: (("cohort-scaling", 55), ("unseen-transfer", 11),
             ("unseen-transfer", 22), ("unseen-transfer", 33),
             ("unseen-transfer", 44)),
        55: (("cohort-scaling", 55), ("unseen-transfer", 11),
             ("unseen-transfer", 22), ("unseen-transfer", 33),
             ("unseen-transfer", 44)),
    }
    for rank, uses in local_uses.items():
        job = FitJob(
            "local", None, None, "fold-0", "local_transformer", 7,
            (names[rank - 1],),
        )
        assert question_uses(job, names) == uses

    invalid = []
    duplicate = list(names)
    duplicate[-1] = duplicate[0]
    invalid.append((duplicate, coverage))
    reordered = dict(coverage)
    reordered["fold-0"] = tuple(reversed(reordered["fold-0"]))
    invalid.append((names, reordered))
    unknown = dict(coverage)
    unknown["fold-1"] = (*unknown["fold-1"], "UNKNOWN")
    invalid.append((names, unknown))
    no_core = dict(coverage)
    no_core["fold-0"] = tuple(
        name for name in no_core["fold-0"] if name != names[0]
    )
    invalid.append((names, no_core))
    no_unseen = dict(coverage)
    no_unseen["calibration"] = tuple(
        name for name in no_unseen["calibration"] if name != names[-1]
    )
    invalid.append((names, no_unseen))
    for master, phases in invalid:
        try:
            expected_fit_jobs(master, phases)
        except ValueError:
            continue
        raise AssertionError("invalid physical fit schedule was accepted")


def verify_runner_boundaries() -> None:
    assert "torch" not in sys.modules
    names = synthetic_master()
    coverage = synthetic_coverage(names)
    evaluable = {
        phase.phase: phase.evaluable for phase in coverage.phases
    }
    counts = runner.preflight_counts(coverage)
    assert counts.phases == (53, 52, 51)
    assert counts.fits == EXPECTED_FIT_COUNT
    assert counts.prediction_records == EXPECTED_PREDICTION_RECORDS
    assert counts.prediction_values == EXPECTED_PREDICTION_RECORDS
    assert runner.EXPECTED_PREFLIGHT_COUNTS.prediction_values == \
        EXPECTED_PREDICTION_VALUES

    phases = list(coverage.phases)
    series = list(phases[0].series)
    target = series[0]
    extra_rows = 6
    series[0] = SeriesCoverage(
        target.series, target.train_rows,
        target.validation_rows + extra_rows, target.timestamp_sha256,
    )
    phases[0] = PhaseCoverage(phases[0].phase, tuple(series))
    changed = runner.preflight_counts(ScalingCoverage(tuple(phases)))
    predictions_per_core_series = (
        len(MODES) * len(SEEDS) * len(TRAINING_COHORTS) *
        (len(POOLED_MODELS) + 1) +
        len(TRAINING_COHORTS) + len(SEEDS)
    )
    assert predictions_per_core_series == 129
    assert changed.prediction_records == counts.prediction_records
    assert changed.prediction_values - counts.prediction_values == \
        extra_rows * predictions_per_core_series
    assert tuple(
        job.phase for job in runner.phase_jobs(
            expected_fit_jobs(names, evaluable), "fold-1",
        )
    ) == ("fold-1",) * 405

    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-spools-",
    ) as directory_name:
        directory = Path(directory_name).resolve()
        spools = {
            phase: directory / f"{phase}.jsonl"
            for phase in ("fold-0", "fold-1")
        }
        spools["fold-0"].write_text(
            '{"index": 0}\n{"index": 2}\n', encoding="ascii",
        )
        spools["fold-1"].write_text(
            '{"index": 1}\n', encoding="ascii",
        )
        output = directory / "ordered.jsonl"
        runner.merge_spools(
            ("fold-0", "fold-1", "fold-0"), spools, output,
        )
        assert output.read_text(encoding="ascii") == (
            '{"index": 0}\n{"index": 1}\n{"index": 2}\n'
        )
        spools["fold-1"].write_text(
            '{"index": 1}\n{"index": 3}\n', encoding="ascii",
        )
        try:
            runner.merge_spools(
                ("fold-0", "fold-1", "fold-0"), spools,
                directory / "extra.jsonl",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("extra phase spool record was accepted")

        source = directory / "publish-source"
        target = directory / "publish-target"
        source.write_text("complete\n", encoding="ascii")
        directory_fd = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            runner._publish_complete(
                directory_fd, source.name, directory_fd, target.name,
            )
        finally:
            os.close(directory_fd)
        assert not source.exists()
        assert target.read_text(encoding="ascii") == "complete\n"

    orchestrator = SimpleNamespace(
        attempt_path="experiments/exact-attempt.json",
        commands={"validate": ("tools/run_universe_scaling.py", "validate",
                               "experiments/exact-attempt.json")},
        primary_python=SimpleNamespace(
            path=sys.executable, validate_live=lambda _: None,
        ),
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            sys, "argv",
            ["tools/run_universe_scaling.py",
             "experiments/exact-attempt.json"],
        ))
        stack.enter_context(patch.object(
            runner, "_require_isolated_execution",
        ))
        stack.enter_context(patch.object(runner, "_require_exact_launch"))
        stack.enter_context(patch.object(runner, "_validate_environment"))
        stack.enter_context(patch.object(runner, "_validate_source"))
        stack.enter_context(patch.object(runner, "_validate_stage_paths"))
        runner._validate_orchestrator(orchestrator)
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            sys, "argv", ["tools/run_universe_scaling.py", "other.json"],
        ))
        stack.enter_context(patch.object(
            runner, "_require_isolated_execution",
        ))
        stack.enter_context(patch.object(runner, "_require_exact_launch"))
        stack.enter_context(patch.object(runner, "_validate_source"))
        stack.enter_context(patch.object(runner, "_validate_stage_paths"))
        try:
            runner._validate_orchestrator(orchestrator)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong orchestrator argv was accepted")

    stage_attempt = SimpleNamespace(
        commands={
            "calibrate": (
                "tools/run_universe_scaling.py", "calibrate", "attempt.json",
            ),
        },
        torch_probe=SimpleNamespace(
            package_tree=SimpleNamespace(root="/attested/torch"),
        ),
    )
    order = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            sys, "argv", list(stage_attempt.commands["calibrate"]),
        ))
        stack.enter_context(patch.object(
            runner, "_require_isolated_execution",
            side_effect=lambda **_: order.append("isolated"),
        ))
        stack.enter_context(patch.object(
            runner, "_require_exact_launch",
            side_effect=lambda: order.append("launch"),
        ))
        for name, label in (
            ("_validate_environment", "environment"),
            ("_validate_runtime", "runtime"),
            ("_validate_source", "source"),
            ("_expose_torch_package", "expose"),
            ("_validate_stage_paths", "paths"),
        ):
            stack.enter_context(patch.object(
                runner, name,
                side_effect=lambda *_, label=label:
                order.append(label),
            ))
        runner._validate_stage(stage_attempt, "calibrate")
    assert order == [
        "isolated", "launch", "environment",
        "runtime", "source", "expose", "paths",
    ]

    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": "reports/mock/.pycache",
    }
    with patch.dict(runner.os.environ, environment, clear=True):
        runner._validate_environment(SimpleNamespace(environment=environment))
        try:
            runner._validate_environment(SimpleNamespace(environment={}))
        except ValueError:
            pass
        else:
            raise AssertionError("ambient runner environment was accepted")
    with patch.dict(
        runner.os.environ,
        environment | {
            "LC_CTYPE": "UTF-8",
            "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0",
        },
        clear=True,
    ):
        runner._validate_environment(SimpleNamespace(environment=environment))
    with patch.dict(
        runner.os.environ, environment | {"MASSIVE_API_KEY": "secret"},
        clear=True,
    ):
        try:
            runner._validate_environment(
                SimpleNamespace(environment=environment),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("ambient secret reached the runner")

    with tempfile.TemporaryDirectory(
        prefix="compose-mini-attested-torch-",
    ) as directory_name:
        directory = Path(directory_name)
        repository = directory / "repository"
        standard_library = directory / "stdlib"
        package = directory / "site-packages/torch"
        repository.mkdir()
        standard_library.mkdir()
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            "ORIGIN = 'attested'\n", encoding="ascii",
        )
        parent = str(package.parent.resolve())
        package_tree = SimpleNamespace(root=str(package.resolve()))
        for shadow in ("module", "package"):
            shadow_root = repository / shadow
            shadow_root.mkdir()
            marker = shadow_root / "shadow-ran"
            if shadow == "module":
                target = shadow_root / "torch.py"
            else:
                target = shadow_root / "torch/__init__.py"
                target.parent.mkdir()
            target.write_text(
                f"open({str(marker)!r}, 'w').write('ran')\n",
                encoding="ascii",
            )
            paths = [str(standard_library), str(shadow_root)]
            exposures = []
            runtime = SimpleNamespace(
                primary_python=SimpleNamespace(
                    path=sys.executable, validate_live=lambda _: None,
                ),
                torch_probe=SimpleNamespace(
                    python=SimpleNamespace(
                        path=sys.executable,
                        validate_live=lambda _:
                        exposures.append(parent in sys.path),
                    ),
                    package_tree=package_tree,
                ),
            )

            def attest(path: Path) -> object:
                assert path == package.resolve()
                assert parent not in sys.path
                return package_tree

            with patch.object(
                runner, "ROOT", shadow_root,
            ), patch.object(
                sys, "path", paths,
            ), patch.object(
                runner, "source_tree", side_effect=attest,
            ):
                runner._validate_runtime(runtime, "calibrate")
                runner._expose_torch_package(package)
                assert sys.path == [
                    str(standard_library), parent,
                ]
                imported = __import__("torch")
                assert imported.ORIGIN == "attested"
                assert Path(imported.__file__).resolve().parent == \
                    package.resolve()
                sys.modules.pop("torch")
            assert exposures == [False]
            assert not marker.exists()

        paths = [str(standard_library), str(repository)]
        runtime.torch_probe.python.validate_live = lambda _: None
        with patch.object(
            runner, "ROOT", repository,
        ), patch.object(
            sys, "path", paths,
        ), patch.object(
            runner, "source_tree", return_value=object(),
        ):
            try:
                runner._validate_runtime(runtime, "calibrate")
            except ValueError:
                pass
            else:
                raise AssertionError("changed Torch tree was accepted")
            assert parent not in sys.path

        (standard_library / "torch.py").write_text(
            "raise AssertionError('competing torch resolved')\n",
            encoding="ascii",
        )
        with patch.object(
            runner, "ROOT", repository,
        ), patch.object(
            sys, "path", paths,
        ), patch.object(
            runner, "source_tree", return_value=package_tree,
        ):
            try:
                runner._validate_runtime(runtime, "calibrate")
                runner._expose_torch_package(package)
            except ValueError:
                pass
            else:
                raise AssertionError("competing Torch resolver was accepted")
            assert parent not in sys.path

        paths.insert(1, parent)
        with patch.object(
            runner, "ROOT", repository,
        ), patch.object(
            sys, "path", paths,
        ), patch.object(
            runner, "source_tree", return_value=package_tree,
        ):
            try:
                runner._validate_runtime(runtime, "calibrate")
                runner._expose_torch_package(package)
            except ValueError:
                pass
            else:
                raise AssertionError("pre-exposed Torch path was accepted")

        sys.modules["torch.injected"] = object()
        try:
            with patch.object(
                runner, "ROOT", repository,
            ), patch.object(
                sys, "path", [str(standard_library), str(repository)],
            ), patch.object(
                runner, "source_tree", return_value=package_tree,
            ):
                try:
                    runner._validate_runtime(runtime, "calibrate")
                    runner._expose_torch_package(package)
                except ValueError:
                    pass
                else:
                    raise AssertionError("preloaded Torch was accepted")
        finally:
            sys.modules.pop("torch.injected")

    child = SimpleNamespace(wait=lambda timeout=None: 0)
    with patch.dict(
        runner.os.environ, {"MASSIVE_API_KEY": "secret"}, clear=True,
    ), patch.object(
        runner.subprocess, "Popen", return_value=child,
    ) as spawned:
        assert runner._run(("child",), environment) == 0
    assert spawned.call_args.kwargs["env"] == environment

    class Process:
        pid = 41

        def __init__(self, timeout: bool) -> None:
            self.timeout = timeout
            self.waits = 0

        def wait(self, timeout: int | None = None) -> None:
            self.waits += 1
            if self.timeout and timeout == 2 and self.waits == 1:
                raise subprocess.TimeoutExpired("runner", timeout)

    def leader_only(_pid: int, signum: int) -> None:
        if signum == 0:
            raise ProcessLookupError

    process = Process(False)
    with patch.object(
        runner.os, "killpg", side_effect=leader_only,
    ) as killed:
        runner._terminate(process)
    assert [call.args for call in killed.call_args_list] == [
        (41, signal.SIGTERM), (41, 0),
    ]
    assert process.waits == 1

    process = Process(False)
    with patch.object(runner.os, "killpg") as killed:
        runner._terminate(process)
    assert [call.args for call in killed.call_args_list] == [
        (41, signal.SIGTERM), (41, 0), (41, signal.SIGKILL),
    ]
    assert process.waits == 1

    process = Process(True)
    with patch.object(runner.os, "killpg") as killed:
        runner._terminate(process)
    assert [call.args for call in killed.call_args_list] == [
        (41, signal.SIGTERM), (41, 0), (41, signal.SIGKILL),
    ]
    assert process.waits == 2

    with tempfile.TemporaryDirectory(
        prefix="compose-mini-process-group-",
    ) as directory_name:
        heartbeat = Path(directory_name) / "heartbeat"
        descendant = (
            "from pathlib import Path\n"
            "import signal,time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"path=Path({str(heartbeat)!r})\n"
            "path.write_text(str(time.monotonic_ns()))\n"
            "print('ready', flush=True)\n"
            "while True:\n"
            " path.write_text(str(time.monotonic_ns()))\n"
            " time.sleep(0.01)\n"
        )
        leader = (
            "import signal,subprocess,sys\n"
            "def stop(*_): raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            f"child=subprocess.Popen([sys.executable,'-c',{descendant!r}],"
            "stdout=subprocess.PIPE,text=True)\n"
            "assert child.stdout.readline().strip() == 'ready'\n"
            "print(child.pid, flush=True)\n"
            "signal.pause()\n"
        )
        process = subprocess.Popen(
            (sys.executable, "-c", leader),
            stdout=subprocess.PIPE, text=True, start_new_session=True,
        )
        assert process.stdout is not None
        process.stdout.readline()
        try:
            runner._terminate(process)
            time.sleep(0.05)
            first = heartbeat.read_text()
            time.sleep(0.05)
            assert heartbeat.read_text() == first, \
                "TERM-ignoring descendant survived"
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    attempt = SimpleNamespace(
        commands={
            stage: ("tools/run_universe_scaling.py", stage, "attempt.json")
            for stage in ("validate", "preflight", "calibrate", "analyze")
        } | {
            "finalizer_prefix": (
                "tools/finalize_universe_scaling.py",
                "attempt.json", "outcome.json",
            ),
        },
        environment=environment,
        outputs={"outcome": "outcome.json"},
        primary_python=SimpleNamespace(path="/bound/primary"),
        protocol={
            "finalizer_python_flags": FINALIZER_PYTHON_FLAGS,
            "runner_primary_python_flags": RUNNER_PRIMARY_PYTHON_FLAGS,
            "runner_torch_python_flags": RUNNER_TORCH_PYTHON_FLAGS,
        },
        run_dir="reports/mock",
        torch_argv=("/bound/torch",),
    )

    def controller_case(
        exits: Mapping[str, int] = {},
        errors: tuple[str, ...] = (),
        setup_error: bool = False,
        mask_after_analysis: bool = False,
        signal_stage: str | None = None,
    ) -> tuple[int, list[tuple[object, ...]], object, list[object]]:
        commands: list[tuple[object, ...]] = []
        finalizers: list[tuple[object, ...]] = []
        handlers: dict[int, object] = {}
        signal_changes: list[object] = []
        analysis_finished = failed_mask = False

        def command_stage(command: object) -> str:
            values = tuple(command)
            if attempt.commands["finalizer_prefix"][0] in values:
                return "finalizer"
            return next(
                name for name in (
                    "validate", "preflight", "calibrate", "analyze",
                ) if name in values
            )

        def run(command: object, child_environment: object) -> int:
            nonlocal analysis_finished
            values = tuple(command)
            commands.append(values)
            assert dict(child_environment) == environment
            stage = command_stage(values)
            if stage == "finalizer":
                finalizers.append(values)
            if signal_stage == stage:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            if stage in errors:
                raise RuntimeError(f"{stage} failed")
            if stage == "analyze":
                analysis_finished = True
            return exits.get(stage, 0)

        def set_handler(signum: int, handler: object) -> None:
            handlers[signum] = handler
            signal_changes.append(handler)

        def mask(how: int, _signals: object) -> set[int]:
            nonlocal failed_mask
            if mask_after_analysis and analysis_finished and \
                    how == signal.SIG_BLOCK and not failed_mask:
                failed_mask = True
                raise OSError("post-analysis mask failed")
            return set()

        with ExitStack() as stack:
            stack.enter_context(patch.object(
                runner, "read_attempt", return_value=attempt,
            ))
            stack.enter_context(patch.object(
                runner, "_validate_orchestrator",
            ))
            mkdir = stack.enter_context(patch.object(
                runner, "mkdir_nofollow",
                side_effect=OSError("setup failed") if setup_error else None,
            ))
            stack.enter_context(patch.object(
                runner, "_run", side_effect=run,
            ))
            stack.enter_context(patch.object(
                runner, "_utc_now",
                side_effect=("2026-07-24T00:00:00Z",
                             "2026-07-24T00:01:00Z"),
            ))
            stack.enter_context(patch.object(
                runner.signal, "getsignal",
                side_effect=lambda signum: f"original-{signum}",
            ))
            stack.enter_context(patch.object(
                runner.signal, "signal", side_effect=set_handler,
            ))
            stack.enter_context(patch.object(
                runner.signal, "pthread_sigmask", side_effect=mask,
            ))
            stack.enter_context(patch.object(
                runner.signal, "sigpending", return_value=set(),
            ))
            result = runner.execute(Path("attempt.json"))
        assert len(finalizers) == 1
        assert all(handler != signal.SIG_IGN for handler in signal_changes)
        return result, finalizers, mkdir, commands

    result, finalizers, mkdir, commands = controller_case(
        {"analyze": 3},
    )
    assert result == 3
    mkdir.assert_called_once_with(ROOT / "reports/mock")
    assert tuple(commands[:-1]) == (
        (
            "/bound/primary", *RUNNER_PRIMARY_PYTHON_FLAGS,
            *attempt.commands["validate"],
        ),
        (
            "/bound/primary", *RUNNER_PRIMARY_PYTHON_FLAGS,
            *attempt.commands["preflight"],
        ),
        (
            "/bound/torch", *RUNNER_TORCH_PYTHON_FLAGS,
            *attempt.commands["calibrate"],
        ),
        (
            "/bound/primary", *RUNNER_PRIMARY_PYTHON_FLAGS,
            *attempt.commands["analyze"],
        ),
    )
    assert finalizers[0][-6:] == (
        "--stage", "analysis", "--exit", "3",
        "--status", "gate-failure",
    )

    cases = (
        ({}, (), True, 1, "setup", 1, "setup-failure"),
        ({"preflight": 1}, (), False, 1, "preflight", 1,
         "preflight-failure"),
        ({"calibrate": 4}, (), False, 4, "experiment", 4,
         "experiment-failure"),
        ({"analyze": 2}, (), False, 2, "analysis", 2,
         "analysis-integrity-failure"),
        ({}, ("analyze",), False, 2, "analysis", 2,
         "analysis-integrity-failure"),
        ({"finalizer": 5}, (), False, 5, "analysis", 0, "pass"),
        ({}, ("finalizer",), False, 2, "analysis", 0, "pass"),
    )
    for (
        exits, errors, setup_error, expected, stage, terminal, status,
    ) in cases:
        result, finalizers, _, _ = controller_case(
            exits, errors, setup_error,
        )
        assert result == expected
        assert finalizers[0][-6:] == (
            "--stage", stage, "--exit", str(terminal),
            "--status", status,
        )

    for analysis_exit in (0, 3):
        result, finalizers, _, _ = controller_case(
            {"analyze": analysis_exit}, mask_after_analysis=True,
        )
        assert result == 2
        assert finalizers[0][-6:] == (
            "--stage", "analysis", "--exit", "2",
            "--status", "analysis-integrity-failure",
        )

    result, finalizers, _, _ = controller_case(
        signal_stage="finalizer",
    )
    assert result == 128 + signal.SIGTERM
    assert finalizers[0][-6:] == (
        "--stage", "analysis", "--exit", "0", "--status", "pass",
    )
    result, finalizers, _, _ = controller_case(
        signal_stage="validate",
    )
    assert result == 128 + signal.SIGTERM
    assert finalizers[0][-6:] == (
        "--stage", "preflight", "--exit", str(128 + signal.SIGTERM),
        "--status", "preflight-failure",
    )


def verify_finalizer_signal_subprocess() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-finalizer-signal-",
    ) as directory_name:
        directory = Path(directory_name).resolve()
        started = directory / "started"
        release = directory / "release"
        completed = directory / "completed"
        log = directory / "stages.log"
        stage_script = directory / "stage.py"
        stage_script.write_text(
            "from pathlib import Path\n"
            "import sys,time\n"
            "stage=sys.argv[1]\n"
            f"log=Path({str(log)!r})\n"
            "with log.open('a') as file: file.write(stage+'\\n')\n"
            "if stage == 'finalizer':\n"
            f" Path({str(started)!r}).open('x').write('started')\n"
            f" release=Path({str(release)!r})\n"
            " deadline=time.monotonic()+5\n"
            " while not release.exists():\n"
            "  if time.monotonic()>=deadline: raise SystemExit(2)\n"
            "  time.sleep(0.01)\n"
            f" Path({str(completed)!r}).open('x').write('completed')\n",
            encoding="ascii",
        )
        commands = {
            stage: (str(stage_script), stage)
            for stage in ("validate", "preflight", "calibrate", "analyze")
        } | {"finalizer_prefix": (str(stage_script), "finalizer")}
        (directory / "reports").mkdir()
        script = (
            "from contextlib import ExitStack\n"
            "from pathlib import Path\n"
            "from types import SimpleNamespace\n"
            "from unittest.mock import patch\n"
            "import sys\n"
            f"sys.path.insert(0,{str(ROOT)!r})\n"
            "import tools.run_universe_scaling as runner\n"
            f"commands={commands!r}\n"
            "attempt=SimpleNamespace("
            "commands=commands,environment={},run_dir='reports/mock',"
            "primary_python=SimpleNamespace(path=sys.executable),"
            f"protocol={{'finalizer_python_flags':"
            f"{FINALIZER_PYTHON_FLAGS!r},"
            f"'runner_primary_python_flags':"
            f"{RUNNER_PRIMARY_PYTHON_FLAGS!r},"
            f"'runner_torch_python_flags':"
            f"{RUNNER_TORCH_PYTHON_FLAGS!r}}},"
            "torch_argv=(sys.executable,))\n"
            "with ExitStack() as stack:\n"
            f" stack.enter_context(patch.object(runner,'ROOT',"
            f"Path({str(directory)!r})))\n"
            " stack.enter_context(patch.object("
            "runner,'read_attempt',return_value=attempt))\n"
            " stack.enter_context(patch.object("
            "runner,'_validate_orchestrator'))\n"
            " code=runner.execute(Path('attempt.json'))\n"
            "raise SystemExit(code)\n"
        )
        process = subprocess.Popen(
            (sys.executable, "-c", script),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 3
            while not started.exists():
                if process.poll() is not None:
                    output, error = process.communicate()
                    raise AssertionError((process.returncode, output, error))
                if time.monotonic() >= deadline:
                    raise AssertionError("finalizer did not start")
                time.sleep(0.01)
            os.kill(process.pid, signal.SIGTERM)
            time.sleep(0.05)
            release.write_text("release", encoding="ascii")
            output, error = process.communicate(timeout=5)
            assert process.returncode == 128 + signal.SIGTERM, (
                process.returncode, output, error,
            )
            assert started.read_text() == "started"
            assert completed.read_text() == "completed"
            assert log.read_text().splitlines() == [
                "validate", "preflight", "calibrate",
                "analyze", "finalizer",
            ]
        finally:
            release.touch(exist_ok=True)
            if process.poll() is None:
                process.kill()
                process.wait()


def verify_market_truth_derivation() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-truth-",
    ) as directory_name:
        root = Path(directory_name)
        names = synthetic_master()
        manifest = root / "manifest.json"
        calendar = root / "calendar.json"
        csv_paths = tuple(root / f"{name}.csv" for name in names)
        write_json(manifest, {
            "adjusted": True,
            "declared_on": "2025-12-01",
            "eligibility_date": "2025-12-01",
            "end": "2026-01-03",
            "interval_minutes": 30,
            "purpose": "test",
            "schema": 1,
            "series": [
                {"stratum": "test", "ticker": name} for name in names
            ],
            "session": "regular",
            "start": "2026-01-01",
        })
        write_json(calendar, {"test": True})
        csv = (
            "timestamp,open,high,low,close,volume\n"
            "2026-01-01T00:00:00Z,99,101,98,100,1\n"
            "2026-01-02T00:00:00Z,101,103,100,102,1\n"
            "2026-01-03T00:00:00Z,103,105,102,104,1\n"
        )
        for path in csv_paths:
            path.write_text(csv, encoding="ascii")
        missing = dict(EXPECTED_MISSING)
        truth_row = (
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
        )
        coverage = ScalingCoverage(tuple(
            PhaseCoverage(phase, tuple(
                SeriesCoverage(
                    name, 1, int(name not in missing[phase]),
                    timestamp_grid_sha256((truth_row,))
                    if name not in missing[phase] else
                    timestamp_grid_sha256(()),
                )
                for name in names
            ))
            for phase in PHASES
        ))
        calls = 0
        sample_stops = []
        calibration_stop = expected_protocol()[
            "calendar"
        ]["calibration"][-1][-1]
        assert calibration_stop == 4_943

        def sampled(
            *_: object, opportunity_stop: int | None = None,
        ) -> SessionSamples:
            sample_stops.append(opportunity_stop)
            return SessionSamples(
                (SampleRows(0, 1, 2, 0),), 5_505,
            )

        def packed(*_: object) -> PackedRows:
            nonlocal calls
            name = names[calls // len(PHASES)]
            phase = PHASES[calls % len(PHASES)]
            validation = int(name not in missing[phase])
            calls += 1
            row = SampleRows(0, 1, 2, 0)
            return PackedRows(
                (row, row) if validation else (row,),
                (1, validation),
            )

        with file_tools.freeze_inputs(
            (manifest, calendar, *csv_paths),
        ) as frozen:
            csv_paths[0].write_text(
                csv.replace(",101,103,100,102,", ",999,1000,998,999,"),
                encoding="ascii",
            )
            with patch.object(
                finalizer.SessionCalendar, "read",
                return_value=SimpleNamespace(),
            ), patch.object(
                finalizer, "session_samples", side_effect=sampled,
            ), patch.object(
                finalizer, "pack_rows", side_effect=packed,
            ):
                derived = finalizer.derive_market_truth(
                    frozen[0].snapshot, frozen[1].snapshot,
                    {
                        name: item.snapshot
                        for name, item in zip(
                            names, frozen[2:], strict=True,
                        )
                    },
                    coverage, expected_protocol(),
                )
        assert calls == 55 * len(PHASES)
        assert sample_stops == [calibration_stop] * 55
        assert derived.coverage == coverage
        assert tuple(
            len(phase.evaluable) for phase in derived.coverage.phases
        ) == (53, 52, 51)
        assert finalizer.LOCKS["reserved_test_materialized_samples"] == 0
        first = derived.rows[("fold-0", names[0])][0]
        assert (
            first.reference_price, first.outcome_price, first.actual_return,
        ) == (101.0, 104.0, log(104.0 / 101.0))

        changed = list(coverage.phases)
        first_phase = changed[0]
        first_series = list(first_phase.series)
        first_series[0] = SeriesCoverage(
            first_series[0].series, first_series[0].train_rows,
            first_series[0].validation_rows, "0" * 64,
        )
        changed[0] = PhaseCoverage(first_phase.phase, tuple(first_series))
        calls = 0
        sample_stops.clear()
        with patch.object(
            finalizer.SessionCalendar, "read",
            return_value=SimpleNamespace(),
        ), patch.object(
            finalizer, "session_samples", side_effect=sampled,
        ), patch.object(finalizer, "pack_rows", side_effect=packed):
            try:
                finalizer.derive_market_truth(
                    manifest, calendar,
                    dict(zip(names, csv_paths, strict=True)),
                    ScalingCoverage(tuple(changed)), expected_protocol(),
                )
            except ValueError:
                pass
            else:
                raise AssertionError("changed armed truth was accepted")
        assert sample_stops == [calibration_stop] * 55


def scaling_finalizer_fixture() -> tuple[
    tuple[str, ...], ScalingCoverage,
    list[dict[str, object]], list[dict[str, object]], MarketTruth,
]:
    names = synthetic_master()
    market = (
        PredictionTruth(
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
            100.0, 104.0, log(104.0 / 100.0),
        ),
        PredictionTruth(
            "2026-01-04T00:00:00Z",
            "2026-01-05T00:00:00Z",
            "2026-01-06T00:00:00Z",
            100.0, 98.0, log(98.0 / 100.0),
        ),
    )
    grid = tuple(
        (row.as_of, row.entry_time, row.target_time) for row in market
    )
    coverage = ScalingCoverage(tuple(
        PhaseCoverage(phase.phase, tuple(
            SeriesCoverage(
                item.series, item.train_rows,
                2 if item.validation_rows else 0,
                timestamp_grid_sha256(grid)
                if item.validation_rows else timestamp_grid_sha256(()),
            )
            for item in phase.series
        ))
        for phase in synthetic_coverage(names).phases
    ))
    evaluable = {
        phase.phase: phase.evaluable for phase in coverage.phases
    }
    rows = {
        (phase.phase, item.series): (
            item.train_rows, item.validation_rows,
        )
        for phase in coverage.phases for item in phase.series
    }
    fits = []
    for job in expected_fit_jobs(names, evaluable):
        provenance = fit_provenance_id(job)
        fixed = job.kind == "pooled" and job.mode == "fixed-update"
        epoch = job.kind == "local" or job.mode == "fixed-epoch"
        fit = (
            SimpleNamespace(best_checkpoint=1) if fixed else
            SimpleNamespace(best_epoch=1, epochs_trained=11)
            if epoch else None
        )
        fits.append(runner._fit_record(
            job,
            hashlib.sha256(f"model:{provenance}".encode()).hexdigest(),
            fit, rows, names,
        ))
    closure = validate_fit_ledger(fits, names, coverage)
    model_prediction = {
        "global_ridge": 0.01,
        "global_mlp": 0.02,
        "panel_transformer": 0.03,
        "conditioned_panel_transformer": 0.025,
        "local_transformer": 0.015,
    }
    predictions = []
    for job in closure.jobs:
        provenance = fit_provenance_id(job)
        predicted = model_prediction[job.model] + (
            0.0001 * SEEDS.index(job.seed)
            if job.seed is not None else 0.0
        )
        for series in required_prediction_series(
            job, names, closure.evaluable,
        ):
            predictions.append({
                "grid_sha256": closure.timestamp_sha256[
                    (job.phase, series)
                ],
                "model_fingerprint": closure.fingerprints[provenance],
                "phase": job.phase,
                "predictions": encode_f32le_base64(tuple(
                    predicted + 0.001 * index
                    for index in range(
                        closure.rows[(job.phase, series)][1],
                    )
                )),
                "provenance_id": provenance,
                "schema": 2,
                "series": series,
            })
    truth = MarketTruth(coverage, {
        (phase.phase, item.series):
        market if item.validation_rows else ()
        for phase in coverage.phases for item in phase.series
    })
    return names, coverage, fits, predictions, truth


def reject_finalizer_fit(
    values: list[dict[str, object]], names: tuple[str, ...],
    coverage: ScalingCoverage,
) -> None:
    try:
        validate_fit_ledger(values, names, coverage)
    except ValueError:
        return
    raise AssertionError("invalid fit ledger was accepted")


def reject_finalizer_predictions(
    values: Iterable[Mapping[str, object]], closure: object,
    truth: MarketTruth,
) -> None:
    try:
        validate_prediction_ledger(values, closure, truth)
    except ValueError:
        return
    raise AssertionError("invalid prediction ledger was accepted")


def prediction_record_index(
    values: list[dict[str, object]], closure: object, series: str,
    **axes: object,
) -> int:
    """Locate one physical prediction by its fit axes and destination."""
    matches = tuple(
        index for index, record in enumerate(values)
        if record["series"] == series and all(
            getattr(
                closure.jobs_by_id[record["provenance_id"]], name,
            ) == value
            for name, value in axes.items()
        )
    )
    if len(matches) != 1:
        raise AssertionError(f"expected one physical prediction: {matches}")
    return matches[0]


def replace_prediction(
    values: list[dict[str, object]], index: int, value: float,
) -> list[dict[str, object]]:
    """Return a ledger copy with one complete prediction vector replaced."""
    changed = deepcopy(values)
    count = changed[index]["predictions"]["count"]
    changed[index]["predictions"] = encode_f32le_base64(
        (value,) * count,
    )
    return changed


def fake_paired_comparison(
    candidate: dict[str, tuple[ForecastPoint, ...]],
    reference: dict[str, tuple[ForecastPoint, ...]],
    **_: object,
) -> dict[str, object]:
    gains = {
        name: sum(
            abs(right.actual_return - right.predicted_return) -
            abs(left.actual_return - left.predicted_return)
            for left, right in zip(
                candidate[name], reference[name], strict=True,
            )
        ) / len(candidate[name])
        for name in candidate
    }
    mean = sum(gains.values()) / len(gains)
    return {
        "common_dates": ("2026-01-03",),
        "mean_gain": mean,
        "per_stock_mean_gain": gains,
        "wins": sum(value > 0 for value in gains.values()),
        "ties": sum(value == 0 for value in gains.values()),
        "losses": sum(value < 0 for value in gains.values()),
        "intervals": {
            str(block): (mean, mean) for block in (5, 10, 20)
        },
        "effective_count": {
            "value": None,
            "included": tuple(gains),
            "excluded": (),
            "reason": "synthetic",
        },
    }


def verify_scaling_finalizer() -> None:
    names, coverage, fits, predictions, truth = scaling_finalizer_fixture()
    assert tuple(
        len(phase.evaluable) for phase in coverage.phases
    ) == (53, 52, 51)
    closure = validate_fit_ledger(fits, names, coverage)
    assert len(closure.jobs) == EXPECTED_FIT_COUNT
    parsed = validate_prediction_ledger(
        (record for record in predictions), closure, truth,
    )

    invalid_fits = (
        fits[:-1],
        [*fits, fits[-1]],
        [fits[1], fits[0], *fits[2:]],
        [{**fits[0], "provenance_id": "0" * 64}, *fits[1:]],
        [{**fits[0], "policy_selected": True}, *fits[1:]],
        [{
            **fits[0],
            "coverage": [{
                **fits[0]["coverage"][0], "validation_rows": 0,
            }, *fits[0]["coverage"][1:]],
        }, *fits[1:]],
        [{
            **fits[0],
            "coverage": [{
                **fits[0]["coverage"][0],
                "train_rows": fits[0]["coverage"][0]["train_rows"] + 1,
            }, *fits[0]["coverage"][1:]],
        }, *fits[1:]],
    )
    for invalid in invalid_fits:
        reject_finalizer_fit(invalid, names, coverage)

    epoch_indices = (
        next(
            index for index, item in enumerate(fits)
            if item["kind"] == "pooled" and item["mode"] == "fixed-epoch"
        ),
        next(
            index for index, item in enumerate(fits)
            if item["kind"] == "local"
        ),
    )

    def epoch_record(
        source: Mapping[str, object], selected: int, trained: int,
    ) -> dict[str, object]:
        coverage_rows = source["coverage"]
        rows_per_epoch = (
            sum(item["train_rows"] for item in coverage_rows) +
            FIXED_EPOCH_BUDGET.batch_size - 1
        ) // FIXED_EPOCH_BUDGET.batch_size
        return {
            **source,
            "epochs_trained": trained,
            "optimizer_updates": trained * rows_per_epoch,
            "selected_epoch": selected,
        }

    for index in epoch_indices:
        boundary = deepcopy(fits)
        boundary[index] = epoch_record(boundary[index], 90, 100)
        validate_fit_ledger(boundary, names, coverage)
        for mutate in (
            lambda item: item["budget"].update(epochs=99),
            lambda item: item.update(epochs_trained=0, optimizer_updates=0),
            lambda item: item.update(epochs_trained=101),
            lambda item: item.update(selected_epoch=0),
            lambda item: item.update(selected_epoch=12),
            lambda item: item.update(
                selected_epoch=2, epochs_trained=11,
            ),
            lambda item: item.update(
                selected_epoch=1, epochs_trained=12,
            ),
            lambda item: item.update(optimizer_updates=(
                item["optimizer_updates"] + 1
            )),
        ):
            invalid = deepcopy(fits)
            mutate(invalid[index])
            reject_finalizer_fit(invalid, names, coverage)

    for index in (
        next(
            index for index, item in enumerate(fits)
            if item["kind"] == "ridge"
        ),
        next(
            index for index, item in enumerate(fits)
            if item["mode"] == "fixed-update"
        ),
    ):
        invalid = deepcopy(fits)
        invalid[index]["epochs_trained"] = 1
        reject_finalizer_fit(invalid, names, coverage)

    assert len(predictions) == EXPECTED_PREDICTION_RECORDS == 14_216
    assert set(predictions[0]) == finalizer.PREDICTION_FIELDS
    assert parsed.records == EXPECTED_PREDICTION_RECORDS
    assert parsed.stored_values == sum(
        record["predictions"]["count"] for record in predictions
    ) == 2 * EXPECTED_PREDICTION_RECORDS
    assert parsed.stored_values != parsed.records
    assert parsed.synthesized_zero_values == sum(
        len(rows) for rows in truth.rows.values()
    ) == 312

    physical_keys = {
        (*finalizer._family(job), series)
        for job in closure.jobs
        for series in required_prediction_series(
            job, names, closure.evaluable,
        )
    }
    zero_series = sum(bool(rows) for rows in truth.rows.values())
    assert len(parsed.metrics) == len(physical_keys) + zero_series
    assert set(parsed.calibration) == {
        key for key in physical_keys if key[3] == "calibration"
    }
    assert all(key[3] == "calibration" for key in parsed.calibration)
    assert {key[3] for key in parsed.metrics} == set(PHASES)

    ensemble_series = names[44]
    ensemble_jobs = tuple(
        job for job in closure.jobs
        if (
            job.kind, job.mode, job.cohort, job.phase, job.model
        ) == (
            "pooled", "fixed-update", 44, "calibration",
            "panel_transformer",
        )
    )
    assert tuple(job.seed for job in ensemble_jobs) == SEEDS
    by_identity = {
        (record["provenance_id"], record["series"]): record
        for record in predictions
    }
    seed_predictions = tuple(
        decode_f32le_base64(by_identity[
            (fit_provenance_id(job), ensemble_series)
        ]["predictions"])
        for job in ensemble_jobs
    )
    expected_ensemble = tuple(
        fmean(values)
        for values in zip(*seed_predictions, strict=True)
    )
    ensemble_key = (*finalizer._family(ensemble_jobs[0]), ensemble_series)
    assert parsed.calibration[ensemble_key] == expected_ensemble

    first_key, first_values = next(iter(parsed.calibration.items()))
    expected_truth = truth.rows[(first_key[3], first_key[-1])][0]
    first_point = finalizer._points(
        truth.rows[(first_key[3], first_key[-1])], first_values,
    )[0]
    assert (
        first_point.target_time, first_point.actual_return,
        first_point.reference_price, first_point.outcome_price,
    ) == (
        expected_truth.target_time, expected_truth.actual_return,
        expected_truth.reference_price, expected_truth.outcome_price,
    )

    try:
        finalizer._family_key(
            parsed, "unseen-transfer", "fixed-update", 44,
            "calibration", ensemble_series,
            "conditioned_panel_transformer",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("conditioned unseen prediction was accepted")

    def changed_record(**fields: object) -> list[dict[str, object]]:
        changed = deepcopy(predictions)
        changed[0].update(fields)
        return changed

    def changed_payload(**fields: object) -> list[dict[str, object]]:
        changed = deepcopy(predictions)
        changed[0]["predictions"].update(fields)
        return changed

    wrong_phase = next(
        phase for phase in PHASES if phase != predictions[0]["phase"]
    )
    wrong_series = next(
        name for name in names if name != predictions[0]["series"]
    )
    invalid_predictions = (
        lambda: predictions[1:],
        lambda: [*predictions, predictions[-1]],
        lambda: [predictions[1], predictions[0], *predictions[2:]],
        lambda: changed_record(schema=1),
        lambda: changed_record(provenance_id="0" * 64),
        lambda: changed_record(model_fingerprint="0" * 64),
        lambda: changed_record(phase=wrong_phase),
        lambda: changed_record(series=wrong_series),
        lambda: changed_record(grid_sha256="0" * 64),
        lambda: changed_record(backtest_run=False),
        lambda: changed_payload(encoding="f64le-base64"),
        lambda: changed_payload(count=0),
        lambda: changed_payload(base64="!"),
        lambda: changed_payload(unexpected=True),
    )
    for invalid in invalid_predictions:
        reject_finalizer_predictions(invalid(), closure, truth)

    for value in (float("nan"), float("inf"), -float("inf")):
        encoded = base64.b64encode(
            struct.pack("<2f", value, value),
        ).decode("ascii")
        reject_finalizer_predictions(
            changed_record(predictions={
                "encoding": "f32le-base64",
                "count": 2,
                "base64": encoded,
            }),
            closure, truth,
        )
    reject_finalizer_predictions(
        changed_record(predictions={
            "encoding": "f32le-base64",
            "count": 2,
            "base64": base64.b64encode(b"\0" * 9).decode("ascii"),
        }),
        closure, truth,
    )

    truth_rows = dict(truth.rows)
    truth_key = next(key for key, rows in truth_rows.items() if rows)
    row = truth_rows[truth_key][0]
    truth_rows[truth_key] = (PredictionTruth(
        row.as_of, row.entry_time, "2026-01-04T00:00:00Z",
        row.reference_price, row.outcome_price, row.actual_return,
    ), *truth_rows[truth_key][1:])
    reject_finalizer_predictions(
        predictions, closure, MarketTruth(coverage, truth_rows),
    )

    finite_index = prediction_record_index(
        predictions, closure, names[44], kind="pooled",
        mode="fixed-update", cohort=44, phase="fold-0",
        model="panel_transformer", seed=SEEDS[0],
    )
    finite = validate_prediction_ledger(
        replace_prediction(predictions, finite_index, 0.05),
        closure, truth,
    )
    finite_key = (
        *finalizer._family(
            closure.jobs_by_id[predictions[finite_index]["provenance_id"]],
        ),
        names[44],
    )
    assert finite.metrics[finite_key] != parsed.metrics[finite_key]

    with patch(
        "tools.finalize_universe_scaling.paired_comparison",
        fake_paired_comparison,
    ):
        baseline = build_development_summary(closure, parsed)
        assert baseline["prediction_evidence"] == {
            "schema": 2,
            "records": EXPECTED_PREDICTION_RECORDS,
            "stored_values": 2 * EXPECTED_PREDICTION_RECORDS,
            "synthesized_zero_values": 312,
        }
        assert baseline["model_binding_role"] == \
            "cross-ledger-consistency-not-independent-execution-proof"
        assert all(
            "conditioned_panel_transformer" not in view["metrics"]
            for result in baseline["results"]
            if result["question"] == "unseen-transfer"
            for view in result["views"].values()
        )
        paired = baseline["paired_calibration"]
        assert tuple(paired) == ("fixed-update", "fixed-epoch")
        comparison_count = 0
        for mode in paired:
            evidence = paired[mode]
            assert tuple(evidence["candidate_vs_baselines"]) == (
                "core:55", "unseen:44",
            )
            assert all(
                tuple(models) == finalizer.CONTROL_MODELS
                for models in evidence["candidate_vs_baselines"].values()
            )
            assert tuple(evidence["breadth_vs_11"]["core"]) == (
                "22", "33", "55",
            )
            assert tuple(evidence["breadth_vs_11"]["unseen"]) == (
                "22", "33", "44",
            )
            comparisons = (
                *(
                    item
                    for models in
                    evidence["candidate_vs_baselines"].values()
                    for item in models.values()
                ),
                *evidence["breadth_vs_11"]["core"].values(),
                *evidence["breadth_vs_11"]["unseen"].values(),
                evidence["unseen_44_vs_33"],
            )
            assert len(comparisons) == 15
            comparison_count += len(comparisons)
            assert all(
                tuple(item["intervals"]) == ("5", "10", "20") and
                "effective_count" in item
                for item in comparisons
            )
        assert comparison_count == 30
        assert len({
            name for name in baseline["gates"] if name != "all_pass"
        }) == 8
        expansion = paired["fixed-update"][
            "breadth_vs_11"
        ]["unseen"]["44"]
        marginal = paired["fixed-update"]["unseen_44_vs_33"]
        assert baseline["gates"]["positive_paired_intervals"][
            "intervals"
        ] == expansion["intervals"]
        assert baseline["gates"]["majority_unseen_improved"]["wins"] == \
            expansion["wins"]
        assert baseline["gates"]["unseen_33_to_44_marginal"][
            "mean_gain"
        ] == marginal["mean_gain"]

        changed_fold = build_development_summary(
            closure, validate_prediction_ledger(
                replace_prediction(predictions, finite_index, 0.05),
                closure, truth,
            ),
        )
        assert changed_fold["results"] != baseline["results"]
        assert changed_fold["gates"] == baseline["gates"]
        assert changed_fold["paired_calibration"] == paired

        fixed_epoch_index = prediction_record_index(
            predictions, closure, names[44], kind="pooled",
            mode="fixed-epoch", cohort=44, phase="calibration",
            model="panel_transformer", seed=SEEDS[0],
        )
        changed_fixed_epoch = build_development_summary(
            closure, validate_prediction_ledger(
                replace_prediction(
                    predictions, fixed_epoch_index, -0.5,
                ),
                closure, truth,
            ),
        )
        assert changed_fixed_epoch["gates"] == baseline["gates"]
        assert changed_fixed_epoch["paired_calibration"][
            "fixed-update"
        ] == paired["fixed-update"]
        assert changed_fixed_epoch["paired_calibration"][
            "fixed-epoch"
        ] != paired["fixed-epoch"]

        gated_index = prediction_record_index(
            predictions, closure, names[44], kind="pooled",
            mode="fixed-update", cohort=44, phase="calibration",
            model="panel_transformer", seed=SEEDS[0],
        )
        changed_gate = build_development_summary(
            closure, validate_prediction_ledger(
                replace_prediction(predictions, gated_index, -0.5),
                closure, truth,
            ),
        )
        assert changed_gate["gates"] != baseline["gates"]
        assert baseline["locks"] == {
            "reserved_test_materialized_samples": 0,
            "policy_selected": False,
            "backtest_run": False,
            "trading_authorized": False,
        }

    transitions = (
        ("preflight", 2, "preflight-failure"),
        ("setup", 2, "setup-failure"),
        ("experiment", 2, "experiment-failure"),
        ("analysis", 2, "analysis-integrity-failure"),
        ("analysis", 3, "gate-failure"),
        ("analysis", 0, "pass"),
        ("experiment", 129, "experiment-failure"),
        ("experiment", 130, "experiment-failure"),
        ("experiment", 143, "experiment-failure"),
    )
    for stage, code, status in transitions:
        _transition(stage, code, status)
    try:
        _transition("experiment", 0, "pass")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid terminal transition was accepted")


def passing_gate_inputs() -> dict[str, object]:
    metrics = {
        "return_mae": 0.9,
        "direction_accuracy": 0.8,
        "close_mae": 0.8,
    }
    controls = {
        view: {
            model: {
                "return_mae": 1.1,
                "direction_accuracy": 0.5,
                "close_mae": 1.0,
            }
            for model in finalizer.CONTROL_MODELS
        }
        for view in ("core", "unseen")
    }
    return {
        "unseen_metrics": dict(metrics),
        "unseen_control_metrics": {
            **metrics, "return_mae": 1.0,
        },
        "core_metrics": dict(metrics),
        "core_control_metrics": {
            **metrics, "return_mae": 1.0,
        },
        "expansion": {
            "intervals": {
                str(block): (0.1, 0.2) for block in (5, 10, 20)
            },
            "wins": 6,
            "per_stock_mean_gain": {
                f"S{index:02d}": 0.1 for index in range(11)
            },
        },
        "marginal": {"mean_gain": 0.0},
        "control_metrics": controls,
        "core_majority": 0.7,
        "unseen_majority": 0.7,
    }


def evaluate_gates(values: Mapping[str, object]) -> dict[str, object]:
    return _gate_results(
        values["unseen_metrics"],
        values["unseen_control_metrics"],
        values["core_metrics"],
        values["core_control_metrics"],
        values["expansion"],
        values["marginal"],
        values["control_metrics"],
        values["core_majority"],
        values["unseen_majority"],
    )


def verify_gate_boundaries() -> None:
    values = passing_gate_inputs()
    values["unseen_metrics"]["return_mae"] = 0.99
    values["core_metrics"]["return_mae"] = 1.01
    gates = evaluate_gates(values)
    assert gates["all_pass"]
    assert gates["unseen_33_to_44_marginal"]["pass"]

    mutations: tuple[tuple[str, Callable[[dict[str, object]], None]], ...] = (
        ("unseen_mae_improvement", lambda item: item[
            "unseen_metrics"
        ].update(return_mae=nextafter(0.99, inf))),
        ("positive_paired_intervals", lambda item: item[
            "expansion"
        ]["intervals"].update({"10": (0.0, 0.2)})),
        ("majority_unseen_improved", lambda item: item[
            "expansion"
        ].update(wins=5)),
        ("core_degradation", lambda item: item[
            "core_metrics"
        ].update(return_mae=nextafter(1.01, inf))),
        ("pooled_and_local_controls", lambda item: item[
            "control_metrics"
        ]["core"]["zero"].update(return_mae=0.9)),
        ("direction_majority", lambda item: item[
            "core_metrics"
        ].update(direction_accuracy=0.7)),
        ("close_mae", lambda item: item[
            "control_metrics"
        ]["unseen"]["zero"].update(close_mae=0.8)),
        ("unseen_33_to_44_marginal", lambda item: item[
            "marginal"
        ].update(mean_gain=nextafter(0.0, -inf))),
    )
    for gate, mutate in mutations:
        invalid = passing_gate_inputs()
        mutate(invalid)
        assert not evaluate_gates(invalid)[gate]["pass"], gate

    invalid = passing_gate_inputs()
    invalid["expansion"]["per_stock_mean_gain"].pop("S10")
    assert not evaluate_gates(invalid)["majority_unseen_improved"]["pass"]


def verify_real_paired_comparison() -> None:
    candidate = {}
    reference = {}
    start = date(2026, 1, 1)
    for stock in range(11):
        candidate_points = []
        reference_points = []
        for offset in range(20):
            target = (start + timedelta(days=offset)).isoformat()
            actual = 0.1 + stock * 0.0001 + offset * 0.00001
            candidate_points.append(ForecastPoint(
                target, actual, actual - 0.01, 100.0, 110.0,
            ))
            reference_points.append(ForecastPoint(
                target, actual, actual - 0.1, 100.0, 110.0,
            ))
        candidate[f"S{stock:02d}"] = tuple(candidate_points)
        reference[f"S{stock:02d}"] = tuple(reference_points)
    result = paired_comparison(
        candidate, reference, block_days=(5, 10, 20),
        replicates=100,
    )
    assert result["wins"] == 11
    assert result["mean_gain"] > 0
    assert all(
        result["intervals"][str(block)][0] > 0
        for block in (5, 10, 20)
    )


def verify_majority_baseline() -> None:
    def points(signs: tuple[int, ...]) -> tuple[ForecastPoint, ...]:
        return tuple(
            ForecastPoint(
                f"2026-01-{index + 1:02d}T00:00:00Z",
                float(sign), 0.0, 100.0, 100.0,
            )
            for index, sign in enumerate(signs)
        )

    flat_majority = points((0, 0, 0, 1, 1, -1))
    tied_up_down = points((1, 1, -1, -1, 0))
    assert finalizer._majority_accuracy({"flat": flat_majority}) == 0.5
    assert finalizer._majority_accuracy({"tie": tied_up_down}) == 0.4
    assert finalizer._majority_accuracy({
        "flat": flat_majority, "tie": tied_up_down,
    }) == 0.45


TERMINALS = {
    "preflight-failure": ("preflight", 2),
    "setup-failure": ("setup", 2),
    "experiment-failure": ("experiment", 2),
    "analysis-integrity-failure": ("analysis", 2),
    "gate-failure": ("analysis", 3),
    "pass": ("analysis", 0),
}


class DirectFinalizerFixture:
    def __init__(
        self, root: Path, status: str,
        experiment_outputs: tuple[str, ...] = ("fits",),
    ) -> None:
        self.root = root.resolve()
        self.experiments = self.root / "experiments"
        self.reports = self.root / "reports"
        self.experiments.mkdir(parents=True)
        self.reports.mkdir()
        self.attempt_path = self.experiments / "direct-attempt.json"
        self.outcome = self.experiments / "direct-outcome.json"
        self.run = self.reports / "direct"
        self.fits = self.run / "fits.jsonl"
        self.predictions = self.run / "predictions.jsonl"
        self.summary = self.run / "summary.json"
        self.attempt_path.write_text('{"schema":1}\n', encoding="ascii")
        if status != "preflight-failure":
            self.run.mkdir()
        present = (
            experiment_outputs if status == "experiment-failure" else
            ("fits", "predictions")
            if status not in ("preflight-failure", "setup-failure") else ()
        )
        for name in present:
            getattr(self, name).write_text("{}\n", encoding="ascii")
        self.trusted = self.root / "trusted.py"
        self.trusted.write_text("trusted = True\n", encoding="ascii")
        support = self.root / "csv"
        support.mkdir()
        self.support = tuple(
            support / f"s{index:02d}.csv" for index in range(55)
        )
        for index, path in enumerate(self.support):
            path.write_text(f"{index}\n", encoding="ascii")
        self.names = synthetic_master()
        outputs = {
            "fits": "reports/direct/fits.jsonl",
            "predictions": "reports/direct/predictions.jsonl",
            "summary": "reports/direct/summary.json",
            "outcome": "experiments/direct-outcome.json",
        }
        prefix = (
            "tools/finalize_universe_scaling.py",
            "experiments/direct-attempt.json",
            outputs["outcome"],
        )
        self.attempt = SimpleNamespace(
            attempt_path="experiments/direct-attempt.json",
            commands={"finalizer_prefix": prefix},
            coverage=synthetic_coverage(self.names),
            finalizer_tree=SimpleNamespace(sha256=sha256(8_000)),
            manifests=(
                SimpleNamespace(file=SimpleNamespace(
                    path=str(self.support[0]),
                )),
            ),
            outputs=outputs,
            primary_python=SimpleNamespace(
                path=sys.executable, sha256=sha256(8_001),
            ),
            protocol=expected_protocol(),
            run_dir="reports/direct",
            run_id="direct",
            session_calendar=SimpleNamespace(path=str(self.support[1])),
        )
        self.success = finalizer.SuccessInputs(
            self.names,
            tuple(SimpleNamespace(path=str(path)) for path in self.support),
            (), self.support,
        )

    def invoke(
        self, status: str,
        writer: Callable[..., None] | None = None,
        *, patch_isolation_boundary: bool = True,
    ) -> dict[str, object]:
        stage, code = TERMINALS[status]
        prefix = self.attempt.commands["finalizer_prefix"]
        argv = (
            *prefix,
            "--started", "2026-07-24T00:00:00Z",
            "--ended", "2026-07-24T00:01:00Z",
            "--stage", stage, "--exit", str(code), "--status", status,
        )

        def analysis(*_: object) -> dict[str, object]:
            return {
                "schema": 1,
                "status": status,
                "gates": {"all_pass": status == "pass"},
            }

        with ExitStack() as stack:
            if patch_isolation_boundary:
                stack.enter_context(patch.object(
                    finalizer, "_require_isolated_execution",
                ))
                stack.enter_context(patch.object(
                    finalizer, "_require_exact_launch",
                ))
            stack.enter_context(patch.object(finalizer, "ROOT", self.root))
            stack.enter_context(patch.object(
                finalizer.ScalingAttempt, "read",
                return_value=self.attempt,
            ))
            stack.enter_context(patch.object(
                finalizer, "_trusted_paths",
                return_value=(self.trusted,),
            ))
            stack.enter_context(patch.object(
                finalizer, "_validate_trusted",
            ))
            stack.enter_context(patch.object(
                finalizer, "_success_inputs",
                return_value=self.success,
            ))
            stack.enter_context(patch.object(
                finalizer, "_validate_success_inputs",
                return_value=self.names,
            ))
            stack.enter_context(patch.object(
                finalizer, "_validate_live_success",
            ))
            stack.enter_context(patch.object(
                finalizer, "derive_market_truth",
                return_value=MarketTruth(self.attempt.coverage, {}),
            ))
            stack.enter_context(patch.object(
                finalizer, "analyze_ledgers", side_effect=analysis,
            ))
            stack.enter_context(patch.object(sys, "argv", list(argv)))
            if writer is not None:
                stack.enter_context(patch.object(
                    finalizer, "write_json_exclusive", writer,
                ))
            return finalizer.finalize(
                self.attempt_path, self.outcome,
                "2026-07-24T00:00:00Z",
                "2026-07-24T00:01:00Z",
                stage, code, status,
            )


def reject_direct(
    fixture: DirectFinalizerFixture, status: str,
    writer: Callable[..., None] | None = None,
) -> None:
    try:
        fixture.invoke(status, writer)
    except (OSError, ValueError):
        return
    raise AssertionError("invalid direct finalization was accepted")


def intercepted_writer(
    target: Path,
    mutate: Callable[[], None] | None = None,
    *, fail_after_verify: bool = False,
    mutate_after: Callable[[], None] | None = None,
) -> Callable[..., None]:
    original = finalizer.write_json_exclusive

    def writer(
        path: Path, value: Mapping[str, object],
        directory_fd: int | None = None,
        before_link: Callable[[], None] | None = None,
        *,
        before_link_with_temp: Callable[[object], None] | None = None,
        on_temp_created: Callable[[object], None] | None = None,
    ) -> None:
        if path != target:
            original(
                path, value, directory_fd, before_link,
                before_link_with_temp=before_link_with_temp,
                on_temp_created=on_temp_created,
            )
            return
        assert before_link is None
        assert before_link_with_temp is not None

        def inject(temporary: object) -> None:
            if mutate is not None:
                mutate()
            before_link_with_temp(temporary)
            if fail_after_verify:
                raise OSError("injected publication failure")

        original(
            path, value, directory_fd,
            before_link_with_temp=inject,
            on_temp_created=on_temp_created,
        )
        if mutate_after is not None:
            mutate_after()

    return writer


def verify_direct_finalization() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-direct-",
    ) as directory:
        root = Path(directory)
        fixture = DirectFinalizerFixture(
            root / "nonisolated", "setup-failure",
        )
        try:
            fixture.invoke(
                "setup-failure", patch_isolation_boundary=False,
            )
        except ValueError as error:
            assert "isolated" in str(error)
        else:
            raise AssertionError("nonisolated direct finalization was accepted")
        assert not fixture.outcome.exists()

        for status in TERMINALS:
            case = root / status
            case.mkdir()
            fixture = DirectFinalizerFixture(case, status)
            result = fixture.invoke(status)
            assert result["status"] == status
            assert fixture.outcome.exists()
            assert fixture.summary.exists() == (
                status in ("gate-failure", "pass")
            )
            expected = {
                "preflight-failure": (False, False),
                "setup-failure": (False, False),
                "experiment-failure": (True, False),
                "analysis-integrity-failure": (True, True),
                "gate-failure": (True, True),
                "pass": (True, True),
            }[status]
            assert (
                result["outputs"]["fits"]["state"] == "present",
                result["outputs"]["predictions"]["state"] == "present",
            ) == expected

        for index, present in enumerate((
            (), ("predictions",), ("fits", "predictions"),
        )):
            fixture = DirectFinalizerFixture(
                root / f"experiment-partial-{index}",
                "experiment-failure", present,
            )
            result = fixture.invoke("experiment-failure")
            assert tuple(
                name for name in ("fits", "predictions")
                if result["outputs"][name]["state"] == "present"
            ) == present

        fixture = DirectFinalizerFixture(root / "no-clobber", "setup-failure")
        fixture.invoke("setup-failure")
        original = fixture.outcome.read_bytes()
        reject_direct(fixture, "setup-failure")
        assert fixture.outcome.read_bytes() == original

    args = finalizer.parse_args([
        "attempt.json", "outcome.json",
        "--started", "2026-07-24T00:00:00Z",
        "--ended", "2026-07-24T00:01:00Z",
        "--stage", "analysis", "--exit", "0", "--status", "pass",
    ])
    assert (args.stage, args.exit_code, args.status) == (
        "analysis", 0, "pass",
    )


def verify_summary_retry() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-retry-",
    ) as directory:
        fixture = DirectFinalizerFixture(Path(directory), "pass")
        writer = intercepted_writer(
            fixture.outcome, fail_after_verify=True,
        )
        reject_direct(fixture, "pass", writer)
        assert fixture.summary.exists()
        assert not fixture.outcome.exists()
        assert not tuple(fixture.experiments.glob(".*.tmp"))
        identity = (fixture.summary.stat().st_dev, fixture.summary.stat().st_ino)
        result = fixture.invoke("pass")
        assert result["status"] == "pass"
        assert identity == (
            fixture.summary.stat().st_dev, fixture.summary.stat().st_ino,
        )

    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-retry-changed-",
    ) as directory:
        fixture = DirectFinalizerFixture(Path(directory), "pass")
        reject_direct(
            fixture, "pass",
            intercepted_writer(fixture.outcome, fail_after_verify=True),
        )
        fixture.summary.write_bytes(fixture.summary.read_bytes() + b" ")
        reject_direct(fixture, "pass")
        assert not fixture.outcome.exists()


def verify_publication_races() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-races-",
    ) as directory:
        root = Path(directory)
        mutations = {
            "preflight-failure": lambda item: (
                item.run.mkdir(),
                item.fits.write_text("{}\n", encoding="ascii"),
            ),
            "setup-failure": lambda item:
                item.fits.write_text("{}\n", encoding="ascii"),
            "experiment-failure": lambda item: (
                item.fits.unlink(),
                item.fits.write_text('{"changed":true}\n', encoding="ascii"),
            ),
            "analysis-integrity-failure": lambda item: (
                item.predictions.unlink(),
                item.predictions.write_text(
                    '{"changed":true}\n', encoding="ascii",
                ),
            ),
        }
        for status, mutate in mutations.items():
            case = root / f"callback-{status}"
            case.mkdir()
            fixture = DirectFinalizerFixture(case, status)
            reject_direct(
                fixture, status,
                intercepted_writer(
                    fixture.outcome, lambda: mutate(fixture),
                ),
            )
            assert not fixture.outcome.exists()
            assert not tuple(fixture.experiments.glob(".*.tmp"))

        case = root / "summary-extra"
        case.mkdir()
        fixture = DirectFinalizerFixture(case, "pass")
        reject_direct(
            fixture, "pass",
            intercepted_writer(
                fixture.summary,
                lambda: (fixture.run / "policy.json").write_text(
                    "{}\n", encoding="ascii",
                ),
            ),
        )
        assert not fixture.summary.exists()
        assert not tuple(fixture.run.glob(".*.tmp"))

        case = root / "csv"
        case.mkdir()
        fixture = DirectFinalizerFixture(case, "pass")
        reject_direct(
            fixture, "pass",
            intercepted_writer(
                fixture.summary,
                lambda: fixture.support[-1].write_text(
                    "changed\n", encoding="ascii",
                ),
            ),
        )
        assert not fixture.summary.exists()

        case = root / "summary-replaced"
        case.mkdir()
        fixture = DirectFinalizerFixture(case, "pass")
        reject_direct(
            fixture, "pass",
            intercepted_writer(fixture.outcome, fail_after_verify=True),
        )
        content = fixture.summary.read_bytes()

        def replace_summary() -> None:
            fixture.summary.unlink()
            fixture.summary.write_bytes(content)

        reject_direct(
            fixture, "pass",
            intercepted_writer(fixture.outcome, replace_summary),
        )
        assert not fixture.outcome.exists()


def verify_parent_replacement_races() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-parent-races-",
    ) as directory:
        root = Path(directory)

        fixture = DirectFinalizerFixture(root / "summary-parent", "pass")
        moved_run = fixture.run.with_name("direct-summary-moved")

        def replace_summary_parent() -> None:
            fixture.run.rename(moved_run)
            fixture.run.mkdir()

        reject_direct(
            fixture, "pass",
            intercepted_writer(fixture.summary, replace_summary_parent),
        )
        assert not fixture.summary.exists()
        assert not tuple(moved_run.glob(".*.tmp"))

        fixture = DirectFinalizerFixture(root / "run-parent", "pass")
        reject_direct(
            fixture, "pass",
            intercepted_writer(fixture.outcome, fail_after_verify=True),
        )
        moved_run = fixture.run.with_name("direct-outcome-moved")

        def replace_run_parent() -> None:
            fixture.run.rename(moved_run)
            fixture.run.mkdir()

        reject_direct(
            fixture, "pass",
            intercepted_writer(fixture.outcome, replace_run_parent),
        )
        assert not fixture.outcome.exists()

        fixture = DirectFinalizerFixture(
            root / "outcome-parent", "setup-failure",
        )
        moved_experiments = fixture.experiments.with_name(
            "experiments-moved",
        )

        def replace_outcome_parent() -> None:
            fixture.experiments.rename(moved_experiments)
            fixture.experiments.mkdir()

        reject_direct(
            fixture, "setup-failure",
            intercepted_writer(fixture.outcome, replace_outcome_parent),
        )
        assert not fixture.outcome.exists()
        assert not tuple(moved_experiments.glob(".*.tmp"))

        fixture = DirectFinalizerFixture(
            root / "outcome-parent-after", "setup-failure",
        )
        moved_experiments = fixture.experiments.with_name(
            "experiments-after-moved",
        )

        def replace_outcome_parent_after() -> None:
            fixture.experiments.rename(moved_experiments)
            fixture.experiments.mkdir()

        reject_direct(
            fixture, "setup-failure",
            intercepted_writer(
                fixture.outcome, mutate_after=replace_outcome_parent_after,
            ),
        )
        assert not fixture.outcome.exists()
        assert (moved_experiments / fixture.outcome.name).exists()


def verify_temp_cleanup_safety() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-temp-cleanup-",
    ) as directory_name:
        root = Path(directory_name)
        for mode in ("owned", "replaced", "hardlinked"):
            directory = root / mode
            directory.mkdir()
            output = directory / "output.json"
            descriptor = os.open(
                directory,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            captured = []

            def reject(temporary: object) -> None:
                captured.append(temporary)
                temporary_path = directory / temporary.name
                if mode == "replaced":
                    temporary_path.unlink()
                    temporary_path.write_text(
                        "unowned\n", encoding="ascii",
                    )
                elif mode == "hardlinked":
                    os.link(temporary_path, directory / "alias")
                raise ValueError("injected callback failure")

            try:
                try:
                    finalizer._publish_exclusive(
                        output, {"status": "test"}, descriptor, reject,
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        "failed exclusive callback was accepted"
                    )
            finally:
                os.close(descriptor)
            assert len(captured) == 1
            temporary = directory / captured[0].name
            if mode == "owned":
                assert not temporary.exists()
            elif mode == "replaced":
                assert temporary.read_text(encoding="ascii") == "unowned\n"
            else:
                assert temporary.exists()
                assert (directory / "alias").stat().st_ino == \
                    temporary.stat().st_ino


def verify_early_temp_capture() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-early-temp-",
    ) as directory_name:
        root = Path(directory_name)
        for mode in ("owned", "replaced", "hardlinked"):
            directory = root / mode
            directory.mkdir()
            output = directory / "output.json"
            descriptor = os.open(
                directory,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )

            def writer(
                path: Path, value: Mapping[str, object],
                directory_fd: int | None = None,
                before_link: Callable[[], None] | None = None,
                *,
                before_link_with_temp: Callable[[object], None] | None = None,
                on_temp_created: Callable[[object], None] | None = None,
            ) -> None:
                assert path == output and before_link is None

                def capture(temporary: object) -> None:
                    assert on_temp_created is not None
                    on_temp_created(temporary)
                    temporary_path = directory / temporary.name
                    if mode == "replaced":
                        temporary_path.unlink()
                        temporary_path.write_text(
                            "unowned\n", encoding="ascii",
                        )
                    elif mode == "hardlinked":
                        os.link(temporary_path, directory / "alias")

                def fail(_: object) -> None:
                    raise OSError("injected write failure")

                file_tools.exclusive_text(
                    path, fail, directory_fd,
                    before_link_with_temp=before_link_with_temp,
                    on_temp_created=capture,
                )

            try:
                try:
                    with patch.object(
                        finalizer, "write_json_exclusive", writer,
                    ):
                        finalizer._publish_exclusive(
                            output, {"status": "test"}, descriptor,
                            lambda _: None,
                        )
                except OSError:
                    pass
                else:
                    raise AssertionError("pre-link write failure was accepted")
            finally:
                os.close(descriptor)
            temporary = tuple(directory.glob(".output.json.*.tmp"))
            if mode == "owned":
                assert not temporary
            elif mode == "replaced":
                assert len(temporary) == 1
                assert temporary[0].read_text(encoding="ascii") == "unowned\n"
            else:
                assert len(temporary) == 1
                assert temporary[0].stat().st_ino == \
                    (directory / "alias").stat().st_ino


def verify_status_closure_rejections() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-scaling-closures-",
    ) as directory:
        root = Path(directory)
        for status in (
            "preflight-failure", "setup-failure",
            "experiment-failure", "analysis-integrity-failure",
        ):
            case = root / f"extra-{status}"
            case.mkdir()
            fixture = DirectFinalizerFixture(case, status)
            fixture.run.mkdir(exist_ok=True)
            (fixture.run / "test.jsonl").write_text("{}\n", encoding="ascii")
            reject_direct(fixture, status)

            case = root / f"alias-{status}"
            case.mkdir()
            fixture = DirectFinalizerFixture(case, status)
            fixture.run.mkdir(exist_ok=True)
            alias = (
                fixture.predictions
                if status == "analysis-integrity-failure" else fixture.fits
            )
            alias.unlink(missing_ok=True)
            source = (
                fixture.fits
                if status == "analysis-integrity-failure" else
                fixture.attempt_path
            )
            os.link(source, alias)
            reject_direct(fixture, status)

        case = root / "symlink"
        case.mkdir()
        fixture = DirectFinalizerFixture(case, "pass")
        moved = fixture.run.with_name("direct-real")
        fixture.run.rename(moved)
        fixture.run.symlink_to(moved, target_is_directory=True)
        reject_direct(fixture, "pass")
        assert not fixture.summary.exists()


def main() -> None:
    verify_attempt()
    verify_isolated_startup()
    verify_input_derivation()
    verify_fit_schedule()
    verify_runner_boundaries()
    verify_finalizer_signal_subprocess()
    verify_market_truth_derivation()
    verify_scaling_finalizer()
    verify_gate_boundaries()
    verify_real_paired_comparison()
    verify_majority_baseline()
    verify_direct_finalization()
    verify_summary_retry()
    verify_publication_races()
    verify_parent_replacement_races()
    verify_temp_cleanup_safety()
    verify_early_temp_capture()
    verify_status_closure_rejections()
    print("universe scaling driver tests passed")


if __name__ == "__main__":
    main()
