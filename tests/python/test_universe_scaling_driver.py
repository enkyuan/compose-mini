#!/usr/bin/env python3
"""Verify the immutable development-only universe-scaling contract."""

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.universe_scaling_contract import (
    CALENDAR_SHA256, CONFIG_SHA256, EXPECTED_BUDGETS, FETCH_SHA256,
    FINALIZER_SOURCE_PATHS, MANIFEST_SHA256, PHASES, SELECTION_FILES,
    SELECTION_SHA256, SOURCE_PATHS, FitJob, ScalingAttempt,
    expected_fit_jobs, expected_protocol, expected_scaling_commands,
    question_uses,
)
from tools.universe_scaling_inputs import ScalingCoverage, fetch_series
from tools.universe_scaling_inputs import (
    EXPECTED_MISSING, common_coverage, selection_binding, selection_paths,
)
from tools.universe_contract import PackedRows


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
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": f"{run_dir}/.pycache",
        },
        "fetch_report": binding(
            "reports/liquid-common-55-20260724-02-fetch.json", FETCH_SHA256,
        ),
        "finalizer_tree": tree(
            repository_root, FINALIZER_SOURCE_PATHS, 7_000,
        ),
        "implementation_commit": "1" * 40,
        "manifests": [
            {
                "size": size,
                **binding(
                    "reports/universe-selection-20260724-06/manifests/"
                    f"liquid-common-{size}.json",
                    digest,
                ),
            }
            for size, digest in MANIFEST_SHA256.items()
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
        protocol["models"].clear()
        assert expected_protocol()["models"]
        assert {
            "tools/analyze_universe.py",
            "tools/arm_universe_scaling.py",
        } <= set(SOURCE_PATHS)
        assert "tools/universe_contract.py" in FINALIZER_SOURCE_PATHS

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
                        root / "data/liquid-common-55-20260724-02" /
                        f"s{index:02d}-30m.csv"
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
    for index, name in (
        (11, "ALTR"), (20, "ZI"), (34, "FYBR"),
        (40, "INFA"), (49, "ENLC"),
    ):
        names[index] = name
    manifest = SimpleNamespace(
        series=tuple(SimpleNamespace(ticker=name) for name in names),
        interval_minutes=30, start="start", end="end",
    )
    bars = {
        name: SimpleNamespace(timestamps=(name,))
        for name in names
    }
    calls = 0

    def samples(timestamps: tuple[str, ...], *_: object) -> object:
        return SimpleNamespace(
            opportunities=5_505, rows=timestamps,
        )

    def packed(rows: tuple[str, ...], *_: object) -> PackedRows:
        nonlocal calls
        name, phase = rows[0], PHASES[calls % len(PHASES)]
        index = names.index(name)
        budget = dict(EXPECTED_BUDGETS)[phase]
        base, remainder = divmod(budget.control_samples, 11)
        train = base + (index < remainder) if index < 11 else 1
        validation = int(name not in dict(EXPECTED_MISSING)[phase])
        calls += 1
        return PackedRows((), (train, validation))

    with patch(
        "tools.universe_scaling_inputs.session_samples",
        side_effect=samples,
    ), patch(
        "tools.universe_scaling_inputs.pack_rows",
        side_effect=packed,
    ):
        coverage = common_coverage(
            manifest, SimpleNamespace(), bars,
        )
    assert calls == 55 * len(PHASES)
    assert tuple(len(item.evaluable) for item in coverage.phases) == (
        52, 51, 50,
    )
    assert tuple(
        (item.phase, item.missing) for item in coverage.phases
    ) == EXPECTED_MISSING
    assert tuple(
        sum(train for _, train, _ in item.counts[:11])
        for item in coverage.phases
    ) == tuple(
        budget.control_samples for _, budget in EXPECTED_BUDGETS
    )
    assert coverage.unseen_missing == ("ENLC",)
    try:
        coverage.require_promotable()
    except ValueError as error:
        assert str(error) == \
            "unseen calibration coverage is incomplete: ENLC"
    else:
        raise AssertionError("incomplete unseen coverage was accepted")
    try:
        ScalingCoverage((), ()).require_promotable()
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


def main() -> None:
    verify_attempt()
    verify_input_derivation()
    verify_fit_schedule()
    print("universe scaling driver tests passed")


if __name__ == "__main__":
    main()
