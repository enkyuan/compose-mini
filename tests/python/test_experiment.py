#!/usr/bin/env python3
"""Verify walk-forward selection, aligned targets, baselines, and JSON reports."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import hashlib
import math
import os
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("experiment tests require PyTorch") from error

from tools.experiment import (
    Candidate, ConstantReturn, RollingMean, Sweep, _authorize_test,
    _candidate_data, _fit_neural, _label_available, _model_fingerprint,
    _panel_data, _panel_selected_epochs, _run_experiment, _selected_epochs,
    _verify_test_state, expected_runs, holdout_split, linear_model,
    purged_split, run_experiment, select_candidates, walk_forward_splits,
    write_predictions, write_report,
)
from tools.backtest import experiment_fingerprint, validate_policy
from tools.data_v1 import (
    CLOSE_RETURN_TARGET, EXECUTABLE_RETURN_TARGET, read_bars,
)
from tools.train import (
    ForecastTransformer, TrainingData, Windows, data_loaders, evaluate,
    feature_lookback, feature_values, fit_epochs, prepare_data,
)
from tools.files import file_sha256
from tools.panel_contract import (
    FINALIZER_SOURCE_PATHS, SOURCE_PATHS, ExecutableBinding, SourceTree,
    TorchIdentity, executable_binding, freeze_panel_execution, source_tree,
)


def close_value(index: int) -> float:
    return 100.0 + 0.08 * index + 0.05 * (index % 7) - 0.03 * (index % 3)


def write_csv(path: Path, count: int = 56, offset_minutes: int = 0) -> None:
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc) + \
        timedelta(minutes=offset_minutes)
    lines = ["timestamp,open,high,low,close,volume"]
    for index in range(count):
        close = close_value(index)
        values = (close - 0.1 - 0.01 * (index % 2), close + 0.4,
                  close - 0.5, close, 1_000.0 + 11.0 * index + index % 5)
        timestamp = (start + timedelta(minutes=index)).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append(timestamp + "," + ",".join(f"{value:.9g}" for value in values))
    path.write_text("\n".join(lines), encoding="ascii")


def write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def scale_csv(path: Path, scale: float) -> None:
    lines = path.read_text(encoding="ascii").splitlines()
    for index in range(1, len(lines)):
        fields = lines[index].split(",")
        fields[1:] = [str(float(value) * scale) for value in fields[1:]]
        lines[index] = ",".join(fields)
    path.write_text("\n".join(lines), encoding="ascii")


def binding(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": file_sha256(path)}


def tree_binding(root: Path, paths: tuple[str, ...]) -> dict[str, object]:
    files = [
        {"path": path, "sha256": file_sha256(root / path)}
        for path in sorted(paths)
    ]
    digest = hashlib.sha256()
    for item in files:
        digest.update(
            item["path"].encode() + b"\0" +
            item["sha256"].encode() + b"\n"
        )
    return {"root": str(root), "files": files, "sha256": digest.hexdigest()}


@contextmanager
def panel_fixture(
    directory: Path, sweep: Sweep, *, mismatched: bool = False,
    mutation: str | None = None,
) -> Iterator[object]:
    relative = directory.relative_to(ROOT)
    source_root = directory / "source"
    for name in SOURCE_PATHS:
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="ascii")
    package_root = directory / "torch"
    package_root.mkdir()
    (package_root / "module.py").write_text("value = 1\n", encoding="ascii")
    python = (directory / "runtime").resolve()
    python.write_text(
        "#!/bin/sh\nprintf '%s\\n' 'fixture runtime'\n", encoding="ascii",
    )
    python.chmod(0o700)
    command_version = "fixture runtime"
    command = ExecutableBinding(
        str(python), file_sha256(python), command_version,
    )
    observed = TorchIdentity(
        executable_binding(python, sys.version), str(torch.__version__),
        torch.version.git_version, torch.version.cuda,
        torch.__config__.show(), source_tree(package_root),
    )
    config = relative / "sweep.json"
    baseline_report = relative / "baseline.json"
    baseline_ledger = relative / "baseline.jsonl"
    inputs = relative / "inputs.json"
    attempt = relative / "attempt.json"
    write_canonical(ROOT / config, {
        "candidates": [asdict(item) for item in sweep.candidates],
        "models": list(sweep.models), "seeds": list(sweep.seeds),
        "folds": sweep.folds, "fold_fraction": sweep.fold_fraction,
        "epochs": sweep.epochs, "patience": sweep.patience,
        "batch_size": sweep.batch_size,
        "target_horizon_bars": sweep.target_horizon_bars,
        "alignment_horizon_bars": sweep.alignment_horizon_bars,
        "target_kind": sweep.target_kind,
    })
    write_canonical(ROOT / baseline_report, {"fixture": True})
    (ROOT / baseline_ledger).write_text("{}\n", encoding="ascii")
    paths = []
    declarations = []
    for index, name in enumerate(("A", "B", "C"), 1):
        path = relative / f"{name}.csv"
        write_csv(ROOT / path, 72, int(mismatched and name == "B"))
        scale_csv(ROOT / path, float(index))
        timestamps, _ = read_bars(ROOT / path)
        digest = hashlib.sha256(
            "".join(f"{timestamp}\n" for timestamp in timestamps).encode()
        ).hexdigest()
        paths.append((name, path))
        declarations.append({
            "name": name, "csv": binding(path),
            "rows": len(timestamps), "first_timestamp": timestamps[0],
            "last_timestamp": timestamps[-1],
            "timestamp_sha256": digest,
        })
    write_canonical(ROOT / inputs, {
        "schema": 1, "series": declarations,
        "baseline_report": binding(baseline_report),
        "baseline_ledger": binding(baseline_ledger),
    })
    run_dir = relative / "run"
    (ROOT / run_dir).mkdir()
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": f"{run_dir}/.pycache",
    }
    argv = ("tools/experiment.py", "fixture")
    write_canonical(ROOT / attempt, {
        "schema": 1, "run_id": "fixture", "status": "armed",
        "run_dir": run_dir.as_posix(),
        "implementation_commit": "0" * 40,
        "input_manifest": binding(inputs), "config": binding(config),
        "baseline_report": binding(baseline_report),
        "baseline_ledger": binding(baseline_ledger),
        "source_tree": tree_binding(source_root, SOURCE_PATHS),
        "finalizer_tree": tree_binding(source_root, FINALIZER_SOURCE_PATHS),
        "primary_python": asdict(command), "uv": asdict(command),
        "torch_argv": [
            str(python), "run", "--offline", "--with", "torch", "python",
        ],
        "torch_probe": asdict(observed),
        "environment": environment,
        "commands": {
            "validate_attempt": ["validate"],
            "preflight": ["preflight"], "experiment": list(argv),
            "analyze": ["analyze"], "finalizer_prefix": ["finalize"],
        },
        "expected_equivalent_runs": expected_runs(sweep, 3),
        "expected_panel_fits":
            len(sweep.seeds) * (len(sweep.candidates) * sweep.folds + 1),
        "outputs": {
            "experiment_report": f"{run_dir}/experiment.json",
            "calibration_ledger": f"{run_dir}/calibration.jsonl",
            "analysis_report": f"{run_dir}/analysis.json",
            "outcome": f"{relative}/outcome.json",
        },
    })
    mutation_paths = {
        "config": ROOT / config,
        "baseline": ROOT / baseline_report,
        "csv": ROOT / paths[0][1],
        "source": source_root / SOURCE_PATHS[0],
    }
    if mutation in mutation_paths:
        with mutation_paths[mutation].open("a", encoding="ascii") as file:
            file.write(" ")
    if mutation == "binary":
        with python.open("a", encoding="ascii") as file:
            file.write(" ")
    if mutation == "output":
        (ROOT / run_dir / "experiment.json").write_text(
            "occupied", encoding="ascii",
        )
    if mutation == "symlink":
        path = ROOT / paths[0][1]
        target = path.with_suffix(".source")
        path.rename(target)
        path.symlink_to(target.name)
    if mutation == "hardlink":
        path = ROOT / paths[1][1]
        path.unlink()
        os.link(ROOT / paths[0][1], path)
    runtime = [
        replace(observed, version=f"{observed.version}-changed")
        if mutation == "runtime" else observed
    ]
    observe = lambda: runtime[0]
    previous = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    if mutation == "cache":
        os.environ["PYTHONPYCACHEPREFIX"] = f"{run_dir}/changed"
    try:
        with freeze_panel_execution(
            attempt, inputs, config, baseline_report, baseline_ledger,
            paths, source_root, observe,
            (*argv, "changed") if mutation == "argv" else argv,
        ) as execution:
            moved = (ROOT / run_dir).with_name(f"{run_dir.name}-bound")
            if mutation == "runtime-after-freeze":
                runtime[0] = replace(
                    observed, version=f"{observed.version}-changed",
                )
            elif mutation == "directory-after-freeze":
                (ROOT / run_dir).rename(moved)
                (ROOT / run_dir).mkdir()
            try:
                yield execution
            finally:
                runtime[0] = observed
                if mutation == "directory-after-freeze":
                    (ROOT / run_dir).rmdir()
                    moved.rename(ROOT / run_dir)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def candidate(name: str, seq_len: int,
              feature_set: str = "ohlcv") -> Candidate:
    return Candidate(name, seq_len, 4, 2, 6, 1, 1e-3, 1e-4,
                     6, min(2, seq_len - 1), 1e-3, feature_set)


def authorize(*models: str) -> Callable[
    [object], Mapping[str, Mapping[str, object]]
]:
    def selected(contract: object) -> Mapping[str, Mapping[str, object]]:
        assert isinstance(contract, dict)
        assert contract["calibration"] and contract["model_fingerprints"]
        assert "test" not in contract
        return {
            model: {
                "model_fingerprints": [
                    item for item in contract["model_fingerprints"]
                    if item["model"] == model
                ],
            }
            for model in models
        }
    return selected


def frozen_policy(contract: Mapping[str, object],
                  model: str) -> dict[str, object]:
    candidate_name = contract["selection"][model]["candidate"]
    configuration = next(
        item for item in contract["sweep"]["candidates"]
        if item["name"] == candidate_name
    )
    fingerprints = [
        item for item in contract["model_fingerprints"]
        if item["model"] == model
    ]
    records = max(len(fingerprints), 1)
    return validate_policy({
        "schema": 2, "action": "cash", "model": model,
        "candidate": candidate_name,
        "feature_set": configuration["feature_set"],
        "target_kind": contract["sweep"]["target_kind"],
        "horizon_bars": contract["sweep"]["target_horizon_bars"],
        "seeds": (
            sorted(contract["sweep"]["seeds"])
            if model in ("transformer", "mlp") else []
        ),
        "series": sorted(item["name"] for item in contract["series"]),
        "initial_cash": 100.0,
        "costs": {"spread_bps": 0.0, "slippage_bps": 0.0, "fee_bps": 0.0},
        "safety_bps": None, "minimum_predicted_log_return": None,
        "selection_objective": "macro_mean_terminal_log_growth",
        "calibration_report": {"path": "calibration.json", "sha256": "1" * 64},
        "calibration_prediction_ledger": {
            "path": "calibration.jsonl", "sha256": "2" * 64,
            "source_records": records, "selected_records": records,
        },
        "model_fingerprints": fingerprints,
        "threshold_trials": [
            {
                "action": "long_above", "safety_bps": 0.0,
                "objective": -0.1, "mean_final_equity": 90.0,
                "mean_gross_turnover": 1.0, "signal_coverage": 1.0,
                "execution_coverage": 1.0, "trade_count": 1,
            },
            {
                "action": "cash", "safety_bps": None, "objective": 0.0,
                "mean_final_equity": 100.0, "mean_gross_turnover": 0.0,
                "signal_coverage": 0.0, "execution_coverage": 0.0,
                "trade_count": 0,
            },
        ],
        "test_grid": contract["test_contract"],
        "calibration_fingerprint": experiment_fingerprint(contract),
    })


def outcomes(dataset: Windows) -> torch.Tensor:
    return torch.stack([torch.stack(dataset[index][1:])
                        for index in range(len(dataset))])


def verify_stationary_features() -> None:
    rows = torch.tensor([
        [100.0, 105.0, 95.0, 102.0, 1_000.0],
        [103.0, 108.0, 101.0, 106.0, 1_300.0],
        [104.0, 107.0, 99.0, 100.0, 900.0],
    ])
    features = feature_values(rows, "stationary-v1")
    scaled = rows.clone()
    scaled[:, :4] *= 7.0
    torch.testing.assert_close(feature_values(scaled, "stationary-v1"), features)
    torch.testing.assert_close(
        features[:, 0] + features[:, 1],
        torch.log(rows[1:, 3] / rows[:-1, 3]),
    )
    torch.testing.assert_close(
        features[:, 2] + features[:, 1].abs() + features[:, 3],
        torch.log(rows[1:, 1] / rows[1:, 2]),
    )
    future = rows.clone()
    future[-1, :4] *= 2.0
    assert torch.equal(feature_values(future, "stationary-v1")[0], features[0])
    assert torch.equal(feature_values(torch.ones(2, 5), "stationary-v1"),
                       torch.zeros(1, 5))
    assert feature_lookback("ohlcv") == 0
    assert feature_lookback("stationary-v1") == 1
    invalid = []
    for column, value in ((0, 0.0), (2, 104.0), (4, -1.0)):
        bad = rows.clone()
        bad[1, column] = value
        invalid.append(bad)
    for bad in invalid:
        try:
            feature_values(bad, "stationary-v1")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid stationary source bar was accepted")


def verify_selection_is_validation_only() -> None:
    candidates = (candidate("first", 3), candidate("second", 5))
    records = [
        {"model": "transformer", "candidate": "first",
         "validation_scaled_mse": 2.0, "test": 0.0},
        {"model": "transformer", "candidate": "second",
         "validation_scaled_mse": 1.0, "test": 100.0},
    ]
    selected = select_candidates(records, ("transformer",), candidates)
    assert selected["transformer"]["candidate"] == "second"
    records[0]["test"], records[1]["test"] = 1_000.0, -1_000.0
    assert select_candidates(records, ("transformer",), candidates) == selected


def verify_selected_epochs() -> None:
    records = [
        {"model": "transformer", "candidate": "raw", "series": "TEST",
         "seed": 7, "best_epoch": 3},
        {"model": "transformer", "candidate": "raw", "series": "TEST",
         "seed": 7, "best_epoch": 7},
    ]
    assert _selected_epochs(records, "transformer", "raw", "TEST", 7) == 3


def verify_ridge() -> None:
    torch.manual_seed(13)
    features, closes = torch.randn(40, 5), torch.ones(40)
    windows = features.unfold(0, 2, 1).transpose(1, 2).reshape(-1, 10)
    weight = torch.linspace(-0.5, 0.5, 10)
    targets = windows @ weight + 0.25
    dataset = Windows(features, targets, closes, closes, 2, 0, 30)
    data = TrainingData(dataset, dataset, dataset, torch.zeros(5),
                        torch.ones(5), torch.tensor(0.0), torch.tensor(1.0))
    torch.testing.assert_close(linear_model(data, 1e-8)(windows[:30]), targets[:30],
                               rtol=1e-5, atol=1e-5)


def verify_model_fingerprint(data: TrainingData) -> None:
    configuration = candidate("fingerprint", data.train.seq_len)

    def fitted() -> ForecastTransformer:
        torch.manual_seed(23)
        model = ForecastTransformer(configuration.config())
        fit_epochs(model, data, 8, 2, 1e-3, 1e-4, 31, torch.device("cpu"))
        return model

    model = fitted()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    first = _model_fingerprint(model, data, configuration)
    assert first == _model_fingerprint(model, data, configuration)

    with torch.no_grad():
        next(model.parameters()).view(-1)[0].add_(1)
    assert _model_fingerprint(model, data, configuration) != first

    restored = ForecastTransformer(configuration.config())
    restored.load_state_dict(state)
    changed_scale = replace(data, target_scale=data.target_scale * 2)
    assert _model_fingerprint(restored, changed_scale, configuration) != first
    assert _model_fingerprint(fitted(), data, configuration) == \
        _model_fingerprint(fitted(), data, configuration)


def verify_caps(directory: Path) -> None:
    path = directory / "oversized.json"
    base = {"name": "oversized", "seq_len": 410, "model_dim": 4,
            "heads": 2, "ff_dim": 6, "layers": 1}
    for model, extra, message in (
        ("linear", {}, "linear feature cap"),
        ("mlp", {"seq_len": 32, "mlp_dim": 100_000}, "MLP parameter cap"),
    ):
        path.write_text(json.dumps({"models": [model], "candidates": [base | extra]}),
                        encoding="utf-8")
        try:
            Sweep.read(path)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"{model} size cap was not enforced")


def verify_horizon_targets(csv: Path) -> TrainingData:
    selected = candidate("horizon", 3)
    config = selected.config()
    implicit = prepare_data(csv, config, 0.7, 0.15, (20, 10, 10))
    explicit = prepare_data(csv, config, 0.7, 0.15, (20, 10, 10),
                            horizon_bars=1)
    torch.testing.assert_close(outcomes(implicit.train), outcomes(explicit.train),
                               rtol=0.0, atol=0.0)
    automatic = prepare_data(csv, config, 0.6, 0.2,
                             horizon_bars=3, split_gap=2)
    assert sum(map(len, (automatic.train, automatic.validation,
                        automatic.test))) + 4 == 51
    def as_of(dataset: Windows, offset: int) -> int:
        return feature_lookback(selected.feature_set) + dataset.start + \
            selected.seq_len - 1 + offset

    assert as_of(automatic.train, len(automatic.train) - 1) + \
        automatic.horizon_bars <= as_of(automatic.validation, 0)
    assert as_of(automatic.validation, len(automatic.validation) - 1) + \
        automatic.horizon_bars <= as_of(automatic.test, 0)

    one = prepare_data(csv, config, 0.7, 0.15, (20, 10, 10), 2,
                       horizon_bars=1, split_gap=2)
    three = prepare_data(csv, config, 0.7, 0.15, (20, 10, 10), 0,
                         horizon_bars=3, split_gap=2)
    _, target, latest, actual = three.train[0]
    torch.testing.assert_close(latest, torch.tensor(close_value(2)))
    torch.testing.assert_close(actual, torch.tensor(close_value(5)))
    torch.testing.assert_close(latest * (target * three.target_scale +
                                         three.target_mean).exp(), actual)
    executable = prepare_data(
        csv, config, 0.7, 0.15, (20, 10, 10), horizon_bars=3,
        split_gap=2, target_kind=EXECUTABLE_RETURN_TARGET,
    )
    _, target, reference, actual = executable.train[0]
    expected_open = close_value(3) - 0.1 - 0.01 * (3 % 2)
    torch.testing.assert_close(reference, torch.tensor(expected_open))
    torch.testing.assert_close(actual, torch.tensor(close_value(5)))
    torch.testing.assert_close(
        reference * (target * executable.target_scale +
                     executable.target_mean).exp(), actual,
    )
    assert executable.target_kind == EXECUTABLE_RETURN_TARGET
    stationary = prepare_data(
        csv, config, 0.7, 0.15, (20, 10, 10), feature_set="stationary-v1",
        horizon_bars=3, split_gap=2,
        target_kind=EXECUTABLE_RETURN_TARGET,
    )
    _, _, reference, actual = stationary.train[0]
    expected_open = close_value(4) - 0.1 - 0.01 * (4 % 2)
    torch.testing.assert_close(reference, torch.tensor(expected_open))
    torch.testing.assert_close(actual, torch.tensor(close_value(6)))
    for left, right in zip((one.train, one.validation, one.test),
                           (three.train, three.validation, three.test), strict=True):
        torch.testing.assert_close(
            torch.stack([left[index][3] for index in range(len(left))]),
            torch.stack([right[index][3] for index in range(len(right))]),
            rtol=0.0, atol=0.0,
        )
    try:
        prepare_data(csv, config, 0.7, 0.15, horizon_bars=0)
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive horizon was accepted")
    try:
        prepare_data(csv, config, 0.7, 0.15, target_kind="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported target kind was accepted")
    bad_open = csv.with_name("bad-open.csv")
    lines = csv.read_text(encoding="ascii").splitlines()
    fields = lines[4].split(",")
    fields[1] = "0"
    lines[4] = ",".join(fields)
    bad_open.write_text("\n".join(lines), encoding="ascii")
    try:
        prepare_data(
            bad_open, config, 0.7, 0.15, target_kind=EXECUTABLE_RETURN_TARGET,
        )
    except ValueError as error:
        assert "reference prices" in str(error)
    else:
        raise AssertionError("nonpositive executable reference was accepted")
    return three


def verify_horizon_reports(csv: Path) -> None:
    ledgers: list[list[dict[str, object]]] = [[], []]
    calibration_ledgers: list[list[dict[str, object]]] = [[], []]
    reports = tuple(
        run_experiment(
            Sweep((candidate("raw-3", 3),), ("last_close",), (3,), 1, 0.2,
                  1, 1, 8, horizon, 3),
            (("SYNTH", csv),), torch.device("cpu"), 2, ledger,
            calibration_ledger, test_authorizer=authorize("last_close"),
            requested_models=frozenset(("last_close",)),
        )
        for horizon, ledger, calibration_ledger in zip(
            (1, 3), ledgers, calibration_ledgers, strict=True,
        )
    )
    for section in ("validation", "calibration", "test"):
        assert [record["targets"] for record in reports[0][section]] == \
            [record["targets"] for record in reports[1][section]]
        assert [record["samples"] for record in reports[0][section]] == \
            [record["samples"] for record in reports[1][section]]
    assert [report["protocol"]["target_horizon_bars"] for report in reports] == [1, 3]
    assert all(report["protocol"]["embargo_bars"] == 2 for report in reports)
    for report, ledger, calibration_ledger in zip(
        reports, ledgers, calibration_ledgers, strict=True,
    ):
        assert report["schema"] == 6
        assert report["protocol"]["target_kind"] == CLOSE_RETURN_TARGET
        assert len(ledger) == report["test"][0]["samples"]
        assert len(calibration_ledger) == report["calibration"][0]["samples"]
        assert all(record["schema"] == 3 and record["split"] == "test" and
                   record["fold"] is None and
                   record["target_kind"] == CLOSE_RETURN_TARGET
                   for record in ledger)
        assert all(record["schema"] == 3 and
                   record["split"] == "calibration" and
                   record["fold"] is None for record in calibration_ledger)
        assert all(record["predicted_log_return"] == 0.0 for record in ledger)
        assert all(record["csv_sha256"] == report["series"][0]["sha256"]
                   for record in ledger)
        assert ledger[0]["target_time"] == report["test"][0]["targets"]["test"][0]
    for record in (*reports[1]["validation"], *reports[1]["test"]):
        targets = record["targets"]
        assert datetime.fromisoformat(targets["validation"][0]) - \
            timedelta(minutes=3) >= datetime.fromisoformat(targets["train"][1])
        if "test" in targets:
            assert datetime.fromisoformat(targets["test"][0]) - \
                timedelta(minutes=3) >= datetime.fromisoformat(
                    targets["validation"][1]
                )
    selection_end = max(record["targets"]["validation"][1]
                        for record in reports[1]["validation"])
    test_start = min(record["targets"]["test"][0]
                     for record in reports[1]["test"])
    assert datetime.fromisoformat(test_start) - timedelta(minutes=3) >= \
        datetime.fromisoformat(selection_end)
    try:
        Sweep((candidate("invalid", 3),), ("last_close",), (3,), 1, 0.2,
              1, 1, 8, 3, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("horizon greater than alignment was accepted")
    executable = run_experiment(
        Sweep((candidate("executable", 3),), ("last_close",), (3,),
              1, 0.2, 1, 1, 8, 3, 3, EXECUTABLE_RETURN_TARGET),
        (("SYNTH", csv),), torch.device("cpu"), 2,
        test_authorizer=authorize("last_close"),
        requested_models=frozenset(("last_close",)),
    )
    assert executable["protocol"]["target_kind"] == EXECUTABLE_RETURN_TARGET
    assert executable["protocol"]["zero_return_reference"] == "open[t + 1]"
    calibration_only = run_experiment(
        Sweep((candidate("executable", 3),), ("last_close",), (3,),
              1, 0.2, 1, 1, 8, 3, 3, EXECUTABLE_RETURN_TARGET),
        (("SYNTH", csv),), torch.device("cpu"), 2,
        evaluate_test=False,
        requested_models=frozenset(),
    )
    assert calibration_only["protocol"]["phase"] == "selection-and-calibration"
    assert calibration_only["test"] == []
    assert calibration_only["test_contract"][0]["samples"] > 0
    policy = {
        "calibration_fingerprint": experiment_fingerprint(calibration_only),
        "model": "last_close", "candidate": "executable",
        "feature_set": "ohlcv", "target_kind": EXECUTABLE_RETURN_TARGET,
        "horizon_bars": 3, "seeds": [], "series": ["SYNTH"],
        "test_grid": calibration_only["test_contract"],
        "model_fingerprints": calibration_only["model_fingerprints"],
    }
    first_reconstruction = json.loads(json.dumps(calibration_only))
    reconstructed = json.loads(json.dumps(calibration_only))
    assert first_reconstruction["model_fingerprints"] == \
        reconstructed["model_fingerprints"]
    reconstructed_policy = json.loads(json.dumps(policy))
    assert _authorize_test(reconstructed, (reconstructed_policy,)) == {
        "last_close": reconstructed_policy,
    }
    reordered = calibration_only | {
        "sweep": calibration_only["sweep"] | {
            "models": ["transformer"], "seeds": [7, 3],
        },
        "selection": {"transformer": {"candidate": "executable"}},
        "model_fingerprints": [
            {
                "model": "transformer", "series": "SYNTH",
                "seed": seed, "epochs": 1, "sha256": str(seed) * 64,
            }
            for seed in (3, 7)
        ],
    }
    seeded_policy = policy | {
        "calibration_fingerprint": experiment_fingerprint(reordered),
        "model": "transformer", "seeds": [3, 7],
        "model_fingerprints": reordered["model_fingerprints"],
    }
    assert _authorize_test(reordered, (seeded_policy,)) == {
        "transformer": seeded_policy,
    }
    changed_digest = seeded_policy | {
        "model_fingerprints": [
            seeded_policy["model_fingerprints"][0] | {"sha256": "0" * 64},
            seeded_policy["model_fingerprints"][1],
        ],
    }
    missing_seed = seeded_policy | {
        "model_fingerprints": seeded_policy["model_fingerprints"][:1],
    }
    extra_model = seeded_policy | {
        "model_fingerprints": [
            *seeded_policy["model_fingerprints"],
            seeded_policy["model_fingerprints"][0] | {"model": "mlp"},
        ],
    }
    reversed_entries = seeded_policy | {
        "model_fingerprints": list(
            reversed(seeded_policy["model_fingerprints"])
        ),
    }
    for policies in (
        (), (policy | {"calibration_fingerprint": "0" * 64},),
        (changed_digest,), (missing_seed,), (extra_model,), (reversed_entries,),
    ):
        try:
            _authorize_test(
                reordered if policies and policies[0].get("model") ==
                "transformer" else calibration_only,
                policies,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid test authorization was accepted")

    multi_contract = run_experiment(
        Sweep(
            (candidate("multi", 3),),
            ("last_close", "rolling_mean"), (3,), 1, 0.2, 1, 1, 8,
            3, 3, EXECUTABLE_RETURN_TARGET,
        ),
        (("SYNTH", csv),), torch.device("cpu"), 4,
        evaluate_test=False, requested_models=frozenset(),
    )
    policies = tuple(
        frozen_policy(multi_contract, model)
        for model in ("last_close", "rolling_mean")
    )
    assert _authorize_test(multi_contract, policies) == {
        policy["model"]: policy for policy in policies
    }
    mismatched = validate_policy(policies[1] | {"candidate": "mismatch"})
    for invalid in ((policies[0], policies[0]), (policies[0], mismatched)):
        try:
            _authorize_test(multi_contract, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid multi-policy authorization was accepted")


def verify_test_state(csv: Path) -> None:
    selected = candidate("retained", 3)
    data = prepare_data(
        csv, selected.config(), 0.7, 0.15, (20, 10, 10),
        feature_set=selected.feature_set,
    )
    torch.manual_seed(7)
    model = ForecastTransformer(selected.config())
    digest = _model_fingerprint(model, data, selected)
    fingerprint = {
        "model": "transformer", "series": "SYNTH", "seed": 7,
        "epochs": 1, "sha256": digest,
    }
    policy = {"model_fingerprints": [fingerprint]}

    class UnreadTestBatches:
        def __iter__(self) -> object:
            raise AssertionError("test batches were read before state verification")

    for fingerprints in ([], [fingerprint, fingerprint]):
        try:
            _verify_test_state(
                model, data, selected, fingerprint,
                {"model_fingerprints": fingerprints}, "SYNTH", 7,
            )
            evaluate(
                model, UnreadTestBatches(), data.target_mean, data.target_scale,
                torch.device("cpu"),
            )
        except ValueError as error:
            assert "frozen policy" in str(error)
        else:
            raise AssertionError("invalid frozen-policy state was accepted")

    with torch.no_grad():
        next(model.parameters()).add_(1.0)

    with patch("tools.experiment.evaluate") as test_evaluate:
        try:
            _verify_test_state(
                model, data, selected, fingerprint, policy, "SYNTH", 7,
            )
            test_evaluate(
                model, UnreadTestBatches(), data.target_mean,
                data.target_scale, torch.device("cpu"),
            )
        except ValueError as error:
            assert "frozen policy" in str(error)
        else:
            raise AssertionError("mutated retained model state was accepted")
        test_evaluate.assert_not_called()


def verify_panel_orchestration() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-source-tree-", dir=ROOT,
    ) as directory:
        package = Path(directory)
        (package / "_C").mkdir()
        (package / "_C" / "_VariableFunctions.pyi").write_text(
            "nested\n", encoding="ascii",
        )
        (package / "_C.cpython-314-darwin.so").write_text(
            "sibling\n", encoding="ascii",
        )
        observed = source_tree(package)
        paths = tuple(item.path for item in observed.files)
        assert paths == tuple(sorted(paths))
        encoded = json.loads(json.dumps(asdict(observed)))
        assert SourceTree.parse(encoded, "package") == observed

    selected = candidate("panel", 3)
    sweep = Sweep(
        (selected,), ("panel_transformer",), (7, 19, 31, 43, 61),
        2, 0.1, 1, 1, 16,
    )
    fixed = replace(
        sweep,
        models=(
            "transformer", "linear", "mlp", "rolling_mean", "last_close",
            "panel_transformer",
        ),
    )
    assert expected_runs(fixed, 3) == 162
    assert expected_runs(sweep, 3) == 45
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-default-", dir=ROOT,
    ) as directory:
        path = Path(directory) / "implicit.json"
        write_canonical(path, {"candidates": [asdict(selected)]})
        assert Sweep.read(path).models == (
            "transformer", "linear", "mlp", "rolling_mean", "last_close",
        )
    try:
        run_experiment(
            sweep, (("A", Path("missing.csv")),), torch.device("cpu"), 45,
            evaluate_test=False, requested_models=frozenset(),
        )
    except ValueError as error:
        assert "bound calibration-only execution" in str(error)
    else:
        raise AssertionError("manifestless panel execution was accepted")

    with patch("tools.experiment.read_bars") as read, \
         patch("tools.experiment._candidate_data") as prepared, \
         patch("tools.experiment._fit_neural") as fitted:
        try:
            _run_experiment(
                sweep, (), torch.device("cpu"), 45, None, None, True,
                None, frozenset(), None,
            )
        except ValueError as error:
            assert "bound calibration-only execution" in str(error)
        else:
            raise AssertionError("test-mode panel execution was accepted")
        read.assert_not_called()
        prepared.assert_not_called()
        fitted.assert_not_called()
    try:
        _authorize_test({
            "sweep": {"models": ["panel_transformer"]},
        }, ())
    except ValueError as error:
        assert "calibration-only" in str(error)
    else:
        raise AssertionError("panel test authorization was accepted")

    timestamps = tuple(f"2026-01-01T00:00:{index:02d}Z" for index in range(10))
    _label_available(timestamps, 2, (5, 1), 1, 1)
    try:
        _label_available(timestamps, 2, (5, 1), 0, 2)
    except ValueError as error:
        assert "unavailable" in str(error)
    else:
        raise AssertionError("unavailable training labels were accepted")

    epoch_records = [
        {
            "model": "panel_transformer", "candidate": "panel",
            "series": series, "fold": fold, "seed": 7,
            "best_epoch": 3 + fold,
        }
        for series in ("A", "B", "C") for fold in range(2)
    ]
    assert _panel_selected_epochs(
        epoch_records, "panel", ("A", "B", "C"), 7, 2,
    ) == 3
    for invalid in (
        epoch_records[:-1],
        [*epoch_records, epoch_records[0]],
        [
            record | {"best_epoch": 9}
            if record["series"] == "C" and record["fold"] == 1 else record
            for record in epoch_records
        ],
    ):
        try:
            _panel_selected_epochs(
                invalid, "panel", ("A", "B", "C"), 7, 2,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid shared panel epochs were accepted")

    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-", dir=ROOT,
    ) as directory:
        with panel_fixture(Path(directory), sweep) as execution:
            report_path = Path(
                execution.attempt.outputs["experiment_report"],
            )
            ledger_path = Path(
                execution.attempt.outputs["calibration_ledger"],
            )
            execution.validate_outputs(report_path, ledger_path)
            for report, ledger in (
                (report_path.with_name("wrong.json"), ledger_path),
                (report_path, None),
            ):
                try:
                    execution.validate_outputs(report, ledger)
                except ValueError as error:
                    assert "outputs do not match" in str(error)
                else:
                    raise AssertionError("unbound panel outputs were accepted")
            try:
                execution.inputs.validate_timestamps(tuple(
                    (name, frozen, ()) for name, frozen in execution.series
                ))
            except ValueError as error:
                assert "empty" in str(error)
            else:
                raise AssertionError("empty timestamp grid was accepted")
            members = [
                prepare_data(
                    frozen.snapshot, selected.config(), 0.7, 0.15,
                    (40, 10, 10),
                )
                for _, frozen in execution.series
            ]
            pooled = _panel_data(members)
            for name in ("train", "validation", "test"):
                combined = getattr(pooled, name)
                parts = [getattr(member, name) for member in members]
                assert len(combined) == sum(map(len, parts))
                assert tuple(combined.datasets) == tuple(parts)
                left = combined[len(parts[0]) - 1][0]
                right = combined[len(parts[0])][0]
                assert torch.equal(left, parts[0][len(parts[0]) - 1][0])
                assert torch.equal(right, parts[1][0][0])
            assert len({
                tuple(member.feature_mean.tolist()) for member in members
            }) == 3
            late = Path(directory) / "late.csv"
            late.write_bytes(execution.series[0][1].snapshot.read_bytes())
            lines = late.read_text(encoding="ascii").splitlines()
            for index in range(50, len(lines)):
                fields = lines[index].split(",")
                fields[1:] = [
                    str(float(value) * 1.5) for value in fields[1:]
                ]
                lines[index] = ",".join(fields)
            late.write_text("\n".join(lines), encoding="ascii")
            late_data = prepare_data(
                late, selected.config(), 0.7, 0.15, (40, 10, 10),
            )
            for left, right in zip(
                (
                    members[0].feature_mean, members[0].feature_scale,
                    members[0].target_mean, members[0].target_scale,
                ),
                (
                    late_data.feature_mean, late_data.feature_scale,
                    late_data.target_mean, late_data.target_scale,
                ),
                strict=True,
            ):
                assert torch.equal(left, right)
            for index in range(len(members[0].train)):
                assert torch.equal(
                    members[0].train[index][0], late_data.train[index][0],
                )
            torch.manual_seed(3)
            shared = ForecastTransformer(selected.config())
            before = [
                _model_fingerprint(shared, member, selected)
                for member in members
            ]
            changed = [members[0], replace(
                members[1], target_scale=members[1].target_scale * 2,
            ), members[2]]
            after = [
                _model_fingerprint(shared, member, selected)
                for member in changed
            ]
            assert before[0] == after[0] and before[1] != after[1] and \
                before[2] == after[2]

            with patch(
                "tools.experiment._panel_selected_epochs",
                side_effect=(1, ValueError("invalid later seed")),
            ), patch("tools.experiment.fit_epochs") as final_fit:
                try:
                    run_experiment(
                        sweep, execution.series, torch.device("cpu"), 45,
                        evaluate_test=False, requested_models=frozenset(),
                        panel_execution=execution,
                    )
                except ValueError as error:
                    assert "invalid later seed" in str(error)
                else:
                    raise AssertionError(
                        "invalid later seed fitted an earlier final model"
                    )
                final_fit.assert_not_called()

            ledger: list[dict[str, object]] = []
            model_ids = []

            def fingerprint(model: object, data: TrainingData,
                            configuration: Candidate) -> str:
                model_ids.append(id(model))
                return _model_fingerprint(model, data, configuration)

            with patch(
                "tools.experiment._fit_neural",
                wraps=_fit_neural,
            ) as validation_fit, patch(
                "tools.experiment.fit_epochs", wraps=fit_epochs,
            ) as calibration_fit, patch(
                "tools.experiment._model_fingerprint",
                side_effect=fingerprint,
            ):
                report = run_experiment(
                    sweep, execution.series, torch.device("cpu"), 45,
                    calibration_prediction_records=ledger,
                    evaluate_test=False, requested_models=frozenset(),
                    panel_execution=execution,
                )
            assert validation_fit.call_count == 10
            assert calibration_fit.call_count == 5
            assert len(set(model_ids)) == 5
            assert all(model_ids.count(model) == 3 for model in set(model_ids))
            expected_validation = [
                (series, fold, seed)
                for series in ("A", "B", "C")
                for fold in range(2) for seed in sweep.seeds
            ]
            assert [
                (record["series"], record["fold"], record["seed"])
                for record in report["validation"]
            ] == expected_validation
            assert [
                (record["series"], record["seed"])
                for record in report["calibration"]
            ] == [
                (series, seed)
                for series in ("A", "B", "C") for seed in sweep.seeds
            ]
            for fold in range(2):
                for seed in sweep.seeds:
                    epochs = {
                        record["best_epoch"] for record in report["validation"]
                        if record["fold"] == fold and record["seed"] == seed
                    }
                    assert len(epochs) == 1
            assert [
                (record["series"], record["seed"], record["target_time"])
                for record in ledger
            ] == sorted(
                (
                    record["series"], record["seed"], record["target_time"]
                )
                for record in ledger
            )
            for seed in sweep.seeds:
                hashes = {
                    item["sha256"] for item in report["model_fingerprints"]
                    if item["seed"] == seed
                }
                assert len(hashes) == 3
            repeated_ledger: list[dict[str, object]] = []
            repeated = run_experiment(
                sweep, execution.series, torch.device("cpu"), 45,
                calibration_prediction_records=repeated_ledger,
                evaluate_test=False, requested_models=frozenset(),
                panel_execution=execution,
            )
            assert repeated == report
            assert repeated_ledger == ledger

        occupied = Path(directory) / "occupied.json"
        occupied.write_text("original", encoding="ascii")
        try:
            write_report(occupied, {"changed": True}, exclusive=True)
        except FileExistsError:
            pass
        else:
            raise AssertionError("exclusive report replaced an existing file")
        assert occupied.read_text(encoding="ascii") == "original"
        target = Path(directory) / "target.json"
        target.write_text("target", encoding="ascii")
        symlink = Path(directory) / "symlink.json"
        symlink.symlink_to(target.name)
        try:
            write_report(symlink, {"changed": True}, exclusive=True)
        except FileExistsError:
            pass
        else:
            raise AssertionError("exclusive report followed a symlink")
        assert target.read_text(encoding="ascii") == "target"
        stable = Path(directory) / "stable"
        moved = Path(directory) / "moved"
        stable.mkdir()
        descriptor = os.open(stable, os.O_RDONLY)
        try:
            stable.rename(moved)
            stable.mkdir()
            write_report(
                stable / "bound.json", {"bound": True}, exclusive=True,
                directory_fd=descriptor,
            )
            assert (moved / "bound.json").is_file()
            assert not (stable / "bound.json").exists()
        finally:
            os.close(descriptor)

    for mutation in (
        "config", "baseline", "csv", "source", "binary", "runtime", "cache",
        "argv", "output", "symlink", "hardlink", "runtime-after-freeze",
        "directory-after-freeze",
    ):
        with tempfile.TemporaryDirectory(
            prefix=f"compose-mini-panel-{mutation}-", dir=ROOT,
        ) as directory, \
             patch("tools.experiment._candidate_data") as prepared, \
             patch("tools.experiment._fit_neural") as fitted, \
             patch("tools.experiment.fit_epochs") as final_fit:
            try:
                with panel_fixture(
                    Path(directory), sweep, mutation=mutation,
                ) as execution:
                    if mutation.endswith("-after-freeze"):
                        run_experiment(
                            sweep, execution.series, torch.device("cpu"), 45,
                            evaluate_test=False,
                            requested_models=frozenset(),
                            panel_execution=execution,
                        )
            except ValueError:
                pass
            else:
                raise AssertionError(f"{mutation} mutation was accepted")
            prepared.assert_not_called()
            fitted.assert_not_called()
            final_fit.assert_not_called()

    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-grid-", dir=ROOT,
    ) as directory, panel_fixture(
        Path(directory), sweep, mismatched=True,
    ) as execution, patch("tools.experiment._fit_neural") as fitted:
        try:
            run_experiment(
                sweep, execution.series, torch.device("cpu"), 45,
                evaluate_test=False, requested_models=frozenset(),
                panel_execution=execution,
            )
        except ValueError as error:
            assert "identical timestamp grids" in str(error)
        else:
            raise AssertionError("mismatched panel timestamps were accepted")
        fitted.assert_not_called()


def main() -> None:
    assert walk_forward_splits(20, 2, 0.2) == ((8, 4), (12, 4))
    assert walk_forward_splits(100, 2, 0.1) == ((70, 10), (80, 10))
    assert walk_forward_splits(100, 2, 0.1, 2) == ((60, 10), (70, 10))
    assert holdout_split(20, 0.2) == (12, 4, 4)
    assert purged_split((12, 4, 4), 2) == (10, 2, 4)
    assert purged_split((12, 4), 2, preserve_last=False) == (10, 2)
    verify_panel_orchestration()
    verify_selection_is_validation_only()
    verify_selected_epochs()
    verify_ridge()
    verify_stationary_features()
    sweep = Sweep(
        (candidate("short", 3), candidate("stationary", 5, "stationary-v1")),
        ("transformer", "linear", "mlp", "rolling_mean", "last_close"),
        (3,), 2, 0.2, 2, 2, 8,
    )
    assert expected_runs(sweep, 1) == 25
    horizon_sweep = Sweep.read(ROOT / "experiments/horizons.example.json")
    assert expected_runs(horizon_sweep, 3) == 117
    feature_path = ROOT / "experiments/executable-h13-features.example.json"
    candidate_defaults = {
        "ff_dim": 32,
        "heads": 2,
        "layers": 1,
        "learning_rate": 0.0003,
        "mlp_dim": 32,
        "model_dim": 16,
        "ridge": 0.001,
        "rolling_window": 8,
        "weight_decay": 0.0001,
    }
    assert json.loads(feature_path.read_text(encoding="utf-8")) == {
        "batch_size": 128,
        "candidates": [
            candidate_defaults | {
                "feature_set": "ohlcv",
                "name": "raw-17",
                "seq_len": 17,
            },
            candidate_defaults | {
                "feature_set": "stationary-v1",
                "name": "stationary-16",
                "seq_len": 16,
            },
        ],
        "epochs": 100,
        "fold_fraction": 0.1,
        "folds": 2,
        "models": ["transformer"],
        "patience": 10,
        "seeds": [7, 19, 31, 43, 61],
        "target_horizon_bars": 13,
        "alignment_horizon_bars": 13,
        "target_kind": EXECUTABLE_RETURN_TARGET,
    }
    feature_sweep = Sweep.read(feature_path)
    assert feature_sweep == Sweep(
        (
            Candidate(
                "raw-17", 17, 16, 2, 32, 1, 3e-4, 1e-4,
                32, 8, 1e-3, "ohlcv",
            ),
            Candidate(
                "stationary-16", 16, 16, 2, 32, 1, 3e-4, 1e-4,
                32, 8, 1e-3, "stationary-v1",
            ),
        ),
        ("transformer",), (7, 19, 31, 43, 61),
        2, 0.1, 100, 10, 128, 13, 13, EXECUTABLE_RETURN_TARGET,
    )
    assert expected_runs(feature_sweep, 3) == 75
    with tempfile.TemporaryDirectory(prefix="compose-mini-experiment-") as directory:
        verify_caps(Path(directory))
        csv = Path(directory) / "series.csv"
        changed = Path(directory) / "changed.csv"
        output = Path(directory) / "report.json"
        repeated_output = Path(directory) / "report-again.json"
        predictions_output = Path(directory) / "predictions.jsonl"
        repeated_predictions_output = Path(directory) / "predictions-again.jsonl"
        calibration_predictions_output = Path(directory) / "calibration.jsonl"
        repeated_calibration_output = Path(directory) / "calibration-again.jsonl"
        write_csv(csv)
        verify_test_state(csv)
        three_horizon = verify_horizon_targets(csv)
        verify_horizon_reports(csv)
        lines = csv.read_text(encoding="ascii").splitlines()
        for index in range(27, len(lines)):
            fields = lines[index].split(",")
            fields[1:] = [str(float(value) * 1.5) for value in fields[1:]]
            lines[index] = ",".join(fields)
        changed.write_text("\n".join(lines), encoding="ascii")
        changed_horizon = prepare_data(
            changed, candidate("horizon-leak", 3).config(), 0.7, 0.15,
            (20, 10, 10), horizon_bars=3, split_gap=2,
        )
        for left, right in zip(
            (three_horizon.feature_mean, three_horizon.feature_scale,
             three_horizon.target_mean, three_horizon.target_scale),
            (changed_horizon.feature_mean, changed_horizon.feature_scale,
             changed_horizon.target_mean, changed_horizon.target_scale), strict=True,
        ):
            assert torch.equal(left, right)
        base = prepare_data(csv, candidate("leak", 5).config(), 0.7, 0.15,
                            (20, 10, 10))
        verify_model_fingerprint(base)
        perturbed = prepare_data(changed, candidate("leak", 5).config(), 0.7, 0.15,
                                 (20, 10, 10))
        for left, right in zip(
            (base.feature_mean, base.feature_scale, base.target_mean, base.target_scale),
            (perturbed.feature_mean, perturbed.feature_scale,
             perturbed.target_mean, perturbed.target_scale), strict=True,
        ):
            assert torch.equal(left, right)
        assert torch.equal(linear_model(base, 1e-3).weight,
                           linear_model(perturbed, 1e-3).weight)
        stationary = prepare_data(
            csv, candidate("stationary-leak", 5).config(), 0.7, 0.15,
            (20, 10, 10), feature_set="stationary-v1",
        )
        stationary_perturbed = prepare_data(
            changed, candidate("stationary-leak", 5).config(), 0.7, 0.15,
            (20, 10, 10), feature_set="stationary-v1",
        )
        for left, right in zip(
            (stationary.feature_mean, stationary.feature_scale,
             stationary.target_mean, stationary.target_scale),
            (stationary_perturbed.feature_mean, stationary_perturbed.feature_scale,
             stationary_perturbed.target_mean, stationary_perturbed.target_scale),
            strict=True,
        ):
            assert torch.equal(left, right)
        short = prepare_data(csv, candidate("short-aligned", 3).config(),
                             0.7, 0.15, (20, 10, 10), 2)
        for left, right in zip(
            (short.train, short.validation, short.test),
            (base.train, base.validation, base.test), strict=True,
        ):
            torch.testing.assert_close(outcomes(left), outcomes(right),
                                       rtol=0.0, atol=0.0)
        raw_history = prepare_data(csv, candidate("raw-history", 6).config(),
                                   0.7, 0.15, (20, 10, 10))
        for raw_split, stationary_split in zip(
            (raw_history.train, raw_history.validation, raw_history.test),
            (stationary.train, stationary.validation, stationary.test), strict=True,
        ):
            torch.testing.assert_close(outcomes(raw_split), outcomes(stationary_split),
                                       rtol=0.0, atol=0.0)
        loader = data_loaders(base, 8, 0)[1]
        constant = ConstantReturn(base)
        constant_metrics = evaluate(constant, loader, base.target_mean,
                                    base.target_scale, torch.device("cpu"))
        assert constant_metrics["direction_accuracy"] == 0.0
        features = next(iter(loader))[0]
        rolling = RollingMean(base, 2)(features) * base.target_scale + base.target_mean
        closes = features[:, :, 3] * base.feature_scale[3] + base.feature_mean[3]
        expected = torch.log(closes[:, 1:] / closes[:, :-1])[:, -2:].mean(1)
        torch.testing.assert_close(rolling, expected)
        horizon_features = next(iter(data_loaders(three_horizon, 8, 0)[1]))[0]
        horizon_rolling = RollingMean(three_horizon, 2)(horizon_features) * \
            three_horizon.target_scale + three_horizon.target_mean
        horizon_closes = horizon_features[:, :, 3] * three_horizon.feature_scale[3] + \
            three_horizon.feature_mean[3]
        horizon_expected = torch.log(
            horizon_closes[:, 1:] / horizon_closes[:, :-1]
        )[:, -2:].mean(1) * 3
        torch.testing.assert_close(horizon_rolling, horizon_expected)
        stationary_three = prepare_data(
            csv, candidate("stationary-horizon", 5).config(), 0.7, 0.15,
            (20, 10, 10), feature_set="stationary-v1", horizon_bars=3,
        )
        stationary_loader = data_loaders(stationary_three, 8, 0)[1]
        stationary_features = next(iter(stationary_loader))[0]
        stationary_rolling = RollingMean(stationary_three, 2)(stationary_features)
        stationary_raw = stationary_features * stationary_three.feature_scale + \
            stationary_three.feature_mean
        stationary_expected = stationary_raw[:, -2:, :2].sum(2).mean(1) * 3
        torch.testing.assert_close(
            stationary_rolling * stationary_three.target_scale +
            stationary_three.target_mean,
            stationary_expected,
        )
        try:
            run_experiment(
                sweep, (("SYNTH", csv),), torch.device("cpu"), 24,
                test_authorizer=authorize(*sweep.models),
                requested_models=frozenset(sweep.models),
            )
        except ValueError as error:
            assert "requires 25 runs" in str(error)
        else:
            raise AssertionError("run limit was not enforced")
        predictions: list[dict[str, object]] = []
        repeated_predictions: list[dict[str, object]] = []
        calibration_predictions: list[dict[str, object]] = []
        repeated_calibration_predictions: list[dict[str, object]] = []
        with patch("tools.experiment.fit_epochs", wraps=fit_epochs) as final_fit, \
             patch("tools.experiment._candidate_data",
                   wraps=_candidate_data) as prepared:
            report = run_experiment(
                sweep, (("SYNTH", csv),), torch.device("cpu"), 25, predictions,
                calibration_predictions,
                test_authorizer=authorize(*sweep.models),
                requested_models=frozenset(sweep.models),
            )
        assert final_fit.call_count == 2
        assert prepared.call_count == sweep.folds * len(sweep.candidates) + len({
            item["candidate"] for item in report["selection"].values()
        })
        assert len({id(call.args[0]) for call in prepared.call_args_list}) == 1
        assert sorted(call.args[3] for call in final_fit.call_args_list) == sorted(
            item["epochs"] for item in report["model_fingerprints"]
            if item["epochs"] is not None
        )
        repeated = run_experiment(
            sweep, (("SYNTH", csv),), torch.device("cpu"), 25,
            repeated_predictions, repeated_calibration_predictions,
            test_authorizer=authorize(*sweep.models),
            requested_models=frozenset(sweep.models),
        )
        assert repeated == report
        assert repeated_predictions == predictions
        assert repeated_calibration_predictions == calibration_predictions
        assert report["schema"] == 6
        assert report["protocol"]["aligned_history_bars"] == 6
        assert report["protocol"]["target_horizon_bars"] == 1
        assert report["protocol"]["alignment_horizon_bars"] == 1
        assert report["protocol"]["embargo_bars"] == 0
        assert report["protocol"]["target_kind"] == CLOSE_RETURN_TARGET
        assert set(report["protocol"]["feature_sets"]) == {"ohlcv", "stationary-v1"}
        paired = report["validation_summary"]["transformer"]["paired_deltas"]
        assert paired["stationary-minus-short"]["return_mae"]["count"] == 2
        assert set(report["selection"]) == set(sweep.models)
        for record in report["test"]:
            assert record["candidate"] == \
                report["selection"][record["model"]]["candidate"]
            assert all(math.isfinite(value) for value in record["metrics"].values())
            assert 0.0 <= record["metrics"]["direction_accuracy"] <= 1.0
        for fold in range(sweep.folds):
            targets = {
                json.dumps(record["targets"], sort_keys=True)
                for record in report["validation"] if record["fold"] == fold
            }
            assert len(targets) == 1
        validation_end = max(
            record["targets"]["validation"][1] for record in report["validation"]
        )
        test_start = min(record["targets"]["test"][0] for record in report["test"])
        assert validation_end < test_start
        write_report(output, report)
        write_report(repeated_output, repeated)
        write_predictions(predictions_output, predictions)
        write_predictions(repeated_predictions_output, repeated_predictions)
        write_predictions(calibration_predictions_output, calibration_predictions)
        write_predictions(repeated_calibration_output,
                          repeated_calibration_predictions)
        assert output.read_bytes() == repeated_output.read_bytes()
        assert predictions_output.read_bytes() == \
            repeated_predictions_output.read_bytes()
        assert calibration_predictions_output.read_bytes() == \
            repeated_calibration_output.read_bytes()
        assert len(calibration_predictions) == sum(
            record["samples"] for record in report["calibration"]
        )
        assert all(record["split"] == "calibration" and
                   record["fold"] is None for record in calibration_predictions)
        assert all(
            item["epochs"] is not None
            for item in report["model_fingerprints"]
            if item["model"] in ("transformer", "mlp")
        )
        assert all(
            item["seed"] is None and item["epochs"] is None
            for item in report["model_fingerprints"]
            if item["model"] in ("linear", "rolling_mean", "last_close")
        )
        assert json.loads(output.read_text(encoding="utf-8")) == report
        assert "NaN" not in output.read_text(encoding="utf-8")
        integrity_sweep = Sweep(
            (candidate("integrity", 3),), ("last_close",), (3,), 1, 0.2, 1, 1, 8,
        )
        original = csv.read_bytes()

        def mutate_source(snapshot: Path) -> tuple[tuple[str, ...], object]:
            timestamps, rows = read_bars(snapshot)
            assert snapshot.read_bytes() == original
            csv.write_bytes(original + b"\n")
            return timestamps, rows

        with patch("tools.experiment.read_bars", side_effect=mutate_source):
            try:
                run_experiment(integrity_sweep, (("SYNTH", csv),),
                               torch.device("cpu"), 2,
                               test_authorizer=authorize("last_close"),
                               requested_models=frozenset(("last_close",)))
            except ValueError as error:
                assert "input changed" in str(error)
            else:
                raise AssertionError("mid-run CSV replacement was accepted")
        csv.write_bytes(original)
    print("experiment tests passed")


if __name__ == "__main__":
    main()
