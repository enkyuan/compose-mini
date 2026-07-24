#!/usr/bin/env python3
"""Validate and analyze one frozen three-stock panel calibration."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
import argparse
import json
import math
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.backtest import Bars, Forecast, load_frozen_bars
from tools.data_v1 import EXECUTABLE_RETURN_TARGET
from tools.files import (
    FrozenInput, freeze_inputs, require_disjoint, series_arg, verify_frozen,
    write_json_exclusive,
)
from tools.panel_contract import (
    HEX, NAME, PanelAttempt, PanelInputs, TorchIdentity,
    _directory_identity, _open_directory,
    read_canonical_json, read_canonical_json_lines,
    regular_file_identities, resolve_fresh_output,
)

LOCAL_MODELS = (
    "transformer", "linear", "mlp", "rolling_mean", "last_close",
)
MODELS = (*LOCAL_MODELS, "panel_transformer")
SEEDED_MODELS = frozenset(("transformer", "mlp", "panel_transformer"))
SEEDS = (7, 19, 31, 43, 61)
SERIES = ("AAPL", "MSFT", "SPY")
METRICS = (
    "return_mse", "return_mae", "direction_accuracy", "close_mae",
    "zero_return_baseline_mae",
)
REPORT_FIELDS = {
    "schema", "protocol", "runtime", "series", "test_contract", "sweep",
    "selection", "validation", "calibration", "model_fingerprints",
    "validation_summary", "test", "summary", "sweep_input",
    "calibration_prediction_ledger",
}
RECORD_FIELDS = {
    "model", "candidate", "series", "feature_set", "fold", "seed", "targets",
    "samples", "validation_scaled_mse", "metrics",
}
TARGET_OFFSET = 29
GAP = 12
EXPECTED_RUNS = 162
EXPECTED_PANEL_FITS = 15
METRIC_REL_TOL = 1e-6
METRIC_ABS_TOL = 1e-8


def _expected_sweep(models: Sequence[str]) -> dict[str, object]:
    return {
        "alignment_horizon_bars": 13,
        "batch_size": 128,
        "candidates": [{
            "feature_set": "ohlcv", "ff_dim": 32, "heads": 2, "layers": 1,
            "learning_rate": 0.0003, "mlp_dim": 32, "model_dim": 16,
            "name": "raw-17", "ridge": 0.001, "rolling_window": 8,
            "seq_len": 17, "weight_decay": 0.0001,
        }],
        "epochs": 100,
        "fold_fraction": 0.1,
        "folds": 2,
        "models": list(models),
        "patience": 10,
        "seeds": list(SEEDS),
        "target_horizon_bars": 13,
        "target_kind": EXECUTABLE_RETURN_TARGET,
    }


def _expected_protocol(run_count: int) -> dict[str, object]:
    return {
        "split": "embargoed expanding walk-forward by target time",
        "selection": "minimum mean validation scaled-return MSE",
        "selection_aggregation": "macro mean over series, folds, and seeds",
        "holdout_aggregation": "macro mean over series and seeds",
        "phase": "selection-and-calibration",
        "calibration_policy": "deferred until policy selection",
        "aligned_history_bars": 17,
        "target_horizon_bars": 13,
        "target_kind": EXECUTABLE_RETURN_TARGET,
        "target_formula": "log(close[t + horizon] / open[t + 1])",
        "zero_return_reference": "open[t + 1]",
        "alignment_horizon_bars": 13,
        "embargo_bars": 12,
        "feature_contract": "experiment-only; artifact V1 remains OHLCV",
        "feature_sets": {
            "ohlcv": ["open", "high", "low", "close", "volume"],
        },
        "folds": 2,
        "fold_fraction": 0.1,
        "run_count": run_count,
        "diagnostic_caps": {
            "linear_flat_features": 2_048,
            "mlp_parameters": 8_388_608,
        },
    }


def _finite(value: object, label: str, minimum: float = 0.0,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result) or result < minimum or \
       maximum is not None and result > maximum:
        raise ValueError(f"{label} is outside its valid range")
    return result


def _integer(value: object, label: str, minimum: int = 0,
             maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or \
       maximum is not None and value > maximum:
        raise ValueError(f"{label} is invalid")
    return value


def _exact_json(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _exact_json(value[key], item) for key, item in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _exact_json(item, other)
            for item, other in zip(value, expected, strict=True)
        )
    return value == expected


def _binding(frozen: FrozenInput) -> dict[str, str]:
    return {"path": str(frozen.source), "sha256": frozen.sha256}


def _empty_summary(models: Sequence[str]) -> dict[str, object]:
    empty = {"count": 0, "mean": None, "stddev": None}
    return {
        model: {
            "by_series": {},
            "return_macro_by_seed": {},
            "return_macro_across_seeds": {
                metric: dict(empty)
                for metric in (
                    "return_mse", "return_mae", "direction_accuracy",
                )
            },
        }
        for model in models
    }


def _stats(values: Sequence[float]) -> dict[str, object]:
    return {
        "count": len(values), "mean": fmean(values), "stddev": pstdev(values),
    }


def _validation_summary(
    records: Sequence[Mapping[str, object]], models: Sequence[str],
) -> dict[str, object]:
    return {
        model: {
            "candidates": {
                "raw-17": {
                    "feature_set": "ohlcv",
                    "metrics": {
                        metric: _stats([
                            float(
                                record["validation_scaled_mse"]
                                if metric == "validation_scaled_mse" else
                                record["metrics"][metric]
                            )
                            for record in records
                            if record["model"] == model
                        ])
                        for metric in ("validation_scaled_mse", *METRICS)
                    },
                },
            },
            "paired_deltas": {},
        }
        for model in models
    }


@dataclass(frozen=True)
class Grid:
    validation: tuple[tuple[dict[str, list[str]], int], ...]
    calibration: dict[str, list[str]]
    calibration_targets: tuple[str, ...]
    calibration_as_of: tuple[str, ...]
    test_contract: dict[str, object]


def _boundary(
    timestamps: Sequence[str], counts: Sequence[int],
) -> dict[str, list[str]]:
    offset, starts = TARGET_OFFSET, []
    for count in counts:
        starts.append(offset)
        offset += count + GAP
    return {
        name: [timestamps[start], timestamps[start + count - 1]]
        for name, start, count in zip(
            ("train", "validation", "test"), starts, counts, strict=False,
        )
    }


def _grid(timestamps: Sequence[str]) -> Grid:
    samples = len(timestamps) - TARGET_OFFSET
    block = int(samples * 0.1)
    initial = samples - 4 * block
    folds = tuple(
        (initial + fold * block - GAP, block - GAP) for fold in range(2)
    )
    holdout = (samples - 2 * block - GAP, block - GAP, block)
    if min(samples, block, initial, *holdout, *(item for split in folds
                                                for item in split)) <= 0:
        raise ValueError("series is too short for the exact panel protocol")
    validation = tuple(
        (_boundary(timestamps, split), split[1]) for split in folds
    )
    calibration = _boundary(timestamps, holdout)
    start = TARGET_OFFSET + holdout[0] + GAP
    targets = tuple(timestamps[start:start + holdout[1]])
    as_of = tuple(timestamps[index - 13]
                  for index in range(start, start + holdout[1]))
    test = calibration["test"]
    return Grid(
        validation, calibration, targets, as_of,
        {
            "samples": holdout[2],
            "first_target_time": test[0],
            "last_target_time": test[1],
        },
    )


def _observe_torch(attempt: PanelAttempt) -> TorchIdentity:
    script = (
        "from dataclasses import asdict\n"
        "import json\n"
        "from tools.experiment import _torch_identity\n"
        "print(json.dumps(asdict(_torch_identity()),"
        "allow_nan=False,sort_keys=True))\n"
    )
    try:
        result = subprocess.run(
            (*attempt.torch_argv, "-c", script),
            cwd=attempt.source_tree.root, check=True, capture_output=True,
            text=True, timeout=300,
        )
        value = json.loads(result.stdout)
    except (
        OSError, subprocess.SubprocessError, json.JSONDecodeError,
    ) as error:
        raise ValueError("cannot observe the bound Torch runtime") from error
    return TorchIdentity.parse(value)


def _validate_stage(
    attempt: PanelAttempt, stage: str, argv: Sequence[str],
) -> None:
    if Path(sys.executable).resolve(strict=True) != \
            Path(attempt.primary_python.path).resolve(strict=True):
        raise ValueError("analyzer is not running under bound primary Python")
    attempt.validate_stage(
        stage, argv, os.environ, attempt.torch_probe,
    )
    attempt.validate_stage(
        stage, argv, os.environ, _observe_torch(attempt),
    )


def _expected_commands(
    attempt_path: Path, attempt: PanelAttempt, inputs: PanelInputs,
) -> dict[str, tuple[str, ...]]:
    common = (
        str(attempt_path), attempt.input_manifest.path, attempt.config.path,
        attempt.baseline_report.path, attempt.baseline_ledger.path,
    )
    series = tuple(f"{item.name}={item.csv.path}" for item in inputs.series)
    analyzer = "tools/analyze_panel.py"
    return {
        "validate_attempt": (
            analyzer, "validate-attempt", *common, *series,
        ),
        "preflight": (analyzer, "preflight", *common, *series),
        "experiment": (
            "tools/experiment.py", attempt.config.path,
            attempt.outputs["experiment_report"], *series,
            "--attempt-manifest", str(attempt_path),
            "--input-manifest", attempt.input_manifest.path,
            "--baseline-report", attempt.baseline_report.path,
            "--baseline-ledger", attempt.baseline_ledger.path,
            "--device", "cpu", "--calibration-only",
            "--calibration-predictions",
            attempt.outputs["calibration_ledger"],
            "--max-runs", str(EXPECTED_RUNS),
        ),
        "analyze": (
            analyzer, "analyze", *common,
            attempt.outputs["experiment_report"],
            attempt.outputs["calibration_ledger"],
            attempt.outputs["analysis_report"], *series,
        ),
        "finalizer_prefix": (
            "tools/finalize_panel_attempt.py", str(attempt_path),
            attempt.outputs["outcome"],
        ),
    }


def _validate_attempt(
    attempt_path: Path, attempt: PanelAttempt, inputs: PanelInputs,
) -> None:
    if Path(attempt.source_tree.root).resolve(strict=True) != ROOT or \
       attempt.expected_equivalent_runs != EXPECTED_RUNS or \
       attempt.expected_panel_fits != EXPECTED_PANEL_FITS or \
       tuple(item.name for item in inputs.series) != SERIES or \
       dict(attempt.commands) != _expected_commands(
           attempt_path, attempt, inputs,
       ):
        raise ValueError("attempt does not match the exact panel protocol")


@dataclass(frozen=True)
class BoundPanel:
    stage: str
    argv: tuple[str, ...]
    attempt: PanelAttempt
    inputs: PanelInputs
    attempt_input: FrozenInput
    input_manifest_input: FrozenInput
    config_input: FrozenInput
    baseline_report_input: FrozenInput
    baseline_ledger_input: FrozenInput
    series: tuple[tuple[str, FrozenInput], ...]
    bars: Mapping[str, Bars]
    frozen: tuple[FrozenInput, ...]
    paths: tuple[Path, ...]
    identities: tuple[tuple[int, int], ...]
    run_directory_fd: int | None
    run_directory_identity: tuple[int, int] | None

    def verify_inputs(self) -> None:
        verify_frozen(self.frozen)
        if regular_file_identities(self.paths) != self.identities:
            raise ValueError("panel input identity changed during the command")
        if self.run_directory_identity is not None and \
           _directory_identity(Path(self.attempt.run_dir)) != \
                self.run_directory_identity:
            raise ValueError("panel run directory changed during the command")
        _validate_stage(self.attempt, self.stage, self.argv)

    def verify(self) -> None:
        self.verify_inputs()
        self.attempt.validate_paths(self.stage)


def _output_paths(
    stage: str, attempt: PanelAttempt, inputs: Sequence[Path],
) -> None:
    names = (
        tuple(attempt.outputs)
        if stage in ("validate_attempt", "preflight")
        else ("analysis_report", "outcome")
    )
    declared = tuple(Path(attempt.outputs[name]) for name in names)
    outputs = (
        declared if stage == "validate_attempt" else
        tuple(resolve_fresh_output(path) for path in declared)
    )
    require_disjoint(inputs, outputs)


@contextmanager
def _freeze_panel(
    stage: str, argv: Sequence[str], attempt_path: Path, inputs_path: Path,
    config_path: Path, baseline_report_path: Path,
    baseline_ledger_path: Path, series: Sequence[tuple[str, Path]],
    extra: Sequence[Path] = (),
) -> Iterator[BoundPanel]:
    direct = (
        attempt_path, inputs_path, config_path, baseline_report_path,
        baseline_ledger_path, *extra, *(path for _, path in series),
    )
    regular_file_identities(direct)
    with freeze_inputs((attempt_path,)) as discovery:
        discovered = PanelAttempt.read(discovery[0].snapshot)
        sources = discovered.source_paths()
        executables = tuple(map(Path, (
            discovered.primary_python.path, discovered.uv.path,
            discovered.torch_probe.python.path,
        )))
        paths = tuple(dict.fromkeys((*direct, *sources, *executables)))
        identities = regular_file_identities(paths)
        with freeze_inputs(paths) as frozen:
            verify_frozen(discovery)
            if regular_file_identities(paths) != identities:
                raise ValueError("panel input identity changed during discovery")
            by_source = dict(zip(paths, frozen, strict=True))
            attempt = PanelAttempt.read(by_source[attempt_path].snapshot)
            if attempt != discovered:
                raise ValueError("attempt changed during source discovery")
            inputs = PanelInputs.read(by_source[inputs_path].snapshot)
            attempt.input_manifest.validate(
                by_source[inputs_path], "input manifest",
            )
            attempt.config.validate(by_source[config_path], "sweep config")
            attempt.baseline_report.validate(
                by_source[baseline_report_path], "baseline report",
            )
            attempt.baseline_ledger.validate(
                by_source[baseline_ledger_path], "baseline ledger",
            )
            for binding, label in (
                (attempt.primary_python, "primary Python"),
                (attempt.uv, "uv"),
                (attempt.torch_probe.python, "Torch Python"),
            ):
                binding.validate(by_source[Path(binding.path)], label)
            source_inputs = {
                item.path: by_source[Path(tree.root) / item.path]
                for tree in (attempt.source_tree, attempt.finalizer_tree)
                for item in tree.files
            }
            attempt.source_tree.validate(
                {
                    item.path: source_inputs[item.path]
                    for item in attempt.source_tree.files
                },
                "implementation",
            )
            attempt.finalizer_tree.validate(
                {
                    item.path: source_inputs[item.path]
                    for item in attempt.finalizer_tree.files
                },
                "finalizer",
            )
            frozen_series = tuple(
                (name, by_source[path]) for name, path in series
            )
            inputs.validate_direct(
                frozen_series, by_source[baseline_report_path],
                by_source[baseline_ledger_path],
            )
            bars = {
                name: load_frozen_bars(item) for name, item in frozen_series
            }
            inputs.validate_timestamps(tuple(
                (name, item, bars[name].timestamps)
                for name, item in frozen_series
            ))
            grids = tuple(item.timestamps for item in bars.values())
            if any(grid != grids[0] for grid in grids[1:]):
                raise ValueError("panel series must share one timestamp grid")
            config = read_canonical_json(by_source[config_path].snapshot)
            if not _exact_json(config, _expected_sweep(MODELS)):
                raise ValueError("config does not match the exact panel sweep")
            _validate_attempt(attempt_path, attempt, inputs)
            _validate_stage(attempt, stage, argv)
            attempt.validate_paths(stage)
            _output_paths(stage, attempt, paths)
            run_fd, run_identity = (
                _open_directory(Path(attempt.run_dir))
                if stage == "analyze" else (None, None)
            )
            try:
                bound = BoundPanel(
                    stage, tuple(argv), attempt, inputs,
                    by_source[attempt_path], by_source[inputs_path],
                    by_source[config_path], by_source[baseline_report_path],
                    by_source[baseline_ledger_path], frozen_series, bars,
                    frozen, paths, identities, run_fd, run_identity,
                )
                yield bound
                (
                    bound.verify_inputs()
                    if stage == "analyze" else bound.verify()
                )
            finally:
                if run_fd is not None:
                    os.close(run_fd)


def _validation_keys(
    names: Sequence[str], panel: bool,
) -> list[tuple[str, str, int, int | None]]:
    keys = [
        (model, name, fold, seed)
        for name in names for fold in range(2) for model in LOCAL_MODELS
        for seed in (SEEDS if model in SEEDED_MODELS else (None,))
    ]
    if panel:
        keys.extend(
            ("panel_transformer", name, fold, seed)
            for name in names for fold in range(2) for seed in SEEDS
        )
    return keys


def _calibration_keys(
    names: Sequence[str], panel: bool,
) -> list[tuple[str, str, int | None]]:
    keys = [
        (model, name, seed)
        for model in LOCAL_MODELS for name in names
        for seed in (SEEDS if model in SEEDED_MODELS else (None,))
    ]
    if panel:
        keys.extend(
            ("panel_transformer", name, seed)
            for name in names for seed in SEEDS
        )
    return keys


def _record(
    value: object, key: tuple[str, str, int | None, int | None],
    targets: Mapping[str, list[str]], samples: int, calibration: bool,
) -> Mapping[str, object]:
    model, series, seed, fold = key
    if not isinstance(value, dict):
        raise ValueError("experiment record is invalid")
    fields = RECORD_FIELDS | ({"epochs"} if calibration else set())
    if not calibration and model in SEEDED_MODELS:
        fields |= {
            "best_validation_scaled_mse", "best_epoch", "epochs_trained",
        }
    identity = {
        "model": model, "series": series, "seed": seed, "fold": fold,
        "candidate": "raw-17", "feature_set": "ohlcv",
    }
    if set(value) != fields or any(
        not _exact_json(value[name], expected)
        for name, expected in identity.items()
    ) or not _exact_json(value["targets"], targets) or \
       _integer(value["samples"], "record samples", 1) != samples:
        raise ValueError("experiment record contract is invalid")
    if type(value["validation_scaled_mse"]) is not float:
        raise ValueError("validation scaled MSE must be a JSON float")
    _finite(value["validation_scaled_mse"], "validation scaled MSE")
    metrics = value["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
        raise ValueError("experiment metrics are invalid")
    for metric in METRICS:
        if type(metrics[metric]) is not float:
            raise ValueError(f"{metric} must be a JSON float")
        _finite(
            metrics[metric], metric, maximum=1.0
            if metric == "direction_accuracy" else None,
        )
    if calibration:
        epochs = value["epochs"]
        if model in SEEDED_MODELS:
            _integer(epochs, "selected epochs", 1, 100)
        elif epochs is not None:
            raise ValueError("deterministic calibration epochs must be null")
    elif model in SEEDED_MODELS:
        best = _integer(value["best_epoch"], "best epoch", 1, 100)
        trained = _integer(value["epochs_trained"], "epochs trained", best, 100)
        _finite(
            value["best_validation_scaled_mse"],
            "best validation scaled MSE",
        )
        if type(value["best_validation_scaled_mse"]) is not float:
            raise ValueError(
                "best validation scaled MSE must be a JSON float"
            )
        if best > trained:
            raise ValueError("best epoch exceeds trained epochs")
    return value


def _runtime(
    value: object, attempt: PanelAttempt | None,
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "device", "python", "torch",
    } or value["device"] != "cpu" or \
       any(not isinstance(value[name], str) or not value[name]
           for name in ("python", "torch")):
        raise ValueError("experiment runtime is invalid")
    if attempt is not None and value != {
        "device": "cpu",
        "python": attempt.torch_probe.python.version.split()[0],
        "torch": attempt.torch_probe.version,
    }:
        raise ValueError("experiment runtime differs from the attempt")


def _provenance(
    value: object, frozen: FrozenInput, label: str,
) -> None:
    if not _exact_json(value, _binding(frozen)):
        raise ValueError(f"{label} provenance is invalid")


def _read_ledger_v3(
    frozen: FrozenInput,
) -> tuple[tuple[Forecast, ...], tuple[str, ...]]:
    values = read_canonical_json_lines(frozen.snapshot)
    forecasts = tuple(Forecast.parse(value) for value in values)
    lines = tuple(
        frozen.snapshot.read_text(encoding="utf-8").splitlines(keepends=True)
    )
    for value in values:
        schema = value.get("schema")
        if type(schema) is not int or schema != 3:
            raise ValueError("calibration ledger rows must use schema 3")
    return forecasts, lines


def _validate_report(
    value: Mapping[str, object], models: Sequence[str],
    bound: BoundPanel, ledger_input: FrozenInput,
    forecasts: Sequence[Forecast], *, panel: bool,
) -> tuple[Mapping[str, object], ...]:
    fields = REPORT_FIELDS | (
        {"attempt_manifest", "input_manifest"} if panel else set()
    )
    run_count = EXPECTED_RUNS if panel else 117
    if set(value) != fields or \
       _integer(value["schema"], "report schema", 6, 6) != 6 or \
       not _exact_json(
           value["protocol"], _expected_protocol(run_count),
       ) or not _exact_json(value["sweep"], _expected_sweep(models)):
        raise ValueError("experiment report protocol is invalid")
    _runtime(value["runtime"], bound.attempt if panel else None)
    if panel:
        if not _exact_json(value["attempt_manifest"], {
            **_binding(bound.attempt_input), "run_id": bound.attempt.run_id,
        }) or not _exact_json(
            value["input_manifest"], _binding(bound.input_manifest_input),
        ):
            raise ValueError("panel report provenance is invalid")
        _provenance(value["sweep_input"], bound.config_input, "config")
    else:
        sweep_input = value["sweep_input"]
        if not isinstance(sweep_input, dict) or set(sweep_input) != {
            "path", "sha256",
        } or not isinstance(sweep_input["path"], str) or \
           not isinstance(sweep_input["sha256"], str) or \
           not HEX.fullmatch(sweep_input["sha256"]):
            raise ValueError("baseline config provenance is invalid")
    if not _exact_json(value["test"], []) or not _exact_json(
        value["summary"], _empty_summary(models),
    ):
        raise ValueError("calibration report contains test results")

    names = tuple(bound.bars)
    grids = {name: _grid(bound.bars[name].timestamps) for name in names}
    expected_series = [
        {
            "name": name, "csv": bound.bars[name].path,
            "rows": len(bound.bars[name].timestamps),
            "sha256": bound.bars[name].sha256,
            "first_timestamp": bound.bars[name].timestamps[0],
            "last_timestamp": bound.bars[name].timestamps[-1],
        }
        for name in names
    ]
    if not _exact_json(value["series"], expected_series) or not _exact_json(
        value["test_contract"], [
            {"series": name, **grids[name].test_contract} for name in names
        ],
    ):
        raise ValueError("experiment series or test grid is invalid")

    validation = value["validation"]
    calibration = value["calibration"]
    if not isinstance(validation, list) or not isinstance(calibration, list):
        raise ValueError("experiment records must be arrays")
    validation_keys = _validation_keys(names, panel)
    if len(validation) != len(validation_keys):
        raise ValueError("validation grid is incomplete")
    checked_validation = []
    for item, (model, name, fold, seed) in zip(
        validation, validation_keys, strict=True,
    ):
        targets, samples = grids[name].validation[fold]
        checked_validation.append(_record(
            item, (model, name, seed, fold), targets, samples, False,
        ))

    calibration_keys = _calibration_keys(names, panel)
    if len(calibration) != len(calibration_keys):
        raise ValueError("calibration grid is incomplete")
    checked_calibration = [
        _record(
            item, (model, name, seed, None), grids[name].calibration,
            len(grids[name].calibration_targets), True,
        )
        for item, (model, name, seed) in zip(
            calibration, calibration_keys, strict=True,
        )
    ]

    selection = value["selection"]
    if not isinstance(selection, dict) or set(selection) != set(models):
        raise ValueError("experiment selection is invalid")
    for model in models:
        expected = {
            "candidate": "raw-17",
            "mean_validation_scaled_mse": fmean(
                float(record["validation_scaled_mse"])
                for record in checked_validation
                if record["model"] == model
            ),
        }
        if not _exact_json(selection[model], expected):
            raise ValueError("experiment selection is invalid")
    if not _exact_json(
        value["validation_summary"],
        _validation_summary(checked_validation, models),
    ):
        raise ValueError("experiment validation summary is invalid")

    fingerprints = value["model_fingerprints"]
    expected_fingerprints = sorted(
        calibration_keys,
        key=lambda item: (
            item[0], item[1], -1 if item[2] is None else item[2],
        ),
    )
    if not isinstance(fingerprints, list) or \
       len(fingerprints) != len(expected_fingerprints):
        raise ValueError("model fingerprints are incomplete")
    epochs = {
        (record["model"], record["series"], record["seed"]): record["epochs"]
        for record in checked_calibration
    }
    for item, key in zip(fingerprints, expected_fingerprints, strict=True):
        if not isinstance(item, dict) or set(item) != {
            "model", "series", "seed", "epochs", "sha256",
        } or not _exact_json(
            [item["model"], item["series"], item["seed"]], list(key),
        ) or not _exact_json(item["epochs"], epochs[key]) or \
           not isinstance(item["sha256"], str) or \
           not HEX.fullmatch(item["sha256"]):
            raise ValueError("model fingerprints are invalid")

    metadata = value["calibration_prediction_ledger"]
    if not _exact_json(metadata, {
        "schema": 3, "path": str(ledger_input.source),
        "records": len(forecasts), "sha256": ledger_input.sha256,
    }):
        raise ValueError("calibration ledger provenance is invalid")
    if panel:
        for fold in range(2):
            for seed in SEEDS:
                copies = [
                    record for record in checked_validation
                    if record["model"] == "panel_transformer" and
                    record["fold"] == fold and record["seed"] == seed
                ]
                shared = {
                    (
                        record["best_validation_scaled_mse"],
                        record["best_epoch"], record["epochs_trained"],
                    )
                    for record in copies
                }
                if len(copies) != len(names) or len(shared) != 1:
                    raise ValueError("panel validation fit is not shared")
        for seed in SEEDS:
            selected = {
                record["epochs"] for record in checked_calibration
                if record["model"] == "panel_transformer" and
                record["seed"] == seed
            }
            if len(selected) != 1:
                raise ValueError("panel calibration epochs are not shared")
    return tuple(checked_calibration)


def _calibration_record_metrics(
    evidence: Sequence[tuple[float, float, float, float]],
) -> dict[str, float]:
    return {
        "return_mse": fmean(
            (prediction - actual) ** 2
            for prediction, actual, _, _ in evidence
        ),
        "return_mae": fmean(
            abs(prediction - actual)
            for prediction, actual, _, _ in evidence
        ),
        "direction_accuracy": fmean(
            _sign(prediction) == _sign(actual)
            for prediction, actual, _, _ in evidence
        ),
        "close_mae": fmean(
            abs(reference * math.exp(prediction) - outcome)
            for prediction, _, reference, outcome in evidence
        ),
        "zero_return_baseline_mae": fmean(
            abs(reference - outcome)
            for _, _, reference, outcome in evidence
        ),
    }


def _validate_ledger(
    forecasts: Sequence[Forecast], calibration: Sequence[Mapping[str, object]],
    bound: BoundPanel,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, dict[str, float]]],
]:
    grids = {name: _grid(bound.bars[name].timestamps) for name in bound.bars}
    predictions: dict[
        tuple[str, str, str], list[tuple[int | None, float]]
    ] = defaultdict(list)
    actuals = {name: {} for name in bound.bars}
    models = tuple(dict.fromkeys(str(record["model"])
                                 for record in calibration))
    offset = 0
    for record in calibration:
        name, model, seed = (
            str(record["series"]), str(record["model"]), record["seed"],
        )
        grid = grids[name]
        rows = forecasts[offset:offset + len(grid.calibration_targets)]
        offset += len(rows)
        if len(rows) != len(grid.calibration_targets):
            raise ValueError("calibration ledger is incomplete")
        indexes = {
            timestamp: index
            for index, timestamp in enumerate(bound.bars[name].timestamps)
        }
        evidence = []
        for row, as_of, target in zip(
            rows, grid.calibration_as_of, grid.calibration_targets,
            strict=True,
        ):
            if (
                row.model, row.series, row.seed, row.as_of, row.target_time
            ) != (model, name, seed, as_of, target) or \
               row.split != "calibration" or row.fold is not None or \
               row.candidate != "raw-17" or row.feature_set != "ohlcv" or \
               row.horizon_bars != 13 or \
               row.target_kind != EXECUTABLE_RETURN_TARGET or \
               row.csv_sha256 != bound.bars[name].sha256:
                raise ValueError("calibration ledger grid is invalid")
            target_index = indexes[target]
            reference = bound.bars[name].opens[target_index - 12]
            outcome = bound.bars[name].closes[target_index]
            actual = math.log(
                outcome / reference
            )
            previous = actuals[name].setdefault(target, actual)
            if previous != actual:
                raise ValueError("calibration actual target changed")
            evidence.append((
                row.predicted_log_return, actual, reference, outcome,
            ))
            predictions[model, name, target].append(
                (seed, row.predicted_log_return),
            )
        expected_metrics = _calibration_record_metrics(evidence)
        metrics = record["metrics"]
        if any(
            not math.isclose(
                float(metrics[metric]), expected,
                rel_tol=METRIC_REL_TOL, abs_tol=METRIC_ABS_TOL,
            )
            for metric, expected in expected_metrics.items()
        ):
            raise ValueError(
                "calibration record metrics differ from the ledger"
            )
    if offset != len(forecasts):
        raise ValueError("calibration ledger contains extra records")

    ensembles = {
        model: {name: {} for name in bound.bars} for model in models
    }
    for model in models:
        seeds = SEEDS if model in SEEDED_MODELS else (None,)
        for name in bound.bars:
            for target in grids[name].calibration_targets:
                rows = predictions[model, name, target]
                if tuple(seed for seed, _ in rows) != seeds:
                    raise ValueError("calibration ensemble seed grid is invalid")
                ensembles[model][name][target] = fmean(
                    prediction for _, prediction in rows
                )
    return actuals, ensembles


def _validation_metrics(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    macro = {
        model: fmean(
            float(record["metrics"]["return_mae"])
            for record in records if record["model"] == model
        )
        for model in MODELS
    }
    local = {
        (record["series"], record["fold"], record["seed"]):
            float(record["metrics"]["return_mae"])
        for record in records if record["model"] == "transformer"
    }
    deltas = {
        name: [] for name in SERIES
    }
    ordered = []
    for record in records:
        if record["model"] != "panel_transformer":
            continue
        key = record["series"], record["fold"], record["seed"]
        delta = float(record["metrics"]["return_mae"]) - local[key]
        deltas[str(record["series"])].append(delta)
        ordered.append(delta)
    if len(ordered) != 30:
        raise ValueError("panel validation pairs are incomplete")
    return {
        "macro_return_mae": macro,
        "paired_panel_minus_local_transformer": {
            "mean_delta": fmean(ordered),
            "wins": sum(value < 0.0 for value in ordered),
            "ties": sum(value == 0.0 for value in ordered),
            "losses": sum(value > 0.0 for value in ordered),
            "per_stock_mean_delta": {
                name: fmean(values) for name, values in deltas.items()
            },
        },
    }


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _calibration_metrics(
    actuals: Mapping[str, Mapping[str, float]],
    predictions: Mapping[str, Mapping[str, Mapping[str, float]]],
    bars: Mapping[str, Bars],
) -> dict[str, object]:
    per_stock: dict[str, object] = {}
    macro = {model: [] for model in MODELS}
    macro_direction = {model: [] for model in MODELS}
    close_improvements = []
    for name in SERIES:
        targets = tuple(actuals[name])
        values = [actuals[name][target] for target in targets]
        if not targets:
            raise ValueError("stock has no calibration targets")
        proportions = {
            "p_up": fmean(value > 0.0 for value in values),
            "p_down": fmean(value < 0.0 for value in values),
            "p_flat": fmean(value == 0.0 for value in values),
        }
        majority = max(proportions.values())
        indexes = {
            timestamp: index
            for index, timestamp in enumerate(bars[name].timestamps)
        }
        models = {}
        for model in MODELS:
            predicted = [predictions[model][name][target] for target in targets]
            return_mae = fmean(
                abs(prediction - actual)
                for prediction, actual in zip(predicted, values, strict=True)
            )
            direction = fmean(
                _sign(prediction) == _sign(actual)
                for prediction, actual in zip(predicted, values, strict=True)
            )
            close_mae = fmean(
                abs(
                    bars[name].opens[indexes[target] - 12] *
                    math.exp(prediction) -
                    bars[name].closes[indexes[target]]
                )
                for target, prediction in zip(
                    targets, predicted, strict=True,
                )
            )
            models[model] = {
                "return_mae": return_mae,
                "direction_accuracy": direction,
                "close_mae": close_mae,
            }
            macro[model].append(return_mae)
            macro_direction[model].append(direction)
        zero_return = fmean(abs(value) for value in values)
        zero_close = fmean(
            abs(
                bars[name].opens[indexes[target] - 12] -
                bars[name].closes[indexes[target]]
            )
            for target in targets
        )
        if zero_close <= 0.0:
            raise ValueError("zero-return close MAE denominator is invalid")
        relative = (
            zero_close - float(models["panel_transformer"]["close_mae"])
        ) / zero_close
        close_improvements.append(relative)
        per_stock[name] = {
            "samples": len(targets),
            "models": models,
            "majority_direction": {**proportions, "reference": majority},
            "zero_return_return_mae": zero_return,
            "zero_return_close_mae": zero_close,
            "panel_close_relative_improvement": relative,
        }
    return {
        "macro_return_mae": {
            model: fmean(values) for model, values in macro.items()
        },
        "macro_direction_accuracy": {
            model: fmean(values)
            for model, values in macro_direction.items()
        },
        "macro_majority_direction": fmean(
            stock["majority_direction"]["reference"]
            for stock in per_stock.values()
        ),
        "mean_panel_close_relative_improvement": fmean(close_improvements),
        "per_stock": per_stock,
    }


def _comparators(values: Mapping[str, float]) -> dict[str, float]:
    return {
        "local_transformer": values["transformer"],
        "mlp": values["mlp"],
        "linear": values["linear"],
        "rolling_mean": values["rolling_mean"],
        "zero_return": values["last_close"],
    }


def _gates(
    validation: Mapping[str, object], calibration: Mapping[str, object],
) -> dict[str, object]:
    validation_macro = validation["macro_return_mae"]
    validation_panel = validation_macro["panel_transformer"]
    validation_comparators = _comparators(validation_macro)
    validation_margin = min(validation_comparators.values()) - validation_panel
    paired = validation["paired_panel_minus_local_transformer"]
    per_stock_delta = paired["per_stock_mean_delta"]
    paired_pass = paired["mean_delta"] < 0.0 and paired["wins"] >= 20 and \
        all(value <= 0.0 for value in per_stock_delta.values())

    calibration_macro = calibration["macro_return_mae"]
    calibration_panel = calibration_macro["panel_transformer"]
    calibration_comparators = _comparators(calibration_macro)
    calibration_margin = min(calibration_comparators.values()) - \
        calibration_panel
    per_stock = calibration["per_stock"]
    zero = {
        name: {
            "panel": stock["models"]["panel_transformer"]["return_mae"],
            "zero_return": stock["zero_return_return_mae"],
            "margin": (
                stock["zero_return_return_mae"] -
                stock["models"]["panel_transformer"]["return_mae"]
            ),
        }
        for name, stock in per_stock.items()
    }
    for value in zero.values():
        value["pass"] = value["margin"] > 0.0

    direction = calibration["macro_direction_accuracy"]["panel_transformer"]
    majority = calibration["macro_majority_direction"]
    stock_direction = {
        name: {
            "panel": stock["models"]["panel_transformer"][
                "direction_accuracy"
            ],
            "minimum": 0.5,
            "margin": stock["models"]["panel_transformer"][
                "direction_accuracy"
            ] - 0.5,
        }
        for name, stock in per_stock.items()
    }
    for value in stock_direction.values():
        value["pass"] = value["margin"] >= 0.0
    close = calibration["mean_panel_close_relative_improvement"]

    gates = {
        "validation_macro_mae": {
            "pass": validation_margin > 0.0,
            "panel": validation_panel,
            "comparators": validation_comparators,
            "margin": validation_margin,
        },
        "validation_paired": {
            "pass": paired_pass,
            "mean_delta": paired["mean_delta"],
            "wins": paired["wins"],
            "required_wins": 20,
            "per_stock_mean_delta": per_stock_delta,
        },
        "calibration_macro_mae": {
            "pass": calibration_margin > 0.0,
            "panel": calibration_panel,
            "comparators": calibration_comparators,
            "margin": calibration_margin,
        },
        "calibration_per_stock_zero": {
            "pass": all(value["pass"] for value in zero.values()),
            "per_stock": zero,
        },
        "calibration_direction": {
            "pass": direction > majority and
                    all(value["pass"] for value in stock_direction.values()),
            "panel_macro": direction,
            "majority_macro": majority,
            "macro_margin": direction - majority,
            "per_stock": stock_direction,
        },
        "calibration_close_mae": {
            "pass": close > 0.0,
            "mean_relative_improvement": close,
            "margin": close,
        },
    }
    gates["all_pass"] = all(
        value["pass"] for value in gates.values()
        if isinstance(value, dict)
    )
    return gates


def _analysis(
    bound: BoundPanel, report_input: FrozenInput, ledger_input: FrozenInput,
) -> tuple[dict[str, object], bool]:
    baseline_forecasts, baseline_lines = _read_ledger_v3(
        bound.baseline_ledger_input,
    )
    baseline = read_canonical_json(bound.baseline_report_input.snapshot)
    baseline_calibration = _validate_report(
        baseline, LOCAL_MODELS, bound, bound.baseline_ledger_input,
        baseline_forecasts, panel=False,
    )
    _validate_ledger(baseline_forecasts, baseline_calibration, bound)

    forecasts, lines = _read_ledger_v3(ledger_input)
    report = read_canonical_json(report_input.snapshot)
    calibration = _validate_report(
        report, MODELS, bound, ledger_input, forecasts, panel=True,
    )
    actuals, predictions = _validate_ledger(
        forecasts, calibration, bound,
    )
    local = set(LOCAL_MODELS)
    if [item for item in report["validation"] if item["model"] in local] != \
            baseline["validation"] or \
       [item for item in report["calibration"] if item["model"] in local] != \
            baseline["calibration"] or \
       [item for item in report["model_fingerprints"]
        if item["model"] in local] != baseline["model_fingerprints"] or \
       tuple(
           line for item, line in zip(forecasts, lines, strict=True)
           if item.model in local
       ) != baseline_lines:
        raise ValueError("live local-model evidence differs from the baseline")

    validation = _validation_metrics(report["validation"])
    calibration_metrics = _calibration_metrics(
        actuals, predictions, bound.bars,
    )
    gates = _gates(validation, calibration_metrics)
    passed = bool(gates["all_pass"])
    return {
        "schema": 1,
        "status": "pass" if passed else "gate-failure",
        "inputs": {
            "run_id": bound.attempt.run_id,
            "attempt": _binding(bound.attempt_input),
            "input_manifest": _binding(bound.input_manifest_input),
            "config": _binding(bound.config_input),
            "baseline_report": _binding(bound.baseline_report_input),
            "baseline_ledger": _binding(bound.baseline_ledger_input),
            "experiment_report": _binding(report_input),
            "calibration_ledger": _binding(ledger_input),
            "series": [
                {"name": name, **_binding(item)}
                for name, item in bound.series
            ],
        },
        "protocol": {
            "candidate": "raw-17",
            "models": list(MODELS),
            "seeds": list(SEEDS),
            "series": list(SERIES),
            "folds": 2,
            "fold_fraction": 0.1,
            "target_horizon_bars": 13,
            "target_kind": EXECUTABLE_RETURN_TARGET,
            "series_equivalent_runs": EXPECTED_RUNS,
            "physical_panel_fits": EXPECTED_PANEL_FITS,
            "validation_pair": "stock/fold/seed",
            "calibration_ensemble":
                "arithmetic mean by model/stock/target before metrics",
            "macro_unit": "stock",
            "majority_reference": "unique actual calibration targets",
        },
        "validation": validation,
        "calibration": calibration_metrics,
        "gates": gates,
    }, passed


def _series(value: str) -> tuple[str, Path]:
    return series_arg(value, NAME)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("validate-attempt", "preflight"):
        command = modes.add_parser(mode)
        command.add_argument("attempt", type=Path)
        command.add_argument("inputs", type=Path)
        command.add_argument("config", type=Path)
        command.add_argument("baseline_report", type=Path)
        command.add_argument("baseline_ledger", type=Path)
        command.add_argument("series", nargs="+", type=_series)
    analyze = modes.add_parser("analyze")
    analyze.add_argument("attempt", type=Path)
    analyze.add_argument("inputs", type=Path)
    analyze.add_argument("config", type=Path)
    analyze.add_argument("baseline_report", type=Path)
    analyze.add_argument("baseline_ledger", type=Path)
    analyze.add_argument("report", type=Path)
    analyze.add_argument("ledger", type=Path)
    analyze.add_argument("output", type=Path)
    analyze.add_argument("series", nargs="+", type=_series)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    stage = args.mode.replace("-", "_")
    try:
        extra = (
            (args.report, args.ledger) if args.mode == "analyze" else ()
        )
        with _freeze_panel(
            stage, sys.argv, args.attempt, args.inputs, args.config,
            args.baseline_report, args.baseline_ledger, args.series, extra,
        ) as bound:
            if args.mode == "analyze":
                by_source = {
                    item.source: item for item in bound.frozen
                }
                report, passed = _analysis(
                    bound, by_source[args.report], by_source[args.ledger],
                )
                output = resolve_fresh_output(args.output)
                if output.parent != Path(bound.attempt.run_dir).resolve():
                    raise ValueError("analysis output left the run directory")
                write_json_exclusive(
                    output, report, bound.run_directory_fd, bound.verify,
                )
        if args.mode != "analyze":
            print(json.dumps(
                {"mode": args.mode, "status": "valid"}, sort_keys=True,
            ))
            return
    except (
        IndexError, KeyError, OSError, OverflowError, TypeError,
        UnicodeError, ValueError,
    ) as error:
        print(f"integrity error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(
        {"output": str(args.output), "status": report["status"]},
        sort_keys=True,
    ))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
