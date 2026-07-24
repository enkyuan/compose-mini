#!/usr/bin/env python3
"""Verify the frozen panel benchmark and its standard-library CLI boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ANALYZER = ROOT / "tools/analyze_panel.py"
FINALIZER = ROOT / "tools/finalize_panel_attempt.py"
CONFIG = ROOT / "experiments/executable-h13-panel.example.json"
INPUTS = ROOT / "experiments/executable-h13-panel-inputs.json"

from tools import (
    analyze_panel as panel_analysis,
    analyze_universe,
    backtest,
    finalize_panel_attempt as panel_finalizer,
    replay_calibration,
    select_policy,
)
from tools.backtest import load_bars
from tools.files import file_sha256, write_json
from tools.panel_contract import (
    BOOTSTRAP_BLOCKS, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
    COMPARISON_PROFILE, FINALIZER_SOURCE_PATHS, LEGACY_PROFILE, SOURCE_PATHS,
    SERIES, TARGET_KIND, PanelAttempt, PanelInputs, PanelProfile,
    expected_panel_commands, expected_panel_sweep, panel_analysis_protocol,
    panel_gates, panel_profile, selected_source_tree,
    validate_panel_analysis,
    mkdir_nofollow, read_canonical_json, read_canonical_json_lines,
    regular_file_identities,
)

LOCAL_MODELS = (
    "transformer", "linear", "mlp", "rolling_mean", "last_close",
)
LEGACY_MODELS = (*LOCAL_MODELS, "panel_transformer")
COMPARISON_MODELS = (
    *LOCAL_MODELS, "panel_transformer", "conditioned_panel_transformer",
)
MODELS = LEGACY_MODELS
SEEDS = (7, 19, 31, 43, 61)
TIMESTAMP_SHA256 = \
    "fdd3c0e647c5312bac7eca3d2837a83d7d223a4e5931251b6af8c310178588e8"
TRANSITIONS = {
    "preflight-failure": ("preflight", 1),
    "setup-failure": ("setup", 1),
    "experiment-failure": ("experiment", 1),
    "analysis-integrity-failure": ("analysis", 2),
    "gate-failure": ("analysis", 3),
    "pass": ("analysis", 0),
}


def expected_config(
    profile: PanelProfile = LEGACY_PROFILE,
) -> dict[str, object]:
    return expected_panel_sweep(profile)


def expected_inputs() -> dict[str, object]:
    common = {
        "first_timestamp": "2024-07-22T13:30:00Z",
        "last_timestamp": "2026-07-21T19:30:00Z",
        "rows": 6488,
        "timestamp_sha256": TIMESTAMP_SHA256,
    }
    return {
        "baseline_ledger": {
            "path": "reports/executable-h13-calibration.jsonl",
            "sha256":
                "8e8f1c9e53e1acaec71cc0abcf73fc402c735973037abd6f5b56bff4afeae2c5",
        },
        "baseline_report": {
            "path": "reports/executable-h13-calibration.json",
            "sha256":
                "0689539de9e7bc0400403b6f7dfe44ead880c11a8fb265d20def48f3e80cfd81",
        },
        "schema": 1,
        "series": [
            {
                "csv": {
                    "path": "data/aapl-30m.csv",
                    "sha256":
                        "a821339ae61f1a7169e2e95f5e221a2"
                        "c9fdbe8d931aa15b9384c0760d394984c",
                },
                "name": "AAPL",
                **common,
            },
            {
                "csv": {
                    "path": "data/msft-30m.csv",
                    "sha256":
                        "715b8e27a73417271054985d8a5366a6"
                        "03b42068afe70bbe809fed83c5f59709",
                },
                "name": "MSFT",
                **common,
            },
            {
                "csv": {
                    "path": "data/spy-30m.csv",
                    "sha256":
                        "486f199189e53134e1385497606b869ee"
                        "77b4a4632efc94574d8016f116e562b",
                },
                "name": "SPY",
                **common,
            },
        ],
    }


def run(
    command: list[object], expected: int = 0,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(item) for item in command], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env=environment,
    )
    assert result.returncode == expected, (
        f"{command} returned {result.returncode}, expected {expected}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def rejects(call: Callable[[], object]) -> None:
    try:
        call()
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("invalid panel contract was accepted")


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def timestamps(rows: int = 170) -> tuple[str, ...]:
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    return tuple(
        (start + timedelta(minutes=index)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for index in range(rows)
    )


def write_canonical_json(path: Path, value: object) -> None:
    write_json(path, value)


def write_panel_csv(
    path: Path, stock: int, timestamp_grid: tuple[str, ...],
) -> None:
    grid = panel_analysis._grid(timestamp_grid)
    targets = tuple(timestamp_grid.index(value)
                    for value in grid.calibration_targets)
    open_values = {
        target - 12: 80.0 + 10.0 * stock + index
        for index, target in enumerate(targets)
    }
    actuals = tuple(
        (0.02 + 0.005 * stock) * (1 if index % 2 == 0 else -1)
        for index in range(len(targets))
    )
    close_values = {
        target: open_values[target - 12] * math.exp(actual)
        for target, actual in zip(targets, actuals, strict=True)
    }
    lines = ["timestamp,open,high,low,close,volume"]
    for index, timestamp in enumerate(timestamp_grid):
        open_value = open_values.get(index, 100.0)
        close_value = close_values.get(index, 100.0)
        high = max(open_value, close_value) + 1.0
        low = min(open_value, close_value) - 1.0
        lines.append(
            f"{timestamp},{open_value:.17g},{high:.17g},{low:.17g},"
            f"{close_value:.17g},{1000 + index}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def report_metrics(return_mae: float) -> dict[str, float]:
    return {
        "return_mse": return_mae ** 2,
        "return_mae": return_mae,
        "direction_accuracy": 0.5,
        "close_mae": return_mae * 100.0,
        "zero_return_baseline_mae": 2.0,
    }


def report_record(
    model: str, name: str, seed: int | None, fold: int | None,
    targets: dict[str, list[str]], samples: int, calibration: bool,
    metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    return_mae = {
        "transformer": 0.02,
        "linear": 0.04,
        "mlp": 0.03,
        "rolling_mean": 0.05,
        "last_close": 0.025,
        "panel_transformer": 0.01,
        "conditioned_panel_transformer": 0.005,
    }[model]
    record: dict[str, object] = {
        "model": model,
        "candidate": "raw-17",
        "series": name,
        "feature_set": "ohlcv",
        "fold": fold,
        "seed": seed,
        "targets": targets,
        "samples": samples,
        "validation_scaled_mse": 1.0,
        "metrics": metrics or report_metrics(return_mae),
    }
    if calibration:
        record["epochs"] = 5 if seed is not None else None
    elif seed is not None:
        record.update({
            "best_validation_scaled_mse": 1.0,
            "best_epoch": 5,
            "epochs_trained": 6,
        })
    return record


def tree_value(root: Path, paths: tuple[str, ...],
               hashes: dict[str, str] | None = None) -> dict[str, object]:
    files = [
        {
            "path": path,
            "sha256": (
                hashes[path] if hashes is not None else file_sha256(root / path)
            ),
        }
        for path in sorted(paths)
    ]
    digest = hashlib.sha256()
    for item in files:
        digest.update(
            item["path"].encode() + b"\0" +
            item["sha256"].encode() + b"\n"
        )
    return {"root": str(root.resolve()), "files": files,
            "sha256": digest.hexdigest()}


class PanelFixture:
    def __init__(
        self, root: Path, profile: PanelProfile = LEGACY_PROFILE,
    ) -> None:
        self.root = root
        self.profile = profile
        self.models = profile.models
        self.panel_models = profile.panel_models
        self.candidate = profile.candidate
        self.reference = profile.reference
        self.expected_runs = profile.expected_runs
        self.expected_panel_fits = profile.expected_panel_fits
        self.analysis_schema = profile.analysis_schema
        self.relative = root.relative_to(ROOT)
        self.run_dir = self.relative / "run"
        (ROOT / self.run_dir).mkdir()
        self.config_path = self.relative / "config.json"
        self.inputs_path = self.relative / "inputs.json"
        self.baseline_report_path = self.relative / "baseline.json"
        self.baseline_ledger_path = self.relative / "baseline.jsonl"
        self.attempt_path = self.relative / "attempt.json"
        self.report_path = self.run_dir / "experiment.json"
        self.ledger_path = self.run_dir / "calibration.jsonl"
        self.analysis_path = self.run_dir / "analysis.json"
        self.outcome_path = self.relative / "outcome.json"
        self.names = ("AAPL", "MSFT", "SPY")
        self.csv_paths = {
            name: self.relative / f"{name.lower()}.csv"
            for name in self.names
        }
        self.timestamp_grid = timestamps(
            600 if profile == COMPARISON_PROFILE else 170
        )
        for stock, name in enumerate(self.names):
            write_panel_csv(
                ROOT / self.csv_paths[name], stock, self.timestamp_grid,
            )
        self.bars = {
            name: load_bars(ROOT / path)
            for name, path in self.csv_paths.items()
        }
        self.grids = {
            name: panel_analysis._grid(self.bars[name].timestamps)
            for name in self.names
        }
        self.actuals = {
            name: {
                target: math.log(
                    self.bars[name].closes[
                        self.bars[name].timestamps.index(target)
                    ] /
                    self.bars[name].opens[
                        self.bars[name].timestamps.index(target) - 12
                    ]
                )
                for target in self.grids[name].calibration_targets
            }
            for name in self.names
        }

        write_canonical_json(
            ROOT / self.config_path, expected_config(profile),
        )
        baseline_records = self._ledger((), "perfect")
        self._write_ledger(self.baseline_ledger_path, baseline_records)
        baseline = self._report(
            (), self.baseline_ledger_path, baseline_records,
        )
        write_canonical_json(ROOT / self.baseline_report_path, baseline)
        self._write_inputs()
        self.environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": f"{self.run_dir}/.pycache",
        }
        self._write_attempt()
        self.write_live(
            "comparison" if profile == COMPARISON_PROFILE else "perfect"
        )

    def _write_inputs(self) -> None:
        declarations = []
        for name in self.names:
            path = self.csv_paths[name]
            bars = self.bars[name]
            digest = hashlib.sha256(
                "".join(
                    f"{timestamp}\n" for timestamp in bars.timestamps
                ).encode("ascii")
            ).hexdigest()
            declarations.append({
                "name": name,
                "csv": {
                    "path": path.as_posix(),
                    "sha256": file_sha256(ROOT / path),
                },
                "rows": len(bars.timestamps),
                "first_timestamp": bars.timestamps[0],
                "last_timestamp": bars.timestamps[-1],
                "timestamp_sha256": digest,
            })
        write_canonical_json(ROOT / self.inputs_path, {
            "schema": 1,
            "series": declarations,
            "baseline_report": {
                "path": self.baseline_report_path.as_posix(),
                "sha256": file_sha256(ROOT / self.baseline_report_path),
            },
            "baseline_ledger": {
                "path": self.baseline_ledger_path.as_posix(),
                "sha256": file_sha256(ROOT / self.baseline_ledger_path),
            },
        })

    def _validation(
        self, panel_models: tuple[str, ...],
    ) -> list[dict[str, object]]:
        records = []
        for model, name, fold, seed in panel_analysis._validation_keys(
            self.names, panel_models,
        ):
            targets, samples = self.grids[name].validation[fold]
            records.append(report_record(
                model, name, seed, fold, targets, samples, False,
            ))
        return records

    def _calibration(
        self, panel_models: tuple[str, ...],
        ledger: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        records = []
        for model, name, seed in panel_analysis._calibration_keys(
            self.names, panel_models,
        ):
            grid = self.grids[name]
            rows = [
                item for item in ledger
                if (
                    item["model"], item["series"], item["seed"]
                ) == (model, name, seed)
            ]
            indexes = {
                timestamp: index
                for index, timestamp in enumerate(
                    self.bars[name].timestamps
                )
            }
            evidence = [
                (
                    float(item["predicted_log_return"]),
                    self.actuals[name][str(item["target_time"])],
                    self.bars[name].opens[
                        indexes[str(item["target_time"])] - 12
                    ],
                    self.bars[name].closes[
                        indexes[str(item["target_time"])]
                    ],
                )
                for item in rows
            ]
            records.append(report_record(
                model, name, seed, None, grid.calibration,
                len(grid.calibration_targets), True,
                panel_analysis._calibration_record_metrics(evidence),
            ))
        return records

    def _ledger(
        self, panel_models: tuple[str, ...], panel_predictions: str,
    ) -> list[dict[str, object]]:
        records = []
        offsets = {7: -0.04, 19: -0.02, 31: 0.0, 43: 0.02, 61: 0.04}
        local_offsets = {
            7: -0.004, 19: -0.002, 31: 0.0, 43: 0.002, 61: 0.004,
        }
        for model, name, seed in panel_analysis._calibration_keys(
            self.names, panel_models,
        ):
            grid = self.grids[name]
            for as_of, target in zip(
                grid.calibration_as_of, grid.calibration_targets, strict=True,
            ):
                actual = self.actuals[name][target]
                if model == "transformer":
                    prediction = actual + 0.02 + local_offsets[seed]
                elif model == "mlp":
                    prediction = actual + 0.03 + local_offsets[seed]
                elif model == "linear":
                    prediction = actual + 0.04
                elif model == "rolling_mean":
                    prediction = actual + 0.05
                elif model == "last_close":
                    prediction = 0.0
                elif model in self.panel_models and \
                        panel_predictions == "exact":
                    prediction = actual
                elif model == "panel_transformer" and \
                        panel_predictions == "comparison":
                    prediction = actual + 0.03 + local_offsets[seed]
                elif model == "conditioned_panel_transformer" and \
                        panel_predictions == "comparison":
                    prediction = actual + local_offsets[seed]
                elif panel_predictions == "perfect":
                    prediction = actual + offsets[seed]
                else:
                    prediction = 0.0
                records.append({
                    "schema": 3,
                    "split": "calibration",
                    "fold": None,
                    "series": name,
                    "model": model,
                    "candidate": "raw-17",
                    "feature_set": "ohlcv",
                    "seed": seed,
                    "csv_sha256": file_sha256(
                        ROOT / self.csv_paths[name]
                    ),
                    "as_of": as_of,
                    "target_time": target,
                    "horizon_bars": 13,
                    "target_kind": "executable-return-v1",
                    "predicted_log_return": prediction,
                })
        return records

    def _write_ledger(
        self, path: Path, records: list[dict[str, object]],
    ) -> None:
        (ROOT / path).write_text(
            "".join(
                json.dumps(
                    record, allow_nan=False, sort_keys=True,
                ) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def _report(
        self, panel_models: tuple[str, ...], ledger_path: Path,
        ledger_records: list[dict[str, object]],
    ) -> dict[str, object]:
        panel = bool(panel_models)
        models = self.models if panel else LOCAL_MODELS
        validation = self._validation(panel_models)
        calibration = self._calibration(panel_models, ledger_records)
        calibration_keys = panel_analysis._calibration_keys(
            self.names, panel_models,
        )
        fingerprint_keys = (
            [
                *sorted(
                    (
                        key for key in calibration_keys
                        if key[0] not in panel_models
                    ),
                    key=lambda item: (
                        item[0], item[1],
                        -1 if item[2] is None else item[2],
                    ),
                ),
                *(
                    (model, name, seed)
                    for model in panel_models
                    for name in self.names for seed in SEEDS
                ),
            ]
            if self.profile == COMPARISON_PROFILE and panel else
            sorted(
                calibration_keys,
                key=lambda item: (
                    item[0], item[1],
                    -1 if item[2] is None else item[2],
                ),
            )
        )
        fingerprints = [
            {
                "model": model,
                "series": name,
                "seed": seed,
                "epochs": 5 if seed is not None else None,
                "sha256": digest_text(f"{model}:{name}:{seed}"),
            }
            for model, name, seed in fingerprint_keys
        ]
        report = {
            "schema": 6,
            "protocol": panel_analysis._expected_protocol(
                self.expected_runs if panel else 117,
                self.profile if panel else None,
            ),
            "runtime": {
                "device": "cpu",
                "python": platform.python_version(),
                "torch": "synthetic-torch",
            },
            "series": [
                {
                    "name": name,
                    "csv": self.csv_paths[name].as_posix(),
                    "rows": len(self.bars[name].timestamps),
                    "sha256": file_sha256(ROOT / self.csv_paths[name]),
                    "first_timestamp": self.bars[name].timestamps[0],
                    "last_timestamp": self.bars[name].timestamps[-1],
                }
                for name in self.names
            ],
            "test_contract": [
                {"series": name, **self.grids[name].test_contract}
                for name in self.names
            ],
            "sweep": (
                expected_panel_sweep(self.profile)
                if panel else panel_analysis._baseline_sweep()
            ),
            "selection": {
                model: {
                    "candidate": "raw-17",
                    "mean_validation_scaled_mse": 1.0,
                }
                for model in models
            },
            "validation": validation,
            "calibration": calibration,
            "model_fingerprints": fingerprints,
            "validation_summary":
                panel_analysis._validation_summary(validation, models),
            "test": [],
            "summary": panel_analysis._empty_summary(models),
            "sweep_input": (
                {
                    "path": self.config_path.as_posix(),
                    "sha256": file_sha256(ROOT / self.config_path),
                }
                if panel else
                {"path": "baseline-config.json", "sha256": "a" * 64}
            ),
            "calibration_prediction_ledger": {
                "schema": 3,
                "path": ledger_path.as_posix(),
                "records": len(ledger_records),
                "sha256": file_sha256(ROOT / ledger_path),
            },
        }
        if panel:
            report.update({
                "attempt_manifest": {
                    "path": self.attempt_path.as_posix(),
                    "sha256": file_sha256(ROOT / self.attempt_path),
                    "run_id": "synthetic-panel",
                },
                "input_manifest": {
                    "path": self.inputs_path.as_posix(),
                    "sha256": file_sha256(ROOT / self.inputs_path),
                },
            })
        return report

    def _write_attempt(self) -> None:
        python = Path(sys.executable).resolve()
        python_version = subprocess.run(
            [str(python), "--version"], check=True, capture_output=True,
            text=True,
        )
        executable = {
            "path": str(python),
            "sha256": file_sha256(python),
            "version":
                (python_version.stdout or python_version.stderr).strip(),
        }
        package = self.root / "synthetic-torch"
        package.mkdir()
        module = package / "module.py"
        module.write_text("synthetic = True\n", encoding="ascii")
        self.torch_module = module
        torch_probe = {
            "python": {
                "path": str(python),
                "sha256": file_sha256(python),
                "version": platform.python_version(),
            },
            "version": "synthetic-torch",
            "git_version": None,
            "cuda_version": None,
            "config": "synthetic-config",
            "package_tree": tree_value(package, ("module.py",)),
        }
        probe = self.root / "torch-probe.json"
        write_canonical_json(probe, torch_probe)
        uv = self.root / "uv"
        uv.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  printf '%s\\n' 'synthetic uv'\n"
            "else\n"
            f"  exec /bin/cat '{probe}'\n"
            "fi\n",
            encoding="ascii",
        )
        uv.chmod(0o700)
        uv_binding = {
            "path": str(uv.resolve()),
            "sha256": file_sha256(uv),
            "version": "synthetic uv",
        }
        outputs = {
            "experiment_report": self.report_path.as_posix(),
            "calibration_ledger": self.ledger_path.as_posix(),
            "analysis_report": self.analysis_path.as_posix(),
            "outcome": self.outcome_path.as_posix(),
        }
        commands = expected_panel_commands(
            self.attempt_path,
            self.inputs_path.as_posix(),
            self.config_path.as_posix(),
            self.baseline_report_path.as_posix(),
            self.baseline_ledger_path.as_posix(),
            outputs,
            PanelInputs.read(ROOT / self.inputs_path),
            self.profile,
        )
        write_canonical_json(ROOT / self.attempt_path, {
            "schema": 1,
            "run_id": "synthetic-panel",
            "status": "armed",
            "run_dir": self.run_dir.as_posix(),
            "implementation_commit": "0" * 40,
            "input_manifest": {
                "path": self.inputs_path.as_posix(),
                "sha256": file_sha256(ROOT / self.inputs_path),
            },
            "config": {
                "path": self.config_path.as_posix(),
                "sha256": file_sha256(ROOT / self.config_path),
            },
            "baseline_report": {
                "path": self.baseline_report_path.as_posix(),
                "sha256": file_sha256(ROOT / self.baseline_report_path),
            },
            "baseline_ledger": {
                "path": self.baseline_ledger_path.as_posix(),
                "sha256": file_sha256(ROOT / self.baseline_ledger_path),
            },
            "source_tree": tree_value(ROOT, SOURCE_PATHS),
            "finalizer_tree": tree_value(ROOT, FINALIZER_SOURCE_PATHS),
            "primary_python": executable,
            "uv": uv_binding,
            "torch_argv": [
                str(uv.resolve()), "run", "--offline", "--with", "torch",
                "python",
            ],
            "torch_probe": torch_probe,
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": f"{self.run_dir}/.pycache",
            },
            "commands": commands,
            "expected_equivalent_runs": self.expected_runs,
            "expected_panel_fits": self.expected_panel_fits,
            "outputs": outputs,
        })

    def write_live(
        self, panel_predictions: str, *, mutate_local: bool = False,
        forbidden_test: bool = False,
    ) -> None:
        records = self._ledger(self.panel_models, panel_predictions)
        self._write_ledger(self.ledger_path, records)
        report = self._report(
            self.panel_models, self.ledger_path, records,
        )
        if mutate_local:
            metrics = report["validation"][0]["metrics"]
            metrics["return_mae"] += 0.001
            report["validation_summary"] = panel_analysis._validation_summary(
                report["validation"], self.models,
            )
        if forbidden_test:
            report["test"] = [{"forbidden": True}]
        write_canonical_json(ROOT / self.report_path, report)

    def analyze(
        self, expected: int, series: tuple[str, ...] | None = None,
    ) -> dict[str, object] | None:
        ordered = self.names if series is None else series
        command = [
            sys.executable, "tools/analyze_panel.py", "analyze",
            self.attempt_path, self.inputs_path, self.config_path,
            self.baseline_report_path, self.baseline_ledger_path,
            self.report_path, self.ledger_path, self.analysis_path,
            *(
                f"{name}={self.csv_paths[name].as_posix()}"
                for name in ordered
            ),
        ]
        result = run(command, expected, self.environment)
        if expected not in (0, 3):
            assert not (ROOT / self.analysis_path).exists()
            return None
        return json.loads(
            (ROOT / self.analysis_path).read_text(encoding="utf-8")
        )

    def validate(self, mode: str, expected: int) -> None:
        command = [
            sys.executable, "tools/analyze_panel.py", mode,
            self.attempt_path, self.inputs_path, self.config_path,
            self.baseline_report_path, self.baseline_ledger_path,
            *(
                f"{name}={self.csv_paths[name].as_posix()}"
                for name in self.names
            ),
        ]
        run(command, expected, self.environment)

    def finalizer_argv(
        self, status: str,
    ) -> tuple[list[object], str, int]:
        stage, code = TRANSITIONS[status]
        argv: list[object] = [
            "tools/finalize_panel_attempt.py",
            self.attempt_path,
            self.outcome_path,
            "--started", "2026-07-23T12:00:00Z",
            "--ended", "2026-07-23T12:01:00Z",
            "--stage", stage,
            "--exit", code,
            "--status", status,
        ]
        return argv, stage, code

    def finalize(self, status: str, expected: int = 0) -> dict[str, object]:
        argv, _, _ = self.finalizer_argv(status)
        run([sys.executable, *argv], expected, self.environment)
        outcome = ROOT / self.outcome_path
        if expected != 0:
            assert not outcome.exists()
            return {}
        return json.loads(outcome.read_text(encoding="utf-8"))


def expected_legacy_analysis(
    fixture: PanelFixture,
) -> dict[str, object]:
    per_stock = {
        "AAPL": {
            "samples": 2,
            "models": {
                "transformer": {
                    "return_mae": 0.020000000000000004,
                    "direction_accuracy": 0.5,
                    "close_mae": 1.6263310942863498,
                },
                "linear": {
                    "return_mae": 0.04,
                    "direction_accuracy": 0.5,
                    "close_mae": 3.2855162560044633,
                },
                "mlp": {
                    "return_mae": 0.03,
                    "direction_accuracy": 0.5,
                    "close_mae": 2.4517757468071224,
                },
                "rolling_mean": {
                    "return_mae": 0.05,
                    "direction_accuracy": 0.5,
                    "close_mae": 4.127635996624072,
                },
                "last_close": {
                    "return_mae": 0.019999963911814446,
                    "direction_accuracy": 0.0,
                    "close_mae": 1.6100044250488281,
                },
                "panel_transformer": {
                    "return_mae": 1.734723475976807e-18,
                    "direction_accuracy": 1.0,
                    "close_mae": 0.0,
                },
            },
            "majority_direction": {
                "p_up": 0.5, "p_down": 0.5, "p_flat": 0.0,
                "reference": 0.5,
            },
            "zero_return_return_mae": 0.019999963911814446,
            "zero_return_close_mae": 1.6100044250488281,
            "panel_close_relative_improvement": 1.0,
        },
        "MSFT": {
            "samples": 2,
            "models": {
                "transformer": {
                    "return_mae": 0.02,
                    "direction_accuracy": 1.0,
                    "close_mae": 1.828540077901181,
                },
                "linear": {
                    "return_mae": 0.04000000000000001,
                    "direction_accuracy": 0.5,
                    "close_mae": 3.6940191156685813,
                },
                "mlp": {
                    "return_mae": 0.03,
                    "direction_accuracy": 0.5,
                    "close_mae": 2.7566159380542246,
                },
                "rolling_mean": {
                    "return_mae": 0.05,
                    "direction_accuracy": 0.5,
                    "close_mae": 4.640843351843209,
                },
                "last_close": {
                    "return_mae": 0.024999973817077546,
                    "direction_accuracy": 0.0,
                    "close_mae": 2.2625770568847656,
                },
                "panel_transformer": {
                    "return_mae": 1.734723475976807e-18,
                    "direction_accuracy": 1.0,
                    "close_mae": 0.0,
                },
            },
            "majority_direction": {
                "p_up": 0.5, "p_down": 0.5, "p_flat": 0.0,
                "reference": 0.5,
            },
            "zero_return_return_mae": 0.024999973817077546,
            "zero_return_close_mae": 2.2625770568847656,
            "panel_close_relative_improvement": 1.0,
        },
        "SPY": {
            "samples": 2,
            "models": {
                "transformer": {
                    "return_mae": 0.020000000000000004,
                    "direction_accuracy": 1.0,
                    "close_mae": 2.0308453119497116,
                },
                "linear": {
                    "return_mae": 0.039999999999999994,
                    "direction_accuracy": 0.5,
                    "close_mae": 4.102716420587875,
                },
                "mlp": {
                    "return_mae": 0.03,
                    "direction_accuracy": 0.5,
                    "close_mae": 3.0616012316607524,
                },
                "rolling_mean": {
                    "return_mae": 0.05,
                    "direction_accuracy": 0.5,
                    "close_mae": 5.154294991117581,
                },
                "last_close": {
                    "return_mae": 0.030000009754387585,
                    "direction_accuracy": 0.0,
                    "close_mae": 3.015228271484375,
                },
                "panel_transformer": {
                    "return_mae": 3.469446951953614e-18,
                    "direction_accuracy": 1.0,
                    "close_mae": 0.0,
                },
            },
            "majority_direction": {
                "p_up": 0.5, "p_down": 0.5, "p_flat": 0.0,
                "reference": 0.5,
            },
            "zero_return_return_mae": 0.030000009754387585,
            "zero_return_close_mae": 3.015228271484375,
            "panel_close_relative_improvement": 1.0,
        },
    }
    inputs = {
        "run_id": "synthetic-panel",
        "attempt": {
            "path": fixture.attempt_path.as_posix(),
            "sha256": file_sha256(ROOT / fixture.attempt_path),
        },
        "input_manifest": {
            "path": fixture.inputs_path.as_posix(),
            "sha256": file_sha256(ROOT / fixture.inputs_path),
        },
        "config": {
            "path": fixture.config_path.as_posix(),
            "sha256": file_sha256(ROOT / fixture.config_path),
        },
        "baseline_report": {
            "path": fixture.baseline_report_path.as_posix(),
            "sha256": file_sha256(ROOT / fixture.baseline_report_path),
        },
        "baseline_ledger": {
            "path": fixture.baseline_ledger_path.as_posix(),
            "sha256": file_sha256(ROOT / fixture.baseline_ledger_path),
        },
        "experiment_report": {
            "path": fixture.report_path.as_posix(),
            "sha256": file_sha256(ROOT / fixture.report_path),
        },
        "calibration_ledger": {
            "path": fixture.ledger_path.as_posix(),
            "sha256": file_sha256(ROOT / fixture.ledger_path),
        },
        "series": [
            {
                "name": name,
                "path": fixture.csv_paths[name].as_posix(),
                "sha256": file_sha256(ROOT / fixture.csv_paths[name]),
            }
            for name in fixture.names
        ],
    }
    validation_macro = {
        "transformer": 0.02,
        "linear": 0.04,
        "mlp": 0.029999999999999995,
        "rolling_mean": 0.05000000000000001,
        "last_close": 0.025000000000000005,
        "panel_transformer": 0.01,
    }
    calibration_macro = {
        "transformer": 0.020000000000000004,
        "linear": 0.04,
        "mlp": 0.03,
        "rolling_mean": 0.05000000000000001,
        "last_close": 0.024999982494426528,
        "panel_transformer": 2.3129646346357427e-18,
    }
    return {
        "schema": 1,
        "status": "pass",
        "inputs": inputs,
        "protocol": {
            "candidate": "raw-17",
            "models": list(LEGACY_MODELS),
            "seeds": list(SEEDS),
            "series": list(SERIES),
            "folds": 2,
            "fold_fraction": 0.1,
            "target_horizon_bars": 13,
            "target_kind": "executable-return-v1",
            "series_equivalent_runs": 162,
            "physical_panel_fits": 15,
            "validation_pair": "stock/fold/seed",
            "calibration_ensemble":
                "arithmetic mean by model/stock/target before metrics",
            "macro_unit": "stock",
            "majority_reference": "unique actual calibration targets",
        },
        "validation": {
            "macro_return_mae": validation_macro,
            "paired_panel_minus_local_transformer": {
                "mean_delta": -0.01,
                "wins": 30,
                "ties": 0,
                "losses": 0,
                "per_stock_mean_delta": {
                    "AAPL": -0.01, "MSFT": -0.01, "SPY": -0.01,
                },
            },
        },
        "calibration": {
            "macro_return_mae": calibration_macro,
            "macro_direction_accuracy": {
                "transformer": 0.8333333333333334,
                "linear": 0.5,
                "mlp": 0.5,
                "rolling_mean": 0.5,
                "last_close": 0.0,
                "panel_transformer": 1.0,
            },
            "macro_majority_direction": 0.5,
            "mean_panel_close_relative_improvement": 1.0,
            "per_stock": per_stock,
        },
        "gates": {
            "validation_macro_mae": {
                "pass": True,
                "panel": 0.01,
                "comparators": {
                    "local_transformer": 0.02,
                    "mlp": 0.029999999999999995,
                    "linear": 0.04,
                    "rolling_mean": 0.05000000000000001,
                    "zero_return": 0.025000000000000005,
                },
                "margin": 0.01,
            },
            "validation_paired": {
                "pass": True,
                "mean_delta": -0.01,
                "wins": 30,
                "required_wins": 20,
                "per_stock_mean_delta": {
                    "AAPL": -0.01, "MSFT": -0.01, "SPY": -0.01,
                },
            },
            "calibration_macro_mae": {
                "pass": True,
                "panel": 2.3129646346357427e-18,
                "comparators": {
                    "local_transformer": 0.020000000000000004,
                    "mlp": 0.03,
                    "linear": 0.04,
                    "rolling_mean": 0.05000000000000001,
                    "zero_return": 0.024999982494426528,
                },
                "margin": 0.02,
            },
            "calibration_per_stock_zero": {
                "pass": True,
                "per_stock": {
                    "AAPL": {
                        "panel": 1.734723475976807e-18,
                        "zero_return": 0.019999963911814446,
                        "margin": 0.019999963911814446,
                        "pass": True,
                    },
                    "MSFT": {
                        "panel": 1.734723475976807e-18,
                        "zero_return": 0.024999973817077546,
                        "margin": 0.024999973817077546,
                        "pass": True,
                    },
                    "SPY": {
                        "panel": 3.469446951953614e-18,
                        "zero_return": 0.030000009754387585,
                        "margin": 0.03000000975438758,
                        "pass": True,
                    },
                },
            },
            "calibration_direction": {
                "pass": True,
                "panel_macro": 1.0,
                "majority_macro": 0.5,
                "macro_margin": 0.5,
                "per_stock": {
                    name: {
                        "panel": 1.0, "minimum": 0.5,
                        "margin": 0.5, "pass": True,
                    }
                    for name in SERIES
                },
            },
            "calibration_close_mae": {
                "pass": True,
                "mean_relative_improvement": 1.0,
                "margin": 1.0,
            },
            "all_pass": True,
        },
    }


def finalizer_fixture(root: Path) -> tuple[Path, Path]:
    relative = root.relative_to(ROOT)
    run_dir = relative / "run"
    package = root / "torch"
    package.mkdir()
    module = package / "module.py"
    module.write_text("fixture = True\n", encoding="ascii")
    python = Path(sys.executable).resolve()
    version = subprocess.run(
        [str(python), "--version"], check=True, capture_output=True, text=True,
    )
    executable = {
        "path": str(python),
        "sha256": file_sha256(python),
        "version": (version.stdout or version.stderr).strip(),
    }
    source_hashes = {path: "0" * 64 for path in SOURCE_PATHS}
    package_tree = tree_value(
        package, ("module.py",), {"module.py": file_sha256(module)},
    )
    attempt = relative / "attempt.json"
    outcome = relative / "outcome.json"
    value = {
        "baseline_ledger": {
            "path": f"{relative}/missing-baseline.jsonl", "sha256": "0" * 64,
        },
        "baseline_report": {
            "path": f"{relative}/missing-baseline.json", "sha256": "0" * 64,
        },
        "commands": {
            "analyze": ["analyze"],
            "experiment": ["experiment"],
            "finalizer_prefix": [
                str(FINALIZER), str(attempt), str(outcome),
            ],
            "preflight": ["preflight"],
            "validate_attempt": ["validate-attempt"],
        },
        "config": {
            "path": f"{relative}/missing-config.json", "sha256": "0" * 64,
        },
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": f"{run_dir}/.pycache",
        },
        "expected_equivalent_runs": 162,
        "expected_panel_fits": 15,
        "finalizer_tree": tree_value(ROOT, FINALIZER_SOURCE_PATHS),
        "implementation_commit": "0" * 40,
        "input_manifest": {
            "path": f"{relative}/missing-inputs.json", "sha256": "0" * 64,
        },
        "outputs": {
            "analysis_report": f"{run_dir}/analysis.json",
            "calibration_ledger": f"{run_dir}/calibration.jsonl",
            "experiment_report": f"{run_dir}/experiment.json",
            "outcome": str(outcome),
        },
        "primary_python": executable,
        "run_dir": str(run_dir),
        "run_id": "fixture",
        "schema": 1,
        "source_tree": tree_value(ROOT, SOURCE_PATHS, source_hashes),
        "status": "armed",
        "torch_argv": [
            str(python), "run", "--offline", "--with", "torch", "python",
        ],
        "torch_probe": {
            "config": "fixture",
            "cuda_version": None,
            "git_version": None,
            "package_tree": package_tree,
            "python": executable,
            "version": "fixture",
        },
        "uv": executable,
    }
    (ROOT / attempt).write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return attempt, outcome


def finalizer_command(
    attempt: Path, outcome: Path, stage: str, code: int, status: str,
) -> list[object]:
    return [
        sys.executable, FINALIZER, attempt, outcome,
        "--started", "2026-07-23T12:00:00Z",
        "--ended", "2026-07-23T12:01:00Z",
        "--stage", stage, "--exit", code, "--status", status,
    ]


def verify_profiles_and_sources() -> None:
    assert LEGACY_PROFILE == PanelProfile(
        LEGACY_MODELS, ("panel_transformer",),
        "panel_transformer", "transformer", 162, 15, 1,
    )
    assert COMPARISON_PROFILE == PanelProfile(
        COMPARISON_MODELS,
        ("panel_transformer", "conditioned_panel_transformer"),
        "conditioned_panel_transformer", "panel_transformer", 207, 30, 2,
    )
    assert TARGET_KIND == panel_analysis.EXECUTABLE_RETURN_TARGET
    for profile in (LEGACY_PROFILE, COMPARISON_PROFILE):
        config = expected_panel_sweep(profile)
        assert panel_profile(config) == profile
        assert panel_analysis_protocol(profile)["models"] == \
            list(profile.models)
        missing = (
            config["models"][:-1]
            if profile == LEGACY_PROFILE else
            [*LOCAL_MODELS, "conditioned_panel_transformer"]
        )
        for invalid in (
            config | {"extra": True},
            config | {"models": missing},
            config | {"models": list(reversed(config["models"]))},
            config | {"seeds": [*config["seeds"][:-1], True]},
            config | {"batch_size": 128.0},
        ):
            rejects(lambda invalid=invalid: panel_profile(invalid))

    assert tuple(
        item.path for item in selected_source_tree(ROOT, SOURCE_PATHS).files
    ) == tuple(sorted(SOURCE_PATHS))
    assert tuple(
        item.path
        for item in selected_source_tree(ROOT, FINALIZER_SOURCE_PATHS).files
    ) == tuple(sorted(FINALIZER_SOURCE_PATHS))
    rejects(lambda: selected_source_tree(
        ROOT, (*SOURCE_PATHS, SOURCE_PATHS[0]),
    ))
    rejects(lambda: selected_source_tree(ROOT, ("missing.py",)))
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-selected-tree-",
    ) as directory:
        root = Path(directory)
        target = root / "target.py"
        target.write_text("value = 1\n", encoding="ascii")
        (root / "linked.py").symlink_to(target)
        rejects(lambda: selected_source_tree(root, ("linked.py",)))
        observed = selected_source_tree(root, ("target.py",))
        expected = hashlib.sha256(
            b"target.py\0" + file_sha256(target).encode("ascii") + b"\n"
        ).hexdigest()
        assert observed.sha256 == expected


def verify_cli_surface() -> None:
    missing = [
        str(path.relative_to(ROOT))
        for path in (ANALYZER, FINALIZER)
        if not path.is_file()
    ]
    assert not missing, f"missing panel tools: {', '.join(missing)}"
    analyzer_help = run([sys.executable, ANALYZER, "--help"]).stdout
    for mode in ("validate-attempt", "preflight", "analyze"):
        assert mode in analyzer_help
    finalizer_help = run([sys.executable, FINALIZER, "--help"]).stdout
    for option in ("--started", "--ended", "--stage", "--exit", "--status"):
        assert option in finalizer_help


def verify_finalizer_transitions() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-finalizer-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        for index, (status, (stage, code)) in enumerate(TRANSITIONS.items()):
            case = root / str(index)
            case.mkdir()
            attempt, outcome = finalizer_fixture(case)
            command = finalizer_command(attempt, outcome, stage, code, status)
            if status in ("gate-failure", "pass"):
                result = run(command, 2)
                assert "terminal" not in result.stderr
                continue
            run(command)
            published = ROOT / outcome
            result = json.loads(published.read_text(encoding="utf-8"))
            assert (result["stage"], result["exit"], result["status"]) == \
                (stage, code, status)
            original = published.read_bytes()
            run(command, 2)
            assert published.read_bytes() == original

        case = root / "invalid"
        case.mkdir()
        attempt, outcome = finalizer_fixture(case)
        for stage, code, status in (
            ("setup", 1, "preflight-failure"),
            ("analysis", 0, "gate-failure"),
            ("analysis", 3, "pass"),
            ("analysis", 256, "analysis-integrity-failure"),
        ):
            result = run(
                finalizer_command(attempt, outcome, stage, code, status), 2,
            )
            assert "terminal" in result.stderr

        target = root / "target.json"
        target.write_text("original", encoding="ascii")
        symlink_case = root / "symlink"
        symlink_case.mkdir()
        attempt, outcome = finalizer_fixture(symlink_case)
        (ROOT / outcome).symlink_to(target)
        run(finalizer_command(
            attempt, outcome, "experiment", 1, "experiment-failure",
        ), 2)
        assert target.read_text(encoding="ascii") == "original"

        copied_case = root / "copied"
        copied_case.mkdir()
        attempt, outcome = finalizer_fixture(copied_case)
        copied = attempt.with_name("copied-attempt.json")
        (ROOT / copied).write_bytes((ROOT / attempt).read_bytes())
        run(finalizer_command(
            copied, outcome, "experiment", 1, "experiment-failure",
        ), 2)
        assert not (ROOT / outcome).exists()


def verify_successful_finalization() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-success-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        for status, analyzer_exit in (("pass", 0), ("gate-failure", 3)):
            case = root / status
            case.mkdir()
            fixture = PanelFixture(case)
            if status == "gate-failure":
                fixture.write_live("zero")
            analysis = fixture.analyze(analyzer_exit)
            assert analysis is not None
            result = fixture.finalize(status)
            assert result["status"] == status
            assert result["attempt"] == {
                "path": fixture.attempt_path.as_posix(),
                "sha256": file_sha256(ROOT / fixture.attempt_path),
                "run_id": "synthetic-panel",
            }
            assert all(result["integrity"]["broader"].values())
            for name, path in (
                ("experiment_report", fixture.report_path),
                ("calibration_ledger", fixture.ledger_path),
                ("analysis_report", fixture.analysis_path),
            ):
                assert result["outputs"][name] == {
                    "path": path.as_posix(),
                    "state": "present",
                    "sha256": file_sha256(ROOT / path),
                }
            assert analysis["inputs"]["experiment_report"] == {
                "path": fixture.report_path.as_posix(),
                "sha256": file_sha256(ROOT / fixture.report_path),
            }
            assert analysis["inputs"]["calibration_ledger"] == {
                "path": fixture.ledger_path.as_posix(),
                "sha256": file_sha256(ROOT / fixture.ledger_path),
            }

        forged = root / "forged"
        forged.mkdir()
        fixture = PanelFixture(forged)
        fixture.write_live("zero")
        fixture.analyze(3)
        analysis = read_canonical_json(ROOT / fixture.analysis_path)
        analysis["status"] = "pass"
        analysis["gates"]["all_pass"] = True
        write_canonical_json(ROOT / fixture.analysis_path, analysis)
        fixture.finalize("pass", 2)


def reject_publication_race(
    fixture: PanelFixture, status: str, mutate: Callable[[], None],
) -> None:
    original = panel_finalizer.write_json_exclusive
    calls = 0

    def intercepted(
        path: Path, value: Mapping[str, object],
        directory_fd: int | None = None,
        before_link: Callable[[], None] | None = None,
    ) -> None:
        nonlocal calls
        calls += 1
        assert callable(before_link)

        def inject() -> None:
            mutate()
            before_link()

        original(path, value, directory_fd, inject)

    argv, stage, code = fixture.finalizer_argv(status)
    with patch.object(
        panel_finalizer, "write_json_exclusive", intercepted,
    ), patch.object(sys, "argv", [str(item) for item in argv]):
        try:
            panel_finalizer.finalize(
                fixture.attempt_path, fixture.outcome_path,
                "2026-07-23T12:00:00Z", "2026-07-23T12:01:00Z",
                stage, code, status,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("publication race crossed the finalizer")
    assert calls == 1
    assert not (ROOT / fixture.outcome_path).exists()


def verify_finalizer_publication_races() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-races-", dir=ROOT,
    ) as directory:
        root = Path(directory)

        torch_case = root / "torch"
        torch_case.mkdir()
        fixture = PanelFixture(torch_case)
        fixture.analyze(0)
        reject_publication_race(
            fixture, "pass",
            lambda: fixture.torch_module.write_text(
                "synthetic = False\n", encoding="ascii",
            ),
        )

        output_case = root / "output"
        output_case.mkdir()
        fixture = PanelFixture(output_case)
        reject_publication_race(
            fixture, "experiment-failure",
            lambda: (ROOT / fixture.analysis_path).write_text(
                "{}\n", encoding="ascii",
            ),
        )


def verify_analysis_publication_race() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-analysis-race-", dir=ROOT,
    ) as directory:
        fixture = PanelFixture(Path(directory))
        attempt = read_canonical_json(ROOT / fixture.attempt_path)
        original = panel_analysis.write_json_exclusive

        def intercepted(
            path: Path, value: Mapping[str, object],
            directory_fd: int | None = None,
            before_link: Callable[[], None] | None = None,
        ) -> None:
            assert directory_fd is not None and callable(before_link)
            run_dir = ROOT / fixture.run_dir
            moved = run_dir.with_name(f"{run_dir.name}-moved")
            run_dir.rename(moved)
            run_dir.symlink_to(moved, target_is_directory=True)
            try:
                original(path, value, directory_fd, before_link)
            finally:
                run_dir.unlink()
                moved.rename(run_dir)

        with patch.object(
            panel_analysis, "write_json_exclusive", intercepted,
        ), patch.object(
            sys, "argv", list(attempt["commands"]["analyze"]),
        ), patch.dict(os.environ, fixture.environment, clear=True):
            try:
                panel_analysis.main()
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError("analysis directory race was accepted")
        assert not (ROOT / fixture.analysis_path).exists()


def verify_rejection_surface() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-contract-",
    ) as directory:
        root = Path(directory)
        duplicate = root / "duplicate.json"
        nonfinite = root / "nonfinite.json"
        duplicate.write_text('{"schema":1,"schema":1}\n', encoding="ascii")
        nonfinite.write_text('{"value":NaN}\n', encoding="ascii")
        real_parent = root / "real-parent"
        linked_parent = root / "linked-parent"
        real_parent.mkdir()
        (real_parent / "input.csv").write_text("x\n", encoding="ascii")
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        try:
            regular_file_identities((linked_parent / "input.csv",))
        except ValueError:
            pass
        else:
            raise AssertionError("symlinked input parent was accepted")
        try:
            mkdir_nofollow(linked_parent / "run")
        except ValueError:
            pass
        else:
            raise AssertionError("symlinked output parent was accepted")
        assert not (real_parent / "run").exists()
        series = ["AAPL=missing-a.csv", "MSFT=missing-b.csv", "SPY=missing-c.csv"]
        for invalid in (duplicate, nonfinite):
            try:
                read_canonical_json(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid canonical JSON was accepted")
            result = subprocess.run(
                [
                    sys.executable, ANALYZER, "validate-attempt",
                    invalid, invalid, invalid, invalid, invalid, *series,
                ],
                cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
            )
            assert result.returncode != 0

        for value in (
            '{"schema":3,"schema":3}\n',
            '{"predicted_log_return":NaN}\n',
            '{ "schema": 3 }\n',
        ):
            invalid = root / "invalid.jsonl"
            invalid.write_text(value, encoding="ascii")
            try:
                read_canonical_json_lines(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid canonical JSONL was accepted")


def verify_gate_boundaries(
    validation: dict[str, object], calibration: dict[str, object],
) -> None:
    assert panel_analysis._gates(validation, calibration)["all_pass"]

    value = deepcopy(validation)
    macro = value["macro_return_mae"]
    macro["panel_transformer"] = macro["transformer"]
    gates = panel_analysis._gates(value, calibration)
    assert not gates["validation_macro_mae"]["pass"]
    assert not gates["all_pass"]

    value = deepcopy(validation)
    value["paired_panel_minus_local_transformer"]["mean_delta"] = 0.0
    assert not panel_analysis._gates(
        value, calibration,
    )["validation_paired"]["pass"]

    value = deepcopy(validation)
    value["paired_panel_minus_local_transformer"]["wins"] = 19
    assert not panel_analysis._gates(
        value, calibration,
    )["validation_paired"]["pass"]

    value = deepcopy(validation)
    per_stock = value[
        "paired_panel_minus_local_transformer"
    ]["per_stock_mean_delta"]
    per_stock["AAPL"] = 0.0
    assert panel_analysis._gates(
        value, calibration,
    )["validation_paired"]["pass"]
    per_stock["AAPL"] = 0.001
    assert not panel_analysis._gates(
        value, calibration,
    )["validation_paired"]["pass"]

    value = deepcopy(calibration)
    macro = value["macro_return_mae"]
    macro["panel_transformer"] = macro["transformer"]
    assert not panel_analysis._gates(
        validation, value,
    )["calibration_macro_mae"]["pass"]

    value = deepcopy(calibration)
    stock = value["per_stock"]["AAPL"]
    stock["models"]["panel_transformer"]["return_mae"] = \
        stock["zero_return_return_mae"]
    gates = panel_analysis._gates(validation, value)
    assert not gates["calibration_per_stock_zero"]["pass"]
    assert not gates["all_pass"]

    value = deepcopy(calibration)
    value["macro_direction_accuracy"]["panel_transformer"] = \
        value["macro_majority_direction"]
    assert not panel_analysis._gates(
        validation, value,
    )["calibration_direction"]["pass"]

    value = deepcopy(calibration)
    value["per_stock"]["AAPL"]["models"]["panel_transformer"][
        "direction_accuracy"
    ] = 0.499
    gates = panel_analysis._gates(validation, value)
    assert not gates["calibration_direction"]["pass"]
    assert not gates["all_pass"]

    value = deepcopy(calibration)
    value["mean_panel_close_relative_improvement"] = 0.0
    assert not panel_analysis._gates(
        validation, value,
    )["calibration_close_mae"]["pass"]


def verify_comparison_gate_boundaries(
    analysis: Mapping[str, object],
) -> None:
    validation = analysis["validation"]
    calibration = analysis["calibration"]
    assert panel_analysis._gates(
        validation, calibration, COMPARISON_PROFILE,
    )["all_pass"]

    value = deepcopy(validation)
    macro = value["macro_return_mae"]
    paired = value["paired_candidate_minus_reference"]
    macro[COMPARISON_PROFILE.candidate] = \
        macro[COMPARISON_PROFILE.reference]
    paired["relative_improvement"] = 0.0
    assert not panel_analysis._gates(
        value, calibration, COMPARISON_PROFILE,
    )["validation_macro_mae"]["pass"]

    for relative, expected in ((0.009999, False), (0.01, True)):
        value = deepcopy(validation)
        value["paired_candidate_minus_reference"][
            "relative_improvement"
        ] = relative
        assert panel_analysis._gates(
            value, calibration, COMPARISON_PROFILE,
        )["validation_macro_mae"]["pass"] is expected

    for wins, ties, expected in ((19, 11, False), (20, 10, True)):
        value = deepcopy(validation)
        paired = value["paired_candidate_minus_reference"]
        paired["wins"], paired["ties"], paired["losses"] = wins, ties, 0
        assert panel_analysis._gates(
            value, calibration, COMPARISON_PROFILE,
        )["validation_paired"]["pass"] is expected

    for axis, key in (("by_stock", "AAPL"), ("by_fold", "0")):
        value = deepcopy(validation)
        value["paired_candidate_minus_reference"][axis][key]["mean"] = 0.0
        assert not panel_analysis._gates(
            value, calibration, COMPARISON_PROFILE,
        )["validation_paired"]["pass"]

    value = deepcopy(validation)
    seed_axis = value["paired_candidate_minus_reference"]["by_seed"]
    for seed in ("7", "19"):
        seed_axis[seed]["mean"] = 0.0
    assert not panel_analysis._gates(
        value, calibration, COMPARISON_PROFILE,
    )["validation_paired"]["pass"]
    seed_axis["19"]["mean"] = -0.001
    assert panel_analysis._gates(
        value, calibration, COMPARISON_PROFILE,
    )["validation_paired"]["pass"]

    for relative, expected in ((0.009999, False), (0.01, True)):
        value = deepcopy(calibration)
        value["relative_improvement_vs_reference"] = relative
        value["bootstrap"]["mae_relative_improvement_lower_025"] = 0.01
        assert panel_analysis._gates(
            validation, value, COMPARISON_PROFILE,
        )["calibration_macro_mae"]["pass"] is expected

    value = deepcopy(calibration)
    value["leave_one_seed_out"]["7"]["relative_improvement"] = 0.0
    assert not panel_analysis._gates(
        validation, value, COMPARISON_PROFILE,
    )["calibration_macro_mae"]["pass"]

    for field in (
        "direction_candidate_minus_reference_lower_025",
        "direction_candidate_minus_majority_lower_025",
    ):
        value = deepcopy(calibration)
        value["bootstrap"][field] = 0.0
        assert not panel_analysis._gates(
            validation, value, COMPARISON_PROFILE,
        )["calibration_direction"]["pass"]

    for reference in ("panel_transformer", "zero_return_return_mae"):
        value = deepcopy(calibration)
        stock = value["per_stock"]["AAPL"]
        if reference == "panel_transformer":
            stock["models"][COMPARISON_PROFILE.candidate]["return_mae"] = \
                stock["models"][reference]["return_mae"]
        else:
            stock["models"][COMPARISON_PROFILE.candidate]["return_mae"] = \
                stock[reference]
        assert not panel_analysis._gates(
            validation, value, COMPARISON_PROFILE,
        )["calibration_per_stock_zero"]["pass"]

    value = deepcopy(calibration)
    value["per_stock"]["AAPL"]["models"][
        COMPARISON_PROFILE.candidate
    ]["direction_accuracy"] = value["per_stock"]["AAPL"]["models"][
        COMPARISON_PROFILE.reference
    ]["direction_accuracy"]
    assert not panel_analysis._gates(
        validation, value, COMPARISON_PROFILE,
    )["calibration_direction"]["pass"]

    for field in (
        "mean_candidate_close_relative_improvement_over_zero",
        "mean_candidate_close_relative_improvement_over_reference",
    ):
        value = deepcopy(calibration)
        value[field] = 0.0
        assert not panel_analysis._gates(
            validation, value, COMPARISON_PROFILE,
        )["calibration_close_mae"]["pass"]


def verify_bootstrap_boundaries() -> None:
    class StubRandom:
        def __init__(self, starts: list[int]) -> None:
            self.starts = iter(starts)

        def randrange(self, stop: int) -> int:
            value = next(self.starts)
            assert 0 <= value < stop
            return value

    for block in BOOTSTRAP_BLOCKS:
        size = block + 5
        assert panel_analysis._block_indexes(
            size, block, StubRandom([1, 0]),
        ) == (*range(1, block + 1), *range(5))
        rejects(lambda block=block: panel_analysis._block_indexes(
            block - 1, block, StubRandom([0]),
        ))
    assert panel_analysis._lower_025(
        tuple(float(index) for index in range(10_000))
    ) == 249.0

    actuals = {
        name: {
            str(index): float(1 if index % 2 == 0 else -1)
            for index in range(39)
        }
        for name in SERIES
    }
    predictions = {
        model: {
            name: {
                str(index): (
                    actuals[name][str(index)]
                    if model == COMPARISON_PROFILE.candidate else
                    0.5 * actuals[name][str(index)]
                )
                for index in range(39)
            }
            for name in SERIES
        }
        for model in COMPARISON_PROFILE.panel_models
    }
    expected = {
        "by_block_rows": {
            "13": {
                "mae_relative_improvement_lower_025": 1.0,
                "direction_candidate_minus_reference_lower_025": 0.0,
                "direction_candidate_minus_majority_lower_025":
                    0.46153846153846156,
            },
            "29": {
                "mae_relative_improvement_lower_025": 1.0,
                "direction_candidate_minus_reference_lower_025": 0.0,
                "direction_candidate_minus_majority_lower_025":
                    0.4871794871794872,
            },
            "39": {
                "mae_relative_improvement_lower_025": 1.0,
                "direction_candidate_minus_reference_lower_025": 0.0,
                "direction_candidate_minus_majority_lower_025":
                    0.4871794871794872,
            },
        },
        "mae_relative_improvement_lower_025": 1.0,
        "direction_candidate_minus_reference_lower_025": 0.0,
        "direction_candidate_minus_majority_lower_025":
            0.46153846153846156,
    }
    assert panel_analysis._bootstrap_metrics(
        actuals, predictions, COMPARISON_PROFILE,
    ) == expected

    class RecordingRandom:
        instances: list[RecordingRandom] = []

        def __init__(self, seed: int) -> None:
            assert seed == BOOTSTRAP_SEED
            self.calls: list[int] = []
            self.toggle = False
            self.instances.append(self)

        def randrange(self, stop: int) -> int:
            self.calls.append(stop)
            self.toggle = not self.toggle
            return 0 if self.toggle else stop - 1

    larger_actuals = {
        name: {
            str(index): float(1 if index % 2 == 0 else -1)
            for index in range(45)
        }
        for name in SERIES
    }
    larger_predictions = {
        model: {
            name: {
                target: (
                    actual
                    if model == COMPARISON_PROFILE.candidate else
                    0.5 * actual
                )
                for target, actual in larger_actuals[name].items()
            }
            for name in SERIES
        }
        for model in COMPARISON_PROFILE.panel_models
    }
    with patch.object(
        panel_analysis, "BOOTSTRAP_REPLICATES", 2,
    ), patch.object(panel_analysis.random, "Random", RecordingRandom):
        panel_analysis._bootstrap_metrics(
            larger_actuals, larger_predictions, COMPARISON_PROFILE,
        )
    assert len(RecordingRandom.instances) == len(BOOTSTRAP_BLOCKS)
    for instance, block in zip(
        RecordingRandom.instances, BOOTSTRAP_BLOCKS, strict=True,
    ):
        assert set(instance.calls) == {45 - block + 1}
        assert len(instance.calls) >= 2

    asymmetric_actuals = {
        "AAPL": {
            "t0": 10.0, "t1": 11.0, "t2": 12.0,
            "t3": 13.0, "t4": 14.0, "t5": 15.0,
        },
        "MSFT": {
            "t0": 20.0, "t1": 22.0, "t2": 24.0,
            "t3": 26.0, "t4": 28.0, "t5": 30.0,
        },
        "SPY": {
            "t0": -3.0, "t1": -1.0, "t2": 1.0,
            "t3": 3.0, "t4": 5.0, "t5": 7.0,
        },
    }
    asymmetric_predictions = {
        model: {
            name: {
                target: actual + (
                    0.5 if model == COMPARISON_PROFILE.candidate else 1.0
                )
                for target, actual in asymmetric_actuals[name].items()
            }
            for name in SERIES
        }
        for model in COMPARISON_PROFILE.panel_models
    }
    expected_streams = {
        (0, 2, 4): {
            "AAPL": (10.0, 12.0, 14.0),
            "MSFT": (20.0, 24.0, 28.0),
            "SPY": (-3.0, 1.0, 5.0),
        },
        (1, 3, 5): {
            "AAPL": (11.0, 13.0, 15.0),
            "MSFT": (22.0, 26.0, 30.0),
            "SPY": (-1.0, 3.0, 7.0),
        },
    }
    batches = ((0, 2, 4), (1, 3, 5)) * len(BOOTSTRAP_BLOCKS)
    pending_batches = iter(batches)
    index_calls: list[tuple[str, tuple[int, ...]]] = []
    stock_calls: list[
        tuple[str, str, tuple[int, ...], tuple[float, ...]]
    ] = []

    def index_spy(
        size: int, block: int, rng: object,
    ) -> tuple[int, ...]:
        assert size == 6 and block in BOOTSTRAP_BLOCKS
        return next(pending_batches)

    def metric_spy(
        actuals: Mapping[str, Mapping[str, float]],
        predictions: Mapping[
            str, Mapping[str, Mapping[str, float]]
        ],
        model: str, indexes: tuple[int, ...],
    ) -> tuple[float, float]:
        index_calls.append((model, indexes))
        for name in SERIES:
            observed = tuple(
                actuals[name][f"t{index}"] for index in indexes
            )
            assert observed == expected_streams[indexes][name]
            stock_calls.append((model, name, indexes, observed))
        return (
            0.5 if model == COMPARISON_PROFILE.candidate else 1.0,
            0.75 if model == COMPARISON_PROFILE.candidate else 0.25,
        )

    def majority_spy(
        actuals: Mapping[str, Mapping[str, float]],
        indexes: tuple[int, ...],
    ) -> float:
        index_calls.append(("majority", indexes))
        for name in SERIES:
            observed = tuple(
                actuals[name][f"t{index}"] for index in indexes
            )
            assert observed == expected_streams[indexes][name]
            stock_calls.append(("majority", name, indexes, observed))
        return 0.5

    with patch.object(
        panel_analysis, "BOOTSTRAP_REPLICATES", 2,
    ), patch.object(
        panel_analysis, "_block_indexes", index_spy,
    ), patch.object(
        panel_analysis, "_sampled_metrics", metric_spy,
    ), patch.object(
        panel_analysis, "_sampled_majority", majority_spy,
    ):
        panel_analysis._bootstrap_metrics(
            asymmetric_actuals, asymmetric_predictions, COMPARISON_PROFILE,
        )
    expected_index_calls = [
        item
        for indexes in batches
        for item in (
            (COMPARISON_PROFILE.candidate, indexes),
            (COMPARISON_PROFILE.reference, indexes),
            ("majority", indexes),
        )
    ]
    assert index_calls == expected_index_calls
    for offset in range(0, len(stock_calls), len(SERIES)):
        group = stock_calls[offset:offset + len(SERIES)]
        assert tuple(item[1] for item in group) == SERIES
        assert len({item[2] for item in group}) == 1

    records = [
        {
            "model": model, "series": "AAPL", "fold": 0, "seed": None,
            "metrics": {"return_mae": 1.0},
        }
        for model in LOCAL_MODELS
    ]
    for stock_index, stock in enumerate(SERIES):
        for fold in range(2):
            for seed_index, seed in enumerate(SEEDS):
                delta = stock_index + fold * 0.1 + seed_index * 0.01
                records.extend((
                    {
                        "model": COMPARISON_PROFILE.reference,
                        "series": stock, "fold": fold, "seed": seed,
                        "metrics": {"return_mae": 1.0},
                    },
                    {
                        "model": COMPARISON_PROFILE.candidate,
                        "series": stock, "fold": fold, "seed": seed,
                        "metrics": {"return_mae": 1.0 + delta},
                    },
                ))
    paired = panel_analysis._validation_metrics(
        records, COMPARISON_PROFILE,
    )["paired_candidate_minus_reference"]
    assert paired["by_stock"] == {
        "AAPL": {
            "count": 10, "mean": 0.07000000000000003,
            "stddev": 0.05196152422706634,
            "minimum": 0.0, "maximum": 0.14000000000000012,
        },
        "MSFT": {
            "count": 10, "mean": 1.07,
            "stddev": 0.05196152422706637,
            "minimum": 1.0, "maximum": 1.1400000000000001,
        },
        "SPY": {
            "count": 10, "mean": 2.07,
            "stddev": 0.05196152422706637,
            "minimum": 2.0, "maximum": 2.14,
        },
    }
    assert paired["by_fold"] == {
        "0": {
            "count": 15, "mean": 1.02,
            "stddev": 0.8166190462306562,
            "minimum": 0.0, "maximum": 2.04,
        },
        "1": {
            "count": 15, "mean": 1.12,
            "stddev": 0.8166190462306562,
            "minimum": 0.10000000000000009, "maximum": 2.14,
        },
    }
    assert paired["by_seed"] == {
        "7": {
            "count": 6, "mean": 1.05,
            "stddev": 0.8180260794538684,
            "minimum": 0.0, "maximum": 2.1,
        },
        "19": {
            "count": 6, "mean": 1.0599999999999998,
            "stddev": 0.8180260794538683,
            "minimum": 0.010000000000000009, "maximum": 2.11,
        },
        "31": {
            "count": 6, "mean": 1.07,
            "stddev": 0.8180260794538684,
            "minimum": 0.020000000000000018, "maximum": 2.12,
        },
        "43": {
            "count": 6, "mean": 1.0799999999999998,
            "stddev": 0.8180260794538684,
            "minimum": 0.030000000000000027, "maximum": 2.13,
        },
        "61": {
            "count": 6, "mean": 1.09,
            "stddev": 0.8180260794538684,
            "minimum": 0.040000000000000036, "maximum": 2.14,
        },
    }


def verify_zero_denominators() -> None:
    records = [
        {
            "model": model, "series": "AAPL", "fold": 0, "seed": None,
            "metrics": {"return_mae": 1.0},
        }
        for model in LOCAL_MODELS
    ]
    for name in SERIES:
        for fold in range(2):
            for seed in SEEDS:
                records.extend((
                    {
                        "model": COMPARISON_PROFILE.reference,
                        "series": name, "fold": fold, "seed": seed,
                        "metrics": {"return_mae": 0.0},
                    },
                    {
                        "model": COMPARISON_PROFILE.candidate,
                        "series": name, "fold": fold, "seed": seed,
                        "metrics": {"return_mae": 0.001},
                    },
                ))
    rejects(lambda: panel_analysis._validation_metrics(
        records, COMPARISON_PROFILE,
    ))

    targets = tuple(f"t{index}" for index in range(20, 59))
    timestamps_grid = tuple(f"t{index}" for index in range(60))
    actuals = {
        name: {
            target: 0.02 * (1 if index % 2 == 0 else -1)
            for index, target in enumerate(targets)
        }
        for name in SERIES
    }

    def inputs(
        reference_offset: float, candidate_offset: float,
        outcomes: float | None = None,
    ) -> tuple[
        dict[str, dict[str, dict[str, float]]],
        dict[str, dict[str, dict[str, dict[int | None, float]]]],
        dict[str, backtest.Bars],
    ]:
        predictions = {
            model: {
                name: {
                    target: actuals[name][target] + (
                        candidate_offset
                        if model == COMPARISON_PROFILE.candidate else
                        reference_offset
                        if model == COMPARISON_PROFILE.reference else 0.03
                    )
                    for target in targets
                }
                for name in SERIES
            }
            for model in COMPARISON_PROFILE.models
        }
        per_seed = {
            model: {
                name: {
                    target: {
                        seed: predictions[model][name][target]
                        for seed in SEEDS
                    }
                    for target in targets
                }
                for name in SERIES
            }
            for model in COMPARISON_PROFILE.models
        }
        bars = {}
        for name in SERIES:
            closes = [100.0] * len(timestamps_grid)
            for target in targets:
                index = timestamps_grid.index(target)
                closes[index] = (
                    outcomes if outcomes is not None else
                    100.0 * math.exp(actuals[name][target])
                )
            bars[name] = backtest.Bars(
                name, "a" * 64, timestamps_grid,
                (100.0,) * len(timestamps_grid), tuple(closes),
            )
        return predictions, per_seed, bars

    predictions, per_seed, bars = inputs(0.0, 0.001)
    rejects(lambda: panel_analysis._calibration_metrics(
        actuals, predictions, bars, per_seed, COMPARISON_PROFILE,
    ))
    rejects(lambda: panel_analysis._bootstrap_metrics(
        actuals, predictions, COMPARISON_PROFILE,
    ))

    predictions, per_seed, bars = inputs(0.01, 0.0)
    for name in SERIES:
        for target in targets:
            per_seed[COMPARISON_PROFILE.reference][name][target] = {
                7: actuals[name][target] + 0.05,
                19: actuals[name][target],
                31: actuals[name][target],
                43: actuals[name][target],
                61: actuals[name][target],
            }
    rejects(lambda: panel_analysis._calibration_metrics(
        actuals, predictions, bars, per_seed, COMPARISON_PROFILE,
    ))

    reference_prediction = 0.1
    outcome = 100.0 * math.exp(reference_prediction)
    predictions, per_seed, bars = inputs(
        reference_prediction - 0.02, 0.001, outcome,
    )
    for name in SERIES:
        for target in targets:
            predictions[COMPARISON_PROFILE.reference][name][target] = \
                reference_prediction
            per_seed[COMPARISON_PROFILE.reference][name][target] = {
                seed: reference_prediction for seed in SEEDS
            }
    rejects(lambda: panel_analysis._calibration_metrics(
        actuals, predictions, bars, per_seed, COMPARISON_PROFILE,
    ))

    predictions, per_seed, bars = inputs(0.01, 0.001, 100.0)
    rejects(lambda: panel_analysis._calibration_metrics(
        actuals, predictions, bars, per_seed, COMPARISON_PROFILE,
    ))


def verify_comparison_semantics() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-comparison-", dir=ROOT,
    ) as directory:
        fixture = PanelFixture(Path(directory), COMPARISON_PROFILE)
        fixture.validate("validate-attempt", 0)
        analysis = fixture.analyze(0)
        assert analysis is not None
        assert analysis["schema"] == 2
        validate_panel_analysis(analysis, COMPARISON_PROFILE)
        experiment = read_canonical_json(ROOT / fixture.report_path)
        assert experiment["protocol"]["panel_conditioning"] == {
            "model": "conditioned_panel_transformer",
            "kind": "learned-series-embedding",
            "series_order": list(fixture.names),
            "initialization": "zeros",
            "application": "additive-before-encoder",
        }
        expected_validation = [
            *panel_analysis._validation_keys(fixture.names, ()),
            *(
                (model, name, fold, seed)
                for model in fixture.panel_models
                for name in fixture.names for fold in range(2)
                for seed in SEEDS
            ),
        ]
        assert [
            (
                record["model"], record["series"],
                record["fold"], record["seed"],
            )
            for record in experiment["validation"]
        ] == expected_validation
        expected_calibration = [
            *panel_analysis._calibration_keys(fixture.names, ()),
            *(
                (model, name, seed)
                for model in fixture.panel_models
                for name in fixture.names for seed in SEEDS
            ),
        ]
        assert [
            (record["model"], record["series"], record["seed"])
            for record in experiment["calibration"]
        ] == expected_calibration
        local_keys = panel_analysis._calibration_keys(fixture.names, ())
        assert [
            (item["model"], item["series"], item["seed"])
            for item in experiment["model_fingerprints"]
        ] == [
            *sorted(
                local_keys,
                key=lambda item: (
                    item[0], item[1],
                    -1 if item[2] is None else item[2],
                ),
            ),
            *(
                (model, name, seed)
                for model in fixture.panel_models
                for name in fixture.names for seed in SEEDS
            ),
        ]
        baseline = read_canonical_json(ROOT / fixture.baseline_report_path)
        local = set(LOCAL_MODELS)
        assert [
            item for item in experiment["validation"]
            if item["model"] in local
        ] == baseline["validation"]
        assert [
            item for item in experiment["calibration"]
            if item["model"] in local
        ] == baseline["calibration"]
        assert [
            item for item in experiment["model_fingerprints"]
            if item["model"] in local
        ] == baseline["model_fingerprints"]
        live_lines = (ROOT / fixture.ledger_path).read_text(
            encoding="utf-8",
        ).splitlines(keepends=True)
        baseline_lines = (ROOT / fixture.baseline_ledger_path).read_text(
            encoding="utf-8",
        ).splitlines(keepends=True)
        live_rows = [
            json.loads(line) for line in live_lines
        ]
        assert [
            line for row, line in zip(live_rows, live_lines, strict=True)
            if row["model"] in local
        ] == baseline_lines
        assert experiment["test"] == []

        paired = analysis["validation"]["paired_candidate_minus_reference"]
        assert (
            paired["candidate_model"], paired["reference_model"],
        ) == (fixture.candidate, fixture.reference)
        assert set(paired["by_stock"]) == set(fixture.names)
        assert set(paired["by_fold"]) == {"0", "1"}
        assert set(paired["by_seed"]) == {str(seed) for seed in SEEDS}
        assert all(
            value["count"] == 10 for value in paired["by_stock"].values()
        )
        assert all(
            value["count"] == 15 for value in paired["by_fold"].values()
        )
        assert all(
            value["count"] == 6 for value in paired["by_seed"].values()
        )

        calibration = analysis["calibration"]
        offsets = {
            7: -0.004, 19: -0.002, 31: 0.0, 43: 0.002, 61: 0.004,
        }
        for seed, offset in offsets.items():
            candidate_mae = abs(offset / 4.0)
            reference_mae = abs(0.03 - offset / 4.0)
            expected = 1.0 - candidate_mae / reference_mae
            assert math.isclose(
                calibration["leave_one_seed_out"][str(seed)][
                    "relative_improvement"
                ],
                expected, rel_tol=0.0, abs_tol=1e-15,
            )
        bootstrap = calibration["bootstrap"]
        assert tuple(bootstrap["by_block_rows"]) == tuple(
            str(block) for block in BOOTSTRAP_BLOCKS
        )
        for name in (
            "mae_relative_improvement_lower_025",
            "direction_candidate_minus_reference_lower_025",
            "direction_candidate_minus_majority_lower_025",
        ):
            assert bootstrap[name] == min(
                block[name] for block in bootstrap["by_block_rows"].values()
            )
        assert analysis["gates"]["all_pass"]
        verify_comparison_gate_boundaries(analysis)
        assert fixture.finalize("pass")["status"] == "pass"

    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-comparison-failure-", dir=ROOT,
    ) as directory:
        fixture = PanelFixture(Path(directory), COMPARISON_PROFILE)
        fixture.write_live("zero")
        analysis = fixture.analyze(3)
        assert analysis is not None and not analysis["gates"]["all_pass"]
        assert fixture.finalize("gate-failure")["status"] == "gate-failure"

    def reject_report(
        root: Path, mutate: Callable[[dict[str, object]], None],
    ) -> None:
        fixture = PanelFixture(root, COMPARISON_PROFILE)
        value = read_canonical_json(ROOT / fixture.report_path)
        mutate(value)
        write_canonical_json(ROOT / fixture.report_path, value)
        fixture.analyze(2)

    def break_validation_fit(
        value: dict[str, object], model: str,
    ) -> None:
        record = next(
            item for item in value["validation"]
            if item["model"] == model
        )
        record["best_epoch"] = 6

    def break_calibration_epochs(
        value: dict[str, object], model: str,
    ) -> None:
        record = next(
            item for item in value["calibration"]
            if item["model"] == model
        )
        record["epochs"] = 6
        fingerprint = next(
            item for item in value["model_fingerprints"]
            if (
                item["model"], item["series"], item["seed"]
            ) == (record["model"], record["series"], record["seed"])
        )
        fingerprint["epochs"] = 6

    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-comparison-rejections-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        mutations: tuple[
            Callable[[dict[str, object]], None], ...
        ] = (
            lambda value: value["validation"].pop(),
            lambda value: value["validation"].__setitem__(
                slice(-2, None),
                list(reversed(value["validation"][-2:])),
            ),
            lambda value: value["calibration"].pop(),
            lambda value: value["calibration"].__setitem__(
                slice(-2, None),
                list(reversed(value["calibration"][-2:])),
            ),
            lambda value: value["model_fingerprints"].pop(),
            lambda value: value["model_fingerprints"].__setitem__(
                slice(-2, None),
                list(reversed(value["model_fingerprints"][-2:])),
            ),
            lambda value: value["protocol"].pop("panel_conditioning"),
            lambda value: value["protocol"]["panel_conditioning"].__setitem__(
                "series_order", ["MSFT", "AAPL", "SPY"],
            ),
            lambda value: break_validation_fit(
                value, "panel_transformer",
            ),
            lambda value: break_validation_fit(
                value, "conditioned_panel_transformer",
            ),
            lambda value: break_calibration_epochs(
                value, "panel_transformer",
            ),
            lambda value: break_calibration_epochs(
                value, "conditioned_panel_transformer",
            ),
            lambda value: value.__setitem__("test", [{"forbidden": True}]),
        )
        for index, mutate in enumerate(mutations):
            case = root / str(index)
            case.mkdir()
            reject_report(case, mutate)

        ledger_case = root / "ledger"
        ledger_case.mkdir()
        fixture = PanelFixture(ledger_case, COMPARISON_PROFILE)
        rows = [
            json.loads(line)
            for line in (ROOT / fixture.ledger_path).read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        rows[-1], rows[-2] = rows[-2], rows[-1]
        fixture._write_ledger(fixture.ledger_path, rows)
        report = read_canonical_json(ROOT / fixture.report_path)
        report["calibration_prediction_ledger"]["sha256"] = file_sha256(
            ROOT / fixture.ledger_path
        )
        write_canonical_json(ROOT / fixture.report_path, report)
        fixture.analyze(2)

        deleted_ledger_case = root / "deleted-ledger"
        deleted_ledger_case.mkdir()
        fixture = PanelFixture(
            deleted_ledger_case, COMPARISON_PROFILE,
        )
        rows = [
            json.loads(line)
            for line in (ROOT / fixture.ledger_path).read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        rows.pop()
        fixture._write_ledger(fixture.ledger_path, rows)
        report = read_canonical_json(ROOT / fixture.report_path)
        report["calibration_prediction_ledger"].update({
            "records": len(rows),
            "sha256": file_sha256(ROOT / fixture.ledger_path),
        })
        write_canonical_json(ROOT / fixture.report_path, report)
        fixture.analyze(2)

        forged_case = root / "forged"
        forged_case.mkdir()
        fixture = PanelFixture(forged_case, COMPARISON_PROFILE)
        analysis = fixture.analyze(0)
        assert analysis is not None

        def forge_calibration_macro(
            value: dict[str, object],
        ) -> None:
            calibration = value["calibration"]
            macro = calibration["macro_return_mae"]
            macro[COMPARISON_PROFILE.candidate] += 0.001
            calibration["relative_improvement_vs_reference"] = 1.0 - (
                macro[COMPARISON_PROFILE.candidate] /
                macro[COMPARISON_PROFILE.reference]
            )

        def forge_paired_macro(value: dict[str, object]) -> None:
            paired = value["validation"][
                "paired_candidate_minus_reference"
            ]
            paired["mean_delta"] += 0.001
            for axis in ("by_stock", "by_fold", "by_seed"):
                for item in paired[axis].values():
                    item["mean"] += 0.001
                    item["minimum"] += 0.001
                    item["maximum"] += 0.001

        def forge_axis_mean(
            value: dict[str, object], axis: str,
        ) -> None:
            item = next(iter(
                value["validation"][
                    "paired_candidate_minus_reference"
                ][axis].values()
            ))
            item["mean"] += 0.001
            item["maximum"] = max(item["maximum"], item["mean"])

        def forge_macro_direction(value: dict[str, object]) -> None:
            value["calibration"]["macro_direction_accuracy"][
                "transformer"
            ] += 0.001

        def forge_macro_majority(value: dict[str, object]) -> None:
            value["calibration"]["macro_majority_direction"] += 0.001

        def forge_close_mean(value: dict[str, object]) -> None:
            value["calibration"][
                "mean_candidate_close_relative_improvement_over_zero"
            ] += 0.001

        def forge_tiny_positive_close(
            value: dict[str, object],
        ) -> None:
            calibration = value["calibration"]
            for stock in calibration["per_stock"].values():
                zero_close = stock["zero_return_close_mae"]
                stock["models"][COMPARISON_PROFILE.reference][
                    "close_mae"
                ] = zero_close
                stock["models"][COMPARISON_PROFILE.candidate][
                    "close_mae"
                ] = zero_close
                stock[
                    "candidate_close_relative_improvement_over_zero"
                ] = 0.0
                stock[
                    "candidate_close_relative_improvement_over_reference"
                ] = 0.0
            calibration[
                "mean_candidate_close_relative_improvement_over_zero"
            ] = 5e-16
            calibration[
                "mean_candidate_close_relative_improvement_over_reference"
            ] = 5e-16

        def forge_last_close_return(value: dict[str, object]) -> None:
            calibration = value["calibration"]
            calibration["per_stock"]["AAPL"]["models"]["last_close"][
                "return_mae"
            ] += 0.001
            calibration["macro_return_mae"]["last_close"] = fmean(
                calibration["per_stock"][name]["models"]["last_close"][
                    "return_mae"
                ]
                for name in SERIES
            )

        def forge_last_close_close(value: dict[str, object]) -> None:
            value["calibration"]["per_stock"]["AAPL"]["models"][
                "last_close"
            ]["close_mae"] += 0.001

        aggregate_mutations = (
            forge_calibration_macro,
            forge_paired_macro,
            lambda value: forge_axis_mean(value, "by_stock"),
            lambda value: forge_axis_mean(value, "by_fold"),
            lambda value: forge_axis_mean(value, "by_seed"),
            forge_macro_direction,
            forge_macro_majority,
            forge_close_mean,
            forge_last_close_return,
            forge_last_close_close,
        )
        for mutate in aggregate_mutations:
            forged = deepcopy(analysis)
            mutate(forged)
            forged["gates"] = panel_gates(
                forged["validation"], forged["calibration"],
                COMPARISON_PROFILE,
            )
            rejects(lambda forged=forged: validate_panel_analysis(
                forged, COMPARISON_PROFILE,
            ))

        tiny_close = deepcopy(analysis)
        forge_tiny_positive_close(tiny_close)
        tiny_close["gates"] = panel_gates(
            tiny_close["validation"], tiny_close["calibration"],
            COMPARISON_PROFILE,
        )
        assert tiny_close["gates"]["calibration_close_mae"]["pass"]
        for field in (
            "candidate_close_relative_improvement_over_zero",
            "candidate_close_relative_improvement_over_reference",
        ):
            assert fmean(
                stock[field]
                for stock in tiny_close["calibration"][
                    "per_stock"
                ].values()
            ) == 0.0
        rejects(lambda: validate_panel_analysis(
            tiny_close, COMPARISON_PROFILE,
        ))

        for mutate in (
            lambda value: value["gates"][
                "validation_macro_mae"
            ].__setitem__("pass", False),
            lambda value: value["protocol"]["bootstrap"].__setitem__(
                "block_rows", [13],
            ),
            lambda value: value["validation"][
                "paired_candidate_minus_reference"
            ].pop("by_seed"),
            lambda value: value["validation"][
                "paired_candidate_minus_reference"
            ].update({
                "candidate_model": COMPARISON_PROFILE.reference,
                "reference_model": COMPARISON_PROFILE.candidate,
            }),
            lambda value: value["calibration"]["bootstrap"].__setitem__(
                "extra", 1,
            ),
            lambda value: value.__setitem__("schema", 1),
        ):
            forged = deepcopy(analysis)
            mutate(forged)
            rejects(lambda forged=forged: validate_panel_analysis(
                forged, COMPARISON_PROFILE,
            ))

        write_canonical_json(ROOT / fixture.analysis_path, tiny_close)
        fixture.finalize("pass", 2)

        analysis["gates"]["validation_macro_mae"]["pass"] = False
        write_canonical_json(ROOT / fixture.analysis_path, analysis)
        fixture.finalize("pass", 2)

        count_case = root / "counts"
        count_case.mkdir()
        fixture = PanelFixture(count_case, COMPARISON_PROFILE)
        analysis = fixture.analyze(0)
        assert analysis is not None
        attempt = read_canonical_json(ROOT / fixture.attempt_path)
        attempt["expected_equivalent_runs"] = 208
        write_canonical_json(ROOT / fixture.attempt_path, attempt)
        report = read_canonical_json(ROOT / fixture.report_path)
        report["attempt_manifest"]["sha256"] = file_sha256(
            ROOT / fixture.attempt_path
        )
        write_canonical_json(ROOT / fixture.report_path, report)
        analysis["inputs"]["attempt"]["sha256"] = file_sha256(
            ROOT / fixture.attempt_path
        )
        analysis["inputs"]["experiment_report"]["sha256"] = file_sha256(
            ROOT / fixture.report_path
        )
        write_canonical_json(ROOT / fixture.analysis_path, analysis)
        fixture.finalize("pass", 2)


def verify_panel_semantics() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-panel-analysis-", dir=ROOT,
    ) as directory:
        fixture = PanelFixture(Path(directory))
        fixture.validate("validate-attempt", 0)
        fixture.validate("preflight", 2)

        for model, path in (
            (PanelInputs, fixture.inputs_path),
            (PanelAttempt, fixture.attempt_path),
        ):
            value = read_canonical_json(ROOT / path)
            value["schema"] = 1.0
            invalid = fixture.relative / f"invalid-{path.name}"
            write_canonical_json(ROOT / invalid, value)
            try:
                model.read(ROOT / invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("floating-point manifest schema accepted")

        binding_root = Path(directory) / "baseline-binding"
        binding_root.mkdir()
        binding = PanelFixture(binding_root)
        baseline = read_canonical_json(ROOT / binding.baseline_report_path)
        baseline["sweep_input"]["sha256"] = int("1" * 64)
        write_canonical_json(ROOT / binding.baseline_report_path, baseline)
        inputs = read_canonical_json(ROOT / binding.inputs_path)
        inputs["baseline_report"]["sha256"] = file_sha256(
            ROOT / binding.baseline_report_path
        )
        write_canonical_json(ROOT / binding.inputs_path, inputs)
        attempt = read_canonical_json(ROOT / binding.attempt_path)
        attempt["baseline_report"]["sha256"] = \
            inputs["baseline_report"]["sha256"]
        attempt["input_manifest"]["sha256"] = file_sha256(
            ROOT / binding.inputs_path
        )
        write_canonical_json(ROOT / binding.attempt_path, attempt)
        binding.write_live("perfect")
        binding.analyze(2)

        report = fixture.analyze(0)
        assert report is not None
        assert report == expected_legacy_analysis(fixture)
        assert report["status"] == "pass"
        assert set(report["inputs"]) == {
            "run_id", "attempt", "input_manifest", "config",
            "baseline_report", "baseline_ledger", "experiment_report",
            "calibration_ledger", "series",
        }
        assert [item["name"] for item in report["inputs"]["series"]] == \
            list(fixture.names)
        assert report["inputs"]["attempt"] == {
            "path": fixture.attempt_path.as_posix(),
            "sha256": file_sha256(ROOT / fixture.attempt_path),
        }

        paired = report["validation"][
            "paired_panel_minus_local_transformer"
        ]
        assert paired["wins"] == 30
        assert paired["ties"] == 0
        assert paired["losses"] == 0
        assert math.isclose(paired["mean_delta"], -0.01, abs_tol=1e-15)
        assert all(
            math.isclose(value, -0.01, abs_tol=1e-15)
            for value in paired["per_stock_mean_delta"].values()
        )

        calibration = report["calibration"]
        assert calibration["macro_majority_direction"] == 0.5
        assert calibration["macro_return_mae"]["panel_transformer"] < 1e-15
        ledger = [
            json.loads(line)
            for line in (ROOT / fixture.ledger_path).read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        first_target = fixture.grids["AAPL"].calibration_targets[0]
        seed_predictions = [
            item["predicted_log_return"]
            for item in ledger
            if item["model"] == "panel_transformer" and
            item["series"] == "AAPL" and
            item["target_time"] == first_target
        ]
        assert len(seed_predictions) == len(SEEDS)
        assert any(
            not math.isclose(
                value, fixture.actuals["AAPL"][first_target], abs_tol=1e-15,
            )
            for value in seed_predictions
        )

        for name in fixture.names:
            stock = calibration["per_stock"][name]
            assert stock["samples"] == 2
            assert stock["majority_direction"] == {
                "p_up": 0.5, "p_down": 0.5, "p_flat": 0.0,
                "reference": 0.5,
            }
            indexes = {
                timestamp: index
                for index, timestamp in enumerate(
                    fixture.bars[name].timestamps
                )
            }
            expected_zero_close = sum(
                abs(
                    fixture.bars[name].opens[indexes[target] - 12] -
                    fixture.bars[name].closes[indexes[target]]
                )
                for target in fixture.grids[name].calibration_targets
            ) / 2
            assert math.isclose(
                stock["zero_return_close_mae"], expected_zero_close,
                abs_tol=1e-12,
            )
            assert stock["models"]["panel_transformer"]["close_mae"] < 1e-12

        verify_gate_boundaries(report["validation"], calibration)

        (ROOT / fixture.analysis_path).unlink()
        fixture.write_live("zero")
        failure = fixture.analyze(3)
        assert failure is not None
        assert failure["status"] == "gate-failure"
        gates = failure["gates"]
        assert not gates["calibration_macro_mae"]["pass"]
        assert gates["calibration_close_mae"]["margin"] == 0.0
        assert not gates["calibration_close_mae"]["pass"]
        assert all(
            value["margin"] == 0.0 and not value["pass"]
            for value in gates[
                "calibration_per_stock_zero"
            ]["per_stock"].values()
        )

        (ROOT / fixture.analysis_path).unlink()
        fixture.write_live("perfect")
        fixture.analyze(2, tuple(reversed(fixture.names)))

        fixture.write_live("perfect", mutate_local=True)
        fixture.analyze(2)

        fixture.write_live("perfect", forbidden_test=True)
        fixture.analyze(2)

        fixture.write_live("perfect")
        report = read_canonical_json(ROOT / fixture.report_path)
        report["calibration"][-1]["metrics"]["return_mae"] = 0.99
        write_canonical_json(ROOT / fixture.report_path, report)
        fixture.analyze(2)

        fixture.write_live("perfect")
        report = read_canonical_json(ROOT / fixture.report_path)
        report["schema"] = 6.0
        write_canonical_json(ROOT / fixture.report_path, report)
        fixture.analyze(2)

        fixture.write_live("perfect")
        report = read_canonical_json(ROOT / fixture.report_path)
        report["validation"][0]["fold"] = False
        write_canonical_json(ROOT / fixture.report_path, report)
        fixture.analyze(2)

        fixture.write_live("perfect")
        report = read_canonical_json(ROOT / fixture.report_path)
        report["validation"][0]["seed"] = 7.0
        write_canonical_json(ROOT / fixture.report_path, report)
        fixture.analyze(2)

        fixture.write_live("perfect")
        ledger = [
            json.loads(line)
            for line in (ROOT / fixture.ledger_path).read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        ledger[0]["schema"] = 2
        fixture._write_ledger(fixture.ledger_path, ledger)
        report = read_canonical_json(ROOT / fixture.report_path)
        report["calibration_prediction_ledger"]["sha256"] = file_sha256(
            ROOT / fixture.ledger_path
        )
        write_canonical_json(ROOT / fixture.report_path, report)
        fixture.analyze(2)

        fixture.write_live("perfect")
        ledger = [
            json.loads(line)
            for line in (ROOT / fixture.ledger_path).read_text(
                encoding="utf-8",
            ).splitlines()
        ]
        row = next(
            item for item in ledger
            if item["model"] == "last_close" and
            item["predicted_log_return"] == 0.0
        )
        row["predicted_log_return"] = 0
        fixture._write_ledger(fixture.ledger_path, ledger)
        report = read_canonical_json(ROOT / fixture.report_path)
        report["calibration_prediction_ledger"]["sha256"] = file_sha256(
            ROOT / fixture.ledger_path
        )
        write_canonical_json(ROOT / fixture.report_path, report)
        fixture.analyze(2)

        fixture.write_live("perfect")
        attempt = read_canonical_json(ROOT / fixture.attempt_path)
        other_python = Path("/usr/bin/python3").resolve(strict=True)
        version = subprocess.run(
            [str(other_python), "--version"], check=True,
            capture_output=True, text=True,
        )
        attempt["primary_python"] = {
            "path": str(other_python),
            "sha256": file_sha256(other_python),
            "version": (version.stdout or version.stderr).strip(),
        }
        write_canonical_json(ROOT / fixture.attempt_path, attempt)
        fixture.analyze(2)


def main() -> None:
    assert COMPARISON_PROFILE.expected_runs == 207
    assert COMPARISON_PROFILE.expected_panel_fits == 30
    assert COMPARISON_PROFILE.analysis_schema == 2
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    inputs = read_canonical_json(INPUTS)
    assert config == expected_config()
    assert inputs == expected_inputs()
    assert PanelInputs.read(INPUTS).series[0].name == "AAPL"
    assert not {
        "tools/analyze_universe.py",
        "tools/fetch_universe.py",
        "tools/fetch_massive.py",
    } & set(SOURCE_PATHS)

    per_fold = sum(
        len(SEEDS) if model in ("transformer", "mlp", "panel_transformer") else 1
        for model in MODELS
    )
    assert 3 * (per_fold * 2 + per_fold) == 162
    comparison_per_fold = sum(
        len(SEEDS) if model in panel_analysis.SEEDED_MODELS else 1
        for model in COMPARISON_MODELS
    )
    assert 3 * (
        comparison_per_fold * 2 + comparison_per_fold
    ) == 207

    assert "panel_transformer" not in backtest.SEEDED_MODELS
    assert "panel_transformer" not in backtest.POLICY_MODELS
    assert "panel_transformer" not in analyze_universe.POLICY_MODELS
    assert "panel_transformer" not in replay_calibration.POLICY_MODELS
    assert "panel_transformer" not in select_policy.POLICY_MODELS
    assert "conditioned_panel_transformer" not in backtest.SEEDED_MODELS
    assert "conditioned_panel_transformer" not in backtest.POLICY_MODELS
    assert "conditioned_panel_transformer" not in analyze_universe.POLICY_MODELS
    assert "conditioned_panel_transformer" not in \
        replay_calibration.POLICY_MODELS
    assert "conditioned_panel_transformer" not in select_policy.POLICY_MODELS

    verify_profiles_and_sources()
    verify_cli_surface()
    verify_panel_semantics()
    verify_bootstrap_boundaries()
    verify_zero_denominators()
    verify_comparison_semantics()
    verify_finalizer_transitions()
    verify_successful_finalization()
    verify_finalizer_publication_races()
    verify_analysis_publication_race()
    verify_rejection_surface()
    print("panel analysis tests passed")


if __name__ == "__main__":
    main()
