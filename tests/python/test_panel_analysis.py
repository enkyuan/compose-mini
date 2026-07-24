#!/usr/bin/env python3
"""Verify the frozen panel benchmark and its standard-library CLI boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
    FINALIZER_SOURCE_PATHS, SOURCE_PATHS, PanelAttempt, PanelInputs,
    mkdir_nofollow, read_canonical_json, read_canonical_json_lines,
    regular_file_identities,
)

LOCAL_MODELS = (
    "transformer", "linear", "mlp", "rolling_mean", "last_close",
)
MODELS = (*LOCAL_MODELS, "panel_transformer")
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


def expected_config() -> dict[str, object]:
    return {
        "alignment_horizon_bars": 13,
        "batch_size": 128,
        "candidates": [{
            "feature_set": "ohlcv",
            "ff_dim": 32,
            "heads": 2,
            "layers": 1,
            "learning_rate": 0.0003,
            "mlp_dim": 32,
            "model_dim": 16,
            "name": "raw-17",
            "ridge": 0.001,
            "rolling_window": 8,
            "seq_len": 17,
            "weight_decay": 0.0001,
        }],
        "epochs": 100,
        "fold_fraction": 0.1,
        "folds": 2,
        "models": list(MODELS),
        "patience": 10,
        "seeds": list(SEEDS),
        "target_horizon_bars": 13,
        "target_kind": "executable-return-v1",
    }


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


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def timestamps() -> tuple[str, ...]:
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    return tuple(
        (start + timedelta(minutes=index)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for index in range(170)
    )


def write_canonical_json(path: Path, value: object) -> None:
    write_json(path, value)


def write_panel_csv(path: Path, stock: int) -> None:
    grid = panel_analysis._grid(timestamps())
    targets = tuple(timestamps().index(value)
                    for value in grid.calibration_targets)
    open_values = {
        targets[0] - 12: 80.0 + 10.0 * stock,
        targets[1] - 12: 120.0 + 10.0 * stock,
    }
    actuals = (0.02 + 0.005 * stock, -0.02 - 0.005 * stock)
    close_values = {
        target: open_values[target - 12] * math.exp(actual)
        for target, actual in zip(targets, actuals, strict=True)
    }
    lines = ["timestamp,open,high,low,close,volume"]
    for index, timestamp in enumerate(timestamps()):
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
    def __init__(self, root: Path) -> None:
        self.root = root
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
        for stock, name in enumerate(self.names):
            write_panel_csv(ROOT / self.csv_paths[name], stock)
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

        write_canonical_json(ROOT / self.config_path, expected_config())
        baseline_records = self._ledger(False, "perfect")
        self._write_ledger(self.baseline_ledger_path, baseline_records)
        baseline = self._report(
            False, self.baseline_ledger_path, baseline_records,
        )
        write_canonical_json(ROOT / self.baseline_report_path, baseline)
        self._write_inputs()
        self.environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": f"{self.run_dir}/.pycache",
        }
        self._write_attempt()
        self.write_live("perfect")

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

    def _validation(self, panel: bool) -> list[dict[str, object]]:
        records = []
        for model, name, fold, seed in panel_analysis._validation_keys(
            self.names, panel,
        ):
            targets, samples = self.grids[name].validation[fold]
            records.append(report_record(
                model, name, seed, fold, targets, samples, False,
            ))
        return records

    def _calibration(
        self, panel: bool, ledger: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        records = []
        for model, name, seed in panel_analysis._calibration_keys(
            self.names, panel,
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
        self, panel: bool, panel_predictions: str,
    ) -> list[dict[str, object]]:
        records = []
        offsets = {7: -0.04, 19: -0.02, 31: 0.0, 43: 0.02, 61: 0.04}
        local_offsets = {
            7: -0.004, 19: -0.002, 31: 0.0, 43: 0.002, 61: 0.004,
        }
        for model, name, seed in panel_analysis._calibration_keys(
            self.names, panel,
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
        self, panel: bool, ledger_path: Path,
        ledger_records: list[dict[str, object]],
    ) -> dict[str, object]:
        models = MODELS if panel else LOCAL_MODELS
        validation = self._validation(panel)
        calibration = self._calibration(panel, ledger_records)
        calibration_keys = panel_analysis._calibration_keys(
            self.names, panel,
        )
        fingerprints = [
            {
                "model": model,
                "series": name,
                "seed": seed,
                "epochs": 5 if seed is not None else None,
                "sha256": digest_text(f"{model}:{name}:{seed}"),
            }
            for model, name, seed in sorted(
                calibration_keys,
                key=lambda item: (
                    item[0], item[1],
                    -1 if item[2] is None else item[2],
                ),
            )
        ]
        report = {
            "schema": 6,
            "protocol": panel_analysis._expected_protocol(
                162 if panel else 117,
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
            "sweep": panel_analysis._expected_sweep(models),
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
        common = [
            self.attempt_path.as_posix(),
            self.inputs_path.as_posix(),
            self.config_path.as_posix(),
            self.baseline_report_path.as_posix(),
            self.baseline_ledger_path.as_posix(),
        ]
        series = [
            f"{name}={self.csv_paths[name].as_posix()}"
            for name in self.names
        ]
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
            "commands": {
                "validate_attempt": [
                    "tools/analyze_panel.py", "validate-attempt",
                    *common, *series,
                ],
                "preflight": [
                    "tools/analyze_panel.py", "preflight",
                    *common, *series,
                ],
                "experiment": [
                    "tools/experiment.py", self.config_path.as_posix(),
                    self.report_path.as_posix(), *series,
                    "--attempt-manifest", self.attempt_path.as_posix(),
                    "--input-manifest", self.inputs_path.as_posix(),
                    "--baseline-report", self.baseline_report_path.as_posix(),
                    "--baseline-ledger", self.baseline_ledger_path.as_posix(),
                    "--device", "cpu", "--calibration-only",
                    "--calibration-predictions",
                    self.ledger_path.as_posix(),
                    "--max-runs", "162",
                ],
                "analyze": [
                    "tools/analyze_panel.py", "analyze", *common,
                    self.report_path.as_posix(), self.ledger_path.as_posix(),
                    self.analysis_path.as_posix(), *series,
                ],
                "finalizer_prefix": [
                    "tools/finalize_panel_attempt.py",
                    self.attempt_path.as_posix(),
                    self.outcome_path.as_posix(),
                ],
            },
            "expected_equivalent_runs": 162,
            "expected_panel_fits": 15,
            "outputs": {
                "experiment_report": self.report_path.as_posix(),
                "calibration_ledger": self.ledger_path.as_posix(),
                "analysis_report": self.analysis_path.as_posix(),
                "outcome": self.outcome_path.as_posix(),
            },
        })

    def write_live(
        self, panel_predictions: str, *, mutate_local: bool = False,
        forbidden_test: bool = False,
    ) -> None:
        records = self._ledger(True, panel_predictions)
        self._write_ledger(self.ledger_path, records)
        report = self._report(True, self.ledger_path, records)
        if mutate_local:
            metrics = report["validation"][0]["metrics"]
            metrics["return_mae"] += 0.001
            report["validation_summary"] = panel_analysis._validation_summary(
                report["validation"], MODELS,
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

    assert "panel_transformer" not in backtest.SEEDED_MODELS
    assert "panel_transformer" not in backtest.POLICY_MODELS
    assert "panel_transformer" not in analyze_universe.POLICY_MODELS
    assert "panel_transformer" not in replay_calibration.POLICY_MODELS
    assert "panel_transformer" not in select_policy.POLICY_MODELS

    verify_cli_surface()
    verify_panel_semantics()
    verify_finalizer_transitions()
    verify_successful_finalization()
    verify_finalizer_publication_races()
    verify_analysis_publication_race()
    verify_rejection_surface()
    print("panel analysis tests passed")


if __name__ == "__main__":
    main()
