#!/usr/bin/env python3
"""Verify frozen universe replay, integrity checks, metrics, and strict gates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from statistics import fmean
import errno
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ANALYZER = ROOT / "tools/analyze_universe.py"
REPLAYER = ROOT / "tools/replay_calibration.py"
missing = [str(path.relative_to(ROOT)) for path in (ANALYZER, REPLAYER)
           if not path.is_file()]
assert not missing, f"missing universe analysis tools: {', '.join(missing)}"

from tools import analyze_universe as analysis
from tools import replay_calibration as replay_tool
from tools.backtest import (
    Costs, experiment_fingerprint, load_bars, read_forecasts,
)
from tools.fetch_universe import fetch_universe
from tools.files import file_sha256, write_json
from tools.select_policy import select_policy

MODELS = ("transformer", "linear", "mlp", "rolling_mean", "last_close")
POLICY_MODELS = ("transformer", "mlp", "linear")
SEEDS = (7, 19, 31, 43, 61)
TARGETS = (29, 30, 31)


def run(
    command: Sequence[object], expected: int = 0,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(item) for item in command], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert result.returncode == expected, (
        f"{command} returned {result.returncode}, expected {expected}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def run_main(function: Callable[[], None], argv: Sequence[object],
             expected: int) -> None:
    with patch.object(sys, "argv", [str(item) for item in argv]), \
         redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        try:
            function()
        except SystemExit as error:
            assert error.code == expected
        else:
            assert expected == 0


def hardlinks_supported(directory: Path) -> bool:
    source = directory / "hardlink-probe-source"
    alias = directory / "hardlink-probe-alias"
    source.write_bytes(b"probe")
    try:
        os.link(source, alias)
    except OSError as error:
        unsupported = {
            errno.EACCES, errno.EPERM, errno.EXDEV, errno.EROFS,
            getattr(errno, "ENOTSUP", -1),
            getattr(errno, "EOPNOTSUPP", -1),
        }
        if error.errno in unsupported:
            return False
        raise
    finally:
        alias.unlink(missing_ok=True)
        source.unlink(missing_ok=True)
    return True


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def write_canonical_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def empty_test_summary() -> dict[str, object]:
    statistics = {"count": 0, "mean": None, "stddev": None}
    return {
        model: {
            "by_series": {},
            "return_macro_by_seed": {},
            "return_macro_across_seeds": {
                metric: dict(statistics)
                for metric in ("return_mse", "return_mae", "direction_accuracy")
            },
        }
        for model in MODELS
    }


def metrics() -> dict[str, float]:
    return {
        "return_mse": 0.0,
        "return_mae": 0.0,
        "direction_accuracy": 0.0,
        "close_mae": 0.0,
        "zero_return_baseline_mae": 1.0,
    }


def manifest_value() -> dict[str, object]:
    return {
        "adjusted": True,
        "declared_on": "2026-07-23",
        "eligibility_date": "2024-07-22",
        "end": "2026-07-21",
        "interval_minutes": 30,
        "purpose": "Synthetic common-stock universe analysis fixture.",
        "schema": 1,
        "series": [
            {"stratum": f"sector-{index:02d}", "ticker": f"T{index:02d}"}
            for index in range(11)
        ],
        "session": "regular",
        "start": "2024-07-22",
    }


def config_value() -> dict[str, object]:
    return json.loads(
        (ROOT / "experiments/executable-h13-universe.example.json").read_text(
            encoding="utf-8",
        )
    )


def timestamps() -> tuple[str, ...]:
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    return tuple(
        (start + timedelta(days=index)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for index in range(50)
    )


def write_csv(path: Path, stock: int) -> None:
    signs = (-1.0, 1.0) if stock == 0 else (1.0, -1.0, 1.0)
    indexes = TARGETS[1:] if stock == 0 else TARGETS
    closes = [100.0] * 50
    for target, sign in zip(indexes, signs, strict=True):
        closes[target] = 100.0 * math.exp(sign * (0.006 + stock * 0.0004))
    lines = ["timestamp,open,high,low,close,volume"]
    for index, (timestamp, close) in enumerate(
        zip(timestamps(), closes, strict=True)
    ):
        high, low = max(100.0, close) + 1.0, min(100.0, close) - 1.0
        lines.append(
            f"{timestamp},100,{high:.9g},{low:.9g},{close:.9g},{1000 + index}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_gap_audit_contract(directory: Path) -> None:
    manifest_path = directory / "gap-manifest.json"
    manifest = manifest_value()
    manifest["series"] = [{"stratum": "generic", "ticker": "AAPL"}]
    write_json(manifest_path, manifest)
    values = (
        ("2024-07-22T13:30:00+00:00", 100.0),
        ("2024-07-22T14:30:00+00:00", 101.0),
    )

    def request(url: str) -> dict[str, object]:
        if "/v3/reference/tickers/" in url:
            return {
                "status": "OK",
                "results": {
                    "ticker": "AAPL", "active": True, "market": "stocks",
                    "locale": "us", "type": "CS", "currency_name": "usd",
                },
            }
        return {
            "status": "OK", "ticker": "AAPL",
            "results": [
                {
                    "t": int(datetime.fromisoformat(timestamp).timestamp() * 1000),
                    "o": close, "h": close, "l": close, "c": close, "v": 1.0,
                }
                for timestamp, close in values
            ],
        }

    report = fetch_universe(
        manifest_path, directory / "gap-csv", directory / "gap-fetch.json",
        key="fake-secret", requester=request,
    )
    parsed = analysis.UniverseManifest.read(manifest_path)
    manifest_input = analysis.FrozenInput(
        manifest_path, manifest_path, file_sha256(manifest_path),
    )
    bars = {"AAPL": load_bars(Path(report["series"][0]["csv"]["path"]))}
    analysis.validate_fetch(report, parsed, manifest_input, bars)

    def rejected(change: Callable[[dict[str, object]], None]) -> None:
        candidate = json.loads(json.dumps(report))
        change(candidate)
        try:
            analysis.validate_fetch(candidate, parsed, manifest_input, bars)
        except ValueError:
            return
        raise AssertionError("mutated gap audit was accepted")

    for change in (
        lambda value: value.update({"fetch_schema": 1}),
        lambda value: value.update({"fetch_schema": 2.0}),
        lambda value: value.update({"gap_policy": "strict"}),
        lambda value: value["series"][0]["csv"]["gap_audit"].update(
            {"scope": "all-session-bins"}
        ),
        lambda value: value["series"][0]["csv"]["gap_audit"].update(
            {"affected_sessions": 0}
        ),
        lambda value: value["series"][0]["csv"]["gap_audit"].update(
            {"internal_gap_count": 0}
        ),
        lambda value: value["series"][0]["csv"]["gap_audit"].update(
            {"internal_gap_count": 1.0}
        ),
        lambda value: value["series"][0]["csv"]["gap_audit"].update(
            {"internal_missing_bins": 0}
        ),
        lambda value: value["series"][0]["csv"]["gap_audit"]["gaps"][0].update(
            {"absent_bins": 2}
        ),
        lambda value: value["series"][0]["csv"]["gap_audit"]["gaps"][0].update(
            {"absent_bins": 1.0}
        ),
    ):
        rejected(change)


def request_contract(
    ticker: str, manifest: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    reference = {
        "path": f"/v3/reference/tickers/{ticker}",
        "query": {"date": manifest["eligibility_date"]},
        "active": True,
        "market": "stocks",
        "locale": "us",
        "type": "CS",
        "currency_name": "usd",
    }
    aggregate = {
        "path": (
            f"/v2/aggs/ticker/{ticker}/range/"
            f"{manifest['interval_minutes']}/minute/"
            f"{manifest['start']}/{manifest['end']}"
        ),
        "query": {"adjusted": "true", "sort": "asc", "limit": "50000"},
    }
    return reference, aggregate


def report_record(
    model: str, series: str, seed: int | None, fold: int | None,
    targets_: Sequence[int], calibration: bool,
) -> dict[str, object]:
    value: dict[str, object] = {
        "model": model,
        "candidate": "raw-17",
        "series": series,
        "feature_set": "ohlcv",
        "fold": fold,
        "seed": seed,
        "targets": {
            "train": [timestamps()[17], timestamps()[18]],
            "validation": [
                timestamps()[targets_[0]], timestamps()[targets_[-1]],
            ],
            "test": [timestamps()[49], timestamps()[49]],
        },
        "samples": len(targets_),
        "validation_scaled_mse": 1.0,
        "metrics": metrics(),
    }
    if calibration:
        value["epochs"] = 5 if seed is not None else None
    elif seed is not None:
        value.update({
            "best_validation_scaled_mse": 1.0,
            "best_epoch": 5,
            "epochs_trained": 6,
        })
    return value


def seeds_for(model: str) -> tuple[int | None, ...]:
    return SEEDS if model in ("transformer", "mlp") else (None,)


class Fixture:
    def __init__(self, root: Path) -> None:
        self.run_dir = root / "run"
        self.cli_run_dir = Path(os.path.relpath(self.run_dir, ROOT))
        assert not self.cli_run_dir.is_absolute()
        self.csv_dir = self.run_dir / "csv"
        self.csv_dir.mkdir(parents=True)
        self.manifest_path = root / "manifest.json"
        self.config_path = root / "config.json"
        self.fetch_path = self.run_dir / "fetch-report.json"
        self.experiment_path = self.run_dir / "experiment.json"
        self.ledger_path = self.run_dir / "calibration.jsonl"
        self.output_dir = root / "outputs"
        self.output_dir.mkdir()
        self.manifest = manifest_value()
        self.config = config_value()
        write_json(self.manifest_path, self.manifest)
        write_json(self.config_path, self.config)
        self.names = tuple(
            item["ticker"] for item in self.manifest["series"]
        )
        self.csv_paths = {
            name: (self.csv_dir / f"{name.lower()}-30m.csv").resolve()
            for name in self.names
        }
        for stock, name in enumerate(self.names):
            write_csv(self.csv_paths[name], stock)
        self.bars = {name: load_bars(path)
                     for name, path in self.csv_paths.items()}
        self.actuals = {
            name: {
                timestamps()[target]: math.log(
                    self.bars[name].closes[target] /
                    self.bars[name].opens[target - 12]
                )
                for target in (TARGETS[1:] if stock == 0 else TARGETS)
            }
            for stock, name in enumerate(self.names)
        }
        self._write_fetch()
        self.experiment, records = self._experiment()
        self._write_ledger(records)
        self.experiment["calibration_prediction_ledger"] = {
            "schema": 3,
            "path": str(self.cli_run_dir / "calibration.jsonl"),
            "records": len(records),
            "sha256": file_sha256(self.ledger_path),
        }
        write_json(self.experiment_path, self.experiment)
        self._write_policies()
        for model in POLICY_MODELS:
            run((
                sys.executable, REPLAYER, self.manifest_path,
                self.cli_run_dir, model,
                self.cli_run_dir / f"backtest-{model}.json",
            ))

    def _write_fetch(self) -> None:
        records = []
        for item in self.manifest["series"]:
            name = item["ticker"]
            reference, aggregate = request_contract(name, self.manifest)
            records.append({
                "ticker": name,
                "stratum": item["stratum"],
                "reference": reference,
                "aggregate": aggregate,
                "csv": {
                    "path": str(self.csv_paths[name]),
                    "rows": 50,
                    "sessions": 50,
                    "source_rows": 50,
                    "sha256": file_sha256(self.csv_paths[name]),
                },
            })
        write_json(self.fetch_path, {
            **{field: self.manifest[field] for field in (
                "schema", "purpose", "declared_on", "eligibility_date",
                "start", "end", "interval_minutes", "adjusted", "session",
            )},
            "manifest": {
                "path": str(self.manifest_path),
                "sha256": file_sha256(self.manifest_path),
            },
            "series": records,
        })

    def _experiment(
        self,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        series = [
            {
                "name": name,
                "csv": str(self.csv_paths[name]),
                "rows": 50,
                "sha256": file_sha256(self.csv_paths[name]),
                "first_timestamp": timestamps()[0],
                "last_timestamp": timestamps()[-1],
            }
            for name in self.names
        ]
        calibration, fingerprints, ledger = [], [], []
        validation = []
        for name in self.names:
            for fold in range(2):
                target = (20 + fold,)
                for model in MODELS:
                    for seed in seeds_for(model):
                        validation.append(report_record(
                            model, name, seed, fold, target, False,
                        ))
        for model in MODELS:
            for stock, name in enumerate(self.names):
                target_indexes = TARGETS[1:] if stock == 0 else TARGETS
                for seed in seeds_for(model):
                    calibration.append(report_record(
                        model, name, seed, None, target_indexes, True,
                    ))
                    fingerprints.append({
                        "model": model,
                        "series": name,
                        "seed": seed,
                        "epochs": 5 if seed is not None else None,
                        "sha256": digest_text(f"{model}:{name}:{seed}"),
                    })
                    for target in target_indexes:
                        actual = self.actuals[name][timestamps()[target]]
                        if model == "transformer":
                            prediction = actual + {
                                7: -0.02, 19: -0.01, 31: 0.0,
                                43: 0.01, 61: 0.02,
                            }[seed]
                        elif model == "mlp":
                            prediction = actual + 0.02 + {
                                7: -0.004, 19: -0.002, 31: 0.0,
                                43: 0.002, 61: 0.004,
                            }[seed]
                        elif model == "linear":
                            prediction = actual + 0.025 + stock * 0.001
                        elif model == "rolling_mean":
                            prediction = -actual
                        else:
                            prediction = 0.0
                        ledger.append({
                            "schema": 3,
                            "split": "calibration",
                            "fold": None,
                            "series": name,
                            "model": model,
                            "candidate": "raw-17",
                            "feature_set": "ohlcv",
                            "seed": seed,
                            "csv_sha256": file_sha256(self.csv_paths[name]),
                            "as_of": timestamps()[target - 13],
                            "target_time": timestamps()[target],
                            "horizon_bars": 13,
                            "target_kind": "executable-return-v1",
                            "predicted_log_return": prediction,
                        })
        fingerprints.sort(key=lambda item: (
            item["model"], item["series"],
            -1 if item["seed"] is None else item["seed"],
        ))
        report = {
            "schema": 6,
            "protocol": json.loads(json.dumps(analysis.EXPECTED_PROTOCOL)),
            "runtime": {
                "device": "cpu", "python": "3.12.13", "torch": "synthetic",
            },
            "series": series,
            "test_contract": [{
                "series": name,
                "samples": 1,
                "first_target_time": timestamps()[49],
                "last_target_time": timestamps()[49],
            } for name in self.names],
            "sweep": self.config,
            "selection": {
                model: {
                    "candidate": "raw-17",
                    "mean_validation_scaled_mse": 1.0,
                }
                for model in MODELS
            },
            "validation": validation,
            "calibration": calibration,
            "model_fingerprints": fingerprints,
            "validation_summary": {},
            "test": [],
            "summary": empty_test_summary(),
            "sweep_input": {
                "path": str(self.config_path),
                "sha256": file_sha256(self.config_path),
            },
        }
        return report, ledger

    def _write_ledger(self, records: Sequence[Mapping[str, object]]) -> None:
        self.ledger_path.write_text(
            "".join(
                json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def _write_policies(self) -> None:
        forecasts = read_forecasts(self.ledger_path)
        experiment_hash = file_sha256(self.experiment_path)
        for model in POLICY_MODELS:
            disagreement = (
                (0.0, 0.5, 1.0)
                if model in ("transformer", "mlp") else (0.0,)
            )
            policy = select_policy(
                self.experiment, forecasts, self.bars, Costs(1, 1, 0),
                (0.0, 3.0, 6.0, 10.0), 100.0, model,
                self.cli_run_dir / "experiment.json", experiment_hash,
                self.cli_run_dir / "calibration.jsonl",
                file_sha256(self.ledger_path), len(forecasts),
                disagreement,
            )
            write_json(self.run_dir / f"policy-{model}.json", policy)

    def analyze(self, name: str, expected: int = 0) -> dict[str, object] | None:
        output = self.output_dir / name
        assert not output.exists()
        run((
            sys.executable, ANALYZER, self.manifest_path, self.config_path,
            self.cli_run_dir, output,
        ), expected)
        if expected not in (0, 3):
            assert not output.exists()
            return None
        return json.loads(output.read_text(encoding="utf-8"))

    def gate_failure(self, name: str) -> dict[str, object]:
        artifacts = (
            self.experiment_path, self.ledger_path,
            *(self.run_dir / f"policy-{model}.json"
              for model in POLICY_MODELS),
            *(self.run_dir / f"backtest-{model}.json"
              for model in POLICY_MODELS),
        )
        saved = {path: path.read_bytes() for path in artifacts}
        try:
            records = [
                json.loads(line)
                for line in self.ledger_path.read_text(
                    encoding="utf-8",
                ).splitlines()
            ]
            for record in records:
                if record["model"] == "transformer":
                    record["predicted_log_return"] = 0.0
            self._write_ledger(records)
            self.experiment = json.loads(
                self.experiment_path.read_text(encoding="utf-8")
            )
            metadata = self.experiment["calibration_prediction_ledger"]
            metadata["sha256"] = file_sha256(self.ledger_path)
            write_canonical_json(self.experiment_path, self.experiment)
            self._write_policies()
            for model in POLICY_MODELS:
                (self.run_dir / f"backtest-{model}.json").unlink()
                run((
                    sys.executable, REPLAYER, self.manifest_path,
                    self.cli_run_dir, model,
                    self.cli_run_dir / f"backtest-{model}.json",
                ))
            report = self.analyze(name, expected=3)
            assert report is not None
            return report
        finally:
            for path, content in saved.items():
                path.write_bytes(content)
            self.experiment = json.loads(
                self.experiment_path.read_text(encoding="utf-8")
            )

    def reject_cli(self, name: str) -> None:
        output = self.output_dir / name
        argv = [
            str(ANALYZER), str(self.manifest_path), str(self.config_path),
            str(self.cli_run_dir), str(output),
        ]
        with patch.object(sys, "argv", argv), \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            try:
                analysis.main()
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError("integrity mutation returned success")
        assert not output.exists()

    def reject_integrity(self) -> None:
        def frozen(source: Path, snapshot: Path | None = None) -> object:
            actual = source if snapshot is None else snapshot
            return analysis.FrozenInput(source, actual, file_sha256(actual))

        named = {
            "run_dir": analysis.FrozenInput(
                self.cli_run_dir, self.run_dir, "",
            ),
            "fetch": frozen(
                self.cli_run_dir / "fetch-report.json", self.fetch_path,
            ),
            "experiment": frozen(
                self.cli_run_dir / "experiment.json", self.experiment_path,
            ),
            "ledger": frozen(
                self.cli_run_dir / "calibration.jsonl", self.ledger_path,
            ),
            **{
                f"policy-{model}": frozen(
                    self.cli_run_dir / f"policy-{model}.json",
                    self.run_dir / f"policy-{model}.json",
                )
                for model in POLICY_MODELS
            },
            **{
                f"backtest-{model}": frozen(
                    self.cli_run_dir / f"backtest-{model}.json",
                    self.run_dir / f"backtest-{model}.json",
                )
                for model in POLICY_MODELS
            },
        }
        try:
            analysis.analyze(
                frozen(self.manifest_path), frozen(self.config_path), named,
                tuple(
                    frozen(path)
                    for path in sorted(self.csv_paths.values())
                ),
            )
        except (
            IndexError, KeyError, OSError, OverflowError, TypeError,
            UnicodeError, ValueError,
        ):
            return
        raise AssertionError("mutation crossed the analysis integrity boundary")


def mutate_json(path: Path, change: Callable[[dict[str, object]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    write_canonical_json(path, value)


def different(value: object) -> object:
    if isinstance(value, dict):
        return value | {"unexpected": True}
    if isinstance(value, str):
        return value + "-mutated"
    if type(value) is int:
        return value + 1
    if isinstance(value, float):
        return value + 0.01
    raise AssertionError(f"no mutation for {type(value).__name__}")


def rejected_mutation(
    fixture: Fixture, name: str, paths: Sequence[Path],
    mutate: Callable[[], None],
) -> None:
    saved = {path: path.read_bytes() for path in paths}
    try:
        mutate()
        if name == "manifest-bytes":
            fixture.reject_cli(f"invalid-{name}.json")
        else:
            fixture.reject_integrity()
    finally:
        for path, content in saved.items():
            path.write_bytes(content)


def refresh_provenance(fixture: Fixture) -> None:
    ledger_hash = file_sha256(fixture.ledger_path)
    experiment = json.loads(fixture.experiment_path.read_text(encoding="utf-8"))
    experiment["calibration_prediction_ledger"]["sha256"] = ledger_hash
    experiment["calibration_prediction_ledger"]["records"] = len(
        fixture.ledger_path.read_text(encoding="utf-8").splitlines()
    )
    write_canonical_json(fixture.experiment_path, experiment)
    experiment_hash = file_sha256(fixture.experiment_path)
    fingerprint = experiment_fingerprint(experiment)
    for model in POLICY_MODELS:
        policy_path = fixture.run_dir / f"policy-{model}.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy["calibration_report"]["sha256"] = experiment_hash
        policy["calibration_prediction_ledger"]["sha256"] = ledger_hash
        policy["calibration_fingerprint"] = fingerprint
        write_canonical_json(policy_path, policy)
        replay_path = fixture.run_dir / f"backtest-{model}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        replay["prediction_ledger"]["sha256"] = ledger_hash
        replay["experiment_report"]["sha256"] = experiment_hash
        replay["policy"]["sha256"] = file_sha256(policy_path)
        write_canonical_json(replay_path, replay)


def prediction_timestamp_mutation(fixture: Fixture) -> None:
    records = [
        json.loads(line)
        for line in fixture.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["target_time"] = timestamps()[29]
    fixture._write_ledger(records)
    refresh_provenance(fixture)


def actual_timestamp_mutation(fixture: Fixture) -> None:
    name = fixture.names[0]
    csv_path = fixture.csv_paths[name]
    lines = csv_path.read_text(encoding="ascii").splitlines()
    lines[31] = lines[31].replace("T14:30:00Z", "T14:31:00Z", 1)
    csv_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    checksum = file_sha256(csv_path)

    fetch = json.loads(fixture.fetch_path.read_text(encoding="utf-8"))
    fetch["series"][0]["csv"]["sha256"] = checksum
    write_canonical_json(fixture.fetch_path, fetch)
    experiment = json.loads(fixture.experiment_path.read_text(encoding="utf-8"))
    experiment["series"][0]["sha256"] = checksum
    write_canonical_json(fixture.experiment_path, experiment)
    records = [
        json.loads(line)
        for line in fixture.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        if record["series"] == name:
            record["csv_sha256"] = checksum
    fixture._write_ledger(records)
    for model in POLICY_MODELS:
        replay_path = fixture.run_dir / f"backtest-{model}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        replay["series"][0]["sha256"] = checksum
        write_canonical_json(replay_path, replay)
    refresh_provenance(fixture)


def reference_bootstrap(
    values: Mapping[str, Mapping[str, Sequence[float]]],
) -> list[float]:
    dates = sorted(next(iter(values.values())))
    generator = random.Random(20260723)
    prepared = [
        tuple(
            (sum(by_date[date]), len(by_date[date]))
            for date in dates
        )
        for by_date in values.values()
    ]
    samples = []
    for _ in range(10_000):
        selected: list[int] = []
        while len(selected) < len(dates):
            start = generator.randrange(len(dates))
            selected.extend(
                (start + offset) % len(dates) for offset in range(5)
            )
        selected = selected[:len(dates)]
        stock_means = [
            sum(stock[index][0] for index in selected) /
            sum(stock[index][1] for index in selected)
            for stock in prepared
        ]
        samples.append(fmean(stock_means))
    samples.sort()
    return [samples[int(0.025 * 9_999)], samples[int(0.975 * 9_999)]]


def test_helpers() -> None:
    blocks = {
        "A": {"2026-01-01": (1.0,), "2026-01-02": (3.0,),
              "2026-01-03": (2.0,)},
        "B": {"2026-01-01": (-2.0,), "2026-01-02": (4.0,),
              "2026-01-03": (1.0,)},
    }
    assert analysis.date_block_bootstrap(blocks) == reference_bootstrap(blocks)
    assert analysis.date_block_bootstrap(blocks) == \
        analysis.date_block_bootstrap(blocks)

    fewer_dates = analysis.effective_count({
        "A": {"d1": 0.0}, "B": {"d1": 1.0},
    })
    assert fewer_dates["value"] is None
    assert fewer_dates["reason"] == "fewer-than-two-aligned-dates"
    fewer_stocks = analysis.effective_count({
        "A": {"d1": -1.0, "d2": 1.0},
        "B": {"d1": 0.0, "d2": 0.0},
    })
    assert fewer_stocks["value"] is None
    assert fewer_stocks["excluded"] == ["B"]
    assert fewer_stocks["reason"] == "fewer-than-two-nonconstant-stocks"
    denominator = analysis.effective_count({
        "A": {"d1": -1.0, "d2": 1.0},
        "B": {"d1": 1.0, "d2": -1.0},
    })
    assert denominator["value"] is None
    assert denominator["reason"] == "nonpositive-or-nonfinite-denominator"
    hand = analysis.effective_count({
        "A": {"d1": 0.0, "d2": 1.0, "d3": 2.0},
        "B": {"d1": 0.0, "d2": 2.0, "d3": 4.0},
        "C": {"d1": 1.0, "d2": 1.0, "d3": 1.0},
    })
    assert math.isclose(hand["value"], 10 / 9)
    assert hand["included"] == ["A", "B"]
    assert hand["excluded"] == ["C"]
    assert hand["reason"] is None
    diversified = analysis.effective_count({
        "A": {"d1": -1.0, "d2": 0.0, "d3": 1.0, "d4": 0.0},
        "B": {"d1": 1.0, "d2": 0.0, "d3": -0.5, "d4": 0.0},
    })
    assert math.isclose(diversified["value"], 34.0)
    assert diversified["value"] > len(diversified["included"])

    strict = analysis.evaluate_gates(
        {
            "transformer": {"return_mae": 1.0, "direction": 0.5},
            "mlp": {"return_mae": 1.0, "direction": 0.0},
            "linear": {"return_mae": 2.0, "direction": 0.0},
            "rolling_mean": {"return_mae": 2.0, "direction": 0.0},
            "last_close": {"return_mae": 2.0, "direction": 0.0},
        },
        0.5, 0.0,
    )
    assert strict["return_mae"]["pass"] is False
    assert strict["direction"]["pass"] is False
    assert strict["close_mae"]["pass"] is False
    assert strict["all_pass"] is False


def test_valid_reports(root: Path) -> Fixture:
    passing = Fixture(root / "passing")
    report = passing.analyze("pass.json")
    assert report is not None and report["status"] == "pass"
    assert report["gates"]["all_pass"] is True
    assert report["protocol"]["seed_ensemble"] == \
        "arithmetic mean before stock/timestamp pairing"
    assert report["protocol"]["macro_unit"] == "stock"
    assert "temporal" not in json.dumps(report["n_eff"], sort_keys=True)
    assert all(
        Path(item["path"]).is_absolute()
        for item in report["inputs"]["csv"]
    )
    assert report["inputs"]["experiment"]["path"] == str(
        passing.cli_run_dir / "experiment.json"
    )

    transformer = report["forecast"]["per_model"]["transformer"]
    first = report["forecast"]["per_stock"][passing.names[0]]
    assert math.isclose(first["models"]["transformer"]["return_mae"], 0.0,
                        rel_tol=0.0, abs_tol=1e-15)
    assert first["majority"] == {
        "p_up": 0.5, "p_down": 0.5, "p_flat": 0.0, "direction": 0.5,
    }
    assert report["forecast"]["per_stock"][passing.names[1]]["majority"][
        "direction"
    ] == 2 / 3
    linear_stock = [
        value["models"]["linear"]["return_mae"]
        for value in report["forecast"]["per_stock"].values()
    ]
    assert math.isclose(
        report["forecast"]["per_model"]["linear"]["return_mae"],
        fmean(linear_stock),
    )
    pooled = sum(
        value["models"]["linear"]["return_mae"] *
        value["samples"]
        for value in report["forecast"]["per_stock"].values()
    ) / sum(value["samples"] for value in report["forecast"]["per_stock"].values())
    assert not math.isclose(fmean(linear_stock), pooled)
    assert transformer["return_mae"] < \
        report["forecast"]["per_model"]["last_close"]["return_mae"]
    assert set(report["policy_resubstitution"]["models"]) == set(POLICY_MODELS)
    for model in POLICY_MODELS:
        value = report["policy_resubstitution"]["models"][model]
        assert set(value["terminal_equity"]) == {
            "forecast_long_cash", "cash", "buy_and_hold", "always_up",
        }
        assert set(value["aggregates"]) == set(value["terminal_equity"])
        assert set(value["forecast_excess_mean_log_growth"]) == {
            "cash", "buy_and_hold", "always_up",
        }

    failed = passing.gate_failure("gate-failure.json")
    assert failed["status"] == "gate-failure"
    assert failed["gates"]["all_pass"] is False
    assert set(failed["policy_resubstitution"]["models"]) == set(POLICY_MODELS)
    assert "value" in failed["n_eff"]
    return passing


def test_output_freshness(fixture: Fixture) -> None:
    commands = {
        "analyzer": (
            sys.executable, ANALYZER, fixture.manifest_path,
            fixture.config_path, fixture.cli_run_dir,
        ),
        "replayer": (
            sys.executable, REPLAYER, fixture.manifest_path,
            fixture.cli_run_dir, "transformer",
        ),
    }
    for tool, command in commands.items():
        for kind in ("existing", "broken"):
            target = fixture.output_dir / f"{tool}-{kind}.json"
            marker = b"must not be overwritten\n"
            if kind == "existing":
                target.write_bytes(marker)
            else:
                target.symlink_to(f"missing-{tool}.json")
            missing_parent = fixture.output_dir / f"missing-{tool}-{kind}"
            assert not missing_parent.exists()
            alias = missing_parent / ".." / target.name
            run((*command, alias), expected=2)
            if kind == "existing":
                assert target.read_bytes() == marker
            else:
                assert target.is_symlink()
                assert os.readlink(target) == f"missing-{tool}.json"


def test_late_membership(fixture: Fixture) -> None:
    analyzer_output = fixture.output_dir / "late-analyzer.json"
    analyzer_late = fixture.run_dir / "late-artifact.json"
    analyzer_verify = analysis.verify_frozen

    def change_after_analyzer_hashes(frozen: object) -> None:
        analyzer_verify(frozen)
        analyzer_late.write_text("{}\n", encoding="utf-8")

    try:
        with patch.object(
            analysis, "analyze", return_value=({"status": "pass"}, True),
        ), patch.object(
            analysis, "verify_frozen",
            side_effect=change_after_analyzer_hashes,
        ):
            run_main(analysis.main, (
                ANALYZER, fixture.manifest_path, fixture.config_path,
                fixture.cli_run_dir, analyzer_output,
            ), expected=2)
        assert not analyzer_output.exists()
    finally:
        analyzer_late.unlink(missing_ok=True)

    replay_output = fixture.output_dir / "late-replay.json"
    replay_late = fixture.csv_dir / "late.csv"
    replay_verify = replay_tool.verify_frozen

    def change_after_replay_hashes(frozen: object) -> None:
        replay_verify(frozen)
        replay_late.write_text("late\n", encoding="ascii")

    try:
        with patch.object(
            replay_tool, "replay", return_value={"results": []},
        ), patch.object(
            replay_tool, "verify_frozen",
            side_effect=change_after_replay_hashes,
        ):
            run_main(replay_tool.main, (
                REPLAYER, fixture.manifest_path, fixture.cli_run_dir,
                "transformer", replay_output,
            ), expected=2)
        assert not replay_output.exists()
    finally:
        replay_late.unlink(missing_ok=True)


def test_hardlink_integrity(fixture: Fixture) -> None:
    if not hardlinks_supported(fixture.output_dir):
        print("hardlink integrity tests skipped: unsupported filesystem")
        return

    config_bytes = fixture.config_path.read_bytes()
    fixture.config_path.unlink()
    os.link(fixture.manifest_path, fixture.config_path)
    try:
        fixture.analyze("hardlink-analyzer.json", expected=2)
    finally:
        fixture.config_path.unlink()
        fixture.config_path.write_bytes(config_bytes)

    ledger_bytes = fixture.ledger_path.read_bytes()
    fixture.ledger_path.unlink()
    os.link(fixture.experiment_path, fixture.ledger_path)
    replay_output = fixture.output_dir / "hardlink-replayer.json"
    try:
        run((
            sys.executable, REPLAYER, fixture.manifest_path,
            fixture.cli_run_dir, "transformer", replay_output,
        ), expected=2)
        assert not replay_output.exists()
    finally:
        fixture.ledger_path.unlink()
        fixture.ledger_path.write_bytes(ledger_bytes)

    policy_path = fixture.run_dir / "policy-transformer.json"
    policy_bytes = policy_path.read_bytes()
    external = fixture.output_dir / "late-hardlink-source.json"
    external.write_bytes(policy_bytes)
    output = fixture.output_dir / "late-hardlink.json"
    original_verify = analysis.verify_frozen

    def substitute_after_hashes(frozen: object) -> None:
        original_verify(frozen)
        policy_path.unlink()
        os.link(external, policy_path)

    try:
        with patch.object(
            analysis, "analyze", return_value=({"status": "pass"}, True),
        ), patch.object(
            analysis, "verify_frozen", side_effect=substitute_after_hashes,
        ):
            run_main(analysis.main, (
                ANALYZER, fixture.manifest_path, fixture.config_path,
                fixture.cli_run_dir, output,
            ), expected=2)
        assert not output.exists()
    finally:
        policy_path.unlink(missing_ok=True)
        policy_path.write_bytes(policy_bytes)
        external.unlink(missing_ok=True)


def test_mutations(fixture: Fixture) -> None:
    rejected_mutation(
        fixture, "manifest-bytes", (fixture.manifest_path,),
        lambda: fixture.manifest_path.write_bytes(
            fixture.manifest_path.read_bytes() + b"\n"
        ),
    )
    rejected_mutation(
        fixture, "config-bytes", (fixture.config_path,),
        lambda: fixture.config_path.write_bytes(
            fixture.config_path.read_bytes() + b"\n"
        ),
    )
    rejected_mutation(
        fixture, "fetch-bytes", (fixture.fetch_path,),
        lambda: fixture.fetch_path.write_bytes(
            fixture.fetch_path.read_bytes() + b"\n"
        ),
    )
    csv_path = fixture.csv_paths[fixture.names[0]]
    rejected_mutation(
        fixture, "csv-bytes", (csv_path,),
        lambda: csv_path.write_bytes(csv_path.read_bytes() + b"\n"),
    )
    rejected_mutation(
        fixture, "experiment-bytes", (fixture.experiment_path,),
        lambda: fixture.experiment_path.write_bytes(
            fixture.experiment_path.read_bytes() + b"\n"
        ),
    )
    rejected_mutation(
        fixture, "ledger-bytes", (fixture.ledger_path,),
        lambda: fixture.ledger_path.write_bytes(
            fixture.ledger_path.read_bytes() + b"\n"
        ),
    )
    for model in POLICY_MODELS:
        path = fixture.run_dir / f"policy-{model}.json"
        rejected_mutation(
            fixture, f"policy-{model}-bytes", (path,),
            lambda path=path: path.write_bytes(path.read_bytes() + b"\n"),
        )
    for model in POLICY_MODELS:
        path = fixture.run_dir / f"backtest-{model}.json"
        rejected_mutation(
            fixture, f"replay-{model}-bytes", (path,),
            lambda path=path: path.write_bytes(path.read_bytes() + b"\n"),
        )

    policy_path = fixture.run_dir / "policy-transformer.json"
    replay_path = fixture.run_dir / "backtest-transformer.json"
    experiment_path = fixture.experiment_path
    ledger_path = fixture.ledger_path
    cases: tuple[tuple[str, Path, Callable[[dict[str, object]], None]], ...] = (
        ("fetch-manifest-hash", fixture.fetch_path,
         lambda value: value["manifest"].update({"sha256": "0" * 64})),
        ("experiment-config-hash", experiment_path,
         lambda value: value["sweep_input"].update({"sha256": "0" * 64})),
        ("experiment-ledger-hash", experiment_path,
         lambda value: value["calibration_prediction_ledger"].update(
             {"sha256": "0" * 64}
         )),
        ("policy-order", policy_path,
         lambda value: value["series"].reverse()),
        ("policy-test-grid", policy_path,
         lambda value: value["test_grid"].reverse()),
        ("policy-fingerprints", policy_path,
         lambda value: value["model_fingerprints"].reverse()),
        ("policy-calibration-fingerprint", policy_path,
         lambda value: value.update({"calibration_fingerprint": "0" * 64})),
        ("replay-policy-provenance", replay_path,
         lambda value: value["policy"].update({"sha256": "0" * 64})),
        ("replay-experiment-provenance", replay_path,
         lambda value: value["experiment_report"].update(
             {"sha256": "0" * 64}
         )),
        ("replay-result-order", replay_path,
         lambda value: value["results"].reverse()),
    )
    for name, path, change in cases:
        rejected_mutation(
            fixture, name, (path,),
            lambda path=path, change=change: mutate_json(path, change),
        )
    for field, expected in analysis.EXPECTED_PROTOCOL.items():
        rejected_mutation(
            fixture, f"protocol-{field}", (experiment_path,),
            lambda field=field, expected=expected: mutate_json(
                experiment_path,
                lambda value: value["protocol"].update({
                    field: different(expected),
                }),
            ),
        )
    rejected_mutation(
        fixture, "protocol-extra", (experiment_path,),
        lambda: mutate_json(
            experiment_path,
            lambda value: value["protocol"].update({"unexpected": True}),
        ),
    )
    calibration_cases = (
        ("calibration-fold",
         lambda value: value["calibration"][0].update({"fold": 0})),
        ("calibration-neural-epochs",
         lambda value: value["calibration"][0].update({"epochs": None})),
        ("calibration-deterministic-epochs",
         lambda value: next(
             item for item in value["calibration"]
             if item["seed"] is None
         ).update({"epochs": 1})),
    )
    for name, change in calibration_cases:
        rejected_mutation(
            fixture, name, (experiment_path,),
            lambda change=change: mutate_json(experiment_path, change),
        )

    cascading = (
        fixture.fetch_path, fixture.experiment_path, fixture.ledger_path,
        *(fixture.run_dir / f"policy-{model}.json" for model in POLICY_MODELS),
        *(fixture.run_dir / f"backtest-{model}.json" for model in POLICY_MODELS),
    )
    rejected_mutation(
        fixture, "prediction-timestamp", cascading,
        lambda: prediction_timestamp_mutation(fixture),
    )
    rejected_mutation(
        fixture, "actual-timestamp", (
            fixture.csv_paths[fixture.names[0]], *cascading,
        ),
        lambda: actual_timestamp_mutation(fixture),
    )


def main() -> None:
    test_helpers()
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-universe-analysis-",
    ) as directory:
        test_gap_audit_contract(Path(directory))
        fixture = test_valid_reports(Path(directory))
        test_output_freshness(fixture)
        test_late_membership(fixture)
        test_hardlink_integrity(fixture)
        test_mutations(fixture)
    print("universe analysis tests passed")


if __name__ == "__main__":
    main()
