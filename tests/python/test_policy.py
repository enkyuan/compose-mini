#!/usr/bin/env python3
"""Verify calibration policy selection and frozen model-state contracts."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtest import Costs, Forecast, load_bars, validate_policy
from tools.data_v1 import EXECUTABLE_RETURN_TARGET
from tools.select_policy import _read_report, select_policy


def write_csv(path: Path, exit_close: float) -> tuple[str, ...]:
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    timestamps = tuple(
        (start + timedelta(minutes=index)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for index in range(5)
    )
    closes = (100.0, exit_close, 100.0, exit_close, 100.0)
    lines = ["timestamp,open,high,low,close,volume"]
    for timestamp, close in zip(timestamps, closes, strict=True):
        lines.append(f"{timestamp},100,120,80,{close},1000")
    path.write_text("\n".join(lines), encoding="ascii")
    return timestamps


def fixture(path: Path, exit_close: float
            ) -> tuple[object, tuple[Forecast, ...], dict[str, object]]:
    timestamps = write_csv(path, exit_close)
    bars = load_bars(path)
    forecasts = tuple(
        Forecast(
            "TEST", "transformer", "raw", "ohlcv", seed, bars.sha256,
            timestamps[0], timestamps[1], 1, prediction,
            "calibration", None, EXECUTABLE_RETURN_TARGET,
        )
        for seed, prediction in ((3, 0.0004), (7, 0.0006))
    )
    records = [
        {
            "model": "transformer", "candidate": "raw", "series": "TEST",
            "feature_set": "ohlcv", "fold": None, "seed": seed,
            "samples": 1,
            "targets": {"validation": [timestamps[1]] * 2},
        }
        for seed in (3, 7)
    ]
    fingerprints = [
        {
            "model": "transformer", "series": "TEST",
            "seed": 3, "epochs": 4, "sha256": "3" * 64,
        },
        {
            "model": "transformer", "series": "TEST",
            "seed": 7, "epochs": 6, "sha256": "4" * 64,
        },
    ]
    report = {
        "schema": 6,
        "protocol": {
            "phase": "selection-and-calibration",
            "target_kind": EXECUTABLE_RETURN_TARGET,
            "target_horizon_bars": 1,
        },
        "selection": {"transformer": {"candidate": "raw"}},
        "sweep": {
            "candidates": [{"name": "raw", "feature_set": "ohlcv"}],
            "seeds": [3, 7], "folds": 2,
        },
        "series": [{"name": "TEST"}], "validation": [],
        "calibration": records, "model_fingerprints": fingerprints, "test": [],
        "test_contract": [{
            "series": "TEST", "samples": 1,
            "first_target_time": timestamps[4],
            "last_target_time": timestamps[4],
        }],
    }
    return bars, forecasts, report


def choose(path: Path, exit_close: float) -> dict[str, object]:
    bars, forecasts, report = fixture(path, exit_close)
    return select_policy(
        report, forecasts, {"TEST": bars}, Costs(0, 0, 0), (0.0, 10.0),
        100.0, "transformer", Path("calibration.json"), "0" * 64,
        Path("calibration.jsonl"), "1" * 64, len(forecasts),
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="compose-mini-policy-") as directory:
        root = Path(directory)
        profitable = choose(root / "profit.csv", 110.0)
        assert profitable["action"] == "long_above"
        assert profitable["safety_bps"] == 0.0
        assert profitable["seeds"] == [3, 7]
        assert len(profitable["threshold_trials"]) == 3
        assert profitable["model_fingerprints"] == [
            {
                "model": "transformer", "series": "TEST",
                "seed": 3, "epochs": 4, "sha256": "3" * 64,
            },
            {
                "model": "transformer", "series": "TEST",
                "seed": 7, "epochs": 6, "sha256": "4" * 64,
            },
        ]
        assert profitable["calibration_report"] == {
            "path": "calibration.json", "sha256": "0" * 64,
        }
        assert profitable["calibration_prediction_ledger"] == {
            "path": "calibration.jsonl", "sha256": "1" * 64,
            "source_records": 2, "selected_records": 2,
        }
        assert choose(root / "loss.csv", 90.0)["action"] == "cash"

        bars, forecasts, report = fixture(root / "incomplete.csv", 110.0)
        try:
            select_policy(
                report, forecasts[:-1], {"TEST": bars}, Costs(0, 0, 0),
                (0.0,), 100.0, "transformer", Path("calibration.json"),
                "0" * 64, Path("calibration.jsonl"), "1" * 64,
                len(forecasts) - 1,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete policy grid was accepted")

        missing_seed = report | {
            "calibration": report["calibration"][:1],
        }
        try:
            select_policy(
                missing_seed, forecasts[:1], {"TEST": bars}, Costs(0, 0, 0),
                (0.0,), 100.0, "transformer", Path("calibration.json"),
                "0" * 64, Path("calibration.jsonl"), "1" * 64, 1,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("missing calibration seed was accepted")

        null_forecasts = tuple(replace(item, seed=None)
                               for item in forecasts[::2])
        null_report = report | {
            "calibration": [record | {"seed": None}
                            for record in report["calibration"][::2]],
        }
        try:
            select_policy(
                null_report, null_forecasts, {"TEST": bars}, Costs(0, 0, 0),
                (0.0,), 100.0, "transformer", Path("calibration.json"),
                "0" * 64, Path("calibration.jsonl"), "1" * 64, 1,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("neural seed grid was treated as deterministic")

        try:
            validate_policy(profitable | {
                "minimum_predicted_log_return": 1.0,
            })
        except ValueError:
            pass
        else:
            raise AssertionError("inconsistent policy threshold was accepted")

        deterministic = profitable | {
            "model": "last_close", "seeds": [],
            "model_fingerprints": [{
                "model": "last_close", "series": "TEST",
                "seed": None, "epochs": None, "sha256": "5" * 64,
            }],
            "calibration_prediction_ledger":
                profitable["calibration_prediction_ledger"] | {
                    "source_records": 1, "selected_records": 1,
                },
        }
        assert validate_policy(deterministic) == deterministic

        fingerprints = profitable["model_fingerprints"]
        for mutation in (
            {"action": "cash", "safety_bps": None,
             "minimum_predicted_log_return": None},
            {"threshold_trials": [{"garbage": True}]},
            {"model_fingerprints": [*fingerprints, fingerprints[0]]},
            {"model_fingerprints": fingerprints[:1]},
            {"model_fingerprints": list(reversed(fingerprints))},
            {"model_fingerprints": [
                fingerprints[0] | {"sha256": "A" * 64}, fingerprints[1],
            ]},
            {"calibration_prediction_ledger": {
                "path": "calibration.jsonl", "sha256": "1" * 64,
                "source_records": 2, "selected_records": 1,
            }},
            {"calibration_fingerprint": int("1" * 64)},
            {"calibration_report": {
                "path": "calibration.json", "sha256": int("1" * 64),
            }},
        ):
            try:
                validate_policy(profitable | mutation)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid frozen policy was accepted")

        invalid_report = root / "invalid-report.json"
        invalid_report.write_text(json.dumps(report | {
            "protocol": report["protocol"] | {"phase": "selection"},
        }),
                                  encoding="utf-8")
        try:
            _read_report(invalid_report)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid calibration report phase was accepted")
        for invalid in (
            {key: value for key, value in report.items() if key != "test"},
            report | {"test": None},
            report | {"test": {}},
            report | {"test": ""},
            report | {"test": 0},
        ):
            invalid_report.write_text(json.dumps(invalid), encoding="utf-8")
            try:
                _read_report(invalid_report)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid calibration report test was accepted")
    print("policy selection tests passed")


if __name__ == "__main__":
    main()
