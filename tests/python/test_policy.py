#!/usr/bin/env python3
"""Verify validation-only policy selection and frozen ensemble contracts."""

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
            timestamps[fold * 2], timestamps[fold * 2 + 1], 1, prediction,
            "validation", fold, EXECUTABLE_RETURN_TARGET,
        )
        for fold in (0, 1)
        for seed, prediction in ((3, 0.0004), (7, 0.0006))
    )
    records = [
        {
            "model": "transformer", "candidate": "raw", "series": "TEST",
            "feature_set": "ohlcv", "fold": fold, "seed": seed,
            "samples": 1,
            "targets": {"validation": [timestamps[fold * 2 + 1]] * 2},
        }
        for fold in (0, 1) for seed in (3, 7)
    ]
    report = {
        "schema": 5,
        "protocol": {
            "phase": "validation", "target_kind": EXECUTABLE_RETURN_TARGET,
            "target_horizon_bars": 1,
        },
        "selection": {"transformer": {"candidate": "raw"}},
        "sweep": {
            "candidates": [{"name": "raw", "feature_set": "ohlcv"}],
            "seeds": [3, 7], "folds": 2,
        },
        "series": [{"name": "TEST"}], "validation": records, "test": [],
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
        100.0, "transformer", Path("validation.json"), "0" * 64,
        Path("validation.jsonl"), "1" * 64, len(forecasts),
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="compose-mini-policy-") as directory:
        root = Path(directory)
        profitable = choose(root / "profit.csv", 110.0)
        assert profitable["action"] == "long_above"
        assert profitable["safety_bps"] == 0.0
        assert profitable["seeds"] == [3, 7]
        assert len(profitable["threshold_trials"]) == 3
        assert choose(root / "loss.csv", 90.0)["action"] == "cash"

        bars, forecasts, report = fixture(root / "incomplete.csv", 110.0)
        try:
            select_policy(
                report, forecasts[:-1], {"TEST": bars}, Costs(0, 0, 0),
                (0.0,), 100.0, "transformer", Path("validation.json"),
                "0" * 64, Path("validation.jsonl"), "1" * 64,
                len(forecasts) - 1,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete policy grid was accepted")

        missing_fold = report | {
            "validation": [record for record in report["validation"]
                           if record["fold"] == 0],
        }
        try:
            select_policy(
                missing_fold, forecasts[:2], {"TEST": bars}, Costs(0, 0, 0),
                (0.0,), 100.0, "transformer", Path("validation.json"),
                "0" * 64, Path("validation.jsonl"), "1" * 64, 2,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("missing validation fold was accepted")

        null_forecasts = tuple(replace(item, seed=None)
                               for item in forecasts[::2])
        null_report = report | {
            "validation": [record | {"seed": None}
                           for record in report["validation"][::2]],
        }
        try:
            select_policy(
                null_report, null_forecasts, {"TEST": bars}, Costs(0, 0, 0),
                (0.0,), 100.0, "transformer", Path("validation.json"),
                "0" * 64, Path("validation.jsonl"), "1" * 64, 2,
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

        for mutation in (
            {"action": "cash", "safety_bps": None,
             "minimum_predicted_log_return": None},
            {"threshold_trials": [{"garbage": True}]},
            {"validation_prediction_ledger": {
                "path": "validation.jsonl", "sha256": "1" * 64,
            }},
            {"validation_fingerprint": int("1" * 64)},
            {"validation_report": {
                "path": "validation.json", "sha256": int("1" * 64),
            }},
        ):
            try:
                validate_policy(profitable | mutation)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid frozen policy was accepted")

        invalid_report = root / "invalid-report.json"
        invalid_report.write_text(json.dumps(report | {"schema": 5.0}),
                                  encoding="utf-8")
        try:
            _read_report(invalid_report)
        except ValueError:
            pass
        else:
            raise AssertionError("noninteger experiment schema was accepted")
    print("policy selection tests passed")


if __name__ == "__main__":
    main()
