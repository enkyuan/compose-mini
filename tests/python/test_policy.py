#!/usr/bin/env python3
"""Verify calibration policy selection and frozen model-state contracts."""

from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import StringIO
from pathlib import Path
import json
import math
import tempfile
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtest import (
    DETERMINISTIC_DISAGREEMENT_GRID, POLICY_SAFETY_GRID,
    SEEDED_DISAGREEMENT_GRID, Costs, Forecast, load_bars,
    policy_disagreement_lambda, select_trial, validate_policy,
)
from tools.data_v1 import EXECUTABLE_RETURN_TARGET
from tools.select_policy import (
    _read_report, _validate_ledger_metadata, main as select_main, select_policy,
)

POLICY_V2_FIELDS = {
    "schema", "action", "model", "candidate", "feature_set", "target_kind",
    "horizon_bars", "seeds", "series", "initial_cash", "costs", "safety_bps",
    "minimum_predicted_log_return", "selection_objective",
    "calibration_report", "calibration_prediction_ledger",
    "model_fingerprints", "threshold_trials", "test_grid",
    "calibration_fingerprint",
}
TRIAL_V2_FIELDS = {
    "action", "safety_bps", "objective", "mean_final_equity",
    "mean_gross_turnover", "signal_coverage", "execution_coverage",
    "trade_count",
}


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


def choose(path: Path, exit_close: float,
           disagreement_values: tuple[float, ...] | None = None
           ) -> dict[str, object]:
    bars, forecasts, report = fixture(path, exit_close)
    options = {} if disagreement_values is None else {
        "disagreement_values": disagreement_values,
    }
    return select_policy(
        report, forecasts, {"TEST": bars}, Costs(0, 0, 0),
        POLICY_SAFETY_GRID if disagreement_values is not None else (0.0, 10.0),
        100.0, "transformer", Path("calibration.json"), "0" * 64,
        Path("calibration.jsonl"), "1" * 64, len(forecasts),
        **options,
    )


def deterministic_fixture(path: Path
                          ) -> tuple[object, tuple[Forecast, ...],
                                     dict[str, object]]:
    bars, forecasts, report = fixture(path, 110.0)
    forecast = replace(forecasts[0], model="last_close", seed=None)
    record = report["calibration"][0] | {
        "model": "last_close", "seed": None,
    }
    fingerprint = {
        "model": "last_close", "series": "TEST", "seed": None,
        "epochs": None, "sha256": "5" * 64,
    }
    report = report | {
        "selection": {"last_close": {"candidate": "raw"}},
        "calibration": [record], "model_fingerprints": [fingerprint],
    }
    return bars, (forecast,), report


def abstention_fixture(path: Path
                       ) -> tuple[object, tuple[Forecast, ...],
                                  dict[str, object]]:
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    timestamps = tuple(
        (start + timedelta(minutes=index)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for index in range(3)
    )
    path.write_text(
        "timestamp,open,high,low,close,volume\n" +
        "\n".join(
            f"{timestamp},100,120,80,{close},1000"
            for timestamp, close in zip(
                timestamps, (100.0, 110.0, 90.0), strict=True,
            )
        ),
        encoding="ascii",
    )
    bars = load_bars(path)
    predictions = ((0.0002, 0.0002), (-0.0001, 0.0005))
    forecasts = tuple(
        Forecast(
            "TEST", "transformer", "raw", "ohlcv", seed, bars.sha256,
            timestamps[index], timestamps[index + 1], 1, prediction,
            "calibration", None, EXECUTABLE_RETURN_TARGET,
        )
        for seed, values in zip((3, 7), zip(*predictions), strict=True)
        for index, prediction in enumerate(values)
    )
    records = [
        {
            "model": "transformer", "candidate": "raw", "series": "TEST",
            "feature_set": "ohlcv", "fold": None, "seed": seed, "samples": 2,
            "targets": {"validation": [timestamps[1], timestamps[2]]},
        }
        for seed in (3, 7)
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
        "calibration": records,
        "model_fingerprints": [
            {
                "model": "transformer", "series": "TEST", "seed": seed,
                "epochs": 4, "sha256": str(seed) * 64,
            }
            for seed in (3, 7)
        ],
        "test": [],
        "test_contract": [{
            "series": "TEST", "samples": 1,
            "first_target_time": timestamps[2],
            "last_target_time": timestamps[2],
        }],
    }
    return bars, forecasts, report


def write_forecasts(path: Path, forecasts: tuple[Forecast, ...]) -> str:
    path.write_text("".join(json.dumps({
        "schema": 3, "split": item.split, "fold": item.fold,
        "series": item.series, "model": item.model,
        "candidate": item.candidate, "feature_set": item.feature_set,
        "seed": item.seed, "csv_sha256": item.csv_sha256,
        "as_of": item.as_of, "target_time": item.target_time,
        "horizon_bars": item.horizon_bars, "target_kind": item.target_kind,
        "predicted_log_return": item.predicted_log_return,
    }) + "\n" for item in forecasts), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def run_cli(root: Path, disagreement: tuple[str, ...] = ()
            ) -> tuple[dict[str, object], dict[str, object]]:
    bars, forecasts, report = fixture(root / f"cli-{len(disagreement)}.csv",
                                      110.0)
    ledger = root / f"cli-{len(disagreement)}.jsonl"
    checksum = write_forecasts(ledger, forecasts)
    report = report | {"calibration_prediction_ledger": {
        "schema": 3, "path": str(ledger), "records": len(forecasts),
        "sha256": checksum,
    }}
    report_path = root / f"cli-{len(disagreement)}-report.json"
    policy_path = root / f"cli-{len(disagreement)}-policy.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    argv = [
        "select_policy.py", str(report_path), str(ledger), str(policy_path),
        f"TEST={bars.path}", "--model", "transformer", "--safety-bps",
        *(("0", "3", "6", "10") if disagreement else ("0", "10")),
        "--initial-cash", "100", "--spread-bps", "0",
        "--slippage-bps", "0", "--fee-bps", "0",
    ]
    if disagreement:
        argv.extend(("--disagreement-lambda", *disagreement))
    output = StringIO()
    with patch.object(sys, "argv", argv), redirect_stdout(output):
        select_main()
    return json.loads(policy_path.read_text()), json.loads(output.getvalue())


def assert_rejected(action: Callable[[], object], message: str) -> None:
    try:
        action()
    except ValueError:
        pass
    else:
        raise AssertionError(message)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="compose-mini-policy-") as directory:
        root = Path(directory)
        profitable = choose(root / "profit.csv", 110.0)
        assert profitable["schema"] == 2
        assert set(profitable) == POLICY_V2_FIELDS
        assert all(set(trial) == TRIAL_V2_FIELDS
                   for trial in profitable["threshold_trials"])
        assert policy_disagreement_lambda(profitable) == 0.0
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

        seeded = choose(
            root / "seeded.csv", 110.0, SEEDED_DISAGREEMENT_GRID,
        )
        assert seeded["schema"] == 3
        assert len(seeded["threshold_trials"]) == 13
        assert [
            (trial["disagreement_lambda"], trial["safety_bps"])
            for trial in seeded["threshold_trials"][:-1]
        ] == [
            (disagreement, safety)
            for disagreement in SEEDED_DISAGREEMENT_GRID
            for safety in POLICY_SAFETY_GRID
        ]
        assert seeded["threshold_trials"][-1]["action"] == "cash"
        assert seeded["threshold_trials"][-1]["disagreement_lambda"] is None

        deterministic_bars, deterministic_forecasts, deterministic_report = \
            deterministic_fixture(root / "deterministic.csv")
        deterministic_policy = select_policy(
            deterministic_report, deterministic_forecasts,
            {"TEST": deterministic_bars}, Costs(0, 0, 0),
            POLICY_SAFETY_GRID, 100.0, "last_close",
            Path("calibration.json"), "0" * 64,
            Path("calibration.jsonl"), "1" * 64,
            len(deterministic_forecasts),
            disagreement_values=DETERMINISTIC_DISAGREEMENT_GRID,
        )
        assert deterministic_policy["schema"] == 3
        assert [
            (trial["disagreement_lambda"], trial["safety_bps"])
            for trial in deterministic_policy["threshold_trials"]
        ] == [
            *[(0.0, safety) for safety in POLICY_SAFETY_GRID],
            (None, None),
        ]

        grid_bars, grid_forecasts, grid_report = fixture(
            root / "grid.csv", 110.0,
        )

        def select_grid(safety: tuple[object, ...],
                        disagreement: tuple[object, ...]) -> object:
            return select_policy(
                grid_report, grid_forecasts, {"TEST": grid_bars},
                Costs(0, 0, 0), safety, 100.0, "transformer",
                Path("calibration.json"), "0" * 64,
                Path("calibration.jsonl"), "1" * 64, len(grid_forecasts),
                disagreement_values=disagreement,
            )

        invalid_safety = (
            tuple(reversed(POLICY_SAFETY_GRID)),
            (*POLICY_SAFETY_GRID, POLICY_SAFETY_GRID[-1]),
            (),
            (False, *POLICY_SAFETY_GRID[1:]),
            (-1.0, *POLICY_SAFETY_GRID[1:]),
            (math.nan, *POLICY_SAFETY_GRID[1:]),
            (math.inf, *POLICY_SAFETY_GRID[1:]),
            (*POLICY_SAFETY_GRID[:-1], 12.0),
        )
        invalid_disagreement = (
            tuple(reversed(SEEDED_DISAGREEMENT_GRID)),
            (*SEEDED_DISAGREEMENT_GRID, SEEDED_DISAGREEMENT_GRID[-1]),
            (),
            (False, *SEEDED_DISAGREEMENT_GRID[1:]),
            (-1.0, *SEEDED_DISAGREEMENT_GRID[1:]),
            (math.nan, *SEEDED_DISAGREEMENT_GRID[1:]),
            (math.inf, *SEEDED_DISAGREEMENT_GRID[1:]),
            (*SEEDED_DISAGREEMENT_GRID[:-1], 2.0),
        )
        with patch("tools.select_policy.run_backtests") as run:
            for safety in invalid_safety:
                assert_rejected(
                    lambda safety=safety: select_grid(
                        safety, SEEDED_DISAGREEMENT_GRID,
                    ),
                    "invalid schema-3 safety grid was accepted",
                )
            for disagreement in invalid_disagreement:
                assert_rejected(
                    lambda disagreement=disagreement: select_grid(
                        POLICY_SAFETY_GRID, disagreement,
                    ),
                    "invalid schema-3 disagreement grid was accepted",
                )
            assert_rejected(
                lambda: select_policy(
                    deterministic_report, deterministic_forecasts,
                    {"TEST": deterministic_bars}, Costs(0, 0, 0),
                    POLICY_SAFETY_GRID, 100.0, "last_close",
                    Path("calibration.json"), "0" * 64,
                    Path("calibration.jsonl"), "1" * 64,
                    len(deterministic_forecasts),
                    disagreement_values=(0.0, 0.5),
                ),
                "deterministic nonzero disagreement grid was accepted",
            )
            run.assert_not_called()

        abstention_bars, abstention_forecasts, abstention_report = \
            abstention_fixture(root / "abstention.csv")
        abstention = select_policy(
            abstention_report, abstention_forecasts,
            {"TEST": abstention_bars}, Costs(0, 0, 0),
            POLICY_SAFETY_GRID, 100.0, "transformer",
            Path("calibration.json"), "0" * 64,
            Path("calibration.jsonl"), "1" * 64,
            len(abstention_forecasts),
            disagreement_values=SEEDED_DISAGREEMENT_GRID,
        )
        assert abstention["action"] == "long_above"
        assert abstention["disagreement_lambda"] == 1.0
        assert abstention["safety_bps"] == 0.0
        assert abstention["threshold_trials"][0]["disagreement_lambda"] == 0.0
        selected_objective = abstention["selection_objective"]
        selected_trial = next(
            trial for trial in abstention["threshold_trials"]
            if trial["action"] == abstention["action"]
            and trial["safety_bps"] == abstention["safety_bps"]
            and trial["disagreement_lambda"] == abstention["disagreement_lambda"]
        )
        assert selected_objective == "macro_mean_terminal_log_growth"
        assert all(
            selected_trial["objective"] > trial["objective"]
            for trial in abstention["threshold_trials"]
            if trial["action"] == "long_above"
            and trial["disagreement_lambda"] == 0.0
        )

        def tied(**values: object) -> dict[str, object]:
            return {
                "objective": 1.0, "mean_gross_turnover": 1.0,
                "safety_bps": 0.0, "disagreement_lambda": 0.0,
            } | values

        def assert_winner(expected: str,
                          pair: tuple[dict[str, object],
                                      dict[str, object]]) -> None:
            assert select_trial(pair)["name"] == expected
            assert select_trial(tuple(reversed(pair)))["name"] == expected

        assert_winner("objective", (
            tied(name="other", objective=0.0),
            tied(name="objective", objective=1.0),
        ))
        assert_winner("turnover", (
            tied(name="other", mean_gross_turnover=2.0),
            tied(name="turnover", mean_gross_turnover=1.0),
        ))
        assert_winner("cash", (
            tied(name="other", safety_bps=10.0),
            tied(name="cash", safety_bps=None),
        ))
        assert_winner("lambda", (
            tied(name="other", disagreement_lambda=1.0),
            tied(name="lambda", disagreement_lambda=0.5),
        ))

        legacy_cli, legacy_summary = run_cli(root)
        explicit_cli, explicit_summary = run_cli(root, ("0", "0.5", "1"))
        assert legacy_cli["schema"] == 2
        assert "disagreement_lambda" not in legacy_summary
        assert explicit_cli["schema"] == 3
        assert explicit_summary["disagreement_lambda"] == \
            explicit_cli["disagreement_lambda"]

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

        strict_reports = (
            report | {"protocol": report["protocol"] | {
                "target_horizon_bars": True,
            }},
            report | {"protocol": report["protocol"] | {
                "target_horizon_bars": 1.0,
            }},
            report | {"protocol": report["protocol"] | {
                "target_horizon_bars": 1.5,
            }},
            report | {"sweep": report["sweep"] | {"seeds": [True, 7]}},
            report | {"sweep": report["sweep"] | {"seeds": [3.0, 7]}},
            report | {"sweep": report["sweep"] | {"seeds": [3.9, 7.9]}},
            report | {"selection": {"transformer": {"candidate": True}}},
            report | {"sweep": report["sweep"] | {
                "candidates": [{"name": "raw", "feature_set": True}],
            }},
            report | {"calibration": [
                report["calibration"][0] | {"samples": True},
                report["calibration"][1],
            ]},
            report | {"calibration": [
                report["calibration"][0] | {"samples": 1.0},
                report["calibration"][1],
            ]},
            report | {"calibration": [
                report["calibration"][0] | {"samples": 1.5},
                report["calibration"][1],
            ]},
            report | {"calibration": [
                report["calibration"][0] | {
                    "targets": {},
                },
                report["calibration"][1],
            ]},
            report | {"series": True},
            report | {"test_contract": True},
            report | {"model_fingerprints": True},
        )
        for invalid in strict_reports:
            try:
                select_policy(
                    invalid, forecasts, {"TEST": bars}, Costs(0, 0, 0),
                    (0.0,), 100.0, "transformer",
                    Path("calibration.json"), "0" * 64,
                    Path("calibration.jsonl"), "1" * 64, len(forecasts),
                )
            except ValueError:
                pass
            else:
                raise AssertionError("malformed experiment report was accepted")

        ledger_metadata = {
            "schema": 3, "path": "calibration.jsonl",
            "records": 2, "sha256": "1" * 64,
        }
        _validate_ledger_metadata(ledger_metadata, "1" * 64, 2)
        for field, value in (
            ("schema", True), ("schema", 3.0),
            ("records", True), ("records", 2.0),
        ):
            try:
                _validate_ledger_metadata(
                    ledger_metadata | {field: value}, "1" * 64, 2,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("malformed ledger metadata was accepted")

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
            {"model": "unsupported"},
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
