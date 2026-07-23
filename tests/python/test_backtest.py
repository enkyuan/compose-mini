#!/usr/bin/env python3
"""Verify execution timing, costs, baselines, and interval reporting."""

from dataclasses import replace
from pathlib import Path
import json
import math
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtest import (
    Costs, Forecast, experiment_fingerprint, load_bars, main as backtest_main,
    read_forecasts, run_backtests, validate_test_experiment,
)
from tools.data_v1 import CLOSE_RETURN_TARGET, EXECUTABLE_RETURN_TARGET
from tools.files import require_disjoint, write_json


def write_csv(path: Path, rows: tuple[tuple[str, float, float], ...]) -> None:
    lines = ["timestamp,open,high,low,close,volume"]
    for timestamp, open_, close in rows:
        lines.append(
            f"{timestamp},{open_},{max(open_, close) + 1},"
            f"{min(open_, close) - 1},{close},1000"
        )
    path.write_text("\n".join(lines), encoding="ascii")


def forecast(checksum: str, as_of: str, target: str, horizon: int,
             prediction: float, model: str = "transformer", seed: int = 7,
             split: str = "test", fold: int | None = None,
             target_kind: str = CLOSE_RETURN_TARGET) -> Forecast:
    return Forecast("TEST", model, "raw", "ohlcv", seed, checksum,
                    as_of, target, horizon, prediction, split, fold, target_kind)


def close(left: float, right: float) -> None:
    assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (left, right)


def verify_policy(directory: Path) -> tuple[object, object, dict[str, object]]:
    path = directory / "bars.csv"
    rows = (
        ("2026-01-30T14:30:00Z", 100.0, 100.0),
        ("2026-01-30T15:00:00Z", 110.0, 120.0),
        ("2026-01-30T15:30:00Z", 90.0, 80.0),
        ("2026-02-02T14:30:00Z", 100.0, 100.0),
        ("2026-02-02T15:00:00Z", 100.0, 105.0),
    )
    write_csv(path, rows)
    bars = load_bars(path)
    predictions = (
        forecast(bars.sha256, rows[0][0], rows[2][0], 2, 0.1),
        forecast(bars.sha256, rows[1][0], rows[3][0], 2, 0.1),
        forecast(bars.sha256, rows[2][0], rows[4][0], 2, -0.1),
    )
    report = run_backtests(predictions, {"TEST": bars}, 100.0, Costs(0, 0, 0))
    strategies = report["results"][0]["strategies"]
    model, always = strategies["forecast_long_cash"], strategies["always_up"]
    close(model["final_equity"], 100.0 * 80.0 / 110.0)
    close(always["final_equity"], 100.0 * 80.0 / 110.0 * 105.0 / 100.0)
    close(strategies["buy_and_hold"]["final_equity"], 100.0 * 105.0 / 110.0)
    close(strategies["cash"]["final_equity"], 100.0)
    close(model["gross_turnover"], 1.0 + 80.0 / 110.0)
    close(model["bar_close_max_drawdown"], 1.0 / 3.0)
    assert model["trade_count"] == 1 and always["trade_count"] == 2
    assert model["decision_count"] == 2
    close(model["signal_coverage"], 2.0 / 3.0)
    close(model["execution_coverage"], 1.0 / 3.0)
    close(model["eligible_entry_hit_rate"], 0.5)
    assert model["trades"][0]["as_of"] == rows[0][0]
    assert model["trades"][0]["predicted_log_return"] == 0.1
    assert [item["period"] for item in model["periods"]["daily"]] == [
        "2026-01-30", "2026-02-02",
    ]
    assert [item["period"] for item in model["periods"]["weekly"]] == [
        "2026-W05", "2026-W06",
    ]
    assert [item["period"] for item in model["periods"]["monthly"]] == [
        "2026-01", "2026-02",
    ]
    return bars, predictions, report


def verify_costs(directory: Path) -> None:
    path = directory / "flat.csv"
    rows = (
        ("2026-02-03T14:30:00Z", 50.0, 50.0),
        ("2026-02-03T15:00:00Z", 100.0, 100.0),
    )
    write_csv(path, rows)
    costs = Costs(2.0, 3.0, 5.0)
    bars = load_bars(path)
    report = run_backtests(
        (forecast(bars.sha256, rows[0][0], rows[1][0], 1, 0.1),),
        {"TEST": bars}, 100.0, costs,
    )
    final = report["results"][0]["strategies"]["forecast_long_cash"][
        "final_equity"
    ]
    expected = 100.0 * (1.0 - costs.impact) * (1.0 - costs.fee) / \
        ((1.0 + costs.impact) * (1.0 + costs.fee))
    close(final, expected)
    threshold = costs.break_even_log_return
    for prediction, count in ((threshold, 0), (threshold + 1e-8, 1)):
        result = run_backtests(
            (forecast(bars.sha256, rows[0][0], rows[1][0], 1, prediction,
                      target_kind=EXECUTABLE_RETURN_TARGET),),
            {"TEST": bars}, 100.0, costs,
        )
        assert result["results"][0]["strategies"]["forecast_long_cash"][
            "trade_count"
        ] == count
        assert result["protocol"]["target_kind"] == EXECUTABLE_RETURN_TARGET


def verify_validation(bars: object, predictions: object) -> None:
    invalid = (
        (replace(predictions[0], target_time=predictions[1].target_time),),
        (predictions[0], predictions[0]),
        (predictions[0], predictions[2]),
        (replace(predictions[0], csv_sha256="0" * 64),),
        (*predictions, replace(predictions[0], model="mlp")),
        (*predictions, replace(
            predictions[0], target_kind=EXECUTABLE_RETURN_TARGET,
        )),
    )
    for values in invalid:
        try:
            run_backtests(values, {"TEST": bars}, 100.0, Costs(0, 0, 0))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid forecast alignment was accepted")
    for costs in (Costs(0, 0, 0),):
        try:
            run_backtests((predictions[0],), {"TEST": bars}, 0.0, costs)
        except ValueError:
            pass
        else:
            raise AssertionError("nonpositive capital was accepted")
    for values in ((-1, 0, 0), (0, -1, 0), (0, 0, 10_000)):
        try:
            Costs(*values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid costs were accepted")


def verify_ensemble(bars: object, predictions: object) -> None:
    streams = tuple(
        replace(item, seed=seed,
                predicted_log_return=item.predicted_log_return + adjustment)
        for seed, adjustment in ((3, -0.02), (7, 0.02))
        for item in predictions
    )
    report = run_backtests(
        streams, {"TEST": bars}, 100.0, Costs(0, 0, 0), ensemble_seeds=True,
        expected_seeds=(3, 7),
    )
    assert len(report["results"]) == 1
    result = report["results"][0]
    assert result["seed_aggregation"] == "arithmetic_mean"
    assert result["seeds"] == [3, 7]
    close(result["strategies"]["forecast_long_cash"]["final_equity"],
          100.0 * 80.0 / 110.0)
    for incomplete in (
        tuple(item for item in streams if item.seed == 3),
        (*streams, replace(streams[0], seed=11)),
    ):
        try:
            run_backtests(
                incomplete, {"TEST": bars}, 100.0, Costs(0, 0, 0),
                ensemble_seeds=True, expected_seeds=(3, 7),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete ensemble was accepted")


def verify_test_report(prediction: Forecast) -> None:
    ledger_hash, policy_hash = "2" * 64, "3" * 64
    fingerprints = [{
        "model": "transformer", "series": "TEST", "seed": 7,
        "epochs": 4, "sha256": "4" * 64,
    }]
    contract = {
        "series": [{"name": "TEST"}], "sweep": {}, "selection": {},
        "validation": [], "calibration": [],
        "model_fingerprints": fingerprints, "test_contract": [],
    }
    policy = {
        "model": "transformer",
        "calibration_fingerprint": experiment_fingerprint(contract),
        "model_fingerprints": fingerprints,
    }
    report = contract | {
        "schema": 6,
        "protocol": {"phase": "selection-calibration-and-test"},
        "policies": [{
            "path": "policy.json", "sha256": policy_hash,
            "model": "transformer",
        }],
        "prediction_ledger": {
            "schema": 3, "path": "predictions.jsonl", "records": 1,
            "sha256": ledger_hash,
        },
        "test": [{"model": "transformer"}],
    }
    assert validate_test_experiment(
        report, ledger_hash, (prediction,), policy_hash, policy,
    ) == report
    invalid = (
        (report | {"policies": [*report["policies"], {"garbage": True}]},
         (prediction,)),
        (report | {"prediction_ledger": report["prediction_ledger"] | {
            "schema": 2.0,
        }}, (prediction,)),
        (report, (prediction, replace(prediction, model="mlp"))),
        (report | {"schema": 5}, (prediction,)),
        (report | {"protocol": {
            "phase": "selection-and-calibration",
        }}, (prediction,)),
        (report | {"model_fingerprints": [
            fingerprints[0] | {"sha256": "5" * 64},
        ]}, (prediction,)),
    )
    for value, forecasts in invalid:
        try:
            validate_test_experiment(
                value, ledger_hash, forecasts, policy_hash, policy,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid test experiment was accepted")


def verify_io(directory: Path, predictions: object,
              report: dict[str, object]) -> None:
    ledger = directory / "predictions.jsonl"
    output, repeated = directory / "report.json", directory / "report-again.json"
    records = []
    for item in predictions:
        records.append({
            "schema": 1, "split": "test", "series": item.series,
            "model": item.model, "candidate": item.candidate,
            "feature_set": item.feature_set, "seed": item.seed,
            "csv_sha256": item.csv_sha256,
            "as_of": item.as_of, "target_time": item.target_time,
            "horizon_bars": item.horizon_bars,
            "predicted_log_return": item.predicted_log_return,
        })
    ledger.write_text("".join(json.dumps(item) + "\n" for item in records),
                      encoding="utf-8")
    assert read_forecasts(ledger) == predictions
    versioned = records[0] | {
        "schema": 2, "fold": None,
        "target_kind": EXECUTABLE_RETURN_TARGET,
    }
    assert Forecast.parse(versioned) == replace(
        predictions[0], target_kind=EXECUTABLE_RETURN_TARGET,
    )
    validation = versioned | {"split": "validation", "fold": 0}
    assert Forecast.parse(validation).split == "validation"
    calibration = versioned | {
        "schema": 3, "split": "calibration", "fold": None,
        "target_kind": EXECUTABLE_RETURN_TARGET,
    }
    assert Forecast.parse(calibration).split == "calibration"
    for invalid in (
        calibration | {"fold": 0},
        validation | {"schema": 3, "fold": None},
    ):
        try:
            Forecast.parse(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid forecast split and fold were accepted")
    for field, value in (
        ("split", "validation"), ("predicted_log_return", math.nan),
        ("predicted_log_return", 10 ** 400), ("as_of", 1),
    ):
        invalid = records[0] | {field: value}
        try:
            Forecast.parse(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {field} was accepted")
    for schema in ([], {}):
        invalid = records[0] | {"schema": schema}
        try:
            Forecast.parse(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid forecast schema was accepted")
        ledger.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
        try:
            read_forecasts(ledger)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid ledger schema was accepted")
    write_json(output, report)
    write_json(repeated, report)
    assert output.read_bytes() == repeated.read_bytes()
    assert json.loads(output.read_text(encoding="utf-8")) == report


def verify_frozen_ledger(directory: Path, bars: object,
                         predictions: object) -> None:
    ledger, output = directory / "mutable.jsonl", directory / "rejected.json"
    records = [
        {
            "schema": 3, "split": "calibration", "fold": None,
            "series": item.series, "model": item.model,
            "candidate": item.candidate, "feature_set": item.feature_set,
            "seed": item.seed, "csv_sha256": item.csv_sha256,
            "as_of": item.as_of, "target_time": item.target_time,
            "horizon_bars": item.horizon_bars,
            "target_kind": item.target_kind,
            "predicted_log_return": item.predicted_log_return,
        }
        for item in predictions
    ]
    ledger.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    original = ledger.read_bytes()

    def mutate(*args: object, **kwargs: object) -> dict[str, object]:
        ledger.write_bytes(original + b"\n")
        return run_backtests(*args, **kwargs)

    argv = [
        "backtest.py", str(ledger), str(output),
        f"TEST={bars.path}", "--spread-bps", "0",
        "--slippage-bps", "0", "--fee-bps", "0",
    ]
    with patch("tools.backtest.run_backtests", side_effect=mutate), \
         patch.object(sys, "argv", argv):
        try:
            backtest_main()
        except SystemExit as error:
            assert "input changed" in str(error)
        else:
            raise AssertionError("mid-run ledger replacement was accepted")
    assert not output.exists()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="compose-mini-backtest-") as directory:
        root = Path(directory)
        bars, predictions, report = verify_policy(root)
        verify_costs(root)
        verify_validation(bars, predictions)
        verify_ensemble(bars, predictions)
        verify_test_report(predictions[0])
        verify_io(root, predictions, report)
        verify_frozen_ledger(root, bars, predictions)
        for inputs, outputs in (((root / "bars.csv",), (root / "bars.csv",)),
                                ((), (root / "same", root / "same"))):
            try:
                require_disjoint(inputs, outputs)
            except ValueError:
                pass
            else:
                raise AssertionError("output alias was accepted")
    print("backtest tests passed")


if __name__ == "__main__":
    main()
