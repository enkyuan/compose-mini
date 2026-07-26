#!/usr/bin/env python3
"""Verify execution timing, costs, baselines, and interval reporting."""

from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import math
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtest import (
    Costs, Forecast, _aggregate_seeds, execute_long, experiment_fingerprint,
    load_bars, main as backtest_main, policy_disagreement_lambda,
    read_forecasts, run_backtests, select_trial, validate_policy,
    validate_test_experiment,
)
from tools.data_v1 import CLOSE_RETURN_TARGET, EXECUTABLE_RETURN_TARGET
from tools.files import require_disjoint, write_json


class NumericSubclass(float):
    pass


class MutableCosts:
    def __init__(self) -> None:
        self.impacts = iter((0.0, 0.5, 0.9))

    @property
    def impact(self) -> float:
        return next(self.impacts)

    @property
    def fee(self) -> float:
        return 0.0


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


def write_forecasts(path: Path, forecasts: tuple[Forecast, ...]) -> None:
    records = (
        {
            "schema": 3, "split": item.split, "fold": item.fold,
            "series": item.series, "model": item.model,
            "candidate": item.candidate, "feature_set": item.feature_set,
            "seed": item.seed, "csv_sha256": item.csv_sha256,
            "as_of": item.as_of, "target_time": item.target_time,
            "horizon_bars": item.horizon_bars,
            "target_kind": item.target_kind,
            "predicted_log_return": item.predicted_log_return,
        }
        for item in forecasts
    )
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )


def policy_trial(safety: float | None, objective: float,
                 disagreement: float | None = None,
                 schema: int = 2) -> dict[str, object]:
    cash = safety is None
    trial = {
        "action": "cash" if cash else "long_above",
        "safety_bps": safety, "objective": objective,
        "mean_final_equity": 100.0 if cash else 101.0,
        "mean_gross_turnover": 0.0 if cash else 1.0,
        "signal_coverage": 0.0 if cash else 1.0,
        "execution_coverage": 0.0 if cash else 1.0,
        "trade_count": 0 if cash else 1,
    }
    return trial if schema == 2 else trial | {
        "disagreement_lambda": disagreement,
    }


def frozen_policy(schema: int, fingerprint: str = "f" * 64,
                  target_time: str = "2026-01-30T15:00:00Z",
                  action: str = "long_above") -> dict[str, object]:
    if schema == 2:
        trials = [
            policy_trial(0.0, 1.0 if action == "long_above" else -1.0),
            policy_trial(None, 0.0),
        ]
        safety, disagreement = (
            (0.0, 0.0) if action == "long_above" else (None, None)
        )
    else:
        trials = [
            policy_trial(
                safety, (
                    1.0 if action == "long_above" and
                    (disagreement, safety) == (0.5, 6.0) else -1.0
                ), disagreement, schema,
            )
            for disagreement in (0.0, 0.5, 1.0)
            for safety in (0.0, 3.0, 6.0, 10.0)
        ]
        trials.append(policy_trial(None, 0.0, None, schema))
        safety, disagreement = (
            (6.0, 0.5) if action == "long_above" else (None, None)
        )
    value = {
        "schema": schema, "action": action, "model": "transformer",
        "candidate": "raw", "feature_set": "ohlcv",
        "target_kind": EXECUTABLE_RETURN_TARGET, "horizon_bars": 1,
        "seeds": [3, 7], "series": ["TEST"], "initial_cash": 100.0,
        "costs": {"spread_bps": 0.0, "slippage_bps": 0.0, "fee_bps": 0.0},
        "safety_bps": safety,
        "minimum_predicted_log_return": (
            None if safety is None else safety / 10_000.0
        ),
        "selection_objective": "macro_mean_terminal_log_growth",
        "calibration_report": {
            "path": "calibration.json", "sha256": "0" * 64,
        },
        "calibration_prediction_ledger": {
            "path": "calibration.jsonl", "sha256": "1" * 64,
            "source_records": 2, "selected_records": 2,
        },
        "model_fingerprints": [
            {
                "model": "transformer", "series": "TEST",
                "seed": 3, "epochs": 4, "sha256": "3" * 64,
            },
            {
                "model": "transformer", "series": "TEST",
                "seed": 7, "epochs": 4, "sha256": "7" * 64,
            },
        ],
        "threshold_trials": trials,
        "test_grid": [{
            "series": "TEST", "samples": 1,
            "first_target_time": target_time, "last_target_time": target_time,
        }],
        "calibration_fingerprint": fingerprint,
    }
    return value if schema == 2 else value | {
        "disagreement_lambda": disagreement,
    }


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

    try:
        Costs(NumericSubclass(0.0), 0.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("numeric cost subclass was accepted")


def verify_long_execution() -> None:
    costs = Costs(2.0, 3.0, 5.0)
    execution = execute_long(100.0, 50.0, 55.0, costs)
    entry, exit_ = 50.0 * (1.0 + costs.impact), 55.0 * (1.0 - costs.impact)
    shares = 100.0 / (entry * (1.0 + costs.fee))
    for actual, expected in (
        (execution.shares, shares),
        (execution.entry_execution_price, entry),
        (execution.exit_execution_price, exit_),
        (execution.entry_notional, shares * entry),
        (execution.exit_notional, shares * 55.0 * (1.0 - costs.impact)),
        (
            execution.cash_after,
            shares * 55.0 * (1.0 - costs.impact) * (1.0 - costs.fee),
        ),
    ):
        close(actual, expected)
    rounding = execute_long(
        100.0, 3066.126885137925, 1.32797250278406, costs,
    )
    assert rounding.exit_notional == \
        rounding.shares * 1.32797250278406 * (1.0 - costs.impact)
    for field in range(3):
        for invalid in (0.0, -1.0, math.nan, math.inf, True):
            values = [100.0, 50.0, 55.0]
            values[field] = invalid
            try:
                execute_long(*values, costs)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid execution input was accepted")
    for values in (
        (100.0, sys.float_info.max, 1.0),
        (sys.float_info.max, sys.float_info.min, 1.0),
    ):
        try:
            execute_long(*values, costs)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite execution result was accepted")
    try:
        execute_long(100.0, 10.0, 10.0, MutableCosts())  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("mutable cost contract was accepted")
    corrupted = Costs(0.0, 0.0, 0.0)
    object.__setattr__(corrupted, "spread_bps", NumericSubclass(0.0))
    try:
        execute_long(100.0, 10.0, 10.0, corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted cost fields were accepted")
    try:
        execute_long(NumericSubclass(100.0), 10.0, 10.0, Costs(0, 0, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("numeric execution subclass was accepted")


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
    assert "decision_signal" not in report["protocol"]
    assert "decision_signal" not in result
    close(result["strategies"]["forecast_long_cash"]["final_equity"],
          100.0 * 80.0 / 110.0)
    members = (
        replace(predictions[0], seed=3, predicted_log_return=0.0),
        replace(predictions[0], seed=7, predicted_log_return=0.5),
    )
    with patch("tools.backtest.pstdev") as disagreement:
        lambda_zero, _ = _aggregate_seeds(
            members, expected_seeds=(3, 7), disagreement_lambda=0.0,
        )
    assert not disagreement.called
    assert lambda_zero[0].predicted_log_return == 0.25
    assert lambda_zero[0].decision_signal == 0.25
    penalized, _ = _aggregate_seeds(
        members, expected_seeds=(3, 7), disagreement_lambda=0.5,
    )
    assert penalized[0].predicted_log_return == 0.25
    assert penalized[0].decision_signal == 0.125
    one_member, _ = _aggregate_seeds(
        members[:1], expected_seeds=(3,), disagreement_lambda=1.0,
    )
    assert one_member[0].decision_signal == \
        one_member[0].predicted_log_return

    default = run_backtests(
        streams, {"TEST": bars}, 100.0, Costs(0, 0, 0),
        ensemble_seeds=True, expected_seeds=(3, 7),
    )
    explicit_zero = run_backtests(
        streams, {"TEST": bars}, 100.0, Costs(0, 0, 0),
        ensemble_seeds=True, expected_seeds=(3, 7),
        disagreement_lambda=0.0,
    )
    assert default["results"] == explicit_zero["results"]
    for left, right in zip(
        default["results"], explicit_zero["results"], strict=True,
    ):
        left_strategy = left["strategies"]["forecast_long_cash"]
        right_strategy = right["strategies"]["forecast_long_cash"]
        assert left_strategy["trades"] == right_strategy["trades"]
        for field in (
            "final_equity", "gross_turnover", "decision_count",
            "signal_coverage", "execution_coverage",
        ):
            assert left_strategy[field] == right_strategy[field]

    traded = run_backtests(
        members, {"TEST": bars}, 100.0, Costs(0, 0, 0),
        safety_bps=2_000.0, ensemble_seeds=True, expected_seeds=(3, 7),
    )
    abstained = run_backtests(
        members, {"TEST": bars}, 100.0, Costs(0, 0, 0),
        safety_bps=2_000.0, ensemble_seeds=True, expected_seeds=(3, 7),
        disagreement_lambda=0.5,
    )
    trade = traded["results"][0]["strategies"]["forecast_long_cash"]
    assert trade["trade_count"] == 1
    assert trade["trades"][0]["predicted_log_return"] == 0.25
    assert "decision_signal" not in trade["trades"][0]
    assert abstained["results"][0]["strategies"]["forecast_long_cash"][
        "trade_count"
    ] == 0

    for incomplete in (
        tuple(item for index, item in enumerate(streams) if index != 1),
        (*streams, replace(streams[0], seed=11)),
        tuple(
            replace(item, as_of=predictions[1].as_of)
            if index == len(predictions) else item
            for index, item in enumerate(streams)
        ),
        tuple(
            replace(item, target_time=predictions[1].target_time)
            if index == len(predictions) else item
            for index, item in enumerate(streams)
        ),
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

    for invalid in (-0.5, True, math.inf, math.nan):
        try:
            run_backtests(
                streams, {"TEST": bars}, 100.0, Costs(0, 0, 0),
                ensemble_seeds=True, expected_seeds=(3, 7),
                disagreement_lambda=invalid,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid disagreement lambda was accepted")
    try:
        run_backtests(
            members, {"TEST": bars}, 100.0, Costs(0, 0, 0),
            disagreement_lambda=0.5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unaggregated disagreement was accepted")
    deterministic = (replace(members[0], seed=None),)
    try:
        run_backtests(
            deterministic, {"TEST": bars}, 100.0, Costs(0, 0, 0),
            ensemble_seeds=True, disagreement_lambda=0.5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("deterministic disagreement was accepted")

    ledger = Path(bars.path).with_name("diagnostic.jsonl")
    output = Path(bars.path).with_name("diagnostic.json")
    write_forecasts(
        ledger, tuple(replace(item, split="calibration") for item in streams),
    )
    argv = [
        "backtest.py", str(ledger), str(output), f"TEST={bars.path}",
        "--spread-bps", "0", "--slippage-bps", "0", "--fee-bps", "0",
        "--disagreement-lambda", "0.5",
    ]
    with patch.object(sys, "argv", argv):
        try:
            backtest_main()
        except SystemExit as error:
            assert "seed disagreement requires ensemble seeds" in str(error)
        else:
            raise AssertionError("diagnostic disagreement bypassed aggregation")


def reject_policies(values: tuple[dict[str, object], ...]) -> None:
    for value in values:
        try:
            validate_policy(value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid disagreement policy was accepted")


def verify_policy_schemas() -> None:
    policy_v2 = frozen_policy(2)
    assert validate_policy(policy_v2) == policy_v2
    assert policy_disagreement_lambda(policy_v2) == 0.0
    reject_policies((policy_v2 | {"disagreement_lambda": 0.0},))

    policy_v3 = frozen_policy(3)
    assert validate_policy(policy_v3) == policy_v3
    assert policy_disagreement_lambda(policy_v3) == 0.5
    assert [
        (item["disagreement_lambda"], item["safety_bps"])
        for item in policy_v3["threshold_trials"][:-1]
    ] == [
        (disagreement, safety)
        for disagreement in (0.0, 0.5, 1.0)
        for safety in (0.0, 3.0, 6.0, 10.0)
    ]

    def changed_trial(index: int, **changes: object) -> dict[str, object]:
        value = deepcopy(policy_v3)
        value["threshold_trials"][index].update(changes)
        return value

    removed_row = deepcopy(policy_v3)
    removed_row["threshold_trials"] = [
        item for item in removed_row["threshold_trials"]
        if item["disagreement_lambda"] != 1.0
    ]
    removed_column = deepcopy(policy_v3)
    removed_column["threshold_trials"] = [
        item for item in removed_column["threshold_trials"]
        if item["safety_bps"] != 10.0
    ]
    extra_row = deepcopy(policy_v3)
    extra_row["threshold_trials"][-1:-1] = [
        policy_trial(safety, -1.0, 1.5, 3)
        for safety in (0.0, 3.0, 6.0, 10.0)
    ]
    extra_column = deepcopy(policy_v3)
    extra_column["threshold_trials"] = [
        policy_trial(safety, -1.0, disagreement, 3)
        for disagreement in (0.0, 0.5, 1.0)
        for safety in (0.0, 3.0, 6.0, 10.0, 12.0)
    ] + [policy_trial(None, 0.0, None, 3)]
    duplicate = deepcopy(policy_v3)
    duplicate["threshold_trials"].insert(
        1, deepcopy(duplicate["threshold_trials"][0]),
    )
    reordered = deepcopy(policy_v3)
    reordered["threshold_trials"][0], reordered["threshold_trials"][1] = \
        reordered["threshold_trials"][1], reordered["threshold_trials"][0]
    missing_field = dict(policy_v3)
    del missing_field["disagreement_lambda"]
    reject_policies((
        missing_field,
        policy_v3 | {"unexpected": True},
        *(policy_v3 | {"disagreement_lambda": value}
          for value in (True, -0.5, math.nan, math.inf)),
        changed_trial(-1, disagreement_lambda=0.0),
        changed_trial(0, disagreement_lambda=None),
        policy_v3 | {"disagreement_lambda": 1.0},
        removed_row, removed_column, extra_row, extra_column,
        duplicate, reordered,
    ))

    deterministic = policy_v3 | {
        "model": "last_close", "seeds": [], "disagreement_lambda": 0.0,
        "model_fingerprints": [{
            "model": "last_close", "series": "TEST",
            "seed": None, "epochs": None, "sha256": "5" * 64,
        }],
        "calibration_prediction_ledger":
            policy_v3["calibration_prediction_ledger"] | {
                "source_records": 1, "selected_records": 1,
            },
        "threshold_trials": [
            policy_trial(
                safety, 1.0 if safety == 6.0 else -1.0, 0.0, 3,
            )
            for safety in (0.0, 3.0, 6.0, 10.0)
        ] + [policy_trial(None, 0.0, None, 3)],
    }
    assert validate_policy(deterministic) == deterministic
    deterministic_trial = deepcopy(deterministic)
    deterministic_trial["threshold_trials"][0]["disagreement_lambda"] = 0.5
    reject_policies((
        deterministic_trial,
        deterministic | {"disagreement_lambda": 0.5},
    ))

    tied = (
        policy_trial(6.0, 1.0, 1.0, 3),
        policy_trial(6.0, 1.0, 0.5, 3),
    )
    assert select_trial(tied)["disagreement_lambda"] == 0.5


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


def verify_policy_cli(directory: Path) -> None:
    csv_path = directory / "policy-bars.csv"
    rows = (
        ("2026-03-02T14:30:00Z", 100.0, 100.0),
        ("2026-03-02T15:00:00Z", 100.0, 101.0),
    )
    write_csv(csv_path, rows)
    bars = load_bars(csv_path)
    forecasts = tuple(
        forecast(
            bars.sha256, rows[0][0], rows[1][0], 1, 0.1,
            seed=seed, target_kind=EXECUTABLE_RETURN_TARGET,
        )
        for seed in (3, 7)
    )
    ledger_path = directory / "policy-predictions.jsonl"
    policy_path = directory / "policy.json"
    experiment_path = directory / "test-experiment.json"
    write_forecasts(ledger_path, forecasts)
    ledger_hash = sha256(ledger_path.read_bytes()).hexdigest()
    contract = {
        "series": [{"name": "TEST"}],
        "sweep": {"seeds": [3, 7]},
        "selection": {"transformer": {"candidate": "raw"}},
        "validation": [], "calibration": [],
        "model_fingerprints": frozen_policy(2)["model_fingerprints"],
        "test_contract": [{
            "series": "TEST", "samples": 1,
            "first_target_time": rows[1][0],
            "last_target_time": rows[1][0],
        }],
    }
    fingerprint = experiment_fingerprint(contract)

    def authorize(policy: dict[str, object]) -> None:
        write_json(policy_path, policy)
        policy_hash = sha256(policy_path.read_bytes()).hexdigest()
        write_json(experiment_path, contract | {
            "schema": 6,
            "protocol": {"phase": "selection-calibration-and-test"},
            "policies": [{
                "path": str(policy_path), "sha256": policy_hash,
                "model": "transformer",
            }],
            "prediction_ledger": {
                "schema": 3, "path": str(ledger_path),
                "records": len(forecasts), "sha256": ledger_hash,
            },
            "test": [{"model": "transformer"}],
        })

    def argv(report: Path, *extra: str) -> list[str]:
        return [
            "backtest.py", str(ledger_path), str(report),
            f"TEST={csv_path}", "--policy", str(policy_path),
            "--experiment-report", str(experiment_path), *extra,
        ]

    for schema, expected in ((2, 0.0), (3, 0.5)):
        policy = frozen_policy(schema, fingerprint, rows[1][0])
        authorize(policy)
        with patch.object(
            sys, "argv", argv(directory / f"long-v{schema}.json"),
        ), patch(
            "tools.backtest.run_backtests", return_value={"results": []},
        ) as run, patch("tools.backtest.write_report"):
            backtest_main()
        assert run.call_args.kwargs["disagreement_lambda"] == expected

    cash = frozen_policy(3, fingerprint, rows[1][0], "cash")
    authorize(cash)
    cash_report = directory / "cash-v3.json"
    with patch.object(sys, "argv", argv(cash_report)):
        backtest_main()
    report = json.loads(cash_report.read_text(encoding="utf-8"))
    assert report["protocol"]["disagreement_lambda"] == 0.0
    assert report["protocol"]["signal"] == "cash"
    for result in report["results"]:
        strategy = result["strategies"]["forecast_long_cash"]
        assert strategy["trade_count"] == 0
        assert strategy["final_equity"] == strategy["initial_equity"] == 100.0

    overrides = (
        ("--disagreement-lambda", "0"),
        ("--model", "transformer"),
        ("--initial-cash", "100"),
        ("--spread-bps", "0"),
        ("--safety-bps", "0"),
        ("--ensemble-seeds",),
    )
    for index, override in enumerate(overrides):
        with patch.object(
            sys, "argv",
            argv(directory / f"override-{index}.json", *override),
        ):
            try:
                backtest_main()
            except SystemExit as error:
                assert str(error) == \
                    "policy mode does not accept diagnostic overrides"
            else:
                raise AssertionError("policy mode accepted a diagnostic override")


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
        verify_long_execution()
        verify_validation(bars, predictions)
        verify_ensemble(bars, predictions)
        verify_policy_schemas()
        verify_test_report(predictions[0])
        verify_policy_cli(root)
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
