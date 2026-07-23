#!/usr/bin/env python3
"""Select and freeze one cost-aware trading policy from calibration data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.backtest import (
    NAME, POLICY_MODELS, SEEDED_MODELS, Costs, Forecast,
    experiment_fingerprint, load_frozen_bars, read_forecasts, run_backtests,
    select_trial, validate_policy,
)
from tools.data_v1 import EXECUTABLE_RETURN_TARGET
from tools.files import (
    freeze_inputs, require_disjoint, series_arg, verify_frozen, write_json,
)


def _read_report(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read experiment report: {error}") from error
    if not isinstance(value, dict) or type(value.get("schema")) is not int or \
       value["schema"] != 6 or \
       not isinstance(value.get("protocol"), dict) or \
       value["protocol"].get("phase") != "selection-and-calibration" or \
       not (isinstance(value.get("test"), list) and not value["test"]):
        raise ValueError("policy selection requires a calibration-only report")
    return value


def _name(value: object, field: str) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError(f"experiment {field} is invalid")
    return value


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"experiment {field} is invalid")
    return value


def _boundary(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 2 or \
       any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"experiment {field} boundary is invalid")
    return value


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and \
        all(byte in "0123456789abcdef" for byte in value)


def _validate_ledger_metadata(value: object, checksum: str,
                              records: int) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema", "path", "records", "sha256",
    } or type(value["schema"]) is not int or value["schema"] != 3 or \
       not isinstance(value["path"], str) or not value["path"] or \
       type(value["records"]) is not int or value["records"] < 1 or \
       not _digest(value["sha256"]) or value["sha256"] != checksum or \
       value["records"] != records:
        raise ValueError("calibration ledger does not match the report")


def _contract(report: Mapping[str, object], model: str
              ) -> tuple[str, str, int, tuple[int, ...], tuple[str, ...],
                         list[Mapping[str, object]], list[dict[str, object]],
                         list[dict[str, object]]]:
    if model not in POLICY_MODELS:
        raise ValueError("policy model is unsupported")
    selection, sweep, protocol = (
        report.get("selection"), report.get("sweep"), report.get("protocol"),
    )
    if not all(isinstance(item, dict)
               for item in (selection, sweep, protocol)):
        raise ValueError("experiment report is missing policy inputs")
    selected = selection.get(model)
    candidates = sweep.get("candidates")
    if not isinstance(selected, dict) or not isinstance(candidates, list) or \
       not candidates or any(not isinstance(item, dict) for item in candidates):
        raise ValueError("experiment report is missing policy inputs")
    candidate = _name(selected.get("candidate"), "candidate")
    candidate_names = [
        _name(item.get("name"), "candidate")
        for item in candidates
    ]
    for item in candidates:
        _name(item.get("feature_set"), "feature set")
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("experiment candidate configuration is invalid")
    configurations = [
        item for item in candidates
        if item["name"] == candidate
    ]
    if len(configurations) != 1:
        raise ValueError("experiment candidate configuration is invalid")
    feature_set = _name(
        configurations[0].get("feature_set"), "feature set",
    )
    target_kind = protocol.get("target_kind")
    if not isinstance(target_kind, str) or \
       target_kind != EXECUTABLE_RETURN_TARGET:
        raise ValueError("policy requires selected executable-return calibration")
    horizon = _integer(
        protocol.get("target_horizon_bars"), "target horizon", 1,
    )

    seeds_value = sweep.get("seeds")
    if not isinstance(seeds_value, list) or not seeds_value:
        raise ValueError("experiment seeds are invalid")
    configured = tuple(sorted(
        _integer(seed, "seed") for seed in seeds_value
    ))
    if len(configured) != len(set(configured)):
        raise ValueError("experiment seeds are invalid")
    seeds = configured if model in SEEDED_MODELS else ()

    series_value = report.get("series")
    if not isinstance(series_value, list) or not series_value or any(
        not isinstance(item, dict) for item in series_value
    ):
        raise ValueError("experiment series are invalid")
    names = tuple(sorted(
        _name(item.get("name"), "series") for item in series_value
    ))
    if len(names) != len(set(names)):
        raise ValueError("experiment series are invalid")

    calibration = report.get("calibration")
    if not isinstance(calibration, list) or not calibration:
        raise ValueError("experiment calibration records are invalid")
    records = []
    for item in calibration:
        if not isinstance(item, dict):
            raise ValueError("experiment calibration records are invalid")
        record_model = _name(item.get("model"), "calibration model")
        record_candidate = _name(
            item.get("candidate"), "calibration candidate",
        )
        _name(item.get("feature_set"), "calibration feature set")
        _name(item.get("series"), "calibration series")
        seed = item.get("seed")
        if seed is not None:
            _integer(seed, "calibration seed")
        _integer(item.get("samples"), "calibration samples", 1)
        targets = item.get("targets")
        if not isinstance(targets, dict):
            raise ValueError("experiment calibration boundaries are invalid")
        _boundary(targets.get("validation"), "calibration validation")
        if item.get("fold") is not None:
            raise ValueError("experiment calibration fold is invalid")
        if record_model == model and record_candidate == candidate:
            records.append(item)
    if not records or \
       {record["seed"] for record in records} != (set(seeds) or {None}):
        raise ValueError("calibration report has an incomplete seed set")

    fingerprint_value = report.get("model_fingerprints")
    if not isinstance(fingerprint_value, list) or not fingerprint_value:
        raise ValueError("experiment model fingerprints are invalid")
    fingerprints = []
    for item in fingerprint_value:
        if not isinstance(item, dict) or set(item) != {
            "model", "series", "seed", "epochs", "sha256",
        }:
            raise ValueError("experiment model fingerprints are invalid")
        fingerprint_model = _name(item["model"], "fingerprint model")
        _name(item["series"], "fingerprint series")
        seed, epochs = item["seed"], item["epochs"]
        if seed is None:
            if epochs is not None:
                raise ValueError("experiment model fingerprints are invalid")
        else:
            _integer(seed, "fingerprint seed")
            _integer(epochs, "fingerprint epochs", 1)
        if not _digest(item["sha256"]):
            raise ValueError("experiment model fingerprints are invalid")
        if fingerprint_model == model:
            fingerprints.append(item)
    if not fingerprints:
        raise ValueError("experiment model fingerprints are incomplete")

    test_grid_value = report.get("test_contract")
    if not isinstance(test_grid_value, list) or \
       len(test_grid_value) != len(names):
        raise ValueError("experiment test grid is invalid")
    test_grid = []
    for item in test_grid_value:
        if not isinstance(item, dict) or set(item) != {
            "series", "samples", "first_target_time", "last_target_time",
        } or _name(item.get("series"), "test series") not in names:
            raise ValueError("experiment test grid is invalid")
        _integer(item.get("samples"), "test samples", 1)
        _boundary(
            [item.get("first_target_time"), item.get("last_target_time")],
            "test",
        )
        test_grid.append(item)
    if len({item["series"] for item in test_grid}) != len(test_grid):
        raise ValueError("experiment test grid is invalid")
    return (
        candidate, feature_set, horizon, seeds, names, records, test_grid,
        fingerprints,
    )


def _validate_grid(forecasts: Sequence[Forecast],
                   records: Sequence[Mapping[str, object]],
                   names: Sequence[str], seeds: Sequence[int]) -> None:
    actual: dict[tuple[object, ...], list[Forecast]] = defaultdict(list)
    for forecast in forecasts:
        actual[(forecast.series, forecast.seed)].append(forecast)
    reported = {
        (record["series"], record["seed"]): record
        for record in records
    }
    expected = {
        (series, seed) for series in names for seed in seeds or (None,)
    }
    if len(reported) != len(records) or set(reported) != expected or \
       set(actual) != expected or any(
           record.get("fold") is not None for record in records
       ):
        raise ValueError("calibration ledger does not cover every selected run")
    for key, record in reported.items():
        rows = sorted(actual[key], key=lambda item: item.target_time)
        boundary = record["targets"]["validation"]
        if len(rows) != record["samples"] or \
           [rows[0].target_time, rows[-1].target_time] != boundary:
            raise ValueError("calibration ledger does not match report boundaries")


def _trial(report: Mapping[str, object], safety_bps: float
           ) -> dict[str, object]:
    strategies = [
        result["strategies"]["forecast_long_cash"]
        for result in report["results"]
    ]
    return {
        "action": "long_above", "safety_bps": safety_bps,
        "objective": fmean(math.log(item["final_equity"] /
                                    item["initial_equity"])
                           for item in strategies),
        "mean_final_equity": fmean(item["final_equity"] for item in strategies),
        "mean_gross_turnover": fmean(item["gross_turnover"]
                                     for item in strategies),
        "signal_coverage": fmean(item["signal_coverage"] for item in strategies),
        "execution_coverage": fmean(item["execution_coverage"]
                                    for item in strategies),
        "trade_count": sum(item["trade_count"] for item in strategies),
    }


def select_policy(report: Mapping[str, object], forecasts: Sequence[Forecast],
                  series: Mapping[str, object], costs: Costs,
                  safety_values: Sequence[float], initial_cash: float,
                  model: str, report_path: Path, report_hash: str,
                  ledger_path: Path, ledger_hash: str,
                  source_records: int) -> dict[str, object]:
    candidate, feature_set, horizon, seeds, names, records, test_grid, \
        fingerprints = _contract(report, model)
    selected = tuple(item for item in forecasts if
                     item.model == model and item.candidate == candidate)
    if not selected or set(series) != set(names) or any(
        item.split != "calibration" or item.fold is not None or
        item.feature_set != feature_set or
        item.target_kind != EXECUTABLE_RETURN_TARGET or
        item.horizon_bars != horizon for item in selected
    ):
        raise ValueError("calibration inputs do not match the selected contract")
    _validate_grid(selected, records, names, seeds)
    trials = [
        _trial(run_backtests(
            selected, series, initial_cash, costs, safety,
            ensemble_seeds=True, expected_seeds=seeds,
        ), safety)
        for safety in safety_values
    ]
    trials.append({
        "action": "cash", "safety_bps": None, "objective": 0.0,
        "mean_final_equity": initial_cash, "mean_gross_turnover": 0.0,
        "signal_coverage": 0.0, "execution_coverage": 0.0,
        "trade_count": 0,
    })
    selected_trial = select_trial(trials)
    safety = selected_trial["safety_bps"]
    policy = {
        "schema": 2, "action": selected_trial["action"], "model": model,
        "candidate": candidate, "feature_set": feature_set,
        "target_kind": EXECUTABLE_RETURN_TARGET, "horizon_bars": horizon,
        "seeds": list(seeds), "series": list(names),
        "initial_cash": initial_cash,
        "costs": {
            "spread_bps": costs.spread_bps,
            "slippage_bps": costs.slippage_bps,
            "fee_bps": costs.fee_bps,
        },
        "safety_bps": safety,
        "minimum_predicted_log_return": (
            None if safety is None else
            costs.break_even_log_return + safety / 10_000.0
        ),
        "selection_objective": "macro_mean_terminal_log_growth",
        "calibration_report": {
            "path": str(report_path), "sha256": report_hash,
        },
        "calibration_prediction_ledger": {
            "path": str(ledger_path), "sha256": ledger_hash,
            "source_records": source_records, "selected_records": len(selected),
        },
        "model_fingerprints": fingerprints,
        "threshold_trials": trials,
        "test_grid": test_grid,
        "calibration_fingerprint": experiment_fingerprint(report),
    }
    return validate_policy(policy)


def _series(value: str) -> tuple[str, Path]:
    return series_arg(value, NAME)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_report", type=Path)
    parser.add_argument("calibration_predictions", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("series", nargs="+", type=_series, metavar="NAME=CSV")
    parser.add_argument("--model", required=True)
    parser.add_argument("--safety-bps", nargs="+", type=float, required=True)
    parser.add_argument("--initial-cash", type=float, default=100.0)
    parser.add_argument("--spread-bps", type=float, required=True)
    parser.add_argument("--slippage-bps", type=float, required=True)
    parser.add_argument("--fee-bps", type=float, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        require_disjoint(
            [args.experiment_report, args.calibration_predictions,
             *(path for _, path in args.series)], [args.policy],
        )
        values = tuple(sorted(set(args.safety_bps)))
        if len(values) != len(args.safety_bps) or any(
            not math.isfinite(value) or not 0.0 <= value < 10_000.0
            for value in values
        ) or not math.isfinite(args.initial_cash) or args.initial_cash <= 0.0:
            raise ValueError("policy grid or initial cash is invalid")
        model = args.model
        if not NAME.fullmatch(model) or model not in POLICY_MODELS:
            raise ValueError("model is invalid")
        paths = dict(args.series)
        if len(paths) != len(args.series):
            raise ValueError("series names must be unique")
        sources = (
            args.experiment_report, args.calibration_predictions,
            *(path for _, path in args.series),
        )
        with freeze_inputs(sources) as frozen:
            report_input, ledger_input = frozen[:2]
            series_inputs = frozen[2:]
            report = _read_report(report_input.snapshot)
            forecasts = read_forecasts(ledger_input.snapshot)
            _validate_ledger_metadata(
                report.get("calibration_prediction_ledger"),
                ledger_input.sha256, len(forecasts),
            )
            bars = {
                name: load_frozen_bars(item)
                for (name, _), item in zip(
                    args.series, series_inputs, strict=True,
                )
            }
            policy = select_policy(
                report, forecasts, bars,
                Costs(args.spread_bps, args.slippage_bps, args.fee_bps),
                values, args.initial_cash, model,
                report_input.source, report_input.sha256,
                ledger_input.source, ledger_input.sha256, len(forecasts),
            )
            verify_frozen(frozen)
            write_json(args.policy, policy)
    except (IndexError, KeyError, OverflowError, TypeError) as error:
        raise SystemExit("policy inputs have invalid nested fields") from error
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "policy": str(args.policy), "action": policy["action"],
        "model": policy["model"], "safety_bps": policy["safety_bps"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
