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
    NAME, Costs, Forecast, experiment_fingerprint, load_bars, read_forecasts,
    run_backtests, select_trial, validate_policy,
)
from tools.data_v1 import EXECUTABLE_RETURN_TARGET
from tools.files import (
    file_sha256 as _sha256, require_disjoint, series_arg, write_json,
)

SEEDED_MODELS = frozenset(("transformer", "mlp"))
POLICY_MODELS = SEEDED_MODELS | {"linear", "rolling_mean", "last_close"}


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


def _contract(report: Mapping[str, object], model: str
              ) -> tuple[str, str, int, tuple[int, ...], tuple[str, ...],
                         list[Mapping[str, object]], list[dict[str, object]],
                         list[dict[str, object]]]:
    try:
        candidate = str(report["selection"][model]["candidate"])
        candidates = report["sweep"]["candidates"]
        configuration = next(item for item in candidates
                             if item["name"] == candidate)
        feature_set = str(configuration["feature_set"])
        target_kind = str(report["protocol"]["target_kind"])
        horizon = int(report["protocol"]["target_horizon_bars"])
        records = [item for item in report["calibration"]
                   if item["model"] == model and item["candidate"] == candidate]
        fingerprints = [item for item in report["model_fingerprints"]
                        if item["model"] == model]
        names = tuple(sorted(item["name"] for item in report["series"]))
        test_grid = list(report["test_contract"])
    except (KeyError, StopIteration, TypeError) as error:
        raise ValueError("experiment report is missing policy inputs") from error
    if not records or not fingerprints or \
       target_kind != EXECUTABLE_RETURN_TARGET or horizon < 1:
        raise ValueError("policy requires selected executable-return calibration")
    try:
        configured = tuple(sorted(int(seed) for seed in report["sweep"]["seeds"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("experiment seeds are invalid") from error
    if model not in POLICY_MODELS or not configured or \
       len(configured) != len(set(configured)):
        raise ValueError("policy model or experiment seeds are invalid")
    seeds = configured if model in SEEDED_MODELS else ()
    if {record["seed"] for record in records} != (set(seeds) or {None}):
        raise ValueError("calibration report has an incomplete seed set")
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
        if not NAME.fullmatch(model):
            raise ValueError("model is invalid")
        paths = dict(args.series)
        if len(paths) != len(args.series):
            raise ValueError("series names must be unique")
        report_hash = _sha256(args.experiment_report)
        ledger_hash = _sha256(args.calibration_predictions)
        report = _read_report(args.experiment_report)
        forecasts = read_forecasts(args.calibration_predictions)
        metadata = report.get("calibration_prediction_ledger")
        if not isinstance(metadata, dict) or metadata.get("schema") != 3 or \
           metadata.get("sha256") != ledger_hash or \
           metadata.get("records") != len(forecasts):
            raise ValueError("calibration ledger does not match the report")
        bars = {name: load_bars(path) for name, path in paths.items()}
        policy = select_policy(
            report, forecasts, bars,
            Costs(args.spread_bps, args.slippage_bps, args.fee_bps),
            values, args.initial_cash, model, args.experiment_report, report_hash,
            args.calibration_predictions, ledger_hash, len(forecasts),
        )
        if _sha256(args.experiment_report) != report_hash or \
           _sha256(args.calibration_predictions) != ledger_hash or any(
               _sha256(Path(item.path)) != item.sha256 for item in bars.values()
           ):
            raise ValueError("policy inputs changed during selection")
        write_json(args.policy, policy)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "policy": str(args.policy), "action": policy["action"],
        "model": policy["model"], "safety_bps": policy["safety_bps"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
