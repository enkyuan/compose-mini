#!/usr/bin/env python3
"""Validate and analyze one frozen common-stock calibration run."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from urllib.parse import quote
import argparse
import json
import math
import os
import random
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.backtest import (
    POLICY_COST_FIELDS, SEEDED_MODELS, Bars, Costs, Forecast,
    experiment_fingerprint, load_frozen_bars, policy_disagreement_lambda,
    read_forecasts, run_backtests, validate_policy,
)
from tools.data_v1 import EXECUTABLE_RETURN_TARGET
from tools.fetch_massive import Bar, scan_regular_bars
from tools.fetch_universe import (
    FETCH_SCHEMA, GAP_POLICY, GAP_SCOPE, UniverseManifest,
)
from tools.files import (
    FrozenInput, freeze_inputs, require_disjoint, verify_frozen, write_json,
)

MODELS = ("transformer", "linear", "mlp", "rolling_mean", "last_close")
POLICY_MODELS = ("transformer", "mlp", "linear")
SEEDS = (7, 19, 31, 43, 61)
EVIDENCE = "descriptive-calibration-resubstitution"
BOOTSTRAP_SEED = 20260723
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_BLOCK_DATES = 5
REPORT_FILES = (
    "fetch-report.json", "experiment.json", "calibration.jsonl",
    *(f"policy-{model}.json" for model in POLICY_MODELS),
    *(f"backtest-{model}.json" for model in POLICY_MODELS),
)
REPORT_FIELDS = {
    "schema", "protocol", "runtime", "series", "test_contract", "sweep",
    "selection", "validation", "calibration", "model_fingerprints",
    "validation_summary", "test", "summary", "sweep_input",
    "calibration_prediction_ledger",
}
EXPECTED_PROTOCOL = {
    "split": "embargoed expanding walk-forward by target time",
    "selection": "minimum mean validation scaled-return MSE",
    "selection_aggregation": "macro mean over series, folds, and seeds",
    "holdout_aggregation": "macro mean over series and seeds",
    "phase": "selection-and-calibration",
    "calibration_policy": "deferred until policy selection",
    "aligned_history_bars": 17,
    "target_horizon_bars": 13,
    "target_kind": EXECUTABLE_RETURN_TARGET,
    "target_formula": "log(close[t + horizon] / open[t + 1])",
    "zero_return_reference": "open[t + 1]",
    "alignment_horizon_bars": 13,
    "embargo_bars": 12,
    "feature_contract": "experiment-only; artifact V1 remains OHLCV",
    "feature_sets": {
        "ohlcv": ["open", "high", "low", "close", "volume"],
    },
    "folds": 2,
    "fold_fraction": 0.1,
    "run_count": 429,
    "diagnostic_caps": {
        "linear_flat_features": 2_048,
        "mlp_parameters": 8_388_608,
    },
}
RECORD_FIELDS = {
    "model", "candidate", "series", "feature_set", "fold", "seed", "targets",
    "samples", "validation_scaled_mse", "metrics",
}
METRIC_FIELDS = {
    "return_mse", "return_mae", "direction_accuracy", "close_mae",
    "zero_return_baseline_mae",
}
FileIdentity = tuple[int, int]


@dataclass(frozen=True)
class DirectoryMembership:
    run_dir: FileIdentity
    csv_dir: FileIdentity
    run_files: frozenset[tuple[str, FileIdentity]]
    csv_files: frozenset[tuple[Path, FileIdentity]]


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("JSON contains a duplicate field")
        value[name] = item
    return value


def _constant(_value: str) -> object:
    raise ValueError("JSON contains a nonfinite number")


def read_json(path: Path, *, canonical: bool = False) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object,
            parse_constant=_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON input: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    if canonical:
        encoded = (
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        if raw != encoded:
            raise ValueError("generated JSON input is not canonical")
    _finite_tree(value)
    return value


def _finite_tree(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("numeric inputs must be finite")
    if isinstance(value, Mapping):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        byte in "0123456789abcdef" for byte in value
    )


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} is invalid")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _absent(path: Path) -> None:
    if os.path.lexists(path):
        raise ValueError("output must be fresh and absent")


def resolve_fresh_output(path: Path) -> Path:
    """Reject lexical aliases before returning one resolved output target."""
    candidates = (path, Path(os.path.abspath(path)))
    for candidate in candidates:
        _absent(candidate)
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("output path is invalid") from error
    _absent(resolved)
    return resolved


def _identity(path: Path, expected: int, label: str) -> FileIdentity:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if stat.S_IFMT(metadata.st_mode) != expected:
        raise ValueError(f"{label} has the wrong file type")
    return metadata.st_dev, metadata.st_ino


def regular_file_identity(path: Path) -> FileIdentity:
    return _identity(path, stat.S_IFREG, "input artifact")


def regular_file_identities(
    paths: Sequence[Path],
) -> tuple[FileIdentity, ...]:
    identities = tuple(regular_file_identity(path) for path in paths)
    if len(set(identities)) != len(identities):
        raise ValueError("input artifacts must not share a file identity")
    return identities


def directory_membership(run_dir: Path) -> DirectoryMembership:
    """Return exact regular-file membership for the run and CSV directories."""
    run_identity = _identity(run_dir, stat.S_IFDIR, "run directory")
    children = tuple(run_dir.iterdir())
    csv_dir = run_dir / "csv"
    csv_identity = _identity(csv_dir, stat.S_IFDIR, "CSV directory")
    run_files = tuple(
        (path.name, regular_file_identity(path))
        for path in children if path != csv_dir
    )
    csv_paths = tuple(csv_dir.iterdir())
    return DirectoryMembership(
        run_identity,
        csv_identity,
        frozenset(run_files),
        frozenset(
            (path.resolve(), regular_file_identity(path))
            for path in csv_paths
        ),
    )


def verify_membership(
    run_dir: Path, expected: DirectoryMembership,
) -> None:
    if directory_membership(run_dir) != expected:
        raise ValueError("run or CSV directory membership changed")


def _expected_config() -> dict[str, object]:
    return {
        "alignment_horizon_bars": 13,
        "batch_size": 128,
        "candidates": [{
            "feature_set": "ohlcv", "ff_dim": 32, "heads": 2, "layers": 1,
            "learning_rate": 0.0003, "mlp_dim": 32, "model_dim": 16,
            "name": "raw-17", "ridge": 0.001, "rolling_window": 8,
            "seq_len": 17, "weight_decay": 0.0001,
        }],
        "epochs": 100,
        "fold_fraction": 0.1,
        "folds": 2,
        "models": list(MODELS),
        "patience": 10,
        "seeds": list(SEEDS),
        "target_horizon_bars": 13,
        "target_kind": EXECUTABLE_RETURN_TARGET,
    }


def validate_config(value: object) -> dict[str, object]:
    if value != _expected_config():
        raise ValueError("config does not match the exact raw-17 protocol")
    _finite_tree(value)
    return value


def _manifest_names(manifest: UniverseManifest) -> tuple[str, ...]:
    names = tuple(item.ticker for item in manifest.series)
    strata = tuple(item.stratum for item in manifest.series)
    if len(names) != 11 or len(set(names)) != 11 or len(set(strata)) != 11:
        raise ValueError("analysis requires 11 unique tickers and strata")
    return names


def _gap_audit(
    timestamps: Sequence[str], minutes: int,
) -> tuple[int, dict[str, object]]:
    observed: list[Bar] = [
        (
            int(datetime.strptime(
                timestamp, "%Y-%m-%dT%H:%M:%SZ",
            ).replace(tzinfo=timezone.utc).timestamp() * 1000),
            1.0, 1.0, 1.0, 1.0, 0.0,
        )
        for timestamp in timestamps
    ]
    selected, sessions, gaps = scan_regular_bars(observed, minutes)
    if len(selected) != len(observed):
        raise ValueError("audited CSV contains non-regular bars")
    return sessions, {
        "scope": GAP_SCOPE,
        "affected_sessions": len({gap["session"] for gap in gaps}),
        "internal_gap_count": len(gaps),
        "internal_missing_bins": sum(
            int(gap["absent_bins"]) for gap in gaps
        ),
        "gaps": gaps,
    }


def validate_fetch(
    value: Mapping[str, object], manifest: UniverseManifest,
    manifest_input: FrozenInput, bars: Mapping[str, Bars],
) -> None:
    legacy_fields = {
        "schema", "purpose", "declared_on", "eligibility_date", "start", "end",
        "interval_minutes", "adjusted", "session", "manifest", "series",
    }
    audited = "fetch_schema" in value
    expected_fields = legacy_fields | (
        {"fetch_schema", "gap_policy"} if audited else set()
    )
    if set(value) != expected_fields or audited and (
        type(value["fetch_schema"]) is not int or \
        value["fetch_schema"] != FETCH_SCHEMA or \
        value["gap_policy"] != GAP_POLICY
    ):
        raise ValueError("fetch report fields are invalid")
    metadata = {
        "schema": manifest.schema,
        "purpose": manifest.purpose,
        "declared_on": str(manifest.declared_on),
        "eligibility_date": str(manifest.eligibility_date),
        "start": str(manifest.start),
        "end": str(manifest.end),
        "interval_minutes": manifest.interval_minutes,
        "adjusted": manifest.adjusted,
        "session": manifest.session,
    }
    provenance = value["manifest"]
    if any(value[name] != item for name, item in metadata.items()) or \
       provenance != {
           "path": str(manifest_input.source),
           "sha256": manifest_input.sha256,
       }:
        raise ValueError("fetch report does not match its manifest")
    records = value["series"]
    if not isinstance(records, list) or len(records) != len(manifest.series):
        raise ValueError("fetch report series are invalid")
    for spec, record in zip(manifest.series, records, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "ticker", "stratum", "reference", "aggregate", "csv",
        } or record["ticker"] != spec.ticker or \
           record["stratum"] != spec.stratum:
            raise ValueError("fetch report series order is invalid")
        reference = {
            "path": f"/v3/reference/tickers/{quote(spec.ticker)}",
            "query": {"date": str(manifest.eligibility_date)},
            "active": True, "market": "stocks", "locale": "us",
            "type": "CS", "currency_name": "usd",
        }
        aggregate = {
            "path": (
                f"/v2/aggs/ticker/{quote(spec.ticker)}/range/"
                f"{manifest.interval_minutes}/minute/"
                f"{manifest.start}/{manifest.end}"
            ),
            "query": {
                "adjusted": str(manifest.adjusted).lower(),
                "sort": "asc", "limit": "50000",
            },
        }
        csv = record["csv"]
        series_bars = bars[spec.ticker]
        csv_fields = {
            "path", "rows", "sessions", "source_rows", "sha256",
        } | ({"gap_audit"} if audited else set())
        if record["reference"] != reference or record["aggregate"] != aggregate or \
           not isinstance(csv, dict) or set(csv) != csv_fields or \
           csv["path"] != series_bars.path or \
           csv["sha256"] != series_bars.sha256 or \
           _integer(csv["rows"], "CSV rows", 1) != len(series_bars.timestamps) or \
           _integer(csv["sessions"], "CSV sessions", 1) > csv["rows"] or \
           _integer(csv["source_rows"], "CSV source rows", 1) < csv["rows"]:
            raise ValueError("fetch request or CSV contract is invalid")
        if audited:
            sessions, audit = _gap_audit(
                series_bars.timestamps, manifest.interval_minutes,
            )
            supplied = json.dumps(
                csv["gap_audit"], allow_nan=False,
                separators=(",", ":"), sort_keys=True,
            )
            expected = json.dumps(
                audit, allow_nan=False, separators=(",", ":"), sort_keys=True,
            )
            if csv["sessions"] != sessions or supplied != expected:
                raise ValueError("fetch gap audit does not match its CSV")


def _expected_empty_summary() -> dict[str, object]:
    empty = {"count": 0, "mean": None, "stddev": None}
    return {
        model: {
            "by_series": {}, "return_macro_by_seed": {},
            "return_macro_across_seeds": {
                metric: dict(empty) for metric in (
                    "return_mse", "return_mae", "direction_accuracy",
                )
            },
        }
        for model in MODELS
    }


def _record_key(value: Mapping[str, object]) -> tuple[object, ...]:
    return value["model"], value["series"], value["seed"]


def _validate_record(value: object, calibration: bool) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("experiment record is invalid")
    expected = RECORD_FIELDS | ({"epochs"} if calibration else set())
    if value.get("seed") is not None and not calibration:
        expected |= {
            "best_validation_scaled_mse", "best_epoch", "epochs_trained",
        }
    if set(value) != expected or value["candidate"] != "raw-17" or \
       value["feature_set"] != "ohlcv" or value["model"] not in MODELS or \
       type(value["samples"]) is not int or value["samples"] < 1 or \
       not isinstance(value["targets"], dict) or \
       set(value["targets"]) != {"train", "validation", "test"} or \
       not isinstance(value["metrics"], dict) or \
       set(value["metrics"]) != METRIC_FIELDS:
        raise ValueError("experiment record contract is invalid")
    if calibration and (
        value["fold"] is not None or
        value["seed"] is None and value["epochs"] is not None or
        value["seed"] is not None and (
            type(value["epochs"]) is not int or value["epochs"] < 1
        )
    ):
        raise ValueError("calibration fold or epochs are invalid")
    for boundary in value["targets"].values():
        if not isinstance(boundary, list) or len(boundary) != 2 or \
           any(not isinstance(item, str) for item in boundary):
            raise ValueError("experiment target boundary is invalid")
    _finite_tree(value)
    return value


def validate_experiment(
    value: Mapping[str, object], config: Mapping[str, object] | None,
    config_input: FrozenInput | None, names: Sequence[str],
    bars: Mapping[str, Bars], ledger_input: FrozenInput,
    forecasts: Sequence[Forecast],
) -> None:
    if set(value) != REPORT_FIELDS or value["schema"] != 6:
        raise ValueError("experiment report schema is invalid")
    if value["protocol"] != EXPECTED_PROTOCOL:
        raise ValueError("experiment protocol is not the exact 429-run contract")
    if value["sweep"] != _expected_config():
        raise ValueError("experiment sweep is not raw-17")
    if config is not None and value["sweep"] != config:
        raise ValueError("experiment sweep differs from the config")
    if config_input is not None and value["sweep_input"] != {
        "path": str(config_input.source), "sha256": config_input.sha256,
    }:
        raise ValueError("experiment config provenance is invalid")
    elif config_input is None:
        sweep_input = value["sweep_input"]
        if not isinstance(sweep_input, dict) or set(sweep_input) != {
            "path", "sha256",
        } or not isinstance(sweep_input["path"], str) or \
           not _digest(sweep_input["sha256"]):
            raise ValueError("experiment config provenance is invalid")
    if value["test"] != [] or value["summary"] != _expected_empty_summary() or \
       "prediction_ledger" in value or "policies" in value:
        raise ValueError("calibration experiment opened a test result")

    series = value["series"]
    if not isinstance(series, list) or len(series) != len(names):
        raise ValueError("experiment series are invalid")
    for name, item in zip(names, series, strict=True):
        series_bars = bars[name]
        if item != {
            "name": name, "csv": series_bars.path,
            "rows": len(series_bars.timestamps), "sha256": series_bars.sha256,
            "first_timestamp": series_bars.timestamps[0],
            "last_timestamp": series_bars.timestamps[-1],
        }:
            raise ValueError("experiment series order or provenance is invalid")
    grid = value["test_contract"]
    if not isinstance(grid, list) or len(grid) != len(names):
        raise ValueError("experiment test grid is invalid")
    for name, item in zip(names, grid, strict=True):
        if not isinstance(item, dict) or set(item) != {
            "series", "samples", "first_target_time", "last_target_time",
        } or item["series"] != name or \
           _integer(item["samples"], "test samples", 1) < 1 or \
           item["first_target_time"] not in bars[name].timestamps or \
           item["last_target_time"] not in bars[name].timestamps:
            raise ValueError("experiment test grid is invalid")

    selection = value["selection"]
    if not isinstance(selection, dict) or set(selection) != set(MODELS):
        raise ValueError("experiment selection is invalid")
    for model in MODELS:
        selected = selection[model]
        if not isinstance(selected, dict) or set(selected) != {
            "candidate", "mean_validation_scaled_mse",
        } or selected["candidate"] != "raw-17":
            raise ValueError("experiment selection is invalid")
        _finite(selected["mean_validation_scaled_mse"], "selection score")

    expected_validation = [
        (model, name, seed, fold)
        for name in names for fold in range(2) for model in MODELS
        for seed in (SEEDS if model in SEEDED_MODELS else (None,))
    ]
    validation = value["validation"]
    if not isinstance(validation, list) or len(validation) != 286:
        raise ValueError("experiment validation grid is invalid")
    actual_validation = []
    for item in validation:
        record = _validate_record(item, False)
        actual_validation.append((*_record_key(record), record["fold"]))
    if actual_validation != expected_validation:
        raise ValueError("experiment validation grid is reordered or incomplete")

    expected_calibration = [
        (model, name, seed)
        for model in MODELS for name in names
        for seed in (SEEDS if model in SEEDED_MODELS else (None,))
    ]
    calibration = value["calibration"]
    if not isinstance(calibration, list) or len(calibration) != 143:
        raise ValueError("experiment calibration grid is invalid")
    if [_record_key(_validate_record(item, True)) for item in calibration] != \
            expected_calibration:
        raise ValueError("experiment calibration grid is reordered or incomplete")

    fingerprints = value["model_fingerprints"]
    if not isinstance(fingerprints, list) or [
        (item.get("model"), item.get("series"), item.get("seed"))
        for item in fingerprints if isinstance(item, dict)
    ] != sorted(expected_calibration):
        raise ValueError("experiment model fingerprints are invalid")
    for item in fingerprints:
        if set(item) != {"model", "series", "seed", "epochs", "sha256"} or \
           not _digest(item["sha256"]) or item["seed"] is None and \
           item["epochs"] is not None or item["seed"] is not None and \
           (type(item["epochs"]) is not int or item["epochs"] < 1):
            raise ValueError("experiment model fingerprints are invalid")

    metadata = value["calibration_prediction_ledger"]
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema", "path", "records", "sha256",
    } or metadata != {
        "schema": 3, "path": str(ledger_input.source),
        "records": len(forecasts), "sha256": ledger_input.sha256,
    }:
        raise ValueError("experiment calibration ledger provenance is invalid")


def read_ledger(frozen: FrozenInput) -> tuple[Forecast, ...]:
    raw = frozen.snapshot.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise ValueError("calibration ledger is empty")
    for line in lines:
        try:
            value = json.loads(
                line, object_pairs_hook=_object, parse_constant=_constant,
            )
            expected = json.dumps(
                value, allow_nan=False, sort_keys=True,
            ) + "\n"
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("calibration ledger line is invalid") from error
        if line != expected:
            raise ValueError("calibration ledger is not canonical")
    return read_forecasts(frozen.snapshot)


def validate_prediction_grid(
    forecasts: Sequence[Forecast], experiment: Mapping[str, object],
    names: Sequence[str], bars: Mapping[str, Bars],
) -> dict[str, dict[str, float]]:
    if any(
        item.split != "calibration" or item.fold is not None or
        item.candidate != "raw-17" or item.feature_set != "ohlcv" or
        item.horizon_bars != 13 or item.target_kind != EXECUTABLE_RETURN_TARGET
        for item in forecasts
    ):
        raise ValueError("ledger contains a non-calibration prediction")
    groups: dict[tuple[str, str, int | None], list[Forecast]] = defaultdict(list)
    order: list[tuple[str, str, int | None]] = []
    for forecast in forecasts:
        key = (forecast.model, forecast.series, forecast.seed)
        if key not in groups:
            order.append(key)
        groups[key].append(forecast)
    calibration = experiment["calibration"]
    expected = [_record_key(item) for item in calibration]
    if order != expected or set(groups) != set(expected):
        raise ValueError("ledger prediction grid is reordered or incomplete")
    actuals: dict[str, dict[str, float]] = {name: {} for name in names}
    for record in calibration:
        key = _record_key(record)
        rows = groups[key]
        if [item.target_time for item in rows] != sorted(
            item.target_time for item in rows
        ) or len(rows) != record["samples"] or \
           [rows[0].target_time, rows[-1].target_time] != \
           record["targets"]["validation"]:
            raise ValueError("ledger does not match calibration boundaries")
        series_bars = bars[str(record["series"])]
        indexes = {
            timestamp: index
            for index, timestamp in enumerate(series_bars.timestamps)
        }
        for item in rows:
            if item.csv_sha256 != series_bars.sha256:
                raise ValueError("ledger CSV hash is invalid")
            try:
                as_of = indexes[item.as_of]
                target = indexes[item.target_time]
            except KeyError as error:
                raise ValueError("prediction timestamp is missing from its CSV") \
                    from error
            if target != as_of + 13 or as_of + 1 >= len(series_bars.opens):
                raise ValueError("prediction timestamps do not match horizon 13")
            actual = math.log(
                series_bars.closes[target] / series_bars.opens[as_of + 1]
            )
            previous = actuals[item.series].setdefault(item.target_time, actual)
            if previous != actual:
                raise ValueError("models do not share one actual target")
    return actuals


def validate_one_policy(
    value: object, model: str, names: Sequence[str],
    experiment: Mapping[str, object], experiment_input: FrozenInput,
    ledger_input: FrozenInput, forecasts: Sequence[Forecast],
) -> dict[str, object]:
    policy = validate_policy(value)
    expected_fingerprints = [
        item for item in experiment["model_fingerprints"]
        if item["model"] == model
    ]
    selected_records = sum(
        item.model == model and item.candidate == "raw-17" and
        item.feature_set == "ohlcv" and item.horizon_bars == 13
        for item in forecasts
    )
    if policy["model"] != model or policy["candidate"] != "raw-17" or \
       policy["feature_set"] != "ohlcv" or \
       policy["target_kind"] != EXECUTABLE_RETURN_TARGET or \
       policy["horizon_bars"] != 13 or \
       policy["seeds"] != (list(SEEDS) if model in SEEDED_MODELS else []) or \
       policy["series"] != sorted(names) or \
       policy["initial_cash"] != 100 or policy["costs"] != {
           "spread_bps": 1.0, "slippage_bps": 1.0, "fee_bps": 0.0,
       } or policy["calibration_report"] != {
           "path": str(experiment_input.source),
           "sha256": experiment_input.sha256,
       } or policy["calibration_prediction_ledger"] != {
           "path": str(ledger_input.source), "sha256": ledger_input.sha256,
           "source_records": len(forecasts),
           "selected_records": selected_records,
       } or policy["test_grid"] != experiment["test_contract"] or \
       policy["model_fingerprints"] != expected_fingerprints or \
       policy["calibration_fingerprint"] != \
       experiment_fingerprint(experiment):
        raise ValueError("policy does not exactly match the experiment contract")
    return policy


def selected_forecasts(
    forecasts: Sequence[Forecast], policy: Mapping[str, object],
) -> tuple[Forecast, ...]:
    selected = tuple(
        item for item in forecasts
        if item.model == policy["model"] and
        item.candidate == policy["candidate"] and
        item.feature_set == policy["feature_set"] and
        item.horizon_bars == policy["horizon_bars"] and
        item.target_kind == policy["target_kind"]
    )
    if not selected:
        raise ValueError("policy selected no calibration forecasts")
    return selected


def build_replay_report(
    policy: Mapping[str, object], policy_input: FrozenInput,
    experiment_input: FrozenInput, ledger_input: FrozenInput,
    forecasts: Sequence[Forecast], bars: Mapping[str, Bars],
) -> dict[str, object]:
    selected = selected_forecasts(forecasts, policy)
    ordered = {name: bars[name] for name in policy["series"]}
    costs = Costs(*(policy["costs"][field] for field in POLICY_COST_FIELDS))
    report = run_backtests(
        selected, ordered, float(policy["initial_cash"]), costs,
        float(policy["safety_bps"] or 0.0), True, policy["seeds"],
        policy["action"] == "cash",
        disagreement_lambda=policy_disagreement_lambda(policy),
    )
    report.update({
        "evidence": EVIDENCE,
        "prediction_ledger": {
            "path": str(ledger_input.source), "sha256": ledger_input.sha256,
            "source_records": len(forecasts),
            "selected_records": len(selected),
        },
        "policy": {
            "path": str(policy_input.source), "sha256": policy_input.sha256,
        },
        "experiment_report": {
            "path": str(experiment_input.source),
            "sha256": experiment_input.sha256,
        },
    })
    return report


def _ensemble(
    forecasts: Sequence[Forecast],
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str, str], list[Forecast]] = defaultdict(list)
    for item in forecasts:
        grouped[(item.model, item.series, item.target_time)].append(item)
    result: dict[str, dict[str, dict[str, float]]] = {
        model: {} for model in MODELS
    }
    for (model, series, target), rows in grouped.items():
        expected = SEEDS if model in SEEDED_MODELS else (None,)
        if tuple(item.seed for item in rows) != expected:
            raise ValueError("prediction ensemble does not use the exact seed set")
        result[model].setdefault(series, {})[target] = fmean(
            item.predicted_log_return for item in rows
        )
    return result


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def date_block_bootstrap(
    values: Mapping[str, Mapping[str, Sequence[float]]],
) -> list[float]:
    if not values:
        raise ValueError("bootstrap values must be nonempty")
    dates = tuple(sorted(set.intersection(
        *(set(by_date) for by_date in values.values())
    )))
    if not dates or any(
           not by_date[date] for by_date in values.values()
           for date in dates
       ):
        raise ValueError("bootstrap stocks need aligned nonempty dates")
    prepared = [
        tuple(
            (sum(float(item) for item in by_date[date]), len(by_date[date]))
            for date in dates
        )
        for by_date in values.values()
    ]
    generator = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected: list[int] = []
        while len(selected) < len(dates):
            start = generator.randrange(len(dates))
            selected.extend(
                (start + offset) % len(dates)
                for offset in range(BOOTSTRAP_BLOCK_DATES)
            )
        selected = selected[:len(dates)]
        stock_means = []
        for stock in prepared:
            total = sum(stock[index][0] for index in selected)
            count = sum(stock[index][1] for index in selected)
            stock_means.append(total / count)
        samples.append(sum(stock_means) / len(stock_means))
    samples.sort()
    return [
        samples[int(0.025 * (BOOTSTRAP_REPLICATES - 1))],
        samples[int(0.975 * (BOOTSTRAP_REPLICATES - 1))],
    ]


def effective_count(
    returns: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    names = sorted(returns)
    dates = sorted(
        set.intersection(*(set(returns[name]) for name in names))
        if names else set()
    )
    if len(dates) < 2:
        return {
            "value": None, "included": names, "excluded": [],
            "reason": "fewer-than-two-aligned-dates",
        }
    columns = {
        name: [float(returns[name][date]) for date in dates]
        for name in names
    }
    if any(not all(math.isfinite(item) for item in values)
           for values in columns.values()):
        raise ValueError("daily policy returns must be finite")
    excluded = [
        name for name, values in columns.items()
        if all(item == values[0] for item in values[1:])
    ]
    included = [name for name in names if name not in excluded]
    if len(included) < 2:
        return {
            "value": None, "included": included, "excluded": excluded,
            "reason": "fewer-than-two-nonconstant-stocks",
        }
    means = {name: fmean(columns[name]) for name in included}
    covariance = {
        (left, right): sum(
            (columns[left][index] - means[left]) *
            (columns[right][index] - means[right])
            for index in range(len(dates))
        ) / (len(dates) - 1)
        for left in included for right in included
    }
    trace = sum(covariance[name, name] for name in included)
    denominator = sum(covariance.values())
    if denominator <= 0.0 or not math.isfinite(denominator):
        return {
            "value": None, "included": included, "excluded": excluded,
            "reason": "nonpositive-or-nonfinite-denominator",
        }
    value = len(included) * trace / denominator
    if not math.isfinite(value):
        return {
            "value": None, "included": included, "excluded": excluded,
            "reason": "nonpositive-or-nonfinite-denominator",
        }
    return {
        "value": value, "included": included, "excluded": excluded,
        "reason": None,
    }


def evaluate_gates(
    per_model: Mapping[str, Mapping[str, float]],
    majority: float, close_improvement: float,
) -> dict[str, object]:
    transformer = per_model["transformer"]
    baselines = {
        model: per_model[model]["return_mae"]
        for model in MODELS if model != "transformer"
    }
    return_mae = transformer["return_mae"] < min(baselines.values())
    direction = transformer["direction"] > majority
    close = close_improvement > 0.0
    return {
        "return_mae": {
            "pass": return_mae,
            "transformer": transformer["return_mae"],
            "baselines": baselines,
        },
        "direction": {
            "pass": direction,
            "transformer": transformer["direction"],
            "majority": majority,
        },
        "close_mae": {
            "pass": close, "mean_relative_improvement": close_improvement,
        },
        "all_pass": return_mae and direction and close,
    }


def forecast_metrics(
    forecasts: Sequence[Forecast], actuals: Mapping[str, Mapping[str, float]],
    bars: Mapping[str, Bars],
) -> tuple[dict[str, object], dict[str, object]]:
    predictions = _ensemble(forecasts)
    per_stock: dict[str, object] = {}
    model_stock: dict[str, list[tuple[float, float]]] = {
        model: [] for model in MODELS
    }
    paired_dates: dict[
        str, dict[str, dict[str, list[float]]]
    ] = {
        model: {} for model in MODELS if model != "transformer"
    }
    close_improvements = []
    for series, actual_by_target in actuals.items():
        targets = tuple(sorted(actual_by_target))
        if not targets:
            raise ValueError("stock has no actual calibration targets")
        actual_values = [actual_by_target[target] for target in targets]
        up = sum(value > 0.0 for value in actual_values) / len(actual_values)
        down = sum(value < 0.0 for value in actual_values) / len(actual_values)
        flat = sum(value == 0.0 for value in actual_values) / len(actual_values)
        majority = max(up, down, flat)
        stock_models = {}
        errors: dict[str, list[float]] = {}
        for model in MODELS:
            if tuple(sorted(predictions[model].get(series, {}))) != targets:
                raise ValueError("model prediction grids do not align")
            values = [predictions[model][series][target] for target in targets]
            errors[model] = [
                abs(prediction - actual)
                for prediction, actual in zip(values, actual_values, strict=True)
            ]
            mae = fmean(errors[model])
            direction = fmean(
                _sign(prediction) == _sign(actual)
                for prediction, actual in zip(values, actual_values, strict=True)
            )
            stock_models[model] = {"return_mae": mae, "direction": direction}
            model_stock[model].append((mae, direction))
        indexes = {
            timestamp: index
            for index, timestamp in enumerate(bars[series].timestamps)
        }
        transformer_closes, zero_closes = [], []
        for target in targets:
            target_index = indexes[target]
            entry_open = bars[series].opens[target_index - 12]
            actual_close = bars[series].closes[target_index]
            predicted_close = entry_open * math.exp(
                predictions["transformer"][series][target]
            )
            transformer_closes.append(abs(predicted_close - actual_close))
            zero_closes.append(abs(entry_open - actual_close))
        zero_mae = fmean(zero_closes)
        if zero_mae <= 0.0 or not math.isfinite(zero_mae):
            raise ValueError("zero-return close MAE denominator is invalid")
        relative = (zero_mae - fmean(transformer_closes)) / zero_mae
        close_improvements.append(relative)
        per_stock[series] = {
            "samples": len(targets), "models": stock_models,
            "majority": {
                "p_up": up, "p_down": down, "p_flat": flat,
                "direction": majority,
            },
            "close_relative_improvement": relative,
        }
        for baseline in paired_dates:
            paired_dates[baseline][series] = defaultdict(list)
            for target, base, transformer in zip(
                targets, errors[baseline], errors["transformer"], strict=True,
            ):
                paired_dates[baseline][series][target[:10]].append(
                    base - transformer
                )
    per_model = {
        model: {
            "return_mae": fmean(item[0] for item in values),
            "direction": fmean(item[1] for item in values),
        }
        for model, values in model_stock.items()
    }
    macro_majority = fmean(
        value["majority"]["direction"] for value in per_stock.values()
    )
    paired = {}
    for baseline, by_stock_dates in paired_dates.items():
        deltas = {
            series: fmean(
                item for values in by_date.values() for item in values
            )
            for series, by_date in by_stock_dates.items()
        }
        paired[baseline] = {
            "per_stock": deltas,
            "counts": {
                "wins": sum(value > 0.0 for value in deltas.values()),
                "ties": sum(value == 0.0 for value in deltas.values()),
                "losses": sum(value < 0.0 for value in deltas.values()),
            },
            "date_block_bootstrap_ci": date_block_bootstrap(by_stock_dates),
        }
    mean_close = fmean(close_improvements)
    forecast = {
        "per_model": per_model, "per_stock": per_stock,
        "macro_majority_direction": macro_majority,
        "mean_close_relative_improvement": mean_close,
        "paired": paired,
    }
    return forecast, evaluate_gates(per_model, macro_majority, mean_close)


def policy_metrics(
    replays: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    models = {}
    for model, replay in replays.items():
        terminal = {
            strategy: {
                result["series"]: result["strategies"][strategy]["final_equity"]
                for result in replay["results"]
            }
            for strategy in (
                "forecast_long_cash", "cash", "buy_and_hold", "always_up",
            )
        }
        aggregates = {
            strategy: {
                "arithmetic_mean": fmean(values.values()),
                "geometric_mean": math.exp(
                    fmean(math.log(value) for value in values.values())
                ),
            }
            for strategy, values in terminal.items()
        }
        forecast_growth = fmean(
            math.log(value / 100.0)
            for value in terminal["forecast_long_cash"].values()
        )
        excess = {
            baseline: forecast_growth - fmean(
                math.log(value / 100.0)
                for value in terminal[baseline].values()
            )
            for baseline in ("cash", "buy_and_hold", "always_up")
        }
        models[model] = {
            "terminal_equity": terminal, "aggregates": aggregates,
            "forecast_excess_mean_log_growth": excess,
        }
    return {"evidence": EVIDENCE, "models": models}


def _daily_returns(replay: Mapping[str, object]) -> dict[str, dict[str, float]]:
    return {
        result["series"]: {
            item["period"]: item["period_return"]
            for item in result["strategies"]["forecast_long_cash"]["periods"][
                "daily"
            ]
        }
        for result in replay["results"]
    }


def analyze(
    manifest_input: FrozenInput, config_input: FrozenInput,
    named: Mapping[str, FrozenInput], csv_inputs: Sequence[FrozenInput],
) -> tuple[dict[str, object], bool]:
    manifest = UniverseManifest.read(manifest_input.snapshot)
    names = _manifest_names(manifest)
    config = validate_config(read_json(config_input.snapshot))
    csv_dir = (named["run_dir"].source / "csv").resolve()
    expected_paths = {
        csv_dir / f"{name.lower()}-{manifest.interval_minutes}m.csv"
        for name in names
    }
    by_path = {item.source: item for item in csv_inputs}
    if set(by_path) != expected_paths or len(by_path) != len(expected_paths):
        raise ValueError("CSV files are missing, extra, or reordered")
    bars = {
        name: load_frozen_bars(by_path[
            csv_dir / f"{name.lower()}-{manifest.interval_minutes}m.csv"
        ])
        for name in names
    }
    fetch = read_json(named["fetch"].snapshot, canonical=True)
    validate_fetch(fetch, manifest, manifest_input, bars)
    ledger = read_ledger(named["ledger"])
    experiment = read_json(named["experiment"].snapshot, canonical=True)
    validate_experiment(
        experiment, config, config_input, names, bars, named["ledger"], ledger,
    )
    actuals = validate_prediction_grid(ledger, experiment, names, bars)
    replays = {}
    for model in POLICY_MODELS:
        policy_input = named[f"policy-{model}"]
        policy_value = read_json(policy_input.snapshot, canonical=True)
        policy = validate_one_policy(
            policy_value, model, names, experiment, named["experiment"],
            named["ledger"], ledger,
        )
        replay = read_json(named[f"backtest-{model}"].snapshot, canonical=True)
        expected = build_replay_report(
            policy, policy_input, named["experiment"], named["ledger"],
            ledger, bars,
        )
        if replay != expected:
            raise ValueError("policy replay is stale, reordered, or mismatched")
        replays[model] = replay
    forecast, gates = forecast_metrics(ledger, actuals, bars)
    policy = policy_metrics(replays)
    n_eff = effective_count(_daily_returns(replays["transformer"]))
    input_report = {
        "manifest": {
            "path": str(manifest_input.source), "sha256": manifest_input.sha256,
        },
        "config": {
            "path": str(config_input.source), "sha256": config_input.sha256,
        },
        "fetch_report": {
            "path": str(named["fetch"].source), "sha256": named["fetch"].sha256,
        },
        "experiment": {
            "path": str(named["experiment"].source),
            "sha256": named["experiment"].sha256,
        },
        "calibration_ledger": {
            "path": str(named["ledger"].source),
            "sha256": named["ledger"].sha256,
        },
        "policies": {
            model: {
                "path": str(named[f"policy-{model}"].source),
                "sha256": named[f"policy-{model}"].sha256,
            }
            for model in POLICY_MODELS
        },
        "replays": {
            model: {
                "path": str(named[f"backtest-{model}"].source),
                "sha256": named[f"backtest-{model}"].sha256,
            }
            for model in POLICY_MODELS
        },
        "csv": [
            {"path": str(item.source), "sha256": item.sha256}
            for item in csv_inputs
        ],
    }
    passed = bool(gates["all_pass"])
    return {
        "schema": 1,
        "status": "pass" if passed else "gate-failure",
        "inputs": input_report,
        "protocol": {
            "ordering": "manifest/fetch/experiment; sorted policy/replay",
            "seed_ensemble":
                "arithmetic mean before stock/timestamp pairing",
            "macro_unit": "stock",
            "majority_reference": "unique actual calibration targets",
            "bootstrap": {
                "unit": "five-date circular block", "replicates": 10_000,
                "seed": BOOTSTRAP_SEED,
            },
            "policy_evidence": EVIDENCE,
            "n_eff_formula": "N * trace(S) / (1' S 1)",
        },
        "forecast": forecast,
        "policy_resubstitution": policy,
        "n_eff": n_eff,
        "gates": gates,
    }, passed


def _sources(
    manifest: Path, config: Path, run_dir: Path, output: Path,
) -> tuple[
    list[Path], tuple[Path, ...], Path, DirectoryMembership,
]:
    resolved_output = resolve_fresh_output(output)
    membership = directory_membership(run_dir)
    expected = {"csv", *REPORT_FILES}
    if {name for name, _identity_ in membership.run_files} | {"csv"} != \
            expected:
        raise ValueError("run directory has missing or extra artifacts")
    csv_paths = tuple(sorted(path for path, _identity_ in membership.csv_files))
    sources = [
        manifest, config, *(run_dir / name for name in REPORT_FILES),
        *csv_paths,
    ]
    regular_file_identities(sources)
    require_disjoint(sources, [resolved_output])
    return sources, csv_paths, resolved_output, membership


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        sources, csv_paths, output, membership = _sources(
            args.manifest, args.config, args.run_dir, args.output,
        )
        with freeze_inputs(sources) as frozen:
            by_source = dict(zip(sources, frozen, strict=True))
            named = {
                "run_dir": FrozenInput(args.run_dir, args.run_dir, ""),
                "fetch": by_source[args.run_dir / "fetch-report.json"],
                "experiment": by_source[args.run_dir / "experiment.json"],
                "ledger": by_source[args.run_dir / "calibration.jsonl"],
                **{
                    f"policy-{model}":
                        by_source[args.run_dir / f"policy-{model}.json"]
                    for model in POLICY_MODELS
                },
                **{
                    f"backtest-{model}":
                        by_source[args.run_dir / f"backtest-{model}.json"]
                    for model in POLICY_MODELS
                },
            }
            report, passed = analyze(
                by_source[args.manifest], by_source[args.config], named,
                tuple(by_source[path] for path in csv_paths),
            )
            verify_frozen(frozen)
            verify_membership(args.run_dir, membership)
            _absent(output)
            write_json(output, report)
    except (
        IndexError, KeyError, OSError, OverflowError, TypeError,
        UnicodeError, ValueError,
    ) as error:
        print(f"integrity error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps({"output": str(output),
                      "status": report["status"]}, sort_keys=True))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
