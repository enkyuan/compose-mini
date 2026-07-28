"""Evaluate the predeclared six-session SPY-residual diagnostic."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
import hashlib
import json
import os

from tools.analyze_spy_residual_shrinkage import (
    _leave_one_out_r2, _paired_mse_metrics, pooled_r2,
)
from tools.files import file_sha256, freeze_inputs, verify_frozen
from tools.finalize_spy_residual import _day, _finite, _population_std
from tools.panel_contract import (
    _absent, _directory_identity, _exact_json, _sha256,
    read_canonical_json, read_canonical_json_lines, selected_source_tree,
)
from tools.relative_context_contract import (
    HORIZON_BARS, SEEDS, ResidualTruthRow,
)
from tools.spy_residual_forward_contract import (
    FORWARD_CALENDAR, FORWARD_CONFIG, FORWARD_RUN_ID, FORWARD_SOURCE_PATHS,
    FORWARD_SOURCES, FORWARD_UNIVERSE, STATE_FINGERPRINTS,
    expected_forward_protocol,
)
from tools.spy_residual_forward_inputs import (
    CandidateLedger, ForwardSeriesPrediction, SeedPrediction, TruthReader,
    TARGET_SESSIONS, _json_line, _prediction_record, _private_identity,
)
from tools.spy_residual_gate import SPY_DIRECTION_SCALE

ROOT = Path(__file__).resolve().parents[1]
FORWARD_RUN_DIR = ROOT / "reports" / FORWARD_RUN_ID
FORWARD_CANDIDATE = FORWARD_RUN_DIR / "candidate.jsonl"
FORWARD_TRUTH_RECEIPT = FORWARD_RUN_DIR / "truth-access.json"
CANDIDATE = "spy-direction-gated-five-seed-mean"
UNCHANGED = "unchanged-five-seed-mean"
ZERO = "zero"
BATCHES = TARGET_SESSIONS * HORIZON_BARS


def _fixed_binding(
    name: str, values: Sequence[tuple[str, str, str]],
) -> dict[str, str]:
    try:
        _, path, sha256 = next(item for item in values if item[0] == name)
    except StopIteration as error:
        raise ValueError("forward provenance binding is missing") from error
    return {"path": path, "sha256": sha256}


def _provenance(value: Mapping[str, object]) -> None:
    """Validate the exact fixed-data closure carried by the candidate."""
    historical = value["historical"]
    sources = value["sources"]
    historical_fields = {
        "attempt", "calibration_fits", "evaluation_grid_sha256",
        "residual_phase_sha256", "runtime_source_tree_sha256",
        "scaler_inputs_sha256", "source_phase_sha256",
        "training_grid_sha256",
    }
    if not isinstance(historical, Mapping) or \
       set(historical) != historical_fields or \
       not _exact_json(
           historical["attempt"],
           _fixed_binding("residual_attempt", FORWARD_SOURCES),
       ) or not _exact_json(
           historical["calibration_fits"],
           _fixed_binding("calibration_fits", FORWARD_SOURCES),
       ) or not isinstance(sources, Mapping) or set(sources) != {
           "forward_calendar", "future_bundle_report",
       } or not _exact_json(
           sources["forward_calendar"],
           _fixed_binding("forward_calendar", (FORWARD_CALENDAR,)),
       ):
        raise ValueError("forward candidate provenance changed")
    for name in historical_fields - {"attempt", "calibration_fits"}:
        _sha256(historical[name], f"forward historical {name}")
    report = sources["future_bundle_report"]
    if not isinstance(report, Mapping) or set(report) != {
        "path", "sha256",
    } or not isinstance(report["path"], str):
        raise ValueError("forward future bundle provenance changed")
    path = Path(report["path"])
    if path.name != "fetch.json" or not path.is_absolute() or \
       path != Path(os.path.abspath(path)):
        raise ValueError("forward future bundle provenance changed")
    _sha256(report["sha256"], "forward future bundle report")


@dataclass(frozen=True, slots=True)
class _CandidateRows:
    triples: tuple[tuple[str, str, str], ...]
    predictions: Mapping[str, Mapping[str, tuple[float, ...]]]
    raw: Mapping[str, tuple[tuple[float, ...], ...]]
    regimes: tuple[str, ...]


def _json_value(value: object) -> object:
    """Copy one finite value into JSON-native containers."""
    try:
        return json.loads(json.dumps(
            value, allow_nan=False, sort_keys=True,
        ))
    except (TypeError, ValueError) as error:
        raise ValueError("forward report value is not finite JSON") from error


def _prediction_rows(
    records: Sequence[Mapping[str, object]],
) -> _CandidateRows:
    rows = tuple(records) if not isinstance(records, (str, bytes)) else ()
    width = len(FORWARD_UNIVERSE)
    if len(rows) != BATCHES * width:
        raise ValueError("forward candidate has the wrong observation count")
    gated = {series: [] for series in FORWARD_UNIVERSE}
    means = {series: [] for series in FORWARD_UNIVERSE}
    raw = {series: [] for series in FORWARD_UNIVERSE}
    triples, regimes = [], []
    for start in range(0, len(rows), width):
        batch = rows[start:start + width]
        first = batch[0]
        try:
            triple = tuple(first[name] for name in (
                "as_of", "entry", "target",
            ))
            regime = first["regime"]
        except (KeyError, TypeError) as error:
            raise ValueError("forward candidate row is invalid") from error
        if len(triple) != 3 or any(
            type(value) is not str or not value for value in triple
        ) or not triple[0] < triple[1] <= triple[2] or \
           type(regime) is not str or any(
               row.get("series") != series or
               tuple(row.get(name) for name in (
                   "as_of", "entry", "target",
               )) != triple or row.get("regime") != regime
               for row, series in zip(
                   batch, FORWARD_UNIVERSE, strict=True,
               )
           ):
            raise ValueError("forward candidate grid changed")
        for row, series in zip(batch, FORWARD_UNIVERSE, strict=True):
            values = row.get("raw_predictions")
            if not isinstance(values, list) or len(values) != len(SEEDS):
                raise ValueError("forward candidate seeds changed")
            try:
                seeds = tuple(
                    SeedPrediction(
                        item["seed"], item["state_fingerprint"],
                        item["prediction"],
                    )
                    for item in values if isinstance(item, Mapping)
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("forward candidate seeds changed") from error
            prediction = ForwardSeriesPrediction(series, seeds)
            if len(seeds) != len(SEEDS) or not _exact_json(
                row, _prediction_record(prediction, triple, regime),
            ):
                raise ValueError("forward candidate row changed")
            raw[series].append(tuple(item.prediction for item in seeds))
            means[series].append(_finite(
                row["mean_prediction"], "forward ensemble prediction",
            ))
            gated[series].append(_finite(
                row["gated_prediction"], "forward gated prediction",
            ))
        triples.append(triple)
        regimes.append(regime)
    dates = tuple(dict.fromkeys(_day(row[2]) for row in triples))
    if len(set(triples)) != BATCHES or len(dates) != TARGET_SESSIONS or any(
        sum(_day(row[2]) == day for row in triples) !=
        BATCHES // TARGET_SESSIONS for day in dates
    ):
        raise ValueError("forward candidate session grid changed")
    return _CandidateRows(
        tuple(triples),
        {
            CANDIDATE: {
                series: tuple(gated[series]) for series in FORWARD_UNIVERSE
            },
            UNCHANGED: {
                series: tuple(means[series]) for series in FORWARD_UNIVERSE
            },
        },
        {series: tuple(raw[series]) for series in FORWARD_UNIVERSE},
        tuple(regimes),
    )


def _truth(
    values: Mapping[str, Sequence[ResidualTruthRow]],
    triples: Sequence[tuple[str, str, str]],
) -> dict[str, tuple[ResidualTruthRow, ...]]:
    if not isinstance(values, Mapping) or tuple(values) != FORWARD_UNIVERSE:
        raise ValueError("forward truth series order changed")
    expected = tuple(triples)
    result = {}
    for series in FORWARD_UNIVERSE:
        rows = tuple(values[series]) if not isinstance(
            values[series], (str, bytes),
        ) and isinstance(values[series], Sequence) else ()
        if len(rows) != len(expected) or any(
            not isinstance(row, ResidualTruthRow) or
            (row.as_of, row.entry, row.target) != triple
            for row, triple in zip(rows, expected, strict=True)
        ):
            raise ValueError(f"{series} forward truth grid changed")
        result[series] = rows
    return result


def _macro(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    predictions: Mapping[str, Sequence[float]],
    score: Callable[[float, float], float],
) -> float:
    return fmean(
        fmean(
            score(row.value, prediction)
            for row, prediction in zip(
                truth[series], predictions[series], strict=True,
            )
        )
        for series in FORWARD_UNIVERSE
    )


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _descriptive(
    truth: Mapping[str, Sequence[ResidualTruthRow]],
    rows: _CandidateRows,
) -> dict[str, object]:
    family = {
        **rows.predictions,
        ZERO: {
            series: (0.0,) * len(rows.triples)
            for series in FORWARD_UNIVERSE
        },
    }
    mae = lambda actual, predicted: abs(actual - predicted)
    direction = lambda actual, predicted: float(
        _sign(actual) == _sign(predicted)
    )

    def cells(regime: str) -> dict[str, object]:
        points = tuple(
            (truth[series][index].value,
             family[CANDIDATE][series][index],
             family[UNCHANGED][series][index])
            for series in FORWARD_UNIVERSE
            for index, label in enumerate(rows.regimes)
            if label == regime
        )
        if not points:
            return {
                "candidate_direction_accuracy": None,
                "candidate_mean_absolute_error": None,
                "mean_squared_error_gain_vs_unchanged_five_seed_mean": None,
                "mean_squared_error_gain_vs_zero": None,
                "observation_count": 0,
            }
        return {
            "candidate_direction_accuracy": fmean(
                direction(actual, candidate)
                for actual, candidate, _ in points
            ),
            "candidate_mean_absolute_error": fmean(
                mae(actual, candidate)
                for actual, candidate, _ in points
            ),
            "mean_squared_error_gain_vs_unchanged_five_seed_mean": fmean(
                (actual - unchanged) ** 2 - (actual - candidate) ** 2
                for actual, candidate, unchanged in points
            ),
            "mean_squared_error_gain_vs_zero": fmean(
                actual ** 2 - (actual - candidate) ** 2
                for actual, candidate, _ in points
            ),
            "observation_count": len(points),
        }

    dispersion = {
        series: tuple(map(_population_std, rows.raw[series]))
        for series in FORWARD_UNIVERSE
    }
    raw_dispersion = tuple(
        value for series in FORWARD_UNIVERSE for value in dispersion[series]
    )
    gated_dispersion = tuple(
        value * SPY_DIRECTION_SCALE
        if rows.regimes[index] == "nonnegative" else 0.0
        for series in FORWARD_UNIVERSE
        for index, value in enumerate(dispersion[series])
    )
    return {
        "direction_accuracy": {
            name: _macro(truth, predictions, direction)
            for name, predictions in family.items()
        },
        "market_regime_cells": {
            regime: cells(regime) for regime in ("negative", "nonnegative")
        },
        "mean_absolute_error": {
            name: _macro(truth, predictions, mae)
            for name, predictions in family.items()
        },
        "seed_dispersion": {
            "mean_gated_population_std": fmean(gated_dispersion),
            "mean_raw_population_std": fmean(raw_dispersion),
        },
    }


def evaluate_forward_candidate(
    records: Sequence[Mapping[str, object]],
    values: Mapping[str, Sequence[ResidualTruthRow]],
) -> dict[str, object]:
    """Report the fixed six-session diagnostic without making a decision."""
    rows = _prediction_rows(records)
    truth = _truth(values, rows.triples)
    zero = _paired_mse_metrics(
        truth, rows.predictions, CANDIDATE, ZERO, block_sessions=(5,),
    )
    unchanged = _paired_mse_metrics(
        truth, rows.predictions, CANDIDATE, UNCHANGED, block_sessions=(5,),
    )
    if any(
        comparison["date_count"] != TARGET_SESSIONS or
        tuple(comparison["intervals"]) != ("5",)
        for comparison in (zero, unchanged)
    ):
        raise ValueError("forward paired comparison grid changed")
    pooled = pooled_r2(truth, rows.predictions[CANDIDATE])
    without = _leave_one_out_r2(
        truth, rows.predictions[CANDIDATE],
    )
    protocol = expected_forward_protocol()
    descriptive = _descriptive(truth, rows)
    descriptive["nondecision_block_intervals"] = {
        reference: {
            "5": _json_value(comparison["intervals"]["5"]),
        }
        for reference, comparison in (
            (ZERO, zero), (UNCHANGED, unchanged),
        )
    }
    return {
        "diagnostic": {
            "paired_squared_error": {
                ZERO: _json_value(zero),
                UNCHANGED: _json_value(unchanged),
            },
            "pooled_raw_residual_r2_vs_zero": pooled,
            "pooled_raw_residual_r2_without_stock": without,
        },
        "interpretation": {
            "candidate_status":
                "unchanged-pending-confirmatory-forward-evidence",
            "output_role": "residual-only-not-executable-return",
            "policy": protocol["gates"]["policy"],
            "uncertainty_role":
                protocol["metrics"]["bootstrap"]["interpretation"],
        },
        "descriptive": descriptive,
        "locks": protocol["locks"],
        "sample": {
            "batches": len(rows.triples),
            "observation_count": len(records),
            "stock_count": len(FORWARD_UNIVERSE),
            "target_session_count": TARGET_SESSIONS,
        },
    }


def _candidate(
    claim: CandidateLedger, path: Path,
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]:
    if type(claim) is not CandidateLedger or \
       claim.path != FORWARD_CANDIDATE or \
       claim.records != BATCHES * len(FORWARD_UNIVERSE) or \
       claim.directory_identity != _directory_identity(claim.path.parent) or \
       claim.identity != _private_identity(claim.path) or \
       claim.sha256 != file_sha256(claim.path):
        raise ValueError("forward candidate binding changed")
    observed = read_canonical_json_lines(path)
    if len(observed) != claim.records + 1:
        raise ValueError("forward candidate closure changed")
    header, records = observed[0], observed[1:]
    fields = {
        "batches", "closure", "evidence_role", "grid_sha256", "records",
        "schema", "type",
    }
    if set(header) != fields or header.get("batches") != BATCHES or \
       header.get("records") != claim.records or \
       header.get("schema") != 1 or \
       header.get("type") != "spy-residual-forward-candidate" or \
       header.get("evidence_role") != "label-free-forward-candidate":
        raise ValueError("forward candidate header changed")
    grid = _sha256(header.get("grid_sha256"), "forward grid")
    closure = header.get("closure")
    closure_fields = {
        "future", "historical", "implementation_tree",
        "prediction_rows_sha256", "protocol", "run", "sources", "states",
    }
    if not isinstance(closure, Mapping) or set(closure) != closure_fields:
        raise ValueError("forward candidate provenance changed")
    _provenance(closure)
    expected_tree = _json_value(asdict(selected_source_tree(
        ROOT, FORWARD_SOURCE_PATHS,
    )))
    expected_states = [
        {"seed": seed, "state_fingerprint": fingerprint}
        for seed, fingerprint in zip(
            SEEDS, STATE_FINGERPRINTS, strict=True,
        )
    ]
    future = closure["future"]
    if not _exact_json(closure["implementation_tree"], expected_tree) or \
       not _exact_json(closure["protocol"], asdict(FORWARD_CONFIG)) or \
       not _exact_json(closure["run"], {
           "directory_identity": list(claim.directory_identity),
           "id": FORWARD_RUN_ID,
       }) or not _exact_json(closure["states"], expected_states) or \
       not isinstance(future, Mapping) or set(future) != {
           "feature_inputs_sha256", "gate_inputs_sha256",
           "timestamp_grid_sha256",
       } or _sha256(
           future["timestamp_grid_sha256"], "forward timestamp grid",
       ) != grid:
        raise ValueError("forward candidate provenance changed")
    _sha256(future["feature_inputs_sha256"], "forward features")
    _sha256(future["gate_inputs_sha256"], "forward gate inputs")
    digest = hashlib.sha256()
    for record in records:
        digest.update(_json_line(record).encode())
    if _sha256(
        closure["prediction_rows_sha256"], "forward prediction rows",
    ) != digest.hexdigest():
        raise ValueError("forward prediction rows changed")
    _prediction_rows(records)
    return header, records


def _receipt(
    candidate: CandidateLedger, grid_sha256: str, path: Path,
) -> Mapping[str, object]:
    expected = {
        "candidate": {
            "directory_identity": list(candidate.directory_identity),
            "identity": list(candidate.identity),
            "path": str(candidate.path),
            "records": candidate.records,
            "sha256": candidate.sha256,
        },
        "grid_sha256": grid_sha256,
        "schema": 1,
        "type": "spy-residual-forward-truth-access",
    }
    observed = read_canonical_json(path)
    if not _exact_json(observed, expected):
        raise ValueError("forward truth access receipt changed")
    return observed


@contextmanager
def finalize_forward_run(
    candidate: CandidateLedger, read_truth: TruthReader,
) -> Iterator[tuple[
    dict[str, object], Callable[[Mapping[str, object]], None],
]]:
    """Keep candidate and receipt evidence frozen until publication."""
    if type(candidate) is not CandidateLedger or not callable(read_truth):
        raise TypeError("forward finalization inputs are invalid")
    _absent(candidate.path.with_name("outcome.json"), "forward outcome")
    _absent(FORWARD_TRUTH_RECEIPT, "forward truth access receipt")
    with freeze_inputs((candidate.path,)) as frozen_candidate:
        if frozen_candidate[0].sha256 != candidate.sha256:
            raise ValueError("forward candidate snapshot changed")
        header, records = _candidate(
            candidate, frozen_candidate[0].snapshot,
        )
        truth = read_truth(candidate)
        with freeze_inputs((FORWARD_TRUTH_RECEIPT,)) as frozen_receipt:
            receipt_identity = _private_identity(FORWARD_TRUTH_RECEIPT)
            _receipt(
                candidate, header["grid_sha256"],
                frozen_receipt[0].snapshot,
            )
            metrics = evaluate_forward_candidate(records, truth)
            closure = header["closure"]
            value = {
                "descriptive": metrics["descriptive"],
                "diagnostic": metrics["diagnostic"],
                "evidence_role":
                    "predeclared-expedited-forward-diagnostic-terminal",
                "inputs": {
                    "candidate": {
                        "directory_identity":
                            list(candidate.directory_identity),
                        "identity": list(candidate.identity),
                        "path": str(candidate.path),
                        "records": candidate.records,
                        "sha256": candidate.sha256,
                    },
                    "truth_access": {
                        "identity": list(receipt_identity),
                        "path": str(FORWARD_TRUTH_RECEIPT),
                        "sha256": frozen_receipt[0].sha256,
                    },
                },
                "integrity": {
                    "config_sha256": FORWARD_CONFIG.sha256,
                    "grid_sha256": header["grid_sha256"],
                    "implementation_tree_sha256":
                        closure["implementation_tree"]["sha256"],
                    "prediction_rows_sha256":
                        closure["prediction_rows_sha256"],
                    "transformer_states": closure["states"],
                },
                "interpretation": metrics["interpretation"],
                "locks": metrics["locks"],
                "run": {
                    "batches": BATCHES,
                    "id": FORWARD_RUN_ID,
                    "observation_count": candidate.records,
                    "stock_count": len(FORWARD_UNIVERSE),
                    "target_session_count": TARGET_SESSIONS,
                },
                "sample": metrics["sample"],
                "schema": 1,
                "type": "spy-residual-forward-outcome",
            }
            value = _json_value(value)
            if not isinstance(value, dict):
                raise ValueError("forward outcome is not an object")
            sealed = _json_line(value)

            def verify(observed: Mapping[str, object]) -> None:
                if _json_line(observed) != sealed:
                    raise ValueError("forward finalized outcome changed")
                if candidate.directory_identity != _directory_identity(
                    candidate.path.parent,
                ) or candidate.identity != _private_identity(
                    candidate.path,
                ) or receipt_identity != _private_identity(
                    FORWARD_TRUTH_RECEIPT,
                ) or file_sha256(candidate.path) != candidate.sha256 or \
                   file_sha256(FORWARD_TRUTH_RECEIPT) != \
                        frozen_receipt[0].sha256:
                    raise ValueError("forward terminal evidence changed")
                verify_frozen((*frozen_candidate, *frozen_receipt))
                _candidate(candidate, frozen_candidate[0].snapshot)
                _receipt(
                    candidate, header["grid_sha256"],
                    frozen_receipt[0].snapshot,
                )

            verify(value)
            yield value, verify
            verify(value)
