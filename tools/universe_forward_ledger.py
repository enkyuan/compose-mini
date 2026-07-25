"""Normalize label-free forward ledgers; callers authenticate attempt bytes."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from statistics import fmean, pstdev
from types import MappingProxyType

from tools.float32 import decode_f32le_base64
from tools.panel_contract import FileBinding, _sha256
from tools.universe_forward_contract import (
    ForwardFitSpec, ForwardPredictionSpec, PassingScalingOutcome,
    _forward_fit_specs, _forward_prediction_specs,
)

FIT_FIELDS = frozenset({
    "schema", "attempt_sha256", "provenance_id", "optimizer_updates",
    "model_fingerprint",
})
PREDICTION_FIELDS = frozenset({
    "schema", "attempt_sha256", "provenance_id", "model_fingerprint",
    "phase", "series", "grid_sha256", "predictions",
})


@dataclass(frozen=True, slots=True)
class ForwardFitRecord:
    spec: ForwardFitSpec
    model_fingerprint: str


@dataclass(frozen=True, slots=True)
class ForwardFitClosure:
    attempt: FileBinding
    target_phase: str
    records: tuple[ForwardFitRecord, ...]
    fingerprints: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ForwardPredictionRecord:
    spec: ForwardPredictionSpec
    model_fingerprint: str
    predictions: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ForwardPredictionEnsemble:
    series: str
    manifest_rank: int
    timestamp_sha256: str
    prediction_mean: tuple[float, ...]
    prediction_pstdev: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ForwardPredictionClosure:
    fits: ForwardFitClosure
    records: tuple[ForwardPredictionRecord, ...]
    ensembles: tuple[ForwardPredictionEnsemble, ...]


def _attempt(value: object) -> FileBinding:
    if not isinstance(value, FileBinding):
        raise ValueError("forward attempt binding is invalid")
    return FileBinding.parse(
        {"path": value.path, "sha256": value.sha256}, "forward attempt",
    )


def _stream(
    values: Iterable[Mapping[str, object]], label: str,
) -> Iterator[Mapping[str, object]]:
    try:
        return iter(values)
    except TypeError as error:
        raise ValueError(f"{label} must be iterable") from error


def _next(
    records: Iterator[Mapping[str, object]], label: str,
) -> Mapping[str, object]:
    try:
        return next(records)
    except StopIteration:
        raise ValueError(f"{label} is missing a record") from None


def _finish(records: Iterator[Mapping[str, object]], label: str) -> None:
    try:
        next(records)
    except StopIteration:
        return
    raise ValueError(f"{label} contains an extra record")


def _record(
    value: object, fields: frozenset[str], label: str,
) -> dict[str, object]:
    if type(value) is not dict or value.keys() != fields:
        raise ValueError(f"{label} fields are invalid")
    if type(value["schema"]) is not int or value["schema"] != 1:
        raise ValueError(f"{label} schema is invalid")
    return value


def _fit_closure(
    attempt: FileBinding, target_phase: str,
    records: tuple[ForwardFitRecord, ...],
) -> ForwardFitClosure:
    fingerprints = {
        record.spec.provenance_id: record.model_fingerprint
        for record in records
    }
    if len(fingerprints) != len(records):
        raise ValueError("forward fit provenance is duplicated")
    return ForwardFitClosure(
        attempt, target_phase, records, MappingProxyType(fingerprints),
    )


def _phase(value: object) -> str:
    if type(value) is not str or value not in ("fold-1", "calibration"):
        raise ValueError("forward target phase is invalid")
    return value


def validate_forward_fit_records(
    values: Iterable[Mapping[str, object]],
    source: PassingScalingOutcome,
    target_phase: str,
    attempt: FileBinding,
) -> ForwardFitClosure:
    """Normalize one exact fit phase; the caller authenticates attempt bytes."""
    binding = _attempt(attempt)
    phase = _phase(target_phase)
    specs = _forward_fit_specs(source, phase)
    stream = _stream(values, "forward fit ledger")
    normalized = []
    for index, spec in enumerate(specs):
        raw = _next(stream, "forward fit ledger")
        record = _record(raw, FIT_FIELDS, f"forward fit[{index}]")
        if _sha256(
            record["attempt_sha256"], "forward fit attempt",
        ) != binding.sha256 or _sha256(
            record["provenance_id"], "forward fit provenance",
        ) != spec.provenance_id or \
                type(record["optimizer_updates"]) is not int or \
                record["optimizer_updates"] != spec.optimizer_updates:
            raise ValueError("forward fit record changed")
        normalized.append(ForwardFitRecord(
            spec,
            _sha256(
                record["model_fingerprint"], "forward model fingerprint",
            ),
        ))
    _finish(stream, "forward fit ledger")
    return _fit_closure(binding, phase, tuple(normalized))


def _normalize_fits(
    value: object, source: PassingScalingOutcome,
    target_phase: str, attempt: FileBinding,
) -> ForwardFitClosure:
    specs = _forward_fit_specs(source, target_phase)
    if type(value) is not ForwardFitClosure or \
            value.attempt != attempt or value.target_phase != target_phase or \
            type(value.target_phase) is not str or \
            type(value.records) is not tuple or \
            len(value.records) != len(specs):
        raise ValueError("forward fit closure changed")
    records = []
    for observed, expected in zip(value.records, specs, strict=True):
        if type(observed) is not ForwardFitRecord or \
                observed.spec != expected:
            raise ValueError("forward fit closure changed")
        records.append(ForwardFitRecord(
            expected,
            _sha256(
                observed.model_fingerprint, "forward model fingerprint",
            ),
        ))
    normalized = _fit_closure(attempt, target_phase, tuple(records))
    if type(value.fingerprints) is not MappingProxyType or \
            dict(value.fingerprints) != dict(normalized.fingerprints):
        raise ValueError("forward fit fingerprints changed")
    return normalized


def _ensembles(
    records: tuple[ForwardPredictionRecord, ...],
    fits: ForwardFitClosure,
) -> tuple[ForwardPredictionEnsemble, ...]:
    width = len(records) // len(fits.records)
    ensembles = []
    for index in range(width):
        seeds = records[index::width]
        mean, deviation = [], []
        for values in zip(
            *(record.predictions for record in seeds), strict=True,
        ):
            mean.append(fmean(values))
            deviation.append(pstdev(values))
        spec = seeds[0].spec
        ensembles.append(ForwardPredictionEnsemble(
            spec.series, spec.manifest_rank, spec.timestamp_sha256,
            tuple(mean), tuple(deviation),
        ))
    return tuple(ensembles)


def validate_forward_prediction_records(
    values: Iterable[Mapping[str, object]],
    source: PassingScalingOutcome,
    target_phase: str,
    attempt: FileBinding,
    fits: ForwardFitClosure,
) -> ForwardPredictionClosure:
    """Normalize one label-free phase; the caller authenticates attempt bytes."""
    binding = _attempt(attempt)
    phase = _phase(target_phase)
    normalized_fits = _normalize_fits(
        fits, source, phase, binding,
    )
    specs = _forward_prediction_specs(source, phase)
    stream = _stream(values, "forward prediction ledger")
    normalized = []
    for index, spec in enumerate(specs):
        raw = _next(stream, "forward prediction ledger")
        record = _record(
            raw, PREDICTION_FIELDS, f"forward prediction[{index}]",
        )
        fingerprint = normalized_fits.fingerprints[
            spec.fit.provenance_id
        ]
        if _sha256(
            record["attempt_sha256"], "forward prediction attempt",
        ) != binding.sha256 or _sha256(
            record["provenance_id"], "forward prediction provenance",
        ) != spec.fit.provenance_id or _sha256(
            record["model_fingerprint"],
            "forward prediction model fingerprint",
        ) != fingerprint or record["phase"] != phase or \
                record["series"] != spec.series or _sha256(
                    record["grid_sha256"], "forward prediction grid",
                ) != spec.timestamp_sha256:
            raise ValueError("forward prediction record changed")
        normalized.append(ForwardPredictionRecord(
            spec, fingerprint,
            decode_f32le_base64(
                record["predictions"],
                expected_count=spec.prediction_count,
            ),
        ))
    _finish(stream, "forward prediction ledger")
    records = tuple(normalized)
    return ForwardPredictionClosure(
        normalized_fits, records, _ensembles(records, normalized_fits),
    )
