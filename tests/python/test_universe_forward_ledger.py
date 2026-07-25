#!/usr/bin/env python3
"""Verify phase-scoped, label-free universe-forward ledgers."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch
import base64
import inspect
import math
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tools.universe_forward_ledger as ledger
from tools.float32 import encode_f32le_base64
from tools.panel_contract import FileBinding
from tools.universe_forward_contract import (
    CheckpointSelection, PassingScalingOutcome, _forward_fit_specs,
    _forward_prediction_specs,
)
from tools.universe_scaling_contract import (
    PHASES, SEEDS, FitJob, PhaseCoverage, ScalingCoverage, SeriesCoverage,
    fit_provenance_id, timestamp_grid_sha256,
)

MASTER = tuple(f"S{index:02}" for index in range(55))
MISSING = {
    "fold-0": frozenset((15, 29)),
    "fold-1": frozenset((15, 29, 40)),
    "calibration": frozenset((15, 29, 33, 40)),
}
ATTEMPT = FileBinding("experiments/forward-attempt.json", "f" * 64)


def raises(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def digest(index: int) -> str:
    return f"{index:064x}"


def source() -> PassingScalingOutcome:
    coverage = ScalingCoverage(tuple(
        PhaseCoverage(phase, tuple(
            SeriesCoverage(
                series, 1_000, 0 if rank in MISSING[phase] else 2,
                timestamp_grid_sha256(()) if rank in MISSING[phase]
                else digest(100 * phase_index + rank),
            )
            for rank, series in enumerate(MASTER, 1)
        ))
        for phase_index, phase in enumerate(PHASES, 1)
    ))
    selections = tuple(
        CheckpointSelection(
            target, prior, seed, index + 1,
            fit_provenance_id(FitJob(
                "pooled", "fixed-update", 55, prior,
                "panel_transformer", seed, MASTER,
            )),
            digest(1_000 + 100 * target_index + index),
        )
        for target_index, (target, prior) in enumerate((
            ("fold-1", "fold-0"), ("calibration", "fold-1"),
        ))
        for index, seed in enumerate(SEEDS)
    )
    bindings = tuple(
        FileBinding(f"reports/run/{name}", digest(index))
        for index, name in enumerate((
            "outcome.json", "attempt.json", "fits.jsonl",
            "predictions.jsonl", "summary.json",
        ), 1)
    )
    return PassingScalingOutcome(
        "run", *bindings, SimpleNamespace(coverage=coverage), selections,
    )


def fit_rows(
    value: PassingScalingOutcome, phase: str,
    attempt: FileBinding = ATTEMPT,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "schema": 1,
            "attempt_sha256": attempt.sha256,
            "provenance_id": spec.provenance_id,
            "optimizer_updates": spec.optimizer_updates,
            "model_fingerprint": digest(2_000 + index),
        }
        for index, spec in enumerate(
            _forward_fit_specs(value, phase), 1,
        )
    )


def prediction_rows(
    value: PassingScalingOutcome, phase: str,
    fits: tuple[dict[str, object], ...],
    attempt: FileBinding = ATTEMPT,
) -> tuple[dict[str, object], ...]:
    fingerprints = {
        item.provenance_id: record["model_fingerprint"]
        for item, record in zip(
            _forward_fit_specs(value, phase), fits, strict=True,
        )
    }
    return tuple(
        {
            "schema": 1,
            "attempt_sha256": attempt.sha256,
            "provenance_id": spec.fit.provenance_id,
            "model_fingerprint": fingerprints[spec.fit.provenance_id],
            "phase": phase,
            "series": spec.series,
            "grid_sha256": spec.timestamp_sha256,
            "predictions": encode_f32le_base64(
                (
                    float(spec.manifest_rank + spec.fit.selection.seed),
                    float(spec.manifest_rank - spec.fit.selection.seed),
                ),
            ),
        }
        for spec in _forward_prediction_specs(value, phase)
    )


def changed(
    values: tuple[dict[str, object], ...], **fields: object,
) -> tuple[dict[str, object], ...]:
    return (values[0] | fields, *values[1:])


def test_phase_scoped_ledgers() -> None:
    value = source()
    for phase, count in (("fold-1", 260), ("calibration", 255)):
        raw_fits = fit_rows(value, phase)
        fits = ledger.validate_forward_fit_records(
            (item for item in raw_fits), value, phase, ATTEMPT,
        )
        raw_predictions = prediction_rows(value, phase, raw_fits)
        with patch.object(
            ledger, "_forward_prediction_specs",
            wraps=_forward_prediction_specs,
        ) as schedule, patch.object(
            Path, "open",
            side_effect=AssertionError("ledger validator opened a file"),
        ):
            predictions = ledger.validate_forward_prediction_records(
                iter(raw_predictions), value, phase, ATTEMPT, fits,
            )
        schedule.assert_called_once_with(value, phase)

        assert tuple(item.spec for item in fits.records) == \
            _forward_fit_specs(value, phase)
        assert tuple(item.spec for item in predictions.records) == \
            _forward_prediction_specs(value, phase)
        assert len(fits.records) == 5
        assert len(predictions.records) == count
        width = count // len(SEEDS)
        expected_ranks = tuple(
            rank for rank in range(1, len(MASTER) + 1)
            if rank not in MISSING[phase]
        )
        assert tuple(
            item.manifest_rank for item in predictions.ensembles
        ) == expected_ranks
        assert len(predictions.ensembles) == width
        phase_index = PHASES.index(phase) + 1
        for index, (rank, ensemble) in enumerate(zip(
            expected_ranks, predictions.ensembles, strict=True,
        )):
            group = predictions.records[index::width]
            assert tuple(
                record.spec.fit.selection.seed for record in group
            ) == SEEDS
            assert all(
                (
                    record.spec.series,
                    record.spec.manifest_rank,
                    record.spec.timestamp_sha256,
                ) == (
                    MASTER[rank - 1], rank,
                    digest(100 * phase_index + rank),
                )
                for record in group
            )
            assert (
                ensemble.series,
                ensemble.timestamp_sha256,
            ) == (
                MASTER[rank - 1],
                digest(100 * phase_index + rank),
            )
            assert ensemble.prediction_mean == (
                (5 * rank + 161) / 5,
                (5 * rank - 161) / 5,
            )
            assert all(
                math.isclose(
                    item, math.sqrt(351.36),
                    rel_tol=1e-15, abs_tol=0.0,
                )
                for item in ensemble.prediction_pstdev
            )
        identical = tuple(
            record | {"predictions": encode_f32le_base64(
                (float(index % width),) * 2,
            )}
            for index, record in enumerate(raw_predictions)
        )
        assert all(
            ensemble.prediction_pstdev == (0.0, 0.0)
            for ensemble in ledger.validate_forward_prediction_records(
                identical, value, phase, ATTEMPT, fits,
            ).ensembles
        )
        assert predictions.fits is not fits
        assert predictions.fits == fits
        assert not hasattr(predictions, "truth")
        assert not hasattr(predictions, "metrics")

        raw_fits[0]["model_fingerprint"] = digest(9_999)
        raw_predictions[0]["predictions"] = encode_f32le_base64((0.0, 0.0))
        assert fits.records[0].model_fingerprint == digest(2_001)
        assert predictions.records[0].predictions != (0.0, 0.0)
        assert predictions.ensembles[0].prediction_mean != (0.0, 0.0)
        try:
            fits.records[0].model_fingerprint = digest(1)  # type: ignore[misc]
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("forward fit record is mutable")
        try:
            fits.fingerprints[fits.records[0].spec.provenance_id] = digest(1)
        except TypeError:
            pass
        else:
            raise AssertionError("forward fingerprints are mutable")
        try:
            predictions.ensembles[0].manifest_rank = 1  # type: ignore[misc]
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("forward prediction ensemble is mutable")


def test_fit_record_rejections() -> None:
    value = source()
    rows = fit_rows(value, "fold-1")
    invalid = (
        changed(rows, schema=True),
        changed(rows, attempt_sha256=digest(1)),
        changed(rows, provenance_id=digest(2)),
        changed(rows, optimizer_updates=True),
        changed(rows, optimizer_updates=rows[0]["optimizer_updates"] + 1),
        changed(rows, model_fingerprint="F" * 64),
        changed(rows, extra=None),
        rows[1:],
        (*rows, rows[0]),
        (rows[1], rows[0], *rows[2:]),
        (rows[0], rows[0], *rows[2:]),
    )
    for records in invalid:
        raises(
            ledger.validate_forward_fit_records,
            records, value, "fold-1", ATTEMPT,
        )
    for attempt in (
        object(),
        replace(ATTEMPT, sha256="F" * 64),
        replace(ATTEMPT, sha256=True),  # type: ignore[arg-type]
        replace(ATTEMPT, path="../attempt.json"),
    ):
        raises(
            ledger.validate_forward_fit_records,
            rows, value, "fold-1", attempt,
        )
    raises(
        ledger.validate_forward_fit_records,
        rows, value, type("Phase", (str,), {})("fold-1"), ATTEMPT,
    )
    raises(
        ledger.validate_forward_fit_records, rows,
        replace(value, outcome=replace(value.outcome, sha256=digest(91))),
        "fold-1", ATTEMPT,
    )
    selections = value.selections
    raises(
        ledger.validate_forward_fit_records, rows,
        replace(value, selections=(
            replace(selections[0], checkpoint=2), *selections[1:],
        )),
        "fold-1", ATTEMPT,
    )

    reads = 0

    def extra() -> object:
        nonlocal reads
        for item in (*rows, rows[0]):
            reads += 1
            yield item
        raise AssertionError("fit validator consumed a seventh record")

    raises(
        ledger.validate_forward_fit_records,
        extra(), value, "fold-1", ATTEMPT,
    )
    assert reads == 6


def test_prediction_record_rejections() -> None:
    value = source()
    raw_fits = fit_rows(value, "fold-1")
    fits = ledger.validate_forward_fit_records(
        raw_fits, value, "fold-1", ATTEMPT,
    )
    rows = prediction_rows(value, "fold-1", raw_fits)
    nan = base64.b64encode(struct.pack("<f", float("nan"))).decode()
    invalid = (
        changed(rows, schema=True),
        changed(rows, attempt_sha256=digest(1)),
        changed(rows, provenance_id=digest(2)),
        changed(rows, model_fingerprint=digest(3)),
        changed(rows, phase="calibration"),
        changed(rows, series="OTHER"),
        changed(rows, grid_sha256=digest(4)),
        changed(rows, predictions=encode_f32le_base64((0.0,))),
        changed(rows, predictions={
            "encoding": "f32le-base64", "count": 1, "base64": nan,
        }),
        changed(rows, extra=None),
        rows[1:],
        (*rows, rows[0]),
        (rows[1], rows[0], *rows[2:]),
        (rows[0], rows[0], *rows[2:]),
    )
    for records in invalid:
        raises(
            ledger.validate_forward_prediction_records,
            records, value, "fold-1", ATTEMPT, fits,
        )
    with patch.object(
        ledger, "_ensembles", wraps=ledger._ensembles,
    ) as aggregate:
        raises(
            ledger.validate_forward_prediction_records,
            (*rows[:-1], rows[-1] | {"schema": True}),
            value, "fold-1", ATTEMPT, fits,
        )
    aggregate.assert_not_called()
    calibration = ledger.validate_forward_fit_records(
        fit_rows(value, "calibration"),
        value, "calibration", ATTEMPT,
    )
    raises(
        ledger.validate_forward_prediction_records,
        rows, value, "fold-1", ATTEMPT, calibration,
    )
    raises(
        ledger.validate_forward_prediction_records,
        rows, value, "fold-1", replace(ATTEMPT, sha256=digest(5)), fits,
    )

    retained = dict(fits.fingerprints)
    forged = replace(fits, fingerprints=MappingProxyType(retained))
    normalized = ledger.validate_forward_prediction_records(
        rows, value, "fold-1", ATTEMPT, forged,
    )
    assert normalized.fits.fingerprints == fits.fingerprints
    assert normalized.fits.fingerprints is not forged.fingerprints
    retained.clear()
    raises(
        ledger.validate_forward_prediction_records,
        rows, value, "fold-1", ATTEMPT, forged,
    )
    raises(
        ledger.validate_forward_prediction_records,
        rows, value, "fold-1", ATTEMPT,
        replace(fits, fingerprints=object()),  # type: ignore[arg-type]
    )

    reads_before_rejection = 0

    def invalid_first() -> object:
        nonlocal reads_before_rejection
        reads_before_rejection += 1
        yield rows[0] | {"schema": True}
        reads_before_rejection += 1
        raise AssertionError("prediction validator read after rejection")

    raises(
        ledger.validate_forward_prediction_records,
        invalid_first(), value, "fold-1", ATTEMPT, fits,
    )
    assert reads_before_rejection == 1

    reads = 0

    def extra() -> object:
        nonlocal reads
        for item in (*rows, rows[0]):
            reads += 1
            yield item
        raise AssertionError("prediction validator read past one extra row")

    raises(
        ledger.validate_forward_prediction_records,
        extra(), value, "fold-1", ATTEMPT, fits,
    )
    assert reads == len(rows) + 1


def test_label_free_schema() -> None:
    banned = {"actual", "label", "price", "return", "truth"}
    assert type(ledger.FIT_FIELDS) is frozenset
    assert type(ledger.PREDICTION_FIELDS) is frozenset
    assert not banned & ledger.FIT_FIELDS
    assert not banned & ledger.PREDICTION_FIELDS
    parameters = set(inspect.signature(
        ledger.validate_forward_prediction_records,
    ).parameters)
    assert not banned & parameters

    value = source()
    fits = fit_rows(value, "fold-1")
    raises(
        ledger.validate_forward_fit_records,
        changed(fits, label=0.0), value, "fold-1", ATTEMPT,
    )
    closure = ledger.validate_forward_fit_records(
        fits, value, "fold-1", ATTEMPT,
    )
    raises(
        ledger.validate_forward_prediction_records,
        changed(
            prediction_rows(value, "fold-1", fits), actual=0.0,
        ),
        value, "fold-1", ATTEMPT, closure,
    )


def main() -> None:
    test_phase_scoped_ledgers()
    test_fit_record_rejections()
    test_prediction_record_rejections()
    test_label_free_schema()


if __name__ == "__main__":
    main()
