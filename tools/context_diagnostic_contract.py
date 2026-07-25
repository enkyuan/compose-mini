"""Freeze one development-only temporal-context comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType

from tools.float32 import decode_f32le_base64, encode_f32le_base64
from tools.panel_contract import (
    FileBinding, _exact_json, _integer, _object, _sha256, _string,
)
from tools.universe_contract import fixed_update_budget, universe_roles
from tools.universe_scaling_contract import (
    EXPECTED_BUDGETS, FitJob, fit_provenance_id,
)

HISTORY_LENGTHS = (17, 34, 68)
PRIMARY_MODEL = "panel_transformer"
CONTROL_MODELS = ("global_ridge", "global_mlp")
MODELS = (*CONTROL_MODELS, PRIMARY_MODEL)
NEURAL_MODELS = ("global_mlp", PRIMARY_MODEL)
RUNTIME_MODELS = ("linear", "mlp", PRIMARY_MODEL)
RUNTIME_TO_PUBLIC = MappingProxyType(dict(zip(
    RUNTIME_MODELS, MODELS, strict=True,
)))
SEEDS = (7, 19, 31, 43, 61)
TARGET_PHASES = ("fold-1", "calibration")
PRIOR_PHASE = MappingProxyType(dict(zip(
    TARGET_PHASES, ("fold-0", "fold-1"), strict=True,
)))
PHASE_RANGES = MappingProxyType({
    "fold-1": ((0, 3_843), (3_855, 4_393)),
    "calibration": ((0, 4_393), (4_405, 4_943)),
})
CONTROL_COHORT = 11
TRAINING_COHORT = 44
EVALUATION_RANKS = tuple(range(45, 56))
MAX_HISTORY = max(HISTORY_LENGTHS)
BATCH_SIZE = 128
SCALER_POLICY = "per-stock-common-68-training-prefix"
EXPECTED_FITS_PER_PHASE = len(HISTORY_LENGTHS) * (
    1 + len(NEURAL_MODELS) * len(SEEDS)
)
EXPECTED_PREDICTIONS_PER_PHASE = (
    EXPECTED_FITS_PER_PHASE * len(EVALUATION_RANKS)
)
_SOURCE_BUDGETS = MappingProxyType(dict(EXPECTED_BUDGETS))


def _candidate(history: int) -> dict[str, object]:
    return {
        "feature_set": "ohlcv",
        "ff_dim": 32,
        "heads": 2,
        "layers": 1,
        "learning_rate": 0.0003,
        "mlp_dim": 32,
        "model_dim": 16,
        "name": f"raw-{history}",
        "ridge": 0.001,
        "rolling_window": 8,
        "seq_len": history,
        "weight_decay": 0.0001,
    }


def expected_context_sweep() -> dict[str, object]:
    """Return a fresh copy of the exact context family."""
    return {
        "alignment_horizon_bars": 13,
        "batch_size": BATCH_SIZE,
        "candidates": list(map(_candidate, HISTORY_LENGTHS)),
        "epochs": 100,
        "fold_fraction": 0.1,
        "folds": 2,
        "models": list(RUNTIME_MODELS),
        "patience": 10,
        "seeds": list(SEEDS),
        "target_horizon_bars": 13,
        "target_kind": "executable-return-v1",
    }


def validate_context_sweep(value: object) -> Mapping[str, object]:
    """Reject any config outside the predeclared family."""
    expected = expected_context_sweep()
    if not _exact_json(value, expected):
        raise ValueError("config does not match the context family")
    return expected


def _items(value: object, count: int, label: str) -> list[object]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} items")
    return value


def _json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ContextScalerInput:
    series: str
    csv_sha256: str
    training_rows: int
    training_grid_sha256: str


def context_scaler_inputs_sha256(
    master: Sequence[str], values: Sequence[ContextScalerInput],
) -> str:
    """Bind every stock's max-history training scaler inputs."""
    names = tuple(master)
    universe_roles(names)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("context scaler inputs are invalid")
    inputs = tuple(values)
    if len(inputs) != len(names):
        raise ValueError("context scaler inputs must contain 55 stocks")
    records = []
    for index, (value, series) in enumerate(zip(
        inputs, names, strict=True,
    )):
        if not isinstance(value, ContextScalerInput) or \
           _string(value.series, f"scaler input {index} series") != series:
            raise ValueError("context scaler input order changed")
        records.append({
            "csv_sha256": _sha256(
                value.csv_sha256, f"{series} scaler csv",
            ),
            "series": series,
            "training_grid_sha256": _sha256(
                value.training_grid_sha256, f"{series} scaler grid",
            ),
            "training_rows": _integer(
                value.training_rows, f"{series} scaler rows",
            ),
        })
    return _json_sha256({
        "inputs": records,
        "max_history": MAX_HISTORY,
        "role": "context-scaler-inputs",
        "scaler_policy": SCALER_POLICY,
        "schema": 1,
    })


@dataclass(frozen=True, slots=True)
class PriorSelection:
    model: str
    seed: int
    selected_checkpoint: int
    source_provenance_id: str
    source_model_fingerprint: str

    @classmethod
    def parse(
        cls, value: object, phase: str, model: str, seed: int,
        members: tuple[str, ...],
    ) -> PriorSelection:
        item = _object(
            value,
            {
                "model", "seed", "selected_checkpoint",
                "source_provenance_id", "source_model_fingerprint",
            },
            "prior selection",
        )
        if _string(item["model"], "prior selection model") != model or \
           _integer(item["seed"], "prior selection seed") != seed:
            raise ValueError("prior selection axes changed")
        checkpoint = _integer(
            item["selected_checkpoint"], "selected checkpoint",
        )
        if checkpoint > _SOURCE_BUDGETS[PRIOR_PHASE[phase]].checkpoints:
            raise ValueError("selected checkpoint exceeds the source grid")
        source = FitJob(
            "pooled", "fixed-update", TRAINING_COHORT,
            PRIOR_PHASE[phase], model, seed, members,
        )
        provenance = _sha256(
            item["source_provenance_id"], "source provenance",
        )
        if provenance != fit_provenance_id(source):
            raise ValueError("source provenance does not match its axes")
        return cls(
            model, seed, checkpoint, provenance,
            _sha256(item["source_model_fingerprint"],
                    "source model fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class ContextPhase:
    phase: str
    source_ranges: tuple[tuple[int, int], tuple[int, int]]
    training_rows: tuple[tuple[str, int], ...]
    evaluation_rows: tuple[tuple[str, int, str], ...]
    training_grid_sha256: str
    evaluation_grid_sha256: str
    scaler_inputs_sha256: str
    updates_per_checkpoint: int
    prior_selections: tuple[PriorSelection, ...]

    @classmethod
    def parse(
        cls, value: object, master: Sequence[str],
    ) -> ContextPhase:
        names = tuple(master)
        roles = universe_roles(names)
        training = dict(roles.transfer_training)[TRAINING_COHORT]
        evaluation = tuple(names[rank - 1] for rank in EVALUATION_RANKS)
        if evaluation != roles.unseen:
            raise ValueError("evaluation ranks do not match universe roles")
        item = _object(
            value,
            {
                "phase", "source_ranges", "training_rows", "evaluation_rows",
                "training_grid_sha256", "evaluation_grid_sha256",
                "scaler_inputs_sha256", "updates_per_checkpoint",
                "prior_selections",
            },
            "context phase",
        )
        phase = _string(item["phase"], "context phase name")
        if phase not in TARGET_PHASES:
            raise ValueError("context phase is not a target phase")
        expected_ranges = PHASE_RANGES[phase]
        if not _exact_json(
            item["source_ranges"], list(map(list, expected_ranges)),
        ):
            raise ValueError("context phase source ranges changed")

        training_rows = []
        for index, (raw, series) in enumerate(zip(
            _items(item["training_rows"], len(training), "training rows"),
            training, strict=True,
        )):
            row = _object(raw, {"series", "count"}, f"training row {index}")
            if row["series"] != series:
                raise ValueError("training row order changed")
            training_rows.append((
                series, _integer(row["count"], f"{series} training count"),
            ))

        evaluation_rows = []
        for index, (raw, series) in enumerate(zip(
            _items(
                item["evaluation_rows"], len(evaluation), "evaluation rows",
            ),
            evaluation, strict=True,
        )):
            row = _object(
                raw, {"series", "count", "grid_sha256"},
                f"evaluation row {index}",
            )
            if row["series"] != series:
                raise ValueError("evaluation row order changed")
            evaluation_rows.append((
                series,
                _integer(row["count"], f"{series} evaluation count"),
                _sha256(row["grid_sha256"], f"{series} evaluation grid"),
            ))

        axes = tuple(
            (model, seed) for model in NEURAL_MODELS for seed in SEEDS
        )
        selections = tuple(
            PriorSelection.parse(raw, phase, model, seed, training)
            for raw, (model, seed) in zip(
                _items(item["prior_selections"], len(axes),
                       "prior selections"),
                axes, strict=True,
            )
        )
        updates = fixed_update_budget(
            sum(
                count for _, count in training_rows[:CONTROL_COHORT]
            ),
            BATCH_SIZE, 1,
        ).updates_per_checkpoint
        if _integer(
            item["updates_per_checkpoint"], "updates per checkpoint",
        ) != updates:
            raise ValueError(
                "updates per checkpoint do not match control rows",
            )
        return cls(
            phase=phase,
            source_ranges=expected_ranges,
            training_rows=tuple(training_rows),
            evaluation_rows=tuple(evaluation_rows),
            training_grid_sha256=_sha256(
                item["training_grid_sha256"], "training grid",
            ),
            evaluation_grid_sha256=_sha256(
                item["evaluation_grid_sha256"], "evaluation grid",
            ),
            scaler_inputs_sha256=_sha256(
                item["scaler_inputs_sha256"], "scaler inputs",
            ),
            updates_per_checkpoint=updates,
            prior_selections=selections,
        )


def _phase_value(phase: ContextPhase) -> dict[str, object]:
    return {
        "evaluation_grid_sha256": phase.evaluation_grid_sha256,
        "evaluation_rows": [
            {"count": count, "grid_sha256": grid, "series": series}
            for series, count, grid in phase.evaluation_rows
        ],
        "phase": phase.phase,
        "prior_selections": list(map(asdict, phase.prior_selections)),
        "scaler_inputs_sha256": phase.scaler_inputs_sha256,
        "source_ranges": list(map(list, phase.source_ranges)),
        "training_grid_sha256": phase.training_grid_sha256,
        "training_rows": [
            {"count": count, "series": series}
            for series, count in phase.training_rows
        ],
        "updates_per_checkpoint": phase.updates_per_checkpoint,
    }


def _validated_phase(
    phase: ContextPhase, master: Sequence[str],
) -> ContextPhase:
    if not isinstance(phase, ContextPhase):
        raise ValueError("context phase type changed")
    parsed = ContextPhase.parse(_phase_value(phase), master)
    if parsed != phase:
        raise ValueError("context phase changed during validation")
    return parsed


def parse_context_phases(
    value: object, master: Sequence[str],
) -> tuple[ContextPhase, ...]:
    """Parse both target phases in their frozen order."""
    phases = tuple(
        ContextPhase.parse(item, master)
        for item in _items(value, len(TARGET_PHASES), "context phases")
    )
    if tuple(item.phase for item in phases) != TARGET_PHASES:
        raise ValueError("context phase order changed")
    return phases


@dataclass(frozen=True, slots=True)
class ContextFit:
    phase: str
    model: str
    history: int
    seed: int | None
    members: tuple[str, ...]
    optimizer_updates: int
    selected_checkpoint: int | None
    source_provenance_id: str | None
    source_model_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class ContextPrediction:
    fit: ContextFit
    series: str
    prediction_count: int
    grid_sha256: str


@dataclass(frozen=True, slots=True)
class ContextFitEvidence:
    fit: ContextFit
    provenance_id: str
    state_fingerprint: str
    training_loss: float


@dataclass(frozen=True, slots=True)
class ContextPredictionEvidence:
    prediction: ContextPrediction
    fit_provenance_id: str
    state_fingerprint: str
    values: tuple[float, ...]


def expected_context_fits(
    master: Sequence[str], phase: ContextPhase,
) -> tuple[ContextFit, ...]:
    """Return each physical context fit exactly once."""
    phase = _validated_phase(phase, master)
    members = tuple(series for series, _ in phase.training_rows)
    selections = {
        (item.model, item.seed): item for item in phase.prior_selections
    }
    fits = []
    for history in HISTORY_LENGTHS:
        fits.append(ContextFit(
            phase.phase, "global_ridge", history, None, members,
            0, None, None, None,
        ))
        for model in NEURAL_MODELS:
            for seed in SEEDS:
                source = selections[model, seed]
                fits.append(ContextFit(
                    phase.phase, model, history, seed, members,
                    source.selected_checkpoint * phase.updates_per_checkpoint,
                    source.selected_checkpoint, source.source_provenance_id,
                    source.source_model_fingerprint,
                ))
    if len(fits) != EXPECTED_FITS_PER_PHASE:
        raise ValueError("context fit closure changed")
    return tuple(fits)


def expected_context_predictions(
    master: Sequence[str], phase: ContextPhase,
) -> tuple[ContextPrediction, ...]:
    """Return every fit-by-unseen-series prediction record."""
    phase = _validated_phase(phase, master)
    predictions = tuple(
        ContextPrediction(fit, series, count, grid)
        for fit in expected_context_fits(master, phase)
        for series, count, grid in phase.evaluation_rows
    )
    if len(predictions) != EXPECTED_PREDICTIONS_PER_PHASE:
        raise ValueError("context prediction closure changed")
    return predictions


def context_provenance_id(
    fit: ContextFit, phase: ContextPhase,
    source_failure_sha256: str, config_sha256: str,
) -> str:
    """Bind one fit to every frozen context treatment."""
    if not isinstance(fit, ContextFit) or not isinstance(phase, ContextPhase):
        raise ValueError("context provenance inputs are invalid")
    master = (
        *(series for series, _ in phase.training_rows),
        *(series for series, _, _ in phase.evaluation_rows),
    )
    phase = _validated_phase(phase, master)
    if fit not in expected_context_fits(master, phase):
        raise ValueError("context fit is outside the frozen family")
    payload = {
        "config_sha256": _sha256(config_sha256, "context config"),
        "fit": asdict(fit),
        "phase": _phase_value(phase),
        "scaler_policy": SCALER_POLICY,
        "source_failure_sha256": _sha256(
            source_failure_sha256, "source failure",
        ),
    }
    return _json_sha256(payload)


def context_family_sha256() -> str:
    """Hash the predeclared model family independently of any run."""
    return _json_sha256(expected_context_sweep())


def context_phase_sha256(phase: ContextPhase) -> str:
    """Hash every frozen axis and grid in one validated target phase."""
    if not isinstance(phase, ContextPhase):
        raise ValueError("context phase digest input is invalid")
    return _json_sha256(_phase_value(phase))


def _loss(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("training loss must be numeric")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("training loss must be finite and nonnegative") \
            from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("training loss must be finite and nonnegative")
    return result


def _fit_value(fit: ContextFit) -> dict[str, object]:
    value = asdict(fit)
    value["members"] = list(fit.members)
    return value


def context_fit_record(
    fit: ContextFit, provenance_id: str, state_fingerprint: str,
    training_loss: float,
) -> dict[str, object]:
    """Serialize one fitted state without adding selection behavior."""
    if not isinstance(fit, ContextFit):
        raise ValueError("context fit record is invalid")
    return {
        "fit": _fit_value(fit),
        "provenance_id": _sha256(provenance_id, "context provenance"),
        "schema": 1,
        "state_fingerprint": _sha256(
            state_fingerprint, "context state fingerprint",
        ),
        "training_loss": _loss(training_loss),
    }


def validate_context_fit_records(
    value: object, master: Sequence[str], phase: ContextPhase,
    source_failure_sha256: str, config_sha256: str,
) -> tuple[ContextFitEvidence, ...]:
    """Require the complete ordered physical-fit ledger for one phase."""
    expected = expected_context_fits(master, phase)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or \
       len(value) != len(expected):
        raise ValueError("context fit ledger has the wrong closure")
    records = []
    for index, (raw, fit) in enumerate(zip(value, expected, strict=True)):
        item = _object(
            raw,
            {
                "schema", "fit", "provenance_id", "state_fingerprint",
                "training_loss",
            },
            f"context fit[{index}]",
        )
        provenance = context_provenance_id(
            fit, phase, source_failure_sha256, config_sha256,
        )
        if _integer(item["schema"], "context fit schema") != 1 or \
           not _exact_json(item["fit"], _fit_value(fit)) or \
           _sha256(item["provenance_id"], "context fit provenance") != \
                provenance:
            raise ValueError("context fit ledger order or provenance changed")
        records.append(ContextFitEvidence(
            fit, provenance,
            _sha256(
                item["state_fingerprint"], "context state fingerprint",
            ),
            _loss(item["training_loss"]),
        ))
    return tuple(records)


def context_prediction_record(
    prediction: ContextPrediction, fit: ContextFitEvidence,
    values: Sequence[float],
) -> dict[str, object]:
    """Serialize one label-free prediction vector for a bound fitted state."""
    if not isinstance(prediction, ContextPrediction) or \
       not isinstance(fit, ContextFitEvidence) or prediction.fit != fit.fit:
        raise ValueError("context prediction fit is invalid")
    payload = encode_f32le_base64(values)
    if payload["count"] != prediction.prediction_count:
        raise ValueError("context prediction count changed")
    return {
        "fit_provenance_id": fit.provenance_id,
        "grid_sha256": prediction.grid_sha256,
        "history": prediction.fit.history,
        "model": prediction.fit.model,
        "phase": prediction.fit.phase,
        "prediction_count": prediction.prediction_count,
        "predictions": payload,
        "schema": 1,
        "seed": prediction.fit.seed,
        "series": prediction.series,
        "state_fingerprint": fit.state_fingerprint,
    }


def validate_context_prediction_records(
    value: object, master: Sequence[str], phase: ContextPhase,
    fit_records: object, source_failure_sha256: str, config_sha256: str,
) -> tuple[ContextPredictionEvidence, ...]:
    """Require the complete ordered prediction ledger for one phase."""
    expected = expected_context_predictions(master, phase)
    fitted = validate_context_fit_records(
        fit_records, master, phase, source_failure_sha256, config_sha256,
    )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or \
       len(value) != len(expected):
        raise ValueError("context prediction inputs have the wrong closure")
    by_fit = {item.fit: item for item in fitted}
    records = []
    fields = {
        "schema", "phase", "model", "history", "seed", "series",
        "prediction_count", "grid_sha256", "fit_provenance_id",
        "state_fingerprint", "predictions",
    }
    for index, (raw, prediction) in enumerate(zip(
        value, expected, strict=True,
    )):
        item = _object(raw, fields, f"context prediction[{index}]")
        fit = by_fit[prediction.fit]
        axes = (
            item["phase"], item["model"], item["history"], item["seed"],
            item["series"], item["prediction_count"],
        )
        expected_axes = (
            prediction.fit.phase, prediction.fit.model,
            prediction.fit.history, prediction.fit.seed,
            prediction.series, prediction.prediction_count,
        )
        if _integer(item["schema"], "context prediction schema") != 1 or \
           not _exact_json(list(axes), list(expected_axes)) or \
           _sha256(item["grid_sha256"], "context prediction grid") != \
                prediction.grid_sha256 or \
           _sha256(
               item["fit_provenance_id"], "context prediction provenance",
           ) != fit.provenance_id or \
           _sha256(
               item["state_fingerprint"], "context prediction state",
           ) != fit.state_fingerprint:
            raise ValueError("context prediction closure changed")
        records.append(ContextPredictionEvidence(
            prediction, fit.provenance_id, fit.state_fingerprint,
            decode_f32le_base64(
                item["predictions"],
                expected_count=prediction.prediction_count,
            ),
        ))
    return tuple(records)


def _run_identity(value: object) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("context run identity is invalid")
    return (
        _integer(value[0], "context run device", 0),
        _integer(value[1], "context run inode", 1),
    )


@dataclass(frozen=True, slots=True)
class ContextReceipt:
    """Bind complete phase evidence to one immutable run directory."""

    phase: str
    attempt: FileBinding
    fits: FileBinding
    predictions: FileBinding
    evaluation_grid_sha256: str
    family_sha256: str
    phase_sha256: str
    source_tree_sha256: str
    run_identity: tuple[int, int]
    fit_count: int
    prediction_count: int

    @classmethod
    def parse(cls, value: object) -> ContextReceipt:
        item = _object(
            value,
            {
                "schema", "phase", "attempt", "fits", "predictions",
                "evaluation_grid_sha256", "family_sha256",
                "phase_sha256", "source_tree_sha256",
                "run_identity", "fit_count", "prediction_count",
            },
            "context receipt",
        )
        phase = _string(item["phase"], "context receipt phase")
        if _integer(item["schema"], "context receipt schema") != 1 or \
           phase not in TARGET_PHASES:
            raise ValueError("context receipt identity is invalid")
        attempt = FileBinding.parse(item["attempt"], "receipt attempt")
        fits = FileBinding.parse(item["fits"], "receipt fits")
        predictions = FileBinding.parse(
            item["predictions"], "receipt predictions",
        )
        if len({attempt.path, fits.path, predictions.path}) != 3:
            raise ValueError("context receipt paths must be distinct")
        return cls(
            phase, attempt, fits, predictions,
            _sha256(
                item["evaluation_grid_sha256"], "receipt evaluation grid",
            ),
            _sha256(item["family_sha256"], "receipt family"),
            _sha256(item["phase_sha256"], "receipt phase"),
            _sha256(item["source_tree_sha256"], "receipt source tree"),
            _run_identity(item["run_identity"]),
            _integer(item["fit_count"], "receipt fit count"),
            _integer(item["prediction_count"], "receipt prediction count"),
        )

    def value(self) -> dict[str, object]:
        return {
            "attempt": asdict(self.attempt),
            "evaluation_grid_sha256": self.evaluation_grid_sha256,
            "family_sha256": self.family_sha256,
            "fit_count": self.fit_count,
            "fits": asdict(self.fits),
            "phase": self.phase,
            "phase_sha256": self.phase_sha256,
            "prediction_count": self.prediction_count,
            "predictions": asdict(self.predictions),
            "run_identity": list(self.run_identity),
            "schema": 1,
            "source_tree_sha256": self.source_tree_sha256,
        }

    def validate(
        self, phase: ContextPhase, attempt: FileBinding,
        fits: FileBinding, predictions: FileBinding,
        source_tree_sha256: str, run_identity: tuple[int, int],
    ) -> None:
        if not isinstance(phase, ContextPhase) or \
           self != ContextReceipt(
               phase.phase, attempt, fits, predictions,
               phase.evaluation_grid_sha256, context_family_sha256(),
               context_phase_sha256(phase),
               _sha256(source_tree_sha256, "receipt source tree"),
               _run_identity(run_identity),
               EXPECTED_FITS_PER_PHASE, EXPECTED_PREDICTIONS_PER_PHASE,
           ):
            raise ValueError("context receipt does not bind the phase")
