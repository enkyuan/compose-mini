"""Freeze one development-only temporal-context comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType

from tools.panel_contract import (
    _exact_json, _integer, _object, _sha256, _string,
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
                "updates_per_checkpoint", "prior_selections",
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
            phase, expected_ranges, tuple(training_rows),
            tuple(evaluation_rows),
            _sha256(item["training_grid_sha256"], "training grid"),
            _sha256(item["evaluation_grid_sha256"], "evaluation grid"),
            updates, selections,
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
    return hashlib.sha256(json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode()).hexdigest()
