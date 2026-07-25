#!/usr/bin/env python3
"""Verify the frozen, Torch-free temporal-context family."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import copy
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.context_diagnostic_contract import (
    BATCH_SIZE, CONTROL_COHORT, CONTROL_MODELS, EVALUATION_RANKS,
    HISTORY_LENGTHS, MAX_HISTORY, MODELS, PHASE_RANGES, PRIMARY_MODEL,
    RUNTIME_MODELS, RUNTIME_TO_PUBLIC, SCALER_POLICY, SEEDS, TARGET_PHASES,
    ContextPhase,
    context_provenance_id, expected_context_fits,
    expected_context_predictions, expected_context_sweep,
    parse_context_phases, validate_context_sweep,
)
from tools.panel_contract import read_canonical_json
from tools.universe_scaling_contract import FitJob, fit_provenance_id

CONFIG = ROOT / "experiments/executable-h13-context.example.json"
MASTER = tuple(f"S{index:02d}" for index in range(55))
SOURCE_FAILURE = hashlib.sha256(b"source-failure").hexdigest()
CONFIG_SHA256 = hashlib.sha256(b"context-config").hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def raises(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def selection_value(
    phase: str, model: str, seed: int, checkpoint: int,
) -> dict[str, object]:
    source_phase = {"fold-1": "fold-0", "calibration": "fold-1"}[phase]
    job = FitJob(
        "pooled", "fixed-update", 44, source_phase, model, seed, MASTER[:44],
    )
    return {
        "model": model,
        "seed": seed,
        "selected_checkpoint": checkpoint,
        "source_model_fingerprint": digest(f"{phase}-{model}-{seed}-model"),
        "source_provenance_id": fit_provenance_id(job),
    }


def phase_value(phase: str) -> dict[str, object]:
    training_rows = [
        {"count": 100 + index, "series": series}
        for index, series in enumerate(MASTER[:44])
    ]
    selections = [
        selection_value(phase, model, seed, index + 1)
        for model in ("global_mlp", "panel_transformer")
        for index, seed in enumerate(SEEDS)
    ]
    return {
        "evaluation_grid_sha256": digest(f"{phase}-evaluation"),
        "evaluation_rows": [
            {
                "count": 20 + index,
                "grid_sha256": digest(f"{phase}-{series}"),
                "series": series,
            }
            for index, series in enumerate(MASTER[44:])
        ],
        "phase": phase,
        "prior_selections": selections,
        "source_ranges": list(map(list, PHASE_RANGES[phase])),
        "training_grid_sha256": digest(f"{phase}-training"),
        "training_rows": training_rows,
        "updates_per_checkpoint": (
            sum(
                row["count"] for row in training_rows[:CONTROL_COHORT]
            ) + BATCH_SIZE - 1
        ) // BATCH_SIZE,
    }


def test_exact_sweep() -> None:
    value = read_canonical_json(CONFIG)
    assert validate_context_sweep(value) == expected_context_sweep()
    assert HISTORY_LENGTHS == (17, 34, 68)
    assert MAX_HISTORY == 68
    assert PRIMARY_MODEL == "panel_transformer"
    assert CONTROL_MODELS == ("global_ridge", "global_mlp")
    assert MODELS == (*CONTROL_MODELS, PRIMARY_MODEL)
    assert RUNTIME_MODELS == ("linear", "mlp", "panel_transformer")
    assert tuple(RUNTIME_TO_PUBLIC.items()) == tuple(zip(
        RUNTIME_MODELS, MODELS, strict=True,
    ))
    assert SCALER_POLICY == "per-stock-common-68-training-prefix"
    assert TARGET_PHASES == ("fold-1", "calibration")
    assert EVALUATION_RANKS == tuple(range(45, 56))
    assert PHASE_RANGES == {
        "fold-1": ((0, 3_843), (3_855, 4_393)),
        "calibration": ((0, 4_393), (4_405, 4_943)),
    }
    assert (CONTROL_COHORT, BATCH_SIZE) == (11, 128)
    assert "torch" not in sys.modules

    for mutate in (
        lambda item: item["candidates"].reverse(),
        lambda item: item["candidates"].pop(),
        lambda item: item["candidates"].append(item["candidates"][0]),
        lambda item: item["models"].reverse(),
        lambda item: item["models"].pop(),
        lambda item: item["models"].append("linear"),
        lambda item: item["seeds"].reverse(),
        lambda item: item["seeds"].pop(),
        lambda item: item["seeds"].append(SEEDS[0]),
        lambda item: item.update({"extra": True}),
    ):
        invalid = copy.deepcopy(value)
        mutate(invalid)
        raises(validate_context_sweep, invalid)

    invalid = copy.deepcopy(value)
    invalid["candidates"][1]["weight_decay"] = 0.0
    raises(validate_context_sweep, invalid)


def test_phase_family_and_closure() -> None:
    phases = parse_context_phases(
        [phase_value(phase) for phase in TARGET_PHASES], MASTER,
    )
    assert tuple(phase.phase for phase in phases) == TARGET_PHASES
    assert all(len(phase.training_rows) == 44 for phase in phases)
    assert all(len(phase.evaluation_rows) == 11 for phase in phases)
    assert all(len(phase.prior_selections) == 10 for phase in phases)

    fits = tuple(
        fit for phase in phases for fit in expected_context_fits(MASTER, phase)
    )
    predictions = tuple(
        prediction
        for phase in phases
        for prediction in expected_context_predictions(MASTER, phase)
    )
    assert len(fits) == 66
    assert len(predictions) == 726
    assert all(
        len(expected_context_fits(MASTER, phase)) == 33
        for phase in phases
    )
    assert all(
        len(expected_context_predictions(MASTER, phase)) == 363
        for phase in phases
    )
    assert len(set(fits)) == len(fits)

    phase = phases[0]
    phase_fits = expected_context_fits(MASTER, phase)
    for history in HISTORY_LENGTHS:
        group = tuple(fit for fit in phase_fits if fit.history == history)
        assert len(group) == 11
        assert tuple(fit.model for fit in group).count("global_ridge") == 1
        assert tuple(fit.model for fit in group).count("global_mlp") == 5
        assert tuple(fit.model for fit in group).count(
            "panel_transformer",
        ) == 5

    for model in ("global_mlp", "panel_transformer"):
        for seed in SEEDS:
            group = tuple(
                fit for fit in phase_fits
                if fit.model == model and fit.seed == seed
            )
            assert len(group) == 3
            assert len({fit.optimizer_updates for fit in group}) == 1
            assert len({fit.source_provenance_id for fit in group}) == 1
            assert len({fit.source_model_fingerprint for fit in group}) == 1

    phase_predictions = expected_context_predictions(MASTER, phase)
    assert sum(item.prediction_count for item in phase_predictions) == (
        33 * sum(count for _, count, _ in phase.evaluation_rows)
    )
    assert tuple(
        item.series for item in phase_predictions[:11]
    ) == MASTER[44:]


def test_phase_rejections() -> None:
    values = [phase_value(phase) for phase in TARGET_PHASES]
    for invalid in (list(reversed(values)), values[:1], [*values, values[0]]):
        raises(parse_context_phases, invalid, MASTER)

    mutations = (
        lambda item: item["training_rows"].reverse(),
        lambda item: item["training_rows"].pop(),
        lambda item: item["training_rows"].append(item["training_rows"][0]),
        lambda item: item["evaluation_rows"].reverse(),
        lambda item: item["evaluation_rows"].pop(),
        lambda item: item["evaluation_rows"].append(
            item["evaluation_rows"][0],
        ),
        lambda item: item["prior_selections"].reverse(),
        lambda item: item["prior_selections"].pop(),
        lambda item: item["prior_selections"].append(
            item["prior_selections"][0],
        ),
        lambda item: item.update({"source_ranges": [[0, 1], [2, 3]]}),
        lambda item: item.update({"training_grid_sha256": "invalid"}),
        lambda item: item.update({"extra": True}),
    )
    for mutate in mutations:
        invalid = phase_value(TARGET_PHASES[0])
        mutate(invalid)
        raises(ContextPhase.parse, invalid, MASTER)

    expected_updates = phase_value(TARGET_PHASES[0])[
        "updates_per_checkpoint"
    ]
    for invalid_updates in (expected_updates - 1, expected_updates + 1):
        invalid = phase_value(TARGET_PHASES[0])
        invalid["updates_per_checkpoint"] = invalid_updates
        raises(ContextPhase.parse, invalid, MASTER)

    for field, value in (
        ("selected_checkpoint", 0),
        ("source_provenance_id", digest("wrong")),
        ("source_model_fingerprint", "invalid"),
        ("seed", 1),
        ("model", "global_ridge"),
    ):
        invalid = phase_value(TARGET_PHASES[0])
        invalid["prior_selections"][0][field] = value
        raises(ContextPhase.parse, invalid, MASTER)

    invalid = phase_value(TARGET_PHASES[0])
    invalid["training_rows"][0]["count"] = 0
    raises(ContextPhase.parse, invalid, MASTER)
    invalid = phase_value(TARGET_PHASES[0])
    invalid["evaluation_rows"][0]["grid_sha256"] = "invalid"
    raises(ContextPhase.parse, invalid, MASTER)
    raises(ContextPhase.parse, phase_value(TARGET_PHASES[0]), MASTER[:-1])


def test_provenance_binds_every_treatment() -> None:
    phases = parse_context_phases(
        [phase_value(name) for name in TARGET_PHASES], MASTER,
    )
    phase = phases[0]
    fits = tuple(
        fit for item in phases for fit in expected_context_fits(MASTER, item)
    )
    identifiers = tuple(
        context_provenance_id(
            fit, next(
                item for item in phases if item.phase == fit.phase
            ), SOURCE_FAILURE, CONFIG_SHA256,
        )
        for fit in fits
    )
    assert len(set(identifiers)) == len(identifiers)
    fit = next(
        item for item in fits
        if item.phase == phase.phase and item.model == "global_mlp"
    )
    original = context_provenance_id(
        fit, phase, SOURCE_FAILURE, CONFIG_SHA256,
    )
    assert context_provenance_id(
        fit, phase, digest("other-source"), CONFIG_SHA256,
    ) != original
    assert context_provenance_id(
        fit, phase, SOURCE_FAILURE, digest("other-config"),
    ) != original

    for changed in (
        replace(phase, training_grid_sha256=digest("other-training-grid")),
        replace(
            phase, evaluation_grid_sha256=digest("other-evaluation-grid"),
        ),
    ):
        changed_fit = next(
            item for item in expected_context_fits(MASTER, changed)
            if item.model == fit.model and item.seed == fit.seed
            and item.history == fit.history
        )
        assert context_provenance_id(
            changed_fit, changed, SOURCE_FAILURE, CONFIG_SHA256,
        ) != original

    first, *rest = phase.training_rows
    changed = replace(
        phase,
        training_rows=((first[0], first[1] + BATCH_SIZE), *rest),
        training_grid_sha256=digest("larger-training-grid"),
        updates_per_checkpoint=phase.updates_per_checkpoint + 1,
    )
    changed_fit = next(
        item for item in expected_context_fits(MASTER, changed)
        if item.model == fit.model and item.seed == fit.seed
        and item.history == fit.history
    )
    assert context_provenance_id(
        changed_fit, changed, SOURCE_FAILURE, CONFIG_SHA256,
    ) != original

    selection = phase.prior_selections[0]
    changed = replace(
        phase,
        prior_selections=(
            replace(
                selection,
                selected_checkpoint=selection.selected_checkpoint + 1,
            ),
            *phase.prior_selections[1:],
        ),
    )
    changed_fit = next(
        item for item in expected_context_fits(MASTER, changed)
        if item.model == fit.model and item.seed == fit.seed
        and item.history == fit.history
    )
    assert context_provenance_id(
        changed_fit, changed, SOURCE_FAILURE, CONFIG_SHA256,
    ) != original
    raises(
        context_provenance_id,
        replace(fit, members=fit.members[1:]),
        phase,
        SOURCE_FAILURE,
        CONFIG_SHA256,
    )
    raises(
        context_provenance_id,
        object(),
        phase,
        SOURCE_FAILURE,
        CONFIG_SHA256,
    )

    try:
        phase.phase = "calibration"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("context phase is mutable")


def main() -> None:
    test_exact_sweep()
    test_phase_family_and_closure()
    test_phase_rejections()
    test_provenance_binds_every_treatment()


if __name__ == "__main__":
    main()
