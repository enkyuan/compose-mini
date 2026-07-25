#!/usr/bin/env python3
"""Verify seed ensembles and the frozen temporal-context decision."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from math import exp
from pathlib import Path
from unittest.mock import patch
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.context_diagnostic_contract import (
    BATCH_SIZE, CONTROL_COHORT, PHASE_RANGES, SEEDS,
    ContextPhase, ContextPredictionEvidence, context_phase_sha256,
    expected_context_predictions,
)
from tools.finalize_context_diagnostic import (
    BOOTSTRAP_SEED, ContextTruthRow, _select_context_history,
    evaluate_context_phase,
)
from tools.universe_scaling import (
    BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_REPLICATES, paired_comparison,
)
from tools.universe_scaling_contract import (
    FitJob, fit_provenance_id, timestamp_grid_sha256,
)

MASTER = tuple(f"S{index:02d}" for index in range(55))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def raises(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except (TypeError, ValueError):
        return
    raise AssertionError("expected context finalizer failure")


def truth_rows() -> tuple[ContextTruthRow, ...]:
    rows = []
    for index in range(20):
        day = date(2026, 1, 1) + timedelta(days=index)
        prefix = day.isoformat()
        actual = 0.01 if index % 2 else -0.01
        rows.append(ContextTruthRow(
            f"{prefix}T10:00:00Z", f"{prefix}T10:30:00Z",
            f"{prefix}T16:00:00Z", 100.0, 100.0 * exp(actual),
        ))
    return tuple(rows)


def phase_for(name: str = "fold-1") -> ContextPhase:
    rows = truth_rows()
    grid = timestamp_grid_sha256(tuple(
        (row.as_of, row.entry_time, row.target_time) for row in rows
    ))
    training = [
        {"count": 2, "series": series} for series in MASTER[:44]
    ]
    prior = {"fold-1": "fold-0", "calibration": "fold-1"}[name]
    return ContextPhase.parse({
        "evaluation_grid_sha256": digest(f"{name}-evaluation"),
        "evaluation_rows": [
            {"count": len(rows), "grid_sha256": grid, "series": series}
            for series in MASTER[44:]
        ],
        "phase": name,
        "prior_selections": [
            {
                "model": model,
                "seed": seed,
                "selected_checkpoint": 1,
                "source_model_fingerprint": digest(
                    f"{prior}-{model}-{seed}-state",
                ),
                "source_provenance_id": fit_provenance_id(FitJob(
                    "pooled", "fixed-update", 44, prior,
                    model, seed, MASTER[:44],
                )),
            }
            for model in ("global_mlp", "panel_transformer")
            for seed in SEEDS
        ],
        "source_ranges": list(map(list, PHASE_RANGES[name])),
        "scaler_inputs_sha256": digest(f"{name}-scalers"),
        "training_grid_sha256": digest(f"{name}-training"),
        "training_rows": training,
        "updates_per_checkpoint": (
            sum(row["count"] for row in training[:CONTROL_COHORT]) +
            BATCH_SIZE - 1
        ) // BATCH_SIZE,
    }, MASTER)


def truth_for() -> dict[str, tuple[ContextTruthRow, ...]]:
    return {series: truth_rows() for series in MASTER[44:]}


def evidence_for(
    phase: ContextPhase,
) -> tuple[ContextPredictionEvidence, ...]:
    actual = tuple(row.actual_return for row in truth_rows())
    offsets = dict(zip(SEEDS, (-0.002, -0.001, 0.0, 0.001, 0.002)))
    records = []
    for prediction in expected_context_predictions(MASTER, phase):
        fit = prediction.fit
        if fit.model == "panel_transformer":
            scale = {17: 0.0, 34: 1.0, 68: 0.5}[fit.history]
            offset = offsets[fit.seed]
        elif fit.model == "global_mlp":
            scale, offset = 0.25, offsets[fit.seed]
        else:
            scale = offset = 0.0
        values = tuple(value * scale + offset for value in actual)
        records.append(ContextPredictionEvidence(
            prediction, digest(f"{fit}-provenance"),
            digest(f"{fit}-state"), values,
        ))
    return tuple(records)


def test_seed_ensemble_precedes_metrics() -> dict[str, object]:
    phase = phase_for()
    with patch(
        "tools.finalize_context_diagnostic.paired_comparison",
        wraps=paired_comparison,
    ) as compare:
        result = evaluate_context_phase(
            MASTER, phase, evidence_for(phase), truth_for(),
        )
    panel = result["descriptive_metrics"]["panel_transformer"]
    assert panel["34"]["return_mae"] < 1e-12
    assert abs(panel["17"]["return_mae"] - 0.01) < 1e-15
    assert compare.call_count == 2
    assert all(
        call.kwargs == {
            "block_days": BOOTSTRAP_BLOCK_DAYS,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        }
        for call in compare.call_args_list
    )
    for comparison in result["primary"].values():
        assert comparison["mean_gain"] > 0
        assert len(comparison["common_dates"]) == 20
        assert all(
            interval[0] > 0
            for interval in comparison["intervals"].values()
        )
    return result


def test_evidence_and_truth_closure() -> None:
    phase = phase_for()
    evidence = evidence_for(phase)
    truth = truth_for()
    assert "actual_return" not in ContextTruthRow.__dataclass_fields__
    raises(evaluate_context_phase, MASTER, phase, evidence[:-1], truth)
    changed = truth | {
        MASTER[44]: (
            replace(
                truth[MASTER[44]][0],
                target_time="2026-01-01T16:30:00Z",
            ),
            *truth[MASTER[44]][1:],
        ),
    }
    raises(evaluate_context_phase, MASTER, phase, evidence, changed)


def test_primary_only_decision(result: dict[str, object]) -> None:
    phases = (phase_for(), phase_for("calibration"))
    evaluations = {
        "fold-1": result,
        "calibration": deepcopy(result) | {
            "phase": "calibration",
            "phase_sha256": context_phase_sha256(phases[1]),
        },
    }
    assert _select_context_history(phases, evaluations) == {
        "qualifies": {"34": True, "68": True},
        "selected_history": 34,
    }

    fallback = deepcopy(evaluations)
    fallback["fold-1"]["primary"]["34"]["intervals"]["5"] = (0.0, 1.0)
    assert _select_context_history(
        phases, fallback,
    )["selected_history"] == 68

    retain = deepcopy(fallback)
    retain["calibration"]["primary"]["68"]["intervals"]["20"] = (-0.1, 1.0)
    assert _select_context_history(
        phases, retain,
    )["selected_history"] == 17
    retain["fold-1"]["descriptive_metrics"] = {"control": "wins"}
    assert _select_context_history(
        phases, retain,
    )["selected_history"] == 17

    missing = deepcopy(evaluations)
    del missing["fold-1"]["primary"]["34"]["intervals"]["10"]
    raises(_select_context_history, phases, missing)

    malformed = deepcopy(evaluations)
    malformed["fold-1"]["schema"] = True
    raises(_select_context_history, phases, malformed)
    malformed = deepcopy(evaluations)
    malformed["fold-1"]["primary"]["34"]["intervals"]["5"] = (
        10 ** 10_000, 10 ** 10_000,
    )
    raises(_select_context_history, phases, malformed)

    forged = replace(phases[0], source_ranges=((0, 1), (2, 3)))
    malformed = deepcopy(evaluations)
    malformed["fold-1"]["phase_sha256"] = context_phase_sha256(forged)
    raises(_select_context_history, (forged, phases[1]), malformed)


def main() -> None:
    result = test_seed_ensemble_precedes_metrics()
    test_evidence_and_truth_closure()
    test_primary_only_decision(result)
    print("context diagnostic finalizer tests passed")


if __name__ == "__main__":
    main()
